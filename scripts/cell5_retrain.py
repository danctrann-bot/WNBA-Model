# ============================================================
# cell5_retrain.py
# Weekly retrain — pulls latest game + team stat data,
# rebuilds rolling features, retrains all models,
# saves trained model artifacts for use by other scripts.
#
# Runs: every Sunday 8am ET via GitHub Actions
# ============================================================

import requests
import pandas as pd
import numpy as np
import pickle
import os
import time
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from config import API_KEY, BASE_URL, HEADERS, get_sheet, CURRENT_SEASON

print("=== WEEKLY RETRAIN STARTING ===")

# ── PULL GAMES ──
def pull_games(seasons=list(range(2021, CURRENT_SEASON + 1))):
    all_games = []
    for season in seasons:
        for season_type in [2, 3]:
            cursor = None
            while True:
                params = {"seasons[]": season, "season_type": season_type, "per_page": 100}
                if cursor:
                    params["cursor"] = cursor
                r = requests.get(f"{BASE_URL}/games", headers=HEADERS, params=params)
                if r.status_code == 429:
                    time.sleep(5); continue
                if r.status_code != 200:
                    break
                data  = r.json()
                games = data.get("data", [])
                if not games:
                    break
                for g in games:
                    if g.get("status") not in ["Final","post","final"]:
                        continue
                    all_games.append({
                        "game_id":      g["id"],
                        "date":         g["date"][:10],
                        "season":       g["season"],
                        "postseason":   g["postseason"],
                        "season_type":  "Postseason" if g["postseason"] else "Regular Season",
                        "status":       g["status"],
                        "home_team_id": g["home_team"]["id"],
                        "home_team":    g["home_team"]["full_name"],
                        "home_abbr":    g["home_team"]["abbreviation"],
                        "away_team_id": g["visitor_team"]["id"],
                        "away_team":    g["visitor_team"]["full_name"],
                        "away_abbr":    g["visitor_team"]["abbreviation"],
                        "home_score":   g.get("home_score"),
                        "away_score":   g.get("away_score"),
                    })
                next_cursor = data.get("meta", {}).get("next_cursor")
                if next_cursor:
                    cursor = next_cursor
                    time.sleep(0.1)
                else:
                    break
            time.sleep(0.2)
        print(f"  {season}: {len([g for g in all_games if g['season']==season])} games")
    df = pd.DataFrame(all_games)
    df = df.sort_values("date").reset_index(drop=True)
    print(f"Total games: {len(df)}")
    return df

# ── PULL TEAM STATS ──
def pull_team_stats(game_ids):
    all_stats = []
    for i, gid in enumerate(game_ids):
        if i % 100 == 0:
            print(f"  Team stats: {i}/{len(game_ids)}")
        r = requests.get(f"{BASE_URL}/team_stats", headers=HEADERS,
                         params={"game_ids[]": gid, "per_page": 100})
        if r.status_code == 429:
            time.sleep(5)
            r = requests.get(f"{BASE_URL}/team_stats", headers=HEADERS,
                             params={"game_ids[]": gid, "per_page": 100})
        if r.status_code != 200:
            continue
        for stat in r.json().get("data", []):
            fga = stat.get("fga") or 0; fgm = stat.get("fgm") or 0
            fg3m = stat.get("fg3m") or 0; fta = stat.get("fta") or 0
            pts = stat.get("pts") or 0; tov = stat.get("turnover") or 0
            oreb = stat.get("oreb") or 0
            ts_den = 2*(fga+0.44*fta)
            efg = (fgm+0.5*fg3m)/fga if fga > 0 else 0
            ts  = pts/ts_den if ts_den > 0 else 0
            tov_den = fga+0.44*fta+tov
            tov_pct = tov/tov_den if tov_den > 0 else 0
            all_stats.append({
                "game_id":  gid,
                "team_id":  stat["team"]["id"],
                "team":     stat["team"]["full_name"],
                "abbr":     stat["team"]["abbreviation"],
                "pts": pts, "fgm": fgm, "fga": fga,
                "fg3m": fg3m, "fg3a": stat.get("fg3a") or 0,
                "ftm": stat.get("ftm") or 0, "fta": fta,
                "oreb": oreb, "dreb": stat.get("dreb") or 0,
                "ast": stat.get("ast") or 0, "tov": tov,
                "stl": stat.get("stl") or 0, "blk": stat.get("blk") or 0,
                "efg_pct": round(efg,4), "ts_pct": round(ts,4),
                "tov_pct": round(tov_pct,4),
            })
        time.sleep(0.1)
    return pd.DataFrame(all_stats)

# ── FEATURE ENGINEERING ──
def build_features(df_games, df_team_stats):
    df = df_games.copy()
    df["date"] = pd.to_datetime(df["date"])

    df = df.merge(df_team_stats.rename(columns={
        "team_id":"home_team_id","pts":"home_pts_box","fgm":"home_fgm",
        "fga":"home_fga","fg3m":"home_fg3m","fg3a":"home_fg3a",
        "ftm":"home_ftm","fta":"home_fta","oreb":"home_oreb","dreb":"home_dreb",
        "ast":"home_ast","tov":"home_tov","stl":"home_stl","blk":"home_blk",
        "efg_pct":"home_efg","ts_pct":"home_ts","tov_pct":"home_tov_pct",
    }), on=["game_id","home_team_id"], how="left")

    df = df.merge(df_team_stats.rename(columns={
        "team_id":"away_team_id","pts":"away_pts_box","fgm":"away_fgm",
        "fga":"away_fga","fg3m":"away_fg3m","fg3a":"away_fg3a",
        "ftm":"away_ftm","fta":"away_fta","oreb":"away_oreb","dreb":"away_dreb",
        "ast":"away_ast","tov":"away_tov","stl":"away_stl","blk":"away_blk",
        "efg_pct":"away_efg","ts_pct":"away_ts","tov_pct":"away_tov_pct",
    }), on=["game_id","away_team_id"], how="left")

    df["home_pts"]   = df["home_score"].combine_first(df["home_pts_box"])
    df["away_pts"]   = df["away_score"].combine_first(df["away_pts_box"])
    df["total_pts"]  = df["home_pts"] + df["away_pts"]
    df["point_diff"] = df["home_pts"] - df["away_pts"]
    df["home_win"]   = (df["home_pts"] > df["away_pts"]).astype(int)

    df["home_pace"] = df["home_fga"] - df["home_oreb"] + df["home_tov"] + 0.44*df["home_fta"]
    df["away_pace"] = df["away_fga"] - df["away_oreb"] + df["away_tov"] + 0.44*df["away_fta"]
    df["home_ppp"]  = df["home_pts"] / df["home_pace"].replace(0, np.nan)
    df["away_ppp"]  = df["away_pts"] / df["away_pace"].replace(0, np.nan)
    df["home_dpp"]  = df["away_pts"] / df["away_pace"].replace(0, np.nan)
    df["away_dpp"]  = df["home_pts"] / df["home_pace"].replace(0, np.nan)

    def rmean(s, w, m=3):
        return s.shift(1).rolling(w, min_periods=m).mean()

    df = df.sort_values(["home_team","date"])
    g  = df.groupby("home_team")
    for col,alias in [("home_pts","home_pts"),("home_efg","home_efg"),
                      ("home_ts","home_ts"),("home_tov_pct","home_tov"),
                      ("home_pace","home_pace"),("home_ppp","home_ppp"),("home_dpp","home_dpp")]:
        df[f"{alias}_L10"] = g[col].transform(lambda x: rmean(x,10))
        if alias in ["home_pts","home_pace"]:
            df[f"{alias}_L5"] = g[col].transform(lambda x: rmean(x,5))
    df["home_pts_allowed_L10"] = g["away_pts"].transform(lambda x: rmean(x,10))

    df = df.sort_values(["away_team","date"])
    g  = df.groupby("away_team")
    for col,alias in [("away_pts","away_pts"),("away_efg","away_efg"),
                      ("away_ts","away_ts"),("away_tov_pct","away_tov"),
                      ("away_pace","away_pace"),("away_ppp","away_ppp"),("away_dpp","away_dpp")]:
        df[f"{alias}_L10"] = g[col].transform(lambda x: rmean(x,10))
        if alias in ["away_pts","away_pace"]:
            df[f"{alias}_L5"] = g[col].transform(lambda x: rmean(x,5))
    df["away_pts_allowed_L10"] = g["home_pts"].transform(lambda x: rmean(x,10))

    df = df.sort_values("date").reset_index(drop=True)

    lg_avg_dpp = df["home_dpp_L10"].mean()
    if pd.isna(lg_avg_dpp) or lg_avg_dpp == 0:
        lg_avg_dpp = 1.0
    df["proj_home_pts"] = df["home_ppp_L10"] * (df["away_dpp_L10"]/lg_avg_dpp) * df["home_pace_L10"]
    df["proj_away_pts"] = df["away_ppp_L10"] * (df["home_dpp_L10"]/lg_avg_dpp) * df["away_pace_L10"]
    df["proj_total"]    = df["proj_home_pts"] + df["proj_away_pts"]

    df = df.sort_values(["home_team","date"])
    df["home_rest"] = df.groupby("home_team")["date"].transform(
        lambda x: x.diff().dt.days.shift(1).fillna(7).clip(upper=7))
    df = df.sort_values(["away_team","date"])
    df["away_rest"] = df.groupby("away_team")["date"].transform(
        lambda x: x.diff().dt.days.shift(1).fillna(7).clip(upper=7))
    df = df.sort_values("date").reset_index(drop=True)

    df["home_b2b"]       = (df["home_rest"]==1).astype(int)
    df["away_b2b"]       = (df["away_rest"]==1).astype(int)
    df["is_playoff"]     = df["postseason"].astype(int)
    df["rest_advantage"] = df["home_rest"] - df["away_rest"]
    df["day_of_season"]  = df.groupby("season")["date"].transform(
        lambda x: (x - x.min()).dt.days)

    return df, lg_avg_dpp

FEATURE_COLS = [
    "home_pts_L10","away_pts_L10","home_pts_L5","away_pts_L5",
    "home_ppp_L10","away_ppp_L10","home_efg_L10","away_efg_L10",
    "home_ts_L10","away_ts_L10","home_tov_L10","away_tov_L10",
    "home_pts_allowed_L10","away_pts_allowed_L10",
    "home_dpp_L10","away_dpp_L10","home_pace_L10","away_pace_L10",
    "proj_home_pts","proj_away_pts","proj_total",
    "home_rest","away_rest","rest_advantage","home_b2b","away_b2b",
    "is_playoff","day_of_season",
]

# ── RUN ──
print("Pulling games...")
df_games = pull_games()

print("Pulling team stats...")
df_ts = pull_team_stats(df_games["game_id"].tolist())

print("Building features...")
df_merged, lg_avg_dpp = build_features(df_games, df_ts)

df_model = df_merged[FEATURE_COLS + ["home_pts","away_pts","total_pts","home_win","season"]].dropna()
df_train = df_model[df_model["season"] <= 2025]
print(f"Training on {len(df_train)} rows")

X      = df_train[FEATURE_COLS]
sc     = StandardScaler()
X_s    = sc.fit_transform(X)

m_home  = Ridge(alpha=1.0); m_home.fit(X_s,  df_train["home_pts"])
m_away  = Ridge(alpha=1.0); m_away.fit(X_s,  df_train["away_pts"])
m_total = Ridge(alpha=1.0); m_total.fit(X_s, df_train["total_pts"])
m_win   = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
m_win.fit(X_s, df_train["home_win"])

# Save models
os.makedirs("models", exist_ok=True)
artifacts = {
    "model_home":   m_home,
    "model_away":   m_away,
    "model_total":  m_total,
    "model_win":    m_win,
    "scaler":       sc,
    "feature_cols": FEATURE_COLS,
    "lg_avg_dpp":   lg_avg_dpp,
    "df_games":     df_games,
    "df_team_stats":df_ts,
    "df_merged":    df_merged,
}
with open("models/game_models.pkl","wb") as f:
    pickle.dump(artifacts, f)

print("Game models saved to models/game_models.pkl")

# Save raw data to Sheets for reference
spreadsheet = get_sheet()
from gspread_dataframe import set_with_dataframe

for tab_name, df_save in [("Games", df_games), ("Team Stats", df_ts)]:
    try:
        ws = spreadsheet.worksheet(tab_name)
        ws.clear()
    except:
        ws = spreadsheet.add_worksheet(title=tab_name, rows=5000, cols=30)
    set_with_dataframe(ws, df_save.reset_index(drop=True))
    print(f"Saved {tab_name} to Sheets")

print("=== RETRAIN COMPLETE ===")
