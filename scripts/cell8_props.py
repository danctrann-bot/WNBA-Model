# ============================================================
# cell8_props.py
# Trains prop models (PTS/REB/AST/3PM), generates live
# signals by comparing projections to book prop lines.
#
# Runs: daily 12:15pm ET via GitHub Actions
# ============================================================

import requests
import pandas as pd
import numpy as np
import pickle
import math
import time
import os
from datetime import date, timedelta, datetime
import pytz
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from gspread_dataframe import set_with_dataframe
from config import (API_KEY, BASE_URL, HEADERS, get_sheet,
                    CURRENT_SEASON, SEASON_START_DATE,
                    PROP_VENDORS, EDGE_THRESHOLDS_PROP)

print("=== PROPS SIGNALS STARTING ===")

ET       = pytz.timezone("America/New_York")
GAME_DAY = datetime.now(ET).strftime("%Y-%m-%d")

# ── LOAD GAME MODELS ──
GAME_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "game_models.pkl")
with open(GAME_MODEL_PATH, "rb") as f:
    art = pickle.load(f)
df_games  = art["df_games"]
df_merged = art.get("df_merged", pd.DataFrame())
print("Game models loaded.")

# ── FEATURE SETS ──
PTS_FEATURES  = ["pts_L10","pts_L5","min_L10","min_L5",
                 "ts_L10","usg_L10","pts_pm_L10",
                 "opp_pts_allowed","is_playoff","day_of_season"]
REB_FEATURES  = ["reb_L10","reb_L5","min_L10","min_L5",
                 "reb_pm_L10","is_playoff","day_of_season"]
AST_FEATURES  = ["ast_L10","ast_L5","min_L10","min_L5",
                 "ast_pm_L10","usg_L10","is_playoff","day_of_season"]
FG3M_FEATURES = ["fg3a_L10","fg3a_L5","fg3a_pm_L10",
                 "fg3m_L10","fg3m_L5","fg3_pct_L10",
                 "min_L10","min_L5","usg_L10",
                 "is_playoff","day_of_season"]
PROP_CONFIGS  = [
    ("pts",  PTS_FEATURES,  "pts"),
    ("reb",  REB_FEATURES,  "reb"),
    ("ast",  AST_FEATURES,  "ast"),
    ("fg3m", FG3M_FEATURES, "fg3m"),
]


# ============================================================
# STEP 1 — TRAIN PROP MODELS
# Pulls historical player game logs, builds rolling features,
# trains one Ridge regression per prop market.
# Caches to models/prop_models.pkl so it only retrains
# when the cache is missing (Sunday retrain clears it).
# ============================================================

PROP_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "prop_models.pkl")

def pull_player_logs(seasons=[2022,2023,2024,2025]):
    all_rows = []
    for season in seasons:
        print(f"  {season}...")
        cursor = None
        while True:
            params = {"seasons[]": season, "per_page": 100}
            if cursor:
                params["cursor"] = cursor
            r = requests.get(f"{BASE_URL}/player_stats",
                             headers=HEADERS, params=params)
            if r.status_code == 429:
                time.sleep(5); continue
            if r.status_code != 200:
                break
            data = r.json()
            rows = data.get("data", [])
            if not rows:
                break
            for row in rows:
                player  = row.get("player", {}) or {}
                team    = row.get("team",   {}) or {}
                game    = row.get("game",   {}) or {}
                min_raw = row.get("min","0") or "0"
                try:
                    minutes = float(str(min_raw).split(":")[0]) \
                              if ":" in str(min_raw) else float(min_raw)
                except:
                    minutes = 0
                if minutes < 3:
                    continue
                fga  = row.get("fga")      or 0
                fta  = row.get("fta")      or 0
                pts  = row.get("pts")      or 0
                reb  = row.get("reb")      or 0
                ast  = row.get("ast")      or 0
                tov  = row.get("turnover") or 0
                fg3m = row.get("fg3m")     or 0
                fg3a = row.get("fg3a")     or 0
                ts_den  = 2*(fga+0.44*fta)
                ts      = pts/ts_den if ts_den > 0 else 0
                usg     = (fga+0.44*fta+tov)/minutes if minutes > 0 else 0
                fg3_pct = fg3m/fg3a if fg3a > 0 else 0
                all_rows.append({
                    "player_id":   player.get("id"),
                    "player_name": f"{player.get('first_name','')} {player.get('last_name','')}".strip(),
                    "team_id":     team.get("id"),
                    "team_abbr":   team.get("abbreviation"),
                    "game_id":     game.get("id"),
                    "season":      game.get("season") or season,
                    "minutes":     round(minutes, 1),
                    "pts":  pts,  "reb":  reb,  "ast":  ast,
                    "fg3m": fg3m, "fg3a": fg3a,
                    "fga":  fga,  "fta":  fta,  "tov":  tov,
                    "ts_pct":   round(ts,      4),
                    "usg":      round(usg,     4),
                    "fg3_pct":  round(fg3_pct, 4),
                    "pts_pm":   round(pts/minutes,  4) if minutes > 0 else 0,
                    "reb_pm":   round(reb/minutes,  4) if minutes > 0 else 0,
                    "ast_pm":   round(ast/minutes,  4) if minutes > 0 else 0,
                    "fg3a_pm":  round(fg3a/minutes, 4) if minutes > 0 else 0,
                    "fg3m_pm":  round(fg3m/minutes, 4) if minutes > 0 else 0,
                })
            nc = data.get("meta", {}).get("next_cursor")
            if nc:
                cursor = nc; time.sleep(0.1)
            else:
                break
        print(f"    {season}: {len([r for r in all_rows if r['season']==season])} rows")
        time.sleep(0.2)
    return pd.DataFrame(all_rows)


def build_prop_features(df, game_dates):
    df = df.merge(game_dates, on="game_id", how="left")
    df = df.dropna(subset=["date"])
    df = df.sort_values(["player_id","date"]).reset_index(drop=True)

    def rmean(s, w, m=3):
        return s.shift(1).rolling(w, min_periods=m).mean()

    grp = df.groupby("player_id")
    for col, alias in [
        ("pts","pts"),("reb","reb"),("ast","ast"),
        ("minutes","min"),("ts_pct","ts"),("usg","usg"),
        ("pts_pm","pts_pm"),("reb_pm","reb_pm"),("ast_pm","ast_pm"),
        ("fg3m","fg3m"),("fg3a","fg3a"),("fg3_pct","fg3_pct"),
        ("fg3a_pm","fg3a_pm"),("fg3m_pm","fg3m_pm"),
    ]:
        df[f"{alias}_L10"] = grp[col].transform(lambda x: rmean(x, 10))
        df[f"{alias}_L5"]  = grp[col].transform(lambda x: rmean(x, 5))

    df = df.merge(
        df_games[["game_id","postseason","season"]].rename(
            columns={"season":"game_season"}),
        on="game_id", how="left"
    )
    df["is_playoff"]    = df["postseason"].fillna(False).astype(int)
    df["day_of_season"] = df.groupby("game_season")["date"].transform(
        lambda x: (x - x.min()).dt.days)

    if "home_pts_allowed_L10" in df_merged.columns and not df_merged.empty:
        opp = df_merged[["game_id","home_team_id","away_team_id",
                          "home_pts_allowed_L10","away_pts_allowed_L10"]].copy()
        df = df.merge(opp, on="game_id", how="left")
        df["opp_pts_allowed"] = np.where(
            df["team_id"] == df["home_team_id"],
            df["away_pts_allowed_L10"],
            df["home_pts_allowed_L10"]
        )
        df["opp_pts_allowed"] = df["opp_pts_allowed"].fillna(82.0)
    else:
        df["opp_pts_allowed"] = 82.0

    return df


def train_prop_models():
    print("Training prop models...")
    df_logs = pull_player_logs()

    game_dates = df_games[["game_id","date"]].copy()
    game_dates["date"] = pd.to_datetime(game_dates["date"])

    df = build_prop_features(df_logs, game_dates)

    all_feats = list(set(PTS_FEATURES+REB_FEATURES+AST_FEATURES+FG3M_FEATURES))
    df_m      = df.dropna(subset=all_feats+["pts","reb","ast","fg3m"])
    df_t      = df_m[df_m["season"] <= 2025]
    print(f"Training on {len(df_t)} rows")

    prop_models  = {}
    prop_scalers = {}
    for prop, features, target in PROP_CONFIGS:
        sc = StandardScaler()
        m  = Ridge(alpha=1.0)
        m.fit(sc.fit_transform(df_t[features]), df_t[target])
        prop_models[prop]  = m
        prop_scalers[prop] = sc
        print(f"  {prop} model trained")

    os.makedirs(os.path.dirname(PROP_MODEL_PATH), exist_ok=True)
    with open(PROP_MODEL_PATH, "wb") as f2:
        pickle.dump({"models": prop_models, "scalers": prop_scalers}, f2)
    print("Prop models saved.")
    return prop_models, prop_scalers


# Load or train
if os.path.exists(PROP_MODEL_PATH):
    with open(PROP_MODEL_PATH, "rb") as f:
        pa = pickle.load(f)
    prop_models  = pa["models"]
    prop_scalers = pa["scalers"]
    print("Prop models loaded from cache.")
else:
    prop_models, prop_scalers = train_prop_models()


# ============================================================
# STEP 2 — HELPERS
# ============================================================

def american_to_prob(odds):
    odds = float(odds)
    return (-odds/(-odds+100)) if odds < 0 else (100/(odds+100))

def calc_ev(prob, odds, stake=100):
    odds    = float(odds)
    win_amt = stake*(100/-odds) if odds < 0 else stake*(odds/100)
    return round((prob*win_amt) - ((1-prob)*stake), 2)

def prop_edge(proj, line, over_odds, under_odds):
    if not line:
        return None
    diff       = proj - line
    over_prob  = 1/(1+math.exp(-diff/2.0))
    under_prob = 1 - over_prob
    o = float(over_odds)  if over_odds  else -115.0
    u = float(under_odds) if under_odds else -115.0
    bo = american_to_prob(o); bu = american_to_prob(u)
    t  = bo + bu
    bof = bo/t; buf = bu/t
    over_ev  = calc_ev(over_prob,  o)
    under_ev = calc_ev(under_prob, u)
    best     = "OVER" if over_ev >= under_ev else "UNDER"
    bev      = over_ev   if best == "OVER" else under_ev
    bodds    = o         if best == "OVER" else u
    bprob    = over_prob if best == "OVER" else under_prob
    bedge    = round((over_prob  - bof)*100, 1) if best == "OVER" \
               else round((under_prob - buf)*100, 1)
    bbook    = bof if best == "OVER" else buf
    return {
        "diff":       round(diff, 1),
        "best_side":  best,
        "best_ev":    bev,
        "best_odds":  int(bodds),
        "best_prob":  round(bprob*100, 1),
        "best_edge":  bedge,
        "book_prob":  round(bbook*100, 1),
    }

def get_active_players(team_id):
    r = requests.get(f"{BASE_URL}/players/active", headers=HEADERS,
                     params={"team_ids[]": team_id, "per_page": 20})
    if r.status_code != 200:
        return []
    return r.json().get("data", [])

def get_player_stats(player_id, n=10):
    r = requests.get(f"{BASE_URL}/player_stats", headers=HEADERS,
                     params={"player_ids[]": player_id,
                             "seasons[]":    CURRENT_SEASON,
                             "per_page":     n})
    if r.status_code != 200:
        return None
    rows = r.json().get("data", [])
    if len(rows) < 3:
        return None

    pts_l,reb_l,ast_l,min_l,ts_l,usg_l = [],[],[],[],[],[]
    pts_pm,reb_pm,ast_pm                = [],[],[]
    fg3m_l,fg3a_l,fg3pct_l             = [],[],[]
    fg3apm_l,fg3mpm_l                   = [],[]

    for row in rows[:n]:
        min_raw = row.get("min","0") or "0"
        try:
            minutes = float(str(min_raw).split(":")[0]) \
                      if ":" in str(min_raw) else float(min_raw)
        except:
            minutes = 0
        if minutes < 3:
            continue
        pts  = row.get("pts")      or 0
        reb  = row.get("reb")      or 0
        ast  = row.get("ast")      or 0
        fg3m = row.get("fg3m")     or 0
        fg3a = row.get("fg3a")     or 0
        fga  = row.get("fga")      or 0
        fta  = row.get("fta")      or 0
        tov  = row.get("turnover") or 0
        ts_den  = 2*(fga+0.44*fta)
        ts      = pts/ts_den if ts_den > 0 else 0
        usg     = (fga+0.44*fta+tov)/minutes if minutes > 0 else 0
        fg3_pct = fg3m/fg3a if fg3a > 0 else 0

        pts_l.append(pts);   reb_l.append(reb);   ast_l.append(ast)
        min_l.append(minutes); ts_l.append(ts);   usg_l.append(usg)
        pts_pm.append(pts/minutes  if minutes > 0 else 0)
        reb_pm.append(reb/minutes  if minutes > 0 else 0)
        ast_pm.append(ast/minutes  if minutes > 0 else 0)
        fg3m_l.append(fg3m); fg3a_l.append(fg3a); fg3pct_l.append(fg3_pct)
        fg3apm_l.append(fg3a/minutes if minutes > 0 else 0)
        fg3mpm_l.append(fg3m/minutes if minutes > 0 else 0)

    def avg(lst, k=None):
        s = lst[-k:] if k else lst
        return round(sum(s)/len(s), 4) if s else None

    return {
        "pts_L10":     avg(pts_l),    "pts_L5":      avg(pts_l, 5),
        "reb_L10":     avg(reb_l),    "reb_L5":      avg(reb_l, 5),
        "ast_L10":     avg(ast_l),    "ast_L5":      avg(ast_l, 5),
        "min_L10":     avg(min_l),    "min_L5":      avg(min_l, 5),
        "ts_L10":      avg(ts_l),     "usg_L10":     avg(usg_l),
        "pts_pm_L10":  avg(pts_pm),
        "reb_pm_L10":  avg(reb_pm),
        "ast_pm_L10":  avg(ast_pm),
        "fg3m_L10":    avg(fg3m_l),   "fg3m_L5":     avg(fg3m_l, 5),
        "fg3a_L10":    avg(fg3a_l),   "fg3a_L5":     avg(fg3a_l, 5),
        "fg3_pct_L10": avg(fg3pct_l),
        "fg3a_pm_L10": avg(fg3apm_l),
        "fg3m_pm_L10": avg(fg3mpm_l),
    }

def get_props(game_id):
    all_props = []
    for vendor in PROP_VENDORS:
        r = requests.get(f"{BASE_URL}/odds/player_props", headers=HEADERS,
                         params={"game_id": game_id,
                                 "vendors[]": vendor,
                                 "per_page": 100})
        time.sleep(0.1)
        if r.status_code != 200:
            continue
        data = r.json().get("data", [])
        if data:
            all_props.extend(data); break

    organized = {}
    for p in all_props:
        pid   = p.get("player_id")
        if not pid:
            continue
        ptype = p.get("prop_type","").lower()
        line  = p.get("line_value")
        mkt   = p.get("market", {}) or {}
        v     = p.get("vendor","?")

        if any(k in ptype for k in ["point","pts","score"]):
            key = "pts"
        elif any(k in ptype for k in ["rebound","reb"]):
            key = "reb"
        elif any(k in ptype for k in ["assist","ast"]):
            key = "ast"
        elif any(k in ptype for k in ["three","3pt","3pm","threes","three_point"]):
            key = "fg3m"
        else:
            continue

        if mkt.get("type") != "over_under":
            continue
        if pid not in organized:
            organized[pid] = {}
        if key not in organized[pid]:
            organized[pid][key] = {
                "line":       float(line) if line else None,
                "over_odds":  mkt.get("over_odds"),
                "under_odds": mkt.get("under_odds"),
                "vendor":     v,
            }
    return organized

def project(stats, opp_pts_allowed, is_playoff, day_of_season):
    results = {}
    for prop, features, _t in PROP_CONFIGS:
        fv = {}
        for f in features:
            if f == "opp_pts_allowed":
                fv[f] = opp_pts_allowed or 82.0
            elif f == "is_playoff":
                fv[f] = is_playoff
            elif f == "day_of_season":
                fv[f] = day_of_season
            else:
                fv[f] = stats.get(f) or 0
        X    = pd.DataFrame([fv])[features].fillna(0)
        X_sc = prop_scalers[prop].transform(X)
        results[prop] = max(0, round(prop_models[prop].predict(X_sc)[0], 1))
    return results


# ============================================================
# STEP 3 — LOAD TODAY'S GAMES FROM PROJECTIONS SHEET
# ============================================================

spreadsheet = get_sheet()

try:
    proj_ws   = spreadsheet.worksheet("Projections")
    proj_data = proj_ws.get_all_records()
    games_df  = pd.DataFrame(proj_data)
except Exception as e:
    print(f"Could not load Projections tab: {e}")
    print("Run cell6_projections.py first.")
    exit()

if games_df.empty:
    print("No games in Projections tab — run cell6 first.")
    exit()

print(f"Loaded {len(games_df)} game(s) from Projections tab")


# ============================================================
# STEP 4 — GENERATE PROP SIGNALS FOR EACH GAME
# ============================================================

season_start  = pd.to_datetime(SEASON_START_DATE)
day_of_szn    = max((pd.to_datetime(GAME_DAY) - season_start).days, 0)
all_signals   = []
all_proj_rows = []

for _, g in games_df.iterrows():
    game_id   = g.get("game_id")
    game_str  = str(g.get("game","")).strip()
    tip_off   = g.get("tip_off_et","")
    is_playoff= int(g.get("is_playoff", 0))
    if not game_id:
        continue

    parts     = game_str.split(" @ ")
    home_abbr = parts[-1] if len(parts) > 1 else ""
    away_abbr = parts[0]  if len(parts) > 1 else ""

    print(f"\n  {game_str}  {tip_off}")

    # Fetch prop lines
    props = get_props(game_id)
    time.sleep(0.15)
    print(f"    {len(props)} players with lines")

    # Opponent defensive context
    opp_pts_allowed = 82.0
    if "away_pts_allowed_L10" in df_merged.columns and not df_merged.empty:
        recent = df_merged[df_merged["away_abbr"] == home_abbr].tail(1)
        if not recent.empty:
            opp_pts_allowed = float(
                recent["away_pts_allowed_L10"].values[0] or 82)

    # Get team IDs from df_merged
    for abbr in [home_abbr, away_abbr]:
        matches = df_merged[df_merged["home_abbr"] == abbr]["home_team_id"] \
            if not df_merged.empty else pd.Series()
        if matches.empty and not df_merged.empty:
            matches = df_merged[df_merged["away_abbr"] == abbr]["away_team_id"]
        if matches.empty:
            continue
        team_id = int(matches.iloc[-1])

        players = get_active_players(team_id)
        time.sleep(0.15)

        for player in players:
            pid   = player["id"]
            pname = f"{player.get('first_name','')} {player.get('last_name','')}".strip()

            stats = get_player_stats(pid)
            time.sleep(0.1)
            if not stats or not stats.get("pts_L10"):
                continue
            if (stats.get("min_L10") or 0) < 10:
                continue

            proj = project(stats, opp_pts_allowed, is_playoff, day_of_szn)
            book = props.get(pid, {})

            row = {
                "game_date":     GAME_DAY,
                "tip_off":       tip_off,
                "game":          game_str,
                "team":          abbr,
                "player":        pname,
                "player_id":     pid,
                "min_L10":       round(stats.get("min_L10") or 0, 1),
                "proj_pts":      proj["pts"],
                "proj_reb":      proj["reb"],
                "proj_ast":      proj["ast"],
                "proj_3pm":      proj["fg3m"],
                "book_pts_line": book.get("pts", {}).get("line",""),
                "book_reb_line": book.get("reb", {}).get("line",""),
                "book_ast_line": book.get("ast", {}).get("line",""),
                "book_3pm_line": book.get("fg3m",{}).get("line",""),
                "action":        "PASS",
            }

            signals = []
            for pk, pv in [("pts",  proj["pts"]),
                            ("reb",  proj["reb"]),
                            ("ast",  proj["ast"]),
                            ("fg3m", proj["fg3m"])]:
                if pk not in book:
                    continue
                b     = book[pk]
                line  = b.get("line")
                o_odds= b.get("over_odds",  -115)
                u_odds= b.get("under_odds", -115)
                if not line:
                    continue
                ei = prop_edge(pv, line, o_odds, u_odds)
                if not ei:
                    continue
                if abs(ei["diff"]) >= EDGE_THRESHOLDS_PROP[pk]:
                    label = "3PM" if pk == "fg3m" else pk.upper()
                    signals.append({
                        "game_date":  GAME_DAY,
                        "tip_off":    tip_off,
                        "game":       game_str,
                        "team":       abbr,
                        "player":     pname,
                        "player_id":  pid,
                        "prop":       label,
                        "side":       ei["best_side"],
                        "line":       line,
                        "odds":       ei["best_odds"],
                        "proj":       pv,
                        "diff":       ei["diff"],
                        "model_prob": ei["best_prob"],
                        "book_prob":  ei["book_prob"],
                        "edge_pct":   ei["best_edge"],
                        "ev":         ei["best_ev"],
                        "vendor":     b["vendor"],
                    })

            if signals:
                row["action"] = " | ".join(
                    f"{s['prop']} {s['side']} {s['line']}"
                    for s in signals)
                all_signals.extend(signals)

            all_proj_rows.append(row)


# ============================================================
# STEP 5 — SAVE TO SHEETS
# ============================================================

print(f"\nTotal prop signals: {len(all_signals)}")

if all_signals:
    print("\nFlagged bets:")
    for s in sorted(all_signals, key=lambda x: x["ev"], reverse=True):
        print(f"  {s['player']} ({s['team']})  "
              f"{s['prop']} {s['side']} {s['line']} @ {s['odds']:+d}  "
              f"proj {s['proj']} (diff {s['diff']:+.1f})  "
              f"edge {s['edge_pct']:+.1f}%  EV {s['ev']:+.2f}")

for tab_name, rows in [
    ("Player Props", all_proj_rows),
    ("Props Signals", all_signals),
]:
    if not rows:
        continue
    df_out = pd.DataFrame(rows)
    try:
        ws = spreadsheet.worksheet(tab_name)
        ws.clear()
    except:
        ws = spreadsheet.add_worksheet(title=tab_name, rows=500, cols=25)
    set_with_dataframe(ws, df_out)
    print(f"Saved {len(rows)} rows to '{tab_name}' tab.")

print("=== PROPS COMPLETE ===")
