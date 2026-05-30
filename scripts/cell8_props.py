# ============================================================
# cell8_props.py
# Player props model with H2H matchup context.
# Markets: PTS / REB / AST / 3PM
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
# H2H features added as lightweight context columns
# They adjust the projection slightly when a player has
# recent history against this specific opponent.
PTS_FEATURES  = ["pts_L10","pts_L5","min_L10","min_L5",
                 "ts_L10","usg_L10","pts_pm_L10",
                 "opp_pts_allowed","is_playoff","day_of_season",
                 "h2h_pts_avg","h2h_games"]
REB_FEATURES  = ["reb_L10","reb_L5","min_L10","min_L5",
                 "reb_pm_L10","is_playoff","day_of_season",
                 "h2h_reb_avg","h2h_games"]
AST_FEATURES  = ["ast_L10","ast_L5","min_L10","min_L5",
                 "ast_pm_L10","usg_L10","is_playoff","day_of_season",
                 "h2h_ast_avg","h2h_games"]
FG3M_FEATURES = ["fg3a_L10","fg3a_L5","fg3a_pm_L10",
                 "fg3m_L10","fg3m_L5","fg3_pct_L10",
                 "min_L10","min_L5","usg_L10",
                 "is_playoff","day_of_season",
                 "h2h_fg3m_avg","h2h_games"]

PROP_CONFIGS = [
    ("pts",  PTS_FEATURES,  "pts"),
    ("reb",  REB_FEATURES,  "reb"),
    ("ast",  AST_FEATURES,  "ast"),
    ("fg3m", FG3M_FEATURES, "fg3m"),
]

# ── PROP MODEL PATH ──
PROP_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "prop_models.pkl")


def pull_player_logs(seasons=[2022,2023,2024,2025]):
    all_rows = []
    for season in seasons:
        print(f"  {season}...")
        cursor = None
        while True:
            params = {"seasons[]": season, "per_page": 100}
            if cursor: params["cursor"] = cursor
            r = requests.get(f"{BASE_URL}/player_stats",
                             headers=HEADERS, params=params)
            if r.status_code == 429: time.sleep(5); continue
            if r.status_code != 200: break
            data = r.json(); rows = data.get("data",[])
            if not rows: break
            for row in rows:
                player  = row.get("player",{}) or {}
                team    = row.get("team",{})   or {}
                game    = row.get("game",{})   or {}
                min_raw = row.get("min","0") or "0"
                try:
                    minutes = float(str(min_raw).split(":")[0]) \
                              if ":" in str(min_raw) else float(min_raw)
                except: minutes = 0
                if minutes < 3: continue
                fga=row.get("fga") or 0; fta=row.get("fta") or 0
                pts=row.get("pts") or 0; reb=row.get("reb") or 0
                ast=row.get("ast") or 0; tov=row.get("turnover") or 0
                fg3m=row.get("fg3m") or 0; fg3a=row.get("fg3a") or 0
                ts_den=2*(fga+0.44*fta)
                ts=pts/ts_den if ts_den>0 else 0
                usg=(fga+0.44*fta+tov)/minutes if minutes>0 else 0
                fg3_pct=fg3m/fg3a if fg3a>0 else 0
                # H2H columns default to 0 at training time
                # They get populated at inference time for live games
                all_rows.append({
                    "player_id":   player.get("id"),
                    "player_name": f"{player.get('first_name','')} {player.get('last_name','')}".strip(),
                    "team_id":     team.get("id"),
                    "team_abbr":   team.get("abbreviation"),
                    "game_id":     game.get("id"),
                    "opp_team_id": game.get("home_team_id") if team.get("id") != game.get("home_team_id") else game.get("visitor_team_id"),
                    "season":      game.get("season") or season,
                    "minutes":     round(minutes,1),
                    "pts":pts,"reb":reb,"ast":ast,
                    "fg3m":fg3m,"fg3a":fg3a,"fga":fga,"fta":fta,"tov":tov,
                    "ts_pct":round(ts,4),"usg":round(usg,4),
                    "fg3_pct":round(fg3_pct,4),
                    "pts_pm":round(pts/minutes,4) if minutes>0 else 0,
                    "reb_pm":round(reb/minutes,4) if minutes>0 else 0,
                    "ast_pm":round(ast/minutes,4) if minutes>0 else 0,
                    "fg3a_pm":round(fg3a/minutes,4) if minutes>0 else 0,
                    "fg3m_pm":round(fg3m/minutes,4) if minutes>0 else 0,
                    # H2H defaults — 0 means no h2h context (neutral effect)
                    "h2h_pts_avg":  0.0,
                    "h2h_reb_avg":  0.0,
                    "h2h_ast_avg":  0.0,
                    "h2h_fg3m_avg": 0.0,
                    "h2h_games":    0,
                })
            nc = data.get("meta",{}).get("next_cursor")
            if nc: cursor=nc; time.sleep(0.1)
            else: break
        print(f"    {season}: {len([r for r in all_rows if r['season']==season])} rows")
        time.sleep(0.2)
    return pd.DataFrame(all_rows)


def build_prop_features(df, game_dates):
    df = df.merge(game_dates, on="game_id", how="left")
    df = df.dropna(subset=["date"])
    df = df.sort_values(["player_id","date"]).reset_index(drop=True)

    def rmean(s,w,m=3): return s.shift(1).rolling(w,min_periods=m).mean()

    grp = df.groupby("player_id")
    for col,alias in [
        ("pts","pts"),("reb","reb"),("ast","ast"),
        ("minutes","min"),("ts_pct","ts"),("usg","usg"),
        ("pts_pm","pts_pm"),("reb_pm","reb_pm"),("ast_pm","ast_pm"),
        ("fg3m","fg3m"),("fg3a","fg3a"),("fg3_pct","fg3_pct"),
        ("fg3a_pm","fg3a_pm"),("fg3m_pm","fg3m_pm"),
    ]:
        df[f"{alias}_L10"] = grp[col].transform(lambda x: rmean(x,10))
        df[f"{alias}_L5"]  = grp[col].transform(lambda x: rmean(x,5))

    df = df.merge(
        df_games[["game_id","postseason","season"]].rename(
            columns={"season":"game_season"}),
        on="game_id", how="left"
    )
    df["is_playoff"]    = df["postseason"].fillna(False).astype(int)
    df["day_of_season"] = df.groupby("game_season")["date"].transform(
        lambda x: (x-x.min()).dt.days)

    if "home_pts_allowed_L10" in df_merged.columns and not df_merged.empty:
        opp = df_merged[["game_id","home_team_id","away_team_id",
                          "home_pts_allowed_L10","away_pts_allowed_L10"]].copy()
        df = df.merge(opp, on="game_id", how="left")
        df["opp_pts_allowed"] = np.where(
            df["team_id"]==df["home_team_id"],
            df["away_pts_allowed_L10"], df["home_pts_allowed_L10"])
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
    df_m = df.dropna(subset=[f for f in all_feats
                              if f not in ["h2h_pts_avg","h2h_reb_avg",
                                           "h2h_ast_avg","h2h_fg3m_avg","h2h_games"]]
                     + ["pts","reb","ast","fg3m"])
    # fill h2h cols with 0 for training (no h2h at training time)
    for col in ["h2h_pts_avg","h2h_reb_avg","h2h_ast_avg","h2h_fg3m_avg","h2h_games"]:
        df_m[col] = df_m[col].fillna(0)

    df_t = df_m[df_m["season"] <= 2025]
    print(f"Training on {len(df_t)} rows")

    prop_models  = {}
    prop_scalers = {}
    for prop, features, target in PROP_CONFIGS:
        sc = StandardScaler()
        m  = Ridge(alpha=1.0)
        m.fit(sc.fit_transform(df_t[features].fillna(0)), df_t[target])
        prop_models[prop]  = m
        prop_scalers[prop] = sc
        print(f"  {prop} model trained")

    os.makedirs(os.path.dirname(PROP_MODEL_PATH), exist_ok=True)
    with open(PROP_MODEL_PATH,"wb") as f2:
        pickle.dump({"models":prop_models,"scalers":prop_scalers}, f2)
    print("Prop models saved.")
    return prop_models, prop_scalers


if os.path.exists(PROP_MODEL_PATH):
    with open(PROP_MODEL_PATH,"rb") as f:
        pa = pickle.load(f)
    prop_models  = pa["models"]
    prop_scalers = pa["scalers"]
    print("Prop models loaded from cache.")
else:
    prop_models, prop_scalers = train_prop_models()


# ── HELPERS ──

def american_to_prob(odds):
    odds = float(odds)
    return (-odds/(-odds+100)) if odds < 0 else (100/(odds+100))

def calc_ev(prob, odds, stake=100):
    odds=float(odds)
    win_amt=stake*(100/-odds) if odds<0 else stake*(odds/100)
    return round((prob*win_amt)-((1-prob)*stake),2)

def prop_edge(proj, line, over_odds, under_odds):
    if not line: return None
    diff=proj-line
    over_prob=1/(1+math.exp(-diff/2.0)); under_prob=1-over_prob
    o=float(over_odds) if over_odds else -115.0
    u=float(under_odds) if under_odds else -115.0
    bo=american_to_prob(o); bu=american_to_prob(u); t=bo+bu
    bof=bo/t; buf=bu/t
    over_ev=calc_ev(over_prob,o); under_ev=calc_ev(under_prob,u)
    best="OVER" if over_ev>=under_ev else "UNDER"
    bev=over_ev if best=="OVER" else under_ev
    bodds=o if best=="OVER" else u
    bprob=over_prob if best=="OVER" else under_prob
    bedge=round((over_prob-bof)*100,1) if best=="OVER" else round((under_prob-buf)*100,1)
    bbook=bof if best=="OVER" else buf
    return {"diff":round(diff,1),"best_side":best,"best_ev":bev,
            "best_odds":int(bodds),"best_prob":round(bprob*100,1),
            "best_edge":bedge,"book_prob":round(bbook*100,1)}

def get_active_players(team_id):
    r=requests.get(f"{BASE_URL}/players/active",headers=HEADERS,
                   params={"team_ids[]":team_id,"per_page":20})
    return r.json().get("data",[]) if r.status_code==200 else []

def get_player_stats(player_id, n=10):
    r=requests.get(f"{BASE_URL}/player_stats",headers=HEADERS,
                   params={"player_ids[]":player_id,"seasons[]":CURRENT_SEASON,"per_page":n})
    if r.status_code!=200: return None
    rows=r.json().get("data",[])
    if len(rows)<3: return None
    pts_l,reb_l,ast_l,min_l,ts_l,usg_l=[],[],[],[],[],[]
    pts_pm,reb_pm,ast_pm=[],[],[]
    fg3m_l,fg3a_l,fg3pct_l,fg3apm_l,fg3mpm_l=[],[],[],[],[]
    for row in rows[:n]:
        min_raw=row.get("min","0") or "0"
        try: minutes=float(str(min_raw).split(":")[0]) if ":" in str(min_raw) else float(min_raw)
        except: minutes=0
        if minutes<3: continue
        pts=row.get("pts") or 0; reb=row.get("reb") or 0; ast=row.get("ast") or 0
        fg3m=row.get("fg3m") or 0; fg3a=row.get("fg3a") or 0
        fga=row.get("fga") or 0; fta=row.get("fta") or 0; tov=row.get("turnover") or 0
        ts_den=2*(fga+0.44*fta)
        ts=pts/ts_den if ts_den>0 else 0
        usg=(fga+0.44*fta+tov)/minutes if minutes>0 else 0
        fg3_pct=fg3m/fg3a if fg3a>0 else 0
        pts_l.append(pts); reb_l.append(reb); ast_l.append(ast)
        min_l.append(minutes); ts_l.append(ts); usg_l.append(usg)
        pts_pm.append(pts/minutes if minutes>0 else 0)
        reb_pm.append(reb/minutes if minutes>0 else 0)
        ast_pm.append(ast/minutes if minutes>0 else 0)
        fg3m_l.append(fg3m); fg3a_l.append(fg3a); fg3pct_l.append(fg3_pct)
        fg3apm_l.append(fg3a/minutes if minutes>0 else 0)
        fg3mpm_l.append(fg3m/minutes if minutes>0 else 0)
    def avg(lst,k=None):
        s=lst[-k:] if k else lst
        return round(sum(s)/len(s),4) if s else None
    return {
        "pts_L10":avg(pts_l),"pts_L5":avg(pts_l,5),
        "reb_L10":avg(reb_l),"reb_L5":avg(reb_l,5),
        "ast_L10":avg(ast_l),"ast_L5":avg(ast_l,5),
        "min_L10":avg(min_l),"min_L5":avg(min_l,5),
        "ts_L10":avg(ts_l),"usg_L10":avg(usg_l),
        "pts_pm_L10":avg(pts_pm),"reb_pm_L10":avg(reb_pm),"ast_pm_L10":avg(ast_pm),
        "fg3m_L10":avg(fg3m_l),"fg3m_L5":avg(fg3m_l,5),
        "fg3a_L10":avg(fg3a_l),"fg3a_L5":avg(fg3a_l,5),
        "fg3_pct_L10":avg(fg3pct_l),
        "fg3a_pm_L10":avg(fg3apm_l),"fg3m_pm_L10":avg(fg3mpm_l),
    }


def get_h2h_stats(player_id, opp_team_id, n=8):
    """
    Pull player's stats from games against this specific opponent
    in the current season. Returns averages for pts/reb/ast/fg3m
    and the number of h2h games found.
    Uses current season only — avoids stale pre-trade data.
    """
    r = requests.get(f"{BASE_URL}/player_stats", headers=HEADERS,
                     params={"player_ids[]": player_id,
                             "seasons[]":    CURRENT_SEASON,
                             "per_page":     50})
    time.sleep(0.1)
    if r.status_code != 200:
        return {"h2h_pts_avg":0,"h2h_reb_avg":0,
                "h2h_ast_avg":0,"h2h_fg3m_avg":0,"h2h_games":0}

    rows = r.json().get("data",[])
    h2h  = []

    for row in rows:
        game   = row.get("game",{}) or {}
        home_t = game.get("home_team_id")
        away_t = game.get("visitor_team_id")
        # check if opponent team was in this game
        if opp_team_id not in [home_t, away_t]:
            continue
        min_raw = row.get("min","0") or "0"
        try:
            minutes = float(str(min_raw).split(":")[0]) \
                      if ":" in str(min_raw) else float(min_raw)
        except:
            minutes = 0
        if minutes < 3:
            continue
        h2h.append({
            "pts":  row.get("pts")  or 0,
            "reb":  row.get("reb")  or 0,
            "ast":  row.get("ast")  or 0,
            "fg3m": row.get("fg3m") or 0,
        })

    if not h2h:
        return {"h2h_pts_avg":0,"h2h_reb_avg":0,
                "h2h_ast_avg":0,"h2h_fg3m_avg":0,"h2h_games":0}

    def avg(key):
        return round(sum(g[key] for g in h2h)/len(h2h), 2)

    return {
        "h2h_pts_avg":  avg("pts"),
        "h2h_reb_avg":  avg("reb"),
        "h2h_ast_avg":  avg("ast"),
        "h2h_fg3m_avg": avg("fg3m"),
        "h2h_games":    len(h2h),
    }


def get_props(game_id):
    all_props=[]
    for vendor in PROP_VENDORS:
        r=requests.get(f"{BASE_URL}/odds/player_props",headers=HEADERS,
                       params={"game_id":game_id,"vendors[]":vendor,"per_page":100})
        time.sleep(0.1)
        if r.status_code!=200: continue
        data=r.json().get("data",[])
        if data: all_props.extend(data); break
    organized={}
    for p in all_props:
        pid=p.get("player_id")
        if not pid: continue
        ptype=p.get("prop_type","").lower()
        line=p.get("line_value")
        mkt=p.get("market",{}) or {}
        v=p.get("vendor","?")
        if any(k in ptype for k in ["point","pts","score"]): key="pts"
        elif any(k in ptype for k in ["rebound","reb"]): key="reb"
        elif any(k in ptype for k in ["assist","ast"]): key="ast"
        elif any(k in ptype for k in ["three","3pt","3pm","threes","three_point"]): key="fg3m"
        else: continue
        if mkt.get("type")!="over_under": continue
        if pid not in organized: organized[pid]={}
        if key not in organized[pid]:
            organized[pid][key]={"line":float(line) if line else None,
                                  "over_odds":mkt.get("over_odds"),
                                  "under_odds":mkt.get("under_odds"),"vendor":v}
    return organized


def project(stats, h2h, opp_pts_allowed, is_playoff, day_of_season):
    """
    Project all 4 prop markets for a player.
    H2H stats are included as lightweight context — they carry
    low weight in the model since sample sizes are small.
    """
    results = {}
    for prop, features, _t in PROP_CONFIGS:
        fv = {}
        for f in features:
            if f == "opp_pts_allowed":  fv[f] = opp_pts_allowed or 82.0
            elif f == "is_playoff":     fv[f] = is_playoff
            elif f == "day_of_season":  fv[f] = day_of_season
            elif f in h2h:              fv[f] = h2h[f]
            else:                       fv[f] = stats.get(f) or 0
        X    = pd.DataFrame([fv])[features].fillna(0)
        X_sc = prop_scalers[prop].transform(X)
        results[prop] = max(0, round(prop_models[prop].predict(X_sc)[0], 1))
    return results


# ── LOAD TODAY'S GAMES ──
spreadsheet = get_sheet()

try:
    proj_ws   = spreadsheet.worksheet("Projections")
    proj_data = proj_ws.get_all_records()
    games_df  = pd.DataFrame(proj_data)
except Exception as e:
    print(f"Could not load Projections: {e}"); exit()

if games_df.empty:
    print("No games in Projections tab."); exit()

print(f"Loaded {len(games_df)} game(s) from Projections tab")

# ── GENERATE SIGNALS ──
season_start  = pd.to_datetime(SEASON_START_DATE)
day_of_szn    = max((pd.to_datetime(GAME_DAY)-season_start).days, 0)
all_signals   = []
all_proj_rows = []

for _, g in games_df.iterrows():
    game_id    = g.get("game_id")
    game_str   = str(g.get("game","")).strip()
    tip_off    = g.get("tip_off_et","")
    is_playoff = int(g.get("is_playoff", 0))
    if not game_id: continue

    parts     = game_str.split(" @ ")
    home_abbr = parts[-1] if len(parts)>1 else ""
    away_abbr = parts[0]  if len(parts)>1 else ""

    print(f"\n  {game_str}  {tip_off}")

    props = get_props(game_id); time.sleep(0.15)
    print(f"    {len(props)} players with lines")

    opp_pts_allowed = 82.0
    if "away_pts_allowed_L10" in df_merged.columns and not df_merged.empty:
        recent = df_merged[df_merged["away_abbr"]==home_abbr].tail(1)
        if not recent.empty:
            opp_pts_allowed = float(recent["away_pts_allowed_L10"].values[0] or 82)

    # Get team IDs and opponent team IDs
    team_pairs = []
    for abbr, opp_abbr in [(home_abbr, away_abbr), (away_abbr, home_abbr)]:
        if df_merged.empty: continue
        t_match = df_merged[df_merged["home_abbr"]==abbr]["home_team_id"]
        if t_match.empty:
            t_match = df_merged[df_merged["away_abbr"]==abbr]["away_team_id"]
        o_match = df_merged[df_merged["home_abbr"]==opp_abbr]["home_team_id"]
        if o_match.empty:
            o_match = df_merged[df_merged["away_abbr"]==opp_abbr]["away_team_id"]
        if not t_match.empty and not o_match.empty:
            team_pairs.append((int(t_match.iloc[-1]), abbr, int(o_match.iloc[-1])))

    for team_id, team_abbr, opp_team_id in team_pairs:
        players = get_active_players(team_id); time.sleep(0.15)

        for player in players:
            pid   = player["id"]
            pname = f"{player.get('first_name','')} {player.get('last_name','')}".strip()

            stats = get_player_stats(pid); time.sleep(0.1)
            if not stats or not stats.get("pts_L10"): continue
            if (stats.get("min_L10") or 0) < 10: continue

            # H2H stats — current season only
            h2h = get_h2h_stats(pid, opp_team_id)

            proj = project(stats, h2h, opp_pts_allowed, is_playoff, day_of_szn)
            book = props.get(pid, {})

            row = {
                "game_date":     GAME_DAY,
                "tip_off":       tip_off,
                "game":          game_str,
                "team":          team_abbr,
                "player":        pname,
                "player_id":     pid,
                "min_L10":       round(stats.get("min_L10") or 0, 1),
                "proj_pts":      proj["pts"],
                "proj_reb":      proj["reb"],
                "proj_ast":      proj["ast"],
                "proj_3pm":      proj["fg3m"],
                "h2h_games":     h2h["h2h_games"],
                "h2h_pts_avg":   h2h["h2h_pts_avg"],
                "h2h_reb_avg":   h2h["h2h_reb_avg"],
                "h2h_ast_avg":   h2h["h2h_ast_avg"],
                "h2h_fg3m_avg":  h2h["h2h_fg3m_avg"],
                "book_pts_line": book.get("pts",{}).get("line",""),
                "book_reb_line": book.get("reb",{}).get("line",""),
                "book_ast_line": book.get("ast",{}).get("line",""),
                "book_3pm_line": book.get("fg3m",{}).get("line",""),
                "action":        "PASS",
            }

            signals = []
            for pk, pv in [("pts",proj["pts"]),("reb",proj["reb"]),
                            ("ast",proj["ast"]),("fg3m",proj["fg3m"])]:
                if pk not in book: continue
                b=book[pk]; line=b.get("line")
                o_odds=b.get("over_odds",-115); u_odds=b.get("under_odds",-115)
                if not line: continue
                ei=prop_edge(pv,line,o_odds,u_odds)
                if not ei: continue
                if abs(ei["diff"])>=EDGE_THRESHOLDS_PROP[pk]:
                    label="3PM" if pk=="fg3m" else pk.upper()
                    signals.append({
                        "game_date":GAME_DAY,"tip_off":tip_off,"game":game_str,
                        "team":team_abbr,"player":pname,"player_id":pid,
                        "prop":label,"side":ei["best_side"],"line":line,
                        "odds":ei["best_odds"],"proj":pv,"diff":ei["diff"],
                        "model_prob":ei["best_prob"],"book_prob":ei["book_prob"],
                        "edge_pct":ei["best_edge"],"ev":ei["best_ev"],
                        "vendor":b["vendor"],
                        "h2h_games":h2h["h2h_games"],
                        "h2h_avg":h2h.get(f"h2h_{pk}_avg",0),
                    })

            if signals:
                row["action"]=" | ".join(
                    f"{s['prop']} {s['side']} {s['line']}" for s in signals)
                all_signals.extend(signals)
            all_proj_rows.append(row)

print(f"\nTotal prop signals: {len(all_signals)}")
if all_signals:
    for s in sorted(all_signals, key=lambda x: x["ev"], reverse=True):
        h2h_note = f" [H2H {s['h2h_games']}g avg {s['h2h_avg']}]" if s["h2h_games"] > 0 else ""
        print(f"  {s['player']} ({s['team']})  "
              f"{s['prop']} {s['side']} {s['line']} @ {s['odds']:+d}  "
              f"EV {s['ev']:+.2f}{h2h_note}")

for tab_name, rows in [("Player Props",all_proj_rows),("Props Signals",all_signals)]:
    if not rows: continue
    df_out = pd.DataFrame(rows)
    try:
        ws=spreadsheet.worksheet(tab_name); ws.clear()
    except:
        ws=spreadsheet.add_worksheet(title=tab_name,rows=500,cols=30)
    set_with_dataframe(ws, df_out)
    print(f"Saved {len(rows)} rows to '{tab_name}'.")

print("=== PROPS COMPLETE ===")
