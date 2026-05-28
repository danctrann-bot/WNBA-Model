# ============================================================
# cell6_projections.py
# Pulls today's upcoming WNBA games, builds projections,
# fetches book odds, saves to Google Sheets Projections tab.
#
# Runs: daily 12pm ET via GitHub Actions
# ============================================================

import requests
import pandas as pd
import numpy as np
import pickle
import time
import math
from datetime import date, timedelta, datetime
import pytz
from config import API_KEY, BASE_URL, HEADERS, get_sheet, CURRENT_SEASON, SEASON_START_DATE
from config import LG_AVG_DPP, LG_AVG_PPP, LG_AVG_PACE, PROP_VENDORS
from gspread_dataframe import set_with_dataframe

print("=== PROJECTIONS STARTING ===")

ET = pytz.timezone("America/New_York")
GAME_DAY = datetime.now(ET).strftime("%Y-%m-%d")
print(f"Game day: {GAME_DAY}")

# ── LOAD MODELS ──
with open("models/game_models.pkl","rb") as f:
    art = pickle.load(f)

model_home   = art["model_home"]
model_away   = art["model_away"]
model_total  = art["model_total"]
model_win    = art["model_win"]
scaler       = art["scaler"]
FEATURE_COLS = art["feature_cols"]
lg_avg_dpp   = art["lg_avg_dpp"]
df_merged    = art["df_merged"]

# ── GET TODAY'S GAMES ──
def get_upcoming_games(game_day):
    upcoming = []
    game_day_dt = datetime.strptime(game_day, "%Y-%m-%d")
    dates_to_check = [game_day, (game_day_dt + timedelta(days=1)).strftime("%Y-%m-%d")]
    seen = set()
    for d in dates_to_check:
        r = requests.get(f"{BASE_URL}/games", headers=HEADERS,
                         params={"dates[]": d, "per_page": 25})
        time.sleep(0.1)
        if r.status_code != 200:
            continue
        for g in r.json().get("data", []):
            if g.get("status") != "pre" or g["id"] in seen:
                continue
            try:
                utc_dt = datetime.strptime(g["date"][:19], "%Y-%m-%dT%H:%M:%S")
                et_dt  = pytz.utc.localize(utc_dt).astimezone(ET)
                if et_dt.strftime("%Y-%m-%d") == game_day:
                    g["et_time"] = et_dt.strftime("%I:%M %p ET")
                    upcoming.append(g)
                    seen.add(g["id"])
            except:
                pass
    return upcoming

def get_team_recent_stats(team_id, n=10):
    r = requests.get(f"{BASE_URL}/team_stats", headers=HEADERS,
                     params={"team_ids[]": team_id, "per_page": n, "seasons[]": CURRENT_SEASON})
    if r.status_code != 200:
        return None
    rows = r.json().get("data", [])
    if not rows:
        return None
    pts_l,efg_l,ts_l,tov_l,pace_l,ppp_l = [],[],[],[],[],[]
    for stat in rows[:n]:
        fga=stat.get("fga") or 0; fgm=stat.get("fgm") or 0
        fg3m=stat.get("fg3m") or 0; fta=stat.get("fta") or 0
        pts=stat.get("pts") or 0; tov=stat.get("turnover") or 0
        oreb=stat.get("oreb") or 0
        pace = fga-oreb+tov+0.44*fta
        if pace <= 0: pace = LG_AVG_PACE
        efg = (fgm+0.5*fg3m)/fga if fga > 0 else 0
        ts_den = 2*(fga+0.44*fta)
        ts  = pts/ts_den if ts_den > 0 else 0
        tov_den = fga+0.44*fta+tov
        tov_pct = tov/tov_den if tov_den > 0 else 0
        ppp = pts/pace
        pts_l.append(pts); efg_l.append(efg); ts_l.append(ts)
        tov_l.append(tov_pct); pace_l.append(pace); ppp_l.append(ppp)
    def avg(lst, n=None):
        s = lst[-n:] if n else lst
        return round(sum(s)/len(s),4) if s else None
    return {"pts_L10":avg(pts_l),"pts_L5":avg(pts_l,5),"efg_L10":avg(efg_l),
            "ts_L10":avg(ts_l),"tov_L10":avg(tov_l),"pace_L10":avg(pace_l),"ppp_L10":avg(ppp_l)}

def get_team_defensive_stats(team_id, n=10):
    r = requests.get(f"{BASE_URL}/team_stats", headers=HEADERS,
                     params={"team_ids[]": team_id, "per_page": n, "seasons[]": CURRENT_SEASON})
    if r.status_code != 200: return None
    rows = r.json().get("data", [])
    if not rows: return None
    game_ids = [x["game"]["id"] for x in rows[:n] if x.get("game")]
    opp_pts,opp_ppp = [],[]
    for gid in game_ids:
        r2 = requests.get(f"{BASE_URL}/team_stats", headers=HEADERS,
                          params={"game_ids[]": gid, "per_page": 10})
        time.sleep(0.1)
        if r2.status_code != 200: continue
        for s in r2.json().get("data", []):
            if s["team"]["id"] != team_id:
                fga=s.get("fga") or 0; pts=s.get("pts") or 0
                tov=s.get("turnover") or 0; oreb=s.get("oreb") or 0
                fta=s.get("fta") or 0
                pace=fga-oreb+tov+0.44*fta
                if pace<=0: pace=LG_AVG_PACE
                opp_pts.append(pts); opp_ppp.append(pts/pace)
    def avg(lst): return round(sum(lst)/len(lst),4) if lst else None
    return {"pts_allowed_L10":avg(opp_pts),"dpp_L10":avg(opp_ppp)}

def get_rest_days(team_id, before_date):
    r = requests.get(f"{BASE_URL}/games", headers=HEADERS,
                     params={"team_ids[]": team_id,"per_page": 5,
                             "seasons[]": CURRENT_SEASON,"end_date": before_date})
    if r.status_code != 200: return 3
    games = [g for g in r.json().get("data",[]) if g.get("status") in ["post","Final","final"]]
    if not games: return 3
    dates   = sorted([g["date"][:10] for g in games], reverse=True)
    last_day= pd.to_datetime(dates[0])
    return min(int((pd.to_datetime(before_date)-last_day).days),7)

def get_odds_for_game(game_id):
    for vendor in PROP_VENDORS:
        r = requests.get(f"{BASE_URL}/odds", headers=HEADERS,
                         params={"game_id": game_id})
        time.sleep(0.1)
        if r.status_code != 200: continue
        odds_list = r.json().get("data",[])
        if not odds_list: continue
        by_vendor = {o["vendor"]: o for o in odds_list}
        sel = None
        for book in PROP_VENDORS:
            if book in by_vendor:
                sel = by_vendor[book]; break
        if not sel and odds_list:
            sel = odds_list[0]
        if sel:
            return {
                "vendor":            sel.get("vendor","?"),
                "home_ml":           sel.get("moneyline_home_odds"),
                "away_ml":           sel.get("moneyline_away_odds"),
                "spread_home":       sel.get("spread_home_value"),
                "spread_home_odds":  sel.get("spread_home_odds"),
                "total":             sel.get("total_value"),
                "over_odds":         sel.get("total_over_odds"),
                "under_odds":        sel.get("total_under_odds"),
            }
    return None

def build_feature_row(home_off, away_off, home_def, away_def,
                       home_rest, away_rest, is_playoff, day_of_season):
    h_ppp  = max(0.9, min(home_off.get("ppp_L10") or LG_AVG_PPP, 1.5))
    a_ppp  = max(0.9, min(away_off.get("ppp_L10") or LG_AVG_PPP, 1.5))
    h_dpp  = max(0.9, min(home_def.get("dpp_L10") or LG_AVG_DPP, 1.5))
    a_dpp  = max(0.9, min(away_def.get("dpp_L10") or LG_AVG_DPP, 1.5))
    h_pace = max(55,  min(home_off.get("pace_L10") or LG_AVG_PACE, 85))
    a_pace = max(55,  min(away_off.get("pace_L10") or LG_AVG_PACE, 85))
    proj_home  = h_ppp * (a_dpp/LG_AVG_DPP) * h_pace
    proj_away  = a_ppp * (h_dpp/LG_AVG_DPP) * a_pace
    return pd.DataFrame([{
        "home_pts_L10": home_off.get("pts_L10") or 82,
        "away_pts_L10": away_off.get("pts_L10") or 82,
        "home_pts_L5":  home_off.get("pts_L5")  or 82,
        "away_pts_L5":  away_off.get("pts_L5")  or 82,
        "home_ppp_L10": h_ppp, "away_ppp_L10": a_ppp,
        "home_efg_L10": home_off.get("efg_L10") or 0.48,
        "away_efg_L10": away_off.get("efg_L10") or 0.48,
        "home_ts_L10":  home_off.get("ts_L10")  or 0.54,
        "away_ts_L10":  away_off.get("ts_L10")  or 0.54,
        "home_tov_L10": home_off.get("tov_L10") or 0.14,
        "away_tov_L10": away_off.get("tov_L10") or 0.14,
        "home_pts_allowed_L10": home_def.get("pts_allowed_L10") or 82,
        "away_pts_allowed_L10": away_def.get("pts_allowed_L10") or 82,
        "home_dpp_L10": h_dpp, "away_dpp_L10": a_dpp,
        "home_pace_L10": h_pace, "away_pace_L10": a_pace,
        "proj_home_pts": round(proj_home,2),
        "proj_away_pts": round(proj_away,2),
        "proj_total":    round(proj_home+proj_away,2),
        "home_rest": home_rest, "away_rest": away_rest,
        "rest_advantage": home_rest-away_rest,
        "home_b2b": int(home_rest==1), "away_b2b": int(away_rest==1),
        "is_playoff": is_playoff,
        "day_of_season": day_of_season,
    }])

# ── RUN ──
upcoming = get_upcoming_games(GAME_DAY)
print(f"Games today: {len(upcoming)}")

season_start = pd.to_datetime(SEASON_START_DATE)
day_of_szn   = max((pd.to_datetime(GAME_DAY) - season_start).days, 0)
results      = []

for game in upcoming:
    home = game["home_team"]; away = game["visitor_team"]
    game_id = game["id"]; tip_off = game.get("et_time","?")
    is_playoff = int(game.get("postseason", False))
    print(f"  {away['abbreviation']} @ {home['abbreviation']} {tip_off}")

    home_off  = get_team_recent_stats(home["id"]); time.sleep(0.1)
    away_off  = get_team_recent_stats(away["id"]); time.sleep(0.1)
    home_def  = get_team_defensive_stats(home["id"]); time.sleep(0.1)
    away_def  = get_team_defensive_stats(away["id"]); time.sleep(0.1)
    home_rest = get_rest_days(home["id"], GAME_DAY)
    away_rest = get_rest_days(away["id"], GAME_DAY); time.sleep(0.1)
    odds      = get_odds_for_game(game_id); time.sleep(0.1)

    if not home_off or not away_off or not home_def or not away_def:
        print(f"    Skipping — not enough 2026 data")
        continue

    X        = build_feature_row(home_off, away_off, home_def, away_def,
                                  home_rest, away_rest, is_playoff, day_of_szn)
    X_scaled = scaler.transform(X)

    proj_home_pts  = model_home.predict(X_scaled)[0]
    proj_away_pts  = model_away.predict(X_scaled)[0]
    proj_total_pts = model_total.predict(X_scaled)[0]
    win_prob_home  = model_win.predict_proba(X_scaled)[0][1]

    results.append({
        "game_date":         GAME_DAY,
        "tip_off_et":        tip_off,
        "season":            CURRENT_SEASON,
        "game_id":           game_id,
        "game":              f"{away['abbreviation']} @ {home['abbreviation']}",
        "home_team":         home["full_name"],
        "away_team":         away["full_name"],
        "is_playoff":        is_playoff,
        "home_rest_days":    home_rest,
        "away_rest_days":    away_rest,
        "proj_home_pts":     round(proj_home_pts, 1),
        "proj_away_pts":     round(proj_away_pts, 1),
        "proj_total":        round(proj_total_pts, 1),
        "proj_home_win_pct": round(win_prob_home * 100, 1),
        "proj_away_win_pct": round((1-win_prob_home) * 100, 1),
        "odds_vendor":       odds["vendor"]      if odds else "",
        "book_home_ml":      odds["home_ml"]     if odds else "",
        "book_away_ml":      odds["away_ml"]     if odds else "",
        "book_total_line":   odds["total"]       if odds else "",
        "book_over_odds":    odds["over_odds"]   if odds else "",
        "book_under_odds":   odds["under_odds"]  if odds else "",
        "book_spread_home":  odds["spread_home"] if odds else "",
        "book_spread_home_odds": odds["spread_home_odds"] if odds else "",
    })

df_proj = pd.DataFrame(results)
print(f"Projections built: {len(df_proj)}")

spreadsheet = get_sheet()
try:
    ws = spreadsheet.worksheet("Projections"); ws.clear()
except:
    ws = spreadsheet.add_worksheet(title="Projections", rows=100, cols=30)
set_with_dataframe(ws, df_proj)
print("Saved to Projections tab.")
print("=== PROJECTIONS COMPLETE ===")
