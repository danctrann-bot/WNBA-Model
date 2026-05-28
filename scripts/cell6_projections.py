import requests
import pandas as pd
import numpy as np
import pytz
import time
import math
from datetime import datetime, date, timedelta
from gspread_dataframe import set_with_dataframe

# ============================================================
# CELL 6 — DAILY PROJECTIONS + ODDS
# Primary odds source:  BallDontLie /wnba/v1/odds
# Fallback odds source: The Odds API (when BDL returns empty)
# ============================================================

# ── KEYS ──
# These should already be in memory from Cell 1
# If you get NameError run Cell 1 first
ODDS_API_KEY = "b6e855143f7a293a7f4a1a677c696f2f"  # replace with your key
ODDS_API_BASE = "https://api.the-odds-api.com/v4"

ET                = pytz.timezone("America/New_York")
CURRENT_SEASON    = 2026
SEASON_START_DATE = "2026-05-01"
GAME_DAY          = datetime.now(ET).strftime("%Y-%m-%d")

print(f"Eastern date today:  {datetime.now(ET).strftime('%Y-%m-%d %H:%M')} ET")
print(f"Pulling games for:   {GAME_DAY}")

# ── STEP 1 — GET TODAY'S WNBA GAMES FROM BALLDONTLIE ──

def get_games_for_day(game_day_et):
    all_games   = []
    seen        = set()
    game_day_dt = datetime.strptime(game_day_et, "%Y-%m-%d")
    dates_to_check = [
        game_day_et,
        (game_day_dt + timedelta(days=1)).strftime("%Y-%m-%d"),
    ]
    for d in dates_to_check:
        response = requests.get(
            f"{BASE_URL}/games",
            headers=HEADERS,
            params={"dates[]": d, "per_page": 25}
        )
        time.sleep(0.1)
        if response.status_code != 200:
            continue
        for g in response.json().get("data", []):
            if g.get("status") != "pre" or g["id"] in seen:
                continue
            try:
                utc_dt = datetime.strptime(g["date"][:19], "%Y-%m-%dT%H:%M:%S")
                et_dt  = pytz.utc.localize(utc_dt).astimezone(ET)
                if et_dt.strftime("%Y-%m-%d") == game_day_et:
                    g["et_time"] = et_dt.strftime("%I:%M %p ET")
                    all_games.append(g)
                    seen.add(g["id"])
            except:
                pass
    return all_games

todays_games = get_games_for_day(GAME_DAY)

if not todays_games:
    print(f"No upcoming WNBA games on {GAME_DAY} ET.")
    print("Nothing to do — exiting Cell 6.")
else:
    print(f"Found {len(todays_games)} game(s):")
    for g in todays_games:
        print(f"  {g['visitor_team']['abbreviation']} @ "
              f"{g['home_team']['abbreviation']}  {g.get('et_time','?')}")


# ── STEP 2 — ROLLING TEAM STATS ──

def get_team_rolling_stats(team_id, n=10):
    r = requests.get(
        f"{BASE_URL}/team_stats", headers=HEADERS,
        params={"team_ids[]": team_id, "per_page": n,
                "seasons[]": CURRENT_SEASON}
    )
    if r.status_code != 200:
        return None
    rows = r.json().get("data", [])
    if not rows:
        return None

    pts_l, efg_l, ts_l, tov_l, pace_l, ppp_l = [], [], [], [], [], []

    for stat in rows[:n]:
        fga  = stat.get("fga")      or 0
        fgm  = stat.get("fgm")      or 0
        fg3m = stat.get("fg3m")     or 0
        fta  = stat.get("fta")      or 0
        pts  = stat.get("pts")      or 0
        tov  = stat.get("turnover") or 0
        oreb = stat.get("oreb")     or 0

        pace = fga - oreb + tov + 0.44 * fta
        if pace <= 0:
            pace = 68.17

        efg     = (fgm + 0.5 * fg3m) / fga if fga > 0 else 0
        ts_den  = 2 * (fga + 0.44 * fta)
        ts      = pts / ts_den if ts_den > 0 else 0
        tov_den = fga + 0.44 * fta + tov
        tov_pct = tov / tov_den if tov_den > 0 else 0
        ppp     = pts / pace if pace > 0 else 0

        pts_l.append(pts);   efg_l.append(efg)
        ts_l.append(ts);     tov_l.append(tov_pct)
        pace_l.append(pace); ppp_l.append(ppp)

    def avg(lst, k=None):
        s = lst[-k:] if k else lst
        return round(sum(s) / len(s), 4) if s else None

    return {
        "pts_L10":  avg(pts_l),  "pts_L5":  avg(pts_l, 5),
        "efg_L10":  avg(efg_l),  "ts_L10":  avg(ts_l),
        "tov_L10":  avg(tov_l),  "pace_L10":avg(pace_l),
        "ppp_L10":  avg(ppp_l),
    }


def get_team_defensive_stats(team_id, n=10):
    r = requests.get(
        f"{BASE_URL}/team_stats", headers=HEADERS,
        params={"team_ids[]": team_id, "per_page": n,
                "seasons[]": CURRENT_SEASON}
    )
    if r.status_code != 200:
        return None
    rows = r.json().get("data", [])
    if not rows:
        return None

    game_ids = [x["game"]["id"] for x in rows[:n] if x.get("game")]
    opp_pts, opp_ppp = [], []

    for gid in game_ids:
        r2 = requests.get(
            f"{BASE_URL}/team_stats", headers=HEADERS,
            params={"game_ids[]": gid, "per_page": 10}
        )
        time.sleep(0.1)
        if r2.status_code != 200:
            continue
        for s in r2.json().get("data", []):
            if s["team"]["id"] != team_id:
                fga  = s.get("fga")      or 0
                pts  = s.get("pts")      or 0
                tov  = s.get("turnover") or 0
                oreb = s.get("oreb")     or 0
                fta  = s.get("fta")      or 0
                pace = fga - oreb + tov + 0.44 * fta
                if pace <= 0:
                    pace = 68.17
                opp_pts.append(pts)
                opp_ppp.append(pts / pace)

    def avg(lst):
        return round(sum(lst) / len(lst), 4) if lst else None

    return {
        "pts_allowed_L10": avg(opp_pts),
        "dpp_L10":         avg(opp_ppp),
    }


def get_rest_days(team_id, before_date):
    r = requests.get(
        f"{BASE_URL}/games", headers=HEADERS,
        params={"team_ids[]": team_id, "per_page": 5,
                "seasons[]": CURRENT_SEASON, "end_date": before_date}
    )
    if r.status_code != 200:
        return 3
    games = [g for g in r.json().get("data", [])
             if g.get("status") in ["post", "Final", "final"]]
    if not games:
        return 3
    dates   = sorted([g["date"][:10] for g in games], reverse=True)
    last_day= pd.to_datetime(dates[0])
    return min(int((pd.to_datetime(before_date) - last_day).days), 7)


# ── STEP 3 — ODDS FETCHING ──
# Primary: BallDontLie
# Fallback: The Odds API

def get_bdl_odds(game_id):
    """Pull game lines from BallDontLie."""
    r = requests.get(
        f"{BASE_URL}/odds", headers=HEADERS,
        params={"game_id": game_id}
    )
    time.sleep(0.1)
    if r.status_code != 200:
        return None
    odds_list = r.json().get("data", [])
    if not odds_list:
        return None

    # Prefer FanDuel then DraftKings then whatever's available
    vendors  = ["fanduel","draftkings","betmgm","caesars","pinnacle"]
    by_vendor= {o["vendor"]: o for o in odds_list}
    sel      = None
    for v in vendors:
        if v in by_vendor:
            sel = by_vendor[v]; break
    if not sel:
        sel = odds_list[0]

    return {
        "source":           f"BDL/{sel.get('vendor','?')}",
        "home_ml":          sel.get("moneyline_home_odds"),
        "away_ml":          sel.get("moneyline_away_odds"),
        "spread_home":      sel.get("spread_home_value"),
        "spread_home_odds": sel.get("spread_home_odds"),
        "total":            sel.get("total_value"),
        "over_odds":        sel.get("total_over_odds"),
        "under_odds":       sel.get("total_under_odds"),
    }


def get_odds_api_lines(home_abbr, away_abbr):
    """
    Pull WNBA game lines from The Odds API.
    Matches by team abbreviation against team names in the response.
    """
    # WNBA sport key on The Odds API
    sport    = "basketball_wnba"
    markets  = "h2h,spreads,totals"
    regions  = "us"

    r = requests.get(
        f"{ODDS_API_BASE}/sports/{sport}/odds",
        params={
            "apiKey":    ODDS_API_KEY,
            "regions":   regions,
            "markets":   markets,
            "oddsFormat":"american",
        }
    )
    time.sleep(0.2)

    if r.status_code != 200:
        print(f"    Odds API error: {r.status_code} — {r.text[:100]}")
        return None

    games = r.json()
    if not games:
        return None

    # Match game by team names — Odds API uses full city names
    # Map common abbreviations to name fragments for matching
    abbr_map = {
        "LV":  "las vegas", "DAL": "dallas",   "NY":  "new york",
        "CHI": "chicago",   "ATL": "atlanta",  "IND": "indiana",
        "SEA": "seattle",   "PHX": "phoenix",  "CON": "connecticut",
        "MIN": "minnesota", "WAS": "washington","GS": "golden state",
        "LA":  "los angeles",
    }

    home_frag = abbr_map.get(home_abbr, home_abbr.lower())
    away_frag = abbr_map.get(away_abbr, away_abbr.lower())

    matched = None
    for g in games:
        ht = g.get("home_team","").lower()
        at = g.get("away_team","").lower()
        if home_frag in ht and away_frag in at:
            matched = g; break
        if away_frag in ht and home_frag in at:
            # Teams are swapped in Odds API response
            matched = g; break

    if not matched:
        print(f"    Odds API: no match for {away_abbr} @ {home_abbr}")
        return None

    # Extract best available lines (prefer DraftKings then FanDuel)
    bookmakers   = {b["key"]: b for b in matched.get("bookmakers", [])}
    pref_books   = ["draftkings","fanduel","betmgm","caesars","bovada"]
    sel_book     = None
    for bk in pref_books:
        if bk in bookmakers:
            sel_book = bookmakers[bk]; break
    if not sel_book and bookmakers:
        sel_book = list(bookmakers.values())[0]
    if not sel_book:
        return None

    markets_data = {m["key"]: m for m in sel_book.get("markets", [])}

    result = {"source": f"OddsAPI/{sel_book['key']}"}

    # Determine home/away alignment
    home_is_home = home_frag in matched.get("home_team","").lower()

    # ── h2h (moneyline) ──
    if "h2h" in markets_data:
        outcomes = {o["name"].lower(): o["price"]
                    for o in markets_data["h2h"]["outcomes"]}
        for name, price in outcomes.items():
            if home_frag in name:
                result["home_ml"] = price
            elif away_frag in name:
                result["away_ml"] = price

    # ── spreads ──
    if "spreads" in markets_data:
        for o in markets_data["spreads"]["outcomes"]:
            name  = o["name"].lower()
            point = o.get("point", 0)
            price = o["price"]
            if home_frag in name:
                result["spread_home"]      = point
                result["spread_home_odds"] = price
            elif away_frag in name:
                result["spread_away"]      = point
                result["spread_away_odds"] = price

    # ── totals ──
    if "totals" in markets_data:
        for o in markets_data["totals"]["outcomes"]:
            side  = o["name"].lower()
            point = o.get("point", 0)
            price = o["price"]
            if "over" in side:
                result["total"]     = point
                result["over_odds"] = price
            elif "under" in side:
                result["under_odds"] = price

    return result


def get_game_odds(game_id, home_abbr, away_abbr):
    """
    Try BallDontLie first. If empty, fall back to The Odds API.
    Returns standardized odds dict or None.
    """
    # Primary: BallDontLie
    odds = get_bdl_odds(game_id)
    if odds:
        print(f"    Lines: {odds['source']}")
        return odds

    print(f"    BDL lines empty — trying The Odds API...")

    # Fallback: The Odds API
    odds = get_odds_api_lines(home_abbr, away_abbr)
    if odds:
        print(f"    Lines: {odds['source']}")
        return odds

    print(f"    No lines available yet from either source")
    return None


# ── STEP 4 — BUILD FEATURE ROW + PROJECT ──

LG_AVG_DPP  = 1.1999
LG_AVG_PPP  = 1.2131
LG_AVG_PACE = 68.17

FEATURE_COLS = [
    "home_pts_L10","away_pts_L10","home_pts_L5","away_pts_L5",
    "home_ppp_L10","away_ppp_L10","home_efg_L10","away_efg_L10",
    "home_ts_L10","away_ts_L10","home_tov_L10","away_tov_L10",
    "home_pts_allowed_L10","away_pts_allowed_L10",
    "home_dpp_L10","away_dpp_L10","home_pace_L10","away_pace_L10",
    "proj_home_pts","proj_away_pts","proj_total",
    "home_rest","away_rest","rest_advantage",
    "home_b2b","away_b2b","is_playoff","day_of_season",
]

def build_feature_row(home_off, away_off, home_def, away_def,
                       home_rest, away_rest, is_playoff, day_of_season):
    h_ppp  = max(0.9, min(home_off.get("ppp_L10")  or LG_AVG_PPP,  1.5))
    a_ppp  = max(0.9, min(away_off.get("ppp_L10")  or LG_AVG_PPP,  1.5))
    h_dpp  = max(0.9, min(home_def.get("dpp_L10")  or LG_AVG_DPP,  1.5))
    a_dpp  = max(0.9, min(away_def.get("dpp_L10")  or LG_AVG_DPP,  1.5))
    h_pace = max(55,  min(home_off.get("pace_L10") or LG_AVG_PACE, 85))
    a_pace = max(55,  min(away_off.get("pace_L10") or LG_AVG_PACE, 85))

    proj_home = h_ppp * (a_dpp / LG_AVG_DPP) * h_pace
    proj_away = a_ppp * (h_dpp / LG_AVG_DPP) * a_pace

    return pd.DataFrame([{
        "home_pts_L10":        home_off.get("pts_L10")          or 82,
        "away_pts_L10":        away_off.get("pts_L10")          or 82,
        "home_pts_L5":         home_off.get("pts_L5")           or 82,
        "away_pts_L5":         away_off.get("pts_L5")           or 82,
        "home_ppp_L10":        h_ppp,
        "away_ppp_L10":        a_ppp,
        "home_efg_L10":        home_off.get("efg_L10")          or 0.48,
        "away_efg_L10":        away_off.get("efg_L10")          or 0.48,
        "home_ts_L10":         home_off.get("ts_L10")           or 0.54,
        "away_ts_L10":         away_off.get("ts_L10")           or 0.54,
        "home_tov_L10":        home_off.get("tov_L10")          or 0.14,
        "away_tov_L10":        away_off.get("tov_L10")          or 0.14,
        "home_pts_allowed_L10":home_def.get("pts_allowed_L10")  or 82,
        "away_pts_allowed_L10":away_def.get("pts_allowed_L10")  or 82,
        "home_dpp_L10":        h_dpp,
        "away_dpp_L10":        a_dpp,
        "home_pace_L10":       h_pace,
        "away_pace_L10":       a_pace,
        "proj_home_pts":       round(proj_home, 2),
        "proj_away_pts":       round(proj_away, 2),
        "proj_total":          round(proj_home + proj_away, 2),
        "home_rest":           home_rest,
        "away_rest":           away_rest,
        "rest_advantage":      home_rest - away_rest,
        "home_b2b":            int(home_rest == 1),
        "away_b2b":            int(away_rest == 1),
        "is_playoff":          is_playoff,
        "day_of_season":       day_of_season,
    }])


# ── STEP 5 — RUN PROJECTIONS ──

season_start = pd.to_datetime(SEASON_START_DATE)
day_of_szn   = max((pd.to_datetime(GAME_DAY) - season_start).days, 0)

results = []

for game in todays_games:
    home       = game["home_team"]
    away       = game["visitor_team"]
    game_id    = game["id"]
    tip_off    = game.get("et_time", "?")
    is_playoff = int(game.get("postseason", False))

    print(f"\n{away['abbreviation']} @ {home['abbreviation']}  {tip_off}")

    # Pull rolling stats
    home_off  = get_team_rolling_stats(home["id"]);    time.sleep(0.1)
    away_off  = get_team_rolling_stats(away["id"]);    time.sleep(0.1)
    home_def  = get_team_defensive_stats(home["id"]);  time.sleep(0.15)
    away_def  = get_team_defensive_stats(away["id"]);  time.sleep(0.15)
    home_rest = get_rest_days(home["id"], GAME_DAY);   time.sleep(0.1)
    away_rest = get_rest_days(away["id"], GAME_DAY);   time.sleep(0.1)

    if not home_off or not away_off or not home_def or not away_def:
        print(f"  Skipping — not enough 2026 data yet")
        continue

    # Pull odds — BDL first, Odds API fallback
    odds = get_game_odds(
        game_id,
        home["abbreviation"],
        away["abbreviation"]
    )
    time.sleep(0.1)

    # Build features and project
    X        = build_feature_row(home_off, away_off, home_def, away_def,
                                  home_rest, away_rest, is_playoff, day_of_szn)
    X_scaled = scaler_final.transform(X)

    proj_home_pts  = model_home_final.predict(X_scaled)[0]
    proj_away_pts  = model_away_final.predict(X_scaled)[0]
    proj_total_pts = model_total_final.predict(X_scaled)[0]
    win_prob_home  = model_win_final.predict_proba(X_scaled)[0][1]

    print(f"  Proj: {away['abbreviation']} {proj_away_pts:.1f}  "
          f"{home['abbreviation']} {proj_home_pts:.1f}  "
          f"Total {proj_total_pts:.1f}  "
          f"Home win {win_prob_home:.1%}")

    if odds:
        print(f"  ML: {odds.get('away_ml','?')} / {odds.get('home_ml','?')}  "
              f"Total: {odds.get('total','?')}  "
              f"Spread: {odds.get('spread_home','?')}")
    else:
        print(f"  No lines yet — projections saved, re-run for signals")

    results.append({
        "game_date":             GAME_DAY,
        "tip_off_et":            tip_off,
        "season":                CURRENT_SEASON,
        "game_id":               game_id,
        "game":                  f"{away['abbreviation']} @ {home['abbreviation']}",
        "home_team":             home["full_name"],
        "away_team":             away["full_name"],
        "is_playoff":            is_playoff,
        "home_rest_days":        home_rest,
        "away_rest_days":        away_rest,
        "proj_home_pts":         round(proj_home_pts,  1),
        "proj_away_pts":         round(proj_away_pts,  1),
        "proj_total":            round(proj_total_pts, 1),
        "proj_home_win_pct":     round(win_prob_home * 100, 1),
        "proj_away_win_pct":     round((1 - win_prob_home) * 100, 1),
        "odds_source":           odds["source"]           if odds else "",
        "book_home_ml":          odds.get("home_ml","")  if odds else "",
        "book_away_ml":          odds.get("away_ml","")  if odds else "",
        "book_total_line":       odds.get("total","")    if odds else "",
        "book_over_odds":        odds.get("over_odds","")if odds else "",
        "book_under_odds":       odds.get("under_odds","")if odds else "",
        "book_spread_home":      odds.get("spread_home","")    if odds else "",
        "book_spread_home_odds": odds.get("spread_home_odds","")if odds else "",
    })


# ── STEP 6 — SAVE TO SHEETS ──

if results:
    df_proj = pd.DataFrame(results)
    print(f"\nProjections built: {len(df_proj)} game(s)")

    try:
        proj_sheet = spreadsheet.worksheet("Projections")
        proj_sheet.clear()
    except:
        proj_sheet = spreadsheet.add_worksheet(
            title="Projections", rows=100, cols=30)
    set_with_dataframe(proj_sheet, df_proj)
    print("Saved to Projections tab.")
    print("Run Cell 7 for game signals, Cell 8 for prop signals.")
else:
    print("\nNo projections generated.")
    print("Check that 2026 team stats are available in BallDontLie.")
