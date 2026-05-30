# ============================================================
# cell7_signals.py — game betting signals
# Reads Projections tab, calculates ML/spread/total signals
# WITH full EV calculation. Saves to Signals tab.
# ============================================================

import pandas as pd
import numpy as np
import math
from config import get_sheet
from gspread_dataframe import set_with_dataframe

print("=== SIGNALS STARTING ===")

spreadsheet = get_sheet()

try:
    ws   = spreadsheet.worksheet("Projections")
    data = ws.get_all_records()
    df   = pd.DataFrame(data)
except Exception as e:
    print(f"Could not load Projections: {e}"); exit()

if df.empty:
    print("No projections found — run cell6 first."); exit()

print(f"Loaded {len(df)} projections")

# ── HELPERS ──

def american_to_prob(odds):
    try:
        odds = float(odds)
        if odds == 0: return None
        return (-odds / (-odds + 100)) if odds < 0 else (100 / (odds + 100))
    except:
        return None

def devig(p1, p2):
    if not p1 or not p2: return None, None
    t = p1 + p2
    return p1/t, p2/t

def calc_ev(prob, odds, stake=100):
    try:
        odds    = float(odds)
        win_amt = stake*(100/-odds) if odds < 0 else stake*(odds/100)
        return round((prob*win_amt) - ((1-prob)*stake), 2)
    except:
        return None

def soft_spread(margin, scale=8):
    return 1/(1+math.exp(-margin/scale))

def safe_float(val, default=0):
    try: return float(val)
    except: return default

# ── GENERATE SIGNALS ──
rows = []

for _, r in df.iterrows():
    proj_home   = safe_float(r.get("proj_home_pts"))
    proj_away   = safe_float(r.get("proj_away_pts"))
    proj_total  = safe_float(r.get("proj_total"))
    proj_margin = proj_home - proj_away
    proj_win_home = safe_float(r.get("proj_home_win_pct"), 50) / 100

    home_ml     = r.get("book_home_ml","")
    away_ml     = r.get("book_away_ml","")
    total       = r.get("book_total_line","")
    o_odds      = r.get("book_over_odds","")
    u_odds      = r.get("book_under_odds","")
    spread      = r.get("book_spread_home","")
    spread_odds = r.get("book_spread_home_odds","")

    game_str  = str(r.get("game",""))
    home_name = game_str.split(" @ ")[-1] if " @ " in game_str else ""
    away_name = game_str.split(" @ ")[0]  if " @ " in game_str else ""

    row = {
        "game_date":        r.get("game_date",""),
        "tip_off_et":       r.get("tip_off_et",""),
        "game":             game_str,
        "game_id":          r.get("game_id",""),
        "proj_home_pts":    proj_home,
        "proj_away_pts":    proj_away,
        "proj_total":       proj_total,
        "proj_home_win_pct":round(proj_win_home*100,1),
        "book_home_ml":     home_ml,
        "book_away_ml":     away_ml,
        "book_total_line":  total,
        "book_over_odds":   o_odds,
        "book_under_odds":  u_odds,
        "book_spread_home": spread,
        "book_spread_home_odds": spread_odds,
        "ml_signal":        "No edge",
        "ml_ev":            0,
        "ml_edge_pct":      0,
        "ml_bet_side":      "",
        "ml_bet_odds":      "",
        "total_signal":     "No edge",
        "total_ev":         0,
        "total_edge_pct":   0,
        "total_diff":       0,
        "spread_signal":    "No edge",
        "spread_ev":        0,
        "spread_edge_pct":  0,
        "spread_diff":      0,
        "ACTION":           "PASS",
    }

    actions = []

    # ── MONEYLINE ──
    if home_ml and away_ml and str(home_ml).strip() and str(away_ml).strip():
        raw_home = american_to_prob(home_ml)
        raw_away = american_to_prob(away_ml)
        fair_home, fair_away = devig(raw_home, raw_away)
        if fair_home and fair_away:
            home_edge = round((proj_win_home - fair_home)*100, 1)
            away_edge = round(((1-proj_win_home) - fair_away)*100, 1)
            home_ev   = calc_ev(proj_win_home,   home_ml) or 0
            away_ev   = calc_ev(1-proj_win_home, away_ml) or 0

            row["home_edge_pct"]   = home_edge
            row["away_edge_pct"]   = away_edge
            row["home_ev_per_100"] = home_ev
            row["away_ev_per_100"] = away_ev

            if home_edge >= 3:
                row["ml_signal"]   = f"BET {home_name} ML {home_ml}"
                row["ml_ev"]       = home_ev
                row["ml_edge_pct"] = home_edge
                row["ml_bet_side"] = home_name
                row["ml_bet_odds"] = str(home_ml)
                actions.append(f"ML {home_name} (EV {home_ev:+.2f})")
            elif away_edge >= 3:
                row["ml_signal"]   = f"BET {away_name} ML {away_ml}"
                row["ml_ev"]       = away_ev
                row["ml_edge_pct"] = away_edge
                row["ml_bet_side"] = away_name
                row["ml_bet_odds"] = str(away_ml)
                actions.append(f"ML {away_name} (EV {away_ev:+.2f})")

    # ── TOTAL ──
    if total and o_odds and u_odds and str(total).strip():
        try:
            book_line  = float(total)
            diff       = proj_total - book_line
            over_prob  = 1/(1+math.exp(-diff/6))
            under_prob = 1-over_prob
            raw_o = american_to_prob(o_odds)
            raw_u = american_to_prob(u_odds)
            fair_o, fair_u = devig(raw_o, raw_u)
            if fair_o and fair_u:
                over_edge  = round((over_prob  - fair_o)*100, 1)
                under_edge = round((under_prob - fair_u)*100, 1)
                over_ev    = calc_ev(over_prob,  o_odds) or 0
                under_ev   = calc_ev(under_prob, u_odds) or 0

                row["total_diff"] = round(diff, 1)

                if diff >= 3 and over_edge >= 3:
                    row["total_signal"]   = f"OVER {total} @ {o_odds}"
                    row["total_ev"]       = over_ev
                    row["total_edge_pct"] = over_edge
                    actions.append(f"OVER {total} (EV {over_ev:+.2f})")
                elif diff <= -3 and under_edge >= 3:
                    row["total_signal"]   = f"UNDER {total} @ {u_odds}"
                    row["total_ev"]       = under_ev
                    row["total_edge_pct"] = under_edge
                    actions.append(f"UNDER {total} (EV {under_ev:+.2f})")
        except:
            pass

    # ── SPREAD ──
    if spread and spread_odds and str(spread).strip():
        try:
            book_spread   = float(spread)
            diff          = proj_margin - book_spread
            home_ats_prob = soft_spread(diff)
            away_ats_prob = 1 - home_ats_prob
            home_ats_ev   = calc_ev(home_ats_prob, spread_odds) or 0
            away_spread   = -book_spread
            # estimate away odds as mirror
            try:
                away_sp_odds = float(spread_odds)
                away_sp_odds = -away_sp_odds if away_sp_odds < 0 else -away_sp_odds
            except:
                away_sp_odds = -110
            away_ats_ev   = calc_ev(away_ats_prob, away_sp_odds) or 0
            spread_edge   = round((home_ats_prob - 0.5238)*100, 1)
            away_sp_edge  = round((away_ats_prob - 0.5238)*100, 1)

            row["spread_diff"] = round(diff, 1)

            if diff >= 2.5 and spread_edge >= 2.5:
                row["spread_signal"]   = f"BET {home_name} {book_spread:+.1f} @ {spread_odds}"
                row["spread_ev"]       = home_ats_ev
                row["spread_edge_pct"] = spread_edge
                actions.append(f"SPREAD {home_name} {book_spread:+.1f} (EV {home_ats_ev:+.2f})")
            elif diff <= -2.5 and away_sp_edge >= 2.5:
                row["spread_signal"]   = f"BET {away_name} {away_spread:+.1f}"
                row["spread_ev"]       = away_ats_ev
                row["spread_edge_pct"] = away_sp_edge
                actions.append(f"SPREAD {away_name} {away_spread:+.1f} (EV {away_ats_ev:+.2f})")
        except:
            pass

    if actions:
        row["ACTION"] = " | ".join(actions)

    rows.append(row)

df_signals = pd.DataFrame(rows)

# ── SUMMARY ──
def count_signals(df, col):
    if col not in df.columns: return 0
    return int((df[col].notna() & (df[col] != "No edge") & (df[col] != "")).sum())

ml_count  = count_signals(df_signals, "ml_signal")
tot_count = count_signals(df_signals, "total_signal")
spd_count = count_signals(df_signals, "spread_signal")
print(f"Signals: {ml_count + tot_count + spd_count} total "
      f"({ml_count} ML, {tot_count} totals, {spd_count} spread)")

if ml_count + tot_count + spd_count > 0:
    flagged = df_signals[df_signals["ACTION"] != "PASS"]
    for _, r in flagged.iterrows():
        print(f"  {r['game']}  →  {r['ACTION']}")

try:
    ws = spreadsheet.worksheet("Signals"); ws.clear()
except:
    ws = spreadsheet.add_worksheet(title="Signals", rows=100, cols=50)
set_with_dataframe(ws, df_signals)
print("Saved to Signals tab.")
print("=== SIGNALS COMPLETE ===")
