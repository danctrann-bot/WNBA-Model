# ============================================================
# cell7_signals.py
# Reads Projections tab, calculates betting signals for
# ML / Spread / Total, saves to Signals tab.
#
# Runs: daily 12pm ET (after cell6) via GitHub Actions
# ============================================================

import pandas as pd
import numpy as np
import math
from config import get_sheet
from gspread_dataframe import set_with_dataframe

print("=== SIGNALS STARTING ===")

spreadsheet = get_sheet()

# ── LOAD PROJECTIONS ──
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
        return (-odds/(-odds+100)) if odds < 0 else (100/(odds+100))
    except:
        return None

def devig(p1, p2):
    """Remove vig from two raw probabilities."""
    if not p1 or not p2: return None, None
    total = p1 + p2
    return p1/total, p2/total

def calc_ev(prob, odds, stake=100):
    try:
        odds    = float(odds)
        win_amt = stake*(100/-odds) if odds < 0 else stake*(odds/100)
        return round((prob*win_amt) - ((1-prob)*stake), 2)
    except:
        return None

def soft_spread(margin, scale=8):
    """Convert projected margin to ATS probability."""
    return 1/(1+math.exp(-margin/scale))

# ── GENERATE SIGNALS ──
rows = []

for _, r in df.iterrows():
    proj_home = float(r.get("proj_home_pts") or 0)
    proj_away = float(r.get("proj_away_pts") or 0)
    proj_total= float(r.get("proj_total") or 0)
    proj_margin = proj_home - proj_away

    home_ml = r.get("book_home_ml") or ""
    away_ml = r.get("book_away_ml") or ""
    total   = r.get("book_total_line") or ""
    o_odds  = r.get("book_over_odds") or ""
    u_odds  = r.get("book_under_odds") or ""
    spread  = r.get("book_spread_home") or ""
    spread_odds = r.get("book_spread_home_odds") or ""
    proj_win_home = float(r.get("proj_home_win_pct") or 50) / 100

    row = {
        "game_date":  r.get("game_date",""),
        "tip_off_et": r.get("tip_off_et",""),
        "game":       r.get("game",""),
        "game_id":    r.get("game_id",""),
        "proj_home":  proj_home,
        "proj_away":  proj_away,
        "proj_total": proj_total,
        "book_home_ml": home_ml,
        "book_away_ml": away_ml,
        "book_total_line": total,
        "book_over_odds":  o_odds,
        "book_under_odds": u_odds,
        "book_spread_home": spread,
    }

    # ── MONEYLINE ──
    if home_ml and away_ml:
        raw_home = american_to_prob(home_ml)
        raw_away = american_to_prob(away_ml)
        fair_home, fair_away = devig(raw_home, raw_away)
        if fair_home and fair_away:
            home_edge = round((proj_win_home - fair_home)*100, 1)
            away_edge = round(((1-proj_win_home) - fair_away)*100, 1)
            home_ev   = calc_ev(proj_win_home, home_ml)
            away_ev   = calc_ev(1-proj_win_home, away_ml)
            row["home_win_prob"] = round(proj_win_home*100,1)
            row["fair_home_prob"]= round(fair_home*100,1)
            row["home_edge_pct"] = home_edge
            row["away_edge_pct"] = away_edge
            row["home_ev_per_100"] = home_ev
            row["away_ev_per_100"] = away_ev

            home_name = r.get("game","").split(" @ ")[-1]
            away_name = r.get("game","").split(" @ ")[0]
            if home_edge >= 3:
                row["ml_signal"] = f"BET {home_name} ML {home_ml} (edge {home_edge:+.1f}%)"
            elif away_edge >= 3:
                row["ml_signal"] = f"BET {away_name} ML {away_ml} (edge {away_edge:+.1f}%)"
            else:
                row["ml_signal"] = "No edge"

    # ── TOTAL ──
    if total and o_odds and u_odds:
        try:
            book_line = float(total)
            diff      = proj_total - book_line
            over_prob = 1/(1+math.exp(-diff/6))
            under_prob= 1-over_prob
            raw_o = american_to_prob(o_odds)
            raw_u = american_to_prob(u_odds)
            fair_o, fair_u = devig(raw_o, raw_u)
            if fair_o and fair_u:
                over_edge  = round((over_prob - fair_o)*100,1)
                under_edge = round((under_prob - fair_u)*100,1)
                over_ev    = calc_ev(over_prob,  o_odds)
                under_ev   = calc_ev(under_prob, u_odds)
                row["total_diff"]       = round(diff,1)
                row["total_over_edge"]  = over_edge
                row["total_under_edge"] = under_edge
                row["total_ev_over"]    = over_ev
                row["total_ev_under"]   = under_ev

                if diff >= 3 and over_edge >= 3:
                    row["total_signal"] = f"OVER {total} {o_odds} (proj {proj_total:.1f}, diff {diff:+.1f})"
                elif diff <= -3 and under_edge >= 3:
                    row["total_signal"] = f"UNDER {total} {u_odds} (proj {proj_total:.1f}, diff {diff:+.1f})"
                else:
                    row["total_signal"] = "No edge"
        except:
            pass

    # ── SPREAD ──
    if spread and spread_odds:
        try:
            book_spread = float(spread)
            # model projected margin vs book spread
            diff = proj_margin - book_spread
            home_ats_prob = soft_spread(diff)
            away_ats_prob = 1 - home_ats_prob
            home_ats_ev   = calc_ev(home_ats_prob, spread_odds)
            # away spread is implied as -spread at similar odds
            away_spread_val = -book_spread
            away_spread_odds = spread_odds  # approximation
            away_ats_ev = calc_ev(away_ats_prob, away_spread_odds)
            spread_edge = round((home_ats_prob - 0.5238)*100, 1)
            row["spread_diff"]       = round(diff,1)
            row["home_ats_prob"]     = round(home_ats_prob*100,1)
            row["spread_edge_pct"]   = spread_edge
            row["spread_ev_per_100"] = home_ats_ev

            game  = r.get("game","")
            home_n= game.split(" @ ")[-1]
            away_n= game.split(" @ ")[0]
            if diff >= 2.5 and spread_edge >= 2.5:
                row["spread_signal"] = f"BET {home_n} {spread:+.1f} {spread_odds} (diff {diff:+.1f})"
            elif diff <= -2.5 and (100-spread_edge) >= 2.5:
                away_sp = f"{-book_spread:+.1f}"
                row["spread_signal"] = f"BET {away_n} {away_sp} (diff {diff:+.1f})"
            else:
                row["spread_signal"] = "No edge"
        except:
            pass

    rows.append(row)

df_signals = pd.DataFrame(rows)

# Summary
ml_bets  = df_signals[df_signals["ml_signal"].notna() & (df_signals["ml_signal"] != "No edge")]["ml_signal"] if "ml_signal" in df_signals.columns else pd.Series()
tot_bets = df_signals[df_signals["total_signal"].notna() & (df_signals["total_signal"] != "No edge")]["total_signal"] if "total_signal" in df_signals.columns else pd.Series()
spd_bets = df_signals[df_signals["spread_signal"].notna() & (df_signals["spread_signal"] != "No edge")]["spread_signal"] if "spread_signal" in df_signals.columns else pd.Series()
total_flags = len(ml_bets) + len(tot_bets) + len(spd_bets)
print(f"Signals: {total_flags} total ({len(ml_bets)} ML, {len(tot_bets)} total, {len(spd_bets)} spread)")

try:
    ws = spreadsheet.worksheet("Signals"); ws.clear()
except:
    ws = spreadsheet.add_worksheet(title="Signals", rows=100, cols=40)
set_with_dataframe(ws, df_signals)
print("Saved to Signals tab.")
print("=== SIGNALS COMPLETE ===")
