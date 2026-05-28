# ============================================================
# config.py — shared constants for all scripts
# All secrets come from GitHub Actions environment variables
# Set these in: repo Settings → Secrets → Actions
# ============================================================

import os
import json
import gspread
from google.oauth2.service_account import Credentials

# ── API ──
API_KEY  = os.environ["BALLDONTLIE_KEY"]
BASE_URL = "https://api.balldontlie.io/wnba/v1"
HEADERS  = {"Authorization": API_KEY}

# ── GOOGLE SHEETS ──
SHEET_ID = os.environ["GOOGLE_SHEETS_ID"]

def get_sheet():
    """Returns an authenticated gspread spreadsheet object."""
    creds_json = os.environ["GOOGLE_CREDENTIALS"]
    creds_dict = json.loads(creds_json)
    scopes = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive",
    ]
    creds       = Credentials.from_service_account_info(creds_dict, scopes=scopes)
    client      = gspread.authorize(creds)
    spreadsheet = client.open_by_key(SHEET_ID)
    return spreadsheet

# ── MODEL CONSTANTS ──
CURRENT_SEASON    = 2026
SEASON_START_DATE = "2026-05-01"

LG_AVG_DPP  = 1.1999
LG_AVG_PPP  = 1.2131
LG_AVG_PACE = 68.17

PROP_VENDORS = ["fanduel", "draftkings", "betmgm", "caesars", "pinnacle"]

EDGE_THRESHOLDS_GAME = {
    "ml":     3.0,
    "total":  3.0,
    "spread": 2.5,
}
EDGE_THRESHOLDS_PROP = {
    "pts":  1.5,
    "reb":  0.75,
    "ast":  0.5,
    "fg3m": 0.5,
}

print("Config loaded.")
