# EDGE — WNBA Betting Model

Automated WNBA scoring projection model for moneyline, spread, total, and player prop signals. Runs daily via GitHub Actions and serves a mobile-friendly dashboard via GitHub Pages.

---

## Setup (one time, ~15 minutes)

### 1. Add GitHub Secrets

Go to **Settings → Secrets and variables → Actions → New repository secret** and add:

| Secret | Where to find it |
|--------|-----------------|
| `BALLDONTLIE_KEY` | app.balldontlie.io — your API key |
| `GOOGLE_SHEETS_ID` | Your Google Sheet URL: `.../spreadsheets/d/`**THIS_PART**`/edit` |
| `GOOGLE_CREDENTIALS` | Full JSON from Google Service Account (see step 2) |

### 2. Google Service Account

1. Go to [console.cloud.google.com](https://console.cloud.google.com)
2. Select your project → **APIs & Services → Credentials → Create Credentials → Service Account**
3. Name it `wnba-model-bot` → Done
4. Click the service account → **Keys → Add Key → JSON** → download
5. Open the JSON file, copy all contents → paste as `GOOGLE_CREDENTIALS` secret
6. Share your Google Sheet with the service account email (e.g. `wnba-model-bot@your-project.iam.gserviceaccount.com`) → give **Editor** access

### 3. Enable GitHub Pages

Go to **Settings → Pages → Source → GitHub Actions** → Save

### 4. Set your Sheet ID in the dashboard

Open `app/index.html` and replace `YOUR_SHEET_ID_HERE` with your actual Sheet ID.
Commit and push — GitHub Pages will deploy automatically.

### 5. Publish your Google Sheet

In Google Sheets: **File → Share → Publish to web**
Publish each of these tabs as CSV: `Projections`, `Signals`, `Props Signals`, `Player Props`

### 6. Add your iPhone home screen

After GitHub Pages deploys (takes ~2 min after first push):
1. Open Safari on iPhone
2. Go to `https://YOUR_USERNAME.github.io/wnba-model`
3. Tap the share icon → **Add to Home Screen**
4. Name it **EDGE** → Add

---

## How it runs

| Time | Job |
|------|-----|
| Sunday 8:00 AM ET | `cell5_retrain.py` — full model retrain on latest data |
| Daily 12:00 PM ET | `cell6` → `cell7` → `cell8` — projections + all signals |
| Daily 3:00 PM ET  | Odds refresh — re-runs signals with updated book lines |

You can also trigger any run manually: **Actions → select workflow → Run workflow**

---

## File structure

```
wnba-model/
├── .github/workflows/
│   ├── daily.yml        # main scheduler
│   └── pages.yml        # deploys app/ to GitHub Pages
├── scripts/
│   ├── config.py        # shared constants + auth
│   ├── cell5_retrain.py # weekly model retrain
│   ├── cell6_projections.py
│   ├── cell7_signals.py
│   └── cell8_props.py
├── app/
│   └── index.html       # mobile dashboard
└── requirements.txt
```

---

## Backtest results (2022–2025)

| Market | Win rate | ROI | Sig |
|--------|----------|-----|-----|
| Game lines (all) | 63.7% | +18.2% | ✓ |
| 3PM props | 68.3% | +27.7% | ✓ p=0.000 |
| Rebounds | 66.1% | +25.7% | ✓ p=0.000 |
| Assists | 62.2% | +16.2% | ✓ |
| Points | 60.7% | +14.0% | ✓ |

Results use rolling average as proxy for book line. Expect live ROI lower — 8–15% sustained is a strong real-world result. Track CLV from day one.

---

## Manual trigger

Go to **Actions → WNBA Model — Daily Run → Run workflow** to run on demand.
Useful for testing, off-schedule games, or playoff dates.
