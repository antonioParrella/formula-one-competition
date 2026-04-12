# F1 Tipping Competition

A fully automated multiplayer F1 tipping competition. Players submit picks via SurveyMars before each race. A Python pipeline scores them against real results from the OpenF1 API and publishes a static leaderboard website to GitHub Pages — no manual work required each race weekend.

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Configuration](#configuration)
- [Running the Pipeline](#running-the-pipeline)
- [Project Structure](#project-structure)
- [Scoring Rules](#scoring-rules)

---

## Prerequisites

- **Python 3.10 or higher** — [Download here](https://www.python.org/downloads/)
- **Git** — [Download here](https://git-scm.com/downloads) (needed for committing changes)
- No other tools are required.

---

## Setup

We provide setup scripts for both platforms. If you have already run the setup scripts, skip to [Configuration](#configuration).

### Quick Start (Automated)

**Windows:**

1. Double-click `setup.bat`
2. Wait for the script to finish — it creates a virtual environment and installs dependencies

**macOS / Linux:**

```bash
cd formula-one-competition  # enter project directory
chmod +x setup.sh           # first time only
./setup.sh
```

### Manual Setup

```bash
# 1. Clone the repository (if not already done)
git clone <repo-url>
cd formula-one-competition

# 2. Create a virtual environment
python -m venv venv

# 3. Activate it
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

# 4. Install dependencies
pip install -r config/requirements.txt
```

---

## Configuration

### 1. SurveyMars API Credentials

The pipeline fetches tips via the SurveyMars API using OAuth2. You need your account ID and secret.

The credentials file is already in the repo at `config/.env`. Edit it with your credentials:

```
SURVEYMARS_ACCOUNT_ID=your_account_id_here
SURVEYMARS_SECRET=your_secret_here
```

You can find these in your SurveyMars account settings under API credentials.

> **Note:** The `.env` file is in `config/` folder (not project root) so it works on both Mac and Windows.

### 2. Survey Link

The survey link displayed on the website is stored in `docs/survey_config.json`. Edit this file to update the link:

```json
{
  "surveyUrl": "https://surveymars.com/q/your-survey-id"
}
```

Get the link from your SurveyMars survey URL or share link.

### 3. GitHub Pages (optional)

To publish the website automatically:

1. Push the repo to GitHub (must be public for free hosting)
2. Go to repo Settings → Pages → Source: **Deploy from branch** → `main` → `/docs`
3. Add `.nojekyll` to `docs/` (already included)
4. The site will be live at `https://{username}.github.io/{repo-name}`

Add GitHub secrets (`SURVEYMARS_ACCOUNT_ID` and `SURVEYMARS_SECRET`) under repo Settings → Secrets → Actions so the CI workflow can use them.

---

## Running the Pipeline

Activate your virtual environment first:

```bash
# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### Full Pipeline (all rounds)

```bash
python scripts/pipeline.py
```

This runs all 5 steps:

1. **Fetch tips** — calls SurveyMars API to get all survey responses
2. **Fetch results** — calls OpenF1 API to get race results for existing rounds
3. **Score rounds** — applies scoring rules to each round
4. **Aggregate standings** — builds season standings from all scored rounds
5. **Build website** — injects data into `docs/index.html`

### Single Round

```bash
python scripts/pipeline.py --round 4
```

Only processes round 4. Useful for re-running a specific round after fixing a bug or updating data.

### Tips Only

```bash
python scripts/pipeline.py --round 5 --tip-only
```

Fetches and saves tips from SurveyMars but skips results fetching, scoring, and site building. Useful for collecting tips before the race when results aren't available yet.

This is the typical workflow during race weekend:

```bash
# Before the race (Sunday or Monday) — just grab tips
python scripts/pipeline.py --round 5 --tip-only

# After the race — get results, score, build site
python scripts/pipeline.py --round 5
```

### Force Re-fetch

```bash
python scripts/pipeline.py --round 3 --force
```

Re-fetches tips even if a file already exists. Useful when survey responses have changed or were corrected.

### Manual Site Rebuild

If you've already scored rounds and just want to regenerate the website:

```bash
python -c "import sys; sys.path.insert(0, 'src'); from build_site import SiteBuilder; SiteBuilder().build_and_save()"
```

### Manual Standings Rebuild

```bash
python -c "import sys; sys.path.insert(0, 'src'); from aggregator import Aggregator; Aggregator().build_and_save()"
```

### Individual Module Usage

Each pipeline module is importable and can be run separately:

```python
import sys
sys.path.insert(0, 'src')

from survey_mars import SurveyMarsClient
from survey_index import SurveyIndex
from tips_parser import TipsParser
from scorer import Scorer
from fetch_results import ResultsFetcher
from aggregator import Aggregator
from build_site import SiteBuilder

# Scoring a specific round
scorer = Scorer(round_num=3)
scorer.score_and_save()

# Fetching results for a specific round (2026 season)
fetcher = ResultsFetcher(round_num=3, year=2026)
fetcher.fetch_and_save()

# Fetching the latest survey index
client = SurveyMarsClient()
client.authenticate()
index = SurveyIndex(client).fetch()
index.print_summary()
```

---

## Data Flow

```
SurveyMars API ──→ data/raw/tips/ ──┐
                                     ├──→ scorer.py ──→ data/processed/ ──→ aggregate.py
OpenF1 API ──────→ data/raw/results/│                                    ↓
                                     └───────────────────────────────── build_site.py
                                                                        ↓
                                                             docs/index.html
                                                                        ↓
                                                            GitHub Pages (live)
```

- `/data/raw/` is **write-once** — never edit these files manually
- `/data/processed/` is **always derived** — delete and regenerate anytime by re-running the pipeline
- `/data/overrides/` holds manual corrections (e.g. late submission penalties) — only created when needed

---

## Scoring Rules

### Main Race

| Result | Condition | Points |
|---|---|---|
| `exact` | Driver picked in correct position | 5 |
| `close` | Driver one position away | 3 |
| `top10` | Driver anywhere else in top 10 | 1 |
| `miss` | Driver not in top 10 | 0 |
| Underdog bonus | Driver outside championship top 10 (not round 1) | ×2 multiplier |

### Sprint (on sprint weekends)

- **5 points** for each correct sprint position (P1, P2, P3)
- Scored separately but added to round total in standings

### DNFs

- **15 points** per tipped driver who actually DNFs
- Driver must have **started** the race (DNS doesn't count)
- **5 picks per season** total, allocatable to any round (0–5 per round)

### Penalties

- **-5 points** per missed practice session (late submission)
- Submission after qualifying: **scores zero**
- Total weekend points **cannot go negative** (floored at 0)

---

## Project Structure

```
├── scripts/
│   └── pipeline.py        # Main entry point — runs all 5 steps
├── src/
│   ├── survey_mars.py     # SurveyMarsClient — OAuth2 + API requests
│   ├── survey_index.py    # SurveyIndex — fetches & indexes surveys by publish date
│   ├── tips_parser.py     # TipsParser — fetches responses, saves raw tips
│   ├── fetch_results.py   # ResultsFetcher — gets OpenF1 race results
│   ├── scorer.py          # Scorer — applies scoring rules per round
│   ├── aggregator.py     # Aggregator — builds season standings
│   ├── build_site.py      # SiteBuilder — injects data into index.html
│   ├── race_utils.py     # clean_race_name(), DRIVER_MAP
│   ├── leaderboard.py    # ResultAggregator — OpenF1 data wrapper
│   └── tips_reader.py    # Reads previously saved tips
├── config/
│   ├── requirements.txt   # Python dependencies
│   ├── .env.example       # Environment variable template
│   ├── .gitignore
│   └── .env               # Local environment variables
├── setup.bat              # Windows setup script
├── setup.sh               # macOS/Linux setup script
├── docs/
│   ├── index.html         # Website — data injected by build_site.py
│   ├── .nojekyll          # Disables Jekyll on GitHub Pages
│   └── survey_config.json # Survey configuration
├── data/
│   ├── raw/               # Write-once source data (tips + results)
│   ├── processed/         # Derived data (scored + standings)
│   └── overrides/         # Manual corrections
├── notebooks/             # Jupyter notebooks for exploration
├── tests/                 # Test files
└── .github/
    └── workflows/
        └── score.yml      # GitHub Actions CI/CD pipeline
```

Each module has a docstring at the top with usage examples. See `ARCHITECTURE.md` for detailed module documentation.
