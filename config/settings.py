"""
FinSight — central settings.

All environment-dependent configuration lives here so every layer
(ingestion, transformation, scoring, narrative, reporting) imports
from one place instead of reading os.environ ad hoc.
"""
from __future__ import annotations

import os
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = BASE_DIR / "config"
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"
REPORTS_DIR = BASE_DIR / "reports" / "output"
LOG_DIR = BASE_DIR / "logs"
DB_PATH = DATA_DIR / "finsight.db"

for d in (DATA_DIR, CACHE_DIR, REPORTS_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# SEC EDGAR
# ---------------------------------------------------------------------------
# SEC requires a descriptive User-Agent with contact info on every request,
# or it will start throttling / blocking the caller. Set this via env var
# before running ingestion against the live API.
SEC_USER_AGENT = os.environ.get(
    "SEC_USER_AGENT", "FinSight Research Tool contact@example.com"
)
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
SEC_COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
SEC_COMPANYCONCEPT_URL = (
    "https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/{taxonomy}/{tag}.json"
)
# SEC rate limit is 10 req/s; we stay well under it.
SEC_REQUEST_DELAY_SECONDS = float(os.environ.get("SEC_REQUEST_DELAY_SECONDS", "0.15"))

# ---------------------------------------------------------------------------
# Gemini (narrative / RAG engine)
# ---------------------------------------------------------------------------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
GEMINI_API_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
# Production (real Airflow, concurrent task writes) should point this at
# Postgres via env var, e.g.:
#   DATABASE_URL=postgresql+psycopg2://finsight:password@localhost:5432/finsight
# SQLite remains the zero-setup default for local dev/demo runs -- see
# database/storage.py for why SQLite alone isn't safe under concurrent
# Airflow workers (RC-009 in tests/REGRESSION_CASES.md).
DATABASE_URL = os.environ.get("DATABASE_URL", f"sqlite:///{DB_PATH}")

# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
MODEL_DIR = BASE_DIR / "scoring" / "artifacts"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
XGB_MODEL_PATH = MODEL_DIR / "distress_model.json"
DISTRESS_LABEL_THRESHOLD = 0.5  # probability above which a company is flagged

# ---------------------------------------------------------------------------
# Company / sector universe
# ---------------------------------------------------------------------------
def load_universe() -> dict:
    with open(CONFIG_DIR / "companies.yaml", "r") as f:
        return yaml.safe_load(f)


def flat_company_list() -> list[dict]:
    """Flatten sectors -> list of {ticker, cik, name, sector}."""
    universe = load_universe()
    companies = []
    for sector, payload in universe["sectors"].items():
        for c in payload["companies"]:
            companies.append({**c, "sector": sector})
    return companies


LOOKBACK_YEARS = load_universe().get("lookback_years", 7)
