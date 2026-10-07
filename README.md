# FinSight — Financial Data Platform

[![CI](https://github.com/rishabhsatishjain7/FinSight/actions/workflows/ci.yml/badge.svg)](https://github.com/rishabhsatishjain7/FinSight/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A modular financial distress-screening platform: XBRL ingestion → ratio/Z-score
transformation → XGBoost + SHAP scoring → Gemini-powered narrative generation →
automated PDF reporting, orchestrated with Airflow.

## Screenshots

![FinSight dashboard — company screen overview](docs/screenshots/dashboard-overview.png)

![FinSight company detail — SHAP drivers, sector outliers, ratio trends](docs/screenshots/company-detail.png)

## Architecture

```
SEC EDGAR (XBRL) ──▶ ingestion ──▶ transformation ──▶ scoring ──▶ narrative ──▶ reporting
                     (multi-tag    (30+ ratios,        (XGBoost +   (Gemini      (structured
                      fallback,     sector Z-scores)     SHAP)        RAG)         PDF)
                      schema
                      evolution)
```

| Layer | Module | Responsibility |
|---|---|---|
| Ingestion | `ingestion/sec_edgar_client.py` | Rate-limited, cached SEC EDGAR XBRL fetch |
| Ingestion | `ingestion/xbrl_parser.py` | **Multi-tag fallback** + schema evolution handling across GAAP taxonomy changes |
| Transformation | `transformation/ratio_engine.py` | 40 financial ratios (liquidity, leverage, profitability, efficiency, cash flow, growth, distress signals) |
| Transformation | `transformation/zscore_benchmark.py` | Sector-relative Z-score standardization |
| Scoring | `scoring/xgboost_model.py` | XGBoost distress classifier + SHAP explainability |
| Scoring | `scoring/train.py` | Feature matrix construction + weak-supervision label generation |
| Narrative | `narrative/context_builder.py` | Structured RAG context assembly (grounds LLM in computed facts) |
| Narrative | `narrative/gemini_client.py` | Gemini REST API integration for analyst narratives |
| Reporting | `reporting/pdf_generator.py` | Per-company + multi-company screen PDF reports |
| Reporting | `reporting/charts.py` | SHAP waterfall + multi-year ratio trend charts embedded in reports |
| Orchestration | `pipeline.py` | Stage functions shared by CLI and Airflow |
| Orchestration | `dags/finsight_pipeline_dag.py` | Airflow DAGs: daily pipeline + weekly model retrain |
| Storage | `database/storage.py` | SQLite (default) or Postgres (via `DATABASE_URL`) persistence between pipeline stages |
| Dashboard | `webapp/main.py` | Read-only FastAPI backend over the storage layer — no duplicated pipeline logic |
| Dashboard | `webapp/static/` | Interactive frontend: company screen table, SHAP drivers, ratio trends, outliers, narrative, PDF download |

Coverage: 14 companies across 4 sectors (technology, retail, energy, industrials),
7 years of 10-K history — configured entirely in `config/companies.yaml`. Adding a
new company requires **no code changes**, only a config entry.

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

export SEC_USER_AGENT="Your Name your-email@example.com"   # required by SEC EDGAR
export GEMINI_API_KEY="your-gemini-api-key"

# Optional: point at Postgres instead of the default local SQLite file.
# No code changes needed either way -- Storage picks the backend from this URL.
export DATABASE_URL="postgresql+psycopg2://user:password@localhost:5432/finsight"
```

## Running

**Offline demo (no external network/API keys needed)** — synthetic but
structurally real XBRL data, exercises the full ratio → Z-score → XGBoost →
SHAP path:

```bash
python3 scripts/run_demo.py
```

**Full pipeline against live SEC EDGAR + Gemini:**

```bash
python3 pipeline.py --stage all
# or individually:
python3 pipeline.py --stage ingest
python3 pipeline.py --stage transform
python3 pipeline.py --stage score
python3 pipeline.py --stage narrate
python3 pipeline.py --stage report
```

**Under Airflow:**

Airflow is **not** in `requirements.txt` — install it in its own,
separate virtual environment (see `requirements-airflow.txt` for why and
the exact commands). Airflow 2.9.3 requires `sqlalchemy<2.0`, which
directly conflicts with this project's `SQLAlchemy==2.0.36`; there's no
version of Airflow 2.x that accepts SQLAlchemy 2.x, so the two can never
share one environment. This also matches how Airflow is normally deployed
in practice — its own environment, pinned against its own official
constraints file, not mixed into an application's requirements.txt.

```bash
# in Airflow's own separate venv (see requirements-airflow.txt):
cp dags/finsight_pipeline_dag.py $AIRFLOW_HOME/dags/
# ensure the FinSight project root is on PYTHONPATH for Airflow workers
airflow dags trigger finsight_daily_pipeline
```

Two DAGs are registered:
- `finsight_daily_pipeline` — weekdays 06:00, full ingest→report chain
- `finsight_weekly_retrain` — Sundays 05:00, refits the XGBoost model on accumulated history

**Dashboard** — run after either of the above (it reads whatever the
pipeline has already written; it doesn't compute anything itself):

```bash
uvicorn webapp.main:app --reload
# open http://localhost:8000
```

A dense, table-first screen of every scored company with risk badges;
click a row to expand SHAP drivers, sector-relative outliers, ratio trend
sparklines, the Gemini narrative (if generated), and a PDF download link
(if `pipeline.py --stage report` has been run). Deep-link to a specific
company with `?company=TICKER`.

## Tests

```bash
pytest tests/ -v
```

93 tests (89 core + 4 Postgres integration) cover multi-tag fallback
resolution, schema-evolution merging,
restatement handling, fiscal-year calendar alignment, currency-unit safety,
ratio null-safety, robust sector Z-score benchmarking (including outlier
resistance), label-leakage guards on the distress model's feature
construction, storage round-trips (SQLite and, when reachable, Postgres —
including a live 12-thread concurrent-write test), XGBoost scoring/SHAP
explanation, PDF/chart generation, Gemini retry/backoff + narrative
caching, and the dashboard's FastAPI backend (risk-band boundaries,
storage-driven company listing, 404 handling). `tests/REGRESSION_CASES.md`
documents 12 real defects found during development (root cause → fix →
guarding test) — this is the "structured regression case" log referenced
in project docs.

## Limitations

- **Live SEC EDGAR / Gemini runs need credentials**: set `SEC_USER_AGENT`
  (required by SEC EDGAR's terms) and `GEMINI_API_KEY` before running
  `pipeline.py` against live data; without them only the offline demo works.
  The sandbox this was built in also blocks `data.sec.gov` and
  `generativelanguage.googleapis.com`, so live end-to-end ingestion and
  narrative generation were verified only up to the demo/tests level here —
  everything runs live once you supply those two env vars in an environment
  with outbound access to those hosts.
- **The demo uses synthetic data**: `scripts/run_demo.py` generates
  structurally-real XBRL payloads (same shape as real SEC responses) rather
  than pulling from EDGAR, and the demo narrative stage is skipped without
  `GEMINI_API_KEY`. PDF generation was verified end-to-end with a synthetic
  context.
- **Airflow lives in its own venv**: `apache-airflow` is deliberately not in
  `requirements.txt` — Airflow 2.9.3 requires `sqlalchemy<2.0`, which
  conflicts with this project's `SQLAlchemy==2.0.36` (see
  `requirements-airflow.txt`). The DAG file's syntax/structure was
  validated; it imports the same `pipeline.py` stage functions as the CLI,
  so DAG correctness is a thin wrapper over already-tested code.
- **4 Postgres integration tests auto-skip** without a reachable Postgres at
  `PG_TEST_URL` (`tests/test_storage.py`). CI provides one via a service
  container, so all 93 tests run there: locally `pytest tests/ -v` reports
  `93 passed` with Postgres up, or `89 passed, 4 skipped` without it.
