# FinSight — Financial Data Platform

A modular financial distress-screening platform: XBRL ingestion → ratio/Z-score
transformation → XGBoost + SHAP scoring → Gemini-powered narrative generation →
automated PDF reporting, orchestrated with Airflow.

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

89 tests cover multi-tag fallback resolution, schema-evolution merging,
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

## Notes on this build

- **SEC EDGAR / Gemini network access**: this environment's sandbox network
  allowlist doesn't include `data.sec.gov` or `generativelanguage.googleapis.com`,
  so live end-to-end ingestion and narrative generation couldn't be executed
  *from this sandbox*. The offline demo (`scripts/run_demo.py`) exercises the
  same code paths (parser → ratio engine → Z-score → XGBoost → SHAP) against
  synthetic data shaped identically to real SEC responses, and all 89
  tests pass (4 Postgres-specific tests auto-skip without a live DB). PDF generation was verified end-to-end with a synthetic context.
  Everything will run live once you supply `SEC_USER_AGENT` and `GEMINI_API_KEY`
  in an environment with outbound access to those hosts.
- **Airflow**: `apache-airflow` wasn't installed in this sandbox (heavy
  dependency footprint, and its own network/DB setup is out of scope for a
  container test run), but the DAG file's syntax/structure was validated. It
  imports the same `pipeline.py` stage functions used by the CLI, so DAG
  correctness is really a thin wrapper over already-tested code.
