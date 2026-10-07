# FinSight — Project Overview

A modular financial distress-screening platform: SEC XBRL ingestion → ratio
computation & sector-relative benchmarking → XGBoost + SHAP scoring →
Gemini-powered narrative generation → automated PDF reporting with charts,
orchestrated with Airflow and backed by a dual SQLite/Postgres storage layer.

This document describes the platform **as it currently stands**, including
three hardening passes completed after the initial build. It's meant as a
standing reference to pick the project back up from — architecture, what's
been verified and how, and what's still open.

---

## 1. Architecture

```
SEC EDGAR (XBRL) ──▶ ingestion ──▶ transformation ──▶ scoring ──▶ narrative ──▶ reporting
                     (multi-tag    (40 ratios,          (XGBoost +   (Gemini      (PDF +
                      fallback,     robust sector        SHAP,        RAG,         SHAP/trend
                      schema        Z-scores,             anti-       grounded     charts)
                      evolution,    median/MAD)           leakage     context)
                      restate-                            features)
                      ments)
                                          │
                                          ▼
                            SQLite (dev) / Postgres (prod)
                              storage layer, via DATABASE_URL
                                          │
                                          ▼
                              Airflow: daily pipeline DAG +
                                weekly retrain DAG
```

| Layer | Module | Responsibility |
|---|---|---|
| Ingestion | `ingestion/sec_edgar_client.py` | Rate-limited, cached SEC EDGAR XBRL fetch |
| Ingestion | `ingestion/xbrl_parser.py` | Multi-tag fallback, schema-evolution merging, restatement resolution, fiscal-year calendar alignment, currency-unit safety |
| Transformation | `transformation/ratio_engine.py` | 40 financial ratios (liquidity, leverage, profitability, efficiency, cash flow, growth, distress signals) |
| Transformation | `transformation/zscore_benchmark.py` | Sector-relative Z-scores — **robust median/MAD method by default**, classic mean/stdev still available via `method="standard"` |
| Scoring | `scoring/xgboost_model.py` | XGBoost distress classifier + SHAP explainability + **stratified K-fold cross-validated AUC** (in-sample AUC alone is not trusted — see §3) |
| Scoring | `scoring/train.py` | Feature matrix construction with an explicit **label-leakage guard**: the 5 ratios used to build the label are excluded from the model's own feature set |
| Narrative | `narrative/context_builder.py` | Structured RAG context assembly, plus a content hash used for caching |
| Narrative | `narrative/gemini_client.py` | Gemini REST API integration with exponential-backoff retry on transient failures; system prompt forbids introducing numbers not present in context |
| Reporting | `reporting/pdf_generator.py` | Per-company + multi-company screen PDF reports |
| Reporting | `reporting/charts.py` | SHAP waterfall chart + multi-year ratio trend lines, **direction-aware** (a rising debt-to-equity renders as deteriorating, not improving) |
| Orchestration | `pipeline.py` | Stage functions (`stage_ingest`, `stage_transform`, `stage_score`, `stage_narrate`, `stage_report`) shared by CLI and Airflow |
| Orchestration | `dags/finsight_pipeline_dag.py` | Airflow DAGs: daily pipeline (weekdays 06:00) + weekly model retrain (Sundays 05:00) |
| Storage | `database/storage.py` | **SQLAlchemy Core**, dialect-aware — SQLite by default, Postgres via `DATABASE_URL`, no application code changes to switch |
| Config | `config/companies.yaml` / `config/settings.py` | 15 companies, 4 sectors, 7-year lookback — add a company with zero code changes |
| Dashboard | `webapp/main.py` | Read-only FastAPI backend over the storage layer — storage-driven (shows whatever tickers actually have data), not config-driven |
| Dashboard | `webapp/static/` | No-build-step frontend (vanilla HTML/CSS/JS): dense company screen table, inline-expanding detail panel with SHAP driver bars, ratio trend sparklines, sector outliers, narrative, PDF download |

---

## 2. Data coverage

15 companies across 4 sectors (technology, retail, energy, industrials), 7
years of 10-K history. Adding a company is a config-only change in
`config/companies.yaml` (ticker, CIK, sector) — no code changes anywhere
in the pipeline.

---

## 3. The three hardening passes

These were built after the initial platform, each found and fixed a real
issue rather than just adding polish, and each is backed by a test that
proves the specific failure mode is closed.

### 3.1 Robust sector Z-score benchmarking

**Problem:** with only ~4 companies per sector, a single unusual company
inflates the classic mean/stdev enough to compress every *other* company's
Z-score toward zero — exactly the distinction a distress screen most needs
to preserve.

**Fix:** `ZScoreBenchmark` now defaults to a modified Z-score using
**median and MAD** (median absolute deviation, scaled by 1.4826 to be
comparable to a standard Z-score under normality) instead of mean/stdev.
The classic method remains available via `method="standard"` for anyone
who wants it.

**Proof:** a planted-outlier test shows the classic method collapsing two
genuinely different companies (margins 0.10 vs 0.12) to a Z-score gap of
under 0.02 — indistinguishable — once a wild outlier (100) is added to
their 4-company sector. The robust method keeps them 0.5+ apart, correctly
ordered relative to the sector median.

### 3.2 PDF charts (SHAP waterfall + ratio trend lines)

**What's there:** `reporting/charts.py` renders matplotlib charts to raw
PNG bytes, embedded via ReportLab's `Image` flowable (not
`reportlab.lib.utils.ImageReader`, which is for the low-level canvas API
and doesn't work as a platypus flowable — this was found and fixed during
implementation).

- **SHAP waterfall**: horizontal bars for the top drivers, colored by
  whether they push distress probability up (red) or down (green),
  labeled with the base rate → scored probability.
- **Ratio trend lines**: up to 4 headline ratios (current ratio, net
  margin, debt-to-equity, ROA) plotted across all available fiscal years,
  when at least 2 years of history exist.

**A real bug caught by actually looking at the output, not just checking
it didn't crash:** the trend line color logic originally assumed "value
went down = red (bad), value went up = green (good)" for every ratio. That
is backwards for ratios where a *lower* value is healthier — a rising
debt-to-equity (worse) was rendering green, the same color as a genuinely
improving ratio. Fixed with a `HIGHER_IS_BETTER` per-ratio direction map
and `_is_improving()` in `reporting/charts.py`. This was only caught by
converting a generated PDF to PNG and visually inspecting it — a reminder
that "renders without crashing" and "renders correct information" are
different bars for anything visual.

**Data plumbing added to support this:** `CompanyContext` gained
`ratio_history` (multi-year, per-ratio) and `base_value` (the SHAP
explainer's base rate) fields; `distress_scores` gained a `base_value`
column (with an idempotent migration so existing databases don't need to
be dropped); `pipeline.py::_build_ratio_history()` reshapes stored
per-year ratios into the per-ratio time series the trend charts need.

### 3.3 Postgres support for real Airflow concurrency

**Problem:** the original storage layer used the `sqlite3` module directly
with `INSERT OR REPLACE` syntax. SQLite serializes writers at the file
level; under real Airflow parallelism (concurrent per-company task
instances, or a retry racing a still-running attempt), this eventually
raises `database is locked`.

**Fix:** `database/storage.py` was rewritten on **SQLAlchemy Core**,
parameterized by a `DATABASE_URL` environment variable (defaulting to the
same local SQLite file path for zero-setup dev/demo use). Switching to
Postgres in production is purely a config change:

```bash
export DATABASE_URL="postgresql+psycopg2://user:password@host:5432/finsight"
```

Upserts are implemented per-dialect (`INSERT ... ON CONFLICT DO UPDATE`,
which both SQLite and Postgres support natively via SQLAlchemy's
dialect-specific `insert()`), rather than SQLite-only syntax. Schema
migrations (e.g. the `base_value` column added in §3.2) are idempotent —
checked via `sqlalchemy.inspect()` before issuing `ALTER TABLE`, so an
existing database doesn't need to be dropped on upgrade.

**This was verified against a real, running Postgres 16 instance**, not
just asserted to be dialect-compatible in theory:
- Full CRUD round-trip (ratios, Z-scores, distress scores, narratives) —
  all correct.
- Upsert/overwrite semantics — re-saving a value replaces it in place,
  doesn't duplicate it.
- **The actual point of the fix**: 12 real threads writing concurrently to
  the same table, 0 errors. An equivalent SQLite test with the schema
  still being created concurrently by multiple threads failed on 10 of 12
  writers with lock contention / "table already exists" errors — a
  concrete demonstration of the failure mode this fix closes, not a
  hypothetical one.

Postgres-backed tests (`tests/test_storage.py::test_postgres_*`) skip
automatically if no Postgres instance is reachable at the configured test
URL, so the suite stays portable for environments without a database
service (e.g. a plain CI runner) while still running for real whenever
Postgres is available.

### 3.4 Fiscal-year calendar alignment and currency-unit safety

Under the item originally framed as "unit-scale normalization." Before
writing any code, checked SEC's own API documentation and sample payloads
to verify the assumption — turned out the original framing was wrong: SEC's
`companyfacts` endpoint returns raw, unscaled monetary values (Apple's
FY2023 revenue really is `383285000000`), so there was no thousands-scaling
bug to fix. Two real issues were found instead:

- **Non-calendar fiscal years**: a January-31-FYE filer's SEC-labeled
  `fy=2023` (period ending 2023-01-31) covers 11 of its 12 months in
  calendar 2022, so it was being grouped with December-FYE peers' actual
  2023 figures. `ingestion/xbrl_parser.py::_calendar_align_year` now
  re-buckets each resolved value by the calendar year containing the
  majority of its fiscal period (via the period end date), enabled by
  default and toggleable. The original SEC-reported fiscal year and period
  end date are preserved on every `LineItem` for traceability.
- **Silent foreign-currency fallback**: SEC's own docs give the example of
  a filer reporting the same concept in multiple currencies. The original
  unit-selection logic fell back to *whatever currency was present* if USD
  wasn't available, which would silently treat a EUR/CAD figure as USD.
  Monetary fields now require an actual USD entry; non-monetary fields
  (share counts) are unaffected.

### 3.5 Two earlier fixes (context, for completeness)

Before the three passes above, two other real issues were found and fixed:

- **XBRL restatement handling**: a single GAAP tag can carry multiple
  entries for the same fiscal year (the original figure, and a later
  restated figure reported again as the prior-year comparative in a
  subsequent filing). The parser now keeps only the entry with the latest
  `filed` date per fiscal year, within each tag, before merging across the
  multi-tag fallback chain.
- **Distress label leakage**: the composite label was built from 5
  ratios that were *also* being fed to the model as features, so the model
  was partly just reconstructing the label formula. `build_full_matrix`
  (all 40 ratios, label construction only) and `build_feature_matrix` (35
  ratios, model training) are now separate, and `build_training_labels`
  raises `ValueError` if accidentally called on the trimmed matrix. Also
  added stratified K-fold cross-validated AUC alongside in-sample AUC,
  since in-sample AUC alone saturates near 1.0 on a dataset this small
  regardless of leakage and isn't informative on its own.

### 3.6 Gemini resilience and caching

Two gaps closed together, since both were about making the narrative stage
production-safe rather than a happy-path demo:

- **Retry/backoff**: transient failures (HTTP 429 rate limits, 5xx server
  errors, connection drops, timeouts) are retried with exponential backoff
  and jitter. Non-transient failures (400/401/403/404) fail immediately —
  a request Gemini has already rejected as invalid will fail identically
  on every retry, so retrying just burns time and quota.
- **Content-hash caching**: `CompanyContext.content_hash()` hashes the
  exact rendered prompt text (not the raw dataclass — the prompt is
  exactly what Gemini sees). `pipeline.py::stage_narrate` checks this hash
  against what's stored for that company/year before calling Gemini; if
  unchanged, the existing narrative is reused rather than regenerated.
  Caught one subtlety while testing this: a ratio that never actually
  surfaces in the rendered prompt (not a notable outlier, no YoY history,
  not a top SHAP driver) can change without changing the hash — that's
  correct behavior, not a bug, since Gemini genuinely never saw that value
  either way and the narrative wouldn't change.

### 3.7 Dashboard: FastAPI backend + no-build-step frontend

A read-only viewer over the storage layer — `webapp/main.py` duplicates no
pipeline logic, it just queries `Storage` and shapes JSON. The frontend
(`webapp/static/`) is vanilla HTML/CSS/JS with zero external dependencies
(no CDN, no build step, no charting library) so it works offline and
loads instantly: SHAP driver bars and ratio-trend sparklines are hand-
drawn (styled divs and inline SVG respectively) rather than pulled from a
JS charting library. Design is dense and table-first — built for scanning
many companies quickly the way a credit analyst actually works, not a
marketing-style card grid — with monospace figures throughout for
tabular alignment and a single reserved accent color for interactive
elements, kept visually separate from the three-color risk-band system.

Verified with real tools, not assumption: curled every endpoint, then
loaded the live page in actual Chromium via Playwright (not
`wkhtmltoimage` — its bundled WebKit turned out not to support `fetch()`
at all, which would have produced a false "broken" result) and clicked
through real interactions — row expansion, search filtering, a narrow
viewport. That verification pass caught four real issues before they
shipped (all documented as RC-012 in `tests/REGRESSION_CASES.md`):

- The API only recognized the 15 real tracked companies, not
  `scripts/run_demo.py`'s synthetic universe — fixed by making the
  dashboard storage-driven (`Storage.list_tickers()`) instead of
  config-driven.
- `run_demo.py` itself never persisted distress scores to storage (only
  printed a sample), so the dashboard had nothing to show even after the
  first fix — fixed by routing every company through the same
  `explain_single` + `save_distress_score` path the real pipeline uses.
- The sector-outlier table colored by raw z-score sign rather than
  whether the deviation is actually good or bad for that ratio — the same
  category of bug as §3.2's trend-chart fix, independently reintroduced in
  a different part of the UI. Fixed the same way (routed through
  `HIGHER_IS_BETTER`), and renamed the CSS classes from `z-pos`/`z-neg` to
  `z-favorable`/`z-concerning` so the sign/meaning mismatch that invited
  the bug can't silently reappear.
- A responsive CSS rule hid table columns by a class that only existed on
  header cells, not the dynamically-generated data cells — misaligning
  the table on narrow viewports. Fixed with position-based (`nth-child`)
  selectors instead.

---

## 4. Testing

**93 tests, all passing** (89 always run; 4 Postgres-specific tests run
whenever a Postgres instance is reachable, otherwise skip cleanly):

| File | Covers |
|---|---|
| `test_xbrl_parser.py` | Multi-tag fallback, schema evolution, restatements, fiscal-year calendar alignment, currency-unit safety |
| `test_ratio_engine.py` | Ratio correctness, null-safety on missing inputs |
| `test_zscore_benchmark.py` | Robust vs. standard Z-scoring, outlier resistance, sector isolation |
| `test_train.py` | Label-leakage guard (signal ratios excluded from features) |
| `test_xgboost_model.py` | Training, scoring, SHAP explanation, schema-drift resilience |
| `test_storage.py` | SQLite + Postgres round-trips, upserts, **live concurrent-write test** |
| `test_charts.py` | Chart generation validity, direction-aware trend coloring |
| `test_pdf_generator.py` | Report generation with/without outliers, SHAP data, ratio history |
| `test_gemini_client.py` | Retry/backoff (retryable vs. non-retryable errors, exhausted retries, backoff growth), content hashing |
| `test_pipeline_narrate_caching.py` | Integration-level: Gemini actually skipped/called correctly based on cache state |
| `test_webapp.py` | Dashboard API: risk-band boundaries, storage-driven company listing, config fallback metadata, 404 handling |

`tests/REGRESSION_CASES.md` documents **12 real defects** found during
development, each as Symptom → Root Cause → Fix → Regression Test:
1. Silent revenue gaps for pre-2018 filers (tag fallback)
2. ZeroDivisionError crashing the whole ingestion batch
3. Sector Z-scores skewed by tiny peer cohorts (led to the min-peers
   threshold; later hardened further by the robust method in §3.1)
4. XGBoost/SHAP feature mismatch after schema changes
5. Gemini narrative hallucinating unreported figures
6. XBRL restatements silently kept the stale original value
7. Distress labels leaking into model features
8. Ratio trend charts colored the wrong direction for "lower is better" ratios
9. SQLite unsafe under concurrent Airflow task writers
10. Fiscal-year misalignment across non-calendar filers + silent foreign-currency fallback
11. No resilience to transient Gemini failures, no narrative caching
12. Dashboard build: ticker mismatch between API and demo data, unsaved
    demo distress scores, outlier-table color direction bug, mobile
    column-alignment bug (four sub-issues, all caught by actually running
    the thing in a real browser rather than assuming it worked)

---

## 5. Everything from the original improvement list is now closed, plus a dashboard

All five original items have been implemented and verified: robust
Z-score benchmarking, PDF charts, Postgres support, fiscal-year/currency
correctness in ingestion, and Gemini resilience + caching. A dashboard
(§3.7) was added on top — a FastAPI backend plus a dependency-free
frontend for actually browsing the scored companies rather than only
reading PDFs one at a time. There's always more that could be done on a
project like this (see the honest caveats scattered through §3 — e.g. the
calendar-alignment heuristic is an approximation, not an exact overlap
calculation), but nothing is currently sitting half-finished or
claimed-but-unverified.

---

## 6. Running it

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

export SEC_USER_AGENT="Your Name your-email@example.com"
export GEMINI_API_KEY="your-gemini-api-key"
export DATABASE_URL="postgresql+psycopg2://user:password@localhost:5432/finsight"  # optional, defaults to SQLite

python3 scripts/run_demo.py     # offline, synthetic data, no external network needed
python3 pipeline.py --stage all # live SEC EDGAR + Gemini
pytest tests/ -v                # full suite; Postgres tests auto-skip if no DB reachable

uvicorn webapp.main:app --reload  # dashboard, after either of the above has written data
# open http://localhost:8000
```

See `AGENT_SETUP.md` for a step-by-step runbook (setup → verify → run →
push to GitHub) written for an agent like Cline to execute directly.
