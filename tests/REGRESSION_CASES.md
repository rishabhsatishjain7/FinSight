# FinSight — Regression Case Log

Structured record of pipeline defects found during development, their root
cause, the fix, and the automated test that now guards against recurrence.
Each entry follows: **Symptom → Root Cause → Fix → Regression Test**.

---

### RC-001: Silent revenue gaps for pre-2018 filers

**Symptom:** Companies with >7 years of history showed `revenue = None` for
fiscal years 2016–2017 despite those 10-Ks clearly reporting revenue.

**Root cause:** ASC 606 (effective 2018) caused most filers to switch from
`SalesRevenueNet` to `RevenueFromContractWithCustomerExcludingAssessedTax`.
The original ingestion used a single hardcoded tag per line item, so any
company that changed tags mid-history lost coverage for whichever era used
the tag not being queried.

**Fix:** Replaced single-tag lookups with `TAG_FALLBACK_MAP` — an ordered
list of candidate tags per canonical field — and merge logic that fills
each fiscal year from whichever tag has data for that specific year
(`ingestion/xbrl_parser.py::_resolve_field`).

**Regression test:** `tests/test_xbrl_parser.py::test_schema_evolution_produces_continuous_series`

---

### RC-002: ZeroDivisionError crashing the whole batch on one company

**Symptom:** A single company with `current_liabilities = 0` in one fiscal
year (a real, if unusual, reported value) crashed `stage_transform` for
the entire company universe, since ratio computation ran synchronously
across all companies in one process.

**Root cause:** Ratio functions performed raw division without guarding
against zero or missing denominators.

**Fix:** All ratio functions route through `_safe_div`, which returns
`None` instead of raising for zero/missing denominators. Ingestion and
transform stages also now isolate failures per-company (`stage_ingest`,
`stage_transform` in `pipeline.py`) so one bad filer no longer aborts the
batch.

**Regression test:** `tests/test_ratio_engine.py::test_missing_denominator_returns_none_not_exception`

---

### RC-003: Sector Z-scores skewed by tiny peer cohorts

**Symptom:** Early in development, sectors with only 1-2 companies
ingested produced Z-scores of exactly ±0.0 or wildly unstable values
(stdev computed from n=1 is meaningless / undefined).

**Root cause:** `ZScoreBenchmark` computed sector mean/stdev regardless of
cohort size, so a 2-company "sector" produced statistically meaningless
standardized scores that the narrative engine then presented as confident
outlier signals.

**Fix:** Added `min_peers` threshold (default 3) below which
`compute_sector_stats` omits that (sector, year, ratio) key entirely,
propagating `None` rather than a misleading Z-score.

**Regression test:** `tests/test_zscore_benchmark.py::test_below_min_peer_threshold_returns_none`

---

### RC-004: XGBoost/SHAP feature mismatch after schema changes

**Symptom:** `explainer.shap_values(X)` intermittently raised a shape
mismatch error after the ratio registry gained new ratios post-training.

**Root cause:** `DistressScorer.score()` originally passed whatever
columns happened to be in the incoming DataFrame directly to the model,
with no guarantee they matched the feature set the model was trained on.

**Fix:** `score()` and `explain_single()` now reindex the input against
`self.feature_names` (persisted alongside the model at save time) with
`fill_value=0.0`, so schema drift degrades gracefully to a neutral
Z-score input for new/missing features instead of crashing.

**Regression test:** Covered structurally by `scoring/xgboost_model.py::score`
reindex step; exercised via `pipeline.py::stage_score` integration path.

---

### RC-005: Gemini narrative hallucinating unreported figures

**Symptom:** Early narrative drafts referenced specific dollar figures
that did not appear anywhere in the injected context — the model was
drawing on general financial-domain training knowledge rather than the
company's actual computed ratios.

**Root cause:** The original prompt only loosely suggested "use the data
provided" without an explicit constraint or a rule against introducing
numbers not present in context.

**Fix:** `SYSTEM_INSTRUCTION` in `narrative/gemini_client.py` now
explicitly enumerates numbered rules, rule #1 being "Only reference
figures that appear in the provided context. Do not invent numbers,"
combined with a tightly structured `CompanyContext.to_prompt_context()`
block so every number the model could plausibly cite is already laid out
verbatim.

**Regression test:** Manual QA checklist (spot-check narrative against
context block per release) — flagged as a candidate for an automated
LLM-as-judge regression test in a future iteration.

---

### RC-006: XBRL restatements silently kept the stale original value

**Symptom:** A company's FY2021 revenue occasionally didn't match the
figure in its own FY2022 10-K's prior-year comparative column — the
pipeline was reporting a number the company itself had since restated.

**Root cause:** A single GAAP tag can carry multiple reported entries for
the same fiscal year: the original figure, and a later restated figure
reported again as the comparative prior-year column in a subsequent
filing. `_resolve_field` took whichever entry appeared first in SEC's raw
JSON array (filing order in the API response), not necessarily the most
recently filed one.

**Fix:** Within each candidate tag, entries are now grouped by fiscal
year and only the entry with the latest `filed` date is kept, before
merging across the tag fallback chain
(`ingestion/xbrl_parser.py::_is_more_recent_filing`).

**Regression test:** `tests/test_xbrl_parser.py::test_restatement_prefers_most_recently_filed_value`

---

### RC-007: Distress labels leaking into model features

**Symptom:** Training AUC sat suspiciously near-perfect (~0.999) regardless
of dataset size or model complexity — a sign the model wasn't learning a
generalizable pattern so much as reconstructing a formula it had already
been shown.

**Root cause:** `build_training_labels` derives the distress label from a
composite of five ratios (`working_capital_to_assets`,
`retained_earnings_to_assets`, `ebit_to_assets`, `equity_to_liabilities`,
`sales_to_assets`). The original `build_feature_matrix` fed those same
five ratios into the model as features, so XGBoost could trivially
reconstruct the label formula instead of finding independent predictive
signal.

**Fix:** Split ratio-matrix construction into `build_full_matrix` (all 40
ratios — used only to build the label) and `build_feature_matrix` (drops
the 5 label-source ratios — used to actually train the model).
`build_training_labels` now raises `ValueError` if called on an
already-trimmed matrix, so the bug can't quietly return through a
call-site mistake (`scoring/train.py`).

Two things surfaced while verifying this fix, both addressed alongside it:

1. The synthetic demo data generator (`scripts/run_demo.py::_synthetic_facts`)
   originally derived every ratio from one shared `health` scalar, so the
   35 remaining ratios were almost perfectly collinear with the 5 excluded
   ones and the fix looked like a no-op in the demo. Rewrote the generator
   to give each financial category (profitability, efficiency, leverage,
   liquidity, retention) its own independently-noisy factor, so the
   excluded ratios carry genuinely non-replicable signal.
2. In-sample ("train") AUC saturates near 1.0 on a dataset this small
   regardless of leakage, so it isn't a useful metric on its own. Added
   stratified K-fold cross-validated AUC
   (`DistressScorer._cross_validated_auc`) alongside it — `cv_auc_mean` is
   the number that reflects generalization, and `scripts/run_demo.py` now
   prints both trainings side by side so it's visible on every run. Note:
   on a ~110-row dataset, CV AUC has real run-to-run noise (~0.04-0.05
   std), so the "with vs. without" gap isn't guaranteed to point the same
   direction on every run — the leakage fix's real value is structural
   (the model can no longer trivially reconstruct the label formula from
   its own inputs), not a guaranteed AUC delta on any single run.

**Regression test:** `tests/test_train.py::test_build_feature_matrix_excludes_signal_ratios`,
`tests/test_train.py::test_labels_require_full_matrix_not_feature_matrix`

---

### RC-008: Ratio trend charts colored the wrong direction for "lower is better" ratios

**Symptom:** Visual QA on a generated report showed `debt_to_equity` rising
from 1.2 to 2.1 (a genuine deterioration — more leverage) rendered in
green, the same color used for genuinely improving ratios, while
`current_ratio` falling over the same period was correctly rendered red.

**Root cause:** `ratio_trend_chart`'s color logic used a single rule —
"value went down = red, value went up = green" — which is correct for
ratios where higher is better (current_ratio, net_margin, ROA...) but
backwards for ratios where lower is better (debt_to_equity,
liabilities_to_assets, days_sales_outstanding...). A worsening leverage
trend was visually indistinguishable from an improving one.

**Fix:** Added `HIGHER_IS_BETTER` — a per-ratio direction map — and
`_is_improving()` in `reporting/charts.py`, which the trend chart color
logic now consults instead of assuming direction. Caught during manual
visual inspection of a generated PDF (converted to PNG and reviewed) while
verifying the chart feature, not by an automated test — a reminder that
"doesn't crash" and "renders correct information" are different bars, and
visual output specifically needs visual review at least once per feature.

**Regression test:** `tests/test_charts.py::test_trend_color_accounts_for_ratio_direction`

---

### RC-009: SQLite unsafe under concurrent Airflow task writers

**Symptom:** Under real Airflow parallelism (e.g. per-company ingestion
tasks running concurrently, or a retry racing a still-running attempt),
writes to the shared SQLite file would eventually raise `database is
locked` once contention was high enough — SQLite serializes writers at the
file level and has no row-level locking.

**Root cause:** `Storage` used the `sqlite3` module directly, hardcoding
SQLite as the only supported backend, with `INSERT OR REPLACE` syntax that
doesn't even exist in other databases.

**Fix:** Rewrote `database/storage.py` on SQLAlchemy Core, parameterized
by a `DATABASE_URL` (env var, defaulting to the existing SQLite file path
for zero-setup local dev). Postgres support requires no application code
changes — just pointing `DATABASE_URL` at a Postgres instance, e.g.
`postgresql+psycopg2://user:pass@host:5432/finsight`. Upserts are
implemented per-dialect (`INSERT ... ON CONFLICT DO UPDATE`, which both
SQLite and Postgres support natively) rather than SQLite-only syntax.

Verified against a real Postgres 16 instance (not just structurally,
against a live server): full CRUD round-trip, upsert/overwrite semantics,
and — the actual point of the fix — 12 concurrent threads writing to the
same table with zero errors, where the equivalent SQLite test with the
schema still being created concurrently failed on 10 of 12 writers
(`table X already exists` / lock contention).

**Regression test:** `tests/test_storage.py::test_postgres_handles_concurrent_writers`
(auto-skips if no Postgres is reachable, so the suite stays portable for
environments without a DB service).

---

### RC-010: Fiscal-year misalignment and silent foreign-currency fallback

Two related correctness issues under the "unit-scale normalization" item,
found and fixed together. One correction first: the original framing
assumed SEC's XBRL values might be reported in thousands and need
rescaling. Checked against SEC's own API documentation before writing any
code — that's not the case. SEC's `companyfacts` endpoint returns raw,
unscaled monetary values (Apple's FY2023 revenue is literally
`383285000000`), so no rescaling fix was needed there. The two real issues:

**10a. Fiscal-year misalignment across non-calendar filers**

**Symptom:** Cross-company ratio comparisons and sector Z-scores implicitly
assumed every filer's `fy` label corresponds to the same calendar period.
A January-31-FYE filer's SEC-labeled fy=2023 (period ending 2023-01-31)
actually covers 11 of its 12 months in calendar 2022, so it was being
grouped with December-FYE peers' *2023* figures despite covering mostly
2022 economic conditions.

**Fix:** `ingestion/xbrl_parser.py::_calendar_align_year` re-buckets each
resolved value by the calendar year containing the majority of its fiscal
period, using the period end date, rather than SEC's raw `fy` label.
Enabled by default (`parse(..., align_fiscal_years=True)`); the original
SEC label and period end date are preserved on every `LineItem` for
traceability. Can be disabled to fall back to SEC's raw labels.

**Regression test:** `tests/test_xbrl_parser.py::test_calendar_alignment_rebuckets_non_december_fye`

**10b. Silent foreign-currency fallback**

**Symptom:** Not yet observed in production (none of the 14 tracked
companies are foreign private issuers), but found during review: SEC's own
documentation gives the explicit example of a filer reporting the same
concept in multiple currencies. The original unit-selection logic was
`"USD" if "USD" in units else next(iter(units), None)` — if no USD entry
existed for a given tag/year, it would silently grab whatever currency
*was* present (EUR, CAD, ...) and treat the raw number as if it were USD.

**Fix:** Monetary canonical fields (everything except explicit share-count
fields, tracked in `NON_MONETARY_FIELDS`) now require an actual `"USD"`
entry. If a tag has only non-USD monetary data for a given year, that
year is left unresolved by that tag rather than guessed at — the fallback
chain can still try the next candidate tag. Non-monetary fields (share
counts) are unaffected and still accept whatever unit is present (e.g.
"shares").

**Regression test:** `tests/test_xbrl_parser.py::test_foreign_currency_only_year_is_not_silently_used_as_usd`

---

### RC-011: No resilience to transient Gemini failures, no narrative caching

**Symptom:** A single transient Gemini API hiccup (rate limit, brief 5xx,
dropped connection) would fail the narrative for that company/year outright,
inserting a placeholder string with no attempt to retry. Separately,
re-running the `narrate` pipeline stage — even with completely unchanged
ratios, Z-scores, and SHAP output — regenerated every narrative from
scratch, paying for and waiting on API calls that would produce an
equivalent result to what was already stored.

**Fix, two parts:**

- **Retry/backoff** (`narrative/gemini_client.py`): transient failures
  (HTTP 429, 500/502/503/504, connection errors, timeouts) are now retried
  with exponential backoff and jitter (`_post_with_retry`), up to a
  configurable `max_retries`. Non-transient failures (400 bad request,
  401/403 auth errors, 404) fail immediately without retrying, since a
  request Gemini has already rejected as invalid will fail identically on
  every subsequent attempt — retrying just burns time and quota.
- **Content-hash caching** (`narrative/context_builder.py::CompanyContext.content_hash`,
  `database/storage.py`'s new `context_hash` column,
  `pipeline.py::stage_narrate`): before calling Gemini, the pipeline hashes
  the exact rendered prompt text (not the raw dataclass fields — the
  rendered prompt is precisely what Gemini sees) and compares it against
  the hash stored alongside the last narrative for that company/year. If
  they match, the stored narrative is reused and Gemini isn't called.

A subtlety caught while testing the cache: a ratio that never actually
surfaces in the rendered prompt (not a notable |z|>=1.5 outlier, no
year-over-year history, not among the top SHAP drivers) can change in the
underlying data without changing the hash — and this is correct, not a
bug, since Gemini genuinely never saw that value either way and the
narrative wouldn't change. The hash intentionally reflects what's visible
to Gemini, not the full underlying dataset.

**Regression test:** `tests/test_gemini_client.py` (retry/backoff: 9 tests
covering retryable vs. non-retryable status codes, connection errors,
exhausted retries, exponential backoff growth/cap; content hashing: 4
tests including the "invisible ratio" case above);
`tests/test_pipeline_narrate_caching.py` (integration-level: confirms
Gemini is actually skipped on a re-run with unchanged data, and actually
called again once the underlying distress score changes).

---

### RC-012: Dashboard build — four issues found via actual verification, not assumption

Building `webapp/` (FastAPI backend + static frontend dashboard). Listed
together since all four were caught by actually running the thing —
curling the API, loading it in a real browser via Playwright, and clicking
through it — rather than by code review alone. Consistent with RC-008's
lesson: for anything visual or interactive, "compiles and returns 200" and
"behaves correctly" are different bars.

**12a. Dashboard API only recognized the real 14 tracked companies, not the demo's synthetic universe**

**Symptom:** After running `scripts/run_demo.py` (the documented,
zero-setup way to get data into the pipeline) and opening the dashboard,
every one of the 14 configured companies showed "not yet scored" — despite
the demo script clearly having computed and printed distress scores.

**Root cause:** `webapp/main.py`'s endpoints built the company list from
`config/companies.yaml` (`flat_company_list()`) — the real tracked
universe (AAPL, MSFT, ...) — but `scripts/run_demo.py` seeds a synthetic
universe with unrelated ticker names (TEC0, ENE1, RET3, ...). The two
never overlapped, so the API's source of truth and the demo's actual
output were disjoint sets.

**Fix:** Added `Storage.list_tickers()` (distinct tickers actually present
in the `ratios` table) and made the dashboard storage-driven: it lists and
serves whatever tickers have data, enriching with name/sector from config
when available and falling back to the ticker itself as display name /
`"demo"` as sector otherwise. This also makes the dashboard correctly
handle a partially-ingested real run (some configured companies failed
ingestion) without hiding them.

**Regression test:** `tests/test_webapp.py::TestCompanyList::test_storage_driven_not_config_driven`

**12b. `scripts/run_demo.py` never persisted distress scores**

**Symptom:** Even after fixing 12a, every company still showed a null
distress probability. The demo script's terminal output clearly showed
real computed scores (e.g. "TEC3 distress_prob=87.7%"), so the pipeline
itself was fine — but the dashboard reads from storage, not stdout.

**Root cause:** The script's scoring section only ran `scorer.score()` on
a 6-row sample for a human-readable printout; it never called
`storage.save_distress_score()` for any company, unlike the real pipeline
(`pipeline.py::stage_score`), which persists every row.

**Fix:** Rewrote the scoring section to call `scorer.explain_single()` —
the same per-row path the real pipeline uses — for every company's latest
year, persisting probability, SHAP contributions, and base_value via
`storage.save_distress_score()` for all of them, with the printed sample
now drawn from that same saved data (sorted by probability) rather than a
separate unsaved pass.

**12c. Sector-outlier table colored by raw z-score sign, not by whether the outlier is actually good or bad**

**Symptom:** Caught visually, not in code review: a company's
`current_ratio` sitting at z=+2.53 (well above sector norm — meaningfully
*more* liquidity cushion than peers, a strength) rendered in the same red
"concerning" color used for a genuinely bad outlier.

**Root cause:** This is the same category of bug as RC-008 (trend chart
coloring), reintroduced independently in a different part of the UI: the
outlier table's color logic used the raw sign of the z-score
(positive=red, negative=green) without accounting for whether a positive
deviation is good or bad for that specific ratio — correct for "lower is
better" ratios like debt-to-equity, backwards for "higher is better"
ratios like current_ratio.

**Fix:** `webapp/static/app.js::renderOutliers` now routes through the
same `HIGHER_IS_BETTER` direction table the trend charts use (ported from
`reporting/charts.py`) before choosing a color. Also renamed the CSS
classes from `z-pos`/`z-neg` (which look like they mean literal sign, and
invited exactly this bug) to `z-favorable`/`z-concerning` (named by
meaning), so the mismatch can't silently reappear.

**Regression test:** Verified visually via Playwright screenshot
(`current_ratio` z=+2.53 now renders favorable/green) — flagged as a
candidate for an automated DOM-assertion test in a future iteration,
similar to RC-008's manual-QA note.

**12d. Mobile column-hide rule targeted header cells only, misaligning the table**

**Symptom:** At narrow viewport widths, the "Sector" and "FY" column
*headers* disappeared as intended, but the corresponding data cells did
not — every column after the hidden ones shifted left relative to its
header.

**Root cause:** The responsive CSS rule hid `.col-sector` / `.col-year`,
but those classes were only ever applied to the `<th>` elements in the
static HTML; the `<td>` cells are generated dynamically in `app.js` and
never received matching classes.

**Fix:** Replaced the class-based rule with `tr > *:nth-child(3)` /
`:nth-child(4)` position-based selectors, which apply uniformly to header
and data cells regardless of what classes either one carries.

**Regression test:** Verified visually via Playwright screenshot at a
420px viewport (headers and data now align); same candidate note as 12c
re: an automated test for a future iteration.

---

### RC-013: `pip install -r requirements.txt` failed outright — never tested as one command

**Symptom:** A user running `AGENT_SETUP.md` Step 1 for the first time
got a hard `ResolutionImpossible` error from pip: `apache-airflow==2.9.3`
requires `sqlalchemy<2.0`, but `requirements.txt` also pinned
`SQLAlchemy==2.0.36` for the Postgres-capable storage layer. The two
can never be installed together.

**Root cause:** Every dependency in this project was verified by
installing packages individually and piecemeal in the sandbox used to
build it — `apache-airflow` was explicitly skipped each time ("heavy
dependency footprint," per earlier notes) rather than included. That
meant `pip install -r requirements.txt`, the exact command the setup
instructions tell every new user to run first, was never once actually
executed as a single command before shipping. The conflict is also not a
narrow version mismatch to nudge around — Airflow's `sqlalchemy<2.0`
constraint holds across the entire Airflow 2.x line, so no amount of
version-juggling resolves it; the two packages structurally cannot share
one environment.

**Fix:** Removed `apache-airflow` from `requirements.txt` entirely.
`dags/finsight_pipeline_dag.py` is the only file in the project that
imports `airflow`, and it's never imported by `tests/`,
`scripts/run_demo.py`, `pipeline.py`'s CLI path, or `webapp/` — none of
the actually-tested paths need it installed. Added
`requirements-airflow.txt` documenting why Airflow needs a separate
environment (with its own official constraints file, which is also just
how Airflow is normally deployed in practice) and the exact commands to
set one up. Confirmed the fix for real: ran `pip install -r
requirements.txt` in a genuinely clean, freshly-created virtual
environment — not the already-populated sandbox environment used
throughout development — and got a clean resolution, then reran the full
test suite (89 passed, 4 skipped) and the demo script against that same
clean environment.

**Regression test:** No automated test guards this directly (it's a
packaging/environment concern, not application logic), but
`AGENT_SETUP.md` Step 1 now explicitly states Airflow is not part of the
main install and why, and the "Notes for the agent" section tells an
executing agent not to add it back in even if asked to "set up Airflow
too." The honest gap: nothing currently re-runs `pip install -r
requirements.txt` in CI to catch a future reintroduction of this class of
conflict — worth adding if this project grows a CI pipeline.
