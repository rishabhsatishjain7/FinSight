# Agent Setup & Deploy Instructions

This file is a runbook, not documentation to read passively — execute it
top to bottom, command by command, confirming each step worked before
moving to the next. If a step fails, stop and report the failure rather
than skipping ahead.

Goal: get this project (backend pipeline + dashboard frontend) running
locally, verify it actually works, then push it to GitHub as a new
repository.

---

## Step 0 — Check prerequisites

Run each of these and confirm a sane version is printed. If any are
missing, stop and tell the user what to install before continuing.

```bash
python3 --version      # need 3.11+
git --version
gh --version            # GitHub CLI -- optional, see Step 7
```

---

## Step 1 — Create a virtual environment and install dependencies

From the project root:

```bash
python3 -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

This installs pandas, XGBoost, SHAP, ReportLab, matplotlib, SQLAlchemy,
FastAPI/uvicorn (dashboard backend), and the rest of the stack. XGBoost
and SHAP can take a couple of minutes to install — that's normal, not a
hang. This has been verified to install cleanly with no dependency
conflicts in a fresh venv.

Airflow is intentionally **not** installed here — `requirements.txt` does
not include it. Steps 2–4 below (tests, demo, dashboard) don't need it;
`dags/finsight_pipeline_dag.py` is the only file that imports `airflow`,
and it's meant to be copied into a separate Airflow deployment, not run
from this venv. If the user specifically asks to set up Airflow, read
`requirements-airflow.txt` for why it's separate and the exact commands —
do not try to `pip install apache-airflow` into this venv; it will
conflict with `SQLAlchemy==2.0.36` and fail.

---

## Step 2 — Verify the test suite passes

```bash
pytest tests/ -v
```

Expect **89 passed, 4 skipped** (the 4 skipped are Postgres-specific
tests — they auto-skip unless a Postgres instance is reachable at
`localhost:5432`, which is expected on a fresh machine with no DB
running). If anything *fails* (not skips), stop and report it — don't
proceed to a GitHub push with a broken test suite.

---

## Step 3 — Run the offline demo to confirm the backend pipeline works

```bash
python3 scripts/run_demo.py
```

This runs the full ingestion → ratio engine → Z-score → XGBoost → SHAP
pipeline against synthetic data, entirely offline (no SEC EDGAR or Gemini
API key needed), and **saves distress scores to `data/finsight.db`** —
this is what the dashboard in the next step actually displays. It should
finish in well under a minute and print training metrics and sample
distress scores.

---

## Step 4 — Launch and verify the dashboard

The dashboard is a FastAPI backend (`webapp/main.py`) serving a static
frontend (`webapp/static/`) that reads whatever the pipeline has written
to the database — it's a viewer, not a second pipeline, so Step 3 must
run first or the dashboard will correctly show an empty state.

```bash
uvicorn webapp.main:app --reload
```

Open **http://localhost:8000** in a browser. You should see:
- A summary strip (high/elevated/low risk counts) in the top bar
- A dense table of companies with distress probabilities and risk badges
- Clicking a row expands an inline detail panel: SHAP driver bars, sector
  outlier table, ratio trend sparklines, and a narrative section (the
  narrative will say "No narrative generated for this period yet" unless
  you've run the live pipeline with `GEMINI_API_KEY` set — that's
  expected for the offline demo, not a bug)

If the page loads but shows "No scored companies yet," Step 3 wasn't run
or `data/finsight.db` is missing — go back and run it. If the page can't
reach the API at all, confirm uvicorn is actually running and nothing
else is bound to port 8000.

Stop the server (Ctrl+C) before moving on.

---

## Step 5 — Confirm the README is in good shape

`README.md` and `PROJECT_OVERVIEW.md` already exist at the project root —
don't create a duplicate or overwrite them. Skim `README.md` and confirm
it still accurately describes the setup commands above (it should, but
things drift). Only edit it if something is actually wrong or stale;
don't rewrite it wholesale.

---

## Step 6 — Initialize git and make the first commit

Check whether this is already a git repo first:

```bash
git status
```

If it says "not a git repository", initialize one:

```bash
git init
git add -A
git status --porcelain   # review what's staged before committing
```

Sanity-check the staged file list — none of these should appear:
`venv/`, `__pycache__/`, `data/finsight.db`, `data/cache/*.json`,
`reports/output/*.pdf`, `.env`. If any of those are staged, `.gitignore`
isn't being picked up correctly (check it exists at the project root and
re-run `git status`) — fix that before committing, don't commit secrets
or generated artifacts.

```bash
git commit -m "Initial commit: FinSight financial distress screening platform"
```

---

## Step 7 — Create the GitHub repository and push

**If `gh` (GitHub CLI) is installed and authenticated** (`gh auth status`
succeeds):

```bash
gh repo create FinSight --private --source=. --remote=origin --push
```

Use `--public` instead of `--private` if the user wants it public — ask
if unclear rather than guessing.

**If `gh` is not available or not authenticated**, do this instead:

1. Tell the user to create a new empty repository at
   https://github.com/new (no README/license/gitignore — this repo
   already has those, adding them on GitHub's side will conflict), and
   get the resulting remote URL (e.g.
   `https://github.com/<username>/FinSight.git`).
2. Then run:

```bash
git branch -M main
git remote add origin <the URL the user gives you>
git push -u origin main
```

Do not guess a GitHub username or repo URL — ask the user for it if `gh`
isn't available.

---

## Step 8 — Confirm

```bash
git log --oneline -1
git remote -v
```

Report back the repo URL so the user can open it in a browser. Done.

---

## Notes for the agent

- Don't attempt to run `pipeline.py --stage all` (the live SEC EDGAR /
  Gemini path) as part of this setup — it needs `SEC_USER_AGENT` and
  `GEMINI_API_KEY` env vars the user hasn't necessarily set yet. The demo
  script in Step 3 is the correct verification path here, and the
  dashboard works fine against its output.
- The dashboard (`webapp/main.py`) is read-only and storage-driven: it
  shows whatever tickers actually have data in `data/finsight.db`,
  whether that's the demo's synthetic universe (TEC0, ENE1, ...) or the
  real tracked companies from `config/companies.yaml`. Don't "fix" it to
  only show the config list — that's intentional.
- Don't force-push, don't overwrite an existing remote without asking,
  and don't commit anything that looks like a credential or API key.
- Don't add `apache-airflow` back into `requirements.txt` or try to
  install it alongside this project's dependencies, even if asked to "set
  up Airflow too" — it needs its own separate environment (see
  `requirements-airflow.txt`). Mixing the two will fail with a dependency
  resolution error, not a warning.
- If `pytest` fails or either script errors, stop and surface the exact
  error — don't try to silently patch around it without telling the user
  what broke.
