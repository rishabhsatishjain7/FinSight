"""
FinSight — dashboard backend.

A thin read-only API over the existing Storage layer (database/storage.py)
-- no new data model, no duplicated business logic. Every endpoint here
just queries what the pipeline already wrote (ratios, Z-scores, distress
scores, narratives) and shapes it into JSON for the static frontend in
webapp/static/.

Run with:
    uvicorn webapp.main:app --reload
Then open http://localhost:8000

The pipeline must have been run at least once (scripts/run_demo.py for
offline synthetic data, or pipeline.py --stage all for live data) before
there's anything to show -- this app is a viewer, not a second pipeline.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from config.settings import REPORTS_DIR, flat_company_list
from database.storage import Storage

app = FastAPI(title="FinSight Dashboard API")

STATIC_DIR = Path(__file__).resolve().parent / "static"

RISK_BANDS = [
    (0.70, "high"),
    (0.40, "elevated"),
    (0.0, "low"),
]


def _risk_band(probability: float) -> str:
    for threshold, label in RISK_BANDS:
        if probability >= threshold:
            return label
    return "low"


def _latest_year(ratios_by_year: dict) -> int | None:
    return max(ratios_by_year.keys()) if ratios_by_year else None


def _company_metadata(ticker: str) -> dict:
    """
    Looks up name/sector from config/companies.yaml when the ticker is one
    of the tracked real companies. Falls back to the ticker itself as the
    display name and "demo" as the sector for anything else -- this is what
    lets the dashboard work against scripts/run_demo.py's synthetic
    universe (TEC0, ENE1, ...), which has no config entry, without treating
    that as an error.
    """
    configured = {c["ticker"]: c for c in flat_company_list()}
    if ticker in configured:
        return {"name": configured[ticker]["name"], "sector": configured[ticker]["sector"]}
    return {"name": ticker, "sector": "demo"}


@app.get("/api/companies")
def list_companies():
    """
    Summary row per company actually present in storage (see
    Storage.list_tickers -- this is storage-driven, not config-driven, so
    it reflects whatever the pipeline produced, whether that's the real
    tracked universe or scripts/run_demo.py's synthetic one).
    """
    storage = Storage()
    results = []

    for ticker in storage.list_tickers():
        meta = _company_metadata(ticker)
        ratios_by_year = storage.get_ratios(ticker)
        year = _latest_year(ratios_by_year)
        score = storage.get_distress_score(ticker, year) if year is not None else None

        results.append(
            {
                "ticker": ticker,
                "name": meta["name"],
                "sector": meta["sector"],
                "fiscal_year": year,
                "distress_probability": score["distress_probability"] if score else None,
                "risk_band": _risk_band(score["distress_probability"]) if score else None,
            }
        )

    return {"companies": results}


@app.get("/api/companies/{ticker}")
def company_detail(ticker: str):
    ticker = ticker.upper()
    storage = Storage()
    if ticker not in storage.list_tickers():
        raise HTTPException(status_code=404, detail=f"No pipeline data for {ticker}")
    company = _company_metadata(ticker)

    ratios_by_year = storage.get_ratios(ticker)
    z_by_year = storage.get_z_scores(ticker)
    year = _latest_year(ratios_by_year)
    if year is None:
        raise HTTPException(status_code=404, detail=f"No pipeline data for {ticker} yet")

    score = storage.get_distress_score(ticker, year)
    narrative_record = storage.get_narrative_record(ticker, year)

    # Multi-year history for the 4 headline ratios, for trend sparklines --
    # same selection pipeline.py uses for PDF trend charts.
    trend_ratio_names = ["current_ratio", "net_margin", "debt_to_equity", "return_on_assets"]
    ratio_history = {}
    for name in trend_ratio_names:
        series = {y: r.get(name) for y, r in ratios_by_year.items() if r.get(name) is not None}
        if len(series) >= 2:
            ratio_history[name] = series

    outliers = sorted(
        (
            (name, z)
            for name, z in (z_by_year.get(year) or {}).items()
            if z is not None and abs(z) >= 1.5
        ),
        key=lambda t: abs(t[1]),
        reverse=True,
    )

    report_filename = f"{ticker}_{year}_distress_report.pdf"
    report_available = (REPORTS_DIR / report_filename).exists()

    return {
        "ticker": ticker,
        "name": company["name"],
        "sector": company["sector"],
        "fiscal_year": year,
        "distress_probability": score["distress_probability"] if score else None,
        "risk_band": _risk_band(score["distress_probability"]) if score else None,
        "base_value": score.get("base_value", 0.0) if score else None,
        "shap_contributions": score["contributions"][:8] if score else [],
        "ratios": ratios_by_year.get(year, {}),
        "outliers": [{"ratio": name, "z_score": z, "raw_value": ratios_by_year[year].get(name)} for name, z in outliers],
        "ratio_history": ratio_history,
        "narrative": narrative_record["narrative_text"] if narrative_record else None,
        "report_available": report_available,
        "report_url": f"/api/companies/{ticker}/report" if report_available else None,
    }


@app.get("/api/companies/{ticker}/report")
def company_report(ticker: str):
    ticker = ticker.upper()
    storage = Storage()
    ratios_by_year = storage.get_ratios(ticker)
    year = _latest_year(ratios_by_year)
    if year is None:
        raise HTTPException(status_code=404, detail=f"No report available for {ticker}")

    path = REPORTS_DIR / f"{ticker}_{year}_distress_report.pdf"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"No PDF report generated yet for {ticker}")
    return FileResponse(path, media_type="application/pdf", filename=path.name)


@app.get("/api/summary")
def summary():
    """Headline counts for the dashboard's summary strip."""
    data = list_companies()["companies"]
    scored = [c for c in data if c["risk_band"] is not None]
    return {
        "total": len(data),
        "scored": len(scored),
        "high": sum(1 for c in scored if c["risk_band"] == "high"),
        "elevated": sum(1 for c in scored if c["risk_band"] == "elevated"),
        "low": sum(1 for c in scored if c["risk_band"] == "low"),
        "sectors": sorted({c["sector"] for c in data}),
    }


# Mount the static frontend last, so /api/* routes above take precedence.
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
