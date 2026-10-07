"""
FinSight — pipeline orchestration.

Each function here is a discrete, idempotent stage that reads its inputs
from and writes its outputs to the SQLite storage layer. This is
deliberate: it's the same boundary Airflow uses between tasks (dags/
finsight_pipeline_dag.py just wraps these functions in PythonOperators),
so the pipeline runs identically whether invoked via `python pipeline.py`
for local dev/debugging or via the Airflow scheduler in production.
"""
from __future__ import annotations

import argparse
import logging
import sys
import uuid
from datetime import datetime, timezone

from config.settings import flat_company_list
from database.storage import Storage
from ingestion.sec_edgar_client import SECEdgarClient
from ingestion.xbrl_parser import XBRLParser
from narrative.context_builder import CompanyContext
from narrative.gemini_client import GeminiNarrativeClient
from reporting.pdf_generator import PDFReportGenerator, TREND_RATIO_NAMES
from scoring.xgboost_model import DistressScorer
from transformation.ratio_engine import RatioEngine
from transformation.zscore_benchmark import ZScoreBenchmark

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("finsight.pipeline")


def stage_ingest(companies: list[dict] | None = None, force_refresh: bool = False) -> dict:
    """Stage 1: Pull raw XBRL companyfacts from SEC EDGAR, cache + persist."""
    companies = companies or flat_company_list()
    client = SECEdgarClient()
    storage = Storage()

    processed, failed = 0, 0
    for company in companies:
        try:
            raw = client.fetch_company_facts(company["cik"], force_refresh=force_refresh)
            storage.save_raw_facts(
                company["ticker"], company["cik"], raw, datetime.now(timezone.utc).isoformat()
            )
            processed += 1
        except Exception as exc:  # noqa: BLE001
            logger.error("Ingestion failed for %s: %s", company["ticker"], exc)
            failed += 1

    logger.info("Ingestion complete: %d ok, %d failed", processed, failed)
    return {"processed": processed, "failed": failed}


def stage_transform(companies: list[dict] | None = None) -> dict:
    """Stage 2: Parse raw facts (multi-tag fallback) -> compute ratios -> sector Z-scores."""
    companies = companies or flat_company_list()
    parser = XBRLParser()
    ratio_engine = RatioEngine()
    zscorer = ZScoreBenchmark()
    storage = Storage()
    client = SECEdgarClient()  # cache-backed, no network hit if already fetched

    ratios_by_company = {}
    sector_map = {}
    for company in companies:
        try:
            raw = client.fetch_company_facts(company["cik"])
        except Exception as exc:
            logger.warning("No cached facts for %s, skipping transform: %s", company["ticker"], exc)
            continue

        cf = parser.parse(company["ticker"], company["cik"], company["sector"], raw)
        year_ratios = ratio_engine.compute_all_years(cf)
        ratios_by_company[company["ticker"]] = year_ratios
        sector_map[company["ticker"]] = company["sector"]
        storage.save_ratios(company["ticker"], company["sector"], year_ratios)

    z_scores = zscorer.score_all(ratios_by_company, sector_map)
    for ticker, year_data in z_scores.items():
        sector = sector_map[ticker]
        storage.save_z_scores(ticker, sector, year_data)

    logger.info(
        "Transform complete: %d companies, %d ratios/company",
        len(ratios_by_company),
        ratio_engine.feature_count,
    )
    return {"companies": len(ratios_by_company), "ratio_count": ratio_engine.feature_count}


def stage_score(companies: list[dict] | None = None) -> dict:
    """Stage 3: XGBoost distress probability + SHAP attribution per company/year."""
    companies = companies or flat_company_list()
    storage = Storage()
    scorer = DistressScorer()
    scorer.load()

    scored = 0
    for company in companies:
        ticker = company["ticker"]
        z_by_year = storage.get_z_scores(ticker)
        if not z_by_year:
            continue
        for year, z_scores in z_by_year.items():
            import pandas as pd

            row = pd.Series(z_scores).apply(lambda v: v if v is not None else 0.0)
            explanation = scorer.explain_single(row)
            storage.save_distress_score(
                ticker,
                year,
                explanation["distress_probability"],
                explanation["contributions"],
                datetime.now(timezone.utc).isoformat(),
                base_value=explanation["base_value"],
            )
            scored += 1

    logger.info("Scoring complete: %d company-years scored", scored)
    return {"scored": scored}


def stage_narrate(companies: list[dict] | None = None, latest_year_only: bool = True) -> dict:
    """
    Stage 4: Build RAG context per company/year and call Gemini for narrative text.

    Cache-aware: before calling Gemini, checks whether a narrative already
    exists for this (ticker, year) whose stored content_hash matches the
    CURRENT context's content_hash (see CompanyContext.content_hash). If it
    matches, nothing about the underlying ratios/Z-scores/SHAP output has
    changed since the narrative was last generated, so the existing
    narrative is reused and Gemini isn't called again.
    """
    companies = companies or flat_company_list()
    storage = Storage()
    gemini = GeminiNarrativeClient()

    generated = 0
    cached = 0
    for company in companies:
        ticker = company["ticker"]
        ratios_by_year = storage.get_ratios(ticker)
        z_by_year = storage.get_z_scores(ticker)
        if not ratios_by_year:
            continue

        years = sorted(ratios_by_year.keys())
        target_years = [years[-1]] if latest_year_only else years

        for year in target_years:
            score_row = _fetch_score_row(storage, ticker, year)
            if score_row is None:
                continue

            prior_ratios = ratios_by_year.get(year - 1, {})
            context = CompanyContext(
                ticker=ticker,
                name=company["name"],
                sector=company["sector"],
                fiscal_year=year,
                ratios=ratios_by_year.get(year, {}),
                z_scores=z_by_year.get(year, {}),
                distress_probability=score_row["distress_probability"],
                shap_contributions=score_row["contributions"],
                prior_year_ratios=prior_ratios,
                base_value=score_row.get("base_value", 0.0),
            )
            content_hash = context.content_hash()

            existing = storage.get_narrative_record(ticker, year)
            if existing is not None and existing["context_hash"] == content_hash:
                logger.debug("Narrative cache hit for %s FY%d, skipping Gemini call", ticker, year)
                cached += 1
                continue

            narrative = gemini.generate_narrative(context)
            storage.save_narrative(
                ticker, year, narrative, datetime.now(timezone.utc).isoformat(), context_hash=content_hash
            )
            generated += 1

    logger.info(
        "Narrative generation complete: %d generated, %d reused from cache", generated, cached
    )
    return {"generated": generated, "cached": cached}


def _fetch_score_row(storage: Storage, ticker: str, year: int) -> dict | None:
    return storage.get_distress_score(ticker, year)


def stage_report(companies: list[dict] | None = None) -> dict:
    """Stage 5: Render PDF reports (per company + combined screen summary)."""
    companies = companies or flat_company_list()
    storage = Storage()
    pdf_gen = PDFReportGenerator()

    contexts = []
    for company in companies:
        ticker = company["ticker"]
        ratios_by_year = storage.get_ratios(ticker)
        z_by_year = storage.get_z_scores(ticker)
        if not ratios_by_year:
            continue
        year = max(ratios_by_year.keys())
        score_row = _fetch_score_row(storage, ticker, year)
        if score_row is None:
            continue

        narrative_text = storage.get_narrative(ticker, year) or "[No narrative generated for this period.]"
        context = CompanyContext(
            ticker=ticker,
            name=company["name"],
            sector=company["sector"],
            fiscal_year=year,
            ratios=ratios_by_year.get(year, {}),
            z_scores=z_by_year.get(year, {}),
            distress_probability=score_row["distress_probability"],
            shap_contributions=score_row["contributions"],
            prior_year_ratios=ratios_by_year.get(year - 1, {}),
            ratio_history=_build_ratio_history(ratios_by_year),
            base_value=score_row.get("base_value", 0.0),
        )
        pdf_gen.generate_company_report(context, narrative_text)
        contexts.append(context)

    if contexts:
        pdf_gen.generate_multi_company_summary(contexts)

    logger.info("Reporting complete: %d company reports generated", len(contexts))
    return {"reports": len(contexts)}


def _build_ratio_history(
    ratios_by_year: dict[int, dict[str, float | None]],
    ratio_names: tuple[str, ...] = TREND_RATIO_NAMES,
) -> dict[str, dict[int, float]]:
    """
    Reshape {year: {ratio: value}} -> {ratio: {year: value}} for the fixed
    set of headline ratios the PDF renders as trend charts (see
    reporting/pdf_generator.py::TREND_RATIO_NAMES). None values are dropped
    rather than plotted, since a gap year shouldn't be drawn as zero.
    """
    history: dict[str, dict[int, float]] = {name: {} for name in ratio_names}
    for year, ratios in ratios_by_year.items():
        for name in ratio_names:
            value = ratios.get(name)
            if value is not None:
                history[name][year] = value
    return {name: years for name, years in history.items() if years}


def run_full_pipeline(companies: list[dict] | None = None) -> dict:
    """Runs all five stages end-to-end. Equivalent to one DAG run."""
    run_id = str(uuid.uuid4())
    storage = Storage()
    storage.log_run(run_id, datetime.now(timezone.utc).isoformat())

    try:
        results = {
            "ingest": stage_ingest(companies),
            "transform": stage_transform(companies),
            "score": stage_score(companies),
            "narrate": stage_narrate(companies),
            "report": stage_report(companies),
        }
        storage.complete_run(
            run_id,
            datetime.now(timezone.utc).isoformat(),
            "success",
            results["ingest"]["processed"],
            results["ingest"]["failed"],
        )
        return results
    except Exception as exc:
        storage.complete_run(
            run_id, datetime.now(timezone.utc).isoformat(), "failed", 0, 0, notes=str(exc)
        )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FinSight pipeline runner")
    parser.add_argument(
        "--stage",
        choices=["ingest", "transform", "score", "narrate", "report", "all"],
        default="all",
    )
    parser.add_argument("--force-refresh", action="store_true")
    args = parser.parse_args()

    stage_map = {
        "ingest": lambda: stage_ingest(force_refresh=args.force_refresh),
        "transform": stage_transform,
        "score": stage_score,
        "narrate": stage_narrate,
        "report": stage_report,
        "all": run_full_pipeline,
    }

    result = stage_map[args.stage]()
    logger.info("Result: %s", result)
    sys.exit(0)
