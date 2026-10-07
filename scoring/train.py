"""
FinSight — training entry point for the distress scoring model.

Run standalone:
    python -m scoring.train

Label strategy:
    No public labeled bankruptcy/default dataset ships with this repo, so
    training labels are built with a transparent, documented heuristic
    (`build_training_labels`) over the Z-scored ratio panel: a company/year
    is labeled "distressed" if it falls in the weakest sector decile on a
    composite of the five distress-signal ratios (Altman-Z-style inputs).
    This is a placeholder — swap in real outcome labels (downgrades,
    covenant breaches, Chapter 11 filings) for production use by replacing
    this function; the rest of the pipeline (feature building, training,
    SHAP explanation) is label-source agnostic.

Avoiding label leakage:
    The five ratios used to construct the label (DISTRESS_SIGNAL_RATIOS)
    are EXCLUDED from the feature matrix the model actually trains on
    (`build_feature_matrix`). If they were left in, the model would just
    be learning to reconstruct the label-generating formula rather than
    finding independent predictive signal in the other 35 ratios — that
    inflates validation metrics (near-perfect AUC) without producing a
    model that generalizes to real, independently-sourced labels later.
    `build_full_matrix` (all ratios) is used only for label construction;
    `build_feature_matrix` (signal ratios dropped) is what actually gets
    passed to `DistressScorer.train`.
"""
from __future__ import annotations

import logging

import pandas as pd

from scoring.xgboost_model import DistressScorer

logger = logging.getLogger("finsight.scoring.train")

DISTRESS_SIGNAL_RATIOS = [
    "working_capital_to_assets",
    "retained_earnings_to_assets",
    "ebit_to_assets",
    "equity_to_liabilities",
    "sales_to_assets",
]


def build_full_matrix(
    z_scores_by_company: dict[str, dict[int, dict[str, float | None]]]
) -> pd.DataFrame:
    """
    Flatten {ticker: {year: {ratio: z}}} into a (ticker, fiscal_year) x ratio
    DataFrame containing ALL ratios, including the five DISTRESS_SIGNAL_RATIOS.
    This is the label-construction source, not the model's training input —
    see `build_feature_matrix` for the version actually fed to XGBoost.
    """
    rows = []
    for ticker, year_data in z_scores_by_company.items():
        for year, ratios in year_data.items():
            row = {"ticker": ticker, "fiscal_year": year, **ratios}
            rows.append(row)
    df = pd.DataFrame(rows).set_index(["ticker", "fiscal_year"])
    return df.apply(pd.to_numeric, errors="coerce")


def build_training_labels(full_df: pd.DataFrame, quantile: float = 0.15) -> pd.Series:
    """
    Composite distress signal = mean of the five Altman-Z-style ratio Z-scores.
    Bottom `quantile` of the composite (weakest fundamentals relative to
    sector peers) is labeled 1 (distressed), rest 0.

    Takes the FULL ratio matrix (must include DISTRESS_SIGNAL_RATIOS columns) —
    pass `build_full_matrix` output here, not `build_feature_matrix` output.
    """
    available = [c for c in DISTRESS_SIGNAL_RATIOS if c in full_df.columns]
    if not available:
        raise ValueError(
            "None of DISTRESS_SIGNAL_RATIOS found in the input DataFrame — "
            "did you pass build_feature_matrix() output instead of build_full_matrix()?"
        )
    composite = full_df[available].mean(axis=1, skipna=True)
    threshold = composite.quantile(quantile)
    labels = (composite <= threshold).astype(int)
    return labels


def build_feature_matrix(full_df: pd.DataFrame) -> pd.DataFrame:
    """
    The five ratios used to construct the label are dropped here so the
    model can't just learn to reconstruct the label formula (see module
    docstring, "Avoiding label leakage"). Everything else — 35 of the 40
    registered ratios — remains as model input.
    """
    to_drop = [c for c in DISTRESS_SIGNAL_RATIOS if c in full_df.columns]
    return full_df.drop(columns=to_drop)


def train_and_save(z_scores_by_company: dict[str, dict[int, dict[str, float | None]]]) -> dict:
    full_df = build_full_matrix(z_scores_by_company)
    full_df = full_df.dropna(axis=0, how="all")
    full_df = full_df.fillna(0.0)

    labels = build_training_labels(full_df)
    feature_df = build_feature_matrix(full_df)

    scorer = DistressScorer()
    metrics = scorer.train(feature_df, labels)
    metrics["excluded_label_source_features"] = [
        c for c in DISTRESS_SIGNAL_RATIOS if c in full_df.columns
    ]
    scorer.save()
    logger.info("Model saved. Metrics: %s", metrics)
    return metrics


if __name__ == "__main__":
    import json

    from ingestion.sec_edgar_client import SECEdgarClient
    from ingestion.xbrl_parser import XBRLParser
    from transformation.ratio_engine import RatioEngine
    from transformation.zscore_benchmark import ZScoreBenchmark
    from config.settings import flat_company_list

    logging.basicConfig(level=logging.INFO)

    companies = flat_company_list()
    client = SECEdgarClient()
    parser = XBRLParser()
    ratio_engine = RatioEngine()
    zscorer = ZScoreBenchmark()

    ratios_by_company = {}
    sector_map = {}
    for company in companies:
        try:
            raw = client.fetch_company_facts(company["cik"])
        except Exception as exc:
            logger.warning("Skipping %s: %s", company["ticker"], exc)
            continue
        cf = parser.parse(company["ticker"], company["cik"], company["sector"], raw)
        ratios_by_company[company["ticker"]] = ratio_engine.compute_all_years(cf)
        sector_map[company["ticker"]] = company["sector"]

    z_scores = zscorer.score_all(ratios_by_company, sector_map)
    metrics = train_and_save(z_scores)  # handles label-vs-feature split internally
    print(json.dumps(metrics, indent=2))
