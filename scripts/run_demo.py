"""
FinSight — offline demo runner.

Runs ingestion -> transform -> score against synthetic (but structurally
real) XBRL data for a small synthetic universe, entirely offline. Useful
for:
  - CI / sandboxed environments without SEC EDGAR / Gemini network access
  - Quickly sanity-checking the pipeline after a code change
  - Onboarding: see real output shape without needing API keys

For the real thing (live SEC EDGAR + Gemini), use `python pipeline.py --stage all`
with SEC_USER_AGENT and GEMINI_API_KEY set in the environment.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from database.storage import Storage
from ingestion.xbrl_parser import XBRLParser
from scoring.train import build_feature_matrix, build_full_matrix, build_training_labels
from scoring.xgboost_model import DistressScorer
from transformation.ratio_engine import RatioEngine
from transformation.zscore_benchmark import ZScoreBenchmark

random.seed(42)

SECTORS = ["technology", "retail", "energy", "industrials"]
SYNTHETIC_UNIVERSE = [
    {"ticker": f"{sector[:3].upper()}{i}", "sector": sector, "cik": f"999{sector[:2]}{i}"}
    for sector in SECTORS
    for i in range(4)
]


def _synthetic_facts(base_revenue: float, growth: float, health: float, years: range) -> dict:
    """
    health in [0, 1] is a latent "true quality" scalar, but it does NOT
    directly determine every line item. Liquidity, leverage, profitability,
    and asset efficiency each get their own noisy factor (correlated with
    health, but with substantial independent variance layered on top) so
    that no single ratio -- or small group of ratios -- perfectly encodes
    the others. This matters for the demo specifically: the distress label
    is built from 5 ratios (working_capital_to_assets, retained_earnings_to_assets,
    ebit_to_assets, equity_to_liabilities, sales_to_assets), and those 5 are
    excluded from the model's feature set to avoid label leakage (see
    scoring/train.py). If every ratio were a deterministic function of one
    shared `health` scalar, the remaining 35 ratios would trivially recover
    the same signal and the leakage fix would look like a no-op. Independent
    per-factor noise means the excluded ratios carry information the other
    35 genuinely don't fully replicate, so removing them should measurably
    reduce (not eliminate) model performance -- which is the honest,
    realistic behavior we want this demo to show.
    """

    def entry(fy, val):
        return {"fy": fy, "fp": "FY", "form": "10-K", "val": val, "filed": f"{fy+1}-02-15"}

    def factor(center: float, sigma: float, lo: float = 0.02, hi: float = 0.98) -> float:
        """health-correlated but independently noisy factor in [lo, hi]."""
        return min(max(center + random.gauss(0, sigma), lo), hi)

    revenue, cogs, gp, oi, ni, assets, cur_assets, cur_liab, liab, equity, re_ = (
        {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
    )
    rev = base_revenue
    for fy in years:
        rev *= 1 + growth + random.uniform(-0.02, 0.02)

        # Independent per-category factors -- each nudged toward `health`
        # but with its own noise draw, so they diverge from one another.
        profitability = factor(0.15 + 0.55 * health, sigma=0.18)
        efficiency = factor(0.20 + 0.60 * health, sigma=0.18)
        leverage = factor(0.20 + 0.55 * health, sigma=0.18)
        liquidity = factor(0.20 + 0.55 * health, sigma=0.18)
        retention = factor(0.15 + 0.55 * health, sigma=0.20)

        margin = 0.03 + 0.30 * profitability
        c = rev * (1 - margin)
        g = rev - c
        op = g * (0.2 + 0.5 * factor(0.3 + 0.4 * health, sigma=0.15))
        n = op * (0.5 + 0.4 * factor(0.5 + 0.3 * health, sigma=0.15))

        asset_intensity = 1.2 + 1.3 * (1 - efficiency)
        a = rev * asset_intensity
        ca = a * (0.15 + 0.35 * liquidity)
        cl = a * (0.35 - 0.22 * liquidity)
        li = a * (0.75 - 0.45 * leverage)
        eq = max(a - li, a * 0.02)  # keep equity nominally positive
        r = eq * (0.10 + 0.55 * retention)

        y = int(fy)
        revenue[y] = entry(y, rev)
        cogs[y] = entry(y, c)
        gp[y] = entry(y, g)
        oi[y] = entry(y, op)
        ni[y] = entry(y, n)
        assets[y] = entry(y, a)
        cur_assets[y] = entry(y, ca)
        cur_liab[y] = entry(y, cl)
        liab[y] = entry(y, li)
        equity[y] = entry(y, eq)
        re_[y] = entry(y, r)

    def series(d):
        return {"units": {"USD": list(d.values())}}

    return {
        "facts": {
            "us-gaap": {
                "RevenueFromContractWithCustomerExcludingAssessedTax": series(revenue),
                "CostOfRevenue": series(cogs),
                "GrossProfit": series(gp),
                "OperatingIncomeLoss": series(oi),
                "NetIncomeLoss": series(ni),
                "Assets": series(assets),
                "AssetsCurrent": series(cur_assets),
                "LiabilitiesCurrent": series(cur_liab),
                "Liabilities": series(liab),
                "StockholdersEquity": series(equity),
                "RetainedEarningsAccumulatedDeficit": series(re_),
            }
        }
    }


def main():
    parser = XBRLParser()
    ratio_engine = RatioEngine()
    zscorer = ZScoreBenchmark(min_peers=3)
    storage = Storage()

    ratios_by_company = {}
    sector_map = {}

    print(f"Generating synthetic 7-year history for {len(SYNTHETIC_UNIVERSE)} companies...")
    for company in SYNTHETIC_UNIVERSE:
        health = random.uniform(0.05, 0.95)
        growth = random.uniform(-0.03, 0.12)
        raw = _synthetic_facts(
            base_revenue=random.uniform(5e8, 5e10),
            growth=growth,
            health=health,
            years=range(2017, 2024),
        )
        cf = parser.parse(company["ticker"], company["cik"], company["sector"], raw)
        year_ratios = ratio_engine.compute_all_years(cf)
        ratios_by_company[company["ticker"]] = year_ratios
        sector_map[company["ticker"]] = company["sector"]
        storage.save_ratios(company["ticker"], company["sector"], year_ratios)

    print(f"Computed {ratio_engine.feature_count} ratios/company/year.")

    z_scores = zscorer.score_all(ratios_by_company, sector_map)
    for ticker, year_data in z_scores.items():
        storage.save_z_scores(ticker, sector_map[ticker], year_data)
    print("Sector-relative Z-scores computed.")

    full_df = build_full_matrix(z_scores).dropna(how="all").fillna(0.0)
    labels = build_training_labels(full_df)
    feature_df = build_feature_matrix(full_df)  # label-source ratios dropped, see scoring/train.py

    # --- Leakage sanity check -------------------------------------------------
    # Train once WITH the label-source ratios still in the feature set (the
    # bug), once WITHOUT (the fix), and compare cross-validated AUC. In-sample
    # ("train") AUC saturates near 1.0 either way on a dataset this small and
    # isn't informative here -- cv_auc_mean is the number that should actually
    # drop when the leaked features are removed.
    print(f"\n{len(feature_df)} company-year rows, {labels.sum()} labeled distressed.")
    print("Leakage check -- training with vs. without the 5 label-source ratios:")

    leaky_scorer = DistressScorer()
    leaky_metrics = leaky_scorer.train(full_df, labels)
    print(
        f"  WITH label-source ratios ({full_df.shape[1]} features):    "
        f"cv_auc_mean={leaky_metrics['cv_auc_mean']:.4f}  train_auc={leaky_metrics['train_auc']:.4f}"
    )

    scorer = DistressScorer()
    metrics = scorer.train(feature_df, labels)
    print(
        f"  WITHOUT label-source ratios ({feature_df.shape[1]} features): "
        f"cv_auc_mean={metrics['cv_auc_mean']:.4f}  train_auc={metrics['train_auc']:.4f}"
    )
    print(
        "  (train_auc barely moves either way -- the model memorizes ~110 training "
        "rows regardless of leakage, so it's not informative here. cv_auc_mean is "
        "the honest comparison; note the gap between the two runs above is within "
        "typical cross-validation noise (~0.04-0.05 std on a dataset this size), so "
        "don't over-read a single run -- the leakage fix's real value is structural "
        "(the model can no longer trivially reconstruct the label formula), not a "
        "guaranteed AUC delta on every run.)"
    )

    # The deployed/saved model is always the leakage-free one.
    scorer.save()

    # Score + explain every company's LATEST year, persisting each one via
    # explain_single -- the same per-row path pipeline.py::stage_score uses
    # against live data -- so SHAP contributions and base_value are saved
    # too, not just the raw probability. This is what lets the webapp
    # dashboard (webapp/main.py) show real data after running this script;
    # it reads distress_scores from storage, not from this script's stdout.
    print("\nScoring every company's latest year and saving to storage...")
    from datetime import datetime, timezone

    latest_rows = feature_df.groupby(level=0).tail(1)
    explanations = {}
    for idx, row in latest_rows.iterrows():
        ticker, year = idx
        explanation = scorer.explain_single(row)
        explanations[ticker] = explanation
        storage.save_distress_score(
            ticker,
            year,
            explanation["distress_probability"],
            explanation["contributions"],
            datetime.now(timezone.utc).isoformat(),
            base_value=explanation["base_value"],
        )

    print(f"Saved distress scores for {len(latest_rows)} companies.")
    print("\nSample (top 6 by distress probability):")
    preview = sorted(explanations.items(), key=lambda kv: kv[1]["distress_probability"], reverse=True)[:6]
    for ticker, exp in preview:
        top_factor = exp["contributions"][0]
        print(
            f"  {ticker:8s} distress_prob={exp['distress_probability']:.1%}  "
            f"top_factor={top_factor['feature']} (shap={top_factor['shap_value']:+.4f})"
        )

    print("\nDemo pipeline run complete. SQLite DB at data/finsight.db")
    print("Launch the dashboard with: uvicorn webapp.main:app --reload")


if __name__ == "__main__":
    main()
