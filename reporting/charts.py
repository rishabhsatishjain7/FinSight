"""
FinSight — chart generation for PDF reports.

Renders matplotlib charts to raw PNG bytes (never touching disk), which
reporting/pdf_generator.py wraps in io.BytesIO for ReportLab's Image
flowable. Two chart types:

    SHAP waterfall  — shows how each top feature pushes the distress
                       probability up or down from the model's base rate,
                       in the order of contribution magnitude.
    Ratio trend line — 3+ year trajectory for a handful of key ratios, so
                       a reader can see whether a company is stable,
                       improving, or deteriorating rather than reading a
                       single-year snapshot in isolation.

Kept deliberately separate from pdf_generator.py so chart logic can be
unit tested (chart bytes decode as valid PNGs) without spinning up a full
PDF document.
"""
from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")  # headless backend -- no display, safe for server/pipeline use
import matplotlib.pyplot as plt

# Palette consistent with reporting/pdf_generator.py's risk bands / table styling.
COLOR_RISK_UP = "#B3261E"      # a feature pushing distress probability higher
COLOR_RISK_DOWN = "#2E7D32"    # a feature pushing distress probability lower
COLOR_NEUTRAL = "#374151"
COLOR_GRID = "#E5E7EB"


def _fig_to_png_bytes(fig, dpi: int = 150) -> bytes:
    """
    Renders to raw PNG bytes rather than reportlab.lib.utils.ImageReader --
    ImageReader is meant for the low-level canvas API (canvas.drawImage);
    the platypus Image flowable used in pdf_generator.py wants a path or a
    plain file-like object, so callers wrap these bytes in io.BytesIO
    themselves (once for sizing via ImageReader, once fresh for the Image
    flowable, since a BytesIO can only be read once).
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


# Whether a HIGHER value of this ratio is the financially healthier
# direction. Used only to color trend lines correctly -- a rising
# debt_to_equity is deteriorating (should render red), while a rising
# current_ratio is improving (should render green). Ratios not listed
# default to "higher is better" in `_is_improving`, which is the more
# common case among ratios likely to be trend-charted.
HIGHER_IS_BETTER: dict[str, bool] = {
    "current_ratio": True,
    "quick_ratio": True,
    "cash_ratio": True,
    "net_margin": True,
    "gross_margin": True,
    "operating_margin": True,
    "return_on_assets": True,
    "return_on_equity": True,
    "interest_coverage": True,
    "free_cash_flow": True,
    "fcf_margin": True,
    "debt_to_equity": False,
    "debt_to_assets": False,
    "liabilities_to_assets": False,
    "long_term_debt_to_equity": False,
    "days_sales_outstanding": False,
    "days_inventory_outstanding": False,
}


def _is_improving(ratio_name: str, first_value: float, last_value: float) -> bool:
    higher_is_better = HIGHER_IS_BETTER.get(ratio_name, True)
    rising = last_value > first_value
    return rising if higher_is_better else not rising


def shap_waterfall_chart(
    contributions: list[dict],
    base_value: float,
    distress_probability: float,
    top_n: int = 8,
    figsize: tuple[float, float] = (6.5, 3.6),
) -> bytes:
    """
    contributions: list of {"feature": str, "shap_value": float, "feature_value": float},
    already sorted by |shap_value| descending (this is the shape
    DistressScorer.explain_single returns).
    """
    top = contributions[:top_n]
    top = list(reversed(top))  # largest contribution at the top of a horizontal chart

    features = [c["feature"] for c in top]
    values = [c["shap_value"] for c in top]
    colors = [COLOR_RISK_UP if v > 0 else COLOR_RISK_DOWN for v in values]

    fig, ax = plt.subplots(figsize=figsize)
    y_pos = range(len(features))
    ax.barh(y_pos, values, color=colors, height=0.6, zorder=3)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(features, fontsize=8)
    ax.axvline(0, color=COLOR_NEUTRAL, linewidth=0.8, zorder=2)
    ax.set_xlabel("SHAP contribution to distress probability (log-odds)", fontsize=8)
    ax.set_title(
        f"Top drivers  \u2014  base rate {base_value:.2f}  \u2192  "
        f"scored {distress_probability:.1%}",
        fontsize=9,
        color=COLOR_NEUTRAL,
        loc="left",
    )
    ax.grid(axis="x", color=COLOR_GRID, linewidth=0.6, zorder=0)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()

    return _fig_to_png_bytes(fig)


def ratio_trend_chart(
    ratio_name: str,
    history: dict[int, float],
    figsize: tuple[float, float] = (3.1, 2.2),
) -> bytes | None:
    """
    history: {fiscal_year: value}. Returns None (rather than an empty/broken
    chart) if there isn't enough history to plot a meaningful trend.
    """
    years = sorted(history.keys())
    if len(years) < 2:
        return None

    values = [history[y] for y in years]

    fig, ax = plt.subplots(figsize=figsize)
    trend_color = COLOR_RISK_DOWN if _is_improving(ratio_name, values[0], values[-1]) else COLOR_RISK_UP
    ax.plot(years, values, marker="o", color=trend_color, linewidth=1.8, markersize=4, zorder=3)
    ax.fill_between(years, values, min(values), color=trend_color, alpha=0.08, zorder=1)

    ax.set_title(ratio_name, fontsize=9, color=COLOR_NEUTRAL, loc="left")
    ax.set_xticks(years)
    ax.tick_params(axis="both", labelsize=7)
    ax.grid(axis="y", color=COLOR_GRID, linewidth=0.6, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()

    return _fig_to_png_bytes(fig)


def ratio_trend_grid(
    ratio_history: dict[str, dict[int, float]],
    ratio_names: list[str],
) -> list[tuple[str, bytes]]:
    """Generate trend charts for a fixed set of ratio names, skipping any without enough history."""
    charts = []
    for name in ratio_names:
        history = ratio_history.get(name)
        if not history:
            continue
        img = ratio_trend_chart(name, history)
        if img is not None:
            charts.append((name, img))
    return charts
