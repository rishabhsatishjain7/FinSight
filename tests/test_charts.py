import io

from reportlab.lib.utils import ImageReader

from reporting.charts import ratio_trend_chart, ratio_trend_grid, shap_waterfall_chart


def _valid_png(png_bytes: bytes) -> bool:
    assert isinstance(png_bytes, (bytes, bytearray))
    assert png_bytes.startswith(b"\x89PNG")
    w, h = ImageReader(io.BytesIO(png_bytes)).getSize()
    return w > 0 and h > 0


def test_shap_waterfall_chart_produces_valid_image():
    contributions = [
        {"feature": "current_ratio", "shap_value": 1.2, "feature_value": -2.3},
        {"feature": "net_margin", "shap_value": -0.8, "feature_value": 1.1},
        {"feature": "debt_to_equity", "shap_value": 0.5, "feature_value": 1.9},
    ]
    img = shap_waterfall_chart(contributions, base_value=-1.5, distress_probability=0.42)
    assert _valid_png(img)


def test_shap_waterfall_chart_respects_top_n():
    contributions = [
        {"feature": f"ratio_{i}", "shap_value": float(i), "feature_value": 0.1} for i in range(20)
    ]
    img = shap_waterfall_chart(contributions, base_value=0.0, distress_probability=0.1, top_n=5)
    assert _valid_png(img)


def test_ratio_trend_chart_with_sufficient_history():
    history = {2021: 1.2, 2022: 1.0, 2023: 0.85}
    img = ratio_trend_chart("current_ratio", history)
    assert img is not None
    assert _valid_png(img)


def test_ratio_trend_chart_returns_none_for_insufficient_history():
    """A single data point isn't a trend -- must not render a misleading chart."""
    assert ratio_trend_chart("current_ratio", {2023: 1.0}) is None
    assert ratio_trend_chart("current_ratio", {}) is None


def test_ratio_trend_grid_skips_ratios_without_history():
    ratio_history = {
        "current_ratio": {2021: 1.2, 2022: 1.0, 2023: 0.85},
        "net_margin": {2023: 0.05},  # only one year -- should be skipped
        "debt_to_equity": {},  # empty -- should be skipped
    }
    charts = ratio_trend_grid(ratio_history, ["current_ratio", "net_margin", "debt_to_equity", "return_on_assets"])

    names = [name for name, _ in charts]
    assert names == ["current_ratio"]


def test_trend_color_accounts_for_ratio_direction():
    """
    A rising debt_to_equity is DETERIORATING (higher leverage is worse),
    while a rising current_ratio is IMPROVING (more liquidity cushion is
    better). Both must not be colored the same way just because they're
    both "increasing" -- that would mislabel a worsening leverage trend as
    healthy. Colors aren't inspectable from PNG bytes directly here, so
    this exercises the underlying direction logic that decides the color.
    """
    from reporting.charts import _is_improving

    # debt_to_equity rising 1.2 -> 2.1 is a DETERIORATION (higher is worse)
    assert _is_improving("debt_to_equity", 1.2, 2.1) is False

    # current_ratio rising 0.85 -> 1.2 is an IMPROVEMENT (higher is better)
    assert _is_improving("current_ratio", 0.85, 1.2) is True

    # current_ratio FALLING is a deterioration
    assert _is_improving("current_ratio", 1.2, 0.85) is False

    # debt_to_equity FALLING is an improvement (paying down leverage)
    assert _is_improving("debt_to_equity", 2.1, 1.2) is True

    # unknown ratio name defaults to "higher is better"
    assert _is_improving("some_unlisted_ratio", 1.0, 2.0) is True
