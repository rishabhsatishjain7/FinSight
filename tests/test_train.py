import pandas as pd

from scoring.train import (
    DISTRESS_SIGNAL_RATIOS,
    build_feature_matrix,
    build_full_matrix,
    build_training_labels,
)


def _synthetic_z_scores() -> dict:
    """Small synthetic z-score panel spanning strong/weak fundamentals."""
    data = {}
    for i in range(20):
        weak = i < 4  # bottom 20% deliberately weak on the signal ratios
        data[f"T{i}"] = {
            2023: {
                "working_capital_to_assets": -2.0 if weak else 0.5,
                "retained_earnings_to_assets": -1.8 if weak else 0.3,
                "ebit_to_assets": -1.5 if weak else 0.4,
                "equity_to_liabilities": -1.9 if weak else 0.2,
                "sales_to_assets": -1.2 if weak else 0.1,
                "current_ratio": -0.3 if weak else 0.1,  # unrelated ratio, mild noise only
                "net_margin": 0.05,
            }
        }
    return data


def test_build_full_matrix_includes_signal_ratios():
    full_df = build_full_matrix(_synthetic_z_scores())
    for ratio in DISTRESS_SIGNAL_RATIOS:
        assert ratio in full_df.columns


def test_build_feature_matrix_excludes_signal_ratios():
    """
    The core anti-leakage guarantee: none of the ratios used to build the
    label may appear in the matrix actually passed to the model. Without
    this, the model just learns to reconstruct the label formula instead of
    finding independent predictive signal.
    """
    full_df = build_full_matrix(_synthetic_z_scores())
    feature_df = build_feature_matrix(full_df)

    for ratio in DISTRESS_SIGNAL_RATIOS:
        assert ratio not in feature_df.columns

    # Non-signal ratios must still be present
    assert "current_ratio" in feature_df.columns
    assert "net_margin" in feature_df.columns


def test_labels_require_full_matrix_not_feature_matrix():
    """build_training_labels must reject a matrix that's already had the
    signal ratios dropped -- calling it on build_feature_matrix() output
    is the exact bug this test prevents from silently reappearing."""
    full_df = build_full_matrix(_synthetic_z_scores())
    feature_df = build_feature_matrix(full_df)

    import pytest

    with pytest.raises(ValueError):
        build_training_labels(feature_df)


def test_labels_flag_weakest_composite_cohort():
    full_df = build_full_matrix(_synthetic_z_scores())
    labels = build_training_labels(full_df, quantile=0.20)

    # The 4 deliberately "weak" companies (T0-T3) should be flagged distressed
    flagged = set(labels[labels == 1].index.get_level_values("ticker"))
    assert {"T0", "T1", "T2", "T3"}.issubset(flagged)
