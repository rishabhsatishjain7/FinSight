from transformation.zscore_benchmark import ZScoreBenchmark
import pytest


def test_zscore_zero_when_at_sector_mean():
    ratios = {
        "A": {2023: {"net_margin": 0.10}},
        "B": {2023: {"net_margin": 0.20}},
        "C": {2023: {"net_margin": 0.30}},
    }
    sectors = {"A": "tech", "B": "tech", "C": "tech"}

    z = ZScoreBenchmark(min_peers=3).score_all(ratios, sectors)
    # mean = 0.20, B sits exactly on the mean -> z = 0
    assert z["B"][2023]["net_margin"] == 0.0


def test_below_min_peer_threshold_returns_none():
    ratios = {
        "A": {2023: {"net_margin": 0.10}},
        "B": {2023: {"net_margin": 0.20}},
    }
    sectors = {"A": "tech", "B": "tech"}

    z = ZScoreBenchmark(min_peers=3).score_all(ratios, sectors)
    # Only 2 peers, below min_peers=3 -> stats not trusted -> None
    assert z["A"][2023]["net_margin"] is None


def test_companies_only_compared_within_same_sector():
    ratios = {
        "A": {2023: {"net_margin": 0.10}},
        "B": {2023: {"net_margin": 0.20}},
        "C": {2023: {"net_margin": 0.30}},
        "X": {2023: {"net_margin": 0.90}},  # different sector, should not pollute tech stats
        "Y": {2023: {"net_margin": 0.95}},
        "Z": {2023: {"net_margin": 0.99}},
    }
    sectors = {"A": "tech", "B": "tech", "C": "tech", "X": "energy", "Y": "energy", "Z": "energy"}

    z = ZScoreBenchmark(min_peers=3).score_all(ratios, sectors)
    # tech mean should still be 0.20, unaffected by energy's much higher margins
    assert z["B"][2023]["net_margin"] == 0.0


def test_missing_value_yields_none_zscore():
    ratios = {
        "A": {2023: {"net_margin": 0.10}},
        "B": {2023: {"net_margin": None}},
        "C": {2023: {"net_margin": 0.30}},
        "D": {2023: {"net_margin": 0.20}},
    }
    sectors = {"A": "tech", "B": "tech", "C": "tech", "D": "tech"}

    z = ZScoreBenchmark(min_peers=3).score_all(ratios, sectors)
    assert z["B"][2023]["net_margin"] is None


def test_robust_method_is_default():
    zb = ZScoreBenchmark(min_peers=3)
    assert zb.method == "robust"


def test_robust_method_resists_single_outlier_distortion():
    """
    The core fix: A, B, C have genuinely different (and fairly close)
    margins -- 0.10, 0.12, 0.11 -- while D is a wild outlier at 100 (e.g. a
    data quality issue or a company with an extreme one-off gain). Under
    the classic mean/stdev method, D inflates the sector stdev so much that
    A, B, and C's Z-scores collapse to nearly identical values, destroying
    the very distinctions a distress screen depends on. The robust
    (median/MAD) method should keep A, B, C meaningfully separated.
    """
    ratios = {
        "A": {2023: {"net_margin": 0.10}},
        "B": {2023: {"net_margin": 0.12}},
        "C": {2023: {"net_margin": 0.11}},
        "D": {2023: {"net_margin": 100.0}},  # extreme outlier
    }
    sectors = {"A": "tech", "B": "tech", "C": "tech", "D": "tech"}

    z_standard = ZScoreBenchmark(min_peers=3, method="standard").score_all(ratios, sectors)
    z_robust = ZScoreBenchmark(min_peers=3, method="robust").score_all(ratios, sectors)

    # Standard method: A and B's z-scores are nearly indistinguishable --
    # the outlier's inflated stdev swamps the real difference between them.
    standard_gap = abs(z_standard["A"][2023]["net_margin"] - z_standard["B"][2023]["net_margin"])
    assert standard_gap < 0.02

    # Robust method: same two companies remain clearly distinguishable.
    robust_gap = abs(z_robust["A"][2023]["net_margin"] - z_robust["B"][2023]["net_margin"])
    assert robust_gap > 0.5

    # And robust A/B/C ordering matches the actual ratio ordering (A < C < B).
    za = z_robust["A"][2023]["net_margin"]
    zb = z_robust["B"][2023]["net_margin"]
    zc = z_robust["C"][2023]["net_margin"]
    assert za < zc < zb


def test_standard_method_still_available_and_matches_classic_formula():
    """The old mean/stdev behavior remains selectable via method='standard'."""
    ratios = {
        "A": {2023: {"net_margin": 0.10}},
        "B": {2023: {"net_margin": 0.20}},
        "C": {2023: {"net_margin": 0.30}},
    }
    sectors = {"A": "tech", "B": "tech", "C": "tech"}

    z = ZScoreBenchmark(min_peers=3, method="standard").score_all(ratios, sectors)
    # mean=0.20, pstdev of [0.10,0.20,0.30] = sqrt(mean((x-0.2)^2)) = sqrt(0.00667) ~= 0.08165
    assert z["A"][2023]["net_margin"] == pytest.approx(-1.2247, rel=1e-3)
