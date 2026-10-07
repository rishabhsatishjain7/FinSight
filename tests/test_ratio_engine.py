from ingestion.xbrl_parser import XBRLParser
from transformation.ratio_engine import RATIO_REGISTRY, RatioEngine


def test_registry_has_30_plus_ratios():
    assert len(RATIO_REGISTRY) >= 30


def test_current_ratio_computed_correctly(clean_company_facts):
    parser = XBRLParser()
    cf = parser.parse("TEST", "0000000001", "technology", clean_company_facts)
    engine = RatioEngine()

    ratios_2023 = engine.compute_year(cf, 2023)
    # current_assets=900_000, current_liabilities=430_000
    assert round(ratios_2023["current_ratio"], 4) == round(900_000 / 430_000, 4)


def test_net_margin_computed_correctly(clean_company_facts):
    parser = XBRLParser()
    cf = parser.parse("TEST", "0000000001", "technology", clean_company_facts)
    engine = RatioEngine()

    ratios_2021 = engine.compute_year(cf, 2021)
    # net_income=100_000, revenue=1_000_000
    assert ratios_2021["net_margin"] == 0.1


def test_missing_denominator_returns_none_not_exception(sparse_company_facts):
    parser = XBRLParser()
    cf = parser.parse("SPARSE", "0000000003", "retail", sparse_company_facts)
    engine = RatioEngine()

    ratios = engine.compute_year(cf, 2022)
    assert ratios["debt_to_equity"] is None  # no equity/debt reported at all
    assert ratios["current_ratio"] is None  # no current_liabilities reported


def test_growth_ratio_requires_prior_year(clean_company_facts):
    parser = XBRLParser()
    cf = parser.parse("TEST", "0000000001", "technology", clean_company_facts)
    engine = RatioEngine()

    ratios_2021 = engine.compute_year(cf, 2021)  # no 2020 data available
    assert ratios_2021["revenue_growth_yoy"] is None

    ratios_2022 = engine.compute_year(cf, 2022)
    expected = (1_150_000 - 1_000_000) / 1_000_000
    assert round(ratios_2022["revenue_growth_yoy"], 4) == round(expected, 4)
