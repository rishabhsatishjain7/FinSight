"""
Shared fixtures. Synthetic XBRL companyfacts payloads let the whole
pipeline (parser -> ratios -> Z-scores -> scoring) be tested offline,
without hitting SEC EDGAR — same shape as the real API response, just
hand-built numbers with known expected outputs.
"""
from __future__ import annotations

import pytest


def _facts(tag: str, entries: list[dict], taxonomy: str = "us-gaap") -> dict:
    return {taxonomy: {tag: {"units": {"USD": entries}}}}


def _merge(*fact_dicts: dict) -> dict:
    merged: dict = {"us-gaap": {}, "dei": {}}
    for fd in fact_dicts:
        for taxonomy, tags in fd.items():
            merged.setdefault(taxonomy, {}).update(tags)
    return merged


def _entry(fy: int, val: float, fp: str = "FY", form: str = "10-K", filed: str = None) -> dict:
    return {
        "fy": fy,
        "fp": fp,
        "form": form,
        "val": val,
        "filed": filed or f"{fy + 1}-02-15",
        "start": f"{fy - 1}-01-01",
        "end": f"{fy}-12-31",
    }


@pytest.fixture
def clean_company_facts() -> dict:
    """A company that reports consistently under modern GAAP tags, 3 years."""
    facts = _merge(
        _facts("RevenueFromContractWithCustomerExcludingAssessedTax", [
            _entry(2021, 1_000_000), _entry(2022, 1_150_000), _entry(2023, 1_300_000),
        ]),
        _facts("CostOfRevenue", [
            _entry(2021, 600_000), _entry(2022, 670_000), _entry(2023, 720_000),
        ]),
        _facts("GrossProfit", [
            _entry(2021, 400_000), _entry(2022, 480_000), _entry(2023, 580_000),
        ]),
        _facts("OperatingIncomeLoss", [
            _entry(2021, 150_000), _entry(2022, 190_000), _entry(2023, 240_000),
        ]),
        _facts("NetIncomeLoss", [
            _entry(2021, 100_000), _entry(2022, 130_000), _entry(2023, 170_000),
        ]),
        _facts("Assets", [
            _entry(2021, 2_000_000), _entry(2022, 2_200_000), _entry(2023, 2_450_000),
        ]),
        _facts("AssetsCurrent", [
            _entry(2021, 800_000), _entry(2022, 850_000), _entry(2023, 900_000),
        ]),
        _facts("LiabilitiesCurrent", [
            _entry(2021, 400_000), _entry(2022, 420_000), _entry(2023, 430_000),
        ]),
        _facts("Liabilities", [
            _entry(2021, 900_000), _entry(2022, 950_000), _entry(2023, 1_000_000),
        ]),
        _facts("StockholdersEquity", [
            _entry(2021, 1_100_000), _entry(2022, 1_250_000), _entry(2023, 1_450_000),
        ]),
        _facts("RetainedEarningsAccumulatedDeficit", [
            _entry(2021, 500_000), _entry(2022, 600_000), _entry(2023, 730_000),
        ]),
    )
    return {"facts": facts}


@pytest.fixture
def schema_evolution_company_facts() -> dict:
    """
    A company that reports revenue under the LEGACY tag pre-2018-style
    (SalesRevenueNet) for early years and switches to the MODERN ASC 606
    tag for later years -- exactly the multi-tag fallback / schema
    evolution scenario the parser exists to handle. No single tag alone
    covers all three years.
    """
    facts = _merge(
        _facts("SalesRevenueNet", [_entry(2016, 500_000), _entry(2017, 540_000)]),
        _facts("RevenueFromContractWithCustomerExcludingAssessedTax", [_entry(2018, 610_000)]),
        _facts("Assets", [_entry(2016, 1_000_000), _entry(2017, 1_050_000), _entry(2018, 1_100_000)]),
    )
    return {"facts": facts}


@pytest.fixture
def restated_company_facts() -> dict:
    """
    A company where FY2021 revenue was originally reported as 900,000 in
    the FY2021 10-K, then restated to 950,000 when it reappeared as the
    prior-year comparative figure in the FY2022 10-K (filed later). The
    parser must surface the restated (later-filed) value, not the original.
    """
    facts = _merge(
        _facts("Revenues", [
            _entry(2021, 900_000, filed="2022-02-10"),   # original FY2021 filing
            _entry(2021, 950_000, filed="2023-02-15"),   # restated, reported again in FY2022 10-K
            _entry(2022, 1_200_000, filed="2023-02-15"),
        ]),
        _facts("Assets", [
            _entry(2021, 1_500_000, filed="2022-02-10"),
            _entry(2022, 1_650_000, filed="2023-02-15"),
        ]),
    )
    return {"facts": facts}


@pytest.fixture
def non_calendar_fye_company_facts() -> dict:
    """
    A retailer-style filer with a January 31 fiscal year end (e.g.
    Walmart's real FYE). SEC labels the year ending 2023-01-31 as fy=2023,
    but 11 of that period's 12 months are actually calendar 2022 --
    calendar alignment should re-bucket it to 2022. A year ending
    2022-01-31 (fy=2022) should align to 2021.
    """
    facts = _merge(
        _facts("Revenues", [
            {"fy": 2022, "fp": "FY", "form": "10-K", "val": 500_000, "filed": "2022-03-15",
             "start": "2021-02-01", "end": "2022-01-31"},
            {"fy": 2023, "fp": "FY", "form": "10-K", "val": 560_000, "filed": "2023-03-15",
             "start": "2022-02-01", "end": "2023-01-31"},
        ]),
        _facts("Assets", [
            {"fy": 2022, "fp": "FY", "form": "10-K", "val": 900_000, "filed": "2022-03-15",
             "start": "2021-02-01", "end": "2022-01-31"},
            {"fy": 2023, "fp": "FY", "form": "10-K", "val": 950_000, "filed": "2023-03-15",
             "start": "2022-02-01", "end": "2023-01-31"},
        ]),
    )
    return {"facts": facts}


@pytest.fixture
def foreign_currency_only_company_facts() -> dict:
    """
    A filer that reports revenue only in EUR for 2021 (no USD entry at
    all that year -- e.g. a foreign private issuer's early filing before
    USD reporting was added), then switches to USD-denominated reporting
    from 2022 onward. The parser must NOT silently treat the EUR 2021
    figure as if it were USD.
    """
    facts = {
        "us-gaap": {
            "Revenues": {
                "units": {
                    "EUR": [
                        {"fy": 2021, "fp": "FY", "form": "10-K", "val": 400_000, "filed": "2022-02-15",
                         "start": "2020-01-01", "end": "2021-12-31"},
                    ],
                    "USD": [
                        {"fy": 2022, "fp": "FY", "form": "10-K", "val": 550_000, "filed": "2023-02-15",
                         "start": "2021-01-01", "end": "2022-12-31"},
                    ],
                }
            }
        }
    }
    return {"facts": facts}


@pytest.fixture
def sparse_company_facts() -> dict:
    """A company with large gaps in coverage -- ratios touching missing fields must return None, not raise."""
    facts = _merge(
        _facts("Revenues", [_entry(2022, 300_000)]),
        _facts("Assets", [_entry(2022, 900_000)]),
        # No liabilities, no equity, no cost of revenue reported at all.
    )
    return {"facts": facts}
