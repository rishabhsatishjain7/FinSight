from ingestion.xbrl_parser import XBRLParser


def test_clean_company_resolves_all_core_fields(clean_company_facts):
    parser = XBRLParser()
    cf = parser.parse("TEST", "0000000001", "technology", clean_company_facts)

    assert cf.get("revenue", 2023) == 1_300_000
    assert cf.get("net_income", 2022) == 130_000
    assert cf.years_available() == [2021, 2022, 2023]


def test_schema_evolution_produces_continuous_series(schema_evolution_company_facts):
    """
    The core regression case: revenue is tagged SalesRevenueNet in 2016-2017
    and RevenueFromContractWithCustomerExcludingAssessedTax in 2018. A naive
    single-tag lookup would return revenue for only one of the two eras.
    The parser must merge both into one continuous 3-year series.
    """
    parser = XBRLParser()
    cf = parser.parse("EVOLVE", "0000000002", "industrials", schema_evolution_company_facts)

    assert cf.get("revenue", 2016) == 500_000
    assert cf.get("revenue", 2017) == 540_000
    assert cf.get("revenue", 2018) == 610_000
    assert cf.years_available() == [2016, 2017, 2018]


def test_tag_resolution_log_records_which_tag_won(schema_evolution_company_facts):
    parser = XBRLParser()
    cf = parser.parse("EVOLVE", "0000000002", "industrials", schema_evolution_company_facts)

    revenue_log = next(e for e in cf.tag_resolution_log if e["field"] == "revenue")
    # First candidate tag in the fallback chain that resolved ANY data wins
    # as the "primary" tag_used, even though later years used a different tag.
    assert revenue_log["tag_used"] in (
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet",
    )
    assert set(revenue_log["years_resolved"]) == {2016, 2017, 2018}


def test_restatement_prefers_most_recently_filed_value(restated_company_facts):
    """
    FY2021 revenue was reported twice under the same tag: 900,000 (filed
    2022-02-10, the original 10-K) and 950,000 (filed 2023-02-15, restated
    as the prior-year comparative in the FY2022 10-K). The parser must
    surface the later-filed, restated figure.
    """
    parser = XBRLParser()
    cf = parser.parse("RESTATE", "0000000005", "retail", restated_company_facts)

    assert cf.get("revenue", 2021) == 950_000  # restated value, not the original 900,000
    assert cf.get("revenue", 2022) == 1_200_000


def test_missing_fields_are_simply_absent_not_raised(sparse_company_facts):
    parser = XBRLParser()
    cf = parser.parse("SPARSE", "0000000003", "retail", sparse_company_facts)

    assert cf.get("revenue", 2022) == 300_000
    assert cf.get("total_liabilities", 2022) is None  # never reported, must not raise
    assert cf.get("total_equity", 2022) is None


def test_no_tag_in_chain_present_returns_empty_series():
    parser = XBRLParser()
    cf = parser.parse("EMPTY", "0000000004", "energy", {"facts": {}})
    assert cf.years_available() == []
    assert cf.get("revenue", 2023) is None


def test_calendar_alignment_rebuckets_non_december_fye(non_calendar_fye_company_facts):
    """
    A January 31 FYE filer's SEC-labeled fy=2023 (period ending 2023-01-31)
    covers mostly calendar 2022 and should be re-bucketed there; fy=2022
    (ending 2022-01-31) should align to 2021.
    """
    parser = XBRLParser()
    cf = parser.parse("JANFYE", "0000000006", "retail", non_calendar_fye_company_facts)

    assert cf.years_available() == [2021, 2022]
    assert cf.get("revenue", 2021) == 500_000  # SEC fy=2022, aligned to calendar 2021
    assert cf.get("revenue", 2022) == 560_000  # SEC fy=2023, aligned to calendar 2022

    # The original SEC label and period end date are preserved for traceability.
    line_item = cf.series["revenue"][2022]
    assert line_item.reported_fiscal_year == 2023
    assert line_item.period_end == "2023-01-31"


def test_calendar_alignment_can_be_disabled(non_calendar_fye_company_facts):
    """align_fiscal_years=False preserves SEC's raw fy labels unchanged."""
    parser = XBRLParser()
    cf = parser.parse(
        "JANFYE", "0000000006", "retail", non_calendar_fye_company_facts, align_fiscal_years=False
    )
    assert cf.years_available() == [2022, 2023]
    assert cf.get("revenue", 2023) == 560_000


def test_december_fye_unaffected_by_calendar_alignment(clean_company_facts):
    """Standard Dec-31 FYE filers should be identical with alignment on or off."""
    parser = XBRLParser()
    aligned = parser.parse("TEST", "0000000001", "technology", clean_company_facts)
    unaligned = parser.parse(
        "TEST", "0000000001", "technology", clean_company_facts, align_fiscal_years=False
    )
    assert aligned.years_available() == unaligned.years_available() == [2021, 2022, 2023]
    assert aligned.get("revenue", 2023) == unaligned.get("revenue", 2023)


def test_foreign_currency_only_year_is_not_silently_used_as_usd(foreign_currency_only_company_facts):
    """
    RC-010 regression: a tag with ONLY a EUR entry for 2021 (no USD at
    all) must not be resolved as if the EUR figure were USD. The 2021
    revenue should simply be absent; only the genuinely USD-denominated
    2022 figure should resolve.
    """
    parser = XBRLParser()
    cf = parser.parse("EURCO", "0000000007", "technology", foreign_currency_only_company_facts)

    assert cf.get("revenue", 2021) is None  # EUR-only year must NOT resolve
    assert cf.get("revenue", 2022) == 550_000  # genuine USD year resolves normally
    assert cf.series["revenue"][2022].unit == "USD"


def test_non_monetary_field_still_resolves_via_shares_unit():
    """Share-count fields legitimately use a 'shares' unit, not USD -- the
    USD-only requirement added for monetary fields must not affect them."""
    facts = {
        "us-gaap": {
            "CommonStockSharesOutstanding": {
                "units": {
                    "shares": [
                        {"fy": 2023, "fp": "FY", "form": "10-K", "val": 1_000_000,
                         "filed": "2024-02-15", "start": "2023-01-01", "end": "2023-12-31"},
                    ]
                }
            }
        }
    }
    parser = XBRLParser()
    cf = parser.parse("SHARECO", "0000000008", "technology", {"facts": facts})
    assert cf.get("shares_outstanding", 2023) == 1_000_000
