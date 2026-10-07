"""
FinSight — XBRL parser: multi-tag fallback + schema evolution handling.

The problem this solves:
    US-GAAP taxonomy tags are not stable across time or across filers.
    "Revenue" alone might show up under any of:
        Revenues
        RevenueFromContractWithCustomerExcludingAssessedTax   (post-ASC 606, 2018+)
        RevenueFromContractWithCustomerIncludingAssessedTax
        SalesRevenueNet                                       (pre-2018, legacy)
        SalesRevenueGoodsNet
    A pipeline that hardcodes one tag silently drops years of history or
    entire filers the moment the taxonomy shifts. This module instead
    defines each canonical line item as an ORDERED list of candidate tags
    and walks the list per fiscal year, taking the first tag that has data
    for that period. This is the "multi-tag fallback" referenced in the
    resume bullet.

    Schema evolution is handled by keying everything off (fiscal_year,
    canonical_field) rather than off any single tag name, so a company
    that migrates tags mid-history still produces one continuous series.

    Restatements are handled separately: a single tag can carry more than
    one reported value for the same fiscal year (the original figure, plus
    a later restated figure reported in a subsequent filing's comparative
    columns). For each tag, we keep only the entry with the latest "filed"
    date per fiscal year before merging across the fallback chain, so the
    parser reflects the most recently reported figure rather than whichever
    entry happens to appear first in SEC's JSON array.

    Currency-unit safety: SEC's XBRL API reports monetary values in raw
    (unscaled) units of whatever currency the filer used -- Apple's FY2023
    revenue really is 383285000000, not "in thousands" (verified against
    SEC's own API documentation and sample payloads; earlier project notes
    assumed a thousands-rescaling issue that turned out not to apply to
    this specific endpoint). The real currency risk is different: a tag's
    `units` object can contain more than one currency (SEC's docs give the
    example of a company reporting "net profits ... in U.S. dollars and in
    Canadian dollars"), and the original implementation silently fell back
    to *whatever unit happened to be present* if "USD" wasn't one of them
    -- which would silently treat a EUR or CAD figure as if it were USD.
    Monetary canonical fields (everything except explicit share-count
    fields) now require an actual "USD" entry; if a tag only has non-USD
    monetary data for a given filer/year, that tag is skipped for that
    year rather than guessed at, so the fallback chain can try the next
    candidate tag (or the year is left unresolved, rather than silently
    wrong).

    Fiscal-year calendar alignment: not every filer's fiscal year ends in
    December (Walmart's ends January 31; many retailers report FYE
    late-January/early-February). Grouping strictly by SEC's own `fy` label
    can misalign filers relative to calendar-year peers by several months
    -- e.g. a company's fy=2023 that actually ended 2023-01-31 covers
    mostly *calendar* 2022. `_calendar_align_year` re-buckets each resolved
    value by the calendar year containing the majority of its fiscal
    period (using the period end date), so cross-company ratio comparisons
    and sector Z-scores line up on economically comparable periods rather
    than each filer's own label. The originally reported SEC fiscal year
    and the period end date are preserved on every LineItem for
    traceability. This can be disabled via `parse(..., align_fiscal_years=False)`
    to use SEC's raw fy labels unchanged.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger("finsight.ingestion.xbrl_parser")

# ---------------------------------------------------------------------------
# Canonical line items -> ordered candidate GAAP tags (highest priority first)
# ---------------------------------------------------------------------------
TAG_FALLBACK_MAP: dict[str, list[str]] = {
    "revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",
    ],
    "cost_of_revenue": [
        "CostOfRevenue",
        "CostOfGoodsAndServicesSold",
        "CostOfGoodsSold",
        "CostOfServices",
    ],
    "gross_profit": ["GrossProfit"],
    "operating_income": [
        "OperatingIncomeLoss",
    ],
    "net_income": [
        "NetIncomeLoss",
        "ProfitLoss",
        "NetIncomeLossAvailableToCommonStockholdersBasic",
    ],
    "ebit": ["OperatingIncomeLoss"],
    "interest_expense": [
        "InterestExpense",
        "InterestExpenseDebt",
        "InterestIncomeExpenseNet",
    ],
    "total_assets": ["Assets"],
    "current_assets": ["AssetsCurrent"],
    "cash_and_equivalents": [
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
        "Cash",
    ],
    "inventory": ["InventoryNet"],
    "receivables": [
        "AccountsReceivableNetCurrent",
        "ReceivablesNetCurrent",
    ],
    "total_liabilities": ["Liabilities"],
    "current_liabilities": ["LiabilitiesCurrent"],
    "long_term_debt": [
        "LongTermDebtNoncurrent",
        "LongTermDebt",
        "LongTermDebtAndCapitalLeaseObligations",
    ],
    "short_term_debt": [
        "ShortTermBorrowings",
        "DebtCurrent",
        "LongTermDebtCurrent",
    ],
    "total_equity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
    "retained_earnings": ["RetainedEarningsAccumulatedDeficit"],
    "operating_cash_flow": [
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ],
    "capex": [
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsForCapitalImprovements",
    ],
    "depreciation_amortization": [
        "DepreciationDepletionAndAmortization",
        "DepreciationAmortizationAndAccretionNet",
        "Depreciation",
    ],
    "shares_outstanding": [
        "CommonStockSharesOutstanding",
        "WeightedAverageNumberOfDilutedSharesOutstanding",
        "WeightedAverageNumberOfSharesOutstandingBasic",
    ],
    "market_cap_proxy_shares": [
        "EntityCommonStockSharesOutstanding",
    ],
    "dividends_paid": ["PaymentsOfDividends", "PaymentsOfDividendsCommonStock"],
    "working_capital_current_liab": ["LiabilitiesCurrent"],
}

TAXONOMY_ORDER = ["us-gaap", "ifrs-full", "dei"]  # search order across taxonomies

# Canonical fields that are share counts, not currency amounts -- these
# legitimately use a "shares" unit rather than "USD" and shouldn't be held
# to the USD-only requirement applied to every other (monetary) field.
NON_MONETARY_FIELDS = {"shares_outstanding", "market_cap_proxy_shares"}


@dataclass
class LineItem:
    fiscal_year: int  # calendar-aligned year (see module docstring), used as the series key
    fiscal_period: str  # "FY", "Q1", "Q2", "Q3"
    value: float
    tag_used: str
    unit: str
    filed: str
    form: str
    reported_fiscal_year: int | None = None  # SEC's own `fy` label, pre-alignment
    period_end: str = ""  # raw "end" date from the XBRL entry, e.g. "2023-01-31"


@dataclass
class CompanyFinancials:
    ticker: str
    cik: str
    sector: str
    series: dict[str, dict[int, LineItem]] = field(default_factory=dict)
    tag_resolution_log: list[dict] = field(default_factory=list)

    def get(self, field_name: str, fiscal_year: int) -> float | None:
        item = self.series.get(field_name, {}).get(fiscal_year)
        return item.value if item else None

    def years_available(self) -> list[int]:
        years: set[int] = set()
        for field_series in self.series.values():
            years.update(field_series.keys())
        return sorted(years)


class XBRLParser:
    """
    Resolves canonical financial line items from raw SEC companyfacts JSON,
    applying multi-tag fallback and normalizing to one row per fiscal year.
    """

    def __init__(self, tag_map: dict[str, list[str]] = None):
        self.tag_map = tag_map or TAG_FALLBACK_MAP

    def parse(
        self,
        ticker: str,
        cik: str,
        sector: str,
        raw_facts: dict,
        annual_only: bool = True,
        align_fiscal_years: bool = True,
    ) -> CompanyFinancials:
        result = CompanyFinancials(ticker=ticker, cik=cik, sector=sector)
        facts_by_taxonomy = raw_facts.get("facts", {})

        for canonical_field, candidate_tags in self.tag_map.items():
            is_monetary = canonical_field not in NON_MONETARY_FIELDS
            resolved_series, tag_used, taxonomy_used = self._resolve_field(
                facts_by_taxonomy, candidate_tags, annual_only, is_monetary, align_fiscal_years
            )
            if resolved_series:
                result.series[canonical_field] = resolved_series
                result.tag_resolution_log.append(
                    {
                        "field": canonical_field,
                        "tag_used": tag_used,
                        "taxonomy": taxonomy_used,
                        "years_resolved": sorted(resolved_series.keys()),
                    }
                )
            else:
                logger.debug(
                    "No tag in fallback chain resolved for %s (%s): tried %s",
                    canonical_field,
                    ticker,
                    candidate_tags,
                )

        return result

    def _resolve_field(
        self,
        facts_by_taxonomy: dict,
        candidate_tags: list[str],
        annual_only: bool,
        is_monetary: bool,
        align_fiscal_years: bool,
    ) -> tuple[dict[int, LineItem], str | None, str | None]:
        """
        Walk candidate tags in priority order. The FIRST tag that yields any
        usable data wins for years it covers; if that tag has gaps, later
        tags in the fallback chain are used to fill those specific years
        (schema evolution: a filer switching tags mid-history still yields
        one continuous series instead of two partial ones).

        Restatements: SEC filers frequently re-report a prior fiscal year's
        figure in a later filing (e.g. a 2022 revenue figure appears both in
        the FY2022 10-K and again, restated, in the FY2023 10-K's
        comparative prior-year column). A single tag can therefore carry
        MULTIPLE entries for the same fiscal year. Within each tag, we first
        resolve exactly one value per fiscal year by keeping whichever entry
        has the latest "filed" date — the most recently reported figure for
        that year — rather than whichever entry happens to appear first in
        the raw JSON array (which is filing order, not correctness order).
        Restatement resolution operates on SEC's raw fy label (the reporting
        period identity), before any calendar-year re-bucketing below.
        """
        merged: dict[int, LineItem] = {}
        winning_tag = None
        winning_taxonomy = None

        for tag in candidate_tags:
            # fy -> (entry, taxonomy, unit_key), resolved for restatements
            best_per_year: dict[int, tuple[dict, str, str]] = {}

            for taxonomy in TAXONOMY_ORDER:
                tag_data = facts_by_taxonomy.get(taxonomy, {}).get(tag)
                if not tag_data:
                    continue

                units = tag_data.get("units", {})
                if is_monetary:
                    # Monetary fields must be an actual USD-denominated
                    # entry -- never silently substitute another currency
                    # a filer might report alongside it (see module
                    # docstring, "Currency-unit safety").
                    if "USD" not in units:
                        continue
                    unit_key = "USD"
                else:
                    unit_key = next(iter(units), None)  # e.g. "shares"
                    if unit_key is None:
                        continue

                for entry in units[unit_key]:
                    fy = entry.get("fy")
                    fp = entry.get("fp")
                    form = entry.get("form", "")
                    if fy is None:
                        continue
                    if annual_only and fp != "FY":
                        continue
                    if annual_only and "10-K" not in form:
                        continue

                    existing = best_per_year.get(fy)
                    if existing is None or self._is_more_recent_filing(entry, existing[0]):
                        best_per_year[fy] = (entry, taxonomy, unit_key)

            for fy, (entry, taxonomy, unit_key) in best_per_year.items():
                target_year = (
                    _calendar_align_year(entry.get("end"), fy) if align_fiscal_years else fy
                )

                # Don't overwrite a year already filled by a higher-priority tag
                if target_year in merged:
                    continue

                merged[target_year] = LineItem(
                    fiscal_year=target_year,
                    fiscal_period=entry.get("fp") or "FY",
                    value=float(entry["val"]),
                    tag_used=tag,
                    unit=unit_key,
                    filed=entry.get("filed", ""),
                    form=entry.get("form", ""),
                    reported_fiscal_year=fy,
                    period_end=entry.get("end", ""),
                )
                if winning_tag is None:
                    winning_tag = tag
                    winning_taxonomy = taxonomy

        return merged, winning_tag, winning_taxonomy

    @staticmethod
    def _is_more_recent_filing(candidate: dict, current_best: dict) -> bool:
        """
        True if `candidate` was filed later than `current_best` and should
        replace it as the value for that fiscal year (i.e. candidate is a
        restatement of current_best). Filed dates are ISO "YYYY-MM-DD"
        strings, which sort correctly as plain strings. Entries missing a
        "filed" date never displace an existing resolved value.
        """
        candidate_filed = candidate.get("filed")
        current_filed = current_best.get("filed")
        if not candidate_filed:
            return False
        if not current_filed:
            return True
        return candidate_filed > current_filed


def _calendar_align_year(period_end: str | None, reported_fiscal_year: int) -> int:
    """
    Re-buckets a fiscal year by the calendar year containing the majority
    of its ~12-month period, using the period end date.

    Heuristic: a fiscal year ending in January-May covers mostly the
    PRIOR calendar year (e.g. FYE 2023-01-31 covers 2022-02-01 through
    2023-01-31 -- 11 of 12 months are in 2022), so it's labeled year-1.
    A fiscal year ending in June-December covers mostly the SAME calendar
    year it ends in (e.g. FYE 2023-09-30 covers 2022-10-01 through
    2023-09-30 -- 9 of 12 months are in 2023), so it's labeled that year.
    June is a genuine 6/6 tie, broken toward the ending year by convention.

    This is an approximation, not an exact overlap calculation -- it's
    accurate for the standard ~12-month annual reporting period every
    company here uses, but would need adjusting for a genuine short/stub
    fiscal year (e.g. following a fiscal-year-end change).

    Falls back to `reported_fiscal_year` unchanged if `period_end` is
    missing or unparseable, so malformed data degrades to the old
    (SEC-label) behavior rather than raising.
    """
    if not period_end:
        return reported_fiscal_year
    try:
        year_str, month_str, _ = period_end.split("-")
        year, month = int(year_str), int(month_str)
    except (ValueError, AttributeError):
        return reported_fiscal_year
    return year - 1 if month <= 5 else year
