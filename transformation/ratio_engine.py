"""
FinSight — ratio computation engine.

Computes 30+ financial ratios per company per fiscal year from the
canonical line items produced by ingestion.xbrl_parser. Designed for
extensibility: every ratio is a small function registered in RATIO_REGISTRY,
so adding coverage for a new company requires zero changes here — only a
config entry in companies.yaml. Adding a new ratio is a one-function change.

All ratios degrade gracefully to None (rather than raising) when a required
line item is missing for that fiscal year, since XBRL coverage is uneven
across filers/years.
"""
from __future__ import annotations

from typing import Callable

from ingestion.xbrl_parser import CompanyFinancials

RatioFn = Callable[[CompanyFinancials, int], float | None]


def _safe_div(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


def _prev(cf: CompanyFinancials, field: str, year: int) -> float | None:
    return cf.get(field, year - 1)


def _avg(cf: CompanyFinancials, field: str, year: int) -> float | None:
    cur = cf.get(field, year)
    prev = _prev(cf, field, year)
    if cur is None or prev is None:
        return cur
    return (cur + prev) / 2


# ---------------------------------------------------------------------------
# Ratio definitions, grouped by category
# ---------------------------------------------------------------------------

def _liquidity_ratios() -> dict[str, RatioFn]:
    return {
        "current_ratio": lambda cf, y: _safe_div(
            cf.get("current_assets", y), cf.get("current_liabilities", y)
        ),
        "quick_ratio": lambda cf, y: _safe_div(
            _sub(cf.get("current_assets", y), cf.get("inventory", y)),
            cf.get("current_liabilities", y),
        ),
        "cash_ratio": lambda cf, y: _safe_div(
            cf.get("cash_and_equivalents", y), cf.get("current_liabilities", y)
        ),
        "working_capital": lambda cf, y: _sub(
            cf.get("current_assets", y), cf.get("current_liabilities", y)
        ),
        "working_capital_ratio": lambda cf, y: _safe_div(
            _sub(cf.get("current_assets", y), cf.get("current_liabilities", y)),
            cf.get("total_assets", y),
        ),
        "operating_cash_flow_ratio": lambda cf, y: _safe_div(
            cf.get("operating_cash_flow", y), cf.get("current_liabilities", y)
        ),
    }


def _leverage_ratios() -> dict[str, RatioFn]:
    return {
        "debt_to_equity": lambda cf, y: _safe_div(
            _total_debt(cf, y), cf.get("total_equity", y)
        ),
        "debt_to_assets": lambda cf, y: _safe_div(
            _total_debt(cf, y), cf.get("total_assets", y)
        ),
        "liabilities_to_assets": lambda cf, y: _safe_div(
            cf.get("total_liabilities", y), cf.get("total_assets", y)
        ),
        "equity_multiplier": lambda cf, y: _safe_div(
            cf.get("total_assets", y), cf.get("total_equity", y)
        ),
        "interest_coverage": lambda cf, y: _safe_div(
            cf.get("ebit", y), cf.get("interest_expense", y)
        ),
        "long_term_debt_to_equity": lambda cf, y: _safe_div(
            cf.get("long_term_debt", y), cf.get("total_equity", y)
        ),
        "long_term_debt_to_capitalization": lambda cf, y: _safe_div(
            cf.get("long_term_debt", y),
            _add(cf.get("long_term_debt", y), cf.get("total_equity", y)),
        ),
    }


def _profitability_ratios() -> dict[str, RatioFn]:
    return {
        "gross_margin": lambda cf, y: _safe_div(cf.get("gross_profit", y), cf.get("revenue", y)),
        "operating_margin": lambda cf, y: _safe_div(
            cf.get("operating_income", y), cf.get("revenue", y)
        ),
        "net_margin": lambda cf, y: _safe_div(cf.get("net_income", y), cf.get("revenue", y)),
        "return_on_assets": lambda cf, y: _safe_div(
            cf.get("net_income", y), _avg(cf, "total_assets", y)
        ),
        "return_on_equity": lambda cf, y: _safe_div(
            cf.get("net_income", y), _avg(cf, "total_equity", y)
        ),
        "return_on_invested_capital": lambda cf, y: _safe_div(
            cf.get("ebit", y),
            _add(cf.get("total_equity", y), _total_debt(cf, y)),
        ),
        "ebitda_margin": lambda cf, y: _safe_div(
            _add(cf.get("ebit", y), cf.get("depreciation_amortization", y)),
            cf.get("revenue", y),
        ),
    }


def _efficiency_ratios() -> dict[str, RatioFn]:
    return {
        "asset_turnover": lambda cf, y: _safe_div(
            cf.get("revenue", y), _avg(cf, "total_assets", y)
        ),
        "inventory_turnover": lambda cf, y: _safe_div(
            cf.get("cost_of_revenue", y), _avg(cf, "inventory", y)
        ),
        "receivables_turnover": lambda cf, y: _safe_div(
            cf.get("revenue", y), _avg(cf, "receivables", y)
        ),
        "days_sales_outstanding": lambda cf, y: _safe_div(
            _avg(cf, "receivables", y), cf.get("revenue", y)
        )
        and _safe_div(_avg(cf, "receivables", y), cf.get("revenue", y)) * 365,
        "days_inventory_outstanding": lambda cf, y: _safe_div(
            _avg(cf, "inventory", y), cf.get("cost_of_revenue", y)
        )
        and _safe_div(_avg(cf, "inventory", y), cf.get("cost_of_revenue", y)) * 365,
        "fixed_asset_turnover": lambda cf, y: _safe_div(
            cf.get("revenue", y),
            _sub(cf.get("total_assets", y), cf.get("current_assets", y)),
        ),
    }


def _cash_flow_ratios() -> dict[str, RatioFn]:
    return {
        "free_cash_flow": lambda cf, y: _sub(cf.get("operating_cash_flow", y), cf.get("capex", y)),
        "fcf_margin": lambda cf, y: _safe_div(
            _sub(cf.get("operating_cash_flow", y), cf.get("capex", y)), cf.get("revenue", y)
        ),
        "cash_flow_to_debt": lambda cf, y: _safe_div(
            cf.get("operating_cash_flow", y), _total_debt(cf, y)
        ),
        "capex_to_revenue": lambda cf, y: _safe_div(cf.get("capex", y), cf.get("revenue", y)),
        "dividend_payout_ratio": lambda cf, y: _safe_div(
            cf.get("dividends_paid", y), cf.get("net_income", y)
        ),
    }


def _growth_ratios() -> dict[str, RatioFn]:
    return {
        "revenue_growth_yoy": lambda cf, y: _pct_change(cf.get("revenue", y), _prev(cf, "revenue", y)),
        "net_income_growth_yoy": lambda cf, y: _pct_change(
            cf.get("net_income", y), _prev(cf, "net_income", y)
        ),
        "asset_growth_yoy": lambda cf, y: _pct_change(
            cf.get("total_assets", y), _prev(cf, "total_assets", y)
        ),
        "equity_growth_yoy": lambda cf, y: _pct_change(
            cf.get("total_equity", y), _prev(cf, "total_equity", y)
        ),
    }


def _distress_signal_ratios() -> dict[str, RatioFn]:
    """Ratios with known predictive power in distress models (Altman-Z-style inputs)."""
    return {
        "working_capital_to_assets": lambda cf, y: _safe_div(
            _sub(cf.get("current_assets", y), cf.get("current_liabilities", y)),
            cf.get("total_assets", y),
        ),
        "retained_earnings_to_assets": lambda cf, y: _safe_div(
            cf.get("retained_earnings", y), cf.get("total_assets", y)
        ),
        "ebit_to_assets": lambda cf, y: _safe_div(cf.get("ebit", y), cf.get("total_assets", y)),
        "equity_to_liabilities": lambda cf, y: _safe_div(
            cf.get("total_equity", y), cf.get("total_liabilities", y)
        ),
        "sales_to_assets": lambda cf, y: _safe_div(cf.get("revenue", y), cf.get("total_assets", y)),
    }


def _sub(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return a - b


def _add(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return a + b


def _total_debt(cf: CompanyFinancials, year: int) -> float | None:
    lt = cf.get("long_term_debt", year)
    st = cf.get("short_term_debt", year)
    if lt is None and st is None:
        return None
    return (lt or 0) + (st or 0)


def _pct_change(cur: float | None, prev: float | None) -> float | None:
    if cur is None or prev is None or prev == 0:
        return None
    return (cur - prev) / abs(prev)


RATIO_REGISTRY: dict[str, RatioFn] = {
    **_liquidity_ratios(),
    **_leverage_ratios(),
    **_profitability_ratios(),
    **_efficiency_ratios(),
    **_cash_flow_ratios(),
    **_growth_ratios(),
    **_distress_signal_ratios(),
}


class RatioEngine:
    """Computes the full ratio set for a company across all available fiscal years."""

    def __init__(self, registry: dict[str, RatioFn] = None):
        self.registry = registry or RATIO_REGISTRY

    def compute_year(self, cf: CompanyFinancials, year: int) -> dict[str, float | None]:
        return {name: fn(cf, year) for name, fn in self.registry.items()}

    def compute_all_years(self, cf: CompanyFinancials) -> dict[int, dict[str, float | None]]:
        return {year: self.compute_year(cf, year) for year in cf.years_available()}

    @property
    def feature_count(self) -> int:
        return len(self.registry)
