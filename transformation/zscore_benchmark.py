"""
FinSight — sector-relative Z-score benchmarking.

A ratio in isolation is hard to interpret (is a 12% net margin good?
depends entirely on sector). This module standardizes every ratio against
its sector peer distribution for the same fiscal year, so downstream
scoring/reporting works with "N standard deviations from sector norm"
instead of raw ratio values.

Two standardization methods are supported:

    "standard" (classic Z-score):
        z = (value - sector_mean) / sector_stdev

    "robust" (modified Z-score, median/MAD-based) -- the DEFAULT:
        z = 0.6745 * (value - sector_median) / sector_MAD

The sector cohorts here are small (4 companies/sector in the default
config). With n=4, a single unusual company pulls the mean and inflates
the stdev enough to compress every OTHER company's Z-score toward zero --
exactly the case a distress screen most needs to get right. Mean/stdev
give every observation equal leverage over the estimate; median/MAD do
not, since the median only cares about the middle-ranked value and MAD is
similarly insensitive to how far a single outlier sits from the pack. The
0.6745 scale factor (technically 1/Phi^-1(0.75)) makes the robust score
comparable in magnitude to a standard Z-score under a normal distribution,
so downstream |z| >= 1.5 "outlier" thresholds behave consistently either
way.

Sector stats are computed cross-sectionally per (sector, fiscal_year, ratio)
so a company is only ever compared to peers reporting for the same period.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Literal

ZScoreMethod = Literal["standard", "robust"]

# Scales MAD to be a consistent estimator of the standard deviation under
# a normal distribution (1 / Phi^-1(0.75) ~= 1.4826), then the classic
# 0.6745 modified-Z-score constant is folded in below at call sites.
_MAD_TO_STDEV_SCALE = 1.4826


@dataclass
class SectorStats:
    center: float  # mean (standard) or median (robust)
    spread: float  # stdev (standard) or scaled MAD (robust)
    n: int
    method: ZScoreMethod

    # Backward-compatible aliases -- existing callers/tests that expect
    # `.mean` / `.stdev` on a "standard"-method SectorStats keep working.
    @property
    def mean(self) -> float:
        return self.center

    @property
    def stdev(self) -> float:
        return self.spread


class ZScoreBenchmark:
    def __init__(self, min_peers: int = 3, method: ZScoreMethod = "robust"):
        # min_peers: below this, we don't trust the spread estimate and
        # return None rather than a misleading Z-score.
        self.min_peers = min_peers
        self.method = method

    def compute_sector_stats(
        self,
        ratios_by_company: dict[str, dict[int, dict[str, float | None]]],
        company_sector_map: dict[str, str],
    ) -> dict[tuple[str, int, str], SectorStats]:
        """
        ratios_by_company: {ticker: {year: {ratio_name: value}}}
        Returns: {(sector, year, ratio_name): SectorStats}
        """
        buckets: dict[tuple[str, int, str], list[float]] = {}

        for ticker, year_data in ratios_by_company.items():
            sector = company_sector_map.get(ticker)
            if sector is None:
                continue
            for year, ratios in year_data.items():
                for ratio_name, value in ratios.items():
                    if value is None:
                        continue
                    key = (sector, year, ratio_name)
                    buckets.setdefault(key, []).append(value)

        stats: dict[tuple[str, int, str], SectorStats] = {}
        for key, values in buckets.items():
            if len(values) < self.min_peers:
                continue
            if self.method == "robust":
                center = statistics.median(values)
                mad = statistics.median([abs(v - center) for v in values])
                spread = mad * _MAD_TO_STDEV_SCALE
            else:
                center = statistics.mean(values)
                spread = statistics.pstdev(values)  # population stdev across full sector cohort
            stats[key] = SectorStats(center=center, spread=spread, n=len(values), method=self.method)
        return stats

    def score_company(
        self,
        ticker: str,
        sector: str,
        year: int,
        ratios: dict[str, float | None],
        sector_stats: dict[tuple[str, int, str], SectorStats],
    ) -> dict[str, float | None]:
        """Return {ratio_name: z_score} for one company/year, sector-relative."""
        z_scores: dict[str, float | None] = {}
        for ratio_name, value in ratios.items():
            key = (sector, year, ratio_name)
            stat = sector_stats.get(key)
            if value is None or stat is None or stat.spread == 0:
                z_scores[ratio_name] = None
                continue

            if stat.method == "robust":
                z_scores[ratio_name] = 0.6745 * (value - stat.center) / stat.spread
            else:
                z_scores[ratio_name] = (value - stat.center) / stat.spread
        return z_scores

    def score_all(
        self,
        ratios_by_company: dict[str, dict[int, dict[str, float | None]]],
        company_sector_map: dict[str, str],
    ) -> dict[str, dict[int, dict[str, float | None]]]:
        """Full pipeline: compute cohort stats, then Z-score every company/year."""
        sector_stats = self.compute_sector_stats(ratios_by_company, company_sector_map)

        z_by_company: dict[str, dict[int, dict[str, float | None]]] = {}
        for ticker, year_data in ratios_by_company.items():
            sector = company_sector_map.get(ticker)
            if sector is None:
                continue
            z_by_company[ticker] = {}
            for year, ratios in year_data.items():
                z_by_company[ticker][year] = self.score_company(
                    ticker, sector, year, ratios, sector_stats
                )
        return z_by_company
