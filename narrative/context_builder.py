"""
FinSight — RAG context builder.

Assembles a structured, grounded context block for a single company from
the outputs of every upstream layer (raw ratios, sector Z-scores, SHAP
distress attributions, trend deltas). This is injected into the Gemini
prompt so the narrative model is reasoning over *actual computed numbers*
rather than hallucinating financial analysis — "structured context
injection" in the resume bullet.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field


@dataclass
class CompanyContext:
    ticker: str
    name: str
    sector: str
    fiscal_year: int
    ratios: dict[str, float | None]
    z_scores: dict[str, float | None]
    distress_probability: float
    shap_contributions: list[dict]
    base_value: float = 0.0
    prior_year_ratios: dict[str, float | None] = field(default_factory=dict)
    # Multi-year history for trend charts: {ratio_name: {year: value}}.
    # Only populated for a handful of key ratios (see pipeline.py::_build_ratio_history) --
    # not the full 40-ratio panel for every year, to keep this lightweight.
    ratio_history: dict[str, dict[int, float]] = field(default_factory=dict)

    def notable_outliers(self, z_threshold: float = 1.5) -> list[tuple[str, float]]:
        """Ratios that are more than z_threshold sector-stdevs from the norm."""
        outliers = [
            (name, z) for name, z in self.z_scores.items() if z is not None and abs(z) >= z_threshold
        ]
        outliers.sort(key=lambda t: abs(t[1]), reverse=True)
        return outliers

    def yoy_deltas(self) -> dict[str, float]:
        deltas = {}
        for ratio_name, cur_val in self.ratios.items():
            prev_val = self.prior_year_ratios.get(ratio_name)
            if cur_val is not None and prev_val is not None and prev_val != 0:
                deltas[ratio_name] = (cur_val - prev_val) / abs(prev_val)
        return deltas

    def to_prompt_context(self) -> str:
        """Render this context as a compact, structured text block for the LLM prompt."""
        lines = [
            f"COMPANY: {self.name} ({self.ticker})",
            f"SECTOR: {self.sector}",
            f"FISCAL YEAR: {self.fiscal_year}",
            f"MODEL DISTRESS PROBABILITY: {self.distress_probability:.1%}",
            "",
            "TOP SHAP RISK/STRENGTH DRIVERS (feature, contribution, raw z-score value):",
        ]
        for c in self.shap_contributions[:5]:
            direction = "increases" if c["shap_value"] > 0 else "decreases"
            lines.append(
                f"  - {c['feature']}: {direction} distress probability "
                f"(SHAP={c['shap_value']:.4f}, z={c['feature_value']:.2f})"
            )

        lines.append("")
        lines.append("NOTABLE SECTOR OUTLIERS (|z| >= 1.5):")
        outliers = self.notable_outliers()
        if outliers:
            for name, z in outliers[:8]:
                raw = self.ratios.get(name)
                raw_str = f"{raw:.3f}" if raw is not None else "n/a"
                lines.append(f"  - {name}: z={z:.2f}, raw value={raw_str}")
        else:
            lines.append("  - None; company tracks close to sector norms.")

        deltas = self.yoy_deltas()
        if deltas:
            lines.append("")
            lines.append("YEAR-OVER-YEAR CHANGES (largest movers):")
            sorted_deltas = sorted(deltas.items(), key=lambda t: abs(t[1]), reverse=True)
            for name, delta in sorted_deltas[:6]:
                lines.append(f"  - {name}: {delta:+.1%} vs prior year")

        return "\n".join(lines)

    def content_hash(self) -> str:
        """
        Stable hash of everything that would actually change the narrative
        Gemini generates: the same rendered prompt context. Used by
        pipeline.py::stage_narrate to skip regenerating (and re-paying for)
        a narrative when the underlying data hasn't changed since the last
        pipeline run. Deliberately hashes the rendered prompt text rather
        than the dataclass fields directly, since to_prompt_context() is
        exactly what Gemini sees -- any change that would change its output
        necessarily changes this string.
        """
        return hashlib.sha256(self.to_prompt_context().encode("utf-8")).hexdigest()
