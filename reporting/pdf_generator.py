"""
FinSight — PDF report generator.

Renders one structured analytical report per company (or a combined
multi-company screen) covering: headline distress score, key ratio table,
sector Z-score outliers, SHAP driver chart, and the Gemini narrative.
This is the final stage of the pipeline — the piece that turns a
6-company screen from a manual spreadsheet exercise into a sub-2-minute
automated report run.
"""
from __future__ import annotations

import io
import logging
from datetime import datetime
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from config.settings import REPORTS_DIR
from narrative.context_builder import CompanyContext
from reporting.charts import ratio_trend_grid, shap_waterfall_chart

logger = logging.getLogger("finsight.reporting.pdf_generator")

# Fixed set of headline ratios shown as trend charts when history is available --
# one representative from each major category so the report isn't just the
# outlier list, but also shows baseline trajectory even for "boring" ratios.
TREND_RATIO_NAMES = ["current_ratio", "net_margin", "debt_to_equity", "return_on_assets"]

RISK_BANDS = [
    (0.70, "HIGH RISK", colors.HexColor("#B3261E")),
    (0.40, "ELEVATED RISK", colors.HexColor("#C77800")),
    (0.0, "LOW RISK", colors.HexColor("#2E7D32")),
]


def _risk_band(probability: float) -> tuple[str, "colors.Color"]:
    for threshold, label, color in RISK_BANDS:
        if probability >= threshold:
            return label, color
    return "LOW RISK", colors.HexColor("#2E7D32")


class PDFReportGenerator:
    def __init__(self, output_dir: Path = REPORTS_DIR):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.styles = getSampleStyleSheet()
        self._register_custom_styles()

    def _register_custom_styles(self):
        self.styles.add(
            ParagraphStyle(
                name="FinSightTitle",
                fontSize=20,
                leading=24,
                spaceAfter=4,
                textColor=colors.HexColor("#111827"),
            )
        )
        self.styles.add(
            ParagraphStyle(
                name="FinSightSubtitle",
                fontSize=11,
                textColor=colors.HexColor("#4B5563"),
                spaceAfter=16,
            )
        )
        self.styles.add(
            ParagraphStyle(
                name="SectionHeader",
                fontSize=13,
                leading=16,
                spaceBefore=14,
                spaceAfter=6,
                textColor=colors.HexColor("#1F2937"),
            )
        )
        self.styles.add(
            ParagraphStyle(
                name="NarrativeBody",
                fontSize=10,
                leading=15,
                spaceAfter=8,
                textColor=colors.HexColor("#111827"),
            )
        )

    def generate_company_report(
        self,
        context: CompanyContext,
        narrative_text: str,
        filename: str | None = None,
    ) -> Path:
        filename = filename or f"{context.ticker}_{context.fiscal_year}_distress_report.pdf"
        path = self.output_dir / filename

        doc = SimpleDocTemplate(
            str(path),
            pagesize=letter,
            topMargin=0.6 * inch,
            bottomMargin=0.6 * inch,
            leftMargin=0.7 * inch,
            rightMargin=0.7 * inch,
        )
        story = []

        story.append(Paragraph(f"{context.name} ({context.ticker})", self.styles["FinSightTitle"]))
        story.append(
            Paragraph(
                f"FinSight Distress Screen &middot; {context.sector.title()} Sector &middot; "
                f"FY{context.fiscal_year} &middot; Generated {datetime.now():%Y-%m-%d %H:%M}",
                self.styles["FinSightSubtitle"],
            )
        )

        risk_label, risk_color = _risk_band(context.distress_probability)
        headline_table = Table(
            [[f"Distress Probability: {context.distress_probability:.1%}", risk_label]],
            colWidths=[3.5 * inch, 2.5 * inch],
        )
        headline_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (1, 0), (1, 0), risk_color),
                    ("TEXTCOLOR", (1, 0), (1, 0), colors.white),
                    ("FONTSIZE", (0, 0), (-1, -1), 11),
                    ("FONTNAME", (1, 0), (1, 0), "Helvetica-Bold"),
                    ("ALIGN", (1, 0), (1, 0), "CENTER"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("TOPPADDING", (0, 0), (-1, -1), 8),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ]
            )
        )
        story.append(headline_table)
        story.append(Spacer(1, 0.15 * inch))

        story.append(Paragraph("Analyst Narrative", self.styles["SectionHeader"]))
        for para in narrative_text.split("\n\n"):
            if para.strip():
                story.append(Paragraph(para.strip(), self.styles["NarrativeBody"]))

        story.append(Paragraph("Top Model Drivers (SHAP)", self.styles["SectionHeader"]))
        driver_rows = [["Feature", "SHAP Contribution", "Z-Score Value"]]
        for c in context.shap_contributions[:8]:
            driver_rows.append(
                [c["feature"], f"{c['shap_value']:+.4f}", f"{c['feature_value']:.2f}"]
            )
        story.append(self._styled_table(driver_rows))
        story.append(Spacer(1, 0.1 * inch))
        if context.shap_contributions:
            story.append(self._shap_waterfall_flowable(context))

        story.append(Paragraph("Sector-Relative Outliers", self.styles["SectionHeader"]))
        outliers = context.notable_outliers()
        if outliers:
            outlier_rows = [["Ratio", "Z-Score", "Raw Value"]]
            for name, z in outliers[:10]:
                raw = context.ratios.get(name)
                raw_str = f"{raw:.3f}" if raw is not None else "n/a"
                outlier_rows.append([name, f"{z:+.2f}", raw_str])
            story.append(self._styled_table(outlier_rows))
        else:
            story.append(Paragraph("No ratios deviate more than 1.5σ from sector norms.", self.styles["NarrativeBody"]))

        trend_flowable = self._ratio_trend_flowable(context)
        if trend_flowable is not None:
            story.append(Paragraph("Ratio Trends", self.styles["SectionHeader"]))
            story.append(trend_flowable)

        story.append(Paragraph("Full Ratio Panel", self.styles["SectionHeader"]))
        ratio_rows = [["Ratio", "Value"]]
        for name, value in sorted(context.ratios.items()):
            value_str = f"{value:.4f}" if value is not None else "n/a"
            ratio_rows.append([name, value_str])
        story.append(self._styled_table(ratio_rows, font_size=8))

        doc.build(story)
        logger.info("Generated report: %s", path)
        return path

    def generate_multi_company_summary(
        self, contexts: list[CompanyContext], output_filename: str = "screen_summary.pdf"
    ) -> Path:
        """Combined summary table across the whole screen, plus one page per company detail."""
        path = self.output_dir / output_filename
        doc = SimpleDocTemplate(str(path), pagesize=letter)
        story = [
            Paragraph("FinSight Multi-Company Distress Screen", self.styles["FinSightTitle"]),
            Paragraph(
                f"{len(contexts)} companies &middot; Generated {datetime.now():%Y-%m-%d %H:%M}",
                self.styles["FinSightSubtitle"],
            ),
        ]

        rows = [["Ticker", "Sector", "FY", "Distress Prob.", "Risk Band"]]
        sorted_ctx = sorted(contexts, key=lambda c: c.distress_probability, reverse=True)
        for c in sorted_ctx:
            label, _ = _risk_band(c.distress_probability)
            rows.append([c.ticker, c.sector, str(c.fiscal_year), f"{c.distress_probability:.1%}", label])
        story.append(self._styled_table(rows))
        story.append(PageBreak())

        doc.build(story)
        logger.info("Generated summary report: %s", path)
        return path

    def _shap_waterfall_flowable(self, context: CompanyContext) -> Image:
        png_bytes = shap_waterfall_chart(
            context.shap_contributions,
            base_value=context.base_value,
            distress_probability=context.distress_probability,
        )
        iw, ih = ImageReader(io.BytesIO(png_bytes)).getSize()
        display_width = 6.5 * inch
        display_height = display_width * (ih / iw)
        return Image(io.BytesIO(png_bytes), width=display_width, height=display_height)

    def _ratio_trend_flowable(self, context: CompanyContext) -> Table | None:
        """
        Trend charts laid out 2-per-row in a borderless table, so they sit
        side by side rather than stacking full-width. Returns None (skip
        the section entirely) if there isn't enough multi-year history for
        any of the headline ratios.
        """
        charts = ratio_trend_grid(context.ratio_history, TREND_RATIO_NAMES)
        if not charts:
            return None

        cell_width = 3.1 * inch
        images = []
        for name, png_bytes in charts:
            iw, ih = ImageReader(io.BytesIO(png_bytes)).getSize()
            display_height = cell_width * (ih / iw)
            images.append(Image(io.BytesIO(png_bytes), width=cell_width, height=display_height))

        rows = [images[i : i + 2] for i in range(0, len(images), 2)]
        if len(rows[-1]) == 1:
            rows[-1].append("")  # pad so the table stays rectangular

        table = Table(rows, colWidths=[cell_width + 0.2 * inch] * 2)
        table.setStyle(
            TableStyle(
                [
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        return table

    def _styled_table(self, rows: list[list[str]], font_size: int = 9) -> Table:
        table = Table(rows, repeatRows=1)
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F2937")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), font_size),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D1D5DB")),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F9FAFB")]),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        return table
