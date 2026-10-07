from narrative.context_builder import CompanyContext
from reporting.pdf_generator import PDFReportGenerator


def _sample_context() -> CompanyContext:
    return CompanyContext(
        ticker="TEST",
        name="Test Corp",
        sector="technology",
        fiscal_year=2023,
        ratios={"current_ratio": 0.85, "net_margin": 0.03, "debt_to_equity": 2.1},
        z_scores={"current_ratio": -2.3, "net_margin": -1.8, "debt_to_equity": 1.9},
        distress_probability=0.75,
        shap_contributions=[
            {"feature": "current_ratio", "shap_value": 1.2, "feature_value": -2.3},
            {"feature": "net_margin", "shap_value": 0.8, "feature_value": -1.8},
        ],
        base_value=-0.9,
        prior_year_ratios={"current_ratio": 1.05, "net_margin": 0.06, "debt_to_equity": 1.7},
        ratio_history={
            "current_ratio": {2021: 1.20, 2022: 1.05, 2023: 0.85},
            "net_margin": {2021: 0.07, 2022: 0.06, 2023: 0.03},
        },
    )


def test_generate_company_report_creates_nonempty_pdf(tmp_path):
    gen = PDFReportGenerator(output_dir=tmp_path)
    path = gen.generate_company_report(_sample_context(), "Sample narrative text about distress.")

    assert path.exists()
    assert path.suffix == ".pdf"
    assert path.stat().st_size > 500  # not an empty/broken PDF stub
    assert path.read_bytes().startswith(b"%PDF")


def test_generate_company_report_handles_no_outliers(tmp_path):
    """A company with no |z|>=1.5 outliers must still render (falls back to
    the 'no outliers' message rather than an empty/broken table)."""
    ctx = _sample_context()
    ctx.z_scores = {"current_ratio": 0.1, "net_margin": -0.2}
    gen = PDFReportGenerator(output_dir=tmp_path)
    path = gen.generate_company_report(ctx, "Narrative for a boring, average company.")
    assert path.exists()
    assert path.stat().st_size > 500


def test_generate_company_report_handles_no_shap_contributions(tmp_path):
    """No SHAP contributions (e.g. a company that failed scoring) must not
    crash report generation -- the waterfall chart section is just skipped."""
    ctx = _sample_context()
    ctx.shap_contributions = []
    gen = PDFReportGenerator(output_dir=tmp_path)
    path = gen.generate_company_report(ctx, "Narrative text.")
    assert path.exists()
    assert path.read_bytes().startswith(b"%PDF")


def test_generate_company_report_handles_no_ratio_history(tmp_path):
    """No trend history (e.g. first fiscal year on record) must not crash --
    the Ratio Trends section is simply omitted."""
    ctx = _sample_context()
    ctx.ratio_history = {}
    gen = PDFReportGenerator(output_dir=tmp_path)
    path = gen.generate_company_report(ctx, "Narrative text.")
    assert path.exists()
    assert path.read_bytes().startswith(b"%PDF")


def test_report_with_charts_is_larger_than_without():
    """Sanity check that the chart-bearing report actually embeds image
    data, not just placeholder text -- file size should meaningfully exceed
    a chart-free report of otherwise identical content."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        gen = PDFReportGenerator(output_dir=Path(td))

        with_charts = _sample_context()
        path_with = gen.generate_company_report(with_charts, "Narrative.", filename="with_charts.pdf")

        without_charts = _sample_context()
        without_charts.shap_contributions = []
        without_charts.ratio_history = {}
        path_without = gen.generate_company_report(without_charts, "Narrative.", filename="without_charts.pdf")

        assert path_with.stat().st_size > path_without.stat().st_size


def test_generate_multi_company_summary(tmp_path):
    gen = PDFReportGenerator(output_dir=tmp_path)
    contexts = [_sample_context()]
    path = gen.generate_multi_company_summary(contexts)
    assert path.exists()
    assert path.read_bytes().startswith(b"%PDF")
