import pytest
from fastapi.testclient import TestClient

import webapp.main as webapp_module
from database.storage import Storage


@pytest.fixture
def client(tmp_path, monkeypatch):
    storage = Storage(db_path=tmp_path / "test_webapp.db")
    monkeypatch.setattr(webapp_module, "Storage", lambda: storage)
    return TestClient(webapp_module.app), storage


def _seed(storage: Storage, ticker: str, sector: str, probability: float, year: int = 2023):
    storage.save_ratios(ticker, sector, {year: {"current_ratio": 1.2, "net_margin": 0.05}})
    storage.save_z_scores(ticker, sector, {year: {"current_ratio": -2.0, "net_margin": 0.3}})
    storage.save_distress_score(
        ticker,
        year,
        probability,
        [{"feature": "current_ratio", "shap_value": 0.4, "feature_value": -2.0}],
        "2024-01-01T00:00:00",
        base_value=-0.5,
    )


class TestSummary:
    def test_empty_database_returns_zeros(self, client):
        c, _ = client
        resp = c.get("/api/summary")
        assert resp.status_code == 200
        assert resp.json() == {"total": 0, "scored": 0, "high": 0, "elevated": 0, "low": 0, "sectors": []}

    def test_counts_by_risk_band(self, client):
        c, storage = client
        _seed(storage, "HIGH1", "technology", 0.85)
        _seed(storage, "MID1", "technology", 0.50)
        _seed(storage, "LOW1", "retail", 0.10)
        resp = c.get("/api/summary").json()
        assert resp["total"] == 3
        assert resp["scored"] == 3
        assert resp["high"] == 1
        assert resp["elevated"] == 1
        assert resp["low"] == 1
        # None of these tickers are in config/companies.yaml, so sector
        # metadata correctly falls back to "demo" (see
        # TestCompanyList.test_storage_driven_not_config_driven for the
        # config-matched case) -- this test is about count correctness,
        # not sector passthrough.
        assert resp["sectors"] == ["demo"]

    def test_sector_passthrough_for_configured_tickers(self, client):
        """A ticker that IS in config/companies.yaml should report its real
        configured sector, not fall back to 'demo'."""
        c, storage = client
        _seed(storage, "AAPL", "technology", 0.2)  # AAPL is in config/companies.yaml
        resp = c.get("/api/summary").json()
        assert resp["sectors"] == ["technology"]


class TestRiskBandThresholds:
    @pytest.mark.parametrize(
        "probability,expected_band",
        [
            (0.95, "high"),
            (0.70, "high"),  # boundary: >= 0.70 is high
            (0.69, "elevated"),
            (0.40, "elevated"),  # boundary: >= 0.40 is elevated
            (0.39, "low"),
            (0.0, "low"),
        ],
    )
    def test_risk_band_boundaries(self, client, probability, expected_band):
        c, storage = client
        _seed(storage, "EDGE1", "technology", probability)
        resp = c.get("/api/companies").json()
        assert resp["companies"][0]["risk_band"] == expected_band


class TestCompanyList:
    def test_storage_driven_not_config_driven(self, client):
        """
        The core fix verified here: the dashboard must show whatever
        tickers actually have data in storage, including synthetic demo
        tickers that don't appear in config/companies.yaml -- not just the
        14 real tracked companies.
        """
        c, storage = client
        _seed(storage, "TEC0", "technology", 0.2)  # a scripts/run_demo.py-style synthetic ticker

        companies = c.get("/api/companies").json()["companies"]
        tickers = [co["ticker"] for co in companies]
        assert "TEC0" in tickers

    def test_unconfigured_ticker_gets_fallback_metadata(self, client):
        c, storage = client
        _seed(storage, "SYNTH9", "technology", 0.2)
        companies = c.get("/api/companies").json()["companies"]
        synth = next(co for co in companies if co["ticker"] == "SYNTH9")
        assert synth["name"] == "SYNTH9"  # falls back to ticker as display name
        assert synth["sector"] == "demo"  # falls back to "demo", not a crash

    def test_company_with_no_score_yet_has_null_fields(self, client):
        c, storage = client
        storage.save_ratios("NOSCORE", "technology", {2023: {"net_margin": 0.05}})
        companies = c.get("/api/companies").json()["companies"]
        row = next(co for co in companies if co["ticker"] == "NOSCORE")
        assert row["distress_probability"] is None
        assert row["risk_band"] is None


class TestCompanyDetail:
    def test_unknown_ticker_returns_404(self, client):
        c, _ = client
        resp = c.get("/api/companies/NOPE")
        assert resp.status_code == 404

    def test_detail_shape(self, client):
        c, storage = client
        _seed(storage, "DETAIL1", "technology", 0.6)
        resp = c.get("/api/companies/DETAIL1")
        assert resp.status_code == 200
        body = resp.json()
        assert body["ticker"] == "DETAIL1"
        assert body["distress_probability"] == 0.6
        assert body["risk_band"] == "elevated"
        assert body["base_value"] == -0.5
        assert len(body["shap_contributions"]) == 1
        assert body["report_available"] is False
        assert body["report_url"] is None

    def test_ticker_is_case_insensitive(self, client):
        c, storage = client
        _seed(storage, "CASE1", "technology", 0.3)
        resp = c.get("/api/companies/case1")
        assert resp.status_code == 200
        assert resp.json()["ticker"] == "CASE1"

    def test_outliers_only_include_significant_z_scores(self, client):
        c, storage = client
        storage.save_ratios("OUT1", "technology", {2023: {"a": 1.0, "b": 2.0}})
        storage.save_z_scores("OUT1", "technology", {2023: {"a": 2.0, "b": 0.5}})  # only "a" clears 1.5
        storage.save_distress_score("OUT1", 2023, 0.3, [], "2024-01-01T00:00:00")

        body = c.get("/api/companies/OUT1").json()
        outlier_names = [o["ratio"] for o in body["outliers"]]
        assert "a" in outlier_names
        assert "b" not in outlier_names

    def test_no_narrative_returns_null_not_error(self, client):
        c, storage = client
        _seed(storage, "NONARR1", "technology", 0.3)
        body = c.get("/api/companies/NONARR1").json()
        assert body["narrative"] is None


class TestReportEndpoint:
    def test_no_report_returns_404(self, client):
        c, storage = client
        _seed(storage, "NOREPORT1", "technology", 0.3)
        resp = c.get("/api/companies/NOREPORT1/report")
        assert resp.status_code == 404

    def test_unknown_ticker_report_returns_404(self, client):
        c, _ = client
        resp = c.get("/api/companies/NOPE/report")
        assert resp.status_code == 404
