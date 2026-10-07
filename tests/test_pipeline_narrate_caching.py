import pipeline as pipeline_module
from database.storage import Storage
from narrative.gemini_client import GeminiNarrativeClient

SAMPLE_COMPANIES = [{"ticker": "ACME", "name": "Acme Corp", "sector": "technology", "cik": "0000000001"}]


def _seed_storage(storage: Storage, distress_probability: float = 0.4):
    storage.save_ratios("ACME", "technology", {2023: {"net_margin": 0.05}})
    storage.save_z_scores("ACME", "technology", {2023: {"net_margin": -2.0}})
    storage.save_distress_score(
        "ACME",
        2023,
        distress_probability,
        [{"feature": "net_margin", "shap_value": 0.3, "feature_value": -2.0}],
        "2024-01-01T00:00:00",
        base_value=-0.5,
    )


def test_stage_narrate_skips_gemini_when_content_unchanged(tmp_path, monkeypatch):
    storage = Storage(db_path=tmp_path / "test.db")
    monkeypatch.setattr(pipeline_module, "Storage", lambda: storage)

    call_count = {"n": 0}

    def fake_generate_narrative(self, context, temperature=0.3):
        call_count["n"] += 1
        return f"Narrative #{call_count['n']}"

    monkeypatch.setattr(GeminiNarrativeClient, "generate_narrative", fake_generate_narrative)

    _seed_storage(storage)

    first_run = pipeline_module.stage_narrate(SAMPLE_COMPANIES)
    assert first_run == {"generated": 1, "cached": 0}
    assert call_count["n"] == 1

    # Re-run with IDENTICAL underlying data: Gemini must not be called again.
    second_run = pipeline_module.stage_narrate(SAMPLE_COMPANIES)
    assert second_run == {"generated": 0, "cached": 1}
    assert call_count["n"] == 1

    # The stored narrative from the first run is preserved, not overwritten with nothing.
    assert storage.get_narrative("ACME", 2023) == "Narrative #1"


def test_stage_narrate_regenerates_when_underlying_data_changes(tmp_path, monkeypatch):
    storage = Storage(db_path=tmp_path / "test.db")
    monkeypatch.setattr(pipeline_module, "Storage", lambda: storage)

    call_count = {"n": 0}

    def fake_generate_narrative(self, context, temperature=0.3):
        call_count["n"] += 1
        return f"Narrative #{call_count['n']}"

    monkeypatch.setattr(GeminiNarrativeClient, "generate_narrative", fake_generate_narrative)

    _seed_storage(storage, distress_probability=0.4)
    pipeline_module.stage_narrate(SAMPLE_COMPANIES)
    assert call_count["n"] == 1

    # A materially different distress score changes the rendered prompt
    # (the probability line always appears), so this must trigger a real
    # regeneration rather than reusing the stale cached narrative.
    _seed_storage(storage, distress_probability=0.9)
    result = pipeline_module.stage_narrate(SAMPLE_COMPANIES)

    assert result == {"generated": 1, "cached": 0}
    assert call_count["n"] == 2
    assert storage.get_narrative("ACME", 2023) == "Narrative #2"
