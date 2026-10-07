import tempfile
from pathlib import Path

import pytest
import sqlalchemy as sa

from database.storage import Storage


@pytest.fixture
def storage(tmp_path) -> Storage:
    return Storage(db_path=tmp_path / "test_finsight.db")


PG_TEST_URL = "postgresql+psycopg2://postgres:finsight_test@localhost:5432/finsight_test"


def _postgres_available() -> bool:
    try:
        engine = sa.create_engine(PG_TEST_URL, future=True)
        with engine.connect():
            return True
    except Exception:
        return False


@pytest.fixture
def pg_storage():
    """
    Real Postgres-backed Storage, skipped automatically if no Postgres is
    reachable at PG_TEST_URL (e.g. CI environments without a DB service).
    Tables are dropped and recreated per test for isolation.
    """
    if not _postgres_available():
        pytest.skip("Postgres not available at PG_TEST_URL -- skipping Postgres-specific tests")
    s = Storage(database_url=PG_TEST_URL)
    from database.storage import metadata

    metadata.drop_all(s.engine)
    s._init_schema()
    yield s
    metadata.drop_all(s.engine)


def test_ratios_round_trip(storage):
    ratios_by_year = {2023: {"net_margin": 0.12, "current_ratio": 1.5}}
    storage.save_ratios("ACME", "technology", ratios_by_year)

    result = storage.get_ratios("ACME")
    assert result[2023]["net_margin"] == 0.12
    assert result[2023]["current_ratio"] == 1.5


def test_z_scores_round_trip(storage):
    z_by_year = {2023: {"net_margin": -0.8, "current_ratio": None}}
    storage.save_z_scores("ACME", "technology", z_by_year)

    result = storage.get_z_scores("ACME")
    assert result[2023]["net_margin"] == -0.8
    assert result[2023]["current_ratio"] is None


def test_distress_score_round_trip(storage):
    contributions = [{"feature": "net_margin", "shap_value": 0.5, "feature_value": -0.8}]
    storage.save_distress_score("ACME", 2023, 0.73, contributions, "2024-01-01T00:00:00")

    result = storage.get_distress_score("ACME", 2023)
    assert result["distress_probability"] == 0.73
    assert result["contributions"] == contributions


def test_distress_score_missing_returns_none(storage):
    assert storage.get_distress_score("NOPE", 2023) is None


def test_narrative_round_trip(storage):
    storage.save_narrative("ACME", 2023, "Some analyst narrative text.", "2024-01-01T00:00:00")
    assert storage.get_narrative("ACME", 2023) == "Some analyst narrative text."


def test_narrative_missing_returns_none(storage):
    assert storage.get_narrative("NOPE", 2023) is None


def test_narrative_record_includes_context_hash(storage):
    storage.save_narrative(
        "ACME", 2023, "Narrative text.", "2024-01-01T00:00:00", context_hash="abc123"
    )
    record = storage.get_narrative_record("ACME", 2023)
    assert record["narrative_text"] == "Narrative text."
    assert record["context_hash"] == "abc123"


def test_narrative_record_missing_returns_none(storage):
    assert storage.get_narrative_record("NOPE", 2023) is None


def test_narrative_context_hash_defaults_to_none_when_not_provided(storage):
    storage.save_narrative("ACME", 2023, "Narrative text.", "2024-01-01T00:00:00")
    record = storage.get_narrative_record("ACME", 2023)
    assert record["context_hash"] is None


def test_save_ratios_overwrites_on_replay(storage):
    """Re-running a pipeline stage (e.g. Airflow task retry) must overwrite,
    not duplicate, prior values for the same (ticker, year, ratio)."""
    storage.save_ratios("ACME", "technology", {2023: {"net_margin": 0.10}})
    storage.save_ratios("ACME", "technology", {2023: {"net_margin": 0.15}})

    result = storage.get_ratios("ACME")
    assert result[2023]["net_margin"] == 0.15
    assert len(result) == 1


# ---------------------------------------------------------------------------
# Postgres-specific: same test bodies as above, run against a real Postgres
# instance to prove the SQLAlchemy rewrite behaves identically across
# backends, not just structurally similar. Skipped automatically if
# Postgres isn't reachable (see _postgres_available above).
# ---------------------------------------------------------------------------


def test_postgres_ratios_round_trip(pg_storage):
    pg_storage.save_ratios("ACME", "technology", {2023: {"net_margin": 0.12, "current_ratio": 1.5}})
    result = pg_storage.get_ratios("ACME")
    assert result[2023]["net_margin"] == 0.12
    assert result[2023]["current_ratio"] == 1.5


def test_postgres_upsert_overwrites_not_duplicates(pg_storage):
    pg_storage.save_ratios("ACME", "technology", {2023: {"net_margin": 0.10}})
    pg_storage.save_ratios("ACME", "technology", {2023: {"net_margin": 0.15}})

    result = pg_storage.get_ratios("ACME")
    assert result[2023]["net_margin"] == 0.15
    assert len(result) == 1


def test_postgres_distress_score_round_trip(pg_storage):
    contributions = [{"feature": "net_margin", "shap_value": 0.5, "feature_value": -0.8}]
    pg_storage.save_distress_score("ACME", 2023, 0.73, contributions, "2024-01-01T00:00:00", base_value=-1.2)

    result = pg_storage.get_distress_score("ACME", 2023)
    assert result["distress_probability"] == 0.73
    assert result["contributions"] == contributions
    assert result["base_value"] == -1.2


def test_postgres_handles_concurrent_writers():
    """
    RC-009 regression: the whole point of the Postgres migration. 12
    threads writing concurrently to the same table must all succeed --
    this is the exact scenario (parallel Airflow task instances) that
    causes 'database is locked' under SQLite once write contention is high
    enough. Uses its own dedicated database so it doesn't interfere with
    other Postgres-backed tests running in parallel.
    """
    if not _postgres_available():
        pytest.skip("Postgres not available at PG_TEST_URL -- skipping concurrency test")

    import threading

    from database.storage import metadata

    setup_storage = Storage(database_url=PG_TEST_URL)
    metadata.drop_all(setup_storage.engine)
    setup_storage._init_schema()

    errors = []

    def write_worker(worker_id: int):
        try:
            s = Storage(database_url=PG_TEST_URL)
            for i in range(15):
                s.save_ratios(f"CONC{worker_id}", "technology", {2020 + i % 5: {"net_margin": 0.1 * i}})
        except Exception as exc:  # noqa: BLE001
            errors.append((worker_id, type(exc).__name__, str(exc)))

    threads = [threading.Thread(target=write_worker, args=(i,)) for i in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [], f"Concurrent writers failed: {errors}"

    # Spot check a couple of workers actually persisted their data
    verify_storage = Storage(database_url=PG_TEST_URL)
    assert verify_storage.get_ratios("CONC0") != {}
    assert verify_storage.get_ratios("CONC11") != {}

    metadata.drop_all(setup_storage.engine)
