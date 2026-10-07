"""
FinSight — storage layer.

Persists intermediate results between pipeline stages. Airflow tasks are
independent processes (or even independent workers), so ratios/Z-scores/
scores/narratives are written here rather than passed in-memory — this is
what lets each DAG task be independently retryable.

Backend: SQLite by default (zero setup, fine for local dev and the demo
script), or Postgres in production via the DATABASE_URL env var --
    DATABASE_URL=postgresql+psycopg2://user:pass@host:5432/finsight

Why SQLite alone isn't enough for real Airflow: multiple task instances
(e.g. per-company parallel ingestion, or a retry racing a still-running
attempt) can write concurrently. SQLite serializes writers at the file
level and raises "database is locked" once a writer holds the lock past
another connection's busy-timeout -- a real failure mode under Airflow's
default parallelism, not a hypothetical one (RC-009). Postgres handles
concurrent writers natively via MVCC + row-level locking, so switching the
DATABASE_URL env var is the whole migration; no application code changes.

SQLAlchemy Core (not the ORM) is used here rather than raw sqlite3, purely
to get one codebase that works against both backends -- parameter binding
(:name vs positional ?) and upsert syntax (INSERT OR REPLACE vs INSERT ...
ON CONFLICT) differ between SQLite and Postgres, and SQLAlchemy abstracts
both without needing two parallel implementations of this class.
"""
from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from config.settings import DATABASE_URL

metadata = sa.MetaData()

raw_facts = sa.Table(
    "raw_facts",
    metadata,
    sa.Column("ticker", sa.Text, primary_key=True),
    sa.Column("cik", sa.Text, nullable=False),
    sa.Column("fetched_at", sa.Text, nullable=False),
    sa.Column("payload_json", sa.Text, nullable=False),
)

ratios_table = sa.Table(
    "ratios",
    metadata,
    sa.Column("ticker", sa.Text, primary_key=True),
    sa.Column("sector", sa.Text, nullable=False),
    sa.Column("fiscal_year", sa.Integer, primary_key=True),
    sa.Column("ratio_name", sa.Text, primary_key=True),
    sa.Column("value", sa.Float, nullable=True),
)

z_scores_table = sa.Table(
    "z_scores",
    metadata,
    sa.Column("ticker", sa.Text, primary_key=True),
    sa.Column("sector", sa.Text, nullable=False),
    sa.Column("fiscal_year", sa.Integer, primary_key=True),
    sa.Column("ratio_name", sa.Text, primary_key=True),
    sa.Column("z_value", sa.Float, nullable=True),
)

distress_scores_table = sa.Table(
    "distress_scores",
    metadata,
    sa.Column("ticker", sa.Text, primary_key=True),
    sa.Column("fiscal_year", sa.Integer, primary_key=True),
    sa.Column("distress_probability", sa.Float, nullable=False),
    sa.Column("shap_contributions_json", sa.Text, nullable=False),
    sa.Column("base_value", sa.Float, nullable=False, server_default="0.0"),
    sa.Column("scored_at", sa.Text, nullable=False),
)

narratives_table = sa.Table(
    "narratives",
    metadata,
    sa.Column("ticker", sa.Text, primary_key=True),
    sa.Column("fiscal_year", sa.Integer, primary_key=True),
    sa.Column("narrative_text", sa.Text, nullable=False),
    sa.Column("context_hash", sa.Text, nullable=True),
    sa.Column("generated_at", sa.Text, nullable=False),
)

pipeline_runs_table = sa.Table(
    "pipeline_runs",
    metadata,
    sa.Column("run_id", sa.Text, primary_key=True),
    sa.Column("started_at", sa.Text, nullable=False),
    sa.Column("completed_at", sa.Text, nullable=True),
    sa.Column("status", sa.Text, nullable=False),
    sa.Column("companies_processed", sa.Integer, server_default="0"),
    sa.Column("companies_failed", sa.Integer, server_default="0"),
    sa.Column("notes", sa.Text, nullable=True),
)


class Storage:
    def __init__(self, db_path: Path | None = None, database_url: str | None = None):
        """
        db_path: legacy/local-dev convenience -- if given, builds a SQLite
                 URL from this file path (kept for backward compatibility
                 with existing call sites and tests).
        database_url: full SQLAlchemy URL, e.g. a Postgres connection
                 string. Takes precedence over db_path. Falls back to
                 config.settings.DATABASE_URL (itself controlled by the
                 DATABASE_URL env var) when neither is given.
        """
        if database_url:
            url = database_url
        elif db_path is not None:
            url = f"sqlite:///{db_path}"
        else:
            url = DATABASE_URL

        self.engine = sa.create_engine(url, future=True)
        self.dialect = self.engine.dialect.name  # "sqlite" or "postgresql"
        self._init_schema()

    @contextmanager
    def _conn(self):
        with self.engine.begin() as conn:  # begin() auto-commits on clean exit
            yield conn

    def _init_schema(self):
        metadata.create_all(self.engine)
        self._migrate()

    def _migrate(self):
        """
        Idempotent migrations for columns added after initial release, so
        an existing database doesn't need to be dropped. metadata.create_all
        only creates missing TABLES, not missing COLUMNS on tables that
        already exist.
        """
        inspector = sa.inspect(self.engine)
        if "distress_scores" in inspector.get_table_names():
            existing_cols = {c["name"] for c in inspector.get_columns("distress_scores")}
            if "base_value" not in existing_cols:
                with self.engine.begin() as conn:
                    conn.execute(
                        sa.text(
                            "ALTER TABLE distress_scores ADD COLUMN base_value "
                            "REAL NOT NULL DEFAULT 0.0"
                        )
                    )
        if "narratives" in inspector.get_table_names():
            existing_cols = {c["name"] for c in inspector.get_columns("narratives")}
            if "context_hash" not in existing_cols:
                with self.engine.begin() as conn:
                    conn.execute(sa.text("ALTER TABLE narratives ADD COLUMN context_hash TEXT"))

    def _upsert(self, table: sa.Table, rows: list[dict]):
        """Dialect-aware upsert: INSERT ... ON CONFLICT DO UPDATE for both
        SQLite and Postgres (syntax differs, semantics are the same)."""
        if not rows:
            return
        insert_fn = pg_insert if self.dialect == "postgresql" else sqlite_insert
        pk_cols = [c.name for c in table.primary_key.columns]
        update_cols = [c.name for c in table.columns if c.name not in pk_cols]

        with self._conn() as conn:
            stmt = insert_fn(table).values(rows)
            if update_cols:
                stmt = stmt.on_conflict_do_update(
                    index_elements=pk_cols,
                    set_={col: getattr(stmt.excluded, col) for col in update_cols},
                )
            else:
                stmt = stmt.on_conflict_do_nothing(index_elements=pk_cols)
            conn.execute(stmt)

    # ------------------------------------------------------------------
    def save_raw_facts(self, ticker: str, cik: str, payload: dict, fetched_at: str):
        self._upsert(
            raw_facts,
            [{"ticker": ticker, "cik": cik, "fetched_at": fetched_at, "payload_json": json.dumps(payload)}],
        )

    def save_ratios(self, ticker: str, sector: str, ratios_by_year: dict[int, dict[str, float | None]]):
        rows = [
            {"ticker": ticker, "sector": sector, "fiscal_year": year, "ratio_name": name, "value": value}
            for year, ratios in ratios_by_year.items()
            for name, value in ratios.items()
        ]
        self._upsert(ratios_table, rows)

    def save_z_scores(self, ticker: str, sector: str, z_by_year: dict[int, dict[str, float | None]]):
        rows = [
            {"ticker": ticker, "sector": sector, "fiscal_year": year, "ratio_name": name, "z_value": value}
            for year, z_scores in z_by_year.items()
            for name, value in z_scores.items()
        ]
        self._upsert(z_scores_table, rows)

    def save_distress_score(
        self,
        ticker: str,
        fiscal_year: int,
        probability: float,
        shap_contributions: list,
        scored_at: str,
        base_value: float = 0.0,
    ):
        self._upsert(
            distress_scores_table,
            [
                {
                    "ticker": ticker,
                    "fiscal_year": fiscal_year,
                    "distress_probability": probability,
                    "shap_contributions_json": json.dumps(shap_contributions),
                    "base_value": base_value,
                    "scored_at": scored_at,
                }
            ],
        )

    def save_narrative(
        self, ticker: str, fiscal_year: int, text: str, generated_at: str, context_hash: str | None = None
    ):
        self._upsert(
            narratives_table,
            [
                {
                    "ticker": ticker,
                    "fiscal_year": fiscal_year,
                    "narrative_text": text,
                    "context_hash": context_hash,
                    "generated_at": generated_at,
                }
            ],
        )

    # ------------------------------------------------------------------
    def list_tickers(self) -> list[str]:
        """
        Distinct tickers that actually have ratio data, regardless of
        whether they appear in config/companies.yaml. The webapp dashboard
        uses this rather than the static config list, since scripts/run_demo.py
        seeds a synthetic universe (TEC0, ENE1, ...) that doesn't match the
        real tracked tickers (AAPL, MSFT, ...) -- the dashboard should show
        whatever the pipeline actually produced, not assume it matches config.
        """
        with self._conn() as conn:
            rows = conn.execute(sa.select(ratios_table.c.ticker).distinct()).fetchall()
            return sorted(r[0] for r in rows)

    def get_ratios(self, ticker: str) -> dict[int, dict[str, float | None]]:
        with self._conn() as conn:
            rows = conn.execute(
                sa.select(ratios_table.c.fiscal_year, ratios_table.c.ratio_name, ratios_table.c.value)
                .where(ratios_table.c.ticker == ticker)
            ).fetchall()
            result: dict[int, dict[str, float | None]] = {}
            for fy, ratio_name, value in rows:
                result.setdefault(fy, {})[ratio_name] = value
            return result

    def get_z_scores(self, ticker: str) -> dict[int, dict[str, float | None]]:
        with self._conn() as conn:
            rows = conn.execute(
                sa.select(z_scores_table.c.fiscal_year, z_scores_table.c.ratio_name, z_scores_table.c.z_value)
                .where(z_scores_table.c.ticker == ticker)
            ).fetchall()
            result: dict[int, dict[str, float | None]] = {}
            for fy, ratio_name, value in rows:
                result.setdefault(fy, {})[ratio_name] = value
            return result

    def get_distress_score(self, ticker: str, fiscal_year: int) -> dict | None:
        """Returns {'distress_probability': float, 'contributions': list[dict], 'base_value': float} or None."""
        with self._conn() as conn:
            row = conn.execute(
                sa.select(
                    distress_scores_table.c.distress_probability,
                    distress_scores_table.c.shap_contributions_json,
                    distress_scores_table.c.base_value,
                ).where(
                    (distress_scores_table.c.ticker == ticker)
                    & (distress_scores_table.c.fiscal_year == fiscal_year)
                )
            ).fetchone()
            if row is None:
                return None
            probability, contributions_json, base_value = row
            return {
                "distress_probability": probability,
                "contributions": json.loads(contributions_json),
                "base_value": base_value,
            }

    def get_narrative(self, ticker: str, fiscal_year: int) -> str | None:
        with self._conn() as conn:
            row = conn.execute(
                sa.select(narratives_table.c.narrative_text).where(
                    (narratives_table.c.ticker == ticker) & (narratives_table.c.fiscal_year == fiscal_year)
                )
            ).fetchone()
            return row[0] if row else None

    def get_narrative_record(self, ticker: str, fiscal_year: int) -> dict | None:
        """Returns {'narrative_text': str, 'context_hash': str | None} or None.
        Used by pipeline.py::stage_narrate to check whether a cached narrative's
        content_hash still matches the current data before deciding to skip
        regenerating it via Gemini."""
        with self._conn() as conn:
            row = conn.execute(
                sa.select(narratives_table.c.narrative_text, narratives_table.c.context_hash).where(
                    (narratives_table.c.ticker == ticker) & (narratives_table.c.fiscal_year == fiscal_year)
                )
            ).fetchone()
            if row is None:
                return None
            return {"narrative_text": row[0], "context_hash": row[1]}

    def log_run(self, run_id: str, started_at: str, status: str = "running"):
        self._upsert(pipeline_runs_table, [{"run_id": run_id, "started_at": started_at, "status": status}])

    def complete_run(
        self, run_id: str, completed_at: str, status: str, processed: int, failed: int, notes: str = ""
    ):
        with self._conn() as conn:
            conn.execute(
                sa.update(pipeline_runs_table)
                .where(pipeline_runs_table.c.run_id == run_id)
                .values(
                    completed_at=completed_at,
                    status=status,
                    companies_processed=processed,
                    companies_failed=failed,
                    notes=notes,
                )
            )
