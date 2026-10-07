"""
FinSight — Airflow DAG.

Orchestrates the five-stage pipeline (ingest -> transform -> score -> narrate
-> report) as a linear dependency chain, plus a weekly model retrain branch.
Each task wraps a function from pipeline.py, so tasks are independently
retryable and the exact same code path runs whether triggered by Airflow
or invoked directly via `python pipeline.py --stage <name>` for local dev.

To install into an Airflow deployment:
    cp dags/finsight_pipeline_dag.py $AIRFLOW_HOME/dags/
    # and make sure the FinSight project root is on PYTHONPATH for the
    # Airflow workers (e.g. via a plugins/ symlink or PYTHONPATH env var).
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.trigger_rule import TriggerRule

# Ensure project root is importable from the Airflow worker context.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline import stage_ingest, stage_narrate, stage_report, stage_score, stage_transform  # noqa: E402
from scoring.train import train_and_save  # noqa: E402
from database.storage import Storage  # noqa: E402
from config.settings import flat_company_list  # noqa: E402

DEFAULT_ARGS = {
    "owner": "finsight",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}


def _retrain_model_callable(**context):
    """Weekly retrain: pull latest z-scores for the whole universe and refit XGBoost."""
    storage = Storage()
    companies = flat_company_list()
    z_scores_by_company = {c["ticker"]: storage.get_z_scores(c["ticker"]) for c in companies}
    z_scores_by_company = {k: v for k, v in z_scores_by_company.items() if v}
    metrics = train_and_save(z_scores_by_company)
    context["ti"].xcom_push(key="retrain_metrics", value=metrics)


# ---------------------------------------------------------------------------
# Daily pipeline: ingest -> transform -> score -> narrate -> report
# ---------------------------------------------------------------------------
with DAG(
    dag_id="finsight_daily_pipeline",
    description="FinSight: ingestion -> ratio/Z-score transform -> XGBoost scoring -> Gemini narrative -> PDF report",
    default_args=DEFAULT_ARGS,
    schedule_interval="0 6 * * 1-5",  # weekdays 06:00
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["finsight", "finance", "xbrl"],
) as daily_dag:

    ingest_task = PythonOperator(
        task_id="ingest_xbrl_filings",
        python_callable=stage_ingest,
        op_kwargs={"force_refresh": False},
    )

    transform_task = PythonOperator(
        task_id="compute_ratios_and_zscores",
        python_callable=stage_transform,
    )

    score_task = PythonOperator(
        task_id="score_distress_xgboost_shap",
        python_callable=stage_score,
    )

    narrate_task = PythonOperator(
        task_id="generate_gemini_narratives",
        python_callable=stage_narrate,
    )

    report_task = PythonOperator(
        task_id="render_pdf_reports",
        python_callable=stage_report,
        trigger_rule=TriggerRule.ALL_DONE,  # still render for companies that scored ok
    )

    ingest_task >> transform_task >> score_task >> narrate_task >> report_task


# ---------------------------------------------------------------------------
# Weekly retrain: refit the XGBoost distress model on the accumulated
# ratio/Z-score history once enough new fiscal periods have landed.
# ---------------------------------------------------------------------------
with DAG(
    dag_id="finsight_weekly_retrain",
    description="FinSight: weekly XGBoost distress model retrain",
    default_args=DEFAULT_ARGS,
    schedule_interval="0 5 * * 0",  # Sundays 05:00
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["finsight", "ml", "retrain"],
) as retrain_dag:

    retrain_task = PythonOperator(
        task_id="retrain_distress_model",
        python_callable=_retrain_model_callable,
    )
