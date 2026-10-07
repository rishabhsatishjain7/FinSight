"""
FinSight — SEC EDGAR client.

Thin, rate-limited HTTP wrapper around SEC's XBRL "companyfacts" API.
Caches raw JSON to disk per CIK so re-runs of the pipeline (or DAG retries)
don't re-hit the API unnecessarily.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import requests

from config.settings import (
    CACHE_DIR,
    SEC_COMPANYFACTS_URL,
    SEC_REQUEST_DELAY_SECONDS,
    SEC_SUBMISSIONS_URL,
    SEC_USER_AGENT,
)

logger = logging.getLogger("finsight.ingestion.sec_edgar")


class SECEdgarClient:
    """Fetches and caches raw company facts (XBRL) from SEC EDGAR."""

    def __init__(self, user_agent: str = SEC_USER_AGENT, cache_dir: Path = CACHE_DIR):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"})
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get(self, url: str) -> dict:
        time.sleep(SEC_REQUEST_DELAY_SECONDS)  # stay under SEC's 10 req/s limit
        resp = self.session.get(url, timeout=30)
        if resp.status_code == 404:
            raise FileNotFoundError(f"SEC EDGAR 404: {url}")
        resp.raise_for_status()
        return resp.json()

    def _cache_path(self, cik: str, kind: str) -> Path:
        return self.cache_dir / f"{cik}_{kind}.json"

    def fetch_company_facts(self, cik: str, force_refresh: bool = False) -> dict:
        """
        Fetch the full XBRL "companyfacts" payload for a CIK — every tag,
        every taxonomy (us-gaap, dei, ...), every filed period, in one call.
        This is the primary ingestion source; the ratio engine and XBRL
        parser both operate on this structure.
        """
        cache_path = self._cache_path(cik, "companyfacts")
        if cache_path.exists() and not force_refresh:
            logger.info("Cache hit for CIK %s companyfacts", cik)
            return json.loads(cache_path.read_text())

        url = SEC_COMPANYFACTS_URL.format(cik=cik)
        logger.info("Fetching companyfacts for CIK %s from %s", cik, url)
        try:
            data = self._get(url)
        except FileNotFoundError:
            logger.warning("No companyfacts found for CIK %s", cik)
            raise

        cache_path.write_text(json.dumps(data))
        return data

    def fetch_submissions(self, cik: str, force_refresh: bool = False) -> dict:
        """Fetch filing history/metadata (form types, dates) for a CIK."""
        cache_path = self._cache_path(cik, "submissions")
        if cache_path.exists() and not force_refresh:
            return json.loads(cache_path.read_text())

        url = SEC_SUBMISSIONS_URL.format(cik=cik)
        data = self._get(url)
        cache_path.write_text(json.dumps(data))
        return data

    def fetch_all(self, companies: list[dict], force_refresh: bool = False) -> dict[str, dict]:
        """
        Fetch companyfacts for a full list of {ticker, cik, name, sector} dicts.
        Individual company failures are logged and skipped rather than
        aborting the whole batch — this is what lets the pipeline degrade
        gracefully when one filer's data is missing or malformed.
        """
        results: dict[str, dict] = {}
        for company in companies:
            ticker = company["ticker"]
            try:
                results[ticker] = self.fetch_company_facts(
                    company["cik"], force_refresh=force_refresh
                )
            except Exception as exc:  # noqa: BLE001 — deliberately broad at batch boundary
                logger.error("Failed to fetch %s (CIK %s): %s", ticker, company["cik"], exc)
        return results
