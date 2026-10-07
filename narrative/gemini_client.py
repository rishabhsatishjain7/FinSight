"""
FinSight — Gemini narrative engine.

Calls the Google Gemini REST API directly (no SDK dependency) to turn a
structured CompanyContext into a readable analyst narrative. The prompt is
explicitly constrained to only use figures present in the injected context
block, which is what makes this "RAG" rather than free-form generation —
the model is grounded in retrieved/computed facts, not trained financial
knowledge that could be stale or fabricated.

Retry/backoff: transient failures (429 rate limits, 5xx server errors,
connection drops, timeouts) are retried with exponential backoff + jitter.
Non-transient failures (400 bad request, 401/403 auth, 404) fail
immediately — retrying a request Gemini has already rejected as invalid
just burns time and quota for the same outcome.

Caching: narrative generation is content-addressed via
CompanyContext.content_hash() (pipeline.py::stage_narrate is the caller
that actually uses this — see database/storage.py's `context_hash` column
on the narratives table). Re-running the narrate stage on data that hasn't
changed since the last run reuses the stored narrative instead of calling
Gemini again, since the exact same structured context would just produce
an equivalent narrative at real API cost.
"""
from __future__ import annotations

import logging
import random
import time

import requests

from config.settings import GEMINI_API_KEY, GEMINI_API_URL, GEMINI_MODEL
from narrative.context_builder import CompanyContext

logger = logging.getLogger("finsight.narrative.gemini_client")

SYSTEM_INSTRUCTION = """You are a financial analyst writing a concise distress-screening \
narrative for an internal credit/equity research report. You will be given a structured \
context block containing computed financial ratios, sector-relative Z-scores, and SHAP \
attributions from a distress classification model.

Rules:
1. Only reference figures that appear in the provided context. Do not invent numbers.
2. Write 3-4 short paragraphs: (a) overall distress read, (b) key drivers from SHAP, \
(c) notable sector outliers, (d) trend/trajectory based on YoY deltas.
3. Be direct about risk level. Do not hedge with generic disclaimers.
4. Plain prose, no markdown headers, no bullet lists — this is inserted into a formatted PDF \
report that already has its own headers.
"""

# Status codes worth retrying: rate limiting and server-side transient errors.
# Everything else (400/401/403/404/...) means the request itself is wrong
# and will fail identically on every retry.
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


class GeminiAPIError(RuntimeError):
    """Raised when Gemini returns a non-retryable error or retries are exhausted."""


class GeminiNarrativeClient:
    def __init__(
        self,
        api_key: str = GEMINI_API_KEY,
        model: str = GEMINI_MODEL,
        max_retries: int = 4,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
    ):
        if not api_key:
            logger.warning(
                "GEMINI_API_KEY not set — narrative generation will fail until configured."
            )
        self.api_key = api_key
        self.model = model
        self.url = GEMINI_API_URL.format(model=model)
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.session = requests.Session()

    def generate_narrative(self, context: CompanyContext, temperature: float = 0.3) -> str:
        prompt_context = context.to_prompt_context()

        payload = {
            "systemInstruction": {"parts": [{"text": SYSTEM_INSTRUCTION}]},
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": (
                                "Here is the structured financial context for this company. "
                                "Write the analyst narrative per the rules above.\n\n"
                                f"{prompt_context}"
                            )
                        }
                    ],
                }
            ],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": 700,
            },
        }

        resp = self._post_with_retry(payload)
        data = resp.json()
        try:
            candidates = data["candidates"]
            parts = candidates[0]["content"]["parts"]
            return "".join(p.get("text", "") for p in parts).strip()
        except (KeyError, IndexError) as exc:
            logger.error("Unexpected Gemini response shape: %s", data)
            raise GeminiAPIError("Gemini returned no usable narrative content") from exc

    def _post_with_retry(self, payload: dict) -> requests.Response:
        """
        Exponential backoff with jitter: delay = min(max_delay, base_delay * 2**attempt) + jitter.
        Jitter avoids every concurrent caller retrying in lockstep against a
        rate-limited endpoint (thundering herd).
        """
        last_exc: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                resp = self.session.post(
                    self.url,
                    params={"key": self.api_key},
                    json=payload,
                    timeout=60,
                )
            except requests.exceptions.RequestException as exc:
                # Connection errors / timeouts -- always transient, always retryable.
                last_exc = exc
                if attempt >= self.max_retries:
                    break
                self._sleep_backoff(attempt, reason=f"{type(exc).__name__}: {exc}")
                continue

            if resp.status_code == 200:
                return resp

            if resp.status_code not in RETRYABLE_STATUS_CODES:
                logger.error("Gemini API non-retryable error %s: %s", resp.status_code, resp.text[:500])
                resp.raise_for_status()  # raises requests.HTTPError

            last_exc = requests.exceptions.HTTPError(
                f"{resp.status_code} {resp.reason} for url: {resp.url}", response=resp
            )
            if attempt >= self.max_retries:
                break
            self._sleep_backoff(attempt, reason=f"HTTP {resp.status_code}")

        raise GeminiAPIError(
            f"Gemini API call failed after {self.max_retries + 1} attempts: {last_exc}"
        ) from last_exc

    def _sleep_backoff(self, attempt: int, reason: str):
        delay = min(self.max_delay, self.base_delay * (2 ** attempt))
        jitter = random.uniform(0, delay * 0.25)
        total = delay + jitter
        logger.warning(
            "Gemini call failed (%s), retrying in %.1fs (attempt %d/%d)",
            reason, total, attempt + 1, self.max_retries,
        )
        time.sleep(total)

    def generate_batch(self, contexts: list[CompanyContext]) -> dict[str, str]:
        """Generate narratives for multiple companies; failures are isolated per-company."""
        narratives = {}
        for ctx in contexts:
            try:
                narratives[ctx.ticker] = self.generate_narrative(ctx)
            except Exception as exc:  # noqa: BLE001
                logger.error("Narrative generation failed for %s: %s", ctx.ticker, exc)
                narratives[ctx.ticker] = (
                    "[Narrative unavailable — Gemini API call failed. "
                    "See structured metrics above.]"
                )
        return narratives
