from unittest.mock import MagicMock, patch

import pytest
import requests

from narrative.context_builder import CompanyContext
from narrative.gemini_client import GeminiAPIError, GeminiNarrativeClient, RETRYABLE_STATUS_CODES


def _sample_context() -> CompanyContext:
    return CompanyContext(
        ticker="TEST",
        name="Test Corp",
        sector="technology",
        fiscal_year=2023,
        ratios={"net_margin": 0.05},
        z_scores={"net_margin": -1.2},
        distress_probability=0.6,
        shap_contributions=[{"feature": "net_margin", "shap_value": 0.5, "feature_value": -1.2}],
    )


def _mock_response(status_code: int, body: dict | None = None) -> MagicMock:
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status_code
    resp.reason = "Error"
    resp.url = "https://example.test/gemini"
    resp.json.return_value = body or {}
    resp.text = str(body or {})

    def raise_for_status():
        if status_code >= 400:
            raise requests.exceptions.HTTPError(f"{status_code} error", response=resp)

    resp.raise_for_status.side_effect = raise_for_status
    return resp


SUCCESS_BODY = {"candidates": [{"content": {"parts": [{"text": "A narrative."}]}}]}


class TestRetryBackoff:
    def test_succeeds_immediately_without_retry(self):
        client = GeminiNarrativeClient(api_key="fake-key", max_retries=3, base_delay=0.01)
        client.session.post = MagicMock(return_value=_mock_response(200, SUCCESS_BODY))

        result = client.generate_narrative(_sample_context())
        assert result == "A narrative."
        assert client.session.post.call_count == 1

    def test_retries_on_429_then_succeeds(self):
        client = GeminiNarrativeClient(api_key="fake-key", max_retries=3, base_delay=0.01)
        client.session.post = MagicMock(
            side_effect=[_mock_response(429), _mock_response(429), _mock_response(200, SUCCESS_BODY)]
        )

        with patch("time.sleep"):  # don't actually wait during tests
            result = client.generate_narrative(_sample_context())

        assert result == "A narrative."
        assert client.session.post.call_count == 3

    def test_retries_on_5xx_then_succeeds(self):
        client = GeminiNarrativeClient(api_key="fake-key", max_retries=3, base_delay=0.01)
        client.session.post = MagicMock(
            side_effect=[_mock_response(503), _mock_response(200, SUCCESS_BODY)]
        )

        with patch("time.sleep"):
            result = client.generate_narrative(_sample_context())

        assert result == "A narrative."
        assert client.session.post.call_count == 2

    def test_retries_on_connection_error_then_succeeds(self):
        client = GeminiNarrativeClient(api_key="fake-key", max_retries=3, base_delay=0.01)
        client.session.post = MagicMock(
            side_effect=[requests.exceptions.ConnectionError("refused"), _mock_response(200, SUCCESS_BODY)]
        )

        with patch("time.sleep"):
            result = client.generate_narrative(_sample_context())

        assert result == "A narrative."
        assert client.session.post.call_count == 2

    def test_does_not_retry_on_400_bad_request(self):
        """A malformed request will fail identically on every retry -- must fail fast, not burn attempts."""
        client = GeminiNarrativeClient(api_key="fake-key", max_retries=3, base_delay=0.01)
        client.session.post = MagicMock(return_value=_mock_response(400))

        with pytest.raises(requests.exceptions.HTTPError):
            client.generate_narrative(_sample_context())

        assert client.session.post.call_count == 1  # no retries attempted

    def test_does_not_retry_on_401_unauthorized(self):
        client = GeminiNarrativeClient(api_key="bad-key", max_retries=3, base_delay=0.01)
        client.session.post = MagicMock(return_value=_mock_response(401))

        with pytest.raises(requests.exceptions.HTTPError):
            client.generate_narrative(_sample_context())

        assert client.session.post.call_count == 1

    def test_gives_up_after_max_retries_exhausted(self):
        client = GeminiNarrativeClient(api_key="fake-key", max_retries=2, base_delay=0.01)
        client.session.post = MagicMock(return_value=_mock_response(503))

        with patch("time.sleep"):
            with pytest.raises(GeminiAPIError):
                client.generate_narrative(_sample_context())

        assert client.session.post.call_count == 3  # initial attempt + 2 retries

    def test_backoff_delay_grows_exponentially_and_is_capped(self):
        client = GeminiNarrativeClient(
            api_key="fake-key", max_retries=5, base_delay=1.0, max_delay=4.0
        )
        client.session.post = MagicMock(return_value=_mock_response(503))

        sleep_calls = []
        with patch("time.sleep", side_effect=lambda d: sleep_calls.append(d)):
            with pytest.raises(GeminiAPIError):
                client.generate_narrative(_sample_context())

        # delays before jitter: 1, 2, 4, 4, 4 (capped at max_delay=4.0); jitter adds up to 25% on top
        assert len(sleep_calls) == 5
        assert all(d >= 1.0 for d in sleep_calls)
        assert all(d <= 4.0 * 1.25 for d in sleep_calls)
        assert sleep_calls[0] < sleep_calls[2]  # grows before hitting the cap

    def test_generate_batch_isolates_failures_per_company(self):
        client = GeminiNarrativeClient(api_key="fake-key", max_retries=0, base_delay=0.01)
        client.session.post = MagicMock(return_value=_mock_response(500))

        with patch("time.sleep"):
            results = client.generate_batch([_sample_context()])

        assert "[Narrative unavailable" in results["TEST"]


class TestContentHash:
    def test_identical_context_produces_identical_hash(self):
        ctx1 = _sample_context()
        ctx2 = _sample_context()
        assert ctx1.content_hash() == ctx2.content_hash()

    def test_changed_outlier_ratio_changes_hash(self):
        """A ratio that surfaces in the prompt (here, as a notable |z|>=1.5
        outlier) must change the hash when its value changes -- this is
        content Gemini actually sees."""
        ctx1 = _sample_context()
        ctx1.z_scores = {"net_margin": -2.0}  # crosses the notable-outlier threshold
        ctx2 = _sample_context()
        ctx2.z_scores = {"net_margin": -2.0}
        ctx2.ratios = {"net_margin": 0.99}  # changes the raw value shown alongside the outlier
        assert ctx1.content_hash() != ctx2.content_hash()

    def test_ratio_change_invisible_to_prompt_does_not_change_hash(self):
        """
        This is intentional, not a bug: if a ratio isn't a notable outlier,
        has no YoY history, and isn't referenced by SHAP, its raw value
        never appears in the rendered prompt -- Gemini genuinely can't see
        the difference, so the narrative wouldn't change either. The hash
        should reflect exactly what's visible to Gemini, nothing more.
        """
        ctx1 = _sample_context()  # net_margin z=-1.2, below the 1.5 outlier threshold
        ctx2 = _sample_context()
        ctx2.ratios = {"net_margin": 0.99}
        assert ctx1.content_hash() == ctx2.content_hash()

    def test_changed_distress_probability_changes_hash(self):
        ctx1 = _sample_context()
        ctx2 = _sample_context()
        ctx2.distress_probability = 0.95
        assert ctx1.content_hash() != ctx2.content_hash()

    def test_hash_is_deterministic_across_calls(self):
        ctx = _sample_context()
        assert ctx.content_hash() == ctx.content_hash()
