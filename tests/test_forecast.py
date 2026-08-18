from __future__ import annotations

import json
import unittest
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx

from prophet_trader.config import Settings
from prophet_trader.forecast import (
    FORECAST_SCHEMA,
    ForecastRequestTimeout,
    GeminiForecaster,
    parse_forecast_payload,
)
from prophet_trader.models import Market


def _settings() -> Settings:
    return Settings(
        root=Path("/tmp/prophet-trader-forecast-test"),
        trading_mode="paper",
        kalshi_env="demo",
        live_trading_enabled=False,
        allow_production=False,
        kalshi_api_key_id=None,
        kalshi_private_key_path=None,
        kalshi_private_key_pem=None,
        gemini_api_key=None,
        gemini_model="gemini-3.1-pro-preview",
        gemini_reasoning_level="high",
        cycle_seconds=7200,
        min_days_to_close=Decimal("2"),
        max_days_to_close=Decimal("14"),
        stop_hours_before_resolution=Decimal("3"),
        max_close_resolution_gap_hours=Decimal("1"),
        refresh_move_threshold=Decimal("0.10"),
        position_scale_contracts=Decimal("100"),
        min_actionable_edge=Decimal("0.02"),
        max_markets_per_cycle=8,
        paper_starting_cash=Decimal("200"),
        max_target_contracts=Decimal("50"),
        max_order_contracts=Decimal("25"),
        max_market_risk_dollars=Decimal("25"),
        max_total_exposure_dollars=Decimal("100"),
        max_spread=Decimal("0.15"),
        min_liquidity_dollars=Decimal("1"),
        min_evidence_sources=1,
        block_high_uncertainty=True,
        state_db_path=Path("/tmp/prophet-trader-forecast-test/state.sqlite3"),
        kill_switch_path=Path("/tmp/prophet-trader-forecast-test/STOP"),
        request_timeout_seconds=30,
        max_http_retries=3,
        gemini_max_output_tokens=2000,
    )


def _market() -> Market:
    now = datetime(2026, 7, 22, 12, 0, tzinfo=timezone.utc)
    return Market(
        ticker="KXTEST-26JUL31-YES",
        event_ticker="KXTEST-26JUL31",
        series_ticker="KXTEST",
        category="Economics",
        title="Will the test statistic exceed 100?",
        subtitle="Test statistic above 100 by July 31",
        yes_sub_title="Above 100",
        no_sub_title="At or below 100",
        rules_primary="Resolves YES if the official release exceeds 100.",
        rules_secondary="The first unrevised release controls.",
        settlement_sources=(
            {"name": "Official release", "url": "https://official.example/release"},
        ),
        close_time=now + timedelta(days=7),
        expected_expiration_time=now + timedelta(days=7),
        yes_bid=Decimal("0.41"),
        yes_ask=Decimal("0.44"),
        no_bid=Decimal("0.56"),
        no_ask=Decimal("0.59"),
        yes_bid_size=Decimal("20"),
        yes_ask_size=Decimal("25"),
        liquidity_dollars=Decimal("500"),
        volume=Decimal("1000"),
        open_interest=Decimal("300"),
        can_close_early=False,
        early_close_condition="",
    )


def _payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "probability_yes": 0.63,
        "rationale": "Current official data and the base rate put the outcome above the quote.",
        "confidence": "medium",
        "key_sources": [
            {"title": "Official release", "url": "https://official.example/latest"}
        ],
    }
    payload.update(overrides)
    return payload


class ForecastPayloadTests(unittest.TestCase):
    def test_accepts_exact_small_schema(self) -> None:
        forecast = parse_forecast_payload(
            _payload(), model="gemini-3.1-pro-preview", response_id="response-1"
        )
        self.assertEqual(forecast.probability_yes, Decimal("0.63"))
        self.assertEqual(forecast.confidence, "medium")
        self.assertEqual(forecast.response_id, "response-1")
        self.assertEqual(len(forecast.evidence), 1)

    def test_rejects_extra_fields_and_overlong_rationale(self) -> None:
        with self.assertRaisesRegex(ValueError, "unexpected fields"):
            parse_forecast_payload(_payload(extra=True), model="test")
        with self.assertRaisesRegex(ValueError, "250 words"):
            parse_forecast_payload(_payload(rationale="word " * 251), model="test")

    def test_rejects_more_than_two_sources(self) -> None:
        sources = [
            {"title": str(index), "url": f"https://example.com/{index}"}
            for index in range(3)
        ]
        with self.assertRaisesRegex(ValueError, "key_sources"):
            parse_forecast_payload(_payload(key_sources=sources), model="test")


class _FakeResponse:
    response_id = "gemini-response-1"

    def __init__(self, text: str) -> None:
        self.text = text

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "response_id": self.response_id,
            "candidates": [
                {
                    "grounding_metadata": {
                        "web_search_queries": ["official statistic latest"]
                    }
                }
            ],
            "usage_metadata": {
                "prompt_token_count": 100,
                "cached_content_token_count": 10,
                "thoughts_token_count": 50,
                "candidates_token_count": 25,
                "total_token_count": 175,
            },
            "text": self.text,
        }


class _FakeModels:
    def __init__(self, result: Any) -> None:
        self.result = result
        self.calls: list[dict[str, Any]] = []

    def generate_content(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


class _FakeClient:
    def __init__(self, result: Any) -> None:
        self.models = _FakeModels(result)


class GeminiForecasterTests(unittest.TestCase):
    def test_uses_high_reasoning_search_grounding_and_strict_json(self) -> None:
        client = _FakeClient(_FakeResponse(json.dumps(_payload())))
        forecaster = GeminiForecaster(_settings(), client=client)
        prepared = forecaster.prepare(
            _market(), slot_iso="2026-07-22T12:00:00+00:00", context={}
        )
        response = forecaster.request(prepared)
        forecast = forecaster.parse(response)

        self.assertEqual(len(client.models.calls), 1)
        request = client.models.calls[0]
        self.assertEqual(request["model"], "gemini-3.1-pro-preview")
        config = request["config"]
        self.assertEqual(str(config.thinking_config.thinking_level), "ThinkingLevel.HIGH")
        self.assertIsNotNone(config.tools[0].google_search)
        self.assertEqual(config.response_mime_type, "application/json")
        self.assertEqual(config.response_json_schema, FORECAST_SCHEMA)
        self.assertEqual(config.max_output_tokens, 2000)
        self.assertEqual(response.provider_request_id, "gemini-response-1")
        self.assertEqual(response.reasoning_tokens, 50)
        self.assertEqual(response.search_queries, ("official statistic latest",))
        self.assertEqual(forecast.confidence, "medium")

    def test_prefers_sdk_parsed_payload_after_raw_response_is_available(self) -> None:
        response = _FakeResponse("truncated JSON")
        client = _FakeClient(response)
        forecaster = GeminiForecaster(_settings(), client=client)
        prepared = forecaster.prepare(
            _market(), slot_iso="2026-07-22T12:00:00+00:00", context={}
        )
        provider_response = forecaster.request(prepared)
        provider_response.raw_response["parsed"] = _payload()

        forecast = forecaster.parse(provider_response)

        self.assertEqual(forecast.probability_yes, Decimal("0.63"))

    def test_reports_max_tokens_instead_of_generic_invalid_json(self) -> None:
        response = _FakeResponse('{"probability_yes": 0.5')
        client = _FakeClient(response)
        forecaster = GeminiForecaster(_settings(), client=client)
        prepared = forecaster.prepare(
            _market(), slot_iso="2026-07-22T12:00:00+00:00", context={}
        )
        provider_response = forecaster.request(prepared)
        provider_response.raw_response["candidates"][0]["finish_reason"] = "MAX_TOKENS"

        with self.assertRaisesRegex(RuntimeError, "2,000-token output budget"):
            forecaster.parse(provider_response)

        self.assertEqual(len(client.models.calls), 1)

    def test_provider_unavailable_is_not_retried_in_same_call(self) -> None:
        client = _FakeClient(RuntimeError("503 UNAVAILABLE: high demand"))
        forecaster = GeminiForecaster(_settings(), client=client)
        prepared = forecaster.prepare(
            _market(), slot_iso="2026-07-22T12:00:00+00:00", context={}
        )

        with self.assertRaisesRegex(RuntimeError, "503 UNAVAILABLE"):
            forecaster.request(prepared)

        self.assertEqual(len(client.models.calls), 1)

    def test_timeout_issues_exactly_one_provider_request(self) -> None:
        client = _FakeClient(httpx.ReadTimeout("timed out"))
        forecaster = GeminiForecaster(_settings(), client=client)
        prepared = forecaster.prepare(
            _market(), slot_iso="2026-07-22T12:00:00+00:00", context={}
        )

        with self.assertRaises(ForecastRequestTimeout):
            forecaster.request(prepared)

        self.assertEqual(len(client.models.calls), 1)

    def test_deadline_exceeded_is_unknown_and_not_retried(self) -> None:
        class DeadlineExceeded(RuntimeError):
            code = 504

        client = _FakeClient(DeadlineExceeded("504 DEADLINE_EXCEEDED"))
        forecaster = GeminiForecaster(_settings(), client=client)
        prepared = forecaster.prepare(
            _market(), slot_iso="2026-07-22T12:00:00+00:00", context={}
        )

        with self.assertRaises(ForecastRequestTimeout):
            forecaster.request(prepared)

        self.assertEqual(len(client.models.calls), 1)

    def test_requires_gemini_key_without_injected_client(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "GEMINI_API_KEY"):
            GeminiForecaster(_settings())


if __name__ == "__main__":
    unittest.main()
