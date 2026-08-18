from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

import httpx
from google import genai
from google.genai import types

from prophet_trader.config import Settings
from prophet_trader.models import Evidence, Forecast, Market, decimal_value


FORECAST_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "probability_yes": {"type": "number", "minimum": 0, "maximum": 1},
        "rationale": {
            "type": "string",
            "description": "Concise rationale, preferably 60 words and never over 250 words.",
        },
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
        "key_sources": {
            "type": "array",
            "maxItems": 2,
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Short source title."},
                    "url": {
                        "type": "string",
                        "description": "Short canonical source URL, not a grounding redirect URL.",
                    },
                },
                "required": ["title", "url"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["probability_yes", "rationale", "confidence", "key_sources"],
    "additionalProperties": False,
}


SYSTEM_PROMPT = """You are an independent prediction-market forecaster replicating the paper's
directional forecasting method. Resolve the binary contract exactly as written. Use Google Search
grounding for current evidence and prioritize primary settlement sources. Treat the market quote
as context rather than truth. Return only the requested JSON. Keep the rationale at or below 250
words. Prefer roughly 60 words. Include no more than two decisive sources, and use each
source's short canonical URL rather than a Google grounding redirect URL."""


@dataclass(frozen=True)
class PreparedForecast:
    ticker: str
    prompt: str
    prompt_version: str
    context_hash: str


@dataclass(frozen=True)
class ProviderForecastResponse:
    provider_request_id: str | None
    text: str
    raw_response: dict[str, Any]
    input_tokens: int
    cached_tokens: int
    reasoning_tokens: int
    output_tokens: int
    total_tokens: int
    search_queries: tuple[str, ...]
    duration_ms: int
    estimated_cost_dollars: Decimal


class ForecastRequestTimeout(RuntimeError):
    def __init__(self, message: str, *, provider_request_id: str | None = None) -> None:
        super().__init__(message)
        self.provider_request_id = provider_request_id


class ForecastQuotaExceeded(RuntimeError):
    def __init__(self, message: str, *, provider_request_id: str | None = None) -> None:
        super().__init__(message)
        self.provider_request_id = provider_request_id


def _market_prompt(market: Market, slot_iso: str, context: dict[str, Any]) -> str:
    sources = "\n".join(
        f"- {item.get('name', '')}: {item.get('url', '')}" for item in market.settlement_sources
    ) or "- None listed"
    context_json = json.dumps(context, sort_keys=True, separators=(",", ":"), default=str)
    return f"""Forecast whether this Kalshi contract resolves YES.

Two-hour decision slot (UTC): {slot_iso}
Ticker: {market.ticker}
Event: {market.title}
Market subtitle: {market.subtitle}
YES outcome: {market.yes_sub_title or 'YES as defined by the rules'}
NO outcome: {market.no_sub_title or 'NO as defined by the rules'}
Expected resolution time: {market.expected_expiration_time.isoformat()}
Trading close time: {market.close_time.isoformat()}
Primary rules: {market.rules_primary}
Secondary rules: {market.rules_secondary or '(none)'}
Named settlement sources:
{sources}
Early-close condition: {market.early_close_condition or '(none)'}

Executable market context (do not anchor on it):
- YES bid/ask: {market.yes_bid}/{market.yes_ask}
- NO bid/ask: {market.no_bid}/{market.no_ask}
- Buy-YES/Buy-NO displayed depth dollars: {market.buy_yes_depth_dollars}/{market.buy_no_depth_dollars}
- Volume/open interest: {market.volume}/{market.open_interest}

Continuously refreshed lifecycle and official-source context:
{context_json}

Return probability_yes from 0 to 1, a rationale of at most 250 words, confidence as low/medium/high,
and at most two key_sources. Prefer a rationale near 60 words and short canonical source URLs.
Do not return any other fields."""


def prepare_forecast(
    market: Market,
    *,
    slot_iso: str,
    context: dict[str, Any],
    prompt_version: str,
) -> PreparedForecast:
    prompt = _market_prompt(market, slot_iso, context)
    material = json.dumps(
        {
            "system": SYSTEM_PROMPT,
            "prompt": prompt,
            "prompt_version": prompt_version,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return PreparedForecast(
        ticker=market.ticker,
        prompt=prompt,
        prompt_version=prompt_version,
        context_hash=hashlib.sha256(material.encode("utf-8")).hexdigest(),
    )


def parse_forecast_payload(
    payload: dict[str, Any],
    *,
    model: str,
    response_id: str | None = None,
) -> Forecast:
    allowed = {"probability_yes", "rationale", "confidence", "key_sources"}
    if set(payload) != allowed:
        raise ValueError("forecast JSON contains missing or unexpected fields")
    probability = decimal_value(payload.get("probability_yes"), Decimal("-1"))
    if not probability.is_finite() or not Decimal("0") <= probability <= Decimal("1"):
        raise ValueError("model returned probability_yes outside [0,1]")

    rationale = str(payload.get("rationale", "")).strip()
    if not rationale:
        raise ValueError("model returned an empty rationale")
    if len(rationale.split()) > 250:
        raise ValueError("model rationale exceeds 250 words")

    confidence = str(payload.get("confidence", "")).strip().lower()
    if confidence not in {"low", "medium", "high"}:
        raise ValueError("model returned invalid confidence")

    raw_sources = payload.get("key_sources")
    if not isinstance(raw_sources, list) or len(raw_sources) > 2:
        raise ValueError("model returned invalid key_sources")
    evidence: list[Evidence] = []
    seen: set[str] = set()
    for item in raw_sources:
        if not isinstance(item, dict) or set(item) != {"title", "url"}:
            raise ValueError("model returned malformed key source")
        url = str(item.get("url", "")).strip()
        if url.startswith(("http://", "https://")) and url not in seen:
            evidence.append(Evidence(claim=str(item.get("title", "")).strip(), source_url=url))
            seen.add(url)

    uncertainty = {"high": "low", "medium": "medium", "low": "high"}[confidence]
    return Forecast(
        probability_yes=probability,
        rationale=rationale,
        resolution_clear=True,
        resolution_clarity_reason="Contract interpretation included in strict paper forecast",
        uncertainty=uncertainty,
        confidence=confidence,
        evidence=tuple(evidence),
        model=model,
        response_id=response_id,
        raw=dict(payload),
    )


def _provider_request_id(value: Any) -> str | None:
    response_id = getattr(value, "response_id", None)
    if response_id:
        return str(response_id)
    response = getattr(value, "response", None)
    headers = getattr(response, "headers", None)
    if headers:
        for name in ("x-request-id", "x-goog-request-id", "x-correlation-id"):
            if headers.get(name):
                return str(headers[name])
    return None


def _search_queries(raw: dict[str, Any]) -> tuple[str, ...]:
    queries: list[str] = []
    for candidate in raw.get("candidates") or []:
        if not isinstance(candidate, dict):
            continue
        metadata = candidate.get("grounding_metadata") or candidate.get("groundingMetadata") or {}
        for query in metadata.get("web_search_queries") or metadata.get("webSearchQueries") or []:
            text = str(query).strip()
            if text and text not in queries:
                queries.append(text)
    return tuple(queries)


def _usage(raw: dict[str, Any]) -> tuple[int, int, int, int, int]:
    usage = raw.get("usage_metadata") or raw.get("usageMetadata") or {}
    value = lambda snake, camel: int(usage.get(snake) or usage.get(camel) or 0)
    return (
        value("prompt_token_count", "promptTokenCount"),
        value("cached_content_token_count", "cachedContentTokenCount"),
        value("thoughts_token_count", "thoughtsTokenCount"),
        value("candidates_token_count", "candidatesTokenCount"),
        value("total_token_count", "totalTokenCount"),
    )


def estimate_cost(
    settings: Settings,
    *,
    input_tokens: int,
    cached_tokens: int,
    reasoning_tokens: int,
    output_tokens: int,
    search_queries: int,
) -> Decimal:
    million = Decimal("1000000")
    uncached = max(0, input_tokens - cached_tokens)
    return (
        Decimal(uncached) * settings.gemini_input_cost_per_million / million
        + Decimal(cached_tokens) * settings.gemini_cached_input_cost_per_million / million
        + Decimal(reasoning_tokens + output_tokens)
        * settings.gemini_output_cost_per_million
        / million
        + Decimal(search_queries) * settings.gemini_search_cost_per_query
    )


def _is_quota_error(exc: Exception) -> bool:
    text = str(exc).lower()
    code = getattr(exc, "code", None)
    return (
        "insufficient_quota" in text
        or "insufficient quota" in text
        or "billing quota" in text
        or "spend limit" in text
        or (code == 429 and "quota" in text)
    )


def _is_timeout_error(exc: Exception) -> bool:
    """Classify provider deadline failures as unknown-outcome timeouts.

    The Google client can surface a request deadline as a generic server error
    carrying HTTP 504 instead of an ``httpx`` timeout. Treating both forms the
    same preserves the no-retry guarantee when the provider may have processed
    the request.
    """
    text = str(exc).lower()
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    return (
        isinstance(exc, (TimeoutError, httpx.TimeoutException))
        or str(code) == "504"
        or "timed out" in text
        or "timeout" in text
        or "deadline_exceeded" in text
        or "deadline exceeded" in text
        or "deadline expired" in text
    )


class GeminiForecaster:
    def __init__(self, settings: Settings, *, client: Any | None = None) -> None:
        self.settings = settings
        if client is None:
            if not settings.gemini_api_key:
                raise RuntimeError("GEMINI_API_KEY is required for forecasting")
            client = genai.Client(
                api_key=settings.gemini_api_key,
                http_options=types.HttpOptions(
                    timeout=settings.request_timeout_seconds * 1000,
                    retry_options=types.HttpRetryOptions(attempts=1),
                ),
            )
        self.client = client

    def prepare(
        self,
        market: Market,
        *,
        slot_iso: str,
        context: dict[str, Any],
    ) -> PreparedForecast:
        return prepare_forecast(
            market,
            slot_iso=slot_iso,
            context=context,
            prompt_version=self.settings.forecast_prompt_version,
        )

    def request(self, prepared: PreparedForecast) -> ProviderForecastResponse:
        started = time.monotonic()
        try:
            response = self.client.models.generate_content(
                model=self.settings.gemini_model,
                contents=prepared.prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    thinking_config=types.ThinkingConfig(thinking_level="high"),
                    tools=[types.Tool(google_search=types.GoogleSearch())],
                    response_mime_type="application/json",
                    response_json_schema=FORECAST_SCHEMA,
                    max_output_tokens=self.settings.gemini_max_output_tokens,
                    candidate_count=1,
                ),
            )
        except Exception as exc:
            provider_id = _provider_request_id(exc)
            if _is_quota_error(exc):
                raise ForecastQuotaExceeded(str(exc), provider_request_id=provider_id) from exc
            if _is_timeout_error(exc):
                raise ForecastRequestTimeout(str(exc), provider_request_id=provider_id) from exc
            raise

        duration_ms = round((time.monotonic() - started) * 1000)
        raw = response.model_dump(mode="json", by_alias=False, exclude_none=True)
        input_tokens, cached_tokens, reasoning_tokens, output_tokens, total_tokens = _usage(raw)
        queries = _search_queries(raw)
        provider_id = _provider_request_id(response)
        return ProviderForecastResponse(
            provider_request_id=provider_id,
            text=str(response.text or ""),
            raw_response=raw,
            input_tokens=input_tokens,
            cached_tokens=cached_tokens,
            reasoning_tokens=reasoning_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            search_queries=queries,
            duration_ms=duration_ms,
            estimated_cost_dollars=estimate_cost(
                self.settings,
                input_tokens=input_tokens,
                cached_tokens=cached_tokens,
                reasoning_tokens=reasoning_tokens,
                output_tokens=output_tokens,
                search_queries=len(queries),
            ),
        )

    def parse(self, response: ProviderForecastResponse) -> Forecast:
        # The SDK exposes the schema-decoded object separately. Prefer it when present;
        # the raw response has already been journaled by the engine before this method runs.
        payload = response.raw_response.get("parsed")
        if payload is None:
            finish_reasons = {
                str(candidate.get("finish_reason") or candidate.get("finishReason") or "")
                for candidate in response.raw_response.get("candidates") or []
                if isinstance(candidate, dict)
            }
            if "MAX_TOKENS" in finish_reasons:
                raise RuntimeError(
                    "Gemini exhausted the 2,000-token output budget before completing JSON"
                )
            if not response.text:
                raise RuntimeError("Gemini returned no forecast JSON")
            try:
                payload = json.loads(response.text)
            except json.JSONDecodeError as exc:
                raise RuntimeError("Gemini forecast was not valid JSON") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("Gemini forecast JSON was not an object")
        return parse_forecast_payload(
            payload,
            model=self.settings.gemini_model,
            response_id=response.provider_request_id,
        )
