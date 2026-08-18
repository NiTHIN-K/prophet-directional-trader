from __future__ import annotations

import hashlib
import ipaddress
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, Protocol, Sequence
from urllib.parse import urljoin, urlparse

import requests

from prophet_trader.config import Settings
from prophet_trader.engine import discover_markets, get_market
from prophet_trader.models import Market, ZERO, parse_timestamp
from prophet_trader.store import StateStore
from prophet_trader.strategy import market_eligibility


LOGGER = logging.getLogger(__name__)


class EventBroker(Protocol):
    def snapshot(self) -> Any: ...


@dataclass(frozen=True)
class SourceSnapshot:
    url: str
    fingerprint: str
    etag: str | None
    last_modified: str | None
    status_code: int


class OfficialSourceFetcher:
    """Fetch a bounded public source document without forwarding credentials."""

    def __init__(self, settings: Settings, *, session: requests.Session | None = None) -> None:
        self.settings = settings
        self.session = session or requests.Session()

    @staticmethod
    def validate_url(url: str) -> None:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("official source URL must be public HTTP(S)")
        hostname = parsed.hostname.rstrip(".").lower()
        if (
            hostname == "localhost"
            or hostname.endswith(".localhost")
            or hostname.endswith(".local")
        ):
            raise ValueError("local official source URLs are not allowed")
        try:
            address = ipaddress.ip_address(hostname)
        except ValueError:
            return
        if not address.is_global:
            raise ValueError("non-public official source IP addresses are not allowed")

    def fetch(self, url: str) -> SourceSnapshot:
        current = url
        for _ in range(4):
            self.validate_url(current)
            response = self.session.get(
                current,
                headers={
                    "Accept": "text/html,application/json,application/pdf,text/plain,*/*;q=0.2",
                    "User-Agent": "prophet-directional-trader/0.1 official-source-monitor",
                },
                timeout=self.settings.official_source_timeout_seconds,
                allow_redirects=False,
                stream=True,
            )
            if response.status_code in {301, 302, 303, 307, 308}:
                location = response.headers.get("Location")
                response.close()
                if not location:
                    raise RuntimeError("official source redirect omitted Location")
                current = urljoin(current, location)
                continue
            if not 200 <= response.status_code < 300:
                status = response.status_code
                response.close()
                raise RuntimeError(f"official source returned HTTP {status}")

            content_hash = hashlib.sha256()
            seen = 0
            for chunk in response.iter_content(chunk_size=65536):
                if not chunk:
                    continue
                remaining = self.settings.official_source_max_bytes - seen
                if remaining <= 0:
                    content_hash.update(b"\0TRUNCATED")
                    break
                content_hash.update(chunk[:remaining])
                seen += min(len(chunk), remaining)
                if len(chunk) > remaining:
                    content_hash.update(b"\0TRUNCATED")
                    break

            etag = response.headers.get("ETag")
            last_modified = response.headers.get("Last-Modified")
            status_code = response.status_code
            response.close()
            material = json.dumps(
                {
                    "body_sha256": content_hash.hexdigest(),
                    "etag": etag,
                    "last_modified": last_modified,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            return SourceSnapshot(
                url=current,
                fingerprint=hashlib.sha256(material).hexdigest(),
                etag=etag,
                last_modified=last_modified,
                status_code=status_code,
            )
        raise RuntimeError("official source exceeded redirect limit")


def market_metadata_payload(market: Market) -> dict[str, Any]:
    """Material non-price fields whose change can alter interpretation or lifecycle."""
    return {
        "ticker": market.ticker,
        "event_ticker": market.event_ticker,
        "series_ticker": market.series_ticker,
        "category": market.category,
        "title": market.title,
        "subtitle": market.subtitle,
        "yes_sub_title": market.yes_sub_title,
        "no_sub_title": market.no_sub_title,
        "rules_primary": market.rules_primary,
        "rules_secondary": market.rules_secondary,
        "close_time": market.close_time.isoformat(),
        "expected_expiration_time": market.expected_expiration_time.isoformat(),
        "can_close_early": market.can_close_early,
        "early_close_condition": market.early_close_condition,
        "status": market.status,
        "price_ranges": list(market.price_ranges),
    }


def metadata_fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _decimal_or_none(value: Any) -> Decimal | None:
    if value in {None, ""}:
        return None
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return None
    return parsed if parsed.is_finite() else None


def _valid_midpoint(market: Market) -> Decimal | None:
    if market.status.lower() not in {"open", "active", ""}:
        return None
    if not (
        market.yes_bid.is_finite()
        and market.yes_ask.is_finite()
        and market.yes_bid_size.is_finite()
        and market.yes_ask_size.is_finite()
        and ZERO < market.yes_bid <= market.yes_ask < Decimal("1")
        and market.yes_bid_size > ZERO
        and market.yes_ask_size > ZERO
    ):
        return None
    midpoint = market.midpoint
    return midpoint if midpoint.is_finite() and ZERO < midpoint < Decimal("1") else None


class EventWatcher:
    """Continuously refresh context without ever invoking the forecasting model."""

    def __init__(
        self,
        settings: Settings,
        *,
        client: Any,
        store: StateStore,
        broker: EventBroker,
        source_fetcher: OfficialSourceFetcher | None = None,
    ) -> None:
        self.settings = settings
        self.client = client
        self.store = store
        self.broker = broker
        self.source_fetcher = source_fetcher or OfficialSourceFetcher(settings)

    def _collect_markets(
        self,
        *,
        tickers: Sequence[str] | None,
        now: datetime,
    ) -> dict[str, Market]:
        discovered = discover_markets(self.client, tickers=tickers)
        explicit = {item.strip() for item in (tickers or ()) if item.strip()}
        portfolio = self.broker.snapshot()
        held = set(portfolio.positions)

        if explicit:
            selected = {market.ticker: market for market in discovered if market.ticker}
        else:
            eligible = [
                market
                for market in discovered
                if market_eligibility(market, self.settings, now=now).eligible
            ]
            eligible.sort(
                key=lambda market: (
                    0 if market.ticker in held else 1,
                    -market.executable_depth_dollars,
                    -market.volume,
                    market.close_time,
                    market.ticker,
                )
            )
            selected = {
                market.ticker: market
                for market in eligible[: self.settings.max_markets_per_cycle]
            }

        tracked = self.store.event_watch_tickers() | held
        if explicit:
            tracked &= explicit
        for ticker in sorted(tracked):
            if ticker in selected:
                continue
            try:
                selected[ticker] = get_market(self.client, ticker)
            except Exception as exc:
                LOGGER.warning("unable to refresh event-watched market %s: %s", ticker, exc)
        return selected

    def _source_poll_due(
        self,
        state: dict[str, Any] | None,
        now: datetime,
    ) -> bool:
        if not state or not state.get("source_checked_at"):
            return True
        checked = parse_timestamp(state["source_checked_at"])
        if checked is None:
            return True
        return (now - checked).total_seconds() >= self.settings.official_source_poll_seconds

    @staticmethod
    def _is_official_source(label: str, url: str) -> bool:
        parsed = urlparse(url)
        hostname = (parsed.hostname or "").rstrip(".").lower()
        official_suffixes = (
            ".gov",
            ".gov.uk",
            ".gc.ca",
            ".gouv.fr",
            ".europa.eu",
            ".mil",
            ".int",
        )
        if any(hostname == suffix[1:] or hostname.endswith(suffix) for suffix in official_suffixes):
            return True
        normalized_label = label.strip().lower()
        official_markers = (
            "official",
            "government",
            "ministry",
            "department",
            "commission",
            "regulator",
            "filing",
            "investor relations",
            "press release",
        )
        return any(marker in normalized_label for marker in official_markers)

    def _official_source_urls(self, market: Market) -> list[str]:
        sources = list(market.settlement_sources)
        if market.event_ticker:
            try:
                metadata = self.client.get_event_metadata(market.event_ticker)
            except Exception as exc:
                LOGGER.warning(
                    "unable to load official settlement sources for %s: %s",
                    market.ticker,
                    exc,
                )
            else:
                raw_sources = metadata.get("settlement_sources", [])
                if isinstance(raw_sources, list):
                    sources = [item for item in raw_sources if isinstance(item, dict)]

        for evidence in self.store.latest_forecast_evidence(market.ticker):
            sources.append(
                {
                    "name": str(evidence.get("claim", "")),
                    "url": str(evidence.get("source_url", "")),
                }
            )

        urls: list[str] = []
        for source in sources:
            url = str(source.get("url", "")).strip()
            label = str(source.get("name", ""))
            if url and self._is_official_source(label, url) and url not in urls:
                urls.append(url)
        return urls

    def _check_sources(
        self,
        market: Market,
        *,
        checked_at: str,
        now: datetime,
        fetch_cache: dict[str, SourceSnapshot | Exception],
        warnings: list[str],
    ) -> list[str]:
        changed: list[str] = []
        for url in self._official_source_urls(market):
            existing = self.store.official_source_state(market.ticker, url)
            if existing and existing.get("error"):
                previous_check = parse_timestamp(existing.get("checked_at"))
                if (
                    previous_check is not None
                    and (now - previous_check).total_seconds()
                    < self.settings.official_source_error_backoff_seconds
                ):
                    continue
            if url not in fetch_cache:
                try:
                    fetch_cache[url] = self.source_fetcher.fetch(url)
                except Exception as exc:
                    fetch_cache[url] = exc
            result = fetch_cache[url]
            if isinstance(result, Exception):
                message = f"{type(result).__name__}: {result}"
                self.store.upsert_official_source_state(
                    market.ticker,
                    url,
                    fingerprint=None,
                    etag=None,
                    last_modified=None,
                    status_code=None,
                    error=message,
                    checked_at=checked_at,
                )
                warnings.append(f"{market.ticker} source {url}: {message}")
                continue
            if existing and existing.get("fingerprint") not in {None, result.fingerprint}:
                changed.append(url)
            self.store.upsert_official_source_state(
                market.ticker,
                url,
                fingerprint=result.fingerprint,
                etag=result.etag,
                last_modified=result.last_modified,
                status_code=result.status_code,
                error=None,
                checked_at=checked_at,
            )
        return changed

    def poll_once(
        self,
        *,
        tickers: Sequence[str] | None = None,
        confirm_live: bool = False,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        now_text = now.isoformat()
        summary: dict[str, Any] = {
            "mode": self.settings.trading_mode,
            "observed_at": now_text,
            "monitored": 0,
            "baselined": 0,
            "detected": 0,
            "forecasting_invocations": 0,
            "context_updates": [],
            "errors": [],
            "source_warnings": [],
        }
        markets = self._collect_markets(tickers=tickers, now=now)
        summary["monitored"] = len(markets)
        summary["legacy_pending_cancelled"] = self.store.cancel_pending_event_triggers()
        fetch_cache: dict[str, SourceSnapshot | Exception] = {}

        for market in markets.values():
            state = self.store.event_watch_state(market.ticker)
            midpoint = _valid_midpoint(market)
            old_reference = _decimal_or_none(state.get("price_reference")) if state else None
            new_reference = old_reference if old_reference is not None else midpoint
            metadata = market_metadata_payload(market)
            fingerprint = metadata_fingerprint(metadata)
            trigger_types: list[str] = []
            payload: dict[str, Any] = {}

            if old_reference is not None and midpoint is not None:
                movement = abs(midpoint - old_reference)
                if movement >= self.settings.event_price_move_threshold:
                    trigger_types.append("price_movement_3c")
                    payload["price_movement"] = {
                        "from": str(old_reference),
                        "to": str(midpoint),
                        "absolute_move": str(movement),
                    }
                    new_reference = midpoint

            if state and state.get("metadata_fingerprint") != fingerprint:
                trigger_types.append("market_metadata_lifecycle_change")
                try:
                    previous_metadata = json.loads(state.get("metadata_json") or "{}")
                except json.JSONDecodeError:
                    previous_metadata = {}
                changed_fields = sorted(
                    key
                    for key in set(previous_metadata) | set(metadata)
                    if previous_metadata.get(key) != metadata.get(key)
                )
                payload["metadata_changed_fields"] = changed_fields

            source_checked_at: str | None = None
            if self._source_poll_due(state, now):
                source_checked_at = now_text
                changed_urls = self._check_sources(
                    market,
                    checked_at=now_text,
                    now=now,
                    fetch_cache=fetch_cache,
                    warnings=summary["source_warnings"],
                )
                if changed_urls:
                    trigger_types.append("official_source_or_scheduled_release_update")
                    payload["official_source_urls"] = changed_urls

            self.store.upsert_event_watch_state(
                market.ticker,
                price_reference=new_reference,
                metadata_fingerprint=fingerprint,
                metadata=metadata,
                source_checked_at=source_checked_at,
            )

            if state is None:
                summary["baselined"] += 1
                continue
            if not trigger_types:
                continue
            summary["detected"] += 1
            event_id = self.store.record_context_event(
                market.ticker,
                trigger_types,
                payload,
                detected_at=now_text,
            )
            summary["context_updates"].append(
                {"id": event_id, "ticker": market.ticker, "types": trigger_types}
            )
        summary["pending"] = 0
        return summary
