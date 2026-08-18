from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Protocol, Sequence

from prophet_trader.config import Settings
from prophet_trader.execution import PortfolioSnapshot
from prophet_trader.forecast import (
    ForecastQuotaExceeded,
    ForecastRequestTimeout,
    PreparedForecast,
    ProviderForecastResponse,
)
from prophet_trader.kalshi import KalshiClient
from prophet_trader.models import (
    Direction,
    Forecast,
    Market,
    ONE,
    OrderIntent,
    Position,
    Signal,
    ZERO,
)
from prophet_trader.risk import RiskManager
from prophet_trader.store import StateStore
from prophet_trader.strategy import (
    brier_signal,
    build_order_intent,
    forecast_eligibility,
    market_eligibility,
    should_reforecast,
)


LOGGER = logging.getLogger(__name__)


class Broker(Protocol):
    def reconcile_and_cancel(self) -> None: ...

    def blocked_tickers(self) -> set[str]: ...

    def snapshot(self) -> PortfolioSnapshot: ...

    def execute(
        self,
        *,
        cycle_id: str,
        market: Market,
        intent: OrderIntent,
    ) -> dict[str, Any]: ...


class Forecaster(Protocol):
    def prepare(
        self,
        market: Market,
        *,
        slot_iso: str,
        context: dict[str, Any],
    ) -> PreparedForecast: ...

    def request(self, prepared: PreparedForecast) -> ProviderForecastResponse: ...

    def parse(self, response: ProviderForecastResponse) -> Forecast: ...


@dataclass
class WorkingPortfolio:
    positions: dict[str, Position]
    available_cash_dollars: Decimal
    total_exposure_dollars: Decimal

    @classmethod
    def from_snapshot(cls, snapshot: PortfolioSnapshot) -> "WorkingPortfolio":
        return cls(
            positions=dict(snapshot.positions),
            available_cash_dollars=snapshot.available_cash_dollars,
            total_exposure_dollars=snapshot.total_exposure_dollars,
        )


def _event_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    nested = payload.get("event")
    event = dict(nested) if isinstance(nested, dict) else dict(payload)
    if "markets" not in event and isinstance(payload.get("markets"), list):
        event["markets"] = payload["markets"]
    return event


def get_market(
    client: KalshiClient,
    ticker: str,
    *,
    event_cache: dict[str, dict[str, Any]] | None = None,
) -> Market:
    """Load a market plus its event metadata, which contains category/source context."""
    payload = client.get_market(ticker)
    nested = payload.get("market")
    market_data = nested if isinstance(nested, dict) else payload
    event_ticker = str(market_data.get("event_ticker", ""))
    event: dict[str, Any] = {}
    if event_ticker:
        cache = event_cache if event_cache is not None else {}
        if event_ticker not in cache:
            cache[event_ticker] = _event_from_payload(client.get_event(event_ticker))
        event = cache[event_ticker]
    return Market.from_api(market_data, event)


def discover_markets(
    client: KalshiClient,
    *,
    tickers: Sequence[str] | None = None,
    max_pages: int = 5,
    event_cache: dict[str, dict[str, Any]] | None = None,
) -> list[Market]:
    """Discover nested open markets, or resolve an explicit ticker allowlist."""
    cache = event_cache if event_cache is not None else {}
    if tickers:
        markets: list[Market] = []
        for ticker in dict.fromkeys(item.strip() for item in tickers if item.strip()):
            markets.append(get_market(client, ticker, event_cache=cache))
        return markets

    markets = []
    seen: set[str] = set()
    for raw_event in client.list_open_events(max_pages=max_pages):
        event = _event_from_payload(raw_event)
        event_ticker = str(event.get("event_ticker", ""))
        if event_ticker:
            cache[event_ticker] = event
        raw_markets = event.get("markets", [])
        if not isinstance(raw_markets, list):
            continue
        for raw_market in raw_markets:
            if not isinstance(raw_market, dict):
                continue
            try:
                market = Market.from_api(raw_market, event)
            except ValueError as exc:
                LOGGER.warning("skipping malformed Kalshi market: %s", exc)
                continue
            if market.ticker and market.ticker not in seen:
                seen.add(market.ticker)
                markets.append(market)
    return markets


def _hold_signal(market: Market, position: Position, reason: str) -> Signal:
    return Signal(
        direction=Direction.FLAT,
        probability_yes=market.midpoint,
        actionable_edge=ZERO,
        raw_target_contracts=position.contracts,
        target_position=position.contracts,
        execution_yes_price=None,
        outcome_cost=None,
        reason=reason,
    )


def _flat_signal(market: Market, reason: str) -> Signal:
    return Signal(
        direction=Direction.FLAT,
        probability_yes=market.midpoint,
        actionable_edge=ZERO,
        raw_target_contracts=ZERO,
        target_position=ZERO,
        execution_yes_price=None,
        outcome_cost=None,
        reason=reason,
    )


def _market_is_executable(market: Market, *, delta: Decimal) -> bool:
    """Return whether the book can execute the required rebalance direction.

    Entries are pre-screened on two-sided depth, but a held position must still
    be reducible when only the relevant exit side remains quoted.
    """
    if market.status.lower() not in {"open", "active", ""} or not delta.is_finite():
        return False
    if delta > ZERO:
        return (
            market.yes_ask.is_finite()
            and market.yes_ask_size.is_finite()
            and ZERO < market.yes_ask < ONE
            and market.yes_ask_size > ZERO
        )
    if delta < ZERO:
        return (
            market.yes_bid.is_finite()
            and market.yes_bid_size.is_finite()
            and ZERO < market.yes_bid < ONE
            and market.yes_bid_size > ZERO
        )
    return True


def two_hour_slot(now: datetime) -> str:
    epoch = int(now.astimezone(timezone.utc).timestamp())
    slot_epoch = epoch - (epoch % 7200)
    return datetime.fromtimestamp(slot_epoch, timezone.utc).isoformat()


class TraderEngine:
    """One fill-aware, bid/ask-aware Brier rebalancing cycle."""

    def __init__(
        self,
        settings: Settings,
        *,
        client: KalshiClient,
        forecaster: Forecaster,
        store: StateStore,
        broker: Broker,
        risk: RiskManager,
    ) -> None:
        self.settings = settings
        self.client = client
        self.forecaster = forecaster
        self.store = store
        self.broker = broker
        self.risk = risk

    @staticmethod
    def _summary(mode: str, cycle_id: str) -> dict[str, Any]:
        return {
            "cycle_id": cycle_id,
            "mode": mode,
            "discovered": 0,
            "eligible": 0,
            "forecasted": 0,
            "forecast_requests_queued": 0,
            "signals": 0,
            "orders_submitted": 0,
            "skipped": 0,
            "orders": [],
            "errors": [],
        }

    def _record_blocked(
        self,
        *,
        cycle_id: str,
        market: Market,
        position: Position,
        signal: Signal,
        reason: str,
    ) -> None:
        self.store.record_decision(
            cycle_id,
            market,
            signal,
            position,
            None,
            allowed=False,
            reason=reason,
        )

    def _rebalance(
        self,
        *,
        cycle_id: str,
        market: Market,
        signal: Signal,
        portfolio: WorkingPortfolio,
        summary: dict[str, Any],
    ) -> None:
        position = portfolio.positions.get(market.ticker, Position(market.ticker, ZERO))
        delta = signal.target_position - position.contracts
        if not _market_is_executable(market, delta=delta):
            reason = "cannot rebalance because the market or current top-of-book is not executable"
            self._record_blocked(
                cycle_id=cycle_id,
                market=market,
                position=position,
                signal=signal,
                reason=reason,
            )
            summary["skipped"] += 1
            return

        intent = build_order_intent(
            market,
            signal,
            position,
            self.settings,
            total_exposure_dollars=portfolio.total_exposure_dollars,
            available_cash_dollars=portfolio.available_cash_dollars,
        )
        if intent is None:
            at_target = signal.target_position == position.contracts
            reason = (
                "target already satisfied"
                if at_target
                else "rebalance is below one contract or blocked by available risk capacity"
            )
            self.store.record_decision(
                cycle_id,
                market,
                signal,
                position,
                None,
                allowed=at_target,
                reason=reason,
            )
            summary["skipped"] += 1
            return

        risk_decision = self.risk.check_order(
            intent,
            total_exposure_dollars=portfolio.total_exposure_dollars,
            available_cash_dollars=portfolio.available_cash_dollars,
        )
        self.store.record_decision(
            cycle_id,
            market,
            signal,
            position,
            intent,
            allowed=risk_decision.allowed,
            reason=risk_decision.reason if not risk_decision.allowed else intent.reason,
        )
        if not risk_decision.allowed:
            summary["skipped"] += 1
            return

        result = self.broker.execute(cycle_id=cycle_id, market=market, intent=intent)
        summary["orders_submitted"] += 1
        summary["orders"].append(
            {
                "ticker": market.ticker,
                "book_side": intent.book_side,
                "count": str(intent.count),
                "yes_price": str(intent.yes_price),
                "reduce_only": intent.reduce_only,
                "order_id": str(result.get("order_id", "")),
                "client_order_id": str(result.get("client_order_id", "")),
                "simulated": bool(result.get("simulated", False)),
            }
        )

        if self.settings.trading_mode == "paper":
            refreshed = self.broker.snapshot()
            portfolio.positions = dict(refreshed.positions)
            portfolio.available_cash_dollars = refreshed.available_cash_dollars
            portfolio.total_exposure_dollars = refreshed.total_exposure_dollars
        else:
            # Reserve the maximum opening loss even if the GTC order has not filled yet.
            portfolio.total_exposure_dollars += intent.opening_risk_dollars
            portfolio.available_cash_dollars = max(
                ZERO, portfolio.available_cash_dollars - intent.opening_risk_dollars
            )

    def run_cycle(
        self,
        *,
        tickers: Sequence[str] | None = None,
        confirm_live: bool = False,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        cycle_id = str(uuid.uuid4())
        summary = self._summary(self.settings.trading_mode, cycle_id)
        summary["two_hour_slot"] = two_hour_slot(now)
        self.store.start_cycle(cycle_id, self.settings.trading_mode)

        try:
            if not self.settings.forecasting_enabled:
                summary["skipped"] += 1
                summary["errors"].append("forecasting is disabled")
                self.store.finish_cycle(cycle_id, "skipped", summary)
                return summary
            circuit = self.store.forecast_circuit()
            if circuit.get("state") == "open":
                summary["skipped"] += 1
                summary["errors"].append(
                    f"forecast circuit open: {circuit.get('reason') or 'unknown reason'}"
                )
                self.store.finish_cycle(cycle_id, "skipped", summary)
                return summary
            if self.settings.trading_mode == "live":
                self.risk.assert_live_enabled(confirm_live=confirm_live)
                status = self.client.exchange_status()
                trading_active = status.get("trading_active")
                if trading_active not in {True, "true", "True", 1}:
                    summary["skipped"] += 1
                    summary["errors"].append("Kalshi reports trading_active=false")
                    self.store.finish_cycle(cycle_id, "skipped", summary)
                    return summary

            # Resting orders are stale at the next two-hour decision point. Reconcile actual
            # fills first, then cancel their remainder before calculating target deltas.
            self.broker.reconcile_and_cancel()
            blocked_tickers = self.broker.blocked_tickers()

            kill = self.risk.check_kill_switch()
            if not kill.allowed:
                summary["skipped"] += 1
                summary["errors"].append(kill.reason)
                self.store.finish_cycle(cycle_id, "skipped", summary)
                return summary

            portfolio = WorkingPortfolio.from_snapshot(self.broker.snapshot())
            explicit = {item.strip() for item in (tickers or ()) if item.strip()}
            scoped_positions = {
                ticker: position
                for ticker, position in portfolio.positions.items()
                if not explicit or ticker in explicit
            }
            event_cache: dict[str, dict[str, Any]] = {}
            tracked_markets: dict[str, Market] = {}
            forced: set[str] = set()

            # Existing inventory is checked even if broad discovery no longer returns it.
            for ticker, position in scoped_positions.items():
                try:
                    market = get_market(self.client, ticker, event_cache=event_cache)
                except Exception as exc:
                    summary["errors"].append(
                        f"{ticker}: unable to refresh held market ({type(exc).__name__}: {exc})"
                    )
                    summary["skipped"] += 1
                    continue
                tracked_markets[ticker] = market
                eligibility = market_eligibility(
                    market,
                    self.settings,
                    now=now,
                    enforce_discovery_horizon=False,
                )
                if not eligibility.eligible:
                    signal = _flat_signal(
                        market, f"forced unwind of held inventory: {eligibility.reason}"
                    )
                    if ticker in blocked_tickers:
                        self._record_blocked(
                            cycle_id=cycle_id,
                            market=market,
                            position=position,
                            signal=signal,
                            reason="unresolved prior order blocks another submission",
                        )
                        summary["skipped"] += 1
                    else:
                        self._rebalance(
                            cycle_id=cycle_id,
                            market=market,
                            signal=signal,
                            portfolio=portfolio,
                            summary=summary,
                        )
                    forced.add(ticker)

            discovered = discover_markets(
                self.client,
                tickers=tickers,
                event_cache=event_cache,
            )
            summary["discovered"] = len(discovered)
            candidates: dict[str, Market] = {
                market.ticker: market for market in discovered if market.ticker
            }
            for ticker, market in tracked_markets.items():
                if ticker not in forced:
                    candidates.setdefault(ticker, market)

            eligible: list[Market] = []
            for market in candidates.values():
                if market.ticker in forced:
                    continue
                eligibility = market_eligibility(
                    market,
                    self.settings,
                    now=now,
                    enforce_discovery_horizon=market.ticker not in scoped_positions,
                )
                if not eligibility.eligible:
                    summary["skipped"] += 1
                    continue
                if market.ticker in blocked_tickers:
                    position = portfolio.positions.get(
                        market.ticker, Position(market.ticker, ZERO)
                    )
                    self._record_blocked(
                        cycle_id=cycle_id,
                        market=market,
                        position=position,
                        signal=_hold_signal(
                            market, position, "unresolved prior order; preserve actual inventory"
                        ),
                        reason="unresolved prior order blocks another submission",
                    )
                    summary["skipped"] += 1
                    continue
                last_fill_price, last_fill_side = self.store.get_last_fill_context(market.ticker)
                if not should_reforecast(
                    market,
                    last_fill_price,
                    self.settings,
                    last_fill_book_side=last_fill_side,
                ):
                    summary["skipped"] += 1
                    continue
                eligible.append(market)

            eligible.sort(
                key=lambda market: (
                    0 if market.ticker in scoped_positions else 1,
                    -market.executable_depth_dollars,
                    -market.volume,
                    market.close_time,
                    market.ticker,
                )
            )
            eligible = eligible[
                : min(self.settings.max_markets_per_cycle, self.settings.max_forecasts_per_cycle)
            ]
            summary["eligible"] = len(eligible)

            queued: list[tuple[Market, PreparedForecast, int]] = []
            slot_iso = str(summary["two_hour_slot"])
            for market in eligible:
                position = portfolio.positions.get(
                    market.ticker, Position(market.ticker, ZERO)
                )
                prepared = self.forecaster.prepare(
                    market,
                    slot_iso=slot_iso,
                    context=self.store.forecast_context_snapshot(market.ticker),
                )
                reservation = self.store.queue_forecast_request(
                    client_request_id=str(uuid.uuid4()),
                    cycle_id=cycle_id,
                    ticker=market.ticker,
                    two_hour_slot=slot_iso,
                    model=self.settings.gemini_model,
                    prompt_version=prepared.prompt_version,
                    context_hash=prepared.context_hash,
                    reserved_cost_dollars=self.settings.forecast_reserve_cost_dollars,
                    daily_limit_dollars=self.settings.daily_forecast_spend_limit,
                    now=now,
                )
                if not reservation.get("queued"):
                    reason = str(reservation.get("reason") or "forecast request not queued")
                    self._record_blocked(
                        cycle_id=cycle_id,
                        market=market,
                        position=position,
                        signal=_hold_signal(market, position, reason),
                        reason=reason,
                    )
                    summary["skipped"] += 1
                    if "daily forecast-spend limit" in reason:
                        summary["errors"].append(reason)
                    continue
                queued.append((market, prepared, int(reservation["request_id"])))
            summary["forecast_requests_queued"] = len(queued)

            for index, (market, prepared, request_id) in enumerate(queued):
                position = portfolio.positions.get(
                    market.ticker, Position(market.ticker, ZERO)
                )
                if not self.store.start_forecast_request(request_id):
                    summary["skipped"] += 1
                    continue
                try:
                    provider_response = self.forecaster.request(prepared)
                except ForecastRequestTimeout as exc:
                    reason = f"forecast status UNKNOWN after timeout: {exc}"
                    self.store.finish_forecast_request(
                        request_id,
                        status="unknown",
                        error=reason,
                        provider_request_id=exc.provider_request_id,
                    )
                    self.store.cancel_queued_forecasts(
                        cycle_id=cycle_id,
                        reason="cancelled after timed-out UNKNOWN request",
                    )
                    self._record_blocked(
                        cycle_id=cycle_id,
                        market=market,
                        position=position,
                        signal=_hold_signal(market, position, reason),
                        reason=reason,
                    )
                    summary["errors"].append(f"{market.ticker}: {reason}")
                    summary["skipped"] += len(queued) - index
                    break
                except ForecastQuotaExceeded as exc:
                    reason = f"forecast quota exhausted: {exc}"
                    self.store.finish_forecast_request(
                        request_id,
                        status="failed",
                        error=reason,
                        provider_request_id=exc.provider_request_id,
                    )
                    cancelled = self.store.open_forecast_circuit(reason)
                    self._record_blocked(
                        cycle_id=cycle_id,
                        market=market,
                        position=position,
                        signal=_hold_signal(market, position, reason),
                        reason=reason,
                    )
                    summary["errors"].append(f"{market.ticker}: {reason}")
                    summary["circuit_opened"] = True
                    summary["queued_cancelled"] = cancelled
                    summary["skipped"] += len(queued) - index
                    break
                except Exception as exc:
                    reason = f"forecast failed ({type(exc).__name__}: {exc})"
                    self.store.finish_forecast_request(
                        request_id,
                        status="failed",
                        error=reason,
                        provider_request_id=getattr(exc, "provider_request_id", None),
                    )
                    self._record_blocked(
                        cycle_id=cycle_id,
                        market=market,
                        position=position,
                        signal=_hold_signal(market, position, reason),
                        reason=reason,
                    )
                    summary["errors"].append(f"{market.ticker}: {reason}")
                    summary["skipped"] += 1
                    continue

                self.store.record_provider_response(
                    request_id,
                    provider_request_id=provider_response.provider_request_id,
                    raw_response=provider_response.raw_response,
                    input_tokens=provider_response.input_tokens,
                    cached_tokens=provider_response.cached_tokens,
                    reasoning_tokens=provider_response.reasoning_tokens,
                    output_tokens=provider_response.output_tokens,
                    total_tokens=provider_response.total_tokens,
                    search_queries=provider_response.search_queries,
                    duration_ms=provider_response.duration_ms,
                    estimated_cost_dollars=provider_response.estimated_cost_dollars,
                )
                daily_spend = self.store.daily_forecast_spend(now=now)
                summary["daily_forecast_spend_dollars"] = str(daily_spend)
                if daily_spend > self.settings.daily_forecast_spend_limit:
                    reason = "daily forecast-spend limit exceeded by provider usage"
                    self.store.finish_forecast_request(
                        request_id, status="parse_failed", error=reason
                    )
                    cancelled = self.store.open_forecast_circuit(reason)
                    self._record_blocked(
                        cycle_id=cycle_id,
                        market=market,
                        position=position,
                        signal=_hold_signal(market, position, reason),
                        reason=reason,
                    )
                    summary["errors"].append(f"{market.ticker}: {reason}")
                    summary["circuit_opened"] = True
                    summary["queued_cancelled"] = cancelled
                    summary["skipped"] += len(queued) - index
                    break
                try:
                    forecast = self.forecaster.parse(provider_response)
                except Exception as exc:
                    reason = f"forecast parse failed ({type(exc).__name__}: {exc})"
                    self.store.finish_forecast_request(
                        request_id, status="parse_failed", error=reason
                    )
                    self._record_blocked(
                        cycle_id=cycle_id,
                        market=market,
                        position=position,
                        signal=_hold_signal(market, position, reason),
                        reason=reason,
                    )
                    summary["errors"].append(f"{market.ticker}: {reason}")
                    summary["skipped"] += 1
                    continue

                self.store.record_forecast(
                    cycle_id,
                    market,
                    forecast,
                    request_id=request_id,
                )
                self.store.finish_forecast_request(request_id, status="completed")
                summary["forecasted"] += 1
                # Search/reasoning can take long enough for the quote to move. Refresh before
                # sizing; the crossing limit still protects against a later adverse move.
                try:
                    market = get_market(
                        self.client,
                        market.ticker,
                        event_cache=event_cache,
                    )
                except Exception as exc:
                    reason = f"post-forecast market refresh failed ({type(exc).__name__}: {exc})"
                    self._record_blocked(
                        cycle_id=cycle_id,
                        market=market,
                        position=position,
                        signal=_hold_signal(market, position, reason),
                        reason=reason,
                    )
                    summary["errors"].append(f"{market.ticker}: {reason}")
                    summary["skipped"] += 1
                    continue

                refreshed_market_gate = market_eligibility(
                    market,
                    self.settings,
                    now=now,
                    enforce_discovery_horizon=market.ticker not in scoped_positions,
                )
                if not refreshed_market_gate.eligible:
                    reason = f"post-forecast market guardrail: {refreshed_market_gate.reason}"
                    if position.contracts != ZERO:
                        self._rebalance(
                            cycle_id=cycle_id,
                            market=market,
                            signal=_flat_signal(market, f"forced unwind: {reason}"),
                            portfolio=portfolio,
                            summary=summary,
                        )
                    else:
                        self._record_blocked(
                            cycle_id=cycle_id,
                            market=market,
                            position=position,
                            signal=_hold_signal(market, position, reason),
                            reason=reason,
                        )
                        summary["skipped"] += 1
                    continue

                forecast_gate = forecast_eligibility(forecast, self.settings)
                if not forecast_gate.eligible:
                    if position.contracts != ZERO:
                        signal = _flat_signal(
                            market, f"forecast guardrail requires unwind: {forecast_gate.reason}"
                        )
                        self._rebalance(
                            cycle_id=cycle_id,
                            market=market,
                            signal=signal,
                            portfolio=portfolio,
                            summary=summary,
                        )
                    else:
                        self._record_blocked(
                            cycle_id=cycle_id,
                            market=market,
                            position=position,
                            signal=_hold_signal(market, position, forecast_gate.reason),
                            reason=forecast_gate.reason,
                        )
                        summary["skipped"] += 1
                    continue

                signal = brier_signal(market, forecast, self.settings)
                if signal.direction is not Direction.FLAT:
                    summary["signals"] += 1
                self._rebalance(
                    cycle_id=cycle_id,
                    market=market,
                    signal=signal,
                    portfolio=portfolio,
                    summary=summary,
                )

            self.store.finish_cycle(cycle_id, "completed", summary)
            return summary
        except Exception as exc:
            summary["errors"].append(f"{type(exc).__name__}: {exc}")
            self.store.finish_cycle(cycle_id, "failed", summary)
            raise
