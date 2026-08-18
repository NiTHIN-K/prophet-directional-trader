from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN, ROUND_HALF_UP

from prophet_trader.config import Settings
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


HOURS_PER_DAY = Decimal("24")
SECONDS_PER_HOUR = Decimal("3600")


@dataclass(frozen=True)
class Eligibility:
    eligible: bool
    reason: str


def market_eligibility(
    market: Market,
    settings: Settings,
    *,
    now: datetime | None = None,
    enforce_discovery_horizon: bool = True,
) -> Eligibility:
    """Apply the paper's ex-ante eligibility rules using API-visible fields."""
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    status = market.status.lower()
    if status not in {"open", "active", ""}:
        return Eligibility(False, f"market status is {market.status!r}")
    if (
        market.category.strip().lower() == "mentions"
        or "MENTION" in market.series_ticker.upper()
    ):
        return Eligibility(False, "MENTIONS category is excluded by the paper")

    until_close_hours = Decimal(str((market.close_time - now).total_seconds())) / SECONDS_PER_HOUR
    until_resolution_hours = (
        Decimal(str((market.expected_expiration_time - now).total_seconds())) / SECONDS_PER_HOUR
    )
    if until_resolution_hours <= settings.stop_hours_before_resolution:
        return Eligibility(False, "inside the paper's pre-resolution trading cutoff")

    if enforce_discovery_horizon:
        min_hours = settings.min_days_to_close * HOURS_PER_DAY
        max_hours = settings.max_days_to_close * HOURS_PER_DAY
        if until_close_hours < min_hours or until_close_hours > max_hours:
            return Eligibility(False, "close time is outside the configured 2-14 day horizon")

    close_gap_hours = Decimal(
        str((market.close_time - market.expected_expiration_time).total_seconds())
    ) / SECONDS_PER_HOUR
    if close_gap_hours > settings.max_close_resolution_gap_hours:
        return Eligibility(False, "close time trails expected resolution by more than one hour")
    if not market.rules_primary.strip():
        return Eligibility(False, "missing primary resolution rules")
    if market.can_close_early and not market.early_close_condition.strip():
        return Eligibility(False, "early close is possible but its condition is unspecified")
    if not market.yes_bid.is_finite() or not market.yes_ask.is_finite():
        return Eligibility(False, "YES top-of-book is incomplete or invalid")
    if not (ZERO < market.yes_bid <= market.yes_ask < ONE):
        return Eligibility(False, "YES top-of-book is incomplete or invalid")
    if market.spread > settings.max_spread:
        return Eligibility(False, "spread is above the configured guardrail")
    if (
        not market.yes_bid_size.is_finite()
        or not market.yes_ask_size.is_finite()
        or market.yes_bid_size <= ZERO
        or market.yes_ask_size <= ZERO
    ):
        return Eligibility(False, "displayed top-of-book size is incomplete or invalid")
    if market.executable_depth_dollars < settings.min_liquidity_dollars:
        return Eligibility(
            False,
            "two-sided executable top-of-book depth is below the configured guardrail",
        )
    return Eligibility(True, "eligible")


def forecast_eligibility(forecast: Forecast, settings: Settings) -> Eligibility:
    if not forecast.resolution_clear:
        return Eligibility(
            False,
            f"model rejected resolution clarity: {forecast.resolution_clarity_reason}",
        )
    unique_sources = {
        evidence.source_url.strip()
        for evidence in forecast.evidence
        if evidence.source_url.strip().startswith(("http://", "https://"))
    }
    if len(unique_sources) < settings.min_evidence_sources:
        return Eligibility(False, "forecast lacks the configured number of distinct web sources")
    if settings.block_high_uncertainty and forecast.uncertainty == "high":
        return Eligibility(False, "high-uncertainty forecasts are blocked")
    return Eligibility(True, "eligible")


def _whole_contracts(value: Decimal) -> Decimal:
    return value.quantize(Decimal("1"), rounding=ROUND_HALF_UP)


def brier_signal(market: Market, forecast: Forecast, settings: Settings) -> Signal:
    """Return the bid/ask-aware Brier target used by the paper's momentum strategy.

    A global coefficient in the Brier gradient is economically arbitrary. The paper's
    case studies imply roughly 100 contracts per unit probability edge, so the scale is
    explicit and configurable rather than presented as a published constant.
    """
    p = forecast.probability_yes
    direction = Direction.FLAT
    edge = ZERO
    execution_yes_price: Decimal | None = None
    outcome_cost: Decimal | None = None
    reason = "forecast lies inside the executable bid-ask band"

    if p > market.yes_ask:
        edge = p - market.yes_ask
        if edge >= settings.min_actionable_edge:
            direction = Direction.BUY_YES
            execution_yes_price = market.yes_ask
            outcome_cost = market.yes_ask
            reason = "model probability exceeds the prevailing YES ask"
        else:
            edge = ZERO
            reason = "YES edge is below the fee/slippage buffer"
    elif p < market.yes_bid:
        edge = market.yes_bid - p
        if edge >= settings.min_actionable_edge:
            direction = Direction.BUY_NO
            # V2 is one YES-price book: ask YES at this price to buy NO.
            execution_yes_price = market.yes_bid
            outcome_cost = ONE - market.yes_bid
            reason = "model probability is below the prevailing YES bid"
        else:
            edge = ZERO
            reason = "NO edge is below the fee/slippage buffer"

    raw_target = _whole_contracts(edge * settings.position_scale_contracts)
    if direction is Direction.BUY_NO:
        raw_target = -raw_target
    if direction is Direction.FLAT or raw_target == ZERO:
        raw_target = ZERO
        direction = Direction.FLAT
        execution_yes_price = None
        outcome_cost = None

    capped_target = raw_target
    if abs(capped_target) > settings.max_target_contracts:
        capped_target = settings.max_target_contracts.copy_sign(capped_target)
        reason += "; target capped by MAX_TARGET_CONTRACTS"

    if outcome_cost and outcome_cost > ZERO:
        max_by_market_risk = (
            settings.max_market_risk_dollars / outcome_cost
        ).quantize(Decimal("1"), rounding=ROUND_DOWN)
        if abs(capped_target) > max_by_market_risk:
            capped_target = max_by_market_risk.copy_sign(capped_target)
            reason += "; target capped by MAX_MARKET_RISK_DOLLARS"

    return Signal(
        direction=direction if capped_target != ZERO else Direction.FLAT,
        probability_yes=p,
        actionable_edge=edge,
        raw_target_contracts=raw_target,
        target_position=capped_target,
        execution_yes_price=execution_yes_price,
        outcome_cost=outcome_cost,
        reason=reason,
    )


def _opening_contracts(current: Decimal, target: Decimal) -> Decimal:
    if target == ZERO:
        return ZERO
    if current == ZERO or current * target > ZERO:
        return max(ZERO, abs(target) - abs(current))
    return abs(target)


def build_order_intent(
    market: Market,
    signal: Signal,
    position: Position,
    settings: Settings,
    *,
    total_exposure_dollars: Decimal,
    available_cash_dollars: Decimal,
) -> OrderIntent | None:
    current = position.contracts
    target = signal.target_position
    delta = target - current
    if delta == ZERO:
        return None

    displayed_size = market.yes_ask_size if delta > ZERO else market.yes_bid_size
    count = min(abs(delta), settings.max_order_contracts, displayed_size)
    count = count.quantize(Decimal("1"), rounding=ROUND_DOWN)
    if count < Decimal("1"):
        return None

    book_side = "bid" if delta > ZERO else "ask"
    yes_price = market.yes_ask if book_side == "bid" else market.yes_bid
    side_cost = yes_price if book_side == "bid" else ONE - yes_price
    desired_after_this_order = current + count if delta > ZERO else current - count
    opening = min(count, _opening_contracts(current, desired_after_this_order))
    opening_risk = opening * side_cost

    risk_reducing = opening == ZERO
    if not risk_reducing:
        room = max(ZERO, settings.max_total_exposure_dollars - total_exposure_dollars)
        affordable = min(room, max(ZERO, available_cash_dollars))
        max_opening = (affordable / side_cost).quantize(Decimal("1"), rounding=ROUND_DOWN)
        if max_opening <= ZERO:
            return None
        if opening > max_opening:
            reduction = opening - max_opening
            count -= reduction
            opening = max_opening
            opening_risk = opening * side_cost
        if count < Decimal("1"):
            return None

    final_position = current + count if book_side == "bid" else current - count
    reduce_only = abs(final_position) <= abs(current) and (
        final_position == ZERO or current * final_position >= ZERO
    )
    return OrderIntent(
        ticker=market.ticker,
        book_side=book_side,
        count=count,
        yes_price=yes_price,
        current_position=current,
        target_position=target,
        opening_risk_dollars=opening_risk,
        reduce_only=reduce_only,
        reason=signal.reason,
    )


def should_reforecast(
    market: Market,
    last_fill_reference_price: Decimal | None,
    settings: Settings,
    last_fill_book_side: str | None = None,
) -> bool:
    if last_fill_reference_price is None:
        return True
    executable_price = (
        market.yes_ask
        if last_fill_book_side == "bid"
        else market.yes_bid
        if last_fill_book_side == "ask"
        else market.midpoint
    )
    return abs(executable_price - last_fill_reference_price) >= settings.refresh_move_threshold
