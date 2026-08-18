from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Any


ZERO = Decimal("0")
ONE = Decimal("1")


def decimal_value(value: Any, default: Decimal = ZERO) -> Decimal:
    if value is None or value == "":
        return default
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return default


def parse_timestamp(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value).strip().replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


class Direction(str, Enum):
    BUY_YES = "buy_yes"
    BUY_NO = "buy_no"
    FLAT = "flat"


@dataclass(frozen=True)
class Evidence:
    claim: str
    source_url: str
    published_at: str | None = None


@dataclass(frozen=True)
class Forecast:
    probability_yes: Decimal
    rationale: str
    resolution_clear: bool
    resolution_clarity_reason: str
    uncertainty: str
    confidence: str = "medium"
    evidence: tuple[Evidence, ...] = ()
    data_as_of: str | None = None
    model: str = ""
    response_id: str | None = None
    raw: dict[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        if not ZERO <= self.probability_yes <= ONE:
            raise ValueError("probability_yes must be between 0 and 1")
        if self.uncertainty not in {"low", "medium", "high"}:
            raise ValueError("uncertainty must be low, medium, or high")
        if self.confidence not in {"low", "medium", "high"}:
            raise ValueError("confidence must be low, medium, or high")


@dataclass(frozen=True)
class Market:
    ticker: str
    event_ticker: str
    series_ticker: str
    category: str
    title: str
    subtitle: str
    yes_sub_title: str
    no_sub_title: str
    rules_primary: str
    rules_secondary: str
    settlement_sources: tuple[dict[str, str], ...]
    close_time: datetime
    expected_expiration_time: datetime
    yes_bid: Decimal
    yes_ask: Decimal
    no_bid: Decimal
    no_ask: Decimal
    yes_bid_size: Decimal
    yes_ask_size: Decimal
    liquidity_dollars: Decimal
    volume: Decimal
    open_interest: Decimal
    can_close_early: bool
    early_close_condition: str
    status: str = "open"
    price_ranges: tuple[dict[str, str], ...] = ()
    raw: dict[str, Any] = field(default_factory=dict, compare=False)

    @property
    def midpoint(self) -> Decimal:
        return (self.yes_bid + self.yes_ask) / Decimal("2")

    @property
    def spread(self) -> Decimal:
        return self.yes_ask - self.yes_bid

    @staticmethod
    def _displayed_depth_dollars(price: Decimal, size: Decimal) -> Decimal:
        """Return executable dollars at one displayed level, failing closed."""
        if not price.is_finite() or not size.is_finite():
            return ZERO
        if not ZERO < price < ONE or size <= ZERO:
            return ZERO
        return price * size

    @property
    def buy_yes_depth_dollars(self) -> Decimal:
        """Displayed dollars available to buy YES at the current YES ask."""
        return self._displayed_depth_dollars(self.yes_ask, self.yes_ask_size)

    @property
    def buy_no_depth_dollars(self) -> Decimal:
        """Displayed dollars available to buy NO at the implied NO ask."""
        return self._displayed_depth_dollars(ONE - self.yes_bid, self.yes_bid_size)

    @property
    def executable_depth_dollars(self) -> Decimal:
        """Conservative two-sided top-of-book depth used by the liquidity gate.

        Kalshi deprecated its aggregate liquidity fields in February 2026, so
        tradability is measured from current displayed quotes and sizes instead.
        Requiring the smaller directional depth keeps the pre-forecast screen
        agnostic to which side the model may ultimately choose.
        """
        return min(self.buy_yes_depth_dollars, self.buy_no_depth_dollars)

    @classmethod
    def from_api(cls, data: dict[str, Any], event: dict[str, Any] | None = None) -> "Market":
        event = event or {}
        close_time = parse_timestamp(data.get("close_time"))
        expected = parse_timestamp(
            data.get("expected_expiration_time")
            or data.get("expiration_time")
            or data.get("latest_expiration_time")
        )
        if close_time is None or expected is None:
            raise ValueError(f"market {data.get('ticker', '<unknown>')} lacks lifecycle timestamps")

        yes_bid = decimal_value(
            data.get("yes_bid_dollars"), decimal_value(data.get("yes_bid")) / 100
        )
        yes_ask = decimal_value(
            data.get("yes_ask_dollars"), decimal_value(data.get("yes_ask")) / 100
        )
        no_bid = decimal_value(data.get("no_bid_dollars"), decimal_value(data.get("no_bid")) / 100)
        no_ask = decimal_value(data.get("no_ask_dollars"), decimal_value(data.get("no_ask")) / 100)

        # Older responses may omit one side. Binary complements are exact on Kalshi's book.
        if no_ask == ZERO and yes_bid > ZERO:
            no_ask = ONE - yes_bid
        if no_bid == ZERO and yes_ask > ZERO:
            no_bid = ONE - yes_ask

        settlement_sources = tuple(
            {
                "name": str(item.get("name", "")),
                "url": str(item.get("url", "")),
            }
            for item in event.get("settlement_sources", [])
            if isinstance(item, dict)
        )
        return cls(
            ticker=str(data.get("ticker", "")),
            event_ticker=str(data.get("event_ticker") or event.get("event_ticker", "")),
            series_ticker=str(data.get("series_ticker") or event.get("series_ticker", "")),
            category=str(data.get("category") or event.get("category", "")),
            title=str(data.get("title") or event.get("title", "")),
            subtitle=str(data.get("subtitle") or event.get("sub_title", "")),
            yes_sub_title=str(data.get("yes_sub_title", "")),
            no_sub_title=str(data.get("no_sub_title", "")),
            rules_primary=str(data.get("rules_primary", "")),
            rules_secondary=str(data.get("rules_secondary", "")),
            settlement_sources=settlement_sources,
            close_time=close_time,
            expected_expiration_time=expected,
            yes_bid=yes_bid,
            yes_ask=yes_ask,
            no_bid=no_bid,
            no_ask=no_ask,
            yes_bid_size=decimal_value(data.get("yes_bid_size_fp") or data.get("yes_bid_size")),
            yes_ask_size=decimal_value(data.get("yes_ask_size_fp") or data.get("yes_ask_size")),
            liquidity_dollars=decimal_value(
                data.get("liquidity_dollars"),
                decimal_value(data.get("liquidity")) / Decimal("100"),
            ),
            volume=decimal_value(data.get("volume_fp") or data.get("volume")),
            open_interest=decimal_value(data.get("open_interest_fp") or data.get("open_interest")),
            can_close_early=bool(data.get("can_close_early", False)),
            early_close_condition=str(data.get("early_close_condition", "")),
            status=str(data.get("status", "open")),
            price_ranges=tuple(data.get("price_ranges") or ()),
            raw=dict(data),
        )


@dataclass(frozen=True)
class Signal:
    direction: Direction
    probability_yes: Decimal
    actionable_edge: Decimal
    raw_target_contracts: Decimal
    target_position: Decimal
    execution_yes_price: Decimal | None
    outcome_cost: Decimal | None
    reason: str


@dataclass(frozen=True)
class Position:
    ticker: str
    contracts: Decimal
    exposure_dollars: Decimal = ZERO


@dataclass(frozen=True)
class OrderIntent:
    ticker: str
    book_side: str
    count: Decimal
    yes_price: Decimal
    current_position: Decimal
    target_position: Decimal
    opening_risk_dollars: Decimal
    reduce_only: bool
    reason: str

    def __post_init__(self) -> None:
        if self.book_side not in {"bid", "ask"}:
            raise ValueError("book_side must be bid or ask")
        if self.count <= ZERO:
            raise ValueError("order count must be positive")
        if not ZERO < self.yes_price < ONE:
            raise ValueError("yes_price must be between 0 and 1")
