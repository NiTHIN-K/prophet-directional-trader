from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from prophet_trader.config import Settings
from prophet_trader.kalshi import AccountSnapshot, KalshiClient
from prophet_trader.models import Market, OrderIntent, Position, ZERO, decimal_value
from prophet_trader.store import StateStore


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PortfolioSnapshot:
    positions: dict[str, Position]
    available_cash_dollars: Decimal
    total_exposure_dollars: Decimal


class PaperBroker:
    def __init__(self, settings: Settings, store: StateStore) -> None:
        self.settings = settings
        self.store = store

    def reconcile_and_cancel(self) -> None:
        return

    def blocked_tickers(self) -> set[str]:
        return set()

    def snapshot(self) -> PortfolioSnapshot:
        positions = self.store.paper_positions()
        exposure = sum((item.exposure_dollars for item in positions.values()), ZERO)
        cash = self.store.paper_cash(self.settings.paper_starting_cash)
        return PortfolioSnapshot(positions, cash, exposure)

    def execute(
        self,
        *,
        cycle_id: str,
        market: Market,
        intent: OrderIntent,
    ) -> dict[str, Any]:
        client_order_id = str(uuid.uuid4())
        position = self.store.apply_paper_fill(
            cycle_id=cycle_id,
            client_order_id=client_order_id,
            market=market,
            intent=intent,
            starting_cash=self.settings.paper_starting_cash,
        )
        return {
            "client_order_id": client_order_id,
            "order_id": f"paper-{client_order_id}",
            "fill_count": str(intent.count),
            "remaining_count": "0",
            "position": str(position.contracts),
            "simulated": True,
        }


class LiveBroker:
    def __init__(self, settings: Settings, store: StateStore, client: KalshiClient) -> None:
        self.settings = settings
        self.store = store
        self.client = client

    @staticmethod
    def _order_payload(payload: dict[str, Any]) -> dict[str, Any]:
        nested = payload.get("order")
        return nested if isinstance(nested, dict) else payload

    def reconcile_and_cancel(self) -> None:
        """Use actual exchange fill state, then cancel every stale bot remainder."""
        for stored in self.store.open_live_orders():
            exchange_order_id = stored.get("exchange_order_id")
            if not exchange_order_id:
                # A prior submission may have timed out after acceptance. Locate it by our UUID.
                matches = self.client.list_orders(ticker=str(stored["ticker"]), max_pages=2)
                match = next(
                    (
                        item
                        for item in matches
                        if str(item.get("client_order_id")) == str(stored["client_order_id"])
                    ),
                    None,
                )
                if match:
                    exchange_order_id = match.get("order_id")
                else:
                    LOGGER.warning(
                        "unresolved prior submission; refusing to resubmit "
                        "ticker=%s client_order_id=%s",
                        stored["ticker"],
                        stored["client_order_id"],
                    )
                    continue
            response = self.client.get_order(str(exchange_order_id))
            order = self._order_payload(response)
            fill_count = decimal_value(order.get("fill_count_fp") or order.get("fill_count"))
            remaining = decimal_value(
                order.get("remaining_count_fp") or order.get("remaining_count")
            )
            status = str(order.get("status", "submitted"))
            if remaining > ZERO:
                cancel_response = self.client.cancel_order(str(exchange_order_id))
                response = {"order": order, "cancel": cancel_response}
                remaining = ZERO
                status = "canceled"
            elif status not in {"executed", "filled", "canceled"}:
                status = "filled" if fill_count > ZERO else "canceled"
            self.store.update_live_order(
                str(stored["client_order_id"]),
                exchange_order_id=str(exchange_order_id),
                fill_count=fill_count,
                remaining_count=remaining,
                status=status,
                response=response,
            )

    def blocked_tickers(self) -> set[str]:
        """Tickers with an unresolved or still-open submission must not be resubmitted."""
        return {str(item["ticker"]) for item in self.store.open_live_orders()}

    def snapshot(self) -> PortfolioSnapshot:
        positions = {item.ticker: item for item in self.client.list_positions()}
        account: AccountSnapshot = self.client.get_account()
        exposure = sum((item.exposure_dollars for item in positions.values()), ZERO)
        return PortfolioSnapshot(positions, account.available_cash_dollars, exposure)

    def execute(
        self,
        *,
        cycle_id: str,
        market: Market,
        intent: OrderIntent,
    ) -> dict[str, Any]:
        client_order_id = str(uuid.uuid4())
        self.store.record_live_submission(
            cycle_id=cycle_id,
            client_order_id=client_order_id,
            market=market,
            intent=intent,
            response=None,
            status="pending",
        )
        try:
            response = self.client.create_order(
                ticker=market.ticker,
                client_order_id=client_order_id,
                book_side=intent.book_side,
                count=intent.count,
                yes_price=intent.yes_price,
                reduce_only=intent.reduce_only,
            )
        except Exception:
            # Keep the pending record. The next cycle must reconcile it before another order.
            LOGGER.exception(
                "order submission outcome is unknown; persisted for reconciliation ticker=%s",
                market.ticker,
            )
            raise
        order = self._order_payload(response)
        fill_count = decimal_value(order.get("fill_count") or order.get("fill_count_fp"))
        remaining = decimal_value(
            order.get("remaining_count") or order.get("remaining_count_fp")
        )
        status = "filled" if remaining == ZERO else ("partial" if fill_count > ZERO else "resting")
        self.store.record_live_submission(
            cycle_id=cycle_id,
            client_order_id=client_order_id,
            market=market,
            intent=intent,
            response=response,
            status=status,
        )
        return response
