from __future__ import annotations

import base64
import json
import random
import time
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Iterator
from urllib.parse import urlparse

import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

from prophet_trader.config import Settings
from prophet_trader.models import Position, decimal_value


class KalshiAPIError(RuntimeError):
    def __init__(self, method: str, path: str, status_code: int, detail: str) -> None:
        super().__init__(f"Kalshi {method} {path} returned {status_code}: {detail}")
        self.method = method
        self.path = path
        self.status_code = status_code
        self.detail = detail


@dataclass(frozen=True)
class AccountSnapshot:
    available_cash_dollars: Decimal
    portfolio_value_dollars: Decimal

    @property
    def equity_dollars(self) -> Decimal:
        return self.available_cash_dollars + self.portfolio_value_dollars


class KalshiClient:
    """Small current-V2 client with official RSA-PSS request signing."""

    def __init__(
        self,
        settings: Settings,
        *,
        session: requests.Session | None = None,
        sleep: Callable[[float], None] = time.sleep,
        now_ms: Callable[[], int] | None = None,
    ) -> None:
        self.settings = settings
        self.base_url = settings.kalshi_base_url.rstrip("/")
        self.session = session or requests.Session()
        self.sleep = sleep
        self.now_ms = now_ms or (lambda: int(time.time() * 1000))
        self._private_key: Any | None = None

    def _load_private_key(self) -> Any:
        if self._private_key is not None:
            return self._private_key
        pem: bytes
        if self.settings.kalshi_private_key_pem:
            pem = self.settings.kalshi_private_key_pem.encode("utf-8")
        elif self.settings.kalshi_private_key_path:
            pem = Path(self.settings.kalshi_private_key_path).read_bytes()
        else:
            raise RuntimeError("KALSHI_PRIVATE_KEY_PATH or KALSHI_PRIVATE_KEY_PEM is required")
        self._private_key = serialization.load_pem_private_key(pem, password=None)
        return self._private_key

    def signature(self, timestamp: str, method: str, full_path: str) -> str:
        path_without_query = full_path.split("?", 1)[0]
        message = f"{timestamp}{method.upper()}{path_without_query}".encode("utf-8")
        signature = self._load_private_key().sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH,
            ),
            hashes.SHA256(),
        )
        return base64.b64encode(signature).decode("ascii")

    def _auth_headers(self, method: str, path: str) -> dict[str, str]:
        if not self.settings.kalshi_api_key_id:
            raise RuntimeError("KALSHI_API_KEY_ID (or legacy KALSHI_API_KEY) is required")
        timestamp = str(self.now_ms())
        full_path = urlparse(self.base_url + path).path
        return {
            "KALSHI-ACCESS-KEY": self.settings.kalshi_api_key_id,
            "KALSHI-ACCESS-TIMESTAMP": timestamp,
            "KALSHI-ACCESS-SIGNATURE": self.signature(timestamp, method, full_path),
        }

    @staticmethod
    def _error_detail(response: requests.Response) -> str:
        try:
            payload = response.json()
        except ValueError:
            return response.text[:500]
        if isinstance(payload, dict):
            return str(payload.get("message") or payload.get("error") or payload)[:500]
        return str(payload)[:500]

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        payload: dict[str, Any] | None = None,
        authenticated: bool = False,
        expected: tuple[int, ...] = (200,),
    ) -> dict[str, Any]:
        method = method.upper()
        last_error: Exception | None = None
        for attempt in range(self.settings.max_http_retries + 1):
            headers = {"Accept": "application/json"}
            if payload is not None:
                headers["Content-Type"] = "application/json"
            if authenticated:
                headers.update(self._auth_headers(method, path))
            try:
                response = self.session.request(
                    method,
                    self.base_url + path,
                    params=params,
                    json=payload,
                    headers=headers,
                    timeout=self.settings.request_timeout_seconds,
                )
            except requests.RequestException as exc:
                last_error = exc
                retryable = True
            else:
                if response.status_code in expected:
                    if not response.content:
                        return {}
                    try:
                        result = response.json()
                    except ValueError as exc:
                        raise KalshiAPIError(
                            method, path, response.status_code, "non-JSON response"
                        ) from exc
                    if not isinstance(result, dict):
                        raise KalshiAPIError(
                            method, path, response.status_code, "unexpected response shape"
                        )
                    return result
                retryable = response.status_code == 429 or response.status_code >= 500
                last_error = KalshiAPIError(
                    method,
                    path,
                    response.status_code,
                    self._error_detail(response),
                )
            if not retryable or attempt >= self.settings.max_http_retries:
                assert last_error is not None
                raise last_error
            delay = min(4.0, 0.25 * (2**attempt)) + random.uniform(0, 0.1)
            self.sleep(delay)
        raise AssertionError("unreachable")

    @staticmethod
    def _pages(
        fetch: Callable[[str | None], dict[str, Any]],
        collection_key: str,
        *,
        max_pages: int,
    ) -> Iterator[dict[str, Any]]:
        cursor: str | None = None
        for _ in range(max_pages):
            payload = fetch(cursor)
            items = payload.get(collection_key, [])
            if not isinstance(items, list):
                raise RuntimeError(f"Kalshi response field {collection_key!r} is not a list")
            for item in items:
                if isinstance(item, dict):
                    yield item
            next_cursor = payload.get("cursor")
            if not next_cursor or next_cursor == cursor:
                return
            cursor = str(next_cursor)

    def list_open_events(self, *, max_pages: int = 5) -> list[dict[str, Any]]:
        def fetch(cursor: str | None) -> dict[str, Any]:
            params: dict[str, Any] = {
                "limit": 200,
                "status": "open",
                "with_nested_markets": "true",
            }
            if cursor:
                params["cursor"] = cursor
            return self._request("GET", "/events", params=params)

        return list(self._pages(fetch, "events", max_pages=max_pages))

    def get_event(self, event_ticker: str) -> dict[str, Any]:
        return self._request(
            "GET",
            f"/events/{event_ticker}",
            params={"with_nested_markets": "true"},
        )

    def get_event_metadata(self, event_ticker: str) -> dict[str, Any]:
        return self._request("GET", f"/events/{event_ticker}/metadata")

    def get_market(self, ticker: str) -> dict[str, Any]:
        return self._request("GET", f"/markets/{ticker}")

    def exchange_status(self) -> dict[str, Any]:
        return self._request("GET", "/exchange/status")

    def get_account(self) -> AccountSnapshot:
        payload = self._request("GET", "/portfolio/balance", authenticated=True)
        if payload.get("balance_dollars") not in {None, ""}:
            balance = decimal_value(payload.get("balance_dollars"))
        else:
            balance = decimal_value(payload.get("balance")) / Decimal("100")
        # Kalshi currently documents portfolio_value as cents.
        portfolio = decimal_value(payload.get("portfolio_value")) / Decimal("100")
        return AccountSnapshot(balance, portfolio)

    def list_positions(self, *, max_pages: int = 10) -> list[Position]:
        def fetch(cursor: str | None) -> dict[str, Any]:
            params: dict[str, Any] = {"limit": 1000, "count_filter": "position"}
            if cursor:
                params["cursor"] = cursor
            return self._request(
                "GET", "/portfolio/positions", params=params, authenticated=True
            )

        positions: list[Position] = []
        for item in self._pages(fetch, "market_positions", max_pages=max_pages):
            contracts = decimal_value(item.get("position_fp") or item.get("position"))
            if contracts:
                positions.append(
                    Position(
                        ticker=str(item.get("ticker", "")),
                        contracts=contracts,
                        exposure_dollars=abs(decimal_value(item.get("market_exposure_dollars"))),
                    )
                )
        return positions

    def get_order(self, order_id: str) -> dict[str, Any]:
        return self._request(
            "GET", f"/portfolio/orders/{order_id}", authenticated=True
        )

    def list_orders(
        self,
        *,
        ticker: str | None = None,
        status: str | None = None,
        max_pages: int = 5,
    ) -> list[dict[str, Any]]:
        def fetch(cursor: str | None) -> dict[str, Any]:
            params: dict[str, Any] = {"limit": 1000}
            if ticker:
                params["ticker"] = ticker
            if status:
                params["status"] = status
            if cursor:
                params["cursor"] = cursor
            return self._request(
                "GET", "/portfolio/orders", params=params, authenticated=True
            )

        return list(self._pages(fetch, "orders", max_pages=max_pages))

    def create_order(
        self,
        *,
        ticker: str,
        client_order_id: str,
        book_side: str,
        count: Decimal,
        yes_price: Decimal,
        reduce_only: bool,
    ) -> dict[str, Any]:
        if book_side not in {"bid", "ask"}:
            raise ValueError("V2 order side must be bid or ask")
        if not count.is_finite() or count <= 0:
            raise ValueError("order count must be a positive finite Decimal")
        if not yes_price.is_finite() or not Decimal("0") < yes_price < Decimal("1"):
            raise ValueError("order price must be strictly between 0 and 1")
        payload = {
            "ticker": ticker,
            "client_order_id": client_order_id,
            "side": book_side,
            "count": format(count.quantize(Decimal("0.01")), "f"),
            "price": format(yes_price.quantize(Decimal("0.0001")), "f"),
            "time_in_force": "good_till_canceled",
            "self_trade_prevention_type": "taker_at_cross",
            "post_only": False,
            "cancel_order_on_pause": True,
            "reduce_only": reduce_only,
            "subaccount": 0,
            "exchange_index": 0,
        }
        return self._request(
            "POST",
            "/portfolio/events/orders",
            payload=payload,
            authenticated=True,
            expected=(201,),
        )

    def cancel_order(self, order_id: str) -> dict[str, Any]:
        return self._request(
            "DELETE",
            f"/portfolio/events/orders/{order_id}",
            params={"subaccount": 0},
            authenticated=True,
        )

    @staticmethod
    def redacted_json(payload: dict[str, Any]) -> str:
        """Serialize an API result for the journal; no request credentials are present."""
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
