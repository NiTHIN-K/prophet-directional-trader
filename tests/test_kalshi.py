from __future__ import annotations

import base64
import json
import sys
import unittest
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import requests
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prophet_trader.kalshi import KalshiClient  # noqa: E402


def json_response(status_code: int, payload: dict[str, object] | None = None) -> requests.Response:
    response = requests.Response()
    response.status_code = status_code
    response.headers["Content-Type"] = "application/json"
    response._content = b"" if payload is None else json.dumps(payload).encode("utf-8")
    return response


class KalshiClientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        cls.public_key = cls.private_key.public_key()
        cls.private_key_pem = cls.private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        ).decode("ascii")

    def settings(self, **overrides: object) -> SimpleNamespace:
        values: dict[str, object] = {
            "kalshi_base_url": "https://api.test.example/trade-api/v2/",
            "kalshi_api_key_id": "test-key-id",
            "kalshi_private_key_pem": self.private_key_pem,
            "kalshi_private_key_path": None,
            "max_http_retries": 0,
            "request_timeout_seconds": 4.5,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def assert_valid_signature(
        self,
        signature: str,
        *,
        timestamp: str,
        method: str,
        path: str,
    ) -> None:
        self.public_key.verify(
            base64.b64decode(signature),
            f"{timestamp}{method.upper()}{path}".encode("utf-8"),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH,
            ),
            hashes.SHA256(),
        )

    def test_signature_uppercases_method_and_strips_query(self) -> None:
        client = KalshiClient(self.settings())

        signature = client.signature(
            "1720000000123",
            "get",
            "/trade-api/v2/portfolio/orders?limit=25",
        )

        self.assert_valid_signature(
            signature,
            timestamp="1720000000123",
            method="GET",
            path="/trade-api/v2/portfolio/orders",
        )
        with self.assertRaises(InvalidSignature):
            self.assert_valid_signature(
                signature,
                timestamp="1720000000123",
                method="GET",
                path="/trade-api/v2/portfolio/orders?limit=25",
            )

    def test_auth_headers_sign_the_full_api_path(self) -> None:
        client = KalshiClient(self.settings(), now_ms=lambda: 1720000000123)

        headers = client._auth_headers("get", "/portfolio/orders?limit=5")

        self.assertEqual(headers["KALSHI-ACCESS-KEY"], "test-key-id")
        self.assertEqual(headers["KALSHI-ACCESS-TIMESTAMP"], "1720000000123")
        self.assertEqual(
            set(headers),
            {
                "KALSHI-ACCESS-KEY",
                "KALSHI-ACCESS-TIMESTAMP",
                "KALSHI-ACCESS-SIGNATURE",
            },
        )
        self.assert_valid_signature(
            headers["KALSHI-ACCESS-SIGNATURE"],
            timestamp="1720000000123",
            method="GET",
            path="/trade-api/v2/portfolio/orders",
        )

    def test_auth_headers_require_an_api_key_id(self) -> None:
        client = KalshiClient(self.settings(kalshi_api_key_id=None))

        with self.assertRaisesRegex(RuntimeError, "KALSHI_API_KEY_ID"):
            client._auth_headers("GET", "/portfolio/balance")

    def test_signature_requires_private_key_material(self) -> None:
        client = KalshiClient(
            self.settings(kalshi_private_key_pem=None, kalshi_private_key_path=None)
        )

        with self.assertRaisesRegex(RuntimeError, "KALSHI_PRIVATE_KEY"):
            client.signature("1720000000123", "GET", "/trade-api/v2/portfolio/balance")

    def test_public_market_request_does_not_require_or_send_credentials(self) -> None:
        session = Mock(spec=requests.Session)
        session.request.return_value = json_response(
            200,
            {"market": {"ticker": "KXTEST-26"}},
        )
        client = KalshiClient(
            self.settings(
                kalshi_api_key_id=None,
                kalshi_private_key_pem=None,
                kalshi_private_key_path=None,
            ),
            session=session,
        )

        result = client.get_market("KXTEST-26")

        self.assertEqual(result, {"market": {"ticker": "KXTEST-26"}})
        session.request.assert_called_once_with(
            "GET",
            "https://api.test.example/trade-api/v2/markets/KXTEST-26",
            params=None,
            json=None,
            headers={"Accept": "application/json"},
            timeout=4.5,
        )

    def test_create_order_sends_exact_v2_payload_and_authenticated_headers(self) -> None:
        session = Mock(spec=requests.Session)
        session.request.return_value = json_response(
            201,
            {"order": {"order_id": "order-1"}},
        )
        client = KalshiClient(
            self.settings(),
            session=session,
            now_ms=lambda: 1720000000123,
        )

        result = client.create_order(
            ticker="KXTEST-26",
            client_order_id="client-order-1",
            book_side="bid",
            count=Decimal("1.236"),
            yes_price=Decimal("0.42555"),
            reduce_only=False,
        )

        self.assertEqual(result, {"order": {"order_id": "order-1"}})
        session.request.assert_called_once()
        args, kwargs = session.request.call_args
        self.assertEqual(
            args,
            ("POST", "https://api.test.example/trade-api/v2/portfolio/events/orders"),
        )
        self.assertIsNone(kwargs["params"])
        self.assertEqual(kwargs["timeout"], 4.5)
        self.assertEqual(
            kwargs["json"],
            {
                "ticker": "KXTEST-26",
                "client_order_id": "client-order-1",
                "side": "bid",
                "count": "1.24",
                "price": "0.4256",
                "time_in_force": "good_till_canceled",
                "self_trade_prevention_type": "taker_at_cross",
                "post_only": False,
                "cancel_order_on_pause": True,
                "reduce_only": False,
                "subaccount": 0,
                "exchange_index": 0,
            },
        )
        headers = kwargs["headers"]
        self.assertEqual(headers["Accept"], "application/json")
        self.assertEqual(headers["Content-Type"], "application/json")
        self.assertEqual(headers["KALSHI-ACCESS-KEY"], "test-key-id")
        self.assert_valid_signature(
            headers["KALSHI-ACCESS-SIGNATURE"],
            timestamp=headers["KALSHI-ACCESS-TIMESTAMP"],
            method="POST",
            path="/trade-api/v2/portfolio/events/orders",
        )

    def test_cancel_order_uses_path_id_and_subaccount_query_parameter(self) -> None:
        session = Mock(spec=requests.Session)
        session.request.return_value = json_response(200, {"order": {"status": "canceled"}})
        client = KalshiClient(
            self.settings(),
            session=session,
            now_ms=lambda: 1720000000123,
        )

        result = client.cancel_order("order-1")

        self.assertEqual(result, {"order": {"status": "canceled"}})
        session.request.assert_called_once()
        args, kwargs = session.request.call_args
        self.assertEqual(
            args,
            ("DELETE", "https://api.test.example/trade-api/v2/portfolio/events/orders/order-1"),
        )
        self.assertEqual(kwargs["params"], {"subaccount": 0})
        self.assertIsNone(kwargs["json"])
        self.assertNotIn("Content-Type", kwargs["headers"])
        self.assert_valid_signature(
            kwargs["headers"]["KALSHI-ACCESS-SIGNATURE"],
            timestamp=kwargs["headers"]["KALSHI-ACCESS-TIMESTAMP"],
            method="DELETE",
            path="/trade-api/v2/portfolio/events/orders/order-1",
        )

    def test_retry_refreshes_auth_but_preserves_the_idempotency_key_and_payload(self) -> None:
        session = Mock(spec=requests.Session)
        session.request.side_effect = [
            json_response(429, {"error": "rate limited"}),
            json_response(201, {"order": {"order_id": "order-1"}}),
        ]
        sleep = Mock()
        timestamps = iter((1720000000123, 1720000000456))
        client = KalshiClient(
            self.settings(max_http_retries=1),
            session=session,
            sleep=sleep,
            now_ms=lambda: next(timestamps),
        )

        with patch("prophet_trader.kalshi.random.uniform", return_value=0.0):
            client.create_order(
                ticker="KXTEST-26",
                client_order_id="stable-client-order-id",
                book_side="ask",
                count=Decimal("2"),
                yes_price=Decimal("0.61"),
                reduce_only=True,
            )

        self.assertEqual(session.request.call_count, 2)
        first_call, second_call = session.request.call_args_list
        self.assertEqual(first_call.kwargs["json"], second_call.kwargs["json"])
        self.assertEqual(
            first_call.kwargs["json"]["client_order_id"],
            "stable-client-order-id",
        )
        self.assertEqual(
            first_call.kwargs["headers"]["KALSHI-ACCESS-TIMESTAMP"],
            "1720000000123",
        )
        self.assertEqual(
            second_call.kwargs["headers"]["KALSHI-ACCESS-TIMESTAMP"],
            "1720000000456",
        )
        for call in (first_call, second_call):
            headers = call.kwargs["headers"]
            self.assert_valid_signature(
                headers["KALSHI-ACCESS-SIGNATURE"],
                timestamp=headers["KALSHI-ACCESS-TIMESTAMP"],
                method="POST",
                path="/trade-api/v2/portfolio/events/orders",
            )
        sleep.assert_called_once_with(0.25)

    def test_create_order_rejects_invalid_v2_side(self) -> None:
        client = KalshiClient(self.settings())

        with self.assertRaisesRegex(ValueError, "bid or ask"):
            client.create_order(
                ticker="KXTEST-26",
                client_order_id="client-order-1",
                book_side="yes",
                count=Decimal("1"),
                yes_price=Decimal("0.50"),
                reduce_only=False,
            )


if __name__ == "__main__":
    unittest.main()
