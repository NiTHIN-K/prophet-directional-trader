from __future__ import annotations

import json
import sys
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from prophet_trader.engine import TraderEngine
from prophet_trader.execution import PaperBroker
from prophet_trader.forecast import (
    ForecastQuotaExceeded,
    ForecastRequestTimeout,
    PreparedForecast,
    ProviderForecastResponse,
)
from prophet_trader.models import Forecast, Market
from prophet_trader.risk import RiskManager
from prophet_trader.store import StateStore
from tests.test_strategy import NOW, make_forecast, make_market, make_settings


def market_api_payload(market: Market) -> dict[str, Any]:
    return {
        "ticker": market.ticker,
        "event_ticker": market.event_ticker,
        "title": market.title,
        "subtitle": market.subtitle,
        "yes_sub_title": market.yes_sub_title,
        "no_sub_title": market.no_sub_title,
        "rules_primary": market.rules_primary,
        "rules_secondary": market.rules_secondary,
        "close_time": market.close_time.isoformat(),
        "expected_expiration_time": market.expected_expiration_time.isoformat(),
        "yes_bid_dollars": str(market.yes_bid),
        "yes_ask_dollars": str(market.yes_ask),
        "no_bid_dollars": str(market.no_bid),
        "no_ask_dollars": str(market.no_ask),
        "yes_bid_size_fp": str(market.yes_bid_size),
        "yes_ask_size_fp": str(market.yes_ask_size),
        "liquidity_dollars": str(market.liquidity_dollars),
        "volume_fp": str(market.volume),
        "open_interest_fp": str(market.open_interest),
        "can_close_early": market.can_close_early,
        "early_close_condition": market.early_close_condition,
        "status": market.status,
    }


class OfflineKalshi:
    """Only public metadata reads used by an explicit-ticker paper cycle exist here."""

    def __init__(self, market: Market) -> None:
        self.market = market
        self.refreshed_market: Market | None = None
        self.market_reads = 0
        self.event_reads = 0
        self.authenticated_calls: list[str] = []

    def get_market(self, ticker: str) -> dict[str, Any]:
        self.market_reads += 1
        if ticker != self.market.ticker:
            raise AssertionError(f"unexpected ticker read: {ticker}")
        selected = self.refreshed_market if self.market_reads > 1 else self.market
        return {"market": market_api_payload(selected or self.market)}

    def get_event(self, event_ticker: str) -> dict[str, Any]:
        self.event_reads += 1
        if event_ticker != self.market.event_ticker:
            raise AssertionError(f"unexpected event read: {event_ticker}")
        return {
            "event": {
                "event_ticker": self.market.event_ticker,
                "series_ticker": self.market.series_ticker,
                "category": self.market.category,
                "title": self.market.title,
                "settlement_sources": list(self.market.settlement_sources),
            }
        }

    def list_open_events(self, *, max_pages: int = 5) -> list[dict[str, Any]]:
        raise AssertionError("explicit-ticker test must not make a broad discovery request")

    def exchange_status(self) -> dict[str, Any]:
        self.authenticated_calls.append("exchange_status")
        raise AssertionError("paper mode must not query live exchange status")

    def create_order(self, **kwargs: Any) -> dict[str, Any]:
        self.authenticated_calls.append("create_order")
        raise AssertionError("paper mode must not create a Kalshi order")

    def cancel_order(self, order_id: str) -> dict[str, Any]:
        self.authenticated_calls.append("cancel_order")
        raise AssertionError("paper mode must not cancel a Kalshi order")

    def list_orders(self, **kwargs: Any) -> list[dict[str, Any]]:
        self.authenticated_calls.append("list_orders")
        raise AssertionError("paper mode must not read live orders")

    def list_positions(self) -> list[Any]:
        self.authenticated_calls.append("list_positions")
        raise AssertionError("paper mode must not read live positions")

    def get_account(self) -> Any:
        self.authenticated_calls.append("get_account")
        raise AssertionError("paper mode must not read the live account")


class OfflineForecaster:
    def __init__(self, forecast: Forecast) -> None:
        self.result = forecast
        self.calls: list[str] = []
        self.exception: Exception | None = None

    def prepare(
        self,
        market: Market,
        *,
        slot_iso: str,
        context: dict[str, Any],
    ) -> PreparedForecast:
        return PreparedForecast(
            ticker=market.ticker,
            prompt=f"{market.ticker}:{slot_iso}",
            prompt_version="test-v1",
            context_hash=f"hash:{market.ticker}:{slot_iso}",
        )

    def request(self, prepared: PreparedForecast) -> ProviderForecastResponse:
        self.calls.append(prepared.ticker)
        if self.exception is not None:
            raise self.exception
        return ProviderForecastResponse(
            provider_request_id=f"provider-{len(self.calls)}",
            text="{}",
            raw_response={"raw": True},
            input_tokens=100,
            cached_tokens=0,
            reasoning_tokens=50,
            output_tokens=25,
            total_tokens=175,
            search_queries=("test query",),
            duration_ms=10,
            estimated_cost_dollars=Decimal("0.01"),
        )

    def parse(self, response: ProviderForecastResponse) -> Forecast:
        return self.result


class TraderEnginePaperCycleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        root = Path(self.temporary_directory.name)
        self.settings = replace(
            make_settings(),
            root=root,
            state_db_path=root / "state" / "trader.sqlite3",
            kill_switch_path=root / "STOP",
            trading_mode="paper",
            forecasting_enabled=True,
            runtime_environment="production",
        )
        # Kalshi deprecated liquidity_dollars in February 2026; real responses now
        # report zero even when the displayed top of book has executable depth.
        self.market = replace(make_market(), liquidity_dollars=Decimal("0"))
        self.client = OfflineKalshi(self.market)
        self.forecaster = OfflineForecaster(make_forecast("0.70"))
        self.store = StateStore(self.settings.state_db_path)
        self.broker = PaperBroker(self.settings, self.store)
        self.engine = TraderEngine(
            self.settings,
            client=self.client,  # type: ignore[arg-type]
            forecaster=self.forecaster,  # type: ignore[arg-type]
            store=self.store,
            broker=self.broker,
            risk=RiskManager(self.settings),
        )

    def test_full_paper_cycle_uses_fakes_only_and_persists_complete_journal(self) -> None:
        summary = self.engine.run_cycle(tickers=[self.market.ticker], now=NOW)

        self.assertEqual(summary["mode"], "paper")
        self.assertEqual(summary["discovered"], 1)
        self.assertEqual(summary["eligible"], 1)
        self.assertEqual(summary["forecasted"], 1)
        self.assertEqual(summary["signals"], 1)
        self.assertEqual(summary["orders_submitted"], 1)
        self.assertEqual(summary["skipped"], 0)
        self.assertEqual(summary["errors"], [])
        self.assertEqual(len(summary["orders"]), 1)
        self.assertTrue(summary["orders"][0]["simulated"])
        self.assertEqual(summary["orders"][0]["ticker"], self.market.ticker)
        self.assertEqual(summary["orders"][0]["book_side"], "bid")
        self.assertEqual(summary["orders"][0]["count"], "26")

        self.assertEqual(self.client.authenticated_calls, [])
        self.assertEqual(self.client.market_reads, 2)
        self.assertEqual(self.client.event_reads, 1)
        self.assertEqual(self.forecaster.calls, [self.market.ticker])

        # Reopen the journal to prove records were committed, rather than relying on the
        # original StateStore object's in-memory state.
        reopened = StateStore(self.settings.state_db_path)
        cycle = reopened.recent_cycles(1)[0]
        self.assertEqual(cycle["cycle_id"], summary["cycle_id"])
        self.assertEqual(cycle["mode"], "paper")
        self.assertEqual(cycle["status"], "completed")
        persisted_summary = json.loads(cycle["summary_json"])
        self.assertEqual(persisted_summary["orders_submitted"], 1)
        self.assertTrue(persisted_summary["orders"][0]["simulated"])

        decisions = reopened.recent_decisions(10)
        self.assertEqual(len(decisions), 1)
        self.assertEqual(decisions[0]["ticker"], self.market.ticker)
        self.assertEqual(decisions[0]["direction"], "buy_yes")
        self.assertEqual(decisions[0]["allowed"], 1)
        self.assertEqual(Decimal(decisions[0]["order_count"]), Decimal("26"))

        snapshot = PaperBroker(self.settings, reopened).snapshot()
        self.assertEqual(
            snapshot.positions[self.market.ticker].contracts,
            Decimal("26"),
        )
        self.assertEqual(snapshot.total_exposure_dollars, Decimal("11.44"))
        self.assertEqual(snapshot.available_cash_dollars, Decimal("188.56"))
        self.assertEqual(
            reopened.get_last_fill_reference(self.market.ticker),
            self.market.yes_ask,
        )
        self.assertEqual(reopened.open_live_orders(), [])

        with reopened.connection() as connection:
            counts = connection.execute(
                """
                SELECT
                    (SELECT COUNT(*) FROM forecasts WHERE cycle_id=?) AS forecasts,
                    (SELECT COUNT(*) FROM decisions WHERE cycle_id=?) AS decisions,
                    (SELECT COUNT(*) FROM orders WHERE cycle_id=? AND mode='paper') AS paper_orders,
                    (SELECT COUNT(*) FROM orders WHERE cycle_id=? AND mode='live') AS live_orders
                """,
                (summary["cycle_id"],) * 4,
            ).fetchone()
        self.assertEqual(counts["forecasts"], 1)
        self.assertEqual(counts["decisions"], 1)
        self.assertEqual(counts["paper_orders"], 1)
        self.assertEqual(counts["live_orders"], 0)

    def test_rebalances_against_post_forecast_quote_refresh(self) -> None:
        self.client.refreshed_market = replace(
            self.market,
            yes_bid=Decimal("0.58"),
            yes_ask=Decimal("0.60"),
            no_bid=Decimal("0.40"),
            no_ask=Decimal("0.42"),
            yes_ask_size=Decimal("4"),
        )

        summary = self.engine.run_cycle(tickers=[self.market.ticker], now=NOW)

        self.assertEqual(summary["orders_submitted"], 1)
        self.assertEqual(summary["orders"][0]["yes_price"], "0.60")
        self.assertEqual(summary["orders"][0]["count"], "4")

    def test_post_forecast_depth_disappearance_fails_closed(self) -> None:
        self.client.refreshed_market = replace(
            self.market,
            yes_ask_size=Decimal("0"),
        )

        summary = self.engine.run_cycle(tickers=[self.market.ticker], now=NOW)

        self.assertEqual(summary["forecasted"], 1)
        self.assertEqual(summary["orders_submitted"], 0)
        self.assertEqual(summary["skipped"], 1)
        decision = self.store.recent_decisions(1)[0]
        self.assertEqual(decision["allowed"], 0)
        self.assertIn("post-forecast market guardrail", decision["reason"])

    def test_held_short_can_exit_when_only_yes_ask_is_executable(self) -> None:
        self.forecaster.result = make_forecast("0.20")
        first = self.engine.run_cycle(tickers=[self.market.ticker], now=NOW)
        self.assertEqual(first["orders_submitted"], 1)
        self.assertLess(
            self.broker.snapshot().positions[self.market.ticker].contracts,
            Decimal("0"),
        )

        one_sided = replace(
            self.market,
            yes_bid=Decimal("0"),
            yes_ask=Decimal("0.03"),
            no_bid=Decimal("0.97"),
            no_ask=Decimal("1"),
            yes_bid_size=Decimal("0"),
            yes_ask_size=Decimal("100"),
        )
        self.client.market = one_sided
        self.client.refreshed_market = one_sided

        second = self.engine.run_cycle(
            tickers=[self.market.ticker], now=NOW + timedelta(hours=2)
        )

        self.assertEqual(second["forecasted"], 0)
        self.assertEqual(second["orders_submitted"], 1)
        self.assertEqual(second["orders"][0]["book_side"], "bid")
        self.assertTrue(second["orders"][0]["reduce_only"])
        self.assertNotIn(self.market.ticker, self.broker.snapshot().positions)
        self.assertEqual(self.forecaster.calls, [self.market.ticker])

    def test_duplicate_cycles_cannot_submit_duplicate_forecasts(self) -> None:
        self.forecaster.result = make_forecast("0.42")
        first = self.engine.run_cycle(tickers=[self.market.ticker], now=NOW)
        self.assertEqual(first["forecasted"], 1)
        second = self.engine.run_cycle(tickers=[self.market.ticker], now=NOW)
        self.assertEqual(second["forecasted"], 0)
        self.assertEqual(self.forecaster.calls, [self.market.ticker])
        decision = self.store.recent_decisions(1)[0]
        self.assertIn("duplicate forecast key", decision["reason"])

    def test_timeout_is_unknown_and_never_issues_a_second_request(self) -> None:
        self.forecaster.exception = ForecastRequestTimeout(
            "deadline exceeded", provider_request_id="provider-timeout-1"
        )

        first = self.engine.run_cycle(tickers=[self.market.ticker], now=NOW)
        second = self.engine.run_cycle(
            tickers=[self.market.ticker], now=NOW + timedelta(hours=2)
        )

        self.assertEqual(len(self.forecaster.calls), 1)
        self.assertEqual(first["forecasted"], 0)
        self.assertEqual(second["forecasted"], 0)
        request = self.store.recent_forecast_requests(1)[0]
        self.assertEqual(request["status"], "unknown")
        self.assertEqual(request["provider_request_id"], "provider-timeout-1")

    def test_insufficient_quota_opens_circuit_immediately(self) -> None:
        self.forecaster.exception = ForecastQuotaExceeded("insufficient_quota")

        first = self.engine.run_cycle(tickers=[self.market.ticker], now=NOW)
        second = self.engine.run_cycle(
            tickers=[self.market.ticker], now=NOW + timedelta(hours=2)
        )

        self.assertTrue(first["circuit_opened"])
        self.assertEqual(self.store.forecast_circuit()["state"], "open")
        self.assertEqual(second["forecasted"], 0)
        self.assertEqual(len(self.forecaster.calls), 1)

    def test_daily_spend_limit_fails_closed_before_second_call(self) -> None:
        self.settings = replace(
            self.settings,
            daily_forecast_spend_limit=Decimal("0.10"),
            forecast_reserve_cost_dollars=Decimal("0.10"),
        )
        self.engine.settings = self.settings
        self.forecaster.result = make_forecast("0.42")

        first = self.engine.run_cycle(tickers=[self.market.ticker], now=NOW)
        second = self.engine.run_cycle(
            tickers=[self.market.ticker], now=NOW + timedelta(hours=2)
        )

        self.assertEqual(first["forecasted"], 1)
        self.assertEqual(second["forecasted"], 0)
        self.assertEqual(self.forecaster.calls, [self.market.ticker])
        self.assertTrue(
            any("daily forecast-spend limit" in error for error in second["errors"])
        )


if __name__ == "__main__":
    unittest.main()
