from __future__ import annotations

import sys
import tempfile
import unittest
from dataclasses import replace
from datetime import timedelta
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from prophet_trader.events import (  # noqa: E402
    EventWatcher,
    OfficialSourceFetcher,
    SourceSnapshot,
)
from prophet_trader.store import StateStore  # noqa: E402
from tests.test_engine import market_api_payload  # noqa: E402
from tests.test_strategy import NOW, make_market, make_settings  # noqa: E402


class OfflineEventClient:
    def __init__(self, market: Any) -> None:
        self.market = market

    def get_market(self, ticker: str) -> dict[str, Any]:
        if ticker != self.market.ticker:
            raise AssertionError(f"unexpected ticker: {ticker}")
        return {"market": market_api_payload(self.market)}

    def get_event(self, event_ticker: str) -> dict[str, Any]:
        if event_ticker != self.market.event_ticker:
            raise AssertionError(f"unexpected event: {event_ticker}")
        return {
            "event": {
                "event_ticker": self.market.event_ticker,
                "series_ticker": self.market.series_ticker,
                "category": self.market.category,
                "title": self.market.title,
                "settlement_sources": list(self.market.settlement_sources),
            }
        }

    def get_event_metadata(self, event_ticker: str) -> dict[str, Any]:
        if event_ticker != self.market.event_ticker:
            raise AssertionError(f"unexpected event metadata: {event_ticker}")
        return {"settlement_sources": list(self.market.settlement_sources)}

    def list_open_events(self, *, max_pages: int = 5) -> list[dict[str, Any]]:
        raise AssertionError("explicit-ticker watcher test must not scan all events")


class OfflineBroker:
    def snapshot(self) -> Any:
        return SimpleNamespace(positions={})


class SequenceSourceFetcher:
    def __init__(self, fingerprints: list[str]) -> None:
        self.fingerprints = iter(fingerprints)
        self.calls: list[str] = []

    def fetch(self, url: str) -> SourceSnapshot:
        self.calls.append(url)
        return SourceSnapshot(
            url=url,
            fingerprint=next(self.fingerprints),
            etag=None,
            last_modified=None,
            status_code=200,
        )


class EventWatcherTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        root = Path(self.temporary_directory.name)
        self.settings = replace(
            make_settings(),
            root=root,
            state_db_path=root / "state" / "events.sqlite3",
            official_source_poll_seconds=60,
            event_trigger_cooldown_seconds=300,
            max_event_triggers_per_poll=2,
        )
        self.market = make_market(
            settlement_sources=(
                {"name": "Official release", "url": "https://official.example/release"},
            )
        )
        self.client = OfflineEventClient(self.market)
        self.store = StateStore(self.settings.state_db_path)

    def watcher(self, fingerprints: list[str]) -> EventWatcher:
        return EventWatcher(
            self.settings,
            client=self.client,
            store=self.store,
            broker=OfflineBroker(),
            source_fetcher=SequenceSourceFetcher(fingerprints),  # type: ignore[arg-type]
        )

    def test_first_observation_only_seeds_baselines(self) -> None:
        summary = self.watcher(["source-v1"]).poll_once(
            tickers=[self.market.ticker],
            now=NOW,
        )

        self.assertEqual(summary["monitored"], 1)
        self.assertEqual(summary["baselined"], 1)
        self.assertEqual(summary["detected"], 0)
        self.assertEqual(summary["context_updates"], [])
        self.assertEqual(summary["forecasting_invocations"], 0)
        state = self.store.event_watch_state(self.market.ticker)
        self.assertIsNotNone(state)
        assert state is not None
        self.assertEqual(state["price_reference"], "0.42")

    def test_three_cent_midpoint_move_updates_context_without_forecast(self) -> None:
        watcher = self.watcher(["source-v1"])
        watcher.poll_once(tickers=[self.market.ticker], now=NOW)
        self.client.market = replace(
            self.market,
            yes_bid=self.market.yes_bid + self.settings.event_price_move_threshold,
            yes_ask=self.market.yes_ask + self.settings.event_price_move_threshold,
        )

        summary = watcher.poll_once(
            tickers=[self.market.ticker],
            now=NOW + timedelta(seconds=30),
        )

        self.assertEqual(summary["detected"], 1)
        self.assertEqual(summary["context_updates"][0]["types"], ["price_movement_3c"])
        self.assertEqual(summary["forecasting_invocations"], 0)
        with self.store.connection() as connection:
            self.assertEqual(
                connection.execute("SELECT COUNT(*) FROM forecast_requests").fetchone()[0],
                0,
            )

    def test_market_metadata_change_updates_context_without_forecast(self) -> None:
        watcher = self.watcher(["source-v1"])
        watcher.poll_once(tickers=[self.market.ticker], now=NOW)
        self.client.market = replace(self.market, rules_primary="Updated official market rule")

        summary = watcher.poll_once(
            tickers=[self.market.ticker],
            now=NOW + timedelta(seconds=30),
        )

        self.assertEqual(summary["detected"], 1)
        self.assertEqual(
            summary["context_updates"][0]["types"],
            ["market_metadata_lifecycle_change"],
        )

    def test_inactive_terminal_book_does_not_replace_price_reference(self) -> None:
        watcher = self.watcher(["source-v1"])
        watcher.poll_once(tickers=[self.market.ticker], now=NOW)
        self.client.market = replace(
            self.market,
            status="finalized",
            yes_bid=Decimal("0"),
            yes_ask=Decimal("1"),
            no_bid=Decimal("0"),
            no_ask=Decimal("1"),
            yes_bid_size=Decimal("0"),
            yes_ask_size=Decimal("0"),
        )

        summary = watcher.poll_once(
            tickers=[self.market.ticker],
            now=NOW + timedelta(seconds=30),
        )

        self.assertEqual(summary["detected"], 1)
        self.assertEqual(
            summary["context_updates"][0]["types"],
            ["market_metadata_lifecycle_change"],
        )
        state = self.store.event_watch_state(self.market.ticker)
        assert state is not None
        self.assertEqual(state["price_reference"], "0.42")

    def test_official_source_change_updates_context_without_forecast(self) -> None:
        watcher = self.watcher(["source-v1", "source-v2"])
        watcher.poll_once(tickers=[self.market.ticker], now=NOW)

        summary = watcher.poll_once(
            tickers=[self.market.ticker],
            now=NOW + timedelta(seconds=61),
        )

        self.assertEqual(summary["detected"], 1)
        self.assertEqual(
            summary["context_updates"][0]["types"],
            ["official_source_or_scheduled_release_update"],
        )

    def test_public_source_url_validation_rejects_local_targets(self) -> None:
        for url in (
            "file:///etc/passwd",
            "http://localhost/report",
            "http://127.0.0.1/report",
            "http://10.0.0.5/report",
        ):
            with self.subTest(url=url), self.assertRaises(ValueError):
                OfficialSourceFetcher.validate_url(url)

    def test_generic_media_homepage_is_not_treated_as_official_source(self) -> None:
        self.client.market = replace(
            self.market,
            settlement_sources=(
                {"name": "Reuters", "url": "https://www.reuters.com/"},
            ),
        )
        fetcher = SequenceSourceFetcher(["must-not-be-used"])
        watcher = EventWatcher(
            self.settings,
            client=self.client,
            store=self.store,
            broker=OfflineBroker(),
            source_fetcher=fetcher,  # type: ignore[arg-type]
        )

        summary = watcher.poll_once(tickers=[self.market.ticker], now=NOW)

        self.assertEqual(summary["baselined"], 1)
        self.assertEqual(fetcher.calls, [])


if __name__ == "__main__":
    unittest.main()
