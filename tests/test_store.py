from __future__ import annotations

import json
import sys
import tempfile
import unittest
from dataclasses import replace
from decimal import Decimal
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from prophet_trader.execution import PaperBroker
from prophet_trader.models import OrderIntent
from prophet_trader.store import StateStore
from tests.test_strategy import make_market, make_settings


class PaperBrokerStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self.settings = replace(
            make_settings(),
            root=self.root,
            state_db_path=self.root / "state" / "trader.sqlite3",
            kill_switch_path=self.root / "STOP",
            paper_starting_cash=Decimal("200"),
        )
        self.store = StateStore(self.settings.state_db_path)
        self.broker = PaperBroker(self.settings, self.store)
        self.market = make_market()

    def intent(
        self,
        *,
        book_side: str,
        count: str,
        current_position: str,
        target_position: str,
        reduce_only: bool,
    ) -> OrderIntent:
        yes_price = self.market.yes_ask if book_side == "bid" else self.market.yes_bid
        opening_risk = (
            Decimal("0")
            if reduce_only
            else Decimal(count)
            * (yes_price if book_side == "bid" else Decimal("1") - yes_price)
        )
        return OrderIntent(
            ticker=self.market.ticker,
            book_side=book_side,
            count=Decimal(count),
            yes_price=yes_price,
            current_position=Decimal(current_position),
            target_position=Decimal(target_position),
            opening_risk_dollars=opening_risk,
            reduce_only=reduce_only,
            reason="offline test order",
        )

    def test_paper_fill_and_cycle_journal_survive_store_reopen(self) -> None:
        cycle_id = "paper-cycle-1"
        self.store.start_cycle(cycle_id, "paper")

        result = self.broker.execute(
            cycle_id=cycle_id,
            market=self.market,
            intent=self.intent(
                book_side="bid",
                count="3",
                current_position="0",
                target_position="3",
                reduce_only=False,
            ),
        )
        summary = {"orders_submitted": 1, "orders": [result]}
        self.store.finish_cycle(cycle_id, "completed", summary)

        self.assertTrue(result["simulated"])
        self.assertEqual(result["fill_count"], "3")
        self.assertEqual(result["position"], "3")

        snapshot = self.broker.snapshot()
        self.assertEqual(snapshot.positions[self.market.ticker].contracts, Decimal("3"))
        self.assertEqual(snapshot.total_exposure_dollars, Decimal("1.32"))
        self.assertEqual(snapshot.available_cash_dollars, Decimal("198.68"))
        self.assertEqual(
            self.store.get_last_fill_reference(self.market.ticker),
            self.market.yes_ask,
        )
        self.assertEqual(self.store.open_live_orders(), [])

        reopened = StateStore(self.settings.state_db_path)
        reopened_snapshot = PaperBroker(self.settings, reopened).snapshot()
        self.assertEqual(
            reopened_snapshot.positions[self.market.ticker].contracts,
            Decimal("3"),
        )
        cycle = reopened.recent_cycles(1)[0]
        self.assertEqual(cycle["cycle_id"], cycle_id)
        self.assertEqual(cycle["mode"], "paper")
        self.assertEqual(cycle["status"], "completed")
        self.assertEqual(json.loads(cycle["summary_json"])["orders_submitted"], 1)

        with reopened.connection() as connection:
            order = connection.execute(
                "SELECT mode, status, fill_count, remaining_count, response_json "
                "FROM orders WHERE cycle_id=?",
                (cycle_id,),
            ).fetchone()
        self.assertIsNotNone(order)
        assert order is not None
        self.assertEqual(order["mode"], "paper")
        self.assertEqual(order["status"], "filled")
        self.assertEqual(Decimal(order["fill_count"]), Decimal("3"))
        self.assertEqual(Decimal(order["remaining_count"]), Decimal("0"))
        self.assertTrue(json.loads(order["response_json"])["simulated"])

    def test_paper_reduce_only_fill_flattens_position_and_releases_risk(self) -> None:
        cycle_id = "paper-cycle-flat"
        self.store.start_cycle(cycle_id, "paper")
        self.broker.execute(
            cycle_id=cycle_id,
            market=self.market,
            intent=self.intent(
                book_side="bid",
                count="3",
                current_position="0",
                target_position="3",
                reduce_only=False,
            ),
        )

        result = self.broker.execute(
            cycle_id=cycle_id,
            market=self.market,
            intent=self.intent(
                book_side="ask",
                count="3",
                current_position="3",
                target_position="0",
                reduce_only=True,
            ),
        )

        self.assertEqual(result["position"], "0")
        snapshot = self.broker.snapshot()
        self.assertEqual(snapshot.positions, {})
        self.assertEqual(snapshot.total_exposure_dollars, Decimal("0"))
        self.assertEqual(snapshot.available_cash_dollars, Decimal("199.88"))
        with self.store.connection() as connection:
            self.assertEqual(connection.execute("PRAGMA foreign_keys").fetchone()[0], 1)
            order_count = connection.execute(
                "SELECT COUNT(*) AS count FROM orders WHERE cycle_id=? AND mode='paper'",
                (cycle_id,),
            ).fetchone()["count"]
        self.assertEqual(order_count, 2)


if __name__ == "__main__":
    unittest.main()
