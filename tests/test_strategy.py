from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from prophet_trader.config import Settings
from prophet_trader.models import Direction, Evidence, Forecast, Market, Position
from prophet_trader.strategy import (
    brier_signal,
    build_order_intent,
    forecast_eligibility,
    market_eligibility,
    should_reforecast,
)


NOW = datetime(2026, 7, 22, 12, 0, tzinfo=timezone.utc)


def make_settings(**overrides: object) -> Settings:
    root = Path("/tmp/prophet-trader-tests")
    settings = Settings(
        root=root,
        trading_mode="paper",
        kalshi_env="demo",
        live_trading_enabled=False,
        allow_production=False,
        kalshi_api_key_id=None,
        kalshi_private_key_path=None,
        kalshi_private_key_pem=None,
        gemini_api_key=None,
        gemini_model="test-model",
        gemini_reasoning_level="high",
        cycle_seconds=7200,
        min_days_to_close=Decimal("2"),
        max_days_to_close=Decimal("14"),
        stop_hours_before_resolution=Decimal("3"),
        max_close_resolution_gap_hours=Decimal("1"),
        refresh_move_threshold=Decimal("0.10"),
        position_scale_contracts=Decimal("100"),
        min_actionable_edge=Decimal("0.02"),
        max_markets_per_cycle=8,
        paper_starting_cash=Decimal("200"),
        max_target_contracts=Decimal("1000"),
        max_order_contracts=Decimal("1000"),
        max_market_risk_dollars=Decimal("1000"),
        max_total_exposure_dollars=Decimal("1000"),
        max_spread=Decimal("0.15"),
        min_liquidity_dollars=Decimal("1"),
        min_evidence_sources=2,
        block_high_uncertainty=True,
        state_db_path=root / "state.sqlite3",
        kill_switch_path=root / "STOP",
        request_timeout_seconds=30,
        max_http_retries=3,
        forecasting_enabled=False,
        scheduler_enabled=False,
        runtime_environment="test",
    )
    return replace(settings, **overrides)


def make_market(**overrides: object) -> Market:
    close_time = NOW + timedelta(days=7)
    market = Market(
        ticker="TEST-MARKET",
        event_ticker="TEST-EVENT",
        series_ticker="TEST-SERIES",
        category="Economics",
        title="Will the test event occur?",
        subtitle="",
        yes_sub_title="Yes",
        no_sub_title="No",
        rules_primary="Resolves YES if the specified event occurs.",
        rules_secondary="",
        settlement_sources=({"name": "Official source", "url": "https://example.test"},),
        close_time=close_time,
        expected_expiration_time=close_time,
        yes_bid=Decimal("0.40"),
        yes_ask=Decimal("0.44"),
        no_bid=Decimal("0.56"),
        no_ask=Decimal("0.60"),
        yes_bid_size=Decimal("100"),
        yes_ask_size=Decimal("100"),
        liquidity_dollars=Decimal("1000"),
        volume=Decimal("10000"),
        open_interest=Decimal("500"),
        can_close_early=False,
        early_close_condition="",
        status="open",
    )
    return replace(market, **overrides)


def make_forecast(
    probability_yes: str,
    *,
    resolution_clear: bool = True,
    uncertainty: str = "medium",
    evidence: tuple[Evidence, ...] | None = None,
) -> Forecast:
    if evidence is None:
        evidence = (
            Evidence("Official observation", "https://one.example/report"),
            Evidence("Independent observation", "https://two.example/report"),
        )
    return Forecast(
        probability_yes=Decimal(probability_yes),
        rationale="A test forecast grounded in contemporaneous evidence.",
        resolution_clear=resolution_clear,
        resolution_clarity_reason="clear" if resolution_clear else "ambiguous rule",
        uncertainty=uncertainty,
        evidence=evidence,
        data_as_of=NOW.isoformat(),
        model="test-model",
    )


class BrierSignalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = make_settings()
        self.market = make_market()

    def test_yes_target_is_scaled_probability_edge_over_yes_ask(self) -> None:
        signal = brier_signal(self.market, make_forecast("0.70"), self.settings)

        self.assertIs(signal.direction, Direction.BUY_YES)
        self.assertEqual(signal.actionable_edge, Decimal("0.26"))
        self.assertEqual(signal.raw_target_contracts, Decimal("26"))
        self.assertEqual(signal.target_position, Decimal("26"))
        self.assertEqual(signal.execution_yes_price, Decimal("0.44"))
        self.assertEqual(signal.outcome_cost, Decimal("0.44"))

    def test_no_target_is_negative_and_uses_yes_bid_boundary(self) -> None:
        signal = brier_signal(self.market, make_forecast("0.20"), self.settings)

        self.assertIs(signal.direction, Direction.BUY_NO)
        self.assertEqual(signal.actionable_edge, Decimal("0.20"))
        self.assertEqual(signal.raw_target_contracts, Decimal("-20"))
        self.assertEqual(signal.target_position, Decimal("-20"))
        self.assertEqual(signal.execution_yes_price, Decimal("0.40"))
        self.assertEqual(signal.outcome_cost, Decimal("0.60"))

    def test_forecast_inside_bid_ask_band_has_flat_target(self) -> None:
        signal = brier_signal(self.market, make_forecast("0.42"), self.settings)

        self.assertIs(signal.direction, Direction.FLAT)
        self.assertEqual(signal.actionable_edge, Decimal("0"))
        self.assertEqual(signal.raw_target_contracts, Decimal("0"))
        self.assertEqual(signal.target_position, Decimal("0"))
        self.assertIsNone(signal.execution_yes_price)
        self.assertIsNone(signal.outcome_cost)

    def test_edge_outside_spread_but_below_buffer_has_flat_target(self) -> None:
        signal = brier_signal(self.market, make_forecast("0.39"), self.settings)

        self.assertIs(signal.direction, Direction.FLAT)
        self.assertEqual(signal.actionable_edge, Decimal("0"))
        self.assertEqual(signal.target_position, Decimal("0"))


class RebalancingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = make_settings()
        self.market = make_market()

    def intent_for(self, probability_yes: str, actual_position: str):
        signal = brier_signal(self.market, make_forecast(probability_yes), self.settings)
        return build_order_intent(
            self.market,
            signal,
            Position(self.market.ticker, Decimal(actual_position)),
            self.settings,
            total_exposure_dollars=Decimal("0"),
            available_cash_dollars=Decimal("1000"),
        )

    def test_rebalance_delta_is_computed_from_actual_filled_position(self) -> None:
        intent = self.intent_for("0.70", "7")

        self.assertIsNotNone(intent)
        assert intent is not None
        self.assertEqual(intent.current_position, Decimal("7"))
        self.assertEqual(intent.target_position, Decimal("26"))
        self.assertEqual(intent.book_side, "bid")
        self.assertEqual(intent.count, Decimal("19"))
        self.assertEqual(intent.yes_price, Decimal("0.44"))
        self.assertFalse(intent.reduce_only)

    def test_rebalance_is_capped_by_displayed_top_of_book_size(self) -> None:
        shallow_market = replace(self.market, yes_ask_size=Decimal("4"))
        signal = brier_signal(shallow_market, make_forecast("0.70"), self.settings)

        intent = build_order_intent(
            shallow_market,
            signal,
            Position(shallow_market.ticker, Decimal("0")),
            self.settings,
            total_exposure_dollars=Decimal("0"),
            available_cash_dollars=Decimal("1000"),
        )

        self.assertIsNotNone(intent)
        assert intent is not None
        self.assertEqual(intent.count, Decimal("4"))

    def test_rebalance_reduces_same_side_position_without_opening_risk(self) -> None:
        signal = brier_signal(self.market, make_forecast("0.54"), self.settings)
        intent = build_order_intent(
            self.market,
            signal,
            Position(self.market.ticker, Decimal("26")),
            self.settings,
            total_exposure_dollars=self.settings.max_total_exposure_dollars,
            available_cash_dollars=Decimal("0"),
        )

        self.assertIsNotNone(intent)
        assert intent is not None
        self.assertEqual(intent.target_position, Decimal("10"))
        self.assertEqual(intent.book_side, "ask")
        self.assertEqual(intent.count, Decimal("16"))
        self.assertEqual(intent.opening_risk_dollars, Decimal("0"))
        self.assertTrue(intent.reduce_only)

    def test_rebalance_reverses_from_yes_to_no(self) -> None:
        intent = self.intent_for("0.20", "8")

        self.assertIsNotNone(intent)
        assert intent is not None
        self.assertEqual(intent.current_position, Decimal("8"))
        self.assertEqual(intent.target_position, Decimal("-20"))
        self.assertEqual(intent.book_side, "ask")
        self.assertEqual(intent.count, Decimal("28"))
        self.assertEqual(intent.opening_risk_dollars, Decimal("12.00"))
        self.assertFalse(intent.reduce_only)

    def test_rebalance_reverses_from_no_to_yes(self) -> None:
        intent = self.intent_for("0.70", "-5")

        self.assertIsNotNone(intent)
        assert intent is not None
        self.assertEqual(intent.current_position, Decimal("-5"))
        self.assertEqual(intent.target_position, Decimal("26"))
        self.assertEqual(intent.book_side, "bid")
        self.assertEqual(intent.count, Decimal("31"))
        self.assertEqual(intent.opening_risk_dollars, Decimal("11.44"))
        self.assertFalse(intent.reduce_only)


class ReforecastTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = make_settings()
        self.market = make_market()

    def test_never_filled_market_is_always_reforecast(self) -> None:
        self.assertTrue(should_reforecast(self.market, None, self.settings))

    def test_move_below_ten_cent_threshold_does_not_reforecast(self) -> None:
        self.assertFalse(
            should_reforecast(self.market, Decimal("0.321"), self.settings)
        )

    def test_move_at_ten_cent_threshold_reforecasts(self) -> None:
        self.assertTrue(should_reforecast(self.market, Decimal("0.32"), self.settings))

    def test_threshold_is_direction_agnostic(self) -> None:
        self.assertTrue(should_reforecast(self.market, Decimal("0.53"), self.settings))


class MarketEligibilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = make_settings()

    def test_baseline_market_is_eligible(self) -> None:
        result = market_eligibility(make_market(), self.settings, now=NOW)
        self.assertTrue(result.eligible, result.reason)

    def test_deprecated_zero_liquidity_metadata_does_not_block_displayed_depth(self) -> None:
        market = make_market(liquidity_dollars=Decimal("0"))

        result = market_eligibility(market, self.settings, now=NOW)

        self.assertTrue(result.eligible, result.reason)
        self.assertEqual(market.buy_yes_depth_dollars, Decimal("44.00"))
        self.assertEqual(market.buy_no_depth_dollars, Decimal("60.00"))
        self.assertEqual(market.executable_depth_dollars, Decimal("44.00"))

    def test_paper_market_exclusions(self) -> None:
        cases = {
            "non-open market": make_market(status="closed"),
            "MENTIONS category": make_market(category="  MeNtIoNs  "),
            "inside resolution cutoff": make_market(
                close_time=NOW + timedelta(hours=3),
                expected_expiration_time=NOW + timedelta(hours=3),
            ),
            "close-resolution mismatch": make_market(
                close_time=NOW + timedelta(days=7, hours=2),
                expected_expiration_time=NOW + timedelta(days=7),
            ),
            "missing primary rules": make_market(rules_primary="  "),
            "unspecified early close": make_market(
                can_close_early=True,
                early_close_condition="",
            ),
        }

        for name, market in cases.items():
            with self.subTest(name=name):
                result = market_eligibility(market, self.settings, now=NOW)
                self.assertFalse(result.eligible, result.reason)

    def test_discovery_horizon_includes_two_and_fourteen_day_boundaries(self) -> None:
        for days in (2, 14):
            with self.subTest(days=days):
                close = NOW + timedelta(days=days)
                result = market_eligibility(
                    make_market(close_time=close, expected_expiration_time=close),
                    self.settings,
                    now=NOW,
                )
                self.assertTrue(result.eligible, result.reason)

    def test_discovery_horizon_rejects_markets_outside_two_to_fourteen_days(self) -> None:
        for delta in (timedelta(days=2, seconds=-1), timedelta(days=14, seconds=1)):
            with self.subTest(delta=delta):
                close = NOW + delta
                result = market_eligibility(
                    make_market(close_time=close, expected_expiration_time=close),
                    self.settings,
                    now=NOW,
                )
                self.assertFalse(result.eligible, result.reason)

    def test_tracked_market_can_skip_discovery_horizon_but_not_resolution_cutoff(self) -> None:
        close = NOW + timedelta(days=1)
        tracked = make_market(close_time=close, expected_expiration_time=close)

        result = market_eligibility(
            tracked,
            self.settings,
            now=NOW,
            enforce_discovery_horizon=False,
        )
        self.assertTrue(result.eligible, result.reason)

        cutoff = replace(
            tracked,
            close_time=NOW + timedelta(hours=3),
            expected_expiration_time=NOW + timedelta(hours=3),
        )
        result = market_eligibility(
            cutoff,
            self.settings,
            now=NOW,
            enforce_discovery_horizon=False,
        )
        self.assertFalse(result.eligible, result.reason)

    def test_engineering_spread_and_liquidity_guards_fail_closed(self) -> None:
        cases = {
            "wide spread": make_market(yes_bid=Decimal("0.20"), yes_ask=Decimal("0.40")),
            "shallow YES entry depth": make_market(yes_ask_size=Decimal("2")),
            "shallow NO entry depth": make_market(yes_bid_size=Decimal("1")),
            "invalid top of book": make_market(yes_bid=Decimal("0")),
            "missing bid size": make_market(yes_bid_size=Decimal("0")),
            "non-finite ask": make_market(yes_ask=Decimal("NaN")),
            "non-finite ask size": make_market(yes_ask_size=Decimal("NaN")),
        }
        for name, market in cases.items():
            with self.subTest(name=name):
                result = market_eligibility(market, self.settings, now=NOW)
                self.assertFalse(result.eligible, result.reason)


class ForecastEligibilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = make_settings()

    def test_clear_forecast_with_distinct_sources_is_eligible(self) -> None:
        result = forecast_eligibility(make_forecast("0.60"), self.settings)
        self.assertTrue(result.eligible, result.reason)

    def test_ambiguous_resolution_is_rejected(self) -> None:
        result = forecast_eligibility(
            make_forecast("0.60", resolution_clear=False),
            self.settings,
        )
        self.assertFalse(result.eligible, result.reason)

    def test_duplicate_source_urls_do_not_satisfy_evidence_requirement(self) -> None:
        evidence = (
            Evidence("First claim", "https://same.example/report"),
            Evidence("Second claim", "https://same.example/report"),
        )
        result = forecast_eligibility(
            make_forecast("0.60", evidence=evidence),
            self.settings,
        )
        self.assertFalse(result.eligible, result.reason)

    def test_high_uncertainty_is_rejected_when_guard_is_enabled(self) -> None:
        result = forecast_eligibility(
            make_forecast("0.60", uncertainty="high"),
            self.settings,
        )
        self.assertFalse(result.eligible, result.reason)


if __name__ == "__main__":
    unittest.main()
