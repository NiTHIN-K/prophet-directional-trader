from __future__ import annotations

import argparse
import json
import logging
import os
import time
import uuid
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

from prophet_trader.config import Settings
from prophet_trader.engine import TraderEngine, discover_markets
from prophet_trader.events import EventWatcher
from prophet_trader.execution import LiveBroker, PaperBroker
from prophet_trader.forecast import GeminiForecaster
from prophet_trader.kalshi import KalshiClient
from prophet_trader.risk import RiskManager
from prophet_trader.store import StateStore
from prophet_trader.strategy import market_eligibility


LOGGER = logging.getLogger(__name__)


def _json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, default=str)


def _settings_for_command(settings: Settings, *, live: bool) -> Settings:
    if live and settings.strict_paper_replication:
        raise RuntimeError("live execution is disabled in strict paper-replication mode")
    if live and settings.trading_mode != "live":
        raise RuntimeError("live execution also requires TRADING_MODE=live")
    selected = replace(settings, trading_mode="live" if live else "paper")
    selected.validate()
    return selected


def _doctor(settings: Settings) -> dict[str, Any]:
    private_configured = bool(
        settings.kalshi_private_key_path or settings.kalshi_private_key_pem
    )
    private_validation = "missing"
    if private_configured:
        try:
            KalshiClient(settings).signature("0", "GET", "/trade-api/v2/exchange/status")
        except Exception as exc:
            private_validation = f"invalid ({type(exc).__name__})"
        else:
            private_validation = "valid"

    key_source = "missing"
    if os.getenv("KALSHI_API_KEY_ID"):
        key_source = "KALSHI_API_KEY_ID"
    elif os.getenv("KALSHI_API_KEY"):
        key_source = "KALSHI_API_KEY (legacy key-ID alias)"
    elif settings.kalshi_api_key_id:
        key_source = "configured programmatically"

    return {
        "configured_trading_mode": settings.trading_mode,
        "command_default_mode": "paper",
        "strict_paper_replication": settings.strict_paper_replication,
        "kalshi_environment": settings.kalshi_env,
        "kalshi_api_key_id_configured": bool(settings.kalshi_api_key_id),
        "kalshi_api_key_source": key_source,
        "kalshi_private_key_configured": private_configured,
        "kalshi_private_key_validation": private_validation,
        "gemini_api_key_configured": bool(settings.gemini_api_key),
        "gemini_model": settings.gemini_model,
        "gemini_reasoning_level": settings.gemini_reasoning_level,
        "gemini_max_output_tokens": settings.gemini_max_output_tokens,
        "forecasting_enabled": settings.forecasting_enabled,
        "scheduler_enabled": settings.scheduler_enabled,
        "daily_forecast_spend_limit": str(settings.daily_forecast_spend_limit),
        "live_trading_enabled": settings.live_trading_enabled,
        "production_enabled": settings.allow_production,
        "kill_switch_present": settings.kill_switch_path.exists(),
        "state_database": str(settings.state_db_path),
        "ready_for_paper_forecasting": bool(
            settings.gemini_api_key
            and settings.forecasting_enabled
            and settings.scheduler_enabled
            and settings.runtime_environment != "test"
        ),
        "ready_for_live": False if settings.strict_paper_replication else bool(
            settings.trading_mode == "live"
            and settings.gemini_api_key
            and settings.kalshi_api_key_id
            and private_validation == "valid"
            and settings.live_trading_enabled
            and not settings.kill_switch_path.exists()
        ),
    }


def _scan(settings: Settings, *, tickers: Sequence[str] | None, limit: int) -> dict[str, Any]:
    client = KalshiClient(settings)
    markets = discover_markets(client, tickers=tickers)
    rows: list[dict[str, Any]] = []
    rejection_reasons: Counter[str] = Counter()
    eligible_count = 0
    for market in markets:
        eligibility = market_eligibility(market, settings)
        if eligibility.eligible:
            eligible_count += 1
        else:
            rejection_reasons[eligibility.reason] += 1
        rows.append(
            {
                "ticker": market.ticker,
                "title": market.title,
                "category": market.category,
                "close_time": market.close_time.isoformat(),
                "expected_expiration_time": market.expected_expiration_time.isoformat(),
                "yes_bid": str(market.yes_bid),
                "yes_ask": str(market.yes_ask),
                "yes_bid_size": str(market.yes_bid_size),
                "yes_ask_size": str(market.yes_ask_size),
                "spread": str(market.spread),
                "buy_yes_depth_dollars": str(market.buy_yes_depth_dollars),
                "buy_no_depth_dollars": str(market.buy_no_depth_dollars),
                "executable_depth_dollars": str(market.executable_depth_dollars),
                "reported_liquidity_dollars_deprecated": str(market.liquidity_dollars),
                "eligible": eligibility.eligible,
                "reason": eligibility.reason,
            }
        )
    rows.sort(
        key=lambda item: (
            not item["eligible"],
            -float(item["executable_depth_dollars"]),
            item["ticker"],
        )
    )
    return {
        "discovered": len(markets),
        "eligible": eligible_count,
        "rejection_reasons": dict(rejection_reasons.most_common()),
        "returned": min(limit, len(rows)),
        "markets": rows[:limit],
    }


def _engine(settings: Settings) -> TraderEngine:
    store = StateStore(settings.state_db_path)
    client = KalshiClient(settings)
    risk = RiskManager(settings)
    broker = (
        LiveBroker(settings, store, client)
        if settings.trading_mode == "live"
        else PaperBroker(settings, store)
    )
    return TraderEngine(
        settings,
        client=client,
        forecaster=GeminiForecaster(settings),
        store=store,
        broker=broker,
        risk=risk,
    )


def _event_watcher(settings: Settings) -> EventWatcher:
    store = StateStore(settings.state_db_path)
    client = KalshiClient(settings)
    return EventWatcher(
        settings,
        client=client,
        store=store,
        broker=PaperBroker(settings, store),
    )


def _acquire_scheduler(settings: Settings, store: StateStore, owner_id: str) -> None:
    if settings.runtime_environment == "test":
        raise RuntimeError("test environments cannot start the scheduler")
    if not settings.scheduler_enabled:
        raise RuntimeError("scheduler is disabled; set SCHEDULER_ENABLED=true explicitly")
    lease_seconds = max(
        settings.cycle_seconds + 600,
        settings.request_timeout_seconds * settings.max_forecasts_per_cycle + 600,
    )
    if not store.acquire_scheduler_leader(
        name="strict-paper-forecaster",
        owner_id=owner_id,
        lease_seconds=lease_seconds,
    ):
        raise RuntimeError("another scheduler leader is already running")


def _status(settings: Settings, limit: int) -> dict[str, Any]:
    store = StateStore(settings.state_db_path)
    cycles = store.recent_cycles(limit)
    for cycle in cycles:
        raw_summary = cycle.get("summary_json")
        if raw_summary:
            try:
                cycle["summary"] = json.loads(raw_summary)
            except json.JSONDecodeError:
                cycle["summary"] = raw_summary
        cycle.pop("summary_json", None)
    return {
        "state_database": str(settings.state_db_path),
        "cycles": cycles,
        "recent_decisions": store.recent_decisions(limit * 5),
        "recent_context_events": store.recent_event_triggers(limit * 5),
        "forecast_circuit": store.forecast_circuit(),
        "daily_forecast_spend_dollars": str(store.daily_forecast_spend()),
        "recent_forecast_requests": store.recent_forecast_requests(limit * 5),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prophet-trader",
        description="Strict paper-replication Kalshi directional trader.",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("doctor", help="validate local configuration without network access")
    scan = subparsers.add_parser("scan", help="inspect market eligibility without forecasting")
    scan.add_argument("--ticker", action="append", dest="tickers")
    scan.add_argument("--limit", type=int, default=20)

    def add_run_arguments(command: argparse.ArgumentParser) -> None:
        command.add_argument("--ticker", action="append", dest="tickers")
        command.add_argument("--live", action="store_true", help="disabled in strict paper mode")
        command.add_argument("--confirm-live", action="store_true")

    once = subparsers.add_parser("run-once", help="run one fixed-slot forecast/rebalance cycle")
    add_run_arguments(once)
    watch_once = subparsers.add_parser(
        "watch-once",
        help="refresh price, source, and lifecycle context without forecasting",
    )
    add_run_arguments(watch_once)
    daemon = subparsers.add_parser("daemon", help="run on UTC-aligned two-hour slots")
    add_run_arguments(daemon)
    status = subparsers.add_parser("status", help="show recent local journal state")
    status.add_argument("--limit", type=int, default=10)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        settings = Settings.from_env(Path.cwd())
        if args.command == "doctor":
            print(_json(_doctor(settings)))
            return 0
        if args.command == "scan":
            if args.limit <= 0:
                parser.error("--limit must be positive")
            print(_json(_scan(settings, tickers=args.tickers, limit=args.limit)))
            return 0
        if args.command == "status":
            if args.limit <= 0:
                parser.error("--limit must be positive")
            print(_json(_status(settings, args.limit)))
            return 0

        if args.confirm_live and not args.live:
            parser.error("--confirm-live is only valid together with --live")
        settings = _settings_for_command(settings, live=args.live)
        if args.command == "watch-once":
            print(
                _json(
                    _event_watcher(settings).poll_once(
                        tickers=args.tickers,
                        confirm_live=args.confirm_live,
                    )
                )
            )
            return 0

        owner_id = str(uuid.uuid4())
        leader_store = StateStore(settings.state_db_path)
        _acquire_scheduler(settings, leader_store, owner_id)
        try:
            engine = _engine(settings)
        except Exception:
            leader_store.release_scheduler_leader(
                name="strict-paper-forecaster", owner_id=owner_id
            )
            raise
        if args.command == "run-once":
            try:
                print(
                    _json(
                        engine.run_cycle(
                            tickers=args.tickers,
                            confirm_live=args.confirm_live,
                        )
                    )
                )
            finally:
                leader_store.release_scheduler_leader(
                    name="strict-paper-forecaster", owner_id=owner_id
                )
            return 0

        if args.command == "daemon":
            try:
                while True:
                    _acquire_scheduler(settings, leader_store, owner_id)
                    result = engine.run_cycle(
                        tickers=args.tickers,
                        confirm_live=args.confirm_live,
                    )
                    print(_json(result), flush=True)
                    remaining = max(
                        1.0,
                        settings.cycle_seconds - (time.time() % settings.cycle_seconds),
                    )
                    LOGGER.info("next UTC-aligned cycle in %.0f seconds", remaining)
                    time.sleep(remaining)
            finally:
                leader_store.release_scheduler_leader(
                    name="strict-paper-forecaster", owner_id=owner_id
                )
    except KeyboardInterrupt:
        LOGGER.info("stopped")
        return 130
    except Exception as exc:
        LOGGER.error("%s: %s", type(exc).__name__, exc)
        return 1
    return 0
