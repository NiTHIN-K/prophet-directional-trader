from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from unittest.mock import Mock, patch

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from prophet_trader.cli import _doctor, build_parser, main  # noqa: E402
from prophet_trader.config import Settings  # noqa: E402
from prophet_trader.store import StateStore  # noqa: E402


def make_settings(root: Path, **overrides: object) -> Settings:
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
        gemini_model="gemini-3.1-pro-preview",
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
        max_target_contracts=Decimal("50"),
        max_order_contracts=Decimal("25"),
        max_market_risk_dollars=Decimal("25"),
        max_total_exposure_dollars=Decimal("100"),
        max_spread=Decimal("0.15"),
        min_liquidity_dollars=Decimal("1"),
        min_evidence_sources=2,
        block_high_uncertainty=True,
        state_db_path=root / "state" / "trader.sqlite3",
        kill_switch_path=root / "STOP",
        request_timeout_seconds=30,
        max_http_retries=3,
        forecasting_enabled=False,
        scheduler_enabled=False,
        runtime_environment="test",
    )
    return replace(settings, **overrides)


class DoctorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        cls.private_key_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        ).decode("ascii")

    def test_doctor_reports_readiness_without_printing_credentials(self) -> None:
        key_id = "kalshi-key-id-that-must-not-leak"
        gemini_key = "gemini-key-that-must-not-leak"
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            settings = make_settings(
                root,
                kalshi_api_key_id=key_id,
                kalshi_private_key_pem=self.private_key_pem,
                gemini_api_key=gemini_key,
                forecasting_enabled=True,
                scheduler_enabled=True,
                runtime_environment="production",
            )
            output = io.StringIO()
            with (
                patch.dict(
                    os.environ,
                    {"KALSHI_API_KEY": key_id, "GEMINI_API_KEY": gemini_key},
                    clear=True,
                ),
                patch.object(Settings, "from_env", return_value=settings),
                redirect_stdout(output),
            ):
                exit_code = main(["doctor"])

        rendered = output.getvalue()
        payload = json.loads(rendered)
        self.assertEqual(exit_code, 0)
        self.assertTrue(payload["kalshi_api_key_id_configured"])
        self.assertEqual(
            payload["kalshi_api_key_source"],
            "KALSHI_API_KEY (legacy key-ID alias)",
        )
        self.assertTrue(payload["kalshi_private_key_configured"])
        self.assertEqual(payload["kalshi_private_key_validation"], "valid")
        self.assertTrue(payload["gemini_api_key_configured"])
        self.assertTrue(payload["ready_for_paper_forecasting"])
        self.assertFalse(payload["ready_for_live"])
        self.assertEqual(payload["command_default_mode"], "paper")
        self.assertNotIn(key_id, rendered)
        self.assertNotIn(gemini_key, rendered)
        self.assertNotIn("BEGIN PRIVATE KEY", rendered)

    def test_doctor_readiness_is_false_when_required_material_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            settings = make_settings(
                Path(temporary_directory),
                kalshi_api_key_id="configured-key-id",
            )
            payload = _doctor(settings)

        self.assertFalse(payload["kalshi_private_key_configured"])
        self.assertEqual(payload["kalshi_private_key_validation"], "missing")
        self.assertFalse(payload["ready_for_paper_forecasting"])
        self.assertFalse(payload["ready_for_live"])

    def test_production_live_readiness_requires_the_production_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            settings = make_settings(
                Path(temporary_directory),
                kalshi_env="production",
                allow_production=False,
                kalshi_api_key_id="configured-key-id",
                kalshi_private_key_pem=self.private_key_pem,
                gemini_api_key="configured-gemini-key",
            )

            payload = _doctor(settings)

        self.assertFalse(payload["ready_for_live"])


class ParserAndCommandSafetyTests(unittest.TestCase):
    def test_run_parser_defaults_to_paper_and_accepts_both_live_acknowledgements(self) -> None:
        parser = build_parser()

        paper_args = parser.parse_args(["run-once"])
        watch_args = parser.parse_args(["watch-once", "--ticker", "KXWATCH-26"])
        live_args = parser.parse_args(
            ["daemon", "--ticker", "KXTEST-26", "--live", "--confirm-live"]
        )

        self.assertFalse(paper_args.live)
        self.assertFalse(paper_args.confirm_live)
        self.assertFalse(watch_args.live)
        self.assertEqual(watch_args.tickers, ["KXWATCH-26"])
        self.assertTrue(live_args.live)
        self.assertTrue(live_args.confirm_live)
        self.assertEqual(live_args.tickers, ["KXTEST-26"])

    def test_test_environment_cannot_start_scheduler(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            settings = make_settings(root, runtime_environment="test")
            with (
                patch.object(Settings, "from_env", return_value=settings),
                patch("prophet_trader.cli._engine") as engine_factory,
            ):
                exit_code = main(["run-once"])

        self.assertEqual(exit_code, 1)
        engine_factory.assert_not_called()


class StatusTests(unittest.TestCase):
    def test_status_reads_and_decodes_local_journal_without_network_clients(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            settings = make_settings(root)
            store = StateStore(settings.state_db_path)
            store.start_cycle("cycle-1", "paper")
            store.finish_cycle(
                "cycle-1",
                "completed",
                {"forecasted": 3, "orders_submitted": 1},
            )

            output = io.StringIO()
            with (
                patch.object(Settings, "from_env", return_value=settings),
                patch("prophet_trader.cli.KalshiClient") as kalshi_client,
                patch("prophet_trader.cli.GeminiForecaster") as forecaster,
                redirect_stdout(output),
            ):
                exit_code = main(["status", "--limit", "2"])

            payload = json.loads(output.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["state_database"], str(settings.state_db_path))
        self.assertEqual(len(payload["cycles"]), 1)
        self.assertEqual(payload["cycles"][0]["cycle_id"], "cycle-1")
        self.assertEqual(payload["cycles"][0]["status"], "completed")
        self.assertEqual(
            payload["cycles"][0]["summary"],
            {"forecasted": 3, "orders_submitted": 1},
        )
        self.assertNotIn("summary_json", payload["cycles"][0])
        self.assertEqual(payload["recent_decisions"], [])
        kalshi_client.assert_not_called()
        forecaster.assert_not_called()


if __name__ == "__main__":
    unittest.main()
