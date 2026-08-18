from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from prophet_trader.config import Settings
from prophet_trader.models import OrderIntent, ZERO


@dataclass(frozen=True)
class RiskDecision:
    allowed: bool
    reason: str


class RiskManager:
    """Fail-closed pre-trade checks that are engineering additions to the paper."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def check_kill_switch(self) -> RiskDecision:
        path: Path = self.settings.kill_switch_path
        if path.exists():
            return RiskDecision(False, f"kill switch is present at {path}")
        return RiskDecision(True, "kill switch clear")

    def check_order(
        self,
        intent: OrderIntent,
        *,
        total_exposure_dollars: Decimal,
        available_cash_dollars: Decimal,
    ) -> RiskDecision:
        kill = self.check_kill_switch()
        if not kill.allowed:
            return kill
        if intent.count > self.settings.max_order_contracts:
            return RiskDecision(False, "order exceeds MAX_ORDER_CONTRACTS")
        if intent.opening_risk_dollars > self.settings.max_market_risk_dollars:
            return RiskDecision(False, "order exceeds MAX_MARKET_RISK_DOLLARS")
        if intent.opening_risk_dollars > ZERO:
            projected = total_exposure_dollars + intent.opening_risk_dollars
            if projected > self.settings.max_total_exposure_dollars:
                return RiskDecision(False, "order exceeds MAX_TOTAL_EXPOSURE_DOLLARS")
            if intent.opening_risk_dollars > available_cash_dollars:
                return RiskDecision(False, "insufficient available cash")
        return RiskDecision(True, "pre-trade checks passed")

    def assert_live_enabled(self, *, confirm_live: bool) -> None:
        if not confirm_live:
            raise RuntimeError("live execution requires the --confirm-live command-line gate")
        if not self.settings.live_trading_enabled:
            raise RuntimeError("live execution requires LIVE_TRADING_ENABLED=true")
        if not self.settings.has_kalshi_auth:
            raise RuntimeError(
                "live execution requires a Kalshi key ID and matching RSA private key"
            )
        if self.settings.kalshi_env == "production" and not self.settings.allow_production:
            raise RuntimeError(
                "production requires ALLOW_PRODUCTION=true in addition to live gates"
            )
