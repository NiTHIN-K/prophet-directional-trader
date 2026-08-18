from __future__ import annotations

import os
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path


TRUE_VALUES = {"1", "true", "yes", "on"}


def load_dotenv(path: Path) -> None:
    """Load a conservative KEY=VALUE file without overriding the process environment."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or not key.replace("_", "").isalnum() or key[0].isdigit():
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key, value.replace("\\n", "\n"))


def _bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    return default if value is None else value.strip().lower() in TRUE_VALUES


def _int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value in {None, ""} else int(value)


def _decimal(name: str, default: str) -> Decimal:
    value = os.getenv(name)
    return Decimal(default if value in {None, ""} else value)


@dataclass(frozen=True)
class Settings:
    root: Path
    trading_mode: str
    kalshi_env: str
    live_trading_enabled: bool
    allow_production: bool
    kalshi_api_key_id: str | None
    kalshi_private_key_path: Path | None
    kalshi_private_key_pem: str | None
    gemini_api_key: str | None
    gemini_model: str
    gemini_reasoning_level: str
    cycle_seconds: int
    min_days_to_close: Decimal
    max_days_to_close: Decimal
    stop_hours_before_resolution: Decimal
    max_close_resolution_gap_hours: Decimal
    refresh_move_threshold: Decimal
    position_scale_contracts: Decimal
    min_actionable_edge: Decimal
    max_markets_per_cycle: int
    paper_starting_cash: Decimal
    max_target_contracts: Decimal
    max_order_contracts: Decimal
    max_market_risk_dollars: Decimal
    max_total_exposure_dollars: Decimal
    max_spread: Decimal
    min_liquidity_dollars: Decimal
    min_evidence_sources: int
    block_high_uncertainty: bool
    state_db_path: Path
    kill_switch_path: Path
    request_timeout_seconds: int
    max_http_retries: int
    forecasting_enabled: bool = False
    scheduler_enabled: bool = False
    runtime_environment: str = "production"
    strict_paper_replication: bool = True
    gemini_max_output_tokens: int = 2000
    max_forecasts_per_cycle: int = 4
    daily_forecast_spend_limit: Decimal = Decimal("1.00")
    forecast_reserve_cost_dollars: Decimal = Decimal("0.15")
    gemini_input_cost_per_million: Decimal = Decimal("2.00")
    gemini_cached_input_cost_per_million: Decimal = Decimal("0.20")
    gemini_output_cost_per_million: Decimal = Decimal("12.00")
    gemini_search_cost_per_query: Decimal = Decimal("0.014")
    forecast_prompt_version: str = "gemini-paper-v2"
    event_price_move_threshold: Decimal = Decimal("0.03")
    official_source_poll_seconds: int = 300
    event_trigger_cooldown_seconds: int = 300
    max_event_triggers_per_poll: int = 2
    official_source_timeout_seconds: int = 15
    official_source_max_bytes: int = 1000000
    official_source_error_backoff_seconds: int = 21600

    @property
    def kalshi_base_url(self) -> str:
        if self.kalshi_env == "production":
            return "https://external-api.kalshi.com/trade-api/v2"
        return "https://external-api.demo.kalshi.co/trade-api/v2"

    @property
    def has_kalshi_auth(self) -> bool:
        return bool(
            self.kalshi_api_key_id
            and (self.kalshi_private_key_path or self.kalshi_private_key_pem)
        )

    def validate(self) -> None:
        if self.trading_mode not in {"paper", "live"}:
            raise ValueError("TRADING_MODE must be paper or live")
        if self.kalshi_env not in {"demo", "production"}:
            raise ValueError("KALSHI_ENV must be demo or production")
        if self.gemini_reasoning_level != "high":
            raise ValueError("strict paper replication requires GEMINI_REASONING_LEVEL=high")
        if self.strict_paper_replication and self.trading_mode != "paper":
            raise ValueError("strict paper replication permits paper trading only")
        if self.strict_paper_replication and self.live_trading_enabled:
            raise ValueError("LIVE_TRADING_ENABLED must be false in strict paper replication")
        if self.runtime_environment == "test" and (self.forecasting_enabled or self.scheduler_enabled):
            raise ValueError("test environments cannot enable forecasting or the scheduler")
        if self.min_days_to_close < 0 or self.max_days_to_close <= self.min_days_to_close:
            raise ValueError("market horizon bounds are invalid")
        if not Decimal("0") <= self.min_actionable_edge < Decimal("1"):
            raise ValueError("MIN_ACTIONABLE_EDGE must be in [0,1)")
        if self.position_scale_contracts <= 0:
            raise ValueError("POSITION_SCALE_CONTRACTS must be positive")
        if self.max_markets_per_cycle <= 0:
            raise ValueError("MAX_MARKETS_PER_CYCLE must be positive")
        if self.cycle_seconds != 7200:
            raise ValueError("strict paper replication requires CYCLE_SECONDS=7200")
        if not 1 <= self.gemini_max_output_tokens <= 2000:
            raise ValueError("GEMINI_MAX_OUTPUT_TOKENS must be between 1 and 2000")
        if self.max_forecasts_per_cycle <= 0:
            raise ValueError("MAX_FORECASTS_PER_CYCLE must be positive")
        if self.daily_forecast_spend_limit <= 0:
            raise ValueError("DAILY_FORECAST_SPEND_LIMIT must be positive")
        if not Decimal("0") < self.forecast_reserve_cost_dollars <= self.daily_forecast_spend_limit:
            raise ValueError("FORECAST_RESERVE_COST_DOLLARS must be positive and within the daily limit")
        if not Decimal("0") < self.event_price_move_threshold < Decimal("1"):
            raise ValueError("EVENT_PRICE_MOVE_THRESHOLD must be between 0 and 1")
        if self.official_source_poll_seconds < 60:
            raise ValueError("OFFICIAL_SOURCE_POLL_SECONDS must be at least 60")
        if self.event_trigger_cooldown_seconds < 0:
            raise ValueError("EVENT_TRIGGER_COOLDOWN_SECONDS cannot be negative")
        if self.max_event_triggers_per_poll <= 0:
            raise ValueError("MAX_EVENT_TRIGGERS_PER_POLL must be positive")
        if self.official_source_timeout_seconds <= 0:
            raise ValueError("OFFICIAL_SOURCE_TIMEOUT_SECONDS must be positive")
        if self.official_source_max_bytes < 1024:
            raise ValueError("OFFICIAL_SOURCE_MAX_BYTES must be at least 1024")
        if self.official_source_error_backoff_seconds < self.official_source_poll_seconds:
            raise ValueError(
                "OFFICIAL_SOURCE_ERROR_BACKOFF_SECONDS must not be shorter than the poll interval"
            )

    @classmethod
    def from_env(cls, root: Path | None = None) -> "Settings":
        root = (root or Path.cwd()).resolve()
        load_dotenv(root / ".env")

        private_path_text = os.getenv("KALSHI_PRIVATE_KEY_PATH", "").strip()
        private_path = None
        if private_path_text:
            candidate = Path(private_path_text).expanduser()
            private_path = candidate if candidate.is_absolute() else root / candidate

        state_path = Path(os.getenv("STATE_DB_PATH", "state/trader.sqlite3"))
        if not state_path.is_absolute():
            state_path = root / state_path
        kill_path = Path(os.getenv("KILL_SWITCH_PATH", "STOP"))
        if not kill_path.is_absolute():
            kill_path = root / kill_path

        settings = cls(
            root=root,
            trading_mode=os.getenv("TRADING_MODE", "paper").strip().lower(),
            kalshi_env=os.getenv("KALSHI_ENV", "demo").strip().lower(),
            live_trading_enabled=_bool("LIVE_TRADING_ENABLED", False),
            allow_production=_bool("ALLOW_PRODUCTION", False),
            kalshi_api_key_id=(
                os.getenv("KALSHI_API_KEY_ID") or os.getenv("KALSHI_API_KEY") or ""
            ).strip() or None,
            kalshi_private_key_path=private_path,
            kalshi_private_key_pem=os.getenv("KALSHI_PRIVATE_KEY_PEM") or None,
            gemini_api_key=os.getenv("GEMINI_API_KEY") or None,
            gemini_model=os.getenv("GEMINI_MODEL", "gemini-3.1-pro-preview").strip(),
            gemini_reasoning_level=os.getenv("GEMINI_REASONING_LEVEL", "high").strip().lower(),
            cycle_seconds=_int("CYCLE_SECONDS", 7200),
            min_days_to_close=_decimal("MIN_DAYS_TO_CLOSE", "2"),
            max_days_to_close=_decimal("MAX_DAYS_TO_CLOSE", "14"),
            stop_hours_before_resolution=_decimal("STOP_HOURS_BEFORE_RESOLUTION", "3"),
            max_close_resolution_gap_hours=_decimal("MAX_CLOSE_RESOLUTION_GAP_HOURS", "1"),
            refresh_move_threshold=_decimal("REFRESH_MOVE_THRESHOLD", "0.10"),
            position_scale_contracts=_decimal("POSITION_SCALE_CONTRACTS", "100"),
            min_actionable_edge=_decimal("MIN_ACTIONABLE_EDGE", "0.02"),
            max_markets_per_cycle=_int("MAX_MARKETS_PER_CYCLE", 8),
            paper_starting_cash=_decimal("PAPER_STARTING_CASH", "200"),
            max_target_contracts=_decimal("MAX_TARGET_CONTRACTS", "50"),
            max_order_contracts=_decimal("MAX_ORDER_CONTRACTS", "25"),
            max_market_risk_dollars=_decimal("MAX_MARKET_RISK_DOLLARS", "25"),
            max_total_exposure_dollars=_decimal("MAX_TOTAL_EXPOSURE_DOLLARS", "100"),
            max_spread=_decimal("MAX_SPREAD", "0.15"),
            min_liquidity_dollars=_decimal("MIN_LIQUIDITY_DOLLARS", "1"),
            min_evidence_sources=_int("MIN_EVIDENCE_SOURCES", 1),
            block_high_uncertainty=_bool("BLOCK_HIGH_UNCERTAINTY", True),
            state_db_path=state_path,
            kill_switch_path=kill_path,
            request_timeout_seconds=_int("REQUEST_TIMEOUT_SECONDS", 30),
            max_http_retries=_int("MAX_HTTP_RETRIES", 3),
            forecasting_enabled=_bool("FORECASTING_ENABLED", False),
            scheduler_enabled=_bool("SCHEDULER_ENABLED", False),
            runtime_environment=os.getenv("APP_ENV", "production").strip().lower(),
            strict_paper_replication=_bool("STRICT_PAPER_REPLICATION", True),
            gemini_max_output_tokens=_int("GEMINI_MAX_OUTPUT_TOKENS", 2000),
            max_forecasts_per_cycle=_int("MAX_FORECASTS_PER_CYCLE", 4),
            daily_forecast_spend_limit=_decimal("DAILY_FORECAST_SPEND_LIMIT", "1.00"),
            forecast_reserve_cost_dollars=_decimal(
                "FORECAST_RESERVE_COST_DOLLARS", "0.15"
            ),
            gemini_input_cost_per_million=_decimal(
                "GEMINI_INPUT_COST_PER_MILLION", "2.00"
            ),
            gemini_cached_input_cost_per_million=_decimal(
                "GEMINI_CACHED_INPUT_COST_PER_MILLION", "0.20"
            ),
            gemini_output_cost_per_million=_decimal(
                "GEMINI_OUTPUT_COST_PER_MILLION", "12.00"
            ),
            gemini_search_cost_per_query=_decimal(
                "GEMINI_SEARCH_COST_PER_QUERY", "0.014"
            ),
            forecast_prompt_version=os.getenv(
                "FORECAST_PROMPT_VERSION", "gemini-paper-v2"
            ).strip(),
            event_price_move_threshold=_decimal("EVENT_PRICE_MOVE_THRESHOLD", "0.03"),
            official_source_poll_seconds=_int("OFFICIAL_SOURCE_POLL_SECONDS", 300),
            event_trigger_cooldown_seconds=_int("EVENT_TRIGGER_COOLDOWN_SECONDS", 300),
            max_event_triggers_per_poll=_int("MAX_EVENT_TRIGGERS_PER_POLL", 2),
            official_source_timeout_seconds=_int("OFFICIAL_SOURCE_TIMEOUT_SECONDS", 15),
            official_source_max_bytes=_int("OFFICIAL_SOURCE_MAX_BYTES", 1000000),
            official_source_error_backoff_seconds=_int(
                "OFFICIAL_SOURCE_ERROR_BACKOFF_SECONDS", 21600
            ),
        )
        settings.validate()
        return settings
