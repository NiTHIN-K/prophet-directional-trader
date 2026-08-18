from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterator

from prophet_trader.models import Forecast, Market, ONE, OrderIntent, Position, Signal, ZERO


def utc_now_text() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_text(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


class StateStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        # SQLite foreign-key enforcement is connection-local, not a persistent DB setting.
        connection.execute("PRAGMA foreign_keys=ON")
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self.connection() as connection:
            connection.executescript(
                """
                PRAGMA journal_mode=WAL;
                PRAGMA foreign_keys=ON;

                CREATE TABLE IF NOT EXISTS cycles (
                    cycle_id TEXT PRIMARY KEY,
                    mode TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    status TEXT NOT NULL,
                    summary_json TEXT
                );

                CREATE TABLE IF NOT EXISTS market_state (
                    ticker TEXT PRIMARY KEY,
                    last_fill_reference_price TEXT,
                    last_fill_book_side TEXT,
                    last_fill_at TEXT,
                    last_forecast_price TEXT,
                    last_forecast_at TEXT,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS forecasts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cycle_id TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    model TEXT NOT NULL,
                    response_id TEXT,
                    request_id INTEGER,
                    probability_yes TEXT NOT NULL,
                    yes_bid TEXT NOT NULL,
                    yes_ask TEXT NOT NULL,
                    resolution_clear INTEGER NOT NULL,
                    uncertainty TEXT NOT NULL,
                    confidence TEXT NOT NULL DEFAULT 'medium',
                    rationale TEXT NOT NULL,
                    evidence_json TEXT NOT NULL,
                    raw_json TEXT NOT NULL,
                    FOREIGN KEY(cycle_id) REFERENCES cycles(cycle_id)
                );

                CREATE TABLE IF NOT EXISTS decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cycle_id TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    edge TEXT NOT NULL,
                    current_position TEXT NOT NULL,
                    target_position TEXT NOT NULL,
                    order_count TEXT,
                    book_side TEXT,
                    yes_price TEXT,
                    allowed INTEGER NOT NULL,
                    reason TEXT NOT NULL,
                    FOREIGN KEY(cycle_id) REFERENCES cycles(cycle_id)
                );

                CREATE TABLE IF NOT EXISTS orders (
                    client_order_id TEXT PRIMARY KEY,
                    exchange_order_id TEXT,
                    cycle_id TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    book_side TEXT NOT NULL,
                    count TEXT NOT NULL,
                    yes_price TEXT NOT NULL,
                    fill_count TEXT NOT NULL DEFAULT '0',
                    remaining_count TEXT NOT NULL,
                    status TEXT NOT NULL,
                    reference_price TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    response_json TEXT NOT NULL,
                    FOREIGN KEY(cycle_id) REFERENCES cycles(cycle_id)
                );

                CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status, mode);
                CREATE INDEX IF NOT EXISTS idx_forecasts_ticker ON forecasts(ticker, created_at);

                CREATE TABLE IF NOT EXISTS forecast_requests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    client_request_id TEXT NOT NULL UNIQUE,
                    cycle_id TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    two_hour_slot TEXT NOT NULL,
                    model TEXT NOT NULL,
                    prompt_version TEXT NOT NULL,
                    context_hash TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    received_at TEXT,
                    completed_at TEXT,
                    provider_request_id TEXT,
                    raw_response_json TEXT,
                    input_tokens INTEGER,
                    cached_tokens INTEGER,
                    reasoning_tokens INTEGER,
                    output_tokens INTEGER,
                    total_tokens INTEGER,
                    search_queries_json TEXT,
                    duration_ms INTEGER,
                    reserved_cost_dollars TEXT NOT NULL,
                    estimated_cost_dollars TEXT,
                    error TEXT,
                    FOREIGN KEY(cycle_id) REFERENCES cycles(cycle_id),
                    UNIQUE(ticker, two_hour_slot, model, prompt_version, context_hash)
                );

                CREATE INDEX IF NOT EXISTS idx_forecast_requests_status
                    ON forecast_requests(status, created_at);
                CREATE INDEX IF NOT EXISTS idx_forecast_requests_daily
                    ON forecast_requests(created_at, status);
                CREATE UNIQUE INDEX IF NOT EXISTS idx_forecast_once_per_two_hour_slot
                    ON forecast_requests(ticker, two_hour_slot, model, prompt_version);

                CREATE TABLE IF NOT EXISTS forecast_circuit (
                    id INTEGER PRIMARY KEY CHECK(id = 1),
                    state TEXT NOT NULL,
                    reason TEXT,
                    opened_at TEXT,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS scheduler_leader (
                    name TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    acquired_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS paper_positions (
                    ticker TEXT PRIMARY KEY,
                    contracts TEXT NOT NULL,
                    risk_dollars TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS paper_account (
                    id INTEGER PRIMARY KEY CHECK(id = 1),
                    cash_dollars TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS event_watch_state (
                    ticker TEXT PRIMARY KEY,
                    price_reference TEXT,
                    metadata_fingerprint TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    source_checked_at TEXT,
                    last_trigger_at TEXT,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS official_source_state (
                    ticker TEXT NOT NULL,
                    url TEXT NOT NULL,
                    fingerprint TEXT,
                    etag TEXT,
                    last_modified TEXT,
                    status_code INTEGER,
                    error TEXT,
                    checked_at TEXT NOT NULL,
                    PRIMARY KEY(ticker, url)
                );

                CREATE TABLE IF NOT EXISTS event_triggers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    trigger_types_json TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    detected_at TEXT NOT NULL,
                    processed_at TEXT,
                    cycle_id TEXT,
                    error TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_event_triggers_status
                    ON event_triggers(status, detected_at);
                """
            )
            self._ensure_column(connection, "market_state", "last_fill_book_side", "TEXT")
            self._ensure_column(connection, "forecasts", "request_id", "INTEGER")
            self._ensure_column(
                connection, "forecasts", "confidence", "TEXT NOT NULL DEFAULT 'medium'"
            )
            connection.execute(
                """
                INSERT INTO forecast_circuit(id, state, updated_at)
                VALUES (1, 'closed', ?)
                ON CONFLICT(id) DO NOTHING
                """,
                (utc_now_text(),),
            )

    @staticmethod
    def _ensure_column(
        connection: sqlite3.Connection,
        table: str,
        column: str,
        definition: str,
    ) -> None:
        columns = {str(row[1]) for row in connection.execute(f"PRAGMA table_info({table})")}
        if column not in columns:
            connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    def start_cycle(self, cycle_id: str, mode: str) -> None:
        with self.connection() as connection:
            connection.execute(
                "INSERT INTO cycles(cycle_id, mode, started_at, status) VALUES (?, ?, ?, ?)",
                (cycle_id, mode, utc_now_text(), "running"),
            )

    def finish_cycle(self, cycle_id: str, status: str, summary: dict[str, Any]) -> None:
        with self.connection() as connection:
            connection.execute(
                "UPDATE cycles SET finished_at=?, status=?, summary_json=? WHERE cycle_id=?",
                (utc_now_text(), status, json_text(summary), cycle_id),
            )

    def get_last_fill_reference(self, ticker: str) -> Decimal | None:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT last_fill_reference_price FROM market_state WHERE ticker=?", (ticker,)
            ).fetchone()
        if not row or row["last_fill_reference_price"] in {None, ""}:
            return None
        return Decimal(row["last_fill_reference_price"])

    def get_last_fill_context(self, ticker: str) -> tuple[Decimal | None, str | None]:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT last_fill_reference_price, last_fill_book_side "
                "FROM market_state WHERE ticker=?",
                (ticker,),
            ).fetchone()
        if not row or row["last_fill_reference_price"] in {None, ""}:
            return None, None
        return Decimal(row["last_fill_reference_price"]), row["last_fill_book_side"]

    def record_fill_reference(
        self,
        ticker: str,
        reference_price: Decimal,
        book_side: str | None = None,
    ) -> None:
        now = utc_now_text()
        with self.connection() as connection:
            connection.execute(
                """
                INSERT INTO market_state(
                    ticker, last_fill_reference_price, last_fill_book_side,
                    last_fill_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(ticker) DO UPDATE SET
                    last_fill_reference_price=excluded.last_fill_reference_price,
                    last_fill_book_side=excluded.last_fill_book_side,
                    last_fill_at=excluded.last_fill_at,
                    updated_at=excluded.updated_at
                """,
                (ticker, str(reference_price), book_side, now, now),
            )

    def record_forecast(
        self,
        cycle_id: str,
        market: Market,
        forecast: Forecast,
        *,
        request_id: int | None = None,
    ) -> None:
        now = utc_now_text()
        evidence = [
            {
                "claim": item.claim,
                "source_url": item.source_url,
                "published_at": item.published_at,
            }
            for item in forecast.evidence
        ]
        with self.connection() as connection:
            connection.execute(
                """
                INSERT INTO forecasts(
                    cycle_id, ticker, created_at, model, response_id, request_id,
                    probability_yes, yes_bid, yes_ask, resolution_clear, uncertainty, confidence, rationale,
                    evidence_json, raw_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    cycle_id,
                    market.ticker,
                    now,
                    forecast.model,
                    forecast.response_id,
                    request_id,
                    str(forecast.probability_yes),
                    str(market.yes_bid),
                    str(market.yes_ask),
                    int(forecast.resolution_clear),
                    forecast.uncertainty,
                    forecast.confidence,
                    forecast.rationale,
                    json_text(evidence),
                    json_text(forecast.raw),
                ),
            )
            connection.execute(
                """
                INSERT INTO market_state(ticker, last_forecast_price, last_forecast_at, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(ticker) DO UPDATE SET
                    last_forecast_price=excluded.last_forecast_price,
                    last_forecast_at=excluded.last_forecast_at,
                    updated_at=excluded.updated_at
                """,
                (market.ticker, str(market.midpoint), now, now),
            )

    def record_decision(
        self,
        cycle_id: str,
        market: Market,
        signal: Signal,
        position: Position,
        intent: OrderIntent | None,
        *,
        allowed: bool,
        reason: str,
    ) -> None:
        with self.connection() as connection:
            connection.execute(
                """
                INSERT INTO decisions(
                    cycle_id, ticker, created_at, direction, edge, current_position,
                    target_position, order_count, book_side, yes_price, allowed, reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    cycle_id,
                    market.ticker,
                    utc_now_text(),
                    signal.direction.value,
                    str(signal.actionable_edge),
                    str(position.contracts),
                    str(signal.target_position),
                    str(intent.count) if intent else None,
                    intent.book_side if intent else None,
                    str(intent.yes_price) if intent else None,
                    int(allowed),
                    reason,
                ),
            )

    def paper_positions(self) -> dict[str, Position]:
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT ticker, contracts, risk_dollars FROM paper_positions"
            ).fetchall()
        return {
            row["ticker"]: Position(
                row["ticker"], Decimal(row["contracts"]), Decimal(row["risk_dollars"])
            )
            for row in rows
            if Decimal(row["contracts"]) != ZERO
        }

    def paper_total_exposure(self) -> Decimal:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT COALESCE(SUM(CAST(risk_dollars AS REAL)), 0) AS risk FROM paper_positions"
            ).fetchone()
        return Decimal(str(row["risk"] if row else 0))

    @staticmethod
    def _ensure_paper_account(
        connection: sqlite3.Connection,
        starting_cash: Decimal,
    ) -> Decimal:
        row = connection.execute(
            "SELECT cash_dollars FROM paper_account WHERE id=1"
        ).fetchone()
        if row:
            return Decimal(row["cash_dollars"])

        # Migrate an older risk-only journal conservatively by treating existing exposure as
        # already paid for. New databases simply start at PAPER_STARTING_CASH.
        rows = connection.execute(
            "SELECT risk_dollars FROM paper_positions"
        ).fetchall()
        legacy_exposure = sum((Decimal(item["risk_dollars"]) for item in rows), ZERO)
        cash = max(ZERO, starting_cash - legacy_exposure)
        connection.execute(
            "INSERT INTO paper_account(id, cash_dollars, updated_at) VALUES (1, ?, ?)",
            (str(cash), utc_now_text()),
        )
        return cash

    def paper_cash(self, starting_cash: Decimal) -> Decimal:
        with self.connection() as connection:
            return self._ensure_paper_account(connection, starting_cash)

    def apply_paper_fill(
        self,
        *,
        cycle_id: str,
        client_order_id: str,
        market: Market,
        intent: OrderIntent,
        starting_cash: Decimal,
    ) -> Position:
        now = utc_now_text()
        with self.connection() as connection:
            row = connection.execute(
                "SELECT contracts FROM paper_positions WHERE ticker=?",
                (market.ticker,),
            ).fetchone()
            current_contracts = Decimal(row["contracts"]) if row else ZERO
            contracts = (
                current_contracts + intent.count
                if intent.book_side == "bid"
                else current_contracts - intent.count
            )

            if intent.book_side == "bid":
                closing = (
                    min(intent.count, abs(current_contracts))
                    if current_contracts < ZERO
                    else ZERO
                )
                opening = intent.count - closing
                cash_delta = closing * (ONE - intent.yes_price) - opening * intent.yes_price
            else:
                closing = (
                    min(intent.count, current_contracts)
                    if current_contracts > ZERO
                    else ZERO
                )
                opening = intent.count - closing
                cash_delta = closing * intent.yes_price - opening * (ONE - intent.yes_price)

            cash = self._ensure_paper_account(connection, starting_cash)
            cash_after = cash + cash_delta
            if cash_after < ZERO:
                raise RuntimeError("paper fill would make simulated cash negative")

            side_cost = market.yes_ask if contracts > ZERO else market.no_ask
            risk = abs(contracts) * side_cost if contracts else ZERO
            response = {
                "simulated": True,
                "fill_count": str(intent.count),
                "remaining_count": "0",
                "cash_delta": str(cash_delta),
                "cash_after": str(cash_after),
            }
            connection.execute(
                """
                INSERT INTO paper_positions(ticker, contracts, risk_dollars, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(ticker) DO UPDATE SET
                    contracts=excluded.contracts,
                    risk_dollars=excluded.risk_dollars,
                    updated_at=excluded.updated_at
                """,
                (market.ticker, str(contracts), str(risk), now),
            )
            connection.execute(
                "UPDATE paper_account SET cash_dollars=?, updated_at=? WHERE id=1",
                (str(cash_after), now),
            )
            connection.execute(
                """
                INSERT INTO orders(
                    client_order_id, exchange_order_id, cycle_id, ticker, mode, book_side,
                    count, yes_price, fill_count, remaining_count, status, reference_price,
                    created_at, updated_at, response_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    client_order_id,
                    f"paper-{client_order_id}",
                    cycle_id,
                    market.ticker,
                    "paper",
                    intent.book_side,
                    str(intent.count),
                    str(intent.yes_price),
                    str(intent.count),
                    "0",
                    "filled",
                    str(market.midpoint),
                    now,
                    now,
                    json_text(response),
                ),
            )
        self.record_fill_reference(market.ticker, intent.yes_price, intent.book_side)
        return Position(market.ticker, contracts, risk)

    def record_live_submission(
        self,
        *,
        cycle_id: str,
        client_order_id: str,
        market: Market,
        intent: OrderIntent,
        response: dict[str, Any] | None,
        status: str,
    ) -> None:
        response = response or {}
        order = response.get("order") if isinstance(response.get("order"), dict) else response
        fill_count = str(order.get("fill_count") or order.get("fill_count_fp") or "0")
        remaining = str(
            order.get("remaining_count") or order.get("remaining_count_fp") or intent.count
        )
        now = utc_now_text()
        with self.connection() as connection:
            connection.execute(
                """
                INSERT INTO orders(
                    client_order_id, exchange_order_id, cycle_id, ticker, mode, book_side,
                    count, yes_price, fill_count, remaining_count, status, reference_price,
                    created_at, updated_at, response_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(client_order_id) DO UPDATE SET
                    exchange_order_id=excluded.exchange_order_id,
                    fill_count=excluded.fill_count,
                    remaining_count=excluded.remaining_count,
                    status=excluded.status,
                    updated_at=excluded.updated_at,
                    response_json=excluded.response_json
                """,
                (
                    client_order_id,
                    order.get("order_id"),
                    cycle_id,
                    market.ticker,
                    "live",
                    intent.book_side,
                    str(intent.count),
                    str(intent.yes_price),
                    fill_count,
                    remaining,
                    status,
                    str(market.midpoint),
                    now,
                    now,
                    json_text(response),
                ),
            )
        if Decimal(fill_count) > ZERO:
            self.record_fill_reference(market.ticker, intent.yes_price, intent.book_side)

    def open_live_orders(self) -> list[dict[str, Any]]:
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM orders
                WHERE mode='live' AND status IN ('pending', 'submitted', 'resting', 'partial')
                ORDER BY created_at
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def update_live_order(
        self,
        client_order_id: str,
        *,
        exchange_order_id: str | None,
        fill_count: Decimal,
        remaining_count: Decimal,
        status: str,
        response: dict[str, Any],
    ) -> None:
        with self.connection() as connection:
            connection.execute(
                """
                UPDATE orders SET exchange_order_id=COALESCE(?, exchange_order_id), fill_count=?,
                    remaining_count=?, status=?, updated_at=?, response_json=?
                WHERE client_order_id=?
                """,
                (
                    exchange_order_id,
                    str(fill_count),
                    str(remaining_count),
                    status,
                    utc_now_text(),
                    json_text(response),
                    client_order_id,
                ),
            )
        if fill_count > ZERO:
            with self.connection() as connection:
                row = connection.execute(
                    "SELECT ticker, reference_price FROM orders WHERE client_order_id=?",
                    (client_order_id,),
                ).fetchone()
            if row:
                self.record_fill_reference(row["ticker"], Decimal(row["reference_price"]))

    @staticmethod
    def _request_cost(row: sqlite3.Row) -> Decimal:
        if row["status"] in {"cancelled", "failed"}:
            return ZERO
        reserved = Decimal(row["reserved_cost_dollars"] or "0")
        estimated = Decimal(row["estimated_cost_dollars"] or "0")
        if row["status"] in {"queued", "in_progress", "unknown"}:
            return max(reserved, estimated)
        return estimated if estimated > ZERO else reserved

    def daily_forecast_spend(self, *, now: datetime | None = None) -> Decimal:
        now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        day_start = now.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT status, reserved_cost_dollars, estimated_cost_dollars "
                "FROM forecast_requests WHERE created_at>=?",
                (day_start,),
            ).fetchall()
        return sum((self._request_cost(row) for row in rows), ZERO)

    def queue_forecast_request(
        self,
        *,
        client_request_id: str,
        cycle_id: str,
        ticker: str,
        two_hour_slot: str,
        model: str,
        prompt_version: str,
        context_hash: str,
        reserved_cost_dollars: Decimal,
        daily_limit_dollars: Decimal,
        now: datetime,
    ) -> dict[str, Any]:
        now = now.astimezone(timezone.utc)
        now_text = now.isoformat()
        day_start = now.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            circuit = connection.execute(
                "SELECT state, reason FROM forecast_circuit WHERE id=1"
            ).fetchone()
            if circuit and circuit["state"] == "open":
                return {"queued": False, "reason": f"forecast circuit open: {circuit['reason']}"}
            unknown = connection.execute(
                "SELECT id FROM forecast_requests WHERE ticker=? AND status='unknown' LIMIT 1",
                (ticker,),
            ).fetchone()
            if unknown:
                return {"queued": False, "reason": "unreconciled UNKNOWN forecast request"}
            duplicate = connection.execute(
                """
                SELECT id, status FROM forecast_requests
                WHERE ticker=? AND two_hour_slot=? AND model=?
                  AND prompt_version=? AND context_hash=?
                """,
                (ticker, two_hour_slot, model, prompt_version, context_hash),
            ).fetchone()
            if duplicate:
                return {
                    "queued": False,
                    "reason": "duplicate forecast key",
                    "request_id": int(duplicate["id"]),
                    "status": str(duplicate["status"]),
                }
            rows = connection.execute(
                "SELECT status, reserved_cost_dollars, estimated_cost_dollars "
                "FROM forecast_requests WHERE created_at>=?",
                (day_start,),
            ).fetchall()
            spend = sum((self._request_cost(row) for row in rows), ZERO)
            if spend + reserved_cost_dollars > daily_limit_dollars:
                return {
                    "queued": False,
                    "reason": "daily forecast-spend limit",
                    "daily_spend": str(spend),
                }
            try:
                cursor = connection.execute(
                    """
                    INSERT INTO forecast_requests(
                        client_request_id, cycle_id, ticker, two_hour_slot, model,
                        prompt_version, context_hash, status, created_at,
                        reserved_cost_dollars
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, 'queued', ?, ?)
                    """,
                    (
                        client_request_id,
                        cycle_id,
                        ticker,
                        two_hour_slot,
                        model,
                        prompt_version,
                        context_hash,
                        now_text,
                        str(reserved_cost_dollars),
                    ),
                )
            except sqlite3.IntegrityError:
                return {"queued": False, "reason": "duplicate forecast key"}
            return {"queued": True, "request_id": int(cursor.lastrowid)}

    def start_forecast_request(self, request_id: int) -> bool:
        with self.connection() as connection:
            cursor = connection.execute(
                """
                UPDATE forecast_requests SET status='in_progress', started_at=?
                WHERE id=? AND status='queued'
                """,
                (utc_now_text(), request_id),
            )
        return cursor.rowcount == 1

    def record_provider_response(
        self,
        request_id: int,
        *,
        provider_request_id: str | None,
        raw_response: dict[str, Any],
        input_tokens: int,
        cached_tokens: int,
        reasoning_tokens: int,
        output_tokens: int,
        total_tokens: int,
        search_queries: tuple[str, ...],
        duration_ms: int,
        estimated_cost_dollars: Decimal,
    ) -> None:
        """Persist the unparsed provider result and accounting before JSON parsing."""
        with self.connection() as connection:
            connection.execute(
                """
                UPDATE forecast_requests SET
                    status='received', received_at=?, provider_request_id=?,
                    raw_response_json=?, input_tokens=?, cached_tokens=?,
                    reasoning_tokens=?, output_tokens=?, total_tokens=?,
                    search_queries_json=?, duration_ms=?, estimated_cost_dollars=?
                WHERE id=? AND status='in_progress'
                """,
                (
                    utc_now_text(),
                    provider_request_id,
                    json_text(raw_response),
                    input_tokens,
                    cached_tokens,
                    reasoning_tokens,
                    output_tokens,
                    total_tokens,
                    json_text(search_queries),
                    duration_ms,
                    str(estimated_cost_dollars),
                    request_id,
                ),
            )

    def finish_forecast_request(
        self,
        request_id: int,
        *,
        status: str,
        error: str | None = None,
        provider_request_id: str | None = None,
    ) -> None:
        if status not in {"completed", "parse_failed", "failed", "unknown", "cancelled"}:
            raise ValueError("invalid forecast request terminal status")
        with self.connection() as connection:
            connection.execute(
                """
                UPDATE forecast_requests SET status=?, completed_at=?, error=?,
                    provider_request_id=COALESCE(?, provider_request_id)
                WHERE id=?
                """,
                (status, utc_now_text(), error, provider_request_id, request_id),
            )

    def cancel_queued_forecasts(self, *, cycle_id: str | None, reason: str) -> int:
        where = "status='queued'"
        parameters: list[Any] = [utc_now_text(), reason]
        if cycle_id is not None:
            where += " AND cycle_id=?"
            parameters.append(cycle_id)
        with self.connection() as connection:
            cursor = connection.execute(
                f"UPDATE forecast_requests SET status='cancelled', completed_at=?, error=? WHERE {where}",
                parameters,
            )
        return cursor.rowcount

    def forecast_circuit(self) -> dict[str, Any]:
        with self.connection() as connection:
            row = connection.execute("SELECT * FROM forecast_circuit WHERE id=1").fetchone()
        return dict(row) if row else {"state": "closed"}

    def open_forecast_circuit(self, reason: str) -> int:
        now = utc_now_text()
        with self.connection() as connection:
            connection.execute(
                """
                UPDATE forecast_circuit
                SET state='open', reason=?, opened_at=?, updated_at=? WHERE id=1
                """,
                (reason, now, now),
            )
        return self.cancel_queued_forecasts(cycle_id=None, reason=f"circuit opened: {reason}")

    def reset_forecast_circuit(self) -> None:
        with self.connection() as connection:
            connection.execute(
                """
                UPDATE forecast_circuit
                SET state='closed', reason=NULL, opened_at=NULL, updated_at=? WHERE id=1
                """,
                (utc_now_text(),),
            )

    def forecast_context_snapshot(self, ticker: str) -> dict[str, Any]:
        state = self.event_watch_state(ticker)
        with self.connection() as connection:
            source_rows = connection.execute(
                """
                SELECT url, fingerprint, etag, last_modified, status_code, error, checked_at
                FROM official_source_state WHERE ticker=? ORDER BY url
                """,
                (ticker,),
            ).fetchall()
        metadata: dict[str, Any] = {}
        if state:
            try:
                metadata = json.loads(state.get("metadata_json") or "{}")
            except (json.JSONDecodeError, TypeError):
                metadata = {}
        return {
            "watch": {
                "price_reference": state.get("price_reference") if state else None,
                "metadata": metadata,
            },
            "official_sources": [
                {
                    key: row[key]
                    for key in ("url", "fingerprint", "etag", "last_modified", "status_code", "error")
                }
                for row in source_rows
            ],
        }

    def acquire_scheduler_leader(
        self,
        *,
        name: str,
        owner_id: str,
        lease_seconds: int,
        now: datetime | None = None,
    ) -> bool:
        now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        expires = now + timedelta(seconds=lease_seconds)
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT owner_id, expires_at FROM scheduler_leader WHERE name=?", (name,)
            ).fetchone()
            if row and row["owner_id"] != owner_id and row["expires_at"] > now.isoformat():
                return False
            connection.execute(
                """
                INSERT INTO scheduler_leader(name, owner_id, acquired_at, expires_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET owner_id=excluded.owner_id,
                    acquired_at=excluded.acquired_at, expires_at=excluded.expires_at
                """,
                (name, owner_id, now.isoformat(), expires.isoformat()),
            )
        return True

    def release_scheduler_leader(self, *, name: str, owner_id: str) -> None:
        with self.connection() as connection:
            connection.execute(
                "DELETE FROM scheduler_leader WHERE name=? AND owner_id=?", (name, owner_id)
            )

    def event_watch_state(self, ticker: str) -> dict[str, Any] | None:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM event_watch_state WHERE ticker=?", (ticker,)
            ).fetchone()
        return dict(row) if row else None

    def event_watch_tickers(self) -> set[str]:
        with self.connection() as connection:
            rows = connection.execute("SELECT ticker FROM event_watch_state").fetchall()
        return {str(row["ticker"]) for row in rows}

    def upsert_event_watch_state(
        self,
        ticker: str,
        *,
        price_reference: Decimal | None,
        metadata_fingerprint: str,
        metadata: dict[str, Any],
        source_checked_at: str | None = None,
    ) -> None:
        now = utc_now_text()
        with self.connection() as connection:
            connection.execute(
                """
                INSERT INTO event_watch_state(
                    ticker, price_reference, metadata_fingerprint, metadata_json,
                    source_checked_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(ticker) DO UPDATE SET
                    price_reference=excluded.price_reference,
                    metadata_fingerprint=excluded.metadata_fingerprint,
                    metadata_json=excluded.metadata_json,
                    source_checked_at=COALESCE(
                        excluded.source_checked_at, event_watch_state.source_checked_at
                    ),
                    updated_at=excluded.updated_at
                """,
                (
                    ticker,
                    str(price_reference) if price_reference is not None else None,
                    metadata_fingerprint,
                    json_text(metadata),
                    source_checked_at,
                    now,
                ),
            )

    def record_event_trigger_time(self, ticker: str, observed_at: str) -> None:
        with self.connection() as connection:
            connection.execute(
                "UPDATE event_watch_state SET last_trigger_at=?, updated_at=? WHERE ticker=?",
                (observed_at, utc_now_text(), ticker),
            )

    def official_source_state(self, ticker: str, url: str) -> dict[str, Any] | None:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM official_source_state WHERE ticker=? AND url=?",
                (ticker, url),
            ).fetchone()
        return dict(row) if row else None

    def upsert_official_source_state(
        self,
        ticker: str,
        url: str,
        *,
        fingerprint: str | None,
        etag: str | None,
        last_modified: str | None,
        status_code: int | None,
        error: str | None,
        checked_at: str,
    ) -> None:
        with self.connection() as connection:
            connection.execute(
                """
                INSERT INTO official_source_state(
                    ticker, url, fingerprint, etag, last_modified,
                    status_code, error, checked_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(ticker, url) DO UPDATE SET
                    fingerprint=COALESCE(excluded.fingerprint, official_source_state.fingerprint),
                    etag=excluded.etag,
                    last_modified=excluded.last_modified,
                    status_code=excluded.status_code,
                    error=excluded.error,
                    checked_at=excluded.checked_at
                """,
                (
                    ticker,
                    url,
                    fingerprint,
                    etag,
                    last_modified,
                    status_code,
                    error,
                    checked_at,
                ),
            )

    def enqueue_event_trigger(
        self,
        ticker: str,
        trigger_types: list[str],
        payload: dict[str, Any],
        *,
        detected_at: str,
    ) -> int:
        with self.connection() as connection:
            row = connection.execute(
                """
                SELECT id, trigger_types_json, payload_json FROM event_triggers
                WHERE ticker=? AND status='pending' ORDER BY id LIMIT 1
                """,
                (ticker,),
            ).fetchone()
            if row:
                existing_types = json.loads(row["trigger_types_json"])
                existing_payload = json.loads(row["payload_json"])
                merged_types = sorted(set(existing_types) | set(trigger_types))
                existing_payload.update(payload)
                connection.execute(
                    """
                    UPDATE event_triggers
                    SET trigger_types_json=?, payload_json=?, detected_at=? WHERE id=?
                    """,
                    (json_text(merged_types), json_text(existing_payload), detected_at, row["id"]),
                )
                return int(row["id"])
            cursor = connection.execute(
                """
                INSERT INTO event_triggers(
                    ticker, trigger_types_json, payload_json, status, detected_at
                ) VALUES (?, ?, ?, 'pending', ?)
                """,
                (ticker, json_text(sorted(set(trigger_types))), json_text(payload), detected_at),
            )
            return int(cursor.lastrowid)

    def record_context_event(
        self,
        ticker: str,
        event_types: list[str],
        payload: dict[str, Any],
        *,
        detected_at: str,
    ) -> int:
        with self.connection() as connection:
            cursor = connection.execute(
                """
                INSERT INTO event_triggers(
                    ticker, trigger_types_json, payload_json, status,
                    detected_at, processed_at, error
                ) VALUES (?, ?, ?, 'context_only', ?, ?, ?)
                """,
                (
                    ticker,
                    json_text(sorted(set(event_types))),
                    json_text(payload),
                    detected_at,
                    detected_at,
                    "event-driven forecasting disabled by strict paper replication",
                ),
            )
            return int(cursor.lastrowid)

    def cancel_pending_event_triggers(self) -> int:
        now = utc_now_text()
        with self.connection() as connection:
            cursor = connection.execute(
                """
                UPDATE event_triggers
                SET status='context_only', processed_at=?,
                    error='event-driven forecasting disabled by strict paper replication'
                WHERE status='pending'
                """,
                (now,),
            )
        return cursor.rowcount

    def pending_event_triggers(self, limit: int) -> list[dict[str, Any]]:
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM event_triggers WHERE status='pending'
                ORDER BY detected_at, id LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def finish_event_trigger(
        self,
        trigger_id: int,
        *,
        status: str,
        cycle_id: str | None,
        error: str | None = None,
    ) -> None:
        with self.connection() as connection:
            connection.execute(
                """
                UPDATE event_triggers
                SET status=?, processed_at=?, cycle_id=?, error=? WHERE id=?
                """,
                (status, utc_now_text(), cycle_id, error, trigger_id),
            )

    def recent_event_triggers(self, limit: int = 50) -> list[dict[str, Any]]:
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT * FROM event_triggers ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(row) for row in rows]

    def latest_forecast_evidence(self, ticker: str) -> list[dict[str, Any]]:
        with self.connection() as connection:
            row = connection.execute(
                """
                SELECT evidence_json FROM forecasts
                WHERE ticker=? ORDER BY id DESC LIMIT 1
                """,
                (ticker,),
            ).fetchone()
        if not row:
            return []
        try:
            evidence = json.loads(row["evidence_json"])
        except (json.JSONDecodeError, TypeError):
            return []
        return [item for item in evidence if isinstance(item, dict)]

    def recent_cycles(self, limit: int = 10) -> list[dict[str, Any]]:
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT * FROM cycles ORDER BY started_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(row) for row in rows]

    def recent_forecast_requests(self, limit: int = 50) -> list[dict[str, Any]]:
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT id, client_request_id, cycle_id, ticker, two_hour_slot, model,
                    prompt_version, context_hash, status, created_at, started_at,
                    received_at, completed_at, provider_request_id, input_tokens,
                    cached_tokens, reasoning_tokens, output_tokens, total_tokens,
                    search_queries_json, duration_ms, reserved_cost_dollars,
                    estimated_cost_dollars, error
                FROM forecast_requests ORDER BY id DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def recent_decisions(self, limit: int = 50) -> list[dict[str, Any]]:
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT * FROM decisions ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(row) for row in rows]
