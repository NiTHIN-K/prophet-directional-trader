# Operations

## Fixed-cadence cron setup

`run-once` acquires the single-leader lease and deduplicates the current UTC two-hour slot. A
second invocation in the same slot cannot submit the same forecast again.

```cron
0 */2 * * * cd /path/to/prophet-directional-trader && .venv/bin/prophet-trader run-once >> state/forecast.log 2>&1
* * * * * cd /path/to/prophet-directional-trader && .venv/bin/prophet-trader watch-once >> state/watcher.log 2>&1
```

The first entry runs forecasts every two hours. The second continuously refreshes price,
lifecycle, and official-source context without forecasting.

Cron does not reset the account. `PAPER_STARTING_CASH` initializes `paper_account` only when the
SQLite database is new. Every later process opens the same database and continues from its stored
cash, positions, fills, and decisions.

## Health checks

```bash
prophet-trader doctor
prophet-trader status --limit 20
```

Review these status fields:

- latest cycle timestamps and completion status;
- recent forecast request status (`completed`, `unknown`, `failed`, or `parse_failed`);
- daily estimated provider spend;
- circuit state and reason;
- decisions and paper orders;
- recent context events.

The watcher log may contain source warnings for blocked, slow, or unsupported pages. Those
warnings do not trigger forecasts and do not stop market/lifecycle refreshes.

## Kill switch

Create the configured kill-switch file before the next cycle:

```bash
touch STOP
```

The engine refuses to forecast or rebalance while it exists. Remove it only after reviewing the
failure and current shadow positions.

## Database continuity

The default database is `state/trader.sqlite3`, with SQLite WAL sidecars during operation. Keep
the database and its sidecars together when moving an active installation. Stop cron or the
daemon before taking a filesystem copy.

To start an intentionally separate experiment, configure a different `STATE_DB_PATH`; do not
overwrite the existing journal.

## Cost controls

The system reserves `FORECAST_RESERVE_COST_DOLLARS` before each call and refuses a reservation
that could cross `DAILY_FORECAST_SPEND_LIMIT`. Keep the reservation large enough for the chosen
model, output ceiling, and grounded searches. The stored post-response estimate is based on
reported token and search usage.
