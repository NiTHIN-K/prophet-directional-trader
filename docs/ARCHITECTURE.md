# Architecture

Prophet Directional Trader separates scheduled decisions from continuous context refreshes. This
keeps the paper’s fixed cadence intact while retaining current market and settlement information.

## Scheduled decision path

1. Acquire the SQLite scheduler lease.
2. Reconcile existing inventory and any unresolved order state.
3. Refresh Kalshi market and event metadata.
4. Apply lifecycle, horizon, spread, and displayed-depth eligibility gates.
5. For previously traded markets, require a 10¢ move in the executable price for the last fill
   side.
6. Reserve daily spend and insert the unique forecast request before contacting the provider.
7. Persist the raw response and usage metadata before parsing structured JSON.
8. Refresh the quote after forecasting, then derive the Brier target from executable bid/ask
   prices.
9. Apply portfolio and order risk limits.
10. Simulate the fill and atomically update cash, positions, orders, and fill references.

No eligible market receives more than one provider call per cycle. The database uniqueness key
is `(ticker, two_hour_slot, model, prompt_version, context_hash)`, and a second unique index limits
each ticker to one request per model/prompt slot even if context changes during that slot.

## Context path

The watcher can poll much more frequently than the forecast loop. It records only:

- a 3¢ or larger executable midpoint movement;
- an official-source or scheduled-release change;
- a market metadata or lifecycle change.

Watcher events update the context snapshot in SQLite. They never invoke the forecast provider,
queue a request, or bypass the two-hour decision cadence. Inactive and terminal books cannot
replace the last valid executable price reference.

## Storage model

SQLite is the source of truth for runtime state:

| Table | Responsibility |
| --- | --- |
| `cycles` | Scheduler-run status and summary |
| `forecast_requests` | Deduplication, request lifecycle, tokens, duration, cost, raw response |
| `forecast_circuit` | Fail-closed provider availability state |
| `forecasts` | Parsed probabilities, rationale, confidence, and evidence |
| `decisions` | Signal, target, order intent, allow/deny result, and reason |
| `orders` | Paper fills and any reconciled order metadata |
| `paper_account` | Durable shadow cash balance |
| `paper_positions` | Filled net contracts and risk by ticker |
| `market_state` | Last fill price, side, and timestamp |
| `event_watch_state` | Last valid price and metadata/source baselines |
| `event_triggers` | Context-only change journal |
| `scheduler_leader` | Single-leader lease |

WAL mode provides durable local journaling while allowing status reads during normal operation.

## Failure semantics

- **Provider deadline:** mark the request `UNKNOWN`, cancel queued work for the cycle, and do not
  retry. The market remains blocked until the outcome is reconciled.
- **Insufficient quota:** open the circuit immediately and cancel queued requests.
- **Daily limit:** reservations fail closed before submission; reported usage above the ceiling
  also opens the circuit.
- **Invalid response:** keep the raw response and usage record, mark parsing failed, and place no
  order.
- **Stale quote:** refresh after forecasting and place no new exposure if guardrails no longer
  pass.
- **One-sided book:** new entries still require two-sided depth; held inventory may use the
  executable side needed to reduce risk.

## Trust boundaries

Kalshi market data and provider output are untrusted inputs. Structured schemas, decimal bounds,
public-source URL validation, quote refreshes, and order risk checks sit between those inputs and
the shadow ledger. Credentials are loaded only from the process environment or a local `.env`
file and are never stored in SQLite.
