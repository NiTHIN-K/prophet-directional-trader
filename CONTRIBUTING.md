# Contributing

Contributions are welcome when they preserve the project’s paper-only safety properties and
auditable request lifecycle.

## Local setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -v
```

Tests must remain offline. Use injected clients and temporary SQLite databases; never make a
provider request or start the scheduler from a test process.

## Change guidelines

1. Add regression coverage for behavior changes.
2. Keep provider calls deduplicated, budget-reserved, and journaled before parsing.
3. Preserve timeout-as-unknown and no-automatic-retry semantics.
4. Keep event polling context-only.
5. Keep strict replication mode incapable of live execution.
6. Do not commit credentials, private keys, databases, or runtime logs.

Run the complete suite before opening a pull request:

```bash
python -m compileall -q src tests
python -m unittest discover -s tests -v
```
