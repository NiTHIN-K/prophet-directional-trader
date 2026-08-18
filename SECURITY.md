# Security Policy

## Reporting a vulnerability

Do not disclose credential-handling, signature, scheduler, or execution vulnerabilities in a
public issue. Use the repository’s private vulnerability-reporting form under the **Security**
tab and include reproduction steps, affected configuration, and expected impact.

Never include API keys, private-key material, `.env` contents, or a production database in a
report. Revoke any credential that may have been exposed before sharing redacted diagnostics.

## Operational boundaries

- Strict replication mode permits paper trading only.
- Test environments reject forecasting and scheduler startup.
- Local secrets, ledgers, logs, and the kill-switch file are excluded from version control.
- Official-source fetching rejects local and non-public IP targets and bounds response size.
- Unknown provider outcomes are not retried automatically.
