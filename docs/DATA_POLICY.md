# Data Policy

This repository does not track bulk market data, weather data, generated model artifacts, live ledgers, or backtest reports.

Excluded local paths include:

- `work/`
- `data/`
- `reports/`
- `logs/`
- `cache/`
- `vendor/`
- `*.sqlite`
- `*.parquet`
- `*.jsonl`
- `*.jsonl.gz`

The code can generate or consume these artifacts locally, but they should remain machine-local unless a specific sanitized dataset is intentionally packaged later.

## Rationale

The generated artifacts are large, expensive to reproduce, and may contain operational context such as raw API payload paths, live/paper ledger details, or private research notes. Keeping them out of the public repository reduces accidental leakage and keeps the codebase reviewable.

## Adding Public Data Later

If a small sample dataset is added later, it should be:

- synthetic or explicitly public
- small enough for normal Git history
- free of API keys, wallet identifiers, live ledger details, and private run paths
- documented with source, license, and generation steps

