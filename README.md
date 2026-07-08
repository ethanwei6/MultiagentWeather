# MultiagentWeather

Research tooling for daily-high temperature prediction-market analysis. The codebase contains data-pipeline, weather-model validation, backtesting, settlement-audit, and forward-shadow recorder modules for studying daily-high weather markets.

This public repository is code-first. It intentionally excludes private agent briefs, local memory files, credentials, raw market/weather data, SQLite stores, live ledgers, generated reports, caches, and other machine-local artifacts.

## Repository Layout

- `daily_weather_pipeline/` - pipeline, modeling, validation, backtesting, settlement, and shadow-recorder modules.
- `tests/` - unit tests covering the public source tree.
- `docs/` - public project notes, data policy, and security guidance.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m unittest discover -s tests
python3 -m daily_weather_pipeline --help
```

Most offline validation commands expect local data artifacts under `work/`. Those artifacts are not included in the public repository because they are large, generated, and may contain private operational context.

Online data-pull commands require local API credentials supplied through environment variables or a local `.env` file. Credentials must never be committed.

## Public Scope

The public repo is intended to preserve the reproducible engineering surface:

- schema construction and artifact builders
- weather-model validation utilities
- anti-overfit and robustness harnesses
- backtesting framework code
- settlement-label audit tooling
- forward-shadow recorder logic
- unit tests for the above

Private operational runbooks, live trading keys, agent handoff briefs, and generated research artifacts stay outside version control.

## Validation

Before pushing, run:

```bash
python3 -m py_compile daily_weather_pipeline/*.py
python3 -m unittest discover -s tests
```

Also scan the staged files for credentials. The release process for this repo should treat any high-confidence secret finding as a hard stop.

