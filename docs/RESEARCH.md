# Research package

The package targets Python 3.12 on a POSIX system. Parquet operations use Polars 1.36.1. SQLite, the decimal S01 allocator, the empirical probability routines and the test fixtures use the standard library. The corpus writer uses POSIX advisory locks and atomic file replacement.

| Module | Public research functionality |
| --- | --- |
| `weather_model_runner`, `weather_model_improvement`, `weather_model_validation` | Score a temperature bucket with historical forecast errors, apply observed-high constraints, check label availability, and verify the origin of the supplied baseline package. |
| `weather_corpus_snapshot`, `weather_corpus_coverage`, `weather_corpus_extension` | Construct and check immutable SQLite corpus snapshots, evaluate weather coverage by city and source, and promote a validated update. |
| `live_forward_shadow_recorder`, `live_forward_shadow_scoring`, `shadow_risk` | Record timestamped quotes and forecasts, score a decision, and apply a dry-run pre-trade check. The executor returns an assessment without submitting an order. |
| `backtesting_engine`, `venue_execution` | Simulate fills and fees, walk the order book for all-or-none fills, reserve capital, and assess a fixed-cent adverse-price sensitivity. |
| `s01_parity_engine` | Evaluate historical S01 signal eligibility, scheduled decision windows, daily cash sleeves, venue minimums, and joint worst-case loss across a city–date event. |
| `physical_weather_contract` | Check settlement-source station identities, local-day GFS coverage, forecast availability and consistency between physical observations and market rules. This belongs to a later research line. |

## Reproduce an S01 risk calculation

This synthetic NO position has five shares, a 60¢ fill, and a 1¢ total fee. The two scenario payouts explain why joint event risk depends on contracts held together.

```python
from decimal import Decimal as D
from daily_weather_pipeline.s01_parity_engine import (
    PositionLeg, joint_event_maximum_loss, scenario_pnl,
)

leg = PositionLeg(
    event_id="sample-city-date", market_id="temperature-bucket",
    token_id="fixture-no", side="NO", shares=D("5"),
    gross_notional_usd=D("3"), fee_usd=D("0.01"), cash_outlay_usd=D("3.01"),
)
assert scenario_pnl([leg], "temperature-bucket") == D("-3.01")
assert scenario_pnl([leg], "some-other-bucket") == D("1.99")
assert joint_event_maximum_loss([leg], ["temperature-bucket"]) == D("3.01")
```

## Inputs for full research runs

Run `python -m daily_weather_pipeline --help` to list ingestion and corpus commands. `python -m daily_weather_pipeline --work-root /path/to/experiment init` creates empty research stores at the specified location. Full historical validation and replay require compatible market definitions, weather snapshots, quotes and outcomes acquired by the researcher. Some legacy replay configurations expect the historical artifact names specified in the module.

Baseline forecast generation and legacy ingestion integrate with a separately supplied checkout containing the `weather_strategy/` Python package. Configure its parent directory with `--source-repo /path/to/checkout` where supported, or with `WEATHER_SOURCE_REPO` before starting Python. The older `DAILYWEATHER_SOURCE_REPO` variable is also accepted. The default is `vendor/weather_strategy` relative to the working directory. An ordinary installation of this public package does not provide that separate checkout or the historical data. Corpus storage tests use a neutral synthetic baseline, and separate tests check import provenance and model arithmetic.

The public recorder exposes read-only market data and scoring. The private order transport, account ledger, runtime deployment bindings, immutable archive identity manifests, and archive-specific decisive execution-proof replay are outside this release. The portable S01 engine reproduces the policy calculations. Historical execution requires the original decision state, fills and reconciliations. The public showcase uses synthetic trade examples and retains the aggregate historical model audit as a separate evaluation.

## Offline verification

```sh
python -m unittest discover -s tests
```

Tests exercise synthetic SQLite corpora, probability and time cutoffs, fill-or-kill depth, cash reservations, S01 thresholds and joint event risk. The two small NOAA fixtures are public station metadata and GFS object/index/range metadata. Each file records its source, observation timestamp and integrity hashes. The fixtures support offline station and forecast-provenance checks. They contain no private market records or bulk weather data.
