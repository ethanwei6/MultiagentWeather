from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
import csv
import json
import math
import sqlite3
import sys
from collections import defaultdict
from contextlib import closing
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Callable, Iterable, Mapping, Optional
from zoneinfo import ZoneInfo

from .cli import DEFAULT_SOURCE_REPO
from .weather_corpus_snapshot import reject_corpus_database_sidecars


DEFAULT_START_DATE = "2025-10-13"
DEFAULT_END_DATE = "2026-06-29"
EPSILON = 1e-6


@dataclass(frozen=True)
class MarketRow:
    universe_row_id: str
    normalized_city: str
    country_region: Optional[str]
    latitude: float
    longitude: float
    timezone_name: str
    market_date_local: date
    bucket_label: str
    bucket_type: str
    threshold_low_f: Optional[float]
    threshold_high_f: Optional[float]
    units: Optional[str]
    resolved_outcome: Optional[int]


@dataclass(frozen=True)
class ForecastRow:
    forecast_id: str
    normalized_city: str
    country_region: Optional[str]
    latitude: float
    longitude: float
    timezone_name: str
    local_date: date
    decision_time_utc: datetime
    decision_hour_utc: int
    forecast_run_time_utc: Optional[datetime]
    source_name: str
    model_name: Optional[str]
    temperature_2m_max_f: float
    hourly_temperature_2m_max_f: Optional[float]
    feature_json: Mapping[str, Any]
    data_quality_flags: tuple[str, ...]


@dataclass(frozen=True)
class RealizedRow:
    normalized_city: str
    local_date: date
    temperature_2m_max_f: float
    hourly_temperature_2m_max_f: Optional[float]
    source_name: str
    data_quality_flags: tuple[str, ...]


@dataclass
class ValidationRow:
    row_id: str
    universe_row_id: str
    forecast_id: str
    normalized_city: str
    country_region: Optional[str]
    timezone_name: str
    market_date_local: date
    decision_time_utc: datetime
    decision_hour_utc: int
    local_decision_hour: float
    lead_days: int
    day_type: str
    source_name: str
    model_name: Optional[str]
    bucket_label: str
    bucket_type: str
    bucket_shape: str
    threshold_low_f: Optional[float]
    threshold_high_f: Optional[float]
    threshold_distance_f: Optional[float]
    threshold_distance_bucket: str
    realized_high_f: float
    forecast_high_f: float
    forecast_error_f: float
    outcome: int
    resolved_outcome: Optional[int]
    resolved_outcome_mismatch: bool
    current_probability: float
    mean_probability: float
    model_probabilities: dict[str, float]
    forecast_quality_flags: tuple[str, ...]
    realized_quality_flags: tuple[str, ...]
    feature_json: Mapping[str, Any]
    latitude: float
    longitude: float


@dataclass(frozen=True)
class DecisionQuoteRow:
    universe_row_id: str
    decision_time_utc: datetime
    side: str
    forecast_source_name: Optional[str]
    mid: Optional[float]
    bid: Optional[float]
    ask: Optional[float]
    spread: Optional[float]
    bid_size: Optional[float]
    ask_size: Optional[float]
    liquidity_depth: Optional[float]
    price_stale_seconds: Optional[float]
    resolved_outcome_yes: int
    quote_quality_flags: tuple[str, ...]


def resolve_training_corpus_end_date(
    end_date: Optional[date | str] = None,
    *,
    now_utc: Optional[datetime] = None,
) -> date:
    if end_date is not None:
        if isinstance(end_date, date):
            return end_date
        return date.fromisoformat(str(end_date))
    now = now_utc or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    return now.astimezone(timezone.utc).date()


class ReliabilityCalibrator:
    def __init__(
        self,
        *,
        name: str,
        group_fn: Callable[[ValidationRow], str],
        smoothing: float = 200.0,
        min_bin_count: int = 25,
    ) -> None:
        self.name = name
        self.group_fn = group_fn
        self.smoothing = float(smoothing)
        self.min_bin_count = int(min_bin_count)
        self.group_bins: dict[str, dict[int, dict[str, float]]] = {}
        self.global_bins: dict[int, dict[str, float]] = {}

    def fit(self, rows: list[ValidationRow], probabilities: Mapping[str, float]) -> None:
        grouped: dict[str, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
        global_bins: dict[int, list[int]] = defaultdict(list)
        for row in rows:
            probability = probabilities[row.row_id]
            bucket = _probability_bin(probability)
            grouped[self.group_fn(row)][bucket].append(row.outcome)
            global_bins[bucket].append(row.outcome)
        self.group_bins = {
            group: {
                bucket: {
                    "n": len(values),
                    "actual_rate": sum(values) / len(values),
                }
                for bucket, values in buckets.items()
                if values
            }
            for group, buckets in grouped.items()
        }
        self.global_bins = {
            bucket: {
                "n": len(values),
                "actual_rate": sum(values) / len(values),
            }
            for bucket, values in global_bins.items()
            if values
        }

    def transform(self, row: ValidationRow, probability: float) -> float:
        bucket = _probability_bin(probability)
        group = self.group_fn(row)
        stats = (self.group_bins.get(group) or {}).get(bucket)
        if stats is None or stats["n"] < self.min_bin_count:
            stats = self.global_bins.get(bucket)
        if stats is None:
            return _clip_probability(probability)
        n = float(stats["n"])
        empirical = float(stats["actual_rate"])
        calibrated = (empirical * n + probability * self.smoothing) / (n + self.smoothing)
        return _clip_probability(calibrated)

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "smoothing": self.smoothing,
            "min_bin_count": self.min_bin_count,
            "group_bins": self.group_bins,
            "global_bins": self.global_bins,
        }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Validate DailyWeather weather probabilities against realized highs")
    parser.add_argument("--work-root", default=".")
    parser.add_argument("--source-repo", default=str(DEFAULT_SOURCE_REPO))
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=DEFAULT_END_DATE)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--output-dir", default="work/reports/weather_model_validation")
    parser.add_argument("--candidate-dir", default="work/data/backtest/weather_model_candidates")
    parser.add_argument("--max-coordinate-epochs", type=int, default=3)
    args = parser.parse_args(argv)

    work_root = Path(args.work_root).resolve()
    source_repo = Path(args.source_repo).expanduser().resolve()
    strategy = _load_weather_strategy(source_repo)
    start = date.fromisoformat(args.start_date)
    end = date.fromisoformat(args.end_date)
    output = run_validation(
        work_root=work_root,
        source_repo=source_repo,
        strategy=strategy,
        start=start,
        end=end,
        train_fraction=args.train_fraction,
        output_dir=Path(args.output_dir),
        candidate_dir=Path(args.candidate_dir),
        max_coordinate_epochs=args.max_coordinate_epochs,
    )
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


def run_validation(
    *,
    work_root: Path,
    source_repo: Path,
    strategy: Mapping[str, Any],
    start: date,
    end: date,
    train_fraction: float,
    output_dir: Path,
    candidate_dir: Path,
    max_coordinate_epochs: int,
) -> dict[str, Any]:
    paths = _artifact_paths(work_root)
    markets = _load_markets(paths["market_db"], start=start, end=end)
    forecasts = _load_forecasts(paths["weather_db"], start=start, end=end)
    realized = _load_realized(paths["weather_db"], start=start, end=end)
    rows = _build_validation_rows(strategy, markets, forecasts, realized)
    if len(rows) < 100:
        raise ValueError(f"Need at least 100 validation rows; got {len(rows)}")
    train_rows, test_rows, split = _time_split(rows, train_fraction=train_fraction)
    baseline_probabilities = {row.row_id: row.current_probability for row in rows}

    fitted_candidates = _fit_candidates(
        strategy,
        rows=rows,
        train_rows=train_rows,
        test_rows=test_rows,
        baseline_probabilities=baseline_probabilities,
        max_coordinate_epochs=max_coordinate_epochs,
    )
    baseline_metrics = {
        "train": _metrics(train_rows, baseline_probabilities),
        "test": _metrics(test_rows, baseline_probabilities),
        "all": _metrics(rows, baseline_probabilities),
    }
    accepted = [
        candidate
        for candidate in fitted_candidates
        if _improves_both(candidate["metrics"]["test"], baseline_metrics["test"])
    ]
    accepted.sort(key=lambda item: (item["metrics"]["test"]["brier"], item["metrics"]["test"]["log_loss"]))
    best_probabilities = accepted[0]["probabilities"] if accepted else baseline_probabilities
    best_name = accepted[0]["name"] if accepted else "current_weighted_consensus"

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_dir = (work_root / output_dir).resolve() if not output_dir.is_absolute() else output_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    candidate_output_dir = (work_root / candidate_dir).resolve() if not candidate_dir.is_absolute() else candidate_dir
    candidate_output_dir.mkdir(parents=True, exist_ok=True)

    diagnostics = _diagnostics(
        rows=rows,
        train_rows=train_rows,
        test_rows=test_rows,
        baseline_probabilities=baseline_probabilities,
        best_probabilities=best_probabilities,
        best_name=best_name,
        fitted_candidates=fitted_candidates,
        accepted=accepted,
    )
    forecast_high_metrics = _forecast_high_metrics(forecasts, realized, split["cutoff_decision_time_utc"])
    model_family_metrics = _model_probability_metrics(test_rows)
    market_diagnostics = _market_edge_diagnostics(
        paths["decision_db"],
        test_rows=test_rows,
        baseline_probabilities=baseline_probabilities,
        candidate_probabilities=best_probabilities,
        candidate_name=best_name,
    )

    summary = {
        "role": "Weather Model Agent",
        "source_repo": str(source_repo),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_start_date": start.isoformat(),
        "target_end_date": end.isoformat(),
        "artifacts_used": {key: str(value) for key, value in paths.items()},
        "input_counts": {
            "market_rows": len(markets),
            "forecast_rows": len(forecasts),
            "realized_rows": len(realized),
            "validation_rows": len(rows),
            "train_rows": len(train_rows),
            "test_rows": len(test_rows),
        },
        "split": split,
        "baseline_metrics": baseline_metrics,
        "best_probability_series": best_name,
        "accepted_candidate_names": [candidate["name"] for candidate in accepted],
        "candidate_comparison": [
            _candidate_summary(candidate, baseline_metrics["test"])
            for candidate in fitted_candidates
        ],
        "forecast_high_metrics": forecast_high_metrics,
        "diagnostics": diagnostics["summary"],
        "model_family_metrics": model_family_metrics,
        "old_best_source_model_family_test": (model_family_metrics.get("by_source_model_family") or [None])[0],
        "market_relative_diagnostics": market_diagnostics["summary"],
        "validation_notes": _validation_notes(rows, forecasts, accepted),
        "market_edge_claim": "Quote-covered diagnostics are reported where compact point-in-time quotes exist. They are not a market-beating claim and do not include full execution/trading replay.",
    }

    artifacts = _write_reports(
        timestamp=timestamp,
        report_dir=report_dir,
        candidate_output_dir=candidate_output_dir,
        summary=summary,
        diagnostics=diagnostics,
        model_family_metrics=model_family_metrics,
        accepted=accepted,
        baseline_probabilities=baseline_probabilities,
        best_probabilities=best_probabilities,
        market_diagnostics=market_diagnostics,
        rows=rows,
        train_rows=train_rows,
        test_rows=test_rows,
    )
    summary["output_artifacts"] = artifacts
    summary_path = Path(artifacts["accuracy_json"])
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def _artifact_paths(work_root: Path) -> dict[str, Path]:
    base = work_root / "work" / "data" / "backtest"
    return {
        "market_db": base / "daily_weather_market_universe.sqlite",
        "weather_db": base / "daily_weather_weather_history.sqlite",
        "decision_db": base / "daily_weather_decision_artifact.sqlite",
    }


def _is_weather_strategy_module_name(name: str) -> bool:
    return name == "weather_strategy" or name.startswith("weather_strategy.")


def _resolved_module_origin_path(value: Any, *, module_name: str) -> Path:
    try:
        return Path(value).expanduser().resolve(strict=True)
    except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "weather_strategy module origin is missing or unreadable: "
            f"{module_name}:{value!r}"
        ) from exc


def _module_origin_paths(module_name: str, module: Any) -> set[Path]:
    if module is None:
        raise RuntimeError(
            f"weather_strategy module origin is unprovable: {module_name}:None"
        )
    origins: set[Path] = set()
    module_file = getattr(module, "__file__", None)
    if module_file:
        origins.add(
            _resolved_module_origin_path(module_file, module_name=module_name)
        )

    spec = getattr(module, "__spec__", None)
    spec_origin = getattr(spec, "origin", None)
    if spec_origin and str(spec_origin).lower() not in {
        "built-in",
        "frozen",
        "namespace",
    }:
        origins.add(
            _resolved_module_origin_path(spec_origin, module_name=module_name)
        )

    package_paths = getattr(module, "__path__", None)
    if package_paths is None and spec is not None:
        package_paths = getattr(spec, "submodule_search_locations", None)
    if package_paths is not None:
        try:
            for package_path in package_paths:
                origins.add(
                    _resolved_module_origin_path(
                        package_path,
                        module_name=module_name,
                    )
                )
        except TypeError as exc:
            raise RuntimeError(
                f"weather_strategy package path is unprovable: {module_name}"
            ) from exc
    if not origins:
        raise RuntimeError(
            f"weather_strategy module origin is unprovable: {module_name}"
        )
    return origins


def _assert_weather_strategy_module_origins(package_root: Path) -> None:
    expected_root = Path(package_root).resolve(strict=True)
    for module_name, module in sorted(sys.modules.items()):
        if not _is_weather_strategy_module_name(module_name):
            continue
        for origin in _module_origin_paths(module_name, module):
            try:
                origin.relative_to(expected_root)
            except ValueError as exc:
                raise RuntimeError(
                    "weather_strategy module resolved outside the exact vendored "
                    f"package tree: {module_name}:{origin}:expected={expected_root}"
                ) from exc


def _resolved_sys_path_entry(value: Any) -> Optional[Path]:
    try:
        return Path(value).expanduser().resolve()
    except (OSError, RuntimeError, TypeError, ValueError):
        return None


def _load_weather_strategy(source_repo: Path) -> dict[str, Any]:
    try:
        resolved_source_repo = Path(source_repo).expanduser().resolve(strict=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Source repo not found: {source_repo}") from exc
    if not resolved_source_repo.is_dir():
        raise FileNotFoundError(f"Source repo is not a directory: {source_repo}")
    package_path = resolved_source_repo / "weather_strategy"
    if package_path.is_symlink():
        raise RuntimeError(
            f"weather_strategy package may not be a symlink: {package_path}"
        )
    try:
        package_root = package_path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"weather_strategy package not found in source repo: {package_path}"
        ) from exc
    if not package_root.is_dir() or package_root.parent != resolved_source_repo:
        raise RuntimeError(
            "weather_strategy package must be the exact source_repo/weather_strategy "
            f"directory: {package_root}"
        )

    original_sys_path = list(sys.path)
    original_modules = {
        name: module
        for name, module in sys.modules.items()
        if _is_weather_strategy_module_name(name)
    }
    sys.path[:] = [
        str(resolved_source_repo),
        *[
            entry
            for entry in original_sys_path
            if _resolved_sys_path_entry(entry) != resolved_source_repo
        ],
    ]
    try:
        _assert_weather_strategy_module_origins(package_root)
        from weather_strategy.forecast import (  # type: ignore
            ConsensusForecastEngine,
            EmpiricalEnsembleModel,
            FeatureAwareNormalModel,
            HourlyCurveMaxModel,
            KernelSmoothingModel,
            ParametricNormalModel,
        )
        from weather_strategy.models import (  # type: ignore
            CityConfig,
            ForecastDistribution,
            TemperatureBucket,
        )

        engine = ConsensusForecastEngine()
        models = (
            EmpiricalEnsembleModel(),
            KernelSmoothingModel(sigma_floor_f=1.5, name="kernel_tight"),
            KernelSmoothingModel(sigma_floor_f=3.0, name="kernel_wide"),
            ParametricNormalModel(
                sigma_floor_f=2.0,
                sigma_multiplier=1.0,
                name="normal_parametric",
            ),
            ParametricNormalModel(
                sigma_floor_f=4.0,
                sigma_multiplier=1.75,
                name="normal_conservative",
            ),
            FeatureAwareNormalModel(sigma_floor_f=3.0),
            HourlyCurveMaxModel(sigma_floor_f=2.5),
        )
        _assert_weather_strategy_module_origins(package_root)
        return {
            "CityConfig": CityConfig,
            "ForecastDistribution": ForecastDistribution,
            "TemperatureBucket": TemperatureBucket,
            "models": models,
            "source_weights": dict(engine.source_weights),
            "model_weights": dict(engine.model_weights),
        }
    except Exception:
        for name in tuple(sys.modules):
            if _is_weather_strategy_module_name(name):
                sys.modules.pop(name, None)
        sys.modules.update(original_modules)
        raise
    finally:
        sys.path[:] = original_sys_path


def _connect_readonly(
    path: Path,
    *,
    immutable: bool = False,
    reject_sidecars: bool = False,
    sidecar_reason_prefix: str = "corpus_database",
    allow_zero_length_wal: bool = False,
) -> sqlite3.Connection:
    if not path.exists():
        raise FileNotFoundError(path)
    if reject_sidecars:
        reject_corpus_database_sidecars(
            path,
            reason_prefix=sidecar_reason_prefix,
            allow_zero_length_wal=allow_zero_length_wal,
        )
    suffix = "&immutable=1" if immutable else ""
    conn = sqlite3.connect(f"file:{path}?mode=ro{suffix}", uri=True, timeout=30)
    try:
        if reject_sidecars:
            reject_corpus_database_sidecars(
                path,
                reason_prefix=sidecar_reason_prefix,
                allow_zero_length_wal=allow_zero_length_wal,
            )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("PRAGMA query_only=ON")
    except Exception:
        conn.close()
        raise
    return conn


def _load_markets(
    path: Path,
    *,
    start: date,
    end: date,
    immutable: bool = False,
    reject_sidecars: bool = False,
    sidecar_reason_prefix: str = "corpus_database",
    allow_zero_length_wal: bool = False,
) -> list[MarketRow]:
    with closing(
        _connect_readonly(
            path,
            immutable=immutable,
            reject_sidecars=reject_sidecars,
            sidecar_reason_prefix=sidecar_reason_prefix,
            allow_zero_length_wal=allow_zero_length_wal,
        )
    ) as conn:
        rows = conn.execute(
            """
            SELECT universe_row_id, normalized_city, country_region, latitude, longitude, timezone,
                   market_date_local, bucket_label, bucket_type, threshold_low_f, threshold_high_f,
                   units, resolved_outcome
            FROM market_universe
            WHERE market_date_local >= ?
              AND market_date_local <= ?
              AND normalized_city IS NOT NULL
              AND timezone IS NOT NULL
              AND latitude IS NOT NULL
              AND longitude IS NOT NULL
              AND bucket_label IS NOT NULL
            ORDER BY market_date_local, normalized_city, universe_row_id
            """,
            (start.isoformat(), end.isoformat()),
        ).fetchall()
    return [
        MarketRow(
            universe_row_id=str(row["universe_row_id"]),
            normalized_city=str(row["normalized_city"]),
            country_region=row["country_region"],
            latitude=float(row["latitude"]),
            longitude=float(row["longitude"]),
            timezone_name=str(row["timezone"]),
            market_date_local=date.fromisoformat(str(row["market_date_local"])),
            bucket_label=str(row["bucket_label"]),
            bucket_type=str(row["bucket_type"] or "unknown"),
            threshold_low_f=_optional_float(row["threshold_low_f"]),
            threshold_high_f=_optional_float(row["threshold_high_f"]),
            units=row["units"],
            resolved_outcome=_optional_int(row["resolved_outcome"]),
        )
        for row in rows
    ]


def _load_forecasts(
    path: Path,
    *,
    start: date,
    end: date,
    immutable: bool = False,
    reject_sidecars: bool = False,
    sidecar_reason_prefix: str = "corpus_database",
    allow_zero_length_wal: bool = False,
) -> list[ForecastRow]:
    with closing(
        _connect_readonly(
            path,
            immutable=immutable,
            reject_sidecars=reject_sidecars,
            sidecar_reason_prefix=sidecar_reason_prefix,
            allow_zero_length_wal=allow_zero_length_wal,
        )
    ) as conn:
        rows = conn.execute(
            """
            SELECT forecast_id, normalized_city, country_region, latitude, longitude, timezone,
                   local_date, decision_time_utc, forecast_run_time_utc, source_name, model_name,
                   temperature_2m_max_f, hourly_temperature_2m_max_f, feature_json, data_quality_flags
            FROM forecast_snapshots
            WHERE local_date >= ?
              AND local_date <= ?
              AND temperature_2m_max_f IS NOT NULL
              AND decision_time_utc IS NOT NULL
            ORDER BY decision_time_utc, local_date, normalized_city, source_name
            """,
            (start.isoformat(), end.isoformat()),
        ).fetchall()
    forecasts: list[ForecastRow] = []
    for row in rows:
        decision_time = _parse_datetime(str(row["decision_time_utc"]))
        if decision_time is None:
            continue
        forecasts.append(
            ForecastRow(
                forecast_id=str(row["forecast_id"]),
                normalized_city=str(row["normalized_city"]),
                country_region=row["country_region"],
                latitude=float(row["latitude"]),
                longitude=float(row["longitude"]),
                timezone_name=str(row["timezone"]),
                local_date=date.fromisoformat(str(row["local_date"])),
                decision_time_utc=decision_time,
                decision_hour_utc=decision_time.hour,
                forecast_run_time_utc=_parse_datetime(row["forecast_run_time_utc"]),
                source_name=str(row["source_name"]),
                model_name=row["model_name"],
                temperature_2m_max_f=float(row["temperature_2m_max_f"]),
                hourly_temperature_2m_max_f=_optional_float(row["hourly_temperature_2m_max_f"]),
                feature_json=_json_mapping(row["feature_json"]),
                data_quality_flags=_json_flags(row["data_quality_flags"]),
            )
        )
    return forecasts


def _load_realized(
    path: Path,
    *,
    start: date,
    end: date,
    immutable: bool = False,
    reject_sidecars: bool = False,
    sidecar_reason_prefix: str = "corpus_database",
    allow_zero_length_wal: bool = False,
) -> dict[tuple[str, date], RealizedRow]:
    with closing(
        _connect_readonly(
            path,
            immutable=immutable,
            reject_sidecars=reject_sidecars,
            sidecar_reason_prefix=sidecar_reason_prefix,
            allow_zero_length_wal=allow_zero_length_wal,
        )
    ) as conn:
        rows = conn.execute(
            """
            SELECT normalized_city, local_date, temperature_2m_max_f, hourly_temperature_2m_max_f,
                   source_name, data_quality_flags
            FROM realized_daily_weather
            WHERE local_date >= ?
              AND local_date <= ?
              AND temperature_2m_max_f IS NOT NULL
            ORDER BY local_date, normalized_city, source_name
            """,
            (start.isoformat(), end.isoformat()),
        ).fetchall()
    realized: dict[tuple[str, date], RealizedRow] = {}
    for row in rows:
        local_date = date.fromisoformat(str(row["local_date"]))
        key = (str(row["normalized_city"]), local_date)
        realized.setdefault(
            key,
            RealizedRow(
                normalized_city=str(row["normalized_city"]),
                local_date=local_date,
                temperature_2m_max_f=float(row["temperature_2m_max_f"]),
                hourly_temperature_2m_max_f=_optional_float(row["hourly_temperature_2m_max_f"]),
                source_name=str(row["source_name"]),
                data_quality_flags=_json_flags(row["data_quality_flags"]),
            ),
        )
    return realized


def _build_validation_rows(
    strategy: Mapping[str, Any],
    markets: list[MarketRow],
    forecasts: list[ForecastRow],
    realized: Mapping[tuple[str, date], RealizedRow],
) -> list[ValidationRow]:
    markets_by_city_date: dict[tuple[str, date], list[MarketRow]] = defaultdict(list)
    for market in markets:
        markets_by_city_date[(market.normalized_city, market.market_date_local)].append(market)
    rows: list[ValidationRow] = []
    for forecast in forecasts:
        matching_markets = markets_by_city_date.get((forecast.normalized_city, forecast.local_date))
        realized_row = realized.get((forecast.normalized_city, forecast.local_date))
        if not matching_markets or realized_row is None:
            continue
        for market in matching_markets:
            lead_days, local_hour = _lead_days_and_local_hour(forecast.decision_time_utc, market.market_date_local, market.timezone_name)
            if lead_days < 0 or lead_days > 3:
                continue
            model_probabilities = _score_model_probabilities(strategy, market, forecast, bias_f=0.0)
            if not model_probabilities:
                continue
            current_probability = _weighted_consensus_probability(
                model_probabilities,
                source_weights=strategy["source_weights"],
                model_weights=strategy["model_weights"],
            )
            mean_probability = _clip_probability(sum(model_probabilities.values()) / len(model_probabilities))
            actual_high = realized_row.temperature_2m_max_f
            outcome = 1 if _bucket_contains(actual_high, market.threshold_low_f, market.threshold_high_f) else 0
            mismatch = market.resolved_outcome is not None and int(market.resolved_outcome) != outcome
            distance = _threshold_distance(forecast.temperature_2m_max_f, market.threshold_low_f, market.threshold_high_f)
            rows.append(
                ValidationRow(
                    row_id=f"{market.universe_row_id}|{forecast.forecast_id}",
                    universe_row_id=market.universe_row_id,
                    forecast_id=forecast.forecast_id,
                    normalized_city=market.normalized_city,
                    country_region=market.country_region,
                    timezone_name=market.timezone_name,
                    market_date_local=market.market_date_local,
                    decision_time_utc=forecast.decision_time_utc,
                    decision_hour_utc=forecast.decision_hour_utc,
                    local_decision_hour=local_hour,
                    lead_days=lead_days,
                    day_type=_day_type(lead_days),
                    source_name=forecast.source_name,
                    model_name=forecast.model_name,
                    bucket_label=market.bucket_label,
                    bucket_type=market.bucket_type,
                    bucket_shape=_bucket_shape(market.threshold_low_f, market.threshold_high_f),
                    threshold_low_f=market.threshold_low_f,
                    threshold_high_f=market.threshold_high_f,
                    threshold_distance_f=distance,
                    threshold_distance_bucket=_distance_bucket(distance),
                    realized_high_f=actual_high,
                    forecast_high_f=forecast.temperature_2m_max_f,
                    forecast_error_f=forecast.temperature_2m_max_f - actual_high,
                    outcome=outcome,
                    resolved_outcome=market.resolved_outcome,
                    resolved_outcome_mismatch=mismatch,
                    current_probability=current_probability,
                    mean_probability=mean_probability,
                    model_probabilities=model_probabilities,
                    forecast_quality_flags=forecast.data_quality_flags,
                    realized_quality_flags=realized_row.data_quality_flags,
                    feature_json=forecast.feature_json,
                    latitude=market.latitude,
                    longitude=market.longitude,
                )
            )
    rows.sort(key=lambda row: (row.decision_time_utc, row.market_date_local, row.normalized_city, row.universe_row_id))
    return rows


def _score_model_probabilities(
    strategy: Mapping[str, Any],
    market: MarketRow,
    forecast: ForecastRow,
    *,
    bias_f: float,
) -> dict[str, float]:
    CityConfig = strategy["CityConfig"]
    ForecastDistribution = strategy["ForecastDistribution"]
    TemperatureBucket = strategy["TemperatureBucket"]
    features = dict(forecast.feature_json or {})
    adjusted_daily = forecast.temperature_2m_max_f + bias_f
    features["temperature_2m_max"] = adjusted_daily
    if forecast.hourly_temperature_2m_max_f is not None:
        features["hourly_temperature_2m_max"] = forecast.hourly_temperature_2m_max_f + bias_f
    city = CityConfig(
        name=market.normalized_city,
        state=market.country_region or "",
        latitude=market.latitude,
        longitude=market.longitude,
        timezone=market.timezone_name,
    )
    distribution = ForecastDistribution(
        city=city,
        target_date=market.market_date_local,
        samples_f=(adjusted_daily,),
        generated_at=forecast.decision_time_utc,
        source=forecast.source_name,
        model_metadata={"features": features, "model_name": forecast.model_name},
    )
    bucket = TemperatureBucket(
        label=market.bucket_label,
        lower_f=market.threshold_low_f,
        upper_f=market.threshold_high_f,
        resolution_unit=market.units,
        resolution_precision=None,
    )
    probabilities: dict[str, float] = {}
    for model in strategy["models"]:
        values = model.probability_by_bucket(distribution, (bucket,))
        value = values.get(bucket.label)
        if value is not None:
            probabilities[f"{forecast.source_name}.{model.name}"] = _clip_probability(float(value))
    return probabilities


def _fit_candidates(
    strategy: Mapping[str, Any],
    *,
    rows: list[ValidationRow],
    train_rows: list[ValidationRow],
    test_rows: list[ValidationRow],
    baseline_probabilities: Mapping[str, float],
    max_coordinate_epochs: int,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []

    candidates.extend(_fit_residual_distribution_candidates(rows=rows, train_rows=train_rows, test_rows=test_rows))

    source_weights, family_weights = _fit_coordinate_weights(
        train_rows,
        source_weights=strategy["source_weights"],
        model_weights=strategy["model_weights"],
        max_epochs=max_coordinate_epochs,
    )
    family_weighted = {
        row.row_id: _weighted_consensus_probability(
            row.model_probabilities,
            source_weights=source_weights,
            model_weights={key: family_weights.get(_model_family(key), 1.0) for key in row.model_probabilities},
        )
        for row in rows
    }
    candidates.append(
        _candidate_payload(
            "source_and_family_weighted_consensus",
            rows,
            train_rows,
            test_rows,
            family_weighted,
            params={"source_weights": source_weights, "family_weights": family_weights},
        )
    )

    tail = _fit_tail_calibration(train_rows, baseline_probabilities)
    tail_probs = {
        row.row_id: _apply_tail_calibration(baseline_probabilities[row.row_id], tail)
        for row in rows
    }
    candidates.append(_candidate_payload("tail_shrinkage_calibration", rows, train_rows, test_rows, tail_probs, params=tail))

    for calibrator in (
        ReliabilityCalibrator(name="lead_time_reliability_calibration", group_fn=lambda row: f"lead:{row.lead_days}"),
        ReliabilityCalibrator(name="bucket_type_reliability_calibration", group_fn=lambda row: f"bucket_type:{row.bucket_type}"),
        ReliabilityCalibrator(name="decision_hour_reliability_calibration", group_fn=lambda row: f"utc_hour:{row.decision_hour_utc}"),
        ReliabilityCalibrator(name="time_of_day_reliability_calibration", group_fn=lambda row: f"{row.day_type}|utc:{row.decision_hour_utc}|local:{_local_hour_bucket(row.local_decision_hour)}"),
        ReliabilityCalibrator(name="threshold_distance_reliability_calibration", group_fn=lambda row: f"threshold_distance:{row.threshold_distance_bucket}"),
    ):
        calibrator.fit(train_rows, baseline_probabilities)
        calibrated = {
            row.row_id: calibrator.transform(row, baseline_probabilities[row.row_id])
            for row in rows
        }
        candidates.append(
            _candidate_payload(
                calibrator.name,
                rows,
                train_rows,
                test_rows,
                calibrated,
                params=calibrator.to_json(),
            )
        )

    bias_by_city = _fit_temperature_biases(train_rows, key_fn=lambda row: row.normalized_city)
    city_bias_probs = _bias_corrected_probabilities(strategy, rows, bias_by_city, key_fn=lambda row: row.normalized_city)
    candidates.append(
        _candidate_payload(
            "city_specific_temperature_bias_correction",
            rows,
            train_rows,
            test_rows,
            city_bias_probs,
            params={"biases_f": bias_by_city},
        )
    )

    bias_by_lead = _fit_temperature_biases(train_rows, key_fn=lambda row: f"lead:{row.lead_days}")
    lead_bias_probs = _bias_corrected_probabilities(strategy, rows, bias_by_lead, key_fn=lambda row: f"lead:{row.lead_days}")
    candidates.append(
        _candidate_payload(
            "lead_time_temperature_bias_correction",
            rows,
            train_rows,
            test_rows,
            lead_bias_probs,
            params={"biases_f": bias_by_lead},
        )
    )

    bias_by_city_lead = _fit_temperature_biases(train_rows, key_fn=lambda row: f"{row.normalized_city}|lead:{row.lead_days}", minimum_count=50)
    city_lead_bias_probs = _bias_corrected_probabilities(
        strategy,
        rows,
        bias_by_city_lead,
        key_fn=lambda row: f"{row.normalized_city}|lead:{row.lead_days}",
    )
    candidates.append(
        _candidate_payload(
            "city_and_lead_temperature_bias_correction",
            rows,
            train_rows,
            test_rows,
            city_lead_bias_probs,
            params={"biases_f": bias_by_city_lead},
        )
        )

    kernel_tight = _family_probabilities(rows, family_name="kernel_tight")
    for calibrator in (
        ReliabilityCalibrator(
            name="kernel_tight_threshold_distance_reliability_calibration",
            group_fn=lambda row: f"threshold_distance:{row.threshold_distance_bucket}",
            smoothing=100.0,
            min_bin_count=40,
        ),
        ReliabilityCalibrator(
            name="kernel_tight_bucket_time_distance_reliability_calibration",
            group_fn=lambda row: f"{row.bucket_type}|{row.day_type}|local:{_local_hour_bucket(row.local_decision_hour)}|dist:{row.threshold_distance_bucket}",
            smoothing=150.0,
            min_bin_count=50,
        ),
    ):
        calibrator.fit(train_rows, kernel_tight)
        calibrated = {
            row.row_id: calibrator.transform(row, kernel_tight[row.row_id])
            for row in rows
        }
        candidates.append(
            _candidate_payload(
                calibrator.name,
                rows,
                train_rows,
                test_rows,
                calibrated,
                params={
                    "base_probability_series": "old_model_family.kernel_tight",
                    **calibrator.to_json(),
                },
            )
        )

    return candidates


def _fit_residual_distribution_candidates(
    *,
    rows: list[ValidationRow],
    train_rows: list[ValidationRow],
    test_rows: list[ValidationRow],
) -> list[dict[str, Any]]:
    specs_by_name: dict[str, list[tuple[str, float, int, float, Callable[[ValidationRow], str]]]] = {
        "residual_global_empirical_distribution": [
            ("global", 1.0, 1, 1.0, lambda row: "__global__"),
        ],
        "residual_hierarchical_empirical_distribution": [
            ("city_lead", 0.45, 35, 120.0, lambda row: f"{row.normalized_city}|lead:{row.lead_days}"),
            ("city", 0.25, 35, 160.0, lambda row: row.normalized_city),
            ("lead_hour", 0.15, 45, 180.0, lambda row: f"lead:{row.lead_days}|local:{_local_hour_bucket(row.local_decision_hour)}"),
            ("lead", 0.10, 45, 180.0, lambda row: f"lead:{row.lead_days}"),
            ("global", 0.05, 1, 1.0, lambda row: "__global__"),
        ],
    }
    candidates: list[dict[str, Any]] = []
    for name, specs in specs_by_name.items():
        residual_sets = _collect_residual_sets(train_rows, specs)
        best_padding = _select_residual_padding(train_rows, residual_sets=residual_sets, specs=specs)
        probabilities = {
            row.row_id: _residual_distribution_probability(
                row,
                residual_sets=residual_sets,
                specs=specs,
                padding_f=best_padding,
            )
            for row in rows
        }
        params = _residual_distribution_params(
            residual_sets=residual_sets,
            specs=specs,
            padding_f=best_padding,
            train_rows=train_rows,
            probabilities=probabilities,
        )
        payload = _candidate_payload(name, rows, train_rows, test_rows, probabilities, params=params)
        payload["forecast_high_metrics"] = _residual_forecast_high_metrics(
            test_rows,
            residual_sets=residual_sets,
            specs=specs,
        )
        candidates.append(payload)

        calibrator = ReliabilityCalibrator(
            name=f"{name}_bucket_distance_reliability_calibration",
            group_fn=lambda row: f"{row.bucket_type}|{row.threshold_distance_bucket}",
            smoothing=120.0,
            min_bin_count=45,
        )
        calibrator.fit(train_rows, probabilities)
        calibrated = {
            row.row_id: calibrator.transform(row, probabilities[row.row_id])
            for row in rows
        }
        calibrated_payload = _candidate_payload(
            calibrator.name,
            rows,
            train_rows,
            test_rows,
            calibrated,
            params={
                "base_probability_series": name,
                "residual_distribution": params,
                "reliability": calibrator.to_json(),
            },
        )
        calibrated_payload["forecast_high_metrics"] = payload["forecast_high_metrics"]
        candidates.append(calibrated_payload)
    return candidates


def _collect_residual_sets(
    rows: list[ValidationRow],
    specs: list[tuple[str, float, int, float, Callable[[ValidationRow], str]]],
) -> dict[str, dict[str, tuple[float, ...]]]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    seen: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        residual = row.realized_high_f - row.forecast_high_f
        for label, _, _, _, key_fn in specs:
            key = key_fn(row)
            seen_key = (label, key)
            if row.forecast_id in seen[seen_key]:
                continue
            seen[seen_key].add(row.forecast_id)
            grouped[label][key].append(residual)
    return {
        label: {
            key: tuple(sorted(values))
            for key, values in values_by_key.items()
            if values
        }
        for label, values_by_key in grouped.items()
    }


def _select_residual_padding(
    rows: list[ValidationRow],
    *,
    residual_sets: Mapping[str, Mapping[str, tuple[float, ...]]],
    specs: list[tuple[str, float, int, float, Callable[[ValidationRow], str]]],
) -> float:
    best: Optional[tuple[float, float, float]] = None
    for padding in (0.0, 0.10, 0.25, 0.40, 0.60, 0.85, 1.10, 1.50):
        probabilities = {
            row.row_id: _residual_distribution_probability(
                row,
                residual_sets=residual_sets,
                specs=specs,
                padding_f=padding,
            )
            for row in rows
        }
        metrics = _metrics(rows, probabilities)
        score = float(metrics["log_loss"]) + float(metrics["brier"])
        if best is None or score < best[0]:
            best = (score, padding, float(metrics["brier"]))
    assert best is not None
    return float(best[1])


def _residual_distribution_probability(
    row: ValidationRow,
    *,
    residual_sets: Mapping[str, Mapping[str, tuple[float, ...]]],
    specs: list[tuple[str, float, int, float, Callable[[ValidationRow], str]]],
    padding_f: float,
) -> float:
    component_probabilities: list[tuple[float, float]] = []
    for label, base_weight, minimum_count, full_count, key_fn in specs:
        values = residual_sets.get(label, {}).get(key_fn(row))
        if not values or len(values) < minimum_count:
            continue
        count_weight = min(1.0, len(values) / max(1.0, full_count))
        weight = float(base_weight) * count_weight
        if weight <= 0:
            continue
        probability = _empirical_residual_interval_probability(
            values,
            lower=None if row.threshold_low_f is None else row.threshold_low_f - row.forecast_high_f - padding_f,
            upper=None if row.threshold_high_f is None else row.threshold_high_f - row.forecast_high_f + padding_f,
        )
        component_probabilities.append((probability, weight))
    if not component_probabilities:
        global_values = residual_sets.get("global", {}).get("__global__")
        if not global_values:
            return row.current_probability
        return _clip_probability(
            _empirical_residual_interval_probability(
                global_values,
                lower=None if row.threshold_low_f is None else row.threshold_low_f - row.forecast_high_f - padding_f,
                upper=None if row.threshold_high_f is None else row.threshold_high_f - row.forecast_high_f + padding_f,
            )
        )
    total_weight = sum(weight for _, weight in component_probabilities)
    probability = sum(probability * weight for probability, weight in component_probabilities) / total_weight
    return _clip_probability(probability)


def _empirical_residual_interval_probability(
    sorted_residuals: tuple[float, ...],
    *,
    lower: Optional[float],
    upper: Optional[float],
) -> float:
    n = len(sorted_residuals)
    if n <= 0:
        return 0.0
    left = 0 if lower is None else bisect_left(sorted_residuals, lower)
    right = n if upper is None else bisect_right(sorted_residuals, upper)
    return max(0, right - left) / n


def _residual_distribution_params(
    *,
    residual_sets: Mapping[str, Mapping[str, tuple[float, ...]]],
    specs: list[tuple[str, float, int, float, Callable[[ValidationRow], str]]],
    padding_f: float,
    train_rows: list[ValidationRow],
    probabilities: Mapping[str, float],
) -> dict[str, Any]:
    return {
        "type": "empirical_residual_cdf",
        "interval_padding_f": padding_f,
        "train_selection_metrics": _metrics(train_rows, probabilities),
        "components": [
            {
                "label": label,
                "base_weight": base_weight,
                "minimum_count": minimum_count,
                "full_count": full_count,
                "group_count": len(residual_sets.get(label, {})),
                "residual_count": sum(len(values) for values in residual_sets.get(label, {}).values()),
            }
            for label, base_weight, minimum_count, full_count, _ in specs
        ],
        "residual_sets": {
            label: {
                key: [round(value, 4) for value in values]
                for key, values in values_by_key.items()
            }
            for label, values_by_key in residual_sets.items()
        },
    }


def _residual_forecast_high_metrics(
    rows: list[ValidationRow],
    *,
    residual_sets: Mapping[str, Mapping[str, tuple[float, ...]]],
    specs: list[tuple[str, float, int, float, Callable[[ValidationRow], str]]],
) -> dict[str, Any]:
    seen: set[str] = set()
    errors: list[float] = []
    for row in rows:
        if row.forecast_id in seen:
            continue
        seen.add(row.forecast_id)
        residual_mean = _predicted_residual_mean(row, residual_sets=residual_sets, specs=specs)
        predicted_high = row.forecast_high_f + residual_mean
        errors.append(predicted_high - row.realized_high_f)
    return _temperature_error_metrics(errors)


def _predicted_residual_mean(
    row: ValidationRow,
    *,
    residual_sets: Mapping[str, Mapping[str, tuple[float, ...]]],
    specs: list[tuple[str, float, int, float, Callable[[ValidationRow], str]]],
) -> float:
    components: list[tuple[float, float]] = []
    for label, base_weight, minimum_count, full_count, key_fn in specs:
        values = residual_sets.get(label, {}).get(key_fn(row))
        if not values or len(values) < minimum_count:
            continue
        count_weight = min(1.0, len(values) / max(1.0, full_count))
        weight = float(base_weight) * count_weight
        if weight <= 0:
            continue
        components.append((sum(values) / len(values), weight))
    if not components:
        values = residual_sets.get("global", {}).get("__global__")
        return sum(values) / len(values) if values else 0.0
    total_weight = sum(weight for _, weight in components)
    return sum(value * weight for value, weight in components) / total_weight


def _family_probabilities(rows: list[ValidationRow], *, family_name: str) -> dict[str, float]:
    probabilities: dict[str, float] = {}
    for row in rows:
        values = [
            probability
            for key, probability in row.model_probabilities.items()
            if _model_family(key) == family_name
        ]
        probabilities[row.row_id] = _clip_probability(sum(values) / len(values)) if values else row.current_probability
    return probabilities


def _fit_coordinate_weights(
    rows: list[ValidationRow],
    *,
    source_weights: Mapping[str, float],
    model_weights: Mapping[str, float],
    max_epochs: int,
) -> tuple[dict[str, float], dict[str, float]]:
    sources = sorted({_model_source(key) for row in rows for key in row.model_probabilities})
    families = sorted({_model_family(key) for row in rows for key in row.model_probabilities})
    family_briers: dict[str, list[float]] = defaultdict(list)
    source_briers: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        source_values: dict[str, list[float]] = defaultdict(list)
        for key, probability in row.model_probabilities.items():
            family = _model_family(key)
            source = _model_source(key)
            family_briers[family].append((_clip_probability(probability) - row.outcome) ** 2)
            source_values[source].append(_clip_probability(probability))
        for source, values in source_values.items():
            source_probability = sum(values) / len(values)
            source_briers[source].append((source_probability - row.outcome) ** 2)

    family_reference = _median([mean(values) for values in family_briers.values() if values]) or 0.1
    source_reference = _median([mean(values) for values in source_briers.values() if values]) or 0.1
    fitted_family_weights = {
        family: _accuracy_weight(mean(family_briers.get(family) or [family_reference]), family_reference)
        for family in families
    }
    for key, value in model_weights.items():
        family = _model_family(key)
        fitted_family_weights[family] = max(0.0, min(3.0, float(value)))
    fitted_source_weights = {
        source: _accuracy_weight(mean(source_briers.get(source) or [source_reference]), source_reference)
        * max(0.25, min(3.0, float(source_weights.get(source, 1.0))))
        for source in sources
    }
    return fitted_source_weights, fitted_family_weights


def _fit_tail_calibration(rows: list[ValidationRow], probabilities: Mapping[str, float]) -> dict[str, float]:
    best: Optional[tuple[float, float, float, float]] = None
    for center_alpha in (0.65, 0.75, 0.85, 0.95, 1.0, 1.05, 1.10):
        for tail_threshold in (0.70, 0.75, 0.80, 0.85, 0.90, 0.925, 0.95, 0.975, 1.0):
            for tail_alpha in (0.25, 0.35, 0.50, 0.65, 0.80, 0.90, 1.0):
                params = {
                    "center_alpha": center_alpha,
                    "tail_threshold": tail_threshold,
                    "tail_alpha": tail_alpha,
                }
                brier = _brier_from_rows(rows, lambda row: _apply_tail_calibration(probabilities[row.row_id], params))
                if best is None or brier < best[0]:
                    best = (brier, center_alpha, tail_threshold, tail_alpha)
    assert best is not None
    return {
        "center_alpha": best[1],
        "tail_threshold": best[2],
        "tail_alpha": best[3],
        "train_brier": round(best[0], 8),
    }


def _fit_temperature_biases(
    rows: list[ValidationRow],
    *,
    key_fn: Callable[[ValidationRow], str],
    minimum_count: int = 20,
    shrink_count: float = 30.0,
) -> dict[str, float]:
    by_key: dict[str, dict[str, Any]] = defaultdict(lambda: {"errors": [], "forecast_ids": set()})
    global_errors_by_forecast: dict[str, float] = {}
    for row in rows:
        if row.forecast_id not in by_key[key_fn(row)]["forecast_ids"]:
            by_key[key_fn(row)]["errors"].append(row.realized_high_f - row.forecast_high_f)
            by_key[key_fn(row)]["forecast_ids"].add(row.forecast_id)
        global_errors_by_forecast.setdefault(row.forecast_id, row.realized_high_f - row.forecast_high_f)
    global_bias = mean(global_errors_by_forecast.values()) if global_errors_by_forecast else 0.0
    biases: dict[str, float] = {}
    for key, payload in by_key.items():
        values = payload["errors"]
        if len(values) < minimum_count:
            continue
        raw = mean(values)
        shrunk = (raw * len(values) + global_bias * shrink_count) / (len(values) + shrink_count)
        biases[key] = round(max(-6.0, min(6.0, shrunk)), 4)
    return biases


def _bias_corrected_probabilities(
    strategy: Mapping[str, Any],
    rows: list[ValidationRow],
    biases: Mapping[str, float],
    *,
    key_fn: Callable[[ValidationRow], str],
) -> dict[str, float]:
    probabilities: dict[str, float] = {}
    for row in rows:
        bias = float(biases.get(key_fn(row), 0.0))
        if bias == 0.0:
            probabilities[row.row_id] = row.current_probability
            continue
        market = MarketRow(
            universe_row_id=row.universe_row_id,
            normalized_city=row.normalized_city,
            country_region=row.country_region,
            latitude=row.latitude,
            longitude=row.longitude,
            timezone_name=row.timezone_name,
            market_date_local=row.market_date_local,
            bucket_label=row.bucket_label,
            bucket_type=row.bucket_type,
            threshold_low_f=row.threshold_low_f,
            threshold_high_f=row.threshold_high_f,
            units=None,
            resolved_outcome=row.resolved_outcome,
        )
        forecast = ForecastRow(
            forecast_id=row.forecast_id,
            normalized_city=row.normalized_city,
            country_region=row.country_region,
            latitude=row.latitude,
            longitude=row.longitude,
            timezone_name=row.timezone_name,
            local_date=row.market_date_local,
            decision_time_utc=row.decision_time_utc,
            decision_hour_utc=row.decision_hour_utc,
            forecast_run_time_utc=None,
            source_name=row.source_name,
            model_name=row.model_name,
            temperature_2m_max_f=row.forecast_high_f,
            hourly_temperature_2m_max_f=_optional_float(row.feature_json.get("hourly_temperature_2m_max")),
            feature_json=row.feature_json,
            data_quality_flags=row.forecast_quality_flags,
        )
        model_probabilities = _score_model_probabilities(strategy, market, forecast, bias_f=bias)
        probabilities[row.row_id] = _weighted_consensus_probability(
            model_probabilities,
            source_weights=strategy["source_weights"],
            model_weights=strategy["model_weights"],
        )
    return probabilities


def _candidate_payload(
    name: str,
    rows: list[ValidationRow],
    train_rows: list[ValidationRow],
    test_rows: list[ValidationRow],
    probabilities: Mapping[str, float],
    *,
    params: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "name": name,
        "params": _json_safe(params),
        "metrics": {
            "train": _metrics(train_rows, probabilities),
            "test": _metrics(test_rows, probabilities),
            "all": _metrics(rows, probabilities),
        },
        "probabilities": dict(probabilities),
    }


def _diagnostics(
    *,
    rows: list[ValidationRow],
    train_rows: list[ValidationRow],
    test_rows: list[ValidationRow],
    baseline_probabilities: Mapping[str, float],
    best_probabilities: Mapping[str, float],
    best_name: str,
    fitted_candidates: list[dict[str, Any]],
    accepted: list[dict[str, Any]],
) -> dict[str, Any]:
    baseline_test = _metrics(test_rows, baseline_probabilities)
    best_test = _metrics(test_rows, best_probabilities)
    by_city = _group_metrics(test_rows, baseline_probabilities, lambda row: row.normalized_city, min_count=25)
    by_lead_days = _group_metrics(test_rows, baseline_probabilities, lambda row: str(row.lead_days), min_count=25)
    by_day_type = _group_metrics(test_rows, baseline_probabilities, lambda row: row.day_type, min_count=25)
    by_decision_hour = _group_metrics(test_rows, baseline_probabilities, lambda row: str(row.decision_hour_utc), min_count=25)
    by_local_hour = _group_metrics(test_rows, baseline_probabilities, lambda row: _local_hour_bucket(row.local_decision_hour), min_count=25)
    by_source = _group_metrics(test_rows, baseline_probabilities, lambda row: row.source_name, min_count=25)
    by_bucket_type = _group_metrics(test_rows, baseline_probabilities, lambda row: row.bucket_type, min_count=25)
    by_bucket_shape = _group_metrics(test_rows, baseline_probabilities, lambda row: row.bucket_shape, min_count=25)
    by_threshold_distance = _group_metrics(test_rows, baseline_probabilities, lambda row: row.threshold_distance_bucket, min_count=25)
    calibration_baseline = _calibration_bins(test_rows, baseline_probabilities)
    calibration_best = _calibration_bins(test_rows, best_probabilities)
    threshold_accuracy = _threshold_accuracy(test_rows, baseline_probabilities)
    overconfident = _largest_bias_groups(
        by_city + by_lead_days + by_decision_hour + by_bucket_type + by_threshold_distance,
        direction="over",
    )
    underconfident = _largest_bias_groups(
        by_city + by_lead_days + by_decision_hour + by_bucket_type + by_threshold_distance,
        direction="under",
    )
    temp_bias_city = _temperature_error_groups(test_rows, lambda row: row.normalized_city, min_count=25)
    temp_bias_lead = _temperature_error_groups(test_rows, lambda row: str(row.lead_days), min_count=25)
    return {
        "summary": {
            "baseline_test": baseline_test,
            "best_test": best_test,
            "best_name": best_name,
            "accepted_candidates": [candidate["name"] for candidate in accepted],
            "candidate_count": len(fitted_candidates),
            "overconfident_groups_top": overconfident[:12],
            "underconfident_groups_top": underconfident[:12],
            "largest_temperature_bias_city_top": temp_bias_city[:12],
            "largest_temperature_bias_lead": temp_bias_lead,
            "resolved_outcome_mismatches": sum(1 for row in rows if row.resolved_outcome_mismatch),
        },
        "tables": {
            "metrics_by_city": by_city,
            "metrics_by_lead_days": by_lead_days,
            "metrics_by_day_type": by_day_type,
            "metrics_by_decision_hour_utc": by_decision_hour,
            "metrics_by_local_hour": by_local_hour,
            "metrics_by_source": by_source,
            "metrics_by_bucket_type": by_bucket_type,
            "metrics_by_bucket_shape": by_bucket_shape,
            "metrics_by_threshold_distance": by_threshold_distance,
            "calibration_baseline_test": calibration_baseline,
            "calibration_best_test": calibration_best,
            "threshold_accuracy_baseline_test": threshold_accuracy,
            "temperature_bias_by_city": temp_bias_city,
            "temperature_bias_by_lead_days": temp_bias_lead,
            "candidate_comparison": [
                _candidate_summary(candidate, baseline_test)
                for candidate in fitted_candidates
            ],
        },
    }


def _metrics(rows: list[ValidationRow], probabilities: Mapping[str, float]) -> dict[str, Any]:
    if not rows:
        return {
            "n": 0,
            "avg_probability": None,
            "actual_rate": None,
            "bias": None,
            "brier": None,
            "log_loss": None,
            "accuracy_at_0_5": None,
        }
    probs = [_clip_probability(probabilities[row.row_id]) for row in rows]
    actual = [row.outcome for row in rows]
    avg_probability = sum(probs) / len(probs)
    actual_rate = sum(actual) / len(actual)
    return {
        "n": len(rows),
        "avg_probability": round(avg_probability, 6),
        "actual_rate": round(actual_rate, 6),
        "bias": round(avg_probability - actual_rate, 6),
        "brier": round(sum((prob - outcome) ** 2 for prob, outcome in zip(probs, actual)) / len(rows), 8),
        "log_loss": round(
            -sum(outcome * math.log(prob) + (1 - outcome) * math.log(1 - prob) for prob, outcome in zip(probs, actual))
            / len(rows),
            8,
        ),
        "accuracy_at_0_5": round(sum((prob >= 0.5) == bool(outcome) for prob, outcome in zip(probs, actual)) / len(rows), 6),
    }


def _group_metrics(
    rows: list[ValidationRow],
    probabilities: Mapping[str, float],
    key_fn: Callable[[ValidationRow], str],
    *,
    min_count: int,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[ValidationRow]] = defaultdict(list)
    for row in rows:
        grouped[key_fn(row)].append(row)
    output = [
        {"group": key, **_metrics(items, probabilities)}
        for key, items in grouped.items()
        if len(items) >= min_count
    ]
    output.sort(key=lambda item: (-int(item["n"]), str(item["group"])))
    return output


def _calibration_bins(rows: list[ValidationRow], probabilities: Mapping[str, float]) -> list[dict[str, Any]]:
    grouped: dict[int, list[ValidationRow]] = defaultdict(list)
    for row in rows:
        grouped[_probability_bin(probabilities[row.row_id])].append(row)
    output = []
    for bucket in range(10):
        items = grouped.get(bucket, [])
        metrics = _metrics(items, probabilities)
        output.append(
            {
                "probability_bin": f"{bucket / 10:.1f}-{(bucket + 1) / 10:.1f}",
                **metrics,
            }
        )
    return output


def _threshold_accuracy(rows: list[ValidationRow], probabilities: Mapping[str, float]) -> list[dict[str, Any]]:
    output = []
    for threshold in (0.50, 0.60, 0.70, 0.80, 0.90):
        high = [row for row in rows if probabilities[row.row_id] >= threshold]
        low = [row for row in rows if probabilities[row.row_id] <= 1.0 - threshold]
        high_rate = sum(row.outcome for row in high) / len(high) if high else None
        low_false_rate = sum(1 - row.outcome for row in low) / len(low) if low else None
        output.append(
            {
                "threshold": threshold,
                "high_probability_n": len(high),
                "high_probability_actual_rate": round(high_rate, 6) if high_rate is not None else None,
                "high_probability_directional_accuracy": round(high_rate, 6) if high_rate is not None else None,
                "low_probability_n": len(low),
                "low_probability_actual_false_rate": round(low_false_rate, 6) if low_false_rate is not None else None,
                "low_probability_directional_accuracy": round(low_false_rate, 6) if low_false_rate is not None else None,
            }
        )
    return output


def _model_probability_metrics(rows: list[ValidationRow]) -> dict[str, list[dict[str, Any]]]:
    family_values: dict[str, list[tuple[float, int]]] = defaultdict(list)
    source_values: dict[str, list[tuple[float, int]]] = defaultdict(list)
    key_values: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for row in rows:
        for key, probability in row.model_probabilities.items():
            family_values[_model_family(key)].append((probability, row.outcome))
            source_values[_model_source(key)].append((probability, row.outcome))
            key_values[key].append((probability, row.outcome))
    return {
        "by_model_family": _probability_tuple_metrics(family_values),
        "by_source": _probability_tuple_metrics(source_values),
        "by_source_model_family": _probability_tuple_metrics(key_values),
    }


def _market_edge_diagnostics(
    decision_db: Path,
    *,
    test_rows: list[ValidationRow],
    baseline_probabilities: Mapping[str, float],
    candidate_probabilities: Mapping[str, float],
    candidate_name: str,
) -> dict[str, Any]:
    quotes = _load_decision_quotes(decision_db)
    quotes_by_key: dict[tuple[str, datetime], list[DecisionQuoteRow]] = defaultdict(list)
    for quote in quotes:
        quotes_by_key[(quote.universe_row_id, quote.decision_time_utc)].append(quote)

    side_rows: list[dict[str, Any]] = []
    matched_quote_ids: set[tuple[str, datetime, str]] = set()
    for row in test_rows:
        matched_quotes = quotes_by_key.get((row.universe_row_id, row.decision_time_utc), [])
        if not matched_quotes:
            continue
        for quote in matched_quotes:
            if quote.forecast_source_name and quote.forecast_source_name != row.source_name:
                continue
            if quote.mid is None:
                continue
            side = quote.side.upper()
            if side not in {"YES", "NO"}:
                continue
            side_outcome = quote.resolved_outcome_yes if side == "YES" else 1 - quote.resolved_outcome_yes
            baseline_event_probability = baseline_probabilities[row.row_id]
            candidate_event_probability = candidate_probabilities[row.row_id]
            baseline_side_probability = baseline_event_probability if side == "YES" else 1.0 - baseline_event_probability
            candidate_side_probability = candidate_event_probability if side == "YES" else 1.0 - candidate_event_probability
            matched_quote_ids.add((quote.universe_row_id, quote.decision_time_utc, side))
            side_rows.append(
                {
                    "row_id": row.row_id,
                    "universe_row_id": row.universe_row_id,
                    "city": row.normalized_city,
                    "lead_days": row.lead_days,
                    "day_type": row.day_type,
                    "decision_hour_utc": row.decision_hour_utc,
                    "bucket_type": row.bucket_type,
                    "bucket_shape": row.bucket_shape,
                    "threshold_distance_bucket": row.threshold_distance_bucket,
                    "side": side,
                    "weather_yes_outcome": row.outcome,
                    "polymarket_yes_outcome": quote.resolved_outcome_yes,
                    "side_outcome": side_outcome,
                    "baseline_probability": _clip_probability(baseline_side_probability),
                    "candidate_probability": _clip_probability(candidate_side_probability),
                    "market_mid": _clip_probability(quote.mid),
                    "bid": quote.bid,
                    "ask": quote.ask,
                    "spread": quote.spread,
                    "bid_size": quote.bid_size,
                    "ask_size": quote.ask_size,
                    "liquidity_depth": quote.liquidity_depth,
                    "price_stale_seconds": quote.price_stale_seconds,
                    "weather_polymarket_mismatch": int(row.outcome != quote.resolved_outcome_yes),
                    "quote_quality_flags": ";".join(quote.quote_quality_flags),
                }
            )
    tradable_rows = [
        item
        for item in side_rows
        if item.get("ask") is not None
        and item.get("bid") is not None
        and item.get("spread") is not None
        and 0.0 <= float(item["ask"]) <= 1.0
        and 0.0 <= float(item["bid"]) <= 1.0
        and float(item["ask"]) > 0.0
        and float(item["spread"]) <= 0.10
        and not item.get("quote_quality_flags")
    ]
    summary = {
        "candidate_name": candidate_name,
        "all_quote_side_rows": len(quotes),
        "matched_test_quote_side_rows": len(side_rows),
        "unique_matched_quote_side_rows": len(matched_quote_ids),
        "tradable_diagnostic_rows": len(tradable_rows),
        "quote_coverage_warning": "Market diagnostics use only compact quote-covered rows and are not a full trading replay.",
        "baseline_vs_polymarket_payout": _market_probability_metrics(side_rows, "baseline_probability"),
        "candidate_vs_polymarket_payout": _market_probability_metrics(side_rows, "candidate_probability"),
        "market_mid_vs_polymarket_payout": _market_probability_metrics(side_rows, "market_mid"),
        "weather_label_mismatch_rate": _rate([item["weather_polymarket_mismatch"] for item in side_rows]),
        "candidate_buy_edge_by_threshold": _edge_threshold_rows(tradable_rows, probability_key="candidate_probability"),
        "baseline_buy_edge_by_threshold": _edge_threshold_rows(tradable_rows, probability_key="baseline_probability"),
        "candidate_incremental_edge_rows": _incremental_edge_summary(tradable_rows),
        "by_side": _market_group_metrics(side_rows, "side", "candidate_probability"),
        "by_bucket_type": _market_group_metrics(side_rows, "bucket_type", "candidate_probability"),
        "by_lead_days": _market_group_metrics(side_rows, "lead_days", "candidate_probability"),
        "by_decision_hour": _market_group_metrics(side_rows, "decision_hour_utc", "candidate_probability"),
    }
    edge_rows = []
    for family_name, probability_key in (("candidate", "candidate_probability"), ("baseline", "baseline_probability")):
        for row in _edge_threshold_rows(tradable_rows, probability_key=probability_key):
            edge_rows.append({"probability_series": family_name, **row})
    return {
        "summary": summary,
        "tables": {
            "edge_thresholds": edge_rows,
            "candidate_by_side": summary["by_side"],
            "candidate_by_bucket_type": summary["by_bucket_type"],
            "candidate_by_lead_days": summary["by_lead_days"],
            "candidate_by_decision_hour": summary["by_decision_hour"],
        },
    }


def _load_decision_quotes(path: Path) -> list[DecisionQuoteRow]:
    if not path.exists():
        return []
    with closing(_connect_readonly(path)) as conn:
        rows = conn.execute(
            """
            SELECT universe_row_id, decision_time_utc, side, forecast_source_name,
                   mid, bid, ask, spread, bid_size, ask_size, liquidity_depth,
                   price_stale_seconds, resolved_outcome, quote_data_quality_flags
            FROM decision_snapshots
            WHERE quote_timestamp_utc IS NOT NULL
              AND resolved_outcome IN (0, 1)
            ORDER BY decision_time_utc, universe_row_id, side
            """
        ).fetchall()
    quotes: list[DecisionQuoteRow] = []
    for row in rows:
        parsed_time = _parse_datetime(row["decision_time_utc"])
        if parsed_time is None:
            continue
        side = str(row["side"] or "").upper()
        if side not in {"YES", "NO"}:
            continue
        quotes.append(
            DecisionQuoteRow(
                universe_row_id=str(row["universe_row_id"]),
                decision_time_utc=parsed_time,
                side=side,
                forecast_source_name=str(row["forecast_source_name"]) if row["forecast_source_name"] else None,
                mid=_optional_float(row["mid"]),
                bid=_optional_float(row["bid"]),
                ask=_optional_float(row["ask"]),
                spread=_optional_float(row["spread"]),
                bid_size=_optional_float(row["bid_size"]),
                ask_size=_optional_float(row["ask_size"]),
                liquidity_depth=_optional_float(row["liquidity_depth"]),
                price_stale_seconds=_optional_float(row["price_stale_seconds"]),
                resolved_outcome_yes=int(row["resolved_outcome"]),
                quote_quality_flags=_json_flags(row["quote_data_quality_flags"]),
            )
        )
    return quotes


def _market_probability_metrics(rows: list[Mapping[str, Any]], probability_key: str) -> dict[str, Any]:
    if not rows:
        return {"n": 0, "avg_probability": None, "actual_rate": None, "bias": None, "brier": None, "log_loss": None}
    probabilities = [_clip_probability(float(row[probability_key])) for row in rows if row.get(probability_key) is not None]
    outcomes = [int(row["side_outcome"]) for row in rows if row.get(probability_key) is not None]
    if not probabilities:
        return {"n": 0, "avg_probability": None, "actual_rate": None, "bias": None, "brier": None, "log_loss": None}
    avg_probability = sum(probabilities) / len(probabilities)
    actual_rate = sum(outcomes) / len(outcomes)
    return {
        "n": len(probabilities),
        "avg_probability": round(avg_probability, 6),
        "actual_rate": round(actual_rate, 6),
        "bias": round(avg_probability - actual_rate, 6),
        "brier": round(sum((prob - outcome) ** 2 for prob, outcome in zip(probabilities, outcomes)) / len(probabilities), 8),
        "log_loss": round(
            -sum(outcome * math.log(prob) + (1 - outcome) * math.log(1 - prob) for prob, outcome in zip(probabilities, outcomes))
            / len(probabilities),
            8,
        ),
    }


def _market_group_metrics(rows: list[Mapping[str, Any]], key: str, probability_key: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key))].append(row)
    output = [
        {"group": group, **_market_probability_metrics(items, probability_key)}
        for group, items in grouped.items()
        if len(items) >= 25
    ]
    output.sort(key=lambda item: (-int(item["n"]), str(item["group"])))
    return output


def _edge_threshold_rows(rows: list[Mapping[str, Any]], *, probability_key: str) -> list[dict[str, Any]]:
    output = []
    for threshold in (0.00, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20):
        selected = [
            row
            for row in rows
            if row.get("ask") is not None
            and float(row[probability_key]) - float(row["ask"]) >= threshold
        ]
        output.append(_edge_summary_row(selected, threshold=threshold, probability_key=probability_key))
    return output


def _incremental_edge_summary(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    selected = [
        row
        for row in rows
        if row.get("ask") is not None
        and float(row["candidate_probability"]) - float(row["ask"]) >= 0.05
        and float(row["baseline_probability"]) - float(row["ask"]) < 0.05
    ]
    return _edge_summary_row(selected, threshold=0.05, probability_key="candidate_probability")


def _edge_summary_row(rows: list[Mapping[str, Any]], *, threshold: float, probability_key: str) -> dict[str, Any]:
    if not rows:
        return {
            "edge_threshold": threshold,
            "n": 0,
            "actual_rate": None,
            "avg_model_probability": None,
            "avg_ask": None,
            "avg_edge_to_ask": None,
            "flat_return_on_notional": None,
            "weather_label_mismatch_rate": None,
        }
    outcomes = [int(row["side_outcome"]) for row in rows]
    probabilities = [float(row[probability_key]) for row in rows]
    asks = [float(row["ask"]) for row in rows]
    edges = [probability - ask for probability, ask in zip(probabilities, asks)]
    returns = [(outcome / ask) - 1.0 for outcome, ask in zip(outcomes, asks) if ask > 0.0]
    mismatches = [int(row.get("weather_polymarket_mismatch") or 0) for row in rows]
    return {
        "edge_threshold": threshold,
        "n": len(rows),
        "actual_rate": round(sum(outcomes) / len(outcomes), 6),
        "avg_model_probability": round(sum(probabilities) / len(probabilities), 6),
        "avg_ask": round(sum(asks) / len(asks), 6),
        "avg_edge_to_ask": round(sum(edges) / len(edges), 6),
        "flat_return_on_notional": round(sum(returns) / len(returns), 6) if returns else None,
        "weather_label_mismatch_rate": _rate(mismatches),
    }


def _rate(values: list[int]) -> Optional[float]:
    if not values:
        return None
    return round(sum(values) / len(values), 6)


def _probability_tuple_metrics(grouped: Mapping[str, list[tuple[float, int]]]) -> list[dict[str, Any]]:
    output = []
    for key, values in grouped.items():
        if not values:
            continue
        probabilities = [_clip_probability(item[0]) for item in values]
        outcomes = [item[1] for item in values]
        avg_probability = sum(probabilities) / len(probabilities)
        actual_rate = sum(outcomes) / len(outcomes)
        output.append(
            {
                "group": key,
                "n": len(values),
                "avg_probability": round(avg_probability, 6),
                "actual_rate": round(actual_rate, 6),
                "bias": round(avg_probability - actual_rate, 6),
                "brier": round(sum((prob - outcome) ** 2 for prob, outcome in zip(probabilities, outcomes)) / len(values), 8),
                "log_loss": round(
                    -sum(outcome * math.log(prob) + (1 - outcome) * math.log(1 - prob) for prob, outcome in zip(probabilities, outcomes))
                    / len(values),
                    8,
                ),
                "accuracy_at_0_5": round(sum((prob >= 0.5) == bool(outcome) for prob, outcome in zip(probabilities, outcomes)) / len(values), 6),
            }
        )
    output.sort(key=lambda item: (item["brier"], item["log_loss"], str(item["group"])))
    return output


def _forecast_high_metrics(
    forecasts: list[ForecastRow],
    realized: Mapping[tuple[str, date], RealizedRow],
    cutoff: str,
) -> dict[str, Any]:
    cutoff_dt = _parse_datetime(cutoff)
    grouped: dict[str, list[float]] = defaultdict(list)
    for forecast in forecasts:
        actual = realized.get((forecast.normalized_city, forecast.local_date))
        if actual is None or cutoff_dt is None:
            continue
        split = "train" if forecast.decision_time_utc <= cutoff_dt else "test"
        error = forecast.temperature_2m_max_f - actual.temperature_2m_max_f
        grouped[split].append(error)
        grouped["all"].append(error)
    output = {}
    for split, errors in grouped.items():
        output[split] = _temperature_error_metrics(errors)
    return output


def _temperature_error_groups(
    rows: list[ValidationRow],
    key_fn: Callable[[ValidationRow], str],
    *,
    min_count: int,
) -> list[dict[str, Any]]:
    seen_forecasts: dict[str, set[str]] = defaultdict(set)
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        key = key_fn(row)
        if row.forecast_id in seen_forecasts[key]:
            continue
        seen_forecasts[key].add(row.forecast_id)
        grouped[key].append(row.forecast_error_f)
    output = [
        {"group": key, **_temperature_error_metrics(errors)}
        for key, errors in grouped.items()
        if len(errors) >= min_count
    ]
    output.sort(key=lambda item: abs(float(item["mean_error_f"] or 0.0)), reverse=True)
    return output


def _temperature_error_metrics(errors: list[float]) -> dict[str, Any]:
    if not errors:
        return {"n": 0, "mean_error_f": None, "mae_f": None, "rmse_f": None, "stdev_error_f": None}
    return {
        "n": len(errors),
        "mean_error_f": round(sum(errors) / len(errors), 4),
        "mae_f": round(sum(abs(value) for value in errors) / len(errors), 4),
        "rmse_f": round(math.sqrt(sum(value * value for value in errors) / len(errors)), 4),
        "stdev_error_f": round(pstdev(errors), 4) if len(errors) >= 2 else 0.0,
    }


def _largest_bias_groups(rows: list[dict[str, Any]], *, direction: str) -> list[dict[str, Any]]:
    filtered = []
    for row in rows:
        n = int(row.get("n") or 0)
        bias = row.get("bias")
        if n < 250 or bias is None:
            continue
        if direction == "over" and float(bias) <= 0.0:
            continue
        if direction == "under" and float(bias) >= 0.0:
            continue
        filtered.append(row)
    filtered.sort(key=lambda item: abs(float(item["bias"])), reverse=True)
    return filtered


def _write_reports(
    *,
    timestamp: str,
    report_dir: Path,
    candidate_output_dir: Path,
    summary: dict[str, Any],
    diagnostics: dict[str, Any],
    model_family_metrics: dict[str, list[dict[str, Any]]],
    accepted: list[dict[str, Any]],
    baseline_probabilities: Mapping[str, float],
    best_probabilities: Mapping[str, float],
    market_diagnostics: Mapping[str, Any],
    rows: list[ValidationRow],
    train_rows: list[ValidationRow],
    test_rows: list[ValidationRow],
) -> dict[str, str]:
    prefix = f"{timestamp}-weather-model"
    artifacts = {
        "accuracy_json": str(report_dir / f"{prefix}-accuracy.json"),
        "accuracy_md": str(report_dir / f"{prefix}-accuracy.md"),
        "calibration_csv": str(report_dir / f"{prefix}-calibration-test.csv"),
        "candidate_comparison_csv": str(report_dir / f"{prefix}-candidate-comparison.csv"),
        "city_diagnostics_csv": str(report_dir / f"{prefix}-city-diagnostics.csv"),
        "lead_diagnostics_csv": str(report_dir / f"{prefix}-lead-diagnostics.csv"),
        "decision_hour_diagnostics_csv": str(report_dir / f"{prefix}-decision-hour-diagnostics.csv"),
        "source_family_diagnostics_csv": str(report_dir / f"{prefix}-source-family-diagnostics.csv"),
        "bucket_diagnostics_csv": str(report_dir / f"{prefix}-bucket-diagnostics.csv"),
        "threshold_distance_diagnostics_csv": str(report_dir / f"{prefix}-threshold-distance-diagnostics.csv"),
        "threshold_accuracy_csv": str(report_dir / f"{prefix}-threshold-accuracy.csv"),
        "market_edge_diagnostics_csv": str(report_dir / f"{prefix}-market-edge-diagnostics.csv"),
    }
    _write_markdown(Path(artifacts["accuracy_md"]), summary, diagnostics)
    _write_csv(Path(artifacts["calibration_csv"]), diagnostics["tables"]["calibration_baseline_test"])
    _write_csv(Path(artifacts["candidate_comparison_csv"]), diagnostics["tables"]["candidate_comparison"])
    _write_csv(Path(artifacts["city_diagnostics_csv"]), diagnostics["tables"]["metrics_by_city"])
    _write_csv(Path(artifacts["lead_diagnostics_csv"]), diagnostics["tables"]["metrics_by_lead_days"])
    _write_csv(Path(artifacts["decision_hour_diagnostics_csv"]), diagnostics["tables"]["metrics_by_decision_hour_utc"])
    _write_csv(Path(artifacts["source_family_diagnostics_csv"]), model_family_metrics["by_source_model_family"])
    bucket_rows = diagnostics["tables"]["metrics_by_bucket_type"] + diagnostics["tables"]["metrics_by_bucket_shape"]
    _write_csv(Path(artifacts["bucket_diagnostics_csv"]), bucket_rows)
    _write_csv(Path(artifacts["threshold_distance_diagnostics_csv"]), diagnostics["tables"]["metrics_by_threshold_distance"])
    _write_csv(Path(artifacts["threshold_accuracy_csv"]), diagnostics["tables"]["threshold_accuracy_baseline_test"])
    market_rows: list[Mapping[str, Any]] = []
    for table_name, table_rows in (market_diagnostics.get("tables") or {}).items():
        if isinstance(table_rows, list):
            for row in table_rows:
                market_rows.append({"table": table_name, **dict(row)})
    _write_csv(Path(artifacts["market_edge_diagnostics_csv"]), market_rows)

    if accepted:
        accepted_path = candidate_output_dir / f"{prefix}-accepted-calibrators.json"
        accepted_payload = {
            "generated_at_utc": summary["generated_at_utc"],
            "acceptance_rule": "Candidate is included only when held-out Brier and log_loss both beat current_weighted_consensus.",
            "baseline_test_metrics": summary["baseline_metrics"]["test"],
            "accepted": [
                {
                    "name": candidate["name"],
                    "params": candidate["params"],
                    "metrics": candidate["metrics"],
                }
                for candidate in accepted
            ],
            "not_promoted_to_live": "Weather-only improvement has not been compared against point-in-time Polymarket quotes or trading replay.",
        }
        accepted_path.write_text(json.dumps(accepted_payload, indent=2, sort_keys=True), encoding="utf-8")
        artifacts["accepted_candidate_calibrators_json"] = str(accepted_path)

    return artifacts


def _write_markdown(path: Path, summary: Mapping[str, Any], diagnostics: Mapping[str, Any]) -> None:
    baseline = summary["baseline_metrics"]["test"]
    best = diagnostics["summary"]["best_test"]
    lines = [
        "# DailyWeather Weather Model Validation",
        "",
        f"Generated: {summary['generated_at_utc']}",
        "",
        "Scope: weather-only probability validation against realized daily highs. This report does not evaluate or claim Polymarket market edge.",
        "",
        "## Validation Set",
        "",
        f"- Window: {summary['target_start_date']} through {summary['target_end_date']}",
        f"- Validation rows: {summary['input_counts']['validation_rows']:,}",
        f"- Train rows: {summary['input_counts']['train_rows']:,}",
        f"- Test rows: {summary['input_counts']['test_rows']:,}",
        f"- Split cutoff decision time: {summary['split']['cutoff_decision_time_utc']}",
        "",
        "## Current Weather Model",
        "",
        f"- Test Brier: {baseline['brier']}",
        f"- Test log loss: {baseline['log_loss']}",
        f"- Test calibration bias: {baseline['bias']} (avg probability minus actual rate)",
        f"- Test accuracy at 0.5: {baseline['accuracy_at_0_5']}",
        "",
        "## Best Held-Out Candidate",
        "",
        f"- Name: {diagnostics['summary']['best_name']}",
        f"- Test Brier: {best['brier']}",
        f"- Test log loss: {best['log_loss']}",
        f"- Accepted candidates: {', '.join(summary['accepted_candidate_names']) if summary['accepted_candidate_names'] else 'none'}",
        "",
        "## Strongest Old Family",
        "",
    ]
    old_best = summary.get("old_best_source_model_family_test") or {}
    if old_best:
        lines.extend(
            [
                f"- Group: {old_best.get('group')}",
                f"- Test Brier: {old_best.get('brier')}",
                f"- Test log loss: {old_best.get('log_loss')}",
                "",
            ]
        )
    else:
        lines.extend(["No old model-family metrics were available.", ""])
    market = summary.get("market_relative_diagnostics") or {}
    if market:
        candidate_market = market.get("candidate_vs_polymarket_payout") or {}
        market_mid = market.get("market_mid_vs_polymarket_payout") or {}
        lines.extend(
            [
                "## Quote-Covered Market Diagnostics",
                "",
                "These diagnostics use only compact point-in-time quote rows and are not a trading replay or market-beating claim.",
                "",
                f"- Matched held-out quote side rows: {market.get('matched_test_quote_side_rows')}",
                f"- Tradable diagnostic rows, bid/ask present and spread <= 10c: {market.get('tradable_diagnostic_rows')}",
                f"- Candidate vs Polymarket payout Brier: {candidate_market.get('brier')}",
                f"- Candidate vs Polymarket payout log loss: {candidate_market.get('log_loss')}",
                f"- Market mid vs Polymarket payout Brier: {market_mid.get('brier')}",
                f"- Market mid vs Polymarket payout log loss: {market_mid.get('log_loss')}",
                f"- Weather/Polymarket label mismatch rate in matched rows: {market.get('weather_label_mismatch_rate')}",
                "",
            ]
        )
    lines.extend(
        [
            "## Main Diagnostics",
            "",
            "- Overconfident groups are where predicted probabilities are higher than realized event frequency.",
            "- Underconfident groups are where predicted probabilities are lower than realized event frequency.",
            "- Forecast high error is forecast minus realized high in Fahrenheit.",
            "",
            "Top overconfident groups:",
            "",
        ]
    )
    lines.extend(_markdown_table(diagnostics["summary"]["overconfident_groups_top"], ("group", "n", "avg_probability", "actual_rate", "bias", "brier")))
    lines.extend(["", "Top underconfident groups:", ""])
    lines.extend(_markdown_table(diagnostics["summary"]["underconfident_groups_top"], ("group", "n", "avg_probability", "actual_rate", "bias", "brier")))
    lines.extend(["", "Largest temperature-bias city groups:", ""])
    lines.extend(_markdown_table(diagnostics["summary"]["largest_temperature_bias_city_top"], ("group", "n", "mean_error_f", "mae_f", "rmse_f")))
    lines.extend(
        [
            "",
            "## Candidate Policy",
            "",
            "Candidate weights/calibrators are written only when they improve held-out Brier and log loss versus the current weighted consensus. No accepted candidate should be promoted to live trading until it also survives point-in-time quote comparison and trading replay.",
            "",
            "## Observed-High Note",
            "",
            "The current weather artifacts do not store partial observed high as of each decision time. This run tests decision-hour and local-time calibration, but it does not test observed-high-aware same-day adjustment because using final realized weather before the decision would leak the label.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _markdown_table(rows: list[Mapping[str, Any]], columns: tuple[str, ...]) -> list[str]:
    if not rows:
        return ["No groups met the reporting threshold."]
    output = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows[:12]:
        output.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return output


def _validation_notes(rows: list[ValidationRow], forecasts: list[ForecastRow], accepted: list[dict[str, Any]]) -> list[str]:
    source_counts: dict[str, int] = defaultdict(int)
    flag_counts: dict[str, int] = defaultdict(int)
    for forecast in forecasts:
        source_counts[forecast.source_name] += 1
        for flag in forecast.data_quality_flags:
            flag_counts[flag] += 1
    notes = [
        "Train/test split is by decision_time_utc, so held-out rows occur strictly after the training cutoff.",
        "Realized daily high is used only as the label and for fitting training-only candidate corrections.",
        "Point-in-time Polymarket diagnostics are limited to compact quote-covered rows and are not a full trading replay.",
    ]
    if flag_counts.get("historical_forecast_endpoint_is_asof_approximation"):
        notes.append("Most broad-history forecasts carry historical_forecast_endpoint_is_asof_approximation; treat this as weather-model validation, not a perfect vendor as-of audit.")
    if len(source_counts) <= 2:
        notes.append("Source-weight testing is limited because the broad validation set is dominated by one Open-Meteo historical forecast source.")
    if not accepted:
        notes.append("No candidate calibrator/weight artifact improved both held-out Brier and log loss; no deployable model-weight file was produced.")
    return notes


def _time_split(rows: list[ValidationRow], *, train_fraction: float) -> tuple[list[ValidationRow], list[ValidationRow], dict[str, Any]]:
    times = sorted({row.decision_time_utc for row in rows})
    if len(times) < 2:
        raise ValueError("Need at least two distinct decision times for time-aware split")
    cutoff_index = max(0, min(len(times) - 2, int(len(times) * train_fraction) - 1))
    cutoff = times[cutoff_index]
    train_rows = [row for row in rows if row.decision_time_utc <= cutoff]
    test_rows = [row for row in rows if row.decision_time_utc > cutoff]
    if not train_rows or not test_rows:
        raise ValueError("Time split produced an empty train or test set")
    return (
        train_rows,
        test_rows,
        {
            "train_fraction_requested": train_fraction,
            "unique_decision_times": len(times),
            "cutoff_decision_time_utc": cutoff.isoformat(),
            "first_decision_time_utc": times[0].isoformat(),
            "last_decision_time_utc": times[-1].isoformat(),
            "first_test_decision_time_utc": min(row.decision_time_utc for row in test_rows).isoformat(),
        },
    )


def _weighted_consensus_probability(
    model_probabilities: Mapping[str, float],
    *,
    source_weights: Mapping[str, float],
    model_weights: Mapping[str, float],
) -> float:
    by_source: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for key, probability in model_probabilities.items():
        model_weight = max(0.0, float(model_weights.get(key, model_weights.get(_model_family(key), 1.0))))
        by_source[_model_source(key)].append((_clip_probability(probability), model_weight))
    weighted_total = 0.0
    weight_sum = 0.0
    for source, values in by_source.items():
        model_weight_sum = sum(weight for _, weight in values)
        if model_weight_sum <= 0:
            continue
        source_probability = sum(prob * weight for prob, weight in values) / model_weight_sum
        source_weight = max(0.0, float(source_weights.get(source, 1.0)))
        weighted_total += source_probability * source_weight
        weight_sum += source_weight
    if weight_sum <= 0:
        return _clip_probability(sum(model_probabilities.values()) / len(model_probabilities))
    return _clip_probability(weighted_total / weight_sum)


def _apply_tail_calibration(probability: float, params: Mapping[str, float]) -> float:
    center_alpha = float(params.get("center_alpha", 1.0))
    tail_threshold = float(params.get("tail_threshold", 1.0))
    tail_alpha = float(params.get("tail_alpha", 1.0))
    probability = 0.5 + center_alpha * (_clip_probability(probability) - 0.5)
    low_threshold = 1.0 - tail_threshold
    if probability >= tail_threshold:
        probability = tail_threshold + tail_alpha * (probability - tail_threshold)
    elif probability <= low_threshold:
        probability = low_threshold + tail_alpha * (probability - low_threshold)
    return _clip_probability(probability)


def _brier_from_rows(rows: list[ValidationRow], predictor: Callable[[ValidationRow], float]) -> float:
    if not rows:
        return float("inf")
    return sum((_clip_probability(predictor(row)) - row.outcome) ** 2 for row in rows) / len(rows)


def _accuracy_weight(brier: float, reference_brier: float) -> float:
    if brier <= 0:
        return 3.0
    raw = reference_brier / brier
    return round(max(0.0, min(3.0, raw)), 4)


def _median(values: list[float]) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def _candidate_summary(candidate: Mapping[str, Any], baseline_test: Mapping[str, Any]) -> dict[str, Any]:
    test = candidate["metrics"]["test"]
    summary = {
        "name": candidate["name"],
        "test_n": test["n"],
        "test_brier": test["brier"],
        "test_log_loss": test["log_loss"],
        "test_bias": test["bias"],
        "baseline_test_brier": baseline_test["brier"],
        "baseline_test_log_loss": baseline_test["log_loss"],
        "delta_brier_vs_baseline": round(float(test["brier"]) - float(baseline_test["brier"]), 8),
        "delta_log_loss_vs_baseline": round(float(test["log_loss"]) - float(baseline_test["log_loss"]), 8),
        "accepted": _improves_both(test, baseline_test),
    }
    high_metrics = candidate.get("forecast_high_metrics")
    if isinstance(high_metrics, Mapping):
        summary["candidate_high_mae_f"] = high_metrics.get("mae_f")
        summary["candidate_high_rmse_f"] = high_metrics.get("rmse_f")
        summary["candidate_high_mean_error_f"] = high_metrics.get("mean_error_f")
    return summary


def _improves_both(candidate_test: Mapping[str, Any], baseline_test: Mapping[str, Any]) -> bool:
    if candidate_test.get("brier") is None or candidate_test.get("log_loss") is None:
        return False
    return (
        float(candidate_test["brier"]) < float(baseline_test["brier"]) - 1e-9
        and float(candidate_test["log_loss"]) < float(baseline_test["log_loss"]) - 1e-9
    )


def _lead_days_and_local_hour(decision_time_utc: datetime, target: date, timezone_name: str) -> tuple[int, float]:
    try:
        zone = ZoneInfo(timezone_name)
    except Exception:
        zone = timezone.utc
    local_dt = decision_time_utc.astimezone(zone)
    return (target - local_dt.date()).days, local_dt.hour + local_dt.minute / 60.0


def _day_type(lead_days: int) -> str:
    if lead_days == 0:
        return "same_day"
    if lead_days == 1:
        return "next_day"
    if lead_days == 2:
        return "two_day"
    return f"lead_{lead_days}"


def _local_hour_bucket(local_hour: float) -> str:
    if local_hour < 6:
        return "00-06"
    if local_hour < 12:
        return "06-12"
    if local_hour < 15:
        return "12-15"
    if local_hour < 18:
        return "15-18"
    return "18-24"


def _bucket_contains(value: float, lower: Optional[float], upper: Optional[float]) -> bool:
    if lower is not None and value < lower:
        return False
    if upper is not None and value > upper:
        return False
    return True


def _bucket_shape(lower: Optional[float], upper: Optional[float]) -> str:
    if lower is None and upper is None:
        return "unknown"
    if lower is None:
        return "lower_tail"
    if upper is None:
        return "upper_tail"
    return "bounded"


def _threshold_distance(forecast_high_f: float, lower: Optional[float], upper: Optional[float]) -> Optional[float]:
    thresholds = [value for value in (lower, upper) if value is not None]
    if not thresholds:
        return None
    return min(abs(forecast_high_f - value) for value in thresholds)


def _distance_bucket(distance: Optional[float]) -> str:
    if distance is None:
        return "unknown"
    if distance < 1:
        return "0-1F"
    if distance < 2:
        return "1-2F"
    if distance < 4:
        return "2-4F"
    if distance < 8:
        return "4-8F"
    return "8F+"


def _probability_bin(probability: float) -> int:
    return min(9, max(0, int(_clip_probability(probability) * 10)))


def _model_source(key: str) -> str:
    return key.rsplit(".", 1)[0] if "." in key else key


def _model_family(key: str) -> str:
    return key.rsplit(".", 1)[1] if "." in key else "unknown"


def _clip_probability(value: float) -> float:
    return max(EPSILON, min(1.0 - EPSILON, float(value)))


def _parse_datetime(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _optional_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _json_mapping(value: Any) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _json_flags(value: Any) -> tuple[str, ...]:
    if not value:
        return ()
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return (str(value),)
    if isinstance(parsed, list):
        return tuple(str(item) for item in parsed if item)
    if isinstance(parsed, dict):
        return tuple(str(key) for key, item in parsed.items() if item)
    return (str(parsed),)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
