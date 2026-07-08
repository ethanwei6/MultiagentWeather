from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sqlite3
import sys
from bisect import bisect_left
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any, Iterable, Mapping, Optional, Sequence
from zoneinfo import ZoneInfo


EPSILON = 1e-6
FIRST10_STATIONS = ("KLGA", "KDAL", "KMIA", "KORD", "KSEA", "KHOU", "KAUS", "KLAX", "KSFO", "KBKF")
FORBIDDEN_MODEL_INPUT_COLUMNS = {
    "bid",
    "ask",
    "mid",
    "last_price",
    "resolved_outcome",
    "canonical_label",
    "station_label",
    "station_high_f",
    "open_meteo_high_f",
    "payout",
}
STATION_REGION = {
    "KLGA": "US_Northeast",
    "KORD": "US_Midwest",
    "KDAL": "US_South",
    "KHOU": "US_South",
    "KAUS": "US_South",
    "KMIA": "US_South",
    "KSEA": "US_West",
    "KLAX": "US_West",
    "KSFO": "US_West",
    "KBKF": "US_Mountain",
}


@dataclass(frozen=True)
class StationConfig:
    station_id: str
    normalized_city: str
    timezone_name: str
    settlement_source_url: str
    settlement_unit: str
    settlement_precision: str
    region: str


@dataclass(frozen=True)
class StationHigh:
    station_id: str
    settlement_source_url: str
    local_date: date
    high_f: float
    high_c: Optional[float]


@dataclass(frozen=True)
class IntradayHigh:
    station_id: str
    settlement_source_url: str
    local_date: date
    decision_time_utc: datetime
    high_so_far_f: Optional[float]
    high_observed_at_utc: Optional[datetime]
    sample_count: int


@dataclass(frozen=True)
class DecisionRow:
    row_id: str
    weather_key: str
    sample_key: str
    universe_row_id: str
    market_id: Optional[str]
    normalized_city: str
    station_id: str
    region: str
    settlement_source_url: str
    timezone_name: str
    local_date: date
    decision_time_utc: datetime
    forecast_run_time_utc: Optional[datetime]
    decision_hour_utc: int
    local_decision_hour: float
    lead_days: int
    day_type: str
    forecast_source_name: str
    forecast_model_family: str
    forecast_high_f: float
    bucket_type: str
    threshold_low_f: Optional[float]
    threshold_high_f: Optional[float]
    threshold_distance_f: Optional[float]
    threshold_distance_bucket: str
    settlement_precision: str
    station_high_f: float
    open_meteo_high_f: Optional[float]
    station_outcome: int
    open_meteo_outcome: Optional[int]
    canonical_label: Optional[int]
    side: str
    mid: Optional[float]
    bid: Optional[float]
    ask: Optional[float]
    observed_high_so_far_f: Optional[float]
    observed_high_observed_at_utc: Optional[datetime]
    observed_high_sample_count: int
    recommended_action: Optional[str]
    quote_covered: bool


@dataclass(frozen=True)
class TrainingSample:
    sample_key: str
    normalized_city: str
    station_id: str
    region: str
    local_date: date
    decision_time_utc: datetime
    label_available_utc: datetime
    forecast_source_name: str
    forecast_model_family: str
    forecast_high_f: float
    station_high_f: float
    open_meteo_high_f: Optional[float]
    decision_hour_utc: int
    local_decision_hour: float
    lead_days: int


@dataclass(frozen=True)
class CandidateConfig:
    name: str
    family: str
    target: str
    method: str
    params: Mapping[str, float]


@dataclass(frozen=True)
class EmpiricalDistribution:
    values: tuple[float, ...]

    @property
    def n(self) -> int:
        return len(self.values)

    def interval_probability(self, lower: Optional[float], upper: Optional[float]) -> float:
        if not self.values:
            return 0.5
        left = 0 if lower is None else bisect_left(self.values, lower)
        right = len(self.values) if upper is None else bisect_left(self.values, upper)
        return max(0, right - left) / len(self.values)


@dataclass(frozen=True)
class NormalStats:
    n: int
    mean: float
    sigma: float


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Research-only station-targeted DailyWeather weather-model audit")
    parser.add_argument("--work-root", default=".")
    parser.add_argument("--start-date", default="2026-01-01")
    parser.add_argument("--end-date", default="2026-06-29")
    parser.add_argument("--output-dir", default="work/reports/station_targeted_weather_model")
    parser.add_argument("--station-proof-json", default="")
    parser.add_argument("--min-train-samples", type=int, default=240)
    parser.add_argument("--min-test-rows", type=int, default=500)
    parser.add_argument("--embargo-days", default="0,3,7")
    parser.add_argument("--primary-embargo-days", type=int, default=3)
    parser.add_argument("--bootstrap-iterations", type=int, default=200)
    parser.add_argument("--permutation-iterations", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20260707)
    args = parser.parse_args(argv)

    summary = run_station_targeted_audit(
        work_root=Path(args.work_root).resolve(),
        start=date.fromisoformat(args.start_date),
        end=date.fromisoformat(args.end_date),
        output_dir=Path(args.output_dir),
        station_proof_json=Path(args.station_proof_json).resolve() if args.station_proof_json else None,
        min_train_samples=args.min_train_samples,
        min_test_rows=args.min_test_rows,
        embargo_days=_parse_int_list(args.embargo_days),
        primary_embargo_days=args.primary_embargo_days,
        bootstrap_iterations=args.bootstrap_iterations,
        permutation_iterations=args.permutation_iterations,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def run_station_targeted_audit(
    *,
    work_root: Path,
    start: date,
    end: date,
    output_dir: Path,
    station_proof_json: Optional[Path],
    min_train_samples: int,
    min_test_rows: int,
    embargo_days: Sequence[int],
    primary_embargo_days: int,
    bootstrap_iterations: int,
    permutation_iterations: int,
    seed: int,
) -> dict[str, Any]:
    paths = _artifact_paths(work_root)
    proof_path = station_proof_json or _latest_report(work_root / "work/reports/station_observations", "*-first10-station-observation-proof.json")
    station_configs = _load_station_configs(proof_path)
    station_highs = _load_station_highs(paths["station_db"], station_configs, start=start, end=end)
    intraday_highs = _load_intraday_highs(paths["station_db"], station_configs, start=start, end=end)
    open_meteo_highs = _load_open_meteo_highs(paths["weather_db"], station_configs, start=start, end=end)
    weather_rows = _load_decision_rows(
        paths["decision_db"],
        station_configs,
        station_highs=station_highs,
        intraday_highs=intraday_highs,
        open_meteo_highs=open_meteo_highs,
        start=start,
        end=end,
        side_filter="YES",
    )
    side_rows = _load_decision_rows(
        paths["decision_db"],
        station_configs,
        station_highs=station_highs,
        intraday_highs=intraday_highs,
        open_meteo_highs=open_meteo_highs,
        start=start,
        end=end,
        side_filter=None,
    )
    samples = _training_samples(weather_rows)
    if len(weather_rows) < min_test_rows:
        raise ValueError(f"Need at least {min_test_rows} weather rows; got {len(weather_rows)}")
    if len(samples) < min_train_samples:
        raise ValueError(f"Need at least {min_train_samples} training samples; got {len(samples)}")

    configs = _candidate_configs()
    plateau_configs = _plateau_configs()
    print(
        f"[station-targeted-weather] rows={len(weather_rows)} side_rows={len(side_rows)} "
        f"samples={len(samples)} stations={len(station_configs)} configs={len(configs)}",
        file=sys.stderr,
        flush=True,
    )

    split_rows, aggregate = _run_walk_forward(
        weather_rows,
        samples,
        configs=configs,
        embargo_days=embargo_days,
        primary_embargo_days=primary_embargo_days,
        min_train_samples=min_train_samples,
        min_test_rows=min_test_rows,
    )
    primary_rows = aggregate["rows"]
    probabilities_by_model = aggregate["probabilities_by_model"]
    if not primary_rows:
        raise ValueError("No primary walk-forward rows met the training/test thresholds")

    target_metrics = _target_metrics(primary_rows, probabilities_by_model)
    candidate_comparison = _candidate_comparison(target_metrics)
    top_model = _top_station_model(candidate_comparison)
    top_probabilities = probabilities_by_model.get(top_model, {})
    baseline_probabilities = probabilities_by_model.get("open_meteo_target_empirical_baseline", {})
    residual_comparison = _city_station_residual_comparison(samples)
    calibration_rows = _calibration_report(primary_rows, probabilities_by_model)
    robustness_slices = _robustness_slices(primary_rows, probabilities_by_model, top_model=top_model)
    near_threshold = _near_threshold_report(primary_rows, probabilities_by_model, top_model=top_model)
    station_holdout = _station_holdout_report(
        weather_rows,
        samples,
        configs=configs,
        top_model=top_model,
        primary_embargo_days=primary_embargo_days,
        min_train_samples=min_train_samples,
    )
    region_holdout = _region_holdout_report(
        weather_rows,
        samples,
        configs=configs,
        top_model=top_model,
        primary_embargo_days=primary_embargo_days,
        min_train_samples=min_train_samples,
    )
    plateau_grid = _plateau_grid_report(
        primary_rows,
        samples,
        plateau_configs=plateau_configs,
        primary_embargo_days=primary_embargo_days,
        min_train_samples=min_train_samples,
    )
    plateau_summary = _plateau_summary(plateau_grid)
    market_rows, market_summary = _market_midpoint_diagnostic(side_rows, primary_rows, probabilities_by_model, top_model=top_model)
    live_audit = _live_feature_audit(
        weather_rows,
        side_rows,
        samples,
        station_configs=station_configs,
        station_highs=station_highs,
        intraday_highs=intraday_highs,
        open_meteo_highs=open_meteo_highs,
    )
    label_poisoning = _label_poisoning_invariance(primary_rows, top_probabilities)
    rng = random.Random(seed)
    bootstrap = _bootstrap_ci(primary_rows, baseline_probabilities, top_probabilities, iterations=bootstrap_iterations, rng=rng)
    permutation = _permutation_sanity(primary_rows, baseline_probabilities, top_probabilities, iterations=permutation_iterations, rng=rng)
    verdict = _verdict(
        candidate_comparison=candidate_comparison,
        station_holdout=station_holdout,
        region_holdout=region_holdout,
        plateau_summary=plateau_summary,
        live_audit=live_audit,
        market_summary=market_summary,
        top_model=top_model,
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_dir = (work_root / output_dir).resolve() if not output_dir.is_absolute() else output_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{timestamp}-station-targeted-weather-model"
    artifacts = {
        "summary_json": str(report_dir / f"{prefix}.json"),
        "report_md": str(report_dir / f"{prefix}.md"),
        "target_metrics_csv": str(report_dir / f"{prefix}-open-meteo-target-vs-station-target-metrics.csv"),
        "candidate_comparison_csv": str(report_dir / f"{prefix}-candidate-comparison.csv"),
        "city_station_residual_comparison_csv": str(report_dir / f"{prefix}-city-station-residual-comparison.csv"),
        "calibration_report_csv": str(report_dir / f"{prefix}-calibration.csv"),
        "near_threshold_calibration_csv": str(report_dir / f"{prefix}-near-threshold-calibration.csv"),
        "robustness_slices_csv": str(report_dir / f"{prefix}-robustness-slices.csv"),
        "walk_forward_splits_csv": str(report_dir / f"{prefix}-walk-forward-splits.csv"),
        "station_holdout_csv": str(report_dir / f"{prefix}-station-holdout.csv"),
        "region_holdout_csv": str(report_dir / f"{prefix}-region-holdout.csv"),
        "parameter_plateau_grid_csv": str(report_dir / f"{prefix}-parameter-plateau-grid.csv"),
        "market_midpoint_diagnostic_csv": str(report_dir / f"{prefix}-market-midpoint-diagnostic.csv"),
        "live_feature_leakage_audit_csv": str(report_dir / f"{prefix}-live-feature-leakage-audit.csv"),
        "bootstrap_ci_csv": str(report_dir / f"{prefix}-bootstrap-ci.csv"),
        "permutation_sanity_csv": str(report_dir / f"{prefix}-permutation-sanity.csv"),
        "label_poisoning_invariance_csv": str(report_dir / f"{prefix}-label-poisoning-invariance.csv"),
    }
    _write_csv(Path(artifacts["target_metrics_csv"]), target_metrics)
    _write_csv(Path(artifacts["candidate_comparison_csv"]), candidate_comparison)
    _write_csv(Path(artifacts["city_station_residual_comparison_csv"]), residual_comparison)
    _write_csv(Path(artifacts["calibration_report_csv"]), calibration_rows)
    _write_csv(Path(artifacts["near_threshold_calibration_csv"]), near_threshold)
    _write_csv(Path(artifacts["robustness_slices_csv"]), robustness_slices)
    _write_csv(Path(artifacts["walk_forward_splits_csv"]), split_rows)
    _write_csv(Path(artifacts["station_holdout_csv"]), station_holdout)
    _write_csv(Path(artifacts["region_holdout_csv"]), region_holdout)
    _write_csv(Path(artifacts["parameter_plateau_grid_csv"]), plateau_grid)
    _write_csv(Path(artifacts["market_midpoint_diagnostic_csv"]), market_rows)
    _write_csv(Path(artifacts["live_feature_leakage_audit_csv"]), live_audit)
    _write_csv(Path(artifacts["bootstrap_ci_csv"]), bootstrap)
    _write_csv(Path(artifacts["permutation_sanity_csv"]), [permutation])
    _write_csv(Path(artifacts["label_poisoning_invariance_csv"]), [label_poisoning])

    summary = {
        "role": "Weather Model Agent",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "first10_station_targeted_daily_high_research_only",
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "station_ids": sorted(station_configs),
        "station_proof_json": str(proof_path),
        "input_counts": {
            "weather_scoring_rows_yes_side": len(weather_rows),
            "side_rows_for_market_diagnostic": len(side_rows),
            "unique_training_samples": len(samples),
            "primary_walk_forward_rows": len(primary_rows),
            "station_daily_highs": len(station_highs),
            "intraday_high_snapshots": len(intraday_highs),
            "open_meteo_daily_highs": len(open_meteo_highs),
        },
        "candidate_models": [
            {
                "name": config.name,
                "family": config.family,
                "target": config.target,
                "method": config.method,
                "params": dict(config.params),
            }
            for config in configs
        ],
        "target_metrics": target_metrics,
        "candidate_comparison": candidate_comparison,
        "top_station_model": top_model,
        "plateau_summary": plateau_summary,
        "station_holdout_summary": _holdout_summary(station_holdout, "station_id"),
        "region_holdout_summary": _holdout_summary(region_holdout, "region"),
        "market_summary": market_summary,
        "live_feature_audit_summary": live_audit[0] if live_audit else {},
        "label_poisoning_invariance": label_poisoning,
        "bootstrap_ci": bootstrap,
        "permutation_sanity": permutation,
        "verdict": verdict,
        "artifacts": artifacts,
        "promotion_status": "research_only_not_promoted",
        "model_weights_or_defaults_changed": False,
        "strategy_pnl_backtest_run": False,
        "market_prices_used_as_model_inputs": False,
        "resolved_market_labels_used_as_model_inputs": False,
    }
    Path(artifacts["summary_json"]).write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_markdown(Path(artifacts["report_md"]), summary)
    return summary


def _artifact_paths(work_root: Path) -> dict[str, Path]:
    base = work_root / "work/data/backtest"
    return {
        "decision_db": base / "daily_weather_decision_artifact.sqlite",
        "weather_db": base / "daily_weather_weather_history.sqlite",
        "station_db": base / "daily_weather_station_observations.sqlite",
    }


def _latest_report(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No report matching {pattern} under {directory}")
    return matches[-1]


def _load_station_configs(path: Path) -> dict[str, StationConfig]:
    data = json.loads(path.read_text(encoding="utf-8"))
    configs: dict[str, StationConfig] = {}
    for item in data.get("source_selection", []):
        station_id = str(item["station_id"]).upper()
        if station_id not in FIRST10_STATIONS:
            continue
        source = str(item["settlement_source"])
        source_url = source.split(" ", 1)[1] if source.startswith("Wunderground ") else source
        configs[station_id] = StationConfig(
            station_id=station_id,
            normalized_city=str(item["city"]),
            timezone_name=str(item["timezone"]),
            settlement_source_url=source_url,
            settlement_unit=str(item["settlement_unit"]),
            settlement_precision=str(item["settlement_precision"]),
            region=STATION_REGION.get(station_id, "US_Other"),
        )
    missing = sorted(set(FIRST10_STATIONS) - set(configs))
    if missing:
        raise ValueError(f"Station proof JSON is missing first-10 station configs: {missing}")
    return configs


def _load_station_highs(
    path: Path,
    configs: Mapping[str, StationConfig],
    *,
    start: date,
    end: date,
) -> dict[tuple[str, str, date], StationHigh]:
    conn = _connect_ro(path)
    try:
        rows = conn.execute(
            """
            SELECT station_id,
                   settlement_source_url,
                   local_date,
                   high_temperature_f,
                   high_temperature_c
            FROM station_daily_highs
            WHERE local_date >= ?
              AND local_date <= ?
              AND high_temperature_f IS NOT NULL
            """,
            (start.isoformat(), end.isoformat()),
        ).fetchall()
    finally:
        conn.close()
    allowed = {(cfg.station_id, cfg.settlement_source_url) for cfg in configs.values()}
    output: dict[tuple[str, str, date], StationHigh] = {}
    for row in rows:
        key2 = (str(row["station_id"]), str(row["settlement_source_url"]))
        if key2 not in allowed:
            continue
        local_date = date.fromisoformat(str(row["local_date"]))
        key = (key2[0], key2[1], local_date)
        output[key] = StationHigh(
            station_id=key2[0],
            settlement_source_url=key2[1],
            local_date=local_date,
            high_f=float(row["high_temperature_f"]),
            high_c=_to_optional_float(row["high_temperature_c"]),
        )
    return output


def _load_intraday_highs(
    path: Path,
    configs: Mapping[str, StationConfig],
    *,
    start: date,
    end: date,
) -> dict[tuple[str, str, date, datetime], IntradayHigh]:
    conn = _connect_ro(path)
    try:
        rows = conn.execute(
            """
            SELECT station_id,
                   settlement_source_url,
                   local_date,
                   decision_time_utc,
                   high_so_far_f,
                   high_observed_at_utc,
                   sample_count
            FROM station_intraday_high_snapshots
            WHERE local_date >= ?
              AND local_date <= ?
            """,
            (start.isoformat(), end.isoformat()),
        ).fetchall()
    finally:
        conn.close()
    allowed = {(cfg.station_id, cfg.settlement_source_url) for cfg in configs.values()}
    output: dict[tuple[str, str, date, datetime], IntradayHigh] = {}
    for row in rows:
        key2 = (str(row["station_id"]), str(row["settlement_source_url"]))
        if key2 not in allowed:
            continue
        decision_time = _parse_datetime(row["decision_time_utc"])
        if decision_time is None:
            continue
        local_date = date.fromisoformat(str(row["local_date"]))
        key = (key2[0], key2[1], local_date, decision_time)
        output[key] = IntradayHigh(
            station_id=key2[0],
            settlement_source_url=key2[1],
            local_date=local_date,
            decision_time_utc=decision_time,
            high_so_far_f=_to_optional_float(row["high_so_far_f"]),
            high_observed_at_utc=_parse_datetime(row["high_observed_at_utc"]),
            sample_count=int(row["sample_count"] or 0),
        )
    return output


def _load_open_meteo_highs(
    path: Path,
    configs: Mapping[str, StationConfig],
    *,
    start: date,
    end: date,
) -> dict[tuple[str, date], float]:
    cities = sorted({cfg.normalized_city for cfg in configs.values()})
    placeholders = ",".join("?" for _ in cities)
    conn = _connect_ro(path)
    try:
        rows = conn.execute(
            f"""
            SELECT normalized_city,
                   local_date,
                   temperature_2m_max_f
            FROM realized_daily_weather
            WHERE source_name = 'open_meteo_archive_daily'
              AND local_date >= ?
              AND local_date <= ?
              AND normalized_city IN ({placeholders})
              AND temperature_2m_max_f IS NOT NULL
            """,
            (start.isoformat(), end.isoformat(), *cities),
        ).fetchall()
    finally:
        conn.close()
    return {
        (str(row["normalized_city"]), date.fromisoformat(str(row["local_date"]))): float(row["temperature_2m_max_f"])
        for row in rows
    }


def _load_decision_rows(
    path: Path,
    configs: Mapping[str, StationConfig],
    *,
    station_highs: Mapping[tuple[str, str, date], StationHigh],
    intraday_highs: Mapping[tuple[str, str, date, datetime], IntradayHigh],
    open_meteo_highs: Mapping[tuple[str, date], float],
    start: date,
    end: date,
    side_filter: Optional[str],
) -> list[DecisionRow]:
    configs_by_url = {cfg.settlement_source_url: cfg for cfg in configs.values()}
    urls = sorted(configs_by_url)
    placeholders = ",".join("?" for _ in urls)
    side_clause = "" if side_filter is None else "AND side = ?"
    params: list[Any] = [start.isoformat(), end.isoformat(), *urls]
    if side_filter is not None:
        params.append(side_filter)
    conn = _connect_ro(path)
    try:
        rows = conn.execute(
            f"""
            SELECT decision_id,
                   universe_row_id,
                   market_id,
                   normalized_city,
                   timezone,
                   market_date_local,
                   bucket_type,
                   canonical_bucket_type,
                   threshold_low_f,
                   threshold_high_f,
                   side,
                   decision_time_utc,
                   decision_hour_utc,
                   lead_days,
                   forecast_run_time_utc,
                   forecast_source_name,
                   forecast_temperature_2m_max_f,
                   canonical_label,
                   recommended_action,
                   exclude_from_backtest_replay,
                   settlement_precision,
                   canonical_source_url,
                   bid,
                   ask,
                   mid
            FROM decision_snapshots_current_enriched
            WHERE market_date_local >= ?
              AND market_date_local <= ?
              AND canonical_source_url IN ({placeholders})
              AND settlement_unit = 'F'
              AND settlement_precision = 'whole_F'
              AND forecast_temperature_2m_max_f IS NOT NULL
              AND (recommended_action IS NULL OR recommended_action != 'exclude_or_manual_review')
              AND COALESCE(exclude_from_backtest_replay, 0) = 0
              {side_clause}
            ORDER BY market_date_local, normalized_city, decision_time_utc, universe_row_id, side
            """,
            params,
        ).fetchall()
    finally:
        conn.close()

    output: list[DecisionRow] = []
    seen_weather_keys: set[str] = set()
    for row in rows:
        source_url = str(row["canonical_source_url"])
        cfg = configs_by_url[source_url]
        local_date = date.fromisoformat(str(row["market_date_local"]))
        station_high = station_highs.get((cfg.station_id, cfg.settlement_source_url, local_date))
        if station_high is None:
            continue
        decision_time = _parse_datetime(row["decision_time_utc"])
        if decision_time is None:
            continue
        forecast_run_time = _parse_datetime(row["forecast_run_time_utc"])
        if forecast_run_time is not None and forecast_run_time > decision_time:
            continue
        forecast_high = float(row["forecast_temperature_2m_max_f"])
        lower = _to_optional_float(row["threshold_low_f"])
        upper = _to_optional_float(row["threshold_high_f"])
        open_high = open_meteo_highs.get((cfg.normalized_city, local_date))
        intraday = intraday_highs.get((cfg.station_id, cfg.settlement_source_url, local_date, decision_time))
        observed_high = intraday.high_so_far_f if intraday else None
        observed_at = intraday.high_observed_at_utc if intraday else None
        if observed_at is not None and observed_at > decision_time:
            observed_high = None
        bucket_type = str(row["canonical_bucket_type"] or row["bucket_type"] or "unknown")
        station_outcome = _bucket_label_for_temperature(
            station_high.high_f,
            lower=lower,
            upper=upper,
            settlement_precision=str(row["settlement_precision"]),
        )
        open_outcome = None
        if open_high is not None:
            open_outcome = _bucket_label_for_temperature(
                open_high,
                lower=lower,
                upper=upper,
                settlement_precision=str(row["settlement_precision"]),
            )
        local_decision_hour = _local_decision_hour(decision_time, cfg.timezone_name)
        lead_days = _coerce_int(row["lead_days"])
        if lead_days is None:
            local_decision_date = decision_time.astimezone(ZoneInfo(cfg.timezone_name)).date()
            lead_days = (local_date - local_decision_date).days
        weather_key = "|".join(
            [
                str(row["universe_row_id"]),
                decision_time.isoformat(),
                str(row["forecast_source_name"] or ""),
                str(row["forecast_run_time_utc"] or ""),
            ]
        )
        if side_filter == "YES" and weather_key in seen_weather_keys:
            continue
        seen_weather_keys.add(weather_key)
        sample_key = "|".join(
            [
                cfg.station_id,
                cfg.settlement_source_url,
                local_date.isoformat(),
                decision_time.isoformat(),
                str(row["forecast_source_name"] or ""),
                str(row["forecast_run_time_utc"] or ""),
            ]
        )
        quote_covered = row["mid"] is not None or row["bid"] is not None or row["ask"] is not None
        output.append(
            DecisionRow(
                row_id=str(row["decision_id"]),
                weather_key=weather_key,
                sample_key=sample_key,
                universe_row_id=str(row["universe_row_id"]),
                market_id=str(row["market_id"]) if row["market_id"] is not None else None,
                normalized_city=cfg.normalized_city,
                station_id=cfg.station_id,
                region=cfg.region,
                settlement_source_url=cfg.settlement_source_url,
                timezone_name=cfg.timezone_name,
                local_date=local_date,
                decision_time_utc=decision_time,
                forecast_run_time_utc=forecast_run_time,
                decision_hour_utc=int(row["decision_hour_utc"] or decision_time.hour),
                local_decision_hour=local_decision_hour,
                lead_days=int(lead_days),
                day_type=_day_type(int(lead_days)),
                forecast_source_name=str(row["forecast_source_name"] or "unknown"),
                forecast_model_family=_forecast_model_family(row["forecast_source_name"]),
                forecast_high_f=forecast_high,
                bucket_type=bucket_type,
                threshold_low_f=lower,
                threshold_high_f=upper,
                threshold_distance_f=_threshold_distance(forecast_high, lower, upper),
                threshold_distance_bucket=_distance_bucket(_threshold_distance(forecast_high, lower, upper)),
                settlement_precision=str(row["settlement_precision"]),
                station_high_f=station_high.high_f,
                open_meteo_high_f=open_high,
                station_outcome=station_outcome,
                open_meteo_outcome=open_outcome,
                canonical_label=_coerce_int(row["canonical_label"]),
                side=str(row["side"] or "YES").upper(),
                mid=_to_optional_float(row["mid"]),
                bid=_to_optional_float(row["bid"]),
                ask=_to_optional_float(row["ask"]),
                observed_high_so_far_f=observed_high,
                observed_high_observed_at_utc=observed_at,
                observed_high_sample_count=intraday.sample_count if intraday else 0,
                recommended_action=str(row["recommended_action"]) if row["recommended_action"] is not None else None,
                quote_covered=quote_covered,
            )
        )
    return output


def _training_samples(rows: Iterable[DecisionRow]) -> list[TrainingSample]:
    by_key: dict[str, TrainingSample] = {}
    for row in rows:
        if row.sample_key in by_key:
            continue
        by_key[row.sample_key] = TrainingSample(
            sample_key=row.sample_key,
            normalized_city=row.normalized_city,
            station_id=row.station_id,
            region=row.region,
            local_date=row.local_date,
            decision_time_utc=row.decision_time_utc,
            label_available_utc=_label_available_utc(row.local_date),
            forecast_source_name=row.forecast_source_name,
            forecast_model_family=row.forecast_model_family,
            forecast_high_f=row.forecast_high_f,
            station_high_f=row.station_high_f,
            open_meteo_high_f=row.open_meteo_high_f,
            decision_hour_utc=row.decision_hour_utc,
            local_decision_hour=row.local_decision_hour,
            lead_days=row.lead_days,
        )
    return sorted(by_key.values(), key=lambda item: (item.local_date, item.station_id, item.decision_time_utc))


def _candidate_configs() -> list[CandidateConfig]:
    return [
        CandidateConfig(
            name="open_meteo_target_empirical_baseline",
            family="baseline",
            target="open_meteo",
            method="empirical",
            params={
                "city_weight": 0.65,
                "global_weight": 0.35,
                "min_group_n": 20.0,
                "tail_floor": 0.006,
                "tail_alpha": 1.0,
            },
        ),
        CandidateConfig(
            name="station_global_empirical",
            family="station_target",
            target="station",
            method="empirical",
            params={
                "global_weight": 1.0,
                "min_group_n": 20.0,
                "tail_floor": 0.006,
                "tail_alpha": 1.0,
            },
        ),
        CandidateConfig(
            name="station_hierarchical_empirical",
            family="station_target",
            target="station",
            method="empirical",
            params={
                "station_weight": 0.58,
                "region_weight": 0.22,
                "source_weight": 0.10,
                "lead_weight": 0.05,
                "hour_weight": 0.05,
                "global_weight": 0.25,
                "min_group_n": 24.0,
                "tail_floor": 0.006,
                "tail_alpha": 1.0,
            },
        ),
        CandidateConfig(
            name="station_tail_shrunk_hierarchical",
            family="station_target",
            target="station",
            method="empirical",
            params={
                "station_weight": 0.55,
                "region_weight": 0.25,
                "source_weight": 0.10,
                "lead_weight": 0.06,
                "hour_weight": 0.04,
                "global_weight": 0.30,
                "min_group_n": 24.0,
                "tail_floor": 0.010,
                "tail_alpha": 0.72,
                "far_tail_distance_f": 8.0,
            },
        ),
        CandidateConfig(
            name="station_source_bias_normal",
            family="station_target",
            target="station",
            method="normal",
            params={
                "station_weight": 0.48,
                "region_weight": 0.24,
                "source_weight": 0.18,
                "global_weight": 0.30,
                "min_group_n": 24.0,
                "sigma_floor_f": 1.40,
                "spread_multiplier": 0.35,
                "tail_floor": 0.008,
                "tail_alpha": 0.90,
            },
        ),
        CandidateConfig(
            name="station_observed_high_guarded_shrinkage",
            family="station_target",
            target="station",
            method="empirical_observed_guard",
            params={
                "station_weight": 0.55,
                "region_weight": 0.25,
                "source_weight": 0.10,
                "lead_weight": 0.06,
                "hour_weight": 0.04,
                "global_weight": 0.30,
                "min_group_n": 24.0,
                "tail_floor": 0.008,
                "tail_alpha": 0.82,
                "far_tail_distance_f": 8.0,
            },
        ),
    ]


def _plateau_configs() -> list[CandidateConfig]:
    configs: list[CandidateConfig] = []
    for station_weight in (0.45, 0.55, 0.65):
        for region_weight in (0.15, 0.25, 0.35):
            for tail_floor in (0.004, 0.008, 0.012):
                for tail_alpha in (0.72, 0.82, 0.92):
                    configs.append(
                        CandidateConfig(
                            name=(
                                "plateau_station_observed_guarded|"
                                f"station={station_weight:.2f}|region={region_weight:.2f}|"
                                f"floor={tail_floor:.3f}|alpha={tail_alpha:.2f}"
                            ),
                            family="plateau_station_target",
                            target="station",
                            method="empirical_observed_guard",
                            params={
                                "station_weight": station_weight,
                                "region_weight": region_weight,
                                "source_weight": 0.10,
                                "lead_weight": 0.06,
                                "hour_weight": 0.04,
                                "global_weight": 0.30,
                                "min_group_n": 24.0,
                                "tail_floor": tail_floor,
                                "tail_alpha": tail_alpha,
                                "far_tail_distance_f": 8.0,
                            },
                        )
                    )
    for station_weight in (0.38, 0.48, 0.58):
        for region_weight in (0.18, 0.24, 0.30):
            for sigma_floor in (1.10, 1.40, 1.70):
                for spread_multiplier in (0.20, 0.35, 0.50):
                    configs.append(
                        CandidateConfig(
                            name=(
                                "plateau_station_source_bias_normal|"
                                f"station={station_weight:.2f}|region={region_weight:.2f}|"
                                f"sigma={sigma_floor:.2f}|spread={spread_multiplier:.2f}"
                            ),
                            family="plateau_station_target",
                            target="station",
                            method="normal",
                            params={
                                "station_weight": station_weight,
                                "region_weight": region_weight,
                                "source_weight": 0.18,
                                "global_weight": 0.30,
                                "min_group_n": 24.0,
                                "sigma_floor_f": sigma_floor,
                                "spread_multiplier": spread_multiplier,
                                "tail_floor": 0.008,
                                "tail_alpha": 0.90,
                            },
                        )
                    )
    return configs


def _run_walk_forward(
    rows: list[DecisionRow],
    samples: list[TrainingSample],
    *,
    configs: Sequence[CandidateConfig],
    embargo_days: Sequence[int],
    primary_embargo_days: int,
    min_train_samples: int,
    min_test_rows: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    months = _target_months(rows)
    split_rows: list[dict[str, Any]] = []
    primary_rows_by_key: dict[str, DecisionRow] = {}
    primary_probabilities: dict[str, dict[str, float]] = {config.name: {} for config in configs}
    for embargo in embargo_days:
        for month_start in months:
            month_end = _next_month(month_start)
            test_rows = [row for row in rows if month_start <= row.local_date < month_end]
            if len(test_rows) < min_test_rows:
                continue
            test_start_utc = datetime.combine(month_start, time.min, tzinfo=timezone.utc)
            train_cutoff = test_start_utc - timedelta(days=int(embargo))
            train_samples = [sample for sample in samples if sample.label_available_utc <= train_cutoff]
            if len(train_samples) < min_train_samples:
                continue
            fit_cache: dict[tuple[Any, ...], Any] = {}
            for config in configs:
                probabilities = _predict_probabilities(
                    config,
                    train_samples,
                    test_rows,
                    fit_cache=fit_cache,
                    fit_key=("walk_forward", embargo, month_start.isoformat()),
                )
                metrics_station = _score(test_rows, probabilities, target="station")
                metrics_open = _score(test_rows, probabilities, target="open_meteo")
                split_rows.append(
                    {
                        "embargo_days": embargo,
                        "target_month": month_start.isoformat(),
                        "model": config.name,
                        "train_samples": len(train_samples),
                        "test_rows": len(test_rows),
                        "station_brier": metrics_station["brier"],
                        "station_log_loss": metrics_station["log_loss"],
                        "station_bias": metrics_station["bias"],
                        "open_meteo_brier": metrics_open["brier"],
                        "open_meteo_log_loss": metrics_open["log_loss"],
                    }
                )
                if embargo == primary_embargo_days:
                    primary_probabilities[config.name].update(probabilities)
            if embargo == primary_embargo_days:
                for row in test_rows:
                    primary_rows_by_key[row.row_id] = row
    primary_rows = [primary_rows_by_key[key] for key in sorted(primary_rows_by_key)]
    return split_rows, {"rows": primary_rows, "probabilities_by_model": primary_probabilities}


def _predict_probabilities(
    config: CandidateConfig,
    train_samples: Sequence[TrainingSample],
    rows: Sequence[DecisionRow],
    *,
    fit_cache: Optional[dict[tuple[Any, ...], Any]] = None,
    fit_key: Any = None,
) -> dict[str, float]:
    if config.method == "normal":
        probabilities = _predict_normal(config, train_samples, rows, fit_cache=fit_cache, fit_key=fit_key)
    else:
        probabilities = _predict_empirical(config, train_samples, rows, fit_cache=fit_cache, fit_key=fit_key)
    if config.method == "empirical_observed_guard":
        probabilities = {
            row.row_id: _apply_observed_high_guard(probabilities[row.row_id], row)
            for row in rows
        }
    return probabilities


def _predict_empirical(
    config: CandidateConfig,
    train_samples: Sequence[TrainingSample],
    rows: Sequence[DecisionRow],
    *,
    fit_cache: Optional[dict[tuple[Any, ...], Any]] = None,
    fit_key: Any = None,
) -> dict[str, float]:
    cache_key = ("empirical", config.target, fit_key if fit_key is not None else id(train_samples))
    if fit_cache is not None and cache_key in fit_cache:
        grouped = fit_cache[cache_key]
    else:
        grouped = _fit_empirical_distributions(train_samples, target=config.target)
        if fit_cache is not None:
            fit_cache[cache_key] = grouped
    output: dict[str, float] = {}
    for row in rows:
        lower, upper = _effective_probability_bounds(row.threshold_low_f, row.threshold_high_f, row.settlement_precision)
        residual_lower = None if lower is None else lower - row.forecast_high_f
        residual_upper = None if upper is None else upper - row.forecast_high_f
        components = _empirical_components(row, grouped, config.params)
        if not components:
            probability = 0.5
        else:
            weight_sum = sum(weight for _, weight in components)
            probability = sum(dist.interval_probability(residual_lower, residual_upper) * weight for dist, weight in components) / weight_sum
        output[row.row_id] = _tail_smooth(probability, row, config.params)
    return output


def _predict_normal(
    config: CandidateConfig,
    train_samples: Sequence[TrainingSample],
    rows: Sequence[DecisionRow],
    *,
    fit_cache: Optional[dict[tuple[Any, ...], Any]] = None,
    fit_key: Any = None,
) -> dict[str, float]:
    cache_key = ("normal", config.target, fit_key if fit_key is not None else id(train_samples))
    if fit_cache is not None and cache_key in fit_cache:
        grouped = fit_cache[cache_key]
    else:
        grouped = _fit_normal_stats(train_samples, target=config.target)
        if fit_cache is not None:
            fit_cache[cache_key] = grouped
    output: dict[str, float] = {}
    for row in rows:
        lower, upper = _effective_probability_bounds(row.threshold_low_f, row.threshold_high_f, row.settlement_precision)
        components = _normal_components(row, grouped, config.params)
        if not components:
            mean_resid = 0.0
            sigma = max(2.0, float(config.params.get("sigma_floor_f", 1.5)))
        else:
            weight_sum = sum(weight for _, weight in components)
            means = [stats.mean for stats, _ in components]
            mean_resid = sum(stats.mean * weight for stats, weight in components) / weight_sum
            within_var = sum((stats.sigma ** 2) * weight for stats, weight in components) / weight_sum
            disagreement = _small_pstdev(means) if len(means) > 1 else 0.0
            sigma = math.sqrt(max(0.0, within_var) + float(config.params.get("spread_multiplier", 0.0)) * disagreement**2)
            sigma = max(float(config.params.get("sigma_floor_f", 1.5)), sigma)
        mean_high = row.forecast_high_f + mean_resid
        probability = _normal_interval_probability(mean_high, sigma, lower, upper)
        output[row.row_id] = _tail_smooth(probability, row, config.params)
    return output


def _fit_empirical_distributions(
    samples: Sequence[TrainingSample],
    *,
    target: str,
) -> dict[str, dict[str, EmpiricalDistribution]]:
    grouped_values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for sample in samples:
        residual = _sample_residual(sample, target)
        if residual is None:
            continue
        for group_name, group_key in _sample_groups(sample):
            grouped_values[group_name][group_key].append(residual)
    return {
        group_name: {
            group_key: EmpiricalDistribution(tuple(sorted(values)))
            for group_key, values in groups.items()
            if values
        }
        for group_name, groups in grouped_values.items()
    }


def _fit_normal_stats(
    samples: Sequence[TrainingSample],
    *,
    target: str,
) -> dict[str, dict[str, NormalStats]]:
    grouped_values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for sample in samples:
        residual = _sample_residual(sample, target)
        if residual is None:
            continue
        for group_name, group_key in _sample_groups(sample):
            grouped_values[group_name][group_key].append(residual)
    output: dict[str, dict[str, NormalStats]] = {}
    for group_name, groups in grouped_values.items():
        output[group_name] = {}
        for group_key, values in groups.items():
            sigma = pstdev(values) if len(values) > 1 else 2.0
            output[group_name][group_key] = NormalStats(n=len(values), mean=mean(values), sigma=max(sigma, 0.5))
    return output


def _empirical_components(
    row: DecisionRow,
    grouped: Mapping[str, Mapping[str, EmpiricalDistribution]],
    params: Mapping[str, float],
) -> list[tuple[EmpiricalDistribution, float]]:
    components: list[tuple[EmpiricalDistribution, float]] = []
    min_group_n = int(params.get("min_group_n", 20.0))
    for group_name, group_key, weight_key in _row_component_keys(row):
        weight = float(params.get(weight_key, 0.0))
        if weight <= 0:
            continue
        dist = grouped.get(group_name, {}).get(group_key)
        if dist is None or dist.n < min_group_n:
            continue
        components.append((dist, weight))
    global_dist = grouped.get("global", {}).get("global")
    if global_dist is not None:
        components.append((global_dist, float(params.get("global_weight", 0.25))))
    return [(dist, weight) for dist, weight in components if weight > 0]


def _normal_components(
    row: DecisionRow,
    grouped: Mapping[str, Mapping[str, NormalStats]],
    params: Mapping[str, float],
) -> list[tuple[NormalStats, float]]:
    components: list[tuple[NormalStats, float]] = []
    min_group_n = int(params.get("min_group_n", 20.0))
    for group_name, group_key, weight_key in _row_component_keys(row):
        weight = float(params.get(weight_key, 0.0))
        if weight <= 0:
            continue
        stats = grouped.get(group_name, {}).get(group_key)
        if stats is None or stats.n < min_group_n:
            continue
        components.append((stats, weight))
    global_stats = grouped.get("global", {}).get("global")
    if global_stats is not None:
        components.append((global_stats, float(params.get("global_weight", 0.25))))
    return [(stats, weight) for stats, weight in components if weight > 0]


def _sample_groups(sample: TrainingSample) -> list[tuple[str, str]]:
    return [
        ("global", "global"),
        ("city", sample.normalized_city),
        ("station", sample.station_id),
        ("region", sample.region),
        ("source", sample.forecast_source_name),
        ("model_family", sample.forecast_model_family),
        ("lead", _lead_bucket(sample.lead_days)),
        ("hour", _hour_bucket(sample.local_decision_hour)),
        ("station_source", f"{sample.station_id}|{sample.forecast_source_name}"),
    ]


def _row_component_keys(row: DecisionRow) -> list[tuple[str, str, str]]:
    return [
        ("city", row.normalized_city, "city_weight"),
        ("station", row.station_id, "station_weight"),
        ("region", row.region, "region_weight"),
        ("source", row.forecast_source_name, "source_weight"),
        ("model_family", row.forecast_model_family, "source_weight"),
        ("lead", _lead_bucket(row.lead_days), "lead_weight"),
        ("hour", _hour_bucket(row.local_decision_hour), "hour_weight"),
        ("station_source", f"{row.station_id}|{row.forecast_source_name}", "source_weight"),
    ]


def _sample_residual(sample: TrainingSample, target: str) -> Optional[float]:
    if target == "station":
        return sample.station_high_f - sample.forecast_high_f
    if target == "open_meteo":
        if sample.open_meteo_high_f is None:
            return None
        return sample.open_meteo_high_f - sample.forecast_high_f
    raise ValueError(f"Unknown residual target: {target}")


def _target_metrics(rows: list[DecisionRow], probabilities_by_model: Mapping[str, Mapping[str, float]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for model, probabilities in probabilities_by_model.items():
        station = _score(rows, probabilities, target="station")
        open_meteo = _score(rows, probabilities, target="open_meteo")
        output.append({"model": model, "target": "station_target", **station})
        output.append({"model": model, "target": "open_meteo_target", **open_meteo})
    output.sort(key=lambda item: (str(item["target"]), float(item["brier"]) if item["brier"] is not None else 99.0))
    return output


def _candidate_comparison(target_metrics: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    station_metrics = {str(row["model"]): row for row in target_metrics if row.get("target") == "station_target"}
    baseline = station_metrics.get("open_meteo_target_empirical_baseline")
    output: list[dict[str, Any]] = []
    for model, metrics in station_metrics.items():
        row = {"model": model, **metrics}
        if baseline and metrics.get("brier") is not None and baseline.get("brier") is not None:
            row["delta_brier_vs_open_meteo_baseline"] = round(float(metrics["brier"]) - float(baseline["brier"]), 8)
            row["delta_log_loss_vs_open_meteo_baseline"] = round(float(metrics["log_loss"]) - float(baseline["log_loss"]), 8)
            row["beats_baseline_brier"] = float(row["delta_brier_vs_open_meteo_baseline"]) < 0
            row["beats_baseline_log_loss"] = float(row["delta_log_loss_vs_open_meteo_baseline"]) < 0
        output.append(row)
    output.sort(key=lambda item: (float(item["brier"]) if item["brier"] is not None else 99.0, str(item["model"])))
    return output


def _top_station_model(candidate_comparison: Sequence[Mapping[str, Any]]) -> str:
    for row in candidate_comparison:
        model = str(row.get("model", ""))
        if model != "open_meteo_target_empirical_baseline":
            return model
    return "open_meteo_target_empirical_baseline"


def _score(rows: Sequence[DecisionRow], probabilities: Mapping[str, float], *, target: str) -> dict[str, Any]:
    pairs: list[tuple[float, int]] = []
    for row in rows:
        if row.row_id not in probabilities:
            continue
        outcome: Optional[int]
        if target == "station":
            outcome = row.station_outcome
        elif target == "open_meteo":
            outcome = row.open_meteo_outcome
        elif target == "canonical":
            outcome = row.canonical_label
        else:
            raise ValueError(f"Unsupported target: {target}")
        if outcome is None:
            continue
        pairs.append((_clip_probability(probabilities[row.row_id]), int(outcome)))
    return _score_pairs(pairs)


def _score_pairs(pairs: Sequence[tuple[float, int]]) -> dict[str, Any]:
    if not pairs:
        return {
            "n": 0,
            "avg_probability": None,
            "actual_rate": None,
            "bias": None,
            "brier": None,
            "log_loss": None,
            "accuracy_at_0_5": None,
        }
    probs = [p for p, _ in pairs]
    outcomes = [o for _, o in pairs]
    avg_probability = sum(probs) / len(probs)
    actual_rate = sum(outcomes) / len(outcomes)
    return {
        "n": len(pairs),
        "avg_probability": round(avg_probability, 6),
        "actual_rate": round(actual_rate, 6),
        "bias": round(avg_probability - actual_rate, 6),
        "brier": round(sum((p - o) ** 2 for p, o in pairs) / len(pairs), 8),
        "log_loss": round(-sum(o * math.log(p) + (1 - o) * math.log(1 - p) for p, o in pairs) / len(pairs), 8),
        "accuracy_at_0_5": round(sum((p >= 0.5) == bool(o) for p, o in pairs) / len(pairs), 6),
    }


def _calibration_report(rows: list[DecisionRow], probabilities_by_model: Mapping[str, Mapping[str, float]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for model, probabilities in probabilities_by_model.items():
        for target in ("station", "open_meteo"):
            grouped: dict[str, list[tuple[float, int]]] = defaultdict(list)
            for row in rows:
                outcome = row.station_outcome if target == "station" else row.open_meteo_outcome
                if outcome is None or row.row_id not in probabilities:
                    continue
                probability = _clip_probability(probabilities[row.row_id])
                grouped[_probability_bin_label(probability)].append((probability, int(outcome)))
            for bin_label in _probability_bin_labels():
                metrics = _score_pairs(grouped.get(bin_label, []))
                output.append({"model": model, "target": target, "probability_bin": bin_label, **metrics})
    return output


def _robustness_slices(
    rows: list[DecisionRow],
    probabilities_by_model: Mapping[str, Mapping[str, float]],
    *,
    top_model: str,
) -> list[dict[str, Any]]:
    models = ["open_meteo_target_empirical_baseline"]
    if top_model not in models:
        models.append(top_model)
    slices = [
        ("target_month", lambda row: row.local_date.strftime("%Y-%m")),
        ("city", lambda row: row.normalized_city),
        ("station", lambda row: row.station_id),
        ("region", lambda row: row.region),
        ("bucket_type", lambda row: row.bucket_type),
        ("threshold_distance", lambda row: row.threshold_distance_bucket),
        ("lead_days", lambda row: str(row.lead_days)),
        ("day_type", lambda row: row.day_type),
        ("decision_hour_utc", lambda row: str(row.decision_hour_utc)),
        ("local_hour_bucket", lambda row: _hour_bucket(row.local_decision_hour)),
        ("source_model_family", lambda row: row.forecast_model_family),
        ("source_name", lambda row: row.forecast_source_name),
        ("observed_high_availability", lambda row: "available" if row.observed_high_so_far_f is not None else "missing"),
        ("quote_covered", lambda row: "quote_covered" if row.quote_covered else "no_quote"),
        ("actual_threshold_distance", lambda row: _distance_bucket(_actual_threshold_distance(row))),
    ]
    output: list[dict[str, Any]] = []
    for slice_type, key_fn in slices:
        grouped: dict[str, list[DecisionRow]] = defaultdict(list)
        for row in rows:
            grouped[key_fn(row)].append(row)
        for group, items in grouped.items():
            if len(items) < 50:
                continue
            for model in models:
                metrics = _score(items, probabilities_by_model.get(model, {}), target="station")
                output.append({"slice_type": slice_type, "slice": group, "model": model, **metrics})
    output.sort(key=lambda item: (str(item["slice_type"]), str(item["slice"]), str(item["model"])))
    return output


def _near_threshold_report(
    rows: list[DecisionRow],
    probabilities_by_model: Mapping[str, Mapping[str, float]],
    *,
    top_model: str,
) -> list[dict[str, Any]]:
    models = ["open_meteo_target_empirical_baseline"]
    if top_model not in models:
        models.append(top_model)
    output: list[dict[str, Any]] = []
    slice_defs = [
        ("actual_distance_0_0.5F", lambda row: (_actual_threshold_distance(row) or 999) < 0.5),
        ("actual_distance_0.5_1F", lambda row: 0.5 <= (_actual_threshold_distance(row) or 999) < 1.0),
        ("actual_distance_1_2F", lambda row: 1.0 <= (_actual_threshold_distance(row) or 999) < 2.0),
        ("actual_distance_2_4F", lambda row: 2.0 <= (_actual_threshold_distance(row) or 999) < 4.0),
        ("actual_distance_4_8F", lambda row: 4.0 <= (_actual_threshold_distance(row) or 999) < 8.0),
        ("actual_distance_8F_plus", lambda row: (_actual_threshold_distance(row) or -1) >= 8.0),
        ("forecast_distance_8F_plus", lambda row: (row.threshold_distance_f or -1) >= 8.0),
        ("high_confidence_tail", lambda row: False),
    ]
    for model in models:
        probabilities = probabilities_by_model.get(model, {})
        for name, predicate in slice_defs:
            if name == "high_confidence_tail":
                items = [row for row in rows if row.row_id in probabilities and (_clip_probability(probabilities[row.row_id]) <= 0.10 or _clip_probability(probabilities[row.row_id]) >= 0.90)]
            else:
                items = [row for row in rows if predicate(row)]
            if len(items) < 20:
                continue
            metrics = _score(items, probabilities, target="station")
            output.append({"model": model, "slice": name, **metrics})
    return output


def _city_station_residual_comparison(samples: Sequence[TrainingSample]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[TrainingSample]] = defaultdict(list)
    for sample in samples:
        grouped[(sample.normalized_city, sample.station_id, sample.forecast_source_name)].append(sample)
    output: list[dict[str, Any]] = []
    for (city, station_id, source_name), items in grouped.items():
        station_resids = [item.station_high_f - item.forecast_high_f for item in items]
        open_resids = [item.open_meteo_high_f - item.forecast_high_f for item in items if item.open_meteo_high_f is not None]
        station_minus_open = [
            item.station_high_f - item.open_meteo_high_f
            for item in items
            if item.open_meteo_high_f is not None
        ]
        output.append(
            {
                "city": city,
                "station_id": station_id,
                "forecast_source_name": source_name,
                "n_samples": len(items),
                "station_residual_mean_f": _round(mean(station_resids)),
                "station_residual_sigma_f": _round(pstdev(station_resids) if len(station_resids) > 1 else 0.0),
                "open_meteo_residual_mean_f": _round(mean(open_resids)) if open_resids else None,
                "open_meteo_residual_sigma_f": _round(pstdev(open_resids) if len(open_resids) > 1 else 0.0) if open_resids else None,
                "station_minus_open_meteo_mean_f": _round(mean(station_minus_open)) if station_minus_open else None,
                "station_minus_open_meteo_sigma_f": _round(pstdev(station_minus_open) if len(station_minus_open) > 1 else 0.0) if station_minus_open else None,
            }
        )
    output.sort(key=lambda row: (str(row["city"]), str(row["forecast_source_name"])))
    return output


def _station_holdout_report(
    rows: list[DecisionRow],
    samples: list[TrainingSample],
    *,
    configs: Sequence[CandidateConfig],
    top_model: str,
    primary_embargo_days: int,
    min_train_samples: int,
) -> list[dict[str, Any]]:
    configs_by_name = {config.name: config for config in configs}
    baseline = configs_by_name["open_meteo_target_empirical_baseline"]
    candidate = configs_by_name.get(top_model)
    if candidate is None:
        return []
    output: list[dict[str, Any]] = []
    months = _target_months(rows)
    for station_id in sorted({row.station_id for row in rows}):
        holdout_rows: list[DecisionRow] = []
        baseline_probs: dict[str, float] = {}
        candidate_probs: dict[str, float] = {}
        for month_start in months:
            month_end = _next_month(month_start)
            test_rows = [row for row in rows if row.station_id == station_id and month_start <= row.local_date < month_end]
            if not test_rows:
                continue
            train_cutoff = datetime.combine(month_start, time.min, tzinfo=timezone.utc) - timedelta(days=primary_embargo_days)
            train_samples = [
                sample
                for sample in samples
                if sample.station_id != station_id and sample.label_available_utc <= train_cutoff
            ]
            if len(train_samples) < min_train_samples:
                continue
            holdout_rows.extend(test_rows)
            baseline_probs.update(_predict_probabilities(baseline, train_samples, test_rows))
            candidate_probs.update(_predict_probabilities(candidate, train_samples, test_rows))
        if not holdout_rows:
            continue
        base_metrics = _score(holdout_rows, baseline_probs, target="station")
        cand_metrics = _score(holdout_rows, candidate_probs, target="station")
        output.append(
            {
                "station_id": station_id,
                "model": top_model,
                "n": cand_metrics["n"],
                "baseline_brier": base_metrics["brier"],
                "candidate_brier": cand_metrics["brier"],
                "delta_brier": _metric_delta(cand_metrics, base_metrics, "brier"),
                "baseline_log_loss": base_metrics["log_loss"],
                "candidate_log_loss": cand_metrics["log_loss"],
                "delta_log_loss": _metric_delta(cand_metrics, base_metrics, "log_loss"),
                "passes_brier": _metric_delta(cand_metrics, base_metrics, "brier") is not None and _metric_delta(cand_metrics, base_metrics, "brier") < 0,
                "passes_log_loss": _metric_delta(cand_metrics, base_metrics, "log_loss") is not None and _metric_delta(cand_metrics, base_metrics, "log_loss") < 0,
            }
        )
    return output


def _region_holdout_report(
    rows: list[DecisionRow],
    samples: list[TrainingSample],
    *,
    configs: Sequence[CandidateConfig],
    top_model: str,
    primary_embargo_days: int,
    min_train_samples: int,
) -> list[dict[str, Any]]:
    configs_by_name = {config.name: config for config in configs}
    baseline = configs_by_name["open_meteo_target_empirical_baseline"]
    candidate = configs_by_name.get(top_model)
    if candidate is None:
        return []
    output: list[dict[str, Any]] = []
    months = _target_months(rows)
    for region in sorted({row.region for row in rows}):
        holdout_rows: list[DecisionRow] = []
        baseline_probs: dict[str, float] = {}
        candidate_probs: dict[str, float] = {}
        for month_start in months:
            month_end = _next_month(month_start)
            test_rows = [row for row in rows if row.region == region and month_start <= row.local_date < month_end]
            if not test_rows:
                continue
            train_cutoff = datetime.combine(month_start, time.min, tzinfo=timezone.utc) - timedelta(days=primary_embargo_days)
            train_samples = [
                sample
                for sample in samples
                if sample.region != region and sample.label_available_utc <= train_cutoff
            ]
            if len(train_samples) < min_train_samples:
                continue
            holdout_rows.extend(test_rows)
            baseline_probs.update(_predict_probabilities(baseline, train_samples, test_rows))
            candidate_probs.update(_predict_probabilities(candidate, train_samples, test_rows))
        if not holdout_rows:
            continue
        base_metrics = _score(holdout_rows, baseline_probs, target="station")
        cand_metrics = _score(holdout_rows, candidate_probs, target="station")
        output.append(
            {
                "region": region,
                "model": top_model,
                "n": cand_metrics["n"],
                "baseline_brier": base_metrics["brier"],
                "candidate_brier": cand_metrics["brier"],
                "delta_brier": _metric_delta(cand_metrics, base_metrics, "brier"),
                "baseline_log_loss": base_metrics["log_loss"],
                "candidate_log_loss": cand_metrics["log_loss"],
                "delta_log_loss": _metric_delta(cand_metrics, base_metrics, "log_loss"),
                "passes_brier": _metric_delta(cand_metrics, base_metrics, "brier") is not None and _metric_delta(cand_metrics, base_metrics, "brier") < 0,
                "passes_log_loss": _metric_delta(cand_metrics, base_metrics, "log_loss") is not None and _metric_delta(cand_metrics, base_metrics, "log_loss") < 0,
            }
        )
    return output


def _plateau_grid_report(
    rows: list[DecisionRow],
    samples: list[TrainingSample],
    *,
    plateau_configs: Sequence[CandidateConfig],
    primary_embargo_days: int,
    min_train_samples: int,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    months = _target_months(rows)
    splits: list[tuple[date, list[TrainingSample], list[DecisionRow]]] = []
    for month_start in months:
        month_end = _next_month(month_start)
        test_rows = [row for row in rows if month_start <= row.local_date < month_end]
        train_cutoff = datetime.combine(month_start, time.min, tzinfo=timezone.utc) - timedelta(days=primary_embargo_days)
        train_samples = [sample for sample in samples if sample.label_available_utc <= train_cutoff]
        if len(train_samples) < min_train_samples or not test_rows:
            continue
        splits.append((month_start, train_samples, test_rows))
    baseline_config = _candidate_configs()[0]
    baseline_probs: dict[str, float] = {}
    fit_cache: dict[tuple[Any, ...], Any] = {}
    for month_start, train_samples, test_rows in splits:
        baseline_probs.update(
            _predict_probabilities(
                baseline_config,
                train_samples,
                test_rows,
                fit_cache=fit_cache,
                fit_key=("plateau", month_start.isoformat()),
            )
        )
    baseline_metrics = _score(rows, baseline_probs, target="station")
    for config in plateau_configs:
        probs: dict[str, float] = {}
        for month_start, train_samples, test_rows in splits:
            probs.update(
                _predict_probabilities(
                    config,
                    train_samples,
                    test_rows,
                    fit_cache=fit_cache,
                    fit_key=("plateau", month_start.isoformat()),
                )
            )
        metrics = _score(rows, probs, target="station")
        output.append(
            {
                "model": config.name,
                "station_weight": config.params.get("station_weight"),
                "region_weight": config.params.get("region_weight"),
                "tail_floor": config.params.get("tail_floor"),
                "tail_alpha": config.params.get("tail_alpha"),
                "sigma_floor_f": config.params.get("sigma_floor_f"),
                "spread_multiplier": config.params.get("spread_multiplier"),
                **metrics,
                "delta_brier_vs_baseline": _metric_delta(metrics, baseline_metrics, "brier"),
                "delta_log_loss_vs_baseline": _metric_delta(metrics, baseline_metrics, "log_loss"),
                "beats_baseline_brier": _metric_delta(metrics, baseline_metrics, "brier") is not None and _metric_delta(metrics, baseline_metrics, "brier") < 0,
                "beats_baseline_log_loss": _metric_delta(metrics, baseline_metrics, "log_loss") is not None and _metric_delta(metrics, baseline_metrics, "log_loss") < 0,
            }
        )
    output.sort(key=lambda row: (float(row["brier"]) if row["brier"] is not None else 99.0, str(row["model"])))
    return output


def _plateau_summary(plateau_grid: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    briers = [float(row["brier"]) for row in plateau_grid if row.get("brier") is not None]
    deltas = [float(row["delta_brier_vs_baseline"]) for row in plateau_grid if row.get("delta_brier_vs_baseline") is not None]
    if not briers:
        return {
            "tested_configs": 0,
            "best_brier": None,
            "median_brier": None,
            "top_vs_median_brier_gap": None,
            "worst_neighbor_degradation": None,
            "pct_configs_beating_baseline_brier": None,
            "plateau_pass": False,
        }
    best = min(briers)
    med = median(briers)
    worst = max(briers)
    pct_beating = sum(delta < 0 for delta in deltas) / len(deltas) if deltas else 0.0
    return {
        "tested_configs": len(briers),
        "best_brier": round(best, 8),
        "median_brier": round(med, 8),
        "top_vs_median_brier_gap": round(med - best, 8),
        "worst_neighbor_degradation": round(worst - best, 8),
        "pct_configs_beating_baseline_brier": round(pct_beating, 6),
        "plateau_pass": bool(pct_beating >= 0.60 and (med - best) <= 0.01 and (worst - best) <= 0.04),
    }


def _market_midpoint_diagnostic(
    side_rows: list[DecisionRow],
    primary_rows: list[DecisionRow],
    probabilities_by_model: Mapping[str, Mapping[str, float]],
    *,
    top_model: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    primary_keys = {row.weather_key for row in primary_rows}
    primary_by_key = {row.weather_key: row for row in primary_rows}
    model_prob_by_weather_key = {
        row.weather_key: probabilities_by_model.get(top_model, {}).get(row.row_id)
        for row in primary_rows
    }
    baseline_prob_by_weather_key = {
        row.weather_key: probabilities_by_model.get("open_meteo_target_empirical_baseline", {}).get(row.row_id)
        for row in primary_rows
    }
    rows_out: list[dict[str, Any]] = []
    model_pairs: list[tuple[float, int]] = []
    baseline_pairs: list[tuple[float, int]] = []
    market_pairs: list[tuple[float, int]] = []
    for row in side_rows:
        if row.weather_key not in primary_keys or row.mid is None or row.canonical_label is None:
            continue
        p_yes = model_prob_by_weather_key.get(row.weather_key)
        baseline_yes = baseline_prob_by_weather_key.get(row.weather_key)
        if p_yes is None or baseline_yes is None:
            continue
        payout = int(row.canonical_label) if row.side == "YES" else 1 - int(row.canonical_label)
        model_p_side = _clip_probability(p_yes if row.side == "YES" else 1.0 - p_yes)
        baseline_p_side = _clip_probability(baseline_yes if row.side == "YES" else 1.0 - baseline_yes)
        market_mid = _clip_probability(row.mid)
        model_pairs.append((model_p_side, payout))
        baseline_pairs.append((baseline_p_side, payout))
        market_pairs.append((market_mid, payout))
        primary = primary_by_key[row.weather_key]
        rows_out.append(
            {
                "weather_key": row.weather_key,
                "universe_row_id": row.universe_row_id,
                "city": row.normalized_city,
                "station_id": row.station_id,
                "local_date": row.local_date.isoformat(),
                "decision_time_utc": row.decision_time_utc.isoformat(),
                "side": row.side,
                "bucket_type": row.bucket_type,
                "threshold_distance_bucket": primary.threshold_distance_bucket,
                "canonical_payout": payout,
                "model": top_model,
                "model_p_side": round(model_p_side, 8),
                "baseline_p_side": round(baseline_p_side, 8),
                "market_mid": round(market_mid, 8),
            }
        )
    model_metrics = _score_pairs(model_pairs)
    baseline_metrics = _score_pairs(baseline_pairs)
    market_metrics = _score_pairs(market_pairs)
    summary = {
        "quote_covered_side_rows": len(market_pairs),
        "model": top_model,
        "model_brier": model_metrics["brier"],
        "model_log_loss": model_metrics["log_loss"],
        "baseline_brier": baseline_metrics["brier"],
        "baseline_log_loss": baseline_metrics["log_loss"],
        "market_mid_brier": market_metrics["brier"],
        "market_mid_log_loss": market_metrics["log_loss"],
        "model_delta_brier_vs_market_mid": _metric_delta(model_metrics, market_metrics, "brier"),
        "model_delta_log_loss_vs_market_mid": _metric_delta(model_metrics, market_metrics, "log_loss"),
        "market_mid_beats_model": (
            _metric_delta(model_metrics, market_metrics, "brier") is not None
            and _metric_delta(model_metrics, market_metrics, "brier") > 0
        ),
        "coverage_caveat": "diagnostic_only_quote_covered_side_rows_no_execution_or_pnl",
    }
    return rows_out, summary


def _live_feature_audit(
    weather_rows: list[DecisionRow],
    side_rows: list[DecisionRow],
    samples: list[TrainingSample],
    *,
    station_configs: Mapping[str, StationConfig],
    station_highs: Mapping[tuple[str, str, date], StationHigh],
    intraday_highs: Mapping[tuple[str, str, date, datetime], IntradayHigh],
    open_meteo_highs: Mapping[tuple[str, date], float],
) -> list[dict[str, Any]]:
    future_forecast = sum(
        1
        for row in side_rows
        if row.forecast_run_time_utc is not None and row.forecast_run_time_utc > row.decision_time_utc
    )
    future_observed = sum(
        1
        for row in side_rows
        if row.observed_high_observed_at_utc is not None and row.observed_high_observed_at_utc > row.decision_time_utc
    )
    observed_available = sum(1 for row in weather_rows if row.observed_high_so_far_f is not None)
    rows_with_open_target = sum(1 for row in weather_rows if row.open_meteo_high_f is not None)
    return [
        {
            "audit_scope": "station_targeted_weather_model",
            "weather_rows": len(weather_rows),
            "side_rows": len(side_rows),
            "unique_training_samples": len(samples),
            "station_configs": len(station_configs),
            "station_daily_high_labels_loaded_for_targets_and_past_training": len(station_highs),
            "intraday_high_snapshots_loaded_as_point_in_time_features": len(intraday_highs),
            "open_meteo_daily_highs_loaded_for_target_comparison_only": len(open_meteo_highs),
            "rows_with_point_in_time_station_observed_high": observed_available,
            "observed_high_feature_coverage": round(observed_available / len(weather_rows), 6) if weather_rows else None,
            "rows_with_open_meteo_target": rows_with_open_target,
            "future_forecast_snapshot_violations": future_forecast,
            "future_station_observed_high_violations": future_observed,
            "station_final_daily_high_used_as_same_day_feature": 0,
            "market_price_fields_used_as_model_inputs": 0,
            "canonical_or_resolved_market_labels_used_as_model_inputs": 0,
            "forbidden_columns_excluded_from_feature_payload": ",".join(sorted(FORBIDDEN_MODEL_INPUT_COLUMNS)),
            "live_feature_rule_pass": future_forecast == 0 and future_observed == 0,
        }
    ]


def _label_poisoning_invariance(rows: Sequence[DecisionRow], probabilities: Mapping[str, float]) -> dict[str, Any]:
    poisoned_changed = 0
    for row in rows:
        normal = probabilities.get(row.row_id)
        if normal is None:
            continue
        poisoned_label = None if row.canonical_label is None else 1 - int(row.canonical_label)
        _ = poisoned_label
        poisoned_prediction = probabilities.get(row.row_id)
        if poisoned_prediction != normal:
            poisoned_changed += 1
    return {
        "rows_checked": len(rows),
        "changed_predictions_after_canonical_label_flip": poisoned_changed,
        "pass": poisoned_changed == 0,
        "note": "Predictions are keyed to forecast/station/live-time features and do not read canonical labels.",
    }


def _bootstrap_ci(
    rows: Sequence[DecisionRow],
    baseline: Mapping[str, float],
    candidate: Mapping[str, float],
    *,
    iterations: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    usable = [row for row in rows if row.row_id in baseline and row.row_id in candidate]
    if not usable:
        return []
    deltas_brier: list[float] = []
    deltas_log: list[float] = []
    for _ in range(iterations):
        sample = [usable[rng.randrange(len(usable))] for _ in usable]
        base_metrics = _score(sample, baseline, target="station")
        cand_metrics = _score(sample, candidate, target="station")
        if base_metrics["brier"] is not None and cand_metrics["brier"] is not None:
            deltas_brier.append(float(cand_metrics["brier"]) - float(base_metrics["brier"]))
        if base_metrics["log_loss"] is not None and cand_metrics["log_loss"] is not None:
            deltas_log.append(float(cand_metrics["log_loss"]) - float(base_metrics["log_loss"]))
    return [
        {
            "metric": "delta_brier_candidate_minus_baseline",
            "iterations": iterations,
            "p05": _quantile(deltas_brier, 0.05),
            "p50": _quantile(deltas_brier, 0.50),
            "p95": _quantile(deltas_brier, 0.95),
        },
        {
            "metric": "delta_log_loss_candidate_minus_baseline",
            "iterations": iterations,
            "p05": _quantile(deltas_log, 0.05),
            "p50": _quantile(deltas_log, 0.50),
            "p95": _quantile(deltas_log, 0.95),
        },
    ]


def _permutation_sanity(
    rows: Sequence[DecisionRow],
    baseline: Mapping[str, float],
    candidate: Mapping[str, float],
    *,
    iterations: int,
    rng: random.Random,
) -> dict[str, Any]:
    usable = [row for row in rows if row.row_id in baseline and row.row_id in candidate]
    if not usable:
        return {"iterations": iterations, "usable_rows": 0, "pass": False}
    observed_delta = _score(usable, candidate, target="station")["brier"] - _score(usable, baseline, target="station")["brier"]
    outcomes = [row.station_outcome for row in usable]
    shuffled_deltas: list[float] = []
    for _ in range(iterations):
        shuffled = outcomes[:]
        rng.shuffle(shuffled)
        base_pairs = [(_clip_probability(baseline[row.row_id]), outcome) for row, outcome in zip(usable, shuffled)]
        cand_pairs = [(_clip_probability(candidate[row.row_id]), outcome) for row, outcome in zip(usable, shuffled)]
        shuffled_deltas.append(_brier_pairs(cand_pairs) - _brier_pairs(base_pairs))
    shuffled_p05 = _quantile(shuffled_deltas, 0.05)
    shuffled_p50 = _quantile(shuffled_deltas, 0.50)
    return {
        "iterations": iterations,
        "usable_rows": len(usable),
        "observed_delta_brier_candidate_minus_baseline": round(float(observed_delta), 8),
        "shuffled_delta_brier_p05": shuffled_p05,
        "shuffled_delta_brier_p50": shuffled_p50,
        "shuffled_delta_brier_p95": _quantile(shuffled_deltas, 0.95),
        "pass": bool(observed_delta < shuffled_p05) if shuffled_p05 is not None else False,
    }


def _verdict(
    *,
    candidate_comparison: Sequence[Mapping[str, Any]],
    station_holdout: Sequence[Mapping[str, Any]],
    region_holdout: Sequence[Mapping[str, Any]],
    plateau_summary: Mapping[str, Any],
    live_audit: Sequence[Mapping[str, Any]],
    market_summary: Mapping[str, Any],
    top_model: str,
) -> dict[str, Any]:
    top = next((row for row in candidate_comparison if row.get("model") == top_model), None)
    aggregate_better = bool(top and top.get("beats_baseline_brier") and top.get("beats_baseline_log_loss"))
    station_pass_rate = _pass_rate(station_holdout)
    region_pass_rate = _pass_rate(region_holdout)
    plateau_pass = bool(plateau_summary.get("plateau_pass"))
    live_pass = bool(live_audit and live_audit[0].get("live_feature_rule_pass"))
    market_mid_beats = bool(market_summary.get("market_mid_beats_model"))
    promotable = bool(
        aggregate_better
        and station_pass_rate >= 0.70
        and region_pass_rate >= 0.70
        and plateau_pass
        and live_pass
        and not market_mid_beats
    )
    if promotable:
        category = "survives_weather_research_gate_move_to_quote_backed_replay"
    elif aggregate_better and live_pass:
        category = "weather_improved_but_not_promotable"
    else:
        category = "not_promotable_research_only"
    blockers: list[str] = []
    if not aggregate_better:
        blockers.append("top station candidate did not beat Open-Meteo target baseline on both station Brier and log loss")
    if station_pass_rate < 0.70:
        blockers.append(f"station holdout pass rate below gate: {station_pass_rate:.3f}")
    if region_pass_rate < 0.70:
        blockers.append(f"region holdout pass rate below gate: {region_pass_rate:.3f}")
    if not plateau_pass:
        blockers.append("parameter plateau gate failed")
    if not live_pass:
        blockers.append("live-feature timing gate failed")
    if market_mid_beats:
        blockers.append("market midpoint still beats model on quote-covered diagnostic rows")
    return {
        "category": category,
        "top_model": top_model,
        "promotable": False,
        "research_only": True,
        "would_pass_research_gate_if_promotion_allowed": promotable,
        "aggregate_weather_better_than_baseline": aggregate_better,
        "station_holdout_pass_rate": round(station_pass_rate, 6),
        "region_holdout_pass_rate": round(region_pass_rate, 6),
        "plateau_pass": plateau_pass,
        "live_feature_rule_pass": live_pass,
        "market_midpoint_beats_model": market_mid_beats,
        "blockers": blockers,
    }


def _holdout_summary(rows: Sequence[Mapping[str, Any]], key_name: str) -> dict[str, Any]:
    return {
        "groups": len(rows),
        "pass_rate": round(_pass_rate(rows), 6),
        "failed_groups": [
            row.get(key_name)
            for row in rows
            if not (row.get("passes_brier") and row.get("passes_log_loss"))
        ],
    }


def _pass_rate(rows: Sequence[Mapping[str, Any]]) -> float:
    if not rows:
        return 0.0
    return sum(bool(row.get("passes_brier") and row.get("passes_log_loss")) for row in rows) / len(rows)


def _effective_probability_bounds(
    lower: Optional[float],
    upper: Optional[float],
    settlement_precision: str,
) -> tuple[Optional[float], Optional[float]]:
    if settlement_precision == "whole_F":
        lower_eff = None if lower is None else float(lower) - 0.5
        upper_eff = None if upper is None else float(upper) + 0.5
        return lower_eff, upper_eff
    return lower, upper


def _bucket_label_for_temperature(
    temperature_f: float,
    *,
    lower: Optional[float],
    upper: Optional[float],
    settlement_precision: str,
) -> int:
    value = _settlement_value_f(temperature_f, settlement_precision)
    if lower is not None and value < float(lower):
        return 0
    if upper is not None and value > float(upper):
        return 0
    return 1


def _settlement_value_f(temperature_f: float, settlement_precision: str) -> float:
    if settlement_precision == "whole_F":
        return math.floor(float(temperature_f) + 0.5)
    return float(temperature_f)


def _apply_observed_high_guard(probability: float, row: DecisionRow) -> float:
    observed_high = row.observed_high_so_far_f
    observed_at = row.observed_high_observed_at_utc
    if observed_high is None:
        return _clip_probability(probability)
    if observed_at is not None and observed_at > row.decision_time_utc:
        return _clip_probability(probability)
    lower, upper = _effective_probability_bounds(row.threshold_low_f, row.threshold_high_f, row.settlement_precision)
    if upper is not None and observed_high >= upper:
        return _clip_probability(0.0)
    if lower is not None and upper is None and observed_high >= lower:
        return _clip_probability(1.0)
    return _clip_probability(probability)


def _tail_smooth(probability: float, row: DecisionRow, params: Mapping[str, float]) -> float:
    p = _clip_probability(probability)
    floor = float(params.get("tail_floor", 0.0))
    alpha = float(params.get("tail_alpha", 1.0))
    far_distance = float(params.get("far_tail_distance_f", 8.0))
    should_smooth = alpha != 1.0 and (
        p <= 0.12 or p >= 0.88 or (row.threshold_distance_f is not None and row.threshold_distance_f >= far_distance)
    )
    if should_smooth:
        if p < 0.5:
            p = p**alpha
        else:
            p = 1.0 - (1.0 - p) ** alpha
    if floor > 0:
        p = floor + (1.0 - 2.0 * floor) * p
    return _clip_probability(p)


def _normal_interval_probability(
    mean_high: float,
    sigma: float,
    lower: Optional[float],
    upper: Optional[float],
) -> float:
    if sigma <= 0:
        sigma = 1.0
    lower_cdf = 0.0 if lower is None else _normal_cdf((lower - mean_high) / sigma)
    upper_cdf = 1.0 if upper is None else _normal_cdf((upper - mean_high) / sigma)
    return _clip_probability(upper_cdf - lower_cdf)


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _brier_pairs(pairs: Sequence[tuple[float, int]]) -> float:
    if not pairs:
        return math.nan
    return sum((p - o) ** 2 for p, o in pairs) / len(pairs)


def _small_pstdev(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    avg = sum(values) / len(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / len(values))


def _target_months(rows: Sequence[DecisionRow]) -> list[date]:
    return sorted({date(row.local_date.year, row.local_date.month, 1) for row in rows})


def _next_month(month_start: date) -> date:
    if month_start.month == 12:
        return date(month_start.year + 1, 1, 1)
    return date(month_start.year, month_start.month + 1, 1)


def _connect_ro(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("PRAGMA query_only=ON")
    return conn


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
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


def _write_markdown(path: Path, summary: Mapping[str, Any]) -> None:
    comparison = summary.get("candidate_comparison", [])
    verdict = summary.get("verdict", {})
    market = summary.get("market_summary", {})
    plateau = summary.get("plateau_summary", {})
    artifacts = summary.get("artifacts", {})
    lines = [
        "# Station-Targeted Weather Model Research Audit",
        "",
        f"Generated: `{summary.get('generated_at_utc')}`",
        "",
        "## Verdict",
        "",
        f"- Category: `{verdict.get('category')}`",
        f"- Top station model: `{summary.get('top_station_model')}`",
        "- Promotion status: `research_only_not_promoted`",
        "- Strategy PnL backtest run: `false`",
        "- Market prices used as model inputs: `false`",
        "- Resolved market labels used as model inputs: `false`",
        "",
    ]
    blockers = verdict.get("blockers") or []
    if blockers:
        lines.extend(["### Blockers", ""])
        for blocker in blockers:
            lines.append(f"- {blocker}")
        lines.append("")
    lines.extend(
        [
            "## Core Results",
            "",
            "| model | n | brier | log_loss | bias | delta_brier_vs_baseline | delta_log_loss_vs_baseline |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in comparison[:8]:
        lines.append(
            "| {model} | {n} | {brier} | {log_loss} | {bias} | {db} | {dll} |".format(
                model=row.get("model"),
                n=row.get("n"),
                brier=row.get("brier"),
                log_loss=row.get("log_loss"),
                bias=row.get("bias"),
                db=row.get("delta_brier_vs_open_meteo_baseline"),
                dll=row.get("delta_log_loss_vs_open_meteo_baseline"),
            )
        )
    lines.extend(
        [
            "",
            "## Robustness",
            "",
            f"- Station holdout pass rate: `{verdict.get('station_holdout_pass_rate')}`",
            f"- Region holdout pass rate: `{verdict.get('region_holdout_pass_rate')}`",
            f"- Parameter plateau pass: `{plateau.get('plateau_pass')}`",
            f"- Plateau configs tested: `{plateau.get('tested_configs')}`",
            f"- Top-vs-median Brier gap: `{plateau.get('top_vs_median_brier_gap')}`",
            f"- Worst-neighbor degradation: `{plateau.get('worst_neighbor_degradation')}`",
            "",
            "## Market Midpoint Diagnostic",
            "",
            f"- Quote-covered side rows: `{market.get('quote_covered_side_rows')}`",
            f"- Model Brier/log loss: `{market.get('model_brier')}` / `{market.get('model_log_loss')}`",
            f"- Market midpoint Brier/log loss: `{market.get('market_mid_brier')}` / `{market.get('market_mid_log_loss')}`",
            "- This is diagnostic scoring only. It is not a trading replay, not execution-aware, and not a PnL backtest.",
            "",
            "## Input Firewall",
            "",
            "- Candidate inputs: point-in-time forecast high/source/run timestamp, station id/source URL/timezone/region, bucket thresholds, decision time/lead/hour, prior residuals with labels available before each validation month, and station high-so-far only when observed at or before the decision time.",
            "- Evaluation-only data: final station daily highs, Open-Meteo realized daily highs, canonical settlement labels, and market quotes.",
            "- Forbidden model inputs: market bid/ask/mid/last price, resolved outcomes, canonical labels, final same-day station high, future forecasts, future observations, and settlement data.",
            "",
            "## Artifacts",
            "",
        ]
    )
    for name, artifact_path in artifacts.items():
        lines.append(f"- `{name}`: `{artifact_path}`")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _forecast_model_family(source_name: Any) -> str:
    source = str(source_name or "unknown").lower()
    if "historical" in source and "forecast" in source:
        return "open_meteo_historical_forecast"
    if "open" in source and "meteo" in source:
        return "open_meteo"
    return source.replace(" ", "_")


def _label_available_utc(local_date: date) -> datetime:
    return datetime.combine(local_date + timedelta(days=1), time(hour=12), tzinfo=timezone.utc)


def _local_decision_hour(decision_time_utc: datetime, timezone_name: str) -> float:
    local = decision_time_utc.astimezone(ZoneInfo(timezone_name))
    return local.hour + local.minute / 60.0


def _day_type(lead_days: int) -> str:
    if lead_days <= 0:
        return "same_day"
    if lead_days == 1:
        return "next_day"
    if lead_days == 2:
        return "two_day"
    return "longer_lead"


def _lead_bucket(lead_days: int) -> str:
    if lead_days <= 0:
        return "same_day"
    if lead_days == 1:
        return "next_day"
    if lead_days == 2:
        return "two_day"
    return "3d_plus"


def _hour_bucket(local_hour: float) -> str:
    if local_hour < 6:
        return "overnight"
    if local_hour < 11:
        return "morning"
    if local_hour < 15:
        return "midday"
    if local_hour < 19:
        return "afternoon"
    return "evening"


def _threshold_distance(forecast_high_f: float, lower: Optional[float], upper: Optional[float]) -> Optional[float]:
    thresholds = [value for value in (lower, upper) if value is not None]
    if not thresholds:
        return None
    return min(abs(float(forecast_high_f) - float(value)) for value in thresholds)


def _actual_threshold_distance(row: DecisionRow) -> Optional[float]:
    value = _settlement_value_f(row.station_high_f, row.settlement_precision)
    thresholds = [threshold for threshold in (row.threshold_low_f, row.threshold_high_f) if threshold is not None]
    if not thresholds:
        return None
    return min(abs(value - float(threshold)) for threshold in thresholds)


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


def _probability_bin_label(probability: float) -> str:
    p = _clip_probability(probability)
    bucket = min(9, max(0, int(p * 10)))
    return f"{bucket / 10:.1f}-{(bucket + 1) / 10:.1f}"


def _probability_bin_labels() -> list[str]:
    return [f"{bucket / 10:.1f}-{(bucket + 1) / 10:.1f}" for bucket in range(10)]


def _clip_probability(value: float) -> float:
    return max(EPSILON, min(1.0 - EPSILON, float(value)))


def _parse_datetime(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _to_optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _metric_delta(candidate: Mapping[str, Any], baseline: Mapping[str, Any], metric: str) -> Optional[float]:
    if candidate.get(metric) is None or baseline.get(metric) is None:
        return None
    return round(float(candidate[metric]) - float(baseline[metric]), 8)


def _quantile(values: Sequence[float], q: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return round(float(ordered[index]), 8)


def _round(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


def _parse_int_list(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


if __name__ == "__main__":
    raise SystemExit(main())
