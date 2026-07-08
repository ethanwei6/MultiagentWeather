from __future__ import annotations

import argparse
import json
import math
import random
import sqlite3
import sys
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from statistics import median, mean, pstdev
from typing import Any, Callable, Mapping, Optional

from .weather_model_live_readiness import _audit_decision_artifact_live_features
from .weather_model_robustness import (
    _bootstrap_confidence_intervals,
    _comparison_metrics,
    _permutation_sanity_check,
    _region,
    _walk_forward_market_diagnostics,
)
from .weather_model_validation import (
    DEFAULT_END_DATE,
    DEFAULT_SOURCE_REPO,
    DEFAULT_START_DATE,
    ValidationRow,
    _artifact_paths,
    _build_validation_rows,
    _clip_probability,
    _load_forecasts,
    _load_markets,
    _load_realized,
    _load_weather_strategy,
    _metrics,
    _model_family,
    _write_csv,
)


@dataclass(frozen=True)
class CandidateConfig:
    config_id: str
    family: str
    kind: str
    params: Mapping[str, Any]


@dataclass(frozen=True)
class EmpiricalDistribution:
    values: tuple[float, ...]

    @property
    def n(self) -> int:
        return len(self.values)

    def interval_probability(self, lower: Optional[float], upper: Optional[float]) -> float:
        if not self.values:
            return 0.0
        left = 0 if lower is None else bisect_left(self.values, lower)
        right = len(self.values) if upper is None else bisect_right(self.values, upper)
        return max(0, right - left) / len(self.values)


@dataclass(frozen=True)
class BiasStats:
    n: int
    mean: float
    sigma: float


@dataclass(frozen=True)
class ObservedHighContext:
    observed_high_f: Optional[float]
    observed_at_utc: Optional[datetime]
    is_final_at_decision: bool


ObservedHighContextMap = Mapping[tuple[str, date, datetime], ObservedHighContext]


FOCUS_CITIES = {
    "Madrid, ES",
    "Seoul, KR",
    "Moscow, RU",
    "Tokyo, JP",
    "London, GB",
    "Los Angeles, CA",
    "Houston, TX",
    "Chicago, IL",
}


def _load_observed_high_context(
    path: Path,
    *,
    start: date,
    end: date,
) -> dict[tuple[str, date, datetime], ObservedHighContext]:
    if not path.exists():
        return {}
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("PRAGMA query_only=ON")
    try:
        rows = conn.execute(
            """
            SELECT normalized_city,
                   market_date_local,
                   decision_time_utc,
                   observed_high_so_far_f,
                   observed_high_observed_at_utc,
                   observed_high_is_final_at_decision
            FROM observed_high_snapshots
            WHERE market_date_local >= ?
              AND market_date_local <= ?
              AND observed_high_so_far_f IS NOT NULL
            ORDER BY market_date_local, normalized_city, decision_time_utc
            """,
            (start.isoformat(), end.isoformat()),
        ).fetchall()
    finally:
        conn.close()
    context: dict[tuple[str, date, datetime], ObservedHighContext] = {}
    for row in rows:
        decision_time = _parse_observed_datetime(row["decision_time_utc"])
        if decision_time is None:
            continue
        key = (
            str(row["normalized_city"]),
            date.fromisoformat(str(row["market_date_local"])),
            decision_time,
        )
        context[key] = ObservedHighContext(
            observed_high_f=float(row["observed_high_so_far_f"]),
            observed_at_utc=_parse_observed_datetime(row["observed_high_observed_at_utc"]),
            is_final_at_decision=bool(int(row["observed_high_is_final_at_decision"] or 0)),
        )
    return context


def _parse_observed_datetime(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _observed_high_context_summary(
    rows: list[ValidationRow],
    observed_context: ObservedHighContextMap,
) -> dict[str, Any]:
    if not rows:
        return {
            "primary_rows": 0,
            "rows_with_point_in_time_observed_high": 0,
            "observed_high_fraction": None,
            "future_observed_high_violations": 0,
            "rows_with_final_high_known_at_decision": 0,
        }
    available = 0
    future = 0
    final_known = 0
    for row in rows:
        raw = observed_context.get((row.normalized_city, row.market_date_local, row.decision_time_utc))
        if raw is None:
            continue
        if raw.observed_at_utc is not None and raw.observed_at_utc > row.decision_time_utc:
            future += 1
            continue
        available += 1
        if raw.is_final_at_decision:
            final_known += 1
    return {
        "primary_rows": len(rows),
        "rows_with_point_in_time_observed_high": available,
        "observed_high_fraction": round(available / len(rows), 6),
        "future_observed_high_violations": future,
        "rows_with_final_high_known_at_decision": final_known,
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate new DailyWeather weather-model improvement candidates")
    parser.add_argument("--work-root", default=".")
    parser.add_argument("--source-repo", default=str(DEFAULT_SOURCE_REPO))
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=DEFAULT_END_DATE)
    parser.add_argument("--output-dir", default="work/reports/weather_model_improvement")
    parser.add_argument("--min-train-rows", type=int, default=12000)
    parser.add_argument("--min-test-rows", type=int, default=4000)
    parser.add_argument("--min-inner-train-rows", type=int, default=5000)
    parser.add_argument("--min-inner-test-rows", type=int, default=1000)
    parser.add_argument("--embargo-days", default="0,3,7")
    parser.add_argument("--max-configs", type=int, default=0, help="Optional smoke cap; 0 means full candidate grid")
    parser.add_argument("--bootstrap-iterations", type=int, default=200)
    parser.add_argument("--permutation-iterations", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20260702)
    args = parser.parse_args(argv)

    summary = run_improvement_audit(
        work_root=Path(args.work_root).resolve(),
        source_repo=Path(args.source_repo).expanduser().resolve(),
        start=date.fromisoformat(args.start_date),
        end=date.fromisoformat(args.end_date),
        output_dir=Path(args.output_dir),
        min_train_rows=args.min_train_rows,
        min_test_rows=args.min_test_rows,
        min_inner_train_rows=args.min_inner_train_rows,
        min_inner_test_rows=args.min_inner_test_rows,
        embargo_days=_parse_int_list(args.embargo_days),
        max_configs=args.max_configs,
        bootstrap_iterations=args.bootstrap_iterations,
        permutation_iterations=args.permutation_iterations,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def run_improvement_audit(
    *,
    work_root: Path,
    source_repo: Path,
    start: date,
    end: date,
    output_dir: Path,
    min_train_rows: int,
    min_test_rows: int,
    min_inner_train_rows: int,
    min_inner_test_rows: int,
    embargo_days: list[int],
    max_configs: int,
    bootstrap_iterations: int,
    permutation_iterations: int,
    seed: int,
) -> dict[str, Any]:
    strategy = _load_weather_strategy(source_repo)
    paths = _artifact_paths(work_root)
    markets = _load_markets(paths["market_db"], start=start, end=end)
    forecasts = _load_forecasts(paths["weather_db"], start=start, end=end)
    realized = _load_realized(paths["weather_db"], start=start, end=end)
    rows = _build_validation_rows(strategy, markets, forecasts, realized)
    if len(rows) < 100:
        raise ValueError(f"Need at least 100 validation rows; got {len(rows)}")
    observed_context = _load_observed_high_context(paths["decision_db"], start=start, end=end)

    configs = _candidate_grid(max_configs=max_configs)
    print(
        f"[weather-improvement] rows={len(rows)} configs={len(configs)} observed_contexts={len(observed_context)} "
        f"families={sorted({cfg.family for cfg in configs})}",
        file=sys.stderr,
        flush=True,
    )
    split_rows, plateau_rows, leakage_rows, aggregate = _run_nested_walk_forward(
        rows,
        configs=configs,
        observed_context=observed_context,
        embargo_days=embargo_days,
        min_train_rows=min_train_rows,
        min_test_rows=min_test_rows,
        min_inner_train_rows=min_inner_train_rows,
        min_inner_test_rows=min_inner_test_rows,
    )
    primary_rows = aggregate["rows"]
    baseline_probabilities = aggregate["baseline_probabilities"]
    candidate_comparison = _candidate_comparison(primary_rows, baseline_probabilities, aggregate["candidate_probabilities_by_family"])
    plateau_summary = _plateau_summary(plateau_rows)
    observed_context_summary = _observed_high_context_summary(primary_rows, observed_context)
    bucket_tail = _bucket_tail_diagnostics(
        primary_rows,
        baseline_probabilities,
        aggregate["candidate_probabilities_by_family"],
        observed_context=observed_context,
    )
    city_holdouts, region_holdouts = _holdout_diagnostics(
        rows,
        configs=configs,
        min_train_rows=min_train_rows,
        observed_context=observed_context,
    )
    focused_holdout_slices = _focused_holdout_slices(
        rows,
        configs=configs,
        min_train_rows=min_train_rows,
        observed_context=observed_context,
    )
    focused_primary_diagnostics = _focused_primary_diagnostics(
        primary_rows,
        baseline_probabilities,
        aggregate["candidate_probabilities_by_family"],
        observed_context=observed_context,
    )
    forecast_residual_diagnostics = _forecast_residual_diagnostics(rows)
    unit_bucket_audit = _unit_bucket_audit(primary_rows, baseline_probabilities, aggregate["candidate_probabilities_by_family"])
    alignment_diagnostics = _alignment_diagnostics(rows)
    feature_key_audit = _feature_key_audit(rows)
    old_model_family_comparison = _old_model_family_comparison(primary_rows)
    market_rows, market_summary = _market_comparison(paths["decision_db"], primary_rows, baseline_probabilities, aggregate["candidate_probabilities_by_family"])

    rng = random.Random(seed)
    top_family = _top_family(candidate_comparison)
    top_probabilities = aggregate["candidate_probabilities_by_family"].get(top_family, {})
    bootstrap = _bootstrap_confidence_intervals(
        primary_rows,
        baseline_probabilities,
        top_probabilities,
        iterations=bootstrap_iterations,
        rng=rng,
    ) if top_family else {}
    permutation = _permutation_sanity_check(
        primary_rows,
        baseline_probabilities,
        top_probabilities,
        iterations=permutation_iterations,
        rng=rng,
    ) if top_family else {}
    print("[weather-improvement] running live-feature audit", file=sys.stderr, flush=True)
    live_feature_gate = _audit_decision_artifact_live_features(paths["decision_db"], start=start, end=end)
    print("[weather-improvement] running label-poisoning audit", file=sys.stderr, flush=True)
    label_poisoning = _label_poisoning_invariance(
        rows,
        configs=configs,
        min_train_rows=min_train_rows,
        min_test_rows=min_test_rows,
        observed_context=observed_context,
    )

    decision = _decision(
        candidate_comparison=candidate_comparison,
        plateau_summary=plateau_summary,
        city_holdouts=city_holdouts,
        region_holdouts=region_holdouts,
        live_feature_gate=live_feature_gate["summary"],
        label_poisoning=label_poisoning,
        market_summary=market_summary,
        top_family=top_family,
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_dir = (work_root / output_dir).resolve() if not output_dir.is_absolute() else output_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{timestamp}-weather-model-improvement"
    artifacts = {
        "summary_json": str(report_dir / f"{prefix}.json"),
        "report_md": str(report_dir / f"{prefix}.md"),
        "candidate_comparison_csv": str(report_dir / f"{prefix}-candidate-comparison.csv"),
        "outer_walk_forward_split_metrics_csv": str(report_dir / f"{prefix}-outer-walk-forward-split-metrics.csv"),
        "parameter_plateau_csv": str(report_dir / f"{prefix}-parameter-plateau.csv"),
        "plateau_summary_csv": str(report_dir / f"{prefix}-plateau-summary.csv"),
        "city_holdout_csv": str(report_dir / f"{prefix}-city-holdout.csv"),
        "region_holdout_csv": str(report_dir / f"{prefix}-region-holdout.csv"),
        "bucket_tail_diagnostics_csv": str(report_dir / f"{prefix}-bucket-tail-diagnostics.csv"),
        "focused_primary_diagnostics_csv": str(report_dir / f"{prefix}-focused-primary-diagnostics.csv"),
        "focused_holdout_slices_csv": str(report_dir / f"{prefix}-focused-holdout-slices.csv"),
        "forecast_residual_diagnostics_csv": str(report_dir / f"{prefix}-forecast-residual-diagnostics.csv"),
        "unit_bucket_audit_csv": str(report_dir / f"{prefix}-unit-bucket-audit.csv"),
        "alignment_diagnostics_csv": str(report_dir / f"{prefix}-alignment-diagnostics.csv"),
        "feature_key_audit_csv": str(report_dir / f"{prefix}-feature-key-audit.csv"),
        "old_model_family_comparison_csv": str(report_dir / f"{prefix}-old-model-family-comparison.csv"),
        "market_comparison_csv": str(report_dir / f"{prefix}-market-comparison.csv"),
        "leakage_label_availability_csv": str(report_dir / f"{prefix}-leakage-label-availability.csv"),
        "bootstrap_csv": str(report_dir / f"{prefix}-bootstrap-ci.csv"),
        "permutation_csv": str(report_dir / f"{prefix}-permutation.csv"),
        "live_feature_coverage_csv": str(report_dir / f"{prefix}-live-feature-coverage.csv"),
        "label_poisoning_invariance_csv": str(report_dir / f"{prefix}-label-poisoning-invariance.csv"),
        "diagnosis_report_md": str(report_dir / f"{prefix}-failure-diagnosis.md"),
    }
    _write_csv(Path(artifacts["candidate_comparison_csv"]), candidate_comparison)
    _write_csv(Path(artifacts["outer_walk_forward_split_metrics_csv"]), split_rows)
    _write_csv(Path(artifacts["parameter_plateau_csv"]), plateau_rows)
    _write_csv(Path(artifacts["plateau_summary_csv"]), list(plateau_summary.values()))
    _write_csv(Path(artifacts["city_holdout_csv"]), city_holdouts)
    _write_csv(Path(artifacts["region_holdout_csv"]), region_holdouts)
    _write_csv(Path(artifacts["bucket_tail_diagnostics_csv"]), bucket_tail)
    _write_csv(Path(artifacts["focused_primary_diagnostics_csv"]), focused_primary_diagnostics)
    _write_csv(Path(artifacts["focused_holdout_slices_csv"]), focused_holdout_slices)
    _write_csv(Path(artifacts["forecast_residual_diagnostics_csv"]), forecast_residual_diagnostics)
    _write_csv(Path(artifacts["unit_bucket_audit_csv"]), unit_bucket_audit)
    _write_csv(Path(artifacts["alignment_diagnostics_csv"]), alignment_diagnostics)
    _write_csv(Path(artifacts["feature_key_audit_csv"]), feature_key_audit)
    _write_csv(Path(artifacts["old_model_family_comparison_csv"]), old_model_family_comparison)
    _write_csv(Path(artifacts["market_comparison_csv"]), market_rows)
    _write_csv(Path(artifacts["leakage_label_availability_csv"]), leakage_rows)
    _write_csv(Path(artifacts["bootstrap_csv"]), _flatten_bootstrap(bootstrap, top_family=top_family))
    _write_csv(Path(artifacts["permutation_csv"]), [{"candidate_family": top_family, **permutation}] if permutation else [])
    _write_csv(Path(artifacts["live_feature_coverage_csv"]), [live_feature_gate["summary"]])
    _write_csv(Path(artifacts["label_poisoning_invariance_csv"]), label_poisoning)

    summary = {
        "role": "Weather Model Improvement Agent",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_repo": str(source_repo),
        "target_start_date": start.isoformat(),
        "target_end_date": end.isoformat(),
        "input_counts": {
            "market_rows": len(markets),
            "forecast_rows": len(forecasts),
            "realized_rows_for_evaluation_only": len(realized),
            "validation_rows_for_audit": len(rows),
            "primary_outer_test_rows": len(primary_rows),
            "candidate_config_count": len(configs),
            "candidate_families": sorted({cfg.family for cfg in configs}),
            "observed_high_contexts_loaded": len(observed_context),
        },
        "observed_high_context_summary": observed_context_summary,
        "candidate_comparison": candidate_comparison,
        "top_family": top_family,
        "plateau_summary": plateau_summary,
        "city_holdout_summary": _holdout_summary(city_holdouts),
        "region_holdout_summary": _holdout_summary(region_holdouts),
        "focused_diagnosis_summary": _focused_diagnosis_summary(
            focused_holdout_slices=focused_holdout_slices,
            forecast_residual_diagnostics=forecast_residual_diagnostics,
            unit_bucket_audit=unit_bucket_audit,
            feature_key_audit=feature_key_audit,
        ),
        "bootstrap_confidence_intervals_top_family": bootstrap,
        "permutation_sanity_check_top_family": permutation,
        "live_feature_gate": live_feature_gate["summary"],
        "label_poisoning_invariance_summary": _label_poisoning_summary(label_poisoning),
        "market_summary": market_summary,
        "decision": decision,
        "output_artifacts": artifacts,
        "promotion_status": "research_only_not_promoted",
        "model_weights_or_defaults_changed": False,
        "strategy_pnl_backtest_run": False,
    }
    Path(artifacts["summary_json"]).write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_markdown(Path(artifacts["report_md"]), summary)
    _write_diagnosis_markdown(Path(artifacts["diagnosis_report_md"]), summary)
    return summary


def _candidate_grid(*, max_configs: int) -> list[CandidateConfig]:
    configs: list[CandidateConfig] = []
    for city_weight, exact_padding, tail_floor in (
        (0.12, 0.00, 0.010),
        (0.18, 0.15, 0.015),
        (0.24, 0.30, 0.020),
        (0.30, 0.45, 0.025),
    ):
        configs.append(
            CandidateConfig(
                config_id=f"hier_shrink|city={city_weight:.2f}|exact={exact_padding:.2f}|tail={tail_floor:.3f}",
                family="hierarchical_shrinkage_residual",
                kind="shrinkage",
                params={
                    "city_weight": city_weight,
                    "exact_padding_f": exact_padding,
                    "celsius_exact_padding_f": exact_padding * 0.6,
                    "range_padding_f": exact_padding * 0.3,
                    "tail_floor": tail_floor,
                    "tail_alpha": 0.72,
                },
            )
        )
    for shrink_count, sigma_floor, spread_multiplier in (
        (60.0, 1.15, 0.20),
        (100.0, 1.35, 0.30),
        (160.0, 1.55, 0.45),
        (240.0, 1.80, 0.60),
    ):
        configs.append(
            CandidateConfig(
                config_id=f"source_bias|shrink={shrink_count:.0f}|sigma={sigma_floor:.2f}|spread={spread_multiplier:.2f}",
                family="source_model_family_bias_normal",
                kind="source_bias_normal",
                params={
                    "shrink_count": shrink_count,
                    "sigma_floor_f": sigma_floor,
                    "spread_multiplier": spread_multiplier,
                    "exact_padding_f": 0.12,
                    "tail_floor": 0.010,
                },
            )
        )
    for exact_padding, celsius_padding, tail_floor, tail_alpha in (
        (0.00, 0.00, 0.015, 0.65),
        (0.12, 0.05, 0.020, 0.65),
        (0.25, 0.10, 0.025, 0.55),
        (0.38, 0.16, 0.030, 0.50),
    ):
        configs.append(
            CandidateConfig(
                config_id=f"exact_tail|exact={exact_padding:.2f}|c={celsius_padding:.2f}|tail={tail_floor:.3f}|alpha={tail_alpha:.2f}",
                family="exact_bucket_tail_calibrated",
                kind="shrinkage",
                params={
                    "city_weight": 0.14,
                    "exact_padding_f": exact_padding,
                    "celsius_exact_padding_f": celsius_padding,
                    "range_padding_f": exact_padding * 0.25,
                    "tail_floor": tail_floor,
                    "tail_alpha": tail_alpha,
                },
            )
        )
    for tail_floor, far_tail_weight, city_guard, region_guard in (
        (0.006, 1.00, 0.25, 0.70),
        (0.010, 1.00, 0.40, 0.85),
        (0.014, 0.85, 0.55, 0.95),
        (0.018, 0.70, 0.70, 1.00),
    ):
        configs.append(
            CandidateConfig(
                config_id=(
                    "guarded_shrink|"
                    f"tail={tail_floor:.3f}|far={far_tail_weight:.2f}|city_guard={city_guard:.2f}|region_guard={region_guard:.2f}"
                ),
                family="baseline_guarded_shrinkage",
                kind="shrinkage",
                params={
                    "city_weight": 0.12,
                    "exact_padding_f": 0.0,
                    "celsius_exact_padding_f": 0.0,
                    "range_padding_f": 0.0,
                    "tail_floor": tail_floor,
                    "tail_alpha": 0.65,
                    "far_tail_guard_distance_f": 8.0,
                    "far_tail_baseline_weight": far_tail_weight,
                    "city_cold_start_baseline_weight": city_guard,
                    "region_cold_start_baseline_weight": region_guard,
                    "cold_start_min_count": 35.0,
                },
            )
        )
    for tail_floor, far_tail_weight, city_guard, region_guard in (
        (0.006, 1.00, 0.25, 0.70),
        (0.010, 1.00, 0.40, 0.85),
        (0.014, 0.85, 0.55, 0.95),
        (0.018, 0.70, 0.70, 1.00),
    ):
        configs.append(
            CandidateConfig(
                config_id=(
                    "observed_guarded|"
                    f"tail={tail_floor:.3f}|far={far_tail_weight:.2f}|city_guard={city_guard:.2f}|region_guard={region_guard:.2f}"
                ),
                family="observed_high_guarded_shrinkage",
                kind="observed_high_overlay",
                params={
                    "city_weight": 0.12,
                    "exact_padding_f": 0.0,
                    "celsius_exact_padding_f": 0.0,
                    "range_padding_f": 0.0,
                    "tail_floor": tail_floor,
                    "tail_alpha": 0.65,
                    "far_tail_guard_distance_f": 8.0,
                    "far_tail_baseline_weight": far_tail_weight,
                    "city_cold_start_baseline_weight": city_guard,
                    "region_cold_start_baseline_weight": region_guard,
                    "cold_start_min_count": 35.0,
                },
            )
        )
    for residual_weight, source_weight, baseline_weight, tail_floor in (
        (0.55, 0.30, 0.15, 0.010),
        (0.45, 0.40, 0.15, 0.015),
        (0.60, 0.25, 0.15, 0.020),
        (0.50, 0.25, 0.25, 0.020),
    ):
        configs.append(
            CandidateConfig(
                config_id=f"ensemble|resid={residual_weight:.2f}|source={source_weight:.2f}|base={baseline_weight:.2f}|tail={tail_floor:.3f}",
                family="ensemble_distribution",
                kind="ensemble",
                params={
                    "residual_weight": residual_weight,
                    "source_weight": source_weight,
                    "baseline_weight": baseline_weight,
                    "tail_floor": tail_floor,
                    "tail_alpha": 0.65,
                    "exact_padding_f": 0.12,
                    "celsius_exact_padding_f": 0.05,
                    "spread_multiplier": 0.35,
                    "sigma_floor_f": 1.35,
                    "shrink_count": 120.0,
                },
            )
        )
    if max_configs > 0:
        by_family: dict[str, list[CandidateConfig]] = defaultdict(list)
        for config in configs:
            by_family[config.family].append(config)
        chosen: list[CandidateConfig] = []
        while len(chosen) < max_configs and any(by_family.values()):
            for family in sorted(by_family):
                if by_family[family] and len(chosen) < max_configs:
                    chosen.append(by_family[family].pop(0))
        configs = chosen
    return configs


def _run_nested_walk_forward(
    rows: list[ValidationRow],
    *,
    configs: list[CandidateConfig],
    observed_context: ObservedHighContextMap,
    embargo_days: list[int],
    min_train_rows: int,
    min_test_rows: int,
    min_inner_train_rows: int,
    min_inner_test_rows: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    split_rows: list[dict[str, Any]] = []
    plateau_rows: list[dict[str, Any]] = []
    leakage_rows: list[dict[str, Any]] = []
    aggregate = {
        "rows": [],
        "baseline_probabilities": {},
        "candidate_probabilities_by_family": defaultdict(dict),
    }
    target_months = sorted({row.market_date_local.strftime("%Y-%m") for row in rows})
    for embargo in embargo_days:
        for target_month in target_months:
            month_start = date.fromisoformat(f"{target_month}-01")
            month_end = _next_month(month_start)
            test_rows = [row for row in rows if month_start <= row.market_date_local < month_end]
            if len(test_rows) < min_test_rows:
                continue
            first_test_decision = min(row.decision_time_utc for row in test_rows).date()
            label_cutoff = first_test_decision - timedelta(days=embargo)
            train_rows = [row for row in rows if row.market_date_local < label_cutoff]
            if len(train_rows) < min_train_rows:
                continue
            inner = _inner_split(
                train_rows,
                embargo_days=embargo,
                min_inner_train_rows=min_inner_train_rows,
                min_inner_test_rows=min_inner_test_rows,
            )
            if inner is None:
                continue
            inner_month, inner_train, inner_valid = inner
            print(
                "[weather-improvement] "
                f"embargo={embargo} target_month={target_month} train={len(train_rows)} "
                f"test={len(test_rows)} inner={inner_month}",
                file=sys.stderr,
                flush=True,
            )
            baseline = {row.row_id: row.current_probability for row in test_rows}
            inner_baseline = {row.row_id: row.current_probability for row in inner_valid}
            leakage_rows.append(_leakage_row(target_month, embargo, train_rows, test_rows, label_cutoff, first_test_decision))

            scored: dict[str, dict[str, Any]] = {}
            fit_cache: dict[tuple[str, int], Any] = {}
            for config in configs:
                inner_probs = _predict_config(config, inner_train, inner_valid, fit_cache=fit_cache, observed_context=observed_context)
                outer_probs = _predict_config(config, train_rows, test_rows, fit_cache=fit_cache, observed_context=observed_context)
                inner_metrics = _comparison_metrics(inner_valid, inner_baseline, inner_probs)
                outer_metrics = _comparison_metrics(test_rows, baseline, outer_probs)
                row = {
                    "target_month": target_month,
                    "embargo_days": embargo,
                    "inner_target_month": inner_month,
                    "candidate_family": config.family,
                    "config_id": config.config_id,
                    "inner_baseline_brier": inner_metrics["baseline"]["brier"],
                    "inner_candidate_brier": inner_metrics["candidate"]["brier"],
                    "inner_brier_improvement": inner_metrics["brier_improvement"],
                    "inner_baseline_log_loss": inner_metrics["baseline"]["log_loss"],
                    "inner_candidate_log_loss": inner_metrics["candidate"]["log_loss"],
                    "inner_log_loss_improvement": inner_metrics["log_loss_improvement"],
                    "inner_score": _selection_score(inner_metrics),
                    "inner_beats_baseline_both": inner_metrics["brier_improvement"] > 0 and inner_metrics["log_loss_improvement"] > 0,
                    "outer_baseline_brier": outer_metrics["baseline"]["brier"],
                    "outer_candidate_brier": outer_metrics["candidate"]["brier"],
                    "outer_brier_improvement": outer_metrics["brier_improvement"],
                    "outer_baseline_log_loss": outer_metrics["baseline"]["log_loss"],
                    "outer_candidate_log_loss": outer_metrics["candidate"]["log_loss"],
                    "outer_log_loss_improvement": outer_metrics["log_loss_improvement"],
                    "outer_score": _selection_score(outer_metrics),
                    "outer_beats_baseline_both": outer_metrics["brier_improvement"] > 0 and outer_metrics["log_loss_improvement"] > 0,
                }
                if embargo == 0:
                    plateau_rows.append(row)
                scored[config.config_id] = {"config": config, "row": row, "outer_probs": outer_probs}

            if embargo == 0:
                aggregate["rows"].extend(test_rows)
                aggregate["baseline_probabilities"].update(baseline)

            for family in sorted({config.family for config in configs}):
                family_items = [item for item in scored.values() if item["config"].family == family]
                family_items.sort(key=lambda item: (float(item["row"]["inner_score"]), float(item["row"]["inner_candidate_brier"])))
                if not family_items:
                    continue
                selected = family_items[0]
                for rank, item in enumerate(family_items, start=1):
                    if embargo == 0:
                        item["row"]["inner_rank_in_family"] = rank
                        item["row"]["selected_by_inner_tuning"] = item is selected
                selected_row = dict(selected["row"])
                selected_row.update(
                    {
                        "train_n": len(train_rows),
                        "test_n": len(test_rows),
                        "inner_train_n": len(inner_train),
                        "inner_valid_n": len(inner_valid),
                        "selected_config_id": selected["config"].config_id,
                        "selected_by_inner_tuning": True,
                    }
                )
                split_rows.append(selected_row)
                if embargo == 0:
                    aggregate["candidate_probabilities_by_family"][family].update(selected["outer_probs"])
    return split_rows, plateau_rows, leakage_rows, aggregate


def _predict_config(
    config: CandidateConfig,
    train_rows: list[ValidationRow],
    score_rows: list[ValidationRow],
    *,
    fit_cache: Optional[dict[tuple[str, int], Any]] = None,
    observed_context: Optional[ObservedHighContextMap] = None,
) -> dict[str, float]:
    if config.kind == "shrinkage":
        return _predict_shrinkage(config, train_rows, score_rows, fit_cache=fit_cache)
    if config.kind == "source_bias_normal":
        return _predict_source_bias_normal(config, train_rows, score_rows, fit_cache=fit_cache)
    if config.kind == "observed_high_overlay":
        return _predict_observed_high_overlay(
            config,
            train_rows,
            score_rows,
            fit_cache=fit_cache,
            observed_context=observed_context or {},
        )
    if config.kind == "ensemble":
        residual_config = CandidateConfig(
            config_id=f"{config.config_id}|residual_component",
            family=config.family,
            kind="shrinkage",
            params=config.params,
        )
        source_config = CandidateConfig(
            config_id=f"{config.config_id}|source_component",
            family=config.family,
            kind="source_bias_normal",
            params=config.params,
        )
        residual = _predict_shrinkage(residual_config, train_rows, score_rows, fit_cache=fit_cache)
        source = _predict_source_bias_normal(source_config, train_rows, score_rows, fit_cache=fit_cache)
        rw = float(config.params.get("residual_weight", 0.5))
        sw = float(config.params.get("source_weight", 0.3))
        bw = float(config.params.get("baseline_weight", 0.2))
        total = max(1e-9, rw + sw + bw)
        output = {}
        for row in score_rows:
            probability = (rw * residual[row.row_id] + sw * source[row.row_id] + bw * row.current_probability) / total
            output[row.row_id] = _tail_smooth(_clip_probability(probability), row, config.params)
        return output
    raise ValueError(f"Unsupported candidate kind: {config.kind}")


def _predict_shrinkage(
    config: CandidateConfig,
    train_rows: list[ValidationRow],
    score_rows: list[ValidationRow],
    *,
    fit_cache: Optional[dict[tuple[str, int], Any]] = None,
) -> dict[str, float]:
    cache_key = ("empirical_residuals", id(train_rows))
    if fit_cache is not None and cache_key in fit_cache:
        dists = fit_cache[cache_key]
    else:
        dists = _fit_empirical_residual_distributions(train_rows)
        if fit_cache is not None:
            fit_cache[cache_key] = dists
    output = {}
    for row in score_rows:
        padding = _padding_for_row(row, config.params)
        lower = None if row.threshold_low_f is None else row.threshold_low_f - row.forecast_high_f - padding
        upper = None if row.threshold_high_f is None else row.threshold_high_f - row.forecast_high_f + padding
        components = _shrinkage_components(row, dists, config.params)
        if not components:
            probability = row.current_probability
        else:
            total_weight = sum(weight for _, weight in components)
            probability = sum(dist.interval_probability(lower, upper) * weight for dist, weight in components) / total_weight
        smoothed = _tail_smooth(_clip_probability(probability), row, config.params)
        output[row.row_id] = _apply_baseline_guards(smoothed, row, config.params, _evidence_context(row, dists))
    return output


def _predict_observed_high_overlay(
    config: CandidateConfig,
    train_rows: list[ValidationRow],
    score_rows: list[ValidationRow],
    *,
    fit_cache: Optional[dict[tuple[str, int], Any]] = None,
    observed_context: ObservedHighContextMap,
) -> dict[str, float]:
    base_config = CandidateConfig(
        config_id=f"{config.config_id}|guarded_base",
        family=config.family,
        kind="shrinkage",
        params=config.params,
    )
    base = _predict_shrinkage(base_config, train_rows, score_rows, fit_cache=fit_cache)
    return {
        row.row_id: _apply_observed_high_overlay(base[row.row_id], row, observed_context)
        for row in score_rows
    }


def _apply_observed_high_overlay(
    probability: float,
    row: ValidationRow,
    observed_context: ObservedHighContextMap,
) -> float:
    item = _observed_context_for_row(row, observed_context)
    if item is None or item.observed_high_f is None:
        return _clip_probability(probability)
    observed_high = float(item.observed_high_f)
    if item.is_final_at_decision:
        return _clip_probability(1.0 if _bucket_contains_value(observed_high, row.threshold_low_f, row.threshold_high_f) else 0.0)

    # The final daily high cannot be below the point-in-time high already observed.
    if row.threshold_high_f is not None and observed_high > float(row.threshold_high_f):
        return _clip_probability(0.0)
    if row.threshold_low_f is not None and row.threshold_high_f is None and observed_high >= float(row.threshold_low_f):
        return _clip_probability(1.0)
    return _clip_probability(probability)


def _observed_context_for_row(
    row: ValidationRow,
    observed_context: ObservedHighContextMap,
) -> Optional[ObservedHighContext]:
    item = observed_context.get((row.normalized_city, row.market_date_local, row.decision_time_utc))
    if item is None:
        return None
    if item.observed_at_utc is not None and item.observed_at_utc > row.decision_time_utc:
        return None
    return item


def _observed_high_group(row: ValidationRow, observed_context: ObservedHighContextMap) -> str:
    raw = observed_context.get((row.normalized_city, row.market_date_local, row.decision_time_utc))
    if raw is None or raw.observed_high_f is None:
        return "observed_high_missing"
    if raw.observed_at_utc is not None and raw.observed_at_utc > row.decision_time_utc:
        return "observed_high_future_violation"
    if raw.is_final_at_decision:
        return "observed_high_final_known_at_decision"
    return "observed_high_available"


def _bucket_contains_value(value: float, lower: Optional[float], upper: Optional[float]) -> bool:
    if lower is not None and value < float(lower):
        return False
    if upper is not None and value > float(upper):
        return False
    return True


def _fit_empirical_residual_distributions(train_rows: list[ValidationRow]) -> dict[str, dict[str, EmpiricalDistribution]]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    seen: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in train_rows:
        residual = row.realized_high_f - row.forecast_high_f
        for label, key in _residual_keys(row):
            seen_key = (label, key)
            if row.forecast_id in seen[seen_key]:
                continue
            seen[seen_key].add(row.forecast_id)
            grouped[label][key].append(residual)
    return {
        label: {key: EmpiricalDistribution(tuple(sorted(values))) for key, values in by_key.items() if values}
        for label, by_key in grouped.items()
    }


def _residual_keys(row: ValidationRow) -> list[tuple[str, str]]:
    region = _region(row)
    climate = _climate_key(row)
    source_family = f"{row.source_name}|{_dominant_model_family(row)}"
    return [
        ("city_lead", f"{row.normalized_city}|lead:{row.lead_days}"),
        ("city", row.normalized_city),
        ("region_lead", f"{region}|lead:{row.lead_days}"),
        ("region", region),
        ("climate_lead", f"{climate}|lead:{row.lead_days}"),
        ("climate_month", f"{climate}|month:{row.market_date_local.month:02d}"),
        ("source_family", source_family),
        ("lead", f"lead:{row.lead_days}"),
        ("global", "__global__"),
    ]


def _shrinkage_components(
    row: ValidationRow,
    dists: Mapping[str, Mapping[str, EmpiricalDistribution]],
    params: Mapping[str, float],
) -> list[tuple[EmpiricalDistribution, float]]:
    city_weight = float(params.get("city_weight", 0.18))
    specs = [
        ("city_lead", f"{row.normalized_city}|lead:{row.lead_days}", city_weight, 180.0, 35),
        ("city", row.normalized_city, city_weight * 0.8, 220.0, 35),
        ("region_lead", f"{_region(row)}|lead:{row.lead_days}", 0.22, 110.0, 25),
        ("region", _region(row), 0.18, 140.0, 25),
        ("climate_lead", f"{_climate_key(row)}|lead:{row.lead_days}", 0.16, 120.0, 20),
        ("climate_month", f"{_climate_key(row)}|month:{row.market_date_local.month:02d}", 0.12, 150.0, 20),
        ("source_family", f"{row.source_name}|{_dominant_model_family(row)}", 0.10, 160.0, 30),
        ("lead", f"lead:{row.lead_days}", 0.08, 180.0, 30),
        ("global", "__global__", 0.08, 1.0, 1),
    ]
    components: list[tuple[EmpiricalDistribution, float]] = []
    for label, key, base_weight, shrink_count, minimum_count in specs:
        dist = dists.get(label, {}).get(key)
        if dist is None or dist.n < minimum_count:
            continue
        evidence = dist.n / (dist.n + shrink_count)
        weight = base_weight * evidence
        if weight > 0:
            components.append((dist, weight))
    return components


def _evidence_context(
    row: ValidationRow,
    dists: Mapping[str, Mapping[str, EmpiricalDistribution]],
) -> dict[str, int]:
    return {
        "city_count": max(
            _distribution_count(dists, "city_lead", f"{row.normalized_city}|lead:{row.lead_days}"),
            _distribution_count(dists, "city", row.normalized_city),
        ),
        "region_count": max(
            _distribution_count(dists, "region_lead", f"{_region(row)}|lead:{row.lead_days}"),
            _distribution_count(dists, "region", _region(row)),
        ),
    }


def _distribution_count(
    dists: Mapping[str, Mapping[str, EmpiricalDistribution]],
    label: str,
    key: str,
) -> int:
    dist = dists.get(label, {}).get(key)
    return int(dist.n) if dist is not None else 0


def _apply_baseline_guards(
    probability: float,
    row: ValidationRow,
    params: Mapping[str, float],
    evidence: Mapping[str, int],
) -> float:
    baseline_weight = 0.0
    far_distance = float(params.get("far_tail_guard_distance_f", 0.0))
    if far_distance > 0 and row.threshold_distance_f is not None and row.threshold_distance_f >= far_distance:
        baseline_weight = max(baseline_weight, float(params.get("far_tail_baseline_weight", 0.0)))

    min_count = int(float(params.get("cold_start_min_count", 35.0)))
    if min_count > 0 and int(evidence.get("city_count") or 0) < min_count:
        baseline_weight = max(baseline_weight, float(params.get("city_cold_start_baseline_weight", 0.0)))
    if min_count > 0 and int(evidence.get("region_count") or 0) < min_count:
        baseline_weight = max(baseline_weight, float(params.get("region_cold_start_baseline_weight", 0.0)))

    if baseline_weight <= 0:
        return _clip_probability(probability)
    weight = max(0.0, min(1.0, baseline_weight))
    return _clip_probability(weight * row.current_probability + (1.0 - weight) * probability)


def _predict_source_bias_normal(
    config: CandidateConfig,
    train_rows: list[ValidationRow],
    score_rows: list[ValidationRow],
    *,
    fit_cache: Optional[dict[tuple[str, int], Any]] = None,
) -> dict[str, float]:
    stats_key = ("bias_stats", id(train_rows))
    spread_key = ("forecast_spread", id(score_rows))
    if fit_cache is not None and stats_key in fit_cache:
        stats = fit_cache[stats_key]
    else:
        stats = _fit_bias_stats(train_rows)
        if fit_cache is not None:
            fit_cache[stats_key] = stats
    if fit_cache is not None and spread_key in fit_cache:
        spread_by_key = fit_cache[spread_key]
    else:
        spread_by_key = _forecast_high_spread(score_rows)
        if fit_cache is not None:
            fit_cache[spread_key] = spread_by_key
    output = {}
    for row in score_rows:
        bias, sigma = _source_bias_and_sigma(row, stats, config.params)
        spread = spread_by_key.get(_decision_group_key(row), 0.0)
        sigma = max(float(config.params.get("sigma_floor_f", 1.35)), sigma + float(config.params.get("spread_multiplier", 0.35)) * spread)
        mean_high = row.forecast_high_f + bias
        padding = _padding_for_row(row, config.params)
        lower = None if row.threshold_low_f is None else row.threshold_low_f - padding
        upper = None if row.threshold_high_f is None else row.threshold_high_f + padding
        probability = _normal_interval_probability(mean_high, sigma, lower, upper)
        output[row.row_id] = _tail_smooth(_clip_probability(probability), row, config.params)
    return output


def _fit_bias_stats(rows: list[ValidationRow]) -> dict[str, dict[str, BiasStats]]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    seen: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        residual = row.realized_high_f - row.forecast_high_f
        for label, key in (
            ("source_lead", f"{row.source_name}|lead:{row.lead_days}"),
            ("source", row.source_name),
            ("model_family", _dominant_model_family(row)),
            ("region_source", f"{_region(row)}|{row.source_name}"),
            ("lead", f"lead:{row.lead_days}"),
            ("global", "__global__"),
        ):
            seen_key = (label, key)
            if row.forecast_id in seen[seen_key]:
                continue
            seen[seen_key].add(row.forecast_id)
            grouped[label][key].append(residual)
    return {
        label: {key: _bias_stats(values) for key, values in by_key.items() if values}
        for label, by_key in grouped.items()
    }


def _source_bias_and_sigma(
    row: ValidationRow,
    stats: Mapping[str, Mapping[str, BiasStats]],
    params: Mapping[str, float],
) -> tuple[float, float]:
    shrink = float(params.get("shrink_count", 120.0))
    specs = [
        ("source_lead", f"{row.source_name}|lead:{row.lead_days}", 0.35),
        ("source", row.source_name, 0.25),
        ("region_source", f"{_region(row)}|{row.source_name}", 0.20),
        ("model_family", _dominant_model_family(row), 0.10),
        ("lead", f"lead:{row.lead_days}", 0.06),
        ("global", "__global__", 0.04),
    ]
    components = []
    sigmas = []
    for label, key, base_weight in specs:
        item = stats.get(label, {}).get(key)
        if item is None:
            continue
        evidence = item.n / (item.n + shrink)
        weight = base_weight * evidence
        if weight <= 0:
            continue
        components.append((item.mean, weight))
        sigmas.append((item.sigma, weight))
    if not components:
        return 0.0, float(params.get("sigma_floor_f", 1.35))
    total = sum(weight for _, weight in components)
    bias = sum(value * weight for value, weight in components) / total
    sigma = sum(value * weight for value, weight in sigmas) / total if sigmas else float(params.get("sigma_floor_f", 1.35))
    return max(-5.0, min(5.0, bias)), max(0.5, sigma)


def _bias_stats(values: list[float]) -> BiasStats:
    sigma = pstdev(values) if len(values) > 1 else 1.5
    return BiasStats(n=len(values), mean=mean(values), sigma=max(0.5, sigma))


def _forecast_high_spread(rows: list[ValidationRow]) -> dict[tuple[str, date, datetime], float]:
    grouped: dict[tuple[str, date, datetime], dict[str, float]] = defaultdict(dict)
    for row in rows:
        grouped[_decision_group_key(row)][row.forecast_id] = row.forecast_high_f
    return {
        key: pstdev(list(values.values())) if len(values) > 1 else 0.0
        for key, values in grouped.items()
    }


def _decision_group_key(row: ValidationRow) -> tuple[str, date, datetime]:
    return (row.normalized_city, row.market_date_local, row.decision_time_utc)


def _normal_interval_probability(mean_high: float, sigma: float, lower: Optional[float], upper: Optional[float]) -> float:
    if sigma <= 0:
        sigma = 1.0
    if lower is None and upper is None:
        return 1.0
    if lower is None:
        return _normal_cdf((upper - mean_high) / sigma) if upper is not None else 1.0
    if upper is None:
        return 1.0 - _normal_cdf((lower - mean_high) / sigma)
    return max(0.0, _normal_cdf((upper - mean_high) / sigma) - _normal_cdf((lower - mean_high) / sigma))


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _padding_for_row(row: ValidationRow, params: Mapping[str, float]) -> float:
    width = _bucket_width(row)
    if row.bucket_type == "exact" or (width is not None and width <= 1.05):
        return float(params.get("exact_padding_f", 0.0))
    if width is not None and 1.6 <= width <= 2.0:
        return float(params.get("celsius_exact_padding_f", params.get("exact_padding_f", 0.0)))
    if row.bucket_shape == "bounded":
        return float(params.get("range_padding_f", 0.0))
    return 0.0


def _tail_smooth(probability: float, row: ValidationRow, params: Mapping[str, float]) -> float:
    floor = float(params.get("tail_floor", 0.0))
    alpha = float(params.get("tail_alpha", 1.0))
    if floor <= 0:
        return _clip_probability(probability)
    distance = row.threshold_distance_f
    if distance is None or distance < 4.0:
        return _clip_probability(probability)
    if probability < floor:
        return _clip_probability(floor + alpha * (probability - floor))
    if probability > 1.0 - floor:
        return _clip_probability(1.0 - floor + alpha * (probability - (1.0 - floor)))
    if distance >= 8.0 and probability < 2 * floor:
        return _clip_probability((probability + floor) / 2)
    if distance >= 8.0 and probability > 1.0 - 2 * floor:
        return _clip_probability((probability + (1.0 - floor)) / 2)
    return _clip_probability(probability)


def _bucket_width(row: ValidationRow) -> Optional[float]:
    if row.threshold_low_f is None or row.threshold_high_f is None:
        return None
    return row.threshold_high_f - row.threshold_low_f


def _dominant_model_family(row: ValidationRow) -> str:
    if not row.model_probabilities:
        return "unknown"
    return _model_family(next(iter(row.model_probabilities)))


def _climate_key(row: ValidationRow) -> str:
    lat_band = int(math.floor((row.latitude + 90.0) / 10.0) * 10 - 90)
    lon_band = int(math.floor((row.longitude + 180.0) / 30.0) * 30 - 180)
    return f"lat:{lat_band:+03d}|lon:{lon_band:+04d}"


def _candidate_comparison(
    rows: list[ValidationRow],
    baseline: Mapping[str, float],
    by_family: Mapping[str, Mapping[str, float]],
) -> list[dict[str, Any]]:
    output = []
    for family, probabilities in sorted(by_family.items()):
        metrics = _comparison_metrics(rows, baseline, probabilities)
        output.append(
            {
                "candidate_family": family,
                "n": metrics["candidate"]["n"],
                "baseline_brier": metrics["baseline"]["brier"],
                "candidate_brier": metrics["candidate"]["brier"],
                "brier_improvement": metrics["brier_improvement"],
                "baseline_log_loss": metrics["baseline"]["log_loss"],
                "candidate_log_loss": metrics["candidate"]["log_loss"],
                "log_loss_improvement": metrics["log_loss_improvement"],
                "candidate_bias": metrics["candidate"]["bias"],
                "candidate_accuracy_at_0_5": metrics["candidate"]["accuracy_at_0_5"],
                "beats_baseline_both": metrics["brier_improvement"] > 0 and metrics["log_loss_improvement"] > 0,
            }
        )
    output.sort(key=lambda row: (not bool(row["beats_baseline_both"]), -float(row["brier_improvement"]), -float(row["log_loss_improvement"])))
    return output


def _bucket_tail_diagnostics(
    rows: list[ValidationRow],
    baseline: Mapping[str, float],
    by_family: Mapping[str, Mapping[str, float]],
    *,
    observed_context: ObservedHighContextMap,
) -> list[dict[str, Any]]:
    dims: list[tuple[str, Callable[[ValidationRow], str]]] = [
        ("bucket_type", lambda row: row.bucket_type),
        ("bucket_shape", lambda row: row.bucket_shape),
        ("threshold_distance", lambda row: row.threshold_distance_bucket),
        ("bucket_type_x_threshold_distance", lambda row: f"{row.bucket_type}|{row.threshold_distance_bucket}"),
        ("source_name", lambda row: row.source_name),
        ("lead_days", lambda row: str(row.lead_days)),
        ("region", _region),
        ("observed_high_availability", lambda row: _observed_high_group(row, observed_context)),
    ]
    output = []
    for family, probabilities in sorted(by_family.items()):
        for dimension, key_fn in dims:
            grouped: dict[str, list[ValidationRow]] = defaultdict(list)
            for row in rows:
                grouped[str(key_fn(row))].append(row)
            for group, items in sorted(grouped.items()):
                if len(items) < 50:
                    continue
                metrics = _comparison_metrics(items, baseline, probabilities)
                output.append(
                    {
                        "candidate_family": family,
                        "dimension": dimension,
                        "group": group,
                        "n": len(items),
                        "baseline_brier": metrics["baseline"]["brier"],
                        "candidate_brier": metrics["candidate"]["brier"],
                        "brier_improvement": metrics["brier_improvement"],
                        "baseline_log_loss": metrics["baseline"]["log_loss"],
                        "candidate_log_loss": metrics["candidate"]["log_loss"],
                        "log_loss_improvement": metrics["log_loss_improvement"],
                        "beats_baseline_both": metrics["brier_improvement"] > 0 and metrics["log_loss_improvement"] > 0,
                    }
                )
    return output


def _holdout_diagnostics(
    rows: list[ValidationRow],
    *,
    configs: list[CandidateConfig],
    min_train_rows: int,
    observed_context: ObservedHighContextMap,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cutoff = date(2026, 4, 1)
    city_rows: list[dict[str, Any]] = []
    region_rows: list[dict[str, Any]] = []
    configs_by_family = {family: [config for config in configs if config.family == family] for family in sorted({config.family for config in configs})}
    for holdout_type, key_fn, output in (
        ("city", lambda row: row.normalized_city, city_rows),
        ("region", _region, region_rows),
    ):
        pre_cutoff_rows = [row for row in rows if row.market_date_local < cutoff]
        post_cutoff_by_group: dict[str, list[ValidationRow]] = defaultdict(list)
        pre_cutoff_by_group: dict[str, list[ValidationRow]] = defaultdict(list)
        for row in rows:
            group_key = str(key_fn(row))
            if row.market_date_local >= cutoff:
                post_cutoff_by_group[group_key].append(row)
            else:
                pre_cutoff_by_group[group_key].append(row)
        groups = sorted(set(post_cutoff_by_group) | set(pre_cutoff_by_group))
        for group in groups:
            test_rows = post_cutoff_by_group.get(group, [])
            train_rows = [row for row in pre_cutoff_rows if str(key_fn(row)) != group]
            if len(train_rows) < min_train_rows or len(test_rows) < 1000:
                continue
            print(
                f"[weather-improvement] holdout={holdout_type}:{group} train={len(train_rows)} test={len(test_rows)}",
                file=sys.stderr,
                flush=True,
            )
            inner = _inner_split(train_rows, embargo_days=0, min_inner_train_rows=max(1000, min_train_rows // 3), min_inner_test_rows=500)
            if inner is None:
                continue
            _, inner_train, inner_valid = inner
            baseline = {row.row_id: row.current_probability for row in test_rows}
            inner_baseline = {row.row_id: row.current_probability for row in inner_valid}
            fit_cache: dict[tuple[str, int], Any] = {}
            for family, family_configs in configs_by_family.items():
                selected = _select_config_for_family(
                    family_configs,
                    inner_train,
                    inner_valid,
                    inner_baseline,
                    fit_cache=fit_cache,
                    observed_context=observed_context,
                )
                probabilities = _predict_config(selected, train_rows, test_rows, fit_cache=fit_cache, observed_context=observed_context)
                metrics = _comparison_metrics(test_rows, baseline, probabilities)
                output.append(
                    {
                        "holdout_type": holdout_type,
                        "heldout_group": group,
                        "candidate_family": family,
                        "selected_config_id": selected.config_id,
                        "train_n": len(train_rows),
                        "test_n": len(test_rows),
                        "baseline_brier": metrics["baseline"]["brier"],
                        "candidate_brier": metrics["candidate"]["brier"],
                        "brier_improvement": metrics["brier_improvement"],
                        "baseline_log_loss": metrics["baseline"]["log_loss"],
                        "candidate_log_loss": metrics["candidate"]["log_loss"],
                        "log_loss_improvement": metrics["log_loss_improvement"],
                        "beats_baseline_both": metrics["brier_improvement"] > 0 and metrics["log_loss_improvement"] > 0,
                        "target_date_cutoff": cutoff.isoformat(),
                    }
                )
    return city_rows, region_rows


def _focused_holdout_slices(
    rows: list[ValidationRow],
    *,
    configs: list[CandidateConfig],
    min_train_rows: int,
    observed_context: ObservedHighContextMap,
) -> list[dict[str, Any]]:
    cutoff = date(2026, 4, 1)
    pre_cutoff_rows = [row for row in rows if row.market_date_local < cutoff]
    configs_by_family = {family: [config for config in configs if config.family == family] for family in sorted({config.family for config in configs})}
    output: list[dict[str, Any]] = []
    holdouts: list[tuple[str, str, Callable[[ValidationRow], str]]] = [
        ("region", "US", _region),
    ]
    holdouts.extend(("city", city, lambda row, city=city: row.normalized_city) for city in sorted(FOCUS_CITIES))
    for holdout_type, group, key_fn in holdouts:
        test_rows = [row for row in rows if row.market_date_local >= cutoff and str(key_fn(row)) == group]
        train_rows = [row for row in pre_cutoff_rows if str(key_fn(row)) != group]
        if len(train_rows) < min_train_rows or len(test_rows) < 1000:
            continue
        print(
            f"[weather-improvement] focused-holdout={holdout_type}:{group} train={len(train_rows)} test={len(test_rows)}",
            file=sys.stderr,
            flush=True,
        )
        inner = _inner_split(train_rows, embargo_days=0, min_inner_train_rows=max(1000, min_train_rows // 3), min_inner_test_rows=500)
        if inner is None:
            continue
        _, inner_train, inner_valid = inner
        baseline = {row.row_id: row.current_probability for row in test_rows}
        inner_baseline = {row.row_id: row.current_probability for row in inner_valid}
        fit_cache: dict[tuple[str, int], Any] = {}
        for family, family_configs in configs_by_family.items():
            selected = _select_config_for_family(
                family_configs,
                inner_train,
                inner_valid,
                inner_baseline,
                fit_cache=fit_cache,
                observed_context=observed_context,
            )
            probabilities = _predict_config(selected, train_rows, test_rows, fit_cache=fit_cache, observed_context=observed_context)
            for item in _slice_metrics(test_rows, baseline, probabilities, observed_context=observed_context):
                output.append(
                    {
                        "holdout_type": holdout_type,
                        "heldout_group": group,
                        "candidate_family": family,
                        "selected_config_id": selected.config_id,
                        "train_n": len(train_rows),
                        "test_n": len(test_rows),
                        "target_date_cutoff": cutoff.isoformat(),
                        **item,
                    }
                )
    return output


def _focused_primary_diagnostics(
    rows: list[ValidationRow],
    baseline: Mapping[str, float],
    by_family: Mapping[str, Mapping[str, float]],
    *,
    observed_context: ObservedHighContextMap,
) -> list[dict[str, Any]]:
    scopes = [
        ("US region", [row for row in rows if _region(row) == "US"]),
        ("8F+ threshold distance", [row for row in rows if row.threshold_distance_bucket == "8F+"]),
    ]
    for city in sorted(FOCUS_CITIES):
        scopes.append((f"city:{city}", [row for row in rows if row.normalized_city == city]))
    output: list[dict[str, Any]] = []
    for family, probabilities in sorted(by_family.items()):
        for scope, items in scopes:
            if len(items) < 50:
                continue
            for item in _slice_metrics(items, baseline, probabilities, observed_context=observed_context):
                output.append({"scope": scope, "candidate_family": family, **item})
    return output


def _slice_metrics(
    rows: list[ValidationRow],
    baseline: Mapping[str, float],
    probabilities: Mapping[str, float],
    *,
    observed_context: Optional[ObservedHighContextMap] = None,
) -> list[dict[str, Any]]:
    dims: list[tuple[str, Callable[[ValidationRow], str]]] = [
        ("overall", lambda row: "overall"),
        ("bucket_type", lambda row: row.bucket_type),
        ("bucket_shape", lambda row: row.bucket_shape),
        ("threshold_distance", lambda row: row.threshold_distance_bucket),
        ("bucket_type_x_threshold_distance", lambda row: f"{row.bucket_type}|{row.threshold_distance_bucket}"),
        ("lead_days", lambda row: str(row.lead_days)),
        ("decision_hour_utc", lambda row: str(row.decision_hour_utc)),
        ("local_decision_hour", lambda row: _local_hour_bucket(row.local_decision_hour)),
        ("source_name", lambda row: row.source_name),
        ("day_type", lambda row: row.day_type),
    ]
    if observed_context is not None:
        dims.append(("observed_high_availability", lambda row: _observed_high_group(row, observed_context)))
    output: list[dict[str, Any]] = []
    for dimension, key_fn in dims:
        grouped: dict[str, list[ValidationRow]] = defaultdict(list)
        for row in rows:
            grouped[str(key_fn(row))].append(row)
        for group, items in sorted(grouped.items()):
            if len(items) < 50:
                continue
            metrics = _comparison_metrics(items, baseline, probabilities)
            output.append(
                {
                    "dimension": dimension,
                    "group": group,
                    "n": len(items),
                    "baseline_brier": metrics["baseline"]["brier"],
                    "candidate_brier": metrics["candidate"]["brier"],
                    "brier_improvement": metrics["brier_improvement"],
                    "baseline_log_loss": metrics["baseline"]["log_loss"],
                    "candidate_log_loss": metrics["candidate"]["log_loss"],
                    "log_loss_improvement": metrics["log_loss_improvement"],
                    "baseline_bias": metrics["baseline"]["bias"],
                    "candidate_bias": metrics["candidate"]["bias"],
                    "beats_baseline_both": metrics["brier_improvement"] > 0 and metrics["log_loss_improvement"] > 0,
                }
            )
    return output


def _forecast_residual_diagnostics(rows: list[ValidationRow]) -> list[dict[str, Any]]:
    unique_forecasts = _unique_forecast_rows(rows)
    dims: list[tuple[str, Callable[[ValidationRow], Optional[str]], int]] = [
        ("region", _region, 25),
        ("city", lambda row: row.normalized_city, 25),
        ("focused_city", lambda row: row.normalized_city if row.normalized_city in FOCUS_CITIES else None, 25),
        ("source_name", lambda row: row.source_name, 25),
        ("source_model_name", lambda row: f"{row.source_name}|{row.model_name or 'unknown'}", 25),
        ("region_source", lambda row: f"{_region(row)}|{row.source_name}", 25),
        ("focused_city_source", lambda row: f"{row.normalized_city}|{row.source_name}" if row.normalized_city in FOCUS_CITIES else None, 20),
        ("lead_days", lambda row: str(row.lead_days), 25),
        ("decision_hour_utc", lambda row: str(row.decision_hour_utc), 25),
        ("local_decision_hour", lambda row: _local_hour_bucket(row.local_decision_hour), 25),
        ("target_month", lambda row: row.market_date_local.strftime("%Y-%m"), 25),
        ("coastal_proxy", _coastal_proxy, 25),
    ]
    output: list[dict[str, Any]] = []
    for dimension, key_fn, min_count in dims:
        grouped: dict[str, list[ValidationRow]] = defaultdict(list)
        for row in unique_forecasts:
            key = key_fn(row)
            if key is None:
                continue
            grouped[str(key)].append(row)
        for group, items in sorted(grouped.items()):
            if len(items) < min_count:
                continue
            errors = [row.forecast_error_f for row in items]
            abs_errors = [abs(value) for value in errors]
            output.append(
                {
                    "dimension": dimension,
                    "group": group,
                    "n_forecasts": len(items),
                    "mean_forecast_minus_realized_f": round(mean(errors), 6),
                    "mean_realized_minus_forecast_f": round(-mean(errors), 6),
                    "mae_f": round(mean(abs_errors), 6),
                    "rmse_f": round(math.sqrt(sum(value * value for value in errors) / len(errors)), 6),
                    "sigma_forecast_error_f": round(pstdev(errors), 6) if len(errors) > 1 else 0.0,
                    "forecast_overpredict_rate": round(sum(1 for value in errors if value > 0) / len(errors), 6),
                    "min_market_date_local": min(row.market_date_local for row in items).isoformat(),
                    "max_market_date_local": max(row.market_date_local for row in items).isoformat(),
                }
            )
    return output


def _unit_bucket_audit(
    rows: list[ValidationRow],
    baseline: Mapping[str, float],
    by_family: Mapping[str, Mapping[str, float]],
) -> list[dict[str, Any]]:
    dims: list[tuple[str, Callable[[ValidationRow], str]]] = [
        ("bucket_width", _bucket_width_bucket),
        ("bucket_type_x_width", lambda row: f"{row.bucket_type}|{_bucket_width_bucket(row)}"),
        ("bucket_shape_x_width", lambda row: f"{row.bucket_shape}|{_bucket_width_bucket(row)}"),
    ]
    output: list[dict[str, Any]] = []
    for family, probabilities in sorted(by_family.items()):
        for dimension, key_fn in dims:
            grouped: dict[str, list[ValidationRow]] = defaultdict(list)
            for row in rows:
                grouped[str(key_fn(row))].append(row)
            for group, items in sorted(grouped.items()):
                if len(items) < 50:
                    continue
                metrics = _comparison_metrics(items, baseline, probabilities)
                widths = [_bucket_width(row) for row in items]
                known_widths = [float(value) for value in widths if value is not None]
                output.append(
                    {
                        "candidate_family": family,
                        "dimension": dimension,
                        "group": group,
                        "n": len(items),
                        "mean_bucket_width_f": round(mean(known_widths), 6) if known_widths else None,
                        "baseline_brier": metrics["baseline"]["brier"],
                        "candidate_brier": metrics["candidate"]["brier"],
                        "brier_improvement": metrics["brier_improvement"],
                        "baseline_log_loss": metrics["baseline"]["log_loss"],
                        "candidate_log_loss": metrics["candidate"]["log_loss"],
                        "log_loss_improvement": metrics["log_loss_improvement"],
                        "beats_baseline_both": metrics["brier_improvement"] > 0 and metrics["log_loss_improvement"] > 0,
                    }
                )
    return output


def _alignment_diagnostics(rows: list[ValidationRow]) -> list[dict[str, Any]]:
    grouped: dict[str, list[ValidationRow]] = defaultdict(list)
    for row in rows:
        grouped[row.normalized_city].append(row)
    output: list[dict[str, Any]] = []
    for city, items in sorted(grouped.items()):
        unique = _unique_forecast_rows(items)
        lead_counts = Counter(str(row.lead_days) for row in unique)
        decision_hours = [row.local_decision_hour for row in unique]
        output.append(
            {
                "city": city,
                "region": _region(items[0]),
                "country_region": items[0].country_region,
                "timezone_name": items[0].timezone_name,
                "latitude": items[0].latitude,
                "longitude": items[0].longitude,
                "n_validation_rows": len(items),
                "n_unique_forecasts": len(unique),
                "min_market_date_local": min(row.market_date_local for row in unique).isoformat(),
                "max_market_date_local": max(row.market_date_local for row in unique).isoformat(),
                "lead_day_counts": json.dumps(dict(sorted(lead_counts.items())), sort_keys=True),
                "min_local_decision_hour": round(min(decision_hours), 3) if decision_hours else None,
                "max_local_decision_hour": round(max(decision_hours), 3) if decision_hours else None,
                "negative_or_gt3_lead_rows": sum(1 for row in unique if row.lead_days < 0 or row.lead_days > 3),
                "focus_failure_city": city in FOCUS_CITIES,
                "coastal_proxy": _coastal_proxy(items[0]),
            }
        )
    return output


def _feature_key_audit(rows: list[ValidationRow]) -> list[dict[str, Any]]:
    unique = _unique_forecast_rows(rows)
    counts: Counter[str] = Counter()
    example_types: dict[str, str] = {}
    for row in unique:
        for key, value in dict(row.feature_json or {}).items():
            counts[str(key)] += 1
            example_types.setdefault(str(key), type(value).__name__)
    expected = [
        "elevation",
        "elevation_m",
        "altitude",
        "coastal_distance_km",
        "distance_to_coast_km",
        "observed_high_so_far_f",
        "temperature_2m_max",
        "hourly_temperature_2m_max",
        "cloud_cover_mean",
        "precipitation_sum",
    ]
    all_keys = sorted(set(counts) | set(expected))
    total = len(unique)
    return [
        {
            "feature_key": key,
            "n_unique_forecasts_with_key": counts.get(key, 0),
            "coverage_fraction": round(counts.get(key, 0) / total, 6) if total else None,
            "example_python_type": example_types.get(key),
            "expected_metadata_or_live_adjustment_key": key in expected,
        }
        for key in all_keys
    ]


def _old_model_family_comparison(rows: list[ValidationRow]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[tuple[float, int]]] = defaultdict(list)
    for row in rows:
        for key, probability in row.model_probabilities.items():
            family = _model_family(key)
            source = key.rsplit(".", 1)[0] if "." in key else row.source_name
            grouped[("model_family", family)].append((_clip_probability(probability), row.outcome))
            grouped[("source_x_model_family", f"{source}|{family}")].append((_clip_probability(probability), row.outcome))
    output = []
    for (dimension, group), pairs in sorted(grouped.items()):
        if len(pairs) < 50:
            continue
        metrics = _pair_metrics(pairs)
        output.append({"dimension": dimension, "group": group, **metrics})
    output.sort(key=lambda row: (str(row["dimension"]), float(row["brier"])))
    return output


def _unique_forecast_rows(rows: list[ValidationRow]) -> list[ValidationRow]:
    seen: set[str] = set()
    unique: list[ValidationRow] = []
    for row in rows:
        if row.forecast_id in seen:
            continue
        seen.add(row.forecast_id)
        unique.append(row)
    return unique


def _bucket_width_bucket(row: ValidationRow) -> str:
    width = _bucket_width(row)
    if width is None:
        return "tail_or_open"
    if width <= 1.05:
        return "about_1F_exact"
    if 1.6 <= width <= 2.0:
        return "about_1C_width"
    if width < 4:
        return "other_lt4F"
    return "wide_range"


def _coastal_proxy(row: ValidationRow) -> str:
    city = row.normalized_city
    if city in {"Los Angeles, CA", "San Francisco, CA", "Seattle, WA", "Miami, FL", "Houston, TX", "New York, NY", "London, GB", "Tokyo, JP", "Seoul, KR", "Singapore, SG", "Taipei, TW", "Shanghai, CN", "Shenzhen, CN", "Guangzhou, CN"}:
        return "coastal_or_marine_influenced_city_name_proxy"
    return "inland_or_unknown_city_name_proxy"


def _local_hour_bucket(local_hour: float) -> str:
    if local_hour < 6:
        return "00-06"
    if local_hour < 9:
        return "06-09"
    if local_hour < 12:
        return "09-12"
    if local_hour < 15:
        return "12-15"
    if local_hour < 18:
        return "15-18"
    return "18-24"


def _pair_metrics(pairs: list[tuple[float, int]]) -> dict[str, Any]:
    probs = [_clip_probability(probability) for probability, _ in pairs]
    actual = [int(outcome) for _, outcome in pairs]
    n = len(pairs)
    actual_rate = sum(actual) / n
    avg_probability = sum(probs) / n
    return {
        "n": n,
        "avg_probability": round(avg_probability, 6),
        "actual_rate": round(actual_rate, 6),
        "bias": round(avg_probability - actual_rate, 6),
        "brier": round(sum((probability - outcome) ** 2 for probability, outcome in zip(probs, actual)) / n, 8),
        "log_loss": round(
            -sum(outcome * math.log(probability) + (1 - outcome) * math.log(1 - probability) for probability, outcome in zip(probs, actual)) / n,
            8,
        ),
        "accuracy_at_0_5": round(sum((probability >= 0.5) == bool(outcome) for probability, outcome in zip(probs, actual)) / n, 6),
    }


def _focused_diagnosis_summary(
    *,
    focused_holdout_slices: list[Mapping[str, Any]],
    forecast_residual_diagnostics: list[Mapping[str, Any]],
    unit_bucket_audit: list[Mapping[str, Any]],
    feature_key_audit: list[Mapping[str, Any]],
) -> dict[str, Any]:
    overall_holdouts = [
        row for row in focused_holdout_slices
        if row.get("dimension") == "overall" and row.get("group") == "overall"
    ]
    failing_overall = [
        row for row in overall_holdouts
        if not bool(row.get("beats_baseline_both"))
    ]
    metadata_keys = {
        str(row.get("feature_key")): row
        for row in feature_key_audit
        if bool(row.get("expected_metadata_or_live_adjustment_key"))
    }
    focus_residuals = [
        row for row in forecast_residual_diagnostics
        if row.get("dimension") in {"focused_city", "focused_city_source"}
    ]
    worst_focus_residuals = sorted(
        focus_residuals,
        key=lambda row: abs(float(row.get("mean_forecast_minus_realized_f") or 0.0)),
        reverse=True,
    )[:10]
    unit_failures = [
        row for row in unit_bucket_audit
        if str(row.get("dimension")) in {"bucket_width", "bucket_type_x_width"} and not bool(row.get("beats_baseline_both"))
    ][:20]
    return {
        "focused_holdout_overall_fail_count": len(failing_overall),
        "focused_holdout_overall_count": len(overall_holdouts),
        "largest_abs_focus_source_bias_rows": worst_focus_residuals,
        "unit_width_groups_failing_top_sample": unit_failures,
        "elevation_or_coastal_metadata_available": any(
            int(metadata_keys.get(key, {}).get("n_unique_forecasts_with_key") or 0) > 0
            for key in ("elevation", "elevation_m", "altitude", "coastal_distance_km", "distance_to_coast_km")
        ),
        "observed_high_so_far_in_forecast_features": int(metadata_keys.get("observed_high_so_far_f", {}).get("n_unique_forecasts_with_key") or 0) > 0,
    }


def _select_config_for_family(
    configs: list[CandidateConfig],
    inner_train: list[ValidationRow],
    inner_valid: list[ValidationRow],
    inner_baseline: Mapping[str, float],
    *,
    fit_cache: Optional[dict[tuple[str, int], Any]] = None,
    observed_context: Optional[ObservedHighContextMap] = None,
) -> CandidateConfig:
    best: Optional[tuple[float, float, CandidateConfig]] = None
    for config in configs:
        probabilities = _predict_config(config, inner_train, inner_valid, fit_cache=fit_cache, observed_context=observed_context)
        metrics = _comparison_metrics(inner_valid, inner_baseline, probabilities)
        score = _selection_score(metrics)
        candidate = (score, float(metrics["candidate"]["brier"]), config)
        if best is None or candidate < best:
            best = candidate
    assert best is not None
    return best[2]


def _market_comparison(
    decision_db: Path,
    rows: list[ValidationRow],
    baseline: Mapping[str, float],
    by_family: Mapping[str, Mapping[str, float]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    table_rows = []
    summary = {}
    for family, probabilities in sorted(by_family.items()):
        diagnostics = _walk_forward_market_diagnostics(decision_db, rows, baseline, probabilities)
        summary[family] = diagnostics["summary"]
        table_rows.extend({"candidate_family": family, **row} for row in diagnostics["tables"])
    return table_rows, summary


def _label_poisoning_invariance(
    rows: list[ValidationRow],
    *,
    configs: list[CandidateConfig],
    min_train_rows: int,
    min_test_rows: int,
    observed_context: ObservedHighContextMap,
) -> list[dict[str, Any]]:
    output = []
    for month in sorted({row.market_date_local.strftime("%Y-%m") for row in rows}):
        month_start = date.fromisoformat(f"{month}-01")
        month_end = _next_month(month_start)
        test_rows = [row for row in rows if month_start <= row.market_date_local < month_end]
        if len(test_rows) < min_test_rows:
            continue
        first_test_decision = min(row.decision_time_utc for row in test_rows).date()
        train_rows = [row for row in rows if row.market_date_local < first_test_decision]
        if len(train_rows) < min_train_rows:
            continue
        inner = _inner_split(train_rows, embargo_days=0, min_inner_train_rows=max(1000, min_train_rows // 3), min_inner_test_rows=500)
        if inner is None:
            continue
        _, inner_train, inner_valid = inner
        inner_baseline = {row.row_id: row.current_probability for row in inner_valid}
        poisoned_rows = [_poison_evaluation_fields(row) for row in test_rows]
        fit_cache: dict[tuple[str, int], Any] = {}
        for family in sorted({config.family for config in configs}):
            family_configs = [config for config in configs if config.family == family]
            selected = _select_config_for_family(
                family_configs,
                inner_train,
                inner_valid,
                inner_baseline,
                fit_cache=fit_cache,
                observed_context=observed_context,
            )
            original = _predict_config(selected, train_rows, test_rows, fit_cache=fit_cache, observed_context=observed_context)
            poisoned = _predict_config(selected, train_rows, poisoned_rows, fit_cache=fit_cache, observed_context=observed_context)
            deltas = [abs(original[row.row_id] - poisoned[row.row_id]) for row in test_rows]
            changed = sum(1 for value in deltas if value > 1e-12)
            output.append(
                {
                    "target_month": month,
                    "candidate_family": family,
                    "selected_config_id": selected.config_id,
                    "train_n": len(train_rows),
                    "test_n": len(test_rows),
                    "changed_probability_rows": changed,
                    "max_abs_probability_delta": round(max(deltas) if deltas else 0.0, 14),
                    "mean_abs_probability_delta": round(sum(deltas) / len(deltas), 14) if deltas else 0.0,
                    "live_inference_invariant": changed == 0,
                }
            )
    return output


def _poison_evaluation_fields(row: ValidationRow) -> ValidationRow:
    return replace(
        row,
        realized_high_f=9999.0,
        forecast_error_f=-9999.0,
        outcome=0 if int(row.outcome) else 1,
        resolved_outcome=None if row.resolved_outcome is None else (0 if int(row.resolved_outcome) else 1),
        resolved_outcome_mismatch=not row.resolved_outcome_mismatch,
        realized_quality_flags=("poisoned_for_weather_model_improvement_audit",),
    )


def _decision(
    *,
    candidate_comparison: list[Mapping[str, Any]],
    plateau_summary: Mapping[str, Mapping[str, Any]],
    city_holdouts: list[Mapping[str, Any]],
    region_holdouts: list[Mapping[str, Any]],
    live_feature_gate: Mapping[str, Any],
    label_poisoning: list[Mapping[str, Any]],
    market_summary: Mapping[str, Any],
    top_family: str,
) -> dict[str, Any]:
    family_row = next((row for row in candidate_comparison if row["candidate_family"] == top_family), None)
    weather_pass = bool(family_row and family_row.get("beats_baseline_both"))
    plateau = plateau_summary.get(top_family) or {}
    neighbor_pass_rate = plateau.get("outer_neighbor_pass_rate_median")
    cliff_rate = plateau.get("outer_performance_cliff_rate_median")
    plateau_pass = (
        neighbor_pass_rate is not None
        and cliff_rate is not None
        and float(neighbor_pass_rate) >= 0.60
        and float(cliff_rate) <= 0.40
    )
    city_pass_rate = _family_pass_rate(city_holdouts, top_family)
    region_pass_rate = _family_pass_rate(region_holdouts, top_family)
    holdout_pass = city_pass_rate >= 0.70 and region_pass_rate >= 0.70
    live_pass = (
        int(live_feature_gate.get("live_timing_violation_rows") or 0) == 0
        and float(live_feature_gate.get("weather_model_scoreable_fraction") or 0.0) >= 0.95
        and all(bool(row.get("live_inference_invariant")) for row in label_poisoning if row.get("candidate_family") == top_family)
    )
    market = (market_summary.get(top_family) or {})
    cand_market = market.get("candidate_vs_polymarket_payout") or {}
    mid_market = market.get("market_mid_vs_polymarket_payout") or {}
    market_pass = (
        cand_market.get("brier") is not None
        and mid_market.get("brier") is not None
        and float(cand_market["brier"]) < float(mid_market["brier"])
        and float(cand_market["log_loss"]) < float(mid_market["log_loss"])
    )
    if not weather_pass:
        outcome = "no_candidate_beats_weather_baseline"
    elif not plateau_pass:
        outcome = "research_only_fragile_parameter_plateau"
    elif not holdout_pass:
        outcome = "research_only_fails_city_region_holdouts"
    elif not live_pass:
        outcome = "reject_not_live_feature_safe"
    elif not market_pass:
        outcome = "weather_improved_but_not_market_beating"
    else:
        outcome = "candidate_survives_weather_gate_ready_for_quote_backed_replay"
    return {
        "outcome": outcome,
        "top_family": top_family,
        "weather_pass": weather_pass,
        "plateau_pass": plateau_pass,
        "holdout_pass": holdout_pass,
        "city_holdout_pass_rate": city_pass_rate,
        "region_holdout_pass_rate": region_pass_rate,
        "live_feature_pass": live_pass,
        "market_pass": market_pass,
        "model_weights_or_strategy_defaults_changed": False,
        "strategy_pnl_backtest_run": False,
    }


def _plateau_summary(rows: list[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[tuple[str, int], list[Mapping[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[str(row["candidate_family"])][(str(row["target_month"]), int(row["embargo_days"]))].append(row)
    output: dict[str, dict[str, Any]] = {}
    for family, by_split in grouped.items():
        split_summaries = []
        for items in by_split.values():
            selected_items = [row for row in items if bool(row.get("selected_by_inner_tuning"))]
            selected = selected_items[0] if selected_items else min(items, key=lambda row: float(row["inner_score"]))
            selected_outer = float(selected["outer_brier_improvement"])
            outer_improvements = [float(row["outer_brier_improvement"]) for row in items]
            pass_rate = sum(1 for row in items if bool(row["outer_beats_baseline_both"])) / len(items)
            cliff_rate = sum(
                1
                for row in items
                if selected_outer - float(row["outer_brier_improvement"]) > 0.005 or not bool(row["outer_beats_baseline_both"])
            ) / len(items)
            split_summaries.append(
                {
                    "candidate_family": family,
                    "outer_neighbor_pass_rate": pass_rate,
                    "outer_performance_cliff_rate": cliff_rate,
                    "outer_top_vs_median_brier_gap": max(outer_improvements) - median(outer_improvements),
                    "outer_worst_neighbor_degradation": selected_outer - min(outer_improvements),
                }
            )
        output[family] = {
            "candidate_family": family,
            "n": sum(len(items) for items in by_split.values()),
            "split_count": len(split_summaries),
            "outer_neighbor_pass_rate_median": round(median(row["outer_neighbor_pass_rate"] for row in split_summaries), 6),
            "outer_performance_cliff_rate_median": round(median(row["outer_performance_cliff_rate"] for row in split_summaries), 6),
            "outer_top_vs_median_brier_gap_median": round(median(row["outer_top_vs_median_brier_gap"] for row in split_summaries), 8),
            "outer_worst_neighbor_degradation_median": round(median(row["outer_worst_neighbor_degradation"] for row in split_summaries), 8),
        }
    return output


def _holdout_summary(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    output = {}
    for family in sorted({str(row["candidate_family"]) for row in rows}):
        family_rows = [row for row in rows if str(row["candidate_family"]) == family]
        output[family] = {
            "n": len(family_rows),
            "pass_rate": _family_pass_rate(rows, family),
            "worst_brier_improvement": min((float(row["brier_improvement"]) for row in family_rows), default=None),
            "worst_log_loss_improvement": min((float(row["log_loss_improvement"]) for row in family_rows), default=None),
        }
    return output


def _family_pass_rate(rows: list[Mapping[str, Any]], family: str) -> float:
    family_rows = [row for row in rows if str(row.get("candidate_family")) == family]
    if not family_rows:
        return 0.0
    return sum(1 for row in family_rows if bool(row.get("beats_baseline_both"))) / len(family_rows)


def _label_poisoning_summary(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    output = {}
    for family in sorted({str(row["candidate_family"]) for row in rows}):
        family_rows = [row for row in rows if str(row["candidate_family"]) == family]
        output[family] = {
            "n": len(family_rows),
            "pass_rate": sum(1 for row in family_rows if bool(row["live_inference_invariant"])) / len(family_rows) if family_rows else None,
            "total_changed_probability_rows": sum(int(row["changed_probability_rows"]) for row in family_rows),
            "max_abs_probability_delta": max((float(row["max_abs_probability_delta"]) for row in family_rows), default=0.0),
        }
    return output


def _top_family(candidate_comparison: list[Mapping[str, Any]]) -> str:
    if not candidate_comparison:
        return ""
    return str(candidate_comparison[0]["candidate_family"])


def _selection_score(metrics: Mapping[str, Any]) -> float:
    return float(metrics["candidate"]["brier"]) + float(metrics["candidate"]["log_loss"])


def _inner_split(
    train_rows: list[ValidationRow],
    *,
    embargo_days: int,
    min_inner_train_rows: int,
    min_inner_test_rows: int,
) -> Optional[tuple[str, list[ValidationRow], list[ValidationRow]]]:
    months = sorted({row.market_date_local.strftime("%Y-%m") for row in train_rows}, reverse=True)
    for month in months:
        month_start = date.fromisoformat(f"{month}-01")
        month_end = _next_month(month_start)
        valid_rows = [row for row in train_rows if month_start <= row.market_date_local < month_end]
        if len(valid_rows) < min_inner_test_rows:
            continue
        first_valid_decision = min(row.decision_time_utc for row in valid_rows).date()
        label_cutoff = first_valid_decision - timedelta(days=embargo_days)
        inner_train = [row for row in train_rows if row.market_date_local < label_cutoff]
        if len(inner_train) >= min_inner_train_rows:
            return month, inner_train, valid_rows
    return None


def _leakage_row(
    target_month: str,
    embargo_days: int,
    train_rows: list[ValidationRow],
    test_rows: list[ValidationRow],
    label_cutoff: date,
    first_test_decision: date,
) -> dict[str, Any]:
    return {
        "target_month": target_month,
        "embargo_days": embargo_days,
        "train_n": len(train_rows),
        "test_n": len(test_rows),
        "first_test_decision_date": first_test_decision.isoformat(),
        "label_cutoff_local_date_exclusive": label_cutoff.isoformat(),
        "max_train_target_date": max(row.market_date_local for row in train_rows).isoformat() if train_rows else None,
        "min_test_target_date": min(row.market_date_local for row in test_rows).isoformat() if test_rows else None,
        "train_rows_with_unavailable_labels": sum(1 for row in train_rows if row.market_date_local >= label_cutoff),
        "train_test_city_date_overlap": _overlap_count(train_rows, test_rows, lambda row: f"{row.normalized_city}|{row.market_date_local.isoformat()}"),
        "train_test_universe_overlap": _overlap_count(train_rows, test_rows, lambda row: row.universe_row_id),
    }


def _overlap_count(
    train_rows: list[ValidationRow],
    test_rows: list[ValidationRow],
    key_fn: Callable[[ValidationRow], str],
) -> int:
    train_keys = {str(key_fn(row)) for row in train_rows}
    test_keys = {str(key_fn(row)) for row in test_rows}
    return len(train_keys & test_keys)


def _flatten_bootstrap(bootstrap: Mapping[str, Any], *, top_family: str) -> list[dict[str, Any]]:
    return [{"candidate_family": top_family, "cluster": key, **dict(value)} for key, value in bootstrap.items()]


def _write_markdown(path: Path, summary: Mapping[str, Any]) -> None:
    decision = summary["decision"]
    comparison = summary["candidate_comparison"]
    lines = [
        "# DailyWeather Weather Model Improvement Audit",
        "",
        f"Generated: {summary['generated_at_utc']}",
        "",
        f"Outcome: `{decision['outcome']}`",
        "",
        "- No model weights, paper defaults, live defaults, or strategy settings were changed.",
        "- No strategy PnL backtest was run.",
        "- Core candidates use weather forecasts, metadata, prior train-window weather residuals, and point-in-time observed-high snapshots only.",
        "- Resolved Polymarket labels are used only in quote-covered market diagnostics.",
        "",
        "## Candidate Comparison",
        "",
    ]
    for row in comparison:
        lines.append(
            f"- `{row['candidate_family']}`: Brier/log loss {row['candidate_brier']} / {row['candidate_log_loss']} "
            f"(improvement {row['brier_improvement']} / {row['log_loss_improvement']})"
        )
    lines.extend(
        [
            "",
            "## Gate Summary",
            "",
            f"- Top family: `{summary['top_family']}`",
            f"- Weather pass: {decision['weather_pass']}",
            f"- Plateau pass: {decision['plateau_pass']}",
            f"- Holdout pass: {decision['holdout_pass']} "
            f"(city {decision['city_holdout_pass_rate']}, region {decision['region_holdout_pass_rate']})",
            f"- Live-feature pass: {decision['live_feature_pass']}",
            f"- Market pass: {decision['market_pass']}",
            "",
            "## Observed-High Context",
            "",
            f"- Primary rows with point-in-time observed high: {(summary.get('observed_high_context_summary') or {}).get('rows_with_point_in_time_observed_high')}",
            f"- Future observed-high violations: {(summary.get('observed_high_context_summary') or {}).get('future_observed_high_violations')}",
            f"- Rows with final high known at decision: {(summary.get('observed_high_context_summary') or {}).get('rows_with_final_high_known_at_decision')}",
            "",
            "## Interpretation",
            "",
            "- These candidates target city/region generalization and smoother tails rather than tuning the rejected residual model.",
            "- A candidate is not promotable unless it passes aggregate weather metrics, city/region holdouts, plateau diagnostics, and live-feature checks.",
            "- Market midpoint comparison remains diagnostic only; no strategy replay or PnL backtest was run.",
            "",
            "## Artifacts",
            "",
        ]
    )
    for key, value in summary["output_artifacts"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_diagnosis_markdown(path: Path, summary: Mapping[str, Any]) -> None:
    decision = summary["decision"]
    comparison = summary["candidate_comparison"]
    guarded = next((row for row in comparison if row.get("candidate_family") == "baseline_guarded_shrinkage"), None)
    observed_guarded = next((row for row in comparison if row.get("candidate_family") == "observed_high_guarded_shrinkage"), None)
    focused = summary.get("focused_diagnosis_summary") or {}
    observed_summary = summary.get("observed_high_context_summary") or {}
    region_summary = summary.get("region_holdout_summary") or {}
    city_summary = summary.get("city_holdout_summary") or {}
    top_family = str(summary.get("top_family") or "")
    lines = [
        "# DailyWeather Weather Model Failure Diagnosis",
        "",
        f"Generated: {summary['generated_at_utc']}",
        "",
        f"Verdict: `{decision['outcome']}`",
        "",
        "## Scope",
        "",
        "- This report diagnoses the US region failure, focused city failures, 8F+ far-tail behavior, source residuals, unit-width parsing, date/time alignment, and metadata coverage.",
        "- No model weights, paper defaults, live defaults, strategy settings, or trading/PnL backtests were changed or run.",
        "- Candidate inputs remain weather forecasts, city/date metadata, and train-window weather residuals available before each validation window.",
        "",
        "## Candidate Change Tested",
        "",
    ]
    if guarded:
        lines.append(
            "- Added `baseline_guarded_shrinkage`: a research-only shrinkage family that blends back toward the existing baseline when the train window lacks city/region residual evidence or the threshold is already at least 8F from the forecast high."
        )
        lines.append(
            f"- Guarded family Brier/log loss: {guarded['candidate_brier']} / {guarded['candidate_log_loss']} "
            f"(improvement {guarded['brier_improvement']} / {guarded['log_loss_improvement']})."
        )
    else:
        lines.append("- No guarded candidate row was produced.")
    if observed_guarded:
        lines.append(
            "- Added `observed_high_guarded_shrinkage`: the guarded shrinkage family plus a live-known observed-high overlay. The overlay only uses observed-high snapshots whose observation timestamp is not after the simulated decision time; it makes bounded/range events impossible when the high-so-far already exceeds the upper threshold and upper-tail events certain when the high-so-far already reaches the lower threshold."
        )
        lines.append(
            f"- Observed-high guarded family Brier/log loss: {observed_guarded['candidate_brier']} / {observed_guarded['candidate_log_loss']} "
            f"(improvement {observed_guarded['brier_improvement']} / {observed_guarded['log_loss_improvement']})."
        )
    lines.extend(
        [
            "",
            "## Main Diagnosis",
            "",
            f"- Top aggregate family: `{top_family}`.",
            f"- City holdout pass rate for top family: {decision['city_holdout_pass_rate']}.",
            f"- Region holdout pass rate for top family: {decision['region_holdout_pass_rate']}.",
            f"- Focused holdout overall failures reported: {focused.get('focused_holdout_overall_fail_count')} of {focused.get('focused_holdout_overall_count')}.",
            f"- Elevation/coastal metadata available in forecast feature JSON: {focused.get('elevation_or_coastal_metadata_available')}.",
            f"- Observed-high-so-far present in forecast features: {focused.get('observed_high_so_far_in_forecast_features')}.",
            f"- Point-in-time observed-high rows in primary test set: {observed_summary.get('rows_with_point_in_time_observed_high')} of {observed_summary.get('primary_rows')}.",
            f"- Future observed-high violations in primary test set: {observed_summary.get('future_observed_high_violations')}.",
            "",
            "## Region Holdout Summary",
            "",
        ]
    )
    for family, item in sorted(region_summary.items()):
        lines.append(
            f"- `{family}`: pass rate {item.get('pass_rate')}, worst Brier improvement {item.get('worst_brier_improvement')}, "
            f"worst log-loss improvement {item.get('worst_log_loss_improvement')}."
        )
    lines.extend(["", "## City Holdout Summary", ""])
    for family, item in sorted(city_summary.items()):
        lines.append(
            f"- `{family}`: pass rate {item.get('pass_rate')}, worst Brier improvement {item.get('worst_brier_improvement')}, "
            f"worst log-loss improvement {item.get('worst_log_loss_improvement')}."
        )
    lines.extend(["", "## Largest Focused Source Biases", ""])
    for row in (focused.get("largest_abs_focus_source_bias_rows") or [])[:8]:
        lines.append(
            f"- `{row.get('dimension')}={row.get('group')}`: n={row.get('n_forecasts')}, "
            f"forecast-realized mean {row.get('mean_forecast_minus_realized_f')}F, MAE {row.get('mae_f')}F."
        )
    lines.extend(["", "## Unit And Tail Notes", ""])
    unit_rows = focused.get("unit_width_groups_failing_top_sample") or []
    if unit_rows:
        for row in unit_rows[:8]:
            lines.append(
                f"- `{row.get('candidate_family')}` `{row.get('dimension')}={row.get('group')}`: "
                f"Brier improvement {row.get('brier_improvement')}, log-loss improvement {row.get('log_loss_improvement')}."
            )
    else:
        lines.append("- No unit-width groups failed the baseline in the sampled diagnostic rows.")
    lines.extend(
        [
            "",
            "## Artifact Pointers",
            "",
        ]
    )
    for key in (
        "focused_primary_diagnostics_csv",
        "focused_holdout_slices_csv",
        "forecast_residual_diagnostics_csv",
        "unit_bucket_audit_csv",
        "alignment_diagnostics_csv",
        "feature_key_audit_csv",
        "old_model_family_comparison_csv",
        "report_md",
        "summary_json",
    ):
        value = (summary.get("output_artifacts") or {}).get(key)
        if value:
            lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Promotion Verdict",
            "",
            "- Not promotable. The report may identify safer research directions, but the acceptance gate still requires aggregate weather improvement, city and region holdout survival, plateau stability, live-feature safety, permutation sanity, and market-midpoint comparison before any quote-backed replay or promotion.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _parse_int_list(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def _next_month(value: date) -> date:
    if value.month == 12:
        return date(value.year + 1, 1, 1)
    return date(value.year, value.month + 1, 1)


if __name__ == "__main__":
    sys.exit(main())
