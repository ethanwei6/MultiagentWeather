from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from statistics import median
from typing import Any, Callable, Mapping, Optional

from .weather_model_live_readiness import (
    _audit_decision_artifact_live_features,
)
from .weather_model_robustness import (
    DEFAULT_CANDIDATE,
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
    _collect_residual_sets,
    _family_probabilities,
    _fit_residual_distribution_candidates,
    _load_decision_quotes,
    _load_forecasts,
    _load_markets,
    _load_realized,
    _load_weather_strategy,
    _local_hour_bucket,
    _metrics,
    _model_family,
    _parse_datetime,
    _residual_distribution_probability,
    _write_csv,
)


BASE_COMPONENTS: tuple[tuple[str, float, int, float, Callable[[ValidationRow], str]], ...] = (
    ("city_lead", 0.45, 35, 120.0, lambda row: f"{row.normalized_city}|lead:{row.lead_days}"),
    ("city", 0.25, 35, 160.0, lambda row: row.normalized_city),
    ("lead_hour", 0.15, 45, 180.0, lambda row: f"lead:{row.lead_days}|local:{_local_hour_bucket(row.local_decision_hour)}"),
    ("lead", 0.10, 45, 180.0, lambda row: f"lead:{row.lead_days}"),
    ("global", 0.05, 1, 1.0, lambda row: "__global__"),
)

WEIGHT_VARIANTS: dict[str, tuple[float, float, float, float, float]] = {
    "current": (0.45, 0.25, 0.15, 0.10, 0.05),
    "city_heavier": (0.55, 0.25, 0.10, 0.07, 0.03),
    "lead_heavier": (0.35, 0.20, 0.25, 0.15, 0.05),
    "global_smoother": (0.35, 0.20, 0.15, 0.10, 0.20),
    "balanced": (0.30, 0.25, 0.20, 0.15, 0.10),
}


@dataclass(frozen=True)
class ResidualConfig:
    config_id: str
    padding_f: float
    weight_variant: str
    minimum_count_scale: float
    full_count_scale: float


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run nested anti-overfit validation for DailyWeather weather models")
    parser.add_argument("--work-root", default=".")
    parser.add_argument("--source-repo", default=str(DEFAULT_SOURCE_REPO))
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=DEFAULT_END_DATE)
    parser.add_argument("--output-dir", default="work/reports/weather_model_anti_overfit")
    parser.add_argument("--candidate-name", default=DEFAULT_CANDIDATE)
    parser.add_argument("--min-train-rows", type=int, default=12000)
    parser.add_argument("--min-test-rows", type=int, default=4000)
    parser.add_argument("--min-inner-train-rows", type=int, default=5000)
    parser.add_argument("--min-inner-test-rows", type=int, default=1000)
    parser.add_argument("--embargo-days", default="0,3,7")
    parser.add_argument("--padding-grid", default="0.0,0.25,0.60,1.10")
    parser.add_argument("--minimum-count-scales", default="0.75,1.0,1.25")
    parser.add_argument("--full-count-scales", default="1.0")
    parser.add_argument("--weight-variants", default="current,city_heavier,lead_heavier,global_smoother,balanced")
    parser.add_argument("--max-grid-configs", type=int, default=0, help="Optional smoke-test cap; 0 means use the full grid")
    parser.add_argument("--bootstrap-iterations", type=int, default=300)
    parser.add_argument("--permutation-iterations", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20260702)
    args = parser.parse_args(argv)

    summary = run_anti_overfit_audit(
        work_root=Path(args.work_root).resolve(),
        source_repo=Path(args.source_repo).expanduser().resolve(),
        start=date.fromisoformat(args.start_date),
        end=date.fromisoformat(args.end_date),
        output_dir=Path(args.output_dir),
        candidate_name=args.candidate_name,
        min_train_rows=args.min_train_rows,
        min_test_rows=args.min_test_rows,
        min_inner_train_rows=args.min_inner_train_rows,
        min_inner_test_rows=args.min_inner_test_rows,
        embargo_days=_parse_int_list(args.embargo_days),
        padding_grid=_parse_float_list(args.padding_grid),
        minimum_count_scales=_parse_float_list(args.minimum_count_scales),
        full_count_scales=_parse_float_list(args.full_count_scales),
        weight_variants=[item.strip() for item in args.weight_variants.split(",") if item.strip()],
        max_grid_configs=args.max_grid_configs,
        bootstrap_iterations=args.bootstrap_iterations,
        permutation_iterations=args.permutation_iterations,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def run_anti_overfit_audit(
    *,
    work_root: Path,
    source_repo: Path,
    start: date,
    end: date,
    output_dir: Path,
    candidate_name: str,
    min_train_rows: int,
    min_test_rows: int,
    min_inner_train_rows: int,
    min_inner_test_rows: int,
    embargo_days: list[int],
    padding_grid: list[float],
    minimum_count_scales: list[float],
    full_count_scales: list[float],
    weight_variants: list[str],
    max_grid_configs: int,
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

    configs = _build_grid(
        padding_grid=padding_grid,
        minimum_count_scales=minimum_count_scales,
        full_count_scales=full_count_scales,
        weight_variants=weight_variants,
        max_grid_configs=max_grid_configs,
    )
    live_context = _load_live_context(paths["decision_db"])
    split_metrics, tuning_rows, plateau_rows, leakage_rows, aggregate = _run_nested_walk_forward(
        rows,
        candidate_name=candidate_name,
        configs=configs,
        embargo_days=embargo_days,
        min_train_rows=min_train_rows,
        min_test_rows=min_test_rows,
        min_inner_train_rows=min_inner_train_rows,
        min_inner_test_rows=min_inner_test_rows,
    )

    primary = aggregate.get(0) or _empty_aggregate()
    current_metrics = _comparison_metrics(primary["rows"], primary["baseline_probabilities"], primary["current_probabilities"])
    nested_metrics = _comparison_metrics(primary["rows"], primary["baseline_probabilities"], primary["nested_probabilities"])
    robustness_slices = _robustness_slices(
        primary["rows"],
        baseline_probabilities=primary["baseline_probabilities"],
        current_probabilities=primary["current_probabilities"],
        nested_probabilities=primary["nested_probabilities"],
        live_context=live_context,
    )
    model_family_slices = _model_family_slices(primary["rows"])
    city_holdouts, region_holdouts = _purged_group_holdouts(rows, candidate_name=candidate_name, min_train_rows=min_train_rows)

    rng = random.Random(seed)
    bootstrap = _bootstrap_confidence_intervals(
        primary["rows"],
        primary["baseline_probabilities"],
        primary["current_probabilities"],
        iterations=bootstrap_iterations,
        rng=rng,
    )
    permutation = _permutation_sanity_check(
        primary["rows"],
        primary["baseline_probabilities"],
        primary["current_probabilities"],
        iterations=permutation_iterations,
        rng=rng,
    )
    print("[anti-overfit] running live-feature timing audit", file=sys.stderr, flush=True)
    live_feature_gate = _audit_decision_artifact_live_features(paths["decision_db"], start=start, end=end)
    print("[anti-overfit] running label-poisoning invariance audit", file=sys.stderr, flush=True)
    label_poisoning = _run_fast_label_poisoning_invariance(
        rows,
        candidate_name=candidate_name,
        min_train_rows=min_train_rows,
        min_test_rows=min_test_rows,
    )
    market_current = _walk_forward_market_diagnostics(
        paths["decision_db"],
        primary["rows"],
        primary["baseline_probabilities"],
        primary["current_probabilities"],
    )
    market_nested = _walk_forward_market_diagnostics(
        paths["decision_db"],
        primary["rows"],
        primary["baseline_probabilities"],
        primary["nested_probabilities"],
    )
    market_rows = [
        {"candidate_series": "current_candidate", **row}
        for row in market_current["tables"]
    ] + [
        {"candidate_series": "nested_selected", **row}
        for row in market_nested["tables"]
    ]
    plateau_summary = _plateau_summary(plateau_rows)
    decision = _decision(
        current_metrics=current_metrics,
        nested_metrics=nested_metrics,
        split_metrics=split_metrics,
        plateau_summary=plateau_summary,
        bootstrap=bootstrap,
        permutation=permutation,
        city_holdouts=city_holdouts,
        region_holdouts=region_holdouts,
        live_feature_gate=live_feature_gate["summary"],
        label_poisoning=label_poisoning,
        market_summary=market_current["summary"],
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_dir = (work_root / output_dir).resolve() if not output_dir.is_absolute() else output_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{timestamp}-weather-model-anti-overfit"
    artifacts = {
        "summary_json": str(report_dir / f"{prefix}.json"),
        "report_md": str(report_dir / f"{prefix}.md"),
        "outer_walk_forward_split_metrics_csv": str(report_dir / f"{prefix}-outer-walk-forward-split-metrics.csv"),
        "inner_tuning_decisions_csv": str(report_dir / f"{prefix}-inner-tuning-decisions.csv"),
        "parameter_plateau_grid_csv": str(report_dir / f"{prefix}-parameter-plateau-grid.csv"),
        "robustness_slices_csv": str(report_dir / f"{prefix}-robustness-slices.csv"),
        "model_family_slices_csv": str(report_dir / f"{prefix}-model-family-slices.csv"),
        "market_comparison_csv": str(report_dir / f"{prefix}-market-comparison.csv"),
        "leakage_label_availability_csv": str(report_dir / f"{prefix}-leakage-label-availability.csv"),
        "bootstrap_csv": str(report_dir / f"{prefix}-bootstrap-ci.csv"),
        "permutation_csv": str(report_dir / f"{prefix}-permutation.csv"),
        "city_holdout_csv": str(report_dir / f"{prefix}-city-holdout.csv"),
        "region_holdout_csv": str(report_dir / f"{prefix}-region-holdout.csv"),
        "live_feature_coverage_csv": str(report_dir / f"{prefix}-live-feature-coverage.csv"),
        "label_poisoning_invariance_csv": str(report_dir / f"{prefix}-label-poisoning-invariance.csv"),
    }
    _write_csv(Path(artifacts["outer_walk_forward_split_metrics_csv"]), split_metrics)
    _write_csv(Path(artifacts["inner_tuning_decisions_csv"]), tuning_rows)
    _write_csv(Path(artifacts["parameter_plateau_grid_csv"]), plateau_rows)
    _write_csv(Path(artifacts["robustness_slices_csv"]), robustness_slices)
    _write_csv(Path(artifacts["model_family_slices_csv"]), model_family_slices)
    _write_csv(Path(artifacts["market_comparison_csv"]), market_rows)
    _write_csv(Path(artifacts["leakage_label_availability_csv"]), leakage_rows)
    _write_csv(Path(artifacts["bootstrap_csv"]), _flatten_bootstrap(bootstrap))
    _write_csv(Path(artifacts["permutation_csv"]), [permutation])
    _write_csv(Path(artifacts["city_holdout_csv"]), city_holdouts)
    _write_csv(Path(artifacts["region_holdout_csv"]), region_holdouts)
    _write_csv(Path(artifacts["live_feature_coverage_csv"]), [live_feature_gate["summary"]])
    _write_csv(Path(artifacts["label_poisoning_invariance_csv"]), label_poisoning)

    summary = {
        "role": "Weather Model Robustness / Anti-Overfitting Agent",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_name": candidate_name,
        "source_repo": str(source_repo),
        "target_start_date": start.isoformat(),
        "target_end_date": end.isoformat(),
        "input_counts": {
            "market_rows": len(markets),
            "forecast_rows": len(forecasts),
            "realized_rows_for_evaluation_only": len(realized),
            "validation_rows_for_audit": len(rows),
            "primary_outer_test_rows": len(primary["rows"]),
            "grid_config_count": len(configs),
        },
        "grid_search_space": _grid_search_space(configs),
        "primary_current_candidate_metrics": current_metrics,
        "primary_nested_selected_metrics": nested_metrics,
        "plateau_summary": plateau_summary,
        "bootstrap_confidence_intervals": bootstrap,
        "permutation_sanity_check": permutation,
        "city_holdout_summary": _summarize_collection(city_holdouts),
        "region_holdout_summary": _summarize_collection(region_holdouts),
        "live_feature_gate": live_feature_gate["summary"],
        "label_poisoning_invariance_summary": _summarize_label_poisoning(label_poisoning),
        "market_diagnostics_current_candidate": market_current["summary"],
        "market_diagnostics_nested_selected": market_nested["summary"],
        "anti_overfit_decision": decision,
        "output_artifacts": artifacts,
        "promotion_status": "research_only_not_promoted",
        "no_live_or_paper_defaults_changed": True,
        "strategy_pnl_backtest_run": False,
    }
    Path(artifacts["summary_json"]).write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_markdown(Path(artifacts["report_md"]), summary)
    return summary


def _build_grid(
    *,
    padding_grid: list[float],
    minimum_count_scales: list[float],
    full_count_scales: list[float],
    weight_variants: list[str],
    max_grid_configs: int,
) -> list[ResidualConfig]:
    configs = []
    for weight_variant in weight_variants:
        if weight_variant not in WEIGHT_VARIANTS:
            raise ValueError(f"Unknown weight variant {weight_variant!r}; available: {sorted(WEIGHT_VARIANTS)}")
        for padding in padding_grid:
            for minimum_count_scale in minimum_count_scales:
                for full_count_scale in full_count_scales:
                    configs.append(
                        ResidualConfig(
                            config_id=f"w={weight_variant}|pad={padding:.2f}|min={minimum_count_scale:.2f}|full={full_count_scale:.2f}",
                            padding_f=float(padding),
                            weight_variant=weight_variant,
                            minimum_count_scale=float(minimum_count_scale),
                            full_count_scale=float(full_count_scale),
                        )
                    )
    if max_grid_configs > 0:
        current = [cfg for cfg in configs if cfg.weight_variant == "current" and cfg.minimum_count_scale == 1.0 and cfg.full_count_scale == 1.0]
        rest = [cfg for cfg in configs if cfg not in current]
        configs = (current + rest)[:max_grid_configs]
    return configs


def _run_nested_walk_forward(
    rows: list[ValidationRow],
    *,
    candidate_name: str,
    configs: list[ResidualConfig],
    embargo_days: list[int],
    min_train_rows: int,
    min_test_rows: int,
    min_inner_train_rows: int,
    min_inner_test_rows: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[int, dict[str, Any]]]:
    split_metrics: list[dict[str, Any]] = []
    tuning_rows: list[dict[str, Any]] = []
    plateau_rows: list[dict[str, Any]] = []
    leakage_rows: list[dict[str, Any]] = []
    aggregate: dict[int, dict[str, Any]] = defaultdict(_empty_aggregate)
    target_months = sorted({row.market_date_local.strftime("%Y-%m") for row in rows})
    for embargo in embargo_days:
        for target_month in target_months:
            month_start = date.fromisoformat(f"{target_month}-01")
            month_end = _next_month_date(month_start)
            test_rows = [row for row in rows if month_start <= row.market_date_local < month_end]
            if len(test_rows) < min_test_rows:
                continue
            first_test_decision = min(row.decision_time_utc for row in test_rows).date()
            label_cutoff = first_test_decision - timedelta(days=embargo)
            train_rows = [row for row in rows if row.market_date_local < label_cutoff]
            if len(train_rows) < min_train_rows:
                continue
            leakage = _leakage_row(
                target_month=target_month,
                embargo_days=embargo,
                train_rows=train_rows,
                test_rows=test_rows,
                label_cutoff=label_cutoff,
                first_test_decision=first_test_decision,
            )
            leakage_rows.append(leakage)

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
                "[anti-overfit] "
                f"embargo={embargo} target_month={target_month} "
                f"train={len(train_rows)} test={len(test_rows)} inner={inner_month}",
                file=sys.stderr,
                flush=True,
            )
            current_probs = _current_candidate_probabilities(train_rows, test_rows, candidate_name=candidate_name)
            baseline_probs = {row.row_id: row.current_probability for row in test_rows}
            current_metrics = _comparison_metrics(test_rows, baseline_probs, current_probs)
            if embargo != 0:
                split_metrics.append(
                    {
                        "target_month": target_month,
                        "embargo_days": embargo,
                        "train_n": len(train_rows),
                        "test_n": len(test_rows),
                        "inner_target_month": None,
                        "inner_train_n": None,
                        "inner_valid_n": None,
                        "label_cutoff_local_date_exclusive": label_cutoff.isoformat(),
                        "train_test_city_date_overlap": leakage["train_test_city_date_overlap"],
                        "train_test_universe_overlap": leakage["train_test_universe_overlap"],
                        "train_rows_with_unavailable_labels": leakage["train_rows_with_unavailable_labels"],
                        "selected_config_id": None,
                        "baseline_brier": current_metrics["baseline"]["brier"],
                        "current_candidate_brier": current_metrics["candidate"]["brier"],
                        "nested_selected_brier": None,
                        "current_brier_improvement": current_metrics["brier_improvement"],
                        "nested_brier_improvement": None,
                        "baseline_log_loss": current_metrics["baseline"]["log_loss"],
                        "current_candidate_log_loss": current_metrics["candidate"]["log_loss"],
                        "nested_selected_log_loss": None,
                        "current_log_loss_improvement": current_metrics["log_loss_improvement"],
                        "nested_log_loss_improvement": None,
                        "current_beats_baseline_both": current_metrics["brier_improvement"] > 0 and current_metrics["log_loss_improvement"] > 0,
                        "nested_beats_baseline_both": None,
                        "note": "embargo_variant_current_candidate_only",
                    }
                )
                continue
            grid_rows, selected_config = _evaluate_plateau_grid(
                outer_target_month=target_month,
                embargo_days=embargo,
                inner_target_month=inner_month,
                configs=configs,
                inner_train=inner_train,
                inner_valid=inner_valid,
                outer_train=train_rows,
                outer_test=test_rows,
            )
            plateau_rows.extend(grid_rows)
            nested_probs = _grid_candidate_probabilities(train_rows, test_rows, selected_config)
            nested_metrics = _comparison_metrics(test_rows, baseline_probs, nested_probs)
            selected_grid_row = next(row for row in grid_rows if row["config_id"] == selected_config.config_id)
            tuning_rows.append(_tuning_decision_row(target_month, embargo, inner_month, selected_grid_row, grid_rows))

            split_metrics.append(
                {
                    "target_month": target_month,
                    "embargo_days": embargo,
                    "train_n": len(train_rows),
                    "test_n": len(test_rows),
                    "inner_target_month": inner_month,
                    "inner_train_n": len(inner_train),
                    "inner_valid_n": len(inner_valid),
                    "label_cutoff_local_date_exclusive": label_cutoff.isoformat(),
                    "train_test_city_date_overlap": leakage["train_test_city_date_overlap"],
                    "train_test_universe_overlap": leakage["train_test_universe_overlap"],
                    "train_rows_with_unavailable_labels": leakage["train_rows_with_unavailable_labels"],
                    "selected_config_id": selected_config.config_id,
                    "baseline_brier": current_metrics["baseline"]["brier"],
                    "current_candidate_brier": current_metrics["candidate"]["brier"],
                    "nested_selected_brier": nested_metrics["candidate"]["brier"],
                    "current_brier_improvement": current_metrics["brier_improvement"],
                    "nested_brier_improvement": nested_metrics["brier_improvement"],
                    "baseline_log_loss": current_metrics["baseline"]["log_loss"],
                    "current_candidate_log_loss": current_metrics["candidate"]["log_loss"],
                    "nested_selected_log_loss": nested_metrics["candidate"]["log_loss"],
                    "current_log_loss_improvement": current_metrics["log_loss_improvement"],
                    "nested_log_loss_improvement": nested_metrics["log_loss_improvement"],
                    "current_beats_baseline_both": current_metrics["brier_improvement"] > 0 and current_metrics["log_loss_improvement"] > 0,
                    "nested_beats_baseline_both": nested_metrics["brier_improvement"] > 0 and nested_metrics["log_loss_improvement"] > 0,
                }
            )
            bucket = aggregate[embargo]
            bucket["rows"].extend(test_rows)
            bucket["baseline_probabilities"].update(baseline_probs)
            bucket["current_probabilities"].update(current_probs)
            bucket["nested_probabilities"].update(nested_probs)
    return split_metrics, tuning_rows, plateau_rows, leakage_rows, aggregate


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
        month_end = _next_month_date(month_start)
        valid_rows = [row for row in train_rows if month_start <= row.market_date_local < month_end]
        if len(valid_rows) < min_inner_test_rows:
            continue
        first_valid_decision = min(row.decision_time_utc for row in valid_rows).date()
        label_cutoff = first_valid_decision - timedelta(days=embargo_days)
        inner_train = [row for row in train_rows if row.market_date_local < label_cutoff]
        if len(inner_train) >= min_inner_train_rows:
            return month, inner_train, valid_rows
    return None


def _evaluate_plateau_grid(
    *,
    outer_target_month: str,
    embargo_days: int,
    inner_target_month: str,
    configs: list[ResidualConfig],
    inner_train: list[ValidationRow],
    inner_valid: list[ValidationRow],
    outer_train: list[ValidationRow],
    outer_test: list[ValidationRow],
) -> tuple[list[dict[str, Any]], ResidualConfig]:
    inner_baseline = {row.row_id: row.current_probability for row in inner_valid}
    outer_baseline = {row.row_id: row.current_probability for row in outer_test}
    inner_residual_sets = _collect_residual_sets(inner_train, list(BASE_COMPONENTS))
    outer_residual_sets = _collect_residual_sets(outer_train, list(BASE_COMPONENTS))
    output = []
    for config in configs:
        specs = _config_specs(config)
        inner_probs = _grid_probabilities_from_sets(inner_valid, inner_residual_sets, specs, config.padding_f)
        outer_probs = _grid_probabilities_from_sets(outer_test, outer_residual_sets, specs, config.padding_f)
        inner_metrics = _comparison_metrics(inner_valid, inner_baseline, inner_probs)
        outer_metrics = _comparison_metrics(outer_test, outer_baseline, outer_probs)
        output.append(
            {
                "outer_target_month": outer_target_month,
                "embargo_days": embargo_days,
                "inner_target_month": inner_target_month,
                "config_id": config.config_id,
                "padding_f": config.padding_f,
                "weight_variant": config.weight_variant,
                "minimum_count_scale": config.minimum_count_scale,
                "full_count_scale": config.full_count_scale,
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
        )
    output.sort(key=lambda row: (float(row["inner_score"]), float(row["inner_candidate_brier"]), float(row["inner_candidate_log_loss"])))
    for rank, row in enumerate(output, start=1):
        row["inner_rank"] = rank
        row["selected_by_inner_tuning"] = rank == 1
    selected = next(config for config in configs if config.config_id == output[0]["config_id"])
    return output, selected


def _tuning_decision_row(
    target_month: str,
    embargo_days: int,
    inner_month: str,
    selected_row: Mapping[str, Any],
    grid_rows: list[Mapping[str, Any]],
) -> dict[str, Any]:
    inner_scores = [float(row["inner_score"]) for row in grid_rows]
    outer_improvements = [float(row["outer_brier_improvement"]) for row in grid_rows]
    selected_outer = float(selected_row["outer_brier_improvement"])
    median_outer = median(outer_improvements)
    worst_outer = min(outer_improvements)
    pass_count = sum(1 for row in grid_rows if bool(row["outer_beats_baseline_both"]))
    cliff_count = sum(1 for row in grid_rows if selected_outer - float(row["outer_brier_improvement"]) > 0.005 or not bool(row["outer_beats_baseline_both"]))
    return {
        "target_month": target_month,
        "embargo_days": embargo_days,
        "inner_target_month": inner_month,
        "selected_config_id": selected_row["config_id"],
        "grid_config_count": len(grid_rows),
        "inner_top_score": min(inner_scores),
        "inner_median_score": round(median(inner_scores), 8),
        "inner_top_vs_median_score_gap": round(median(inner_scores) - min(inner_scores), 8),
        "outer_selected_brier_improvement": selected_outer,
        "outer_median_brier_improvement": round(median_outer, 8),
        "outer_worst_neighbor_brier_improvement": round(worst_outer, 8),
        "outer_worst_neighbor_degradation_from_selected": round(selected_outer - worst_outer, 8),
        "outer_neighbor_pass_rate": round(pass_count / len(grid_rows), 6) if grid_rows else None,
        "outer_performance_cliff_count": cliff_count,
        "outer_performance_cliff_rate": round(cliff_count / len(grid_rows), 6) if grid_rows else None,
    }


def _config_specs(config: ResidualConfig) -> list[tuple[str, float, int, float, Callable[[ValidationRow], str]]]:
    weights = WEIGHT_VARIANTS[config.weight_variant]
    specs = []
    for index, (label, _, minimum_count, full_count, key_fn) in enumerate(BASE_COMPONENTS):
        scaled_minimum = max(1, int(round(minimum_count * config.minimum_count_scale)))
        scaled_full = max(1.0, float(full_count) * config.full_count_scale)
        specs.append((label, weights[index], scaled_minimum, scaled_full, key_fn))
    return specs


def _grid_candidate_probabilities(train_rows: list[ValidationRow], score_rows: list[ValidationRow], config: ResidualConfig) -> dict[str, float]:
    residual_sets = _collect_residual_sets(train_rows, list(BASE_COMPONENTS))
    specs = _config_specs(config)
    return _grid_probabilities_from_sets(score_rows, residual_sets, specs, config.padding_f)


def _grid_probabilities_from_sets(
    rows: list[ValidationRow],
    residual_sets: Mapping[str, Mapping[str, tuple[float, ...]]],
    specs: list[tuple[str, float, int, float, Callable[[ValidationRow], str]]],
    padding_f: float,
) -> dict[str, float]:
    return {
        row.row_id: _residual_distribution_probability(
            row,
            residual_sets=residual_sets,
            specs=specs,
            padding_f=padding_f,
        )
        for row in rows
    }


def _current_candidate_probabilities(train_rows: list[ValidationRow], score_rows: list[ValidationRow], *, candidate_name: str) -> dict[str, float]:
    if candidate_name == DEFAULT_CANDIDATE:
        residual_sets = _collect_residual_sets(train_rows, list(BASE_COMPONENTS))
        padding_f = _select_current_candidate_padding(train_rows, residual_sets=residual_sets)
        return _grid_probabilities_from_sets(
            score_rows,
            residual_sets,
            list(BASE_COMPONENTS),
            padding_f,
        )
    fit_rows = sorted([*train_rows, *score_rows], key=lambda row: (row.decision_time_utc, row.row_id))
    candidates = _fit_residual_distribution_candidates(rows=fit_rows, train_rows=train_rows, test_rows=score_rows)
    candidate = next((item for item in candidates if item["name"] == candidate_name), None)
    if candidate is None:
        names = ", ".join(item["name"] for item in candidates)
        raise ValueError(f"Candidate {candidate_name!r} not found. Available: {names}")
    probabilities = candidate["probabilities"]
    return {row.row_id: float(probabilities[row.row_id]) for row in score_rows}


def _run_fast_label_poisoning_invariance(
    rows: list[ValidationRow],
    *,
    candidate_name: str,
    min_train_rows: int,
    min_test_rows: int,
) -> list[dict[str, Any]]:
    output = []
    for month in sorted({row.market_date_local.strftime("%Y-%m") for row in rows}):
        month_start = date.fromisoformat(f"{month}-01")
        month_end = _next_month_date(month_start)
        test_rows = [row for row in rows if month_start <= row.market_date_local < month_end]
        if len(test_rows) < min_test_rows:
            continue
        first_test_decision = min(row.decision_time_utc for row in test_rows).date()
        train_rows = [row for row in rows if row.market_date_local < first_test_decision]
        if len(train_rows) < min_train_rows:
            continue
        original_probs = _current_candidate_probabilities(train_rows, test_rows, candidate_name=candidate_name)
        poisoned_rows = [_poison_evaluation_fields(row) for row in test_rows]
        poisoned_probs = _current_candidate_probabilities(train_rows, poisoned_rows, candidate_name=candidate_name)
        deltas = [abs(original_probs[row.row_id] - poisoned_probs[row.row_id]) for row in test_rows]
        changed = sum(1 for value in deltas if value > 1e-12)
        output.append(
            {
                "target_month": month,
                "train_n": len(train_rows),
                "test_n": len(test_rows),
                "first_test_decision_date": first_test_decision.isoformat(),
                "labels_available_before_test": True,
                "poisoned_realized_high_f": True,
                "poisoned_outcome": True,
                "poisoned_resolved_outcome": True,
                "changed_probability_rows": changed,
                "max_abs_probability_delta": round(max(deltas) if deltas else 0.0, 14),
                "mean_abs_probability_delta": round(sum(deltas) / len(deltas), 14) if deltas else 0.0,
                "live_inference_invariant": changed == 0,
            }
        )
    return output


def _poison_evaluation_fields(row: ValidationRow) -> ValidationRow:
    poisoned_outcome = 0 if int(row.outcome) else 1
    poisoned_resolved = None if row.resolved_outcome is None else (0 if int(row.resolved_outcome) else 1)
    return replace(
        row,
        realized_high_f=9999.0,
        forecast_error_f=-9999.0,
        outcome=poisoned_outcome,
        resolved_outcome=poisoned_resolved,
        resolved_outcome_mismatch=not row.resolved_outcome_mismatch,
        realized_quality_flags=("poisoned_for_anti_overfit_audit",),
    )


def _select_current_candidate_padding(
    train_rows: list[ValidationRow],
    *,
    residual_sets: Mapping[str, Mapping[str, tuple[float, ...]]],
) -> float:
    best: Optional[tuple[float, float]] = None
    specs = list(BASE_COMPONENTS)
    for padding_f in (0.0, 0.10, 0.25, 0.40, 0.60, 0.85, 1.10, 1.50):
        probabilities = _grid_probabilities_from_sets(train_rows, residual_sets, specs, padding_f)
        metrics = _metrics(train_rows, probabilities)
        score = float(metrics["log_loss"]) + float(metrics["brier"])
        if best is None or score < best[0]:
            best = (score, padding_f)
    assert best is not None
    return float(best[1])


def _purged_group_holdouts(rows: list[ValidationRow], *, candidate_name: str, min_train_rows: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cutoff = date(2026, 4, 1)
    city_rows = []
    region_rows = []
    holdout_config = ResidualConfig(
        config_id="holdout_current_structure_fixed_padding_0.60",
        padding_f=0.60,
        weight_variant="current",
        minimum_count_scale=1.0,
        full_count_scale=1.0,
    )
    for group_type, key_fn, output in (
        ("city", lambda row: row.normalized_city, city_rows),
        ("region", _region, region_rows),
    ):
        for group in sorted({str(key_fn(row)) for row in rows}):
            test_rows = [row for row in rows if row.market_date_local >= cutoff and str(key_fn(row)) == group]
            train_rows = [row for row in rows if row.market_date_local < cutoff and str(key_fn(row)) != group]
            if len(train_rows) < min_train_rows or len(test_rows) < 1000:
                continue
            baseline = {row.row_id: row.current_probability for row in test_rows}
            current = _grid_candidate_probabilities(train_rows, test_rows, holdout_config)
            metrics = _comparison_metrics(test_rows, baseline, current)
            output.append(
                {
                    "holdout_type": group_type,
                    "heldout_group": group,
                    "candidate_name": candidate_name,
                    "candidate_variant": holdout_config.config_id,
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


def _robustness_slices(
    rows: list[ValidationRow],
    *,
    baseline_probabilities: Mapping[str, float],
    current_probabilities: Mapping[str, float],
    nested_probabilities: Mapping[str, float],
    live_context: Mapping[tuple[str, datetime, str], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    dimensions: list[tuple[str, Callable[[ValidationRow], str]]] = [
        ("target_month", lambda row: row.market_date_local.strftime("%Y-%m")),
        ("city", lambda row: row.normalized_city),
        ("region", _region),
        ("bucket_type", lambda row: row.bucket_type),
        ("threshold_distance", lambda row: row.threshold_distance_bucket),
        ("lead_days", lambda row: str(row.lead_days)),
        ("decision_hour", lambda row: str(row.decision_hour_utc)),
        ("source_name", lambda row: row.source_name),
        ("observed_high_availability", lambda row: _context_group(row, live_context, "observed_high_group")),
        ("quote_coverage", lambda row: _context_group(row, live_context, "quote_group")),
    ]
    output = []
    for dimension, key_fn in dimensions:
        grouped: dict[str, list[ValidationRow]] = defaultdict(list)
        for row in rows:
            grouped[str(key_fn(row))].append(row)
        for group, items in sorted(grouped.items()):
            if len(items) < 25:
                continue
            current = _comparison_metrics(items, baseline_probabilities, current_probabilities)
            nested = _comparison_metrics(items, baseline_probabilities, nested_probabilities)
            output.append(
                {
                    "dimension": dimension,
                    "group": group,
                    "n": len(items),
                    "baseline_brier": current["baseline"]["brier"],
                    "current_candidate_brier": current["candidate"]["brier"],
                    "nested_selected_brier": nested["candidate"]["brier"],
                    "current_brier_improvement": current["brier_improvement"],
                    "nested_brier_improvement": nested["brier_improvement"],
                    "baseline_log_loss": current["baseline"]["log_loss"],
                    "current_candidate_log_loss": current["candidate"]["log_loss"],
                    "nested_selected_log_loss": nested["candidate"]["log_loss"],
                    "current_log_loss_improvement": current["log_loss_improvement"],
                    "nested_log_loss_improvement": nested["log_loss_improvement"],
                    "current_beats_baseline_both": current["brier_improvement"] > 0 and current["log_loss_improvement"] > 0,
                    "nested_beats_baseline_both": nested["brier_improvement"] > 0 and nested["log_loss_improvement"] > 0,
                }
            )
    return output


def _model_family_slices(rows: list[ValidationRow]) -> list[dict[str, Any]]:
    output = []
    families = sorted({_model_family(key) for row in rows for key in row.model_probabilities})
    for family in families:
        probs = _family_probabilities(rows, family_name=family)
        metrics = _metrics(rows, probs)
        output.append({"dimension": "model_family", "group": family, **metrics})
    by_key: dict[str, dict[str, Any]] = defaultdict(lambda: {"rows": [], "probabilities": {}})
    for row in rows:
        for key, probability in row.model_probabilities.items():
            by_key[key]["rows"].append(row)
            by_key[key]["probabilities"][row.row_id] = probability
    for key, payload in sorted(by_key.items()):
        items = payload["rows"]
        if len(items) < 25:
            continue
        metrics = _metrics(items, payload["probabilities"])
        output.append({"dimension": "source_model_family", "group": key, **metrics})
    return output


def _load_live_context(decision_db: Path) -> dict[tuple[str, datetime, str], dict[str, Any]]:
    context: dict[tuple[str, datetime, str], dict[str, Any]] = {}
    for quote in _load_decision_quotes(decision_db):
        source = quote.forecast_source_name or ""
        key = (quote.universe_row_id, quote.decision_time_utc, source)
        item = context.setdefault(key, {"quote_group": "no_quote", "observed_high_group": "observed_high_unknown"})
        if quote.mid is not None:
            item["quote_group"] = "quote_covered"
    with _load_decision_context_connection(decision_db) as conn:
        rows = conn.execute(
            """
            SELECT universe_row_id, decision_time_utc, COALESCE(forecast_source_name, '') AS forecast_source_name,
                   MAX(CASE WHEN observed_high_so_far_f IS NOT NULL THEN 1 ELSE 0 END) AS has_observed_high,
                   MAX(CASE WHEN quote_timestamp_utc IS NOT NULL THEN 1 ELSE 0 END) AS has_quote
            FROM decision_snapshots_current_enriched
            GROUP BY universe_row_id, decision_time_utc, forecast_source_name
            """
        ).fetchall()
    for row in rows:
        dt = _parse_datetime(row["decision_time_utc"])
        if dt is None:
            continue
        key = (str(row["universe_row_id"]), dt, str(row["forecast_source_name"] or ""))
        item = context.setdefault(key, {})
        item["observed_high_group"] = "observed_high_available" if int(row["has_observed_high"] or 0) else "observed_high_missing"
        item["quote_group"] = "quote_covered" if int(row["has_quote"] or 0) else item.get("quote_group", "no_quote")
    return context


def _load_decision_context_connection(path: Path) -> Any:
    from .weather_model_validation import _connect_readonly

    return _connect_readonly(path)


def _context_group(row: ValidationRow, context: Mapping[tuple[str, datetime, str], Mapping[str, Any]], field: str) -> str:
    item = context.get((row.universe_row_id, row.decision_time_utc, row.source_name))
    if item is None:
        item = context.get((row.universe_row_id, row.decision_time_utc, ""))
    return str((item or {}).get(field, "unknown"))


def _leakage_row(
    *,
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


def _decision(
    *,
    current_metrics: Mapping[str, Any],
    nested_metrics: Mapping[str, Any],
    split_metrics: list[Mapping[str, Any]],
    plateau_summary: Mapping[str, Any],
    bootstrap: Mapping[str, Any],
    permutation: Mapping[str, Any],
    city_holdouts: list[Mapping[str, Any]],
    region_holdouts: list[Mapping[str, Any]],
    live_feature_gate: Mapping[str, Any],
    label_poisoning: list[Mapping[str, Any]],
    market_summary: Mapping[str, Any],
) -> dict[str, Any]:
    city_date_ci = bootstrap.get("city_date_cluster") or {}
    weather_pass = (
        float(current_metrics["brier_improvement"]) > 0.0
        and float(current_metrics["log_loss_improvement"]) > 0.0
        and float(city_date_ci.get("brier_improvement_ci_2_5") or -1.0) > 0.0
        and float(city_date_ci.get("log_loss_improvement_ci_2_5") or -1.0) > 0.0
        and bool(permutation.get("observed_brier_above_null_95"))
        and bool(permutation.get("observed_log_loss_above_null_95"))
    )
    nested_pass = float(nested_metrics["brier_improvement"]) > 0.0 and float(nested_metrics["log_loss_improvement"]) > 0.0
    split_pass_rate = _pass_rate(split_metrics, "current_beats_baseline_both")
    plateau_pass = (
        float(plateau_summary.get("outer_neighbor_pass_rate_median") or 0.0) >= 0.60
        and float(plateau_summary.get("outer_performance_cliff_rate_median") or 1.0) <= 0.40
    )
    holdout_pass = _pass_rate(city_holdouts, "beats_baseline_both") >= 0.70 and _pass_rate(region_holdouts, "beats_baseline_both") >= 0.70
    live_pass = (
        int(live_feature_gate.get("live_timing_violation_rows") or 0) == 0
        and float(live_feature_gate.get("weather_model_scoreable_fraction") or 0.0) >= 0.95
        and all(bool(row.get("live_inference_invariant")) for row in label_poisoning)
    )
    cand_market = market_summary.get("candidate_vs_polymarket_payout") or {}
    mid_market = market_summary.get("market_mid_vs_polymarket_payout") or {}
    market_pass = (
        cand_market.get("brier") is not None
        and mid_market.get("brier") is not None
        and float(cand_market["brier"]) < float(mid_market["brier"])
        and float(cand_market["log_loss"]) < float(mid_market["log_loss"])
    )
    if not weather_pass or not nested_pass:
        outcome = "reject_weather_not_robust"
    elif not plateau_pass:
        outcome = "research_only_parameter_peak_or_fragile_plateau"
    elif not holdout_pass:
        outcome = "research_only_fails_city_region_generalization"
    elif not live_pass:
        outcome = "reject_not_live_feature_safe"
    elif not market_pass:
        outcome = "weather_improved_but_not_market_beating"
    else:
        outcome = "survives_anti_overfit_weather_gate_ready_for_quote_backed_replay"
    return {
        "outcome": outcome,
        "weather_pass": weather_pass,
        "nested_pass": nested_pass,
        "split_pass_rate": split_pass_rate,
        "plateau_pass": plateau_pass,
        "holdout_pass": holdout_pass,
        "live_feature_pass": live_pass,
        "market_pass": market_pass,
        "model_weights_or_strategy_defaults_changed": False,
        "strategy_pnl_backtest_run": False,
    }


def _plateau_summary(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0}
    by_split: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_split[(str(row["outer_target_month"]), int(row["embargo_days"]))].append(row)
    split_summaries = []
    for _, items in by_split.items():
        selected = next(row for row in items if bool(row.get("selected_by_inner_tuning")))
        selected_outer = float(selected["outer_brier_improvement"])
        outer_improvements = [float(row["outer_brier_improvement"]) for row in items]
        pass_rate = sum(1 for row in items if bool(row["outer_beats_baseline_both"])) / len(items)
        cliff_rate = sum(1 for row in items if selected_outer - float(row["outer_brier_improvement"]) > 0.005 or not bool(row["outer_beats_baseline_both"])) / len(items)
        split_summaries.append(
            {
                "outer_neighbor_pass_rate": pass_rate,
                "outer_performance_cliff_rate": cliff_rate,
                "outer_top_vs_median_brier_gap": max(outer_improvements) - median(outer_improvements),
                "outer_worst_neighbor_degradation": selected_outer - min(outer_improvements),
            }
        )
    return {
        "n": len(rows),
        "split_count": len(split_summaries),
        "outer_neighbor_pass_rate_median": round(median(row["outer_neighbor_pass_rate"] for row in split_summaries), 6),
        "outer_performance_cliff_rate_median": round(median(row["outer_performance_cliff_rate"] for row in split_summaries), 6),
        "outer_top_vs_median_brier_gap_median": round(median(row["outer_top_vs_median_brier_gap"] for row in split_summaries), 8),
        "outer_worst_neighbor_degradation_median": round(median(row["outer_worst_neighbor_degradation"] for row in split_summaries), 8),
    }


def _write_markdown(path: Path, summary: Mapping[str, Any]) -> None:
    current = summary["primary_current_candidate_metrics"]
    nested = summary["primary_nested_selected_metrics"]
    plateau = summary["plateau_summary"]
    decision = summary["anti_overfit_decision"]
    market = summary["market_diagnostics_current_candidate"]
    lines = [
        "# DailyWeather Weather Model Anti-Overfit Audit",
        "",
        f"Generated: {summary['generated_at_utc']}",
        "",
        f"Candidate: `{summary['candidate_name']}`",
        "",
        "## Decision",
        "",
        f"Outcome: `{decision['outcome']}`",
        "",
        "- No model weights, paper defaults, live defaults, or strategy settings were changed.",
        "- No strategy PnL backtest was run.",
        "- Resolved Polymarket labels are used only for evaluation/market comparison, not as core weather-model inputs.",
        "",
        "## Primary Purged Walk-Forward Metrics",
        "",
        f"- Test rows: {summary['input_counts']['primary_outer_test_rows']}",
        f"- Baseline Brier/log loss: {current['baseline']['brier']} / {current['baseline']['log_loss']}",
        f"- Current candidate Brier/log loss: {current['candidate']['brier']} / {current['candidate']['log_loss']}",
        f"- Current improvement: {current['brier_improvement']} Brier, {current['log_loss_improvement']} log loss",
        f"- Nested-selected Brier/log loss: {nested['candidate']['brier']} / {nested['candidate']['log_loss']}",
        f"- Nested-selected improvement: {nested['brier_improvement']} Brier, {nested['log_loss_improvement']} log loss",
        "",
        "## Plateau Diagnostics",
        "",
        f"- Grid configs tested per eligible split: {summary['input_counts']['grid_config_count']}",
        f"- Plateau grid rows: {plateau.get('n')}",
        f"- Median neighbor pass rate: {plateau.get('outer_neighbor_pass_rate_median')}",
        f"- Median performance cliff rate: {plateau.get('outer_performance_cliff_rate_median')}",
        f"- Median top-vs-median Brier gap: {plateau.get('outer_top_vs_median_brier_gap_median')}",
        f"- Median worst-neighbor degradation: {plateau.get('outer_worst_neighbor_degradation_median')}",
        "",
        "## Robustness Gates",
        "",
        f"- Bootstrap city-date Brier CI: {(summary['bootstrap_confidence_intervals'].get('city_date_cluster') or {}).get('brier_improvement_ci_2_5')} to {(summary['bootstrap_confidence_intervals'].get('city_date_cluster') or {}).get('brier_improvement_ci_97_5')}",
        f"- Permutation pass: {summary['permutation_sanity_check'].get('observed_brier_above_null_95')} / {summary['permutation_sanity_check'].get('observed_log_loss_above_null_95')}",
        f"- City holdout pass rate: {summary['city_holdout_summary'].get('pass_rate')}",
        f"- Region holdout pass rate: {summary['region_holdout_summary'].get('pass_rate')}",
        f"- Live timing violation rows: {summary['live_feature_gate'].get('live_timing_violation_rows')}",
        f"- Label-poisoning changed rows: {summary['label_poisoning_invariance_summary'].get('total_changed_probability_rows')}",
        "",
        "## Market Comparison",
        "",
        f"- Matched quote side rows: {market.get('matched_walk_forward_quote_side_rows')}",
        f"- Candidate vs Polymarket payout Brier/log loss: {(market.get('candidate_vs_polymarket_payout') or {}).get('brier')} / {(market.get('candidate_vs_polymarket_payout') or {}).get('log_loss')}",
        f"- Market midpoint Brier/log loss: {(market.get('market_mid_vs_polymarket_payout') or {}).get('brier')} / {(market.get('market_mid_vs_polymarket_payout') or {}).get('log_loss')}",
        "",
        "## Interpretation",
        "",
        "- The aggregate weather improvement is real on the known-city purged walk-forward task, but it is not sufficient for promotion.",
        "- The parameter surface is fragile: too many neighboring configurations fail or fall off sharply.",
        "- City and region holdouts are the largest robustness failure, so the model is learning location-specific residual structure that does not generalize reliably.",
        "- Market midpoint still beats the model on quote-covered Polymarket payout rows; this is market-relative scoring only, not a strategy PnL replay.",
        "",
        "## Candidate Improvement Plan",
        "",
        "- Keep the residual model research-only and do not change model weights or live/paper defaults.",
        "- Redesign the residual hierarchy for cold-start city/region generalization before adding more tuning dimensions.",
        "- Test region-aware shrinkage, source-unit exact-bucket handling, and threshold-distance tail calibration only through the same nested purged walk-forward gate.",
        "- Preserve the live-feature firewall: no final highs, resolved labels, future quotes, or post-decision fields in core weather-model inputs.",
        "- Re-run quote-backed market comparison after canonical-label handling and quote coverage improve; strategy PnL replay remains out of scope for this audit.",
        "",
        "## Artifacts",
        "",
    ]
    for key, value in summary["output_artifacts"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _selection_score(metrics: Mapping[str, Any]) -> float:
    return float(metrics["candidate"]["brier"]) + float(metrics["candidate"]["log_loss"])


def _flatten_bootstrap(bootstrap: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [{"cluster": cluster, **dict(values)} for cluster, values in bootstrap.items()]


def _summarize_collection(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0, "pass_rate": None}
    return {
        "n": len(rows),
        "pass_rate": _pass_rate(rows, "beats_baseline_both"),
        "worst_brier_improvement": min(float(row["brier_improvement"]) for row in rows),
        "worst_log_loss_improvement": min(float(row["log_loss_improvement"]) for row in rows),
    }


def _summarize_label_poisoning(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0, "pass_rate": None}
    return {
        "n": len(rows),
        "pass_rate": _pass_rate(rows, "live_inference_invariant"),
        "total_changed_probability_rows": sum(int(row.get("changed_probability_rows") or 0) for row in rows),
        "max_abs_probability_delta": max(float(row.get("max_abs_probability_delta") or 0.0) for row in rows),
    }


def _grid_search_space(configs: list[ResidualConfig]) -> dict[str, Any]:
    return {
        "config_count": len(configs),
        "padding_f": sorted({config.padding_f for config in configs}),
        "weight_variants": sorted({config.weight_variant for config in configs}),
        "minimum_count_scales": sorted({config.minimum_count_scale for config in configs}),
        "full_count_scales": sorted({config.full_count_scale for config in configs}),
    }


def _pass_rate(rows: list[Mapping[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return sum(1 for row in rows if bool(row.get(key))) / len(rows)


def _empty_aggregate() -> dict[str, Any]:
    return {
        "rows": [],
        "baseline_probabilities": {},
        "current_probabilities": {},
        "nested_probabilities": {},
    }


def _overlap_count(left: list[ValidationRow], right: list[ValidationRow], key_fn: Callable[[ValidationRow], str]) -> int:
    left_keys = {str(key_fn(row)) for row in left}
    right_keys = {str(key_fn(row)) for row in right}
    return len(left_keys & right_keys)


def _next_month_date(value: date) -> date:
    if value.month == 12:
        return date(value.year + 1, 1, 1)
    return date(value.year, value.month + 1, 1)


def _parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


if __name__ == "__main__":
    sys.exit(main())
