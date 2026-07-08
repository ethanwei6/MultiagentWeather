from __future__ import annotations

import argparse
import json
import math
import random
import sqlite3
import sys
from collections import defaultdict
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

from .weather_model_validation import (
    DEFAULT_END_DATE,
    DEFAULT_SOURCE_REPO,
    DEFAULT_START_DATE,
    DecisionQuoteRow,
    ValidationRow,
    _artifact_paths,
    _bucket_contains,
    _build_validation_rows,
    _clip_probability,
    _connect_readonly,
    _family_probabilities,
    _fit_residual_distribution_candidates,
    _load_decision_quotes,
    _load_forecasts,
    _load_markets,
    _load_realized,
    _load_weather_strategy,
    _metrics,
    _parse_datetime,
    _time_split,
    _write_csv,
)


DEFAULT_CANDIDATE = "residual_hierarchical_empirical_distribution"
US_STATE_CODES = {
    "AL",
    "AK",
    "AZ",
    "AR",
    "CA",
    "CO",
    "CT",
    "DE",
    "FL",
    "GA",
    "HI",
    "IA",
    "ID",
    "IL",
    "IN",
    "KS",
    "KY",
    "LA",
    "MA",
    "MD",
    "ME",
    "MI",
    "MN",
    "MO",
    "MS",
    "MT",
    "NC",
    "ND",
    "NE",
    "NH",
    "NJ",
    "NM",
    "NV",
    "NY",
    "OH",
    "OK",
    "OR",
    "PA",
    "RI",
    "SC",
    "SD",
    "TN",
    "TX",
    "UT",
    "VA",
    "VT",
    "WA",
    "WI",
    "WV",
    "WY",
}


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run anti-overfitting robustness checks for DailyWeather weather models")
    parser.add_argument("--work-root", default=".")
    parser.add_argument("--source-repo", default=str(DEFAULT_SOURCE_REPO))
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=DEFAULT_END_DATE)
    parser.add_argument("--output-dir", default="work/reports/weather_model_robustness")
    parser.add_argument("--candidate-name", default=DEFAULT_CANDIDATE)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--rolling-min-train-rows", type=int, default=12000)
    parser.add_argument("--rolling-min-test-rows", type=int, default=4000)
    parser.add_argument("--window-train-days", type=int, default=60)
    parser.add_argument("--window-test-days", type=int, default=30)
    parser.add_argument("--bootstrap-iterations", type=int, default=300)
    parser.add_argument("--permutation-iterations", type=int, default=120)
    parser.add_argument("--residual-shuffle-iterations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260701)
    args = parser.parse_args(argv)

    summary = run_robustness_audit(
        work_root=Path(args.work_root).resolve(),
        source_repo=Path(args.source_repo).expanduser().resolve(),
        start=date.fromisoformat(args.start_date),
        end=date.fromisoformat(args.end_date),
        output_dir=Path(args.output_dir),
        candidate_name=args.candidate_name,
        train_fraction=args.train_fraction,
        rolling_min_train_rows=args.rolling_min_train_rows,
        rolling_min_test_rows=args.rolling_min_test_rows,
        window_train_days=args.window_train_days,
        window_test_days=args.window_test_days,
        bootstrap_iterations=args.bootstrap_iterations,
        permutation_iterations=args.permutation_iterations,
        residual_shuffle_iterations=args.residual_shuffle_iterations,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def run_robustness_audit(
    *,
    work_root: Path,
    source_repo: Path,
    start: date,
    end: date,
    output_dir: Path,
    candidate_name: str,
    train_fraction: float,
    rolling_min_train_rows: int,
    rolling_min_test_rows: int,
    window_train_days: int,
    window_test_days: int,
    bootstrap_iterations: int,
    permutation_iterations: int,
    residual_shuffle_iterations: int,
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

    main_train_rows, main_test_rows, main_split = _time_split(rows, train_fraction=train_fraction)
    rng = random.Random(seed)

    rolling_results: list[dict[str, Any]] = []
    aggregate_rows: list[ValidationRow] = []
    aggregate_baseline_probabilities: dict[str, float] = {}
    aggregate_candidate_probabilities: dict[str, float] = {}

    for split_name, train_rows, test_rows in _rolling_month_splits(
        rows,
        min_train_rows=rolling_min_train_rows,
        min_test_rows=rolling_min_test_rows,
    ):
        result = _evaluate_split(
            split_name,
            "rolling_origin_month",
            train_rows,
            test_rows,
            candidate_name=candidate_name,
        )
        rolling_results.append(result["summary"])
        aggregate_rows.extend(test_rows)
        aggregate_baseline_probabilities.update(result["baseline_probabilities"])
        aggregate_candidate_probabilities.update(result["candidate_probabilities"])

    if not aggregate_rows:
        main_result = _evaluate_split(
            "main_time_split",
            "fallback",
            main_train_rows,
            main_test_rows,
            candidate_name=candidate_name,
        )
        aggregate_rows = list(main_test_rows)
        aggregate_baseline_probabilities = dict(main_result["baseline_probabilities"])
        aggregate_candidate_probabilities = dict(main_result["candidate_probabilities"])

    nonoverlap_results = [
        _evaluate_split(split_name, "nonoverlap_window", train_rows, test_rows, candidate_name=candidate_name)["summary"]
        for split_name, train_rows, test_rows in _nonoverlap_window_splits(
            rows,
            train_days=window_train_days,
            test_days=window_test_days,
            min_train_rows=rolling_min_train_rows,
            min_test_rows=rolling_min_test_rows,
        )
    ]

    heldout_city_results = _heldout_group_results(
        rows,
        main_split["cutoff_decision_time_utc"],
        candidate_name=candidate_name,
        group_name="heldout_city",
        key_fn=lambda row: row.normalized_city,
        min_train_rows=rolling_min_train_rows,
        min_test_rows=1000,
    )
    heldout_region_results = _heldout_group_results(
        rows,
        main_split["cutoff_decision_time_utc"],
        candidate_name=candidate_name,
        group_name="heldout_region",
        key_fn=_region,
        min_train_rows=rolling_min_train_rows,
        min_test_rows=1000,
    )

    aggregate_metrics = _comparison_metrics(
        aggregate_rows,
        aggregate_baseline_probabilities,
        aggregate_candidate_probabilities,
    )
    stress_rows = _stress_rows(aggregate_rows, aggregate_baseline_probabilities, aggregate_candidate_probabilities)
    bootstrap = _bootstrap_confidence_intervals(
        aggregate_rows,
        aggregate_baseline_probabilities,
        aggregate_candidate_probabilities,
        iterations=bootstrap_iterations,
        rng=rng,
    )
    permutation = _permutation_sanity_check(
        aggregate_rows,
        aggregate_baseline_probabilities,
        aggregate_candidate_probabilities,
        iterations=permutation_iterations,
        rng=rng,
    )
    residual_shuffle = _train_residual_shuffle_check(
        main_train_rows,
        main_test_rows,
        candidate_name=candidate_name,
        iterations=residual_shuffle_iterations,
        rng=rng,
    )
    settlement_audit = _settlement_label_audit(rows)
    market_diagnostics = _walk_forward_market_diagnostics(
        paths["decision_db"],
        aggregate_rows,
        aggregate_baseline_probabilities,
        aggregate_candidate_probabilities,
    )
    point_in_time_audit = _point_in_time_data_audit(paths)

    decision = _robustness_decision(
        aggregate_metrics=aggregate_metrics,
        rolling_results=rolling_results,
        nonoverlap_results=nonoverlap_results,
        bootstrap=bootstrap,
        permutation=permutation,
        market_diagnostics=market_diagnostics,
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_dir = (work_root / output_dir).resolve() if not output_dir.is_absolute() else output_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{timestamp}-weather-model-robustness"
    artifacts = {
        "summary_json": str(report_dir / f"{prefix}.json"),
        "report_md": str(report_dir / f"{prefix}.md"),
        "split_metrics_csv": str(report_dir / f"{prefix}-split-metrics.csv"),
        "stress_metrics_csv": str(report_dir / f"{prefix}-stress-metrics.csv"),
        "heldout_city_csv": str(report_dir / f"{prefix}-heldout-city.csv"),
        "heldout_region_csv": str(report_dir / f"{prefix}-heldout-region.csv"),
        "bootstrap_csv": str(report_dir / f"{prefix}-bootstrap-ci.csv"),
        "permutation_csv": str(report_dir / f"{prefix}-permutation.csv"),
        "residual_shuffle_csv": str(report_dir / f"{prefix}-residual-shuffle.csv"),
        "settlement_audit_csv": str(report_dir / f"{prefix}-settlement-label-audit.csv"),
        "market_execution_csv": str(report_dir / f"{prefix}-market-execution.csv"),
    }

    split_rows = []
    for table, table_rows in (
        ("rolling_origin_month", rolling_results),
        ("nonoverlap_window", nonoverlap_results),
    ):
        split_rows.extend({"table": table, **row} for row in table_rows)
    _write_csv(Path(artifacts["split_metrics_csv"]), split_rows)
    _write_csv(Path(artifacts["stress_metrics_csv"]), stress_rows)
    _write_csv(Path(artifacts["heldout_city_csv"]), heldout_city_results)
    _write_csv(Path(artifacts["heldout_region_csv"]), heldout_region_results)
    _write_csv(Path(artifacts["bootstrap_csv"]), _flatten_bootstrap(bootstrap))
    _write_csv(Path(artifacts["permutation_csv"]), [permutation])
    _write_csv(Path(artifacts["residual_shuffle_csv"]), residual_shuffle["iterations"])
    _write_csv(Path(artifacts["settlement_audit_csv"]), settlement_audit["tables"])
    _write_csv(Path(artifacts["market_execution_csv"]), market_diagnostics["tables"])

    summary = {
        "role": "Weather Model Robustness / Anti-Overfitting Agent",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_repo": str(source_repo),
        "candidate_name": candidate_name,
        "target_start_date": start.isoformat(),
        "target_end_date": end.isoformat(),
        "artifacts_used": {key: str(value) for key, value in paths.items()},
        "input_counts": {
            "market_rows": len(markets),
            "forecast_rows": len(forecasts),
            "realized_rows": len(realized),
            "validation_rows": len(rows),
            "aggregate_walk_forward_test_rows": len(aggregate_rows),
        },
        "main_time_split": main_split,
        "aggregate_walk_forward_metrics": aggregate_metrics,
        "rolling_origin_months": rolling_results,
        "nonoverlap_windows": nonoverlap_results,
        "heldout_city_summary": _summarize_split_collection(heldout_city_results),
        "heldout_region_summary": _summarize_split_collection(heldout_region_results),
        "bootstrap_confidence_intervals": bootstrap,
        "permutation_sanity_check": permutation,
        "train_residual_shuffle_sanity_check": residual_shuffle["summary"],
        "settlement_label_audit": settlement_audit["summary"],
        "market_diagnostics": market_diagnostics["summary"],
        "point_in_time_data_audit": point_in_time_audit,
        "robustness_decision": decision,
        "output_artifacts": artifacts,
        "promotion_status": "research_only_not_promoted",
        "no_live_or_paper_defaults_changed": True,
    }
    Path(artifacts["summary_json"]).write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_markdown_report(Path(artifacts["report_md"]), summary)
    return summary


def _evaluate_split(
    split_name: str,
    split_type: str,
    train_rows: list[ValidationRow],
    test_rows: list[ValidationRow],
    *,
    candidate_name: str,
) -> dict[str, Any]:
    fit_rows = sorted([*train_rows, *test_rows], key=lambda row: (row.decision_time_utc, row.row_id))
    baseline_probabilities = {row.row_id: row.current_probability for row in fit_rows}
    kernel_tight_probabilities = _family_probabilities(fit_rows, family_name="kernel_tight")
    candidates = _fit_residual_distribution_candidates(rows=fit_rows, train_rows=train_rows, test_rows=test_rows)
    candidate = next((item for item in candidates if item["name"] == candidate_name), None)
    if candidate is None:
        names = ", ".join(item["name"] for item in candidates)
        raise ValueError(f"Candidate {candidate_name!r} not found. Available: {names}")
    candidate_probabilities = {key: float(value) for key, value in candidate["probabilities"].items()}
    metrics = _comparison_metrics(test_rows, baseline_probabilities, candidate_probabilities)
    kernel_metrics = _metrics(test_rows, kernel_tight_probabilities)
    train_start, train_end = _decision_range(train_rows)
    test_start, test_end = _decision_range(test_rows)
    summary = {
        "split_name": split_name,
        "split_type": split_type,
        "train_n": len(train_rows),
        "test_n": len(test_rows),
        "train_start_utc": train_start,
        "train_end_utc": train_end,
        "test_start_utc": test_start,
        "test_end_utc": test_end,
        "test_city_count": len({row.normalized_city for row in test_rows}),
        "test_region_count": len({_region(row) for row in test_rows}),
        "baseline_brier": metrics["baseline"]["brier"],
        "candidate_brier": metrics["candidate"]["brier"],
        "brier_improvement": metrics["brier_improvement"],
        "baseline_log_loss": metrics["baseline"]["log_loss"],
        "candidate_log_loss": metrics["candidate"]["log_loss"],
        "log_loss_improvement": metrics["log_loss_improvement"],
        "candidate_bias": metrics["candidate"]["bias"],
        "kernel_tight_brier": kernel_metrics["brier"],
        "kernel_tight_log_loss": kernel_metrics["log_loss"],
        "beats_baseline_brier": metrics["brier_improvement"] > 0,
        "beats_baseline_log_loss": metrics["log_loss_improvement"] > 0,
        "beats_baseline_both": metrics["brier_improvement"] > 0 and metrics["log_loss_improvement"] > 0,
        "beats_kernel_tight_brier": float(metrics["candidate"]["brier"]) < float(kernel_metrics["brier"]),
        "beats_kernel_tight_log_loss": float(metrics["candidate"]["log_loss"]) < float(kernel_metrics["log_loss"]),
    }
    return {
        "summary": summary,
        "baseline_probabilities": {row.row_id: baseline_probabilities[row.row_id] for row in test_rows},
        "candidate_probabilities": {row.row_id: candidate_probabilities[row.row_id] for row in test_rows},
    }


def _rolling_month_splits(
    rows: list[ValidationRow],
    *,
    min_train_rows: int,
    min_test_rows: int,
) -> list[tuple[str, list[ValidationRow], list[ValidationRow]]]:
    months = sorted({_month_key(row.decision_time_utc) for row in rows})
    splits = []
    for month in months:
        month_start = _month_start(month)
        month_end = _next_month(month_start)
        train_rows = [row for row in rows if row.decision_time_utc < month_start]
        test_rows = [row for row in rows if month_start <= row.decision_time_utc < month_end]
        if len(train_rows) >= min_train_rows and len(test_rows) >= min_test_rows:
            splits.append((f"train_before_{month}_test_{month}", train_rows, test_rows))
    return splits


def _nonoverlap_window_splits(
    rows: list[ValidationRow],
    *,
    train_days: int,
    test_days: int,
    min_train_rows: int,
    min_test_rows: int,
) -> list[tuple[str, list[ValidationRow], list[ValidationRow]]]:
    if not rows:
        return []
    start = min(row.decision_time_utc for row in rows)
    end = max(row.decision_time_utc for row in rows)
    splits = []
    train_start = start
    while train_start + timedelta(days=train_days + test_days) <= end + timedelta(seconds=1):
        train_end = train_start + timedelta(days=train_days)
        test_end = train_end + timedelta(days=test_days)
        train_rows = [row for row in rows if train_start <= row.decision_time_utc < train_end]
        test_rows = [row for row in rows if train_end <= row.decision_time_utc < test_end]
        if len(train_rows) >= min_train_rows and len(test_rows) >= min_test_rows:
            split_name = f"train_{train_start.date()}_{train_end.date()}_test_{test_end.date()}"
            splits.append((split_name, train_rows, test_rows))
        train_start = test_end
    return splits


def _heldout_group_results(
    rows: list[ValidationRow],
    cutoff_iso: str,
    *,
    candidate_name: str,
    group_name: str,
    key_fn: Any,
    min_train_rows: int,
    min_test_rows: int,
) -> list[dict[str, Any]]:
    cutoff = _parse_datetime(cutoff_iso)
    if cutoff is None:
        return []
    groups = sorted({str(key_fn(row)) for row in rows})
    results = []
    for group in groups:
        train_rows = [row for row in rows if row.decision_time_utc <= cutoff and str(key_fn(row)) != group]
        test_rows = [row for row in rows if row.decision_time_utc > cutoff and str(key_fn(row)) == group]
        if len(train_rows) < min_train_rows or len(test_rows) < min_test_rows:
            continue
        result = _evaluate_split(
            f"{group_name}:{group}",
            group_name,
            train_rows,
            test_rows,
            candidate_name=candidate_name,
        )["summary"]
        result["heldout_group"] = group
        results.append(result)
    results.sort(key=lambda row: (not bool(row["beats_baseline_both"]), -int(row["test_n"]), str(row["heldout_group"])))
    return results


def _comparison_metrics(
    rows: list[ValidationRow],
    baseline_probabilities: Mapping[str, float],
    candidate_probabilities: Mapping[str, float],
) -> dict[str, Any]:
    baseline = _metrics(rows, baseline_probabilities)
    candidate = _metrics(rows, candidate_probabilities)
    return {
        "baseline": baseline,
        "candidate": candidate,
        "brier_improvement": round(float(baseline["brier"]) - float(candidate["brier"]), 8),
        "log_loss_improvement": round(float(baseline["log_loss"]) - float(candidate["log_loss"]), 8),
    }


def _stress_rows(
    rows: list[ValidationRow],
    baseline_probabilities: Mapping[str, float],
    candidate_probabilities: Mapping[str, float],
) -> list[dict[str, Any]]:
    dimensions = {
        "city": lambda row: row.normalized_city,
        "region": _region,
        "bucket_type": lambda row: row.bucket_type,
        "bucket_shape": lambda row: row.bucket_shape,
        "lead_days": lambda row: str(row.lead_days),
        "day_type": lambda row: row.day_type,
        "decision_hour_utc": lambda row: str(row.decision_hour_utc),
        "threshold_distance": lambda row: row.threshold_distance_bucket,
        "decision_month": lambda row: _month_key(row.decision_time_utc),
        "source_name": lambda row: row.source_name,
    }
    output = []
    for dimension, key_fn in dimensions.items():
        grouped: dict[str, list[ValidationRow]] = defaultdict(list)
        for row in rows:
            grouped[str(key_fn(row))].append(row)
        for group, items in grouped.items():
            if len(items) < 250:
                continue
            metrics = _comparison_metrics(items, baseline_probabilities, candidate_probabilities)
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
                    "candidate_bias": metrics["candidate"]["bias"],
                    "beats_baseline_both": metrics["brier_improvement"] > 0 and metrics["log_loss_improvement"] > 0,
                }
            )
    output.sort(key=lambda item: (str(item["dimension"]), float(item["brier_improvement"])))
    return output


def _bootstrap_confidence_intervals(
    rows: list[ValidationRow],
    baseline_probabilities: Mapping[str, float],
    candidate_probabilities: Mapping[str, float],
    *,
    iterations: int,
    rng: random.Random,
) -> dict[str, Any]:
    return {
        "universe_row_id_cluster": _cluster_bootstrap(
            rows,
            baseline_probabilities,
            candidate_probabilities,
            key_fn=lambda row: row.universe_row_id,
            iterations=iterations,
            rng=rng,
        ),
        "city_date_cluster": _cluster_bootstrap(
            rows,
            baseline_probabilities,
            candidate_probabilities,
            key_fn=lambda row: f"{row.normalized_city}|{row.market_date_local.isoformat()}",
            iterations=iterations,
            rng=rng,
        ),
    }


def _cluster_bootstrap(
    rows: list[ValidationRow],
    baseline_probabilities: Mapping[str, float],
    candidate_probabilities: Mapping[str, float],
    *,
    key_fn: Any,
    iterations: int,
    rng: random.Random,
) -> dict[str, Any]:
    grouped: dict[str, dict[str, float]] = defaultdict(lambda: {"n": 0.0, "base_brier": 0.0, "cand_brier": 0.0, "base_log": 0.0, "cand_log": 0.0})
    for row in rows:
        key = str(key_fn(row))
        outcome = row.outcome
        base = _clip_probability(baseline_probabilities[row.row_id])
        cand = _clip_probability(candidate_probabilities[row.row_id])
        grouped[key]["n"] += 1.0
        grouped[key]["base_brier"] += (base - outcome) ** 2
        grouped[key]["cand_brier"] += (cand - outcome) ** 2
        grouped[key]["base_log"] += _log_loss_value(base, outcome)
        grouped[key]["cand_log"] += _log_loss_value(cand, outcome)
    clusters = list(grouped.values())
    if not clusters or iterations <= 0:
        return {"iterations": 0, "cluster_count": len(clusters)}
    brier_improvements = []
    log_loss_improvements = []
    for _ in range(iterations):
        selected = [clusters[rng.randrange(len(clusters))] for _ in clusters]
        n = sum(item["n"] for item in selected)
        base_brier = sum(item["base_brier"] for item in selected) / n
        cand_brier = sum(item["cand_brier"] for item in selected) / n
        base_log = sum(item["base_log"] for item in selected) / n
        cand_log = sum(item["cand_log"] for item in selected) / n
        brier_improvements.append(base_brier - cand_brier)
        log_loss_improvements.append(base_log - cand_log)
    observed = _comparison_metrics(rows, baseline_probabilities, candidate_probabilities)
    return {
        "iterations": iterations,
        "cluster_count": len(clusters),
        "observed_brier_improvement": observed["brier_improvement"],
        "observed_log_loss_improvement": observed["log_loss_improvement"],
        "brier_improvement_ci_2_5": round(_quantile(brier_improvements, 0.025), 8),
        "brier_improvement_ci_50": round(_quantile(brier_improvements, 0.50), 8),
        "brier_improvement_ci_97_5": round(_quantile(brier_improvements, 0.975), 8),
        "log_loss_improvement_ci_2_5": round(_quantile(log_loss_improvements, 0.025), 8),
        "log_loss_improvement_ci_50": round(_quantile(log_loss_improvements, 0.50), 8),
        "log_loss_improvement_ci_97_5": round(_quantile(log_loss_improvements, 0.975), 8),
    }


def _permutation_sanity_check(
    rows: list[ValidationRow],
    baseline_probabilities: Mapping[str, float],
    candidate_probabilities: Mapping[str, float],
    *,
    iterations: int,
    rng: random.Random,
) -> dict[str, Any]:
    outcomes = [row.outcome for row in rows]
    base_probs = [_clip_probability(baseline_probabilities[row.row_id]) for row in rows]
    cand_probs = [_clip_probability(candidate_probabilities[row.row_id]) for row in rows]
    observed = _comparison_metrics(rows, baseline_probabilities, candidate_probabilities)
    brier_null = []
    log_null = []
    for _ in range(iterations):
        shuffled = list(outcomes)
        rng.shuffle(shuffled)
        base_brier = sum((prob - outcome) ** 2 for prob, outcome in zip(base_probs, shuffled)) / len(shuffled)
        cand_brier = sum((prob - outcome) ** 2 for prob, outcome in zip(cand_probs, shuffled)) / len(shuffled)
        base_log = sum(_log_loss_value(prob, outcome) for prob, outcome in zip(base_probs, shuffled)) / len(shuffled)
        cand_log = sum(_log_loss_value(prob, outcome) for prob, outcome in zip(cand_probs, shuffled)) / len(shuffled)
        brier_null.append(base_brier - cand_brier)
        log_null.append(base_log - cand_log)
    return {
        "iterations": iterations,
        "observed_brier_improvement": observed["brier_improvement"],
        "observed_log_loss_improvement": observed["log_loss_improvement"],
        "null_brier_improvement_95": round(_quantile(brier_null, 0.95), 8) if brier_null else None,
        "null_log_loss_improvement_95": round(_quantile(log_null, 0.95), 8) if log_null else None,
        "observed_brier_above_null_95": bool(brier_null and observed["brier_improvement"] > _quantile(brier_null, 0.95)),
        "observed_log_loss_above_null_95": bool(log_null and observed["log_loss_improvement"] > _quantile(log_null, 0.95)),
    }


def _train_residual_shuffle_check(
    train_rows: list[ValidationRow],
    test_rows: list[ValidationRow],
    *,
    candidate_name: str,
    iterations: int,
    rng: random.Random,
) -> dict[str, Any]:
    if iterations <= 0:
        return {"summary": {"iterations": 0}, "iterations": []}
    observed = _evaluate_split("main_time_split_observed", "observed", train_rows, test_rows, candidate_name=candidate_name)["summary"]
    residuals = [row.realized_high_f - row.forecast_high_f for row in train_rows]
    rows_out = []
    for idx in range(iterations):
        shuffled = list(residuals)
        rng.shuffle(shuffled)
        shuffled_train = []
        for row, residual in zip(train_rows, shuffled):
            realized_high = row.forecast_high_f + residual
            outcome = 1 if _bucket_contains(realized_high, row.threshold_low_f, row.threshold_high_f) else 0
            shuffled_train.append(
                replace(
                    row,
                    realized_high_f=realized_high,
                    forecast_error_f=row.forecast_high_f - realized_high,
                    outcome=outcome,
                    resolved_outcome_mismatch=(row.resolved_outcome is not None and int(row.resolved_outcome) != outcome),
                )
            )
        result = _evaluate_split(
            f"train_residual_shuffle_{idx + 1}",
            "train_residual_shuffle",
            shuffled_train,
            test_rows,
            candidate_name=candidate_name,
        )["summary"]
        rows_out.append(result)
    brier_values = [float(row["brier_improvement"]) for row in rows_out]
    log_values = [float(row["log_loss_improvement"]) for row in rows_out]
    summary = {
        "iterations": iterations,
        "observed_brier_improvement": observed["brier_improvement"],
        "observed_log_loss_improvement": observed["log_loss_improvement"],
        "shuffled_brier_improvement_median": round(_quantile(brier_values, 0.50), 8),
        "shuffled_brier_improvement_95": round(_quantile(brier_values, 0.95), 8),
        "shuffled_log_loss_improvement_median": round(_quantile(log_values, 0.50), 8),
        "shuffled_log_loss_improvement_95": round(_quantile(log_values, 0.95), 8),
        "observed_brier_above_shuffled_95": observed["brier_improvement"] > _quantile(brier_values, 0.95),
        "observed_log_loss_above_shuffled_95": observed["log_loss_improvement"] > _quantile(log_values, 0.95),
    }
    return {"summary": summary, "iterations": rows_out}


def _settlement_label_audit(rows: list[ValidationRow]) -> dict[str, Any]:
    by_universe: dict[str, ValidationRow] = {}
    for row in rows:
        if row.resolved_outcome is None:
            continue
        by_universe.setdefault(row.universe_row_id, row)
    unique_rows = list(by_universe.values())
    dimensions = {
        "all": lambda row: "all",
        "city": lambda row: row.normalized_city,
        "region": _region,
        "bucket_type": lambda row: row.bucket_type,
        "target_month": lambda row: row.market_date_local.strftime("%Y-%m"),
        "settlement_margin_f": lambda row: _margin_bucket(_settlement_margin_f(row)),
    }
    output = []
    for dimension, key_fn in dimensions.items():
        grouped: dict[str, list[ValidationRow]] = defaultdict(list)
        for row in unique_rows:
            grouped[str(key_fn(row))].append(row)
        for group, items in grouped.items():
            if len(items) < 5:
                continue
            mismatches = [int(item.resolved_outcome_mismatch) for item in items]
            weather_rate = sum(item.outcome for item in items) / len(items)
            polymarket_rate = sum(int(item.resolved_outcome or 0) for item in items) / len(items)
            output.append(
                {
                    "dimension": dimension,
                    "group": group,
                    "n": len(items),
                    "mismatch_n": sum(mismatches),
                    "mismatch_rate": round(sum(mismatches) / len(mismatches), 6),
                    "weather_yes_rate": round(weather_rate, 6),
                    "polymarket_yes_rate": round(polymarket_rate, 6),
                }
            )
    output.sort(key=lambda row: (str(row["dimension"]), -float(row["mismatch_rate"]), -int(row["n"])))
    summary_row = next((row for row in output if row["dimension"] == "all" and row["group"] == "all"), None)
    return {
        "summary": summary_row or {"n": 0, "mismatch_rate": None},
        "tables": output,
    }


def _walk_forward_market_diagnostics(
    decision_db: Path,
    rows: list[ValidationRow],
    baseline_probabilities: Mapping[str, float],
    candidate_probabilities: Mapping[str, float],
) -> dict[str, Any]:
    quotes = _load_decision_quotes(decision_db)
    quotes_by_key: dict[tuple[str, datetime], list[DecisionQuoteRow]] = defaultdict(list)
    for quote in quotes:
        quotes_by_key[(quote.universe_row_id, quote.decision_time_utc)].append(quote)

    side_rows: list[dict[str, Any]] = []
    matched_quote_ids: set[tuple[str, datetime, str]] = set()
    for row in rows:
        for quote in quotes_by_key.get((row.universe_row_id, row.decision_time_utc), []):
            if quote.forecast_source_name and quote.forecast_source_name != row.source_name:
                continue
            if quote.mid is None:
                continue
            side = quote.side.upper()
            side_outcome = quote.resolved_outcome_yes if side == "YES" else 1 - quote.resolved_outcome_yes
            base_event = _clip_probability(baseline_probabilities[row.row_id])
            cand_event = _clip_probability(candidate_probabilities[row.row_id])
            base_side = base_event if side == "YES" else 1.0 - base_event
            cand_side = cand_event if side == "YES" else 1.0 - cand_event
            matched_quote_ids.add((quote.universe_row_id, quote.decision_time_utc, side))
            side_rows.append(
                {
                    "city": row.normalized_city,
                    "region": _region(row),
                    "lead_days": row.lead_days,
                    "decision_hour_utc": row.decision_hour_utc,
                    "bucket_type": row.bucket_type,
                    "threshold_distance_bucket": row.threshold_distance_bucket,
                    "side": side,
                    "side_outcome": side_outcome,
                    "baseline_probability": _clip_probability(base_side),
                    "candidate_probability": _clip_probability(cand_side),
                    "market_mid": _clip_probability(float(quote.mid)),
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
        row
        for row in side_rows
        if row.get("bid") is not None
        and row.get("ask") is not None
        and row.get("spread") is not None
        and not row.get("quote_quality_flags")
        and 0.0 <= float(row["bid"]) <= 1.0
        and 0.0 < float(row["ask"]) <= 1.0
        and float(row["spread"]) <= 0.10
    ]
    execution_rows = []
    for family, probability_key in (("candidate", "candidate_probability"), ("baseline", "baseline_probability")):
        for action in ("buy_at_ask", "sell_at_bid"):
            for threshold in (0.00, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20):
                execution_rows.append(
                    {
                        "table": "execution_threshold",
                        "probability_series": family,
                        "action": action,
                        **_execution_threshold_summary(
                            tradable_rows,
                            probability_key=probability_key,
                            action=action,
                            threshold=threshold,
                        ),
                    }
                )
    for dimension, key in (
        ("side", "side"),
        ("bucket_type", "bucket_type"),
        ("lead_days", "lead_days"),
        ("decision_hour_utc", "decision_hour_utc"),
        ("spread_bucket", "spread_bucket"),
    ):
        grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in side_rows:
            group = _spread_bucket(row.get("spread")) if key == "spread_bucket" else str(row.get(key))
            grouped[group].append(row)
        for group, items in grouped.items():
            if len(items) < 25:
                continue
            execution_rows.append(
                {
                    "table": "market_group",
                    "dimension": dimension,
                    "group": group,
                    "n": len(items),
                    "candidate_brier": _side_probability_metrics(items, "candidate_probability")["brier"],
                    "market_mid_brier": _side_probability_metrics(items, "market_mid")["brier"],
                    "candidate_log_loss": _side_probability_metrics(items, "candidate_probability")["log_loss"],
                    "market_mid_log_loss": _side_probability_metrics(items, "market_mid")["log_loss"],
                    "weather_label_mismatch_rate": _rate([int(item["weather_polymarket_mismatch"]) for item in items]),
                }
            )
    summary = {
        "all_quote_side_rows": len(quotes),
        "matched_walk_forward_quote_side_rows": len(side_rows),
        "unique_matched_quote_side_rows": len(matched_quote_ids),
        "tradable_rows_spread_lte_10c": len(tradable_rows),
        "baseline_vs_polymarket_payout": _side_probability_metrics(side_rows, "baseline_probability"),
        "candidate_vs_polymarket_payout": _side_probability_metrics(side_rows, "candidate_probability"),
        "market_mid_vs_polymarket_payout": _side_probability_metrics(side_rows, "market_mid"),
        "weather_label_mismatch_rate": _rate([int(row["weather_polymarket_mismatch"]) for row in side_rows]),
        "candidate_buy_edge_5c": _execution_threshold_summary(
            tradable_rows,
            probability_key="candidate_probability",
            action="buy_at_ask",
            threshold=0.05,
        ),
        "candidate_sell_edge_5c": _execution_threshold_summary(
            tradable_rows,
            probability_key="candidate_probability",
            action="sell_at_bid",
            threshold=0.05,
        ),
    }
    return {"summary": summary, "tables": execution_rows}


def _side_probability_metrics(rows: list[Mapping[str, Any]], probability_key: str) -> dict[str, Any]:
    valid = [row for row in rows if row.get(probability_key) is not None]
    if not valid:
        return {"n": 0, "avg_probability": None, "actual_rate": None, "bias": None, "brier": None, "log_loss": None}
    probs = [_clip_probability(float(row[probability_key])) for row in valid]
    outcomes = [int(row["side_outcome"]) for row in valid]
    return {
        "n": len(valid),
        "avg_probability": round(sum(probs) / len(probs), 6),
        "actual_rate": round(sum(outcomes) / len(outcomes), 6),
        "bias": round(sum(probs) / len(probs) - sum(outcomes) / len(outcomes), 6),
        "brier": round(sum((prob - outcome) ** 2 for prob, outcome in zip(probs, outcomes)) / len(probs), 8),
        "log_loss": round(sum(_log_loss_value(prob, outcome) for prob, outcome in zip(probs, outcomes)) / len(probs), 8),
    }


def _execution_threshold_summary(
    rows: list[Mapping[str, Any]],
    *,
    probability_key: str,
    action: str,
    threshold: float,
) -> dict[str, Any]:
    selected = []
    for row in rows:
        probability = float(row[probability_key])
        bid = float(row["bid"])
        ask = float(row["ask"])
        if action == "buy_at_ask" and probability - ask >= threshold:
            selected.append(row)
        elif action == "sell_at_bid" and bid - probability >= threshold:
            selected.append(row)
    if not selected:
        return {
            "edge_threshold": threshold,
            "n": 0,
            "actual_rate": None,
            "avg_probability": None,
            "avg_bid": None,
            "avg_ask": None,
            "avg_edge": None,
            "flat_return_on_collateral": None,
            "weather_label_mismatch_rate": None,
        }
    outcomes = [int(row["side_outcome"]) for row in selected]
    probabilities = [float(row[probability_key]) for row in selected]
    bids = [float(row["bid"]) for row in selected]
    asks = [float(row["ask"]) for row in selected]
    if action == "buy_at_ask":
        edges = [probability - ask for probability, ask in zip(probabilities, asks)]
        returns = [(outcome / ask) - 1.0 for outcome, ask in zip(outcomes, asks) if ask > 0.0]
    else:
        edges = [bid - probability for probability, bid in zip(probabilities, bids)]
        returns = [((bid - outcome) / (1.0 - bid)) for outcome, bid in zip(outcomes, bids) if bid < 1.0]
    return {
        "edge_threshold": threshold,
        "n": len(selected),
        "actual_rate": round(sum(outcomes) / len(outcomes), 6),
        "avg_probability": round(sum(probabilities) / len(probabilities), 6),
        "avg_bid": round(sum(bids) / len(bids), 6),
        "avg_ask": round(sum(asks) / len(asks), 6),
        "avg_edge": round(sum(edges) / len(edges), 6),
        "flat_return_on_collateral": round(sum(returns) / len(returns), 6) if returns else None,
        "weather_label_mismatch_rate": _rate([int(row["weather_polymarket_mismatch"]) for row in selected]),
    }


def _point_in_time_data_audit(paths: Mapping[str, Path]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    with _connect_readonly(paths["weather_db"]) as conn:
        row = conn.execute(
            """
            SELECT COUNT(*) AS n,
                   SUM(CASE WHEN forecast_run_time_utc IS NOT NULL AND forecast_run_time_utc > decision_time_utc THEN 1 ELSE 0 END) AS forecast_run_after_decision,
                   SUM(CASE WHEN data_quality_flags LIKE '%historical_forecast_endpoint_is_asof_approximation%' THEN 1 ELSE 0 END) AS asof_approximation_rows
            FROM forecast_snapshots
            """
        ).fetchone()
        output["forecast_snapshots"] = dict(row)
    with _connect_readonly(paths["decision_db"]) as conn:
        row = conn.execute(
            """
            SELECT COUNT(*) AS n,
                   SUM(CASE WHEN quote_timestamp_utc IS NOT NULL THEN 1 ELSE 0 END) AS quote_rows,
                   SUM(CASE WHEN quote_timestamp_utc IS NOT NULL AND quote_timestamp_utc > decision_time_utc THEN 1 ELSE 0 END) AS future_quote_rows,
                   MAX(price_stale_seconds) AS max_price_stale_seconds
            FROM decision_snapshots
            """
        ).fetchone()
        output["decision_snapshots"] = dict(row)
    return output


def _robustness_decision(
    *,
    aggregate_metrics: Mapping[str, Any],
    rolling_results: list[Mapping[str, Any]],
    nonoverlap_results: list[Mapping[str, Any]],
    bootstrap: Mapping[str, Any],
    permutation: Mapping[str, Any],
    market_diagnostics: Mapping[str, Any],
) -> dict[str, Any]:
    rolling_pass_rate = _pass_rate(rolling_results)
    nonoverlap_pass_rate = _pass_rate(nonoverlap_results)
    city_date_ci = bootstrap.get("city_date_cluster") or {}
    weather_ci_pass = (
        float(city_date_ci.get("brier_improvement_ci_2_5") or -1.0) > 0.0
        and float(city_date_ci.get("log_loss_improvement_ci_2_5") or -1.0) > 0.0
    )
    permutation_pass = bool(permutation.get("observed_brier_above_null_95")) and bool(permutation.get("observed_log_loss_above_null_95"))
    aggregate_weather_pass = (
        float(aggregate_metrics["brier_improvement"]) > 0.0
        and float(aggregate_metrics["log_loss_improvement"]) > 0.0
    )
    weather_survives = (
        aggregate_weather_pass
        and weather_ci_pass
        and permutation_pass
        and rolling_pass_rate >= 0.60
        and (not nonoverlap_results or nonoverlap_pass_rate >= 0.50)
    )
    market = market_diagnostics.get("summary") or {}
    candidate_market = market.get("candidate_vs_polymarket_payout") or {}
    market_mid = market.get("market_mid_vs_polymarket_payout") or {}
    buy_5c = market.get("candidate_buy_edge_5c") or {}
    market_survives = (
        candidate_market.get("brier") is not None
        and market_mid.get("brier") is not None
        and float(candidate_market["brier"]) < float(market_mid["brier"])
        and float(candidate_market["log_loss"]) < float(market_mid["log_loss"])
        and buy_5c.get("flat_return_on_collateral") is not None
        and float(buy_5c["flat_return_on_collateral"]) > 0.0
    )
    if weather_survives and market_survives:
        outcome = "survives_robustness_can_move_to_quote_backed_replay"
    elif weather_survives:
        outcome = "improves_weather_prediction_but_does_not_beat_market"
    else:
        outcome = "appears_overfit_or_too_fragile_reject_or_redesign_before_promotion"
    return {
        "outcome": outcome,
        "weather_survives": weather_survives,
        "market_survives": market_survives,
        "aggregate_weather_pass": aggregate_weather_pass,
        "city_date_bootstrap_ci_pass": weather_ci_pass,
        "permutation_pass": permutation_pass,
        "rolling_pass_rate": round(rolling_pass_rate, 6),
        "nonoverlap_pass_rate": round(nonoverlap_pass_rate, 6) if nonoverlap_results else None,
        "model_weights_or_strategy_defaults_changed": False,
    }


def _write_markdown_report(path: Path, summary: Mapping[str, Any]) -> None:
    aggregate = summary["aggregate_walk_forward_metrics"]
    decision = summary["robustness_decision"]
    market = summary["market_diagnostics"]
    settlement = summary["settlement_label_audit"]
    bootstrap = summary["bootstrap_confidence_intervals"]["city_date_cluster"]
    heldout_city = summary["heldout_city_summary"]
    heldout_region = summary["heldout_region_summary"]
    point_in_time = summary["point_in_time_data_audit"]
    lines = [
        "# DailyWeather Weather Model Robustness Audit",
        "",
        f"Generated: {summary['generated_at_utc']}",
        "",
        f"Candidate under audit: `{summary['candidate_name']}`",
        "",
        "## Decision",
        "",
        f"Outcome: `{decision['outcome']}`",
        "",
        "- The candidate remains research-only.",
        "- No `model_weights.json`, paper defaults, live defaults, or strategy settings were changed.",
        "- Market diagnostics use only compact quote-covered rows and are not a full trading replay.",
        "",
        "## Aggregate Walk-Forward Weather Metrics",
        "",
        f"- Test rows: {summary['input_counts']['aggregate_walk_forward_test_rows']:,}",
        f"- Baseline Brier/log loss: {aggregate['baseline']['brier']} / {aggregate['baseline']['log_loss']}",
        f"- Candidate Brier/log loss: {aggregate['candidate']['brier']} / {aggregate['candidate']['log_loss']}",
        f"- Brier improvement: {aggregate['brier_improvement']}",
        f"- Log-loss improvement: {aggregate['log_loss_improvement']}",
        f"- Candidate bias: {aggregate['candidate']['bias']}",
        "",
        "## Robustness Checks",
        "",
        f"- Rolling-origin pass rate: {decision['rolling_pass_rate']}",
        f"- Non-overlap pass rate: {decision['nonoverlap_pass_rate']}",
        f"- Held-out city pass rate: {heldout_city.get('pass_rate')} across {heldout_city.get('n')} city holdouts",
        f"- Held-out region pass rate: {heldout_region.get('pass_rate')} across {heldout_region.get('n')} region holdouts",
        f"- Worst held-out city Brier/log-loss improvement: {heldout_city.get('worst_brier_improvement')} / {heldout_city.get('worst_log_loss_improvement')}",
        f"- Worst held-out region Brier/log-loss improvement: {heldout_region.get('worst_brier_improvement')} / {heldout_region.get('worst_log_loss_improvement')}",
        f"- City-date cluster bootstrap Brier improvement 95% CI: {bootstrap.get('brier_improvement_ci_2_5')} to {bootstrap.get('brier_improvement_ci_97_5')}",
        f"- City-date cluster bootstrap log-loss improvement 95% CI: {bootstrap.get('log_loss_improvement_ci_2_5')} to {bootstrap.get('log_loss_improvement_ci_97_5')}",
        f"- Label-shuffle permutation pass: {decision['permutation_pass']}",
        "",
        "Interpretation: the candidate is robust for the existing city universe when each validation month is trained only on prior data, but it is not robust enough for new-city or new-region cold starts without city history.",
        "",
        "## Market Diagnostics",
        "",
        f"- Matched walk-forward quote side rows: {market.get('matched_walk_forward_quote_side_rows')}",
        f"- Tradable rows with clean quotes and spread <= 10c: {market.get('tradable_rows_spread_lte_10c')}",
        f"- Candidate vs Polymarket payout Brier/log loss: {(market.get('candidate_vs_polymarket_payout') or {}).get('brier')} / {(market.get('candidate_vs_polymarket_payout') or {}).get('log_loss')}",
        f"- Market midpoint Brier/log loss: {(market.get('market_mid_vs_polymarket_payout') or {}).get('brier')} / {(market.get('market_mid_vs_polymarket_payout') or {}).get('log_loss')}",
        f"- Candidate buy-at-ask edge >= 5c flat return: {(market.get('candidate_buy_edge_5c') or {}).get('flat_return_on_collateral')}",
        f"- Candidate sell-at-bid edge >= 5c flat return: {(market.get('candidate_sell_edge_5c') or {}).get('flat_return_on_collateral')}",
        f"- Weather/Polymarket label mismatch rate: {market.get('weather_label_mismatch_rate')}",
        "",
        "## Settlement Label Audit",
        "",
        f"- Unique market buckets audited: {settlement.get('n')}",
        f"- Mismatch count: {settlement.get('mismatch_n')}",
        f"- Mismatch rate: {settlement.get('mismatch_rate')}",
        "",
        "## Point-In-Time Audit",
        "",
        f"- Forecast rows with forecast run after decision time: {(point_in_time.get('forecast_snapshots') or {}).get('forecast_run_after_decision')}",
        f"- Forecast rows carrying historical-as-of approximation flag: {(point_in_time.get('forecast_snapshots') or {}).get('asof_approximation_rows')}",
        f"- Decision rows with future quote timestamp: {(point_in_time.get('decision_snapshots') or {}).get('future_quote_rows')}",
        f"- Max accepted quote staleness seconds: {(point_in_time.get('decision_snapshots') or {}).get('max_price_stale_seconds')}",
        "",
        "## Artifacts",
        "",
    ]
    for key, value in summary["output_artifacts"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _decision_range(rows: list[ValidationRow]) -> tuple[Optional[str], Optional[str]]:
    if not rows:
        return None, None
    return min(row.decision_time_utc for row in rows).isoformat(), max(row.decision_time_utc for row in rows).isoformat()


def _month_key(dt: datetime) -> str:
    return f"{dt.year:04d}-{dt.month:02d}"


def _month_start(month: str) -> datetime:
    year, month_num = (int(part) for part in month.split("-"))
    return datetime(year, month_num, 1, tzinfo=timezone.utc)


def _next_month(dt: datetime) -> datetime:
    if dt.month == 12:
        return datetime(dt.year + 1, 1, 1, tzinfo=timezone.utc)
    return datetime(dt.year, dt.month + 1, 1, tzinfo=timezone.utc)


def _region(row: ValidationRow) -> str:
    region = (row.country_region or "").upper()
    if region in US_STATE_CODES:
        return "US"
    if region in {"GB", "FR", "ES", "RU"} or row.timezone_name.startswith("Europe/"):
        return "Europe"
    if region in {"CN", "SG", "TW", "KR", "JP"} or row.timezone_name.startswith("Asia/"):
        return "Asia"
    return region or "unknown"


def _settlement_margin_f(row: ValidationRow) -> Optional[float]:
    distances = []
    if row.threshold_low_f is not None:
        distances.append(abs(row.realized_high_f - row.threshold_low_f))
    if row.threshold_high_f is not None:
        distances.append(abs(row.realized_high_f - row.threshold_high_f))
    return min(distances) if distances else None


def _margin_bucket(value: Optional[float]) -> str:
    if value is None:
        return "unbounded"
    if value <= 0.5:
        return "0-0.5F"
    if value <= 1.0:
        return "0.5-1F"
    if value <= 2.0:
        return "1-2F"
    if value <= 4.0:
        return "2-4F"
    return "4F+"


def _spread_bucket(value: Any) -> str:
    if value is None:
        return "missing"
    spread = float(value)
    if spread <= 0.02:
        return "0-2c"
    if spread <= 0.05:
        return "2-5c"
    if spread <= 0.10:
        return "5-10c"
    return "10c+"


def _log_loss_value(probability: float, outcome: int) -> float:
    prob = _clip_probability(probability)
    return -(outcome * math.log(prob) + (1 - outcome) * math.log(1 - prob))


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = q * (len(ordered) - 1)
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def _rate(values: list[int]) -> Optional[float]:
    if not values:
        return None
    return round(sum(values) / len(values), 6)


def _pass_rate(rows: list[Mapping[str, Any]]) -> float:
    if not rows:
        return 0.0
    return sum(1 for row in rows if row.get("beats_baseline_both")) / len(rows)


def _summarize_split_collection(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0, "pass_rate": None}
    return {
        "n": len(rows),
        "pass_rate": round(_pass_rate(rows), 6),
        "worst_brier_improvement": min(float(row["brier_improvement"]) for row in rows),
        "worst_log_loss_improvement": min(float(row["log_loss_improvement"]) for row in rows),
    }


def _flatten_bootstrap(bootstrap: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [{"cluster": cluster, **dict(values)} for cluster, values in bootstrap.items()]


if __name__ == "__main__":
    sys.exit(main())
