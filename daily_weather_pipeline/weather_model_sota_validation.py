from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

from .weather_model_robustness import (
    DEFAULT_CANDIDATE,
    _bootstrap_confidence_intervals,
    _comparison_metrics,
    _evaluate_split,
    _pass_rate,
    _permutation_sanity_check,
    _region,
    _stress_rows,
    _walk_forward_market_diagnostics,
)
from .weather_model_validation import (
    DEFAULT_END_DATE,
    DEFAULT_SOURCE_REPO,
    DEFAULT_START_DATE,
    ValidationRow,
    _artifact_paths,
    _build_validation_rows,
    _load_forecasts,
    _load_markets,
    _load_realized,
    _load_weather_strategy,
    _write_csv,
)


SOURCE_NOTES = [
    {
        "topic": "rolling_and_blocked_cv",
        "source": "Cross-validation guidance for temporal/spatial data: use rolling/forward validation and spatial or buffered blocking when dependence makes random splits optimistic.",
        "url": "https://en.wikipedia.org/wiki/Cross-validation_(statistics)#Cross_validation_for_time-series,_spatial_and_spatiotemporal_models",
    },
    {
        "topic": "purged_embargoed_cv",
        "source": "Purged/embargoed validation removes samples whose label horizon overlaps the test set and can add a temporal buffer.",
        "url": "https://en.wikipedia.org/wiki/Purged_cross-validation",
    },
    {
        "topic": "spatial_cv",
        "source": "Spatial CV papers warn that spatially structured test sets may not be independent unless assessment sets force spatial extrapolation or use exclusion buffers.",
        "url": "https://arxiv.org/abs/2303.07334",
    },
    {
        "topic": "weatherbench_probabilistic",
        "source": "WeatherBench Probability uses established probabilistic verification metrics and operational baselines for weather forecasts.",
        "url": "https://arxiv.org/abs/2205.00865",
    },
    {
        "topic": "weatherbench2_operational_protocol",
        "source": "WeatherBench 2 emphasizes operational verification practices, caveats about reanalysis look-ahead, and probabilistic prediction.",
        "url": "https://arxiv.org/abs/2308.15560",
    },
]


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run SOTA-inspired leakage and validation tests for DailyWeather weather models")
    parser.add_argument("--work-root", default=".")
    parser.add_argument("--source-repo", default=str(DEFAULT_SOURCE_REPO))
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=DEFAULT_END_DATE)
    parser.add_argument("--output-dir", default="work/reports/weather_model_sota_validation")
    parser.add_argument("--candidate-name", default=DEFAULT_CANDIDATE)
    parser.add_argument("--min-train-rows", type=int, default=12000)
    parser.add_argument("--min-test-rows", type=int, default=4000)
    parser.add_argument("--bootstrap-iterations", type=int, default=300)
    parser.add_argument("--permutation-iterations", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20260702)
    args = parser.parse_args(argv)

    summary = run_sota_validation(
        work_root=Path(args.work_root).resolve(),
        source_repo=Path(args.source_repo).expanduser().resolve(),
        start=date.fromisoformat(args.start_date),
        end=date.fromisoformat(args.end_date),
        output_dir=Path(args.output_dir),
        candidate_name=args.candidate_name,
        min_train_rows=args.min_train_rows,
        min_test_rows=args.min_test_rows,
        bootstrap_iterations=args.bootstrap_iterations,
        permutation_iterations=args.permutation_iterations,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def run_sota_validation(
    *,
    work_root: Path,
    source_repo: Path,
    start: date,
    end: date,
    output_dir: Path,
    candidate_name: str,
    min_train_rows: int,
    min_test_rows: int,
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

    leakage_scan = _decision_time_split_leakage_scan(rows, min_train_rows=min_train_rows, min_test_rows=min_test_rows)
    primary_results, primary_rows, baseline_probs, candidate_probs = _run_purged_target_months(
        rows,
        candidate_name=candidate_name,
        embargo_days=0,
        min_train_rows=min_train_rows,
        min_test_rows=min_test_rows,
    )
    embargo_3_results, _, _, _ = _run_purged_target_months(
        rows,
        candidate_name=candidate_name,
        embargo_days=3,
        min_train_rows=min_train_rows,
        min_test_rows=min_test_rows,
    )
    embargo_7_results, _, _, _ = _run_purged_target_months(
        rows,
        candidate_name=candidate_name,
        embargo_days=7,
        min_train_rows=min_train_rows,
        min_test_rows=min_test_rows,
    )
    city_holdouts = _purged_group_holdouts(
        rows,
        candidate_name=candidate_name,
        key_fn=lambda row: row.normalized_city,
        group_type="purged_city_holdout",
        min_train_rows=min_train_rows,
        min_test_rows=1000,
    )
    region_holdouts = _purged_group_holdouts(
        rows,
        candidate_name=candidate_name,
        key_fn=_region,
        group_type="purged_region_holdout",
        min_train_rows=min_train_rows,
        min_test_rows=1000,
    )

    rng = random.Random(seed)
    aggregate = _comparison_metrics(primary_rows, baseline_probs, candidate_probs)
    bootstrap = _bootstrap_confidence_intervals(
        primary_rows,
        baseline_probs,
        candidate_probs,
        iterations=bootstrap_iterations,
        rng=rng,
    )
    permutation = _permutation_sanity_check(
        primary_rows,
        baseline_probs,
        candidate_probs,
        iterations=permutation_iterations,
        rng=rng,
    )
    stress = _stress_rows(primary_rows, baseline_probs, candidate_probs)
    market = _walk_forward_market_diagnostics(paths["decision_db"], primary_rows, baseline_probs, candidate_probs)
    decision = _decision(
        aggregate=aggregate,
        bootstrap=bootstrap,
        permutation=permutation,
        purged_months=primary_results,
        city_holdouts=city_holdouts,
        region_holdouts=region_holdouts,
        market=market,
        leakage_scan=leakage_scan,
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_dir = (work_root / output_dir).resolve() if not output_dir.is_absolute() else output_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{timestamp}-weather-model-sota-validation"
    artifacts = {
        "summary_json": str(report_dir / f"{prefix}.json"),
        "report_md": str(report_dir / f"{prefix}.md"),
        "decision_time_leakage_scan_csv": str(report_dir / f"{prefix}-decision-time-leakage-scan.csv"),
        "purged_month_splits_csv": str(report_dir / f"{prefix}-purged-month-splits.csv"),
        "purged_embargo_splits_csv": str(report_dir / f"{prefix}-purged-embargo-splits.csv"),
        "purged_city_holdout_csv": str(report_dir / f"{prefix}-purged-city-holdout.csv"),
        "purged_region_holdout_csv": str(report_dir / f"{prefix}-purged-region-holdout.csv"),
        "stress_metrics_csv": str(report_dir / f"{prefix}-stress-metrics.csv"),
        "bootstrap_csv": str(report_dir / f"{prefix}-bootstrap-ci.csv"),
        "permutation_csv": str(report_dir / f"{prefix}-permutation.csv"),
        "market_execution_csv": str(report_dir / f"{prefix}-market-execution.csv"),
    }
    _write_csv(Path(artifacts["decision_time_leakage_scan_csv"]), leakage_scan)
    _write_csv(Path(artifacts["purged_month_splits_csv"]), primary_results)
    _write_csv(
        Path(artifacts["purged_embargo_splits_csv"]),
        [{"embargo_days": 3, **row} for row in embargo_3_results] + [{"embargo_days": 7, **row} for row in embargo_7_results],
    )
    _write_csv(Path(artifacts["purged_city_holdout_csv"]), city_holdouts)
    _write_csv(Path(artifacts["purged_region_holdout_csv"]), region_holdouts)
    _write_csv(Path(artifacts["stress_metrics_csv"]), stress)
    _write_csv(Path(artifacts["bootstrap_csv"]), _flatten_bootstrap(bootstrap))
    _write_csv(Path(artifacts["permutation_csv"]), [permutation])
    _write_csv(Path(artifacts["market_execution_csv"]), market["tables"])

    summary = {
        "role": "Weather Model Robustness / Anti-Overfitting Agent",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_name": candidate_name,
        "source_repo": str(source_repo),
        "target_start_date": start.isoformat(),
        "target_end_date": end.isoformat(),
        "research_sources": SOURCE_NOTES,
        "input_counts": {
            "market_rows": len(markets),
            "forecast_rows": len(forecasts),
            "realized_rows": len(realized),
            "validation_rows": len(rows),
            "purged_walk_forward_test_rows": len(primary_rows),
        },
        "decision_time_split_leakage_scan_summary": _summarize_leakage_scan(leakage_scan),
        "purged_target_month_metrics": aggregate,
        "purged_target_month_pass_rate": _pass_rate(primary_results),
        "purged_embargo_3_day_pass_rate": _pass_rate(embargo_3_results),
        "purged_embargo_7_day_pass_rate": _pass_rate(embargo_7_results),
        "purged_city_holdout_summary": _summarize_collection(city_holdouts),
        "purged_region_holdout_summary": _summarize_collection(region_holdouts),
        "bootstrap_confidence_intervals": bootstrap,
        "permutation_sanity_check": permutation,
        "market_diagnostics": market["summary"],
        "decision": decision,
        "output_artifacts": artifacts,
        "no_live_or_paper_defaults_changed": True,
    }
    Path(artifacts["summary_json"]).write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_markdown(Path(artifacts["report_md"]), summary)
    return summary


def _run_purged_target_months(
    rows: list[ValidationRow],
    *,
    candidate_name: str,
    embargo_days: int,
    min_train_rows: int,
    min_test_rows: int,
) -> tuple[list[dict[str, Any]], list[ValidationRow], dict[str, float], dict[str, float]]:
    results = []
    aggregate_rows: list[ValidationRow] = []
    baseline_probs: dict[str, float] = {}
    candidate_probs: dict[str, float] = {}
    for month in sorted({row.market_date_local.strftime("%Y-%m") for row in rows}):
        month_start = date.fromisoformat(f"{month}-01")
        month_end = _next_month_date(month_start)
        test_rows = [row for row in rows if month_start <= row.market_date_local < month_end]
        if len(test_rows) < min_test_rows:
            continue
        first_test_decision = min(row.decision_time_utc for row in test_rows).date()
        label_cutoff = first_test_decision - timedelta(days=embargo_days)
        train_rows = [row for row in rows if row.market_date_local < label_cutoff]
        if len(train_rows) < min_train_rows:
            continue
        result = _evaluate_split(
            f"target_month_{month}_embargo_{embargo_days}d",
            "purged_target_month",
            train_rows,
            test_rows,
            candidate_name=candidate_name,
        )
        summary = dict(result["summary"])
        summary.update(
            {
                "target_month": month,
                "embargo_days": embargo_days,
                "label_cutoff_local_date_exclusive": label_cutoff.isoformat(),
                "test_target_start": month_start.isoformat(),
                "test_target_end_exclusive": month_end.isoformat(),
                "train_test_city_date_overlap": _overlap_count(train_rows, test_rows, lambda row: f"{row.normalized_city}|{row.market_date_local.isoformat()}"),
                "train_test_universe_overlap": _overlap_count(train_rows, test_rows, lambda row: row.universe_row_id),
                "train_rows_with_target_on_or_after_first_test_decision_date": sum(1 for row in train_rows if row.market_date_local >= first_test_decision),
            }
        )
        results.append(summary)
        aggregate_rows.extend(test_rows)
        baseline_probs.update(result["baseline_probabilities"])
        candidate_probs.update(result["candidate_probabilities"])
    return results, aggregate_rows, baseline_probs, candidate_probs


def _decision_time_split_leakage_scan(
    rows: list[ValidationRow],
    *,
    min_train_rows: int,
    min_test_rows: int,
) -> list[dict[str, Any]]:
    output = []
    for month in sorted({row.decision_time_utc.strftime("%Y-%m") for row in rows}):
        month_start_dt = datetime(int(month[:4]), int(month[5:]), 1, tzinfo=timezone.utc)
        month_end_dt = datetime(month_start_dt.year + (month_start_dt.month // 12), (month_start_dt.month % 12) + 1, 1, tzinfo=timezone.utc)
        train_rows = [row for row in rows if row.decision_time_utc < month_start_dt]
        test_rows = [row for row in rows if month_start_dt <= row.decision_time_utc < month_end_dt]
        if len(train_rows) < min_train_rows or len(test_rows) < min_test_rows:
            continue
        first_test_decision_date = min(row.decision_time_utc for row in test_rows).date()
        future_label_train_rows = [row for row in train_rows if row.market_date_local >= first_test_decision_date]
        output.append(
            {
                "decision_month": month,
                "train_n": len(train_rows),
                "test_n": len(test_rows),
                "first_test_decision_date": first_test_decision_date.isoformat(),
                "future_or_unresolved_label_train_rows": len(future_label_train_rows),
                "future_or_unresolved_label_train_fraction": round(len(future_label_train_rows) / len(train_rows), 6) if train_rows else None,
                "train_test_city_date_overlap": _overlap_count(train_rows, test_rows, lambda row: f"{row.normalized_city}|{row.market_date_local.isoformat()}"),
                "train_test_universe_overlap": _overlap_count(train_rows, test_rows, lambda row: row.universe_row_id),
            }
        )
    return output


def _purged_group_holdouts(
    rows: list[ValidationRow],
    *,
    candidate_name: str,
    key_fn: Any,
    group_type: str,
    min_train_rows: int,
    min_test_rows: int,
) -> list[dict[str, Any]]:
    cutoff = date(2026, 4, 1)
    output = []
    for group in sorted({str(key_fn(row)) for row in rows}):
        test_rows = [row for row in rows if row.market_date_local >= cutoff and str(key_fn(row)) == group]
        train_rows = [row for row in rows if row.market_date_local < cutoff and str(key_fn(row)) != group]
        if len(train_rows) < min_train_rows or len(test_rows) < min_test_rows:
            continue
        result = _evaluate_split(
            f"{group_type}:{group}",
            group_type,
            train_rows,
            test_rows,
            candidate_name=candidate_name,
        )["summary"]
        result["heldout_group"] = group
        result["target_date_cutoff"] = cutoff.isoformat()
        result["train_test_city_date_overlap"] = _overlap_count(train_rows, test_rows, lambda row: f"{row.normalized_city}|{row.market_date_local.isoformat()}")
        output.append(result)
    output.sort(key=lambda row: (not bool(row["beats_baseline_both"]), float(row["brier_improvement"])))
    return output


def _decision(
    *,
    aggregate: Mapping[str, Any],
    bootstrap: Mapping[str, Any],
    permutation: Mapping[str, Any],
    purged_months: list[Mapping[str, Any]],
    city_holdouts: list[Mapping[str, Any]],
    region_holdouts: list[Mapping[str, Any]],
    market: Mapping[str, Any],
    leakage_scan: list[Mapping[str, Any]],
) -> dict[str, Any]:
    city_date_ci = bootstrap.get("city_date_cluster") or {}
    ci_pass = (
        float(city_date_ci.get("brier_improvement_ci_2_5") or -1.0) > 0.0
        and float(city_date_ci.get("log_loss_improvement_ci_2_5") or -1.0) > 0.0
    )
    weather_pass = (
        float(aggregate["brier_improvement"]) > 0.0
        and float(aggregate["log_loss_improvement"]) > 0.0
        and ci_pass
        and bool(permutation.get("observed_brier_above_null_95"))
        and bool(permutation.get("observed_log_loss_above_null_95"))
    )
    market_summary = market.get("summary") or {}
    cand = market_summary.get("candidate_vs_polymarket_payout") or {}
    mid = market_summary.get("market_mid_vs_polymarket_payout") or {}
    market_pass = (
        cand.get("brier") is not None
        and mid.get("brier") is not None
        and float(cand["brier"]) < float(mid["brier"])
        and float(cand["log_loss"]) < float(mid["log_loss"])
    )
    leakage_detected = any(int(row.get("future_or_unresolved_label_train_rows") or 0) > 0 for row in leakage_scan)
    if not weather_pass:
        outcome = "previous_gain_likely_overfit_or_label_leaked_reject_current_residual_model"
    elif not market_pass:
        outcome = "weather_gain_survives_purged_label_available_validation_but_does_not_beat_market"
    else:
        outcome = "survives_sota_validation_can_move_to_quote_backed_replay"
    return {
        "outcome": outcome,
        "weather_pass": weather_pass,
        "market_pass": market_pass,
        "label_availability_leakage_detected_in_old_decision_time_splits": leakage_detected,
        "purged_target_month_pass_rate": _pass_rate(purged_months),
        "purged_city_holdout_pass_rate": _pass_rate(city_holdouts),
        "purged_region_holdout_pass_rate": _pass_rate(region_holdouts),
        "model_weights_or_strategy_defaults_changed": False,
    }


def _write_markdown(path: Path, summary: Mapping[str, Any]) -> None:
    aggregate = summary["purged_target_month_metrics"]
    decision = summary["decision"]
    market = summary["market_diagnostics"]
    leakage = summary["decision_time_split_leakage_scan_summary"]
    bootstrap = summary["bootstrap_confidence_intervals"]["city_date_cluster"]
    lines = [
        "# DailyWeather SOTA Validation And Leakage Audit",
        "",
        f"Generated: {summary['generated_at_utc']}",
        "",
        f"Candidate: `{summary['candidate_name']}`",
        "",
        "## Decision",
        "",
        f"Outcome: `{decision['outcome']}`",
        "",
        "- The older decision-time rolling splits are contaminated by label-availability leakage.",
        "- The purged target-date splits remove same/adjacent unresolved target labels from training.",
        "- No model weights, paper defaults, live defaults, or strategy settings were changed.",
        "",
        "## Literature-Inspired Tests",
        "",
        "- Purged / embargoed target-date splits for overlapping label horizons.",
        "- Rolling-origin validation by target month, not just decision timestamp.",
        "- Spatial/city/region holdouts for autocorrelation and cold-start risk.",
        "- Proper binary scoring: Brier score and log loss, plus calibration/stress slices.",
        "- Cluster bootstrap and label-shuffle sanity checks.",
        "- Point-in-time market midpoint and bid/ask execution diagnostics.",
        "",
        "## Old Split Leakage Scan",
        "",
        f"- Decision-month splits scanned: {leakage.get('n')}",
        f"- Splits with future/unresolved labels in train: {leakage.get('splits_with_future_labels')}",
        f"- Max train future-label fraction: {leakage.get('max_future_label_fraction')}",
        f"- Max train/test city-date overlap: {leakage.get('max_city_date_overlap')}",
        f"- Max train/test universe overlap: {leakage.get('max_universe_overlap')}",
        "",
        "## Purged Target-Date Walk-Forward Metrics",
        "",
        f"- Test rows: {summary['input_counts']['purged_walk_forward_test_rows']:,}",
        f"- Baseline Brier/log loss: {aggregate['baseline']['brier']} / {aggregate['baseline']['log_loss']}",
        f"- Candidate Brier/log loss: {aggregate['candidate']['brier']} / {aggregate['candidate']['log_loss']}",
        f"- Brier improvement: {aggregate['brier_improvement']}",
        f"- Log-loss improvement: {aggregate['log_loss_improvement']}",
        f"- Purged target-month pass rate: {summary['purged_target_month_pass_rate']}",
        f"- 3-day embargo pass rate: {summary['purged_embargo_3_day_pass_rate']}",
        f"- 7-day embargo pass rate: {summary['purged_embargo_7_day_pass_rate']}",
        f"- City-date cluster Brier CI: {bootstrap.get('brier_improvement_ci_2_5')} to {bootstrap.get('brier_improvement_ci_97_5')}",
        f"- City-date cluster log-loss CI: {bootstrap.get('log_loss_improvement_ci_2_5')} to {bootstrap.get('log_loss_improvement_ci_97_5')}",
        f"- Label-shuffle pass: {summary['permutation_sanity_check'].get('observed_brier_above_null_95')} / {summary['permutation_sanity_check'].get('observed_log_loss_above_null_95')}",
        "",
        "## Spatial Holdouts",
        "",
        f"- Purged city holdout pass rate: {decision.get('purged_city_holdout_pass_rate')}",
        f"- Purged region holdout pass rate: {decision.get('purged_region_holdout_pass_rate')}",
        "",
        "## Market Diagnostics",
        "",
        f"- Matched quote side rows: {market.get('matched_walk_forward_quote_side_rows')}",
        f"- Candidate vs Polymarket payout Brier/log loss: {(market.get('candidate_vs_polymarket_payout') or {}).get('brier')} / {(market.get('candidate_vs_polymarket_payout') or {}).get('log_loss')}",
        f"- Market midpoint Brier/log loss: {(market.get('market_mid_vs_polymarket_payout') or {}).get('brier')} / {(market.get('market_mid_vs_polymarket_payout') or {}).get('log_loss')}",
        f"- Weather/Polymarket mismatch rate: {market.get('weather_label_mismatch_rate')}",
        "",
        "## Sources Consulted",
        "",
    ]
    for item in summary["research_sources"]:
        lines.append(f"- {item['topic']}: {item['url']}")
    lines.extend(["", "## Artifacts", ""])
    for key, value in summary["output_artifacts"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _summarize_leakage_scan(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0}
    return {
        "n": len(rows),
        "splits_with_future_labels": sum(1 for row in rows if int(row.get("future_or_unresolved_label_train_rows") or 0) > 0),
        "max_future_label_fraction": max(float(row.get("future_or_unresolved_label_train_fraction") or 0.0) for row in rows),
        "max_city_date_overlap": max(int(row.get("train_test_city_date_overlap") or 0) for row in rows),
        "max_universe_overlap": max(int(row.get("train_test_universe_overlap") or 0) for row in rows),
    }


def _summarize_collection(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0, "pass_rate": None}
    return {
        "n": len(rows),
        "pass_rate": _pass_rate(rows),
        "worst_brier_improvement": min(float(row["brier_improvement"]) for row in rows),
        "worst_log_loss_improvement": min(float(row["log_loss_improvement"]) for row in rows),
    }


def _flatten_bootstrap(bootstrap: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [{"cluster": cluster, **dict(values)} for cluster, values in bootstrap.items()]


def _overlap_count(left: list[ValidationRow], right: list[ValidationRow], key_fn: Any) -> int:
    left_keys = {str(key_fn(row)) for row in left}
    right_keys = {str(key_fn(row)) for row in right}
    return len(left_keys & right_keys)


def _next_month_date(value: date) -> date:
    if value.month == 12:
        return date(value.year + 1, 1, 1)
    return date(value.year, value.month + 1, 1)


if __name__ == "__main__":
    sys.exit(main())
