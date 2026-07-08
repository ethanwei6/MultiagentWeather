from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

from .weather_model_robustness import DEFAULT_CANDIDATE
from .weather_model_validation import (
    DEFAULT_END_DATE,
    DEFAULT_SOURCE_REPO,
    DEFAULT_START_DATE,
    ValidationRow,
    _artifact_paths,
    _build_validation_rows,
    _connect_readonly,
    _fit_residual_distribution_candidates,
    _load_forecasts,
    _load_markets,
    _load_realized,
    _load_weather_strategy,
    _write_csv,
)


LIVE_FEATURE_FIELDS = (
    "universe_row_id",
    "market_id",
    "normalized_city",
    "country_region",
    "timezone",
    "market_date_local",
    "bucket_label",
    "bucket_type",
    "threshold_low_f",
    "threshold_high_f",
    "side",
    "decision_time_utc",
    "decision_hour_utc",
    "lead_days",
    "forecast_run_time_utc",
    "forecast_source_name",
    "forecast_temperature_2m_max_f",
    "forecast_features_json",
    "observed_high_so_far_f",
    "observed_high_observed_at_utc",
    "observed_high_missing_reason",
    "quote_timestamp_utc",
    "bid",
    "ask",
    "mid",
    "spread",
    "bid_size",
    "ask_size",
    "liquidity_depth",
)

EVALUATION_ONLY_FIELDS = (
    "realized_temperature_2m_max_f",
    "resolved_outcome",
    "canonical_label",
    "canonical_label_source",
    "canonical_bucket_type",
    "open_meteo_current_continuous_label",
    "open_meteo_source_rule_proxy_label",
    "recommended_action",
    "exclude_from_backtest_replay",
    "exclude_from_open_meteo_training",
)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Audit whether DailyWeather weather models can run with live-available data only")
    parser.add_argument("--work-root", default=".")
    parser.add_argument("--source-repo", default=str(DEFAULT_SOURCE_REPO))
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=DEFAULT_END_DATE)
    parser.add_argument("--output-dir", default="work/reports/weather_model_live_readiness")
    parser.add_argument("--candidate-name", default=DEFAULT_CANDIDATE)
    parser.add_argument("--min-train-rows", type=int, default=12000)
    parser.add_argument("--min-test-rows", type=int, default=4000)
    args = parser.parse_args(argv)

    summary = run_live_readiness_audit(
        work_root=Path(args.work_root).resolve(),
        source_repo=Path(args.source_repo).expanduser().resolve(),
        start=date.fromisoformat(args.start_date),
        end=date.fromisoformat(args.end_date),
        output_dir=Path(args.output_dir),
        candidate_name=args.candidate_name,
        min_train_rows=args.min_train_rows,
        min_test_rows=args.min_test_rows,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def run_live_readiness_audit(
    *,
    work_root: Path,
    source_repo: Path,
    start: date,
    end: date,
    output_dir: Path,
    candidate_name: str,
    min_train_rows: int,
    min_test_rows: int,
) -> dict[str, Any]:
    strategy = _load_weather_strategy(source_repo)
    paths = _artifact_paths(work_root)
    markets = _load_markets(paths["market_db"], start=start, end=end)
    forecasts = _load_forecasts(paths["weather_db"], start=start, end=end)
    realized = _load_realized(paths["weather_db"], start=start, end=end)
    rows = _build_validation_rows(strategy, markets, forecasts, realized)
    if len(rows) < 100:
        raise ValueError(f"Need at least 100 validation rows; got {len(rows)}")

    feature_gate = _audit_decision_artifact_live_features(paths["decision_db"], start=start, end=end)
    invariance_rows = _run_label_poisoning_invariance(rows, candidate_name=candidate_name, min_train_rows=min_train_rows, min_test_rows=min_test_rows)
    train_windows = _target_month_training_windows(rows, min_train_rows=min_train_rows, min_test_rows=min_test_rows)
    decision = _decision(feature_gate=feature_gate["summary"], invariance_rows=invariance_rows)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_dir = (work_root / output_dir).resolve() if not output_dir.is_absolute() else output_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{timestamp}-weather-model-live-readiness"
    artifacts = {
        "summary_json": str(report_dir / f"{prefix}.json"),
        "report_md": str(report_dir / f"{prefix}.md"),
        "live_feature_coverage_csv": str(report_dir / f"{prefix}-live-feature-coverage.csv"),
        "live_feature_slices_csv": str(report_dir / f"{prefix}-live-feature-slices.csv"),
        "label_poisoning_invariance_csv": str(report_dir / f"{prefix}-label-poisoning-invariance.csv"),
        "target_month_training_windows_csv": str(report_dir / f"{prefix}-target-month-training-windows.csv"),
    }

    _write_csv(Path(artifacts["live_feature_coverage_csv"]), [feature_gate["summary"]])
    _write_csv(Path(artifacts["live_feature_slices_csv"]), feature_gate["slices"])
    _write_csv(Path(artifacts["label_poisoning_invariance_csv"]), invariance_rows)
    _write_csv(Path(artifacts["target_month_training_windows_csv"]), train_windows)

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
        },
        "live_feature_contract": {
            "allowed_live_feature_fields": LIVE_FEATURE_FIELDS,
            "evaluation_only_fields": EVALUATION_ONLY_FIELDS,
            "rule": "Live scoring rows must be buildable without realized highs, resolved outcomes, canonical labels, or future observations.",
        },
        "decision_artifact_live_feature_summary": feature_gate["summary"],
        "label_poisoning_invariance_summary": _summarize_invariance(invariance_rows),
        "target_month_training_windows_summary": _summarize_training_windows(train_windows),
        "decision": decision,
        "output_artifacts": artifacts,
        "no_live_or_paper_defaults_changed": True,
    }
    Path(artifacts["summary_json"]).write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_markdown(Path(artifacts["report_md"]), summary)
    return summary


def _audit_decision_artifact_live_features(path: Path, *, start: date, end: date) -> dict[str, Any]:
    with _connect_readonly(path) as conn:
        summary = dict(
            conn.execute(
                """
                SELECT
                  COUNT(*) AS decision_rows,
                  SUM(CASE WHEN forecast_run_time_utc IS NOT NULL THEN 1 ELSE 0 END) AS rows_with_forecast_snapshot,
                  SUM(CASE WHEN quote_timestamp_utc IS NOT NULL THEN 1 ELSE 0 END) AS rows_with_quote_snapshot,
                  SUM(CASE WHEN observed_high_so_far_f IS NOT NULL THEN 1 ELSE 0 END) AS rows_with_observed_high_so_far,
                  SUM(CASE WHEN realized_temperature_2m_max_f IS NOT NULL THEN 1 ELSE 0 END) AS rows_with_realized_label_evaluation_only,
                  SUM(CASE WHEN canonical_label IS NOT NULL THEN 1 ELSE 0 END) AS rows_with_canonical_label_evaluation_only,
                  SUM(CASE WHEN forecast_run_time_utc > decision_time_utc THEN 1 ELSE 0 END) AS future_forecast_violations,
                  SUM(CASE WHEN quote_timestamp_utc > decision_time_utc THEN 1 ELSE 0 END) AS future_quote_violations,
                  SUM(CASE WHEN observed_high_observed_at_utc > decision_time_utc THEN 1 ELSE 0 END) AS future_observed_high_violations,
                  SUM(CASE
                        WHEN observed_high_data_quality_flags LIKE '%final_daily_aggregate_used%'
                          OR observed_high_data_quality_flags LIKE '%derived_from_final_daily_aggregate%'
                        THEN 1 ELSE 0
                      END) AS final_daily_aggregate_leak_violations,
                  SUM(CASE WHEN forecast_run_time_utc IS NOT NULL
                             AND forecast_run_time_utc <= decision_time_utc
                           THEN 1 ELSE 0 END) AS weather_model_scoreable_rows
                FROM decision_snapshots_current_enriched
                WHERE market_date_local >= ? AND market_date_local <= ?
                """,
                (start.isoformat(), end.isoformat()),
            ).fetchone()
        )
        slices = []
        slices.extend(_coverage_slice(conn, "decision_hour_utc", "decision_hour_utc", start, end))
        slices.extend(_coverage_slice(conn, "lead_days", "lead_days", start, end))
        slices.extend(_coverage_slice(conn, "bucket_type", "bucket_type", start, end))
        slices.extend(_coverage_slice(conn, "source_name", "COALESCE(forecast_source_name, 'missing')", start, end))
        slices.extend(_coverage_slice(conn, "observed_high_missing_reason", "COALESCE(observed_high_missing_reason, 'available')", start, end))

    decision_rows = int(summary.get("decision_rows") or 0)
    scoreable = int(summary.get("weather_model_scoreable_rows") or 0)
    summary["weather_model_scoreable_fraction"] = round(scoreable / decision_rows, 6) if decision_rows else None
    summary["quote_snapshot_fraction"] = round(int(summary.get("rows_with_quote_snapshot") or 0) / decision_rows, 6) if decision_rows else None
    summary["observed_high_fraction"] = round(int(summary.get("rows_with_observed_high_so_far") or 0) / decision_rows, 6) if decision_rows else None
    summary["live_timing_violation_rows"] = (
        int(summary.get("future_forecast_violations") or 0)
        + int(summary.get("future_quote_violations") or 0)
        + int(summary.get("future_observed_high_violations") or 0)
        + int(summary.get("final_daily_aggregate_leak_violations") or 0)
    )
    summary["evaluation_only_fields_present_note"] = (
        "Realized and canonical settlement labels are present in the enriched artifact for scoring/evaluation, "
        "but must be stripped from live feature rows."
    )
    return {"summary": summary, "slices": slices}


def _coverage_slice(conn: Any, dimension: str, expression: str, start: date, end: date) -> list[dict[str, Any]]:
    rows = conn.execute(
        f"""
        SELECT
          {expression} AS group_value,
          COUNT(*) AS decision_rows,
          SUM(CASE WHEN forecast_run_time_utc IS NOT NULL THEN 1 ELSE 0 END) AS rows_with_forecast_snapshot,
          SUM(CASE WHEN quote_timestamp_utc IS NOT NULL THEN 1 ELSE 0 END) AS rows_with_quote_snapshot,
          SUM(CASE WHEN observed_high_so_far_f IS NOT NULL THEN 1 ELSE 0 END) AS rows_with_observed_high_so_far,
          SUM(CASE WHEN forecast_run_time_utc > decision_time_utc THEN 1 ELSE 0 END) AS future_forecast_violations,
          SUM(CASE WHEN quote_timestamp_utc > decision_time_utc THEN 1 ELSE 0 END) AS future_quote_violations,
          SUM(CASE WHEN observed_high_observed_at_utc > decision_time_utc THEN 1 ELSE 0 END) AS future_observed_high_violations
        FROM decision_snapshots_current_enriched
        WHERE market_date_local >= ? AND market_date_local <= ?
        GROUP BY group_value
        ORDER BY group_value
        """,
        (start.isoformat(), end.isoformat()),
    ).fetchall()
    output = []
    for row in rows:
        item = dict(row)
        item["dimension"] = dimension
        item["group"] = str(item.pop("group_value"))
        total = int(item.get("decision_rows") or 0)
        item["forecast_snapshot_fraction"] = round(int(item.get("rows_with_forecast_snapshot") or 0) / total, 6) if total else None
        item["quote_snapshot_fraction"] = round(int(item.get("rows_with_quote_snapshot") or 0) / total, 6) if total else None
        item["observed_high_fraction"] = round(int(item.get("rows_with_observed_high_so_far") or 0) / total, 6) if total else None
        output.append(item)
    return output


def _run_label_poisoning_invariance(
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
        original_probs = _candidate_probabilities(train_rows, test_rows, candidate_name=candidate_name)
        poisoned_rows = [_poison_evaluation_fields(row) for row in test_rows]
        poisoned_probs = _candidate_probabilities(train_rows, poisoned_rows, candidate_name=candidate_name)
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


def _candidate_probabilities(
    train_rows: list[ValidationRow],
    score_rows: list[ValidationRow],
    *,
    candidate_name: str,
) -> dict[str, float]:
    fit_rows = sorted([*train_rows, *score_rows], key=lambda row: (row.decision_time_utc, row.row_id))
    candidates = _fit_residual_distribution_candidates(rows=fit_rows, train_rows=train_rows, test_rows=score_rows)
    candidate = next((item for item in candidates if item["name"] == candidate_name), None)
    if candidate is None:
        names = ", ".join(item["name"] for item in candidates)
        raise ValueError(f"Candidate {candidate_name!r} not found. Available: {names}")
    probabilities = candidate["probabilities"]
    return {row.row_id: float(probabilities[row.row_id]) for row in score_rows}


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
        realized_quality_flags=("poisoned_for_live_readiness_audit",),
    )


def _target_month_training_windows(
    rows: list[ValidationRow],
    *,
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
        output.append(
            {
                "target_month": month,
                "train_n": len(train_rows),
                "test_n": len(test_rows),
                "first_test_decision_date": first_test_decision.isoformat(),
                "max_train_target_date": max(row.market_date_local for row in train_rows).isoformat(),
                "min_test_target_date": min(row.market_date_local for row in test_rows).isoformat(),
                "train_rows_with_unavailable_labels": sum(1 for row in train_rows if row.market_date_local >= first_test_decision),
                "train_test_city_date_overlap": _overlap_count(train_rows, test_rows, lambda row: f"{row.normalized_city}|{row.market_date_local.isoformat()}"),
                "train_test_universe_overlap": _overlap_count(train_rows, test_rows, lambda row: row.universe_row_id),
            }
        )
    return output


def _decision(*, feature_gate: Mapping[str, Any], invariance_rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    timing_pass = int(feature_gate.get("live_timing_violation_rows") or 0) == 0
    scoreable_fraction = float(feature_gate.get("weather_model_scoreable_fraction") or 0.0)
    scoreable_pass = scoreable_fraction >= 0.95
    invariance_pass = bool(invariance_rows) and all(bool(row.get("live_inference_invariant")) for row in invariance_rows)
    if not timing_pass or not invariance_pass:
        outcome = "reject_not_live_safe"
    elif not scoreable_pass:
        outcome = "live_safe_but_weather_forecast_coverage_incomplete"
    else:
        outcome = "live_feature_gate_passes_for_weather_inference_research_only"
    return {
        "outcome": outcome,
        "timing_pass": timing_pass,
        "weather_model_scoreable_pass": scoreable_pass,
        "label_poisoning_invariance_pass": invariance_pass,
        "market_promotion_pass": False,
        "model_weights_or_strategy_defaults_changed": False,
        "note": "This gate checks live feature safety. It does not override the failed market-midpoint/replay promotion gate.",
    }


def _summarize_invariance(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0, "pass_rate": None}
    return {
        "n": len(rows),
        "pass_rate": sum(1 for row in rows if bool(row.get("live_inference_invariant"))) / len(rows),
        "total_changed_probability_rows": sum(int(row.get("changed_probability_rows") or 0) for row in rows),
        "max_abs_probability_delta": max(float(row.get("max_abs_probability_delta") or 0.0) for row in rows),
    }


def _summarize_training_windows(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0}
    return {
        "n": len(rows),
        "total_train_rows_with_unavailable_labels": sum(int(row.get("train_rows_with_unavailable_labels") or 0) for row in rows),
        "max_train_test_city_date_overlap": max(int(row.get("train_test_city_date_overlap") or 0) for row in rows),
        "max_train_test_universe_overlap": max(int(row.get("train_test_universe_overlap") or 0) for row in rows),
    }


def _write_markdown(path: Path, summary: Mapping[str, Any]) -> None:
    feature = summary["decision_artifact_live_feature_summary"]
    invariance = summary["label_poisoning_invariance_summary"]
    decision = summary["decision"]
    lines = [
        "# DailyWeather Weather Model Live-Readiness Audit",
        "",
        f"Generated: {summary['generated_at_utc']}",
        "",
        f"Candidate: `{summary['candidate_name']}`",
        "",
        "## Decision",
        "",
        f"Outcome: `{decision['outcome']}`",
        "",
        "- Live scoring must use only point-in-time forecast, market, quote, and observed-high fields.",
        "- Realized highs, resolved outcomes, canonical labels, and settlement audit fields are evaluation-only.",
        "- No model weights, paper defaults, live defaults, or strategy settings were changed.",
        "",
        "## Live Feature Gate",
        "",
        f"- Decision rows: {feature.get('decision_rows')}",
        f"- Weather-model scoreable rows: {feature.get('weather_model_scoreable_rows')} ({feature.get('weather_model_scoreable_fraction')})",
        f"- Quote snapshot rows: {feature.get('rows_with_quote_snapshot')} ({feature.get('quote_snapshot_fraction')})",
        f"- Observed-high-so-far rows: {feature.get('rows_with_observed_high_so_far')} ({feature.get('observed_high_fraction')})",
        f"- Future forecast violations: {feature.get('future_forecast_violations')}",
        f"- Future quote violations: {feature.get('future_quote_violations')}",
        f"- Future observed-high violations: {feature.get('future_observed_high_violations')}",
        f"- Final daily aggregate leak violations: {feature.get('final_daily_aggregate_leak_violations')}",
        "",
        "## Label Poisoning Test",
        "",
        f"- Target-month splits tested: {invariance.get('n')}",
        f"- Invariance pass rate: {invariance.get('pass_rate')}",
        f"- Changed probability rows after poisoning labels/final highs: {invariance.get('total_changed_probability_rows')}",
        f"- Max absolute probability delta: {invariance.get('max_abs_probability_delta')}",
        "",
        "## Promotion Status",
        "",
        "- Weather inference live-feature gate passes only for scoreable rows with point-in-time forecast snapshots.",
        "- The model remains research-only because the market-midpoint and quote-backed replay gates have not passed.",
        "",
        "## Artifacts",
        "",
    ]
    for key, value in summary["output_artifacts"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


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
