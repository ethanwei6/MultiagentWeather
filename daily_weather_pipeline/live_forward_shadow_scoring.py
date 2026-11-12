from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

from .backtesting_engine import (
    DecisionSnapshot,
    ObservedHighSnapshot,
    ReplayGeneratedScoreCache,
    SharedScoreContract,
    WeatherModelDecisionContext,
    WeatherModelPrediction,
    WeatherModelRunner,
    validate_weather_model_context_firewall,
)
from .weather_model_runner import ObservedHighGuardedShrinkageWeatherModelRunner
from .weather_model_validation import resolve_training_corpus_end_date


RUNNER_SCORE_FAMILY = "replay_generated_weather_model_runner"
RUNNER_NAME = "ObservedHighGuardedShrinkageWeatherModelRunner"
LEGACY_LIVE_OPEN_METEO_SCORE_FAMILY = "old_no_tail_shadow_live_forecast_adapter"


@dataclass(frozen=True)
class LiveForwardShadowScoreResult:
    score: Optional[SharedScoreContract]
    context: Optional[WeatherModelDecisionContext]
    prediction: Optional[WeatherModelPrediction]
    firewall_violation: Optional[str] = None
    missing_model_output: bool = False


class LiveForwardShadowScoringAdapter:
    """Replay-parity scorer for live-forward shadow decisions."""

    def __init__(
        self,
        *,
        model_runner: WeatherModelRunner,
        score_cache: ReplayGeneratedScoreCache,
        runner_metadata_overrides: Optional[Mapping[str, str]] = None,
    ) -> None:
        self.model_runner = model_runner
        self.score_cache = score_cache
        self.runner_metadata_overrides = dict(runner_metadata_overrides or {})

    def runner_metadata(self) -> Mapping[str, str]:
        return {
            **dict(self.model_runner.metadata()),
            **self.runner_metadata_overrides,
        }

    def score_decision(
        self,
        decision: DecisionSnapshot,
        *,
        settlement_unit: Optional[str] = None,
        settlement_precision: Optional[str] = None,
    ) -> LiveForwardShadowScoreResult:
        try:
            context = WeatherModelDecisionContext.from_decision(decision)
            validate_weather_model_context_firewall(context)
            prediction = self.model_runner.predict(context)
        except ValueError as exc:
            return LiveForwardShadowScoreResult(
                score=None,
                context=None,
                prediction=None,
                firewall_violation=str(exc),
            )
        if prediction is None:
            return LiveForwardShadowScoreResult(
                score=None,
                context=context,
                prediction=None,
                missing_model_output=True,
            )
        score = prediction.to_shared_score(
            side=decision.side,
            observed_high=model_visible_observed_high(decision),
            parser_confidence=decision.parser_confidence,
            settlement_unit=settlement_unit,
            settlement_precision=settlement_precision,
        )
        score = _with_live_shadow_metadata(
            score,
            runner_metadata=self.runner_metadata(),
        )
        self.score_cache.add(context=context, prediction=prediction, score=score)
        return LiveForwardShadowScoreResult(
            score=score,
            context=context,
            prediction=prediction,
        )


def default_live_forward_shadow_scoring_adapter(
    *,
    work_root: Path,
    score_cache: ReplayGeneratedScoreCache,
    training_corpus_end_date: Optional[date | str] = None,
) -> LiveForwardShadowScoringAdapter:
    resolved_end = resolve_training_corpus_end_date(training_corpus_end_date)
    runner_kwargs: dict[str, Any] = {
        "work_root": Path(work_root),
        "end": resolved_end,
    }
    return LiveForwardShadowScoringAdapter(
        model_runner=ObservedHighGuardedShrinkageWeatherModelRunner(**runner_kwargs),
        score_cache=score_cache,
    )




def model_visible_observed_high(decision: DecisionSnapshot) -> Optional[ObservedHighSnapshot]:
    observed = decision.observed_high
    if observed is None:
        return None
    if observed.observed_high_timestamp_utc > decision.decision_time_utc:
        return None
    return observed


def _with_live_shadow_metadata(
    score: SharedScoreContract,
    *,
    runner_metadata: Mapping[str, Any],
) -> SharedScoreContract:
    return SharedScoreContract(
        p_yes=score.p_yes,
        side=score.side,
        p_side=score.p_side,
        model_agreement=score.model_agreement,
        model_count=score.model_count,
        probability_stdev=score.probability_stdev,
        per_source_probabilities=dict(score.per_source_probabilities),
        remaining_exceed_probability=score.remaining_exceed_probability,
        observed_high_aware_confidence=score.observed_high_aware_confidence,
        observed_high=score.observed_high,
        heat_window=score.heat_window,
        parser_confidence=score.parser_confidence,
        settlement_unit=score.settlement_unit,
        settlement_precision=score.settlement_precision,
        scored_at_utc=score.scored_at_utc,
        score_model_version=score.score_model_version,
        score_family=score.score_family,
        metadata={
            **dict(score.metadata),
            "live_forward_shadow_scoring_adapter": "ObservedHighGuardedShrinkageWeatherModelRunner",
            "gate_valid_scoring_path": True,
            "runner_metadata": dict(runner_metadata),
            "pre_fix_artifacts_excluded_from_gate": True,
        },
    )


def shadow_scoring_parity_report(
    *,
    events: list[Mapping[str, Any]],
    score_cache_summary: Mapping[str, Any],
    runner_metadata: Mapping[str, Any],
    firewall_violation_count: int = 0,
    missing_model_output_count: int = 0,
    generated_at_utc: Optional[datetime] = None,
) -> Mapping[str, Any]:
    score_family_counts: dict[str, int] = {}
    runner_metadata_count = 0
    model_hash_metadata_count = 0
    training_corpus_metadata = _training_corpus_metadata_from_events(events)
    for event in events:
        model_output = event.get("model_output") if isinstance(event, Mapping) else {}
        if not isinstance(model_output, Mapping) or model_output.get("status") != "logged":
            continue
        family = str(model_output.get("score_family") or "missing")
        score_family_counts[family] = score_family_counts.get(family, 0) + 1
        metadata = model_output.get("metadata") if isinstance(model_output.get("metadata"), Mapping) else {}
        if isinstance(metadata.get("runner_metadata"), Mapping):
            runner_metadata_count += 1
        if metadata.get("model_code_hash") and metadata.get("model_config_hash"):
            model_hash_metadata_count += 1

    logged_score_count = sum(score_family_counts.values())
    legacy_count = int(score_family_counts.get(LEGACY_LIVE_OPEN_METEO_SCORE_FAMILY, 0))
    runner_count = int(score_family_counts.get(RUNNER_SCORE_FAMILY, 0))
    cache_records = int(score_cache_summary.get("record_count") or 0)
    runner_name = str(runner_metadata.get("runner_name") or "")
    model_code_hashes = list(score_cache_summary.get("model_code_hashes") or [])
    model_config_hashes = list(score_cache_summary.get("model_config_hashes") or [])
    reasons: list[str] = []
    if runner_name != RUNNER_NAME:
        reasons.append("model_runner_is_not_observed_high_guarded")
    if logged_score_count != runner_count:
        reasons.append("not_all_logged_scores_use_runner_score_family")
    if legacy_count:
        reasons.append("legacy_live_open_meteo_adapter_scores_present")
    if logged_score_count and runner_metadata_count != logged_score_count:
        reasons.append("runner_metadata_missing_on_logged_scores")
    if logged_score_count and model_hash_metadata_count != logged_score_count:
        reasons.append("runner_model_hash_metadata_missing_on_logged_scores")
    if cache_records != logged_score_count:
        reasons.append("score_cache_record_count_does_not_match_logged_scores")
    if firewall_violation_count:
        reasons.append("weather_model_context_firewall_violations_present")
    if missing_model_output_count:
        reasons.append("missing_model_outputs_present")
    if not model_code_hashes or not model_config_hashes:
        reasons.append("runner_model_hashes_missing")
    warning_flags = list(training_corpus_metadata.get("warning_flags") or [])
    staleness = _training_corpus_staleness_warning(
        training_corpus_metadata.get("training_corpus_end_date"),
        generated_at_utc=generated_at_utc,
    )
    if staleness["training_corpus_staleness_warning"]:
        for flag in staleness["warning_flags"]:
            if flag not in warning_flags:
                warning_flags.append(flag)
    coverage_metadata_count = int(
        training_corpus_metadata.get("training_corpus_coverage_metadata_count") or 0
    )
    coverage_available_count = int(
        training_corpus_metadata.get("training_corpus_coverage_available_count") or 0
    )
    coverage_fresh_count = int(
        training_corpus_metadata.get("training_corpus_operationally_fresh_count") or 0
    )
    relevant_city_coverage_count = int(
        training_corpus_metadata.get("training_corpus_relevant_city_coverage_count") or 0
    )
    relevant_city_fresh_count = int(
        training_corpus_metadata.get("training_corpus_relevant_city_fresh_count") or 0
    )
    if logged_score_count and coverage_metadata_count != logged_score_count:
        reasons.append("training_corpus_coverage_metadata_missing_on_logged_scores")
    if logged_score_count and coverage_available_count != logged_score_count:
        reasons.append("training_corpus_coverage_unavailable")
    if logged_score_count and coverage_fresh_count != logged_score_count:
        reasons.append("training_corpus_coverage_stale_or_incomplete")
    if logged_score_count and relevant_city_coverage_count != logged_score_count:
        reasons.append("training_corpus_relevant_city_date_coverage_missing")
    if logged_score_count and relevant_city_fresh_count != logged_score_count:
        reasons.append("training_corpus_relevant_city_date_coverage_stale_or_incomplete")
    if "training_corpus_stale_from_score_metadata" in warning_flags:
        reasons.append("training_corpus_stale_from_score_metadata")
    if staleness["training_corpus_staleness_warning"]:
        reasons.append("training_corpus_stale_by_global_date")
    gate_valid = not reasons
    return {
        "gate_valid_for_shadow_deployment": gate_valid,
        "gate_validity_reason": "gate_valid" if gate_valid else ";".join(reasons),
        "scoring_path": (
            "WeatherModelDecisionContext.from_decision -> validate_weather_model_context_firewall -> "
            "ObservedHighGuardedShrinkageWeatherModelRunner.predict -> prediction.to_shared_score -> "
            "ReplayGeneratedScoreCache.add"
        ),
        "score_family_counts": dict(sorted(score_family_counts.items())),
        "legacy_live_open_meteo_adapter_score_count": legacy_count,
        "runner_metadata": dict(runner_metadata),
        "runner_model_versions": list(score_cache_summary.get("model_versions") or []),
        "runner_model_code_hashes": model_code_hashes,
        "runner_model_config_hashes": model_config_hashes,
        "score_cache": dict(score_cache_summary),
        "training_corpus_end_date": training_corpus_metadata.get("training_corpus_end_date"),
        "training_row_count": training_corpus_metadata.get("training_row_count"),
        "training_corpus_configured_end_date": training_corpus_metadata.get(
            "training_corpus_configured_end_date"
        ),
        "training_corpus_loaded_row_count": training_corpus_metadata.get(
            "training_corpus_loaded_row_count"
        ),
        "training_row_count_range": training_corpus_metadata.get("training_row_count_range"),
        "training_corpus_staleness_days": staleness["training_corpus_staleness_days"],
        "training_corpus_staleness_warning": staleness[
            "training_corpus_staleness_warning"
        ],
        "training_corpus_staleness_warning_threshold_days": staleness[
            "training_corpus_staleness_warning_threshold_days"
        ],
        "training_corpus_coverage_metadata_count": coverage_metadata_count,
        "training_corpus_coverage_available_count": coverage_available_count,
        "training_corpus_operationally_fresh_count": coverage_fresh_count,
        "training_corpus_relevant_city_coverage_count": relevant_city_coverage_count,
        "training_corpus_relevant_city_fresh_count": relevant_city_fresh_count,
        "training_corpus_relevant_coverage": list(
            training_corpus_metadata.get("training_corpus_relevant_coverage") or []
        ),
        "warning_flags": warning_flags,
        "firewall_violation_count": firewall_violation_count,
        "missing_model_output_count": missing_model_output_count,
        "scored_row_count": logged_score_count,
        "runner_backed_score_count": runner_count,
        "score_rows_with_runner_metadata": runner_metadata_count,
        "score_rows_with_model_hash_metadata": model_hash_metadata_count,
        "pre_fix_artifacts_excluded_from_gate": True,
    }


def _training_corpus_metadata_from_events(
    events: list[Mapping[str, Any]],
) -> Mapping[str, Any]:
    end_dates: list[str] = []
    configured_end_dates: list[str] = []
    row_counts: list[int] = []
    loaded_row_counts: list[int] = []
    warning_flags: list[str] = []
    coverage_metadata_count = 0
    coverage_available_count = 0
    coverage_fresh_count = 0
    relevant_city_coverage_count = 0
    relevant_city_fresh_count = 0
    relevant_coverage: list[dict[str, Any]] = []
    for event in events:
        model_output = event.get("model_output") if isinstance(event, Mapping) else {}
        if not isinstance(model_output, Mapping) or model_output.get("status") != "logged":
            continue
        metadata = model_output.get("metadata")
        if not isinstance(metadata, Mapping):
            continue
        end_date = metadata.get("training_corpus_end_date")
        if isinstance(end_date, str) and end_date:
            end_dates.append(end_date)
        configured_end = metadata.get("training_corpus_configured_end_date")
        if isinstance(configured_end, str) and configured_end:
            configured_end_dates.append(configured_end)
        row_count = _optional_int(metadata.get("training_row_count"))
        if row_count is None:
            row_count = _optional_int(metadata.get("train_rows"))
        if row_count is not None:
            row_counts.append(row_count)
        loaded_row_count = _optional_int(metadata.get("training_corpus_loaded_row_count"))
        if loaded_row_count is not None:
            loaded_row_counts.append(loaded_row_count)
        if metadata.get("training_corpus_staleness_warning"):
            warning_flags.append("training_corpus_stale_from_score_metadata")
        coverage_available = metadata.get("training_corpus_coverage_available")
        operationally_fresh = metadata.get("training_corpus_operationally_fresh")
        if isinstance(coverage_available, bool) and isinstance(operationally_fresh, bool):
            coverage_metadata_count += 1
        if coverage_available is True:
            coverage_available_count += 1
        if operationally_fresh is True:
            coverage_fresh_count += 1
        market = event.get("market_discovery")
        if not isinstance(market, Mapping):
            market = {}
        parsed_contract = event.get("parsed_contract")
        if not isinstance(parsed_contract, Mapping):
            parsed_contract = {}
        relevant_city = str(
            market.get("normalized_city")
            or parsed_contract.get("city")
            or ""
        )
        relevant_target_date = str(
            market.get("market_date_local")
            or parsed_contract.get("date")
            or ""
        )
        city_coverage = metadata.get("training_corpus_city_coverage")
        city_coverage_matches = (
            isinstance(city_coverage, Mapping)
            and bool(relevant_city)
            and str(city_coverage.get("normalized_city") or "") == relevant_city
        )
        if city_coverage_matches:
            relevant_city_coverage_count += 1
        city_coverage_fresh = city_coverage_matches and _city_coverage_is_fresh(
            city_coverage
        )
        if city_coverage_fresh:
            relevant_city_fresh_count += 1
        relevant_coverage.append(
            {
                "normalized_city": relevant_city or None,
                "market_target_date": relevant_target_date or None,
                "coverage_metadata_available": (
                    isinstance(coverage_available, bool)
                    and isinstance(operationally_fresh, bool)
                ),
                "coverage_available": coverage_available,
                "operationally_fresh": operationally_fresh,
                "relevant_city_coverage_present": city_coverage_matches,
                "relevant_city_date_coverage_fresh": city_coverage_fresh,
                "city_coverage": dict(city_coverage) if isinstance(city_coverage, Mapping) else None,
            }
        )
    return {
        "training_corpus_end_date": max(end_dates) if end_dates else None,
        "training_corpus_configured_end_date": (
            max(configured_end_dates) if configured_end_dates else None
        ),
        "training_row_count": max(row_counts) if row_counts else None,
        "training_corpus_loaded_row_count": (
            max(loaded_row_counts) if loaded_row_counts else None
        ),
        "training_row_count_range": (
            {"min": min(row_counts), "max": max(row_counts)} if row_counts else None
        ),
        "warning_flags": tuple(sorted(set(warning_flags))),
        "training_corpus_coverage_metadata_count": coverage_metadata_count,
        "training_corpus_coverage_available_count": coverage_available_count,
        "training_corpus_operationally_fresh_count": coverage_fresh_count,
        "training_corpus_relevant_city_coverage_count": relevant_city_coverage_count,
        "training_corpus_relevant_city_fresh_count": relevant_city_fresh_count,
        "training_corpus_relevant_coverage": relevant_coverage,
    }


def _city_coverage_is_fresh(city_coverage: Mapping[str, Any]) -> bool:
    required_dates = [str(item) for item in city_coverage.get("required_target_dates") or []]
    incomplete_dates = [
        str(item) for item in city_coverage.get("incomplete_target_dates") or []
    ]
    if not bool(city_coverage.get("operationally_fresh")):
        return False
    if not required_dates or incomplete_dates:
        return False
    if str(city_coverage.get("latest_required_target_date") or "") != str(
        city_coverage.get("latest_complete_target_date") or ""
    ):
        return False
    source_coverage = city_coverage.get("source_coverage")
    if not isinstance(source_coverage, Mapping):
        return False
    for source_kind in ("forecast", "realized"):
        item = source_coverage.get(source_kind)
        if not isinstance(item, Mapping) or list(item.get("missing_target_dates") or []):
            return False
    return True


def _training_corpus_staleness_warning(
    training_corpus_end_date: Any,
    *,
    generated_at_utc: Optional[datetime],
    threshold_days: int = 3,
) -> Mapping[str, Any]:
    generated_at = generated_at_utc or datetime.now(timezone.utc)
    if generated_at.tzinfo is None:
        generated_at = generated_at.replace(tzinfo=timezone.utc)
    else:
        generated_at = generated_at.astimezone(timezone.utc)
    try:
        corpus_end = date.fromisoformat(str(training_corpus_end_date))
    except (TypeError, ValueError):
        return {
            "training_corpus_staleness_days": None,
            "training_corpus_staleness_warning": True,
            "training_corpus_staleness_warning_threshold_days": int(threshold_days),
            "warning_flags": ("training_corpus_metadata_missing",),
        }
    staleness_days = max(0, (generated_at.date() - corpus_end).days)
    warning = staleness_days > int(threshold_days)
    return {
        "training_corpus_staleness_days": staleness_days,
        "training_corpus_staleness_warning": warning,
        "training_corpus_staleness_warning_threshold_days": int(threshold_days),
        "warning_flags": ("training_corpus_stale_gt_3d",) if warning else (),
    }


def _optional_int(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
