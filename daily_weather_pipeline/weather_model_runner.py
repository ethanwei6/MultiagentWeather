from __future__ import annotations

import hashlib
import statistics
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

from .backtesting_engine import (
    WeatherModelDecisionContext,
    WeatherModelPrediction,
    _hash_mapping,
    validate_weather_model_context_firewall,
)
from .weather_model_improvement import (
    CandidateConfig,
    ObservedHighContext,
    _candidate_grid,
    _predict_config,
)
from .weather_model_validation import (
    DEFAULT_END_DATE,
    DEFAULT_SOURCE_REPO,
    DEFAULT_START_DATE,
    ForecastRow,
    MarketRow,
    ValidationRow,
    _artifact_paths,
    _bucket_shape,
    _build_validation_rows,
    _clip_probability,
    _day_type,
    _distance_bucket,
    _lead_days_and_local_hour,
    _load_forecasts,
    _load_markets,
    _load_realized,
    _load_weather_strategy,
    _score_model_probabilities,
    _threshold_distance,
    _weighted_consensus_probability,
)


BaselineProvider = Callable[
    [WeatherModelDecisionContext],
    Tuple[float, Mapping[str, float]],
]


MODEL_VERSION = "observed_high_guarded_shrinkage_research_runner_v1"
DEFAULT_CONFIG_ID = (
    "observed_guarded|tail=0.006|far=1.00|city_guard=0.25|region_guard=0.70"
)


@dataclass
class ObservedHighGuardedShrinkageWeatherModelRunner:
    """Live-like adapter for the research-only observed-high guarded shrinkage candidate."""

    work_root: Path = Path(".")
    source_repo: Path = DEFAULT_SOURCE_REPO
    start: date = date.fromisoformat(DEFAULT_START_DATE)
    end: date = date.fromisoformat(DEFAULT_END_DATE)
    config_id: str = DEFAULT_CONFIG_ID
    label_availability_lag_hours: float = 12.0
    historical_rows: Optional[Sequence[ValidationRow]] = None
    baseline_probability_provider: Optional[BaselineProvider] = None
    model_version: str = MODEL_VERSION
    _strategy: Optional[Mapping[str, Any]] = field(default=None, init=False, repr=False)
    _rows: Optional[list[ValidationRow]] = field(default=None, init=False, repr=False)
    _city_metadata: dict[str, Mapping[str, Any]] = field(
        default_factory=dict, init=False, repr=False
    )
    _last_train_key: Optional[datetime] = field(default=None, init=False, repr=False)
    _last_train_rows: Optional[list[ValidationRow]] = field(default=None, init=False, repr=False)
    _last_fit_cache_key: Optional[datetime] = field(default=None, init=False, repr=False)
    _last_fit_cache: dict[tuple[str, int], Any] = field(
        default_factory=dict, init=False, repr=False
    )

    @property
    def config(self) -> CandidateConfig:
        for candidate in _candidate_grid(max_configs=0):
            if candidate.config_id == self.config_id:
                return candidate
        raise ValueError(f"Unknown observed-high guarded config_id: {self.config_id}")

    def metadata(self) -> Mapping[str, str]:
        return {
            "runner_name": "ObservedHighGuardedShrinkageWeatherModelRunner",
            "model_version": self.model_version,
            "model_code_hash": self.model_code_hash(),
            "model_config_hash": self.model_config_hash(),
            "fixture_only": "false",
            "promotion_status": "research_only_not_promoted",
            "core_weather_model_only": "true",
        }

    def model_code_hash(self) -> str:
        digest = hashlib.sha256()
        for path in (
            Path(__file__),
            Path(__file__).with_name("weather_model_improvement.py"),
            Path(__file__).with_name("weather_model_validation.py"),
        ):
            digest.update(path.name.encode("utf-8"))
            digest.update(path.read_bytes())
        return digest.hexdigest()

    def model_config_hash(self) -> str:
        config = self.config
        return _hash_mapping(
            {
                "model_version": self.model_version,
                "config_id": config.config_id,
                "family": config.family,
                "kind": config.kind,
                "params": dict(config.params),
                "label_availability_lag_hours": self.label_availability_lag_hours,
                "train_rows_cutoff": "local_target_day_end_plus_lag_before_decision_time",
                "research_only": True,
            }
        )

    def predict(
        self,
        context: WeatherModelDecisionContext,
    ) -> Optional[WeatherModelPrediction]:
        validate_weather_model_context_firewall(context)
        self._validate_point_in_time_inputs(context)
        self._load_rows()
        score_row = self._context_to_validation_row(context)
        if score_row is None:
            return None
        baseline = self._baseline_probability(context, score_row)
        if baseline is None:
            return None
        baseline_probability, model_probabilities = baseline
        score_row.current_probability = baseline_probability
        score_row.mean_probability = _mean_probability(model_probabilities, baseline_probability)
        score_row.model_probabilities = dict(model_probabilities)
        train_rows = self._training_rows_for(context.decision_time_utc)
        observed_context = self._observed_context_for(context, score_row)
        fit_cache = self._fit_cache_for(context.decision_time_utc)
        probabilities = _predict_config(
            self.config,
            train_rows,
            [score_row],
            fit_cache=fit_cache,
            observed_context=observed_context,
        )
        p_yes = _clip_probability(probabilities[score_row.row_id])
        per_source = _per_source_probabilities(model_probabilities, p_yes)
        values = list(per_source.values())
        model_agreement = _agreement(values, p_yes)
        probability_stdev = statistics.pstdev(values) if len(values) > 1 else 0.0
        metadata = self.metadata()
        now = datetime.now(timezone.utc)
        p_side = p_yes if context.side.upper() == "YES" else 1.0 - p_yes
        observed_confidence = _observed_high_confidence(context)
        return WeatherModelPrediction(
            p_yes=p_yes,
            model_agreement=model_agreement,
            model_count=len(per_source),
            probability_stdev=probability_stdev,
            per_source_probabilities=per_source,
            remaining_exceed_probability=(
                p_yes if context.threshold_low_f is not None and context.threshold_high_f is None else None
            ),
            observed_high_aware_confidence=observed_confidence,
            model_version=metadata["model_version"],
            model_code_hash=metadata["model_code_hash"],
            model_config_hash=metadata["model_config_hash"],
            feature_provenance={
                **dict(context.feature_provenance),
                "research_candidate": "observed_high_guarded_shrinkage",
                "training_labels": "historical_realized_highs_available_before_decision_time",
            },
            input_source_paths=context.input_source_paths,
            forecast_as_of_utc=context.forecast_run_time_utc,
            prediction_timestamp_utc=now,
            prediction_as_of_utc=context.decision_time_utc,
            metadata={
                "input_hash": context.input_hash,
                "p_yes": p_yes,
                "p_no": 1.0 - p_yes,
                "p_side": p_side,
                "research_only": True,
                "no_profitability_inference": True,
                "config_id": self.config_id,
                "baseline_probability": baseline_probability,
                "train_rows": len(train_rows),
                "label_availability_lag_hours": self.label_availability_lag_hours,
                "forecast_as_of_utc": (
                    context.forecast_run_time_utc.isoformat()
                    if context.forecast_run_time_utc
                    else None
                ),
                "prediction_as_of_utc": context.decision_time_utc.isoformat(),
                "observed_high_omitted_reason": context.observed_high_omitted_reason,
            },
        )

    def _validate_point_in_time_inputs(self, context: WeatherModelDecisionContext) -> None:
        if context.forecast_run_time_utc is not None:
            if context.forecast_run_time_utc > context.decision_time_utc:
                raise ValueError("Weather model context contains future forecast snapshot")
        if context.observed_high_timestamp_utc is not None:
            if context.observed_high_timestamp_utc > context.decision_time_utc:
                raise ValueError("Weather model context contains future observed-high snapshot")

    def _training_rows_for(self, decision_time_utc: datetime) -> list[ValidationRow]:
        if self._last_train_key == decision_time_utc and self._last_train_rows is not None:
            return self._last_train_rows
        rows = [
            row
            for row in self._load_rows()
            if _label_available_at_utc(row, self.label_availability_lag_hours)
            <= decision_time_utc
        ]
        self._last_train_key = decision_time_utc
        self._last_train_rows = rows
        self._last_fit_cache_key = None
        self._last_fit_cache = {}
        return rows

    def _fit_cache_for(self, decision_time_utc: datetime) -> dict[tuple[str, int], Any]:
        if self._last_fit_cache_key != decision_time_utc:
            self._last_fit_cache_key = decision_time_utc
            self._last_fit_cache = {}
        return self._last_fit_cache

    def _load_rows(self) -> list[ValidationRow]:
        if self._rows is not None:
            return self._rows
        if self.historical_rows is not None:
            self._rows = list(self.historical_rows)
            self._index_city_metadata(self._rows)
            return self._rows
        strategy = self._load_strategy()
        paths = _artifact_paths(Path(self.work_root).resolve())
        markets = _load_markets(paths["market_db"], start=self.start, end=self.end)
        forecasts = _load_forecasts(paths["weather_db"], start=self.start, end=self.end)
        realized = _load_realized(paths["weather_db"], start=self.start, end=self.end)
        self._rows = _build_validation_rows(strategy, markets, forecasts, realized)
        self._index_city_metadata(self._rows)
        return self._rows

    def _load_strategy(self) -> Mapping[str, Any]:
        if self._strategy is None:
            self._strategy = _load_weather_strategy(Path(self.source_repo).expanduser().resolve())
        return self._strategy

    def _index_city_metadata(self, rows: Sequence[ValidationRow]) -> None:
        if self._city_metadata:
            return
        for row in rows:
            self._city_metadata.setdefault(
                row.normalized_city,
                {
                    "country_region": row.country_region,
                    "timezone_name": row.timezone_name,
                    "latitude": row.latitude,
                    "longitude": row.longitude,
                },
            )

    def _baseline_probability(
        self,
        context: WeatherModelDecisionContext,
        row: ValidationRow,
    ) -> Optional[Tuple[float, Mapping[str, float]]]:
        if self.baseline_probability_provider is not None:
            probability, probabilities = self.baseline_probability_provider(context)
            source_probs = dict(probabilities)
            if not source_probs:
                source_probs = {"injected_baseline": float(probability)}
            return _clip_probability(float(probability)), source_probs
        market = self._market_row_from_context(context, row)
        forecast = self._forecast_row_from_context(context, row)
        if market is None or forecast is None:
            return None
        strategy = self._load_strategy()
        model_probabilities = _score_model_probabilities(
            strategy,
            market,
            forecast,
            bias_f=0.0,
        )
        if not model_probabilities:
            return None
        probability = _weighted_consensus_probability(
            model_probabilities,
            source_weights=strategy["source_weights"],
            model_weights=strategy["model_weights"],
        )
        return probability, model_probabilities

    def _context_to_validation_row(
        self,
        context: WeatherModelDecisionContext,
    ) -> Optional[ValidationRow]:
        if not context.normalized_city or not context.market_date_local:
            return None
        if context.forecast_temperature_2m_max_f is None:
            return None
        if context.threshold_low_f is None and context.threshold_high_f is None:
            return None
        try:
            target_date = date.fromisoformat(context.market_date_local)
        except ValueError:
            return None
        metadata = self._metadata_for_context(context)
        timezone_name = str(metadata["timezone_name"])
        if context.lead_days is None:
            lead_days, local_hour = _lead_days_and_local_hour(
                context.decision_time_utc,
                target_date,
                timezone_name,
            )
        else:
            lead_days = int(context.lead_days)
            local_hour = _local_hour(context.decision_time_utc, timezone_name)
        forecast_high = float(context.forecast_temperature_2m_max_f)
        distance = _threshold_distance(
            forecast_high,
            context.threshold_low_f,
            context.threshold_high_f,
        )
        bucket_shape = _bucket_shape(context.threshold_low_f, context.threshold_high_f)
        row_id = f"runner:{context.decision_id}:{context.universe_row_id}"
        return ValidationRow(
            row_id=row_id,
            universe_row_id=context.universe_row_id,
            forecast_id=f"{context.decision_id}:forecast",
            normalized_city=context.normalized_city,
            country_region=context.country_region or metadata.get("country_region"),
            timezone_name=timezone_name,
            market_date_local=target_date,
            decision_time_utc=context.decision_time_utc,
            decision_hour_utc=int(context.decision_hour_utc or context.decision_time_utc.hour),
            local_decision_hour=local_hour,
            lead_days=lead_days,
            day_type=_day_type(lead_days),
            source_name=context.forecast_source_name or "unknown_forecast_source",
            model_name=_model_name(context.forecast_features),
            bucket_label=context.bucket_label or context.bucket_type or "unknown_bucket",
            bucket_type=context.bucket_type or "unknown",
            bucket_shape=bucket_shape,
            threshold_low_f=context.threshold_low_f,
            threshold_high_f=context.threshold_high_f,
            threshold_distance_f=distance,
            threshold_distance_bucket=_distance_bucket(distance),
            realized_high_f=forecast_high,
            forecast_high_f=forecast_high,
            forecast_error_f=0.0,
            outcome=0,
            resolved_outcome=None,
            resolved_outcome_mismatch=False,
            current_probability=0.5,
            mean_probability=0.5,
            model_probabilities={},
            forecast_quality_flags=tuple(context.data_quality_flags),
            realized_quality_flags=(),
            feature_json=dict(context.forecast_features),
            latitude=float(metadata["latitude"]),
            longitude=float(metadata["longitude"]),
        )

    def _metadata_for_context(
        self,
        context: WeatherModelDecisionContext,
    ) -> Mapping[str, Any]:
        if not self._city_metadata and self.historical_rows is not None:
            self._index_city_metadata(self.historical_rows)
        city_metadata = self._city_metadata.get(context.normalized_city or "", {})
        return {
            "country_region": context.country_region or city_metadata.get("country_region"),
            "timezone_name": context.timezone or city_metadata.get("timezone_name") or "UTC",
            "latitude": _float_feature(context.forecast_features, ("latitude", "lat"))
            if _float_feature(context.forecast_features, ("latitude", "lat")) is not None
            else float(city_metadata.get("latitude") or 0.0),
            "longitude": _float_feature(context.forecast_features, ("longitude", "lon", "lng"))
            if _float_feature(context.forecast_features, ("longitude", "lon", "lng")) is not None
            else float(city_metadata.get("longitude") or 0.0),
        }

    def _market_row_from_context(
        self,
        context: WeatherModelDecisionContext,
        row: ValidationRow,
    ) -> Optional[MarketRow]:
        if not context.normalized_city or not context.market_date_local:
            return None
        return MarketRow(
            universe_row_id=context.universe_row_id,
            normalized_city=context.normalized_city,
            country_region=row.country_region,
            latitude=row.latitude,
            longitude=row.longitude,
            timezone_name=row.timezone_name,
            market_date_local=row.market_date_local,
            bucket_label=row.bucket_label,
            bucket_type=row.bucket_type,
            threshold_low_f=context.threshold_low_f,
            threshold_high_f=context.threshold_high_f,
            units=None,
            resolved_outcome=None,
        )

    def _forecast_row_from_context(
        self,
        context: WeatherModelDecisionContext,
        row: ValidationRow,
    ) -> Optional[ForecastRow]:
        if context.forecast_temperature_2m_max_f is None:
            return None
        return ForecastRow(
            forecast_id=row.forecast_id,
            normalized_city=row.normalized_city,
            country_region=row.country_region,
            latitude=row.latitude,
            longitude=row.longitude,
            timezone_name=row.timezone_name,
            local_date=row.market_date_local,
            decision_time_utc=context.decision_time_utc,
            decision_hour_utc=row.decision_hour_utc,
            forecast_run_time_utc=context.forecast_run_time_utc,
            source_name=row.source_name,
            model_name=row.model_name,
            temperature_2m_max_f=float(context.forecast_temperature_2m_max_f),
            hourly_temperature_2m_max_f=_float_feature(
                context.forecast_features,
                (
                    "hourly_temperature_2m_max_f",
                    "hourly_temperature_2m_max",
                    "temperature_2m_hourly_max_f",
                ),
            ),
            feature_json=dict(context.forecast_features),
            data_quality_flags=tuple(context.data_quality_flags),
        )

    def _observed_context_for(
        self,
        context: WeatherModelDecisionContext,
        row: ValidationRow,
    ) -> Mapping[tuple[str, date, datetime], ObservedHighContext]:
        if context.observed_high_so_far_f is None:
            return {}
        if context.observed_high_timestamp_utc is not None:
            if context.observed_high_timestamp_utc > context.decision_time_utc:
                return {}
        key = (row.normalized_city, row.market_date_local, row.decision_time_utc)
        return {
            key: ObservedHighContext(
                observed_high_f=float(context.observed_high_so_far_f),
                observed_at_utc=context.observed_high_timestamp_utc,
                is_final_at_decision=bool(context.observed_high_final_at_decision),
            )
        }


def _label_available_at_utc(row: ValidationRow, lag_hours: float) -> datetime:
    try:
        zone = ZoneInfo(row.timezone_name)
    except Exception:
        zone = timezone.utc
    local_end = datetime.combine(
        row.market_date_local + timedelta(days=1),
        datetime.min.time(),
        tzinfo=zone,
    )
    return local_end.astimezone(timezone.utc) + timedelta(hours=float(lag_hours))


def _local_hour(decision_time_utc: datetime, timezone_name: str) -> float:
    try:
        zone = ZoneInfo(timezone_name)
    except Exception:
        zone = timezone.utc
    local_dt = decision_time_utc.astimezone(zone)
    return local_dt.hour + local_dt.minute / 60.0 + local_dt.second / 3600.0


def _model_name(features: Mapping[str, Any]) -> Optional[str]:
    for key in ("model_name", "model", "forecast_model"):
        value = features.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _float_feature(features: Mapping[str, Any], keys: Sequence[str]) -> Optional[float]:
    for key in keys:
        value = features.get(key)
        if value in (None, ""):
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _mean_probability(
    model_probabilities: Mapping[str, float],
    fallback: float,
) -> float:
    if not model_probabilities:
        return _clip_probability(fallback)
    return _clip_probability(sum(float(value) for value in model_probabilities.values()) / len(model_probabilities))


def _per_source_probabilities(
    model_probabilities: Mapping[str, float],
    final_probability: float,
) -> dict[str, float]:
    output = {
        str(key): _clip_probability(float(value))
        for key, value in sorted(model_probabilities.items())
    }
    output["observed_high_guarded_shrinkage"] = _clip_probability(final_probability)
    return output


def _agreement(probabilities: Sequence[float], final_probability: float) -> float:
    if not probabilities:
        return 1.0
    final_yes = final_probability >= 0.5
    agreeing = sum(1 for probability in probabilities if (probability >= 0.5) == final_yes)
    return agreeing / len(probabilities)


def _observed_high_confidence(context: WeatherModelDecisionContext) -> Optional[float]:
    if context.observed_high_so_far_f is None:
        return None
    if context.observed_high_final_at_decision:
        return 1.0
    observed = float(context.observed_high_so_far_f)
    if context.threshold_high_f is not None and observed > float(context.threshold_high_f):
        return 0.95
    if context.threshold_low_f is not None and context.threshold_high_f is None:
        if observed >= float(context.threshold_low_f):
            return 0.95
    return 0.75
