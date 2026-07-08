from __future__ import annotations

import csv
import gzip
import hashlib
import json
import sqlite3
import argparse
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Optional, Protocol, Sequence
from zoneinfo import ZoneInfo


BACKTEST_FRAMEWORK_SCHEMA_VERSION = "dailyweather.backtest.framework.v1"
CANONICAL_LABEL_ARTIFACT = (
    "work/reports/settlement_label_audit/"
    "20260701T090014Z-settlement-label-audit-canonical-labels.csv"
)
SIX_MONTH_V1_FREEZE_MANIFEST = (
    "work/reports/dataset_freeze/20260707T040322Z-six-month-v1-dataset-freeze.json"
)
OFFICIAL_TRADING_REPLAY_SQL_PREDICATE = (
    "quote_timestamp_utc IS NOT NULL "
    "AND COALESCE(exclude_from_backtest_replay, 0) = 0 "
    "AND COALESCE(recommended_action, '') != 'exclude_or_manual_review'"
)
COMPACT_DECISION_VIEW = "decision_snapshots_current_enriched"
WIDE_PRICE_DB_RELATIVE_PATH = "work/data/backtest/daily_weather_price_history.sqlite"
COMPACT_PRICE_DB_RELATIVE_PATH = "work/data/backtest/daily_weather_price_history_compact.sqlite"
FEE_MODEL_ZERO = "zero"
FEE_MODEL_POLYMARKET_WEATHER_TAKER = "polymarket_weather_taker"
DEFAULT_POLYMARKET_WEATHER_TAKER_FEE_RATE = 0.05
FEE_ROUNDING_DECIMALS = 5
DEFAULT_STRATEGY_REPLAY_FEE_MODEL = FEE_MODEL_POLYMARKET_WEATHER_TAKER
DEFAULT_STRATEGY_REPLAY_FEE_RATE = DEFAULT_POLYMARKET_WEATHER_TAKER_FEE_RATE
DEFAULT_DELAY_STRESS_SECONDS = 2
DEFAULT_WORSE_FILL_TICK = 0.01
GEN2_REPRODUCIBILITY_REQUIRED_RUNS = 3


class SkipReason(str, Enum):
    MISSING_SHARED_SCORE = "missing_shared_score"
    MISSING_MODEL_OUTPUT = "missing_model_output"
    INVALID_SHARED_SCORE = "invalid_shared_score"
    MODEL_INPUT_FIREWALL_VIOLATION = "model_input_firewall_violation"
    MISSING_QUOTE_SNAPSHOT = "missing_quote_snapshot"
    MISSING_FORECAST_SNAPSHOT = "missing_forecast_snapshot"
    MISSING_OBSERVED_HIGH_SNAPSHOT = "missing_observed_high_snapshot"
    MISSING_HEAT_WINDOW_FIELDS = "missing_heat_window_fields"
    MISSING_CANONICAL_LABEL = "missing_canonical_label"
    CANONICAL_LABEL_EXCLUDED = "canonical_label_excluded"
    NON_CANONICAL_LABEL_SOURCE = "non_canonical_label_source"
    UNSUPPORTED_SIDE = "unsupported_side"
    FUTURE_QUOTE = "future_quote"
    FUTURE_FORECAST = "future_forecast"
    FUTURE_OBSERVED_HIGH = "future_observed_high"
    STALE_QUOTE = "stale_quote"
    STALE_OBSERVED_HIGH = "stale_observed_high"
    MISSING_EXECUTABLE_PRICE = "missing_executable_price"
    MISSING_MARKET_FIELDS = "missing_market_fields"
    MISSING_SETTLEMENT_PRECISION = "missing_settlement_precision"
    PARSER_CONFIDENCE_LOW = "parser_confidence_low"
    SPREAD_TOO_WIDE = "spread_too_wide"
    ASK_BELOW_MINIMUM = "ask_below_minimum"
    ASK_ABOVE_MAXIMUM = "ask_above_maximum"
    INSUFFICIENT_VISIBLE_SIZE = "insufficient_visible_size"
    MODEL_COUNT_LOW = "model_count_low"
    MODEL_AGREEMENT_LOW = "model_agreement_low"
    PROBABILITY_STDEV_HIGH = "probability_stdev_high"
    OBSERVED_HIGH_CONFIDENCE_LOW = "observed_high_confidence_low"
    UNSUPPORTED_LEAD_DAY = "unsupported_lead_day"
    FAIR_VALUE_BELOW_MINIMUM = "fair_value_below_minimum"
    NET_EDGE_BELOW_MINIMUM = "net_edge_below_minimum"
    COUNTER_EVENT_PROBABILITY_HIGH = "counter_event_probability_high"
    BOUNDED_BUCKET_GATES_NOT_MET = "bounded_bucket_gates_not_met"
    SETTLEMENT_SOURCE_CONFLICT = "settlement_source_conflict"
    PORTFOLIO_CAP_EXCEEDED = "portfolio_cap_exceeded"
    TOKEN_SIDE_CAP_EXCEEDED = "token_side_cap_exceeded"
    WINDOW_CAP_EXCEEDED = "window_cap_exceeded"
    REGION_CAP_EXCEEDED = "region_cap_exceeded"
    CITY_DATE_CAP_EXCEEDED = "city_date_cap_exceeded"
    SETTLEMENT_SOURCE_CAP_EXCEEDED = "settlement_source_cap_exceeded"
    PRIMARY_CITY_DATE_EXPOSURE_EXISTS = "primary_city_date_exposure_exists"
    REQUESTED_NOTIONAL_BELOW_MINIMUM = "requested_notional_below_minimum"
    LEAKAGE_VIOLATION = "leakage_violation"
    STRATEGY_NO_ACTION = "strategy_no_action"


class OrderAction(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class ReplayEventKind(str, Enum):
    SKIPPED = "skipped"
    STRATEGY_EVALUATED = "strategy_evaluated"
    ORDER_INTENT = "order_intent"


class StrategyFamily(str, Enum):
    FORECAST_ONLY = "before_day_forecast_only"
    SAME_DAY_OBSERVED_HIGH = "same_day_observed_high_so_far"
    LATE_DAY_HEAT_WINDOW = "late_day_heat_window_exhaustion"


@dataclass(frozen=True)
class QuoteSnapshot:
    timestamp_utc: datetime
    bid: Optional[float]
    ask: Optional[float]
    mid: Optional[float]
    last_price: Optional[float]
    spread: Optional[float]
    bid_size: Optional[float]
    ask_size: Optional[float]
    liquidity_depth: Optional[float]
    source_payload_id: Optional[str]
    source_cache_path: Optional[str]
    data_quality_flags: tuple[str, ...] = ()

    @property
    def has_executable_bid_ask(self) -> bool:
        return self.bid is not None and self.ask is not None


@dataclass(frozen=True)
class ForecastSnapshot:
    run_time_utc: datetime
    source_name: Optional[str]
    temperature_2m_max_f: Optional[float]
    features: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ObservedHighSnapshot:
    observed_high_f: float
    observed_high_timestamp_utc: datetime
    observation_source: str
    observation_final: bool = False
    source_match_confidence: Optional[float] = None
    source_family: Optional[str] = None
    source_station: Optional[str] = None
    deterministic_state: Optional[str] = None


@dataclass(frozen=True)
class HeatWindowSnapshot:
    local_hour: float
    forecast_peak_hour_local: Optional[float] = None
    remaining_hourly_forecast_max_f: Optional[float] = None
    heat_window_exhaustion_score: Optional[float] = None
    solar_radiation_state: Optional[str] = None
    cloud_cover_state: Optional[str] = None
    wind_state: Optional[str] = None
    season: Optional[str] = None


@dataclass(frozen=True)
class SharedScoreContract:
    p_yes: float
    side: str
    p_side: float
    model_agreement: float
    model_count: int
    probability_stdev: Optional[float] = None
    per_source_probabilities: Mapping[str, float] = field(default_factory=dict)
    remaining_exceed_probability: Optional[float] = None
    observed_high_aware_confidence: Optional[float] = None
    observed_high: Optional[ObservedHighSnapshot] = None
    heat_window: Optional[HeatWindowSnapshot] = None
    parser_confidence: Optional[float] = None
    settlement_unit: Optional[str] = None
    settlement_precision: Optional[str] = None
    scored_at_utc: Optional[datetime] = None
    score_model_version: Optional[str] = None
    score_family: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def p_no(self) -> float:
        return 1.0 - self.p_yes

    @property
    def score_errors(self) -> tuple[str, ...]:
        errors: list[str] = []
        for field_name, probability in (
            ("p_yes", self.p_yes),
            ("p_side", self.p_side),
            ("model_agreement", self.model_agreement),
        ):
            if probability < 0.0 or probability > 1.0:
                errors.append(f"{field_name}_outside_0_1")
        if self.observed_high_aware_confidence is not None and not (
            0.0 <= self.observed_high_aware_confidence <= 1.0
        ):
            errors.append("observed_high_aware_confidence_outside_0_1")
        if self.remaining_exceed_probability is not None and not (
            0.0 <= self.remaining_exceed_probability <= 1.0
        ):
            errors.append("remaining_exceed_probability_outside_0_1")
        if self.heat_window and self.heat_window.heat_window_exhaustion_score is not None:
            if not 0.0 <= self.heat_window.heat_window_exhaustion_score <= 1.0:
                errors.append("heat_window_exhaustion_score_outside_0_1")
        expected_side_probability = self.p_yes if self.side.upper() == "YES" else self.p_no
        if self.side.upper() not in {"YES", "NO"}:
            errors.append("unsupported_side")
        elif abs(self.p_side - expected_side_probability) > 1e-6:
            errors.append("p_side_inconsistent_with_side")
        if self.model_count < 0:
            errors.append("negative_model_count")
        return tuple(errors)

    @property
    def is_valid(self) -> bool:
        return not self.score_errors


FORBIDDEN_WEATHER_MODEL_CONTEXT_FIELDS = (
    "bid",
    "ask",
    "mid",
    "last_price",
    "spread",
    "bid_size",
    "ask_size",
    "canonical_label",
    "canonical_label_source",
    "resolved_outcome",
    "realized_temperature_2m_max_f",
    "settlement_outcome",
    "future_quote",
)


@dataclass(frozen=True)
class WeatherModelDecisionContext:
    decision_id: str
    universe_row_id: str
    market_id: Optional[str]
    slug: Optional[str]
    question: Optional[str]
    normalized_city: Optional[str]
    country_region: Optional[str]
    timezone: Optional[str]
    market_date_local: Optional[str]
    bucket_label: Optional[str]
    bucket_type: Optional[str]
    threshold_low_f: Optional[float]
    threshold_high_f: Optional[float]
    side: str
    token_id: str
    decision_time_utc: datetime
    decision_hour_utc: Optional[int]
    lead_days: Optional[int]
    parser_confidence: Optional[float]
    forecast_run_time_utc: Optional[datetime]
    forecast_source_name: Optional[str]
    forecast_temperature_2m_max_f: Optional[float]
    forecast_features: Mapping[str, Any] = field(default_factory=dict)
    observed_high_so_far_f: Optional[float] = None
    observed_high_timestamp_utc: Optional[datetime] = None
    observed_high_source: Optional[str] = None
    observed_high_final_at_decision: Optional[bool] = None
    observed_high_source_payload_id: Optional[str] = None
    observed_high_raw_cache_path: Optional[str] = None
    observed_high_data_quality_flags: tuple[str, ...] = ()
    observed_high_omitted_reason: Optional[str] = None
    data_quality_flags: tuple[str, ...] = ()
    input_source_paths: tuple[str, ...] = ()
    feature_provenance: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def from_decision(cls, decision: "DecisionSnapshot") -> "WeatherModelDecisionContext":
        forecast = decision.forecast
        observed = decision.observed_high
        observed_high_so_far_f: Optional[float] = None
        observed_high_timestamp_utc: Optional[datetime] = None
        observed_high_source: Optional[str] = None
        observed_high_final_at_decision: Optional[bool] = None
        observed_high_omitted_reason: Optional[str] = None
        if observed is not None:
            if observed.observed_high_timestamp_utc <= decision.decision_time_utc:
                observed_high_so_far_f = observed.observed_high_f
                observed_high_timestamp_utc = observed.observed_high_timestamp_utc
                observed_high_source = observed.observation_source
                observed_high_final_at_decision = observed.observation_final
            else:
                observed_high_omitted_reason = "future_observed_high_not_model_visible"

        input_source_paths = tuple(
            path
            for path in (
                decision.observed_high_raw_cache_path if observed_high_so_far_f is not None else None,
            )
            if path
        )
        provenance = {
            "forecast": "point_in_time_forecast_snapshot",
            "observed_high_so_far": "decision_time_observed_high_snapshot",
            "market_metadata": "decision_artifact_market_fields",
        }
        return cls(
            decision_id=decision.decision_id,
            universe_row_id=decision.universe_row_id,
            market_id=decision.market_id,
            slug=decision.slug,
            question=decision.question,
            normalized_city=decision.normalized_city,
            country_region=decision.country_region,
            timezone=decision.timezone,
            market_date_local=decision.market_date_local,
            bucket_label=decision.bucket_label,
            bucket_type=decision.bucket_type,
            threshold_low_f=decision.threshold_low_f,
            threshold_high_f=decision.threshold_high_f,
            side=decision.side,
            token_id=decision.token_id,
            decision_time_utc=decision.decision_time_utc,
            decision_hour_utc=decision.decision_hour_utc,
            lead_days=decision.lead_days,
            parser_confidence=decision.parser_confidence,
            forecast_run_time_utc=forecast.run_time_utc if forecast else None,
            forecast_source_name=forecast.source_name if forecast else None,
            forecast_temperature_2m_max_f=forecast.temperature_2m_max_f if forecast else None,
            forecast_features=dict(forecast.features) if forecast else {},
            observed_high_so_far_f=observed_high_so_far_f,
            observed_high_timestamp_utc=observed_high_timestamp_utc,
            observed_high_source=observed_high_source,
            observed_high_final_at_decision=observed_high_final_at_decision,
            observed_high_source_payload_id=(
                decision.observed_high_source_payload_id if observed_high_so_far_f is not None else None
            ),
            observed_high_raw_cache_path=(
                decision.observed_high_raw_cache_path if observed_high_so_far_f is not None else None
            ),
            observed_high_data_quality_flags=(
                decision.observed_high_data_quality_flags if observed_high_so_far_f is not None else ()
            ),
            observed_high_omitted_reason=observed_high_omitted_reason,
            data_quality_flags=decision.data_quality_flags,
            input_source_paths=input_source_paths,
            feature_provenance=provenance,
        )

    def to_hash_payload(self) -> Mapping[str, Any]:
        return _json_ready(
            {
                "decision_id": self.decision_id,
                "universe_row_id": self.universe_row_id,
                "market_id": self.market_id,
                "slug": self.slug,
                "question": self.question,
                "normalized_city": self.normalized_city,
                "country_region": self.country_region,
                "timezone": self.timezone,
                "market_date_local": self.market_date_local,
                "bucket_label": self.bucket_label,
                "bucket_type": self.bucket_type,
                "threshold_low_f": self.threshold_low_f,
                "threshold_high_f": self.threshold_high_f,
                "side": self.side,
                "token_id": self.token_id,
                "decision_time_utc": self.decision_time_utc,
                "decision_hour_utc": self.decision_hour_utc,
                "lead_days": self.lead_days,
                "parser_confidence": self.parser_confidence,
                "forecast_run_time_utc": self.forecast_run_time_utc,
                "forecast_source_name": self.forecast_source_name,
                "forecast_temperature_2m_max_f": self.forecast_temperature_2m_max_f,
                "forecast_features": self.forecast_features,
                "observed_high_so_far_f": self.observed_high_so_far_f,
                "observed_high_timestamp_utc": self.observed_high_timestamp_utc,
                "observed_high_source": self.observed_high_source,
                "observed_high_final_at_decision": self.observed_high_final_at_decision,
                "observed_high_source_payload_id": self.observed_high_source_payload_id,
                "observed_high_raw_cache_path": self.observed_high_raw_cache_path,
                "observed_high_data_quality_flags": self.observed_high_data_quality_flags,
                "observed_high_omitted_reason": self.observed_high_omitted_reason,
                "data_quality_flags": self.data_quality_flags,
                "input_source_paths": self.input_source_paths,
                "feature_provenance": self.feature_provenance,
            }
        )

    @property
    def input_hash(self) -> str:
        return _hash_mapping(self.to_hash_payload())


@dataclass(frozen=True)
class WeatherModelPrediction:
    p_yes: float
    model_agreement: float
    model_count: int
    probability_stdev: Optional[float]
    per_source_probabilities: Mapping[str, float]
    remaining_exceed_probability: Optional[float]
    observed_high_aware_confidence: Optional[float]
    model_version: str
    model_code_hash: str
    model_config_hash: str
    feature_provenance: Mapping[str, str]
    input_source_paths: tuple[str, ...]
    forecast_as_of_utc: Optional[datetime]
    prediction_timestamp_utc: datetime
    prediction_as_of_utc: datetime
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def p_no(self) -> float:
        return 1.0 - self.p_yes

    def to_shared_score(
        self,
        *,
        side: str,
        observed_high: Optional[ObservedHighSnapshot],
        parser_confidence: Optional[float],
        settlement_unit: Optional[str],
        settlement_precision: Optional[str],
    ) -> SharedScoreContract:
        side_upper = side.upper()
        p_side = self.p_yes if side_upper == "YES" else self.p_no
        return SharedScoreContract(
            p_yes=self.p_yes,
            side=side_upper,
            p_side=p_side,
            model_agreement=self.model_agreement,
            model_count=self.model_count,
            probability_stdev=self.probability_stdev,
            per_source_probabilities=self.per_source_probabilities,
            remaining_exceed_probability=self.remaining_exceed_probability,
            observed_high_aware_confidence=self.observed_high_aware_confidence,
            observed_high=observed_high,
            parser_confidence=parser_confidence,
            settlement_unit=settlement_unit,
            settlement_precision=settlement_precision,
            scored_at_utc=self.prediction_as_of_utc,
            score_model_version=self.model_version,
            score_family="replay_generated_weather_model_runner",
            metadata={
                **dict(self.metadata),
                "model_version": self.model_version,
                "model_code_hash": self.model_code_hash,
                "model_config_hash": self.model_config_hash,
                "feature_provenance": dict(self.feature_provenance),
                "input_source_paths": list(self.input_source_paths),
                "forecast_as_of_utc": (
                    self.forecast_as_of_utc.isoformat() if self.forecast_as_of_utc else None
                ),
                "prediction_timestamp_utc": self.prediction_timestamp_utc.isoformat(),
                "prediction_as_of_utc": self.prediction_as_of_utc.isoformat(),
                "cache_name": "replay_generated_shared_scores",
            },
        )


class WeatherModelRunner(Protocol):
    def metadata(self) -> Mapping[str, str]:
        ...

    def predict(
        self,
        context: WeatherModelDecisionContext,
    ) -> Optional[WeatherModelPrediction]:
        ...


@dataclass(frozen=True)
class DeterministicFixtureWeatherModelRunner:
    model_version: str = "deterministic_fixture_weather_runner_v1"
    model_count: int = 4
    model_agreement: float = 1.0
    probability_stdev: float = 0.01
    yes_probability_for_yes_side: float = 0.99
    yes_probability_for_no_side: float = 0.01

    def metadata(self) -> Mapping[str, str]:
        return {
            "runner_name": "DeterministicFixtureWeatherModelRunner",
            "model_version": self.model_version,
            "model_code_hash": _hash_mapping(
                {
                    "runner": "DeterministicFixtureWeatherModelRunner",
                    "algorithm": "side_direction_fixture_no_weather_edge_claim_v1",
                }
            ),
            "model_config_hash": _hash_mapping(self.config_payload()),
            "fixture_only": "true",
        }

    def config_payload(self) -> Mapping[str, Any]:
        return {
            "model_count": self.model_count,
            "model_agreement": self.model_agreement,
            "probability_stdev": self.probability_stdev,
            "yes_probability_for_yes_side": self.yes_probability_for_yes_side,
            "yes_probability_for_no_side": self.yes_probability_for_no_side,
        }

    def predict(
        self,
        context: WeatherModelDecisionContext,
    ) -> Optional[WeatherModelPrediction]:
        validate_weather_model_context_firewall(context)
        metadata = self.metadata()
        p_yes = (
            self.yes_probability_for_yes_side
            if context.side.upper() == "YES"
            else self.yes_probability_for_no_side
        )
        now = datetime.now(timezone.utc)
        return WeatherModelPrediction(
            p_yes=p_yes,
            model_agreement=self.model_agreement,
            model_count=self.model_count,
            probability_stdev=self.probability_stdev,
            per_source_probabilities={"deterministic_fixture": p_yes},
            remaining_exceed_probability=1.0 - p_yes,
            observed_high_aware_confidence=(
                0.95 if context.observed_high_so_far_f is not None else None
            ),
            model_version=self.model_version,
            model_code_hash=metadata["model_code_hash"],
            model_config_hash=metadata["model_config_hash"],
            feature_provenance=context.feature_provenance,
            input_source_paths=context.input_source_paths,
            forecast_as_of_utc=context.forecast_run_time_utc,
            prediction_timestamp_utc=now,
            prediction_as_of_utc=context.decision_time_utc,
            metadata={
                "fixture_only": True,
                "no_profitability_inference": True,
                "input_hash": context.input_hash,
            },
        )


@dataclass(frozen=True)
class ReplayGeneratedScoreRecord:
    cache_name: str
    decision_id: str
    universe_row_id: str
    generated_at_utc: datetime
    model_version: str
    model_code_hash: str
    model_config_hash: str
    input_hash: str
    prediction_hash: str
    p_yes: float
    p_no: float
    p_side: float
    side: str
    model_agreement: float
    model_count: int
    probability_stdev: Optional[float]
    per_source_probabilities: Mapping[str, float]
    remaining_exceed_probability: Optional[float]
    observed_high_aware_confidence: Optional[float]
    score_model_version: Optional[str]
    score_family: Optional[str]
    feature_provenance: Mapping[str, str]
    input_source_paths: tuple[str, ...]
    forecast_as_of_utc: Optional[datetime]
    prediction_timestamp_utc: datetime

    def to_json(self) -> Mapping[str, Any]:
        return _json_ready(self.__dict__)


class ReplayGeneratedScoreCache:
    cache_name = "replay_generated_shared_scores"

    def __init__(
        self,
        *,
        stream_path: Optional[Path] = None,
        compress: bool = False,
        keep_records: bool = True,
    ) -> None:
        self.records: list[ReplayGeneratedScoreRecord] = []
        self.stream_path = Path(stream_path) if stream_path is not None else None
        self.compress = bool(compress or (self.stream_path and self.stream_path.suffix == ".gz"))
        self.keep_records = keep_records
        self._stream_handle: Optional[Any] = None
        self._record_count = 0
        self._model_versions: set[str] = set()
        self._model_code_hashes: set[str] = set()
        self._model_config_hashes: set[str] = set()
        self._input_hash_count = 0
        self._prediction_hash_count = 0

    def add(
        self,
        *,
        context: WeatherModelDecisionContext,
        prediction: WeatherModelPrediction,
        score: SharedScoreContract,
    ) -> ReplayGeneratedScoreRecord:
        record = ReplayGeneratedScoreRecord(
            cache_name=self.cache_name,
            decision_id=context.decision_id,
            universe_row_id=context.universe_row_id,
            generated_at_utc=datetime.now(timezone.utc),
            model_version=prediction.model_version,
            model_code_hash=prediction.model_code_hash,
            model_config_hash=prediction.model_config_hash,
            input_hash=context.input_hash,
            prediction_hash=_hash_mapping(
                {
                    "p_yes": prediction.p_yes,
                    "model_agreement": prediction.model_agreement,
                    "model_count": prediction.model_count,
                    "probability_stdev": prediction.probability_stdev,
                    "per_source_probabilities": prediction.per_source_probabilities,
                    "remaining_exceed_probability": prediction.remaining_exceed_probability,
                    "observed_high_aware_confidence": prediction.observed_high_aware_confidence,
                    "model_version": prediction.model_version,
                    "model_code_hash": prediction.model_code_hash,
                    "model_config_hash": prediction.model_config_hash,
                    "input_hash": context.input_hash,
                }
            ),
            p_yes=score.p_yes,
            p_no=score.p_no,
            p_side=score.p_side,
            side=score.side,
            model_agreement=score.model_agreement,
            model_count=score.model_count,
            probability_stdev=score.probability_stdev,
            per_source_probabilities=score.per_source_probabilities,
            remaining_exceed_probability=score.remaining_exceed_probability,
            observed_high_aware_confidence=score.observed_high_aware_confidence,
            score_model_version=score.score_model_version,
            score_family=score.score_family,
            feature_provenance=prediction.feature_provenance,
            input_source_paths=prediction.input_source_paths,
            forecast_as_of_utc=prediction.forecast_as_of_utc,
            prediction_timestamp_utc=prediction.prediction_timestamp_utc,
        )
        self._record_count += 1
        self._model_versions.add(record.model_version)
        self._model_code_hashes.add(record.model_code_hash)
        self._model_config_hashes.add(record.model_config_hash)
        if record.input_hash:
            self._input_hash_count += 1
        if record.prediction_hash:
            self._prediction_hash_count += 1
        if self.keep_records:
            self.records.append(record)
        if self.stream_path is not None:
            self._write_stream_record(record)
        return record

    def is_reusable(
        self,
        record: ReplayGeneratedScoreRecord,
        *,
        context: WeatherModelDecisionContext,
        runner_metadata: Mapping[str, str],
    ) -> bool:
        return (
            record.cache_name == self.cache_name
            and record.input_hash == context.input_hash
            and record.model_code_hash == runner_metadata.get("model_code_hash")
            and record.model_config_hash == runner_metadata.get("model_config_hash")
            and record.model_version == runner_metadata.get("model_version")
        )

    def write_jsonl(self, path: Path) -> None:
        path = Path(path)
        if self.stream_path is not None:
            self.close()
            if path.resolve() == self.stream_path.resolve():
                return
            if not self.records:
                raise RuntimeError(
                    "Cannot rewrite streamed score cache to a second path without retained records"
                )
        path.parent.mkdir(parents=True, exist_ok=True)
        open_handle = gzip.open if path.suffix == ".gz" else open
        with open_handle(path, "wt", encoding="utf-8") as handle:
            for record in self.records:
                handle.write(self._record_jsonl(record))

    def close(self) -> None:
        if self._stream_handle is not None:
            self._stream_handle.close()
            self._stream_handle = None

    def _write_stream_record(self, record: ReplayGeneratedScoreRecord) -> None:
        if self.stream_path is None:
            return
        if self._stream_handle is None:
            self.stream_path.parent.mkdir(parents=True, exist_ok=True)
            if self.compress:
                self._stream_handle = gzip.open(
                    self.stream_path,
                    "wt",
                    encoding="utf-8",
                )
            else:
                self._stream_handle = self.stream_path.open("w", encoding="utf-8")
        self._stream_handle.write(self._record_jsonl(record))

    @staticmethod
    def _record_jsonl(record: ReplayGeneratedScoreRecord) -> str:
        return json.dumps(record.to_json(), sort_keys=True) + "\n"

    def report_summary(self, *, artifact_path: Optional[Path] = None) -> Mapping[str, Any]:
        model_versions = sorted(self._model_versions)
        code_hashes = sorted(self._model_code_hashes)
        config_hashes = sorted(self._model_config_hashes)
        return {
            "cache_name": self.cache_name,
            "record_count": self._record_count,
            "artifact_path": str(artifact_path) if artifact_path is not None else None,
            "artifact_compression": "gzip" if self.compress else "none",
            "streaming_write": self.stream_path is not None,
            "records_retained_in_memory": self.keep_records,
            "model_versions": model_versions,
            "model_code_hashes": code_hashes,
            "model_config_hashes": config_hashes,
            "input_hashes_recorded": self._input_hash_count,
            "prediction_hashes_recorded": self._prediction_hash_count,
            "reuse_rule": (
                "reuse only when model_version, model_code_hash, model_config_hash, "
                "and point-in-time input_hash all match"
            ),
            "invalidation_rules": [
                "invalidate_on_weather_model_code_hash_change",
                "invalidate_on_weather_model_config_hash_change",
                "invalidate_on_point_in_time_input_hash_change",
                "invalidate_on_feature_provenance_or_source_path_change",
            ],
        }


def validate_weather_model_context_firewall(
    context: WeatherModelDecisionContext,
) -> tuple[str, ...]:
    payload = context.to_hash_payload()
    leaked = tuple(
        sorted(
            field_name
            for field_name in FORBIDDEN_WEATHER_MODEL_CONTEXT_FIELDS
            if field_name in payload
        )
    )
    if leaked:
        raise ValueError(f"Weather model context contains forbidden fields: {', '.join(leaked)}")
    for key in context.forecast_features:
        key_lower = str(key).lower()
        if (
            "resolved" in key_lower
            or "canonical_label" in key_lower
            or "realized" in key_lower
            or key_lower.startswith("actual_")
        ):
            leaked = tuple(sorted((*leaked, f"forecast_features.{key}")))
    if leaked:
        raise ValueError(f"Weather model context contains forbidden fields: {', '.join(leaked)}")
    return leaked


@dataclass(frozen=True)
class CanonicalLabel:
    universe_row_id: str
    market_id: str
    canonical_label: int
    canonical_label_source: str
    recommended_action: str
    exclude_from_backtest_replay: bool
    canonical_bucket_type: str
    source_family: str
    source_station: str
    source_url: str
    settlement_unit: str
    settlement_precision: str
    resolution_date_text: str
    resolution_date_conflict: bool
    open_meteo_current_continuous_label: Optional[int]
    open_meteo_source_rule_proxy_label: Optional[int]
    audit_flags: tuple[str, ...] = ()

    @property
    def is_replay_allowed(self) -> bool:
        return (
            not self.exclude_from_backtest_replay
            and self.recommended_action != "exclude_or_manual_review"
        )

    @property
    def is_polymarket_canonical(self) -> bool:
        source = self.canonical_label_source.lower()
        return "polymarket" in source or "telonex" in source

    def payout_for_side(self, side: str) -> int:
        side_upper = side.upper()
        if side_upper == "YES":
            return int(self.canonical_label)
        if side_upper == "NO":
            return 1 - int(self.canonical_label)
        raise ValueError(f"Unsupported token side: {side}")

    @classmethod
    def from_csv_row(cls, row: Mapping[str, str]) -> "CanonicalLabel":
        return cls(
            universe_row_id=row.get("universe_row_id", ""),
            market_id=row.get("market_id", ""),
            canonical_label=_int_value(row.get("canonical_label")),
            canonical_label_source=row.get("canonical_label_source", ""),
            recommended_action=row.get("recommended_action", ""),
            exclude_from_backtest_replay=_boolish(row.get("exclude_from_backtest_replay")),
            canonical_bucket_type=row.get("canonical_bucket_type", ""),
            source_family=row.get("source_family", ""),
            source_station=row.get("source_station", ""),
            source_url=row.get("source_url", ""),
            settlement_unit=row.get("settlement_unit", ""),
            settlement_precision=row.get("settlement_precision", ""),
            resolution_date_text=row.get("resolution_date_text", ""),
            resolution_date_conflict=_boolish(row.get("resolution_date_conflict")),
            open_meteo_current_continuous_label=_optional_int(
                row.get("open_meteo_current_continuous_label")
            ),
            open_meteo_source_rule_proxy_label=_optional_int(
                row.get("open_meteo_source_rule_proxy_label")
            ),
            audit_flags=_pipe_flags(row.get("audit_flags")),
        )

    @classmethod
    def from_sqlite_row(cls, row: Mapping[str, Any]) -> Optional["CanonicalLabel"]:
        if row.get("canonical_label") in (None, ""):
            return None
        return cls(
            universe_row_id=str(row.get("universe_row_id") or ""),
            market_id=str(row.get("market_id") or ""),
            canonical_label=_int_value(row.get("canonical_label")),
            canonical_label_source=str(row.get("canonical_label_source") or ""),
            recommended_action=str(row.get("recommended_action") or ""),
            exclude_from_backtest_replay=_boolish(row.get("exclude_from_backtest_replay")),
            canonical_bucket_type=str(row.get("canonical_bucket_type") or ""),
            source_family=str(row.get("canonical_source_family") or row.get("source_family") or ""),
            source_station=str(row.get("canonical_source_station") or row.get("source_station") or ""),
            source_url=str(row.get("canonical_source_url") or row.get("source_url") or ""),
            settlement_unit=str(row.get("settlement_unit") or ""),
            settlement_precision=str(row.get("settlement_precision") or ""),
            resolution_date_text=str(row.get("resolution_date_text") or ""),
            resolution_date_conflict=_boolish(row.get("resolution_date_conflict")),
            open_meteo_current_continuous_label=_optional_int(
                row.get("open_meteo_current_continuous_label")
            ),
            open_meteo_source_rule_proxy_label=_optional_int(
                row.get("open_meteo_source_rule_proxy_label")
            ),
            audit_flags=_pipe_flags(row.get("settlement_audit_flags") or row.get("audit_flags")),
        )


@dataclass(frozen=True)
class DecisionSnapshot:
    decision_id: str
    universe_row_id: str
    market_id: Optional[str]
    slug: Optional[str]
    question: Optional[str]
    normalized_city: Optional[str]
    country_region: Optional[str]
    timezone: Optional[str]
    market_date_local: Optional[str]
    bucket_label: Optional[str]
    bucket_type: Optional[str]
    threshold_low_f: Optional[float]
    threshold_high_f: Optional[float]
    side: str
    yes_token_id: Optional[str]
    no_token_id: Optional[str]
    token_id: str
    decision_time_utc: datetime
    decision_hour_utc: Optional[int]
    lead_days: Optional[int]
    quote_store: Optional[str]
    quote: Optional[QuoteSnapshot]
    forecast: Optional[ForecastSnapshot]
    data_quality_flags: tuple[str, ...] = ()
    parser_confidence: Optional[float] = None
    observed_high: Optional[ObservedHighSnapshot] = None
    observed_high_source_payload_id: Optional[str] = None
    observed_high_raw_cache_path: Optional[str] = None
    observed_high_data_quality_flags: tuple[str, ...] = ()
    observed_high_missing_reason: Optional[str] = None
    decision_local_hour: Optional[float] = None
    realized_temperature_2m_max_f: Optional[float] = None
    resolved_outcome: Optional[int] = None
    settlement_source: Optional[str] = None

    @classmethod
    def from_sqlite_row(cls, row: Mapping[str, Any]) -> "DecisionSnapshot":
        quote_timestamp = _parse_utc(row.get("quote_timestamp_utc"))
        quote = None
        if quote_timestamp is not None:
            quote = QuoteSnapshot(
                timestamp_utc=quote_timestamp,
                bid=_optional_float(row.get("bid")),
                ask=_optional_float(row.get("ask")),
                mid=_optional_float(row.get("mid")),
                last_price=_optional_float(row.get("last_price")),
                spread=_optional_float(row.get("spread")),
                bid_size=_optional_float(row.get("bid_size")),
                ask_size=_optional_float(row.get("ask_size")),
                liquidity_depth=_optional_float(row.get("liquidity_depth")),
                source_payload_id=row.get("quote_source_payload_id"),
                source_cache_path=row.get("quote_source_cache_path"),
                data_quality_flags=_json_flags(row.get("quote_data_quality_flags")),
            )

        forecast_run_time = _parse_utc(row.get("forecast_run_time_utc"))
        forecast = None
        if forecast_run_time is not None:
            forecast = ForecastSnapshot(
                run_time_utc=forecast_run_time,
                source_name=row.get("forecast_source_name"),
                temperature_2m_max_f=_optional_float(row.get("forecast_temperature_2m_max_f")),
                features=_json_mapping(row.get("forecast_features_json")),
            )

        observed_high = None
        observed_high_timestamp = _parse_utc(row.get("observed_high_observed_at_utc"))
        observed_high_value = _optional_float(row.get("observed_high_so_far_f"))
        if observed_high_timestamp is not None and observed_high_value is not None:
            observed_high = ObservedHighSnapshot(
                observed_high_f=observed_high_value,
                observed_high_timestamp_utc=observed_high_timestamp,
                observation_source=str(row.get("observed_high_source") or ""),
                observation_final=_boolish(row.get("observed_high_is_final_at_decision")),
                source_match_confidence=None,
                source_family=row.get("canonical_source_family"),
                source_station=row.get("canonical_source_station"),
                deterministic_state=None,
            )

        return cls(
            decision_id=str(row.get("decision_id") or ""),
            universe_row_id=str(row.get("universe_row_id") or ""),
            market_id=row.get("market_id"),
            slug=row.get("slug"),
            question=row.get("question"),
            normalized_city=row.get("normalized_city"),
            country_region=row.get("country_region"),
            timezone=row.get("timezone"),
            market_date_local=row.get("market_date_local"),
            bucket_label=row.get("bucket_label"),
            bucket_type=row.get("bucket_type"),
            threshold_low_f=_optional_float(row.get("threshold_low_f")),
            threshold_high_f=_optional_float(row.get("threshold_high_f")),
            side=str(row.get("side") or "").upper(),
            yes_token_id=row.get("yes_token_id"),
            no_token_id=row.get("no_token_id"),
            token_id=str(row.get("token_id") or ""),
            decision_time_utc=_require_utc(row.get("decision_time_utc"), "decision_time_utc"),
            decision_hour_utc=_optional_int(row.get("decision_hour_utc")),
            lead_days=_optional_int(row.get("lead_days")),
            quote_store=row.get("quote_store"),
            quote=quote,
            forecast=forecast,
            data_quality_flags=_json_flags(row.get("data_quality_flags")),
            parser_confidence=_optional_float(row.get("parser_confidence")),
            observed_high=observed_high,
            observed_high_source_payload_id=row.get("observed_high_source_payload_id"),
            observed_high_raw_cache_path=row.get("observed_high_raw_cache_path"),
            observed_high_data_quality_flags=_json_flags(row.get("observed_high_data_quality_flags")),
            observed_high_missing_reason=row.get("observed_high_missing_reason"),
            decision_local_hour=_optional_float(row.get("decision_local_hour")),
            realized_temperature_2m_max_f=_optional_float(row.get("realized_temperature_2m_max_f")),
            resolved_outcome=_optional_int(row.get("resolved_outcome")),
            settlement_source=row.get("settlement_source"),
        )


@dataclass(frozen=True)
class StrategyInput:
    decision_id: str
    universe_row_id: str
    market_id: Optional[str]
    slug: Optional[str]
    question: Optional[str]
    normalized_city: Optional[str]
    country_region: Optional[str]
    timezone: Optional[str]
    market_date_local: Optional[str]
    bucket_label: Optional[str]
    bucket_type: Optional[str]
    threshold_low_f: Optional[float]
    threshold_high_f: Optional[float]
    side: str
    token_id: str
    decision_time_utc: datetime
    decision_hour_utc: Optional[int]
    lead_days: Optional[int]
    quote: QuoteSnapshot
    forecast: ForecastSnapshot
    score: Optional[SharedScoreContract] = None


@dataclass(frozen=True)
class OrderIntent:
    strategy_name: str
    decision_id: str
    token_id: str
    side: str
    action: OrderAction
    requested_notional_usd: Optional[float] = None
    requested_shares: Optional[float] = None
    limit_price: Optional[float] = None
    reason: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StrategyDecision:
    orders: tuple[OrderIntent, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


class StrategyPlugin(Protocol):
    @property
    def name(self) -> str:
        ...

    def evaluate(self, snapshot: StrategyInput) -> StrategyDecision:
        ...


class NoopStrategyPlugin:
    name = "noop"

    def evaluate(self, snapshot: StrategyInput) -> StrategyDecision:
        return StrategyDecision()


@dataclass(frozen=True)
class ConservativeFVEdgeConfig:
    order_notional_usd: float = 1.0
    allowed_lead_days: tuple[int, ...] = (1, 2)
    yes_min_probability: float = 0.70
    no_min_probability: float = 0.86
    no_counter_event_probability_max: float = 0.14
    model_count_min: int = 3
    model_agreement_min: float = 1.00
    open_bucket_probability_stdev_max: float = 0.20
    bounded_probability_stdev_max: float = 0.10
    bounded_min_probability: float = 0.98
    bounded_min_ask: float = 0.70
    bounded_min_net_edge: float = 0.10
    max_spread: float = 0.10
    min_ask: float = 0.125
    yes_side_min_ask: float = 0.20
    max_ask: float = 0.95
    no_side_max_ask: float = 0.94
    min_net_edge_floor: float = 0.03
    min_net_edge_fraction_of_remaining: float = 0.08
    uncertainty_buffer_base: float = 0.02
    probability_stdev_buffer_multiplier: float = 0.50

    def as_report_dict(self) -> Mapping[str, Any]:
        return {
            "order_notional_usd": self.order_notional_usd,
            "allowed_lead_days": list(self.allowed_lead_days),
            "yes_min_probability": self.yes_min_probability,
            "no_min_probability": self.no_min_probability,
            "no_counter_event_probability_max": self.no_counter_event_probability_max,
            "model_count_min": self.model_count_min,
            "model_agreement_min": self.model_agreement_min,
            "open_bucket_probability_stdev_max": self.open_bucket_probability_stdev_max,
            "bounded_probability_stdev_max": self.bounded_probability_stdev_max,
            "bounded_min_probability": self.bounded_min_probability,
            "bounded_min_ask": self.bounded_min_ask,
            "bounded_min_net_edge": self.bounded_min_net_edge,
            "max_spread": self.max_spread,
            "min_ask": self.min_ask,
            "yes_side_min_ask": self.yes_side_min_ask,
            "max_ask": self.max_ask,
            "no_side_max_ask": self.no_side_max_ask,
            "min_net_edge_floor": self.min_net_edge_floor,
            "min_net_edge_fraction_of_remaining": self.min_net_edge_fraction_of_remaining,
            "uncertainty_buffer_base": self.uncertainty_buffer_base,
            "probability_stdev_buffer_multiplier": self.probability_stdev_buffer_multiplier,
        }


class ConservativeFVEdgeStrategy:
    name = "conservative_fv_edge_baseline"

    def __init__(self, config: ConservativeFVEdgeConfig = ConservativeFVEdgeConfig()) -> None:
        self.config = config

    def evaluate(self, snapshot: StrategyInput) -> StrategyDecision:
        reasons: list[SkipReason] = []
        details: dict[str, Any] = {}
        quote = snapshot.quote
        score = snapshot.score
        if score is None:
            return _strategy_skip(SkipReason.MISSING_SHARED_SCORE)
        if snapshot.lead_days not in self.config.allowed_lead_days:
            reasons.append(SkipReason.UNSUPPORTED_LEAD_DAY)
        if quote.ask is None or quote.bid is None:
            reasons.append(SkipReason.MISSING_EXECUTABLE_PRICE)
        if quote.spread is None or quote.spread > self.config.max_spread:
            reasons.append(SkipReason.SPREAD_TOO_WIDE)
        if quote.ask is not None:
            min_ask = self.config.yes_side_min_ask if snapshot.side == "YES" else self.config.min_ask
            max_ask = self.config.no_side_max_ask if snapshot.side == "NO" else self.config.max_ask
            if quote.ask < min_ask:
                reasons.append(SkipReason.ASK_BELOW_MINIMUM)
            if quote.ask > max_ask:
                reasons.append(SkipReason.ASK_ABOVE_MAXIMUM)
        if score.model_count < self.config.model_count_min:
            reasons.append(SkipReason.MODEL_COUNT_LOW)
        if score.model_agreement < self.config.model_agreement_min:
            reasons.append(SkipReason.MODEL_AGREEMENT_LOW)
        if score.probability_stdev is None:
            reasons.append(SkipReason.PROBABILITY_STDEV_HIGH)
        elif _is_bounded_bucket(snapshot.bucket_type):
            if score.probability_stdev > self.config.bounded_probability_stdev_max:
                reasons.append(SkipReason.PROBABILITY_STDEV_HIGH)
        elif score.probability_stdev > self.config.open_bucket_probability_stdev_max:
            reasons.append(SkipReason.PROBABILITY_STDEV_HIGH)

        p_side = score.p_side
        if snapshot.side == "YES":
            if score.p_yes < self.config.yes_min_probability:
                reasons.append(SkipReason.FAIR_VALUE_BELOW_MINIMUM)
        elif snapshot.side == "NO":
            if score.p_no < self.config.no_min_probability:
                reasons.append(SkipReason.FAIR_VALUE_BELOW_MINIMUM)
            if score.p_yes > self.config.no_counter_event_probability_max:
                reasons.append(SkipReason.COUNTER_EVENT_PROBABILITY_HIGH)
        else:
            reasons.append(SkipReason.MISSING_MARKET_FIELDS)

        entry_edge = None
        uncertainty_buffer = None
        net_edge = None
        edge_floor = None
        if quote.ask is not None:
            entry_edge = p_side - quote.ask
            uncertainty_buffer = self._uncertainty_buffer(score)
            net_edge = entry_edge - uncertainty_buffer
            edge_floor = self._edge_floor(quote.ask)
            if net_edge < edge_floor:
                reasons.append(SkipReason.NET_EDGE_BELOW_MINIMUM)
            if _is_bounded_bucket(snapshot.bucket_type):
                if (
                    p_side < self.config.bounded_min_probability
                    or quote.ask < self.config.bounded_min_ask
                    or net_edge < self.config.bounded_min_net_edge
                ):
                    reasons.append(SkipReason.BOUNDED_BUCKET_GATES_NOT_MET)
        details.update(
            {
                "p_yes": score.p_yes,
                "p_side": p_side,
                "entry_edge": entry_edge,
                "uncertainty_buffer": uncertainty_buffer,
                "net_edge": net_edge,
                "edge_floor": edge_floor,
                "quote_source_payload_id": quote.source_payload_id,
                "quote_source_cache_path": quote.source_cache_path,
            }
        )

        unique_reasons = tuple(dict.fromkeys(reasons))
        if unique_reasons:
            return StrategyDecision(
                metadata={
                    "candidate_skip_reasons": [reason.value for reason in unique_reasons],
                    "candidate_gate_details": details,
                }
            )
        return StrategyDecision(
            orders=(
                OrderIntent(
                    strategy_name=self.name,
                    decision_id=snapshot.decision_id,
                    token_id=snapshot.token_id,
                    side=snapshot.side,
                    action=OrderAction.BUY,
                    requested_notional_usd=self.config.order_notional_usd,
                    limit_price=quote.ask,
                    reason="conservative_fv_edge_baseline_entry",
                    metadata=details,
                ),
            ),
            metadata={"candidate_gate_details": details},
        )

    def _edge_floor(self, ask: float) -> float:
        return max(
            self.config.min_net_edge_floor,
            self.config.min_net_edge_fraction_of_remaining * (1.0 - ask),
        )

    def _uncertainty_buffer(self, score: SharedScoreContract) -> float:
        return self.config.uncertainty_buffer_base + (
            (score.probability_stdev or 0.0) * self.config.probability_stdev_buffer_multiplier
        )


@dataclass(frozen=True)
class HighConfidenceNOTailConfig:
    order_notional_usd: float = 1.0
    sizing_mode: str = "flat"
    sizing_bankroll_usd: float = 100.0
    kelly_fraction: float = 0.50
    min_order_notional_usd: float = 1.0
    max_order_notional_usd: Optional[float] = None
    allowed_lead_days: tuple[int, ...] = (1, 2)
    no_min_probability: float = 0.86
    no_counter_event_probability_max: float = 0.14
    model_count_min: int = 3
    model_agreement_min: float = 1.00
    open_bucket_probability_stdev_max: float = 0.20
    bounded_probability_stdev_max: float = 0.10
    bounded_min_probability: float = 0.98
    bounded_min_ask: float = 0.70
    bounded_min_net_edge: float = 0.10
    max_spread: float = 0.10
    min_ask: float = 0.125
    no_side_max_ask: float = 0.94
    min_net_edge: float = 0.05

    def as_report_dict(self) -> Mapping[str, Any]:
        return {
            "order_notional_usd": self.order_notional_usd,
            "sizing_mode": self.sizing_mode,
            "sizing_bankroll_usd": self.sizing_bankroll_usd,
            "kelly_fraction": self.kelly_fraction,
            "min_order_notional_usd": self.min_order_notional_usd,
            "max_order_notional_usd": self.max_order_notional_usd,
            "allowed_lead_days": list(self.allowed_lead_days),
            "no_min_probability": self.no_min_probability,
            "no_counter_event_probability_max": self.no_counter_event_probability_max,
            "model_count_min": self.model_count_min,
            "model_agreement_min": self.model_agreement_min,
            "open_bucket_probability_stdev_max": self.open_bucket_probability_stdev_max,
            "bounded_probability_stdev_max": self.bounded_probability_stdev_max,
            "bounded_min_probability": self.bounded_min_probability,
            "bounded_min_ask": self.bounded_min_ask,
            "bounded_min_net_edge": self.bounded_min_net_edge,
            "max_spread": self.max_spread,
            "min_ask": self.min_ask,
            "no_side_max_ask": self.no_side_max_ask,
            "min_net_edge": self.min_net_edge,
        }


class HighConfidenceNOTailStrategy:
    name = "old_high_confidence_no_tail"

    def __init__(self, config: HighConfidenceNOTailConfig = HighConfidenceNOTailConfig()) -> None:
        self.config = config

    def evaluate(self, snapshot: StrategyInput) -> StrategyDecision:
        reasons: list[SkipReason] = []
        details: dict[str, Any] = {}
        quote = snapshot.quote
        score = snapshot.score
        if score is None:
            return _strategy_skip(SkipReason.MISSING_SHARED_SCORE)
        if snapshot.side != "NO":
            reasons.append(SkipReason.UNSUPPORTED_SIDE)
        if snapshot.lead_days not in self.config.allowed_lead_days:
            reasons.append(SkipReason.UNSUPPORTED_LEAD_DAY)
        if quote.ask is None or quote.bid is None:
            reasons.append(SkipReason.MISSING_EXECUTABLE_PRICE)
        if quote.spread is None or quote.spread > self.config.max_spread:
            reasons.append(SkipReason.SPREAD_TOO_WIDE)
        if quote.ask is not None:
            if quote.ask < self.config.min_ask:
                reasons.append(SkipReason.ASK_BELOW_MINIMUM)
            if quote.ask > self.config.no_side_max_ask:
                reasons.append(SkipReason.ASK_ABOVE_MAXIMUM)
        if score.model_count < self.config.model_count_min:
            reasons.append(SkipReason.MODEL_COUNT_LOW)
        if score.model_agreement < self.config.model_agreement_min:
            reasons.append(SkipReason.MODEL_AGREEMENT_LOW)
        if score.probability_stdev is None:
            reasons.append(SkipReason.PROBABILITY_STDEV_HIGH)
        elif _is_bounded_bucket(snapshot.bucket_type):
            if score.probability_stdev > self.config.bounded_probability_stdev_max:
                reasons.append(SkipReason.PROBABILITY_STDEV_HIGH)
        elif score.probability_stdev > self.config.open_bucket_probability_stdev_max:
            reasons.append(SkipReason.PROBABILITY_STDEV_HIGH)

        entry_edge = None
        net_edge = None
        requested_notional = None
        if score.p_no < self.config.no_min_probability:
            reasons.append(SkipReason.FAIR_VALUE_BELOW_MINIMUM)
        if score.p_yes > self.config.no_counter_event_probability_max:
            reasons.append(SkipReason.COUNTER_EVENT_PROBABILITY_HIGH)
        if quote.ask is not None:
            entry_edge = score.p_no - quote.ask
            net_edge = entry_edge
            if net_edge < self.config.min_net_edge:
                reasons.append(SkipReason.NET_EDGE_BELOW_MINIMUM)
            if _is_bounded_bucket(snapshot.bucket_type):
                if (
                    score.p_no < self.config.bounded_min_probability
                    or quote.ask < self.config.bounded_min_ask
                    or net_edge < self.config.bounded_min_net_edge
                ):
                    reasons.append(SkipReason.BOUNDED_BUCKET_GATES_NOT_MET)
            requested_notional = self._requested_notional(ask=quote.ask, p_no=score.p_no)
            if requested_notional < self.config.min_order_notional_usd:
                reasons.append(SkipReason.REQUESTED_NOTIONAL_BELOW_MINIMUM)

        details.update(
            {
                "p_yes": score.p_yes,
                "p_no": score.p_no,
                "p_side": score.p_side,
                "entry_edge": entry_edge,
                "net_edge": net_edge,
                "quote_source_payload_id": quote.source_payload_id,
                "quote_source_cache_path": quote.source_cache_path,
                "sizing_mode": self.config.sizing_mode,
                "kelly_fraction": self.config.kelly_fraction,
                "requested_notional_usd": requested_notional,
            }
        )
        unique_reasons = tuple(dict.fromkeys(reasons))
        if unique_reasons:
            return StrategyDecision(
                metadata={
                    "candidate_skip_reasons": [reason.value for reason in unique_reasons],
                    "candidate_gate_details": details,
                }
            )
        return StrategyDecision(
            orders=(
                OrderIntent(
                    strategy_name=self.name,
                    decision_id=snapshot.decision_id,
                    token_id=snapshot.token_id,
                    side=snapshot.side,
                    action=OrderAction.BUY,
                    requested_notional_usd=requested_notional,
                    limit_price=quote.ask,
                    reason="old_high_confidence_no_tail_entry",
                    metadata=details,
                ),
            ),
            metadata={"candidate_gate_details": details},
        )

    def _requested_notional(self, *, ask: float, p_no: float) -> float:
        if self.config.sizing_mode == "flat":
            notional = self.config.order_notional_usd
        elif self.config.sizing_mode == "fractional_kelly":
            if ask >= 1.0:
                return 0.0
            raw_kelly = max(0.0, (p_no - ask) / max(1e-9, 1.0 - ask))
            notional = self.config.sizing_bankroll_usd * self.config.kelly_fraction * raw_kelly
        else:
            return 0.0
        if self.config.max_order_notional_usd is not None:
            notional = min(notional, self.config.max_order_notional_usd)
        return max(0.0, notional)


@dataclass(frozen=True)
class ReplayFrame:
    decision: DecisionSnapshot
    canonical_label: Optional[CanonicalLabel]
    skip_reasons: tuple[SkipReason, ...]
    score: Optional[SharedScoreContract] = None

    def strategy_input(self) -> StrategyInput:
        if self.decision.quote is None:
            raise ValueError("Cannot build StrategyInput without a quote snapshot")
        if self.decision.forecast is None:
            raise ValueError("Cannot build StrategyInput without a forecast snapshot")
        return StrategyInput(
            decision_id=self.decision.decision_id,
            universe_row_id=self.decision.universe_row_id,
            market_id=self.decision.market_id,
            slug=self.decision.slug,
            question=self.decision.question,
            normalized_city=self.decision.normalized_city,
            country_region=self.decision.country_region,
            timezone=self.decision.timezone,
            market_date_local=self.decision.market_date_local,
            bucket_label=self.decision.bucket_label,
            bucket_type=self.decision.bucket_type,
            threshold_low_f=self.decision.threshold_low_f,
            threshold_high_f=self.decision.threshold_high_f,
            side=self.decision.side,
            token_id=self.decision.token_id,
            decision_time_utc=self.decision.decision_time_utc,
            decision_hour_utc=self.decision.decision_hour_utc,
            lead_days=self.decision.lead_days,
            quote=self.decision.quote,
            forecast=self.decision.forecast,
            score=self.score,
        )


@dataclass(frozen=True)
class LeakageViolation:
    code: str
    decision_id: str
    universe_row_id: str
    details: str
    severity: str = "error"


class LeakageValidator(Protocol):
    def validate(self, frame: ReplayFrame) -> Sequence[LeakageViolation]:
        ...


@dataclass(frozen=True)
class DefaultLeakageValidator:
    require_polymarket_labels: bool = True

    def validate(self, frame: ReplayFrame) -> Sequence[LeakageViolation]:
        decision = frame.decision
        violations: list[LeakageViolation] = []
        if decision.quote and decision.quote.timestamp_utc > decision.decision_time_utc:
            violations.append(
                _violation(
                    decision,
                    "future_quote",
                    f"quote timestamp {decision.quote.timestamp_utc.isoformat()} is after decision",
                )
            )
        if decision.forecast and decision.forecast.run_time_utc > decision.decision_time_utc:
            violations.append(
                _violation(
                    decision,
                    "future_forecast",
                    f"forecast run {decision.forecast.run_time_utc.isoformat()} is after decision",
                )
            )
        if (
            self.require_polymarket_labels
            and frame.canonical_label is not None
            and not frame.canonical_label.is_polymarket_canonical
        ):
            violations.append(
                _violation(
                    decision,
                    "non_canonical_label_source",
                    frame.canonical_label.canonical_label_source,
                )
            )
        if decision.forecast:
            leaked_keys = sorted(
                key
                for key in decision.forecast.features
                if "realized" in key.lower() or key.lower().startswith("actual_")
            )
            if leaked_keys:
                violations.append(
                    _violation(
                        decision,
                        "forecast_contains_realized_feature",
                        ",".join(leaked_keys),
                    )
                )
        if frame.score:
            if frame.score.scored_at_utc and frame.score.scored_at_utc > decision.decision_time_utc:
                violations.append(
                    _violation(
                        decision,
                        "future_score",
                        frame.score.scored_at_utc.isoformat(),
                    )
                )
            if (
                frame.score.observed_high
                and frame.score.observed_high.observed_high_timestamp_utc > decision.decision_time_utc
            ):
                violations.append(
                    _violation(
                        decision,
                        "future_observed_high",
                        frame.score.observed_high.observed_high_timestamp_utc.isoformat(),
                    )
                )
            leaked_score_keys = sorted(
                key
                for key in frame.score.metadata
                if "resolved" in key.lower()
                or "canonical_label" in key.lower()
                or "realized_high" in key.lower()
            )
            if leaked_score_keys:
                violations.append(
                    _violation(
                        decision,
                        "score_metadata_contains_outcome_key",
                        ",".join(leaked_score_keys),
                    )
                )
        return violations


@dataclass(frozen=True)
class PointInTimePolicy:
    max_quote_staleness_seconds: int = 90 * 60
    require_quote: bool = True
    require_forecast: bool = True
    require_canonical_label: bool = True
    require_executable_bid_ask: bool = True
    require_shared_score: bool = False

    def skip_reasons(
        self,
        decision: DecisionSnapshot,
        label: Optional[CanonicalLabel],
        score: Optional[SharedScoreContract] = None,
    ) -> tuple[SkipReason, ...]:
        reasons: list[SkipReason] = []
        if self.require_shared_score and score is None:
            reasons.append(SkipReason.MISSING_SHARED_SCORE)
        if score is not None and not score.is_valid:
            reasons.append(SkipReason.INVALID_SHARED_SCORE)
        if self.require_quote and decision.quote is None:
            reasons.append(SkipReason.MISSING_QUOTE_SNAPSHOT)
        if self.require_forecast and decision.forecast is None:
            reasons.append(SkipReason.MISSING_FORECAST_SNAPSHOT)
        if self.require_canonical_label and label is None:
            reasons.append(SkipReason.MISSING_CANONICAL_LABEL)
        if label is not None:
            if not label.is_replay_allowed:
                reasons.append(SkipReason.CANONICAL_LABEL_EXCLUDED)
            if not label.is_polymarket_canonical:
                reasons.append(SkipReason.NON_CANONICAL_LABEL_SOURCE)
        if decision.quote is not None:
            if decision.quote.timestamp_utc > decision.decision_time_utc:
                reasons.append(SkipReason.FUTURE_QUOTE)
            stale_seconds = (
                decision.decision_time_utc - decision.quote.timestamp_utc
            ).total_seconds()
            if stale_seconds > self.max_quote_staleness_seconds:
                reasons.append(SkipReason.STALE_QUOTE)
            if self.require_executable_bid_ask and not decision.quote.has_executable_bid_ask:
                reasons.append(SkipReason.MISSING_EXECUTABLE_PRICE)
        if (
            decision.forecast is not None
            and decision.forecast.run_time_utc > decision.decision_time_utc
        ):
            reasons.append(SkipReason.FUTURE_FORECAST)
        if score and score.scored_at_utc and score.scored_at_utc > decision.decision_time_utc:
            reasons.append(SkipReason.LEAKAGE_VIOLATION)
        if (
            score
            and score.observed_high
            and score.observed_high.observed_high_timestamp_utc > decision.decision_time_utc
        ):
            reasons.append(SkipReason.FUTURE_OBSERVED_HIGH)
        return tuple(dict.fromkeys(reasons))


@dataclass(frozen=True)
class UniversalGateConfig:
    parser_confidence_min: float = 0.90
    max_quote_age_seconds: int = 90 * 60
    base_max_spread: float = 0.10
    strict_max_spread: float = 0.05
    min_ask: float = 0.125
    yes_side_min_ask: float = 0.20
    max_ask: float = 0.95
    no_side_max_ask: float = 0.94
    base_model_count_min: int = 3
    forecast_only_model_agreement_min: float = 1.00
    same_day_model_agreement_min: float = 0.80
    same_day_observed_high_confidence_min: float = 0.75
    late_day_heat_window_exhaustion_min: float = 0.80
    bounded_model_agreement_min: float = 1.00
    bounded_probability_stdev_max: Optional[float] = None
    nominal_order_usd_for_size_check: Optional[float] = None


@dataclass(frozen=True)
class GateCheck:
    name: str
    passed: bool
    reason: Optional[SkipReason] = None
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class UniversalGateResult:
    family: StrategyFamily
    passed: bool
    skip_reasons: tuple[SkipReason, ...]
    checks: tuple[GateCheck, ...]


def evaluate_universal_gates(
    frame: ReplayFrame,
    *,
    family: StrategyFamily,
    config: UniversalGateConfig = UniversalGateConfig(),
) -> UniversalGateResult:
    checks: list[GateCheck] = []
    reasons: list[SkipReason] = list(frame.skip_reasons)
    decision = frame.decision
    quote = decision.quote
    label = frame.canonical_label
    score = frame.score

    def add_check(
        name: str,
        passed: bool,
        reason: Optional[SkipReason] = None,
        **details: Any,
    ) -> None:
        checks.append(GateCheck(name=name, passed=passed, reason=reason, details=details))
        if not passed and reason is not None:
            reasons.append(reason)

    market_fields_present = all(
        (
            decision.normalized_city,
            decision.timezone,
            decision.market_date_local,
            decision.token_id,
            decision.yes_token_id,
            decision.no_token_id,
        )
    )
    add_check("market_fields_present", bool(market_fields_present), SkipReason.MISSING_MARKET_FIELDS)

    parser_confidence = (
        score.parser_confidence if score and score.parser_confidence is not None else decision.parser_confidence
    )
    add_check(
        "parser_confidence",
        parser_confidence is not None and parser_confidence >= config.parser_confidence_min,
        SkipReason.PARSER_CONFIDENCE_LOW,
        parser_confidence=parser_confidence,
        minimum=config.parser_confidence_min,
    )

    settlement_precision_ok = True
    if (decision.bucket_type or "").lower() in {"exact", "range"}:
        settlement_precision_ok = bool(
            (score and score.settlement_unit and score.settlement_precision)
            or (label and label.settlement_unit and label.settlement_precision)
        )
    add_check(
        "settlement_precision_known",
        settlement_precision_ok,
        SkipReason.MISSING_SETTLEMENT_PRECISION,
    )

    label_conflict = bool(
        label
        and (
            label.resolution_date_conflict
            or "contradictory" in "|".join(label.audit_flags).lower()
        )
    )
    add_check(
        "settlement_source_not_conflicted",
        not label_conflict,
        SkipReason.SETTLEMENT_SOURCE_CONFLICT,
    )

    if quote is None:
        add_check("quote_present", False, SkipReason.MISSING_QUOTE_SNAPSHOT)
    else:
        quote_age = (decision.decision_time_utc - quote.timestamp_utc).total_seconds()
        add_check(
            "quote_point_in_time",
            quote.timestamp_utc <= decision.decision_time_utc,
            SkipReason.FUTURE_QUOTE,
            quote_timestamp_utc=quote.timestamp_utc.isoformat(),
        )
        add_check(
            "quote_fresh",
            0 <= quote_age <= config.max_quote_age_seconds,
            SkipReason.STALE_QUOTE,
            quote_age_seconds=quote_age,
            max_quote_age_seconds=config.max_quote_age_seconds,
        )
        add_check("bid_ask_present", quote.has_executable_bid_ask, SkipReason.MISSING_EXECUTABLE_PRICE)
        if quote.spread is not None:
            max_spread = config.strict_max_spread if family == StrategyFamily.LATE_DAY_HEAT_WINDOW else config.base_max_spread
            add_check(
                "spread_within_limit",
                quote.spread <= max_spread,
                SkipReason.SPREAD_TOO_WIDE,
                spread=quote.spread,
                max_spread=max_spread,
            )
        if quote.ask is not None:
            min_ask = config.yes_side_min_ask if decision.side == "YES" else config.min_ask
            max_ask = config.no_side_max_ask if decision.side == "NO" else config.max_ask
            add_check("ask_above_minimum", quote.ask >= min_ask, SkipReason.ASK_BELOW_MINIMUM, ask=quote.ask, min_ask=min_ask)
            add_check("ask_below_maximum", quote.ask <= max_ask, SkipReason.ASK_ABOVE_MAXIMUM, ask=quote.ask, max_ask=max_ask)
        if config.nominal_order_usd_for_size_check is not None and quote.ask:
            required_shares = config.nominal_order_usd_for_size_check / quote.ask
            add_check(
                "visible_ask_size_covers_nominal_order",
                quote.ask_size is not None and quote.ask_size >= required_shares,
                SkipReason.INSUFFICIENT_VISIBLE_SIZE,
                required_shares=required_shares,
                ask_size=quote.ask_size,
            )

    if score is None:
        add_check("shared_score_present", False, SkipReason.MISSING_SHARED_SCORE)
    else:
        add_check("shared_score_valid", score.is_valid, SkipReason.INVALID_SHARED_SCORE, errors=score.score_errors)
        add_check(
            "model_count_minimum",
            score.model_count >= config.base_model_count_min,
            SkipReason.MODEL_COUNT_LOW,
            model_count=score.model_count,
            minimum=config.base_model_count_min,
        )
        agreement_min = _agreement_minimum(family, decision.bucket_type, config)
        add_check(
            "model_agreement_minimum",
            score.model_agreement >= agreement_min,
            SkipReason.MODEL_AGREEMENT_LOW,
            model_agreement=score.model_agreement,
            minimum=agreement_min,
        )
        if config.bounded_probability_stdev_max is not None and (decision.bucket_type or "").lower() in {"exact", "range"}:
            add_check(
                "bounded_probability_stdev",
                score.probability_stdev is not None
                and score.probability_stdev <= config.bounded_probability_stdev_max,
                SkipReason.PROBABILITY_STDEV_HIGH,
                probability_stdev=score.probability_stdev,
                maximum=config.bounded_probability_stdev_max,
            )
        if family in {StrategyFamily.SAME_DAY_OBSERVED_HIGH, StrategyFamily.LATE_DAY_HEAT_WINDOW}:
            add_check(
                "observed_high_present",
                score.observed_high is not None,
                SkipReason.MISSING_OBSERVED_HIGH_SNAPSHOT,
            )
            if score.observed_high is not None:
                add_check(
                    "observed_high_point_in_time",
                    score.observed_high.observed_high_timestamp_utc <= decision.decision_time_utc,
                    SkipReason.FUTURE_OBSERVED_HIGH,
                )
            add_check(
                "observed_high_aware_confidence",
                score.observed_high_aware_confidence is not None
                and score.observed_high_aware_confidence >= config.same_day_observed_high_confidence_min,
                SkipReason.OBSERVED_HIGH_CONFIDENCE_LOW,
                confidence=score.observed_high_aware_confidence,
                minimum=config.same_day_observed_high_confidence_min,
            )
        if family == StrategyFamily.LATE_DAY_HEAT_WINDOW:
            heat = score.heat_window
            add_check("heat_window_present", heat is not None, SkipReason.MISSING_HEAT_WINDOW_FIELDS)
            add_check(
                "heat_window_exhaustion",
                heat is not None
                and heat.heat_window_exhaustion_score is not None
                and heat.heat_window_exhaustion_score >= config.late_day_heat_window_exhaustion_min,
                SkipReason.MISSING_HEAT_WINDOW_FIELDS,
                heat_window_exhaustion_score=heat.heat_window_exhaustion_score if heat else None,
                minimum=config.late_day_heat_window_exhaustion_min,
            )

    unique_reasons = tuple(dict.fromkeys(reasons))
    return UniversalGateResult(
        family=family,
        passed=not unique_reasons and all(check.passed for check in checks),
        skip_reasons=unique_reasons,
        checks=tuple(checks),
    )


def _agreement_minimum(
    family: StrategyFamily,
    bucket_type: Optional[str],
    config: UniversalGateConfig,
) -> float:
    if (bucket_type or "").lower() in {"exact", "range"}:
        return config.bounded_model_agreement_min
    if family == StrategyFamily.FORECAST_ONLY:
        return config.forecast_only_model_agreement_min
    return config.same_day_model_agreement_min


@dataclass(frozen=True)
class PortfolioCaps:
    bankroll_usd: float
    window_bankroll_fraction: float = 0.25
    per_position_window_fraction: float = 0.25
    per_city_date_total_fraction: float = 0.15
    per_region_total_fraction: float = 0.35
    per_token_side_total_fraction: float = 0.25
    per_settlement_source_total_fraction: float = 0.50
    max_primary_exposures_per_city_date_per_run: int = 1

    @property
    def window_cap_usd(self) -> float:
        return self.bankroll_usd * self.window_bankroll_fraction

    @property
    def per_position_cap_usd(self) -> float:
        return self.window_cap_usd * self.per_position_window_fraction

    @property
    def city_date_cap_usd(self) -> float:
        return self.bankroll_usd * self.per_city_date_total_fraction

    @property
    def region_cap_usd(self) -> float:
        return self.bankroll_usd * self.per_region_total_fraction

    @property
    def token_side_cap_usd(self) -> float:
        return self.bankroll_usd * self.per_token_side_total_fraction

    @property
    def settlement_source_cap_usd(self) -> float:
        return self.bankroll_usd * self.per_settlement_source_total_fraction


@dataclass(frozen=True)
class ExposureIntent:
    token_id: str
    side: str
    city: str
    market_date_local: str
    region: str
    settlement_source_family: str
    decision_hour_utc: int
    notional_usd: float
    primary_exposure: bool = True
    release_time_utc: Optional[datetime] = None

    @property
    def city_date_key(self) -> str:
        return f"{self.city}|{self.market_date_local}"

    @property
    def token_side_key(self) -> str:
        return f"{self.token_id}|{self.side.upper()}"


@dataclass(frozen=True)
class PortfolioCapResult:
    passed: bool
    skip_reasons: tuple[SkipReason, ...]
    exposures_after: Mapping[str, float]


class PortfolioCapValidator:
    def __init__(self, caps: PortfolioCaps) -> None:
        self.caps = caps

    def validate(
        self,
        existing: Iterable[ExposureIntent],
        proposed: ExposureIntent,
    ) -> PortfolioCapResult:
        all_exposures = list(existing) + [proposed]
        reasons: list[SkipReason] = []
        by_window = _sum_by(all_exposures, lambda item: str(item.decision_hour_utc))
        by_region = _sum_by(all_exposures, lambda item: item.region)
        by_city_date = _sum_by(all_exposures, lambda item: item.city_date_key)
        by_token_side = _sum_by(all_exposures, lambda item: item.token_side_key)
        by_source = _sum_by(all_exposures, lambda item: item.settlement_source_family)
        primary_city_run_count = sum(
            1
            for item in all_exposures
            if item.primary_exposure
            and item.city_date_key == proposed.city_date_key
        )

        if proposed.notional_usd > self.caps.per_position_cap_usd:
            reasons.append(SkipReason.PORTFOLIO_CAP_EXCEEDED)
        if by_window[str(proposed.decision_hour_utc)] > self.caps.window_cap_usd:
            reasons.append(SkipReason.WINDOW_CAP_EXCEEDED)
        if by_region[proposed.region] > self.caps.region_cap_usd:
            reasons.append(SkipReason.REGION_CAP_EXCEEDED)
        if by_city_date[proposed.city_date_key] > self.caps.city_date_cap_usd:
            reasons.append(SkipReason.CITY_DATE_CAP_EXCEEDED)
        if by_token_side[proposed.token_side_key] > self.caps.token_side_cap_usd:
            reasons.append(SkipReason.TOKEN_SIDE_CAP_EXCEEDED)
        if by_source[proposed.settlement_source_family] > self.caps.settlement_source_cap_usd:
            reasons.append(SkipReason.SETTLEMENT_SOURCE_CAP_EXCEEDED)
        if primary_city_run_count > self.caps.max_primary_exposures_per_city_date_per_run:
            reasons.append(SkipReason.PRIMARY_CITY_DATE_EXPOSURE_EXISTS)

        return PortfolioCapResult(
            passed=not reasons,
            skip_reasons=tuple(dict.fromkeys(reasons)),
            exposures_after={
                "window_usd": by_window[str(proposed.decision_hour_utc)],
                "region_usd": by_region[proposed.region],
                "city_date_usd": by_city_date[proposed.city_date_key],
                "token_side_usd": by_token_side[proposed.token_side_key],
                "settlement_source_usd": by_source[proposed.settlement_source_family],
            },
        )


class PortfolioState:
    def __init__(self, caps: PortfolioCaps) -> None:
        self.caps = caps
        self.validator = PortfolioCapValidator(caps)
        self.exposures: list[ExposureIntent] = []

    def check(self, proposed: ExposureIntent) -> PortfolioCapResult:
        return self.validator.validate(self.exposures, proposed)

    def reserve(self, proposed: ExposureIntent) -> PortfolioCapResult:
        result = self.check(proposed)
        if result.passed:
            self.exposures.append(proposed)
        return result

    def release_expired(self, as_of_utc: datetime) -> int:
        as_of = _ensure_utc(as_of_utc)
        before = len(self.exposures)
        self.exposures = [
            exposure
            for exposure in self.exposures
            if exposure.release_time_utc is None
            or _ensure_utc(exposure.release_time_utc) > as_of
        ]
        return before - len(self.exposures)

    def summary(self) -> Mapping[str, Any]:
        return {
            "reserved_exposure_count": len(self.exposures),
            "reserved_notional_usd": sum(item.notional_usd for item in self.exposures),
            "cap_dimensions": PORTFOLIO_CAP_SCHEMA["cap_dimensions"],
            "cap_values_usd": {
                "utc_window": self.caps.window_cap_usd,
                "per_position": self.caps.per_position_cap_usd,
                "city_date": self.caps.city_date_cap_usd,
                "region": self.caps.region_cap_usd,
                "token_side": self.caps.token_side_cap_usd,
                "settlement_source_family": self.caps.settlement_source_cap_usd,
            },
            "one_primary_exposure_per_city_date": (
                self.caps.max_primary_exposures_per_city_date_per_run == 1
            ),
        }


def _sum_by(items: Iterable[ExposureIntent], key_fn: Any) -> dict[str, float]:
    totals: dict[str, float] = defaultdict(float)
    for item in items:
        totals[str(key_fn(item))] += float(item.notional_usd)
    return totals


class DecisionArtifactLoader(Protocol):
    def iter_decisions(
        self,
        *,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 0,
    ) -> Iterator[DecisionSnapshot]:
        ...


class LabelStore(Protocol):
    def get_for_decision(self, decision: DecisionSnapshot) -> Optional[CanonicalLabel]:
        ...


class ScoreStore(Protocol):
    def get_for_decision(self, decision: DecisionSnapshot) -> Optional[SharedScoreContract]:
        ...


class NullScoreStore:
    def get_for_decision(self, decision: DecisionSnapshot) -> Optional[SharedScoreContract]:
        return None


class DictScoreStore:
    def __init__(self, scores: Mapping[str, SharedScoreContract]) -> None:
        self.scores = dict(scores)

    def get_for_decision(self, decision: DecisionSnapshot) -> Optional[SharedScoreContract]:
        return self.scores.get(decision.decision_id) or self.scores.get(decision.universe_row_id)


class SQLiteDecisionArtifactLoader:
    def __init__(self, decision_db: Path, *, use_current_scope: bool = True) -> None:
        self.decision_db = Path(decision_db)
        self.use_current_scope = use_current_scope

    def iter_decisions(
        self,
        *,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 0,
    ) -> Iterator[DecisionSnapshot]:
        if not self.decision_db.exists():
            raise FileNotFoundError(self.decision_db)
        with sqlite3.connect(f"file:{self.decision_db}?mode=ro", uri=True) as conn:
            conn.row_factory = sqlite3.Row
            relation = "decision_snapshots_current" if self.use_current_scope else "decision_snapshots"
            if not _sqlite_relation_exists(conn, relation):
                raise RuntimeError(f"Missing decision artifact relation: {relation}")
            where: list[str] = []
            params: list[Any] = []
            if start_date is not None:
                where.append("market_date_local >= ?")
                params.append(start_date)
            if end_date is not None:
                where.append("market_date_local <= ?")
                params.append(end_date)
            query = f"SELECT * FROM {relation}"
            if where:
                query += " WHERE " + " AND ".join(where)
            query += " ORDER BY decision_time_utc, market_id, universe_row_id, side"
            if limit:
                query += " LIMIT ?"
                params.append(limit)
            for row in conn.execute(query, params):
                yield DecisionSnapshot.from_sqlite_row(dict(row))


class SQLiteDecisionFrameLoader:
    def __init__(
        self,
        decision_db: Path,
        *,
        relation: str = COMPACT_DECISION_VIEW,
        official_trading_replay_only: bool = False,
        require_compact_quote_store: bool = True,
        point_in_time_policy: PointInTimePolicy = PointInTimePolicy(),
    ) -> None:
        self.decision_db = Path(decision_db)
        self.relation = relation
        self.official_trading_replay_only = official_trading_replay_only
        self.require_compact_quote_store = require_compact_quote_store
        self.point_in_time_policy = point_in_time_policy

    def iter_frames(
        self,
        *,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 0,
    ) -> Iterator[ReplayFrame]:
        if not self.decision_db.exists():
            raise FileNotFoundError(self.decision_db)
        with sqlite3.connect(f"file:{self.decision_db}?mode=ro", uri=True) as conn:
            conn.row_factory = sqlite3.Row
            if not _sqlite_relation_exists(conn, self.relation):
                raise RuntimeError(f"Missing decision artifact relation: {self.relation}")
            columns = _sqlite_columns(conn, self.relation)
            _require_columns(
                columns,
                {
                    "decision_id",
                    "universe_row_id",
                    "decision_time_utc",
                    "quote_timestamp_utc",
                    "quote_source_payload_id",
                    "quote_source_cache_path",
                    "quote_data_quality_flags",
                    "canonical_label",
                    "recommended_action",
                    "exclude_from_backtest_replay",
                },
                self.relation,
            )
            where: list[str] = []
            params: list[Any] = []
            if start_date is not None:
                where.append("market_date_local >= ?")
                params.append(start_date)
            if end_date is not None:
                where.append("market_date_local <= ?")
                params.append(end_date)
            if self.require_compact_quote_store:
                where.append("COALESCE(quote_store, 'compact') = 'compact'")
            if self.official_trading_replay_only:
                where.append(f"({OFFICIAL_TRADING_REPLAY_SQL_PREDICATE})")
            query = f"SELECT * FROM {self.relation}"
            if where:
                query += " WHERE " + " AND ".join(where)
            query += " ORDER BY decision_time_utc, market_id, universe_row_id, side"
            if limit:
                query += " LIMIT ?"
                params.append(limit)
            for row in conn.execute(query, params):
                row_dict = dict(row)
                decision = DecisionSnapshot.from_sqlite_row(row_dict)
                label = CanonicalLabel.from_sqlite_row(row_dict)
                score = _score_from_enriched_row(row_dict, decision)
                yield ReplayFrame(
                    decision=decision,
                    canonical_label=label,
                    score=score,
                    skip_reasons=self.point_in_time_policy.skip_reasons(decision, label, score),
                )


class CompactQuoteHistoryLookup:
    """Read-only compact quote lookup used for execution-realism stress tests."""

    def __init__(self, compact_price_db: Path) -> None:
        self.compact_price_db = Path(compact_price_db)
        if not self.compact_price_db.exists():
            raise FileNotFoundError(self.compact_price_db)
        self._conn = sqlite3.connect(f"file:{self.compact_price_db}?mode=ro", uri=True)
        self._conn.row_factory = sqlite3.Row

    @classmethod
    def from_work_root(cls, work_root: Path) -> "CompactQuoteHistoryLookup":
        return cls(Path(work_root).resolve() / COMPACT_PRICE_DB_RELATIVE_PATH)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "CompactQuoteHistoryLookup":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def quote_after(
        self,
        *,
        token_id: str,
        earliest_time_utc: datetime,
        max_wait_seconds: int = 90 * 60,
    ) -> Optional[QuoteSnapshot]:
        if not token_id:
            return None
        earliest = _ensure_utc(earliest_time_utc)
        row = self._conn.execute(
            """
            SELECT
              q.timestamp_utc,
              q.bid,
              q.ask,
              q.mid,
              q.last_price,
              q.spread,
              q.bid_size,
              q.ask_size,
              q.liquidity_depth,
              q.data_quality_flags,
              s.source_payload_id,
              s.source_cache_path
            FROM compact_tokens t
            JOIN compact_quote_history q ON q.token_ref = t.token_ref
            LEFT JOIN compact_sources s ON s.source_ref = q.source_ref
            WHERE t.token_id = ? AND q.timestamp_utc >= ?
            ORDER BY q.timestamp_utc ASC
            LIMIT 1
            """,
            (token_id, earliest.isoformat()),
        ).fetchone()
        if row is None:
            return None
        quote = _quote_snapshot_from_compact_row(dict(row))
        if quote is None:
            return None
        if (quote.timestamp_utc - earliest).total_seconds() > max_wait_seconds:
            return None
        return quote


class CanonicalLabelStore:
    def __init__(self, labels: Iterable[CanonicalLabel]) -> None:
        self.by_universe_row_id: dict[str, CanonicalLabel] = {}
        self.by_market_id: dict[str, CanonicalLabel] = {}
        for label in labels:
            if label.universe_row_id:
                self.by_universe_row_id[label.universe_row_id] = label
            if label.market_id:
                self.by_market_id[label.market_id] = label

    @classmethod
    def from_csv(cls, path: Path) -> "CanonicalLabelStore":
        with Path(path).open(newline="", encoding="utf-8") as handle:
            return cls(CanonicalLabel.from_csv_row(row) for row in csv.DictReader(handle))

    def get_for_decision(self, decision: DecisionSnapshot) -> Optional[CanonicalLabel]:
        label = self.by_universe_row_id.get(decision.universe_row_id)
        if label is not None:
            return label
        if decision.market_id:
            return self.by_market_id.get(decision.market_id)
        return None


@dataclass
class ReplayDataset:
    decision_loader: DecisionArtifactLoader
    label_store: LabelStore
    score_store: ScoreStore = field(default_factory=NullScoreStore)
    point_in_time_policy: PointInTimePolicy = field(default_factory=PointInTimePolicy)

    def iter_frames(
        self,
        *,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 0,
    ) -> Iterator[ReplayFrame]:
        for decision in self.decision_loader.iter_decisions(
            start_date=start_date,
            end_date=end_date,
            limit=limit,
        ):
            label = self.label_store.get_for_decision(decision)
            score = self.score_store.get_for_decision(decision)
            yield ReplayFrame(
                decision=decision,
                canonical_label=label,
                score=score,
                skip_reasons=self.point_in_time_policy.skip_reasons(decision, label, score),
            )


@dataclass(frozen=True)
class ReplayEvent:
    kind: ReplayEventKind
    decision_id: str
    decision_time_utc: datetime
    skip_reasons: tuple[SkipReason, ...] = ()
    leakage_violations: tuple[LeakageViolation, ...] = ()
    order_intent: Optional[OrderIntent] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ReplayRun:
    strategy_name: str
    report: Mapping[str, Any]
    events: tuple[ReplayEvent, ...]
    order_intents: tuple[OrderIntent, ...]


@dataclass(frozen=True)
class ValidationOnlyRun:
    report: Mapping[str, Any]
    gate_results: tuple[UniversalGateResult, ...]


@dataclass(frozen=True)
class CandidateReplayRun:
    report: Mapping[str, Any]
    order_intents: tuple[OrderIntent, ...]
    fills: tuple[Fill, ...]
    settlements: tuple[Settlement, ...]
    score_cache_records: tuple[ReplayGeneratedScoreRecord, ...] = ()
    trade_records: tuple["TradeAccountingRecord", ...] = ()


class ValidationOnlyEngine:
    def __init__(
        self,
        dataset: ReplayDataset,
        *,
        gate_config: UniversalGateConfig = UniversalGateConfig(),
        leakage_validators: Sequence[LeakageValidator] = (DefaultLeakageValidator(),),
    ) -> None:
        self.dataset = dataset
        self.gate_config = gate_config
        self.leakage_validators = tuple(leakage_validators)

    def validate(
        self,
        *,
        families: Sequence[StrategyFamily] = (
            StrategyFamily.FORECAST_ONLY,
            StrategyFamily.SAME_DAY_OBSERVED_HIGH,
            StrategyFamily.LATE_DAY_HEAT_WINDOW,
        ),
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 0,
        collect_gate_results: bool = True,
    ) -> ValidationOnlyRun:
        builder = ValidationOnlyReportBuilder()
        gate_results: list[UniversalGateResult] = []
        for frame in self.dataset.iter_frames(start_date=start_date, end_date=end_date, limit=limit):
            violations = tuple(
                violation
                for validator in self.leakage_validators
                for violation in validator.validate(frame)
            )
            builder.add_frame(frame, leakage_violations=violations)
            for family in families:
                result = evaluate_universal_gates(frame, family=family, config=self.gate_config)
                if collect_gate_results:
                    gate_results.append(result)
                builder.add_gate_result(result)
        return ValidationOnlyRun(report=builder.to_report(), gate_results=tuple(gate_results))


class ValidationOnlyReportBuilder:
    def __init__(self) -> None:
        self.total_frames = 0
        self.score_rows = 0
        self.observed_high_rows = 0
        self.heat_window_rows = 0
        self.quote_rows = 0
        self.forecast_rows = 0
        self.canonical_label_rows = 0
        self.quote_source_cache_path_rows = 0
        self.quote_source_payload_id_rows = 0
        self.observed_high_raw_cache_path_rows = 0
        self.max_quote_staleness_seconds: Optional[float] = None
        self.label_actions: Counter[str] = Counter()
        self.leakage_violations: Counter[str] = Counter()
        self.frame_skip_reasons: Counter[str] = Counter()
        self.gate_pass_counts: Counter[str] = Counter()
        self.gate_skip_reasons: Counter[str] = Counter()
        self.gate_check_failures: Counter[str] = Counter()
        self.coverage: dict[str, Counter[str]] = defaultdict(Counter)

    def add_frame(
        self,
        frame: ReplayFrame,
        *,
        leakage_violations: Sequence[LeakageViolation],
    ) -> None:
        decision = frame.decision
        self.total_frames += 1
        if frame.score is not None:
            self.score_rows += 1
            if frame.score.observed_high is not None:
                self.observed_high_rows += 1
            if frame.score.heat_window is not None:
                self.heat_window_rows += 1
        elif decision.observed_high is not None:
            self.observed_high_rows += 1
        if decision.quote is not None:
            self.quote_rows += 1
            if decision.quote.source_cache_path:
                self.quote_source_cache_path_rows += 1
            if decision.quote.source_payload_id:
                self.quote_source_payload_id_rows += 1
            quote_staleness = (decision.decision_time_utc - decision.quote.timestamp_utc).total_seconds()
            if quote_staleness >= 0:
                self.max_quote_staleness_seconds = (
                    quote_staleness
                    if self.max_quote_staleness_seconds is None
                    else max(self.max_quote_staleness_seconds, quote_staleness)
                )
        if decision.forecast is not None:
            self.forecast_rows += 1
        if decision.observed_high_raw_cache_path:
            self.observed_high_raw_cache_path_rows += 1
        if frame.canonical_label is not None:
            self.canonical_label_rows += 1
            self.label_actions[frame.canonical_label.recommended_action or "missing_action"] += 1
        for reason in frame.skip_reasons:
            self.frame_skip_reasons[reason.value] += 1
        for violation in leakage_violations:
            self.leakage_violations[violation.code] += 1
        self.coverage["side"][decision.side or "missing"] += 1
        self.coverage["city"][decision.normalized_city or "missing"] += 1
        self.coverage["bucket_type"][decision.bucket_type or "missing"] += 1
        self.coverage["lead_days"][str(decision.lead_days)] += 1
        self.coverage["decision_hour_utc"][str(decision.decision_hour_utc)] += 1

    def add_gate_result(self, result: UniversalGateResult) -> None:
        family = result.family.value
        if result.passed:
            self.gate_pass_counts[family] += 1
        for reason in result.skip_reasons:
            self.gate_skip_reasons[f"{family}:{reason.value}"] += 1
        for check in result.checks:
            if not check.passed:
                self.gate_check_failures[f"{family}:{check.name}"] += 1

    def to_report(self) -> Mapping[str, Any]:
        schema_gaps: list[str] = []
        if self.observed_high_rows == 0:
            schema_gaps.append("observed_high_as_of_decision_not_joined_or_not_present")
        if self.heat_window_rows == 0:
            schema_gaps.append("heat_window_features_not_joined_or_not_present")
        if self.score_rows == 0:
            schema_gaps.append("shared_score_store_not_materialized_for_decision_rows")
        schema_gaps.append("portfolio_state_join_not_materialized_for_validation_only_runs")
        return {
            "schema_version": BACKTEST_FRAMEWORK_SCHEMA_VERSION,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "mode": "validation_only_no_trading_replay_no_pnl",
            "no_pnl_statement": (
                "This report validates data/schema readiness only. It does not run "
                "strategy PnL replay, optimize parameters, or claim profitability."
            ),
            "score_contract_coverage": {
                "decision_frames": self.total_frames,
                "frames_with_shared_score": self.score_rows,
                "frames_with_observed_high_snapshot": self.observed_high_rows,
                "frames_with_heat_window_fields": self.heat_window_rows,
            },
            "artifact_coverage": {
                "frames_with_quote": self.quote_rows,
                "frames_with_forecast": self.forecast_rows,
                "frames_with_canonical_label": self.canonical_label_rows,
                "frames_with_quote_source_cache_path": self.quote_source_cache_path_rows,
                "frames_with_quote_source_payload_id": self.quote_source_payload_id_rows,
                "frames_with_observed_high_raw_cache_path": self.observed_high_raw_cache_path_rows,
                "max_quote_staleness_seconds_seen": self.max_quote_staleness_seconds,
            },
            "universal_gate_summary": {
                "pass_counts_by_family": dict(sorted(self.gate_pass_counts.items())),
                "skip_reason_counts_by_family": dict(sorted(self.gate_skip_reasons.items())),
                "failed_check_counts_by_family": dict(sorted(self.gate_check_failures.items())),
            },
            "canonical_label_action_counts": dict(sorted(self.label_actions.items())),
            "frame_skip_reason_counts": dict(sorted(self.frame_skip_reasons.items())),
            "leakage_validation": {
                "violation_counts": dict(sorted(self.leakage_violations.items())),
            },
            "portfolio_cap_readiness": {
                "cap_contract_available": True,
                "cap_dimensions": PORTFOLIO_CAP_SCHEMA["cap_dimensions"],
                "portfolio_state_join_available": False,
            },
            "anti_overfit_readiness": {
                "walk_forward_required_before_pnl": True,
                "parameter_optimization_performed": False,
                "pnl_replay_performed": False,
                "required_future_schema": "anti_overfit_validation_schema",
            },
            "coverage": {
                key: dict(counter.most_common(200))
                for key, counter in sorted(self.coverage.items())
            },
            "schema_gaps": schema_gaps,
            "required_future_report_sections": VALIDATION_ONLY_REPORT_SCHEMA["required_sections"],
        }


class ReplayEngine:
    def __init__(
        self,
        dataset: ReplayDataset,
        *,
        leakage_validators: Sequence[LeakageValidator] = (DefaultLeakageValidator(),),
    ) -> None:
        self.dataset = dataset
        self.leakage_validators = tuple(leakage_validators)

    def replay(
        self,
        strategy: StrategyPlugin,
        *,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 0,
    ) -> ReplayRun:
        events: list[ReplayEvent] = []
        order_intents: list[OrderIntent] = []
        report_builder = ReplayReportBuilder(strategy_name=strategy.name)
        for frame in self.dataset.iter_frames(start_date=start_date, end_date=end_date, limit=limit):
            violations = tuple(
                violation
                for validator in self.leakage_validators
                for violation in validator.validate(frame)
            )
            if violations and SkipReason.LEAKAGE_VIOLATION not in frame.skip_reasons:
                skip_reasons = frame.skip_reasons + (SkipReason.LEAKAGE_VIOLATION,)
            else:
                skip_reasons = frame.skip_reasons
            report_builder.add_frame(frame, skip_reasons=skip_reasons, leakage_violations=violations)
            if skip_reasons:
                events.append(
                    ReplayEvent(
                        kind=ReplayEventKind.SKIPPED,
                        decision_id=frame.decision.decision_id,
                        decision_time_utc=frame.decision.decision_time_utc,
                        skip_reasons=skip_reasons,
                        leakage_violations=violations,
                    )
                )
                continue
            strategy_decision = strategy.evaluate(frame.strategy_input())
            events.append(
                ReplayEvent(
                    kind=ReplayEventKind.STRATEGY_EVALUATED,
                    decision_id=frame.decision.decision_id,
                    decision_time_utc=frame.decision.decision_time_utc,
                    metadata=strategy_decision.metadata,
                )
            )
            if not strategy_decision.orders:
                report_builder.add_skip_reason(SkipReason.STRATEGY_NO_ACTION)
            for order in strategy_decision.orders:
                order_intents.append(order)
                events.append(
                    ReplayEvent(
                        kind=ReplayEventKind.ORDER_INTENT,
                        decision_id=frame.decision.decision_id,
                        decision_time_utc=frame.decision.decision_time_utc,
                        order_intent=order,
                    )
                )
        return ReplayRun(
            strategy_name=strategy.name,
            report=report_builder.to_report(),
            events=tuple(events),
            order_intents=tuple(order_intents),
        )


@dataclass(frozen=True)
class AccountingAssumptions:
    buy_price_source: str = "ask"
    sell_price_source: str = "bid"
    fee_model: str = FEE_MODEL_ZERO
    fee_rate: float = 0.0
    fill_policy: str = "explicit_fill_only"


@dataclass(frozen=True)
class FeeModelConfig:
    fee_model: str = FEE_MODEL_ZERO
    fee_rate: Optional[float] = None
    rounding_decimals: int = FEE_ROUNDING_DECIMALS

    @property
    def resolved_fee_rate(self) -> float:
        if self.fee_model == FEE_MODEL_POLYMARKET_WEATHER_TAKER:
            return (
                DEFAULT_POLYMARKET_WEATHER_TAKER_FEE_RATE
                if self.fee_rate is None
                else float(self.fee_rate)
            )
        return 0.0

    def as_report_dict(self) -> Mapping[str, Any]:
        return {
            "fee_model": self.fee_model,
            "fee_rate": self.resolved_fee_rate,
            "fee_formula": (
                "shares * feeRate * price * (1 - price)"
                if self.fee_model == FEE_MODEL_POLYMARKET_WEATHER_TAKER
                else "0"
            ),
            "fee_rounding_decimals": self.rounding_decimals,
            "maker_fee_usd": 0.0,
            "liquidity_role": (
                "taker_for_buy_at_ask_and_sell_at_bid"
                if self.fee_model == FEE_MODEL_POLYMARKET_WEATHER_TAKER
                else "no_fee_model"
            ),
        }


def _round_fee_usd(value: float, decimals: int = FEE_ROUNDING_DECIMALS) -> float:
    quant = Decimal("1").scaleb(-decimals)
    return float(Decimal(str(value)).quantize(quant, rounding=ROUND_HALF_UP))


def calculate_fill_fee_usd(
    *,
    shares: float,
    price: float,
    fee_config: FeeModelConfig = FeeModelConfig(),
) -> float:
    if fee_config.fee_model == FEE_MODEL_ZERO:
        return 0.0
    if fee_config.fee_model != FEE_MODEL_POLYMARKET_WEATHER_TAKER:
        raise ValueError(f"Unsupported fee model: {fee_config.fee_model}")
    fee = (
        Decimal(str(shares))
        * Decimal(str(fee_config.resolved_fee_rate))
        * Decimal(str(price))
        * (Decimal("1") - Decimal(str(price)))
    )
    return _round_fee_usd(float(fee), fee_config.rounding_decimals)


@dataclass(frozen=True)
class Fill:
    token_id: str
    side: str
    action: OrderAction
    timestamp_utc: datetime
    price: float
    shares: float
    fee_usd: float = 0.0
    fee_model: str = FEE_MODEL_ZERO
    fee_rate: float = 0.0
    liquidity_role: str = "none"
    decision_id: Optional[str] = None

    @property
    def notional_usd(self) -> float:
        return self.price * self.shares


@dataclass(frozen=True)
class FillPolicy:
    allow_partial: bool = False
    min_fill_shares: float = 0.0
    policy_name: str = "no_partial_fill_visible_size_must_cover_order"


@dataclass(frozen=True)
class FillResult:
    fill: Optional[Fill]
    status: str
    skip_reason: Optional[SkipReason]
    requested_shares: Optional[float]
    filled_shares: float = 0.0
    visible_size: Optional[float] = None
    price_source: Optional[str] = None


@dataclass(frozen=True)
class PositionSnapshot:
    token_id: str
    side: str
    shares: float
    cost_usd: float
    last_quote_bid: Optional[float] = None
    last_quote_ask: Optional[float] = None
    decision_time_utc: Optional[datetime] = None


class ExitHoldAction(str, Enum):
    HOLD = "HOLD"
    EXIT = "EXIT"
    REDUCE = "REDUCE"


@dataclass(frozen=True)
class ExitHoldDecision:
    action: ExitHoldAction
    reason: str
    limit_price: Optional[float] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


def evaluate_exit_hold_primitive(
    position: PositionSnapshot,
    *,
    score: Optional[SharedScoreContract],
    quote: Optional[QuoteSnapshot],
    hold_min_probability: float = 0.50,
) -> ExitHoldDecision:
    if score is None:
        return ExitHoldDecision(
            action=ExitHoldAction.HOLD,
            reason=SkipReason.MISSING_SHARED_SCORE.value,
        )
    if quote is None or quote.bid is None:
        return ExitHoldDecision(
            action=ExitHoldAction.HOLD,
            reason=SkipReason.MISSING_EXECUTABLE_PRICE.value,
        )
    if score.p_side < hold_min_probability:
        return ExitHoldDecision(
            action=ExitHoldAction.EXIT,
            reason="p_side_below_hold_min_probability",
            limit_price=quote.bid,
            metadata={"p_side": score.p_side, "hold_min_probability": hold_min_probability},
        )
    return ExitHoldDecision(
        action=ExitHoldAction.HOLD,
        reason="p_side_above_hold_min_probability",
        limit_price=quote.bid,
        metadata={"p_side": score.p_side, "hold_min_probability": hold_min_probability},
    )


@dataclass
class Position:
    token_id: str
    side: str
    shares: float = 0.0
    cost_usd: float = 0.0


@dataclass(frozen=True)
class Settlement:
    token_id: str
    side: str
    shares: float
    payout_per_share: int
    cash_delta_usd: float
    label_source: str


@dataclass(frozen=True)
class TradeAccountingRecord:
    decision_id: str
    universe_row_id: str
    market_id: Optional[str]
    slug: Optional[str]
    question: Optional[str]
    normalized_city: Optional[str]
    country_region: Optional[str]
    timezone: Optional[str]
    market_date_local: Optional[str]
    bucket_type: Optional[str]
    bucket_label: Optional[str]
    side: str
    token_id: str
    lead_days: Optional[int]
    decision_hour_utc: Optional[int]
    decision_time_utc: datetime
    fill_timestamp_utc: datetime
    fill_price: float
    shares: float
    gross_no_fee_cost_usd: float
    fee_usd: float
    fee_model: str
    fee_rate: float
    liquidity_role: str
    cost_usd: float
    payout_per_share: int
    payout_usd: float
    gross_no_fee_realized_pnl_usd: float
    realized_pnl_usd: float
    net_fee_adjusted_realized_pnl_usd: float
    gross_no_fee_roi_on_cost: Optional[float]
    roi_on_cost: Optional[float]
    net_fee_adjusted_roi_on_cost: Optional[float]
    gross_no_fee_outcome: str
    outcome: str
    quote_mid: Optional[float]
    market_midpoint_cost_usd: Optional[float]
    market_midpoint_realized_pnl_usd: Optional[float]
    quote_source_payload_id: Optional[str]
    quote_source_cache_path: Optional[str]
    label_source: str
    score_p_yes: Optional[float]
    score_p_side: Optional[float]
    score_model_version: Optional[str]

    def to_report_dict(self) -> Mapping[str, Any]:
        return _json_ready(self.__dict__)


@dataclass
class ReplayLedger:
    starting_cash_usd: float
    assumptions: AccountingAssumptions = field(default_factory=AccountingAssumptions)
    cash_usd: float = field(init=False)
    positions: dict[tuple[str, str], Position] = field(default_factory=dict)
    fills: list[Fill] = field(default_factory=list)
    settlements: list[Settlement] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.cash_usd = float(self.starting_cash_usd)

    def apply_fill(self, fill: Fill) -> Position:
        key = (fill.token_id, fill.side.upper())
        position = self.positions.setdefault(key, Position(token_id=fill.token_id, side=fill.side.upper()))
        if fill.action == OrderAction.BUY:
            self.cash_usd -= fill.notional_usd + fill.fee_usd
            position.cost_usd += fill.notional_usd + fill.fee_usd
            position.shares += fill.shares
        elif fill.action == OrderAction.SELL:
            if fill.shares > position.shares + 1e-9:
                raise ValueError("Cannot sell more shares than the ledger holds")
            average_cost = position.cost_usd / position.shares if position.shares else 0.0
            self.cash_usd += fill.notional_usd - fill.fee_usd
            position.shares -= fill.shares
            position.cost_usd = max(0.0, position.cost_usd - average_cost * fill.shares)
        else:
            raise ValueError(f"Unsupported order action: {fill.action}")
        self.fills.append(fill)
        return position

    def settle_position(self, token_id: str, side: str, label: CanonicalLabel) -> Settlement:
        key = (token_id, side.upper())
        position = self.positions.get(key)
        if position is None or position.shares <= 0:
            raise ValueError("No open position to settle")
        payout_per_share = label.payout_for_side(side)
        cash_delta = position.shares * payout_per_share
        settlement = Settlement(
            token_id=token_id,
            side=side.upper(),
            shares=position.shares,
            payout_per_share=payout_per_share,
            cash_delta_usd=cash_delta,
            label_source=label.canonical_label_source,
        )
        self.cash_usd += cash_delta
        position.shares = 0.0
        position.cost_usd = 0.0
        self.settlements.append(settlement)
        return settlement


class CandidateReplayReportBuilder:
    def __init__(
        self,
        *,
        strategy_name: str,
        strategy_config: ConservativeFVEdgeConfig,
        starting_cash_usd: float,
        row_limit: int,
        execution_enabled: bool = False,
        fill_policy: FillPolicy = FillPolicy(),
        fee_config: FeeModelConfig = FeeModelConfig(),
        model_runner_metadata: Optional[Mapping[str, str]] = None,
        portfolio_caps: Optional[PortfolioCaps] = None,
    ) -> None:
        self.strategy_name = strategy_name
        self.strategy_config = strategy_config
        self.starting_cash_usd = starting_cash_usd
        self.row_limit = row_limit
        self.execution_enabled = execution_enabled
        self.fill_policy = fill_policy
        self.fee_config = fee_config
        self.model_runner_metadata = dict(model_runner_metadata or {})
        self.portfolio_caps = portfolio_caps
        self.total_frames = 0
        self.strategy_input_rows = 0
        self.order_count = 0
        self.fill_count = 0
        self.settlement_count = 0
        self.generated_score_rows = 0
        self.model_output_missing_rows = 0
        self.model_input_firewall_violation_rows = 0
        self.requested_notional_usd = 0.0
        self.filled_notional_usd = 0.0
        self.fees_paid_usd = 0.0
        self.cash_delta_usd = 0.0
        self.quote_rows = 0
        self.quote_source_cache_path_rows = 0
        self.quote_source_payload_id_rows = 0
        self.canonical_label_rows = 0
        self.skip_reasons: Counter[str] = Counter()
        self.leakage_violations: Counter[str] = Counter()
        self.label_actions: Counter[str] = Counter()
        self.coverage: dict[str, Counter[str]] = defaultdict(Counter)
        self.quote_quality_flags: Counter[str] = Counter()
        self.quote_source_cache_path_samples: list[str] = []
        self.fill_status_counts: Counter[str] = Counter()
        self.portfolio_state_summary: Mapping[str, Any] = {}
        self.score_cache_summary: Mapping[str, Any] = {}

    def add_frame(
        self,
        frame: ReplayFrame,
        *,
        skip_reasons: Sequence[SkipReason],
        leakage_violations: Sequence[LeakageViolation],
    ) -> None:
        decision = frame.decision
        self.total_frames += 1
        if not skip_reasons:
            self.strategy_input_rows += 1
        for reason in skip_reasons:
            self.skip_reasons[reason.value] += 1
        for violation in leakage_violations:
            self.leakage_violations[violation.code] += 1
        if decision.quote is not None:
            self.quote_rows += 1
            if decision.quote.source_cache_path:
                self.quote_source_cache_path_rows += 1
                if len(self.quote_source_cache_path_samples) < 5:
                    self.quote_source_cache_path_samples.append(decision.quote.source_cache_path)
            if decision.quote.source_payload_id:
                self.quote_source_payload_id_rows += 1
            for flag in decision.quote.data_quality_flags:
                self.quote_quality_flags[flag] += 1
        if frame.canonical_label is not None:
            self.canonical_label_rows += 1
            self.label_actions[frame.canonical_label.recommended_action or "missing_action"] += 1
        self.coverage["side"][decision.side or "missing"] += 1
        self.coverage["city"][decision.normalized_city or "missing"] += 1
        self.coverage["bucket_type"][decision.bucket_type or "missing"] += 1
        self.coverage["lead_days"][str(decision.lead_days)] += 1
        self.coverage["decision_hour_utc"][str(decision.decision_hour_utc)] += 1

    def add_strategy_skip_reasons(self, reasons: Sequence[SkipReason]) -> None:
        if not reasons:
            self.skip_reasons[SkipReason.STRATEGY_NO_ACTION.value] += 1
            return
        for reason in reasons:
            self.skip_reasons[reason.value] += 1

    def add_generated_score(self) -> None:
        self.generated_score_rows += 1

    def add_model_output_missing(self) -> None:
        self.model_output_missing_rows += 1

    def add_model_input_firewall_violation(self) -> None:
        self.model_input_firewall_violation_rows += 1

    def add_order(self, order: OrderIntent) -> None:
        self.order_count += 1
        self.requested_notional_usd += float(order.requested_notional_usd or 0.0)

    def add_fill_result(self, result: FillResult) -> None:
        self.fill_status_counts[result.status] += 1
        if result.skip_reason is not None:
            self.skip_reasons[result.skip_reason.value] += 1

    def add_fill(self, fill: Fill) -> None:
        self.fill_count += 1
        self.filled_notional_usd += fill.notional_usd
        self.fees_paid_usd += fill.fee_usd

    def add_settlement(self, settlement: Settlement) -> None:
        self.settlement_count += 1

    def set_portfolio_state_summary(self, summary: Mapping[str, Any]) -> None:
        self.portfolio_state_summary = dict(summary)

    def set_score_cache_summary(self, summary: Mapping[str, Any]) -> None:
        self.score_cache_summary = dict(summary)

    def to_report(
        self,
        *,
        ledger: ReplayLedger,
        fixed_strategy_accounting: Optional[Mapping[str, Any]] = None,
    ) -> Mapping[str, Any]:
        self.cash_delta_usd = ledger.cash_usd - self.starting_cash_usd
        blocker_reasons = sorted(
            reason
            for reason, count in self.skip_reasons.items()
            if count == self.total_frames and self.total_frames > 0
        )
        fixture_runner_used = self.model_runner_metadata.get("fixture_only") == "true"
        research_only_runner = (
            self.model_runner_metadata.get("promotion_status") == "research_only_not_promoted"
            or self.model_runner_metadata.get("research_only_not_promoted") == "true"
        )
        profitability_blockers = [
            "small_validation_replay_only",
            "walk_forward_and_parameter_plateau_gates_not_run_for_this_candidate_report",
        ]
        if fixture_runner_used:
            profitability_blockers.append("fixture_runner_not_a_real_weather_model")
        if research_only_runner:
            profitability_blockers.append("research_only_weather_model_not_promoted")
        if not self.model_runner_metadata:
            profitability_blockers.append("weather_model_runner_not_plugged_in")
        report: dict[str, Any] = {
            "schema_version": BACKTEST_FRAMEWORK_SCHEMA_VERSION,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "mode": (
                "fixed_strategy_accounting_replay_no_optimization"
                if self.execution_enabled
                else "candidate_replay_small_baseline_no_optimization"
            ),
            "strategy_name": self.strategy_name,
            "strategy_candidate": "conservative_fv_edge",
            "baseline_settings": dict(self.strategy_config.as_report_dict()),
            "row_limit": self.row_limit,
            "small_validation_replay_only": True,
            "parameter_search_performed": False,
            "profitability_claim": (
                "exact_fixed_replay_accounting_only"
                if self.execution_enabled
                else "none"
            ),
            "profitability_claim_allowed": False,
            "profitability_claim_scope": (
                "realized_accounting_for_this_exact_fixed_parameter_replay_only"
                if self.execution_enabled
                else "none"
            ),
            "profitability_claim_blockers": profitability_blockers,
            "official_trading_replay_sql_predicate": OFFICIAL_TRADING_REPLAY_SQL_PREDICATE,
            "execution_assumptions": {
                "buy_price_source": "ask",
                "sell_price_source": "bid",
                "entry_policy": "baseline_order_intents_only_when_runner_score_and_gates_pass",
                "exit_policy": "exit_hold_primitives_available_no_economic_exit_replay_in_report",
                "partial_fill_policy": self.fill_policy.policy_name,
                "orders_executed": self.execution_enabled,
                "fee_model": self.fee_config.as_report_dict(),
                "fees_included_in_cash_cost_pnl_roi": True,
                "order_latency_model": "not_modeled",
                "order_latency_stress_tested": False,
                "order_latency_blocker": (
                    "current accounting frames expose the decision quote snapshot but not a "
                    "next-quote-after-delay lookup"
                ),
            },
            "decision_counts": {
                "total_decision_rows_seen": self.total_frames,
                "rows_available_for_strategy_input": self.strategy_input_rows,
                "order_intents": self.order_count,
                "fills": self.fill_count,
                "settlements": self.settlement_count,
            },
            "weather_model_runner": {
                "metadata": self.model_runner_metadata,
                "fixture_runner_used": fixture_runner_used,
                "real_weather_model_runner_plugged_in": (
                    not fixture_runner_used
                    and bool(self.model_runner_metadata)
                ),
                "research_only_not_promoted": research_only_runner,
                "model_outputs_generated": self.generated_score_rows,
                "missing_model_outputs": self.model_output_missing_rows,
                "model_input_firewall_violations": self.model_input_firewall_violation_rows,
            },
            "model_input_firewall": {
                "forbidden_fields": list(FORBIDDEN_WEATHER_MODEL_CONTEXT_FIELDS),
                "live_valid_model_inputs_only": True,
                "market_prices_passed_to_runner": False,
                "labels_or_outcomes_passed_to_runner": False,
                "future_weather_observations_passed_to_runner": False,
            },
            "replay_generated_score_cache": self.score_cache_summary,
            "accounting_summary": {
                "starting_cash_usd": self.starting_cash_usd,
                "ending_cash_usd": ledger.cash_usd,
                "cash_delta_usd": self.cash_delta_usd,
                "requested_notional_usd": self.requested_notional_usd,
                "filled_notional_usd": self.filled_notional_usd,
                "fees_paid_usd": self.fees_paid_usd,
                "filled_cost_including_fees_usd": self.filled_notional_usd + self.fees_paid_usd,
                "open_positions": sum(
                    1 for position in ledger.positions.values() if position.shares > 1e-9
                ),
                "accounting_is_interpretable": self.execution_enabled and self.fill_count > 0,
                "economic_strategy_results_performed": self.execution_enabled,
                "fill_status_counts": dict(sorted(self.fill_status_counts.items())),
            },
            "portfolio_state": self.portfolio_state_summary,
            "skip_reason_counts": dict(sorted(self.skip_reasons.items())),
            "candidate_replay_blockers": blocker_reasons,
            "canonical_label_action_counts": dict(sorted(self.label_actions.items())),
            "leakage_validation": {
                "violation_counts": dict(sorted(self.leakage_violations.items())),
            },
            "quote_source_audit": {
                "frames_with_quote": self.quote_rows,
                "frames_with_quote_source_cache_path": self.quote_source_cache_path_rows,
                "frames_with_quote_source_payload_id": self.quote_source_payload_id_rows,
                "quote_quality_flag_counts": dict(sorted(self.quote_quality_flags.items())),
                "sample_quote_source_cache_paths": self.quote_source_cache_path_samples,
            },
            "coverage": {
                key: dict(counter.most_common(200))
                for key, counter in sorted(self.coverage.items())
            },
            "anti_overfit_readiness": {
                "walk_forward_required_before_profitability_claim": True,
                "parameter_optimization_performed": False,
                "parameter_plateau_test_performed": False,
                "passes_anti_overfit_reporting_gates": False,
            },
        }
        if fixed_strategy_accounting is not None:
            report["fixed_strategy_accounting"] = fixed_strategy_accounting
        return report


def run_candidate_replay(
    dataset: Any,
    strategy: StrategyPlugin,
    *,
    strategy_config: ConservativeFVEdgeConfig,
    starting_cash_usd: float = 100.0,
    limit: int = 250,
    leakage_validators: Sequence[LeakageValidator] = (DefaultLeakageValidator(),),
    model_runner: Optional[WeatherModelRunner] = None,
    score_cache: Optional[ReplayGeneratedScoreCache] = None,
    execute_orders: bool = False,
    fill_policy: FillPolicy = FillPolicy(),
    fee_config: FeeModelConfig = FeeModelConfig(),
    portfolio_caps: Optional[PortfolioCaps] = None,
) -> CandidateReplayRun:
    ledger = ReplayLedger(
        starting_cash_usd=starting_cash_usd,
        assumptions=AccountingAssumptions(
            fee_model=fee_config.fee_model,
            fee_rate=fee_config.resolved_fee_rate,
            fill_policy=fill_policy.policy_name,
        ),
    )
    model_runner_metadata = dict(model_runner.metadata()) if model_runner is not None else {}
    portfolio_state = PortfolioState(portfolio_caps or PortfolioCaps(bankroll_usd=starting_cash_usd))
    builder = CandidateReplayReportBuilder(
        strategy_name=strategy.name,
        strategy_config=strategy_config,
        starting_cash_usd=starting_cash_usd,
        row_limit=limit,
        execution_enabled=execute_orders,
        fill_policy=fill_policy,
        fee_config=fee_config,
        model_runner_metadata=model_runner_metadata,
        portfolio_caps=portfolio_state.caps,
    )
    order_intents: list[OrderIntent] = []
    trade_records: list[TradeAccountingRecord] = []
    cache = score_cache or ReplayGeneratedScoreCache()
    point_policy = PointInTimePolicy(require_shared_score=True)
    for source_frame in dataset.iter_frames(limit=limit):
        frame = _frame_with_replay_generated_score(
            source_frame,
            model_runner=model_runner,
            score_cache=cache,
            point_policy=point_policy,
            builder=builder,
        )
        violations = tuple(
            violation
            for validator in leakage_validators
            for violation in validator.validate(frame)
        )
        skip_reasons = _frame_skip_reasons_with_leakage(frame.skip_reasons, violations)
        builder.add_frame(frame, skip_reasons=skip_reasons, leakage_violations=violations)
        if skip_reasons:
            continue
        decision = strategy.evaluate(frame.strategy_input())
        if not decision.orders:
            builder.add_strategy_skip_reasons(_metadata_skip_reasons(decision.metadata))
            continue
        for order in decision.orders:
            portfolio_state.release_expired(frame.decision.decision_time_utc)
            exposure = _exposure_intent_from_order(order, frame.decision)
            portfolio_result = portfolio_state.reserve(exposure)
            if not portfolio_result.passed:
                builder.add_strategy_skip_reasons(portfolio_result.skip_reasons)
                continue
            builder.add_order(order)
            order_intents.append(order)
            if not execute_orders:
                continue
            fill_result = simulate_order_fill(
                order,
                frame.decision,
                fill_policy=fill_policy,
                fee_config=fee_config,
            )
            builder.add_fill_result(fill_result)
            if fill_result.fill is None:
                continue
            fill = fill_result.fill
            ledger.apply_fill(fill)
            builder.add_fill(fill)
            if order.action == OrderAction.BUY and frame.canonical_label is not None:
                settlement = ledger.settle_position(order.token_id, order.side, frame.canonical_label)
                builder.add_settlement(settlement)
                trade_records.append(
                    _trade_record_from_fill_and_settlement(
                        frame=frame,
                        order=order,
                        fill=fill,
                        settlement=settlement,
                    )
                )
    builder.set_portfolio_state_summary(portfolio_state.summary())
    builder.set_score_cache_summary(cache.report_summary())
    accounting_report = (
        _fixed_strategy_accounting_report(
            trade_records=trade_records,
            fill_status_counts=builder.fill_status_counts,
            order_intents=order_intents,
            ledger=ledger,
            starting_cash_usd=starting_cash_usd,
            fee_config=fee_config,
        )
        if execute_orders
        else None
    )
    return CandidateReplayRun(
        report=builder.to_report(ledger=ledger, fixed_strategy_accounting=accounting_report),
        order_intents=tuple(order_intents),
        fills=tuple(ledger.fills),
        settlements=tuple(ledger.settlements),
        score_cache_records=tuple(cache.records),
        trade_records=tuple(trade_records),
    )


def run_conservative_fv_edge_candidate_replay(
    dataset: Any,
    *,
    config: ConservativeFVEdgeConfig = ConservativeFVEdgeConfig(),
    starting_cash_usd: float = 100.0,
    limit: int = 250,
    model_runner: Optional[WeatherModelRunner] = None,
    score_cache: Optional[ReplayGeneratedScoreCache] = None,
    execute_orders: bool = False,
    fill_policy: FillPolicy = FillPolicy(),
    fee_config: FeeModelConfig = FeeModelConfig(),
    portfolio_caps: Optional[PortfolioCaps] = None,
) -> CandidateReplayRun:
    return run_candidate_replay(
        dataset,
        ConservativeFVEdgeStrategy(config),
        strategy_config=config,
        starting_cash_usd=starting_cash_usd,
        limit=limit,
        model_runner=model_runner,
        score_cache=score_cache,
        execute_orders=execute_orders,
        fill_policy=fill_policy,
        fee_config=fee_config,
        portfolio_caps=portfolio_caps,
    )


@dataclass(frozen=True)
class ThroughputDiagnosticVariantSpec:
    key: str
    label: str
    group: str
    description: str
    config: ConservativeFVEdgeConfig
    portfolio_caps: PortfolioCaps
    loosened_dimensions: tuple[str, ...] = ()


@dataclass
class ThroughputDiagnosticVariantState:
    spec: ThroughputDiagnosticVariantSpec
    starting_cash_usd: float
    fee_config: FeeModelConfig
    fill_policy: FillPolicy = field(default_factory=FillPolicy)

    def __post_init__(self) -> None:
        self.strategy = ConservativeFVEdgeStrategy(self.spec.config)
        self.portfolio_state = PortfolioState(self.spec.portfolio_caps)
        self.ledger = ReplayLedger(
            starting_cash_usd=self.starting_cash_usd,
            assumptions=AccountingAssumptions(
                fee_model=self.fee_config.fee_model,
                fee_rate=self.fee_config.resolved_fee_rate,
                fill_policy=self.fill_policy.policy_name,
            ),
        )
        self.pre_cap_candidate_rows = 0
        self.order_intents: list[OrderIntent] = []
        self.trade_records: list[TradeAccountingRecord] = []
        self.fill_status_counts: Counter[str] = Counter()
        self.strategy_skip_counts: Counter[str] = Counter()
        self.cap_block_counts: Counter[str] = Counter()
        self.coverage: dict[str, Counter[str]] = defaultdict(Counter)
        self.coverage_pnl: dict[str, Counter[str]] = defaultdict(Counter)
        self.unresolved_fills = 0

    def process_frame(self, frame: ReplayFrame) -> None:
        if frame.skip_reasons:
            for reason in frame.skip_reasons:
                self.strategy_skip_counts[reason.value] += 1
            return
        decision = self.strategy.evaluate(frame.strategy_input())
        if not decision.orders:
            for reason in _metadata_skip_reasons(decision.metadata):
                self.strategy_skip_counts[reason.value] += 1
            return
        for order in decision.orders:
            self.pre_cap_candidate_rows += 1
            self.portfolio_state.release_expired(frame.decision.decision_time_utc)
            exposure = _exposure_intent_from_order(order, frame.decision)
            cap_result = self.portfolio_state.reserve(exposure)
            if not cap_result.passed:
                for reason in cap_result.skip_reasons:
                    self.cap_block_counts[reason.value] += 1
                continue
            self.order_intents.append(order)
            fill_result = simulate_order_fill(
                order,
                frame.decision,
                fill_policy=self.fill_policy,
                fee_config=self.fee_config,
            )
            self.fill_status_counts[fill_result.status] += 1
            if fill_result.fill is None:
                continue
            fill = fill_result.fill
            self.ledger.apply_fill(fill)
            if order.action == OrderAction.BUY and frame.canonical_label is not None:
                settlement = self.ledger.settle_position(
                    order.token_id,
                    order.side,
                    frame.canonical_label,
                )
                record = _trade_record_from_fill_and_settlement(
                    frame=frame,
                    order=order,
                    fill=fill,
                    settlement=settlement,
                )
                self.trade_records.append(record)
                _add_throughput_coverage(self.coverage, self.coverage_pnl, record, frame.decision)
            else:
                self.unresolved_fills += 1

    @property
    def fill_count(self) -> int:
        return len(self.ledger.fills)

    def summary(self) -> Mapping[str, Any]:
        accounting = _fixed_strategy_accounting_report(
            trade_records=self.trade_records,
            fill_status_counts=self.fill_status_counts,
            order_intents=self.order_intents,
            ledger=self.ledger,
            starting_cash_usd=self.starting_cash_usd,
            fee_config=self.fee_config,
        )
        accounting = dict(accounting)
        accounting.pop("all_trades", None)
        pnl = accounting.get("pnl", {})
        return {
            "variant_key": self.spec.key,
            "label": self.spec.label,
            "group": self.spec.group,
            "description": self.spec.description,
            "loosened_dimensions": list(self.spec.loosened_dimensions),
            "config": dict(self.spec.config.as_report_dict()),
            "portfolio_caps": _portfolio_caps_summary(self.spec.portfolio_caps),
            "pre_cap_candidate_rows": self.pre_cap_candidate_rows,
            "order_intents": len(self.order_intents),
            "fills": self.fill_count,
            "settlements": len(self.ledger.settlements),
            "unresolved_fills": self.unresolved_fills,
            "cap_block_counts": dict(sorted(self.cap_block_counts.items())),
            "strategy_skip_counts_top": dict(self.strategy_skip_counts.most_common(20)),
            "accounting": accounting,
            "coverage": _throughput_coverage_report(self.coverage, self.coverage_pnl),
            "risk_checks": _throughput_risk_checks(accounting),
            "frontier_metrics": {
                "trade_count": self.fill_count,
                "estimated_gross_no_fee_pnl_usd": pnl.get("gross_no_fee_realized_pnl_usd"),
                "estimated_net_fee_adjusted_pnl_usd": pnl.get("net_fee_adjusted_realized_pnl_usd"),
                "win_count": pnl.get("win_count"),
                "loss_count": pnl.get("loss_count"),
                "flat_count": pnl.get("flat_count"),
                "roi_before_fees": pnl.get("roi_before_fees_on_deployed_capital"),
                "roi_after_fees": pnl.get("roi_after_fees_on_deployed_capital"),
                "total_fees_usd": pnl.get("total_fees_usd"),
                "fee_flips": pnl.get("trades_flipped_profitable_to_unprofitable_by_fees"),
            },
        }


def _portfolio_caps_summary(caps: PortfolioCaps) -> Mapping[str, Any]:
    return {
        "bankroll_usd": caps.bankroll_usd,
        "window_bankroll_fraction": caps.window_bankroll_fraction,
        "per_position_window_fraction": caps.per_position_window_fraction,
        "per_city_date_total_fraction": caps.per_city_date_total_fraction,
        "per_region_total_fraction": caps.per_region_total_fraction,
        "per_token_side_total_fraction": caps.per_token_side_total_fraction,
        "per_settlement_source_total_fraction": caps.per_settlement_source_total_fraction,
        "max_primary_exposures_per_city_date_per_run": caps.max_primary_exposures_per_city_date_per_run,
    }


def _throughput_variant_specs(
    *,
    starting_cash_usd: float,
) -> tuple[ThroughputDiagnosticVariantSpec, ...]:
    base = ConservativeFVEdgeConfig()
    base_caps = PortfolioCaps(bankroll_usd=starting_cash_usd)
    loose_city_date_caps = replace(
        base_caps,
        per_city_date_total_fraction=0.50,
        max_primary_exposures_per_city_date_per_run=3,
    )
    moderate_caps = replace(
        base_caps,
        per_city_date_total_fraction=0.50,
        per_settlement_source_total_fraction=1.00,
        max_primary_exposures_per_city_date_per_run=3,
    )

    def spec(
        key: str,
        label: str,
        group: str,
        description: str,
        config: ConservativeFVEdgeConfig,
        *,
        caps: PortfolioCaps = base_caps,
        dims: Sequence[str] = (),
    ) -> ThroughputDiagnosticVariantSpec:
        return ThroughputDiagnosticVariantSpec(
            key=key,
            label=label,
            group=group,
            description=description,
            config=config,
            portfolio_caps=caps,
            loosened_dimensions=tuple(dims),
        )

    return (
        spec("baseline", "baseline current gates", "frontier", "Current fixed Conservative FV Edge baseline.", base),
        spec(
            "loosen_min_fair_value",
            "min fair value loosened",
            "one_gate",
            "YES min FV 0.70->0.65 and NO min FV 0.86->0.82.",
            replace(base, yes_min_probability=0.65, no_min_probability=0.82),
            dims=("min_fair_value",),
        ),
        spec(
            "loosen_net_edge_floor",
            "net edge floor loosened",
            "one_gate",
            "Net edge floor 0.03->0.02 and remaining-fraction 0.08->0.05.",
            replace(base, min_net_edge_floor=0.02, min_net_edge_fraction_of_remaining=0.05),
            dims=("net_edge_floor",),
        ),
        spec(
            "loosen_max_spread",
            "max spread loosened",
            "one_gate",
            "Max spread 0.10->0.15.",
            replace(base, max_spread=0.15),
            dims=("max_spread",),
        ),
        spec(
            "loosen_yes_min_ask",
            "YES min ask loosened",
            "one_gate",
            "YES min ask 0.20->0.15.",
            replace(base, yes_side_min_ask=0.15),
            dims=("yes_min_ask",),
        ),
        spec(
            "loosen_general_min_ask",
            "general min ask loosened",
            "one_gate",
            "General min ask 0.125->0.10.",
            replace(base, min_ask=0.10),
            dims=("general_min_ask",),
        ),
        spec(
            "loosen_max_ask",
            "max ask loosened",
            "one_gate",
            "YES/general max ask 0.95->0.98.",
            replace(base, max_ask=0.98),
            dims=("max_ask",),
        ),
        spec(
            "loosen_no_max_ask",
            "NO max ask loosened",
            "one_gate",
            "NO max ask 0.94->0.97.",
            replace(base, no_side_max_ask=0.97),
            dims=("no_max_ask",),
        ),
        spec(
            "loosen_no_counter_event_cap",
            "NO counter-event cap loosened",
            "one_gate",
            "NO counter-event probability cap 0.14->0.18.",
            replace(base, no_counter_event_probability_max=0.18),
            dims=("no_counter_event_cap",),
        ),
        spec(
            "loosen_bounded_min_fv",
            "bounded bucket min FV loosened",
            "one_gate",
            "Bounded bucket min probability 0.98->0.95.",
            replace(base, bounded_min_probability=0.95),
            dims=("bounded_bucket_min_fv",),
        ),
        spec(
            "loosen_bounded_min_price",
            "bounded bucket min price loosened",
            "one_gate",
            "Bounded bucket min ask 0.70->0.60.",
            replace(base, bounded_min_ask=0.60),
            dims=("bounded_bucket_min_price",),
        ),
        spec(
            "loosen_bounded_edge_floor",
            "bounded bucket edge floor loosened",
            "one_gate",
            "Bounded bucket min net edge 0.10->0.05.",
            replace(base, bounded_min_net_edge=0.05),
            dims=("bounded_bucket_edge_floor",),
        ),
        spec(
            "loosen_model_agreement",
            "model agreement threshold loosened",
            "one_gate",
            "Model agreement threshold 1.00->0.90.",
            replace(base, model_agreement_min=0.90),
            dims=("model_agreement_threshold",),
        ),
        spec(
            "loosen_probability_stdev",
            "probability stdev cap loosened",
            "one_gate",
            "Open stdev cap 0.20->0.25 and bounded stdev cap 0.10->0.15.",
            replace(base, open_bucket_probability_stdev_max=0.25, bounded_probability_stdev_max=0.15),
            dims=("probability_stdev_cap",),
        ),
        spec(
            "loosen_lead_day_filter",
            "lead-day filter loosened",
            "one_gate",
            "Allowed lead days (1,2)->(1,2,3).",
            replace(base, allowed_lead_days=(1, 2, 3)),
            dims=("lead_day_filter",),
        ),
        spec(
            "include_same_day",
            "same-day inclusion",
            "one_gate",
            "Allowed lead days (1,2)->(0,1,2).",
            replace(base, allowed_lead_days=(0, 1, 2)),
            dims=("same_day_inclusion",),
        ),
        spec(
            "loosen_city_date_exposure_cap",
            "city/date exposure cap loosened",
            "one_gate",
            "Allow up to 3 primary exposures per city/date and raise city/date cap to 50% bankroll.",
            base,
            caps=loose_city_date_caps,
            dims=("city_date_exposure_cap",),
        ),
        spec(
            "frontier_edge_only",
            "slightly looser edge only",
            "frontier",
            "Frontier variant: only net edge floor loosened.",
            replace(base, min_net_edge_floor=0.02, min_net_edge_fraction_of_remaining=0.05),
            dims=("net_edge_floor",),
        ),
        spec(
            "frontier_no_cap_only",
            "slightly looser NO cap only",
            "frontier",
            "Frontier variant: only NO counter-event cap loosened.",
            replace(base, no_counter_event_probability_max=0.18),
            dims=("no_counter_event_cap",),
        ),
        spec(
            "frontier_bounded_only",
            "slightly looser bounded-bucket gates only",
            "frontier",
            "Frontier variant: bounded min FV, bounded min ask, bounded edge, and bounded stdev loosened.",
            replace(
                base,
                bounded_min_probability=0.95,
                bounded_min_ask=0.60,
                bounded_min_net_edge=0.05,
                bounded_probability_stdev_max=0.15,
            ),
            dims=("bounded_bucket_gates",),
        ),
        spec(
            "frontier_lead_day_only",
            "looser lead-day inclusion only",
            "frontier",
            "Frontier variant: allowed lead days (1,2)->(0,1,2,3).",
            replace(base, allowed_lead_days=(0, 1, 2, 3)),
            dims=("lead_day_filter", "same_day_inclusion"),
        ),
        spec(
            "frontier_price_caps_only",
            "looser price caps only",
            "frontier",
            "Frontier variant: min ask, YES min ask, max ask, and NO max ask loosened.",
            replace(base, min_ask=0.10, yes_side_min_ask=0.15, max_ask=0.98, no_side_max_ask=0.97),
            dims=("price_caps",),
        ),
        spec(
            "frontier_combined_moderate",
            "combined moderate throughput diagnostic variant",
            "frontier",
            "Diagnostic combination of moderate gate loosening plus source/city-date cap expansion to reveal post-gate throughput.",
            replace(
                base,
                allowed_lead_days=(0, 1, 2, 3),
                yes_min_probability=0.65,
                no_min_probability=0.82,
                no_counter_event_probability_max=0.18,
                model_agreement_min=0.95,
                open_bucket_probability_stdev_max=0.25,
                bounded_probability_stdev_max=0.15,
                bounded_min_probability=0.95,
                bounded_min_ask=0.60,
                bounded_min_net_edge=0.05,
                max_spread=0.15,
                min_ask=0.10,
                yes_side_min_ask=0.15,
                max_ask=0.98,
                no_side_max_ask=0.97,
                min_net_edge_floor=0.02,
                min_net_edge_fraction_of_remaining=0.05,
            ),
            caps=moderate_caps,
            dims=("combined_moderate_diagnostic",),
        ),
    )


def run_conservative_fv_edge_throughput_diagnostic(
    dataset: Any,
    *,
    starting_cash_usd: float = 100.0,
    limit: int = 0,
    model_runner: Optional[WeatherModelRunner] = None,
    score_cache: Optional[ReplayGeneratedScoreCache] = None,
    leakage_validators: Sequence[LeakageValidator] = (DefaultLeakageValidator(),),
    fee_config: FeeModelConfig = FeeModelConfig(
        fee_model=FEE_MODEL_POLYMARKET_WEATHER_TAKER,
        fee_rate=DEFAULT_POLYMARKET_WEATHER_TAKER_FEE_RATE,
    ),
    excluded_quote_rows: Optional[Mapping[str, Any]] = None,
    progress_every: int = 0,
) -> Mapping[str, Any]:
    baseline_config = ConservativeFVEdgeConfig()
    specs = _throughput_variant_specs(starting_cash_usd=starting_cash_usd)
    states = [
        ThroughputDiagnosticVariantState(
            spec=spec,
            starting_cash_usd=starting_cash_usd,
            fee_config=fee_config,
            fill_policy=FillPolicy(allow_partial=False),
        )
        for spec in specs
    ]
    variants_by_key = {state.spec.key: state for state in states}
    cache = score_cache or ReplayGeneratedScoreCache()
    point_policy = PointInTimePolicy(require_shared_score=True)
    total_rows = 0
    model_outputs = 0
    missing_model_outputs = 0
    firewall_violations = 0
    leakage_violations: Counter[str] = Counter()
    frame_skip_counts: Counter[str] = Counter()
    cumulative_funnel: Counter[str] = Counter()
    individual_gate_pass: Counter[str] = Counter()
    individual_gate_fail: Counter[str] = Counter()
    runner_metadata = dict(model_runner.metadata()) if model_runner is not None else {}
    for source_frame in dataset.iter_frames(limit=limit):
        total_rows += 1
        if progress_every and total_rows % progress_every == 0:
            print(f"throughput-diagnostic rows={total_rows}", flush=True)
        frame, generated_score, missing_model, firewall_violation = _score_frame_for_throughput(
            source_frame,
            model_runner=model_runner,
            score_cache=cache,
            point_policy=point_policy,
        )
        if generated_score:
            model_outputs += 1
        if missing_model:
            missing_model_outputs += 1
        if firewall_violation:
            firewall_violations += 1
        violations = tuple(
            violation
            for validator in leakage_validators
            for violation in validator.validate(frame)
        )
        for violation in violations:
            leakage_violations[violation.code] += 1
        if violations:
            reasons = _frame_skip_reasons_with_leakage(frame.skip_reasons, violations)
            frame = ReplayFrame(
                decision=frame.decision,
                canonical_label=frame.canonical_label,
                score=frame.score,
                skip_reasons=reasons,
            )
        for reason in frame.skip_reasons:
            frame_skip_counts[reason.value] += 1
        gate_flags = _baseline_throughput_gate_flags(frame, baseline_config)
        _add_gate_funnel_counts(cumulative_funnel, individual_gate_pass, individual_gate_fail, gate_flags)
        for state in states:
            state.process_frame(frame)
    variant_summaries = {state.spec.key: state.summary() for state in states}
    baseline_summary = variant_summaries["baseline"]
    one_gate_tests = _one_gate_loosen_summary(variant_summaries, baseline_summary)
    frontier = {
        key: variant_summaries[key]
        for key in (
            "baseline",
            "frontier_edge_only",
            "frontier_no_cap_only",
            "frontier_bounded_only",
            "frontier_lead_day_only",
            "frontier_price_caps_only",
            "frontier_combined_moderate",
        )
    }
    return {
        "schema_version": BACKTEST_FRAMEWORK_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "conservative_fv_edge_throughput_diagnostic_no_optimization",
        "diagnostic_only": True,
        "no_parameter_optimization": True,
        "no_strategy_setting_change": True,
        "no_live_trading_recommendation": True,
        "weather_model_runner": {
            "metadata": runner_metadata,
            "fixture_runner_used": runner_metadata.get("fixture_only") == "true",
            "real_weather_model_runner_plugged_in": (
                runner_metadata.get("fixture_only") == "false"
                and bool(runner_metadata)
            ),
            "research_only_not_promoted": (
                runner_metadata.get("promotion_status") == "research_only_not_promoted"
            ),
            "model_outputs_generated": model_outputs,
            "missing_model_outputs": missing_model_outputs,
            "model_input_firewall_violations": firewall_violations,
        },
        "fee_model": fee_config.as_report_dict(),
        "official_trading_replay_sql_predicate": OFFICIAL_TRADING_REPLAY_SQL_PREDICATE,
        "total_official_rows": total_rows,
        "row_limit": limit,
        "gate_funnel": {
            "cumulative_counts": dict(cumulative_funnel),
            "individual_gate_pass_counts": dict(individual_gate_pass),
            "individual_gate_fail_counts": dict(individual_gate_fail),
            "stage_order": _throughput_gate_stage_order(),
        },
        "frame_skip_counts": dict(sorted(frame_skip_counts.items())),
        "leakage_validation": {"violation_counts": dict(sorted(leakage_violations.items()))},
        "excluded_quote_rows_for_settlement_or_manual_review": dict(excluded_quote_rows or {}),
        "variants": variant_summaries,
        "one_gate_loosen_tests": one_gate_tests,
        "trade_frequency_frontier": frontier,
        "diagnostic_interpretation": _throughput_interpretation(
            baseline_summary=baseline_summary,
            one_gate_tests=one_gate_tests,
            frontier=frontier,
        ),
        "execution_realism": {
            "buy_at_ask_visible_size_logic_preserved": True,
            "order_latency_model": "not_modeled",
            "order_latency_stress_tested": False,
            "blocker": (
                "Need compact quote-history lookup for first quote after decision_time + delay; "
                "current decision frames expose only the decision quote snapshot."
            ),
        },
    }


def _score_frame_for_throughput(
    frame: ReplayFrame,
    *,
    model_runner: Optional[WeatherModelRunner],
    score_cache: ReplayGeneratedScoreCache,
    point_policy: PointInTimePolicy,
) -> tuple[ReplayFrame, bool, bool, bool]:
    if frame.score is not None:
        scored = ReplayFrame(
            decision=frame.decision,
            canonical_label=frame.canonical_label,
            score=frame.score,
            skip_reasons=point_policy.skip_reasons(frame.decision, frame.canonical_label, frame.score),
        )
        return scored, False, False, False
    if model_runner is None:
        scored = ReplayFrame(
            decision=frame.decision,
            canonical_label=frame.canonical_label,
            score=None,
            skip_reasons=point_policy.skip_reasons(frame.decision, frame.canonical_label, None),
        )
        return scored, False, True, False
    try:
        context = WeatherModelDecisionContext.from_decision(frame.decision)
        validate_weather_model_context_firewall(context)
        prediction = model_runner.predict(context)
    except ValueError:
        reasons = list(point_policy.skip_reasons(frame.decision, frame.canonical_label, None))
        reasons.append(SkipReason.MODEL_INPUT_FIREWALL_VIOLATION)
        return (
            ReplayFrame(
                decision=frame.decision,
                canonical_label=frame.canonical_label,
                score=None,
                skip_reasons=tuple(dict.fromkeys(reasons)),
            ),
            False,
            False,
            True,
        )
    if prediction is None:
        reasons = list(point_policy.skip_reasons(frame.decision, frame.canonical_label, None))
        reasons.append(SkipReason.MISSING_MODEL_OUTPUT)
        return (
            ReplayFrame(
                decision=frame.decision,
                canonical_label=frame.canonical_label,
                score=None,
                skip_reasons=tuple(dict.fromkeys(reasons)),
            ),
            False,
            True,
            False,
        )
    score = prediction.to_shared_score(
        side=frame.decision.side,
        observed_high=_model_visible_observed_high(frame.decision),
        parser_confidence=frame.decision.parser_confidence,
        settlement_unit=frame.canonical_label.settlement_unit if frame.canonical_label else None,
        settlement_precision=frame.canonical_label.settlement_precision if frame.canonical_label else None,
    )
    score_cache.add(context=context, prediction=prediction, score=score)
    return (
        ReplayFrame(
            decision=frame.decision,
            canonical_label=frame.canonical_label,
            score=score,
            skip_reasons=point_policy.skip_reasons(frame.decision, frame.canonical_label, score),
        ),
        True,
        False,
        False,
    )


def _throughput_gate_stage_order() -> tuple[str, ...]:
    return (
        "total_official_rows",
        "rows_with_model_output",
        "rows_with_executable_bid_ask",
        "rows_passing_lead_day_gate",
        "rows_passing_spread_gate",
        "rows_passing_ask_min_max_gate",
        "rows_passing_fair_value_minimum",
        "rows_passing_net_edge_minimum",
        "rows_passing_model_agreement",
        "rows_passing_probability_stdev",
        "rows_passing_no_counter_event_cap",
        "rows_passing_bounded_bucket_gates",
    )


def _baseline_throughput_gate_flags(
    frame: ReplayFrame,
    config: ConservativeFVEdgeConfig,
) -> Mapping[str, bool]:
    decision = frame.decision
    quote = decision.quote
    score = frame.score
    flags = {stage: False for stage in _throughput_gate_stage_order()}
    flags["total_official_rows"] = True
    if score is None:
        return flags
    flags["rows_with_model_output"] = True
    executable = quote is not None and quote.bid is not None and quote.ask is not None
    flags["rows_with_executable_bid_ask"] = executable
    flags["rows_passing_lead_day_gate"] = decision.lead_days in config.allowed_lead_days
    flags["rows_passing_spread_gate"] = (
        quote is not None and quote.spread is not None and quote.spread <= config.max_spread
    )
    ask_ok = False
    if quote is not None and quote.ask is not None:
        min_ask = config.yes_side_min_ask if decision.side == "YES" else config.min_ask
        max_ask = config.no_side_max_ask if decision.side == "NO" else config.max_ask
        ask_ok = quote.ask >= min_ask and quote.ask <= max_ask
    flags["rows_passing_ask_min_max_gate"] = ask_ok
    if decision.side == "YES":
        fair_value_ok = score.p_yes >= config.yes_min_probability
    elif decision.side == "NO":
        fair_value_ok = score.p_no >= config.no_min_probability
    else:
        fair_value_ok = False
    flags["rows_passing_fair_value_minimum"] = fair_value_ok
    net_edge_ok = False
    if quote is not None and quote.ask is not None:
        strategy = ConservativeFVEdgeStrategy(config)
        net_edge = score.p_side - quote.ask - strategy._uncertainty_buffer(score)
        net_edge_ok = net_edge >= strategy._edge_floor(quote.ask)
    flags["rows_passing_net_edge_minimum"] = net_edge_ok
    flags["rows_passing_model_agreement"] = (
        score.model_count >= config.model_count_min
        and score.model_agreement >= config.model_agreement_min
    )
    if score.probability_stdev is None:
        stdev_ok = False
    elif _is_bounded_bucket(decision.bucket_type):
        stdev_ok = score.probability_stdev <= config.bounded_probability_stdev_max
    else:
        stdev_ok = score.probability_stdev <= config.open_bucket_probability_stdev_max
    flags["rows_passing_probability_stdev"] = stdev_ok
    flags["rows_passing_no_counter_event_cap"] = (
        decision.side != "NO" or score.p_yes <= config.no_counter_event_probability_max
    )
    bounded_ok = True
    if _is_bounded_bucket(decision.bucket_type):
        if quote is None or quote.ask is None:
            bounded_ok = False
        else:
            net_edge = score.p_side - quote.ask - ConservativeFVEdgeStrategy(config)._uncertainty_buffer(score)
            bounded_ok = (
                score.p_side >= config.bounded_min_probability
                and quote.ask >= config.bounded_min_ask
                and net_edge >= config.bounded_min_net_edge
            )
    flags["rows_passing_bounded_bucket_gates"] = bounded_ok
    return flags


def _add_gate_funnel_counts(
    cumulative: Counter[str],
    individual_pass: Counter[str],
    individual_fail: Counter[str],
    flags: Mapping[str, bool],
) -> None:
    still_passing = True
    for stage in _throughput_gate_stage_order():
        passed = bool(flags.get(stage))
        if passed:
            individual_pass[stage] += 1
        else:
            individual_fail[stage] += 1
        if still_passing and passed:
            cumulative[stage] += 1
        else:
            still_passing = False


def _one_gate_loosen_summary(
    variants: Mapping[str, Mapping[str, Any]],
    baseline: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    baseline_pre_cap = int(baseline.get("pre_cap_candidate_rows") or 0)
    baseline_orders = int(baseline.get("order_intents") or 0)
    baseline_fills = int(baseline.get("fills") or 0)
    rows: list[Mapping[str, Any]] = []
    for key, summary in variants.items():
        if summary.get("group") != "one_gate":
            continue
        rows.append(
            {
                "variant_key": key,
                "label": summary.get("label"),
                "loosened_dimensions": summary.get("loosened_dimensions"),
                "pre_cap_candidate_rows": summary.get("pre_cap_candidate_rows"),
                "additional_pre_cap_candidate_rows_vs_baseline": (
                    int(summary.get("pre_cap_candidate_rows") or 0) - baseline_pre_cap
                ),
                "order_intents": summary.get("order_intents"),
                "additional_order_intents_vs_baseline": (
                    int(summary.get("order_intents") or 0) - baseline_orders
                ),
                "fills": summary.get("fills"),
                "additional_fills_vs_baseline": int(summary.get("fills") or 0) - baseline_fills,
                "cap_block_counts": summary.get("cap_block_counts"),
                "estimated_net_fee_adjusted_pnl_usd": summary.get("frontier_metrics", {}).get(
                    "estimated_net_fee_adjusted_pnl_usd"
                ),
                "roi_after_fees": summary.get("frontier_metrics", {}).get("roi_after_fees"),
            }
        )
    return rows


def _add_throughput_coverage(
    coverage: dict[str, Counter[str]],
    coverage_pnl: dict[str, Counter[str]],
    record: TradeAccountingRecord,
    decision: DecisionSnapshot,
) -> None:
    dimensions = {
        "city": record.normalized_city or "missing",
        "country_region": decision.country_region or "missing",
        "timezone": decision.timezone or "missing",
        "decision_hour_utc": str(record.decision_hour_utc),
        "market_month": _trade_market_month_key(record),
        "market_week": _trade_market_week_key(record),
        "side": record.side or "missing",
        "bucket_type": record.bucket_type or "missing",
        "lead_day": str(record.lead_days),
        "ask_price_bucket": _price_bucket(record.fill_price),
        "spread_bucket": _spread_bucket(decision.quote.spread if decision.quote else None),
        "visible_size_bucket": _visible_size_bucket(decision.quote.ask_size if decision.quote else None),
    }
    for dimension, bucket in dimensions.items():
        coverage[dimension][bucket] += 1
        coverage_pnl[dimension][bucket] += record.net_fee_adjusted_realized_pnl_usd


def _throughput_coverage_report(
    coverage: Mapping[str, Counter[str]],
    coverage_pnl: Mapping[str, Counter[str]],
) -> Mapping[str, Mapping[str, Any]]:
    report: dict[str, Mapping[str, Any]] = {}
    for dimension, counter in sorted(coverage.items()):
        report[dimension] = {
            key: {
                "trades": count,
                "net_fee_adjusted_pnl_usd": coverage_pnl.get(dimension, Counter()).get(key, 0.0),
            }
            for key, count in counter.most_common(200)
        }
    return report


def _price_bucket(value: Optional[float]) -> str:
    if value is None:
        return "missing"
    if value < 0.20:
        return "<0.20"
    if value < 0.40:
        return "0.20-0.40"
    if value < 0.60:
        return "0.40-0.60"
    if value < 0.80:
        return "0.60-0.80"
    if value <= 0.95:
        return "0.80-0.95"
    return ">0.95"


def _spread_bucket(value: Optional[float]) -> str:
    if value is None:
        return "missing"
    if value <= 0.01:
        return "<=0.01"
    if value <= 0.03:
        return "0.01-0.03"
    if value <= 0.05:
        return "0.03-0.05"
    if value <= 0.10:
        return "0.05-0.10"
    return ">0.10"


def _visible_size_bucket(value: Optional[float]) -> str:
    if value is None:
        return "missing"
    if value < 1:
        return "<1"
    if value < 5:
        return "1-5"
    if value < 20:
        return "5-20"
    if value < 100:
        return "20-100"
    return ">=100"


@dataclass(frozen=True)
class Gen2ValidationVariantSpec:
    key: str
    label: str
    stage: str
    strategy_family: str
    strategy: StrategyPlugin
    config_report: Mapping[str, Any]
    portfolio_caps: PortfolioCaps
    parameters: Mapping[str, Any]


@dataclass
class Gen2ValidationVariantState:
    spec: Gen2ValidationVariantSpec
    starting_cash_usd: float
    fee_config: FeeModelConfig
    fill_policy: FillPolicy = field(default_factory=FillPolicy)

    def __post_init__(self) -> None:
        self.portfolio_state = PortfolioState(self.spec.portfolio_caps)
        self.ledger = self._new_ledger()
        self.delayed_ledger = self._new_ledger()
        self.worse_fill_ledger = self._new_ledger()
        self.pre_cap_candidate_rows = 0
        self.order_intents: list[OrderIntent] = []
        self.trade_records: list[TradeAccountingRecord] = []
        self.delayed_trade_records: list[TradeAccountingRecord] = []
        self.worse_fill_trade_records: list[TradeAccountingRecord] = []
        self.fill_status_counts: Counter[str] = Counter()
        self.delayed_fill_status_counts: Counter[str] = Counter()
        self.worse_fill_status_counts: Counter[str] = Counter()
        self.strategy_skip_counts: Counter[str] = Counter()
        self.cap_block_counts: Counter[str] = Counter()
        self.unresolved_fills = 0

    def _new_ledger(self) -> ReplayLedger:
        return ReplayLedger(
            starting_cash_usd=self.starting_cash_usd,
            assumptions=AccountingAssumptions(
                fee_model=self.fee_config.fee_model,
                fee_rate=self.fee_config.resolved_fee_rate,
                fill_policy=self.fill_policy.policy_name,
            ),
        )

    def process_frame(
        self,
        frame: ReplayFrame,
        *,
        delayed_quote_lookup: Optional[CompactQuoteHistoryLookup],
        delay_seconds: int,
        worse_fill_tick: float,
    ) -> None:
        if frame.skip_reasons:
            for reason in frame.skip_reasons:
                self.strategy_skip_counts[reason.value] += 1
            return
        decision = self.spec.strategy.evaluate(frame.strategy_input())
        if not decision.orders:
            for reason in _metadata_skip_reasons(decision.metadata):
                self.strategy_skip_counts[reason.value] += 1
            return
        for order in decision.orders:
            self.pre_cap_candidate_rows += 1
            self.portfolio_state.release_expired(frame.decision.decision_time_utc)
            exposure = _exposure_intent_from_order(order, frame.decision)
            cap_result = self.portfolio_state.reserve(exposure)
            if not cap_result.passed:
                for reason in cap_result.skip_reasons:
                    self.cap_block_counts[reason.value] += 1
                continue
            self.order_intents.append(order)
            base_fill_result = simulate_order_fill(
                order,
                frame.decision,
                fill_policy=self.fill_policy,
                fee_config=self.fee_config,
            )
            self._apply_fill_result(
                frame=frame,
                order=order,
                fill_result=base_fill_result,
                ledger=self.ledger,
                records=self.trade_records,
                fill_status_counts=self.fill_status_counts,
            )
            self._run_worse_fill_stress(
                frame=frame,
                order=order,
                worse_fill_tick=worse_fill_tick,
            )
            self._run_delayed_entry_stress(
                frame=frame,
                order=order,
                delayed_quote_lookup=delayed_quote_lookup,
                delay_seconds=delay_seconds,
            )

    def _run_worse_fill_stress(
        self,
        *,
        frame: ReplayFrame,
        order: OrderIntent,
        worse_fill_tick: float,
    ) -> None:
        quote = frame.decision.quote
        if quote is None:
            self.worse_fill_status_counts["missing_quote"] += 1
            return
        stress_quote = _quote_with_worse_fill_tick(
            quote,
            action=order.action,
            tick=worse_fill_tick,
        )
        stress_decision = replace(frame.decision, quote=stress_quote)
        stress_frame = replace(frame, decision=stress_decision)
        fill_result = simulate_order_fill(
            order,
            stress_decision,
            fill_policy=self.fill_policy,
            fee_config=self.fee_config,
        )
        self._apply_fill_result(
            frame=stress_frame,
            order=order,
            fill_result=fill_result,
            ledger=self.worse_fill_ledger,
            records=self.worse_fill_trade_records,
            fill_status_counts=self.worse_fill_status_counts,
        )

    def _run_delayed_entry_stress(
        self,
        *,
        frame: ReplayFrame,
        order: OrderIntent,
        delayed_quote_lookup: Optional[CompactQuoteHistoryLookup],
        delay_seconds: int,
    ) -> None:
        if delayed_quote_lookup is None:
            self.delayed_fill_status_counts["delayed_lookup_unavailable"] += 1
            return
        earliest = frame.decision.decision_time_utc + timedelta(seconds=delay_seconds)
        delayed_quote = delayed_quote_lookup.quote_after(
            token_id=order.token_id,
            earliest_time_utc=earliest,
        )
        if delayed_quote is None:
            self.delayed_fill_status_counts["missing_delayed_quote"] += 1
            return
        stress_decision = replace(frame.decision, quote=delayed_quote)
        stress_frame = replace(frame, decision=stress_decision)
        fill_result = simulate_order_fill(
            order,
            stress_decision,
            fill_policy=self.fill_policy,
            fee_config=self.fee_config,
            fill_timestamp_utc=delayed_quote.timestamp_utc,
        )
        self._apply_fill_result(
            frame=stress_frame,
            order=order,
            fill_result=fill_result,
            ledger=self.delayed_ledger,
            records=self.delayed_trade_records,
            fill_status_counts=self.delayed_fill_status_counts,
        )

    def _apply_fill_result(
        self,
        *,
        frame: ReplayFrame,
        order: OrderIntent,
        fill_result: FillResult,
        ledger: ReplayLedger,
        records: list[TradeAccountingRecord],
        fill_status_counts: Counter[str],
    ) -> None:
        fill_status_counts[fill_result.status] += 1
        if fill_result.fill is None:
            return
        fill = fill_result.fill
        ledger.apply_fill(fill)
        if order.action == OrderAction.BUY and frame.canonical_label is not None:
            settlement = ledger.settle_position(order.token_id, order.side, frame.canonical_label)
            records.append(
                _trade_record_from_fill_and_settlement(
                    frame=frame,
                    order=order,
                    fill=fill,
                    settlement=settlement,
                )
            )
        else:
            self.unresolved_fills += 1

    def base_accounting(self) -> Mapping[str, Any]:
        return _compact_accounting_report(
            _fixed_strategy_accounting_report(
                trade_records=self.trade_records,
                fill_status_counts=self.fill_status_counts,
                order_intents=self.order_intents,
                ledger=self.ledger,
                starting_cash_usd=self.starting_cash_usd,
                fee_config=self.fee_config,
            )
        )

    def stress_accounting(self) -> Mapping[str, Mapping[str, Any]]:
        delayed = _compact_accounting_report(
            _fixed_strategy_accounting_report(
                trade_records=self.delayed_trade_records,
                fill_status_counts=self.delayed_fill_status_counts,
                order_intents=self.order_intents,
                ledger=self.delayed_ledger,
                starting_cash_usd=self.starting_cash_usd,
                fee_config=self.fee_config,
            )
        )
        worse_fill = _compact_accounting_report(
            _fixed_strategy_accounting_report(
                trade_records=self.worse_fill_trade_records,
                fill_status_counts=self.worse_fill_status_counts,
                order_intents=self.order_intents,
                ledger=self.worse_fill_ledger,
                starting_cash_usd=self.starting_cash_usd,
                fee_config=self.fee_config,
            )
        )
        return {"delayed_entry_2s": delayed, "worse_fill_1tick": worse_fill}

    def summary(self) -> Mapping[str, Any]:
        accounting = self.base_accounting()
        stress = self.stress_accounting()
        pnl = accounting.get("pnl", {})
        delayed_pnl = stress["delayed_entry_2s"].get("pnl", {})
        worse_pnl = stress["worse_fill_1tick"].get("pnl", {})
        return {
            "variant_key": self.spec.key,
            "label": self.spec.label,
            "stage": self.spec.stage,
            "strategy_family": self.spec.strategy_family,
            "parameters": dict(self.spec.parameters),
            "config": dict(self.spec.config_report),
            "portfolio_caps": _portfolio_caps_summary(self.spec.portfolio_caps),
            "pre_cap_candidate_rows": self.pre_cap_candidate_rows,
            "order_intents": len(self.order_intents),
            "fills": len(self.ledger.fills),
            "settlements": len(self.ledger.settlements),
            "cap_block_counts": dict(sorted(self.cap_block_counts.items())),
            "strategy_skip_counts_top": dict(self.strategy_skip_counts.most_common(20)),
            "base_accounting": accounting,
            "stress_accounting": stress,
            "monthly_window_split_pnl": accounting.get("pnl_by_month", {}),
            "pnl_by_target_date": _trade_slice_report(
                self.trade_records,
                lambda record: record.market_date_local or "missing",
            ),
            "city_region_timezone_concentration": {
                "city": accounting.get("pnl_by_city", {}),
                "country_region": _trade_slice_report(
                    self.trade_records,
                    lambda record: record.country_region or "missing",
                ),
                "timezone": _trade_slice_report(
                    self.trade_records,
                    lambda record: record.timezone or "missing",
                ),
            },
            "ranking_metrics": {
                "net_fee_adjusted_pnl_usd": pnl.get("net_fee_adjusted_realized_pnl_usd"),
                "roi_after_fees": pnl.get("roi_after_fees_on_deployed_capital"),
                "trade_count": len(self.trade_records),
                "win_count": pnl.get("win_count"),
                "loss_count": pnl.get("loss_count"),
                "total_fees_usd": pnl.get("total_fees_usd"),
                "delayed_entry_net_pnl_usd": delayed_pnl.get("net_fee_adjusted_realized_pnl_usd"),
                "worse_fill_net_pnl_usd": worse_pnl.get("net_fee_adjusted_realized_pnl_usd"),
            },
        }


def _compact_accounting_report(report: Mapping[str, Any]) -> Mapping[str, Any]:
    compact = dict(report)
    compact.pop("all_trades", None)
    return compact


def _gen2_variant_specs(starting_cash_usd: float) -> list[Gen2ValidationVariantSpec]:
    specs: list[Gen2ValidationVariantSpec] = []
    conservative_config = ConservativeFVEdgeConfig()
    conservative_caps = PortfolioCaps(bankroll_usd=starting_cash_usd)
    specs.append(
        Gen2ValidationVariantSpec(
            key="stage_a_conservative_fv_edge_baseline",
            label="Stage A Conservative FV Edge baseline",
            stage="A_baseline_realism",
            strategy_family="conservative_fv_edge",
            strategy=ConservativeFVEdgeStrategy(conservative_config),
            config_report=conservative_config.as_report_dict(),
            portfolio_caps=conservative_caps,
            parameters={
                "lead_day_variant": "current_1_2",
                "fee_model_default": DEFAULT_STRATEGY_REPLAY_FEE_MODEL,
            },
        )
    )
    baseline_no_config = _old_no_tail_config(
        starting_cash_usd=starting_cash_usd,
        no_cap=0.14,
        bounded_min_ask=0.70,
        lead_days=(1, 2),
        sizing_mode="fractional_kelly",
        kelly_fraction=0.50,
        flat_order_size=None,
        max_order_notional_usd=_max_order_notional_for_caps(starting_cash_usd, 0.25, True),
    )
    specs.append(
        _old_no_tail_spec(
            key="stage_a_old_no_tail_baseline_no14_pos25_kelly50_window_bprice70",
            label="Stage A old high-confidence NO-tail baseline",
            stage="A_baseline_realism",
            config=baseline_no_config,
            starting_cash_usd=starting_cash_usd,
            position_cap_fraction=0.25,
            window_bankroll=True,
            parameters={
                "no_counter_event_cap": 0.14,
                "position_cap_fraction": 0.25,
                "sizing_mode": "fractional_kelly",
                "kelly_fraction": 0.50,
                "bankroll_scope": "window_bankroll",
                "bounded_min_price": 0.70,
                "lead_day_variant": "current_1_2",
            },
        )
    )

    for position_cap in (0.10, 0.20, 0.25, 0.35):
        for window_bankroll in (True, False):
            bankroll_label = "window_bankroll" if window_bankroll else "whole_bankroll"
            for flat_size in (1.0, 2.0, 5.0):
                config = _old_no_tail_config(
                    starting_cash_usd=starting_cash_usd,
                    no_cap=0.14,
                    bounded_min_ask=0.70,
                    lead_days=(1, 2),
                    sizing_mode="flat",
                    flat_order_size=flat_size,
                    kelly_fraction=None,
                    max_order_notional_usd=_max_order_notional_for_caps(
                        starting_cash_usd,
                        position_cap,
                        window_bankroll,
                    ),
                )
                specs.append(
                    _old_no_tail_spec(
                        key=(
                            f"stage_b_old_no_tail_no14_pos{_pct_key(position_cap)}_"
                            f"flat{int(flat_size)}_{bankroll_label}_bprice70"
                        ),
                        label="Stage B old NO-tail cap/sizing flat",
                        stage="B_cap_sizing_grid",
                        config=config,
                        starting_cash_usd=starting_cash_usd,
                        position_cap_fraction=position_cap,
                        window_bankroll=window_bankroll,
                        parameters={
                            "no_counter_event_cap": 0.14,
                            "position_cap_fraction": position_cap,
                            "sizing_mode": "flat",
                            "flat_order_size_usd": flat_size,
                            "bankroll_scope": bankroll_label,
                            "bounded_min_price": 0.70,
                            "lead_day_variant": "current_1_2",
                        },
                    )
                )
            for kelly_fraction in (0.25, 0.50, 0.75):
                config = _old_no_tail_config(
                    starting_cash_usd=starting_cash_usd,
                    no_cap=0.14,
                    bounded_min_ask=0.70,
                    lead_days=(1, 2),
                    sizing_mode="fractional_kelly",
                    flat_order_size=None,
                    kelly_fraction=kelly_fraction,
                    max_order_notional_usd=_max_order_notional_for_caps(
                        starting_cash_usd,
                        position_cap,
                        window_bankroll,
                    ),
                )
                specs.append(
                    _old_no_tail_spec(
                        key=(
                            f"stage_b_old_no_tail_no14_pos{_pct_key(position_cap)}_"
                            f"kelly{_pct_key(kelly_fraction)}_{bankroll_label}_bprice70"
                        ),
                        label="Stage B old NO-tail cap/sizing Kelly",
                        stage="B_cap_sizing_grid",
                        config=config,
                        starting_cash_usd=starting_cash_usd,
                        position_cap_fraction=position_cap,
                        window_bankroll=window_bankroll,
                        parameters={
                            "no_counter_event_cap": 0.14,
                            "position_cap_fraction": position_cap,
                            "sizing_mode": "fractional_kelly",
                            "kelly_fraction": kelly_fraction,
                            "bankroll_scope": bankroll_label,
                            "bounded_min_price": 0.70,
                            "lead_day_variant": "current_1_2",
                        },
                    )
                )

    for no_cap in (0.10, 0.12, 0.14, 0.16):
        for bounded_price in (0.70, 0.75):
            for lead_days, lead_label in (((1, 2), "current_1_2"), ((0, 1, 2), "same_day_plus_1_2")):
                config = _old_no_tail_config(
                    starting_cash_usd=starting_cash_usd,
                    no_cap=no_cap,
                    bounded_min_ask=bounded_price,
                    lead_days=lead_days,
                    sizing_mode="fractional_kelly",
                    flat_order_size=None,
                    kelly_fraction=0.50,
                    max_order_notional_usd=_max_order_notional_for_caps(
                        starting_cash_usd,
                        0.25,
                        True,
                    ),
                )
                specs.append(
                    _old_no_tail_spec(
                        key=(
                            f"stage_c_old_no_tail_no{_pct_key(no_cap)}_pos25_"
                            f"kelly50_window_bprice{int(bounded_price * 100)}_{lead_label}"
                        ),
                        label="Stage C old NO-tail signal perturbation",
                        stage="C_signal_perturbation_grid",
                        config=config,
                        starting_cash_usd=starting_cash_usd,
                        position_cap_fraction=0.25,
                        window_bankroll=True,
                        parameters={
                            "no_counter_event_cap": no_cap,
                            "position_cap_fraction": 0.25,
                            "sizing_mode": "fractional_kelly",
                            "kelly_fraction": 0.50,
                            "bankroll_scope": "window_bankroll",
                            "bounded_min_price": bounded_price,
                            "lead_day_variant": lead_label,
                            "same_day_inclusion": 0 in lead_days,
                        },
                    )
                )
    return _dedupe_variant_specs(specs)


def _old_no_tail_config(
    *,
    starting_cash_usd: float,
    no_cap: float,
    bounded_min_ask: float,
    lead_days: tuple[int, ...],
    sizing_mode: str,
    flat_order_size: Optional[float],
    kelly_fraction: Optional[float],
    max_order_notional_usd: float,
) -> HighConfidenceNOTailConfig:
    return HighConfidenceNOTailConfig(
        order_notional_usd=flat_order_size or 1.0,
        sizing_mode=sizing_mode,
        sizing_bankroll_usd=starting_cash_usd,
        kelly_fraction=kelly_fraction or 0.0,
        max_order_notional_usd=max_order_notional_usd,
        allowed_lead_days=lead_days,
        no_min_probability=1.0 - no_cap,
        no_counter_event_probability_max=no_cap,
        bounded_min_ask=bounded_min_ask,
    )


def _old_no_tail_spec(
    *,
    key: str,
    label: str,
    stage: str,
    config: HighConfidenceNOTailConfig,
    starting_cash_usd: float,
    position_cap_fraction: float,
    window_bankroll: bool,
    parameters: Mapping[str, Any],
) -> Gen2ValidationVariantSpec:
    caps = _grid_portfolio_caps(
        starting_cash_usd=starting_cash_usd,
        position_cap_fraction=position_cap_fraction,
        window_bankroll=window_bankroll,
    )
    return Gen2ValidationVariantSpec(
        key=key,
        label=label,
        stage=stage,
        strategy_family="old_high_confidence_no_tail",
        strategy=HighConfidenceNOTailStrategy(config),
        config_report=config.as_report_dict(),
        portfolio_caps=caps,
        parameters=dict(parameters),
    )


def _grid_portfolio_caps(
    *,
    starting_cash_usd: float,
    position_cap_fraction: float,
    window_bankroll: bool,
) -> PortfolioCaps:
    window_fraction = 0.25 if window_bankroll else 1.0
    per_position_window_fraction = position_cap_fraction / window_fraction
    return PortfolioCaps(
        bankroll_usd=starting_cash_usd,
        window_bankroll_fraction=window_fraction,
        per_position_window_fraction=per_position_window_fraction,
        per_city_date_total_fraction=max(0.15, position_cap_fraction),
        per_region_total_fraction=0.35,
        per_token_side_total_fraction=max(0.25, position_cap_fraction),
        per_settlement_source_total_fraction=1.0,
        max_primary_exposures_per_city_date_per_run=1,
    )


def _max_order_notional_for_caps(
    starting_cash_usd: float,
    position_cap_fraction: float,
    window_bankroll: bool,
) -> float:
    window_cap = starting_cash_usd * (0.25 if window_bankroll else 1.0)
    position_cap = starting_cash_usd * position_cap_fraction
    return min(window_cap, position_cap)


def _pct_key(value: float) -> str:
    return str(int(round(value * 100)))


def _dedupe_variant_specs(specs: Sequence[Gen2ValidationVariantSpec]) -> list[Gen2ValidationVariantSpec]:
    deduped: dict[str, Gen2ValidationVariantSpec] = {}
    for spec in specs:
        deduped.setdefault(spec.key, spec)
    return list(deduped.values())


OLD_NO_TAIL_LIVE_FORWARD_SHADOW_PROFILE_KEY = (
    "old_high_confidence_no_tail_no14_flat2_window_bankroll_bprice70"
)
OLD_NO_TAIL_LIVE_FORWARD_SHADOW_SCHEMA_VERSION = (
    "dailyweather.old_no_tail.live_forward_shadow.v1"
)


OLD_NO_TAIL_LIVE_FORWARD_SHADOW_PREREGISTRATION: Mapping[str, Any] = {
    "registered_before_forward_collection": True,
    "profile_key": OLD_NO_TAIL_LIVE_FORWARD_SHADOW_PROFILE_KEY,
    "profile_family": "old_high_confidence_no_tail",
    "frozen_semantics": {
        "strategy_family": "old_high_confidence_no_tail",
        "counter_event_cap": "no14",
        "sizing": "flat_2_usd",
        "bankroll_scope": "window_bankroll",
        "bounded_min_price": "bprice70",
    },
    "expected_trades_per_utc_day": {
        "basis": "prospective monitoring bandwidth only; not derived from historical PnL",
        "point_estimate": 1.0,
        "low": 0.0,
        "high": 4.0,
        "decision_windows_per_day": 4,
    },
    "hit_rate_confidence_bands": {
        "basis": "forward shadow settlement analysis only; not expected performance",
        "primary_interval": "wilson_95pct",
        "secondary_interval": "wilson_80pct",
        "minimum_resolved_would_trades_before_interpretation": 30,
        "labels": {
            "below_coinflip": "95pct_upper_bound_below_0.50",
            "inconclusive": "95pct_interval_crosses_0.50",
            "supportive_weather_hit_rate": "80pct_lower_bound_at_or_above_0.50_after_minimum_sample",
        },
    },
    "claim_boundaries": [
        "read_only_shadow_logging_only",
        "no_real_orders",
        "no_historical_pnl_expected_performance_claim",
        "no_live_money_promotion_from_shadow_log_alone",
    ],
}


def old_no_tail_live_forward_shadow_spec(
    *,
    starting_cash_usd: float = 100.0,
) -> Gen2ValidationVariantSpec:
    config = _old_no_tail_config(
        starting_cash_usd=starting_cash_usd,
        no_cap=0.14,
        bounded_min_ask=0.70,
        lead_days=(1, 2),
        sizing_mode="flat",
        flat_order_size=2.0,
        kelly_fraction=None,
        max_order_notional_usd=_max_order_notional_for_caps(
            starting_cash_usd,
            0.25,
            True,
        ),
    )
    return _old_no_tail_spec(
        key=OLD_NO_TAIL_LIVE_FORWARD_SHADOW_PROFILE_KEY,
        label="Read-only old NO-tail live-forward shadow",
        stage="forward_shadow_preregistered",
        config=config,
        starting_cash_usd=starting_cash_usd,
        position_cap_fraction=0.25,
        window_bankroll=True,
        parameters={
            "no_counter_event_cap": 0.14,
            "sizing_mode": "flat",
            "flat_order_size_usd": 2.0,
            "bankroll_scope": "window_bankroll",
            "bounded_min_price": 0.70,
            "lead_day_variant": "current_1_2",
            "read_only_shadow": True,
            "real_orders_allowed": False,
        },
    )


def run_old_no_tail_live_forward_shadow_logging(
    dataset: Any,
    *,
    starting_cash_usd: float = 100.0,
    limit: int = 250,
    model_runner: Optional[WeatherModelRunner] = None,
    score_cache: Optional[ReplayGeneratedScoreCache] = None,
    leakage_validators: Sequence[LeakageValidator] = (DefaultLeakageValidator(),),
    progress_every: int = 0,
) -> Mapping[str, Any]:
    spec = old_no_tail_live_forward_shadow_spec(starting_cash_usd=starting_cash_usd)
    portfolio_state = PortfolioState(spec.portfolio_caps)
    cache = score_cache or ReplayGeneratedScoreCache()
    point_policy = PointInTimePolicy(require_shared_score=True, require_canonical_label=False)
    events: list[Mapping[str, Any]] = []
    frame_skip_counts: Counter[str] = Counter()
    no_trade_reason_counts: Counter[str] = Counter()
    cap_block_counts: Counter[str] = Counter()
    leakage_violations: Counter[str] = Counter()
    settlement_counts: Counter[str] = Counter()
    market_discovery_counts: Counter[str] = Counter()
    quote_counts: Counter[str] = Counter()
    forecast_counts: Counter[str] = Counter()
    model_outputs = 0
    missing_model_outputs = 0
    firewall_violations = 0
    would_trade_count = 0
    total_rows = 0
    runner_metadata = dict(model_runner.metadata()) if model_runner is not None else {}
    generated_at = datetime.now(timezone.utc)
    start_time = time.perf_counter()

    for source_frame in dataset.iter_frames(limit=limit):
        total_rows += 1
        if progress_every and total_rows % progress_every == 0:
            print(f"old-no-tail-live-forward-shadow rows={total_rows}", flush=True)
        frame, generated_score, missing_model, firewall_violation = _score_frame_for_throughput(
            source_frame,
            model_runner=model_runner,
            score_cache=cache,
            point_policy=point_policy,
        )
        if generated_score or frame.score is not None:
            model_outputs += 1
        if missing_model:
            missing_model_outputs += 1
        if firewall_violation:
            firewall_violations += 1
        violations = tuple(
            violation
            for validator in leakage_validators
            for violation in validator.validate(frame)
        )
        for violation in violations:
            leakage_violations[violation.code] += 1
        if violations:
            frame = replace(
                frame,
                skip_reasons=_frame_skip_reasons_with_leakage(frame.skip_reasons, violations),
            )
        for reason in frame.skip_reasons:
            frame_skip_counts[reason.value] += 1
        event = _old_no_tail_shadow_event(
            frame=frame,
            spec=spec,
            portfolio_state=portfolio_state,
            collected_at_utc=generated_at,
        )
        events.append(event)
        decision = event["decision"]
        settlement_counts[str(event["settlement"].get("status") or "unknown")] += 1
        market_discovery_counts["markets_seen"] += 1
        if event["quote"].get("timestamp_utc"):
            quote_counts["quotes_logged"] += 1
        else:
            quote_counts["quotes_missing"] += 1
        if event["open_meteo_forecast"].get("run_time_utc"):
            forecast_counts["forecasts_logged"] += 1
        else:
            forecast_counts["forecasts_missing"] += 1
        if decision.get("would_trade"):
            would_trade_count += 1
        else:
            for reason in decision.get("no_trade_reasons") or ("unknown_no_trade_reason",):
                no_trade_reason_counts[str(reason)] += 1
        for reason in decision.get("cap_skip_reasons") or ():
            cap_block_counts[str(reason)] += 1

    runtime_seconds = time.perf_counter() - start_time
    return {
        "schema_version": OLD_NO_TAIL_LIVE_FORWARD_SHADOW_SCHEMA_VERSION,
        "generated_at_utc": generated_at.isoformat(),
        "mode": "read_only_live_forward_shadow_logging",
        "profile_key": spec.key,
        "strategy_family": spec.strategy_family,
        "strategy_label": spec.label,
        "read_only": True,
        "real_order_placement_allowed": False,
        "real_orders_placed": 0,
        "fills_simulated": False,
        "ledger_accounting_performed": False,
        "historical_pnl_numbers_are_expected_performance": False,
        "pre_registered_plan": OLD_NO_TAIL_LIVE_FORWARD_SHADOW_PREREGISTRATION,
        "frozen_strategy_config": dict(spec.config_report),
        "frozen_strategy_parameters": dict(spec.parameters),
        "portfolio_caps": _portfolio_caps_summary(spec.portfolio_caps),
        "starting_cash_usd_for_shadow_caps_only": starting_cash_usd,
        "row_limit": limit,
        "rows_seen": total_rows,
        "runtime_seconds": runtime_seconds,
        "rows_per_sec": (total_rows / runtime_seconds) if runtime_seconds else None,
        "market_discovery_counts": dict(sorted(market_discovery_counts.items())),
        "quote_logging_counts": dict(sorted(quote_counts.items())),
        "forecast_logging_counts": dict(sorted(forecast_counts.items())),
        "model_output_counts": {
            "model_outputs_logged": model_outputs,
            "missing_model_outputs": missing_model_outputs,
            "model_input_firewall_violations": firewall_violations,
        },
        "decision_counts": {
            "would_trade_signals": would_trade_count,
            "no_trade_decisions": total_rows - would_trade_count,
        },
        "frame_skip_counts": dict(sorted(frame_skip_counts.items())),
        "no_trade_reason_counts": dict(sorted(no_trade_reason_counts.items())),
        "cap_block_counts": dict(sorted(cap_block_counts.items())),
        "settlement_logging_counts": dict(sorted(settlement_counts.items())),
        "leakage_validation": {"violation_counts": dict(sorted(leakage_violations.items()))},
        "weather_model_runner": {
            "metadata": runner_metadata,
            "fixture_runner_used": runner_metadata.get("fixture_only") == "true",
            "research_only_not_promoted": (
                runner_metadata.get("promotion_status") == "research_only_not_promoted"
            ),
        },
        "score_cache": cache.report_summary(),
        "shadow_portfolio_state": portfolio_state.summary(),
        "shadow_events": events,
    }


def _old_no_tail_shadow_event(
    *,
    frame: ReplayFrame,
    spec: Gen2ValidationVariantSpec,
    portfolio_state: PortfolioState,
    collected_at_utc: datetime,
) -> Mapping[str, Any]:
    decision = frame.decision
    order: Optional[OrderIntent] = None
    strategy_skip_reasons: tuple[SkipReason, ...] = ()
    cap_skip_reasons: tuple[SkipReason, ...] = ()
    exposure_after: Mapping[str, float] = {}
    gate_details: Mapping[str, Any] = {}
    decision_reason = "frame_not_strategy_eligible"
    if frame.skip_reasons:
        strategy_skip_reasons = tuple(frame.skip_reasons)
    else:
        strategy_decision = spec.strategy.evaluate(frame.strategy_input())
        gate_details = dict(strategy_decision.metadata.get("candidate_gate_details") or {})
        strategy_skip_reasons = _metadata_skip_reasons(strategy_decision.metadata)
        if strategy_decision.orders:
            candidate_order = strategy_decision.orders[0]
            portfolio_state.release_expired(decision.decision_time_utc)
            cap_result = portfolio_state.reserve(_exposure_intent_from_order(candidate_order, decision))
            exposure_after = cap_result.exposures_after
            if cap_result.passed:
                order = candidate_order
                decision_reason = candidate_order.reason
            else:
                cap_skip_reasons = cap_result.skip_reasons
                decision_reason = "shadow_portfolio_cap_blocked"
        elif strategy_skip_reasons:
            decision_reason = "strategy_gate_blocked"
        else:
            decision_reason = "strategy_no_action"
            strategy_skip_reasons = (SkipReason.STRATEGY_NO_ACTION,)

    no_trade_reasons = tuple(dict.fromkeys((*strategy_skip_reasons, *cap_skip_reasons)))
    return {
        "schema_version": OLD_NO_TAIL_LIVE_FORWARD_SHADOW_SCHEMA_VERSION,
        "event_type": "old_no_tail_live_forward_shadow_decision",
        "collected_at_utc": collected_at_utc.isoformat(),
        "read_only": True,
        "real_order_placement_allowed": False,
        "profile_key": spec.key,
        "market_discovery": _shadow_market_discovery(decision),
        "quote": _shadow_quote(decision.quote),
        "open_meteo_forecast": _shadow_forecast(decision.forecast, decision),
        "model_output": _shadow_model_output(frame.score),
        "decision": {
            "would_trade": order is not None,
            "decision_reason": decision_reason,
            "would_trade_signal": _shadow_order_intent(order),
            "no_trade_reasons": [reason.value for reason in no_trade_reasons],
            "candidate_gate_details": gate_details,
            "cap_skip_reasons": [reason.value for reason in cap_skip_reasons],
            "shadow_exposure_after": dict(exposure_after),
        },
        "settlement": _shadow_settlement(frame.canonical_label, order.side if order else decision.side),
    }


def _shadow_market_discovery(decision: DecisionSnapshot) -> Mapping[str, Any]:
    return {
        "decision_id": decision.decision_id,
        "universe_row_id": decision.universe_row_id,
        "market_id": decision.market_id,
        "slug": decision.slug,
        "question": decision.question,
        "normalized_city": decision.normalized_city,
        "country_region": decision.country_region,
        "timezone": decision.timezone,
        "market_date_local": decision.market_date_local,
        "bucket_label": decision.bucket_label,
        "bucket_type": decision.bucket_type,
        "threshold_low_f": decision.threshold_low_f,
        "threshold_high_f": decision.threshold_high_f,
        "side": decision.side,
        "yes_token_id": decision.yes_token_id,
        "no_token_id": decision.no_token_id,
        "token_id": decision.token_id,
        "decision_time_utc": decision.decision_time_utc.isoformat(),
        "decision_hour_utc": decision.decision_hour_utc,
        "lead_days": decision.lead_days,
        "quote_store": decision.quote_store,
        "parser_confidence": decision.parser_confidence,
        "data_quality_flags": list(decision.data_quality_flags),
    }


def _shadow_quote(quote: Optional[QuoteSnapshot]) -> Mapping[str, Any]:
    if quote is None:
        return {"status": "missing"}
    return {
        "status": "logged",
        "timestamp_utc": quote.timestamp_utc.isoformat(),
        "bid": quote.bid,
        "ask": quote.ask,
        "mid": quote.mid,
        "last_price": quote.last_price,
        "spread": quote.spread,
        "top_of_book_depth": {
            "bid_size": quote.bid_size,
            "ask_size": quote.ask_size,
            "liquidity_depth": quote.liquidity_depth,
        },
        "source_payload_id": quote.source_payload_id,
        "source_cache_path": quote.source_cache_path,
        "data_quality_flags": list(quote.data_quality_flags),
    }


def _shadow_forecast(
    forecast: Optional[ForecastSnapshot],
    decision: DecisionSnapshot,
) -> Mapping[str, Any]:
    payload: dict[str, Any] = {
        "status": "missing" if forecast is None else "logged",
        "observed_high_so_far": None,
    }
    if forecast is not None:
        payload.update(
            {
                "run_time_utc": forecast.run_time_utc.isoformat(),
                "source_name": forecast.source_name,
                "temperature_2m_max_f": forecast.temperature_2m_max_f,
                "features": dict(forecast.features),
            }
        )
    if decision.observed_high is not None:
        observed = decision.observed_high
        payload["observed_high_so_far"] = {
            "observed_high_f": observed.observed_high_f,
            "observed_high_timestamp_utc": observed.observed_high_timestamp_utc.isoformat(),
            "observation_source": observed.observation_source,
            "observation_final": observed.observation_final,
            "source_family": observed.source_family,
            "source_station": observed.source_station,
            "source_payload_id": decision.observed_high_source_payload_id,
            "raw_cache_path": decision.observed_high_raw_cache_path,
            "data_quality_flags": list(decision.observed_high_data_quality_flags),
        }
    elif decision.observed_high_missing_reason:
        payload["observed_high_so_far"] = {"status": "missing", "reason": decision.observed_high_missing_reason}
    return payload


def _shadow_model_output(score: Optional[SharedScoreContract]) -> Mapping[str, Any]:
    if score is None:
        return {"status": "missing"}
    return {
        "status": "logged",
        "p_yes": score.p_yes,
        "p_no": score.p_no,
        "side": score.side,
        "p_side": score.p_side,
        "model_agreement": score.model_agreement,
        "model_count": score.model_count,
        "probability_stdev": score.probability_stdev,
        "per_source_probabilities": dict(score.per_source_probabilities),
        "remaining_exceed_probability": score.remaining_exceed_probability,
        "observed_high_aware_confidence": score.observed_high_aware_confidence,
        "parser_confidence": score.parser_confidence,
        "settlement_unit": score.settlement_unit,
        "settlement_precision": score.settlement_precision,
        "scored_at_utc": score.scored_at_utc.isoformat() if score.scored_at_utc else None,
        "score_model_version": score.score_model_version,
        "score_family": score.score_family,
        "metadata": dict(score.metadata),
    }


def _shadow_order_intent(order: Optional[OrderIntent]) -> Optional[Mapping[str, Any]]:
    if order is None:
        return None
    return {
        "strategy_name": order.strategy_name,
        "decision_id": order.decision_id,
        "token_id": order.token_id,
        "side": order.side,
        "action": order.action.value,
        "requested_notional_usd": order.requested_notional_usd,
        "requested_shares": order.requested_shares,
        "limit_price": order.limit_price,
        "reason": order.reason,
        "metadata": dict(order.metadata),
    }


def _shadow_settlement(
    label: Optional[CanonicalLabel],
    side: str,
) -> Mapping[str, Any]:
    if label is None:
        return {
            "status": "pending",
            "later_update_required": True,
            "outcome_logged": False,
            "source": None,
        }
    if not label.is_replay_allowed:
        return {
            "status": "excluded_or_manual_review",
            "later_update_required": False,
            "outcome_logged": False,
            "recommended_action": label.recommended_action,
            "source": label.canonical_label_source,
            "audit_flags": list(label.audit_flags),
        }
    return {
        "status": "resolved",
        "later_update_required": False,
        "outcome_logged": True,
        "canonical_label": label.canonical_label,
        "payout_for_shadow_side": label.payout_for_side(side),
        "side": side.upper(),
        "source": label.canonical_label_source,
        "recommended_action": label.recommended_action,
        "settlement_unit": label.settlement_unit,
        "settlement_precision": label.settlement_precision,
        "resolution_date_text": label.resolution_date_text,
        "audit_flags": list(label.audit_flags),
    }


def _throughput_risk_checks(accounting: Mapping[str, Any]) -> Mapping[str, Any]:
    pnl = accounting.get("pnl", {})
    city = accounting.get("pnl_by_city", {})
    hour = accounting.get("pnl_by_decision_hour_utc", {})
    bucket = accounting.get("pnl_by_bucket_type", {})
    return {
        "concentration": accounting.get("concentration"),
        "worst_city": _worst_slice(city),
        "worst_decision_hour_utc": _worst_slice(hour),
        "worst_bucket_type": _worst_slice(bucket),
        "trades_flipped_negative_by_fees": pnl.get("trades_flipped_profitable_to_unprofitable_by_fees"),
        "order_latency_model": "not_modeled",
    }


def _worst_slice(slice_report: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    if not slice_report:
        return None
    key, summary = sorted(
        slice_report.items(),
        key=lambda item: (float(item[1].get("net_fee_adjusted_realized_pnl_usd", item[1].get("realized_pnl_usd", 0.0))), item[0]),
    )[0]
    return {"key": key, **summary}


def _throughput_interpretation(
    *,
    baseline_summary: Mapping[str, Any],
    one_gate_tests: Sequence[Mapping[str, Any]],
    frontier: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any]:
    cap_counts = baseline_summary.get("cap_block_counts", {})
    skip_counts = baseline_summary.get("strategy_skip_counts_top", {})
    limiting_gates = sorted(skip_counts.items(), key=lambda item: item[1], reverse=True)[:8]
    cap_limiters = sorted(cap_counts.items(), key=lambda item: item[1], reverse=True)
    adds_pre_cap = [
        item
        for item in one_gate_tests
        if int(item.get("additional_pre_cap_candidate_rows_vs_baseline") or 0) > 0
    ]
    adds_final = [
        item
        for item in one_gate_tests
        if int(item.get("additional_fills_vs_baseline") or 0) > 0
    ]
    fee_positive_variants = [
        {
            "variant_key": key,
            "label": summary.get("label"),
            "fills": summary.get("fills"),
            "net_fee_adjusted_pnl": summary.get("frontier_metrics", {}).get(
                "estimated_net_fee_adjusted_pnl_usd"
            ),
            "roi_after_fees": summary.get("frontier_metrics", {}).get("roi_after_fees"),
        }
        for key, summary in frontier.items()
        if (summary.get("frontier_metrics", {}).get("estimated_net_fee_adjusted_pnl_usd") or 0.0) >= 0
    ]
    dangerous = [
        {
            "variant_key": key,
            "label": summary.get("label"),
            "reason": "negative_net_fee_adjusted_pnl",
            "net_fee_adjusted_pnl": summary.get("frontier_metrics", {}).get(
                "estimated_net_fee_adjusted_pnl_usd"
            ),
        }
        for key, summary in frontier.items()
        if (summary.get("frontier_metrics", {}).get("estimated_net_fee_adjusted_pnl_usd") or 0.0) < 0
    ]
    return {
        "why_baseline_trades_are_low": [
            "Final fills are capped after baseline gates by portfolio exposure constraints.",
            "Within pre-cap gate evaluation, the largest baseline skip counts identify strict edge/fair-value/bounded-bucket/price gates.",
            "Order latency remains unmodeled and should not be inferred from this diagnostic.",
        ],
        "largest_strategy_gate_skip_counts": limiting_gates,
        "portfolio_cap_limiters": cap_limiters,
        "one_gate_tests_that_add_pre_cap_candidates": adds_pre_cap,
        "one_gate_tests_that_add_final_fills": adds_final,
        "frontier_variants_with_nonnegative_fee_adjusted_pnl": fee_positive_variants,
        "frontier_danger_flags": dangerous,
        "recommended_next_step": (
            "Define a small purged walk-forward/grid protocol using fee-aware accounting and explicit "
            "portfolio-cap variants; do not change live settings from this diagnostic."
        ),
    }


GEN2_STRATEGY_VALIDATION_PASS_CRITERIA: Mapping[str, Any] = {
    "walk_forward_windows": {
        "definition": "target market_date_local calendar months in six_month_v1",
        "required_positive_net_windows": 4,
        "total_expected_windows": 6,
    },
    "concentration_limits": {
        "single_month_abs_pnl_share_max": 0.40,
        "single_city_abs_pnl_share_max": 0.35,
        "single_target_date_abs_pnl_share_max": 0.25,
        "single_region_abs_pnl_share_max": 0.50,
    },
    "risk_limits": {
        "worst_window_net_pnl_min_usd": -5.0,
        "minimum_filled_trades_for_live_candidate": 100,
    },
    "stress_limits": {
        "base_net_pnl_must_be_positive": True,
        "worse_fill_1tick_net_pnl_must_be_positive": True,
        "delayed_entry_2s_net_pnl_must_be_positive": True,
        "delayed_entry_fill_retention_min": 0.90,
    },
    "plateau_limits": {
        "minimum_positive_neighbor_variants": 2,
    },
    "ledger": {
        "full_trade_ledger_must_be_persisted": True,
    },
}


def run_gen2_strategy_validation(
    dataset: Any,
    *,
    starting_cash_usd: float = 100.0,
    limit: int = 0,
    model_runner: Optional[WeatherModelRunner] = None,
    score_cache: Optional[ReplayGeneratedScoreCache] = None,
    delayed_quote_lookup: Optional[CompactQuoteHistoryLookup] = None,
    leakage_validators: Sequence[LeakageValidator] = (DefaultLeakageValidator(),),
    fee_config: FeeModelConfig = FeeModelConfig(
        fee_model=DEFAULT_STRATEGY_REPLAY_FEE_MODEL,
        fee_rate=DEFAULT_STRATEGY_REPLAY_FEE_RATE,
    ),
    delay_seconds: int = DEFAULT_DELAY_STRESS_SECONDS,
    worse_fill_tick: float = DEFAULT_WORSE_FILL_TICK,
    progress_every: int = 0,
) -> tuple[Mapping[str, Any], list[Gen2ValidationVariantState]]:
    specs = _gen2_variant_specs(starting_cash_usd)
    states = [
        Gen2ValidationVariantState(
            spec=spec,
            starting_cash_usd=starting_cash_usd,
            fee_config=fee_config,
            fill_policy=FillPolicy(allow_partial=False),
        )
        for spec in specs
    ]
    cache = score_cache or ReplayGeneratedScoreCache()
    point_policy = PointInTimePolicy(require_shared_score=True)
    total_rows = 0
    model_outputs = 0
    missing_model_outputs = 0
    firewall_violations = 0
    leakage_violations: Counter[str] = Counter()
    frame_skip_counts: Counter[str] = Counter()
    runner_metadata = dict(model_runner.metadata()) if model_runner is not None else {}
    start_time = time.perf_counter()
    for source_frame in dataset.iter_frames(limit=limit):
        total_rows += 1
        if progress_every and total_rows % progress_every == 0:
            print(f"gen2-strategy-validation rows={total_rows}", flush=True)
        frame, generated_score, missing_model, firewall_violation = _score_frame_for_throughput(
            source_frame,
            model_runner=model_runner,
            score_cache=cache,
            point_policy=point_policy,
        )
        if generated_score:
            model_outputs += 1
        if missing_model:
            missing_model_outputs += 1
        if firewall_violation:
            firewall_violations += 1
        violations = tuple(
            violation
            for validator in leakage_validators
            for violation in validator.validate(frame)
        )
        for violation in violations:
            leakage_violations[violation.code] += 1
        if violations:
            frame = replace(
                frame,
                skip_reasons=_frame_skip_reasons_with_leakage(frame.skip_reasons, violations),
            )
        for reason in frame.skip_reasons:
            frame_skip_counts[reason.value] += 1
        for state in states:
            state.process_frame(
                frame,
                delayed_quote_lookup=delayed_quote_lookup,
                delay_seconds=delay_seconds,
                worse_fill_tick=worse_fill_tick,
            )
    runtime_seconds = time.perf_counter() - start_time
    variant_summaries = {state.spec.key: state.summary() for state in states}
    ranked = _rank_gen2_variants(variant_summaries)
    criteria_results = _apply_gen2_reproducibility_gate(
        _evaluate_gen2_pass_fail_criteria(variant_summaries),
        independent_runs_reproduced=1,
        required_runs=GEN2_REPRODUCIBILITY_REQUIRED_RUNS,
        ranked_grid_byte_identical=None,
        full_trade_ledger_byte_identical=None,
    )
    verdict = _gen2_overall_verdict(criteria_results)
    payload = {
        "schema_version": BACKTEST_FRAMEWORK_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "gen2_strategy_validation_fee_aware_walk_forward",
        "dataset_id": "six_month_v1",
        "official_trading_replay_sql_predicate": OFFICIAL_TRADING_REPLAY_SQL_PREDICATE,
        "row_limit": limit,
        "total_official_rows": total_rows,
        "runtime_seconds": runtime_seconds,
        "rows_per_sec": (total_rows / runtime_seconds) if runtime_seconds else None,
        "no_live_trading": True,
        "no_live_setting_promotion": True,
        "no_model_weight_or_default_change": True,
        "no_broad_optimization": True,
        "reproducibility": _gen2_reproducibility_field(
            independent_runs_reproduced=1,
            required_runs=GEN2_REPRODUCIBILITY_REQUIRED_RUNS,
            ranked_grid_byte_identical=None,
            full_trade_ledger_byte_identical=None,
            re_emitted=False,
        ),
        "determinism_controls": {
            "sqlite_row_order": "ORDER BY decision_time_utc, market_id, universe_row_id, side",
            "variant_order_source": "pre_registered_gen2_variant_spec_list",
            "ranking_tie_breakers": [
                "net_fee_adjusted_pnl_usd_desc",
                "trade_count_desc",
                "variant_key_asc",
            ],
            "portfolio_exposure_expiry": (
                "release_expired is called before cap checks; release time is "
                "market local date plus one day at 12:00 local converted to UTC"
            ),
            "stress_state_reuse": (
                "base, delayed-entry, and worse-fill scenarios use separate ledgers; "
                "portfolio caps reserve once per accepted base order intent"
            ),
            "score_cache_role": "audit_cache_only_not_source_of_truth",
        },
        "pre_registered_pass_fail_criteria": GEN2_STRATEGY_VALIDATION_PASS_CRITERIA,
        "strategies_tested": [
            "conservative_fv_edge_baseline",
            "old_high_confidence_no_tail",
        ],
        "staging_protocol": {
            "stage_a": "baseline plus accounting realism checks",
            "stage_b": "cap/sizing grid at fixed signal gates",
            "stage_c": "small signal perturbation grid at representative sizing",
            "stage_d": "ranking and pass/fail stress evaluation of variants",
        },
        "weather_model_runner": {
            "metadata": runner_metadata,
            "fixture_runner_used": runner_metadata.get("fixture_only") == "true",
            "real_weather_model_runner_plugged_in": (
                runner_metadata.get("fixture_only") == "false"
                and bool(runner_metadata)
            ),
            "research_only_not_promoted": (
                runner_metadata.get("promotion_status") == "research_only_not_promoted"
            ),
            "model_outputs_generated": model_outputs,
            "missing_model_outputs": missing_model_outputs,
            "model_input_firewall_violations": firewall_violations,
        },
        "fee_model": fee_config.as_report_dict(),
        "execution_realism": {
            "buy_at_ask_sell_at_bid": True,
            "visible_size_no_partial_fill_policy": FillPolicy(allow_partial=False).policy_name,
            "fee_aware_by_default": fee_config.fee_model == FEE_MODEL_POLYMARKET_WEATHER_TAKER,
            "delayed_entry_stress": {
                "enabled": delayed_quote_lookup is not None,
                "delay_seconds": delay_seconds,
                "lookup": "first compact quote at or after decision_time + delay",
            },
            "worse_fill_stress": {
                "enabled": True,
                "tick": worse_fill_tick,
                "buy_orders": "ask increased by one tick",
                "sell_orders": "bid decreased by one tick",
            },
            "order_latency_model": "stress_test_only_not_full_latency_execution_model",
        },
        "frame_skip_counts": dict(sorted(frame_skip_counts.items())),
        "leakage_validation": {"violation_counts": dict(sorted(leakage_violations.items()))},
        "variants": variant_summaries,
        "ranked_variants": ranked,
        "criteria_results": criteria_results,
        "overall_verdict": verdict,
    }
    return payload, states


def run_gen2_cap_divergence_diagnostic(
    dataset: Any,
    *,
    variant_key: str,
    starting_cash_usd: float = 100.0,
    limit: int = 0,
    model_runner: Optional[WeatherModelRunner] = None,
    score_cache: Optional[ReplayGeneratedScoreCache] = None,
    leakage_validators: Sequence[LeakageValidator] = (DefaultLeakageValidator(),),
    progress_every: int = 0,
) -> Mapping[str, Any]:
    specs = {spec.key: spec for spec in _gen2_variant_specs(starting_cash_usd)}
    if variant_key not in specs:
        raise ValueError(f"Unknown Gen-2 variant key: {variant_key}")
    spec = specs[variant_key]
    permanent_state = PortfolioState(spec.portfolio_caps)
    expiring_state = PortfolioState(spec.portfolio_caps)
    cache = score_cache or ReplayGeneratedScoreCache()
    point_policy = PointInTimePolicy(require_shared_score=True)
    runner_metadata = dict(model_runner.metadata()) if model_runner is not None else {}
    total_rows = 0
    model_outputs = 0
    missing_model_outputs = 0
    firewall_violations = 0
    leakage_violations: Counter[str] = Counter()
    frame_skip_counts: Counter[str] = Counter()
    strategy_skip_counts: Counter[str] = Counter()
    cap_decisions_compared = 0
    pre_cap_candidate_rows = 0
    permanent_cap_blocks: Counter[str] = Counter()
    expiring_cap_blocks: Counter[str] = Counter()
    permanent_passes = 0
    expiring_passes = 0
    first_divergence: Optional[Mapping[str, Any]] = None
    start_time = time.perf_counter()
    for source_frame in dataset.iter_frames(limit=limit):
        total_rows += 1
        if progress_every and total_rows % progress_every == 0:
            print(f"gen2-cap-divergence rows={total_rows}", flush=True)
        frame, generated_score, missing_model, firewall_violation = _score_frame_for_throughput(
            source_frame,
            model_runner=model_runner,
            score_cache=cache,
            point_policy=point_policy,
        )
        if generated_score or frame.score is not None:
            model_outputs += 1
        if missing_model:
            missing_model_outputs += 1
        if firewall_violation:
            firewall_violations += 1
        violations = tuple(
            violation
            for validator in leakage_validators
            for violation in validator.validate(frame)
        )
        for violation in violations:
            leakage_violations[violation.code] += 1
        if violations:
            frame = replace(
                frame,
                skip_reasons=_frame_skip_reasons_with_leakage(frame.skip_reasons, violations),
            )
        if frame.skip_reasons:
            for reason in frame.skip_reasons:
                frame_skip_counts[reason.value] += 1
            continue
        decision = spec.strategy.evaluate(frame.strategy_input())
        if not decision.orders:
            for reason in _metadata_skip_reasons(decision.metadata):
                strategy_skip_counts[reason.value] += 1
            continue
        for order in decision.orders:
            pre_cap_candidate_rows += 1
            released_count = expiring_state.release_expired(frame.decision.decision_time_utc)
            exposure = _exposure_intent_from_order(order, frame.decision)
            permanent_result = permanent_state.check(exposure)
            expiring_result = expiring_state.check(exposure)
            cap_decisions_compared += 1
            permanent_reasons = tuple(reason.value for reason in permanent_result.skip_reasons)
            expiring_reasons = tuple(reason.value for reason in expiring_result.skip_reasons)
            if (
                permanent_result.passed != expiring_result.passed
                or permanent_reasons != expiring_reasons
            ):
                first_divergence = {
                    "cap_decision_index": cap_decisions_compared,
                    "row_index": total_rows,
                    "released_expiring_exposures_before_check": released_count,
                    "decision": _cap_divergence_decision_report(frame.decision),
                    "order": _cap_divergence_order_report(order),
                    "exposure": _cap_divergence_exposure_report(exposure),
                    "permanent_policy": {
                        "description": "103638Z lineage: exposure reservations never expire during replay",
                        "result": _portfolio_cap_result_report(permanent_result),
                        "state_before": permanent_state.summary(),
                    },
                    "expiring_policy": {
                        "description": "current lineage: expired exposures released before cap check",
                        "result": _portfolio_cap_result_report(expiring_result),
                        "state_before_after_release": expiring_state.summary(),
                    },
                }
                break
            if permanent_result.passed:
                permanent_state.reserve(exposure)
                permanent_passes += 1
            else:
                for reason in permanent_result.skip_reasons:
                    permanent_cap_blocks[reason.value] += 1
            if expiring_result.passed:
                expiring_state.reserve(exposure)
                expiring_passes += 1
            else:
                for reason in expiring_result.skip_reasons:
                    expiring_cap_blocks[reason.value] += 1
        if first_divergence is not None:
            break
    runtime_seconds = time.perf_counter() - start_time
    return {
        "schema_version": BACKTEST_FRAMEWORK_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "gen2_cap_divergence_diagnostic",
        "dataset_id": "six_month_v1",
        "official_trading_replay_sql_predicate": OFFICIAL_TRADING_REPLAY_SQL_PREDICATE,
        "row_limit": limit,
        "variant_key": variant_key,
        "variant_parameters": dict(spec.parameters),
        "variant_config": dict(spec.config_report),
        "portfolio_caps": _portfolio_caps_summary(spec.portfolio_caps),
        "total_rows_scanned_until_stop": total_rows,
        "runtime_seconds": runtime_seconds,
        "rows_per_sec": (total_rows / runtime_seconds) if runtime_seconds else None,
        "model_outputs_generated_or_present": model_outputs,
        "missing_model_outputs": missing_model_outputs,
        "model_input_firewall_violations": firewall_violations,
        "leakage_validation": {"violation_counts": dict(sorted(leakage_violations.items()))},
        "frame_skip_counts": dict(sorted(frame_skip_counts.items())),
        "strategy_skip_counts_top": dict(strategy_skip_counts.most_common(20)),
        "pre_cap_candidate_rows_compared_until_stop": pre_cap_candidate_rows,
        "cap_decisions_compared": cap_decisions_compared,
        "first_divergence_found": first_divergence is not None,
        "first_divergence": first_divergence,
        "permanent_policy_counts_before_divergence": {
            "passes": permanent_passes,
            "cap_blocks": dict(sorted(permanent_cap_blocks.items())),
        },
        "expiring_policy_counts_before_divergence": {
            "passes": expiring_passes,
            "cap_blocks": dict(sorted(expiring_cap_blocks.items())),
        },
        "weather_model_runner": {
            "metadata": runner_metadata,
            "fixture_runner_used": runner_metadata.get("fixture_only") == "true",
            "real_weather_model_runner_plugged_in": (
                runner_metadata.get("fixture_only") == "false"
                and bool(runner_metadata)
            ),
            "research_only_not_promoted": (
                runner_metadata.get("promotion_status") == "research_only_not_promoted"
            ),
        },
        "score_cache_role": "audit_cache_only_not_source_of_truth",
        "no_order_execution": True,
        "no_pnl_accounting": True,
        "no_live_trading": True,
    }


def _portfolio_cap_result_report(result: PortfolioCapResult) -> Mapping[str, Any]:
    return {
        "passed": result.passed,
        "skip_reasons": [reason.value for reason in result.skip_reasons],
        "exposures_after": dict(result.exposures_after),
    }


def _cap_divergence_decision_report(decision: DecisionSnapshot) -> Mapping[str, Any]:
    return {
        "decision_id": decision.decision_id,
        "universe_row_id": decision.universe_row_id,
        "market_id": decision.market_id,
        "decision_time_utc": decision.decision_time_utc.isoformat(),
        "decision_hour_utc": decision.decision_hour_utc,
        "market_date_local": decision.market_date_local,
        "normalized_city": decision.normalized_city,
        "country_region": decision.country_region,
        "timezone": decision.timezone,
        "bucket_type": decision.bucket_type,
        "bucket_label": decision.bucket_label,
        "side": decision.side,
        "token_id": decision.token_id,
        "settlement_source": decision.settlement_source,
        "lead_days": decision.lead_days,
    }


def _cap_divergence_order_report(order: OrderIntent) -> Mapping[str, Any]:
    return {
        "strategy_name": order.strategy_name,
        "decision_id": order.decision_id,
        "token_id": order.token_id,
        "side": order.side,
        "action": order.action.value,
        "limit_price": order.limit_price,
        "requested_notional_usd": order.requested_notional_usd,
        "requested_shares": order.requested_shares,
    }


def _cap_divergence_exposure_report(exposure: ExposureIntent) -> Mapping[str, Any]:
    return {
        "token_id": exposure.token_id,
        "side": exposure.side,
        "city": exposure.city,
        "market_date_local": exposure.market_date_local,
        "region": exposure.region,
        "settlement_source_family": exposure.settlement_source_family,
        "decision_hour_utc": exposure.decision_hour_utc,
        "notional_usd": exposure.notional_usd,
        "primary_exposure": exposure.primary_exposure,
        "release_time_utc": (
            exposure.release_time_utc.isoformat()
            if exposure.release_time_utc is not None
            else None
        ),
    }


def _rank_gen2_variants(
    variant_summaries: Mapping[str, Mapping[str, Any]]
) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    for key, summary in variant_summaries.items():
        metrics = summary.get("ranking_metrics", {})
        pnl_value = _float_or_none(metrics.get("net_fee_adjusted_pnl_usd"))
        pnl = 0.0 if pnl_value is None else pnl_value
        rows.append(
            {
                "variant_key": key,
                "label": summary.get("label"),
                "stage": summary.get("stage"),
                "strategy_family": summary.get("strategy_family"),
                "net_fee_adjusted_pnl_usd": metrics.get("net_fee_adjusted_pnl_usd"),
                "roi_after_fees": metrics.get("roi_after_fees"),
                "trade_count": metrics.get("trade_count"),
                "win_count": metrics.get("win_count"),
                "loss_count": metrics.get("loss_count"),
                "total_fees_usd": metrics.get("total_fees_usd"),
                "delayed_entry_net_pnl_usd": metrics.get("delayed_entry_net_pnl_usd"),
                "worse_fill_net_pnl_usd": metrics.get("worse_fill_net_pnl_usd"),
                "parameters": summary.get("parameters"),
                "_sort_pnl": pnl,
            }
        )
    ranked = sorted(
        rows,
        key=lambda row: (
            -(
                _float_or_none(row.get("net_fee_adjusted_pnl_usd"))
                if _float_or_none(row.get("net_fee_adjusted_pnl_usd")) is not None
                else float("-inf")
            ),
            -int(row.get("trade_count") or 0),
            str(row.get("variant_key") or ""),
        ),
    )
    for idx, row in enumerate(ranked, start=1):
        row["rank_by_net_fee_adjusted_pnl"] = idx
        row.pop("_sort_pnl", None)
    return ranked


def _evaluate_gen2_pass_fail_criteria(
    variant_summaries: Mapping[str, Mapping[str, Any]]
) -> Mapping[str, Any]:
    results: dict[str, Mapping[str, Any]] = {}
    neighbor_map = _gen2_neighbor_positive_counts(variant_summaries)
    for key, summary in variant_summaries.items():
        base = summary.get("base_accounting", {})
        pnl = base.get("pnl", {})
        monthly = base.get("pnl_by_month", {})
        month_eval = _walk_forward_month_eval(monthly)
        concentration = _gen2_concentration_eval(summary)
        stress = _gen2_stress_eval(summary)
        ledger = {"full_trade_ledger_persisted": True}
        net_pnl = _float_or_none(pnl.get("net_fee_adjusted_realized_pnl_usd")) or 0.0
        trade_count = int(pnl.get("win_count") or 0) + int(pnl.get("loss_count") or 0) + int(pnl.get("flat_count") or 0)
        criteria = GEN2_STRATEGY_VALIDATION_PASS_CRITERIA
        neighbor_positive = neighbor_map.get(key, 0)
        checks = {
            "net_positive_after_fees": net_pnl > 0.0,
            "positive_in_majority_of_windows": (
                month_eval["positive_windows"]
                >= criteria["walk_forward_windows"]["required_positive_net_windows"]
            ),
            "concentration_bounded": concentration["passed"],
            "worst_window_loss_bounded": (
                month_eval["worst_window_net_pnl_usd"]
                >= criteria["risk_limits"]["worst_window_net_pnl_min_usd"]
            ),
            "survives_worse_fill_stress": stress["worse_fill_passed"],
            "survives_delayed_entry_stress": stress["delayed_entry_passed"],
            "neighboring_variants_not_all_negative": (
                neighbor_positive >= criteria["plateau_limits"]["minimum_positive_neighbor_variants"]
            ),
            "minimum_trade_count_for_live_candidate": (
                trade_count >= criteria["risk_limits"]["minimum_filled_trades_for_live_candidate"]
            ),
            "full_trade_ledger_persisted": ledger["full_trade_ledger_persisted"],
        }
        live_candidate_pass = all(checks.values())
        if live_candidate_pass:
            verdict = "live-candidate"
        elif net_pnl > 0.0 and checks["full_trade_ledger_persisted"]:
            verdict = "research-only"
        else:
            verdict = "rejected"
        results[key] = {
            "variant_key": key,
            "verdict": verdict,
            "checks": checks,
            "walk_forward": month_eval,
            "concentration": concentration,
            "stress": stress,
            "neighbor_positive_variant_count": neighbor_positive,
            "trade_count": trade_count,
            "net_fee_adjusted_pnl_usd": net_pnl,
        }
    return results


def _walk_forward_month_eval(monthly: Mapping[str, Any]) -> Mapping[str, Any]:
    months = [f"2026-{month:02d}" for month in range(1, 7)]
    values: dict[str, float] = {}
    positive = 0
    worst = 0.0
    for month in months:
        summary = monthly.get(month, {}) if isinstance(monthly, Mapping) else {}
        value = _float_or_none(summary.get("net_fee_adjusted_realized_pnl_usd")) or 0.0
        values[month] = value
        if value > 0:
            positive += 1
        worst = min(worst, value)
    return {
        "months": values,
        "positive_windows": positive,
        "total_windows": len(months),
        "worst_window_net_pnl_usd": worst,
    }


def _gen2_concentration_eval(summary: Mapping[str, Any]) -> Mapping[str, Any]:
    base = summary.get("base_accounting", {})
    criteria = GEN2_STRATEGY_VALIDATION_PASS_CRITERIA["concentration_limits"]
    month_share = _max_abs_pnl_share(base.get("pnl_by_month", {}))
    city_share = _max_abs_pnl_share(base.get("pnl_by_city", {}))
    target_date_share = _max_abs_pnl_share(summary.get("pnl_by_target_date", {}))
    region_share = _max_abs_pnl_share(
        summary.get("city_region_timezone_concentration", {}).get("country_region", {})
    )
    checks = {
        "single_month_abs_pnl_share": month_share,
        "single_city_abs_pnl_share": city_share,
        "single_target_date_abs_pnl_share": target_date_share,
        "single_region_abs_pnl_share": region_share,
    }
    passed = (
        month_share <= criteria["single_month_abs_pnl_share_max"]
        and city_share <= criteria["single_city_abs_pnl_share_max"]
        and target_date_share <= criteria["single_target_date_abs_pnl_share_max"]
        and region_share <= criteria["single_region_abs_pnl_share_max"]
    )
    return {"passed": passed, **checks}


def _max_abs_pnl_share(slice_report: Mapping[str, Any]) -> float:
    pnls: list[float] = []
    for value in slice_report.values():
        if isinstance(value, Mapping):
            pnl = _float_or_none(value.get("net_fee_adjusted_realized_pnl_usd"))
            if pnl is None:
                pnl = _float_or_none(value.get("realized_pnl_usd"))
            if pnl is not None:
                pnls.append(pnl)
    total_abs = sum(abs(value) for value in pnls)
    if total_abs <= 1e-12:
        return 0.0
    return max(abs(value) for value in pnls) / total_abs


def _gen2_stress_eval(summary: Mapping[str, Any]) -> Mapping[str, Any]:
    base_pnl = _float_or_none(
        summary.get("base_accounting", {}).get("pnl", {}).get("net_fee_adjusted_realized_pnl_usd")
    ) or 0.0
    base_fills = int(summary.get("fills") or 0)
    stress = summary.get("stress_accounting", {})
    delayed = stress.get("delayed_entry_2s", {})
    worse = stress.get("worse_fill_1tick", {})
    delayed_pnl = _float_or_none(delayed.get("pnl", {}).get("net_fee_adjusted_realized_pnl_usd")) or 0.0
    worse_pnl = _float_or_none(worse.get("pnl", {}).get("net_fee_adjusted_realized_pnl_usd")) or 0.0
    delayed_fills = int(delayed.get("fill_counts", {}).get("actual_fills") or 0)
    retention = (delayed_fills / base_fills) if base_fills else 0.0
    min_retention = GEN2_STRATEGY_VALIDATION_PASS_CRITERIA["stress_limits"]["delayed_entry_fill_retention_min"]
    return {
        "base_net_pnl_usd": base_pnl,
        "delayed_entry_net_pnl_usd": delayed_pnl,
        "worse_fill_net_pnl_usd": worse_pnl,
        "delayed_entry_fill_retention": retention,
        "delayed_entry_passed": delayed_pnl > 0.0 and retention >= min_retention,
        "worse_fill_passed": worse_pnl > 0.0,
    }


def _gen2_neighbor_positive_counts(
    variant_summaries: Mapping[str, Mapping[str, Any]]
) -> Mapping[str, int]:
    counts: dict[str, int] = {}
    for key in sorted(variant_summaries):
        summary = variant_summaries[key]
        params = summary.get("parameters", {})
        positive_neighbors = 0
        for other_key in sorted(variant_summaries):
            other = variant_summaries[other_key]
            if other_key == key:
                continue
            if other.get("strategy_family") != summary.get("strategy_family"):
                continue
            if other.get("stage") not in {summary.get("stage"), "B_cap_sizing_grid", "C_signal_perturbation_grid"}:
                continue
            other_params = other.get("parameters", {})
            shared = sorted(set(params) | set(other_params))
            diffs = sum(1 for item in shared if params.get(item) != other_params.get(item))
            if diffs > 2:
                continue
            pnl = _float_or_none(
                other.get("base_accounting", {}).get("pnl", {}).get("net_fee_adjusted_realized_pnl_usd")
            )
            if pnl is not None and pnl > 0:
                positive_neighbors += 1
        counts[key] = positive_neighbors
    return counts


def _gen2_reproducibility_field(
    *,
    independent_runs_reproduced: int,
    required_runs: int,
    ranked_grid_byte_identical: Optional[bool],
    full_trade_ledger_byte_identical: Optional[bool],
    re_emitted: bool,
) -> Mapping[str, Any]:
    passed = (
        independent_runs_reproduced >= required_runs
        and ranked_grid_byte_identical is True
        and full_trade_ledger_byte_identical is True
    )
    if passed and re_emitted:
        status = "passed_live_candidate_verdict_re_emitted"
    elif passed:
        status = "passed_pending_explicit_re_emission"
    else:
        status = "pending_three_independent_byte_identical_runs"
    return {
        "required_independent_runs": required_runs,
        "independent_runs_reproduced_every_cited_result": independent_runs_reproduced,
        "ranked_grid_csv_byte_identical": ranked_grid_byte_identical,
        "full_trade_ledger_csv_byte_identical": full_trade_ledger_byte_identical,
        "live_candidate_verdict_re_emitted": bool(re_emitted and passed),
        "gate_status": status,
    }


def _apply_gen2_reproducibility_gate(
    criteria_results: Mapping[str, Mapping[str, Any]],
    *,
    independent_runs_reproduced: int,
    required_runs: int,
    ranked_grid_byte_identical: Optional[bool],
    full_trade_ledger_byte_identical: Optional[bool],
) -> Mapping[str, Mapping[str, Any]]:
    passed = (
        independent_runs_reproduced >= required_runs
        and ranked_grid_byte_identical is True
        and full_trade_ledger_byte_identical is True
    )
    gated: dict[str, Mapping[str, Any]] = {}
    for key in sorted(criteria_results):
        result = dict(criteria_results[key])
        pre_reproducibility_verdict = str(result.get("verdict") or "")
        result["pre_reproducibility_verdict"] = pre_reproducibility_verdict
        result["reproducibility"] = _gen2_reproducibility_field(
            independent_runs_reproduced=independent_runs_reproduced,
            required_runs=required_runs,
            ranked_grid_byte_identical=ranked_grid_byte_identical,
            full_trade_ledger_byte_identical=full_trade_ledger_byte_identical,
            re_emitted=passed,
        )
        if pre_reproducibility_verdict == "live-candidate" and not passed:
            result["verdict"] = "replay-gate-pass_pending_reproducibility"
            checks = dict(result.get("checks", {}))
            checks["reproducibility_gate_passed"] = False
            result["checks"] = checks
        elif pre_reproducibility_verdict == "live-candidate":
            checks = dict(result.get("checks", {}))
            checks["reproducibility_gate_passed"] = True
            result["checks"] = checks
        gated[key] = result
    return gated


def _gen2_overall_verdict(criteria_results: Mapping[str, Mapping[str, Any]]) -> Mapping[str, Any]:
    live_candidates = sorted(
        key for key, result in criteria_results.items() if result.get("verdict") == "live-candidate"
    )
    pending_reproducibility = sorted(
        key
        for key, result in criteria_results.items()
        if result.get("verdict") == "replay-gate-pass_pending_reproducibility"
    )
    research_only = sorted(
        key for key, result in criteria_results.items() if result.get("verdict") == "research-only"
    )
    if live_candidates:
        status = "live-candidate"
    elif pending_reproducibility:
        status = "suspended_pending_reproducibility"
    elif research_only:
        status = "research-only"
    else:
        status = "rejected"
    return {
        "status": status,
        "live_candidate_variants": live_candidates,
        "replay_gate_pass_variants": pending_reproducibility,
        "research_only_variants": research_only,
        "rejected_variants": sorted(
            key for key, result in criteria_results.items() if result.get("verdict") == "rejected"
        ),
        "claim_boundary": (
            "No live-trading recommendation or general profitability claim. "
            "A live-candidate verdict may be emitted only after the pre-registered "
            "replay gates pass and the Gen-2 reproducibility gate independently verifies "
            "byte-identical ranked-grid and full-ledger artifacts."
        ),
    }


def _float_or_none(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    return float(value)


class ReplayReportBuilder:
    def __init__(self, *, strategy_name: str) -> None:
        self.strategy_name = strategy_name
        self.total_decision_rows = 0
        self.strategy_input_rows = 0
        self.skip_reasons: Counter[str] = Counter()
        self.leakage_violations: Counter[str] = Counter()
        self.label_actions: Counter[str] = Counter()
        self.coverage: dict[str, Counter[str]] = defaultdict(Counter)

    def add_frame(
        self,
        frame: ReplayFrame,
        *,
        skip_reasons: Sequence[SkipReason],
        leakage_violations: Sequence[LeakageViolation],
    ) -> None:
        decision = frame.decision
        self.total_decision_rows += 1
        if not skip_reasons:
            self.strategy_input_rows += 1
        for reason in skip_reasons:
            self.skip_reasons[reason.value] += 1
        for violation in leakage_violations:
            self.leakage_violations[violation.code] += 1
        if frame.canonical_label is not None:
            self.label_actions[frame.canonical_label.recommended_action or "missing_action"] += 1
        self.coverage["side"][decision.side or "missing"] += 1
        self.coverage["city"][decision.normalized_city or "missing"] += 1
        self.coverage["bucket_type"][decision.bucket_type or "missing"] += 1
        self.coverage["decision_hour_utc"][str(decision.decision_hour_utc)] += 1

    def add_skip_reason(self, reason: SkipReason) -> None:
        self.skip_reasons[reason.value] += 1

    def to_report(self) -> Mapping[str, Any]:
        return {
            "schema_version": BACKTEST_FRAMEWORK_SCHEMA_VERSION,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "strategy_name": self.strategy_name,
            "mode": "framework_only_no_pnl",
            "label_policy": {
                "canonical_label_source": "polymarket_telonex_settlement_labels",
                "canonical_label_artifact": CANONICAL_LABEL_ARTIFACT,
                "open_meteo_realized_highs": "non_canonical_weather_regression_proxy",
                "excluded_recommended_action": "exclude_or_manual_review",
            },
            "decision_counts": {
                "total_decision_rows_seen": self.total_decision_rows,
                "rows_available_for_strategy_input": self.strategy_input_rows,
            },
            "skip_reason_counts": dict(sorted(self.skip_reasons.items())),
            "canonical_label_action_counts": dict(sorted(self.label_actions.items())),
            "leakage_validation": {
                "violation_counts": dict(sorted(self.leakage_violations.items())),
            },
            "coverage": {
                key: dict(counter.most_common(200))
                for key, counter in sorted(self.coverage.items())
            },
            "report_sections_reserved_for_future_strategy_backtests": [
                "weather_model_accuracy",
                "market_price_scoring",
                "execution_realism",
                "walk_forward_validation",
                "parameter_sensitivity",
                "plateau_analysis",
                "selection_bias_controls",
                "accounting_summary",
                "risk_diagnostics",
                "improvement_report",
            ],
        }


BACKTEST_REPORT_SCHEMA: Mapping[str, Any] = {
    "schema_version": BACKTEST_FRAMEWORK_SCHEMA_VERSION,
    "required_sections": [
        "run_metadata",
        "data_window",
        "artifact_versions",
        "strategy_profile",
        "decision_cadence",
        "quote_source",
        "weather_source",
        "label_policy",
        "skip_reason_counts",
        "coverage",
        "leakage_validation",
        "execution_assumptions",
        "accounting_summary",
        "walk_forward_validation",
        "parameter_sensitivity",
        "plateau_analysis",
        "selection_bias_controls",
        "final_holdout_policy",
        "window_distribution",
        "weather_model_accuracy",
        "market_price_scoring",
        "execution_realism",
        "risk_diagnostics",
        "improvement_report",
    ],
    "label_policy": {
        "bucket_labels": "Polymarket/Telonex canonical settlement labels",
        "weather_regression_target": "Open-Meteo realized highs are non-canonical proxies",
        "exclude_rule": "recommended_action=exclude_or_manual_review",
    },
}


ANTI_OVERFIT_VALIDATION_SCHEMA: Mapping[str, Any] = {
    "schema_version": BACKTEST_FRAMEWORK_SCHEMA_VERSION,
    "contract_name": "anti_overfit_walk_forward_parameter_plateau",
    "applies_to": [
        "weather_model_validation",
        "market_relative_scoring",
        "trading_strategy_backtests",
    ],
    "core_rule": (
        "A single best parameter combination is research-only until it survives "
        "purged walk-forward validation, neighboring parameter settings, and "
        "realistic execution stress."
    ),
    "required_controls": [
        "rolling_origin_or_walk_forward_splits",
        "target_date_purging",
        "optional_temporal_embargo_variants",
        "label_availability_cutoff_before_validation_window",
        "final_holdout_not_used_for_parameter_selection",
        "full_search_space_or_grid_reported",
        "nearby_parameter_sensitivity_reported",
        "median_and_worst_window_performance_reported",
        "losing_window_count_reported",
        "city_region_lead_bucket_hour_strata_reported",
        "plausible_worse_fill_spread_liquidity_stress_reported",
    ],
    "weather_model_required_diagnostics": [
        "target_month_out_of_sample_metrics",
        "city_holdout_metrics",
        "region_holdout_metrics",
        "bucket_type_metrics",
        "threshold_distance_metrics",
        "lead_time_metrics",
        "decision_hour_metrics",
        "source_model_family_metrics",
        "bootstrap_confidence_intervals",
        "label_shuffle_or_permutation_sanity_check",
    ],
    "trading_required_diagnostics": [
        "continuous_portfolio_state_replay",
        "buy_at_ask_sell_at_bid_execution",
        "quote_staleness_cap_enforced",
        "liquidity_and_partial_fill_policy",
        "costs_and_spread_paid",
        "parameter_plateau_table",
        "window_return_distribution",
        "window_drawdown_distribution",
        "pnl_concentration_by_city_region_weather_day",
        "worse_fill_delayed_entry_liquidity_stress",
    ],
    "promotion_blocker": (
        "Do not promote a model or strategy if the edge depends on an isolated "
        "parameter peak, post-decision information, or aggregate-only performance."
    ),
}


SCORE_CONTRACT_SCHEMA: Mapping[str, Any] = {
    "schema_version": BACKTEST_FRAMEWORK_SCHEMA_VERSION,
    "contract_name": "shared_side_aware_score_contract",
    "label_visibility": "strategies_must_not_receive_realized_high_resolved_outcome_or_canonical_payout_label",
    "required_fields": [
        "p_yes",
        "p_no",
        "p_side",
        "side",
        "model_agreement",
        "model_count",
        "probability_stdev",
        "per_source_probabilities",
        "remaining_exceed_probability",
        "observed_high_aware_confidence",
        "quote.bid",
        "quote.ask",
        "quote.mid",
        "quote.spread",
        "quote.timestamp_utc",
        "quote.bid_size",
        "quote.ask_size",
        "quote.source_cache_path",
    ],
    "observed_high_fields": [
        "observed_high_f",
        "observed_high_timestamp_utc",
        "observation_source",
        "observation_final",
        "source_match_confidence",
        "source_family",
        "source_station",
        "deterministic_state",
    ],
    "heat_window_fields": [
        "local_hour",
        "forecast_peak_hour_local",
        "remaining_hourly_forecast_max_f",
        "heat_window_exhaustion_score",
        "solar_radiation_state",
        "cloud_cover_state",
        "wind_state",
        "season",
    ],
}


UNIVERSAL_GATE_SCHEMA: Mapping[str, Any] = {
    "schema_version": BACKTEST_FRAMEWORK_SCHEMA_VERSION,
    "gate_groups": {
        "data_gates": [
            "parser_confidence",
            "market_fields_present",
            "forecast_run_time_lte_decision_time",
            "quote_timestamp_lte_decision_time",
            "quote_age_lte_90_minutes",
            "bid_ask_present",
            "manual_review_labels_excluded",
            "settlement_unit_precision_known",
        ],
        "market_gates": [
            "buy_at_ask_sell_at_bid",
            "spread_within_limit",
            "visible_size_covers_order_or_partial_fill_policy",
            "ask_minimum_maximum",
        ],
        "model_gates": [
            "model_count_minimum",
            "model_agreement_minimum",
            "observed_high_aware_confidence_minimum",
            "bounded_probability_stdev_cap",
        ],
        "portfolio_gates": [
            "utc_window_bankroll_cap",
            "region_cap",
            "city_date_cap",
            "per_position_cap",
            "settlement_source_cap",
            "one_primary_exposure_per_city_date_per_run",
        ],
    },
    "skip_reasons": [item.value for item in SkipReason],
}


VALIDATION_ONLY_REPORT_SCHEMA: Mapping[str, Any] = {
    "schema_version": BACKTEST_FRAMEWORK_SCHEMA_VERSION,
    "required_sections": [
        "no_pnl_statement",
        "score_contract_coverage",
        "artifact_coverage",
        "universal_gate_summary",
        "canonical_label_action_counts",
        "frame_skip_reason_counts",
        "leakage_validation",
        "portfolio_cap_readiness",
        "anti_overfit_readiness",
        "coverage",
        "schema_gaps",
    ],
    "forbidden_sections_until_strategy_selection": [
        "pnl_usd",
        "return_pct",
        "sharpe",
        "strategy_ranking",
        "parameter_optimization",
        "best_parameter_only_claim",
    ],
}


PORTFOLIO_CAP_SCHEMA: Mapping[str, Any] = {
    "schema_version": BACKTEST_FRAMEWORK_SCHEMA_VERSION,
    "cap_dimensions": [
        "utc_window",
        "region",
        "city_date",
        "token_side",
        "bucket_type",
        "settlement_source_family",
    ],
    "default_hypotheses": {
        "window_bankroll_fraction": 0.25,
        "per_position_window_fraction": 0.25,
        "per_city_date_total_fraction": "0.10_to_0.15_strategy_spec_hypothesis",
        "max_primary_exposures_per_city_date_per_run": 1,
    },
}


def framework_contract_payload() -> Mapping[str, Any]:
    return {
        "schema_version": BACKTEST_FRAMEWORK_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "implementation_scaffold_only",
        "no_pnl_statement": (
            "Contracts support future validation-only dry runs. They do not run "
            "strategy PnL backtests, optimize parameters, or claim profitability."
        ),
        "source_strategy_spec": "work/reports/strategy_design/20260701T101642Z-daily-high-temperature-strategy-spec.md",
        "canonical_label_artifact": CANONICAL_LABEL_ARTIFACT,
        "score_contract_schema": SCORE_CONTRACT_SCHEMA,
        "universal_gate_schema": UNIVERSAL_GATE_SCHEMA,
        "portfolio_cap_schema": PORTFOLIO_CAP_SCHEMA,
        "anti_overfit_validation_schema": ANTI_OVERFIT_VALIDATION_SCHEMA,
        "validation_only_report_schema": VALIDATION_ONLY_REPORT_SCHEMA,
    }


def write_framework_contract_report(output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = output_dir / f"{timestamp}-backtesting-framework-contracts.json"
    md_path = output_dir / f"{timestamp}-backtesting-framework-contracts.md"
    payload = framework_contract_payload()
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(_framework_contract_markdown(payload), encoding="utf-8")
    return {"json_path": str(json_path), "markdown_path": str(md_path)}


def write_validation_only_report(
    *,
    work_root: Path,
    output_dir: Path,
    limit: int = 0,
    official_trading_replay_only: bool = True,
) -> dict[str, str]:
    work_root = Path(work_root).resolve()
    wide_db = work_root / WIDE_PRICE_DB_RELATIVE_PATH
    if wide_db.exists():
        raise RuntimeError(f"Wide price DB must remain unused for six_month_v1: {wide_db}")
    decision_db = work_root / "work" / "data" / "backtest" / "daily_weather_decision_artifact.sqlite"
    if not decision_db.exists():
        raise FileNotFoundError(decision_db)
    output_dir.mkdir(parents=True, exist_ok=True)
    loader = SQLiteDecisionFrameLoader(
        decision_db,
        relation=COMPACT_DECISION_VIEW,
        official_trading_replay_only=official_trading_replay_only,
        require_compact_quote_store=True,
        point_in_time_policy=PointInTimePolicy(require_shared_score=False),
    )
    validation = ValidationOnlyEngine(loader).validate(limit=limit, collect_gate_results=False)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = output_dir / f"{timestamp}-backtesting-validation-only-dry-replay.json"
    md_path = output_dir / f"{timestamp}-backtesting-validation-only-dry-replay.md"
    payload = {
        **validation.report,
        "dataset_id": "six_month_v1",
        "freeze_manifest": str(work_root / SIX_MONTH_V1_FREEZE_MANIFEST),
        "decision_artifact_db": str(decision_db),
        "compact_only": True,
        "wide_price_db_exists": False,
        "official_trading_replay_sql_predicate": (
            OFFICIAL_TRADING_REPLAY_SQL_PREDICATE if official_trading_replay_only else None
        ),
        "official_trading_replay_only": official_trading_replay_only,
        "row_limit": limit,
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(_validation_only_markdown(payload), encoding="utf-8")
    return {"json_path": str(json_path), "markdown_path": str(md_path)}


def write_conservative_fv_edge_candidate_replay_report(
    *,
    work_root: Path,
    output_dir: Path,
    limit: int = 250,
    starting_cash_usd: float = 100.0,
    weather_runner: str = "fixture",
    source_repo: Optional[Path] = None,
) -> dict[str, str]:
    work_root = Path(work_root).resolve()
    wide_db = work_root / WIDE_PRICE_DB_RELATIVE_PATH
    if wide_db.exists():
        raise RuntimeError(f"Wide price DB must remain unused for six_month_v1: {wide_db}")
    decision_db = work_root / "work" / "data" / "backtest" / "daily_weather_decision_artifact.sqlite"
    if not decision_db.exists():
        raise FileNotFoundError(decision_db)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = ConservativeFVEdgeConfig()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    runner_slug = _weather_runner_slug(weather_runner)
    score_cache_path = output_dir / f"{timestamp}-{runner_slug}-replay-generated-shared-scores.jsonl"
    score_cache = ReplayGeneratedScoreCache()
    model_runner = _build_candidate_weather_model_runner(
        weather_runner,
        work_root=work_root,
        source_repo=source_repo,
    )
    loader = SQLiteDecisionFrameLoader(
        decision_db,
        relation=COMPACT_DECISION_VIEW,
        official_trading_replay_only=True,
        require_compact_quote_store=True,
        point_in_time_policy=PointInTimePolicy(require_shared_score=True),
    )
    replay = run_conservative_fv_edge_candidate_replay(
        loader,
        config=config,
        starting_cash_usd=starting_cash_usd,
        limit=limit,
        model_runner=model_runner,
        score_cache=score_cache,
        execute_orders=False,
    )
    json_path = output_dir / f"{timestamp}-conservative-fv-edge-candidate-replay.json"
    md_path = output_dir / f"{timestamp}-conservative-fv-edge-candidate-replay.md"
    score_cache.write_jsonl(score_cache_path)
    payload = {
        **replay.report,
        "replay_generated_score_cache": score_cache.report_summary(
            artifact_path=score_cache_path
        ),
        "dataset_id": "six_month_v1",
        "freeze_manifest": str(work_root / SIX_MONTH_V1_FREEZE_MANIFEST),
        "decision_artifact_db": str(decision_db),
        "compact_only": True,
        "wide_price_db_exists": False,
        "source_strategy_spec": (
            "work/reports/strategy_design/"
            "20260701T101642Z-daily-high-temperature-strategy-spec.md"
        ),
        "candidate_weather_runner_selection": {
            "requested_runner": weather_runner,
            "runner_slug": runner_slug,
            "execution_enabled": False,
            "research_only_runner_allowed": weather_runner == "observed-high-guarded",
            "fixture_runner_allowed": weather_runner == "fixture",
        },
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(_candidate_replay_markdown(payload), encoding="utf-8")
    return {"json_path": str(json_path), "markdown_path": str(md_path)}


def write_conservative_fv_edge_accounting_replay_report(
    *,
    work_root: Path,
    output_dir: Path,
    limit: int = 2500,
    starting_cash_usd: float = 100.0,
    weather_runner: str = "observed-high-guarded",
    fee_model: str = DEFAULT_STRATEGY_REPLAY_FEE_MODEL,
    fee_rate: Optional[float] = DEFAULT_STRATEGY_REPLAY_FEE_RATE,
    source_repo: Optional[Path] = None,
) -> dict[str, str]:
    work_root = Path(work_root).resolve()
    wide_db = work_root / WIDE_PRICE_DB_RELATIVE_PATH
    if wide_db.exists():
        raise RuntimeError(f"Wide price DB must remain unused for six_month_v1: {wide_db}")
    decision_db = work_root / "work" / "data" / "backtest" / "daily_weather_decision_artifact.sqlite"
    if not decision_db.exists():
        raise FileNotFoundError(decision_db)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = ConservativeFVEdgeConfig()
    fee_config = FeeModelConfig(fee_model=fee_model, fee_rate=fee_rate)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    runner_slug = _weather_runner_slug(weather_runner)
    score_cache_path = output_dir / (
        f"{timestamp}-{runner_slug}-accounting-replay-generated-shared-scores.jsonl.gz"
    )
    score_cache = ReplayGeneratedScoreCache(
        stream_path=score_cache_path,
        compress=True,
        keep_records=False,
    )
    model_runner = _build_candidate_weather_model_runner(
        weather_runner,
        work_root=work_root,
        source_repo=source_repo,
    )
    loader = SQLiteDecisionFrameLoader(
        decision_db,
        relation=COMPACT_DECISION_VIEW,
        official_trading_replay_only=True,
        require_compact_quote_store=True,
        point_in_time_policy=PointInTimePolicy(require_shared_score=True),
    )
    replay = run_conservative_fv_edge_candidate_replay(
        loader,
        config=config,
        starting_cash_usd=starting_cash_usd,
        limit=limit,
        model_runner=model_runner,
        score_cache=score_cache,
        execute_orders=True,
        fill_policy=FillPolicy(allow_partial=False),
        fee_config=fee_config,
    )
    json_path = output_dir / f"{timestamp}-conservative-fv-edge-accounting-replay.json"
    md_path = output_dir / f"{timestamp}-conservative-fv-edge-accounting-replay.md"
    trade_ledger_path = output_dir / f"{timestamp}-conservative-fv-edge-accounting-trade-ledger.csv"
    score_cache.write_jsonl(score_cache_path)
    _write_trade_ledger_csv(trade_ledger_path, replay.trade_records)
    payload = {
        **replay.report,
        "replay_generated_score_cache": score_cache.report_summary(
            artifact_path=score_cache_path
        ),
        "full_trade_ledger": {
            "persisted": True,
            "format": "csv",
            "path": str(trade_ledger_path),
            "trade_count": len(replay.trade_records),
            "all_fills_included": len(replay.trade_records) == len(replay.fills),
        },
        "dataset_id": "six_month_v1",
        "freeze_manifest": str(work_root / SIX_MONTH_V1_FREEZE_MANIFEST),
        "decision_artifact_db": str(decision_db),
        "compact_only": True,
        "wide_price_db_exists": False,
        "source_strategy_spec": (
            "work/reports/strategy_design/"
            "20260701T101642Z-daily-high-temperature-strategy-spec.md"
        ),
        "candidate_weather_runner_selection": {
            "requested_runner": weather_runner,
            "runner_slug": runner_slug,
            "execution_enabled": True,
            "research_only_runner_allowed": weather_runner == "observed-high-guarded",
            "fixture_runner_allowed": weather_runner == "fixture",
        },
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(_accounting_replay_markdown(payload), encoding="utf-8")
    return {
        "json_path": str(json_path),
        "markdown_path": str(md_path),
        "score_cache_path": str(score_cache_path),
        "trade_ledger_path": str(trade_ledger_path),
    }


def write_conservative_fv_edge_throughput_diagnostic_report(
    *,
    work_root: Path,
    output_dir: Path,
    limit: int = 0,
    starting_cash_usd: float = 100.0,
    weather_runner: str = "observed-high-guarded",
    fee_model: str = FEE_MODEL_POLYMARKET_WEATHER_TAKER,
    fee_rate: Optional[float] = DEFAULT_POLYMARKET_WEATHER_TAKER_FEE_RATE,
    source_repo: Optional[Path] = None,
    progress_every: int = 0,
) -> dict[str, Any]:
    work_root = Path(work_root).resolve()
    wide_db = work_root / WIDE_PRICE_DB_RELATIVE_PATH
    if wide_db.exists():
        raise RuntimeError(f"Wide price DB must remain unused for six_month_v1: {wide_db}")
    decision_db = work_root / "work" / "data" / "backtest" / "daily_weather_decision_artifact.sqlite"
    if not decision_db.exists():
        raise FileNotFoundError(decision_db)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    runner_slug = _weather_runner_slug(weather_runner)
    json_path = output_dir / f"{timestamp}-conservative-fv-edge-throughput-diagnostic.json"
    md_path = output_dir / f"{timestamp}-conservative-fv-edge-throughput-diagnostic.md"
    score_cache_path = output_dir / (
        f"{timestamp}-{runner_slug}-throughput-diagnostic-generated-shared-scores.jsonl.gz"
    )
    score_cache = ReplayGeneratedScoreCache(
        stream_path=score_cache_path,
        compress=True,
        keep_records=False,
    )
    model_runner = _build_candidate_weather_model_runner(
        weather_runner,
        work_root=work_root,
        source_repo=source_repo,
    )
    loader = SQLiteDecisionFrameLoader(
        decision_db,
        relation=COMPACT_DECISION_VIEW,
        official_trading_replay_only=True,
        require_compact_quote_store=True,
        point_in_time_policy=PointInTimePolicy(require_shared_score=True),
    )
    fee_config = FeeModelConfig(fee_model=fee_model, fee_rate=fee_rate)
    started = time.perf_counter()
    payload = dict(
        run_conservative_fv_edge_throughput_diagnostic(
            loader,
            starting_cash_usd=starting_cash_usd,
            limit=limit,
            model_runner=model_runner,
            score_cache=score_cache,
            fee_config=fee_config,
            excluded_quote_rows=_count_settlement_excluded_quote_rows(decision_db),
            progress_every=progress_every,
        )
    )
    runtime_seconds = time.perf_counter() - started
    score_cache.write_jsonl(score_cache_path)
    payload.update(
        {
            "dataset_id": "six_month_v1",
            "freeze_manifest": str(work_root / SIX_MONTH_V1_FREEZE_MANIFEST),
            "decision_artifact_db": str(decision_db),
            "compact_only": True,
            "wide_price_db_exists": False,
            "runtime_seconds": runtime_seconds,
            "rows_per_sec": (
                payload.get("total_official_rows", 0) / runtime_seconds
                if runtime_seconds > 0
                else None
            ),
            "replay_generated_score_cache": score_cache.report_summary(
                artifact_path=score_cache_path
            ),
            "source_no_fee_full_baseline": (
                "work/reports/backtesting_engine/"
                "20260707T072924Z-conservative-fv-edge-accounting-replay.json"
            ),
            "boundary": {
                "parameter_optimization_performed": False,
                "strategy_settings_changed": False,
                "live_settings_promoted": False,
                "weather_model_promoted": False,
                "profitability_claim": "none",
            },
        }
    )
    csv_paths = _write_throughput_diagnostic_csvs(payload, output_dir, timestamp)
    payload["csv_artifacts"] = csv_paths
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(_throughput_diagnostic_markdown(payload), encoding="utf-8")
    return {
        "json_path": str(json_path),
        "markdown_path": str(md_path),
        "score_cache_path": str(score_cache_path),
        "csv_paths": csv_paths,
    }


def _count_settlement_excluded_quote_rows(decision_db: Path) -> Mapping[str, Any]:
    with sqlite3.connect(decision_db) as conn:
        if not _sqlite_relation_exists(conn, COMPACT_DECISION_VIEW):
            return {"available": False, "reason": f"{COMPACT_DECISION_VIEW} not found"}
        total_quote_rows = int(
            conn.execute(
                f"""
                SELECT COUNT(*)
                FROM {COMPACT_DECISION_VIEW}
                WHERE quote_timestamp_utc IS NOT NULL
                """
            ).fetchone()[0]
        )
        excluded = int(
            conn.execute(
                f"""
                SELECT COUNT(*)
                FROM {COMPACT_DECISION_VIEW}
                WHERE quote_timestamp_utc IS NOT NULL
                  AND (
                    COALESCE(exclude_from_backtest_replay, 0) != 0
                    OR COALESCE(recommended_action, '') = 'exclude_or_manual_review'
                  )
                """
            ).fetchone()[0]
        )
        actions = {
            str(row[0] or "missing"): int(row[1])
            for row in conn.execute(
                f"""
                SELECT COALESCE(recommended_action, ''), COUNT(*)
                FROM {COMPACT_DECISION_VIEW}
                WHERE quote_timestamp_utc IS NOT NULL
                GROUP BY COALESCE(recommended_action, '')
                """
            )
        }
    return {
        "available": True,
        "quote_timestamp_rows_before_official_filter": total_quote_rows,
        "excluded_for_manual_review_or_backtest_exclusion": excluded,
        "recommended_action_counts_on_quote_rows": actions,
    }


def _write_throughput_diagnostic_csvs(
    payload: Mapping[str, Any],
    output_dir: Path,
    timestamp: str,
) -> Mapping[str, str]:
    paths = {
        "gate_funnel": output_dir / f"{timestamp}-throughput-gate-funnel.csv",
        "one_gate_loosen_tests": output_dir / f"{timestamp}-throughput-one-gate-loosen-tests.csv",
        "variant_frontier": output_dir / f"{timestamp}-throughput-variant-frontier.csv",
        "coverage": output_dir / f"{timestamp}-throughput-coverage.csv",
    }
    total = float(payload.get("total_official_rows") or 0.0)
    cumulative = payload.get("gate_funnel", {}).get("cumulative_counts", {})
    individual_fail = payload.get("gate_funnel", {}).get("individual_gate_fail_counts", {})
    with paths["gate_funnel"].open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("stage", "cumulative_count", "pct_total", "individual_fail_count"),
        )
        writer.writeheader()
        for stage in payload.get("gate_funnel", {}).get("stage_order", ()):
            count = int(cumulative.get(stage) or 0)
            writer.writerow(
                {
                    "stage": stage,
                    "cumulative_count": count,
                    "pct_total": (count / total) if total else None,
                    "individual_fail_count": int(individual_fail.get(stage) or 0),
                }
            )
    with paths["one_gate_loosen_tests"].open("w", newline="", encoding="utf-8") as handle:
        fieldnames = (
            "variant_key",
            "label",
            "loosened_dimensions",
            "pre_cap_candidate_rows",
            "additional_pre_cap_candidate_rows_vs_baseline",
            "order_intents",
            "additional_order_intents_vs_baseline",
            "fills",
            "additional_fills_vs_baseline",
            "estimated_net_fee_adjusted_pnl_usd",
            "roi_after_fees",
        )
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in payload.get("one_gate_loosen_tests", ()):
            writer.writerow({key: row.get(key) for key in fieldnames})
    with paths["variant_frontier"].open("w", newline="", encoding="utf-8") as handle:
        fieldnames = (
            "variant_key",
            "label",
            "group",
            "loosened_dimensions",
            "pre_cap_candidate_rows",
            "order_intents",
            "fills",
            "gross_no_fee_pnl_usd",
            "net_fee_adjusted_pnl_usd",
            "total_fees_usd",
            "win_count",
            "loss_count",
            "flat_count",
            "roi_before_fees",
            "roi_after_fees",
            "fee_flips",
            "cap_block_counts",
        )
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for key, summary in payload.get("trade_frequency_frontier", {}).items():
            metrics = summary.get("frontier_metrics", {})
            writer.writerow(
                {
                    "variant_key": key,
                    "label": summary.get("label"),
                    "group": summary.get("group"),
                    "loosened_dimensions": json.dumps(summary.get("loosened_dimensions")),
                    "pre_cap_candidate_rows": summary.get("pre_cap_candidate_rows"),
                    "order_intents": summary.get("order_intents"),
                    "fills": summary.get("fills"),
                    "gross_no_fee_pnl_usd": metrics.get("estimated_gross_no_fee_pnl_usd"),
                    "net_fee_adjusted_pnl_usd": metrics.get("estimated_net_fee_adjusted_pnl_usd"),
                    "total_fees_usd": metrics.get("total_fees_usd"),
                    "win_count": metrics.get("win_count"),
                    "loss_count": metrics.get("loss_count"),
                    "flat_count": metrics.get("flat_count"),
                    "roi_before_fees": metrics.get("roi_before_fees"),
                    "roi_after_fees": metrics.get("roi_after_fees"),
                    "fee_flips": metrics.get("fee_flips"),
                    "cap_block_counts": json.dumps(summary.get("cap_block_counts")),
                }
            )
    with paths["coverage"].open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ("variant_key", "dimension", "bucket", "trades", "net_fee_adjusted_pnl_usd")
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for variant_key, summary in payload.get("trade_frequency_frontier", {}).items():
            for dimension, buckets in summary.get("coverage", {}).items():
                for bucket, values in buckets.items():
                    writer.writerow(
                        {
                            "variant_key": variant_key,
                            "dimension": dimension,
                            "bucket": bucket,
                            "trades": values.get("trades"),
                            "net_fee_adjusted_pnl_usd": values.get("net_fee_adjusted_pnl_usd"),
                        }
                    )
    return {key: str(path) for key, path in paths.items()}


def _throughput_diagnostic_markdown(payload: Mapping[str, Any]) -> str:
    baseline = payload.get("variants", {}).get("baseline", {})
    interp = payload.get("diagnostic_interpretation", {})
    runner = payload.get("weather_model_runner", {})
    lines = [
        "# Conservative FV Edge Throughput Diagnostic",
        "",
        "Mode: diagnostic-only. No parameter optimization, strategy ranking, live-setting promotion, or live-trading recommendation.",
        "",
        f"Rows scanned: `{payload.get('total_official_rows')}`",
        f"Runtime seconds: `{payload.get('runtime_seconds')}`",
        f"Rows/sec: `{payload.get('rows_per_sec')}`",
        f"Weather runner: `{runner.get('metadata', {}).get('runner_name')}`",
        f"Fixture runner used: `{runner.get('fixture_runner_used')}`",
        f"Model outputs generated/missing: `{runner.get('model_outputs_generated')}` / `{runner.get('missing_model_outputs')}`",
        f"Firewall violations: `{runner.get('model_input_firewall_violations')}`",
        "",
        "## Why Baseline Trade Count Is Low",
        "",
    ]
    for item in interp.get("why_baseline_trades_are_low", ()):
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            f"Baseline pre-cap candidate rows: `{baseline.get('pre_cap_candidate_rows')}`",
            f"Baseline order intents/fills: `{baseline.get('order_intents')}` / `{baseline.get('fills')}`",
            f"Baseline cap blocks: `{baseline.get('cap_block_counts')}`",
            "",
            "## Gate Funnel",
            "",
        ]
    )
    cumulative = payload.get("gate_funnel", {}).get("cumulative_counts", {})
    for stage in payload.get("gate_funnel", {}).get("stage_order", ()):
        lines.append(f"- `{stage}`: `{cumulative.get(stage)}`")
    lines.extend(["", "## Largest Gate Limiters", ""])
    for key, value in interp.get("largest_strategy_gate_skip_counts", ()):
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## One-Gate Loosen Tests", ""])
    for row in payload.get("one_gate_loosen_tests", ()):
        lines.append(
            "- "
            f"`{row.get('variant_key')}`: pre-cap `+{row.get('additional_pre_cap_candidate_rows_vs_baseline')}`, "
            f"fills `+{row.get('additional_fills_vs_baseline')}`, "
            f"net PnL `{row.get('estimated_net_fee_adjusted_pnl_usd')}`"
        )
    lines.extend(["", "## Trade-Frequency Frontier", ""])
    for key, summary in payload.get("trade_frequency_frontier", {}).items():
        metrics = summary.get("frontier_metrics", {})
        lines.append(
            "- "
            f"`{key}`: fills `{metrics.get('trade_count')}`, "
            f"gross PnL `{metrics.get('estimated_gross_no_fee_pnl_usd')}`, "
            f"net PnL `{metrics.get('estimated_net_fee_adjusted_pnl_usd')}`, "
            f"ROI after fees `{metrics.get('roi_after_fees')}`, "
            f"wins/losses `{metrics.get('win_count')}`/`{metrics.get('loss_count')}`"
        )
    lines.extend(["", "## Diagnostic Assessment", ""])
    lines.append("Dimensions adding pre-cap candidates without immediate negative fee-adjusted PnL:")
    for row in interp.get("one_gate_tests_that_add_pre_cap_candidates", ())[:10]:
        lines.append(
            f"- `{row.get('variant_key')}`: +{row.get('additional_pre_cap_candidate_rows_vs_baseline')} pre-cap candidates"
        )
    if not interp.get("one_gate_tests_that_add_pre_cap_candidates"):
        lines.append("- None.")
    lines.append("")
    lines.append("Danger flags:")
    if interp.get("frontier_danger_flags"):
        for row in interp.get("frontier_danger_flags", ()):
            lines.append(f"- `{row.get('variant_key')}`: {row.get('reason')}")
    else:
        lines.append("- No frontier variant had negative aggregate net fee-adjusted PnL in this diagnostic.")
    lines.extend(
        [
            "",
            "## Execution Realism",
            "",
            f"Order latency model: `{payload.get('execution_realism', {}).get('order_latency_model')}`",
            f"Latency blocker: `{payload.get('execution_realism', {}).get('blocker')}`",
            "",
            "## Recommended Next Step",
            "",
            interp.get("recommended_next_step", ""),
            "",
            "## Artifacts",
            "",
        ]
    )
    for key, value in payload.get("csv_artifacts", {}).items():
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    return "\n".join(lines)


def write_weather_model_runner_adapter_validation_report(
    *,
    work_root: Path,
    output_dir: Path,
    limit: int = 25,
    source_repo: Optional[Path] = None,
) -> dict[str, str]:
    from .weather_model_runner import ObservedHighGuardedShrinkageWeatherModelRunner

    work_root = Path(work_root).resolve()
    wide_db = work_root / WIDE_PRICE_DB_RELATIVE_PATH
    if wide_db.exists():
        raise RuntimeError(f"Wide price DB must remain unused for six_month_v1: {wide_db}")
    decision_db = work_root / "work" / "data" / "backtest" / "daily_weather_decision_artifact.sqlite"
    if not decision_db.exists():
        raise FileNotFoundError(decision_db)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    score_cache_path = output_dir / f"{timestamp}-observed-high-guarded-runner-shared-scores.jsonl"
    audit_csv_path = output_dir / f"{timestamp}-observed-high-guarded-runner-firewall-audit.csv"
    json_path = output_dir / f"{timestamp}-observed-high-guarded-runner-validation.json"
    md_path = output_dir / f"{timestamp}-observed-high-guarded-runner-validation.md"

    runner_kwargs: dict[str, Any] = {"work_root": work_root}
    if source_repo is not None:
        runner_kwargs["source_repo"] = Path(source_repo).expanduser().resolve()
    runner = ObservedHighGuardedShrinkageWeatherModelRunner(**runner_kwargs)
    runner_metadata = dict(runner.metadata())
    cache = ReplayGeneratedScoreCache()
    loader = SQLiteDecisionFrameLoader(
        decision_db,
        relation=COMPACT_DECISION_VIEW,
        official_trading_replay_only=True,
        require_compact_quote_store=True,
        point_in_time_policy=PointInTimePolicy(require_shared_score=True),
    )
    leakage_validator = DefaultLeakageValidator()
    counts: Counter[str] = Counter()
    coverage: dict[str, Counter[str]] = defaultdict(Counter)
    leakage_counts: Counter[str] = Counter()
    audit_rows: list[dict[str, Any]] = []
    p_yes_values: list[float] = []
    for frame in loader.iter_frames(limit=limit):
        decision = frame.decision
        counts["decision_frames_seen"] += 1
        coverage["city"][decision.normalized_city or "missing"] += 1
        coverage["bucket_type"][decision.bucket_type or "missing"] += 1
        coverage["lead_days"][str(decision.lead_days)] += 1
        coverage["decision_hour_utc"][str(decision.decision_hour_utc)] += 1
        context = WeatherModelDecisionContext.from_decision(decision)
        if context.observed_high_omitted_reason:
            counts["future_observed_high_snapshots_omitted"] += 1
        row = {
            "decision_id": decision.decision_id,
            "universe_row_id": decision.universe_row_id,
            "decision_time_utc": decision.decision_time_utc.isoformat(),
            "forecast_as_of_utc": (
                context.forecast_run_time_utc.isoformat()
                if context.forecast_run_time_utc
                else ""
            ),
            "input_hash": context.input_hash,
            "status": "pending",
            "p_yes": "",
            "p_no": "",
            "p_side": "",
            "model_version": runner_metadata.get("model_version", ""),
            "model_code_hash": runner_metadata.get("model_code_hash", ""),
            "model_config_hash": runner_metadata.get("model_config_hash", ""),
            "prediction_hash": "",
            "firewall_violation": "",
            "leakage_violation_codes": "",
            "observed_high_omitted_reason": context.observed_high_omitted_reason or "",
        }
        try:
            validate_weather_model_context_firewall(context)
            prediction = runner.predict(context)
        except ValueError as exc:
            counts["model_input_firewall_violations"] += 1
            row["status"] = "firewall_violation"
            row["firewall_violation"] = str(exc)
            audit_rows.append(row)
            continue
        if prediction is None:
            counts["missing_model_outputs"] += 1
            row["status"] = "missing_model_output"
            audit_rows.append(row)
            continue
        score = prediction.to_shared_score(
            side=decision.side,
            observed_high=_model_visible_observed_high(decision),
            parser_confidence=decision.parser_confidence,
            settlement_unit=None,
            settlement_precision=None,
        )
        if not score.is_valid:
            counts["invalid_shared_scores"] += 1
            row["status"] = "invalid_shared_score"
            row["firewall_violation"] = ",".join(score.score_errors)
            audit_rows.append(row)
            continue
        record = cache.add(context=context, prediction=prediction, score=score)
        generated_frame = ReplayFrame(
            decision=decision,
            canonical_label=frame.canonical_label,
            score=score,
            skip_reasons=(),
        )
        violations = leakage_validator.validate(generated_frame)
        for violation in violations:
            leakage_counts[violation.code] += 1
        counts["model_outputs_generated"] += 1
        p_yes_values.append(prediction.p_yes)
        row.update(
            {
                "status": "generated",
                "p_yes": prediction.p_yes,
                "p_no": prediction.p_no,
                "p_side": score.p_side,
                "prediction_hash": record.prediction_hash,
                "leakage_violation_codes": ",".join(
                    sorted(violation.code for violation in violations)
                ),
            }
        )
        audit_rows.append(row)

    cache.write_jsonl(score_cache_path)
    _write_dict_csv(audit_csv_path, audit_rows)
    payload = {
        "schema_version": BACKTEST_FRAMEWORK_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "weather_model_runner_adapter_validation_no_strategy_no_pnl",
        "dataset_id": "six_month_v1",
        "freeze_manifest": str(work_root / SIX_MONTH_V1_FREEZE_MANIFEST),
        "decision_artifact_db": str(decision_db),
        "compact_only": True,
        "official_trading_replay_sql_predicate": OFFICIAL_TRADING_REPLAY_SQL_PREDICATE,
        "row_limit": limit,
        "weather_model_runner": {
            "metadata": runner_metadata,
            "fixture_runner_used": False,
            "research_only_not_promoted": True,
            "model_outputs_generated": counts["model_outputs_generated"],
            "missing_model_outputs": counts["missing_model_outputs"],
            "model_input_firewall_violations": counts["model_input_firewall_violations"],
            "invalid_shared_scores": counts["invalid_shared_scores"],
        },
        "model_input_firewall": {
            "forbidden_fields": list(FORBIDDEN_WEATHER_MODEL_CONTEXT_FIELDS),
            "live_valid_model_inputs_only": True,
            "market_prices_passed_to_runner": False,
            "labels_or_outcomes_passed_to_runner": False,
            "canonical_or_settlement_labels_passed_to_runner": False,
            "future_weather_observations_passed_to_runner": False,
            "future_observed_high_snapshots_omitted": counts[
                "future_observed_high_snapshots_omitted"
            ],
        },
        "decision_counts": dict(sorted(counts.items())),
        "probability_summary": _probability_summary(p_yes_values),
        "replay_generated_score_cache": cache.report_summary(
            artifact_path=score_cache_path
        ),
        "artifact_paths": {
            "summary_json": str(json_path),
            "markdown_report": str(md_path),
            "replay_generated_shared_scores_jsonl": str(score_cache_path),
            "firewall_audit_csv": str(audit_csv_path),
        },
        "coverage": {
            key: dict(counter.most_common(200))
            for key, counter in sorted(coverage.items())
        },
        "leakage_validation": {
            "violation_counts": dict(sorted(leakage_counts.items())),
        },
        "strategy_pnl_backtest_run": False,
        "strategy_evaluation_run": False,
        "model_weights_or_defaults_changed": False,
        "profitability_claim": "none",
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(_weather_model_runner_validation_markdown(payload), encoding="utf-8")
    return {
        "json_path": str(json_path),
        "markdown_path": str(md_path),
        "score_cache_path": str(score_cache_path),
        "firewall_audit_csv": str(audit_csv_path),
    }


def _write_dict_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        if not fieldnames:
            handle.write("")
            return
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_cell(row.get(key, "")) for key in fieldnames})


def _write_trade_ledger_csv(path: Path, trade_records: Sequence[TradeAccountingRecord]) -> None:
    _write_dict_csv(path, [record.to_report_dict() for record in trade_records])


def _csv_cell(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (str, int, float)):
        return value
    if isinstance(value, bool):
        return int(value)
    return json.dumps(_json_ready(value), sort_keys=True)


def _probability_summary(values: Sequence[float]) -> Mapping[str, Any]:
    if not values:
        return {"n": 0, "min": None, "max": None, "mean": None}
    return {
        "n": len(values),
        "min": round(min(values), 8),
        "max": round(max(values), 8),
        "mean": round(sum(values) / len(values), 8),
    }


def _weather_model_runner_validation_markdown(payload: Mapping[str, Any]) -> str:
    runner = payload.get("weather_model_runner", {})
    metadata = runner.get("metadata", {})
    counts = payload.get("decision_counts", {})
    cache = payload.get("replay_generated_score_cache", {})
    leakage = payload.get("leakage_validation", {}).get("violation_counts", {})
    artifacts = payload.get("artifact_paths", {})
    lines = [
        "# Observed-High Guarded Weather Runner Adapter Validation",
        "",
        "Mode: adapter validation only. No strategy evaluation, no PnL backtest, no parameter search, and no profitability claim.",
        "",
        f"Dataset: `{payload.get('dataset_id')}`",
        f"Rows scanned: `{counts.get('decision_frames_seen', 0)}`",
        f"Model outputs generated: `{runner.get('model_outputs_generated', 0)}`",
        f"Missing model outputs: `{runner.get('missing_model_outputs', 0)}`",
        f"Firewall violations: `{runner.get('model_input_firewall_violations', 0)}`",
        f"Invalid shared scores: `{runner.get('invalid_shared_scores', 0)}`",
        f"Research-only model version: `{metadata.get('model_version')}`",
        f"Model code hash: `{metadata.get('model_code_hash')}`",
        f"Model config hash: `{metadata.get('model_config_hash')}`",
        "",
        "## Input Firewall",
        "",
        f"- Market prices passed to runner: `{payload.get('model_input_firewall', {}).get('market_prices_passed_to_runner')}`",
        f"- Labels/outcomes passed to runner: `{payload.get('model_input_firewall', {}).get('labels_or_outcomes_passed_to_runner')}`",
        f"- Settlement labels passed to runner: `{payload.get('model_input_firewall', {}).get('canonical_or_settlement_labels_passed_to_runner')}`",
        f"- Future observed-high snapshots omitted: `{payload.get('model_input_firewall', {}).get('future_observed_high_snapshots_omitted')}`",
        "",
        "## Replay-Generated Score Cache",
        "",
        f"- Cache name: `{cache.get('cache_name')}`",
        f"- Records: `{cache.get('record_count')}`",
        f"- Artifact: `{cache.get('artifact_path')}`",
        "",
        "## Leakage Validation",
        "",
    ]
    if leakage:
        for key, value in sorted(leakage.items()):
            lines.append(f"- `{key}`: `{value}`")
    else:
        lines.append("- No leakage violations recorded for generated scores.")
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- Summary JSON: `{artifacts.get('summary_json')}`",
            f"- Shared-score JSONL: `{artifacts.get('replay_generated_shared_scores_jsonl')}`",
            f"- Firewall audit CSV: `{artifacts.get('firewall_audit_csv')}`",
            "",
            "## Verdict",
            "",
            "Adapter smoke validation completed. The model remains research-only and is not promoted to production, paper defaults, live defaults, model weights, or strategy settings.",
            "",
        ]
    )
    return "\n".join(lines)


def _candidate_replay_markdown(payload: Mapping[str, Any]) -> str:
    counts = payload.get("decision_counts", {})
    accounting = payload.get("accounting_summary", {})
    quote_audit = payload.get("quote_source_audit", {})
    model_runner = payload.get("weather_model_runner", {})
    score_cache = payload.get("replay_generated_score_cache", {})
    leakage = payload.get("leakage_validation", {}).get("violation_counts", {})
    lines = [
        "# Conservative FV Edge Candidate Replay",
        "",
        "Mode: small baseline candidate replay scaffold. No order execution, no parameter search, no strategy comparison, and no profitability claim.",
        "",
        f"Dataset: `{payload.get('dataset_id')}`",
        f"Rows scanned: `{counts.get('total_decision_rows_seen')}`",
        f"Rows available for strategy input: `{counts.get('rows_available_for_strategy_input')}`",
        f"Order intents: `{counts.get('order_intents')}`",
        f"Fills: `{counts.get('fills')}`",
        f"Replay-generated score rows: `{model_runner.get('model_outputs_generated')}`",
        f"Score cache artifact: `{score_cache.get('artifact_path')}`",
        "",
        "## Accounting Assumptions",
        "",
        "- Buys execute at ask.",
        "- Sells execute at bid.",
        "- This first small report does not execute orders or settle positions.",
        "- Fill and exit/hold primitives are available for validation tests only in this phase.",
        f"- Cash delta: `{accounting.get('cash_delta_usd')}`",
        f"- Economic strategy results performed: `{accounting.get('economic_strategy_results_performed')}`",
        "",
        "## Quote Source Audit",
        "",
        f"- Frames with quote source cache path: `{quote_audit.get('frames_with_quote_source_cache_path')}`",
        f"- Frames with quote source payload id: `{quote_audit.get('frames_with_quote_source_payload_id')}`",
        "",
        "## Skip Reasons",
        "",
    ]
    for key, value in sorted(payload.get("skip_reason_counts", {}).items()):
        lines.append(f"- `{key}`: `{value}`")
    if not payload.get("skip_reason_counts"):
        lines.append("- None.")
    lines.extend(["", "## Leakage Validation", ""])
    if leakage:
        for key, value in sorted(leakage.items()):
            lines.append(f"- `{key}`: `{value}`")
    else:
        lines.append("- No leakage violations recorded.")
    lines.extend(
        [
            "",
            "## Profitability Claim",
            "",
            f"`{payload.get('profitability_claim')}`. Profitability claims remain blocked until walk-forward, plateau, and full reporting gates pass.",
            "",
        ]
    )
    return "\n".join(lines)


def _accounting_replay_markdown(payload: Mapping[str, Any]) -> str:
    counts = payload.get("decision_counts", {})
    accounting = payload.get("fixed_strategy_accounting", {})
    pnl = accounting.get("pnl", {})
    fills = accounting.get("fill_counts", {})
    execution_policy = accounting.get("execution_policy", {})
    fee_model = execution_policy.get("fee_model", {})
    runner = payload.get("weather_model_runner", {})
    score_cache = payload.get("replay_generated_score_cache", {})
    trade_ledger = payload.get("full_trade_ledger", {})
    leakage = payload.get("leakage_validation", {}).get("violation_counts", {})
    lines = [
        "# Conservative FV Edge Fixed Accounting Replay",
        "",
        "Mode: fixed-parameter accounting replay. No parameter search, no strategy comparison, and no live-trading recommendation.",
        "",
        f"Dataset: `{payload.get('dataset_id')}`",
        f"Rows scanned: `{counts.get('total_decision_rows_seen')}`",
        f"Weather runner: `{runner.get('metadata', {}).get('runner_name')}`",
        f"Fixture runner used: `{runner.get('fixture_runner_used')}`",
        f"Model outputs generated: `{runner.get('model_outputs_generated')}`",
        f"Missing model outputs: `{runner.get('missing_model_outputs')}`",
        f"Firewall violations: `{runner.get('model_input_firewall_violations')}`",
        f"Score cache artifact: `{score_cache.get('artifact_path')}`",
        f"Full trade ledger: `{trade_ledger.get('path')}`",
        "",
        "## Execution",
        "",
        f"- Order intents: `{fills.get('order_intents')}`",
        f"- Attempted fills: `{fills.get('attempted_fills')}`",
        f"- Actual fills: `{fills.get('actual_fills')}`",
        f"- Skipped fills: `{fills.get('skipped_fills')}`",
        f"- Fills in report counts: `{counts.get('fills')}`",
        f"- Settlements: `{counts.get('settlements')}`",
        f"- Fee model: `{fee_model.get('fee_model')}`",
        f"- Fee rate: `{fee_model.get('fee_rate')}`",
        f"- Order latency model: `{execution_policy.get('order_latency_model')}`",
        "",
        "## Accounting",
        "",
        f"- Gross no-fee cost: `{pnl.get('gross_no_fee_total_cost_usd')}`",
        f"- Total fees: `{pnl.get('total_fees_usd')}`",
        f"- Average fee per trade: `{pnl.get('average_fee_per_trade_usd')}`",
        f"- Total cost: `{pnl.get('total_cost_usd')}`",
        f"- Total payout: `{pnl.get('total_payout_usd')}`",
        f"- Gross no-fee realized PnL: `{pnl.get('gross_no_fee_realized_pnl_usd')}`",
        f"- Net fee-adjusted realized PnL: `{pnl.get('net_fee_adjusted_realized_pnl_usd')}`",
        f"- ROI before fees: `{pnl.get('roi_before_fees_on_deployed_capital')}`",
        f"- ROI after fees: `{pnl.get('roi_after_fees_on_deployed_capital')}`",
        f"- Trades flipped profitable to unprofitable by fees: `{pnl.get('trades_flipped_profitable_to_unprofitable_by_fees')}`",
        f"- Win/loss/flat: `{pnl.get('win_count')}` / `{pnl.get('loss_count')}` / `{pnl.get('flat_count')}`",
        f"- Open positions: `{pnl.get('open_position_count')}`",
        f"- Max drawdown: `{pnl.get('max_drawdown_usd')}`",
        "",
        "## Skip Reasons",
        "",
    ]
    for key, value in sorted(
        payload.get("skip_reason_counts", {}).items(),
        key=lambda item: item[1],
        reverse=True,
    )[:20]:
        lines.append(f"- `{key}`: `{value}`")
    if not payload.get("skip_reason_counts"):
        lines.append("- None.")
    lines.extend(["", "## Leakage Validation", ""])
    if leakage:
        for key, value in sorted(leakage.items()):
            lines.append(f"- `{key}`: `{value}`")
    else:
        lines.append("- No leakage violations recorded.")
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "This reports accounting for this exact fixed replay only. It is not a parameter search, strategy ranking, live-trading recommendation, or general profitability claim.",
            "",
        ]
    )
    return "\n".join(lines)


def write_gen2_strategy_validation_report(
    *,
    work_root: Path,
    output_dir: Path,
    limit: int = 0,
    starting_cash_usd: float = 100.0,
    weather_runner: str = "observed-high-guarded",
    fee_model: str = DEFAULT_STRATEGY_REPLAY_FEE_MODEL,
    fee_rate: Optional[float] = DEFAULT_STRATEGY_REPLAY_FEE_RATE,
    source_repo: Optional[Path] = None,
    progress_every: int = 50000,
) -> dict[str, Any]:
    work_root = Path(work_root).resolve()
    wide_db = work_root / WIDE_PRICE_DB_RELATIVE_PATH
    if wide_db.exists():
        raise RuntimeError(f"Wide price DB must remain unused for six_month_v1: {wide_db}")
    decision_db = work_root / "work" / "data" / "backtest" / "daily_weather_decision_artifact.sqlite"
    if not decision_db.exists():
        raise FileNotFoundError(decision_db)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    runner_slug = _weather_runner_slug(weather_runner)
    score_cache_path = output_dir / (
        f"{timestamp}-{runner_slug}-gen2-strategy-validation-generated-shared-scores.jsonl.gz"
    )
    score_cache = ReplayGeneratedScoreCache(
        stream_path=score_cache_path,
        compress=True,
        keep_records=False,
    )
    model_runner = _build_candidate_weather_model_runner(
        weather_runner,
        work_root=work_root,
        source_repo=source_repo,
    )
    loader = SQLiteDecisionFrameLoader(
        decision_db,
        relation=COMPACT_DECISION_VIEW,
        official_trading_replay_only=True,
        require_compact_quote_store=True,
        point_in_time_policy=PointInTimePolicy(require_shared_score=True),
    )
    fee_config = FeeModelConfig(fee_model=fee_model, fee_rate=fee_rate)
    with CompactQuoteHistoryLookup.from_work_root(work_root) as quote_lookup:
        payload, states = run_gen2_strategy_validation(
            loader,
            starting_cash_usd=starting_cash_usd,
            limit=limit,
            model_runner=model_runner,
            score_cache=score_cache,
            delayed_quote_lookup=quote_lookup,
            fee_config=fee_config,
            delay_seconds=DEFAULT_DELAY_STRESS_SECONDS,
            worse_fill_tick=DEFAULT_WORSE_FILL_TICK,
            progress_every=progress_every,
        )
    score_cache.write_jsonl(score_cache_path)
    json_path = output_dir / f"{timestamp}-gen2-strategy-validation.json"
    md_path = output_dir / f"{timestamp}-gen2-strategy-validation.md"
    csv_paths = _write_gen2_strategy_validation_csvs(
        states=states,
        payload=payload,
        output_dir=output_dir,
        timestamp=timestamp,
    )
    payload = {
        **payload,
        "freeze_manifest": str(work_root / SIX_MONTH_V1_FREEZE_MANIFEST),
        "decision_artifact_db": str(decision_db),
        "compact_price_db": str(work_root / COMPACT_PRICE_DB_RELATIVE_PATH),
        "compact_only": True,
        "wide_price_db_exists": False,
        "code_hashes": {
            "backtesting_engine_py_sha256": _file_sha256(Path(__file__)),
        },
        "excluded_quote_rows_for_settlement_or_manual_review": dict(
            _count_settlement_excluded_quote_rows(decision_db)
        ),
        "replay_generated_score_cache": score_cache.report_summary(
            artifact_path=score_cache_path
        ),
        "csv_artifacts": csv_paths,
        "full_trade_ledger": {
            "persisted": True,
            "format": "csv",
            "path": csv_paths["trade_ledger"],
            "stress_scenarios_included": [
                "base",
                "delayed_entry_2s",
                "worse_fill_1tick",
            ],
            "base_trade_rows": sum(len(state.trade_records) for state in states),
        },
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(_gen2_strategy_validation_markdown(payload), encoding="utf-8")
    return {
        "json_path": str(json_path),
        "markdown_path": str(md_path),
        "score_cache_path": str(score_cache_path),
        **csv_paths,
    }


def write_gen2_cap_divergence_report(
    *,
    work_root: Path,
    output_dir: Path,
    variant_key: str,
    limit: int = 0,
    starting_cash_usd: float = 100.0,
    weather_runner: str = "observed-high-guarded",
    source_repo: Optional[Path] = None,
    progress_every: int = 50000,
) -> dict[str, Any]:
    work_root = Path(work_root).resolve()
    wide_db = work_root / WIDE_PRICE_DB_RELATIVE_PATH
    if wide_db.exists():
        raise RuntimeError(f"Wide price DB must remain unused for six_month_v1: {wide_db}")
    decision_db = work_root / "work" / "data" / "backtest" / "daily_weather_decision_artifact.sqlite"
    if not decision_db.exists():
        raise FileNotFoundError(decision_db)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    runner_slug = _weather_runner_slug(weather_runner)
    score_cache_path = output_dir / (
        f"{timestamp}-{runner_slug}-gen2-cap-divergence-generated-shared-scores.jsonl.gz"
    )
    score_cache = ReplayGeneratedScoreCache(
        stream_path=score_cache_path,
        compress=True,
        keep_records=False,
    )
    model_runner = _build_candidate_weather_model_runner(
        weather_runner,
        work_root=work_root,
        source_repo=source_repo,
    )
    loader = SQLiteDecisionFrameLoader(
        decision_db,
        relation=COMPACT_DECISION_VIEW,
        official_trading_replay_only=True,
        require_compact_quote_store=True,
        point_in_time_policy=PointInTimePolicy(require_shared_score=True),
    )
    payload = run_gen2_cap_divergence_diagnostic(
        loader,
        variant_key=variant_key,
        starting_cash_usd=starting_cash_usd,
        limit=limit,
        model_runner=model_runner,
        score_cache=score_cache,
        progress_every=progress_every,
    )
    score_cache.write_jsonl(score_cache_path)
    payload = {
        **payload,
        "freeze_manifest": str(work_root / SIX_MONTH_V1_FREEZE_MANIFEST),
        "decision_artifact_db": str(decision_db),
        "compact_price_db": str(work_root / COMPACT_PRICE_DB_RELATIVE_PATH),
        "compact_only": True,
        "wide_price_db_exists": False,
        "code_hashes": {
            "backtesting_engine_py_sha256": _file_sha256(Path(__file__)),
        },
        "replay_generated_score_cache": score_cache.report_summary(
            artifact_path=score_cache_path
        ),
    }
    json_path = output_dir / f"{timestamp}-gen2-cap-divergence-report.json"
    md_path = output_dir / f"{timestamp}-gen2-cap-divergence-report.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(_gen2_cap_divergence_markdown(payload), encoding="utf-8")
    return {
        "json_path": str(json_path),
        "markdown_path": str(md_path),
        "score_cache_path": str(score_cache_path),
    }


def write_old_no_tail_live_forward_shadow_report(
    *,
    work_root: Path,
    output_dir: Path,
    limit: int = 250,
    starting_cash_usd: float = 100.0,
    weather_runner: str = "observed-high-guarded",
    source_repo: Optional[Path] = None,
    progress_every: int = 0,
) -> dict[str, Any]:
    work_root = Path(work_root).resolve()
    wide_db = work_root / WIDE_PRICE_DB_RELATIVE_PATH
    if wide_db.exists():
        raise RuntimeError(f"Wide price DB must remain unused for six_month_v1: {wide_db}")
    decision_db = work_root / "work" / "data" / "backtest" / "daily_weather_decision_artifact.sqlite"
    if not decision_db.exists():
        raise FileNotFoundError(decision_db)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    runner_slug = _weather_runner_slug(weather_runner)
    score_cache_path = output_dir / (
        f"{timestamp}-{runner_slug}-old-no-tail-live-forward-shadow-scores.jsonl.gz"
    )
    events_path = output_dir / f"{timestamp}-old-no-tail-live-forward-shadow-events.jsonl"
    json_path = output_dir / f"{timestamp}-old-no-tail-live-forward-shadow.json"
    md_path = output_dir / f"{timestamp}-old-no-tail-live-forward-shadow.md"
    score_cache = ReplayGeneratedScoreCache(
        stream_path=score_cache_path,
        compress=True,
        keep_records=False,
    )
    model_runner = _build_candidate_weather_model_runner(
        weather_runner,
        work_root=work_root,
        source_repo=source_repo,
    )
    loader = SQLiteDecisionFrameLoader(
        decision_db,
        relation=COMPACT_DECISION_VIEW,
        official_trading_replay_only=False,
        require_compact_quote_store=True,
        point_in_time_policy=PointInTimePolicy(
            require_shared_score=True,
            require_canonical_label=False,
        ),
    )
    payload = dict(
        run_old_no_tail_live_forward_shadow_logging(
            loader,
            starting_cash_usd=starting_cash_usd,
            limit=limit,
            model_runner=model_runner,
            score_cache=score_cache,
            progress_every=progress_every,
        )
    )
    score_cache.write_jsonl(score_cache_path)
    events = list(payload.pop("shadow_events", ()))
    with events_path.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(_json_ready(event), sort_keys=True) + "\n")
    payload = {
        **payload,
        "decision_artifact_db": str(decision_db),
        "compact_only": True,
        "wide_price_db_exists": False,
        "replay_generated_score_cache": score_cache.report_summary(
            artifact_path=score_cache_path
        ),
        "shadow_events_artifact": {
            "path": str(events_path),
            "format": "jsonl",
            "event_count": len(events),
        },
        "shadow_events_preview": events[:20],
    }
    json_path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(_old_no_tail_live_forward_shadow_markdown(payload), encoding="utf-8")
    return {
        "json_path": str(json_path),
        "markdown_path": str(md_path),
        "events_jsonl_path": str(events_path),
        "score_cache_path": str(score_cache_path),
    }


def _old_no_tail_live_forward_shadow_markdown(payload: Mapping[str, Any]) -> str:
    counts = payload.get("decision_counts", {})
    plan = payload.get("pre_registered_plan", {})
    expected = plan.get("expected_trades_per_utc_day", {}) if isinstance(plan, Mapping) else {}
    bands = plan.get("hit_rate_confidence_bands", {}) if isinstance(plan, Mapping) else {}
    events = payload.get("shadow_events_artifact", {})
    return "\n".join(
        [
            "# Old NO-tail Live-Forward Shadow Log",
            "",
            "Mode: read-only live-forward shadow logging. No real orders are placed, no fills are simulated, and no historical PnL number is treated as expected performance.",
            "",
            "## Frozen Profile",
            "",
            f"- Profile key: `{payload.get('profile_key')}`",
            f"- Strategy family: `{payload.get('strategy_family')}`",
            "- Semantics: `old_high_confidence_no_tail`, `no14`, flat `$2`, `window_bankroll`, `bprice70`.",
            f"- Real order placement allowed: `{payload.get('real_order_placement_allowed')}`",
            "",
            "## Pre-registration",
            "",
            f"- Expected trades/day point: `{expected.get('point_estimate')}`",
            f"- Expected trades/day band: `{expected.get('low')}` to `{expected.get('high')}`",
            f"- Hit-rate primary interval: `{bands.get('primary_interval')}`",
            f"- Minimum resolved would-trades before interpretation: `{bands.get('minimum_resolved_would_trades_before_interpretation')}`",
            "",
            "## Current Log Summary",
            "",
            f"- Rows seen: `{payload.get('rows_seen')}`",
            f"- Would-trade signals: `{counts.get('would_trade_signals')}`",
            f"- No-trade decisions: `{counts.get('no_trade_decisions')}`",
            f"- Event log: `{events.get('path')}`",
            "",
            "## Boundaries",
            "",
            "- This artifact is forward-evidence collection only.",
            "- Settlement outcomes are logged when canonical labels are available; unresolved rows remain pending.",
            "- Live money remains blocked by the migration brief gates.",
            "",
        ]
    )


def _write_gen2_strategy_validation_csvs(
    *,
    states: Sequence[Gen2ValidationVariantState],
    payload: Mapping[str, Any],
    output_dir: Path,
    timestamp: str,
) -> Mapping[str, str]:
    ranked_path = output_dir / f"{timestamp}-gen2-ranked-grid.csv"
    ledger_path = output_dir / f"{timestamp}-gen2-full-trade-ledger.csv"
    monthly_path = output_dir / f"{timestamp}-gen2-monthly-window-pnl.csv"
    concentration_path = output_dir / f"{timestamp}-gen2-concentration.csv"
    _write_dict_csv(
        ranked_path,
        [
            {
                **{field: value for field, value in row.items() if field != "parameters"},
                "parameters_json": row.get("parameters"),
                "criteria_verdict": payload.get("criteria_results", {})
                .get(row.get("variant_key"), {})
                .get("verdict"),
            }
            for row in payload.get("ranked_variants", [])
        ],
    )
    ledger_rows: list[Mapping[str, Any]] = []
    for state in states:
        for scenario, records in (
            ("base", state.trade_records),
            ("delayed_entry_2s", state.delayed_trade_records),
            ("worse_fill_1tick", state.worse_fill_trade_records),
        ):
            for record in records:
                ledger_rows.append(
                    {
                        "variant_key": state.spec.key,
                        "variant_label": state.spec.label,
                        "stage": state.spec.stage,
                        "strategy_family": state.spec.strategy_family,
                        "stress_scenario": scenario,
                        **record.to_report_dict(),
                    }
                )
    _write_dict_csv(ledger_path, ledger_rows)
    monthly_rows: list[Mapping[str, Any]] = []
    concentration_rows: list[Mapping[str, Any]] = []
    for key, summary in payload.get("variants", {}).items():
        for month, row in summary.get("monthly_window_split_pnl", {}).items():
            monthly_rows.append({"variant_key": key, "month": month, **row})
        for dimension, slice_report in (
            ("city", summary.get("base_accounting", {}).get("pnl_by_city", {})),
            (
                "country_region",
                summary.get("city_region_timezone_concentration", {}).get("country_region", {}),
            ),
            (
                "timezone",
                summary.get("city_region_timezone_concentration", {}).get("timezone", {}),
            ),
            ("target_date", summary.get("pnl_by_target_date", {})),
        ):
            for bucket, row in slice_report.items():
                concentration_rows.append(
                    {
                        "variant_key": key,
                        "dimension": dimension,
                        "bucket": bucket,
                        **row,
                    }
                )
    _write_dict_csv(monthly_path, monthly_rows)
    _write_dict_csv(concentration_path, concentration_rows)
    return {
        "ranked_grid_csv": str(ranked_path),
        "trade_ledger": str(ledger_path),
        "monthly_window_pnl_csv": str(monthly_path),
        "concentration_csv": str(concentration_path),
    }


def write_gen2_reproducibility_audit_report(
    *,
    run_json_paths: Sequence[Path],
    output_dir: Path,
    required_runs: int = GEN2_REPRODUCIBILITY_REQUIRED_RUNS,
) -> dict[str, Any]:
    if len(run_json_paths) < required_runs:
        raise ValueError(f"Need at least {required_runs} Gen-2 run reports for reproducibility audit")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_reports = [_load_gen2_reproducibility_run(Path(path)) for path in run_json_paths]
    baseline = run_reports[0]
    matching_runs = [
        report
        for report in run_reports
        if report["stable_config_fingerprint"] == baseline["stable_config_fingerprint"]
        and report["ranked_grid_csv_sha256"] == baseline["ranked_grid_csv_sha256"]
        and report["trade_ledger_csv_sha256"] == baseline["trade_ledger_csv_sha256"]
    ]
    reproduced_count = len(matching_runs)
    ranked_identical = len({report["ranked_grid_csv_sha256"] for report in run_reports}) == 1
    ledger_identical = len({report["trade_ledger_csv_sha256"] for report in run_reports}) == 1
    config_identical = len({report["stable_config_fingerprint"] for report in run_reports}) == 1
    score_raw_identical = len({report.get("score_cache_raw_sha256") for report in run_reports}) == 1
    score_semantic_identical = (
        len({report.get("score_cache_semantic_sha256_excluding_run_timestamps") for report in run_reports}) == 1
    )
    passed = (
        reproduced_count >= required_runs
        and ranked_identical
        and ledger_identical
        and config_identical
    )
    first_payload = baseline["payload"]
    first_verdict = first_payload.get("overall_verdict", {})
    replay_gate_pass_variants = list(
        first_verdict.get("replay_gate_pass_variants")
        or first_verdict.get("live_candidate_variants")
        or []
    )
    if passed and replay_gate_pass_variants:
        re_emitted = {
            "status": "live-candidate",
            "live_candidate_variants": sorted(replay_gate_pass_variants),
            "research_only_variants": first_verdict.get("research_only_variants", []),
            "rejected_variants": first_verdict.get("rejected_variants", []),
            "claim_boundary": (
                "No live-trading recommendation or general profitability claim. "
                "This verdict was re-emitted only after byte-identical ranked-grid and "
                "full-trade-ledger artifacts reproduced across the required independent runs."
            ),
        }
    else:
        re_emitted = {
            "status": "suspended_pending_reproducibility",
            "live_candidate_variants": [],
            "replay_gate_pass_variants": sorted(replay_gate_pass_variants),
            "claim_boundary": (
                "Live-candidate verdict remains suspended until the required independent "
                "runs reproduce byte-identical ranked-grid and full-trade-ledger artifacts."
            ),
        }
    payload = {
        "schema_version": BACKTEST_FRAMEWORK_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "gen2_reproducibility_audit",
        "required_artifacts": [
            "gen2_strategy_validation_json",
            "gen2_ranked_grid_csv",
            "gen2_full_trade_ledger_csv",
        ],
        "run_count": len(run_reports),
        "runs": [
            {key: value for key, value in report.items() if key != "payload"}
            for report in run_reports
        ],
        "comparisons": {
            "stable_config_fingerprint_identical": config_identical,
            "ranked_grid_csv_byte_identical": ranked_identical,
            "full_trade_ledger_csv_byte_identical": ledger_identical,
            "score_cache_gzip_byte_identical": score_raw_identical,
            "score_cache_semantic_hash_excluding_run_timestamps_identical": score_semantic_identical,
            "score_cache_byte_identical_claim_allowed": score_raw_identical,
            "score_cache_semantic_hash_note": (
                "Semantic score-cache hash excludes generated_at_utc and prediction_timestamp_utc only."
            ),
        },
        "reproducibility": _gen2_reproducibility_field(
            independent_runs_reproduced=reproduced_count,
            required_runs=required_runs,
            ranked_grid_byte_identical=ranked_identical,
            full_trade_ledger_byte_identical=ledger_identical,
            re_emitted=passed,
        ),
        "re_emitted_overall_verdict": re_emitted,
        "old_disputed_artifacts_preserved": True,
        "no_live_trading": True,
        "no_model_promotion": True,
        "no_strategy_promotion": True,
    }
    json_path = output_dir / f"{timestamp}-gen2-reproducibility-audit.json"
    md_path = output_dir / f"{timestamp}-gen2-reproducibility-audit.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(_gen2_reproducibility_audit_markdown(payload), encoding="utf-8")
    return {
        "json_path": str(json_path),
        "markdown_path": str(md_path),
    }


def _load_gen2_reproducibility_run(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    csvs = payload.get("csv_artifacts", {})
    ranked_path = Path(csvs.get("ranked_grid_csv") or "")
    ledger_path = Path(csvs.get("trade_ledger") or "")
    if not ranked_path.exists():
        raise FileNotFoundError(ranked_path)
    if not ledger_path.exists():
        raise FileNotFoundError(ledger_path)
    score_path_text = payload.get("replay_generated_score_cache", {}).get("artifact_path")
    score_summary = _gen2_score_cache_hash_summary(Path(score_path_text)) if score_path_text else {}
    return {
        "run_json_path": str(path),
        "payload": payload,
        "stable_config_fingerprint": _gen2_stable_run_fingerprint(payload),
        "ranked_grid_csv_path": str(ranked_path),
        "ranked_grid_csv_sha256": _file_sha256(ranked_path),
        "ranked_grid_csv_rows": _csv_data_row_count(ranked_path),
        "trade_ledger_csv_path": str(ledger_path),
        "trade_ledger_csv_sha256": _file_sha256(ledger_path),
        "trade_ledger_csv_rows": _csv_data_row_count(ledger_path),
        "total_official_rows": payload.get("total_official_rows"),
        "row_limit": payload.get("row_limit"),
        "overall_verdict_status": payload.get("overall_verdict", {}).get("status"),
        "replay_gate_pass_variant_count": len(
            payload.get("overall_verdict", {}).get("replay_gate_pass_variants", [])
            or payload.get("overall_verdict", {}).get("live_candidate_variants", [])
            or []
        ),
        **score_summary,
    }


def _gen2_stable_run_fingerprint(payload: Mapping[str, Any]) -> str:
    variants = payload.get("variants", {})
    stable_variants = {
        key: {
            "label": summary.get("label"),
            "stage": summary.get("stage"),
            "strategy_family": summary.get("strategy_family"),
            "parameters": summary.get("parameters"),
            "config": summary.get("config"),
            "portfolio_caps": summary.get("portfolio_caps"),
        }
        for key, summary in sorted(variants.items())
    }
    score_cache = payload.get("replay_generated_score_cache", {})
    stable = {
        "mode": payload.get("mode"),
        "dataset_id": payload.get("dataset_id"),
        "official_trading_replay_sql_predicate": payload.get("official_trading_replay_sql_predicate"),
        "row_limit": payload.get("row_limit"),
        "total_official_rows": payload.get("total_official_rows"),
        "fee_model": payload.get("fee_model"),
        "weather_model_runner_metadata": payload.get("weather_model_runner", {}).get("metadata"),
        "pre_registered_pass_fail_criteria": payload.get("pre_registered_pass_fail_criteria"),
        "strategies_tested": payload.get("strategies_tested"),
        "staging_protocol": payload.get("staging_protocol"),
        "determinism_controls": payload.get("determinism_controls"),
        "code_hashes": payload.get("code_hashes"),
        "score_cache_hash_invalidation_metadata": {
            "record_count": score_cache.get("record_count"),
            "input_hashes_recorded": score_cache.get("input_hashes_recorded"),
            "prediction_hashes_recorded": score_cache.get("prediction_hashes_recorded"),
            "model_versions": score_cache.get("model_versions"),
            "model_code_hashes": score_cache.get("model_code_hashes"),
            "model_config_hashes": score_cache.get("model_config_hashes"),
        },
        "variant_specs": stable_variants,
    }
    return _hash_mapping(stable)


def _gen2_score_cache_hash_summary(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    semantic = hashlib.sha256()
    sequence = hashlib.sha256()
    count = 0
    volatile_fields = {"generated_at_utc", "prediction_timestamp_utc"}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            count += 1
            clean = {key: value for key, value in record.items() if key not in volatile_fields}
            semantic.update(json.dumps(clean, sort_keys=True, separators=(",", ":")).encode("utf-8"))
            semantic.update(b"\n")
            sequence.update(
                (
                    f"{record.get('universe_row_id')}|{record.get('decision_id')}|"
                    f"{record.get('input_hash')}|{record.get('prediction_hash')}\n"
                ).encode("utf-8")
            )
    return {
        "score_cache_path": str(path),
        "score_cache_raw_sha256": _file_sha256(path),
        "score_cache_records": count,
        "score_cache_semantic_sha256_excluding_run_timestamps": semantic.hexdigest(),
        "score_cache_input_prediction_sequence_sha256": sequence.hexdigest(),
    }


def _csv_data_row_count(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as handle:
        total = sum(1 for _ in handle)
    return max(total - 1, 0)


def _gen2_reproducibility_audit_markdown(payload: Mapping[str, Any]) -> str:
    repro = payload.get("reproducibility", {})
    verdict = payload.get("re_emitted_overall_verdict", {})
    comparisons = payload.get("comparisons", {})
    lines = [
        "# Gen-2 Reproducibility Audit",
        "",
        "Mode: deterministic replay artifact audit. No live trading, no model promotion, and no strategy promotion.",
        "",
        f"Runs audited: `{payload.get('run_count')}`",
        f"Independent runs reproduced every cited result: `{repro.get('independent_runs_reproduced_every_cited_result')}` / `{repro.get('required_independent_runs')}`",
        f"Ranked-grid CSV byte-identical: `{comparisons.get('ranked_grid_csv_byte_identical')}`",
        f"Full trade-ledger CSV byte-identical: `{comparisons.get('full_trade_ledger_csv_byte_identical')}`",
        f"Stable config fingerprint identical: `{comparisons.get('stable_config_fingerprint_identical')}`",
        f"Score-cache gzip byte-identical: `{comparisons.get('score_cache_gzip_byte_identical')}`",
        f"Score-cache semantic hash excluding run timestamps identical: `{comparisons.get('score_cache_semantic_hash_excluding_run_timestamps_identical')}`",
        f"Verdict status: `{verdict.get('status')}`",
        "",
        "Run Artifacts:",
    ]
    for run in payload.get("runs", []):
        lines.extend(
            [
                f"- `{run.get('run_json_path')}`",
                f"  - ranked: `{run.get('ranked_grid_csv_path')}` sha256 `{run.get('ranked_grid_csv_sha256')}`",
                f"  - ledger: `{run.get('trade_ledger_csv_path')}` sha256 `{run.get('trade_ledger_csv_sha256')}`",
            ]
        )
    lines.extend(
        [
            "",
            "Boundary: this audit only proves deterministic replay artifacts for the cited runs. It is not a live-trading recommendation or a model/strategy promotion.",
            "",
        ]
    )
    return "\n".join(lines)


def _gen2_cap_divergence_markdown(payload: Mapping[str, Any]) -> str:
    divergence = payload.get("first_divergence") or {}
    decision = divergence.get("decision", {}) if isinstance(divergence, Mapping) else {}
    permanent = divergence.get("permanent_policy", {}) if isinstance(divergence, Mapping) else {}
    expiring = divergence.get("expiring_policy", {}) if isinstance(divergence, Mapping) else {}
    permanent_result = permanent.get("result", {}) if isinstance(permanent, Mapping) else {}
    expiring_result = expiring.get("result", {}) if isinstance(expiring, Mapping) else {}
    lines = [
        "# Gen-2 Cap Divergence Diagnostic",
        "",
        "Mode: cap-layer diagnostic only. No order execution, no PnL accounting, no live trading.",
        "",
        f"Variant: `{payload.get('variant_key')}`",
        f"Rows scanned until stop: `{payload.get('total_rows_scanned_until_stop')}`",
        f"Cap decisions compared: `{payload.get('cap_decisions_compared')}`",
        f"First divergence found: `{payload.get('first_divergence_found')}`",
        f"Model outputs generated/present: `{payload.get('model_outputs_generated_or_present')}`",
        f"Missing model outputs: `{payload.get('missing_model_outputs')}`",
        f"Firewall violations: `{payload.get('model_input_firewall_violations')}`",
        "",
        "## First Divergence",
        "",
        f"- Cap decision index: `{divergence.get('cap_decision_index')}`",
        f"- Row index: `{divergence.get('row_index')}`",
        f"- Decision time: `{decision.get('decision_time_utc')}`",
        f"- City/date: `{decision.get('normalized_city')}` / `{decision.get('market_date_local')}`",
        f"- Bucket/side: `{decision.get('bucket_label')}` `{decision.get('bucket_type')}` `{decision.get('side')}`",
        f"- Expiring exposures released before check: `{divergence.get('released_expiring_exposures_before_check')}`",
        f"- Permanent policy passed: `{permanent_result.get('passed')}`, reasons `{permanent_result.get('skip_reasons')}`",
        f"- Expiring policy passed: `{expiring_result.get('passed')}`, reasons `{expiring_result.get('skip_reasons')}`",
        "",
        "## Boundary",
        "",
        "This diagnostic compares the old permanent-reservation cap semantics against the current expiring-reservation semantics for one frozen variant. It is not a strategy replay verdict.",
        "",
    ]
    return "\n".join(lines)


def _gen2_strategy_validation_markdown(payload: Mapping[str, Any]) -> str:
    verdict = payload.get("overall_verdict", {})
    runner = payload.get("weather_model_runner", {})
    score_cache = payload.get("replay_generated_score_cache", {})
    csvs = payload.get("csv_artifacts", {})
    reproducibility = payload.get("reproducibility", {})
    ranked = payload.get("ranked_variants", [])[:10]
    lines = [
        "# Gen-2 Strategy Validation",
        "",
        "Mode: fee-aware fixed-family validation. No live trading, no model promotion, and no live-setting promotion.",
        "",
        f"Dataset: `{payload.get('dataset_id')}`",
        f"Rows scanned: `{payload.get('total_official_rows')}`",
        f"Runtime seconds: `{payload.get('runtime_seconds')}`",
        f"Rows/sec: `{payload.get('rows_per_sec')}`",
        f"Weather runner: `{runner.get('metadata', {}).get('runner_name')}`",
        f"Fixture runner used: `{runner.get('fixture_runner_used')}`",
        f"Model outputs generated: `{runner.get('model_outputs_generated')}`",
        f"Missing model outputs: `{runner.get('missing_model_outputs')}`",
        f"Firewall violations: `{runner.get('model_input_firewall_violations')}`",
        f"Leakage violations: `{payload.get('leakage_validation', {}).get('violation_counts')}`",
        f"Fee model: `{payload.get('fee_model', {}).get('fee_model')}`",
        f"Fee rate: `{payload.get('fee_model', {}).get('fee_rate')}`",
        f"Score cache: `{score_cache.get('artifact_path')}`",
        f"Ranked grid CSV: `{csvs.get('ranked_grid_csv')}`",
        f"Full trade ledger CSV: `{csvs.get('trade_ledger')}`",
        f"Monthly/window CSV: `{csvs.get('monthly_window_pnl_csv')}`",
        f"Concentration CSV: `{csvs.get('concentration_csv')}`",
        "",
        "## Verdict",
        "",
        f"Overall status: `{verdict.get('status')}`",
        f"Live-candidate variants: `{len(verdict.get('live_candidate_variants', []))}`",
        f"Replay-gate pass variants pending reproducibility: `{len(verdict.get('replay_gate_pass_variants', []))}`",
        f"Research-only variants: `{len(verdict.get('research_only_variants', []))}`",
        f"Rejected variants: `{len(verdict.get('rejected_variants', []))}`",
        f"Independent runs reproduced every cited result: `{reproducibility.get('independent_runs_reproduced_every_cited_result')}` / `{reproducibility.get('required_independent_runs')}`",
        f"Reproducibility gate: `{reproducibility.get('gate_status')}`",
        "",
        "## Top Ranked Variants",
        "",
    ]
    if not ranked:
        lines.append("- None.")
    for row in ranked:
        lines.append(
            "- "
            f"`{row.get('rank_by_net_fee_adjusted_pnl')}` `{row.get('variant_key')}` "
            f"net `{row.get('net_fee_adjusted_pnl_usd')}`, trades `{row.get('trade_count')}`, "
            f"delayed `{row.get('delayed_entry_net_pnl_usd')}`, worse-fill `{row.get('worse_fill_net_pnl_usd')}`"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "Ranking is a validation report over the pre-registered grid, not a live recommendation. Replay-gate passes remain suspended until the required independent byte-identical reproducibility audit passes.",
            "",
        ]
    )
    return "\n".join(lines)


def write_fee_restatement_report(
    *,
    input_report_path: Path,
    output_dir: Path,
    fee_model: str = FEE_MODEL_POLYMARKET_WEATHER_TAKER,
    fee_rate: Optional[float] = None,
) -> dict[str, str]:
    input_report_path = Path(input_report_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = json.loads(input_report_path.read_text(encoding="utf-8"))
    fee_config = FeeModelConfig(fee_model=fee_model, fee_rate=fee_rate)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base_slug = input_report_path.stem.replace("-conservative-fv-edge-accounting-replay", "")
    json_path = output_dir / f"{timestamp}-{base_slug}-fee-restatement.json"
    md_path = output_dir / f"{timestamp}-{base_slug}-fee-restatement.md"
    restatement = _fee_restatement_payload(
        source_report_path=input_report_path,
        source_payload=payload,
        fee_config=fee_config,
    )
    json_path.write_text(json.dumps(restatement, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(_fee_restatement_markdown(restatement), encoding="utf-8")
    return {"json_path": str(json_path), "markdown_path": str(md_path)}


def _fee_restatement_payload(
    *,
    source_report_path: Path,
    source_payload: Mapping[str, Any],
    fee_config: FeeModelConfig,
) -> Mapping[str, Any]:
    accounting = source_payload.get("fixed_strategy_accounting", {})
    pnl = accounting.get("pnl", {})
    fill_counts = accounting.get("fill_counts", {})
    baseline_fill_count = int(fill_counts.get("actual_fills") or 0)
    trade_records, source_mode = _available_report_trade_records(accounting)
    restated_trades = [
        _fee_restatement_trade(trade, fee_config=fee_config)
        for trade in trade_records
    ]
    gross_pnl_available = sum(trade["gross_no_fee_realized_pnl_usd"] for trade in restated_trades)
    net_pnl_available = sum(trade["net_fee_adjusted_realized_pnl_usd"] for trade in restated_trades)
    fees_available = sum(trade["fee_usd"] for trade in restated_trades)
    gross_cost_available = sum(trade["gross_no_fee_cost_usd"] for trade in restated_trades)
    net_cost_available = sum(trade["net_cost_usd"] for trade in restated_trades)
    flips_available = sum(
        1
        for trade in restated_trades
        if trade["gross_no_fee_realized_pnl_usd"] > 1e-9
        and trade["net_fee_adjusted_realized_pnl_usd"] <= 1e-9
    )
    available_count = len(restated_trades)
    exact_full_restatement_possible = available_count == baseline_fill_count
    baseline_gross_pnl = _optional_float(
        pnl.get("gross_no_fee_realized_pnl_usd", pnl.get("realized_pnl_usd"))
    )
    baseline_gross_cost = _optional_float(
        pnl.get("gross_no_fee_total_cost_usd", pnl.get("total_cost_usd"))
    )
    order_notional = _optional_float(
        source_payload.get("baseline_settings", {}).get("order_notional_usd")
    )
    missing_trade_count = max(0, baseline_fill_count - available_count)
    max_missing_fee = None
    lower_bound_net_pnl = None
    if (
        not exact_full_restatement_possible
        and baseline_gross_pnl is not None
        and order_notional is not None
    ):
        max_missing_fee = missing_trade_count * order_notional * fee_config.resolved_fee_rate
        lower_bound_net_pnl = baseline_gross_pnl - fees_available - max_missing_fee
    return {
        "schema_version": BACKTEST_FRAMEWORK_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "fee_restatement_from_completed_accounting_report",
        "source_report_path": str(source_report_path),
        "source_trade_record_mode": source_mode,
        "exact_full_restatement_possible": exact_full_restatement_possible,
        "available_trade_records": available_count,
        "baseline_actual_fills": baseline_fill_count,
        "missing_trade_records": missing_trade_count,
        "fee_model": fee_config.as_report_dict(),
        "baseline_full_report_gross_no_fee_pnl_usd": baseline_gross_pnl,
        "baseline_full_report_gross_no_fee_cost_usd": baseline_gross_cost,
        "available_trade_restatement": {
            "gross_no_fee_cost_usd": gross_cost_available,
            "net_cost_usd": net_cost_available,
            "total_fees_usd": fees_available,
            "average_fee_per_available_trade_usd": (
                fees_available / available_count if available_count else None
            ),
            "gross_no_fee_realized_pnl_usd": gross_pnl_available,
            "net_fee_adjusted_realized_pnl_usd": net_pnl_available,
            "roi_before_fees_on_available_cost": (
                gross_pnl_available / gross_cost_available if gross_cost_available > 0 else None
            ),
            "roi_after_fees_on_available_cost": (
                net_pnl_available / net_cost_available if net_cost_available > 0 else None
            ),
            "trades_flipped_profitable_to_unprofitable_by_fees": flips_available,
        },
        "full_report_fee_survival_bound": {
            "is_bound_not_exact": not exact_full_restatement_possible,
            "max_missing_fee_usd": max_missing_fee,
            "net_pnl_lower_bound_after_available_and_max_missing_fees_usd": lower_bound_net_pnl,
            "previous_positive_result_survives_fee_upper_bound": (
                (lower_bound_net_pnl > 0)
                if lower_bound_net_pnl is not None
                else None
            ),
        },
        "restated_available_trades": restated_trades,
        "limitations": [
            "Does not rerun the weather model or strategy.",
            "Exact restatement requires all filled trade records in the source report.",
            "Older no-fee full report stores only selected best/worst trades, so this restatement is partial unless available_trade_records equals baseline_actual_fills.",
        ],
    }


def _available_report_trade_records(
    accounting: Mapping[str, Any],
) -> tuple[list[Mapping[str, Any]], str]:
    if accounting.get("all_trades"):
        return list(accounting.get("all_trades") or ()), "all_trades"
    records: dict[str, Mapping[str, Any]] = {}
    for section in ("best_trades", "worst_trades"):
        for trade in accounting.get(section, ()) or ():
            key = str(trade.get("decision_id") or trade.get("token_id") or len(records))
            records[key] = trade
    return list(records.values()), "best_and_worst_trades_only"


def _fee_restatement_trade(
    trade: Mapping[str, Any],
    *,
    fee_config: FeeModelConfig,
) -> Mapping[str, Any]:
    price = float(trade["fill_price"])
    shares = float(trade["shares"])
    gross_cost = _optional_float(trade.get("gross_no_fee_cost_usd"))
    if gross_cost is None:
        gross_cost = float(trade.get("cost_usd") or 0.0)
    payout = float(trade.get("payout_usd") or 0.0)
    fee = calculate_fill_fee_usd(shares=shares, price=price, fee_config=fee_config)
    net_cost = gross_cost + fee
    gross_pnl = payout - gross_cost
    net_pnl = payout - net_cost
    output = dict(trade)
    output.update(
        {
            "gross_no_fee_cost_usd": gross_cost,
            "fee_usd": fee,
            "fee_model": fee_config.fee_model,
            "fee_rate": fee_config.resolved_fee_rate,
            "net_cost_usd": net_cost,
            "gross_no_fee_realized_pnl_usd": gross_pnl,
            "net_fee_adjusted_realized_pnl_usd": net_pnl,
            "roi_before_fees_on_cost": gross_pnl / gross_cost if gross_cost > 0 else None,
            "roi_after_fees_on_cost": net_pnl / net_cost if net_cost > 0 else None,
            "fee_flipped_profitable_to_unprofitable": gross_pnl > 1e-9 and net_pnl <= 1e-9,
        }
    )
    return output


def _fee_restatement_markdown(payload: Mapping[str, Any]) -> str:
    available = payload.get("available_trade_restatement", {})
    bound = payload.get("full_report_fee_survival_bound", {})
    return "\n".join(
        [
            "# Fee Restatement",
            "",
            f"Source report: `{payload.get('source_report_path')}`",
            f"Fee model: `{payload.get('fee_model', {}).get('fee_model')}`",
            f"Fee rate: `{payload.get('fee_model', {}).get('fee_rate')}`",
            f"Exact full restatement possible: `{payload.get('exact_full_restatement_possible')}`",
            f"Available/baseline trades: `{payload.get('available_trade_records')}` / `{payload.get('baseline_actual_fills')}`",
            "",
            "## Available Trade Records",
            "",
            f"- Gross no-fee PnL: `{available.get('gross_no_fee_realized_pnl_usd')}`",
            f"- Net fee-adjusted PnL: `{available.get('net_fee_adjusted_realized_pnl_usd')}`",
            f"- Total fees: `{available.get('total_fees_usd')}`",
            f"- Average fee per available trade: `{available.get('average_fee_per_available_trade_usd')}`",
            f"- Fee flips: `{available.get('trades_flipped_profitable_to_unprofitable_by_fees')}`",
            "",
            "## Full Report Bound",
            "",
            f"- Baseline full gross no-fee PnL: `{payload.get('baseline_full_report_gross_no_fee_pnl_usd')}`",
            f"- Max missing fee: `{bound.get('max_missing_fee_usd')}`",
            f"- Net PnL lower bound: `{bound.get('net_pnl_lower_bound_after_available_and_max_missing_fees_usd')}`",
            f"- Survives upper-bound fees: `{bound.get('previous_positive_result_survives_fee_upper_bound')}`",
            "",
            "This is a fee restatement only. It does not rerun the weather model, optimize parameters, compare strategies, or promote any live setting.",
            "",
        ]
    )


def _weather_runner_slug(weather_runner: str) -> str:
    normalized = weather_runner.strip().lower().replace("_", "-")
    if normalized in {"fixture", "deterministic-fixture"}:
        return "fixture"
    if normalized in {"observed-high-guarded", "observed-high-guarded-shrinkage"}:
        return "observed-high-guarded"
    raise ValueError(f"Unsupported candidate weather runner: {weather_runner}")


def _build_candidate_weather_model_runner(
    weather_runner: str,
    *,
    work_root: Path,
    source_repo: Optional[Path] = None,
) -> WeatherModelRunner:
    runner_slug = _weather_runner_slug(weather_runner)
    if runner_slug == "fixture":
        return DeterministicFixtureWeatherModelRunner()
    if runner_slug == "observed-high-guarded":
        from .weather_model_runner import ObservedHighGuardedShrinkageWeatherModelRunner

        kwargs: dict[str, Any] = {"work_root": Path(work_root).resolve()}
        if source_repo is not None:
            kwargs["source_repo"] = Path(source_repo).expanduser().resolve()
        return ObservedHighGuardedShrinkageWeatherModelRunner(**kwargs)
    raise AssertionError(runner_slug)


def _validation_only_markdown(payload: Mapping[str, Any]) -> str:
    counts = payload.get("artifact_coverage", {})
    score = payload.get("score_contract_coverage", {})
    leakage = payload.get("leakage_validation", {}).get("violation_counts", {})
    lines = [
        "# Backtesting Validation-Only Dry Replay",
        "",
        "Mode: validation only. No trading replay, strategy PnL, parameter optimization, or profitability claims.",
        "",
        f"Dataset: `{payload.get('dataset_id')}`",
        f"Official predicate applied: `{payload.get('official_trading_replay_only')}`",
        f"Compact only: `{payload.get('compact_only')}`",
        f"Wide DB exists: `{payload.get('wide_price_db_exists')}`",
        "",
        "## Coverage",
        "",
        f"- Decision frames: `{score.get('decision_frames')}`",
        f"- Frames with quotes: `{counts.get('frames_with_quote')}`",
        f"- Frames with quote source cache path: `{counts.get('frames_with_quote_source_cache_path')}`",
        f"- Frames with canonical labels: `{counts.get('frames_with_canonical_label')}`",
        f"- Frames with observed-high snapshots: `{score.get('frames_with_observed_high_snapshot')}`",
        f"- Frames with shared scores: `{score.get('frames_with_shared_score')}`",
        "",
        "## Leakage Validation",
        "",
    ]
    if leakage:
        for key, value in sorted(leakage.items()):
            lines.append(f"- `{key}`: `{value}`")
    else:
        lines.append("- No leakage violations recorded by the validation-only pass.")
    lines.extend(["", "## Schema Gaps", ""])
    for gap in payload.get("schema_gaps", []):
        lines.append(f"- `{gap}`")
    lines.append("")
    return "\n".join(lines)


def _framework_contract_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Backtesting Framework Contracts",
        "",
        "Mode: implementation scaffold only. No strategy PnL backtests, parameter optimization, or profitability claims.",
        "",
        f"Schema version: `{payload['schema_version']}`",
        f"Strategy spec: `{payload['source_strategy_spec']}`",
        f"Canonical label artifact: `{payload['canonical_label_artifact']}`",
        "",
        "## Contract Groups",
        "",
        "- Shared side-aware score contract",
        "- Universal data, market, model, observed-high, and heat-window gates",
        "- Portfolio cap dimensions",
        "- Anti-overfit walk-forward and parameter-plateau schema",
        "- Validation-only report schema",
        "",
        "## Forbidden Until Strategy Selection",
        "",
    ]
    for section in VALIDATION_ONLY_REPORT_SCHEMA["forbidden_sections_until_strategy_selection"]:
        lines.append(f"- `{section}`")
    lines.extend(
        [
            "",
            "## Required Future Validation Sections",
            "",
        ]
    )
    for section in VALIDATION_ONLY_REPORT_SCHEMA["required_sections"]:
        lines.append(f"- `{section}`")
    lines.extend(
        [
            "",
            "## Required Future Anti-Overfit Controls",
            "",
        ]
    )
    for section in ANTI_OVERFIT_VALIDATION_SCHEMA["required_controls"]:
        lines.append(f"- `{section}`")
    lines.append("")
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="DailyWeather backtesting framework scaffolding")
    subparsers = parser.add_subparsers(dest="command", required=True)
    schema = subparsers.add_parser("schema-report", help="Write validation-only contract schemas")
    schema.add_argument("--work-root", default=".", help="Workspace root containing work/")
    schema.add_argument("--output-dir", default=None, help="Optional output directory")
    validation = subparsers.add_parser("validation-report", help="Write no-PnL compact validation-only dry replay report")
    validation.add_argument("--work-root", default=".", help="Workspace root containing work/")
    validation.add_argument("--output-dir", default=None, help="Optional output directory")
    validation.add_argument("--limit", type=int, default=0, help="Optional row limit for smoke validation")
    validation.add_argument(
        "--all-current-rows",
        action="store_true",
        help="Validate all current rows instead of applying the official trading replay predicate",
    )
    candidate = subparsers.add_parser(
        "conservative-fv-edge-report",
        help="Write a small Conservative FV Edge baseline candidate replay report",
    )
    candidate.add_argument("--work-root", default=".", help="Workspace root containing work/")
    candidate.add_argument("--output-dir", default=None, help="Optional output directory")
    candidate.add_argument("--limit", type=int, default=250, help="Small replay row limit")
    candidate.add_argument("--starting-cash-usd", type=float, default=100.0)
    candidate.add_argument(
        "--weather-runner",
        choices=("fixture", "observed-high-guarded"),
        default="fixture",
        help="Weather model runner used live-like at each decision time",
    )
    candidate.add_argument(
        "--source-repo",
        default=None,
        help="Optional old DailyWeather source checkout for the observed-high guarded runner",
    )
    accounting = subparsers.add_parser(
        "conservative-fv-edge-accounting-report",
        help="Write a fixed-parameter Conservative FV Edge accounting replay report",
    )
    accounting.add_argument("--work-root", default=".", help="Workspace root containing work/")
    accounting.add_argument("--output-dir", default=None, help="Optional output directory")
    accounting.add_argument("--limit", type=int, default=2500, help="Replay row limit")
    accounting.add_argument("--starting-cash-usd", type=float, default=100.0)
    accounting.add_argument(
        "--fee-model",
        choices=(FEE_MODEL_ZERO, FEE_MODEL_POLYMARKET_WEATHER_TAKER),
        default=DEFAULT_STRATEGY_REPLAY_FEE_MODEL,
        help=(
            "Execution fee model for simulated fills; defaults to polymarket_weather_taker "
            "for strategy accounting replays. Use --fee-model zero for explicit no-fee replay."
        ),
    )
    accounting.add_argument(
        "--fee-rate",
        type=float,
        default=DEFAULT_STRATEGY_REPLAY_FEE_RATE,
        help="Fee rate override; default is 0.05 for polymarket_weather_taker",
    )
    accounting.add_argument(
        "--weather-runner",
        choices=("fixture", "observed-high-guarded"),
        default="observed-high-guarded",
        help="Weather model runner used live-like at each decision time",
    )
    accounting.add_argument(
        "--source-repo",
        default=None,
        help="Optional old DailyWeather source checkout for the observed-high guarded runner",
    )
    throughput = subparsers.add_parser(
        "conservative-fv-edge-throughput-diagnostic",
        help="Write a diagnostic-only Conservative FV Edge gate throughput report",
    )
    throughput.add_argument("--work-root", default=".", help="Workspace root containing work/")
    throughput.add_argument("--output-dir", default=None, help="Optional output directory")
    throughput.add_argument("--limit", type=int, default=0, help="Optional row limit; default scans all official rows")
    throughput.add_argument("--starting-cash-usd", type=float, default=100.0)
    throughput.add_argument(
        "--fee-model",
        choices=(FEE_MODEL_ZERO, FEE_MODEL_POLYMARKET_WEATHER_TAKER),
        default=FEE_MODEL_POLYMARKET_WEATHER_TAKER,
        help="Execution fee model for diagnostic simulated fills",
    )
    throughput.add_argument(
        "--fee-rate",
        type=float,
        default=DEFAULT_POLYMARKET_WEATHER_TAKER_FEE_RATE,
        help="Fee rate override; default is 0.05 for polymarket_weather_taker",
    )
    throughput.add_argument(
        "--weather-runner",
        choices=("fixture", "observed-high-guarded"),
        default="observed-high-guarded",
        help="Weather model runner used live-like at each decision time",
    )
    throughput.add_argument(
        "--source-repo",
        default=None,
        help="Optional old DailyWeather source checkout for the observed-high guarded runner",
    )
    throughput.add_argument(
        "--progress-every",
        type=int,
        default=50000,
        help="Print progress every N rows during full diagnostic runs; 0 disables progress",
    )
    gen2 = subparsers.add_parser(
        "gen2-strategy-validation-report",
        help="Run staged fee-aware Gen-2 strategy validation over the pre-registered grid",
    )
    gen2.add_argument("--work-root", default=".", help="Workspace root containing work/")
    gen2.add_argument("--output-dir", default=None, help="Optional output directory")
    gen2.add_argument("--limit", type=int, default=0, help="Optional row limit; default scans all official rows")
    gen2.add_argument("--starting-cash-usd", type=float, default=100.0)
    gen2.add_argument(
        "--fee-model",
        choices=(FEE_MODEL_ZERO, FEE_MODEL_POLYMARKET_WEATHER_TAKER),
        default=DEFAULT_STRATEGY_REPLAY_FEE_MODEL,
        help="Execution fee model; defaults to polymarket_weather_taker. Use --fee-model zero only for explicit no-fee diagnostics.",
    )
    gen2.add_argument(
        "--fee-rate",
        type=float,
        default=DEFAULT_STRATEGY_REPLAY_FEE_RATE,
        help="Fee rate override; default is 0.05 for polymarket_weather_taker",
    )
    gen2.add_argument(
        "--weather-runner",
        choices=("fixture", "observed-high-guarded"),
        default="observed-high-guarded",
        help="Weather model runner used live-like at each decision time",
    )
    gen2.add_argument(
        "--source-repo",
        default=None,
        help="Optional old DailyWeather source checkout for the observed-high guarded runner",
    )
    gen2.add_argument(
        "--progress-every",
        type=int,
        default=50000,
        help="Print progress every N rows during full validation runs; 0 disables progress",
    )
    gen2_cap = subparsers.add_parser(
        "gen2-cap-divergence-report",
        help="Compare permanent versus expiring portfolio-cap semantics and stop at the first differing cap decision",
    )
    gen2_cap.add_argument("--work-root", default=".", help="Workspace root containing work/")
    gen2_cap.add_argument("--output-dir", default=None, help="Optional output directory")
    gen2_cap.add_argument(
        "--variant-key",
        default="stage_b_old_no_tail_no14_pos10_flat2_window_bankroll_bprice70",
        help="Frozen Gen-2 variant key to inspect",
    )
    gen2_cap.add_argument("--limit", type=int, default=0, help="Optional row limit; default scans until first divergence")
    gen2_cap.add_argument("--starting-cash-usd", type=float, default=100.0)
    gen2_cap.add_argument(
        "--weather-runner",
        choices=("fixture", "observed-high-guarded"),
        default="observed-high-guarded",
        help="Weather model runner used live-like at each decision time",
    )
    gen2_cap.add_argument(
        "--source-repo",
        default=None,
        help="Optional old DailyWeather source checkout for the observed-high guarded runner",
    )
    gen2_cap.add_argument(
        "--progress-every",
        type=int,
        default=50000,
        help="Print progress every N rows during cap divergence diagnostics; 0 disables progress",
    )
    gen2_audit = subparsers.add_parser(
        "gen2-reproducibility-audit",
        help="Audit completed Gen-2 runs for byte-identical ranked-grid and full-ledger outputs",
    )
    gen2_audit.add_argument(
        "--run-json",
        action="append",
        required=True,
        help="Completed gen2-strategy-validation JSON report; pass once per independent run",
    )
    gen2_audit.add_argument("--output-dir", default=None, help="Optional output directory")
    gen2_audit.add_argument(
        "--required-runs",
        type=int,
        default=GEN2_REPRODUCIBILITY_REQUIRED_RUNS,
        help="Required independent matching runs before re-emitting a live-candidate verdict",
    )
    shadow = subparsers.add_parser(
        "old-no-tail-live-forward-shadow-report",
        help="Write read-only live-forward shadow logs for frozen old NO-tail flat-$2 semantics",
    )
    shadow.add_argument("--work-root", default=".", help="Workspace root containing work/")
    shadow.add_argument("--output-dir", default=None, help="Optional output directory")
    shadow.add_argument("--limit", type=int, default=250, help="Decision row limit for one shadow logging run")
    shadow.add_argument("--starting-cash-usd", type=float, default=100.0)
    shadow.add_argument(
        "--weather-runner",
        choices=("fixture", "observed-high-guarded"),
        default="observed-high-guarded",
        help="Weather model runner used live-like at each decision time",
    )
    shadow.add_argument(
        "--source-repo",
        default=None,
        help="Optional old DailyWeather source checkout for the observed-high guarded runner",
    )
    shadow.add_argument(
        "--progress-every",
        type=int,
        default=0,
        help="Print progress every N rows during shadow logging; 0 disables progress",
    )
    restatement = subparsers.add_parser(
        "fee-restatement-report",
        help="Restate available trade records from a completed accounting report with a fee model",
    )
    restatement.add_argument("--input-report", required=True, help="Completed accounting report JSON")
    restatement.add_argument("--output-dir", default=None, help="Optional output directory")
    restatement.add_argument(
        "--fee-model",
        choices=(FEE_MODEL_ZERO, FEE_MODEL_POLYMARKET_WEATHER_TAKER),
        default=FEE_MODEL_POLYMARKET_WEATHER_TAKER,
        help="Execution fee model used for restatement",
    )
    restatement.add_argument(
        "--fee-rate",
        type=float,
        default=None,
        help="Optional fee rate override; defaults to 0.05 for polymarket_weather_taker",
    )
    runner_validation = subparsers.add_parser(
        "weather-model-runner-validation",
        help="Validate the research-only observed-high guarded weather runner adapter without strategy/PnL",
    )
    runner_validation.add_argument("--work-root", default=".", help="Workspace root containing work/")
    runner_validation.add_argument("--output-dir", default=None, help="Optional output directory")
    runner_validation.add_argument("--limit", type=int, default=25, help="Small adapter validation row limit")
    runner_validation.add_argument("--source-repo", default=None, help="Optional old DailyWeather source checkout")
    args = parser.parse_args(argv)
    if args.command == "schema-report":
        work_root = Path(args.work_root).resolve()
        output_dir = Path(args.output_dir).resolve() if args.output_dir else work_root / "work" / "reports" / "backtesting_engine"
        result = write_framework_contract_report(output_dir)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "validation-report":
        work_root = Path(args.work_root).resolve()
        output_dir = Path(args.output_dir).resolve() if args.output_dir else work_root / "work" / "reports" / "backtesting_engine"
        result = write_validation_only_report(
            work_root=work_root,
            output_dir=output_dir,
            limit=args.limit,
            official_trading_replay_only=not args.all_current_rows,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "conservative-fv-edge-report":
        work_root = Path(args.work_root).resolve()
        output_dir = Path(args.output_dir).resolve() if args.output_dir else work_root / "work" / "reports" / "backtesting_engine"
        source_repo = Path(args.source_repo).expanduser().resolve() if args.source_repo else None
        result = write_conservative_fv_edge_candidate_replay_report(
            work_root=work_root,
            output_dir=output_dir,
            limit=args.limit,
            starting_cash_usd=args.starting_cash_usd,
            weather_runner=args.weather_runner,
            source_repo=source_repo,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "conservative-fv-edge-accounting-report":
        work_root = Path(args.work_root).resolve()
        output_dir = Path(args.output_dir).resolve() if args.output_dir else work_root / "work" / "reports" / "backtesting_engine"
        source_repo = Path(args.source_repo).expanduser().resolve() if args.source_repo else None
        result = write_conservative_fv_edge_accounting_replay_report(
            work_root=work_root,
            output_dir=output_dir,
            limit=args.limit,
            starting_cash_usd=args.starting_cash_usd,
            weather_runner=args.weather_runner,
            fee_model=args.fee_model,
            fee_rate=args.fee_rate,
            source_repo=source_repo,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "conservative-fv-edge-throughput-diagnostic":
        work_root = Path(args.work_root).resolve()
        output_dir = Path(args.output_dir).resolve() if args.output_dir else work_root / "work" / "reports" / "backtesting_engine"
        source_repo = Path(args.source_repo).expanduser().resolve() if args.source_repo else None
        result = write_conservative_fv_edge_throughput_diagnostic_report(
            work_root=work_root,
            output_dir=output_dir,
            limit=args.limit,
            starting_cash_usd=args.starting_cash_usd,
            weather_runner=args.weather_runner,
            fee_model=args.fee_model,
            fee_rate=args.fee_rate,
            source_repo=source_repo,
            progress_every=args.progress_every,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "gen2-strategy-validation-report":
        work_root = Path(args.work_root).resolve()
        output_dir = Path(args.output_dir).resolve() if args.output_dir else work_root / "work" / "reports" / "backtesting_engine"
        source_repo = Path(args.source_repo).expanduser().resolve() if args.source_repo else None
        result = write_gen2_strategy_validation_report(
            work_root=work_root,
            output_dir=output_dir,
            limit=args.limit,
            starting_cash_usd=args.starting_cash_usd,
            weather_runner=args.weather_runner,
            fee_model=args.fee_model,
            fee_rate=args.fee_rate,
            source_repo=source_repo,
            progress_every=args.progress_every,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "gen2-cap-divergence-report":
        work_root = Path(args.work_root).resolve()
        output_dir = Path(args.output_dir).resolve() if args.output_dir else work_root / "work" / "reports" / "backtesting_engine"
        source_repo = Path(args.source_repo).expanduser().resolve() if args.source_repo else None
        result = write_gen2_cap_divergence_report(
            work_root=work_root,
            output_dir=output_dir,
            variant_key=args.variant_key,
            limit=args.limit,
            starting_cash_usd=args.starting_cash_usd,
            weather_runner=args.weather_runner,
            source_repo=source_repo,
            progress_every=args.progress_every,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "gen2-reproducibility-audit":
        output_dir = (
            Path(args.output_dir).resolve()
            if args.output_dir
            else Path.cwd().resolve() / "work" / "reports" / "backtesting_engine"
        )
        result = write_gen2_reproducibility_audit_report(
            run_json_paths=[Path(path).expanduser().resolve() for path in args.run_json],
            output_dir=output_dir,
            required_runs=args.required_runs,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "old-no-tail-live-forward-shadow-report":
        work_root = Path(args.work_root).resolve()
        output_dir = Path(args.output_dir).resolve() if args.output_dir else work_root / "work" / "reports" / "live_forward_shadow"
        source_repo = Path(args.source_repo).expanduser().resolve() if args.source_repo else None
        result = write_old_no_tail_live_forward_shadow_report(
            work_root=work_root,
            output_dir=output_dir,
            limit=args.limit,
            starting_cash_usd=args.starting_cash_usd,
            weather_runner=args.weather_runner,
            source_repo=source_repo,
            progress_every=args.progress_every,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "fee-restatement-report":
        input_report = Path(args.input_report).expanduser().resolve()
        output_dir = (
            Path(args.output_dir).resolve()
            if args.output_dir
            else input_report.parent
        )
        result = write_fee_restatement_report(
            input_report_path=input_report,
            output_dir=output_dir,
            fee_model=args.fee_model,
            fee_rate=args.fee_rate,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "weather-model-runner-validation":
        work_root = Path(args.work_root).resolve()
        output_dir = Path(args.output_dir).resolve() if args.output_dir else work_root / "work" / "reports" / "backtesting_engine"
        source_repo = Path(args.source_repo).expanduser().resolve() if args.source_repo else None
        result = write_weather_model_runner_adapter_validation_report(
            work_root=work_root,
            output_dir=output_dir,
            limit=args.limit,
            source_repo=source_repo,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    raise AssertionError(args.command)


def _sqlite_relation_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        """
        SELECT 1
        FROM sqlite_master
        WHERE name = ?
          AND type IN ('table', 'view')
        """,
        (name,),
    ).fetchone()
    return row is not None


def _sqlite_columns(conn: sqlite3.Connection, relation: str) -> set[str]:
    return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({relation})")}


def _require_columns(columns: set[str], required: set[str], relation: str) -> None:
    missing = sorted(required - columns)
    if missing:
        raise RuntimeError(f"{relation} is missing required columns: {', '.join(missing)}")


def _score_from_enriched_row(
    row: Mapping[str, Any],
    decision: DecisionSnapshot,
) -> Optional[SharedScoreContract]:
    if row.get("p_yes") in (None, "") or row.get("p_side") in (None, ""):
        return None
    observed_high = decision.observed_high
    heat_window = None
    heat_window_score = _optional_float(row.get("heat_window_exhaustion_score"))
    local_hour = _optional_float(row.get("decision_local_hour") or row.get("local_hour"))
    if local_hour is not None or heat_window_score is not None:
        heat_window = HeatWindowSnapshot(
            local_hour=local_hour if local_hour is not None else 0.0,
            forecast_peak_hour_local=_optional_float(row.get("forecast_peak_hour_local")),
            remaining_hourly_forecast_max_f=_optional_float(row.get("remaining_hourly_forecast_max_f")),
            heat_window_exhaustion_score=heat_window_score,
            solar_radiation_state=row.get("solar_radiation_state"),
            cloud_cover_state=row.get("cloud_cover_state"),
            wind_state=row.get("wind_state"),
            season=row.get("season"),
        )
    return SharedScoreContract(
        p_yes=float(row.get("p_yes")),
        side=decision.side,
        p_side=float(row.get("p_side")),
        model_agreement=_optional_float(row.get("model_agreement")) or 0.0,
        model_count=_optional_int(row.get("model_count")) or 0,
        probability_stdev=_optional_float(row.get("probability_stdev")),
        per_source_probabilities=_json_mapping(row.get("per_source_probabilities_json")),
        remaining_exceed_probability=_optional_float(row.get("remaining_exceed_probability")),
        observed_high_aware_confidence=_optional_float(row.get("observed_high_aware_confidence")),
        observed_high=observed_high,
        heat_window=heat_window,
        parser_confidence=_optional_float(row.get("parser_confidence") or decision.parser_confidence),
        settlement_unit=row.get("settlement_unit"),
        settlement_precision=row.get("settlement_precision"),
        scored_at_utc=_parse_utc(row.get("scored_at_utc")),
        score_model_version=row.get("score_model_version"),
        score_family=row.get("score_family"),
        metadata=_json_mapping(row.get("score_metadata_json")),
    )


def _strategy_skip(reason: SkipReason) -> StrategyDecision:
    return StrategyDecision(metadata={"candidate_skip_reasons": [reason.value]})


def _is_bounded_bucket(bucket_type: Optional[str]) -> bool:
    return (bucket_type or "").lower() in {"exact", "range"}


def _metadata_skip_reasons(metadata: Mapping[str, Any]) -> tuple[SkipReason, ...]:
    raw = metadata.get("candidate_skip_reasons", ())
    reasons: list[SkipReason] = []
    if isinstance(raw, str):
        raw = (raw,)
    for value in raw:
        try:
            reasons.append(SkipReason(str(value)))
        except ValueError:
            continue
    return tuple(dict.fromkeys(reasons))


def _frame_skip_reasons_with_leakage(
    skip_reasons: Sequence[SkipReason],
    violations: Sequence[LeakageViolation],
) -> tuple[SkipReason, ...]:
    reasons = list(skip_reasons)
    if violations and SkipReason.LEAKAGE_VIOLATION not in reasons:
        reasons.append(SkipReason.LEAKAGE_VIOLATION)
    return tuple(dict.fromkeys(reasons))


def _trade_record_from_fill_and_settlement(
    *,
    frame: ReplayFrame,
    order: OrderIntent,
    fill: Fill,
    settlement: Settlement,
) -> TradeAccountingRecord:
    decision = frame.decision
    quote = decision.quote
    score = frame.score
    gross_cost = fill.notional_usd
    fee = fill.fee_usd
    cost = gross_cost + fee
    payout = settlement.cash_delta_usd
    gross_realized_pnl = payout - gross_cost
    realized_pnl = payout - cost
    midpoint_cost = None
    midpoint_pnl = None
    if quote and quote.mid is not None and quote.mid > 0:
        midpoint_cost = quote.mid * fill.shares
        midpoint_pnl = payout - midpoint_cost
    return TradeAccountingRecord(
        decision_id=decision.decision_id,
        universe_row_id=decision.universe_row_id,
        market_id=decision.market_id,
        slug=decision.slug,
        question=decision.question,
        normalized_city=decision.normalized_city,
        country_region=decision.country_region,
        timezone=decision.timezone,
        market_date_local=decision.market_date_local,
        bucket_type=decision.bucket_type,
        bucket_label=decision.bucket_label,
        side=order.side.upper(),
        token_id=order.token_id,
        lead_days=decision.lead_days,
        decision_hour_utc=decision.decision_hour_utc,
        decision_time_utc=decision.decision_time_utc,
        fill_timestamp_utc=fill.timestamp_utc,
        fill_price=fill.price,
        shares=fill.shares,
        gross_no_fee_cost_usd=gross_cost,
        fee_usd=fee,
        fee_model=fill.fee_model,
        fee_rate=fill.fee_rate,
        liquidity_role=fill.liquidity_role,
        cost_usd=cost,
        payout_per_share=settlement.payout_per_share,
        payout_usd=payout,
        gross_no_fee_realized_pnl_usd=gross_realized_pnl,
        realized_pnl_usd=realized_pnl,
        net_fee_adjusted_realized_pnl_usd=realized_pnl,
        gross_no_fee_roi_on_cost=(gross_realized_pnl / gross_cost) if gross_cost > 0 else None,
        roi_on_cost=(realized_pnl / cost) if cost > 0 else None,
        net_fee_adjusted_roi_on_cost=(realized_pnl / cost) if cost > 0 else None,
        gross_no_fee_outcome=(
            "win" if gross_realized_pnl > 1e-9 else "loss" if gross_realized_pnl < -1e-9 else "flat"
        ),
        outcome="win" if realized_pnl > 1e-9 else "loss" if realized_pnl < -1e-9 else "flat",
        quote_mid=quote.mid if quote else None,
        market_midpoint_cost_usd=midpoint_cost,
        market_midpoint_realized_pnl_usd=midpoint_pnl,
        quote_source_payload_id=quote.source_payload_id if quote else None,
        quote_source_cache_path=quote.source_cache_path if quote else None,
        label_source=settlement.label_source,
        score_p_yes=score.p_yes if score else None,
        score_p_side=score.p_side if score else None,
        score_model_version=score.score_model_version if score else None,
    )


def _fixed_strategy_accounting_report(
    *,
    trade_records: Sequence[TradeAccountingRecord],
    fill_status_counts: Counter[str],
    order_intents: Sequence[OrderIntent],
    ledger: ReplayLedger,
    starting_cash_usd: float,
    fee_config: FeeModelConfig = FeeModelConfig(),
) -> Mapping[str, Any]:
    gross_no_fee_total_cost = sum(record.gross_no_fee_cost_usd for record in trade_records)
    total_fees = sum(record.fee_usd for record in trade_records)
    total_cost = sum(record.cost_usd for record in trade_records)
    total_payout = sum(record.payout_usd for record in trade_records)
    gross_no_fee_realized_pnl = sum(
        record.gross_no_fee_realized_pnl_usd for record in trade_records
    )
    realized_pnl = sum(record.net_fee_adjusted_realized_pnl_usd for record in trade_records)
    wins = sum(1 for record in trade_records if record.outcome == "win")
    losses = sum(1 for record in trade_records if record.outcome == "loss")
    flats = sum(1 for record in trade_records if record.outcome == "flat")
    gross_wins = sum(1 for record in trade_records if record.gross_no_fee_outcome == "win")
    gross_losses = sum(1 for record in trade_records if record.gross_no_fee_outcome == "loss")
    gross_flats = sum(1 for record in trade_records if record.gross_no_fee_outcome == "flat")
    fee_flips = sum(
        1
        for record in trade_records
        if record.gross_no_fee_realized_pnl_usd > 1e-9
        and record.net_fee_adjusted_realized_pnl_usd <= 1e-9
    )
    attempted_fills = sum(fill_status_counts.values())
    actual_fills = fill_status_counts.get("filled", 0) + fill_status_counts.get("partial_filled", 0)
    skipped_fills = attempted_fills - actual_fills
    cumulative_pnl = 0.0
    peak_pnl = 0.0
    max_drawdown = 0.0
    for record in trade_records:
        cumulative_pnl += record.realized_pnl_usd
        peak_pnl = max(peak_pnl, cumulative_pnl)
        max_drawdown = max(max_drawdown, peak_pnl - cumulative_pnl)
    midpoint_records = [
        record for record in trade_records if record.market_midpoint_realized_pnl_usd is not None
    ]
    midpoint_total_cost = sum(record.market_midpoint_cost_usd or 0.0 for record in midpoint_records)
    midpoint_total_pnl = sum(record.market_midpoint_realized_pnl_usd or 0.0 for record in midpoint_records)
    sorted_best = sorted(trade_records, key=lambda record: record.realized_pnl_usd, reverse=True)
    sorted_worst = sorted(trade_records, key=lambda record: record.realized_pnl_usd)
    return {
        "accounting_complete": True,
        "fixed_strategy_replay_only": True,
        "no_parameter_search": True,
        "no_strategy_ranking": True,
        "no_live_trading_recommendation": True,
        "settlement_labels": "canonical_polymarket_telonex_labels_only",
        "execution_policy": {
            "buy_price_source": "ask",
            "sell_price_source": "bid",
            "partial_fills_enabled": False,
            "missing_ask_policy": "skip_fill",
            "visible_ask_size_policy": "skip_fill_when_visible_size_below_requested_shares",
            "fee_model": fee_config.as_report_dict(),
            "order_latency_model": "not_modeled",
            "order_latency_stress_tested": False,
            "order_latency_blocker": (
                "delayed-entry replay requires a next-quote-after-delay lookup; "
                "current accounting uses the decision quote snapshot"
            ),
        },
        "fill_counts": {
            "order_intents": len(order_intents),
            "attempted_fills": attempted_fills,
            "actual_fills": actual_fills,
            "skipped_fills": skipped_fills,
            "fill_status_counts": dict(sorted(fill_status_counts.items())),
        },
        "cash": {
            "starting_cash_usd": starting_cash_usd,
            "ending_cash_usd": ledger.cash_usd,
            "cash_delta_usd": ledger.cash_usd - starting_cash_usd,
        },
        "pnl": {
            "gross_no_fee_total_cost_usd": gross_no_fee_total_cost,
            "gross_no_fee_realized_pnl_usd": gross_no_fee_realized_pnl,
            "gross_no_fee_roi_on_deployed_capital": (
                (gross_no_fee_realized_pnl / gross_no_fee_total_cost)
                if gross_no_fee_total_cost > 0
                else None
            ),
            "total_fees_usd": total_fees,
            "average_fee_per_trade_usd": (
                total_fees / len(trade_records) if trade_records else None
            ),
            "total_cost_usd": total_cost,
            "total_payout_usd": total_payout,
            "realized_pnl_usd": realized_pnl,
            "net_fee_adjusted_realized_pnl_usd": realized_pnl,
            "roi_on_deployed_capital": (realized_pnl / total_cost) if total_cost > 0 else None,
            "net_fee_adjusted_roi_on_deployed_capital": (
                (realized_pnl / total_cost) if total_cost > 0 else None
            ),
            "roi_before_fees_on_deployed_capital": (
                (gross_no_fee_realized_pnl / gross_no_fee_total_cost)
                if gross_no_fee_total_cost > 0
                else None
            ),
            "roi_after_fees_on_deployed_capital": (
                (realized_pnl / total_cost) if total_cost > 0 else None
            ),
            "fee_impact_usd": gross_no_fee_realized_pnl - realized_pnl,
            "trades_flipped_profitable_to_unprofitable_by_fees": fee_flips,
            "gross_no_fee_win_count": gross_wins,
            "gross_no_fee_loss_count": gross_losses,
            "gross_no_fee_flat_count": gross_flats,
            "win_count": wins,
            "loss_count": losses,
            "flat_count": flats,
            "open_position_count": sum(
                1 for position in ledger.positions.values() if position.shares > 1e-9
            ),
            "open_position_cost_usd": sum(
                position.cost_usd for position in ledger.positions.values() if position.shares > 1e-9
            ),
            "max_drawdown_usd": max_drawdown,
            "drawdown_basis": "realized_pnl_after_each_resolved_trade_no_intraday_mark_to_market",
        },
        "pnl_by_city": _trade_slice_report(trade_records, lambda record: record.normalized_city or "missing"),
        "pnl_by_side": _trade_slice_report(trade_records, lambda record: record.side or "missing"),
        "pnl_by_bucket_type": _trade_slice_report(trade_records, lambda record: record.bucket_type or "missing"),
        "pnl_by_lead_day": _trade_slice_report(trade_records, lambda record: str(record.lead_days)),
        "pnl_by_decision_hour_utc": _trade_slice_report(
            trade_records,
            lambda record: str(record.decision_hour_utc),
        ),
        "pnl_by_month": _trade_slice_report(trade_records, _trade_market_month_key),
        "pnl_by_week": _trade_slice_report(trade_records, _trade_market_week_key),
        "concentration": _trade_concentration_report(trade_records),
        "all_trades": [record.to_report_dict() for record in trade_records],
        "best_trades": [record.to_report_dict() for record in sorted_best[:10]],
        "worst_trades": [record.to_report_dict() for record in sorted_worst[:10]],
        "market_midpoint_diagnostic": {
            "diagnostic_only_not_optimization": True,
            "matched_trade_count": len(midpoint_records),
            "hypothetical_midpoint_cost_usd": midpoint_total_cost,
            "hypothetical_midpoint_realized_pnl_usd": midpoint_total_pnl,
            "actual_ask_gross_no_fee_cost_usd_for_same_trades": sum(
                record.gross_no_fee_cost_usd for record in midpoint_records
            ),
            "actual_fee_usd_for_same_trades": sum(record.fee_usd for record in midpoint_records),
            "actual_ask_cost_usd_for_same_trades": sum(record.cost_usd for record in midpoint_records),
            "actual_gross_no_fee_realized_pnl_usd_for_same_trades": sum(
                record.gross_no_fee_realized_pnl_usd for record in midpoint_records
            ),
            "actual_net_fee_adjusted_realized_pnl_usd_for_same_trades": sum(
                record.realized_pnl_usd for record in midpoint_records
            ),
        },
        "assumptions_and_limitations": [
            "fixed Conservative FV Edge baseline parameters only",
            "research-only observed-high guarded weather runner",
            "canonical settlement labels used only after model scoring for payout/accounting",
            "positions settle immediately against known historical canonical labels for accounting",
            "no intraday mark-to-market drawdown",
            "no fees beyond configured zero-fee replay assumption",
            "no parameter optimization or strategy comparison",
            "not a live-trading recommendation",
        ],
    }


def _trade_slice_report(
    trade_records: Sequence[TradeAccountingRecord],
    key_fn: Any,
) -> Mapping[str, Mapping[str, Any]]:
    buckets: dict[str, list[TradeAccountingRecord]] = defaultdict(list)
    for record in trade_records:
        buckets[str(key_fn(record))].append(record)
    output: dict[str, Mapping[str, Any]] = {}
    for key, records in sorted(buckets.items()):
        gross_cost = sum(record.gross_no_fee_cost_usd for record in records)
        fees = sum(record.fee_usd for record in records)
        cost = sum(record.cost_usd for record in records)
        payout = sum(record.payout_usd for record in records)
        gross_pnl = sum(record.gross_no_fee_realized_pnl_usd for record in records)
        pnl = sum(record.realized_pnl_usd for record in records)
        output[key] = {
            "trades": len(records),
            "gross_no_fee_cost_usd": gross_cost,
            "fees_usd": fees,
            "cost_usd": cost,
            "payout_usd": payout,
            "gross_no_fee_realized_pnl_usd": gross_pnl,
            "realized_pnl_usd": pnl,
            "net_fee_adjusted_realized_pnl_usd": pnl,
            "gross_no_fee_roi_on_deployed_capital": (
                (gross_pnl / gross_cost) if gross_cost > 0 else None
            ),
            "roi_on_deployed_capital": (pnl / cost) if cost > 0 else None,
            "net_fee_adjusted_roi_on_deployed_capital": (
                (pnl / cost) if cost > 0 else None
            ),
            "wins": sum(1 for record in records if record.outcome == "win"),
            "losses": sum(1 for record in records if record.outcome == "loss"),
            "flats": sum(1 for record in records if record.outcome == "flat"),
        }
    return output


def _trade_market_month_key(record: TradeAccountingRecord) -> str:
    if record.market_date_local:
        return str(record.market_date_local)[:7]
    return record.decision_time_utc.strftime("%Y-%m")


def _trade_market_week_key(record: TradeAccountingRecord) -> str:
    if record.market_date_local:
        try:
            iso = datetime.fromisoformat(str(record.market_date_local)).date().isocalendar()
            return f"{iso.year}-W{iso.week:02d}"
        except ValueError:
            return "missing"
    iso = record.decision_time_utc.date().isocalendar()
    return f"{iso.year}-W{iso.week:02d}"


def _trade_market_date_key(record: TradeAccountingRecord) -> str:
    return str(record.market_date_local or "missing")


def _trade_city_date_key(record: TradeAccountingRecord) -> str:
    return f"{record.normalized_city or 'missing'}|{record.market_date_local or 'missing'}"


def _top_group_contribution(
    trade_records: Sequence[TradeAccountingRecord],
    key_fn: Any,
) -> Optional[Mapping[str, Any]]:
    grouped = _trade_slice_report(trade_records, key_fn)
    if not grouped:
        return None
    key, summary = sorted(
        grouped.items(),
        key=lambda item: (-abs(float(item[1]["realized_pnl_usd"])), item[0]),
    )[0]
    return {"key": key, **summary}


def _trade_concentration_report(
    trade_records: Sequence[TradeAccountingRecord],
) -> Mapping[str, Any]:
    if not trade_records:
        return {
            "basis": "absolute_realized_pnl_usd",
            "top_city_by_absolute_pnl": None,
            "top_market_date_by_absolute_pnl": None,
            "top_city_date_by_absolute_pnl": None,
            "top_trade_by_absolute_pnl": None,
            "top_positive_trade": None,
            "top_negative_trade": None,
        }
    top_absolute_trade = sorted(
        trade_records,
        key=lambda record: (-abs(record.realized_pnl_usd), record.decision_id),
    )[0]
    top_positive_trade = sorted(
        trade_records,
        key=lambda record: (-record.realized_pnl_usd, record.decision_id),
    )[0]
    top_negative_trade = sorted(
        trade_records,
        key=lambda record: (record.realized_pnl_usd, record.decision_id),
    )[0]
    return {
        "basis": "absolute_realized_pnl_usd",
        "top_city_by_absolute_pnl": _top_group_contribution(
            trade_records,
            lambda record: record.normalized_city or "missing",
        ),
        "top_market_date_by_absolute_pnl": _top_group_contribution(
            trade_records,
            _trade_market_date_key,
        ),
        "top_city_date_by_absolute_pnl": _top_group_contribution(
            trade_records,
            _trade_city_date_key,
        ),
        "top_trade_by_absolute_pnl": top_absolute_trade.to_report_dict(),
        "top_positive_trade": top_positive_trade.to_report_dict(),
        "top_negative_trade": top_negative_trade.to_report_dict(),
    }


def _frame_with_replay_generated_score(
    frame: ReplayFrame,
    *,
    model_runner: Optional[WeatherModelRunner],
    score_cache: ReplayGeneratedScoreCache,
    point_policy: PointInTimePolicy,
    builder: CandidateReplayReportBuilder,
) -> ReplayFrame:
    if frame.score is not None or model_runner is None:
        return ReplayFrame(
            decision=frame.decision,
            canonical_label=frame.canonical_label,
            score=frame.score,
            skip_reasons=point_policy.skip_reasons(
                frame.decision,
                frame.canonical_label,
                frame.score,
            ),
        )
    try:
        context = WeatherModelDecisionContext.from_decision(frame.decision)
        validate_weather_model_context_firewall(context)
        prediction = model_runner.predict(context)
    except ValueError:
        builder.add_model_input_firewall_violation()
        reasons = list(point_policy.skip_reasons(frame.decision, frame.canonical_label, None))
        reasons.append(SkipReason.MODEL_INPUT_FIREWALL_VIOLATION)
        return ReplayFrame(
            decision=frame.decision,
            canonical_label=frame.canonical_label,
            score=None,
            skip_reasons=tuple(dict.fromkeys(reasons)),
        )
    if prediction is None:
        builder.add_model_output_missing()
        reasons = list(point_policy.skip_reasons(frame.decision, frame.canonical_label, None))
        reasons.append(SkipReason.MISSING_MODEL_OUTPUT)
        return ReplayFrame(
            decision=frame.decision,
            canonical_label=frame.canonical_label,
            score=None,
            skip_reasons=tuple(dict.fromkeys(reasons)),
        )
    score = prediction.to_shared_score(
        side=frame.decision.side,
        observed_high=_model_visible_observed_high(frame.decision),
        parser_confidence=frame.decision.parser_confidence,
        settlement_unit=frame.canonical_label.settlement_unit if frame.canonical_label else None,
        settlement_precision=frame.canonical_label.settlement_precision if frame.canonical_label else None,
    )
    score_cache.add(context=context, prediction=prediction, score=score)
    builder.add_generated_score()
    return ReplayFrame(
        decision=frame.decision,
        canonical_label=frame.canonical_label,
        score=score,
        skip_reasons=point_policy.skip_reasons(frame.decision, frame.canonical_label, score),
    )


def _model_visible_observed_high(decision: DecisionSnapshot) -> Optional[ObservedHighSnapshot]:
    observed = decision.observed_high
    if observed is None:
        return None
    if observed.observed_high_timestamp_utc > decision.decision_time_utc:
        return None
    return observed


def _exposure_intent_from_order(order: OrderIntent, decision: DecisionSnapshot) -> ExposureIntent:
    notional = float(order.requested_notional_usd or 0.0)
    if notional <= 0.0 and order.requested_shares is not None:
        quote = decision.quote
        price = quote.ask if quote and order.action == OrderAction.BUY else quote.bid if quote else None
        notional = float(order.requested_shares) * float(price or order.limit_price or 0.0)
    return ExposureIntent(
        token_id=order.token_id,
        side=order.side,
        city=decision.normalized_city or "missing_city",
        market_date_local=decision.market_date_local or "missing_market_date",
        region=decision.country_region or "missing_region",
        settlement_source_family=decision.settlement_source or "missing_settlement_source",
        decision_hour_utc=decision.decision_hour_utc if decision.decision_hour_utc is not None else -1,
        notional_usd=notional,
        primary_exposure=True,
        release_time_utc=_exposure_release_time_utc(decision),
    )


def _exposure_release_time_utc(decision: DecisionSnapshot) -> Optional[datetime]:
    if not decision.market_date_local:
        return None
    try:
        target_date = datetime.fromisoformat(str(decision.market_date_local)).date()
    except ValueError:
        return None
    release_date = target_date + timedelta(days=1)
    try:
        zone = ZoneInfo(decision.timezone or "UTC")
    except Exception:
        zone = timezone.utc
    local_release = datetime(
        release_date.year,
        release_date.month,
        release_date.day,
        12,
        0,
        0,
        tzinfo=zone,
    )
    return local_release.astimezone(timezone.utc)


def simulate_order_fill(
    order: OrderIntent,
    decision: DecisionSnapshot,
    *,
    fill_policy: FillPolicy = FillPolicy(),
    fee_config: FeeModelConfig = FeeModelConfig(),
    fill_timestamp_utc: Optional[datetime] = None,
) -> FillResult:
    quote = decision.quote
    if quote is None:
        return FillResult(
            fill=None,
            status="missing_quote",
            skip_reason=SkipReason.MISSING_QUOTE_SNAPSHOT,
            requested_shares=order.requested_shares,
        )
    if order.action == OrderAction.BUY:
        price = quote.ask
        visible_size = quote.ask_size
        price_source = "ask"
    elif order.action == OrderAction.SELL:
        price = quote.bid
        visible_size = quote.bid_size
        price_source = "bid"
    else:
        return FillResult(
            fill=None,
            status="unsupported_order_action",
            skip_reason=SkipReason.MISSING_EXECUTABLE_PRICE,
            requested_shares=order.requested_shares,
        )
    if price is None or price <= 0:
        return FillResult(
            fill=None,
            status="missing_executable_price",
            skip_reason=SkipReason.MISSING_EXECUTABLE_PRICE,
            requested_shares=order.requested_shares,
            visible_size=visible_size,
            price_source=price_source,
        )
    requested_shares = order.requested_shares
    if requested_shares is None:
        if order.requested_notional_usd is None:
            return FillResult(
                fill=None,
                status="missing_requested_size",
                skip_reason=SkipReason.MISSING_EXECUTABLE_PRICE,
                requested_shares=None,
                visible_size=visible_size,
                price_source=price_source,
            )
        requested_shares = order.requested_notional_usd / price
    filled_shares = requested_shares
    if visible_size is not None and visible_size < requested_shares:
        if not fill_policy.allow_partial:
            return FillResult(
                fill=None,
                status="insufficient_visible_size",
                skip_reason=SkipReason.INSUFFICIENT_VISIBLE_SIZE,
                requested_shares=requested_shares,
                visible_size=visible_size,
                price_source=price_source,
            )
        filled_shares = max(0.0, visible_size)
    if filled_shares <= fill_policy.min_fill_shares:
        return FillResult(
            fill=None,
            status="partial_fill_below_minimum",
            skip_reason=SkipReason.INSUFFICIENT_VISIBLE_SIZE,
            requested_shares=requested_shares,
            filled_shares=filled_shares,
            visible_size=visible_size,
            price_source=price_source,
        )
    fee_usd = calculate_fill_fee_usd(
        shares=filled_shares,
        price=price,
        fee_config=fee_config,
    )
    fill = Fill(
        token_id=order.token_id,
        side=order.side,
        action=order.action,
        timestamp_utc=_ensure_utc(fill_timestamp_utc) if fill_timestamp_utc else decision.decision_time_utc,
        price=price,
        shares=filled_shares,
        fee_usd=fee_usd,
        fee_model=fee_config.fee_model,
        fee_rate=fee_config.resolved_fee_rate,
        liquidity_role=(
            "taker"
            if fee_config.fee_model == FEE_MODEL_POLYMARKET_WEATHER_TAKER
            else "none"
        ),
        decision_id=order.decision_id,
    )
    return FillResult(
        fill=fill,
        status="filled" if filled_shares == requested_shares else "partial_filled",
        skip_reason=None,
        requested_shares=requested_shares,
        filled_shares=filled_shares,
        visible_size=visible_size,
        price_source=price_source,
    )


def _fill_order_from_quote(order: OrderIntent, decision: DecisionSnapshot) -> Optional[Fill]:
    return simulate_order_fill(order, decision).fill


def _visible_size_covers(fill: Fill, decision: DecisionSnapshot) -> bool:
    result = simulate_order_fill(
        OrderIntent(
            strategy_name="size_check",
            decision_id=fill.decision_id or "",
            token_id=fill.token_id,
            side=fill.side,
            action=fill.action,
            requested_shares=fill.shares,
        ),
        decision,
    )
    return result.fill is not None


def _violation(decision: DecisionSnapshot, code: str, details: str) -> LeakageViolation:
    return LeakageViolation(
        code=code,
        decision_id=decision.decision_id,
        universe_row_id=decision.universe_row_id,
        details=details,
    )


def _json_ready(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {
            str(key): _json_ready(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value


def _hash_mapping(value: Mapping[str, Any]) -> str:
    payload = json.dumps(_json_ready(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_utc(value: Any) -> Optional[datetime]:
    if value in (None, ""):
        return None
    text = str(value)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _require_utc(value: Any, field_name: str) -> datetime:
    parsed = _parse_utc(value)
    if parsed is None:
        raise ValueError(f"Missing required datetime field: {field_name}")
    return parsed


def _quote_snapshot_from_compact_row(row: Mapping[str, Any]) -> Optional[QuoteSnapshot]:
    timestamp = _parse_utc(row.get("timestamp_utc"))
    if timestamp is None:
        return None
    return QuoteSnapshot(
        timestamp_utc=timestamp,
        bid=_optional_float(row.get("bid")),
        ask=_optional_float(row.get("ask")),
        mid=_optional_float(row.get("mid")),
        last_price=_optional_float(row.get("last_price")),
        spread=_optional_float(row.get("spread")),
        bid_size=_optional_float(row.get("bid_size")),
        ask_size=_optional_float(row.get("ask_size")),
        liquidity_depth=_optional_float(row.get("liquidity_depth")),
        source_payload_id=row.get("source_payload_id"),
        source_cache_path=row.get("source_cache_path"),
        data_quality_flags=_json_flags(row.get("data_quality_flags")),
    )


def _quote_with_worse_fill_tick(
    quote: QuoteSnapshot,
    *,
    action: OrderAction,
    tick: float = DEFAULT_WORSE_FILL_TICK,
) -> QuoteSnapshot:
    bid = quote.bid
    ask = quote.ask
    if action == OrderAction.BUY and ask is not None:
        ask = min(1.0, ask + tick)
    elif action == OrderAction.SELL and bid is not None:
        bid = max(0.0, bid - tick)
    spread = quote.spread
    mid = quote.mid
    if bid is not None and ask is not None:
        spread = max(0.0, ask - bid)
        mid = (bid + ask) / 2.0
    return replace(quote, bid=bid, ask=ask, mid=mid, spread=spread)


def _optional_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    return float(value)


def _optional_int(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    return int(float(value))


def _int_value(value: Any) -> int:
    parsed = _optional_int(value)
    if parsed is None:
        raise ValueError("Missing required integer value")
    return parsed


def _boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value in (None, ""):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _json_flags(value: Any) -> tuple[str, ...]:
    if value in (None, ""):
        return ()
    if isinstance(value, (list, tuple)):
        return tuple(str(item) for item in value)
    text = str(value)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return _pipe_flags(text)
    if isinstance(parsed, list):
        return tuple(str(item) for item in parsed)
    if isinstance(parsed, str):
        return (parsed,)
    return (str(parsed),)


def _json_mapping(value: Any) -> Mapping[str, Any]:
    if value in (None, ""):
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, Mapping) else {}


def _pipe_flags(value: Any) -> tuple[str, ...]:
    if value in (None, ""):
        return ()
    return tuple(part for part in str(value).split("|") if part)


if __name__ == "__main__":
    raise SystemExit(main())
