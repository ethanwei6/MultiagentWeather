from __future__ import annotations

import unittest
from datetime import datetime, timezone

from daily_weather_pipeline.backtesting_engine import (
    ReplayGeneratedScoreCache,
    WeatherModelDecisionContext,
)
from daily_weather_pipeline.weather_model_runner import (
    ObservedHighGuardedShrinkageWeatherModelRunner,
)


class ObservedHighGuardedShrinkageWeatherModelRunnerTests(unittest.TestCase):
    def test_firewall_rejects_leaky_forecast_features(self) -> None:
        runner = _runner()
        context = _context(forecast_features={"actual_high_f": 81.0})

        with self.assertRaises(ValueError):
            runner.predict(context)

    def test_firewall_rejects_future_observed_high(self) -> None:
        runner = _runner()
        context = _context(
            observed_high_so_far_f=73.0,
            observed_high_timestamp_utc=datetime(2026, 1, 2, 13, tzinfo=timezone.utc),
        )

        with self.assertRaises(ValueError):
            runner.predict(context)

    def test_identical_point_in_time_inputs_have_stable_prediction_hash_metadata(self) -> None:
        runner = _runner()
        context = _context()

        first = runner.predict(context)
        second = runner.predict(context)

        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        assert first is not None
        assert second is not None
        self.assertEqual(first.p_yes, second.p_yes)
        self.assertEqual(first.p_no, second.p_no)
        self.assertEqual(first.model_version, second.model_version)
        self.assertEqual(first.model_code_hash, second.model_code_hash)
        self.assertEqual(first.model_config_hash, second.model_config_hash)
        self.assertEqual(first.metadata["input_hash"], second.metadata["input_hash"])
        self.assertEqual(first.metadata["input_hash"], context.input_hash)
        self.assertEqual(first.feature_provenance, second.feature_provenance)
        self.assertEqual(first.per_source_probabilities, second.per_source_probabilities)
        self.assertAlmostEqual(first.metadata["p_no"], 1.0 - first.p_yes)
        self.assertAlmostEqual(first.metadata["p_side"], first.p_yes)

        score = first.to_shared_score(
            side=context.side,
            observed_high=None,
            parser_confidence=context.parser_confidence,
            settlement_unit=None,
            settlement_precision=None,
        )
        cache = ReplayGeneratedScoreCache()
        record = cache.add(context=context, prediction=first, score=score)
        self.assertTrue(cache.is_reusable(record, context=context, runner_metadata=runner.metadata()))
        self.assertEqual(record.model_count, first.model_count)
        self.assertEqual(record.model_agreement, first.model_agreement)
        self.assertEqual(record.per_source_probabilities, first.per_source_probabilities)

        changed_runner = _runner(label_availability_lag_hours=24.0)
        self.assertFalse(
            cache.is_reusable(record, context=context, runner_metadata=changed_runner.metadata())
        )

    def test_observed_high_overlay_uses_only_decision_time_observation(self) -> None:
        runner = _runner(probability=0.82)
        context = _context(
            bucket_type="range",
            bucket_label="70-72F",
            threshold_low_f=70.0,
            threshold_high_f=72.0,
            observed_high_so_far_f=73.0,
            observed_high_timestamp_utc=datetime(2026, 1, 2, 11, tzinfo=timezone.utc),
        )

        prediction = runner.predict(context)

        self.assertIsNotNone(prediction)
        assert prediction is not None
        self.assertLess(prediction.p_yes, 0.001)
        self.assertEqual(prediction.observed_high_aware_confidence, 0.95)
        self.assertEqual(prediction.metadata["p_side"], prediction.p_yes)


def _runner(
    *,
    probability: float = 0.42,
    label_availability_lag_hours: float = 12.0,
) -> ObservedHighGuardedShrinkageWeatherModelRunner:
    def baseline_provider(_context: WeatherModelDecisionContext):
        return probability, {
            "open_meteo.kernel_tight": probability,
            "open_meteo.normal_parametric": min(0.99, probability + 0.02),
        }

    return ObservedHighGuardedShrinkageWeatherModelRunner(
        historical_rows=[],
        baseline_probability_provider=baseline_provider,
        label_availability_lag_hours=label_availability_lag_hours,
    )


def _context(
    *,
    forecast_features=None,
    bucket_type: str = "above",
    bucket_label: str = "70F+",
    threshold_low_f: float | None = 70.0,
    threshold_high_f: float | None = None,
    observed_high_so_far_f: float | None = None,
    observed_high_timestamp_utc: datetime | None = None,
) -> WeatherModelDecisionContext:
    if forecast_features is None:
        forecast_features = {
            "model_name": "historical_forecast",
            "latitude": 51.5,
            "longitude": -0.1,
        }
    return WeatherModelDecisionContext(
        decision_id="decision-1",
        universe_row_id="row-1",
        market_id="market-1",
        slug="london-high",
        question="Will London reach 70F?",
        normalized_city="London, GB",
        country_region="GB",
        timezone="Europe/London",
        market_date_local="2026-01-02",
        bucket_label=bucket_label,
        bucket_type=bucket_type,
        threshold_low_f=threshold_low_f,
        threshold_high_f=threshold_high_f,
        side="YES",
        token_id="token-yes",
        decision_time_utc=datetime(2026, 1, 2, 12, tzinfo=timezone.utc),
        decision_hour_utc=12,
        lead_days=0,
        parser_confidence=0.99,
        forecast_run_time_utc=datetime(2026, 1, 2, 6, tzinfo=timezone.utc),
        forecast_source_name="open_meteo_historical_forecast",
        forecast_temperature_2m_max_f=71.0,
        forecast_features=forecast_features,
        observed_high_so_far_f=observed_high_so_far_f,
        observed_high_timestamp_utc=observed_high_timestamp_utc,
        observed_high_source="point_in_time_station" if observed_high_so_far_f is not None else None,
        observed_high_final_at_decision=False if observed_high_so_far_f is not None else None,
        data_quality_flags=("test_fixture",),
        input_source_paths=("/tmp/forecast.json",),
        feature_provenance={
            "forecast": "test_point_in_time_forecast",
            "observed_high_so_far": "test_point_in_time_observed_high",
        },
    )
