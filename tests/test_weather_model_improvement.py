from __future__ import annotations

import unittest
from datetime import date, datetime, timedelta, timezone
from typing import Optional

from daily_weather_pipeline.weather_model_improvement import (
    ObservedHighContext,
    _apply_observed_high_overlay,
)
from daily_weather_pipeline.weather_model_validation import ValidationRow


def _row(*, lower: Optional[float], upper: Optional[float], bucket_type: str = "range") -> ValidationRow:
    return ValidationRow(
        row_id="market|forecast",
        universe_row_id="market",
        forecast_id="forecast",
        normalized_city="Austin, TX",
        country_region="TX",
        timezone_name="America/Chicago",
        market_date_local=date(2026, 1, 2),
        decision_time_utc=datetime(2026, 1, 2, 18, tzinfo=timezone.utc),
        decision_hour_utc=18,
        local_decision_hour=12.0,
        lead_days=0,
        day_type="same_day",
        source_name="historical_forecast_archive",
        model_name="best_match",
        bucket_label="bucket",
        bucket_type=bucket_type,
        bucket_shape="bounded" if lower is not None and upper is not None else "upper_tail",
        threshold_low_f=lower,
        threshold_high_f=upper,
        threshold_distance_f=1.0,
        threshold_distance_bucket="1-2F",
        realized_high_f=74.0,
        forecast_high_f=72.0,
        forecast_error_f=-2.0,
        outcome=1,
        resolved_outcome=1,
        resolved_outcome_mismatch=False,
        current_probability=0.42,
        mean_probability=0.40,
        model_probabilities={"historical_forecast_archive.kernel_tight": 0.42},
        forecast_quality_flags=(),
        realized_quality_flags=(),
        feature_json={},
        latitude=30.2672,
        longitude=-97.7431,
    )


class WeatherModelImprovementTests(unittest.TestCase):
    def test_observed_high_above_bounded_bucket_makes_event_impossible(self) -> None:
        row = _row(lower=70.0, upper=72.0)
        context = {
            (row.normalized_city, row.market_date_local, row.decision_time_utc): ObservedHighContext(
                observed_high_f=73.0,
                observed_at_utc=row.decision_time_utc,
                is_final_at_decision=False,
            )
        }

        probability = _apply_observed_high_overlay(0.42, row, context)

        self.assertAlmostEqual(probability, 0.000001)

    def test_observed_high_above_open_tail_threshold_makes_event_certain(self) -> None:
        row = _row(lower=73.0, upper=None, bucket_type="above")
        context = {
            (row.normalized_city, row.market_date_local, row.decision_time_utc): ObservedHighContext(
                observed_high_f=73.0,
                observed_at_utc=row.decision_time_utc,
                is_final_at_decision=False,
            )
        }

        probability = _apply_observed_high_overlay(0.42, row, context)

        self.assertAlmostEqual(probability, 0.999999)

    def test_future_observed_high_is_ignored(self) -> None:
        row = _row(lower=70.0, upper=72.0)
        context = {
            (row.normalized_city, row.market_date_local, row.decision_time_utc): ObservedHighContext(
                observed_high_f=80.0,
                observed_at_utc=row.decision_time_utc + timedelta(hours=1),
                is_final_at_decision=False,
            )
        }

        probability = _apply_observed_high_overlay(0.42, row, context)

        self.assertEqual(probability, 0.42)


if __name__ == "__main__":
    unittest.main()
