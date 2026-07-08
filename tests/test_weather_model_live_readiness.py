from __future__ import annotations

import unittest
from datetime import date, datetime, timezone

from daily_weather_pipeline.weather_model_live_readiness import _poison_evaluation_fields
from daily_weather_pipeline.weather_model_validation import ValidationRow


class WeatherModelLiveReadinessTests(unittest.TestCase):
    def test_poison_evaluation_fields_preserves_live_inputs(self) -> None:
        row = ValidationRow(
            row_id="market|forecast",
            universe_row_id="market",
            forecast_id="forecast",
            normalized_city="Austin, TX",
            country_region="TX",
            timezone_name="America/Chicago",
            market_date_local=date(2026, 1, 2),
            decision_time_utc=datetime(2026, 1, 1, 18, tzinfo=timezone.utc),
            decision_hour_utc=18,
            local_decision_hour=12.0,
            lead_days=1,
            day_type="next_day",
            source_name="historical_forecast_archive",
            model_name="best_match",
            bucket_label="70F",
            bucket_type="exact",
            bucket_shape="bounded",
            threshold_low_f=70.0,
            threshold_high_f=70.999,
            threshold_distance_f=1.5,
            threshold_distance_bucket="1-2F",
            realized_high_f=73.0,
            forecast_high_f=71.5,
            forecast_error_f=-1.5,
            outcome=1,
            resolved_outcome=1,
            resolved_outcome_mismatch=False,
            current_probability=0.42,
            mean_probability=0.40,
            model_probabilities={"historical_forecast_archive.kernel_tight": 0.42},
            forecast_quality_flags=(),
            realized_quality_flags=(),
            feature_json={"temperature_2m_max": 71.5},
            latitude=30.2672,
            longitude=-97.7431,
        )

        poisoned = _poison_evaluation_fields(row)

        self.assertEqual(poisoned.forecast_high_f, row.forecast_high_f)
        self.assertEqual(poisoned.current_probability, row.current_probability)
        self.assertEqual(poisoned.threshold_low_f, row.threshold_low_f)
        self.assertEqual(poisoned.normalized_city, row.normalized_city)
        self.assertNotEqual(poisoned.realized_high_f, row.realized_high_f)
        self.assertNotEqual(poisoned.outcome, row.outcome)
        self.assertNotEqual(poisoned.resolved_outcome, row.resolved_outcome)
        self.assertEqual(poisoned.realized_quality_flags, ("poisoned_for_live_readiness_audit",))


if __name__ == "__main__":
    unittest.main()
