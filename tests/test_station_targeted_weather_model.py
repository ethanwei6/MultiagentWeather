from __future__ import annotations

import unittest
from datetime import date, datetime, timedelta, timezone

from daily_weather_pipeline.station_targeted_weather_model import (
    FORBIDDEN_MODEL_INPUT_COLUMNS,
    DecisionRow,
    _apply_observed_high_guard,
    _bucket_label_for_temperature,
    _effective_probability_bounds,
)


def _row(
    *,
    lower: float | None,
    upper: float | None,
    observed_high: float | None,
    observed_at_delta_hours: int = 0,
) -> DecisionRow:
    decision_time = datetime(2026, 6, 1, 18, tzinfo=timezone.utc)
    return DecisionRow(
        row_id="decision-1",
        weather_key="weather-1",
        sample_key="sample-1",
        universe_row_id="universe-1",
        market_id="market-1",
        normalized_city="Austin, TX",
        station_id="KAUS",
        region="US_South",
        settlement_source_url="https://www.wunderground.com/history/daily/us/tx/austin/KAUS",
        timezone_name="America/Chicago",
        local_date=date(2026, 6, 1),
        decision_time_utc=decision_time,
        forecast_run_time_utc=decision_time - timedelta(hours=1),
        decision_hour_utc=18,
        local_decision_hour=13.0,
        lead_days=0,
        day_type="same_day",
        forecast_source_name="open_meteo_historical_forecast",
        forecast_model_family="open_meteo_historical_forecast",
        forecast_high_f=75.0,
        bucket_type="exact",
        threshold_low_f=lower,
        threshold_high_f=upper,
        threshold_distance_f=0.0,
        threshold_distance_bucket="0-1F",
        settlement_precision="whole_F",
        station_high_f=75.0,
        open_meteo_high_f=75.0,
        station_outcome=1,
        open_meteo_outcome=1,
        canonical_label=1,
        side="YES",
        mid=0.51,
        bid=0.50,
        ask=0.52,
        observed_high_so_far_f=observed_high,
        observed_high_observed_at_utc=(
            decision_time + timedelta(hours=observed_at_delta_hours)
            if observed_high is not None
            else None
        ),
        observed_high_sample_count=4 if observed_high is not None else 0,
        recommended_action=None,
        quote_covered=True,
    )


class StationTargetedWeatherModelTests(unittest.TestCase):
    def test_whole_f_exact_probability_bounds_match_settlement_rounding(self) -> None:
        self.assertEqual(_effective_probability_bounds(75.0, 75.0, "whole_F"), (74.5, 75.5))
        self.assertEqual(_bucket_label_for_temperature(74.49, lower=75.0, upper=75.0, settlement_precision="whole_F"), 0)
        self.assertEqual(_bucket_label_for_temperature(74.50, lower=75.0, upper=75.0, settlement_precision="whole_F"), 1)
        self.assertEqual(_bucket_label_for_temperature(75.49, lower=75.0, upper=75.0, settlement_precision="whole_F"), 1)
        self.assertEqual(_bucket_label_for_temperature(75.50, lower=75.0, upper=75.0, settlement_precision="whole_F"), 0)

    def test_observed_high_guard_blocks_upper_bounded_bucket_after_crossing(self) -> None:
        row = _row(lower=72.0, upper=75.0, observed_high=75.5)

        probability = _apply_observed_high_guard(0.62, row)

        self.assertAlmostEqual(probability, 0.000001)

    def test_observed_high_guard_makes_above_bucket_certain_after_crossing(self) -> None:
        row = _row(lower=80.0, upper=None, observed_high=79.5)

        probability = _apply_observed_high_guard(0.22, row)

        self.assertAlmostEqual(probability, 0.999999)

    def test_future_observed_high_is_ignored(self) -> None:
        row = _row(lower=72.0, upper=75.0, observed_high=90.0, observed_at_delta_hours=1)

        probability = _apply_observed_high_guard(0.62, row)

        self.assertEqual(probability, 0.62)

    def test_feature_firewall_forbidden_fields_are_explicit(self) -> None:
        expected = {
            "bid",
            "ask",
            "mid",
            "resolved_outcome",
            "canonical_label",
            "station_high_f",
            "open_meteo_high_f",
        }

        self.assertTrue(expected.issubset(FORBIDDEN_MODEL_INPUT_COLUMNS))


if __name__ == "__main__":
    unittest.main()
