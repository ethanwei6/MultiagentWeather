from __future__ import annotations

import unittest
from datetime import datetime, time, timezone

from daily_weather_pipeline.phase_c_convergence_measurement import (
    MarketSideRow,
    QuotePoint,
    StationPoint,
    classify_fact_events,
    measure_quote_path,
)


class PhaseCConvergenceMeasurementTests(unittest.TestCase):
    def test_above_bucket_crossing_classifies_yes_true_and_no_false(self) -> None:
        rows = [
            _market_row(side="YES", bucket_type="above", lower=80.0, upper=None),
            _market_row(side="NO", bucket_type="above", lower=80.0, upper=None),
        ]
        points = {
            ("source", "2026-06-01"): [
                _station_point("2026-06-01T15:00:00+00:00", "10:00:00", 78.4),
                _station_point("2026-06-01T16:00:00+00:00", "11:00:00", 79.6),
            ]
        }

        events = classify_fact_events(rows, points, heat_close_local=time(17, 30))

        self.assertEqual(
            [event.fact_state for event in events],
            ["crossed_open_ended_threshold", "crossed_open_ended_threshold"],
        )
        self.assertEqual([event.side_fact_fv for event in events], [1.0, 0.0])
        self.assertTrue(
            all(event.event_time_utc.isoformat() == "2026-06-01T16:00:00+00:00" for event in events)
        )

    def test_heat_close_rule_is_labeled_near_deterministic(self) -> None:
        row = _market_row(side="NO", bucket_type="above", lower=90.0, upper=None, canonical_label=0)
        points = {
            ("source", "2026-06-01"): [
                _station_point("2026-06-01T20:00:00+00:00", "15:00:00", 84.0),
                _station_point("2026-06-01T22:30:00+00:00", "17:30:00", 86.0),
            ]
        }

        [event] = classify_fact_events([row], points, heat_close_local=time(17, 30))

        self.assertEqual(event.fact_state, "near_dead_above_after_heat_close")
        self.assertEqual(event.event_class, "near_heat_window_close")
        self.assertEqual(event.side_fact_fv, 1.0)
        self.assertTrue(event.fact_matches_canonical_yes)

    def test_quote_path_measures_first_quote_distance_and_repricing(self) -> None:
        row = _market_row(side="YES", bucket_type="above", lower=80.0, upper=None, token_ref=7)
        [event] = classify_fact_events(
            [row],
            {
                ("source", "2026-06-01"): [
                    _station_point("2026-06-01T16:00:00+00:00", "11:00:00", 80.0),
                ]
            },
            heat_close_local=time(17, 30),
        )
        quotes = [
            _quote("2026-06-01T16:03:00+00:00", bid=0.81, ask=0.86, mid=0.835, spread=0.05),
            _quote("2026-06-01T16:20:00+00:00", bid=0.95, ask=0.99, mid=0.97, spread=0.04),
        ]

        measured = measure_quote_path(event, quotes)

        self.assertTrue(measured["primary_measurable"])
        self.assertEqual(measured["next_quote_seconds_after_fact"], 180.0)
        self.assertEqual(measured["first_quote_ask_distance_from_fv"], 0.14)
        self.assertTrue(measured["true_side_stale_ask_gap_gte_5c"])
        self.assertEqual(measured["repriced_mid_seconds"], 1200.0)


def _market_row(
    *,
    side: str,
    bucket_type: str,
    lower: float | None,
    upper: float | None,
    token_ref: int | None = None,
    canonical_label: int = 1,
) -> MarketSideRow:
    return MarketSideRow(
        universe_row_id="u1",
        market_id="m1",
        slug="slug",
        question="question",
        city="City",
        region="Region",
        timezone_name="America/New_York",
        market_date="2026-06-01",
        bucket_label="bucket",
        canonical_bucket_type=bucket_type,
        threshold_low_f=lower,
        threshold_high_f=upper,
        side=side,
        token_id=f"token-{side}",
        yes_token_id="token-YES",
        no_token_id="token-NO",
        canonical_label=canonical_label,
        settlement_unit="F",
        settlement_precision="whole_F",
        source_family="Wunderground",
        station_name="Station",
        source_url="source",
        recommended_action="keep_polymarket_label",
        exclude_from_backtest_replay=0,
        token_ref=token_ref,
        compact_fetch_status="complete:1",
    )


def _station_point(observed_at: str, local_time: str, high: float) -> StationPoint:
    return StationPoint(
        observed_at_utc=datetime.fromisoformat(observed_at),
        local_time=time.fromisoformat(local_time),
        temperature_f=high,
        high_so_far_f=high,
        settlement_value=round(high),
        sample_count=1,
    )


def _quote(timestamp: str, *, bid: float, ask: float, mid: float, spread: float) -> QuotePoint:
    return QuotePoint(
        timestamp_utc=datetime.fromisoformat(timestamp).astimezone(timezone.utc),
        bid=bid,
        ask=ask,
        mid=mid,
        last_price=None,
        spread=spread,
        bid_size=10.0,
        ask_size=12.0,
        liquidity_depth=22.0,
        data_quality_flags="",
    )
