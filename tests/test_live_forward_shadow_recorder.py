from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from daily_weather_pipeline import live_forward_shadow_recorder as recorder


class LiveForwardShadowRecorderTests(unittest.TestCase):
    def test_parse_binary_temperature_market_preserves_no_side_contract(self) -> None:
        raw = {
            "id": "market-1",
            "conditionId": "condition-1",
            "slug": "nyc-high-july-9",
            "question": "Will the high temperature in New York be 100F or above on July 9?",
            "description": "Resolved using whole degrees Fahrenheit.",
            "outcomes": '["Yes", "No"]',
            "clobTokenIds": '["yes-token", "no-token"]',
            "outcomePrices": '["0.09", "0.91"]',
            "endDate": "2026-07-09T16:00:00Z",
        }

        market = recorder.parse_weather_market(raw, today=date(2026, 7, 8))

        self.assertIsNotNone(market)
        assert market is not None
        self.assertEqual(market.city.display_name, "New York, NY")
        self.assertEqual(market.target_date, date(2026, 7, 9))
        self.assertEqual(market.bucket.no_token_id, "no-token")
        self.assertEqual(market.bucket.yes_token_id, "yes-token")
        self.assertEqual(market.bucket.bucket_type, "above")
        self.assertEqual(market.bucket.resolution_unit, "F")

    def test_record_live_forward_shadow_writes_read_only_artifacts(self) -> None:
        city = recorder.CityConfig("Test City", "TC", 0.0, 0.0, "UTC", "KTCY")
        market = recorder.WeatherMarket(
            market_id="market-1",
            condition_id="condition-1",
            slug="test-market",
            event_slug="test-event",
            question="Will the high temperature in Test City be 100F or above on July 9?",
            close_time=datetime(2026, 7, 9, 16, tzinfo=timezone.utc),
            target_date=(datetime.now(timezone.utc) + timedelta(days=1)).date(),
            city=city,
            bucket=recorder.Bucket(
                label="100F or above",
                lower_f=100.0,
                upper_f=None,
                yes_token_id="yes-token",
                no_token_id="no-token",
                yes_price=0.08,
                no_price=0.92,
            ),
        )

        def fake_discover(**kwargs):
            return [market]

        def fake_forecasts(*args, **kwargs):
            fetched_at = kwargs["fetched_at"]
            return [
                recorder.ForecastSource(
                    source_name=f"source-{index}",
                    fetched_at_utc=fetched_at,
                    temperature_2m_max_f=80.0 + index,
                    samples_f=(80.0 + index,),
                    features={},
                    raw_cache_path=f"/tmp/source-{index}.json",
                    payload_id=f"payload-{index}",
                    request_url="https://example.invalid",
                )
                for index in range(3)
            ]

        def fake_station(*args, **kwargs):
            fetched_at = kwargs["fetched_at"]
            return recorder.StationObservationSummary(
                station_id="KTCY",
                status="downloaded",
                fetched_at_utc=fetched_at,
                local_date=market.target_date.isoformat(),
                latest_observed_at_utc=fetched_at,
                latest_temperature_f=78.0,
                high_so_far_f=79.0,
                high_so_far_observed_at_utc=fetched_at,
                observation_count=1,
                raw_cache_path="/tmp/station.csv",
                payload_id="station-payload",
            )

        def fake_book(token_id, *, raw_dir, fetched_at, cache_only):
            price = 0.70 if token_id == "no-token" else 0.30
            return recorder.BookSnapshot(
                token_id=token_id,
                fetched_at_utc=fetched_at,
                payload_id=f"{token_id}-payload",
                raw_cache_path=str(Path(raw_dir) / f"{token_id}.json"),
                best_bid=price - 0.01,
                best_ask=price,
                bid_size=20.0,
                ask_size=15.0,
                mid=price - 0.005,
                spread=0.01,
                top_bids=({"price": price - 0.01, "size": 20.0},),
                top_asks=({"price": price, "size": 15.0},),
            )

        originals = (
            recorder.discover_active_weather_markets,
            recorder.fetch_open_meteo_forecasts,
            recorder.fetch_station_observations,
            recorder.fetch_order_book,
        )
        recorder.discover_active_weather_markets = fake_discover
        recorder.fetch_open_meteo_forecasts = fake_forecasts
        recorder.fetch_station_observations = fake_station
        recorder.fetch_order_book = fake_book
        try:
            with tempfile.TemporaryDirectory() as directory:
                artifacts = recorder.record_live_forward_shadow(
                    work_root=Path(directory),
                    iterations=1,
                    interval_seconds=0,
                )
                summary = artifacts.summary
                events = artifacts.events_path.read_text(encoding="utf-8").strip().splitlines()
                self.assertTrue(artifacts.json_path.exists())
                self.assertTrue(artifacts.markdown_path.exists())
        finally:
            (
                recorder.discover_active_weather_markets,
                recorder.fetch_open_meteo_forecasts,
                recorder.fetch_station_observations,
                recorder.fetch_order_book,
            ) = originals

        self.assertEqual(summary["real_order_placement_allowed"], False)
        self.assertEqual(summary["real_orders_placed"], 0)
        self.assertEqual(summary["fills_simulated_as_truth"], False)
        self.assertEqual(summary["decision_counts"]["would_trade_signals"], 1)
        self.assertEqual(summary["decision_counts"]["raw_would_trade_poll_detections"], 1)
        self.assertEqual(summary["decision_counts"]["would_trade_signal_clusters"], 1)
        self.assertEqual(summary["lead_day_counts"]["priority_1_2"], 1)
        self.assertEqual(summary["quote_quality"]["counts"]["bid_ask_available"], 1)
        self.assertEqual(summary["would_trade_settlement_status_counts"]["pending"], 1)
        self.assertEqual(summary["would_trade_cluster_settlement_status_counts"]["pending"], 1)
        self.assertFalse(summary["forward_collection_progress"]["collection_complete"])
        self.assertEqual(summary["forward_collection_progress"]["unique_market_side_signal_clusters"], 1)
        self.assertEqual(summary["forward_collection_progress"]["unique_unresolved_market_side_signal_clusters"], 1)
        self.assertEqual(summary["forward_collection_progress"]["gate_count_basis"], "deduped_true_opportunity_clusters")
        self.assertEqual(
            summary["live_vs_replay_behavior"]["pre_registered_expected_trades_per_day"]["point_estimate"],
            1.0,
        )
        self.assertEqual(
            summary["live_vs_replay_behavior"]["pre_registered_hit_rate_confidence_bands"]["primary_interval"],
            "wilson_95pct",
        )
        self.assertEqual(
            summary["live_vs_replay_behavior"]["pre_registered_hit_rate_confidence_bands"][
                "minimum_resolved_would_trades_before_interpretation"
            ],
            30,
        )
        self.assertEqual(len(events), 1)
        self.assertIn("observed_NO_best_ask_snapshot_not_a_fill_and_not_truth", events[0])
        self.assertIn("old_high_confidence_no_tail_entry", events[0])

    def test_refresh_pending_would_trade_settlements_resolves_closed_gamma_market(self) -> None:
        event = {
            "decision": {
                "would_trade": True,
                "decision_reason": "old_high_confidence_no_tail_entry",
                "would_trade_signal": {
                    "decision_id": "decision-1",
                    "limit_price": 0.87,
                    "requested_notional_usd": 2.0,
                    "side": "NO",
                    "token_id": "no-token",
                },
            },
            "live_market_metadata": {
                "condition_id": "condition-1",
                "event_slug": "test-weather-event",
                "market_id": "market-1",
                "question": "Will the highest temperature in Test City be 32C?",
                "slug": "test-weather-event-32c",
            },
            "market_discovery": {
                "bucket_label": "NO: 89.6F",
                "decision_id": "decision-1",
                "market_date_local": "2026-07-09",
                "market_id": "market-1",
                "normalized_city": "Test City, TC",
                "side": "NO",
                "slug": "test-weather-event-32c",
                "token_id": "no-token",
            },
            "parsed_contract": {
                "bucket_label": "89.6F",
                "city": "Test City, TC",
                "date": "2026-07-09",
                "no_token_id": "no-token",
                "side": "NO",
                "yes_token_id": "yes-token",
            },
            "quote": {
                "ask": 0.87,
                "bid": 0.86,
                "spread": 0.01,
                "timestamp_utc": "2026-07-08T02:22:43+00:00",
                "top_of_book_depth": {"ask_size": 10.0, "bid_size": 11.0, "liquidity_depth": 21.0},
            },
            "settlement": {"status": "pending", "later_update_required": True},
        }

        original_fetch = recorder._fetch_gamma_event_by_slug

        def fake_fetch(event_slug, *, raw_dir, fetched_at, cache_only):
            return (
                {
                    "markets": [
                        {
                            "active": True,
                            "closed": True,
                            "clobTokenIds": '["yes-token", "no-token"]',
                            "conditionId": "condition-1",
                            "id": "market-1",
                            "outcomePrices": '["0", "1"]',
                            "outcomes": '["Yes", "No"]',
                            "slug": "test-weather-event-32c",
                            "umaResolutionStatuses": '["proposed"]',
                        }
                    ]
                },
                Path(raw_dir) / "gamma-event.json",
                "payload-1",
                f"https://gamma-api.polymarket.com/events/slug/{event_slug}",
            )

        recorder._fetch_gamma_event_by_slug = fake_fetch
        try:
            with tempfile.TemporaryDirectory() as directory:
                report_dir = Path(directory) / "reports"
                report_dir.mkdir()
                (report_dir / "20260708T000000Z-old-no-tail-live-forward-shadow-intraday-events.jsonl").write_text(
                    json.dumps(event) + "\n",
                    encoding="utf-8",
                )
                (report_dir / "20260708T000000Z-old-no-tail-live-forward-shadow-intraday.json").write_text(
                    json.dumps(
                        {
                            "generated_at_utc": "2026-07-08T00:00:00+00:00",
                            "decision_counts": {"shadow_decisions": 1, "would_trade_signals": 1},
                        }
                    ),
                    encoding="utf-8",
                )

                summary = recorder.refresh_pending_would_trade_settlements(
                    report_dir=report_dir,
                    raw_dir=Path(directory) / "raw",
                    run_id="20260708T010000Z",
                    cache_only=False,
                    generated_at=datetime(2026, 7, 8, 1, tzinfo=timezone.utc),
                )
                progress = recorder._forward_collection_progress(
                    report_dir,
                    current_summary={
                        "generated_at_utc": "2026-07-08T01:00:00+00:00",
                        "decision_counts": {"shadow_decisions": 0, "would_trade_signals": 0},
                    },
                    current_settlement_refresh=summary,
                )
        finally:
            recorder._fetch_gamma_event_by_slug = original_fetch

        self.assertEqual(summary["settlement_status_counts"], {"resolved": 1})
        update = summary["updates"][0]
        self.assertEqual(update["settlement"]["canonical_label"], "NO")
        self.assertEqual(update["settlement"]["payout_for_shadow_side"], 1)
        self.assertTrue(update["settlement"]["payout_is_shadow_settlement_only"])
        self.assertFalse(summary["real_order_placement_allowed"])
        self.assertEqual(summary["real_orders_placed"], 0)
        self.assertEqual(summary["pending_would_trade_clusters_scanned"], 1)
        self.assertEqual(summary["settlement_cluster_status_counts"], {"resolved": 1})
        self.assertEqual(summary["resolved_would_trade_clusters"], 1)
        self.assertEqual(progress["resolved_would_trade_signals"], 1)
        self.assertEqual(progress["unresolved_would_trade_signals"], 0)
        self.assertEqual(progress["unique_market_side_signal_clusters"], 1)
        self.assertEqual(progress["unique_resolved_market_side_signal_clusters"], 1)
        self.assertEqual(progress["gate_resolved_would_trade_clusters"], 1)

    def test_forward_progress_deduplicates_repeated_signals_by_true_opportunity(self) -> None:
        base_event = {
            "decision": {
                "would_trade": True,
                "would_trade_signal": {
                    "decision_id": "decision-1",
                    "side": "NO",
                    "token_id": "no-token",
                },
            },
            "live_market_metadata": {
                "condition_id": "condition-1",
                "event_slug": "test-weather-event",
                "market_id": "market-1",
                "slug": "test-weather-event-32c",
            },
            "market_discovery": {
                "bucket_label": "89.6F",
                "decision_id": "decision-1",
                "market_date_local": "2026-07-09",
                "market_id": "market-1",
                "side": "NO",
                "slug": "test-weather-event-32c",
                "token_id": "no-token",
            },
            "parsed_contract": {
                "bucket_label": "89.6F",
                "city": "Test City, TC",
                "date": "2026-07-09",
                "no_token_id": "no-token",
                "side": "NO",
            },
            "settlement": {"status": "pending"},
        }
        repeated_event = json.loads(json.dumps(base_event))
        repeated_event["decision"]["would_trade_signal"]["decision_id"] = "decision-2"
        repeated_event["market_discovery"]["decision_id"] = "decision-2"
        different_date_event = json.loads(json.dumps(base_event))
        different_date_event["decision"]["would_trade_signal"]["decision_id"] = "decision-3"
        different_date_event["market_discovery"]["decision_id"] = "decision-3"
        different_date_event["market_discovery"]["market_date_local"] = "2026-07-10"
        different_date_event["parsed_contract"]["date"] = "2026-07-10"

        with tempfile.TemporaryDirectory() as directory:
            report_dir = Path(directory) / "reports"
            report_dir.mkdir()
            (report_dir / "20260708T000000Z-old-no-tail-live-forward-shadow-intraday-events.jsonl").write_text(
                json.dumps(base_event)
                + "\n"
                + json.dumps(repeated_event)
                + "\n"
                + json.dumps(different_date_event)
                + "\n",
                encoding="utf-8",
            )
            (report_dir / "20260708T000000Z-old-no-tail-live-forward-shadow-intraday.json").write_text(
                json.dumps(
                    {
                        "generated_at_utc": "2026-07-08T00:00:00+00:00",
                        "decision_counts": {"shadow_decisions": 3, "would_trade_signals": 3},
                    }
                ),
                encoding="utf-8",
            )
            progress = recorder._forward_collection_progress(
                report_dir,
                current_summary={
                    "generated_at_utc": "2026-07-08T00:01:00+00:00",
                    "decision_counts": {"shadow_decisions": 0, "would_trade_signals": 0},
                },
            )

        self.assertEqual(progress["would_trade_signals"], 3)
        self.assertEqual(progress["unique_market_side_signal_clusters"], 2)
        self.assertEqual(progress["unique_market_side_signal_cluster_status_counts"], {"pending": 2})
        self.assertEqual(progress["gate_unresolved_would_trade_clusters"], 2)
        self.assertTrue(progress["raw_signals_are_poll_level_not_independent_contracts"])


if __name__ == "__main__":
    unittest.main()
