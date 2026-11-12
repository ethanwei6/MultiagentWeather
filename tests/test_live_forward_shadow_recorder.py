from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

from daily_weather_pipeline.backtesting_engine import WeatherModelDecisionContext
from daily_weather_pipeline import live_forward_shadow_recorder as recorder
from daily_weather_pipeline.live_forward_shadow_scoring import RUNNER_SCORE_FAMILY
from daily_weather_pipeline.weather_model_runner import (
    ObservedHighGuardedShrinkageWeatherModelRunner,
)


class LiveForwardShadowRecorderTests(unittest.TestCase):
    def test_book_payload_keeps_full_levels_separate_from_top_of_book(self) -> None:
        book = recorder.BookSnapshot(
            token_id="token",
            fetched_at_utc=datetime(2026, 7, 13, tzinfo=timezone.utc),
            payload_id="payload",
            raw_cache_path="fixture.json",
            best_bid=0.79,
            best_ask=0.80,
            bid_size=10.0,
            ask_size=1.0,
            mid=0.795,
            spread=0.01,
            top_bids=({"price": 0.79, "size": 10.0},),
            top_asks=({"price": 0.80, "size": 1.0},),
            ask_levels=(
                {"price": 0.80, "size": 1.0},
                {"price": 0.82, "size": 2.0},
            ),
            full_book_complete=True,
            min_order_size_shares=5.0,
            tick_size=0.01,
            fee_rate=0.05,
            fee_rate_source="fixture_current_market_info",
        )

        payload = recorder.book_to_payload(book)

        self.assertEqual(len(payload["top_asks"]), 1)
        self.assertEqual(len(payload["ask_levels"]), 2)
        self.assertTrue(payload["full_book_complete"])
        self.assertEqual(payload["fee_rate"], 0.05)

    def test_city_aliases_do_not_imply_celsius_units(self) -> None:
        for alias in ("NYC", "DC", "Washington DC", "New York City"):
            with self.subTest(alias=alias):
                raw = {
                    "id": f"market-{alias}",
                    "slug": f"{alias.lower().replace(' ', '-')}-high-july-9",
                    "question": f"Will the highest temperature in {alias} be 85 or above on July 9?",
                    "description": "Resolved from the official station high temperature.",
                    "outcomes": '["Yes", "No"]',
                    "clobTokenIds": '["yes-token", "no-token"]',
                }

                market = recorder.parse_weather_market(raw, today=date(2026, 7, 8))

                self.assertIsNotNone(market)
                assert market is not None
                self.assertEqual(market.bucket.lower_f, 85.0)
                self.assertIsNone(market.bucket.upper_f)
                self.assertEqual(market.bucket.bucket_type, "above")

    def test_spelled_out_temperature_units_parse_correctly(self) -> None:
        celsius = recorder.parse_temperature_bucket(
            "Will the highest temperature in Seoul be 30 degrees Celsius on July 9?"
        )
        fahrenheit = recorder.parse_temperature_bucket(
            "Will the highest temperature in New York be 86 degrees Fahrenheit on July 9?"
        )

        self.assertIsNotNone(celsius)
        self.assertIsNotNone(fahrenheit)
        assert celsius is not None
        assert fahrenheit is not None
        self.assertAlmostEqual(celsius[0] or 0.0, 86.0)
        self.assertAlmostEqual((celsius[1] or 0.0) - (celsius[0] or 0.0), 1.7982)
        self.assertAlmostEqual(fahrenheit[0] or 0.0, 86.0)
        self.assertAlmostEqual((fahrenheit[1] or 0.0) - (fahrenheit[0] or 0.0), 0.999)

    def test_celsius_compact_and_glyph_exact_buckets_use_source_unit_width(self) -> None:
        for token, expected_lower in (("30C", 86.0), ("30°C", 86.0), ("37C", 98.6), ("37°C", 98.6)):
            with self.subTest(token=token):
                raw = {
                    "id": f"market-{token}",
                    "slug": f"seoul-high-{token.lower()}",
                    "question": f"Will the highest temperature in Seoul be {token} on July 9?",
                    "description": "The resolution source measures temperatures in whole degrees Celsius.",
                    "outcomes": '["Yes", "No"]',
                    "clobTokenIds": '["yes-token", "no-token"]',
                }

                market = recorder.parse_weather_market(raw, today=date(2026, 7, 8))

                self.assertIsNotNone(market)
                assert market is not None
                self.assertEqual(market.bucket.bucket_type, "exact")
                self.assertEqual(market.bucket.resolution_unit, "C")
                self.assertEqual(market.bucket.resolution_precision, 1.0)
                self.assertAlmostEqual(market.bucket.lower_f or 0.0, expected_lower)
                self.assertAlmostEqual((market.bucket.upper_f or 0.0) - (market.bucket.lower_f or 0.0), 1.7982)

    def test_celsius_and_fahrenheit_ranges_parse_in_source_units(self) -> None:
        celsius = recorder.parse_temperature_bucket("Will the high in Seoul be between 30-31°C on July 9?")
        fahrenheit = recorder.parse_temperature_bucket("Will the high in New York be between 85-90°F on July 9?")
        adjacent_fahrenheit = recorder.parse_binary_temperature_bucket(
            "Will the high in New York be between 85-86°F on July 9?",
            outcomes=["Yes", "No"],
            token_ids=["yes-token", "no-token"],
            prices=[],
        )

        self.assertIsNotNone(celsius)
        self.assertIsNotNone(fahrenheit)
        self.assertIsNotNone(adjacent_fahrenheit)
        assert celsius is not None
        assert fahrenheit is not None
        assert adjacent_fahrenheit is not None
        self.assertAlmostEqual(celsius[0] or 0.0, 86.0)
        self.assertAlmostEqual(celsius[1] or 0.0, 87.8)
        self.assertAlmostEqual(fahrenheit[0] or 0.0, 85.0)
        self.assertAlmostEqual(fahrenheit[1] or 0.0, 90.0)
        self.assertEqual(adjacent_fahrenheit.bucket_type, "exact")

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
                    model_runner=_shadow_test_runner(probability=0.02),
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
        parity = summary["shadow_scoring_parity"]
        self.assertTrue(parity["gate_valid_for_shadow_deployment"])
        self.assertEqual(parity["score_family_counts"], {RUNNER_SCORE_FAMILY: 1})
        self.assertEqual(parity["legacy_live_open_meteo_adapter_score_count"], 0)
        self.assertEqual(parity["score_cache"]["record_count"], 1)
        self.assertEqual(parity["firewall_violation_count"], 0)
        self.assertEqual(parity["missing_model_output_count"], 0)
        self.assertEqual(
            parity["runner_metadata"]["runner_name"],
            "ObservedHighGuardedShrinkageWeatherModelRunner",
        )
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
        event_payload = json.loads(events[0])
        self.assertEqual(event_payload["model_output"]["score_family"], RUNNER_SCORE_FAMILY)
        self.assertIn("runner_metadata", event_payload["model_output"]["metadata"])
        self.assertNotIn("old_no_tail_shadow_live_forecast_adapter", events[0])
        self.assertIn("observed_NO_best_ask_snapshot_not_a_fill_and_not_truth", events[0])
        self.assertIn("old_high_confidence_no_tail_entry", events[0])

    def test_refresh_pending_would_trade_settlements_excludes_proposed_only_uma_market(self) -> None:
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
                            "shadow_scoring_parity": {"gate_valid_for_shadow_deployment": True},
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

        self.assertEqual(summary["settlement_status_counts"], {"unresolved_or_ambiguous": 1})
        update = summary["updates"][0]
        self.assertTrue(update["settlement"]["manual_review_required"])
        self.assertEqual(update["settlement"]["reason"], "gamma_uma_resolution_status_not_final")
        self.assertFalse(summary["real_order_placement_allowed"])
        self.assertEqual(summary["real_orders_placed"], 0)
        self.assertEqual(summary["pending_would_trade_clusters_scanned"], 1)
        self.assertEqual(summary["settlement_cluster_status_counts"], {"unresolved_or_ambiguous": 1})
        self.assertEqual(summary["resolved_would_trade_clusters"], 0)
        self.assertEqual(progress["resolved_would_trade_signals"], 0)
        self.assertEqual(progress["unresolved_would_trade_signals"], 1)
        self.assertEqual(progress["unique_market_side_signal_clusters"], 1)
        self.assertEqual(progress["unique_resolved_market_side_signal_clusters"], 0)
        self.assertEqual(progress["gate_resolved_would_trade_clusters"], 0)
        self.assertEqual(progress["forward_shadow_gate"]["status"], "not_enough_resolved_clusters")
        self.assertEqual(
            progress["gate_excluded_cluster_count_by_reason"],
            {"ambiguous_or_unfinalized_settlement": 1},
        )

    def test_refresh_pending_would_trade_settlements_resolves_final_gamma_market(self) -> None:
        event = _gate_event("decision-1", "2026-07-09", settlement={"status": "pending"})
        original_fetch = recorder._fetch_gamma_event_by_slug

        def fake_fetch(event_slug, *, raw_dir, fetched_at, cache_only):
            return (
                {
                    "markets": [
                        {
                            "active": True,
                            "closed": True,
                            "conditionId": "condition-1",
                            "id": "market-1",
                            "outcomePrices": '["0", "1"]',
                            "outcomes": '["Yes", "No"]',
                            "slug": "test-weather-event-32c",
                            "umaResolutionStatuses": '["resolved"]',
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
                _write_gate_artifact(report_dir, "20260708T000000Z", [event])

                summary = recorder.refresh_pending_would_trade_settlements(
                    report_dir=report_dir,
                    raw_dir=Path(directory) / "raw",
                    run_id="20260708T010000Z",
                    cache_only=False,
                    generated_at=datetime(2026, 7, 8, 1, tzinfo=timezone.utc),
                )
        finally:
            recorder._fetch_gamma_event_by_slug = original_fetch

        self.assertEqual(summary["settlement_status_counts"], {"resolved": 1})
        update = summary["updates"][0]
        self.assertEqual(update["settlement"]["canonical_label"], "NO")
        self.assertEqual(update["settlement"]["payout_for_shadow_side"], 1)
        self.assertTrue(update["settlement"]["payout_is_shadow_settlement_only"])
        self.assertEqual(update["settlement"]["resolved_at_utc"], "2026-07-08T01:00:00+00:00")

    def test_recent_resolved_cluster_recheck_dispute_exits_gate_valid_set(self) -> None:
        event = _gate_event("decision-1", "2026-07-09", settlement={"status": "pending"})
        original_fetch = recorder._fetch_gamma_event_by_slug

        def fake_fetch(event_slug, *, raw_dir, fetched_at, cache_only):
            return (
                {
                    "markets": [
                        {
                            "active": True,
                            "closed": True,
                            "conditionId": "condition-1",
                            "id": "market-1",
                            "outcomePrices": '["0", "1"]',
                            "outcomes": '["Yes", "No"]',
                            "slug": "test-weather-event-32c",
                            "umaResolutionStatuses": '["disputed"]',
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
                _write_gate_artifact(report_dir, "20260708T000000Z", [event])
                _write_settlement_refresh(
                    report_dir,
                    "20260708T000000Z",
                    event,
                    settlement={
                        "status": "resolved",
                        "canonical_label": "NO",
                        "payout_for_shadow_side": 1,
                        "resolved_at_utc": "2026-07-08T00:00:00+00:00",
                    },
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
                anomaly_events_path = Path(summary["artifacts"]["anomaly_events_jsonl"])
                anomaly_events = [
                    json.loads(line)
                    for line in anomaly_events_path.read_text(encoding="utf-8").splitlines()
                ]
        finally:
            recorder._fetch_gamma_event_by_slug = original_fetch

        self.assertEqual(summary["resolved_recheck_cluster_count"], 1)
        self.assertEqual(summary["settlement_recheck_anomaly_count"], 1)
        self.assertEqual(summary["settlement_status_counts"], {"excluded_or_manual_review": 1})
        self.assertEqual(anomaly_events[0]["event_type"], "settlement_recheck_anomaly")
        update = summary["updates"][0]
        self.assertEqual(update["settlement"]["status"], "excluded_or_manual_review")
        self.assertEqual(update["settlement"]["reason"], "resolved_cluster_recheck_no_longer_final")
        self.assertTrue(update["settlement"]["manual_review_required"])
        self.assertEqual(update["settlement_recheck_anomaly"]["severity"], "critical")
        self.assertEqual(progress["unique_resolved_market_side_signal_clusters"], 0)
        self.assertEqual(progress["gate_resolved_would_trade_clusters"], 0)
        self.assertEqual(
            progress["gate_excluded_cluster_count_by_reason"],
            {"manual_review_or_disputed_settlement": 1},
        )

    def test_resolved_cluster_outside_recheck_window_is_not_rechecked(self) -> None:
        event = _gate_event("decision-1", "2026-07-09", settlement={"status": "pending"})
        fetch_calls = 0
        original_fetch = recorder._fetch_gamma_event_by_slug

        def fake_fetch(event_slug, *, raw_dir, fetched_at, cache_only):
            nonlocal fetch_calls
            fetch_calls += 1
            return (
                {"markets": []},
                Path(raw_dir) / "gamma-event.json",
                "payload-1",
                f"https://gamma-api.polymarket.com/events/slug/{event_slug}",
            )

        recorder._fetch_gamma_event_by_slug = fake_fetch
        try:
            with tempfile.TemporaryDirectory() as directory:
                report_dir = Path(directory) / "reports"
                report_dir.mkdir()
                _write_gate_artifact(report_dir, "20260708T000000Z", [event])
                _write_settlement_refresh(
                    report_dir,
                    "20260705T010000Z",
                    event,
                    settlement={
                        "status": "resolved",
                        "canonical_label": "NO",
                        "payout_for_shadow_side": 1,
                        "resolved_at_utc": "2026-07-05T01:00:00+00:00",
                    },
                )

                summary = recorder.refresh_pending_would_trade_settlements(
                    report_dir=report_dir,
                    raw_dir=Path(directory) / "raw",
                    run_id="20260708T010000Z",
                    cache_only=False,
                    generated_at=datetime(2026, 7, 8, 1, tzinfo=timezone.utc),
                )
        finally:
            recorder._fetch_gamma_event_by_slug = original_fetch

        self.assertEqual(fetch_calls, 0)
        self.assertEqual(summary["pending_would_trade_clusters_scanned"], 0)
        self.assertEqual(summary["resolved_recheck_cluster_count"], 0)
        self.assertEqual(summary["settlement_recheck_anomaly_count"], 0)
        self.assertEqual(summary["updates"], [])

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
                        "shadow_scoring_parity": {"gate_valid_for_shadow_deployment": True},
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

    def test_forward_gate_uses_economic_pnl_not_coin_flip_hit_rate(self) -> None:
        not_enough = recorder._evaluate_forward_shadow_gate([])
        loss_making_19_of_30 = recorder._evaluate_forward_shadow_gate(
            _gate_clusters(wins=19, losses=11, entry_price=0.82)
        )
        inconclusive = recorder._evaluate_forward_shadow_gate(
            _gate_clusters(wins=24, losses=6, entry_price=0.75)
        )
        supportive = recorder._evaluate_forward_shadow_gate(
            _gate_clusters(wins=27, losses=3, entry_price=0.70),
            forward_log_days=30,
        )

        self.assertEqual(not_enough["status"], "not_enough_resolved_clusters")
        self.assertEqual(loss_making_19_of_30["status"], "failed")
        self.assertLess(loss_making_19_of_30["total_net_pnl_usd"], 0.0)
        self.assertGreater(loss_making_19_of_30["price_weighted_break_even_hit_rate"], 0.82)
        self.assertEqual(
            loss_making_19_of_30["hit_rate_wilson_95_vs_price_weighted_break_even"],
            "wilson_95_upper_below_price_weighted_break_even",
        )
        self.assertEqual(inconclusive["status"], "inconclusive")
        self.assertLess(inconclusive["economic_uncertainty"]["lower_mean_net_pnl_usd"], 0.0)
        self.assertGreater(inconclusive["economic_uncertainty"]["upper_mean_net_pnl_usd"], 0.0)
        self.assertEqual(supportive["status"], "supportive")
        self.assertGreater(supportive["economic_uncertainty"]["lower_mean_net_pnl_usd"], 0.0)
        self.assertFalse(supportive["production_candidate_requirements"]["resolved_cluster_count_met"])
        self.assertFalse(supportive["production_candidate_requirements"]["evidence_requirements_met"])
        self.assertFalse(supportive["live_permission_granted"])

    def test_forward_economic_gate_counts_no_fill_and_ineligible_as_zero_pnl(self) -> None:
        clusters = _gate_clusters(wins=27, losses=1, entry_price=0.70)
        clusters.extend(
            [
                {
                    "cluster_key": "no-fill",
                    "latest_status": "resolved",
                    "payout_for_shadow_side": 1,
                    "economic_observation": _gate_economic_observation(
                        entry_price=0.70,
                        execution_status="latency_no_fill",
                        available_size=0.0,
                    ),
                },
                {
                    "cluster_key": "ineligible",
                    "latest_status": "resolved",
                    "payout_for_shadow_side": 1,
                    "economic_observation": _gate_economic_observation(
                        entry_price=0.70,
                        execution_status="ineligible",
                        max_notional_usd=1.0,
                    ),
                },
            ]
        )

        result = recorder._evaluate_forward_shadow_gate(clusters)

        self.assertEqual(result["resolved_cluster_count"], 30)
        self.assertEqual(result["venue_executable_cluster_count"], 28)
        self.assertEqual(result["no_fill_cluster_count"], 1)
        self.assertEqual(result["ineligible_cluster_count"], 1)
        self.assertEqual(result["economically_evaluable_cluster_count"], 30)
        self.assertEqual(result["invalid_economic_cluster_count"], 0)

    def test_forward_economic_bootstrap_is_deterministic(self) -> None:
        first = recorder._evaluate_forward_shadow_gate(_gate_clusters(wins=27, losses=3, entry_price=0.70))
        second = recorder._evaluate_forward_shadow_gate(_gate_clusters(wins=27, losses=3, entry_price=0.70))

        self.assertEqual(first["economic_uncertainty"], second["economic_uncertainty"])

    def test_forward_economic_gate_applies_recorded_latency_worse_price(self) -> None:
        event = _gate_event(
            "latency-price",
            "2026-07-09",
            settlement={"status": "resolved", "payout_for_shadow_side": 1},
        )
        event["execution_economics"] = {
            "execution_status": "venue_executable",
            "side": "NO",
            "order_type": "FOK",
            "order_action": "BUY",
            "entry_price": 0.80,
            "latency_entry_price": 0.84,
            "requested_shares": 5.0,
            "venue_executable_notional_usd": 4.20,
            "min_order_size_shares": 5.0,
            "tick_size": 0.01,
            "available_size_at_price": 20.0,
            "weather_taker_fee_rate": 0.05,
            "fee_rate_source": "fixture_current_market_info",
        }
        observation = recorder._economic_observation_from_event(event)
        record = recorder._economic_record_for_cluster(
            {
                "cluster_key": "latency-price",
                "latest_status": "resolved",
                "payout_for_shadow_side": 1,
                "economic_observation": observation,
            }
        )

        self.assertEqual(observation["entry_price"], 0.80)
        self.assertEqual(observation["conservative_entry_price"], 0.84)
        self.assertTrue(record["conservative_price_applied"])
        self.assertAlmostEqual(record["entry_cost_including_fees_usd"], 4.2336)

    def test_forward_economic_gate_uses_full_book_fok_walk_when_required(self) -> None:
        event = _gate_event(
            "full-book-fok",
            "2026-07-09",
            settlement={"status": "resolved", "payout_for_shadow_side": 1},
        )
        event["execution_economics"] = {
            "side": "NO",
            "order_type": "FOK",
            "order_action": "BUY",
            "entry_price": 0.80,
            "worst_executable_price": 0.82,
            "venue_executable_notional_usd": 2.0,
            "min_order_size_shares": 5.0,
            "tick_size": 0.01,
            "weather_taker_fee_rate": 0.05,
            "fee_rate_source": "fixture_current_market_info",
            "available_size_at_price": 1.0,
            "best_ask": 0.80,
            "book_provenance": "fixture-current-book",
            "quote_timestamp_utc": "2026-07-08T02:22:43+00:00",
            "ask_levels": [
                {"price": 0.80, "size": 1.0},
                {"price": 0.82, "size": 2.0},
                {"price": 0.83, "size": 100.0},
            ],
            "full_book_complete": True,
            "require_full_book_execution": True,
        }

        observation = recorder._economic_observation_from_event(event)
        record = recorder._economic_record_for_cluster(
            {
                "cluster_key": "full-book-fok",
                "latest_status": "resolved",
                "payout_for_shadow_side": 1,
                "economic_observation": observation,
            }
        )

        assessment = observation["venue_execution_assessment"]
        self.assertTrue(assessment["eligible"])
        self.assertTrue(assessment["all_or_none_fill"])
        self.assertEqual(assessment["levels_consumed"], 2)
        self.assertAlmostEqual(assessment["fee_usd"], 0.0188)
        self.assertAlmostEqual(record["entry_cost_including_fees_usd"], 2.0188)
        self.assertEqual(
            record["execution_basis"],
            "full_order_book_fok_buy_usdc_walk",
        )

    def test_forward_economic_gate_requires_levels_when_top_cannot_cover(self) -> None:
        event = _gate_event(
            "missing-full-book",
            "2026-07-09",
            settlement={"status": "resolved", "payout_for_shadow_side": 1},
        )
        event["execution_economics"] = {
            "side": "NO",
            "order_type": "FOK",
            "order_action": "BUY",
            "entry_price": 0.80,
            "worst_executable_price": 0.82,
            "venue_executable_notional_usd": 2.0,
            "min_order_size_shares": 5.0,
            "tick_size": 0.01,
            "weather_taker_fee_rate": 0.05,
            "fee_rate_source": "fixture_current_market_info",
            "available_size_at_price": 1.0,
            "best_ask": 0.80,
            "book_provenance": "fixture-current-book",
            "quote_timestamp_utc": "2026-07-08T02:22:43+00:00",
            "require_full_book_execution": True,
        }

        observation = recorder._economic_observation_from_event(event)
        assessment = observation["venue_execution_assessment"]

        self.assertFalse(assessment["eligible"])
        self.assertIn("missing_full_order_book_levels", assessment["reasons"])
        self.assertIn(
            "full_order_book_completeness_unproven",
            assessment["reasons"],
        )

    def test_forward_economic_gate_top_level_can_prove_full_fok(self) -> None:
        event = _gate_event(
            "top-level-fok",
            "2026-07-09",
            settlement={"status": "resolved", "payout_for_shadow_side": 1},
        )
        event["execution_economics"] = {
            "side": "NO",
            "order_type": "FOK",
            "order_action": "BUY",
            "entry_price": 0.80,
            "worst_executable_price": 0.82,
            "venue_executable_notional_usd": 2.0,
            "min_order_size_shares": 5.0,
            "tick_size": 0.01,
            "weather_taker_fee_rate": 0.05,
            "fee_rate_source": "fixture_current_market_info",
            "available_size_at_price": 10.0,
            "best_ask": 0.80,
            "book_provenance": "fixture-current-book",
            "quote_timestamp_utc": "2026-07-08T02:22:43+00:00",
            "require_full_book_execution": True,
        }

        observation = recorder._economic_observation_from_event(event)
        assessment = observation["venue_execution_assessment"]

        self.assertTrue(assessment["eligible"])
        self.assertEqual(
            assessment["execution_basis"],
            "top_level_proves_full_fill",
        )
        self.assertEqual(assessment["full_book_level_count"], 0)
        self.assertTrue(assessment["all_or_none_fill"])

    def test_forward_economic_gate_rejects_future_execution_quote(self) -> None:
        event = _gate_event(
            "future-quote",
            "2026-07-09",
            settlement={"status": "resolved", "payout_for_shadow_side": 1},
        )
        event["execution_economics"] = {
            "side": "NO",
            "order_type": "FOK",
            "order_action": "BUY",
            "entry_price": 0.80,
            "worst_executable_price": 0.82,
            "venue_executable_notional_usd": 2.0,
            "min_order_size_shares": 5.0,
            "tick_size": 0.01,
            "weather_taker_fee_rate": 0.05,
            "fee_rate_source": "fixture_current_market_info",
            "available_size_at_price": 10.0,
            "best_ask": 0.80,
            "book_provenance": "fixture-current-book",
            "quote_timestamp_utc": "2026-07-08T02:22:46+00:00",
            "require_full_book_execution": True,
        }

        observation = recorder._economic_observation_from_event(event)
        assessment = observation["venue_execution_assessment"]

        self.assertFalse(observation["quote_timestamp_valid"])
        self.assertEqual(observation["quote_age_seconds"], -1.0)
        self.assertFalse(assessment["eligible"])
        self.assertIn("invalid_execution_quote_timestamp", assessment["reasons"])

    def test_forward_economic_gate_production_metadata_never_grants_permission(self) -> None:
        result = recorder._evaluate_forward_shadow_gate(
            _gate_clusters(wins=90, losses=10, entry_price=0.70),
            forward_log_days=30,
        )

        self.assertEqual(result["status"], "supportive")
        self.assertTrue(result["production_candidate_requirements"]["evidence_requirements_met"])
        self.assertEqual(
            result["production_candidate_requirements"]["status"],
            "evidence_requirements_met_manual_review_still_required",
        )
        self.assertFalse(result["production_candidate_requirements"]["live_permission_granted"])
        self.assertFalse(result["live_permission_granted"])

    def test_forward_economic_gate_accepts_recorded_subfive_fok_buy(self) -> None:
        observation = _gate_economic_observation(
            entry_price=0.82,
            shares=2.0 / 0.82,
            min_order_size=5.0,
        )
        record = recorder._economic_record_for_cluster(
            {
                "cluster_key": "subfive-fok-buy",
                "latest_status": "resolved",
                "payout_for_shadow_side": 1,
                "economic_observation": observation,
            }
        )

        self.assertTrue(record["economically_evaluable"])
        self.assertEqual(record["execution_status"], "venue_executable")
        self.assertLess(record["shares"], 5.0)
        self.assertEqual(
            record["venue_execution_assessment"]["minimum_size_rule_status"],
            "not_applied_v2_market_buy_amount_is_usdc_and_subminimum_match_observed",
        )

    def test_forward_economic_gate_fails_closed_on_missing_venue_metadata(self) -> None:
        for missing_field in ("fee_rate", "tick_size", "available_size_at_price"):
            with self.subTest(missing_field=missing_field):
                observation = dict(_gate_economic_observation(entry_price=0.82))
                observation[missing_field] = None
                observation["venue_execution_assessment"] = recorder.assess_venue_order(
                    constraints=recorder.VenueExecutionConstraints(
                        min_order_size_shares=observation["min_order_size_shares"],
                        tick_size=observation["tick_size"],
                        fee_rate=observation["fee_rate"],
                    ),
                    intended_shares=observation["shares"],
                    limit_price=observation["conservative_entry_price"],
                    available_depth_shares=observation["available_size_at_price"],
                    intended_notional_usd=observation["notional_usd"],
                    order_type="FOK",
                    side="BUY",
                ).as_report_dict()
                record = recorder._economic_record_for_cluster(
                    {
                        "cluster_key": f"missing-{missing_field}",
                        "latest_status": "resolved",
                        "payout_for_shadow_side": 1,
                        "economic_observation": observation,
                    }
                )
                self.assertFalse(record["economically_evaluable"])

    def test_forward_economic_gate_rejects_off_tick_and_nonmarket_buy(self) -> None:
        off_tick = _gate_economic_observation(entry_price=0.825, tick_size=0.01)
        nonmarket = _gate_economic_observation(entry_price=0.82, order_type="GTC")

        for label, observation in (("off-tick", off_tick), ("nonmarket", nonmarket)):
            with self.subTest(label=label):
                record = recorder._economic_record_for_cluster(
                    {
                        "cluster_key": label,
                        "latest_status": "resolved",
                        "payout_for_shadow_side": 1,
                        "economic_observation": observation,
                    }
                )
                self.assertFalse(record["economically_evaluable"])

    def test_forward_gate_excludes_pending_ambiguous_disputed_and_legacy_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report_dir = Path(directory) / "reports"
            report_dir.mkdir()
            valid_events = [
                _gate_event("win-1", "2026-07-09", settlement={"status": "resolved", "payout_for_shadow_side": 1}),
                _gate_event("loss-1", "2026-07-10", settlement={"status": "resolved", "payout_for_shadow_side": 0}),
                _gate_event("pending-1", "2026-07-11", settlement={"status": "pending"}),
                _gate_event(
                    "ambiguous-1",
                    "2026-07-12",
                    settlement={"status": "unresolved_or_ambiguous", "manual_review_required": True},
                ),
                _gate_event(
                    "disputed-1",
                    "2026-07-13",
                    settlement={"status": "excluded_or_manual_review", "manual_review_required": True},
                ),
            ]
            repeated = _gate_event(
                "win-2",
                "2026-07-09",
                settlement={"status": "resolved", "payout_for_shadow_side": 1},
            )
            valid_events.append(repeated)
            _write_gate_artifact(report_dir, "20260708T000000Z", valid_events)
            _write_gate_artifact(
                report_dir,
                "20260708T001000Z",
                [_gate_event("legacy-1", "2026-07-14", settlement={"status": "resolved", "payout_for_shadow_side": 1})],
                gate_valid=False,
                gate_reason="legacy_live_open_meteo_adapter_scores_present",
            )

            progress = recorder._forward_collection_progress(
                report_dir,
                current_summary={"generated_at_utc": "2026-07-08T00:20:00+00:00"},
            )

        self.assertEqual(progress["would_trade_signals"], 7)
        self.assertEqual(progress["unique_market_side_signal_clusters"], 5)
        self.assertEqual(progress["gate_resolved_would_trade_clusters"], 2)
        self.assertEqual(progress["gate_winning_would_trade_clusters"], 1)
        self.assertEqual(progress["gate_losing_would_trade_clusters"], 1)
        self.assertEqual(progress["forward_shadow_gate"]["status"], "not_enough_resolved_clusters")
        self.assertEqual(
            progress["gate_excluded_cluster_count_by_reason"],
            {
                "ambiguous_or_unfinalized_settlement": 1,
                "manual_review_or_disputed_settlement": 1,
                "pending_settlement": 1,
                "legacy_pre_fix_score_family_artifact": 1,
            },
        )
        self.assertEqual(
            progress["gate_excluded_raw_detection_count_by_reason"],
            {"legacy_pre_fix_score_family_artifact": 1},
        )

    def test_station_fetch_falls_back_to_recent_cache_after_source_error(self) -> None:
        city = recorder.CityConfig("Test City", "TC", 0.0, 0.0, "UTC", "KTCY")
        target_date = date(2026, 7, 9)
        fetched_at = datetime(2026, 7, 9, 18, tzinfo=timezone.utc)

        original_fetch = recorder._get_or_fetch_raw_payload
        original_parse = recorder.parse_iem_csv_payload
        calls = {"fetches": 0}

        def fake_fetch(*, cache_dir, config, request_url, start, end, phase, fetched_at, cache_only):
            calls["fetches"] += 1
            if calls["fetches"] > 1:
                raise RuntimeError("rate limited")
            station_dir = Path(cache_dir) / config.station_id
            station_dir.mkdir(parents=True, exist_ok=True)
            cache_path = station_dir / f"{config.station_id}-{phase}-{start.isoformat()}-{end.isoformat()}-fake.csv"
            cache_path.write_bytes(b"fake station csv")
            return b"fake station csv", cache_path, "payload-1", "downloaded"

        def fake_parse(*args, **kwargs):
            return (
                1,
                [
                    SimpleNamespace(
                        temperature_f=81.0,
                        local_date=target_date.isoformat(),
                        observed_at_utc="2026-07-09T17:00:00+00:00",
                    )
                ],
            )

        recorder._get_or_fetch_raw_payload = fake_fetch
        recorder.parse_iem_csv_payload = fake_parse
        try:
            with tempfile.TemporaryDirectory() as directory:
                base = Path(directory) / "work/logs/live_forward_shadow_intraday/raw"
                first = recorder.fetch_station_observations(
                    city,
                    target_date,
                    raw_dir=base / "run-1/poll-001/station_observations",
                    fetched_at=fetched_at,
                    cache_only=False,
                )
                second = recorder.fetch_station_observations(
                    city,
                    target_date,
                    raw_dir=base / "run-2/poll-001/station_observations",
                    fetched_at=fetched_at,
                    cache_only=False,
                )
        finally:
            recorder._get_or_fetch_raw_payload = original_fetch
            recorder.parse_iem_csv_payload = original_parse

        self.assertEqual(first.status, "downloaded")
        self.assertEqual(second.status, "recent_cache_fallback_after_RuntimeError")
        self.assertEqual(second.high_so_far_f, 81.0)
        self.assertIn("RuntimeError: rate limited", second.error or "")


def _shadow_test_runner(probability: float) -> ObservedHighGuardedShrinkageWeatherModelRunner:
    def baseline_provider(_context: WeatherModelDecisionContext):
        return probability, {
            "open_meteo.kernel_tight": probability,
            "open_meteo.normal_parametric": min(0.99, probability + 0.02),
        }

    runner = ObservedHighGuardedShrinkageWeatherModelRunner(
        historical_rows=[],
        baseline_probability_provider=baseline_provider,
    )

    def fixture_training_summary(*, as_of_utc, staleness_warning_days=3):
        del staleness_warning_days
        corpus_date = (as_of_utc - timedelta(days=1)).date().isoformat()
        city_coverage = {
            "normalized_city": "Test City, TC",
            "required_target_dates": [corpus_date],
            "complete_target_dates": [corpus_date],
            "incomplete_target_dates": [],
            "latest_required_target_date": corpus_date,
            "latest_complete_target_date": corpus_date,
            "operationally_fresh": True,
            "source_coverage": {
                "forecast": {"missing_target_dates": []},
                "realized": {"missing_target_dates": []},
            },
        }
        return {
            "training_corpus_start_date": "2026-01-01",
            "training_corpus_configured_end_date": corpus_date,
            "training_corpus_end_date": corpus_date,
            "training_row_count": 1,
            "training_corpus_loaded_row_count": 1,
            "training_corpus_as_of_utc": as_of_utc.isoformat(),
            "training_corpus_staleness_days": 1,
            "training_corpus_staleness_warning": False,
            "training_corpus_staleness_warning_threshold_days": 3,
            "warning_flags": (),
            "training_corpus_coverage_available": True,
            "training_corpus_operationally_fresh": True,
            "training_corpus_operational_coverage": {
                "coverage_available": True,
                "operationally_fresh": True,
                "city_coverage": [city_coverage],
            },
            "label_availability_lag_hours": 12.0,
        }

    runner.training_corpus_summary = fixture_training_summary
    return runner


def _gate_economic_observation(
    *,
    entry_price: float,
    shares: float = 5.0,
    min_order_size: float = 5.0,
    tick_size: float = 0.01,
    fee_rate: float = 0.05,
    available_size: float = 100.0,
    order_type: str = "FOK",
    execution_status: str = "venue_executable",
    max_notional_usd: float | None = None,
) -> dict[str, object]:
    notional = shares * entry_price
    assessment = recorder.assess_venue_order(
        constraints=recorder.VenueExecutionConstraints(
            min_order_size_shares=min_order_size,
            tick_size=tick_size,
            fee_rate=fee_rate,
        ),
        intended_shares=shares,
        limit_price=entry_price,
        available_depth_shares=available_size,
        intended_notional_usd=notional,
        max_notional_usd=max_notional_usd,
        order_type=order_type,
        side="BUY",
    )
    return {
        "execution_status": execution_status,
        "execution_status_explicit": True,
        "execution_reason": "test_fixture_explicit_execution_status",
        "side": "NO",
        "order_type": order_type,
        "order_action": "BUY",
        "entry_price": entry_price,
        "conservative_entry_price": entry_price,
        "shares": shares,
        "notional_usd": notional,
        "min_order_size_shares": min_order_size,
        "tick_size": tick_size,
        "available_size_at_price": available_size,
        "fee_rate": fee_rate,
        "venue_execution_assessment": assessment.as_report_dict(),
    }


def _gate_clusters(*, wins: int, losses: int, entry_price: float = 0.82) -> list[dict[str, object]]:
    clusters: list[dict[str, object]] = []
    for index in range(wins):
        clusters.append(
            {
                "cluster_key": f"win-{index}",
                "latest_status": "resolved",
                "payout_for_shadow_side": 1,
                "economic_observation": _gate_economic_observation(entry_price=entry_price),
            }
        )
    for index in range(losses):
        clusters.append(
            {
                "cluster_key": f"loss-{index}",
                "latest_status": "resolved",
                "payout_for_shadow_side": 0,
                "economic_observation": _gate_economic_observation(entry_price=entry_price),
            }
        )
    return clusters


def _gate_event(decision_id: str, target_date: str, *, settlement: dict[str, object]) -> dict[str, object]:
    return {
        "decision": {
            "would_trade": True,
            "decision_reason": "old_high_confidence_no_tail_entry",
            "would_trade_signal": {
                "decision_id": decision_id,
                "limit_price": 0.87,
                "requested_notional_usd": 2.0,
                "requested_shares": 5.0,
                "action": "BUY",
                "order_type": "FOK",
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
            "min_order_size": 5.0,
            "tick_size": 0.01,
            "weather_taker_fee_rate": 0.05,
        },
        "market_discovery": {
            "bucket_label": "89.6F",
            "decision_id": decision_id,
            "decision_time_utc": "2026-07-08T02:22:45+00:00",
            "market_date_local": target_date,
            "market_id": "market-1",
            "normalized_city": "Test City, TC",
            "side": "NO",
            "slug": "test-weather-event-32c",
            "token_id": "no-token",
        },
        "parsed_contract": {
            "bucket_label": "89.6F",
            "city": "Test City, TC",
            "date": target_date,
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
        "settlement": settlement,
    }


def _write_gate_artifact(
    report_dir: Path,
    stamp: str,
    events: list[dict[str, object]],
    *,
    gate_valid: bool = True,
    gate_reason: str = "gate_valid",
) -> None:
    events_path = report_dir / f"{stamp}-old-no-tail-live-forward-shadow-intraday-events.jsonl"
    events_path.write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )
    summary_path = report_dir / f"{stamp}-old-no-tail-live-forward-shadow-intraday.json"
    summary_path.write_text(
        json.dumps(
            {
                "generated_at_utc": "2026-07-08T00:00:00+00:00",
                "decision_counts": {
                    "shadow_decisions": len(events),
                    "would_trade_signals": len(events),
                    "raw_would_trade_poll_detections": len(events),
                    "would_trade_signal_clusters": len(events),
                },
                "shadow_scoring_parity": {
                    "gate_valid_for_shadow_deployment": gate_valid,
                    "gate_validity_reason": gate_reason,
                },
                "artifacts": {"events_jsonl": str(events_path)},
            }
        ),
        encoding="utf-8",
    )


def _write_settlement_refresh(
    report_dir: Path,
    stamp: str,
    event: dict[str, object],
    *,
    settlement: dict[str, object],
) -> None:
    identifiers = recorder._settlement_identifiers_from_event(event)
    cluster_key = recorder._would_trade_cluster_key(identifiers)
    refresh_path = report_dir / f"{stamp}-old-no-tail-live-forward-shadow-settlement-refresh.json"
    refresh_path.write_text(
        json.dumps(
            {
                "schema_version": recorder.SETTLEMENT_REFRESH_SCHEMA_VERSION,
                "run_id": stamp,
                "generated_at_utc": datetime.strptime(stamp, "%Y%m%dT%H%M%SZ")
                .replace(tzinfo=timezone.utc)
                .isoformat(),
                "updates": [
                    {
                        "decision_id": identifiers["decision_id"],
                        "decision_ids": [identifiers["decision_id"]],
                        "cluster_key": cluster_key,
                        "market": dict(identifiers),
                        "settlement": settlement,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    unittest.main()
