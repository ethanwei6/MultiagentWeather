from __future__ import annotations

import sqlite3
import tempfile
import unittest
import json
from datetime import date, datetime, timezone
from pathlib import Path

from daily_weather_pipeline.cli import (
    PipelinePaths,
    attach_settlement_labels,
    _bucket_info,
    _clear_decision_snapshots_for_rebuild,
    _compact_quote_snapshot,
    _compact_failure_class,
    _decision_times_for_market,
    _ensure_compact_price_schema,
    _ensure_decision_schema,
    _ensure_market_schema,
    _ensure_price_schema,
    _ensure_weather_schema,
    _filter_price_jobs,
    _filter_compact_jobs,
    _forecast_run_time,
    _hourly_temperature_series,
    _observed_high_snapshot,
    _populate_observed_high_snapshots,
    _sync_compact_status_from_wide,
    _weather_features,
)
from daily_weather_pipeline.global_quote_addon import (
    _global_failure_class,
    _raise_if_projection_requires_acknowledgement,
    select_phase1_available_jobs,
    select_token_day_jobs,
    write_global_projection_report,
    write_global_scope_options_report,
)


class DummyBucket:
    def __init__(self, lower_f, upper_f):
        self.lower_f = lower_f
        self.upper_f = upper_f


class DataPipelineTests(unittest.TestCase):
    def test_global_phase1_available_job_selection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            discovery = root / "discovery.csv"
            sample = root / "sample.json"
            discovery.write_text(
                "\n".join(
                    [
                        "source,source_path,source_row_index,market_id,slug,event_slug,event_title,question,status,temperature_metric,city,market_date_local,end_date_utc,created_at_utc,settled_at_utc,station_id,station_text,resolution_source_type,current_registry_status,current_registry_timezone,proposed_timezone,proposed_latitude,proposed_longitude,geocoding_need,yes_token_id,no_token_id,resolved_yes,quotes_from,quotes_to,has_telonex_quote_window,bucket_type,operator,threshold_low_f,threshold_high_f,units",
                        "telonex,path,75,market-a,slug-a,event-a,Highest temperature in Karachi on June 23?,Will the highest temperature in Karachi be 37°C on June 23?,resolved,high,Karachi,2026-06-23,2026-06-23T12:00:00+00:00,2026-06-21T05:03:44+00:00,2026-06-23T19:19:35+00:00,OPKC,Masroor,wunderground,missing,,Asia/Karachi,,,verify,yes-a,no-a,False,2026-06-21,2026-06-24,True,range,between,98.6,100.4,C",
                        "telonex,path,76,market-b,slug-b,event-b,Lowest temperature in Karachi on June 9?,Will the lowest temperature in Karachi be 26°C on June 9?,resolved,low,Karachi,2027-06-09,2026-06-09T00:00:00+00:00,2026-06-08T18:21:48+00:00,2026-06-09T22:21:16+00:00,OPKC,Masroor,wunderground,missing,,Asia/Karachi,,,verify,yes-b,no-b,False,2026-06-08,2026-06-10,True,range,between,78.8,80.6,C",
                    ]
                ),
                encoding="utf-8",
            )
            sample.write_text(
                json.dumps(
                    {
                        "results": [
                            {
                                "status": "available",
                                "yes_token_id": "yes-a",
                                "question": "Will the highest temperature in Karachi be 37°C on June 23?",
                            },
                            {
                                "status": "available",
                                "yes_token_id": "yes-b",
                                "question": "Will the lowest temperature in Karachi be 26°C on June 9?",
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )

            jobs = select_phase1_available_jobs(discovery, sample, limit_jobs=4)

        self.assertEqual([job["token_id"] for job in jobs], ["yes-a", "no-a", "yes-b", "no-b"])
        self.assertEqual([job["side"] for job in jobs], ["YES", "NO", "YES", "NO"])
        self.assertEqual(jobs[0]["city"], "Karachi")
        self.assertEqual(jobs[2]["temperature_metric"], "low")

    def test_global_token_day_selection_skips_existing_status(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            discovery = root / "discovery.csv"
            compact_db = root / "global.sqlite"
            discovery.write_text(
                "\n".join(
                    [
                        "source,source_path,source_row_index,market_id,slug,event_slug,event_title,question,status,temperature_metric,city,market_date_local,end_date_utc,created_at_utc,settled_at_utc,station_id,station_text,resolution_source_type,current_registry_status,current_registry_timezone,proposed_timezone,proposed_latitude,proposed_longitude,geocoding_need,yes_token_id,no_token_id,resolved_yes,quotes_from,quotes_to,has_telonex_quote_window,bucket_type,operator,threshold_low_f,threshold_high_f,units",
                        "telonex,path,75,market-a,slug-a,event-a,Highest temperature in Karachi on June 23?,Will the highest temperature in Karachi be 37°C on June 23?,resolved,high,Karachi,2026-06-23,2026-06-23T12:00:00+00:00,2026-06-21T05:03:44+00:00,2026-06-23T19:19:35+00:00,OPKC,Masroor,wunderground,missing,,Asia/Karachi,,,verify,yes-a,no-a,False,2026-06-21,2026-06-22,True,range,between,98.6,100.4,C",
                    ]
                ),
                encoding="utf-8",
            )
            with sqlite3.connect(compact_db) as conn:
                _ensure_compact_price_schema(conn)
                conn.execute(
                    "INSERT INTO compact_tokens(token_ref, token_id, market_id, universe_row_id, side) VALUES (1, 'yes-a', 'market-a', 'global:75', 'YES')"
                )
                start_ts = int(datetime(2026, 6, 21, tzinfo=timezone.utc).timestamp())
                end_ts = int(datetime(2026, 6, 22, tzinfo=timezone.utc).timestamp()) - 1
                conn.execute(
                    """
                    INSERT INTO compact_price_fetch_status(
                        token_ref, start_ts, end_ts, status, rows_written, source_rows_seen,
                        duplicate_rows_seen, conflict_rows_seen, error, retryable, updated_at_utc
                    )
                    VALUES (1, ?, ?, 'complete', 10, 10, 0, 0, NULL, 0, '2026-07-07T00:00:00+00:00')
                    """,
                    (start_ts, end_ts),
                )

            jobs = select_token_day_jobs(discovery, compact_db=compact_db, limit_jobs=4)

        self.assertEqual([job["token_id"] for job in jobs], ["no-a", "yes-a", "no-a"])
        self.assertEqual([job["side"] for job in jobs], ["NO", "YES", "NO"])
        self.assertEqual(len(jobs), 3)
        self.assertEqual(jobs[0]["start_ts"], int(datetime(2026, 6, 21, tzinfo=timezone.utc).timestamp()))
        self.assertEqual(jobs[1]["start_ts"], int(datetime(2026, 6, 22, tzinfo=timezone.utc).timestamp()))

    def test_global_failure_classifier_does_not_read_asset_id_as_auth(self) -> None:
        error = (
            "RuntimeError: HTTP 404 for https://api.telonex.io/v1/downloads/polymarket/quotes/"
            "2026-04-10?asset_id=69659477885944328446785987403969580652638789456591358445872575554236683213471"
        )
        self.assertEqual(_global_failure_class(error), "http_404")

    def test_global_projection_report_uses_selector_and_batch_rates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            discovery = root / "discovery.csv"
            compact_db = root / "global.sqlite"
            decision_db = root / "decision.sqlite"
            raw_cache = root / "cache"
            report_dir = root / "reports"
            raw_cache.mkdir()
            report_dir.mkdir()
            discovery.write_text(
                "\n".join(
                    [
                        "source,source_path,source_row_index,market_id,slug,event_slug,event_title,question,status,temperature_metric,city,market_date_local,end_date_utc,created_at_utc,settled_at_utc,station_id,station_text,resolution_source_type,current_registry_status,current_registry_timezone,proposed_timezone,proposed_latitude,proposed_longitude,geocoding_need,yes_token_id,no_token_id,resolved_yes,quotes_from,quotes_to,has_telonex_quote_window,bucket_type,operator,threshold_low_f,threshold_high_f,units",
                        "telonex,path,75,market-a,slug-a,event-a,Highest temperature in Karachi on June 23?,Will the highest temperature in Karachi be 37°C on June 23?,resolved,high,Karachi,2026-06-23,2026-06-23T12:00:00+00:00,2026-06-21T05:03:44+00:00,2026-06-23T19:19:35+00:00,OPKC,Masroor,wunderground,missing,,Asia/Karachi,,,verify,yes-a,no-a,False,2026-06-21,2026-06-22,True,range,between,98.6,100.4,C",
                    ]
                ),
                encoding="utf-8",
            )
            with sqlite3.connect(compact_db) as conn:
                _ensure_compact_price_schema(conn)
                conn.execute(
                    "INSERT INTO compact_tokens(token_ref, token_id, market_id, universe_row_id, side) VALUES (1, 'yes-a', 'market-a', 'global:75', 'YES')"
                )
                start_ts = int(datetime(2026, 6, 21, tzinfo=timezone.utc).timestamp())
                end_ts = int(datetime(2026, 6, 22, tzinfo=timezone.utc).timestamp()) - 1
                conn.execute(
                    """
                    INSERT INTO compact_price_fetch_status(
                        token_ref, start_ts, end_ts, status, rows_written, source_rows_seen,
                        duplicate_rows_seen, conflict_rows_seen, error, retryable, updated_at_utc
                    )
                    VALUES (1, ?, ?, 'complete', 10, 10, 0, 0, NULL, 0, '2026-07-07T00:00:00+00:00')
                    """,
                    (start_ts, end_ts),
                )
            (report_dir / "20260707T000000Z-global-weather-token-day-batch.json").write_text(
                json.dumps(
                    {
                        "jobs_attempted": 10,
                        "jobs_completed": 8,
                        "jobs_failed": 2,
                        "quote_rows_written": 100,
                        "db_growth_bytes": 1000,
                        "elapsed_seconds": 3600,
                    }
                ),
                encoding="utf-8",
            )

            report = write_global_projection_report(
                compact_db=compact_db,
                decision_db=decision_db,
                discovery_csv=discovery,
                raw_cache_dir=raw_cache,
                report_dir=report_dir,
                approval_db_growth_gib=0.0,
                approval_active_days=0.0,
            )
            self.assertTrue(Path(report["report_paths"]["json"]).exists())

        self.assertEqual(report["selector_counts"]["existing_status_jobs"], 1)
        self.assertEqual(report["selector_counts"]["existing_selector_status_jobs"], 1)
        self.assertEqual(report["selector_counts"]["external_status_jobs"], 0)
        self.assertEqual(report["selector_counts"]["remaining_unattempted_jobs"], 3)
        self.assertEqual(report["selector_counts"]["total_selector_jobs"], 4)
        self.assertEqual(report["observed_batch_rates"]["active_jobs_per_hour"], 10.0)
        self.assertTrue(report["approval_required_before_more_online_backfill"])

    def test_global_projection_guard_requires_acknowledgement(self) -> None:
        report = {
            "approval_required_before_more_online_backfill": True,
            "projection": {
                "projected_active_days_remaining": 23.6,
                "projected_db_growth_gib": 328.6,
            },
            "report_paths": {"json": "projection.json"},
        }

        with self.assertRaisesRegex(RuntimeError, "exceeds approval thresholds"):
            _raise_if_projection_requires_acknowledgement(report, acknowledged=False)

        _raise_if_projection_requires_acknowledgement(report, acknowledged=True)

    def test_global_scope_options_report_summarizes_city_metric_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            discovery = root / "discovery.csv"
            compact_db = root / "global.sqlite"
            decision_db = root / "decision.sqlite"
            raw_cache = root / "cache"
            report_dir = root / "reports"
            raw_cache.mkdir()
            report_dir.mkdir()
            discovery.write_text(
                "\n".join(
                    [
                        "source,source_path,source_row_index,market_id,slug,event_slug,event_title,question,status,temperature_metric,city,market_date_local,end_date_utc,created_at_utc,settled_at_utc,station_id,station_text,resolution_source_type,current_registry_status,current_registry_timezone,proposed_timezone,proposed_latitude,proposed_longitude,geocoding_need,yes_token_id,no_token_id,resolved_yes,quotes_from,quotes_to,has_telonex_quote_window,bucket_type,operator,threshold_low_f,threshold_high_f,units",
                        "telonex,path,75,market-a,slug-a,event-a,Highest temperature in Karachi on June 23?,Will the highest temperature in Karachi be 37°C on June 23?,resolved,high,Karachi,2026-06-23,2026-06-23T12:00:00+00:00,2026-06-21T05:03:44+00:00,2026-06-23T19:19:35+00:00,OPKC,Masroor,wunderground,missing,,Asia/Karachi,,,verify,yes-a,no-a,False,2026-06-21,2026-06-22,True,range,between,98.6,100.4,C",
                        "telonex,path,76,market-b,slug-b,event-b,Lowest temperature in Karachi on June 23?,Will the lowest temperature in Karachi be 27°C on June 23?,resolved,low,Karachi,2026-06-23,2026-06-23T12:00:00+00:00,2026-06-21T05:03:44+00:00,2026-06-23T19:19:35+00:00,OPKC,Masroor,wunderground,missing,,Asia/Karachi,,,verify,yes-b,no-b,False,2026-06-21,2026-06-21,True,range,between,80.6,82.4,C",
                    ]
                ),
                encoding="utf-8",
            )
            with sqlite3.connect(compact_db) as conn:
                _ensure_compact_price_schema(conn)
                conn.execute(
                    "INSERT INTO compact_tokens(token_ref, token_id, market_id, universe_row_id, side) VALUES (1, 'yes-a', 'market-a', 'global:75', 'YES')"
                )
                start_ts = int(datetime(2026, 6, 21, tzinfo=timezone.utc).timestamp())
                end_ts = int(datetime(2026, 6, 22, tzinfo=timezone.utc).timestamp()) - 1
                conn.execute(
                    """
                    INSERT INTO compact_price_fetch_status(
                        token_ref, start_ts, end_ts, status, rows_written, source_rows_seen,
                        duplicate_rows_seen, conflict_rows_seen, error, retryable, updated_at_utc
                    )
                    VALUES (1, ?, ?, 'complete', 10, 10, 0, 0, NULL, 0, '2026-07-07T00:00:00+00:00')
                    """,
                    (start_ts, end_ts),
                )
            (report_dir / "20260707T000000Z-global-weather-token-day-batch.json").write_text(
                json.dumps(
                    {
                        "jobs_attempted": 10,
                        "jobs_completed": 8,
                        "jobs_failed": 2,
                        "quote_rows_written": 100,
                        "db_growth_bytes": 1000,
                        "elapsed_seconds": 3600,
                    }
                ),
                encoding="utf-8",
            )

            report = write_global_scope_options_report(
                compact_db=compact_db,
                decision_db=decision_db,
                discovery_csv=discovery,
                raw_cache_dir=raw_cache,
                report_dir=report_dir,
                top_n=5,
            )
            self.assertTrue(Path(report["report_paths"]["json"]).exists())

        self.assertEqual(report["overall"]["market_rows"], 2)
        self.assertEqual(report["overall"]["token_day_jobs"], 6)
        self.assertEqual(report["overall"]["complete_jobs"], 1)
        self.assertEqual(report["overall"]["unattempted_jobs"], 5)
        by_metric = {row["scope"]: row for row in report["scope_tables"]["by_temperature_metric"]}
        self.assertEqual(by_metric["high"]["token_day_jobs"], 4)
        self.assertEqual(by_metric["low"]["token_day_jobs"], 2)

    def test_market_schema_contains_expected_tables(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "market.sqlite"
            with sqlite3.connect(path) as conn:
                _ensure_market_schema(conn)
                tables = {
                    row[0]
                    for row in conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    )
                }
            self.assertIn("market_universe", tables)
            self.assertIn("market_tokens", tables)
            self.assertIn("market_parse_warnings", tables)

    def test_price_schema_initializes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "price.sqlite"
            with sqlite3.connect(path) as conn:
                _ensure_price_schema(conn)
                columns = {
                    row[1]
                    for row in conn.execute("PRAGMA table_info(token_quote_history)")
                }
            self.assertIn("source_cache_path", columns)
            self.assertIn("timestamp_utc", columns)

    def test_decision_schema_preserves_quote_audit_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "decision.sqlite"
            with sqlite3.connect(path) as conn:
                _ensure_decision_schema(conn)
                columns = {
                    row[1]
                    for row in conn.execute("PRAGMA table_info(decision_snapshots)")
                }
                tables = {
                    row[0]
                    for row in conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    )
                }
            self.assertIn("bid_size", columns)
            self.assertIn("ask_size", columns)
            self.assertIn("liquidity_depth", columns)
            self.assertIn("quote_source_cache_path", columns)
            self.assertIn("quote_data_quality_flags", columns)
            self.assertIn("lead_days", columns)
            self.assertIn("decision_hour_utc", columns)
            self.assertIn("decision_artifact_scope", tables)

    def test_decision_rebuild_clear_removes_out_of_window_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "decision.sqlite"
            with sqlite3.connect(path) as conn:
                _ensure_decision_schema(conn)
                conn.execute(
                    """
                    INSERT INTO decision_snapshots(decision_id, universe_row_id, market_date_local, decision_time_utc, created_at_utc)
                    VALUES
                      ('old', 'row-old', '2025-12-31', '2025-12-30T12:00:00+00:00', 'now'),
                      ('target', 'row-target', '2026-01-01', '2025-12-31T12:00:00+00:00', 'now')
                    """
                )
                stats = _clear_decision_snapshots_for_rebuild(
                    conn,
                    start=date(2026, 1, 1),
                    end=date(2026, 6, 29),
                )
                remaining = conn.execute("SELECT COUNT(*) FROM decision_snapshots").fetchone()[0]
            self.assertEqual(stats["total_rows"], 2)
            self.assertEqual(stats["target_window_rows"], 1)
            self.assertEqual(stats["out_of_target_window_rows"], 1)
            self.assertEqual(remaining, 0)

    def test_attach_settlement_labels_exposes_current_join_view(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = PipelinePaths(Path(directory))
            paths.backtest_data.mkdir(parents=True)
            labels_csv = Path(directory) / "labels.csv"
            labels_csv.write_text(
                "\n".join(
                    [
                        "universe_row_id,market_date_local,canonical_label,recommended_action,exclude_from_backtest_replay",
                        "row-target,2026-01-01,1,keep_polymarket_label,0",
                        "row-old,2025-12-31,0,exclude_or_manual_review,1",
                    ]
                )
                + "\n"
            )
            with sqlite3.connect(paths.decision_db) as conn:
                _ensure_decision_schema(conn)
                conn.execute(
                    """
                    INSERT INTO decision_snapshots(decision_id, universe_row_id, market_date_local, decision_time_utc, created_at_utc)
                    VALUES ('target', 'row-target', '2026-01-01', '2025-12-31T12:00:00+00:00', 'now')
                    """
                )
                conn.execute(
                    """
                    INSERT OR REPLACE INTO decision_artifact_scope(
                        scope_id, target_start_date, target_end_date, price_store,
                        max_staleness_minutes, built_at_utc, row_count
                    )
                    VALUES ('current', '2026-01-01', '2026-06-29', 'compact', 90, 'now', 1)
                    """
                )
                conn.commit()
            result = attach_settlement_labels(
                paths,
                start=date(2026, 1, 1),
                end=date(2026, 6, 29),
                labels_csv=labels_csv,
            )
            with sqlite3.connect(paths.decision_db) as conn:
                joined = conn.execute(
                    "SELECT canonical_label, recommended_action FROM decision_snapshots_current_with_labels"
                ).fetchone()
                label_count = conn.execute("SELECT COUNT(*) FROM canonical_settlement_labels").fetchone()[0]
            self.assertEqual(result["rows_imported"], 1)
            self.assertEqual(label_count, 1)
            self.assertEqual(joined, (1, "keep_polymarket_label"))

    def test_hourly_observed_high_uses_only_samples_before_decision(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            raw_path = Path(directory) / "realized.json"
            raw_path.write_text(
                json.dumps(
                    {
                        "hourly": {
                            "time": [
                                "2026-01-02T00:00",
                                "2026-01-02T12:00",
                                "2026-01-02T16:00",
                                "2026-01-02T23:00",
                            ],
                            "temperature_2m": [50.0, 72.0, 80.0, 85.0],
                        }
                    }
                )
            )
            with sqlite3.connect(":memory:") as conn:
                conn.row_factory = sqlite3.Row
                conn.execute(
                    """
                    CREATE TABLE realized_daily_weather (
                        source_payload_id TEXT,
                        raw_cache_path TEXT
                    )
                    """
                )
                conn.execute(
                    "INSERT INTO realized_daily_weather VALUES ('payload-a', ?)",
                    (str(raw_path),),
                )
                realized = conn.execute("SELECT * FROM realized_daily_weather").fetchone()

            snapshot = _observed_high_snapshot(
                city="New York",
                local_date="2026-01-02",
                zone_name="America/New_York",
                decision_time=datetime(2026, 1, 2, 22, 0, tzinfo=timezone.utc),
                realized=realized,
                series_cache={},
            )
            self.assertEqual(snapshot["observed_high_so_far_f"], 80.0)
            self.assertEqual(snapshot["observed_high_observed_at_utc"], "2026-01-02T21:00:00+00:00")
            self.assertEqual(snapshot["observed_high_sample_count"], 3)
            self.assertEqual(snapshot["observed_high_total_sample_count"], 4)
            self.assertEqual(snapshot["observed_high_is_final_at_decision"], 0)
            self.assertNotIn(
                "final_daily_aggregate_not_used",
                snapshot["observed_high_data_quality_flags"],
            )

    def test_observed_high_does_not_fall_back_to_final_daily_aggregate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            raw_path = Path(directory) / "realized.json"
            raw_path.write_text(
                json.dumps(
                    {
                        "daily": {
                            "time": ["2026-01-02"],
                            "temperature_2m_max": [99.0],
                        },
                        "hourly": {
                            "time": [],
                            "temperature_2m": [],
                        },
                    }
                )
            )
            with sqlite3.connect(":memory:") as conn:
                conn.row_factory = sqlite3.Row
                conn.execute(
                    """
                    CREATE TABLE realized_daily_weather (
                        source_payload_id TEXT,
                        raw_cache_path TEXT
                    )
                    """
                )
                conn.execute(
                    "INSERT INTO realized_daily_weather VALUES ('payload-b', ?)",
                    (str(raw_path),),
                )
                realized = conn.execute("SELECT * FROM realized_daily_weather").fetchone()

            snapshot = _observed_high_snapshot(
                city="New York",
                local_date="2026-01-02",
                zone_name="America/New_York",
                decision_time=datetime(2026, 1, 3, 5, 0, tzinfo=timezone.utc),
                realized=realized,
                series_cache={},
            )
            self.assertIsNone(snapshot["observed_high_so_far_f"])
            self.assertEqual(snapshot["observed_high_missing_reason"], "missing_hourly_temperature_observations")
            self.assertIn(
                "final_daily_aggregate_not_used",
                snapshot["observed_high_data_quality_flags"],
            )

    def test_populate_observed_high_snapshots_accepts_default_tuple_connection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = PipelinePaths(Path(directory))
            paths.backtest_data.mkdir(parents=True)
            raw_path = Path(directory) / "realized.json"
            raw_path.write_text(
                json.dumps(
                    {
                        "hourly": {
                            "time": ["2026-01-02T00:00", "2026-01-02T16:00"],
                            "temperature_2m": [50.0, 80.0],
                        }
                    }
                )
            )
            with sqlite3.connect(paths.weather_db) as weather_conn:
                _ensure_weather_schema(weather_conn)
                weather_conn.execute(
                    """
                    INSERT INTO realized_daily_weather(
                        realized_id, normalized_city, latitude, longitude, timezone,
                        local_date, source_name, source_payload_id, raw_cache_path,
                        fetch_timestamp_utc
                    )
                    VALUES ('realized-a', 'New York', 40.7, -74.0, 'America/New_York',
                            '2026-01-02', 'open_meteo_archive', 'payload-a', ?, 'now')
                    """,
                    (str(raw_path),),
                )
            with sqlite3.connect(paths.decision_db) as decision_conn:
                _ensure_decision_schema(decision_conn)
                decision_conn.execute(
                    """
                    INSERT INTO decision_snapshots(
                        decision_id, universe_row_id, normalized_city, timezone,
                        market_date_local, decision_time_utc, created_at_utc
                    )
                    VALUES ('decision-a', 'row-a', 'New York', 'America/New_York',
                            '2026-01-02', '2026-01-02T22:00:00+00:00', 'now')
                    """
                )
                stats = _populate_observed_high_snapshots(
                    paths,
                    decision_conn,
                    start=date(2026, 1, 1),
                    end=date(2026, 6, 29),
                )
                stored = decision_conn.execute(
                    "SELECT observed_high_so_far_f FROM observed_high_snapshots"
                ).fetchone()
            self.assertEqual(stats["rows_written"], 1)
            self.assertEqual(stored, (80.0,))

    def test_hourly_temperature_series_converts_local_times_to_utc(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            raw_path = Path(directory) / "realized.json"
            raw_path.write_text(
                json.dumps(
                    {
                        "hourly": {
                            "time": ["2026-01-02T00:00", "2026-01-02T16:00"],
                            "temperature_2m": [50.0, 80.0],
                        }
                    }
                )
            )
            series, error = _hourly_temperature_series(str(raw_path), "2026-01-02", "America/New_York")
            self.assertIsNone(error)
            self.assertEqual(
                [(instant.isoformat(), value) for instant, value in series],
                [
                    ("2026-01-02T05:00:00+00:00", 50.0),
                    ("2026-01-02T21:00:00+00:00", 80.0),
                ],
            )

    def test_price_job_filter_targets_retryable_cache_misses(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = PipelinePaths(Path(directory))
            paths.backtest_data.mkdir(parents=True)
            jobs = [
                {"token_id": "a", "start_ts": 1, "end_ts": 2},
                {"token_id": "b", "start_ts": 1, "end_ts": 2},
                {"token_id": "c", "start_ts": 1, "end_ts": 2},
            ]
            with sqlite3.connect(paths.price_db) as conn:
                _ensure_price_schema(conn)
                conn.execute(
                    """
                    INSERT INTO price_fetch_status(
                        token_id, start_ts, end_ts, status, rows_written, retryable, updated_at_utc
                    )
                    VALUES ('a', 1, 2, 'complete', 10, 0, 'now'),
                           ('b', 1, 2, 'cache_miss', 0, 1, 'now')
                    """
                )
            selected = _filter_price_jobs(paths, jobs, job_status="cache_miss", limit_jobs=1)
            self.assertEqual([job["token_id"] for job in selected], ["b"])
            retryable = _filter_price_jobs(paths, jobs, job_status="retryable", limit_jobs=0)
            self.assertEqual([job["token_id"] for job in retryable], ["b", "c"])

    def test_compact_quote_snapshot_is_point_in_time_and_auditable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "compact.sqlite"
            with sqlite3.connect(path) as conn:
                conn.row_factory = sqlite3.Row
                _ensure_compact_price_schema(conn)
                conn.execute(
                    "INSERT INTO compact_tokens(token_ref, token_id, market_id, universe_row_id, side) VALUES (1, 'token-a', 'market-a', 'row-a', 'YES')"
                )
                conn.execute(
                    "INSERT INTO compact_sources(source_ref, source_cache_path, source_payload_id, payload_sha256) VALUES (1, '/tmp/raw.parquet', 'payload-a', 'sha-a')"
                )
                conn.execute(
                    """
                    INSERT INTO compact_quote_history(
                        token_ref, timestamp_utc, bid, ask, mid, last_price, spread,
                        bid_size, ask_size, liquidity_depth, source_ref, data_quality_flags
                    )
                    VALUES
                      (1, '2026-01-02T11:59:00+00:00', 0.40, 0.44, 0.42, 0.41, 0.04, 100.0, 120.0, 220.0, 1, '[]'),
                      (1, '2026-01-02T12:01:00+00:00', 0.50, 0.54, 0.52, 0.51, 0.04, 200.0, 220.0, 420.0, 1, '[]')
                    """
                )
                quote = _compact_quote_snapshot(
                    conn,
                    "token-a",
                    datetime(2026, 1, 2, 12, 0, tzinfo=timezone.utc),
                    max_staleness_minutes=90,
                )
                stale_quote = _compact_quote_snapshot(
                    conn,
                    "token-a",
                    datetime(2026, 1, 2, 14, 0, tzinfo=timezone.utc),
                    max_staleness_minutes=30,
                )
            self.assertIsNotNone(quote)
            self.assertEqual(quote["timestamp_utc"], "2026-01-02T11:59:00+00:00")
            self.assertEqual(quote["source_cache_path"], "/tmp/raw.parquet")
            self.assertEqual(quote["bid_size"], 100.0)
            self.assertIsNone(stale_quote)

    def test_compact_status_sync_preserves_failed_retry_state(self) -> None:
        job = {"token_id": "token-a", "start_ts": 1, "end_ts": 2}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "compact.sqlite"
            with sqlite3.connect(path) as conn:
                conn.row_factory = sqlite3.Row
                _ensure_compact_price_schema(conn)
                conn.execute(
                    "INSERT INTO compact_tokens(token_ref, token_id, market_id, universe_row_id, side) VALUES (1, 'token-a', 'market-a', 'row-a', 'YES')"
                )
                conn.execute(
                    """
                    INSERT INTO compact_price_fetch_status(
                        token_ref, start_ts, end_ts, status, rows_written, retryable, updated_at_utc
                    )
                    VALUES (1, 1, 2, 'failed', 0, 1, 'now')
                    """
                )
                _sync_compact_status_from_wide(
                    conn,
                    [job],
                    {"token-a": 1},
                    {("token-a", 1, 2): {"status": "cache_miss", "error": None}},
                )
                status = conn.execute("SELECT status FROM compact_price_fetch_status").fetchone()["status"]
            self.assertEqual(status, "failed")

    def test_compact_failed_job_filter_can_target_transient_errors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = PipelinePaths(Path(directory))
            paths.backtest_data.mkdir(parents=True)
            jobs = [
                {"token_id": "transient", "start_ts": 1, "end_ts": 2},
                {"token_id": "http404", "start_ts": 1, "end_ts": 2},
                {"token_id": "missing", "start_ts": 1, "end_ts": 2},
                {"token_id": "ratelimit", "start_ts": 1, "end_ts": 2},
                {"token_id": "cache", "start_ts": 1, "end_ts": 2},
            ]
            with sqlite3.connect(paths.compact_price_db) as conn:
                _ensure_compact_price_schema(conn)
                conn.executemany(
                    "INSERT INTO compact_tokens(token_ref, token_id, market_id, universe_row_id, side) VALUES (?, ?, ?, ?, 'YES')",
                    [
                        (1, "transient", "market", "row-1"),
                        (2, "http404", "market", "row-2"),
                        (3, "missing", "market", "row-3"),
                        (4, "ratelimit", "market", "row-4"),
                        (5, "cache", "market", "row-5"),
                    ],
                )
                conn.executemany(
                    """
                    INSERT INTO compact_price_fetch_status(
                        token_ref, start_ts, end_ts, status, rows_written, retryable, error, updated_at_utc
                    )
                    VALUES (?, 1, 2, ?, 0, 1, ?, 'now')
                    """,
                    [
                        (1, "failed", "RuntimeError: <urlopen error [Errno 8] nodename nor servname provided, or not known>"),
                        (2, "failed", "RuntimeError: Telonex HTTP 404 for quotes"),
                        (3, "failed", "RuntimeError: Telonex cached missing file for polymarket/quotes/2026-01-07"),
                        (4, "failed", "RuntimeError: Request failed with HTTP 429 rate limit"),
                        (5, "cache_miss", None),
                    ],
                )

            transient = _filter_compact_jobs(
                paths,
                jobs,
                job_status="failed",
                failure_class="transient",
                limit_jobs=0,
            )
            source_unavailable = _filter_compact_jobs(
                paths,
                jobs,
                job_status="failed",
                failure_class="source_unavailable",
                limit_jobs=0,
            )
            rate_limit = _filter_compact_jobs(
                paths,
                jobs,
                job_status="failed",
                failure_class="rate_limit",
                limit_jobs=0,
            )
            unfiltered = _filter_compact_jobs(
                paths,
                jobs,
                job_status="failed",
                failure_class="all",
                limit_jobs=0,
            )

        self.assertEqual([job["token_id"] for job in transient], ["transient"])
        self.assertEqual([job["token_id"] for job in source_unavailable], ["http404", "missing"])
        self.assertEqual([job["token_id"] for job in rate_limit], ["ratelimit"])
        self.assertEqual([job["token_id"] for job in unfiltered], ["transient", "http404", "missing", "ratelimit"])
        self.assertEqual(_compact_failure_class("RuntimeError: HTTP 429 rate limit"), "rate_limit")
        self.assertEqual(_compact_failure_class("RuntimeError: unknown asset id"), "parser_or_token_mismatch")

    def test_bucket_shape_detection(self) -> None:
        self.assertEqual(_bucket_info(DummyBucket(80.0, 80.999))["bucket_type"], "exact")
        self.assertEqual(_bucket_info(DummyBucket(80.0, 85.0))["bucket_type"], "range")
        self.assertEqual(_bucket_info(DummyBucket(90.0, None))["bucket_type"], "above")
        self.assertEqual(_bucket_info(DummyBucket(None, 70.0))["bucket_type"], "below")

    def test_decision_times_use_live_utc_cadence_and_no_future_close(self) -> None:
        row = {
            "market_date_local": "2026-06-18",
            "timezone": "America/New_York",
            "start_at_utc": "2026-06-16T00:00:00+00:00",
            "close_time_utc": "2026-06-18T12:00:00+00:00",
        }
        decisions = _decision_times_for_market(row)
        self.assertTrue(decisions)
        self.assertTrue(all(item.hour in {0, 6, 12, 18} for item in decisions))
        self.assertTrue(all(item < datetime(2026, 6, 18, 12, tzinfo=timezone.utc) for item in decisions))

    def test_forecast_run_time_respects_availability_lag(self) -> None:
        run_time = _forecast_run_time(datetime(2026, 6, 18, 12, 10, tzinfo=timezone.utc))
        self.assertEqual(run_time.isoformat(), "2026-06-18T06:00:00+00:00")

    def test_weather_feature_extraction_from_open_meteo_shape(self) -> None:
        payload = {
            "daily": {
                "time": ["2026-06-18"],
                "temperature_2m_max": [91.5],
                "precipitation_sum": [0.1],
                "wind_speed_10m_max": [11.0],
            },
            "hourly": {
                "time": ["2026-06-18T00:00", "2026-06-18T01:00"],
                "temperature_2m": [82.0, 84.0],
                "apparent_temperature": [85.0, 87.0],
                "relative_humidity_2m": [60.0, 70.0],
                "dew_point_2m": [66.0, 67.0],
                "cloud_cover": [20.0, 40.0],
                "wind_speed_10m": [9.0, 10.0],
                "precipitation": [0.0, 0.02],
            },
        }
        features = _weather_features(payload, date(2026, 6, 18))
        self.assertEqual(features["temperature_2m_max"], 91.5)
        self.assertEqual(features["hourly_temperature_2m_max"], 84.0)
        self.assertEqual(features["relative_humidity_2m_mean"], 65.0)


if __name__ == "__main__":
    unittest.main()
