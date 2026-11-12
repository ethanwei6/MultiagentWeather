from __future__ import annotations

import hashlib
import json
import sqlite3
import tempfile
import unittest
from contextlib import contextmanager
from datetime import date, datetime, timezone
from pathlib import Path
from unittest.mock import patch

from corpus_fixture import fixture_weather_strategy

from daily_weather_pipeline.cli import (
    PipelinePaths,
    _connect,
    _insert_market_row,
    initialize_weather_corpus,
)
from daily_weather_pipeline.live_forward_shadow_recorder import parse_weather_market
from daily_weather_pipeline.weather_corpus_coverage import (
    coverage_allows_city_date,
    weather_corpus_coverage,
)
from daily_weather_pipeline.weather_corpus_extension import (
    extend_market_universe_from_gamma,
    extend_weather_history_for_labelable_pairs,
    gamma_market_to_universe_row,
    iter_cached_gamma_event_payloads,
    market_rows_from_gamma_event_payload,
    pair_label_available_at_utc,
    run_weather_corpus_extension,
)
from daily_weather_pipeline.weather_model_runner import (
    ObservedHighGuardedShrinkageWeatherModelRunner,
)


class WeatherCorpusExtensionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.enterContext(patch(
            "daily_weather_pipeline.weather_model_runner._load_weather_strategy",
            return_value=fixture_weather_strategy(),
        ))

    def test_gamma_market_row_matches_validation_builder_keys(self) -> None:
        raw = _gamma_raw_market(
            city="Beijing",
            date_text="July 7",
            slug_city="beijing",
            market_id="market-beijing-85",
            yes_token="yes-beijing-85",
            no_token="no-beijing-85",
            question="Will the highest temperature in Beijing be 85°F on July 7?",
        )
        market = parse_weather_market(raw, today=date(2026, 7, 9))
        self.assertIsNotNone(market)
        source_bytes = json.dumps(
            {"markets": [raw]}, indent=1, sort_keys=False
        ).encode("utf-8")
        source_file_sha256 = hashlib.sha256(source_bytes).hexdigest()

        row = gamma_market_to_universe_row(
            market,
            raw=raw,
            source_path=Path("/tmp/gamma-event.json"),
            payload_id="payload-gamma",
            source_file_sha256=source_file_sha256,
        )

        self.assertEqual(row["normalized_city"], "Beijing, CN")
        self.assertEqual(row["market_date_local"], "2026-07-07")
        self.assertEqual(row["timezone"], "Asia/Shanghai")
        self.assertEqual(row["threshold_low_f"], 85.0)
        self.assertEqual(row["bucket_type"], "exact")
        self.assertEqual(row["yes_token_id"], "yes-beijing-85")
        self.assertEqual(row["no_token_id"], "no-beijing-85")
        self.assertEqual(
            row["source_payload"]["payload_sha256"], source_file_sha256
        )
        self.assertEqual(len(row["source_payload"]["payload_sha256"]), 64)

    def test_cached_multi_market_event_uses_one_physical_file_identity(self) -> None:
        as_of = datetime(2026, 7, 9, 20, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as directory:
            paths = PipelinePaths(Path(directory))
            first_market = _gamma_raw_market(
                city="Beijing",
                date_text="July 7",
                slug_city="beijing",
                market_id="market-beijing-84-provenance",
                yes_token="yes-1",
                no_token="no-1",
                question="Will the highest temperature in Beijing be 84°F on July 7?",
            )
            second_market = _gamma_raw_market(
                city="Beijing",
                date_text="July 7",
                slug_city="beijing",
                market_id="market-beijing-85-provenance",
                yes_token="yes-2",
                no_token="no-2",
                question="Will the highest temperature in Beijing be 85°F on July 7?",
            )
            source_path, source_bytes = _write_gamma_event_cache(
                paths,
                first_market,
                second_market,
                indent=1,
            )
            expected_sha256 = hashlib.sha256(source_bytes).hexdigest()

            with patch(
                "daily_weather_pipeline.weather_corpus_extension.hashlib.sha256",
                wraps=hashlib.sha256,
            ) as sha256_mock:
                cached = list(
                    iter_cached_gamma_event_payloads(
                        paths,
                        start=date(2026, 7, 7),
                        end=date(2026, 7, 7),
                    )
                )

            self.assertEqual(
                sum(
                    1
                    for call in sha256_mock.call_args_list
                    if call.args and call.args[0] == source_bytes
                ),
                1,
            )
            self.assertEqual(len(cached), 1)
            payload, cached_path, payload_id, source_file_sha256 = cached[0]
            self.assertEqual(cached_path, source_path)
            self.assertEqual(source_file_sha256, expected_sha256)
            self.assertEqual(len(source_file_sha256), 64)

            rows = market_rows_from_gamma_event_payload(
                payload,
                source_path=cached_path,
                payload_id=payload_id,
                source_file_sha256=source_file_sha256,
                as_of_utc=as_of,
                start=date(2026, 7, 7),
                end=date(2026, 7, 7),
            )
            self.assertEqual(len(rows), 2)
            source_payloads = {
                json.dumps(row["source_payload"], sort_keys=True) for row in rows
            }
            self.assertEqual(len(source_payloads), 1)
            self.assertTrue(
                all(
                    row["source_payload"]["payload_sha256"] == expected_sha256
                    for row in rows
                )
            )

            source_path.touch()
            repeated = list(
                iter_cached_gamma_event_payloads(
                    paths,
                    start=date(2026, 7, 7),
                    end=date(2026, 7, 7),
                )
            )
            self.assertEqual(repeated[0][2], payload_id)
            self.assertEqual(repeated[0][3], expected_sha256)

    def test_gamma_rerun_replaces_changed_file_identity_without_orphans(self) -> None:
        as_of = datetime(2026, 7, 9, 20, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as directory:
            paths = PipelinePaths(Path(directory))
            initialize_weather_corpus(paths)
            raw = _gamma_raw_market(
                city="Beijing",
                date_text="July 7",
                slug_city="beijing",
                market_id="market-beijing-rerun-provenance",
                yes_token="yes-beijing-rerun-provenance",
                no_token="no-beijing-rerun-provenance",
                question="Will the highest temperature in Beijing be 85°F on July 7?",
            )
            source_path, first_bytes = _write_gamma_event_cache(
                paths,
                raw,
                indent=1,
            )
            first = extend_market_universe_from_gamma(
                paths=paths,
                start=date(2026, 7, 7),
                end=date(2026, 7, 7),
                as_of_utc=as_of,
                cache_only=True,
                mode="cached",
            )
            with sqlite3.connect(paths.market_db) as conn:
                first_payload_id, first_stored_sha = conn.execute(
                    """
                    SELECT payload_id, payload_sha256
                    FROM source_payloads
                    WHERE source_name = 'polymarket_gamma_public_event'
                    """
                ).fetchone()
            self.assertEqual(
                first_stored_sha, hashlib.sha256(first_bytes).hexdigest()
            )
            self.assertEqual(first["orphan_source_payload_rows_pruned"], 0)

            source_path, second_bytes = _write_gamma_event_cache(
                paths,
                raw,
                indent=4,
            )
            self.assertNotEqual(first_bytes, second_bytes)
            second = extend_market_universe_from_gamma(
                paths=paths,
                start=date(2026, 7, 7),
                end=date(2026, 7, 7),
                as_of_utc=as_of,
                cache_only=True,
                mode="cached",
            )
            with sqlite3.connect(paths.market_db) as conn:
                payload_rows = conn.execute(
                    """
                    SELECT payload_id, payload_sha256
                    FROM source_payloads
                    WHERE source_name = 'polymarket_gamma_public_event'
                    """
                ).fetchall()
                orphan_count = conn.execute(
                    """
                    SELECT COUNT(*)
                    FROM source_payloads AS source
                    WHERE source.source_name = 'polymarket_gamma_public_event'
                      AND NOT EXISTS (
                            SELECT 1
                            FROM market_universe AS market
                            WHERE market.source_payload_id = source.payload_id
                      )
                    """
                ).fetchone()[0]

            self.assertEqual(len(payload_rows), 1)
            self.assertNotEqual(payload_rows[0][0], first_payload_id)
            self.assertEqual(
                payload_rows[0][1], hashlib.sha256(second_bytes).hexdigest()
            )
            self.assertEqual(orphan_count, 0)
            self.assertEqual(second["orphan_source_payload_rows_pruned"], 1)

            third = extend_market_universe_from_gamma(
                paths=paths,
                start=date(2026, 7, 7),
                end=date(2026, 7, 7),
                as_of_utc=as_of,
                cache_only=True,
                mode="cached",
            )
            with sqlite3.connect(paths.market_db) as conn:
                self.assertEqual(
                    conn.execute(
                        """
                        SELECT COUNT(*)
                        FROM source_payloads
                        WHERE source_name = 'polymarket_gamma_public_event'
                        """
                    ).fetchone()[0],
                    1,
                )
            self.assertEqual(third["orphan_source_payload_rows_pruned"], 0)

    def test_gamma_row_rejects_non_sha256_source_file_identity(self) -> None:
        raw = _gamma_raw_market(
            city="Beijing",
            date_text="July 7",
            slug_city="beijing",
            market_id="market-beijing-invalid-provenance",
            yes_token="yes-beijing-invalid-provenance",
            no_token="no-beijing-invalid-provenance",
            question="Will the highest temperature in Beijing be 85°F on July 7?",
        )
        market = parse_weather_market(raw, today=date(2026, 7, 9))
        self.assertIsNotNone(market)

        for invalid in ("a" * 63, "A" * 64, "g" * 64):
            with self.subTest(invalid=invalid[:8]), self.assertRaisesRegex(
                ValueError,
                "lowercase SHA-256",
            ):
                gamma_market_to_universe_row(
                    market,
                    raw=raw,
                    source_path=Path("/tmp/gamma-event.json"),
                    payload_id="payload-gamma-invalid",
                    source_file_sha256=invalid,
                )

    def test_pair_label_availability_is_timezone_specific(self) -> None:
        as_of = datetime(2026, 7, 9, 7, 45, tzinfo=timezone.utc)
        beijing = {"local_date": "2026-07-08", "timezone": "Asia/Shanghai"}
        los_angeles = {"local_date": "2026-07-08", "timezone": "America/Los_Angeles"}

        self.assertLessEqual(pair_label_available_at_utc(beijing), as_of)
        self.assertGreater(pair_label_available_at_utc(los_angeles), as_of)

    def test_weather_extension_skips_pairs_before_label_availability_lag(self) -> None:
        as_of = datetime(2026, 7, 9, 7, 45, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as directory:
            paths = PipelinePaths(Path(directory))
            initialize_weather_corpus(paths)
            _insert_gamma_market(paths, _gamma_raw_market(
                city="Beijing",
                date_text="July 8",
                slug_city="beijing",
                market_id="market-beijing-85",
                yes_token="yes-beijing-85",
                no_token="no-beijing-85",
                question="Will the highest temperature in Beijing be 85°F on July 8?",
            ))
            _insert_gamma_market(paths, _gamma_raw_market(
                city="Los Angeles",
                date_text="July 8",
                slug_city="los-angeles",
                market_id="market-la-70",
                yes_token="yes-la-70",
                no_token="no-la-70",
                question="Will the highest temperature in Los Angeles be 70°F on July 8?",
            ))
            calls: list[str] = []

            def fake_forecast(pair, start_date, end_date, *, paths, cache_only):
                calls.append(str(pair["normalized_city"]))
                payload = _weather_payload(str(pair["local_date"]), 85.0)
                return payload, Path(directory) / "forecast.json", "forecast-payload", "forecast-url", {}

            def fake_realized(pair, start_date, end_date, *, paths, cache_only):
                payload = _weather_payload(str(pair["local_date"]), 85.0)
                return payload, Path(directory) / "realized.json", "realized-payload", "realized-url", {}

            with patch(
                "daily_weather_pipeline.weather_corpus_extension._fetch_historical_forecast_payload",
                side_effect=fake_forecast,
            ), patch(
                "daily_weather_pipeline.weather_corpus_extension._fetch_realized_payload",
                side_effect=fake_realized,
            ):
                result = extend_weather_history_for_labelable_pairs(
                    paths=paths,
                    start=date(2026, 7, 8),
                    end=date(2026, 7, 8),
                    as_of_utc=as_of,
                    cache_only=True,
                )

            self.assertEqual(result["labelable_city_date_pairs"], 1)
            self.assertEqual(calls, ["Beijing, CN"])
            with sqlite3.connect(paths.weather_db) as conn:
                cities = [
                    row[0]
                    for row in conn.execute(
                        "SELECT normalized_city FROM realized_daily_weather ORDER BY normalized_city"
                    )
                ]
            self.assertEqual(cities, ["Beijing, CN"])

    def test_run_extension_advances_training_corpus_without_staleness_warning(self) -> None:
        as_of = datetime(2026, 7, 9, 7, 45, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as directory:
            paths = PipelinePaths(Path(directory))
            initialize_weather_corpus(paths)
            _insert_gamma_market(paths, _gamma_raw_market(
                city="Beijing",
                date_text="July 7",
                slug_city="beijing",
                market_id="market-beijing-85",
                yes_token="yes-beijing-85",
                no_token="no-beijing-85",
                question="Will the highest temperature in Beijing be 85°F on July 7?",
            ))

            def fake_forecast(pair, start_date, end_date, *, paths, cache_only):
                payload = _weather_payload(str(pair["local_date"]), 85.0)
                return payload, Path(directory) / "forecast.json", "forecast-payload", "forecast-url", {}

            def fake_realized(pair, start_date, end_date, *, paths, cache_only):
                payload = _weather_payload(str(pair["local_date"]), 85.0)
                return payload, Path(directory) / "realized.json", "realized-payload", "realized-url", {}

            with patch(
                "daily_weather_pipeline.weather_corpus_extension._fetch_historical_forecast_payload",
                side_effect=fake_forecast,
            ), patch(
                "daily_weather_pipeline.weather_corpus_extension._fetch_realized_payload",
                side_effect=fake_realized,
            ):
                report = run_weather_corpus_extension(
                    paths=paths,
                    start=date(2026, 7, 7),
                    end=date(2026, 7, 7),
                    as_of_utc=as_of,
                    cache_only=True,
                    market_discovery_mode="cached",
                )

        summary = report["training_corpus_summary"]
        self.assertEqual(summary["training_corpus_end_date"], "2026-07-07")
        self.assertGreater(summary["training_row_count"], 0)
        self.assertEqual(summary["training_corpus_staleness_days"], 2)
        self.assertFalse(summary["training_corpus_staleness_warning"])

    def test_corpus_initializer_and_extension_never_touch_replay_stores(self) -> None:
        as_of = datetime(2026, 7, 9, 7, 45, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as directory:
            paths = PipelinePaths(Path(directory))
            initialize_weather_corpus(paths)
            self.assertTrue(paths.market_db.exists())
            self.assertTrue(paths.weather_db.exists())
            self.assertFalse(paths.price_db.exists())
            self.assertFalse(paths.compact_price_db.exists())
            self.assertFalse(paths.decision_db.exists())

            sentinels = {
                paths.price_db: b"wide-price-sentinel",
                paths.compact_price_db: b"compact-price-sentinel",
                paths.decision_db: b"decision-sentinel",
            }
            for path, payload in sentinels.items():
                path.write_bytes(payload)
            _insert_gamma_market(
                paths,
                _gamma_raw_market(
                    city="Beijing",
                    date_text="July 7",
                    slug_city="beijing",
                    market_id="market-beijing-isolation",
                    yes_token="yes-beijing-isolation",
                    no_token="no-beijing-isolation",
                    question="Will the highest temperature in Beijing be 85°F on July 7?",
                ),
            )

            with _patched_weather_fetches(directory):
                report = run_weather_corpus_extension(
                    paths=paths,
                    start=date(2026, 7, 7),
                    end=date(2026, 7, 7),
                    as_of_utc=as_of,
                    cache_only=True,
                    market_discovery_mode="cached",
                )

            self.assertTrue(report["operational_corpus_ready"])
            for path, payload in sentinels.items():
                self.assertEqual(path.read_bytes(), payload)

    def test_pair_limit_does_not_inflate_skipped_existing_count(self) -> None:
        as_of = datetime(2026, 7, 9, 20, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as directory:
            paths = PipelinePaths(Path(directory))
            initialize_weather_corpus(paths)
            for city, slug, suffix, question in (
                ("Austin", "austin", "austin", "Will the highest temperature in Austin be 95°F on July 7?"),
                ("Beijing", "beijing", "beijing", "Will the highest temperature in Beijing be 85°F on July 7?"),
                ("London", "london", "london", "Will the highest temperature in London be 75°F on July 7?"),
            ):
                _insert_gamma_market(
                    paths,
                    _gamma_raw_market(
                        city=city,
                        date_text="July 7",
                        slug_city=slug,
                        market_id=f"market-{suffix}-limit",
                        yes_token=f"yes-{suffix}-limit",
                        no_token=f"no-{suffix}-limit",
                        question=question,
                    ),
                )

            with _patched_weather_fetches(directory):
                first = extend_weather_history_for_labelable_pairs(
                    paths=paths,
                    start=date(2026, 7, 7),
                    end=date(2026, 7, 7),
                    as_of_utc=as_of,
                    cache_only=True,
                    pair_limit=1,
                )
                second = extend_weather_history_for_labelable_pairs(
                    paths=paths,
                    start=date(2026, 7, 7),
                    end=date(2026, 7, 7),
                    as_of_utc=as_of,
                    cache_only=True,
                    pair_limit=1,
                )

            self.assertEqual(first["pair_counts_before_selection"]["complete"], 0)
            self.assertEqual(first["selected_city_date_pairs"], 1)
            self.assertEqual(first["remaining_unselected_city_date_pairs"], 2)
            self.assertEqual(first["skipped_existing_pairs"], 0)
            self.assertEqual(second["pair_counts_before_selection"]["complete"], 1)
            self.assertEqual(second["pair_counts_before_selection"]["pending"], 2)
            self.assertEqual(second["selected_city_date_pairs"], 1)
            self.assertEqual(second["remaining_unselected_city_date_pairs"], 1)
            self.assertEqual(second["skipped_existing_pairs"], 1)

    def test_fresh_city_cannot_hide_stale_city(self) -> None:
        as_of = datetime(2026, 7, 9, 20, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as directory:
            paths = PipelinePaths(Path(directory))
            initialize_weather_corpus(paths)
            for city, slug, suffix, question in (
                ("Beijing", "beijing", "beijing", "Will the highest temperature in Beijing be 85°F on July 7?"),
                ("London", "london", "london", "Will the highest temperature in London be 75°F on July 7?"),
            ):
                _insert_gamma_market(
                    paths,
                    _gamma_raw_market(
                        city=city,
                        date_text="July 7",
                        slug_city=slug,
                        market_id=f"market-{suffix}-coverage",
                        yes_token=f"yes-{suffix}-coverage",
                        no_token=f"no-{suffix}-coverage",
                        question=question,
                    ),
                )

            with _patched_weather_fetches(directory):
                result = extend_weather_history_for_labelable_pairs(
                    paths=paths,
                    start=date(2026, 7, 7),
                    end=date(2026, 7, 7),
                    as_of_utc=as_of,
                    cache_only=True,
                    pair_limit=1,
                )
            coverage = weather_corpus_coverage(
                work_root=paths.root,
                start=date(2026, 7, 7),
                end=date(2026, 7, 7),
                as_of_utc=as_of,
            )

            self.assertEqual(result["complete_city_date_pairs"], 1)
            self.assertEqual(coverage["complete_city_date_pairs"], 1)
            self.assertEqual(coverage["pending_city_date_pairs"], 1)
            self.assertFalse(coverage["operationally_fresh"])
            self.assertEqual(coverage["fresh_city_count"], 1)
            self.assertEqual(coverage["stale_city_count"], 1)
            covered_city = next(
                item["normalized_city"]
                for item in coverage["city_coverage"]
                if item["operationally_fresh"]
            )
            stale_city = next(
                item["normalized_city"]
                for item in coverage["city_coverage"]
                if not item["operationally_fresh"]
            )
            self.assertTrue(
                coverage_allows_city_date(
                    coverage,
                    normalized_city=covered_city,
                    target_date="2026-07-07",
                )
            )
            self.assertFalse(
                coverage_allows_city_date(
                    coverage,
                    normalized_city=stale_city,
                    target_date="2026-07-07",
                )
            )
            runner = ObservedHighGuardedShrinkageWeatherModelRunner(
                work_root=paths.root,
                end=date(2026, 7, 7),
            )
            summary = runner.training_corpus_summary(as_of_utc=as_of)
            self.assertEqual(summary["training_corpus_end_date"], "2026-07-07")
            self.assertEqual(summary["training_corpus_staleness_days"], 2)
            self.assertFalse(summary["training_corpus_operationally_fresh"])
            self.assertTrue(summary["training_corpus_staleness_warning"])
            self.assertIn(
                "training_corpus_city_source_date_coverage_incomplete",
                summary["warning_flags"],
            )

    def test_rerun_is_idempotent_and_skips_complete_pair(self) -> None:
        as_of = datetime(2026, 7, 9, 20, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as directory:
            paths = PipelinePaths(Path(directory))
            initialize_weather_corpus(paths)
            _insert_gamma_market(
                paths,
                _gamma_raw_market(
                    city="Beijing",
                    date_text="July 7",
                    slug_city="beijing",
                    market_id="market-beijing-idempotent",
                    yes_token="yes-beijing-idempotent",
                    no_token="no-beijing-idempotent",
                    question="Will the highest temperature in Beijing be 85°F on July 7?",
                ),
            )
            with _patched_weather_fetches(directory) as fetches:
                first = extend_weather_history_for_labelable_pairs(
                    paths=paths,
                    start=date(2026, 7, 7),
                    end=date(2026, 7, 7),
                    as_of_utc=as_of,
                    cache_only=True,
                )
                calls_after_first = tuple(fetch.call_count for fetch in fetches)
                with sqlite3.connect(paths.weather_db) as conn:
                    counts_after_first = (
                        conn.execute("SELECT COUNT(*) FROM forecast_snapshots").fetchone()[0],
                        conn.execute("SELECT COUNT(*) FROM realized_daily_weather").fetchone()[0],
                    )
                second = extend_weather_history_for_labelable_pairs(
                    paths=paths,
                    start=date(2026, 7, 7),
                    end=date(2026, 7, 7),
                    as_of_utc=as_of,
                    cache_only=True,
                )
                calls_after_second = tuple(fetch.call_count for fetch in fetches)

            with sqlite3.connect(paths.weather_db) as conn:
                counts = (
                    conn.execute("SELECT COUNT(*) FROM forecast_snapshots").fetchone()[0],
                    conn.execute("SELECT COUNT(*) FROM realized_daily_weather").fetchone()[0],
                )
            self.assertEqual(first["complete_city_date_pairs"], 1)
            self.assertEqual(second["selected_city_date_pairs"], 0)
            self.assertEqual(second["skipped_existing_pairs"], 1)
            self.assertEqual(calls_after_first, calls_after_second)
            self.assertEqual(counts, counts_after_first)

    def test_partial_forecast_write_plus_realized_row_is_not_complete(self) -> None:
        as_of = datetime(2026, 7, 9, 20, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as directory:
            paths = PipelinePaths(Path(directory))
            initialize_weather_corpus(paths)
            _insert_gamma_market(
                paths,
                _gamma_raw_market(
                    city="Beijing",
                    date_text="July 7",
                    slug_city="beijing",
                    market_id="market-beijing-partial",
                    yes_token="yes-beijing-partial",
                    no_token="no-beijing-partial",
                    question="Will the highest temperature in Beijing be 85°F on July 7?",
                ),
            )
            with _patched_weather_fetches(directory):
                initial = extend_weather_history_for_labelable_pairs(
                    paths=paths,
                    start=date(2026, 7, 7),
                    end=date(2026, 7, 7),
                    as_of_utc=as_of,
                    cache_only=True,
                )
            self.assertEqual(initial["complete_city_date_pairs"], 1)
            with sqlite3.connect(paths.weather_db) as conn:
                keep_id = conn.execute(
                    "SELECT forecast_id FROM forecast_snapshots ORDER BY decision_time_utc LIMIT 1"
                ).fetchone()[0]
                conn.execute(
                    "DELETE FROM forecast_snapshots WHERE forecast_id != ?",
                    (keep_id,),
                )
                conn.commit()

            partial_coverage = weather_corpus_coverage(
                work_root=paths.root,
                start=date(2026, 7, 7),
                end=date(2026, 7, 7),
                as_of_utc=as_of,
            )
            pair_state = partial_coverage["pair_states"][0]
            source_coverage = pair_state["forecast_source_decision_time_coverage"][0]
            self.assertFalse(partial_coverage["operationally_fresh"])
            self.assertEqual(partial_coverage["complete_city_date_pairs"], 0)
            self.assertEqual(partial_coverage["pending_city_date_pairs"], 1)
            self.assertTrue(pair_state["realized_present"])
            self.assertFalse(pair_state["forecast_present"])
            self.assertFalse(source_coverage["complete_expected_decision_set"])
            self.assertGreater(len(source_coverage["missing_decision_times_utc"]), 0)
            self.assertGreater(len(pair_state["expected_forecast_decision_times_utc"]), 1)

            with _patched_weather_fetches(directory):
                repaired = extend_weather_history_for_labelable_pairs(
                    paths=paths,
                    start=date(2026, 7, 7),
                    end=date(2026, 7, 7),
                    as_of_utc=as_of,
                    cache_only=True,
                )
            self.assertEqual(repaired["pending_city_date_pairs_before_selection"], 1)
            self.assertEqual(repaired["skipped_existing_pairs"], 0)
            self.assertEqual(repaired["selected_city_date_pairs"], 1)
            self.assertEqual(repaired["complete_city_date_pairs"], 1)

    def test_failed_update_is_reported_incomplete_and_not_fresh(self) -> None:
        as_of = datetime(2026, 7, 9, 20, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as directory:
            paths = PipelinePaths(Path(directory))
            initialize_weather_corpus(paths)
            _insert_gamma_market(
                paths,
                _gamma_raw_market(
                    city="Beijing",
                    date_text="July 7",
                    slug_city="beijing",
                    market_id="market-beijing-failed",
                    yes_token="yes-beijing-failed",
                    no_token="no-beijing-failed",
                    question="Will the highest temperature in Beijing be 85°F on July 7?",
                ),
            )

            def fail_realized(*_args, **_kwargs):
                raise RuntimeError("fixture realized fetch failure")

            with _patched_weather_fetches(directory), patch(
                "daily_weather_pipeline.weather_corpus_extension._fetch_realized_payload",
                side_effect=fail_realized,
            ):
                report = run_weather_corpus_extension(
                    paths=paths,
                    start=date(2026, 7, 7),
                    end=date(2026, 7, 7),
                    as_of_utc=as_of,
                    cache_only=True,
                    market_discovery_mode="cached",
                )

            self.assertEqual(report["status"], "incomplete")
            self.assertFalse(report["operational_corpus_ready"])
            self.assertEqual(
                report["weather_extension"]["failed_city_date_pairs"],
                1,
            )
            self.assertEqual(
                report["operational_corpus_coverage"]["failed_city_date_pairs"],
                1,
            )
            self.assertIsNotNone(report["missing_data_product"])


@contextmanager
def _patched_weather_fetches(directory: str):
    def fake_forecast(pair, start_date, end_date, *, paths, cache_only):
        payload = _weather_payload(str(pair["local_date"]), 85.0)
        return payload, Path(directory) / "forecast.json", "forecast-payload", "forecast-url", {}

    def fake_realized(pair, start_date, end_date, *, paths, cache_only):
        payload = _weather_payload(str(pair["local_date"]), 85.0)
        return payload, Path(directory) / "realized.json", "realized-payload", "realized-url", {}

    with patch(
        "daily_weather_pipeline.weather_corpus_extension._fetch_historical_forecast_payload",
        side_effect=fake_forecast,
    ) as forecast_mock, patch(
        "daily_weather_pipeline.weather_corpus_extension._fetch_realized_payload",
        side_effect=fake_realized,
    ) as realized_mock:
        yield forecast_mock, realized_mock


def _insert_gamma_market(paths: PipelinePaths, raw: dict[str, object]) -> None:
    market = parse_weather_market(raw, today=date(2026, 7, 9))
    assert market is not None
    source_path = paths.root / f"gamma-event-{raw['id']}.json"
    source_bytes = json.dumps(
        {"markets": [raw]}, indent=2, sort_keys=True
    ).encode("utf-8")
    source_path.write_bytes(source_bytes)
    row = gamma_market_to_universe_row(
        market,
        raw=raw,
        source_path=source_path,
        payload_id=f"payload-{raw['id']}",
        source_file_sha256=hashlib.sha256(source_bytes).hexdigest(),
    )
    with _connect(paths.market_db) as conn:
        _insert_market_row(conn, row, "test-run")
        conn.commit()


def _write_gamma_event_cache(
    paths: PipelinePaths,
    *markets: dict[str, object],
    indent: int,
) -> tuple[Path, bytes]:
    if not markets:
        raise ValueError("at least one Gamma market is required")
    first = markets[0]
    payload = {
        "id": f"event-{first['id']}",
        "slug": first["eventSlug"],
        "title": first["eventTitle"],
        "markets": list(markets),
    }
    cache_dir = paths.work / "cache" / "polymarket_gamma" / "weather_corpus"
    cache_dir.mkdir(parents=True, exist_ok=True)
    source_path = cache_dir / f"gamma-event-{first['eventSlug']}.json"
    source_bytes = (
        json.dumps(payload, indent=indent, sort_keys=False) + "\n"
    ).encode("utf-8")
    source_path.write_bytes(source_bytes)
    return source_path, source_bytes


def _gamma_raw_market(
    *,
    city: str,
    date_text: str,
    slug_city: str,
    market_id: str,
    yes_token: str,
    no_token: str,
    question: str,
) -> dict[str, object]:
    slug_date = f"{date_text.lower().replace(' ', '-')}-2026"
    return {
        "id": market_id,
        "conditionId": f"condition-{market_id}",
        "slug": f"highest-temperature-in-{slug_city}-on-{slug_date}",
        "eventSlug": f"highest-temperature-in-{slug_city}-on-{slug_date}",
        "eventTitle": f"Highest temperature in {city} on {date_text} 2026?",
        "question": question,
        "outcomes": '["Yes", "No"]',
        "clobTokenIds": f'["{yes_token}", "{no_token}"]',
        "outcomePrices": '["0.50", "0.50"]',
        "endDate": "2026-07-08T12:00:00Z",
    }


def _weather_payload(local_date: str, high_f: float) -> dict[str, object]:
    return {
        "daily": {
            "time": [local_date],
            "temperature_2m_max": [high_f],
            "precipitation_sum": [0.0],
            "wind_speed_10m_max": [4.0],
        },
        "hourly": {
            "time": [f"{local_date}T00:00", f"{local_date}T12:00"],
            "temperature_2m": [high_f - 10.0, high_f],
            "apparent_temperature": [high_f - 9.0, high_f + 1.0],
            "relative_humidity_2m": [50.0, 60.0],
            "dew_point_2m": [40.0, 45.0],
            "cloud_cover": [10.0, 20.0],
            "wind_speed_10m": [3.0, 4.0],
            "precipitation": [0.0, 0.0],
        },
    }


if __name__ == "__main__":
    unittest.main()
