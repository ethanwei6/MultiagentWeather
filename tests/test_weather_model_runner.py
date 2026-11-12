from __future__ import annotations

import sqlite3
import tempfile
import unittest
from contextlib import closing
from datetime import date, datetime, timezone
from pathlib import Path
from unittest.mock import patch

from daily_weather_pipeline.backtesting_engine import (
    ReplayGeneratedScoreCache,
    WeatherModelDecisionContext,
)
from daily_weather_pipeline.weather_model_validation import (
    DEFAULT_END_DATE,
    ValidationRow,
    _connect_readonly,
    resolve_training_corpus_end_date,
)
from daily_weather_pipeline.weather_corpus_snapshot import CorpusSnapshotError
from daily_weather_pipeline.weather_model_runner import (
    ObservedHighGuardedShrinkageWeatherModelRunner,
)


class ObservedHighGuardedShrinkageWeatherModelRunnerTests(unittest.TestCase):
    def test_controlled_candidate_rejects_nonzero_committed_wal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate_dir = (
                root
                / "work"
                / "data"
                / "weather_corpus"
                / ".staging"
                / ("a" * 32)
            )
            candidate_dir.mkdir(parents=True)
            market_db = candidate_dir / "market.sqlite"
            weather_db = candidate_dir / "weather.sqlite"
            for database in (market_db, weather_db):
                with closing(sqlite3.connect(database)) as conn:
                    conn.execute("CREATE TABLE seed(value TEXT NOT NULL)")
                    conn.execute("INSERT INTO seed(value) VALUES ('seed')")
                    conn.commit()

            with closing(sqlite3.connect(market_db)) as conn:
                self.assertEqual(
                    str(conn.execute("PRAGMA journal_mode=WAL").fetchone()[0]).lower(),
                    "wal",
                )
            reader = sqlite3.connect(market_db)
            try:
                reader.execute("BEGIN")
                self.assertEqual(reader.execute("SELECT COUNT(*) FROM seed").fetchone()[0], 1)
                with closing(sqlite3.connect(market_db)) as writer:
                    writer.execute("INSERT INTO seed(value) VALUES ('committed-in-wal')")
                    writer.commit()
                self.assertGreater(Path(f"{market_db}-wal").stat().st_size, 0)

                runner = ObservedHighGuardedShrinkageWeatherModelRunner(
                    work_root=root,
                    corpus_market_db=market_db,
                    corpus_weather_db=weather_db,
                    corpus_override_provenance="copy_on_write_candidate",
                )
                with self.assertRaisesRegex(
                    CorpusSnapshotError,
                    "copy_on_write_candidate_sidecar_present",
                ):
                    runner._corpus_resolution()
            finally:
                reader.close()

    def test_controlled_candidate_allows_zero_wal_but_rejects_shm_without_deletion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate_dir = (
                root
                / "work"
                / "data"
                / "weather_corpus"
                / ".staging"
                / ("b" * 32)
            )
            candidate_dir.mkdir(parents=True)
            market_db = candidate_dir / "market.sqlite"
            weather_db = candidate_dir / "weather.sqlite"
            for database in (market_db, weather_db):
                with closing(sqlite3.connect(database)) as conn:
                    conn.execute("CREATE TABLE seed(value TEXT NOT NULL)")
                    conn.execute("INSERT INTO seed(value) VALUES ('clean-close')")
                    conn.commit()
            zero_wal = Path(f"{market_db}-wal")
            zero_wal.write_bytes(b"")

            runner = ObservedHighGuardedShrinkageWeatherModelRunner(
                work_root=root,
                corpus_market_db=market_db,
                corpus_weather_db=weather_db,
                corpus_override_provenance="copy_on_write_candidate",
            )
            resolution = runner._corpus_resolution()
            self.assertEqual(resolution.source, "copy_on_write_candidate")
            with _connect_readonly(
                market_db,
                immutable=False,
                reject_sidecars=True,
                sidecar_reason_prefix="copy_on_write_candidate",
                allow_zero_length_wal=True,
            ) as conn:
                self.assertEqual(conn.execute("SELECT value FROM seed").fetchone()[0], "clean-close")
            self.assertTrue(zero_wal.exists())
            self.assertEqual(zero_wal.stat().st_size, 0)

            ambiguous_shm = Path(f"{market_db}-shm")
            ambiguous_shm.write_bytes(b"ambiguous")
            with self.assertRaisesRegex(
                CorpusSnapshotError,
                "copy_on_write_candidate_sidecar_present:market.sqlite-shm",
            ):
                ObservedHighGuardedShrinkageWeatherModelRunner(
                    work_root=root,
                    corpus_market_db=market_db,
                    corpus_weather_db=weather_db,
                    corpus_override_provenance="copy_on_write_candidate",
                )._corpus_resolution()
            self.assertTrue(ambiguous_shm.exists())

    def test_user_explicit_override_rejects_even_zero_length_wal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            market_db = root / "market.sqlite"
            weather_db = root / "weather.sqlite"
            for database in (market_db, weather_db):
                with closing(sqlite3.connect(database)) as conn:
                    conn.execute("CREATE TABLE seed(value TEXT NOT NULL)")
                    conn.commit()
            zero_wal = Path(f"{market_db}-wal")
            zero_wal.write_bytes(b"")
            with self.assertRaisesRegex(
                CorpusSnapshotError,
                "explicit_corpus_override_sidecar_present:market.sqlite-wal:0",
            ):
                ObservedHighGuardedShrinkageWeatherModelRunner(
                    corpus_market_db=market_db,
                    corpus_weather_db=weather_db,
                )._corpus_resolution()
            self.assertTrue(zero_wal.exists())

    def test_user_explicit_wal_header_without_sidecars_fails_without_creating_them(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            market_db = root / "market.sqlite"
            weather_db = root / "weather.sqlite"
            for database in (market_db, weather_db):
                with closing(sqlite3.connect(database)) as conn:
                    conn.execute("CREATE TABLE seed(value TEXT NOT NULL)")
                    conn.commit()
            with closing(sqlite3.connect(market_db)) as conn:
                self.assertEqual(
                    str(conn.execute("PRAGMA journal_mode=WAL").fetchone()[0]).lower(),
                    "wal",
                )
            for suffix in ("-wal", "-shm"):
                Path(f"{market_db}{suffix}").unlink(missing_ok=True)

            with self.assertRaisesRegex(
                CorpusSnapshotError,
                "explicit_corpus_override_journal_mode_not_delete:market.sqlite:wal",
            ):
                ObservedHighGuardedShrinkageWeatherModelRunner(
                    corpus_market_db=market_db,
                    corpus_weather_db=weather_db,
                )._corpus_resolution()
            for suffix in ("-wal", "-shm"):
                self.assertFalse(Path(f"{market_db}{suffix}").exists())

    def test_clean_explicit_override_uses_nonimmutable_sidecar_checked_loaders(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            market_db = root / "market.sqlite"
            weather_db = root / "weather.sqlite"
            for database in (market_db, weather_db):
                with closing(sqlite3.connect(database)) as conn:
                    conn.execute("CREATE TABLE seed(value TEXT NOT NULL)")
                    conn.commit()
            runner = ObservedHighGuardedShrinkageWeatherModelRunner(
                corpus_market_db=market_db,
                corpus_weather_db=weather_db,
            )

            with patch.object(runner, "_load_strategy", return_value={}), patch(
                "daily_weather_pipeline.weather_model_runner._load_markets",
                return_value=[],
            ) as load_markets, patch(
                "daily_weather_pipeline.weather_model_runner._load_forecasts",
                return_value=[],
            ) as load_forecasts, patch(
                "daily_weather_pipeline.weather_model_runner._load_realized",
                return_value={},
            ) as load_realized, patch(
                "daily_weather_pipeline.weather_model_runner._build_validation_rows",
                return_value=[],
            ):
                self.assertEqual(runner._load_rows(), [])

            resolution = runner._corpus_resolution()
            self.assertEqual(resolution.source, "explicit_sidecar_free_override")
            for loader in (load_markets, load_forecasts, load_realized):
                self.assertFalse(loader.call_args.kwargs["immutable"])
                self.assertTrue(loader.call_args.kwargs["reject_sidecars"])
                self.assertEqual(
                    loader.call_args.kwargs["sidecar_reason_prefix"],
                    "explicit_sidecar_free_override",
                )
                self.assertFalse(loader.call_args.kwargs["allow_zero_length_wal"])
            late_sidecar = Path(f"{market_db}-wal")
            late_sidecar.write_bytes(b"")
            with self.assertRaisesRegex(
                CorpusSnapshotError,
                "explicit_corpus_override_sidecar_present:market.sqlite-wal:0",
            ):
                runner._corpus_resolution()
            self.assertTrue(late_sidecar.exists())

    def test_default_training_corpus_end_date_resolves_to_current_utc_date(self) -> None:
        resolved = resolve_training_corpus_end_date(
            now_utc=datetime(2026, 7, 9, 15, 30, tzinfo=timezone.utc)
        )

        self.assertEqual(resolved, date(2026, 7, 9))
        self.assertEqual(
            resolve_training_corpus_end_date("2026-07-04"),
            date(2026, 7, 4),
        )
        runner = ObservedHighGuardedShrinkageWeatherModelRunner(
            end=resolved,
            historical_rows=[],
            baseline_probability_provider=lambda _context: (0.5, {"fixture": 0.5}),
        )
        self.assertEqual(runner.end, date(2026, 7, 9))
        self.assertGreater(runner.end, date.fromisoformat(DEFAULT_END_DATE))

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

    def test_training_rows_still_respect_label_available_lag(self) -> None:
        runner = _runner(
            historical_rows=[
                _validation_row("old", local_date=date(2026, 7, 6)),
                _validation_row("not-yet-available", local_date=date(2026, 7, 8)),
            ]
        )

        rows = runner._training_rows_for(datetime(2026, 7, 9, 8, tzinfo=timezone.utc))
        summary = runner.training_corpus_summary(
            as_of_utc=datetime(2026, 7, 9, 8, tzinfo=timezone.utc)
        )

        self.assertEqual([row.row_id for row in rows], ["old"])
        self.assertEqual(summary["training_corpus_end_date"], "2026-07-06")
        self.assertEqual(summary["training_row_count"], 1)


def _runner(
    *,
    probability: float = 0.42,
    label_availability_lag_hours: float = 12.0,
    historical_rows=None,
) -> ObservedHighGuardedShrinkageWeatherModelRunner:
    def baseline_provider(_context: WeatherModelDecisionContext):
        return probability, {
            "open_meteo.kernel_tight": probability,
            "open_meteo.normal_parametric": min(0.99, probability + 0.02),
        }

    return ObservedHighGuardedShrinkageWeatherModelRunner(
        historical_rows=[] if historical_rows is None else historical_rows,
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


def _validation_row(row_id: str, *, local_date: date) -> ValidationRow:
    return ValidationRow(
        row_id=row_id,
        universe_row_id=f"universe-{row_id}",
        forecast_id=f"forecast-{row_id}",
        normalized_city="London, GB",
        country_region="GB",
        timezone_name="UTC",
        market_date_local=local_date,
        decision_time_utc=datetime(2026, 7, 5, 12, tzinfo=timezone.utc),
        decision_hour_utc=12,
        local_decision_hour=12.0,
        lead_days=1,
        day_type="lead_1",
        source_name="open_meteo",
        model_name="historical_forecast",
        bucket_label="70F+",
        bucket_type="above",
        bucket_shape="lower_bound",
        threshold_low_f=70.0,
        threshold_high_f=None,
        threshold_distance_f=1.0,
        threshold_distance_bucket="near",
        realized_high_f=72.0,
        forecast_high_f=71.0,
        forecast_error_f=-1.0,
        outcome=1,
        resolved_outcome=None,
        resolved_outcome_mismatch=False,
        current_probability=0.55,
        mean_probability=0.55,
        model_probabilities={"open_meteo.kernel_tight": 0.55},
        forecast_quality_flags=(),
        realized_quality_flags=(),
        feature_json={},
        latitude=51.5,
        longitude=-0.1,
    )
