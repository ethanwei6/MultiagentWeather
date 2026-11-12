from __future__ import annotations

import json
import sqlite3
import stat
import tempfile
import unittest
from contextlib import closing, contextmanager
from datetime import date, datetime, timezone
from pathlib import Path
from unittest.mock import patch

from daily_weather_pipeline import weather_corpus_snapshot as snapshot_module
from corpus_fixture import fixture_weather_strategy

from daily_weather_pipeline.cli import (
    PipelinePaths,
    _connect,
    _insert_market_row,
    initialize_weather_corpus,
)
from daily_weather_pipeline.live_forward_shadow_recorder import parse_weather_market
from daily_weather_pipeline.weather_corpus_coverage import weather_corpus_coverage
from daily_weather_pipeline.weather_corpus_extension import run_weather_corpus_extension
from daily_weather_pipeline.weather_corpus_snapshot import (
    CorpusSnapshotError,
    CorpusWriterLockError,
    active_corpus_lineage_artifacts,
    active_pointer_path,
    corpus_product_root,
    create_corpus_candidate,
    discard_corpus_candidate,
    exclusive_corpus_writer_lock,
    normalize_corpus_candidate_databases,
    repair_active_snapshot_derived_sidecars,
    resolve_corpus_snapshot,
    writer_lock_path,
)
from daily_weather_pipeline.weather_model_runner import (
    ObservedHighGuardedShrinkageWeatherModelRunner,
)


class WeatherCorpusSnapshotTests(unittest.TestCase):
    def setUp(self) -> None:
        self.enterContext(patch(
            "daily_weather_pipeline.weather_model_runner._load_weather_strategy",
            return_value=fixture_weather_strategy(),
        ))

    def test_durable_replace_fsyncs_both_affected_parent_directories_after_rename(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_parent = root / "source"
            destination_parent = root / "destination"
            source_parent.mkdir()
            destination_parent.mkdir()
            source = source_parent / "payload"
            destination = destination_parent / "payload"
            source.write_text("durable", encoding="utf-8")
            events: list[tuple[str, Path]] = []
            real_replace = snapshot_module.os.replace

            def recording_replace(left, right):
                events.append(("replace", Path(right)))
                return real_replace(left, right)

            def recording_directory_fsync(path):
                events.append(("fsync", Path(path)))

            with patch.object(
                snapshot_module.os,
                "replace",
                side_effect=recording_replace,
            ), patch.object(
                snapshot_module,
                "_fsync_directory",
                side_effect=recording_directory_fsync,
            ):
                snapshot_module._durable_replace(source, destination)

            self.assertEqual(
                events,
                [
                    ("replace", destination),
                    ("fsync", destination_parent),
                    ("fsync", source_parent),
                ],
            )
            self.assertEqual(destination.read_text(encoding="utf-8"), "durable")

    def test_promotion_fsyncs_payload_before_snapshot_rename_and_durably_replaces_pointer(self) -> None:
        as_of = datetime(2026, 7, 9, 20, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as directory:
            paths = _seed_market(Path(directory))
            events: list[tuple[str, Path, Path | None]] = []
            real_fsync_file = snapshot_module._fsync_file
            real_durable_replace = snapshot_module._durable_replace

            def recording_file_fsync(path):
                events.append(("file_fsync", Path(path), None))
                return real_fsync_file(path)

            def recording_durable_replace(source, destination):
                events.append(("replace", Path(source), Path(destination)))
                return real_durable_replace(source, destination)

            with patch.object(
                snapshot_module,
                "_fsync_file",
                side_effect=recording_file_fsync,
            ), patch.object(
                snapshot_module,
                "_durable_replace",
                side_effect=recording_durable_replace,
            ), _patched_weather_fetches(directory):
                _run(paths, as_of=as_of)

            snapshot_rename_index, snapshot_rename = next(
                (index, event)
                for index, event in enumerate(events)
                if event[0] == "replace"
                and event[2] is not None
                and event[2].parent.name == "snapshots"
            )
            staging_dir = snapshot_rename[1]
            fsynced_before_rename = {
                event[1]
                for event in events[:snapshot_rename_index]
                if event[0] == "file_fsync"
            }
            self.assertTrue(
                {
                    staging_dir / "market.sqlite",
                    staging_dir / "weather.sqlite",
                    staging_dir / "manifest.json",
                }.issubset(fsynced_before_rename)
            )
            pointer = active_pointer_path(paths.root)
            self.assertTrue(
                any(event[0] == "replace" and event[2] == pointer for event in events)
            )

    def test_successful_promotion_and_reader_resolution_preserve_replay_stores(self) -> None:
        as_of = datetime(2026, 7, 9, 20, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as directory:
            paths = _seed_market(Path(directory))
            sentinels = {
                paths.price_db: b"wide-price-sentinel",
                paths.compact_price_db: b"compact-price-sentinel",
                paths.decision_db: b"decision-sentinel",
            }
            for path, payload in sentinels.items():
                path.write_bytes(payload)

            with _patched_weather_fetches(directory):
                report = _run(paths, as_of=as_of)

            promotion = report["snapshot_promotion"]
            self.assertEqual(promotion["status"], "promoted")
            self.assertTrue(promotion["pointer_changed"])
            self.assertGreater(report["weather_extension"]["forecast_rows_written"], 0)
            self.assertEqual(
                report["operational_corpus_coverage"]["complete_city_date_pairs"],
                1,
            )
            pointer = active_pointer_path(paths.root)
            self.assertTrue(pointer.exists())
            resolution = resolve_corpus_snapshot(paths.root)
            self.assertEqual(resolution.snapshot_id, promotion["snapshot_id"])
            self.assertEqual(resolution.source, "active_immutable_snapshot")
            self.assertNotEqual(resolution.market_db, paths.market_db)
            self.assertNotEqual(resolution.weather_db, paths.weather_db)
            self.assertTrue(resolution.market_db.is_file())
            self.assertTrue(resolution.weather_db.is_file())
            with sqlite3.connect(
                f"file:{resolution.weather_db}?mode=ro&immutable=1",
                uri=True,
            ) as conn:
                self.assertGreater(
                    conn.execute("SELECT COUNT(*) FROM forecast_snapshots").fetchone()[0],
                    0,
                )
            self.assertFalse(resolution.market_db.stat().st_mode & stat.S_IWUSR)
            self.assertFalse(resolution.weather_db.stat().st_mode & stat.S_IWUSR)
            self.assertFalse(resolution.market_db.parent.stat().st_mode & stat.S_IWUSR)
            self.assertEqual(
                report["operational_corpus_coverage"]["corpus_snapshot_id"],
                resolution.snapshot_id,
            )
            coverage = weather_corpus_coverage(
                work_root=paths.root,
                start=date(2026, 7, 7),
                end=date(2026, 7, 7),
                as_of_utc=as_of,
            )
            self.assertTrue(coverage["operationally_fresh"])
            self.assertEqual(coverage["corpus_snapshot_id"], resolution.snapshot_id)
            runner = ObservedHighGuardedShrinkageWeatherModelRunner(
                work_root=paths.root,
                end=date(2026, 7, 7),
            )
            summary = runner.training_corpus_summary(as_of_utc=as_of)
            self.assertEqual(summary["training_corpus_snapshot_id"], resolution.snapshot_id)
            self.assertEqual(
                summary["training_corpus_snapshot_source"],
                "active_immutable_snapshot",
            )
            lineage = active_corpus_lineage_artifacts(paths.root)
            self.assertEqual(lineage["snapshot_id"], resolution.snapshot_id)
            self.assertEqual(lineage["market_db"], str(resolution.market_db))
            self.assertEqual(lineage["weather_db"], str(resolution.weather_db))
            self.assertEqual(len(lineage["active_pointer_sha256"]), 64)
            self.assertEqual(len(lineage["manifest_sha256"]), 64)
            self.assertEqual(len(lineage["market_db_sha256"]), 64)
            self.assertEqual(len(lineage["weather_db_sha256"]), 64)
            for database in (resolution.market_db, resolution.weather_db):
                self.assertFalse(Path(f"{database}-wal").exists())
                self.assertFalse(Path(f"{database}-shm").exists())
            for path, payload in sentinels.items():
                self.assertEqual(path.read_bytes(), payload)

    def test_validation_failure_retains_prior_active_pointer(self) -> None:
        as_of = datetime(2026, 7, 9, 20, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as directory:
            paths = _seed_market(Path(directory))
            with _patched_weather_fetches(directory):
                first = _run(paths, as_of=as_of)
            self.assertEqual(first["snapshot_promotion"]["status"], "promoted")
            pointer_path = active_pointer_path(paths.root)
            before = pointer_path.read_bytes()
            first_snapshot = resolve_corpus_snapshot(paths.root).snapshot_id

            def incomplete_candidate(*, paths, **_kwargs):
                with _connect(paths.weather_db) as conn:
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO weather_fetch_status(
                            normalized_city, local_date, status, error, retryable, updated_at_utc
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            "Rejected Candidate, XX",
                            "2026-07-07",
                            "failed",
                            "fixture candidate failure",
                            1,
                            as_of.isoformat(),
                        ),
                    )
                    conn.commit()
                return {
                    "status": "incomplete",
                    "labelable_city_date_pairs": 1,
                    "failed_pairs": 1,
                    "selected_city_date_pairs": 1,
                }

            with patch(
                "daily_weather_pipeline.weather_corpus_extension.extend_weather_history_for_labelable_pairs",
                side_effect=incomplete_candidate,
            ):
                failed = _run(paths, as_of=as_of)

            self.assertEqual(
                failed["snapshot_promotion"]["status"],
                "validation_failed_prior_snapshot_retained",
            )
            self.assertFalse(failed["snapshot_promotion"]["pointer_changed"])
            self.assertEqual(pointer_path.read_bytes(), before)
            active = resolve_corpus_snapshot(paths.root)
            self.assertEqual(active.snapshot_id, first_snapshot)
            with sqlite3.connect(
                f"file:{active.weather_db}?mode=ro&immutable=1",
                uri=True,
            ) as conn:
                rejected_rows = conn.execute(
                    "SELECT COUNT(*) FROM weather_fetch_status WHERE normalized_city = ?",
                    ("Rejected Candidate, XX",),
                ).fetchone()[0]
            self.assertEqual(rejected_rows, 0)
            self.assertIn(
                "weather_extension_incomplete",
                failed["snapshot_promotion"]["fail_closed_reasons"],
            )
            staging = corpus_product_root(paths.root) / ".staging"
            self.assertEqual(list(staging.iterdir()) if staging.exists() else [], [])

    def test_idempotent_rerun_keeps_pointer_and_snapshot_directory(self) -> None:
        as_of = datetime(2026, 7, 9, 20, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as directory:
            paths = _seed_market(Path(directory))
            with _patched_weather_fetches(directory) as fetches:
                first = _run(paths, as_of=as_of)
                first_calls = tuple(item.call_count for item in fetches)
                pointer_path = active_pointer_path(paths.root)
                pointer_before = pointer_path.read_bytes()
                snapshot_dirs_before = sorted(
                    path.name
                    for path in (corpus_product_root(paths.root) / "snapshots").iterdir()
                    if path.is_dir()
                )
                second = _run(paths, as_of=as_of)
                second_calls = tuple(item.call_count for item in fetches)

            self.assertEqual(first["snapshot_promotion"]["status"], "promoted")
            self.assertEqual(second["snapshot_promotion"]["status"], "idempotent_no_change")
            self.assertFalse(second["snapshot_promotion"]["pointer_changed"])
            self.assertEqual(pointer_path.read_bytes(), pointer_before)
            self.assertEqual(first_calls, second_calls)
            self.assertEqual(
                sorted(
                    path.name
                    for path in (corpus_product_root(paths.root) / "snapshots").iterdir()
                    if path.is_dir()
                ),
                snapshot_dirs_before,
            )

    def test_writer_contention_and_stale_lock_fail_closed_without_stealing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with exclusive_corpus_writer_lock(root):
                with self.assertRaisesRegex(
                    CorpusWriterLockError,
                    "corpus_writer_lock_held:active_or_unknown_lock",
                ):
                    with exclusive_corpus_writer_lock(root):
                        self.fail("second writer acquired the lock")
            stale_path = writer_lock_path(root)
            stale_path.parent.mkdir(parents=True, exist_ok=True)
            stale_path.write_text(
                json.dumps(
                    {
                        "schema_version": "weather_corpus_writer_lock_v1",
                        "owner_token": "abandoned",
                        "pid": 1,
                        "hostname": "fixture",
                        "acquired_at_utc": "2020-01-01T00:00:00+00:00",
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                CorpusWriterLockError,
                "stale_lock_manual_review_required",
            ):
                with exclusive_corpus_writer_lock(root, stale_after_seconds=1):
                    self.fail("stale lock was silently stolen")
            self.assertTrue(stale_path.exists())
            self.assertIn("abandoned", stale_path.read_text(encoding="utf-8"))

    def test_corrupt_active_pointer_fails_closed(self) -> None:
        as_of = datetime(2026, 7, 9, 20, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as directory:
            paths = _seed_market(Path(directory))
            with _patched_weather_fetches(directory):
                _run(paths, as_of=as_of)
            pointer_path = active_pointer_path(paths.root)
            pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
            pointer["manifest_sha256"] = "0" * 64
            pointer_path.write_text(json.dumps(pointer), encoding="utf-8")

            with self.assertRaisesRegex(
                CorpusSnapshotError,
                "active_corpus_manifest_hash_mismatch",
            ):
                resolve_corpus_snapshot(paths.root)
            coverage = weather_corpus_coverage(
                work_root=paths.root,
                start=date(2026, 7, 7),
                end=date(2026, 7, 7),
                as_of_utc=as_of,
            )
            self.assertFalse(coverage["coverage_available"])
            self.assertFalse(coverage["operationally_fresh"])
            self.assertEqual(
                coverage["fail_closed_reasons"],
                ["active_corpus_snapshot_invalid"],
            )

    def test_snapshot_sidecars_fail_closed_and_narrow_repair_restores_resolution(self) -> None:
        as_of = datetime(2026, 7, 9, 20, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as directory:
            paths = _seed_market(Path(directory))
            with _patched_weather_fetches(directory):
                report = _run(paths, as_of=as_of)
            resolution = resolve_corpus_snapshot(paths.root)
            snapshot_dir = resolution.market_db.parent
            original_mode = stat.S_IMODE(snapshot_dir.stat().st_mode)
            try:
                snapshot_dir.chmod(original_mode | stat.S_IWUSR)
                wal_path = Path(f"{resolution.market_db}-wal")
                wal_path.write_bytes(b"not-safe-to-remove")
                Path(f"{resolution.market_db}-shm").write_bytes(b"derived-shm")
            finally:
                snapshot_dir.chmod(original_mode)

            with self.assertRaisesRegex(
                CorpusSnapshotError,
                "active_corpus_snapshot_sidecar_present",
            ):
                resolve_corpus_snapshot(paths.root)
            with self.assertRaisesRegex(
                CorpusSnapshotError,
                "repair_refused_nonzero_wal",
            ):
                repair_active_snapshot_derived_sidecars(
                    paths.root,
                    expected_snapshot_id=str(report["snapshot_promotion"]["snapshot_id"]),
                )
            self.assertTrue(wal_path.exists())
            try:
                snapshot_dir.chmod(original_mode | stat.S_IWUSR)
                wal_path.write_bytes(b"")
            finally:
                snapshot_dir.chmod(original_mode)
            repaired = repair_active_snapshot_derived_sidecars(
                paths.root,
                expected_snapshot_id=str(report["snapshot_promotion"]["snapshot_id"]),
            )
            self.assertEqual(repaired["status"], "repaired_derived_sidecars_only")
            self.assertEqual(len(repaired["removed"]), 2)
            self.assertFalse(Path(f"{resolution.market_db}-wal").exists())
            self.assertFalse(Path(f"{resolution.market_db}-shm").exists())
            self.assertEqual(
                resolve_corpus_snapshot(paths.root).snapshot_id,
                resolution.snapshot_id,
            )

    def test_candidate_normalization_replaces_database_under_held_reader(self) -> None:
        """A lingering reader must not make candidate normalization order-dependent."""

        with tempfile.TemporaryDirectory() as directory:
            paths = _seed_market(Path(directory))
            candidate = create_corpus_candidate(paths)
            reader = None
            try:
                with closing(sqlite3.connect(candidate.paths.market_db)) as conn:
                    self.assertEqual(
                        str(conn.execute("PRAGMA journal_mode=WAL").fetchone()[0]).lower(),
                        "wal",
                    )
                reader = sqlite3.connect(candidate.paths.market_db)
                reader.execute("BEGIN")
                self.assertEqual(
                    reader.execute("SELECT COUNT(*) FROM market_universe").fetchone()[0],
                    1,
                )
                with closing(sqlite3.connect(candidate.paths.market_db)) as writer:
                    writer.execute("CREATE TABLE normalization_probe(value TEXT NOT NULL)")
                    writer.execute(
                        "INSERT INTO normalization_probe(value) VALUES (?)",
                        ("committed-in-wal",),
                    )
                    writer.commit()
                self.assertGreater(
                    Path(f"{candidate.paths.market_db}-wal").stat().st_size,
                    0,
                )
                original_inode = candidate.paths.market_db.stat().st_ino

                normalize_corpus_candidate_databases(candidate)

                self.assertNotEqual(candidate.paths.market_db.stat().st_ino, original_inode)
                with closing(
                    sqlite3.connect(
                        f"file:{candidate.paths.market_db}?mode=ro&immutable=1",
                        uri=True,
                    )
                ) as conn:
                    self.assertEqual(
                        str(conn.execute("PRAGMA journal_mode").fetchone()[0]).lower(),
                        "delete",
                    )
                    self.assertEqual(
                        conn.execute("SELECT COUNT(*) FROM market_universe").fetchone()[0],
                        1,
                    )
                    self.assertEqual(
                        conn.execute("SELECT value FROM normalization_probe").fetchone()[0],
                        "committed-in-wal",
                    )
                for database in (
                    candidate.paths.market_db,
                    candidate.paths.weather_db,
                ):
                    self.assertFalse(Path(f"{database}-wal").exists())
                    self.assertFalse(Path(f"{database}-shm").exists())
            finally:
                if reader is not None:
                    reader.close()
                discard_corpus_candidate(candidate)


def _run(paths: PipelinePaths, *, as_of: datetime) -> dict[str, object]:
    return run_weather_corpus_extension(
        paths=paths,
        start=date(2026, 7, 7),
        end=date(2026, 7, 7),
        as_of_utc=as_of,
        cache_only=True,
        market_discovery_mode="cached",
    )


def _seed_market(root: Path) -> PipelinePaths:
    paths = PipelinePaths(root)
    initialize_weather_corpus(paths)
    raw = {
        "id": "market-beijing-snapshot",
        "conditionId": "condition-market-beijing-snapshot",
        "slug": "highest-temperature-in-beijing-on-july-7-2026",
        "eventSlug": "highest-temperature-in-beijing-on-july-7-2026",
        "eventTitle": "Highest temperature in Beijing on July 7 2026?",
        "question": "Will the highest temperature in Beijing be 85°F on July 7?",
        "outcomes": '["Yes", "No"]',
        "clobTokenIds": '["yes-beijing-snapshot", "no-beijing-snapshot"]',
        "outcomePrices": '["0.50", "0.50"]',
        "endDate": "2026-07-08T12:00:00Z",
    }
    market = parse_weather_market(raw, today=date(2026, 7, 9))
    assert market is not None
    from daily_weather_pipeline.weather_corpus_extension import gamma_market_to_universe_row

    row = gamma_market_to_universe_row(
        market,
        raw=raw,
        source_path=root / "gamma-event.json",
        payload_id="payload-market-beijing-snapshot",
    )
    with _connect(paths.market_db) as conn:
        _insert_market_row(conn, row, "test-run")
        conn.commit()
    return paths


@contextmanager
def _patched_weather_fetches(directory: str):
    def fake_forecast(pair, start_date, end_date, *, paths, cache_only):
        del start_date, end_date, paths, cache_only
        payload = _weather_payload(str(pair["local_date"]), 85.0)
        return payload, Path(directory) / "forecast.json", "forecast-payload", "forecast-url", {}

    def fake_realized(pair, start_date, end_date, *, paths, cache_only):
        del start_date, end_date, paths, cache_only
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
