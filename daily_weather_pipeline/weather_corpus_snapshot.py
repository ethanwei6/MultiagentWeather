from __future__ import annotations

import hashlib
import json
import os
import shutil
import socket
import sqlite3
import stat
import uuid
from contextlib import closing, contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional

from .cli import PipelinePaths


SNAPSHOT_SCHEMA_VERSION = "weather_corpus_snapshot_v1"
POINTER_SCHEMA_VERSION = "weather_corpus_active_pointer_v1"
DEFAULT_STALE_LOCK_SECONDS = 3600.0


class CorpusSnapshotError(ValueError):
    """The active corpus snapshot cannot be trusted or resolved."""


class CorpusWriterLockError(CorpusSnapshotError):
    """Another writer, or an unresolved stale lock, owns the corpus writer lock."""


@dataclass(frozen=True)
class CorpusResolution:
    market_db: Path
    weather_db: Path
    snapshot_id: Optional[str]
    source: str
    manifest: Optional[Mapping[str, Any]] = None
    manifest_path: Optional[Path] = None
    active_pointer: Optional[Path] = None


@dataclass(frozen=True)
class CorpusCandidate:
    staging_dir: Path
    paths: "CorpusCandidatePaths"
    base_resolution: CorpusResolution


@dataclass(frozen=True)
class CorpusCandidatePaths:
    base: PipelinePaths
    candidate_market_db: Path
    candidate_weather_db: Path

    @property
    def root(self) -> Path:
        return self.base.root

    @property
    def work(self) -> Path:
        return self.base.work

    @property
    def market_db(self) -> Path:
        return self.candidate_market_db

    @property
    def weather_db(self) -> Path:
        return self.candidate_weather_db

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base, name)


def corpus_product_root(work_root: Path) -> Path:
    return Path(work_root).resolve() / "work" / "data" / "weather_corpus"


def active_pointer_path(work_root: Path) -> Path:
    return corpus_product_root(work_root) / "active.json"


def writer_lock_path(work_root: Path) -> Path:
    return corpus_product_root(work_root) / "writer.lock"


def resolve_corpus_snapshot(
    work_root: Path,
    *,
    legacy_market_db: Optional[Path] = None,
    legacy_weather_db: Optional[Path] = None,
) -> CorpusResolution:
    """Resolve and verify one market/weather pair from the atomic pointer.

    A missing pointer may use the legacy seed paths.  Once a pointer exists, any
    corruption fails closed; readers never silently fall back to mutable legacy
    stores.
    """

    root = Path(work_root).resolve()
    product_root = corpus_product_root(root)
    pointer_path = active_pointer_path(root)
    if not pointer_path.exists():
        base = root / "work" / "data" / "backtest"
        market_db = Path(legacy_market_db or base / "daily_weather_market_universe.sqlite")
        weather_db = Path(legacy_weather_db or base / "daily_weather_weather_history.sqlite")
        return CorpusResolution(
            market_db=market_db,
            weather_db=weather_db,
            snapshot_id=None,
            source="legacy_unversioned_seed",
            active_pointer=None,
        )

    try:
        pointer_bytes = pointer_path.read_bytes()
        pointer = json.loads(pointer_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CorpusSnapshotError(
            f"active_corpus_pointer_unreadable:{type(exc).__name__}:{str(exc)[:300]}"
        ) from exc
    if not isinstance(pointer, Mapping) or pointer.get("schema_version") != POINTER_SCHEMA_VERSION:
        raise CorpusSnapshotError("active_corpus_pointer_schema_invalid")
    snapshot_id = str(pointer.get("snapshot_id") or "")
    if (
        len(snapshot_id) != 64
        or any(ch not in "0123456789abcdef" for ch in snapshot_id)
    ):
        raise CorpusSnapshotError("active_corpus_pointer_snapshot_id_invalid")
    snapshot_dir = _confined_path(product_root, pointer.get("snapshot_dir"))
    manifest_path = _confined_path(product_root, pointer.get("manifest"))
    expected_snapshot_dir = (product_root / "snapshots" / snapshot_id).resolve()
    if snapshot_dir != expected_snapshot_dir:
        raise CorpusSnapshotError("active_corpus_pointer_snapshot_directory_mismatch")
    if manifest_path != snapshot_dir / "manifest.json":
        raise CorpusSnapshotError("active_corpus_pointer_manifest_directory_mismatch")
    try:
        manifest_bytes = manifest_path.read_bytes()
    except OSError as exc:
        raise CorpusSnapshotError("active_corpus_manifest_missing") from exc
    if _sha256_bytes(manifest_bytes) != str(pointer.get("manifest_sha256") or ""):
        raise CorpusSnapshotError("active_corpus_manifest_hash_mismatch")
    try:
        manifest = json.loads(manifest_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CorpusSnapshotError("active_corpus_manifest_invalid_json") from exc
    if not isinstance(manifest, Mapping):
        raise CorpusSnapshotError("active_corpus_manifest_not_mapping")
    if manifest.get("schema_version") != SNAPSHOT_SCHEMA_VERSION:
        raise CorpusSnapshotError("active_corpus_manifest_schema_invalid")
    if str(manifest.get("snapshot_id") or "") != snapshot_id:
        raise CorpusSnapshotError("active_corpus_manifest_snapshot_id_mismatch")
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise CorpusSnapshotError("active_corpus_manifest_files_missing")
    market_db = _verified_manifest_file(snapshot_dir, files, "market_db")
    weather_db = _verified_manifest_file(snapshot_dir, files, "weather_db")
    _reject_snapshot_sidecars(market_db, weather_db)
    return CorpusResolution(
        market_db=market_db,
        weather_db=weather_db,
        snapshot_id=snapshot_id,
        source="active_immutable_snapshot",
        manifest=manifest,
        manifest_path=manifest_path,
        active_pointer=pointer_path,
    )


def resolve_explicit_corpus_override(
    market_db: Path,
    weather_db: Path,
) -> CorpusResolution:
    """Validate unversioned caller-supplied corpus paths without assuming immutability.

    Explicit overrides do not carry the active snapshot manifest proof.  They
    must therefore be opened with ordinary read-only SQLite semantics, and any
    WAL/SHM sidecar makes the pair ambiguous and is rejected before use.
    """

    resolved_market = Path(market_db).expanduser().resolve()
    resolved_weather = Path(weather_db).expanduser().resolve()
    for database in (resolved_market, resolved_weather):
        if not database.is_file():
            raise CorpusSnapshotError(
                f"explicit_corpus_override_database_missing:{database}"
            )
    reject_corpus_database_sidecars(
        resolved_market,
        resolved_weather,
        reason_prefix="explicit_corpus_override",
    )
    for database in (resolved_market, resolved_weather):
        journal_mode = _sqlite_header_journal_mode(database)
        if journal_mode != "delete":
            raise CorpusSnapshotError(
                "explicit_corpus_override_journal_mode_not_delete:"
                f"{database.name}:{journal_mode}"
            )
    return CorpusResolution(
        market_db=resolved_market,
        weather_db=resolved_weather,
        snapshot_id=None,
        source="explicit_sidecar_free_override",
    )


def resolve_controlled_corpus_candidate(
    work_root: Path,
    market_db: Path,
    weather_db: Path,
) -> CorpusResolution:
    """Resolve only a pipeline-owned copy-on-write candidate staging pair.

    A cleanly-closed SQLite connection can leave a zero-byte WAL marker on
    some runtimes.  That marker contains no committed frames and is safe to
    tolerate for this lock-protected candidate read.  Nonzero WAL and every
    SHM remain ambiguous and fail closed.  This function never deletes them.
    """

    staging_root = (corpus_product_root(work_root) / ".staging").resolve()
    resolved_market = Path(market_db).expanduser().resolve()
    resolved_weather = Path(weather_db).expanduser().resolve()
    candidate_dir = resolved_market.parent
    if (
        resolved_market.name != "market.sqlite"
        or resolved_weather.name != "weather.sqlite"
        or resolved_weather.parent != candidate_dir
        or candidate_dir.parent != staging_root
        or len(candidate_dir.name) != 32
        or any(ch not in "0123456789abcdef" for ch in candidate_dir.name)
    ):
        raise CorpusSnapshotError("controlled_corpus_candidate_path_invalid")
    for database in (resolved_market, resolved_weather):
        if not database.is_file():
            raise CorpusSnapshotError(
                f"controlled_corpus_candidate_database_missing:{database}"
            )
    reject_corpus_database_sidecars(
        resolved_market,
        resolved_weather,
        reason_prefix="copy_on_write_candidate",
        allow_zero_length_wal=True,
    )
    for database in (resolved_market, resolved_weather):
        journal_mode = _sqlite_header_journal_mode(database)
        if journal_mode != "delete":
            raise CorpusSnapshotError(
                "copy_on_write_candidate_journal_mode_not_delete:"
                f"{database.name}:{journal_mode}"
            )
    return CorpusResolution(
        market_db=resolved_market,
        weather_db=resolved_weather,
        snapshot_id=None,
        source="copy_on_write_candidate",
    )


def active_corpus_lineage_artifacts(work_root: Path) -> Mapping[str, Any]:
    """Return the exact verified files A4/lineage manifests must hash and pin."""

    resolution = resolve_corpus_snapshot(work_root)
    if (
        resolution.snapshot_id is None
        or resolution.manifest is None
        or resolution.manifest_path is None
        or resolution.active_pointer is None
    ):
        raise CorpusSnapshotError("active_immutable_corpus_snapshot_required_for_lineage")
    files = resolution.manifest.get("files")
    if not isinstance(files, Mapping):
        raise CorpusSnapshotError("active_corpus_manifest_files_missing")
    return {
        "snapshot_id": resolution.snapshot_id,
        "active_pointer": str(resolution.active_pointer),
        "active_pointer_sha256": _sha256_file(resolution.active_pointer),
        "manifest": str(resolution.manifest_path),
        "manifest_sha256": _sha256_file(resolution.manifest_path),
        "market_db": str(resolution.market_db),
        "market_db_sha256": str((files.get("market_db") or {}).get("sha256") or ""),
        "weather_db": str(resolution.weather_db),
        "weather_db_sha256": str((files.get("weather_db") or {}).get("sha256") or ""),
        "semantic_corpus_sha256": str(
            resolution.manifest.get("semantic_corpus_sha256") or ""
        ),
    }


def repair_active_snapshot_derived_sidecars(
    work_root: Path,
    *,
    expected_snapshot_id: str,
) -> Mapping[str, Any]:
    """Remove only verified derived WAL/SHM files from the active snapshot.

    Main files must match the active manifest and every WAL must be zero bytes.
    Any mismatch aborts before deletion.
    """

    root = Path(work_root).resolve()
    product_root = corpus_product_root(root)
    pointer_path = active_pointer_path(root)
    try:
        pointer_bytes = pointer_path.read_bytes()
        pointer = json.loads(pointer_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CorpusSnapshotError("repair_active_corpus_pointer_unreadable") from exc
    if not isinstance(pointer, Mapping):
        raise CorpusSnapshotError("repair_active_corpus_pointer_invalid")
    snapshot_id = str(pointer.get("snapshot_id") or "")
    if snapshot_id != expected_snapshot_id:
        raise CorpusSnapshotError(
            f"repair_snapshot_id_mismatch:expected={expected_snapshot_id}:actual={snapshot_id}"
        )
    snapshot_dir = _confined_path(product_root, pointer.get("snapshot_dir"))
    if snapshot_dir != (product_root / "snapshots" / expected_snapshot_id).resolve():
        raise CorpusSnapshotError("repair_snapshot_directory_mismatch")
    manifest_path = _confined_path(product_root, pointer.get("manifest"))
    if manifest_path != snapshot_dir / "manifest.json":
        raise CorpusSnapshotError("repair_manifest_path_mismatch")
    manifest_bytes = manifest_path.read_bytes()
    if _sha256_bytes(manifest_bytes) != str(pointer.get("manifest_sha256") or ""):
        raise CorpusSnapshotError("repair_manifest_hash_mismatch")
    try:
        manifest = json.loads(manifest_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CorpusSnapshotError("repair_manifest_invalid") from exc
    if not isinstance(manifest, Mapping) or str(manifest.get("snapshot_id") or "") != snapshot_id:
        raise CorpusSnapshotError("repair_manifest_snapshot_id_mismatch")
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise CorpusSnapshotError("repair_manifest_files_missing")
    market_db = _verified_manifest_file(snapshot_dir, files, "market_db")
    weather_db = _verified_manifest_file(snapshot_dir, files, "weather_db")
    sidecars: list[tuple[Path, int]] = []
    for database in (market_db, weather_db):
        wal = Path(f"{database}-wal")
        shm = Path(f"{database}-shm")
        if wal.exists() and wal.stat().st_size != 0:
            raise CorpusSnapshotError(
                f"repair_refused_nonzero_wal:{wal}:{wal.stat().st_size}"
            )
        for sidecar in (wal, shm):
            if sidecar.exists():
                sidecars.append((sidecar, sidecar.stat().st_size))
    original_mode = stat.S_IMODE(snapshot_dir.stat().st_mode)
    removed: list[Mapping[str, Any]] = []
    try:
        snapshot_dir.chmod(original_mode | stat.S_IWUSR)
        for sidecar, size in sidecars:
            sidecar.unlink()
            removed.append({"path": str(sidecar), "bytes": size})
    finally:
        snapshot_dir.chmod(original_mode)
    resolution = resolve_corpus_snapshot(root)
    assert resolution.manifest_path is not None
    assert resolution.active_pointer is not None
    return {
        "status": "repaired_derived_sidecars_only",
        "snapshot_id": resolution.snapshot_id,
        "main_files_verified_against_manifest": True,
        "zero_length_wal_required": True,
        "removed": removed,
        "market_db_sha256": _sha256_file(resolution.market_db),
        "weather_db_sha256": _sha256_file(resolution.weather_db),
        "manifest_sha256": _sha256_file(resolution.manifest_path),
        "active_pointer_sha256": _sha256_file(resolution.active_pointer),
    }


@contextmanager
def exclusive_corpus_writer_lock(
    work_root: Path,
    *,
    stale_after_seconds: float = DEFAULT_STALE_LOCK_SECONDS,
) -> Iterator[Mapping[str, Any]]:
    """Acquire the single corpus-writer lock without stealing stale locks."""

    lock_path = writer_lock_path(work_root)
    _ensure_directory_durable(lock_path.parent)
    token = uuid.uuid4().hex
    acquired_at = datetime.now(timezone.utc)
    payload = {
        "schema_version": "weather_corpus_writer_lock_v1",
        "owner_token": token,
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "acquired_at_utc": acquired_at.isoformat(),
    }
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        existing = _read_lock_observation(lock_path, stale_after_seconds=stale_after_seconds)
        state = str(existing.get("state") or "active_or_unknown")
        raise CorpusWriterLockError(
            f"corpus_writer_lock_held:{state}:{json.dumps(existing, sort_keys=True)}"
        ) from exc
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(lock_path.parent)
        yield payload
    finally:
        try:
            current = json.loads(lock_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            current = None
        if isinstance(current, Mapping) and current.get("owner_token") == token:
            lock_path.unlink(missing_ok=True)
            _fsync_directory(lock_path.parent)


def create_corpus_candidate(paths: PipelinePaths) -> CorpusCandidate:
    resolution = resolve_corpus_snapshot(
        paths.root,
        legacy_market_db=paths.market_db,
        legacy_weather_db=paths.weather_db,
    )
    for source in (resolution.market_db, resolution.weather_db):
        if not source.exists():
            raise CorpusSnapshotError(f"corpus_candidate_seed_missing:{source}")
    staging_root = corpus_product_root(paths.root) / ".staging"
    _ensure_directory_durable(staging_root)
    staging_dir = staging_root / uuid.uuid4().hex
    staging_dir.mkdir(parents=True, exist_ok=False)
    _fsync_directory(staging_root)
    market_db = staging_dir / "market.sqlite"
    weather_db = staging_dir / "weather.sqlite"
    try:
        immutable_source = resolution.source == "active_immutable_snapshot"
        _sqlite_backup(
            resolution.market_db,
            market_db,
            immutable_source=immutable_source,
        )
        _sqlite_backup(
            resolution.weather_db,
            weather_db,
            immutable_source=immutable_source,
        )
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise
    candidate_paths = CorpusCandidatePaths(
        base=paths,
        candidate_market_db=market_db,
        candidate_weather_db=weather_db,
    )
    return CorpusCandidate(
        staging_dir=staging_dir,
        paths=candidate_paths,
        base_resolution=resolution,
    )


def discard_corpus_candidate(candidate: CorpusCandidate) -> None:
    if candidate.staging_dir.exists():
        parent = candidate.staging_dir.parent
        shutil.rmtree(candidate.staging_dir)
        _fsync_directory(parent)


def normalize_corpus_candidate_databases(candidate: CorpusCandidate) -> None:
    """Atomically replace both candidates with fsynced DELETE-journal backups."""

    if not candidate.staging_dir.exists():
        raise CorpusSnapshotError("corpus_candidate_staging_missing")
    for database in (candidate.paths.market_db, candidate.paths.weather_db):
        _normalize_candidate_sqlite(database)
    _reject_snapshot_sidecars(
        candidate.paths.market_db,
        candidate.paths.weather_db,
        reason_prefix="corpus_candidate",
    )


def promote_corpus_candidate(
    candidate: CorpusCandidate,
    *,
    validation_summary: Mapping[str, Any],
) -> Mapping[str, Any]:
    if not candidate.staging_dir.exists():
        raise CorpusSnapshotError("corpus_candidate_staging_missing")
    if not bool(validation_summary.get("operationally_fresh")):
        raise CorpusSnapshotError("corpus_candidate_validation_not_fresh")
    normalize_corpus_candidate_databases(candidate)
    semantic_hash = _semantic_corpus_hash(
        candidate.paths.market_db,
        candidate.paths.weather_db,
    )
    active = candidate.base_resolution
    active_semantic_hash = (
        str((active.manifest or {}).get("semantic_corpus_sha256") or "")
        if active.snapshot_id
        else ""
    )
    if active.snapshot_id and active_semantic_hash == semantic_hash:
        discard_corpus_candidate(candidate)
        return {
            "status": "idempotent_no_change",
            "snapshot_id": active.snapshot_id,
            "active_pointer": str(active.active_pointer),
            "semantic_corpus_sha256": semantic_hash,
            "previous_snapshot_id": active.snapshot_id,
            "pointer_changed": False,
        }

    product_root = corpus_product_root(candidate.paths.root)
    snapshot_dir = product_root / "snapshots" / semantic_hash
    manifest_path = candidate.staging_dir / "manifest.json"
    manifest = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "snapshot_id": semantic_hash,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "parent_snapshot_id": active.snapshot_id,
        "semantic_corpus_sha256": semantic_hash,
        "immutable": True,
        "files": {
            "market_db": _file_manifest(candidate.paths.market_db),
            "weather_db": _file_manifest(candidate.paths.weather_db),
        },
        "validation": _jsonable(validation_summary),
    }
    _atomic_write_json(manifest_path, manifest)
    if snapshot_dir.exists():
        existing = _verify_existing_snapshot(snapshot_dir, semantic_hash)
        discard_corpus_candidate(candidate)
        manifest_path = snapshot_dir / "manifest.json"
        manifest = existing
        _fsync_snapshot_tree(snapshot_dir)
    else:
        _ensure_directory_durable(snapshot_dir.parent)
        _fsync_file(candidate.paths.market_db)
        _fsync_file(candidate.paths.weather_db)
        _fsync_file(manifest_path)
        _fsync_directory(candidate.staging_dir)
        _durable_replace(candidate.staging_dir, snapshot_dir)
        manifest_path = snapshot_dir / "manifest.json"
        for path in (
            snapshot_dir / "market.sqlite",
            snapshot_dir / "weather.sqlite",
            manifest_path,
        ):
            path.chmod(0o444)
            _fsync_file(path)
        snapshot_dir.chmod(0o555)
        _fsync_directory(snapshot_dir)
        _fsync_directory(snapshot_dir.parent)
        manifest = _verify_existing_snapshot(snapshot_dir, semantic_hash)
    manifest_bytes = manifest_path.read_bytes()
    pointer = {
        "schema_version": POINTER_SCHEMA_VERSION,
        "snapshot_id": semantic_hash,
        "snapshot_dir": str(snapshot_dir.relative_to(product_root)),
        "manifest": str(manifest_path.relative_to(product_root)),
        "manifest_sha256": _sha256_bytes(manifest_bytes),
        "promoted_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    pointer_path = active_pointer_path(candidate.paths.root)
    _atomic_write_json(pointer_path, pointer)
    resolved = resolve_corpus_snapshot(candidate.paths.root)
    return {
        "status": "promoted",
        "snapshot_id": resolved.snapshot_id,
        "snapshot_dir": str(snapshot_dir),
        "manifest": str(manifest_path),
        "active_pointer": str(pointer_path),
        "semantic_corpus_sha256": semantic_hash,
        "previous_snapshot_id": active.snapshot_id,
        "pointer_changed": True,
    }


def _verified_manifest_file(
    snapshot_dir: Path,
    files: Mapping[str, Any],
    key: str,
) -> Path:
    entry = files.get(key)
    if not isinstance(entry, Mapping):
        raise CorpusSnapshotError(f"active_corpus_manifest_{key}_missing")
    path = _confined_path(snapshot_dir, entry.get("path"))
    if path.parent != snapshot_dir or not path.is_file():
        raise CorpusSnapshotError(f"active_corpus_manifest_{key}_path_invalid")
    if path.stat().st_size != int(entry.get("bytes") or -1):
        raise CorpusSnapshotError(f"active_corpus_manifest_{key}_size_mismatch")
    if _sha256_file(path) != str(entry.get("sha256") or ""):
        raise CorpusSnapshotError(f"active_corpus_manifest_{key}_hash_mismatch")
    return path


def _verify_existing_snapshot(snapshot_dir: Path, snapshot_id: str) -> Mapping[str, Any]:
    manifest_path = snapshot_dir / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CorpusSnapshotError("existing_semantic_snapshot_invalid") from exc
    if not isinstance(manifest, Mapping):
        raise CorpusSnapshotError("existing_semantic_snapshot_manifest_not_mapping")
    if manifest.get("schema_version") != SNAPSHOT_SCHEMA_VERSION:
        raise CorpusSnapshotError("existing_semantic_snapshot_schema_invalid")
    if str(manifest.get("snapshot_id") or "") != snapshot_id:
        raise CorpusSnapshotError("existing_semantic_snapshot_id_mismatch")
    if str(manifest.get("semantic_corpus_sha256") or "") != snapshot_id:
        raise CorpusSnapshotError("existing_semantic_snapshot_hash_mismatch")
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise CorpusSnapshotError("existing_semantic_snapshot_files_missing")
    market_db = _verified_manifest_file(snapshot_dir, files, "market_db")
    weather_db = _verified_manifest_file(snapshot_dir, files, "weather_db")
    _reject_snapshot_sidecars(market_db, weather_db)
    return manifest


def _read_lock_observation(lock_path: Path, *, stale_after_seconds: float) -> Mapping[str, Any]:
    observed_at = datetime.now(timezone.utc)
    try:
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "state": "unreadable_lock_manual_review_required",
            "path": str(lock_path),
            "error": f"{type(exc).__name__}: {str(exc)[:300]}",
        }
    acquired_at_raw = payload.get("acquired_at_utc") if isinstance(payload, Mapping) else None
    try:
        acquired_at = datetime.fromisoformat(str(acquired_at_raw).replace("Z", "+00:00"))
        if acquired_at.tzinfo is None:
            acquired_at = acquired_at.replace(tzinfo=timezone.utc)
        age_seconds = max(0.0, (observed_at - acquired_at.astimezone(timezone.utc)).total_seconds())
    except (TypeError, ValueError):
        age_seconds = None
    stale = age_seconds is not None and age_seconds > float(stale_after_seconds)
    return {
        "state": (
            "stale_lock_manual_review_required" if stale else "active_or_unknown_lock"
        ),
        "path": str(lock_path),
        "age_seconds": age_seconds,
        "stale_after_seconds": float(stale_after_seconds),
        "owner": _jsonable(payload) if isinstance(payload, Mapping) else None,
    }


def _sqlite_backup(
    source: Path,
    destination: Path,
    *,
    immutable_source: bool,
) -> None:
    source_uri = _read_only_sqlite_uri(source, immutable=immutable_source)
    with closing(
        sqlite3.connect(source_uri, uri=True, timeout=30)
    ) as source_conn:
        with closing(sqlite3.connect(destination, timeout=30)) as destination_conn:
            source_conn.backup(destination_conn)
            destination_conn.commit()


def _normalize_candidate_sqlite(path: Path) -> None:
    """Normalize through a fresh inode so lingering readers cannot block it.

    Candidate writers have already closed and committed before this boundary.
    A SQLite backup therefore captures the complete committed view, including
    committed WAL frames, while avoiding an in-place journal-mode transition
    against any read handle that has not yet been released.
    """

    if not path.is_file():
        raise CorpusSnapshotError(f"corpus_candidate_database_missing:{path}")
    if _candidate_sqlite_is_normalized(path):
        return

    temporary = path.with_name(f".{path.name}.normalized-{uuid.uuid4().hex}.tmp")
    temporary_sidecars = tuple(
        Path(f"{temporary}{suffix}") for suffix in ("-wal", "-shm")
    )
    original_mode = stat.S_IMODE(path.stat().st_mode)
    replaced = False
    try:
        source_wal = Path(f"{path}-wal")
        source_shm = Path(f"{path}-shm")
        source_has_committed_wal_or_shm = (
            source_shm.exists()
            or (source_wal.exists() and source_wal.stat().st_size > 0)
        )
        # A WAL-mode header with no sidecars has no outstanding frames to
        # merge.  Immutable read-only is required on older SQLite builds that
        # otherwise try (and fail) to create a missing SHM file.  A zero-byte
        # WAL is equivalent to no frames.  When a nonzero WAL or SHM exists,
        # ordinary read-only mode is mandatory so backup includes the complete
        # committed SQLite view.
        source_uri = _read_only_sqlite_uri(
            path,
            immutable=not source_has_committed_wal_or_shm,
        )
        with closing(
            sqlite3.connect(source_uri, uri=True, timeout=30)
        ) as source_conn:
            with closing(sqlite3.connect(temporary, timeout=30)) as destination_conn:
                source_conn.backup(destination_conn)
                destination_conn.commit()
                journal_mode_row = destination_conn.execute(
                    "PRAGMA journal_mode=DELETE"
                ).fetchone()
                journal_mode = str((journal_mode_row or ("",))[0]).lower()
                quick_check = destination_conn.execute("PRAGMA quick_check").fetchone()
                destination_conn.commit()
        if journal_mode != "delete":
            raise CorpusSnapshotError(
                f"corpus_candidate_non_delete_journal_mode:{path}:{journal_mode}"
            )
        if not quick_check or str(quick_check[0]).lower() != "ok":
            raise CorpusSnapshotError(
                f"corpus_candidate_backup_quick_check_failed:{path}:{quick_check}"
            )
        for sidecar in temporary_sidecars:
            if sidecar.name.endswith("-wal") and sidecar.exists() and sidecar.stat().st_size:
                raise CorpusSnapshotError(
                    f"corpus_candidate_normalized_temporary_nonzero_wal:"
                    f"{sidecar}:{sidecar.stat().st_size}"
                )
            sidecar.unlink(missing_ok=True)
        _reject_snapshot_sidecars(
            temporary,
            reason_prefix="corpus_candidate_normalized_temporary",
        )
        temporary.chmod(original_mode)
        _fsync_file(temporary)
        _durable_replace(temporary, path)
        replaced = True

        # These names belong to the replaced source inode.  The backup above
        # already captured every committed WAL frame into the new main file.
        for suffix in ("-wal", "-shm"):
            Path(f"{path}{suffix}").unlink(missing_ok=True)
        _fsync_directory(path.parent)
        _reject_snapshot_sidecars(path, reason_prefix="corpus_candidate")

        with closing(
            sqlite3.connect(_read_only_sqlite_uri(path, immutable=True), uri=True)
        ) as conn:
            verified_quick_check = conn.execute("PRAGMA quick_check").fetchone()
        verified_mode = _sqlite_header_journal_mode(path)
        if verified_mode != "delete":
            raise CorpusSnapshotError(
                f"corpus_candidate_journal_mode_verification_failed:{path}:{verified_mode}"
            )
        if not verified_quick_check or str(verified_quick_check[0]).lower() != "ok":
            raise CorpusSnapshotError(
                f"corpus_candidate_post_replace_quick_check_failed:{path}:{verified_quick_check}"
            )
    finally:
        temporary.unlink(missing_ok=True)
        for sidecar in temporary_sidecars:
            sidecar.unlink(missing_ok=True)
        if replaced:
            _reject_snapshot_sidecars(path, reason_prefix="corpus_candidate")


def _candidate_sqlite_is_normalized(path: Path) -> bool:
    if any(Path(f"{path}{suffix}").exists() for suffix in ("-wal", "-shm")):
        return False
    if _sqlite_header_journal_mode(path) != "delete":
        return False
    try:
        with closing(
            sqlite3.connect(_read_only_sqlite_uri(path, immutable=True), uri=True)
        ) as conn:
            quick_check = conn.execute("PRAGMA quick_check").fetchone()
            return bool(quick_check and str(quick_check[0]).lower() == "ok")
    except sqlite3.DatabaseError:
        return False


def _sqlite_header_journal_mode(path: Path) -> str:
    """Read SQLite's persistent journal-mode bytes without opening the database."""

    try:
        with path.open("rb") as handle:
            header = handle.read(20)
    except OSError:
        return "unreadable"
    if len(header) < 20 or header[:16] != b"SQLite format 3\x00":
        return "invalid"
    write_version = header[18]
    read_version = header[19]
    if write_version == 1 and read_version == 1:
        return "delete"
    if write_version == 2 and read_version == 2:
        return "wal"
    return f"unsupported_{write_version}_{read_version}"


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _ensure_directory_durable(path: Path) -> None:
    """Create a directory tree and persist each newly-created directory entry."""

    target = Path(path)
    missing: list[Path] = []
    cursor = target
    while not cursor.exists():
        missing.append(cursor)
        if cursor.parent == cursor:
            break
        cursor = cursor.parent
    target.mkdir(parents=True, exist_ok=True)
    for created in reversed(missing):
        _fsync_directory(created)
        if created.parent != created:
            _fsync_directory(created.parent)


def _durable_replace(source: Path, destination: Path) -> None:
    """Replace a path and fsync every directory whose entry changed."""

    source = Path(source)
    destination = Path(destination)
    source_parent = source.parent
    destination_parent = destination.parent
    os.replace(source, destination)
    _fsync_directory(destination_parent)
    if source_parent != destination_parent:
        _fsync_directory(source_parent)


def _fsync_snapshot_tree(snapshot_dir: Path) -> None:
    for path in (
        snapshot_dir / "market.sqlite",
        snapshot_dir / "weather.sqlite",
        snapshot_dir / "manifest.json",
    ):
        _fsync_file(path)
    _fsync_directory(snapshot_dir)
    _fsync_directory(snapshot_dir.parent)


def reject_corpus_database_sidecars(
    *databases: Path,
    reason_prefix: str = "corpus_database",
    allow_zero_length_wal: bool = False,
) -> None:
    """Fail closed on ambiguous sidecars without mutating caller paths."""

    for database in databases:
        shm = Path(f"{database}-shm")
        if shm.exists():
            raise CorpusSnapshotError(
                f"{reason_prefix}_sidecar_present:{shm.name}:{shm.stat().st_size}"
            )
        wal = Path(f"{database}-wal")
        if wal.exists():
            size = wal.stat().st_size
            if allow_zero_length_wal and size == 0:
                continue
            raise CorpusSnapshotError(
                f"{reason_prefix}_sidecar_present:{wal.name}:{size}"
            )


def _reject_snapshot_sidecars(
    *databases: Path,
    reason_prefix: str = "active_corpus_snapshot",
) -> None:
    for database in databases:
        for suffix in ("-wal", "-shm"):
            sidecar = Path(f"{database}{suffix}")
            if sidecar.exists():
                raise CorpusSnapshotError(
                    f"{reason_prefix}_sidecar_present:{sidecar.name}:{sidecar.stat().st_size}"
                )


def _read_only_sqlite_uri(path: Path, *, immutable: bool) -> str:
    suffix = "&immutable=1" if immutable else ""
    return f"file:{path}?mode=ro{suffix}"


def _semantic_corpus_hash(market_db: Path, weather_db: Path) -> str:
    digest = hashlib.sha256()
    digest.update(SNAPSHOT_SCHEMA_VERSION.encode("utf-8"))
    _hash_sqlite_tables(
        digest,
        market_db,
        {
            "market_universe": {"first_seen_run_id", "updated_at_utc"},
            "market_tokens": set(),
            "market_parse_warnings": set(),
            "source_payloads": {"fetched_at_utc"},
        },
    )
    _hash_sqlite_tables(
        digest,
        weather_db,
        {
            "forecast_snapshots": {"fetch_timestamp_utc"},
            "realized_daily_weather": {"fetch_timestamp_utc"},
            "weather_fetch_status": {"updated_at_utc"},
            "source_payloads": {"fetched_at_utc"},
        },
    )
    return digest.hexdigest()


def _hash_sqlite_tables(
    digest: "hashlib._Hash",
    path: Path,
    tables: Mapping[str, set[str]],
) -> None:
    with sqlite3.connect(_read_only_sqlite_uri(path, immutable=True), uri=True) as conn:
        table_names = sorted(tables)
        placeholders = ",".join("?" for _ in table_names)
        schema_rows = conn.execute(
            f"""
            SELECT type, name, tbl_name, COALESCE(sql, '')
            FROM sqlite_master
            WHERE tbl_name IN ({placeholders}) AND type IN ('table', 'index')
            ORDER BY type, name, tbl_name
            """,
            table_names,
        ).fetchall()
        digest.update(
            (json.dumps(_jsonable(schema_rows), sort_keys=True, separators=(",", ":")) + "\n").encode(
                "utf-8"
            )
        )
        for table in sorted(tables):
            columns = [
                str(row[1])
                for row in conn.execute(f'PRAGMA table_info("{table}")').fetchall()
                if str(row[1]) not in tables[table]
            ]
            if not columns:
                raise CorpusSnapshotError(f"semantic_hash_table_missing_or_empty_schema:{table}")
            quoted = ", ".join(f'"{column}"' for column in columns)
            order = ", ".join(f'"{column}"' for column in columns)
            digest.update(f"{path.name}:{table}:{','.join(columns)}\n".encode("utf-8"))
            for row in conn.execute(f'SELECT {quoted} FROM "{table}" ORDER BY {order}'):
                digest.update(
                    (json.dumps(_jsonable(list(row)), sort_keys=True, separators=(",", ":")) + "\n").encode(
                        "utf-8"
                    )
                )


def _file_manifest(path: Path) -> Mapping[str, Any]:
    return {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _ensure_directory_durable(path.parent)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    encoded = (json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n").encode("utf-8")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        _durable_replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _confined_path(root: Path, relative: Any) -> Path:
    value = Path(str(relative or ""))
    if value.is_absolute() or not value.parts:
        raise CorpusSnapshotError("active_corpus_pointer_path_invalid")
    resolved_root = root.resolve()
    resolved = (resolved_root / value).resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise CorpusSnapshotError("active_corpus_pointer_path_escape")
    return resolved


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
