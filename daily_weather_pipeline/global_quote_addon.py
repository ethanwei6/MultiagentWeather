from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
import time
from collections import Counter
from datetime import date, datetime, time as datetime_time, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from .cli import (
    DEFAULT_SOURCE_REPO,
    PipelinePaths,
    _compact_failure_class,
    _compact_token_refs,
    _connect,
    _ensure_compact_price_schema,
    _ensure_compact_tokens,
    _finish_pipeline_run,
    _insert_compact_quote_rows,
    _insert_pipeline_run,
    _load_quote_rows_for_job,
    _upsert_compact_status,
    bootstrap,
)


DEFAULT_DISCOVERY_CSV = Path("work/reports/global_weather_expansion/20260707T041734Z-global-weather-market-discovery-sample.csv")
DEFAULT_SAMPLE_JSON = Path("work/reports/global_weather_expansion/20260707T041734Z-telonex-quote-sample-status.json")
DEFAULT_COMPACT_DB = Path("work/data/global_weather_expansion/global_weather_price_history_compact.sqlite")
DEFAULT_DECISION_DB = Path("work/data/global_weather_expansion/global_weather_decision_artifact.sqlite")
DEFAULT_RAW_CACHE = Path("work/cache/global_weather_expansion/telonex/raw")
DEFAULT_REPORT_DIR = Path("work/reports/global_weather_expansion")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Compact-only global weather quote add-on tooling")
    parser.add_argument("--work-root", default=".", help="Workspace root; used only to resolve default global paths")
    parser.add_argument("--source-repo", default=str(DEFAULT_SOURCE_REPO), help="Existing weather_strategy source checkout")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect = subparsers.add_parser("inspect", help="Inspect existing global compact quote artifacts")
    _add_common_path_args(inspect)

    proof = subparsers.add_parser("proof-backfill", help="Run a tiny compact-only Telonex proof backfill")
    _add_common_path_args(proof)
    proof.add_argument("--limit-jobs", type=int, default=4)
    proof.add_argument("--workers", type=int, default=1)
    proof.add_argument("--progress-every", type=int, default=1)
    proof.add_argument("--telonex-env", default=None, help="Local env file containing TELONEX_API_KEY")
    proof.add_argument("--telonex-hard-timeout-seconds", type=float, default=60.0)

    token_day = subparsers.add_parser("token-day-backfill", help="Run a staged compact-only token-day quote backfill")
    _add_common_path_args(token_day)
    token_day.add_argument("--limit-jobs", type=int, default=100)
    token_day.add_argument("--workers", type=int, default=1)
    token_day.add_argument("--progress-every", type=int, default=10)
    token_day.add_argument("--telonex-env", default=None, help="Local env file containing TELONEX_API_KEY")
    token_day.add_argument("--telonex-hard-timeout-seconds", type=float, default=60.0)
    token_day.add_argument("--approval-db-growth-gib", type=float, default=100.0)
    token_day.add_argument("--approval-active-days", type=float, default=7.0)
    token_day.add_argument(
        "--acknowledge-large-projection",
        action="store_true",
        help="Allow online backfill even when the read-only projection exceeds approval thresholds",
    )

    status_report = subparsers.add_parser("status-report", help="Write a corrected global compact quote status report")
    _add_common_path_args(status_report)
    status_report.add_argument("--batch-report", default=None, help="Optional prior batch JSON to summarize with corrected failure classes")

    projection_report = subparsers.add_parser(
        "projection-report",
        help="Write a read-only global token-day completion projection report",
    )
    _add_common_path_args(projection_report)
    projection_report.add_argument("--approval-db-growth-gib", type=float, default=100.0)
    projection_report.add_argument("--approval-active-days", type=float, default=7.0)

    scope_options = subparsers.add_parser(
        "scope-options-report",
        help="Write a read-only scope-options report for narrowing global token-day backfill",
    )
    _add_common_path_args(scope_options)
    scope_options.add_argument("--top-n", type=int, default=25)

    args = parser.parse_args(argv)
    root = Path(args.work_root).resolve()
    source_repo = Path(args.source_repo).expanduser()
    bootstrap(PipelinePaths(root), source_repo)

    compact_db = _resolve(root, Path(args.compact_db))
    decision_db = _resolve(root, Path(args.decision_db))
    report_dir = _resolve(root, Path(args.report_dir))

    if args.command == "inspect":
        report = inspect_global_artifacts(compact_db=compact_db, decision_db=decision_db)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    if args.command == "proof-backfill":
        result = run_global_quote_proof_backfill(
            root=root,
            source_repo=source_repo,
            discovery_csv=_resolve(root, Path(args.discovery_csv)),
            sample_json=_resolve(root, Path(args.sample_json)),
            compact_db=compact_db,
            decision_db=decision_db,
            raw_cache_dir=_resolve(root, Path(args.raw_cache_dir)),
            report_dir=report_dir,
            limit_jobs=args.limit_jobs,
            workers=args.workers,
            progress_every=args.progress_every,
            telonex_env=Path(args.telonex_env).expanduser() if args.telonex_env else None,
            hard_timeout_seconds=args.telonex_hard_timeout_seconds,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.command == "token-day-backfill":
        result = run_global_token_day_backfill(
            root=root,
            source_repo=source_repo,
            discovery_csv=_resolve(root, Path(args.discovery_csv)),
            compact_db=compact_db,
            decision_db=decision_db,
            raw_cache_dir=_resolve(root, Path(args.raw_cache_dir)),
            report_dir=report_dir,
            limit_jobs=args.limit_jobs,
            workers=args.workers,
            progress_every=args.progress_every,
            telonex_env=Path(args.telonex_env).expanduser() if args.telonex_env else None,
            hard_timeout_seconds=args.telonex_hard_timeout_seconds,
            approval_db_growth_gib=args.approval_db_growth_gib,
            approval_active_days=args.approval_active_days,
            acknowledge_large_projection=args.acknowledge_large_projection,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.command == "status-report":
        result = write_global_status_report(
            compact_db=compact_db,
            decision_db=decision_db,
            raw_cache_dir=_resolve(root, Path(args.raw_cache_dir)),
            report_dir=report_dir,
            batch_report=Path(args.batch_report).expanduser() if args.batch_report else None,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.command == "projection-report":
        result = write_global_projection_report(
            compact_db=compact_db,
            decision_db=decision_db,
            discovery_csv=_resolve(root, Path(args.discovery_csv)),
            raw_cache_dir=_resolve(root, Path(args.raw_cache_dir)),
            report_dir=report_dir,
            approval_db_growth_gib=args.approval_db_growth_gib,
            approval_active_days=args.approval_active_days,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.command == "scope-options-report":
        result = write_global_scope_options_report(
            compact_db=compact_db,
            decision_db=decision_db,
            discovery_csv=_resolve(root, Path(args.discovery_csv)),
            raw_cache_dir=_resolve(root, Path(args.raw_cache_dir)),
            report_dir=report_dir,
            top_n=args.top_n,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    raise AssertionError(args.command)


def _add_common_path_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--compact-db", default=str(DEFAULT_COMPACT_DB))
    parser.add_argument("--decision-db", default=str(DEFAULT_DECISION_DB))
    parser.add_argument("--discovery-csv", default=str(DEFAULT_DISCOVERY_CSV))
    parser.add_argument("--sample-json", default=str(DEFAULT_SAMPLE_JSON))
    parser.add_argument("--raw-cache-dir", default=str(DEFAULT_RAW_CACHE))
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def run_global_quote_proof_backfill(
    *,
    root: Path,
    source_repo: Path,
    discovery_csv: Path,
    sample_json: Path,
    compact_db: Path,
    decision_db: Path,
    raw_cache_dir: Path,
    report_dir: Path,
    limit_jobs: int,
    workers: int,
    progress_every: int,
    telonex_env: Optional[Path],
    hard_timeout_seconds: float,
) -> dict[str, Any]:
    if workers != 1:
        raise ValueError("global proof backfill is intentionally limited to --workers 1")
    bootstrap(PipelinePaths(root), source_repo)
    started = time.monotonic()
    before = inspect_global_artifacts(compact_db=compact_db, decision_db=decision_db)
    jobs = select_phase1_available_jobs(discovery_csv, sample_json, limit_jobs=limit_jobs)
    if not jobs:
        raise RuntimeError("No Phase 1 available Telonex sample jobs could be selected")

    compact_db.parent.mkdir(parents=True, exist_ok=True)
    raw_cache_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    from weather_strategy.telonex import TelonexClient

    hard_timeout = hard_timeout_seconds if hard_timeout_seconds > 0 else None
    client = TelonexClient(
        cache_dir=raw_cache_dir,
        env_path=str(telonex_env or ".env"),
        hard_timeout_seconds=hard_timeout,
    )

    with _connect(compact_db) as conn:
        conn.row_factory = sqlite3.Row
        _ensure_global_compact_schema(conn)
        run_id = _insert_pipeline_run(
            conn,
            "global-quote-proof-backfill",
            {
                "discovery_csv": str(discovery_csv),
                "sample_json": str(sample_json),
                "limit_jobs": limit_jobs,
                "workers": workers,
                "raw_cache_dir": str(raw_cache_dir),
            },
        )
        _ensure_compact_tokens(conn, jobs)
        _upsert_global_market_metadata(conn, jobs)
        conn.commit()

    quote_cache_dirs = [raw_cache_dir / "polymarket" / "quotes"]
    complete = 0
    failed = 0
    source_rows_seen = 0
    rows_inserted = 0
    duplicates = 0
    failures: Counter[str] = Counter()
    job_results: list[dict[str, Any]] = []

    with _connect(compact_db) as conn:
        conn.row_factory = sqlite3.Row
        _ensure_global_compact_schema(conn)
        token_refs = _compact_token_refs(conn)
        source_ref_cache: dict[str, int] = {}
        for index, job in enumerate(jobs, start=1):
            token_ref = token_refs[str(job["token_id"])]
            try:
                rows = _load_quote_rows_for_job(
                    job,
                    client=client,
                    cache_only=False,
                    quote_cache_dirs=quote_cache_dirs,
                )
                metrics = _insert_compact_quote_rows(conn, token_ref, job, rows, source_ref_cache)
                _upsert_compact_status(
                    conn,
                    token_ref,
                    job,
                    "complete",
                    rows=metrics["compact_rows_for_job"],
                    source_rows=metrics["source_rows_seen"],
                    duplicates=metrics["duplicate_rows_collapsed"],
                    conflicts=0,
                    error=None,
                )
                complete += 1
                source_rows_seen += metrics["source_rows_seen"]
                rows_inserted += metrics["unique_rows_inserted"]
                duplicates += metrics["duplicate_rows_collapsed"]
                job_results.append(_job_result(job, "complete", None, metrics))
            except Exception as exc:  # noqa: BLE001
                error = f"{type(exc).__name__}: {str(exc)[:500]}"
                failure_class = _global_failure_class(error)
                failures[failure_class] += 1
                _upsert_compact_status(
                    conn,
                    token_ref,
                    job,
                    "failed",
                    rows=0,
                    source_rows=0,
                    duplicates=0,
                    conflicts=0,
                    error=error,
                )
                failed += 1
                job_results.append(_job_result(job, "failed", error, None))
            conn.commit()
            if progress_every > 0 and index % progress_every == 0:
                print(
                    f"global_quote_proof_jobs={index}/{len(jobs)} "
                    f"complete={complete} failed={failed} elapsed_seconds={time.monotonic() - started:.1f}",
                    file=sys.stderr,
                    flush=True,
                )
        _finish_pipeline_run(conn, run_id, row_count=rows_inserted)
        conn.commit()

    after = inspect_global_artifacts(compact_db=compact_db, decision_db=decision_db)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "tiny compact-only global Telonex quote proof; not a dataset freeze",
        "root_cause_diagnosis": _root_cause_diagnosis(before),
        "paths": {
            "compact_db": str(compact_db),
            "decision_db": str(decision_db),
            "raw_cache_dir": str(raw_cache_dir),
            "discovery_csv": str(discovery_csv),
            "sample_json": str(sample_json),
        },
        "before": before,
        "after": after,
        "jobs_attempted": len(jobs),
        "jobs_completed": complete,
        "jobs_failed": failed,
        "quote_rows_written": rows_inserted,
        "source_rows_seen": source_rows_seen,
        "duplicate_rows_collapsed": duplicates,
        "failure_classes": dict(sorted(failures.items())),
        "job_results": job_results,
        "db_growth_bytes": int(after.get("compact_db_bytes", 0) or 0) - int(before.get("compact_db_bytes", 0) or 0),
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "safe_for_larger_staged_run": complete > 0 and rows_inserted > 0 and not _operational_failure_present(failures),
    }
    if report["safe_for_larger_staged_run"]:
        report["recommended_next_batch"] = {
            "limit_jobs": 100,
            "workers": 1,
            "job_granularity": "token_day",
            "reason": "Proof writes real rows on exact available token-days; use day-level jobs or partial-day tolerant status before any larger staged run.",
        }
    else:
        report["recommended_next_batch"] = {
            "limit_jobs": min(5, max(1, limit_jobs)),
            "workers": 1,
            "reason": "Do not run a larger staged batch until a tiny proof writes quote rows.",
        }
    json_path, md_path = _write_global_quote_report(report_dir, report)
    report["report_paths"] = {"json": str(json_path), "markdown": str(md_path)}
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(_global_quote_report_markdown(report), encoding="utf-8")
    return report


def run_global_token_day_backfill(
    *,
    root: Path,
    source_repo: Path,
    discovery_csv: Path,
    compact_db: Path,
    decision_db: Path,
    raw_cache_dir: Path,
    report_dir: Path,
    limit_jobs: int,
    workers: int,
    progress_every: int,
    telonex_env: Optional[Path],
    hard_timeout_seconds: float,
    approval_db_growth_gib: float,
    approval_active_days: float,
    acknowledge_large_projection: bool,
) -> dict[str, Any]:
    if workers != 1:
        raise ValueError("first staged global token-day backfill is intentionally limited to --workers 1")
    bootstrap(PipelinePaths(root), source_repo)
    started = time.monotonic()
    projection = write_global_projection_report(
        compact_db=compact_db,
        decision_db=decision_db,
        discovery_csv=discovery_csv,
        raw_cache_dir=raw_cache_dir,
        report_dir=report_dir,
        approval_db_growth_gib=approval_db_growth_gib,
        approval_active_days=approval_active_days,
    )
    _raise_if_projection_requires_acknowledgement(projection, acknowledge_large_projection)
    before = inspect_global_artifacts(compact_db=compact_db, decision_db=decision_db)
    jobs = select_token_day_jobs(discovery_csv, compact_db=compact_db, limit_jobs=limit_jobs)
    if not jobs:
        raise RuntimeError("No unattempted global token-day jobs could be selected")

    compact_db.parent.mkdir(parents=True, exist_ok=True)
    raw_cache_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    from weather_strategy.telonex import TelonexClient

    hard_timeout = hard_timeout_seconds if hard_timeout_seconds > 0 else None
    client = TelonexClient(
        cache_dir=raw_cache_dir,
        env_path=str(telonex_env or ".env"),
        hard_timeout_seconds=hard_timeout,
    )

    with _connect(compact_db) as conn:
        conn.row_factory = sqlite3.Row
        _ensure_global_compact_schema(conn)
        run_id = _insert_pipeline_run(
            conn,
            "global-token-day-backfill",
            {
                "discovery_csv": str(discovery_csv),
                "limit_jobs": limit_jobs,
                "workers": workers,
                "raw_cache_dir": str(raw_cache_dir),
                "job_granularity": "token_day",
                "status_selection": "unattempted_only",
            },
        )
        _ensure_compact_tokens(conn, jobs)
        _upsert_global_market_metadata(conn, jobs)
        conn.commit()

    quote_cache_dirs = [raw_cache_dir / "polymarket" / "quotes"]
    complete = 0
    failed = 0
    source_rows_seen = 0
    rows_inserted = 0
    duplicates = 0
    failures: Counter[str] = Counter()
    job_results: list[dict[str, Any]] = []

    with _connect(compact_db) as conn:
        conn.row_factory = sqlite3.Row
        _ensure_global_compact_schema(conn)
        token_refs = _compact_token_refs(conn)
        source_ref_cache: dict[str, int] = {}
        for index, job in enumerate(jobs, start=1):
            token_ref = token_refs[str(job["token_id"])]
            try:
                rows = _load_quote_rows_for_job(
                    job,
                    client=client,
                    cache_only=False,
                    quote_cache_dirs=quote_cache_dirs,
                )
                metrics = _insert_compact_quote_rows(conn, token_ref, job, rows, source_ref_cache)
                _upsert_compact_status(
                    conn,
                    token_ref,
                    job,
                    "complete",
                    rows=metrics["compact_rows_for_job"],
                    source_rows=metrics["source_rows_seen"],
                    duplicates=metrics["duplicate_rows_collapsed"],
                    conflicts=0,
                    error=None,
                )
                complete += 1
                source_rows_seen += metrics["source_rows_seen"]
                rows_inserted += metrics["unique_rows_inserted"]
                duplicates += metrics["duplicate_rows_collapsed"]
                job_results.append(_job_result(job, "complete", None, metrics))
            except Exception as exc:  # noqa: BLE001
                error = f"{type(exc).__name__}: {str(exc)[:500]}"
                failure_class = _global_failure_class(error)
                failures[failure_class] += 1
                _upsert_compact_status(
                    conn,
                    token_ref,
                    job,
                    "failed",
                    rows=0,
                    source_rows=0,
                    duplicates=0,
                    conflicts=0,
                    error=error,
                )
                failed += 1
                job_results.append(_job_result(job, "failed", error, None))
            conn.commit()
            if progress_every > 0 and index % progress_every == 0:
                print(
                    f"global_token_day_jobs={index}/{len(jobs)} "
                    f"complete={complete} failed={failed} elapsed_seconds={time.monotonic() - started:.1f}",
                    file=sys.stderr,
                    flush=True,
                )
        _finish_pipeline_run(conn, run_id, row_count=rows_inserted)
        conn.commit()

    after = inspect_global_artifacts(compact_db=compact_db, decision_db=decision_db)
    stable = not _operational_failure_present(failures)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "staged compact-only global Telonex token-day quote backfill; not a dataset freeze",
        "root_cause_diagnosis": (
            "Token-day backfill avoids the prior full-window failure mode by recording each token/day independently; "
            "one unavailable Telonex day affects only that token-day status row."
        ),
        "paths": {
            "compact_db": str(compact_db),
            "decision_db": str(decision_db),
            "raw_cache_dir": str(raw_cache_dir),
            "discovery_csv": str(discovery_csv),
        },
        "before": before,
        "after": after,
        "jobs_attempted": len(jobs),
        "jobs_completed": complete,
        "jobs_failed": failed,
        "quote_rows_written": rows_inserted,
        "source_rows_seen": source_rows_seen,
        "duplicate_rows_collapsed": duplicates,
        "failure_classes": dict(sorted(failures.items())),
        "job_results": job_results[:25],
        "job_results_truncated": max(0, len(job_results) - 25),
        "db_growth_bytes": int(after.get("compact_db_bytes", 0) or 0) - int(before.get("compact_db_bytes", 0) or 0),
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "workers_stable": stable,
        "safe_for_larger_staged_run": complete > 0 and rows_inserted > 0 and stable,
    }
    report["recommended_next_batch"] = (
        {
            "limit_jobs": 500,
            "workers": 1,
            "job_granularity": "token_day",
            "reason": "workers=1 was operationally stable; scale batch size before increasing concurrency.",
        }
        if report["safe_for_larger_staged_run"]
        else {
            "limit_jobs": 100,
            "workers": 1,
            "job_granularity": "token_day",
            "reason": "Do not increase size or workers until operational failures are cleared.",
        }
    )
    json_path, md_path = _write_global_quote_report(report_dir, report, stem="global-weather-token-day-batch")
    report["report_paths"] = {"json": str(json_path), "markdown": str(md_path)}
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(_global_quote_report_markdown(report), encoding="utf-8")
    return report


def inspect_global_artifacts(*, compact_db: Path, decision_db: Path) -> dict[str, Any]:
    report: dict[str, Any] = {
        "compact_db": str(compact_db),
        "compact_db_exists": compact_db.exists(),
        "compact_db_bytes": compact_db.stat().st_size if compact_db.exists() else 0,
        "decision_db": str(decision_db),
        "decision_db_exists": decision_db.exists(),
        "decision_db_bytes": decision_db.stat().st_size if decision_db.exists() else 0,
    }
    if compact_db.exists():
        with sqlite3.connect(compact_db) as conn:
            conn.row_factory = sqlite3.Row
            tables = _sqlite_tables(conn)
            report["compact_tables"] = sorted(tables)
            if "compact_price_fetch_status" in tables:
                report["quote_status_counts"] = [
                    dict(row)
                    for row in conn.execute(
                        """
                        SELECT status, COUNT(*) AS jobs,
                               COALESCE(SUM(rows_written), 0) AS rows_written,
                               COALESCE(SUM(source_rows_seen), 0) AS source_rows_seen
                        FROM compact_price_fetch_status
                        GROUP BY status
                        ORDER BY status
                        """
                    )
                ]
                failures = Counter()
                for row in conn.execute("SELECT error FROM compact_price_fetch_status WHERE status = 'failed'"):
                    failures[_global_failure_class(row["error"])] += 1
                report["failed_breakdown"] = dict(sorted(failures.items()))
            else:
                report["quote_status_counts"] = []
                report["failed_breakdown"] = {}
            report["compact_quote_history_rows"] = _count_table(conn, "compact_quote_history", tables)
            report["compact_sources_count"] = _count_table(conn, "compact_sources", tables)
            report["compact_tokens_count"] = _count_table(conn, "compact_tokens", tables)
            report["latest_batch_row"] = _latest_pipeline_run(conn, tables)
    else:
        report.update(
            {
                "quote_status_counts": [],
                "failed_breakdown": {},
                "compact_quote_history_rows": 0,
                "compact_sources_count": 0,
                "compact_tokens_count": 0,
                "latest_batch_row": None,
            }
        )
    if decision_db.exists():
        with sqlite3.connect(decision_db) as conn:
            tables = _sqlite_tables(conn)
            report["decision_tables"] = sorted(tables)
            if "global_decision_snapshots" in tables:
                report["decision_artifact_rows"] = _count_table(conn, "global_decision_snapshots", tables)
            elif "decision_snapshots_current" in tables:
                report["decision_artifact_rows"] = _count_table(conn, "decision_snapshots_current", tables)
            elif "decision_snapshots" in tables:
                report["decision_artifact_rows"] = _count_table(conn, "decision_snapshots", tables)
            else:
                report["decision_artifact_rows"] = 0
    else:
        report["decision_artifact_rows"] = 0
    return report


def write_global_status_report(
    *,
    compact_db: Path,
    decision_db: Path,
    raw_cache_dir: Path,
    report_dir: Path,
    batch_report: Optional[Path],
) -> dict[str, Any]:
    report_dir.mkdir(parents=True, exist_ok=True)
    status = inspect_global_artifacts(compact_db=compact_db, decision_db=decision_db)
    report: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "corrected global compact quote status report",
        "after": status,
        "paths": {
            "compact_db": str(compact_db),
            "decision_db": str(decision_db),
            "raw_cache_dir": str(raw_cache_dir),
        },
    }
    if batch_report:
        prior = json.loads(batch_report.read_text(encoding="utf-8"))
        before_failed = Counter(prior.get("before", {}).get("failed_breakdown") or {})
        after_failed = Counter(status.get("failed_breakdown") or {})
        batch_failed = Counter()
        for key, value in after_failed.items():
            diff = int(value) - int(before_failed.get(key, 0))
            if diff > 0:
                batch_failed[key] = diff
        stable = not _operational_failure_present(batch_failed)
        report.update(
            {
                "batch_report_corrected_from": str(batch_report),
                "jobs_attempted": prior.get("jobs_attempted"),
                "jobs_completed": prior.get("jobs_completed"),
                "jobs_failed": prior.get("jobs_failed"),
                "quote_rows_written": prior.get("quote_rows_written"),
                "source_rows_seen": prior.get("source_rows_seen"),
                "db_growth_bytes": prior.get("db_growth_bytes"),
                "failure_classes": dict(sorted(batch_failed.items())),
                "workers_stable": stable,
                "safe_for_larger_staged_run": bool(prior.get("jobs_completed", 0)) and stable,
                "recommended_next_batch": (
                    {
                        "limit_jobs": 500,
                        "workers": 1,
                        "job_granularity": "token_day",
                        "reason": "workers=1 was operationally stable; scale batch size before increasing concurrency.",
                    }
                    if stable
                    else {
                        "limit_jobs": 100,
                        "workers": 1,
                        "job_granularity": "token_day",
                        "reason": "Do not increase size or workers until operational failures are cleared.",
                    }
                ),
            }
        )
    json_path, md_path = _write_global_quote_report(report_dir, report, stem="global-weather-status-report")
    report["report_paths"] = {"json": str(json_path), "markdown": str(md_path)}
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(_global_quote_report_markdown(report), encoding="utf-8")
    return report


def write_global_projection_report(
    *,
    compact_db: Path,
    decision_db: Path,
    discovery_csv: Path,
    raw_cache_dir: Path,
    report_dir: Path,
    approval_db_growth_gib: float,
    approval_active_days: float,
) -> dict[str, Any]:
    report_dir.mkdir(parents=True, exist_ok=True)
    status = inspect_global_artifacts(compact_db=compact_db, decision_db=decision_db)
    batch_summary = _token_day_batch_summary(report_dir)
    status_rows = _status_key_rows(compact_db)
    scope_summary = _summarize_discovery_scopes(discovery_csv, status_rows, batch_summary)["overall"]
    existing_jobs = len(status_rows)
    selector_status_jobs = int(scope_summary["token_day_jobs"]) - int(scope_summary["unattempted_jobs"])
    external_status_jobs = max(0, existing_jobs - selector_status_jobs)
    remaining_jobs = int(scope_summary["unattempted_jobs"])
    total_selector_jobs = int(scope_summary["token_day_jobs"])
    active_jobs_per_hour = float(batch_summary.get("active_jobs_per_hour") or 0.0)
    bytes_per_job = float(batch_summary.get("db_growth_bytes_per_attempted_job") or 0.0)
    rows_per_job = float(batch_summary.get("quote_rows_per_attempted_job") or 0.0)
    raw_cache_bytes = _directory_size(raw_cache_dir)
    raw_cache_bytes_per_status_job = raw_cache_bytes / existing_jobs if existing_jobs > 0 else 0.0
    projected_active_hours = remaining_jobs / active_jobs_per_hour if active_jobs_per_hour > 0 else None
    projected_active_days = projected_active_hours / 24.0 if projected_active_hours is not None else None
    projected_db_growth_bytes = int(round(remaining_jobs * bytes_per_job)) if bytes_per_job > 0 else None
    projected_raw_cache_growth_bytes = (
        int(round(remaining_jobs * raw_cache_bytes_per_status_job)) if raw_cache_bytes_per_status_job > 0 else None
    )
    projected_quote_rows = int(round(remaining_jobs * rows_per_job)) if rows_per_job > 0 else None
    projected_db_growth_gib = _bytes_to_gib(projected_db_growth_bytes)
    projected_raw_cache_growth_gib = _bytes_to_gib(projected_raw_cache_growth_bytes)
    approval_required = bool(
        (projected_db_growth_gib is not None and projected_db_growth_gib >= approval_db_growth_gib)
        or (projected_active_days is not None and projected_active_days >= approval_active_days)
    )
    report: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "read-only global token-day completion projection; not a dataset freeze",
        "paths": {
            "compact_db": str(compact_db),
            "decision_db": str(decision_db),
            "raw_cache_dir": str(raw_cache_dir),
            "discovery_csv": str(discovery_csv),
            "report_dir": str(report_dir),
        },
        "current_status": status,
        "selector_counts": {
            "existing_status_jobs": existing_jobs,
            "existing_selector_status_jobs": selector_status_jobs,
            "external_status_jobs": external_status_jobs,
            "remaining_unattempted_jobs": remaining_jobs,
            "total_selector_jobs": total_selector_jobs,
        },
        "selector_scope_summary": scope_summary,
        "observed_batch_rates": batch_summary,
        "storage_now": {
            "compact_db_bytes": status.get("compact_db_bytes", 0),
            "raw_cache_bytes": raw_cache_bytes,
        },
        "projection": {
            "projected_active_hours_remaining": round(projected_active_hours, 1) if projected_active_hours is not None else None,
            "projected_active_days_remaining": round(projected_active_days, 1) if projected_active_days is not None else None,
            "projected_db_growth_bytes": projected_db_growth_bytes,
            "projected_db_growth_gib": round(projected_db_growth_gib, 1) if projected_db_growth_gib is not None else None,
            "projected_raw_cache_growth_bytes": projected_raw_cache_growth_bytes,
            "projected_raw_cache_growth_gib": round(projected_raw_cache_growth_gib, 1)
            if projected_raw_cache_growth_gib is not None
            else None,
            "projected_quote_rows": projected_quote_rows,
        },
        "approval_thresholds": {
            "approval_db_growth_gib": approval_db_growth_gib,
            "approval_active_days": approval_active_days,
        },
        "approval_required_before_more_online_backfill": approval_required,
        "safe_to_continue_without_checkpoint": not approval_required,
        "recommended_next_action": (
            "Stop online token-day backfill until manager/user approves the projected runtime/storage, "
            "narrows the global scope, or approves implementation of grouped source-file ingestion."
            if approval_required
            else "Continue staged workers=1 token-day backfill with report checkpoints."
        ),
    }
    json_path, md_path = _write_global_quote_report(report_dir, report, stem="global-weather-projection-report")
    report["report_paths"] = {"json": str(json_path), "markdown": str(md_path)}
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(_global_projection_report_markdown(report), encoding="utf-8")
    return report


def write_global_scope_options_report(
    *,
    compact_db: Path,
    decision_db: Path,
    discovery_csv: Path,
    raw_cache_dir: Path,
    report_dir: Path,
    top_n: int,
) -> dict[str, Any]:
    report_dir.mkdir(parents=True, exist_ok=True)
    status = inspect_global_artifacts(compact_db=compact_db, decision_db=decision_db)
    status_rows = _status_key_rows(compact_db)
    batch_summary = _token_day_batch_summary(report_dir)
    scopes = _summarize_discovery_scopes(discovery_csv, status_rows, batch_summary)
    top_n = max(1, top_n)
    report: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "read-only global quote scope-options report; not a dataset freeze",
        "paths": {
            "compact_db": str(compact_db),
            "decision_db": str(decision_db),
            "raw_cache_dir": str(raw_cache_dir),
            "discovery_csv": str(discovery_csv),
            "report_dir": str(report_dir),
        },
        "current_status": status,
        "observed_batch_rates": batch_summary,
        "overall": scopes["overall"],
        "scope_tables": {
            "by_temperature_metric": scopes["by_temperature_metric"],
            "top_cities_by_remaining_jobs": _top_scope_rows(scopes["by_city"], "unattempted_jobs", top_n),
            "top_cities_by_completed_jobs": _top_scope_rows(scopes["by_city"], "complete_jobs", top_n),
            "top_city_metrics_by_remaining_jobs": _top_scope_rows(scopes["by_city_metric"], "unattempted_jobs", top_n),
            "top_city_metrics_by_completed_jobs": _top_scope_rows(scopes["by_city_metric"], "complete_jobs", top_n),
        },
        "table_counts": {
            "city_count": len(scopes["by_city"]),
            "city_metric_count": len(scopes["by_city_metric"]),
        },
        "recommended_next_action": (
            "Use this report to choose an explicit city/metric/date subset before more online backfill, "
            "or approve grouped source-file ingestion. Do not use full selector token-day backfill without "
            "acknowledging the projection guard."
        ),
    }
    json_path, md_path = _write_global_quote_report(report_dir, report, stem="global-weather-scope-options")
    report["report_paths"] = {"json": str(json_path), "markdown": str(md_path)}
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(_global_scope_options_report_markdown(report), encoding="utf-8")
    return report


def select_phase1_available_jobs(discovery_csv: Path, sample_json: Path, *, limit_jobs: int) -> list[dict[str, Any]]:
    available = json.loads(sample_json.read_text(encoding="utf-8")).get("results", [])
    anchors = [row for row in available if row.get("status") == "available"]
    if not anchors:
        return []
    with discovery_csv.open("r", encoding="utf-8", newline="") as handle:
        discovery_rows = list(csv.DictReader(handle))
    selected: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int]] = set()
    for anchor in anchors:
        match = _find_discovery_row(discovery_rows, anchor)
        if not match:
            continue
        anchor_day = _parse_date(anchor.get("day"))
        for side, token_key in (("YES", "yes_token_id"), ("NO", "no_token_id")):
            token_id = (match.get(token_key) or "").strip()
            if not token_id:
                continue
            job = _job_from_discovery_row(match, side=side, token_id=token_id, start_day=anchor_day, end_day=anchor_day)
            key = (job["token_id"], job["start_ts"], job["end_ts"])
            if key in seen:
                continue
            selected.append(job)
            seen.add(key)
            if limit_jobs > 0 and len(selected) >= limit_jobs:
                return selected
    return selected


def select_token_day_jobs(discovery_csv: Path, *, compact_db: Path, limit_jobs: int) -> list[dict[str, Any]]:
    existing = _existing_status_keys(compact_db)
    selected: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int]] = set()
    with discovery_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("has_telonex_quote_window") or "").strip().lower() not in {"true", "1", "yes"}:
                continue
            quote_from = _parse_date(row.get("quotes_from"))
            quote_to = _parse_date(row.get("quotes_to"))
            if quote_from is None or quote_to is None or quote_to < quote_from:
                continue
            for day in _date_days(quote_from, quote_to):
                for side, token_key in (("YES", "yes_token_id"), ("NO", "no_token_id")):
                    token_id = (row.get(token_key) or "").strip()
                    if not token_id:
                        continue
                    job = _job_from_discovery_row(row, side=side, token_id=token_id, start_day=day, end_day=day)
                    key = (job["token_id"], job["start_ts"], job["end_ts"])
                    if key in existing or key in seen:
                        continue
                    selected.append(job)
                    seen.add(key)
                    if limit_jobs > 0 and len(selected) >= limit_jobs:
                        return selected
    return selected


def _existing_status_keys(compact_db: Path) -> set[tuple[str, int, int]]:
    if not compact_db.exists():
        return set()
    with sqlite3.connect(compact_db) as conn:
        conn.row_factory = sqlite3.Row
        tables = _sqlite_tables(conn)
        if "compact_price_fetch_status" not in tables or "compact_tokens" not in tables:
            return set()
        return {
            (str(row["token_id"]), int(row["start_ts"]), int(row["end_ts"]))
            for row in conn.execute(
                """
                SELECT t.token_id, s.start_ts, s.end_ts
                FROM compact_price_fetch_status s
                JOIN compact_tokens t ON t.token_ref = s.token_ref
                """
            )
        }


def _status_key_rows(compact_db: Path) -> dict[tuple[str, int, int], dict[str, Any]]:
    if not compact_db.exists():
        return {}
    with sqlite3.connect(compact_db) as conn:
        conn.row_factory = sqlite3.Row
        tables = _sqlite_tables(conn)
        if "compact_price_fetch_status" not in tables or "compact_tokens" not in tables:
            return {}
        return {
            (str(row["token_id"]), int(row["start_ts"]), int(row["end_ts"])): {
                "status": row["status"],
                "rows_written": int(row["rows_written"] or 0),
                "source_rows_seen": int(row["source_rows_seen"] or 0),
                "error": row["error"],
            }
            for row in conn.execute(
                """
                SELECT t.token_id, s.start_ts, s.end_ts, s.status, s.rows_written, s.source_rows_seen, s.error
                FROM compact_price_fetch_status s
                JOIN compact_tokens t ON t.token_ref = s.token_ref
                """
            )
        }


def _summarize_discovery_scopes(
    discovery_csv: Path,
    status_rows: Mapping[tuple[str, int, int], Mapping[str, Any]],
    batch_summary: Mapping[str, Any],
) -> dict[str, Any]:
    raw_scopes: dict[tuple[str, str], dict[str, Any]] = {}

    def scope_row(scope_type: str, scope_name: str) -> dict[str, Any]:
        key = (scope_type, scope_name or "unknown")
        if key not in raw_scopes:
            raw_scopes[key] = {
                "scope_type": key[0],
                "scope": key[1],
                "market_rows": 0,
                "token_day_jobs": 0,
                "complete_jobs": 0,
                "failed_jobs": 0,
                "unattempted_jobs": 0,
                "quote_rows_written": 0,
                "source_rows_seen": 0,
            }
        return raw_scopes[key]

    with discovery_csv.open("r", encoding="utf-8", newline="") as handle:
        for discovery_row in csv.DictReader(handle):
            if str(discovery_row.get("has_telonex_quote_window") or "").strip().lower() not in {"true", "1", "yes"}:
                continue
            quote_from = _parse_date(discovery_row.get("quotes_from"))
            quote_to = _parse_date(discovery_row.get("quotes_to"))
            if quote_from is None or quote_to is None or quote_to < quote_from:
                continue
            city = (discovery_row.get("city") or "unknown").strip() or "unknown"
            metric = (discovery_row.get("temperature_metric") or "unknown").strip() or "unknown"
            scope_keys = [
                ("overall", "all"),
                ("temperature_metric", metric),
                ("city", city),
                ("city_metric", f"{city}|{metric}"),
            ]
            for scope_type, scope_name in scope_keys:
                scope_row(scope_type, scope_name)["market_rows"] += 1
            for day in _date_days(quote_from, quote_to):
                for side, token_key in (("YES", "yes_token_id"), ("NO", "no_token_id")):
                    token_id = (discovery_row.get(token_key) or "").strip()
                    if not token_id:
                        continue
                    job = _job_from_discovery_row(
                        discovery_row,
                        side=side,
                        token_id=token_id,
                        start_day=day,
                        end_day=day,
                    )
                    key = (job["token_id"], job["start_ts"], job["end_ts"])
                    status = status_rows.get(key)
                    for scope_type, scope_name in scope_keys:
                        _add_scope_job(scope_row(scope_type, scope_name), status)

    summaries = [_scope_summary(row, batch_summary) for row in raw_scopes.values()]
    return {
        "overall": next(row for row in summaries if row["scope_type"] == "overall" and row["scope"] == "all"),
        "by_temperature_metric": sorted(
            [row for row in summaries if row["scope_type"] == "temperature_metric"],
            key=lambda row: row["scope"],
        ),
        "by_city": [row for row in summaries if row["scope_type"] == "city"],
        "by_city_metric": [row for row in summaries if row["scope_type"] == "city_metric"],
    }


def _add_scope_job(scope: dict[str, Any], status: Optional[Mapping[str, Any]]) -> None:
    scope["token_day_jobs"] += 1
    if status is None:
        scope["unattempted_jobs"] += 1
        return
    if status.get("status") == "complete":
        scope["complete_jobs"] += 1
        scope["quote_rows_written"] += int(status.get("rows_written") or 0)
        scope["source_rows_seen"] += int(status.get("source_rows_seen") or 0)
    elif status.get("status") == "failed":
        scope["failed_jobs"] += 1
    else:
        scope["unattempted_jobs"] += 1


def _scope_summary(scope: Mapping[str, Any], batch_summary: Mapping[str, Any]) -> dict[str, Any]:
    total_jobs = int(scope.get("token_day_jobs") or 0)
    complete_jobs = int(scope.get("complete_jobs") or 0)
    unattempted_jobs = int(scope.get("unattempted_jobs") or 0)
    active_jobs_per_hour = float(batch_summary.get("active_jobs_per_hour") or 0.0)
    bytes_per_job = float(batch_summary.get("db_growth_bytes_per_attempted_job") or 0.0)
    rows_per_job = float(batch_summary.get("quote_rows_per_attempted_job") or 0.0)
    projected_hours = unattempted_jobs / active_jobs_per_hour if active_jobs_per_hour > 0 else None
    projected_db_growth_bytes = int(round(unattempted_jobs * bytes_per_job)) if bytes_per_job > 0 else None
    projected_quote_rows = int(round(unattempted_jobs * rows_per_job)) if rows_per_job > 0 else None
    result = dict(scope)
    result.update(
        {
            "complete_job_fraction": round(complete_jobs / total_jobs, 6) if total_jobs > 0 else None,
            "projected_active_hours_remaining": round(projected_hours, 1) if projected_hours is not None else None,
            "projected_active_days_remaining": round(projected_hours / 24.0, 1) if projected_hours is not None else None,
            "projected_db_growth_gib": round(_bytes_to_gib(projected_db_growth_bytes) or 0.0, 3)
            if projected_db_growth_bytes is not None
            else None,
            "projected_quote_rows": projected_quote_rows,
        }
    )
    return result


def _top_scope_rows(rows: list[dict[str, Any]], field: str, top_n: int) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: (int(row.get(field) or 0), int(row.get("token_day_jobs") or 0)), reverse=True)[:top_n]


def _date_days(start: date, end: date) -> Iterable[date]:
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def _find_discovery_row(discovery_rows: Iterable[Mapping[str, str]], anchor: Mapping[str, Any]) -> Optional[Mapping[str, str]]:
    yes_token = str(anchor.get("yes_token_id") or "")
    question = str(anchor.get("question") or "")
    for row in discovery_rows:
        if yes_token and str(row.get("yes_token_id") or "") == yes_token:
            return row
    for row in discovery_rows:
        if question and str(row.get("question") or "") == question:
            return row
    return None


def _job_from_discovery_row(
    row: Mapping[str, str],
    *,
    side: str,
    token_id: str,
    start_day: Optional[date] = None,
    end_day: Optional[date] = None,
) -> dict[str, Any]:
    start_day = start_day or _parse_date(row.get("quotes_from")) or _parse_date(row.get("created_at_utc")) or _parse_date(row.get("market_date_local"))
    end_day = end_day or _parse_date(row.get("quotes_to")) or _parse_date(row.get("end_date_utc")) or start_day
    if start_day is None or end_day is None:
        raise ValueError(f"Discovery row has no usable quote window for token {token_id[:8]}")
    start_ts = int(datetime.combine(start_day, datetime_time(0, 0), tzinfo=timezone.utc).timestamp())
    end_ts = int(datetime.combine(end_day + timedelta(days=1), datetime_time(0, 0), tzinfo=timezone.utc).timestamp()) - 1
    source_row = row.get("source_row_index") or row.get("market_id") or token_id
    universe_row_id = f"global:{source_row}"
    return {
        "token_id": str(token_id),
        "market_id": row.get("market_id"),
        "universe_row_id": universe_row_id,
        "side": side,
        "slug": row.get("slug"),
        "outcome": "Yes" if side == "YES" else "No",
        "start_ts": start_ts,
        "end_ts": end_ts,
        "source_row_index": row.get("source_row_index"),
        "city": row.get("city"),
        "temperature_metric": row.get("temperature_metric"),
        "question": row.get("question"),
        "event_title": row.get("event_title"),
        "event_slug": row.get("event_slug"),
        "market_date_local": row.get("market_date_local"),
        "bucket_type": row.get("bucket_type"),
        "operator": row.get("operator"),
    }


def _parse_date(value: Any) -> Optional[date]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def _ensure_global_compact_schema(conn: sqlite3.Connection) -> None:
    _ensure_compact_price_schema(conn)
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS global_market_universe (
            universe_row_id TEXT PRIMARY KEY,
            market_id TEXT,
            slug TEXT,
            event_slug TEXT,
            event_title TEXT,
            question TEXT,
            city TEXT,
            temperature_metric TEXT,
            market_date_local TEXT,
            bucket_type TEXT,
            operator TEXT,
            source_row_index TEXT,
            updated_at_utc TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_global_market_city_date
            ON global_market_universe(city, market_date_local);
        """
    )


def _upsert_global_market_metadata(conn: sqlite3.Connection, jobs: Iterable[Mapping[str, Any]]) -> None:
    now = datetime.now(timezone.utc).isoformat()
    for job in jobs:
        conn.execute(
            """
            INSERT OR REPLACE INTO global_market_universe(
                universe_row_id, market_id, slug, event_slug, event_title, question,
                city, temperature_metric, market_date_local, bucket_type, operator,
                source_row_index, updated_at_utc
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job.get("universe_row_id"),
                job.get("market_id"),
                job.get("slug"),
                job.get("event_slug"),
                job.get("event_title"),
                job.get("question"),
                job.get("city"),
                job.get("temperature_metric"),
                job.get("market_date_local"),
                job.get("bucket_type"),
                job.get("operator"),
                job.get("source_row_index"),
                now,
            ),
        )


def _sqlite_tables(conn: sqlite3.Connection) -> set[str]:
    return {
        str(row[0])
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type IN ('table', 'view')")
    }


def _count_table(conn: sqlite3.Connection, table: str, tables: set[str]) -> int:
    if table not in tables:
        return 0
    return int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


def _latest_pipeline_run(conn: sqlite3.Connection, tables: set[str]) -> Optional[dict[str, Any]]:
    if "pipeline_runs" not in tables:
        return None
    row = conn.execute(
        """
        SELECT run_id, command, started_at_utc, finished_at_utc, row_count, settings_json
        FROM pipeline_runs
        ORDER BY started_at_utc DESC
        LIMIT 1
        """
    ).fetchone()
    return dict(row) if row else None


def _global_failure_class(error: Optional[str]) -> str:
    base = _compact_failure_class(error)
    if base == "transient":
        return "transient_network"
    if base == "local_cache_missing":
        return "cached_missing"
    if base != "unknown":
        return base
    text = (error or "").lower()
    if (
        "authorization" in text
        or "unauthorized" in text
        or "forbidden" in text
        or "http 401" in text
        or "http error 401" in text
        or "http 403" in text
        or "http error 403" in text
    ):
        return "auth"
    if "database is locked" in text or "sqlite" in text and "locked" in text:
        return "sqlite_lock"
    if base == "unknown" and ("http 5" in text or "500" in text or "502" in text or "503" in text or "504" in text):
        return "api_server"
    return base


def _operational_failure_present(failures: Mapping[str, int]) -> bool:
    operational = {"transient_network", "rate_limit", "auth", "sqlite_lock", "api_server", "parser_or_token_mismatch", "unknown"}
    return any(int(failures.get(key, 0) or 0) > 0 for key in operational)


def _raise_if_projection_requires_acknowledgement(report: Mapping[str, Any], acknowledged: bool) -> None:
    if acknowledged or not report.get("approval_required_before_more_online_backfill"):
        return
    projection = report.get("projection") or {}
    report_paths = report.get("report_paths") or {}
    raise RuntimeError(
        "Global token-day backfill projection exceeds approval thresholds; refusing online fetch before Telonex access. "
        f"Projection report: {report_paths.get('json')}. "
        f"Projected active days: {projection.get('projected_active_days_remaining')}; "
        f"projected DB growth GiB: {projection.get('projected_db_growth_gib')}. "
        "Pass --acknowledge-large-projection only after manager/user approval, or narrow the global scope."
    )


def _token_day_batch_summary(report_dir: Path) -> dict[str, Any]:
    reports: list[dict[str, Any]] = []
    for path in sorted(report_dir.glob("*-global-weather-token-day-batch.json")):
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        attempted = int(report.get("jobs_attempted") or 0)
        elapsed = float(report.get("elapsed_seconds") or 0.0)
        if attempted <= 0 or elapsed <= 0:
            continue
        reports.append(report)
    total_jobs = sum(int(report.get("jobs_attempted") or 0) for report in reports)
    total_completed = sum(int(report.get("jobs_completed") or 0) for report in reports)
    total_failed = sum(int(report.get("jobs_failed") or 0) for report in reports)
    total_rows = sum(int(report.get("quote_rows_written") or 0) for report in reports)
    total_growth = sum(int(report.get("db_growth_bytes") or 0) for report in reports)
    total_elapsed = sum(float(report.get("elapsed_seconds") or 0.0) for report in reports)
    return {
        "batch_report_count": len(reports),
        "jobs_attempted": total_jobs,
        "jobs_completed": total_completed,
        "jobs_failed": total_failed,
        "quote_rows_written": total_rows,
        "db_growth_bytes": total_growth,
        "elapsed_seconds": round(total_elapsed, 2),
        "active_jobs_per_hour": round(total_jobs / total_elapsed * 3600.0, 2) if total_elapsed > 0 else None,
        "quote_rows_per_attempted_job": round(total_rows / total_jobs, 2) if total_jobs > 0 else None,
        "db_growth_bytes_per_attempted_job": round(total_growth / total_jobs, 2) if total_jobs > 0 else None,
    }


def _directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(child.stat().st_size for child in path.rglob("*") if child.is_file())


def _bytes_to_gib(value: Optional[int]) -> Optional[float]:
    if value is None:
        return None
    return value / float(1024**3)


def _root_cause_diagnosis(before: Mapping[str, Any]) -> str:
    if not before.get("compact_db_exists"):
        return (
            "The requested newer global SQLite artifact is absent in this workspace, so the reported "
            "40,000-job/500-failure state cannot be verified here. The available source evidence points "
            "to Phase 1 only: Telonex URL construction and asset_id usage worked for historical Karachi "
            "sample jobs, while current-day July 7 samples were source-unavailable 404s."
        )
    rows = int(before.get("compact_quote_history_rows", 0) or 0)
    status_counts = before.get("quote_status_counts") or []
    if rows == 0 and status_counts:
        return (
            "Existing global compact DB has status rows but no quote rows. Treat prior failure as unproven "
            "operational failure until a tiny workers=1 proof writes rows; do not promote transient/rate-limit "
            "failures to source-unavailable."
        )
    return "Existing global compact DB is present; diagnosis should be based on its status/error breakdown."


def _job_result(job: Mapping[str, Any], status: str, error: Optional[str], metrics: Optional[Mapping[str, int]]) -> dict[str, Any]:
    return {
        "status": status,
        "failure_class": _global_failure_class(error) if error else None,
        "error": error,
        "token_id_prefix": str(job.get("token_id", ""))[:12],
        "side": job.get("side"),
        "city": job.get("city"),
        "temperature_metric": job.get("temperature_metric"),
        "question": job.get("question"),
        "start_time": datetime.fromtimestamp(int(job["start_ts"]), tz=timezone.utc).isoformat(),
        "end_time": datetime.fromtimestamp(int(job["end_ts"]), tz=timezone.utc).isoformat(),
        "source_rows_seen": int(metrics.get("source_rows_seen", 0)) if metrics else 0,
        "unique_rows_inserted": int(metrics.get("unique_rows_inserted", 0)) if metrics else 0,
        "compact_rows_for_job": int(metrics.get("compact_rows_for_job", 0)) if metrics else 0,
    }


def _write_global_quote_report(report_dir: Path, report: Mapping[str, Any], *, stem: str = "global-weather-quote-proof") -> tuple[Path, Path]:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = report_dir / f"{timestamp}-{stem}.json"
    md_path = report_dir / f"{timestamp}-{stem}.md"
    return json_path, md_path


def _global_quote_report_markdown(report: Mapping[str, Any]) -> str:
    failures = report.get("failure_classes") or {}
    return "\n".join(
        [
            "# Global Weather Quote Proof",
            "",
            f"Generated: `{report['generated_at_utc']}`",
            "",
            "## Result",
            "",
            f"- Jobs attempted: `{report['jobs_attempted']}`",
            f"- Jobs completed: `{report['jobs_completed']}`",
            f"- Jobs failed: `{report['jobs_failed']}`",
            f"- Quote rows written: `{report['quote_rows_written']}`",
            f"- Source rows seen: `{report['source_rows_seen']}`",
            f"- Compact DB growth bytes: `{report['db_growth_bytes']}`",
            f"- Failure classes: `{dict(failures)}`",
            f"- Safe for larger staged run: `{report['safe_for_larger_staged_run']}`",
            "",
            "## Diagnosis",
            "",
            str(report.get("root_cause_diagnosis") or ""),
            "",
            "## Paths",
            "",
            f"- Compact DB: `{report['paths']['compact_db']}`",
            f"- Decision DB: `{report['paths']['decision_db']}`",
            f"- Raw cache: `{report['paths']['raw_cache_dir']}`",
            "",
            "## Next Batch",
            "",
            f"- Limit jobs: `{report['recommended_next_batch']['limit_jobs']}`",
            f"- Workers: `{report['recommended_next_batch']['workers']}`",
            f"- Reason: {report['recommended_next_batch']['reason']}",
            "",
        ]
    )


def _global_projection_report_markdown(report: Mapping[str, Any]) -> str:
    selector = report["selector_counts"]
    rates = report["observed_batch_rates"]
    projection = report["projection"]
    return "\n".join(
        [
            "# Global Weather Token-Day Projection",
            "",
            f"Generated: `{report['generated_at_utc']}`",
            "",
            "## Current Status",
            "",
            f"- Existing status jobs: `{selector['existing_status_jobs']}`",
            f"- Existing selector status jobs: `{selector['existing_selector_status_jobs']}`",
            f"- External status jobs: `{selector['external_status_jobs']}`",
            f"- Remaining unattempted jobs: `{selector['remaining_unattempted_jobs']}`",
            f"- Total selector jobs: `{selector['total_selector_jobs']}`",
            f"- Compact quote rows: `{report['current_status'].get('compact_quote_history_rows', 0)}`",
            f"- Compact DB bytes: `{report['current_status'].get('compact_db_bytes', 0)}`",
            "",
            "## Observed Rates",
            "",
            f"- Batch reports used: `{rates['batch_report_count']}`",
            f"- Attempted jobs: `{rates['jobs_attempted']}`",
            f"- Active jobs/hour: `{rates['active_jobs_per_hour']}`",
            f"- Rows/attempted job: `{rates['quote_rows_per_attempted_job']}`",
            f"- DB bytes/attempted job: `{rates['db_growth_bytes_per_attempted_job']}`",
            "",
            "## Projection",
            "",
            f"- Active hours remaining: `{projection['projected_active_hours_remaining']}`",
            f"- Active days remaining: `{projection['projected_active_days_remaining']}`",
            f"- Projected DB growth GiB: `{projection['projected_db_growth_gib']}`",
            f"- Projected raw cache growth GiB: `{projection['projected_raw_cache_growth_gib']}`",
            f"- Projected quote rows: `{projection['projected_quote_rows']}`",
            f"- Approval required before more online backfill: `{report['approval_required_before_more_online_backfill']}`",
            "",
            "## Recommendation",
            "",
            str(report["recommended_next_action"]),
            "",
        ]
    )


def _global_scope_options_report_markdown(report: Mapping[str, Any]) -> str:
    overall = report["overall"]
    rates = report["observed_batch_rates"]
    tables = report["scope_tables"]
    metric_lines = [
        f"- `{row['scope']}`: markets `{row['market_rows']}`, jobs `{row['token_day_jobs']}`, "
        f"complete `{row['complete_jobs']}`, failed `{row['failed_jobs']}`, "
        f"unattempted `{row['unattempted_jobs']}`, projected days `{row['projected_active_days_remaining']}`, "
        f"projected DB GiB `{row['projected_db_growth_gib']}`"
        for row in tables["by_temperature_metric"]
    ]
    city_metric_lines = [
        f"- `{row['scope']}`: markets `{row['market_rows']}`, jobs `{row['token_day_jobs']}`, "
        f"complete `{row['complete_jobs']}`, unattempted `{row['unattempted_jobs']}`, "
        f"projected days `{row['projected_active_days_remaining']}`, projected DB GiB `{row['projected_db_growth_gib']}`"
        for row in tables["top_city_metrics_by_remaining_jobs"][:10]
    ]
    return "\n".join(
        [
            "# Global Weather Scope Options",
            "",
            f"Generated: `{report['generated_at_utc']}`",
            "",
            "## Overall",
            "",
            f"- Market rows: `{overall['market_rows']}`",
            f"- Token-day jobs: `{overall['token_day_jobs']}`",
            f"- Complete jobs: `{overall['complete_jobs']}`",
            f"- Failed jobs: `{overall['failed_jobs']}`",
            f"- Unattempted jobs: `{overall['unattempted_jobs']}`",
            f"- Projected active days remaining: `{overall['projected_active_days_remaining']}`",
            f"- Projected DB growth GiB: `{overall['projected_db_growth_gib']}`",
            "",
            "## Observed Rates",
            "",
            f"- Batch reports used: `{rates['batch_report_count']}`",
            f"- Active jobs/hour: `{rates['active_jobs_per_hour']}`",
            f"- Rows/attempted job: `{rates['quote_rows_per_attempted_job']}`",
            f"- DB bytes/attempted job: `{rates['db_growth_bytes_per_attempted_job']}`",
            "",
            "## By Temperature Metric",
            "",
            *metric_lines,
            "",
            "## Largest City/Metric Scopes By Remaining Jobs",
            "",
            *city_metric_lines,
            "",
            "## Recommendation",
            "",
            str(report["recommended_next_action"]),
            "",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
