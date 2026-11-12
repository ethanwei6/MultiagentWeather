from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sqlite3
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import closing
from dataclasses import dataclass
from datetime import date, datetime, time as datetime_time, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


DEFAULT_SOURCE_REPO = Path(
    os.environ.get("WEATHER_SOURCE_REPO") or os.environ.get("DAILYWEATHER_SOURCE_REPO") or "vendor/weather_strategy"
).expanduser()
DEFAULT_START_DATE = date(2025, 6, 30)
DEFAULT_END_DATE = date(2026, 6, 29)
ENTRY_HOURS_UTC = (0, 6, 12, 18)


@dataclass(frozen=True)
class PipelinePaths:
    root: Path

    @property
    def work(self) -> Path:
        return self.root / "work"

    @property
    def telonex_raw(self) -> Path:
        return self.work / "cache" / "telonex" / "raw"

    @property
    def open_meteo_raw(self) -> Path:
        return self.work / "cache" / "open_meteo" / "raw"

    @property
    def backtest_data(self) -> Path:
        return self.work / "data" / "backtest"

    @property
    def logs(self) -> Path:
        return self.work / "logs" / "data_pipeline"

    @property
    def reports(self) -> Path:
        return self.work / "reports" / "data_quality"

    @property
    def market_db(self) -> Path:
        return self.backtest_data / "daily_weather_market_universe.sqlite"

    @property
    def weather_db(self) -> Path:
        return self.backtest_data / "daily_weather_weather_history.sqlite"

    @property
    def price_db(self) -> Path:
        return self.backtest_data / "daily_weather_price_history.sqlite"

    @property
    def compact_price_db(self) -> Path:
        return self.backtest_data / "daily_weather_price_history_compact.sqlite"

    @property
    def decision_db(self) -> Path:
        return self.backtest_data / "daily_weather_decision_artifact.sqlite"


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build DailyWeather point-in-time-safe data artifacts")
    parser.add_argument("--work-root", default=".", help="Workspace root containing work/")
    parser.add_argument("--source-repo", default=str(DEFAULT_SOURCE_REPO), help="Existing weather_strategy source checkout")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("init", help="Create cache, data, log, and report directories plus SQLite schemas")

    market = subparsers.add_parser("market-universe", help="Build normalized Telonex market universe")
    market.add_argument("--start-date", default=DEFAULT_START_DATE.isoformat())
    market.add_argument("--end-date", default=DEFAULT_END_DATE.isoformat())
    market.add_argument("--market-parquet", default=None, help="Optional explicit Telonex markets parquet")
    market.add_argument("--limit-markets", type=int, default=0, help="Limit parsed market/bucket rows; 0 means no limit")
    market.add_argument("--refresh-telonex-markets", action="store_true", help="Download a fresh Telonex markets dataset into raw cache")
    market.add_argument("--telonex-env", default=None, help="Local env file containing TELONEX_API_KEY")

    price = subparsers.add_parser("price-history", help="Build normalized Telonex quote history for universe tokens")
    price.add_argument("--start-date", default=DEFAULT_START_DATE.isoformat())
    price.add_argument("--end-date", default=DEFAULT_END_DATE.isoformat())
    price.add_argument("--limit-markets", type=int, default=0)
    price.add_argument("--workers", type=int, default=2)
    price.add_argument("--cache-only", action="store_true", help="Use existing quote Parquet caches only; do not call Telonex")
    price.add_argument("--telonex-env", default=None, help="Local env file containing TELONEX_API_KEY")
    price.add_argument("--max-staleness-minutes", type=int, default=90)
    price.add_argument("--sides", choices=("yes", "no", "both"), default="both")
    price.add_argument("--progress-every", type=int, default=500)
    price.add_argument("--limit-jobs", type=int, default=0, help="Limit token/window jobs after status filtering; 0 means no limit")
    price.add_argument(
        "--job-status",
        choices=("incomplete", "retryable", "cache_miss", "failed", "all"),
        default="incomplete",
        help="Select token/window jobs by existing price_fetch_status before online or cache ingestion",
    )

    compact = subparsers.add_parser("price-compact-build", help="Build compact normalized quote history from the wide price DB")
    compact.add_argument("--start-date", default="2026-01-01")
    compact.add_argument("--end-date", default=DEFAULT_END_DATE.isoformat())
    compact.add_argument("--limit-jobs", type=int, default=0)
    compact.add_argument("--job-status", choices=("complete", "all"), default="complete")
    compact.add_argument("--progress-every", type=int, default=100)

    compact_backfill = subparsers.add_parser("price-compact-backfill", help="Backfill compact quote history directly from Telonex")
    compact_backfill.add_argument("--start-date", default="2026-01-01")
    compact_backfill.add_argument("--end-date", default=DEFAULT_END_DATE.isoformat())
    compact_backfill.add_argument("--limit-jobs", type=int, default=5)
    compact_backfill.add_argument("--job-status", choices=("cache_miss", "failed", "retryable"), default="cache_miss")
    compact_backfill.add_argument(
        "--failure-class",
        choices=("all", "transient", "rate_limit", "source_unavailable", "http_404", "local_cache_missing", "parser_or_token_mismatch", "unknown"),
        default="all",
        help="When retrying failed jobs, restrict selection by prior compact failure class",
    )
    compact_backfill.add_argument("--workers", type=int, default=1)
    compact_backfill.add_argument("--progress-every", type=int, default=1)
    compact_backfill.add_argument("--telonex-env", default=None, help="Local env file containing TELONEX_API_KEY")
    compact_backfill.add_argument(
        "--telonex-hard-timeout-seconds",
        type=float,
        default=60.0,
        help="Abort one stalled Telonex HTTP request after N seconds; <=0 disables",
    )

    compact_report = subparsers.add_parser("price-compact-report", help="Write compact price-store data-quality report")
    compact_report.add_argument("--start-date", default="2026-01-01")
    compact_report.add_argument("--end-date", default=DEFAULT_END_DATE.isoformat())

    weather = subparsers.add_parser("weather-history", help="Build Open-Meteo forecast and realized weather history")
    weather.add_argument("--start-date", default=DEFAULT_START_DATE.isoformat())
    weather.add_argument("--end-date", default=DEFAULT_END_DATE.isoformat())
    weather.add_argument("--limit-pairs", type=int, default=0, help="Limit city/date pairs; 0 means no limit")
    weather.add_argument("--workers", type=int, default=2)
    weather.add_argument("--cache-only", action="store_true", help="Use existing Open-Meteo raw JSON only; do not call the network")
    weather.add_argument("--forecast-mode", choices=("single-runs", "historical-forecast"), default="single-runs")

    corpus = subparsers.add_parser("weather-corpus-extend", help="Extend local market/weather corpus for live shadow scoring without Telonex")
    corpus.add_argument("--start-date", default=None, help="Defaults to as-of date minus 4 days")
    corpus.add_argument("--end-date", default=None, help="Defaults to as-of UTC date")
    corpus.add_argument("--as-of-utc", default=None)
    corpus.add_argument("--cache-only", action="store_true", help="Use cached Gamma/Open-Meteo payloads only")
    corpus.add_argument(
        "--market-discovery-mode",
        choices=("cached", "online", "cached-and-online"),
        default="cached-and-online",
    )
    corpus.add_argument("--gamma-city-limit", type=int, default=0, help="Limit default city slug probes; 0 means all")
    corpus.add_argument("--weather-pair-limit", type=int, default=0, help="Limit label-available city/date weather fetches; 0 means all")
    corpus.add_argument("--report-dir", default="work/reports/strategy_review")

    decision = subparsers.add_parser("decision-artifact", help="Join markets, quotes, forecasts, and labels at live cadence")
    decision.add_argument("--start-date", default=DEFAULT_START_DATE.isoformat())
    decision.add_argument("--end-date", default=DEFAULT_END_DATE.isoformat())
    decision.add_argument("--limit-markets", type=int, default=0)
    decision.add_argument("--max-staleness-minutes", type=int, default=90)
    decision.add_argument("--price-store", choices=("compact", "wide"), default="compact")

    settlement = subparsers.add_parser("settlement-labels", help="Attach canonical settlement labels to the decision artifact DB")
    settlement.add_argument("--start-date", default="2026-01-01")
    settlement.add_argument("--end-date", default=DEFAULT_END_DATE.isoformat())
    settlement.add_argument("--labels-csv", default=None, help="Canonical settlement labels CSV; defaults to latest audit artifact")

    report = subparsers.add_parser("report", help="Write JSON and Markdown data-quality report")
    report.add_argument("--start-date", default=DEFAULT_START_DATE.isoformat())
    report.add_argument("--end-date", default=DEFAULT_END_DATE.isoformat())

    args = parser.parse_args(argv)
    paths = PipelinePaths(Path(args.work_root).resolve())
    bootstrap(paths, Path(args.source_repo))

    if args.command == "init":
        initialize(paths)
        print(json.dumps({"status": "ok", "paths": _paths_json(paths)}, indent=2, sort_keys=True))
        return 0
    if args.command == "market-universe":
        result = build_market_universe(
            paths,
            source_repo=Path(args.source_repo),
            start=_parse_date_arg(args.start_date),
            end=_parse_date_arg(args.end_date),
            market_parquet=Path(args.market_parquet).expanduser() if args.market_parquet else None,
            limit_markets=args.limit_markets,
            refresh_telonex_markets=args.refresh_telonex_markets,
            telonex_env=Path(args.telonex_env).expanduser() if args.telonex_env else None,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "price-history":
        result = ingest_price_history(
            paths,
            source_repo=Path(args.source_repo),
            start=_parse_date_arg(args.start_date),
            end=_parse_date_arg(args.end_date),
            limit_markets=args.limit_markets,
            workers=args.workers,
            cache_only=args.cache_only,
            telonex_env=Path(args.telonex_env).expanduser() if args.telonex_env else None,
            max_staleness_minutes=args.max_staleness_minutes,
            sides=args.sides,
            progress_every=args.progress_every,
            limit_jobs=args.limit_jobs,
            job_status=args.job_status,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "price-compact-build":
        result = build_compact_price_history(
            paths,
            start=_parse_date_arg(args.start_date),
            end=_parse_date_arg(args.end_date),
            limit_jobs=args.limit_jobs,
            job_status=args.job_status,
            progress_every=args.progress_every,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "price-compact-backfill":
        result = backfill_compact_price_history(
            paths,
            source_repo=Path(args.source_repo),
            start=_parse_date_arg(args.start_date),
            end=_parse_date_arg(args.end_date),
            limit_jobs=args.limit_jobs,
            job_status=args.job_status,
            failure_class=args.failure_class,
            workers=args.workers,
            progress_every=args.progress_every,
            telonex_env=Path(args.telonex_env).expanduser() if args.telonex_env else None,
            telonex_hard_timeout_seconds=args.telonex_hard_timeout_seconds,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "price-compact-report":
        result = write_compact_price_report(
            paths,
            start=_parse_date_arg(args.start_date),
            end=_parse_date_arg(args.end_date),
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "weather-history":
        result = ingest_weather_history(
            paths,
            start=_parse_date_arg(args.start_date),
            end=_parse_date_arg(args.end_date),
            limit_pairs=args.limit_pairs,
            workers=args.workers,
            cache_only=args.cache_only,
            forecast_mode=args.forecast_mode,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "weather-corpus-extend":
        from .weather_corpus_extension import DEFAULT_LOOKBACK_DAYS, run_weather_corpus_extension
        from .weather_corpus_snapshot import CorpusSnapshotError, CorpusWriterLockError

        as_of = _parse_datetime(args.as_of_utc) if args.as_of_utc else datetime.now(timezone.utc)
        if as_of is None:
            raise ValueError(f"Invalid --as-of-utc: {args.as_of_utc}")
        as_of = as_of.astimezone(timezone.utc) if as_of.tzinfo else as_of.replace(tzinfo=timezone.utc)
        start = _parse_date_arg(args.start_date) if args.start_date else as_of.date() - timedelta(days=DEFAULT_LOOKBACK_DAYS)
        end = _parse_date_arg(args.end_date) if args.end_date else as_of.date()
        try:
            result = run_weather_corpus_extension(
                paths=paths,
                source_repo=Path(args.source_repo).expanduser(),
                start=start,
                end=end,
                as_of_utc=as_of,
                cache_only=args.cache_only,
                market_discovery_mode=args.market_discovery_mode,
                gamma_city_limit=args.gamma_city_limit,
                weather_pair_limit=args.weather_pair_limit,
                report_dir=Path(args.report_dir),
            )
        except CorpusWriterLockError as exc:
            print(
                json.dumps(
                    {
                        "status": "writer_lock_rejected",
                        "operational_corpus_ready": False,
                        "error": str(exc),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 3
        except CorpusSnapshotError as exc:
            print(
                json.dumps(
                    {
                        "status": "active_snapshot_invalid",
                        "operational_corpus_ready": False,
                        "error": str(exc),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 4
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if bool(result.get("operational_corpus_ready")) else 2
    if args.command == "decision-artifact":
        result = build_decision_artifact(
            paths,
            start=_parse_date_arg(args.start_date),
            end=_parse_date_arg(args.end_date),
            limit_markets=args.limit_markets,
            max_staleness_minutes=args.max_staleness_minutes,
            price_store=args.price_store,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "settlement-labels":
        result = attach_settlement_labels(
            paths,
            start=_parse_date_arg(args.start_date),
            end=_parse_date_arg(args.end_date),
            labels_csv=Path(args.labels_csv).expanduser() if args.labels_csv else None,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "report":
        result = write_data_quality_report(
            paths,
            start=_parse_date_arg(args.start_date),
            end=_parse_date_arg(args.end_date),
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    raise AssertionError(args.command)


def bootstrap(paths: PipelinePaths, source_repo: Path) -> None:
    vendor = paths.work / "vendor"
    if vendor.exists() and str(vendor) not in sys.path:
        sys.path.insert(0, str(vendor))
    if source_repo.exists() and str(source_repo) not in sys.path:
        sys.path.insert(0, str(source_repo))


def initialize(paths: PipelinePaths) -> None:
    for path in (
        paths.telonex_raw,
        paths.open_meteo_raw,
        paths.backtest_data,
        paths.logs,
        paths.reports,
    ):
        path.mkdir(parents=True, exist_ok=True)
    with _connect(paths.market_db) as conn:
        _ensure_market_schema(conn)
    with _connect(paths.price_db) as conn:
        _ensure_price_schema(conn)
    with _connect(paths.compact_price_db) as conn:
        _ensure_compact_price_schema(conn)
    with _connect(paths.weather_db) as conn:
        _ensure_weather_schema(conn)
    with _connect(paths.decision_db) as conn:
        _ensure_decision_schema(conn)


def initialize_weather_corpus(paths: PipelinePaths) -> None:
    """Initialize only the stores owned by the market/weather corpus writer.

    The live corpus extension must never create or migrate replay quote or
    decision stores as a side effect.  Keeping this initializer separate from
    :func:`initialize` makes that boundary explicit and regression-testable.
    """

    for path in (
        paths.open_meteo_raw,
        paths.backtest_data,
        paths.logs,
        paths.reports,
    ):
        path.mkdir(parents=True, exist_ok=True)
    with _connect(paths.market_db) as conn:
        _ensure_market_schema(conn)
    with _connect(paths.weather_db) as conn:
        _ensure_weather_schema(conn)


def build_market_universe(
    paths: PipelinePaths,
    *,
    source_repo: Path,
    start: date,
    end: date,
    market_parquet: Optional[Path],
    limit_markets: int,
    refresh_telonex_markets: bool,
    telonex_env: Optional[Path],
) -> dict[str, Any]:
    initialize(paths)
    started = time.monotonic()
    parquet_path = market_parquet
    if refresh_telonex_markets:
        parquet_path = _download_telonex_markets(paths, telonex_env=telonex_env)
    if parquet_path is None:
        parquet_path = _find_market_parquet(paths)
    if parquet_path is None or not parquet_path.exists():
        raise FileNotFoundError("No Telonex markets parquet found; pass --market-parquet or --refresh-telonex-markets")

    records = _load_candidate_market_records(parquet_path, start=start, end=end)
    rows: list[dict[str, Any]] = []
    parse_failures = 0
    filtered_out = 0
    for index, record in enumerate(records):
        if limit_markets > 0 and len(rows) >= limit_markets:
            break
        parsed_rows = _market_record_to_universe_rows(record, source_path=parquet_path, source_index=index, start=start, end=end)
        if parsed_rows is None:
            parse_failures += 1
            continue
        if not parsed_rows:
            filtered_out += 1
            continue
        rows.extend(parsed_rows)

    with _connect(paths.market_db) as conn:
        _ensure_market_schema(conn)
        run_id = _insert_pipeline_run(
            conn,
            "market-universe",
            {
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
                "source_path": str(parquet_path),
                "limit_markets": limit_markets,
            },
        )
        conn.execute("DELETE FROM market_universe WHERE market_date_local >= ? AND market_date_local <= ?", (start.isoformat(), end.isoformat()))
        conn.execute("DELETE FROM market_tokens")
        conn.execute("DELETE FROM market_parse_warnings")
        for row in rows:
            _insert_market_row(conn, row, run_id)
        _finish_pipeline_run(conn, run_id, row_count=len(rows))
        conn.commit()

    report = write_data_quality_report(paths, start=start, end=end)
    return {
        "status": "ok",
        "market_rows_written": len(rows),
        "candidate_records": len(records),
        "parse_failures": parse_failures,
        "filtered_outside_target_window": filtered_out,
        "market_db": str(paths.market_db),
        "report_json": report["json_path"],
        "elapsed_seconds": round(time.monotonic() - started, 2),
    }


def ingest_price_history(
    paths: PipelinePaths,
    *,
    source_repo: Path,
    start: date,
    end: date,
    limit_markets: int,
    workers: int,
    cache_only: bool,
    telonex_env: Optional[Path],
    max_staleness_minutes: int,
    sides: str,
    progress_every: int,
    limit_jobs: int,
    job_status: str,
) -> dict[str, Any]:
    initialize(paths)
    started = time.monotonic()
    market_rows = _load_market_rows(paths, start=start, end=end, limit=limit_markets)
    candidate_jobs = _token_jobs_from_market_rows(market_rows, max_staleness_minutes=max_staleness_minutes, sides=sides)
    token_jobs = _filter_price_jobs(paths, candidate_jobs, job_status=job_status, limit_jobs=limit_jobs)
    if not token_jobs:
        return {
            "status": "ok",
            "candidate_token_jobs": len(candidate_jobs),
            "token_jobs": 0,
            "rows_written": 0,
            "elapsed_seconds": 0.0,
        }

    if cache_only:
        return _ingest_price_history_from_cache(
            paths,
            token_jobs,
            start=start,
            end=end,
            limit_markets=limit_markets,
            max_staleness_minutes=max_staleness_minutes,
            sides=sides,
            progress_every=progress_every,
            candidate_token_jobs=len(candidate_jobs),
            job_status=job_status,
            limit_jobs=limit_jobs,
            started=started,
        )

    from weather_strategy.telonex import TelonexClient

    client = TelonexClient(cache_dir=paths.telonex_raw, env_path=str(telonex_env or ".env"))

    quote_cache_dirs = _quote_cache_dirs(paths)
    rows_written = 0
    complete = 0
    failed = 0
    skipped_existing = 0
    with _connect(paths.price_db) as conn:
        _ensure_price_schema(conn)
        run_id = _insert_pipeline_run(
            conn,
            "price-history",
            {
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
                "limit_markets": limit_markets,
                "cache_only": cache_only,
                "max_staleness_minutes": max_staleness_minutes,
                "sides": sides,
                "candidate_token_jobs": len(candidate_jobs),
                "job_status": job_status,
                "limit_jobs": limit_jobs,
            },
        )

    def run_job(job: dict[str, Any]) -> dict[str, Any]:
        with _connect(paths.price_db) as status_conn:
            _ensure_price_schema(status_conn)
            if _price_job_complete(status_conn, job):
                return {"status": "skipped_existing", "job": job, "rows": []}
        try:
            rows = _load_quote_rows_for_job(job, client=client, cache_only=False, quote_cache_dirs=quote_cache_dirs)
            return {"status": "complete", "job": job, "rows": rows}
        except Exception as exc:  # noqa: BLE001 - capture per-token failures in status table
            return {"status": "failed", "job": job, "error": f"{type(exc).__name__}: {str(exc)[:500]}", "rows": []}

    max_workers = max(1, min(workers, len(token_jobs)))
    if max_workers == 1:
        results = []
        for index, job in enumerate(token_jobs, start=1):
            results.append(run_job(job))
            if progress_every > 0 and index % progress_every == 0:
                _progress(progress_every, f"price_history_network_jobs={index}/{len(token_jobs)} elapsed_seconds={time.monotonic() - started:.1f}")
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(run_job, job) for job in token_jobs]
            results = []
            for index, future in enumerate(as_completed(futures), start=1):
                results.append(future.result())
                if progress_every > 0 and index % progress_every == 0:
                    _progress(progress_every, f"price_history_network_jobs={index}/{len(token_jobs)} elapsed_seconds={time.monotonic() - started:.1f}")

    with _connect(paths.price_db) as conn:
        _ensure_price_schema(conn)
        for result in results:
            job = result["job"]
            if result["status"] == "skipped_existing":
                skipped_existing += 1
                continue
            if result["status"] == "complete":
                inserted = _insert_quote_history(conn, job, result["rows"])
                rows_written += inserted
                complete += 1
                _upsert_price_status(conn, job, "complete", rows=inserted, error=None)
            else:
                failed += 1
                _upsert_price_status(conn, job, "failed", rows=0, error=result.get("error"))
        _finish_pipeline_run(conn, run_id, row_count=rows_written)
        conn.commit()
    report = write_data_quality_report(paths, start=start, end=end)
    return {
        "status": "ok",
        "candidate_token_jobs": len(candidate_jobs),
        "token_jobs": len(token_jobs),
        "complete_jobs": complete,
        "failed_jobs": failed,
        "skipped_existing_jobs": skipped_existing,
        "rows_written": rows_written,
        "price_db": str(paths.price_db),
        "report_json": report["json_path"],
        "elapsed_seconds": round(time.monotonic() - started, 2),
    }


def build_compact_price_history(
    paths: PipelinePaths,
    *,
    start: date,
    end: date,
    limit_jobs: int,
    job_status: str,
    progress_every: int,
) -> dict[str, Any]:
    initialize(paths)
    started = time.monotonic()
    candidate_jobs = _token_jobs_from_market_rows(
        _load_market_rows(paths, start=start, end=end, limit=0),
        max_staleness_minutes=90,
        sides="both",
    )
    source_status = _wide_price_status_by_key(paths)
    with _connect(paths.compact_price_db) as compact_conn:
        _ensure_compact_price_schema(compact_conn)
        run_id = _insert_pipeline_run(
            compact_conn,
            "price-compact-build",
            {
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
                "candidate_token_jobs": len(candidate_jobs),
                "limit_jobs": limit_jobs,
                "job_status": job_status,
            },
        )
        token_refs = _ensure_compact_tokens(compact_conn, candidate_jobs)
        _sync_compact_status_from_wide(compact_conn, candidate_jobs, token_refs, source_status)
        compact_conn.commit()

    copy_jobs = [
        job
        for job in candidate_jobs
        if source_status.get(_job_key(job), {}).get("status") == "complete"
    ]
    if job_status == "complete":
        copy_jobs = _filter_compact_jobs(paths, copy_jobs, job_status="pending_copy", limit_jobs=limit_jobs)
    elif limit_jobs > 0:
        copy_jobs = copy_jobs[:limit_jobs]

    source_rows_seen = 0
    unique_rows_inserted = 0
    duplicate_rows_collapsed = 0
    jobs_copied = 0
    with _connect(paths.price_db) as wide_conn, _connect(paths.compact_price_db) as compact_conn:
        wide_conn.row_factory = sqlite3.Row
        compact_conn.row_factory = sqlite3.Row
        _ensure_compact_price_schema(compact_conn)
        token_refs = _compact_token_refs(compact_conn)
        source_ref_cache: dict[str, int] = {}
        for index, job in enumerate(copy_jobs, start=1):
            token_ref = token_refs[str(job["token_id"])]
            metrics = _copy_wide_job_to_compact(wide_conn, compact_conn, job, token_ref, source_ref_cache)
            source_rows_seen += metrics["source_rows_seen"]
            unique_rows_inserted += metrics["unique_rows_inserted"]
            duplicate_rows_collapsed += metrics["duplicate_rows_collapsed"]
            jobs_copied += 1
            _upsert_compact_status(
                compact_conn,
                token_ref,
                job,
                "complete",
                rows=metrics["compact_rows_for_job"],
                source_rows=metrics["source_rows_seen"],
                duplicates=metrics["duplicate_rows_collapsed"],
                conflicts=0,
                error=None,
            )
            if index % 25 == 0:
                compact_conn.commit()
            if progress_every > 0 and index % progress_every == 0:
                _progress(
                    progress_every,
                    f"price_compact_build_jobs={index}/{len(copy_jobs)} source_rows={source_rows_seen} unique_inserted={unique_rows_inserted} elapsed_seconds={time.monotonic() - started:.1f}",
                )
        _finish_pipeline_run(compact_conn, run_id, row_count=unique_rows_inserted)
        compact_conn.commit()

    validation = _validate_compact_samples(paths, copy_jobs[: min(10, len(copy_jobs))])
    quality = _compact_price_quality(paths)
    return {
        "status": "ok",
        "candidate_token_jobs": len(candidate_jobs),
        "jobs_selected": len(copy_jobs),
        "jobs_copied": jobs_copied,
        "source_rows_seen": source_rows_seen,
        "unique_rows_inserted": unique_rows_inserted,
        "duplicate_rows_collapsed": duplicate_rows_collapsed,
        "compact_price_db": str(paths.compact_price_db),
        "quality": quality,
        "validation": validation,
        "elapsed_seconds": round(time.monotonic() - started, 2),
    }


def backfill_compact_price_history(
    paths: PipelinePaths,
    *,
    source_repo: Path,
    start: date,
    end: date,
    limit_jobs: int,
    job_status: str,
    failure_class: str,
    workers: int,
    progress_every: int,
    telonex_env: Optional[Path],
    telonex_hard_timeout_seconds: float,
) -> dict[str, Any]:
    initialize(paths)
    bootstrap(paths, source_repo)
    started = time.monotonic()
    candidate_jobs = _token_jobs_from_market_rows(
        _load_market_rows(paths, start=start, end=end, limit=0),
        max_staleness_minutes=90,
        sides="both",
    )
    source_status = _wide_price_status_by_key(paths)
    with _connect(paths.compact_price_db) as compact_conn:
        _ensure_compact_price_schema(compact_conn)
        run_id = _insert_pipeline_run(
            compact_conn,
            "price-compact-backfill",
            {
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
                "candidate_token_jobs": len(candidate_jobs),
                "limit_jobs": limit_jobs,
                "job_status": job_status,
                "failure_class": failure_class,
            },
        )
        token_refs = _ensure_compact_tokens(compact_conn, candidate_jobs)
        _sync_compact_status_from_wide(compact_conn, candidate_jobs, token_refs, source_status)
        compact_conn.commit()
    jobs = _filter_compact_jobs(
        paths,
        candidate_jobs,
        job_status=job_status,
        limit_jobs=limit_jobs,
        failure_class=failure_class,
    )
    if not jobs:
        return {
            "status": "ok",
            "candidate_token_jobs": len(candidate_jobs),
            "token_jobs": 0,
            "rows_written": 0,
            "elapsed_seconds": round(time.monotonic() - started, 2),
        }

    from weather_strategy.telonex import TelonexClient

    hard_timeout = telonex_hard_timeout_seconds if telonex_hard_timeout_seconds > 0 else None
    client = TelonexClient(
        cache_dir=paths.telonex_raw,
        env_path=str(telonex_env or ".env"),
        hard_timeout_seconds=hard_timeout,
    )
    quote_cache_dirs = _quote_cache_dirs(paths)

    def run_job(job: dict[str, Any]) -> dict[str, Any]:
        try:
            rows = _load_quote_rows_for_job(job, client=client, cache_only=False, quote_cache_dirs=quote_cache_dirs)
            return {"status": "complete", "job": job, "rows": rows}
        except Exception as exc:  # noqa: BLE001
            return {"status": "failed", "job": job, "error": f"{type(exc).__name__}: {str(exc)[:500]}", "rows": []}

    def write_result(
        compact_conn: sqlite3.Connection,
        token_refs: Mapping[str, int],
        source_ref_cache: dict[str, int],
        result: Mapping[str, Any],
    ) -> dict[str, int]:
        job = result["job"]
        token_ref = token_refs[str(job["token_id"])]
        if result["status"] != "complete":
            _upsert_compact_status(
                compact_conn,
                token_ref,
                job,
                "failed",
                rows=0,
                source_rows=0,
                duplicates=0,
                conflicts=0,
                error=result.get("error"),
            )
            return {
                "complete": 0,
                "failed": 1,
                "source_rows_seen": 0,
                "unique_rows_inserted": 0,
                "duplicate_rows_collapsed": 0,
            }
        metrics = _insert_compact_quote_rows(compact_conn, token_ref, job, result["rows"], source_ref_cache)
        _upsert_compact_status(
            compact_conn,
            token_ref,
            job,
            "complete",
            rows=metrics["compact_rows_for_job"],
            source_rows=metrics["source_rows_seen"],
            duplicates=metrics["duplicate_rows_collapsed"],
            conflicts=0,
            error=None,
        )
        return {
            "complete": 1,
            "failed": 0,
            "source_rows_seen": metrics["source_rows_seen"],
            "unique_rows_inserted": metrics["unique_rows_inserted"],
            "duplicate_rows_collapsed": metrics["duplicate_rows_collapsed"],
        }

    max_workers = max(1, min(workers, len(jobs)))
    complete = 0
    failed = 0
    source_rows_seen = 0
    unique_rows_inserted = 0
    duplicate_rows_collapsed = 0
    validation_jobs: list[Mapping[str, Any]] = []
    if max_workers == 1:
        with _connect(paths.compact_price_db) as compact_conn:
            compact_conn.row_factory = sqlite3.Row
            _ensure_compact_price_schema(compact_conn)
            token_refs = _compact_token_refs(compact_conn)
            source_ref_cache: dict[str, int] = {}
            for index, job in enumerate(jobs, start=1):
                result = run_job(job)
                written = write_result(compact_conn, token_refs, source_ref_cache, result)
                complete += written["complete"]
                failed += written["failed"]
                source_rows_seen += written["source_rows_seen"]
                unique_rows_inserted += written["unique_rows_inserted"]
                duplicate_rows_collapsed += written["duplicate_rows_collapsed"]
                if result["status"] == "complete" and len(validation_jobs) < 5:
                    validation_jobs.append(job)
                compact_conn.commit()
                if progress_every > 0 and index % progress_every == 0:
                    _progress(progress_every, f"price_compact_backfill_jobs={index}/{len(jobs)} elapsed_seconds={time.monotonic() - started:.1f}")
            _finish_pipeline_run(compact_conn, run_id, row_count=unique_rows_inserted)
            compact_conn.commit()
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(run_job, job) for job in jobs]
            results = []
            for index, future in enumerate(as_completed(futures), start=1):
                results.append(future.result())
                if progress_every > 0 and index % progress_every == 0:
                    _progress(progress_every, f"price_compact_backfill_jobs={index}/{len(jobs)} elapsed_seconds={time.monotonic() - started:.1f}")
        with _connect(paths.compact_price_db) as compact_conn:
            compact_conn.row_factory = sqlite3.Row
            _ensure_compact_price_schema(compact_conn)
            token_refs = _compact_token_refs(compact_conn)
            source_ref_cache: dict[str, int] = {}
            for result in results:
                written = write_result(compact_conn, token_refs, source_ref_cache, result)
                complete += written["complete"]
                failed += written["failed"]
                source_rows_seen += written["source_rows_seen"]
                unique_rows_inserted += written["unique_rows_inserted"]
                duplicate_rows_collapsed += written["duplicate_rows_collapsed"]
                if result["status"] == "complete" and len(validation_jobs) < 5:
                    validation_jobs.append(result["job"])
            _finish_pipeline_run(compact_conn, run_id, row_count=unique_rows_inserted)
            compact_conn.commit()

    validation = _validate_compact_self_samples(paths, validation_jobs)
    quality = _compact_price_quality(paths)
    return {
        "status": "ok",
        "candidate_token_jobs": len(candidate_jobs),
        "token_jobs": len(jobs),
        "complete_jobs": complete,
        "failed_jobs": failed,
        "source_rows_seen": source_rows_seen,
        "unique_rows_inserted": unique_rows_inserted,
        "duplicate_rows_collapsed": duplicate_rows_collapsed,
        "compact_price_db": str(paths.compact_price_db),
        "quality": quality,
        "validation": validation,
        "elapsed_seconds": round(time.monotonic() - started, 2),
    }


def write_compact_price_report(paths: PipelinePaths, *, start: date, end: date) -> dict[str, Any]:
    initialize(paths)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    quality = _compact_price_quality(paths)
    wide_bytes = paths.price_db.stat().st_size if paths.price_db.exists() else 0
    compact_bytes = paths.compact_price_db.stat().st_size if paths.compact_price_db.exists() else 0
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_start_date": start.isoformat(),
        "target_end_date": end.isoformat(),
        "paths": _paths_json(paths),
        "compact_prices": quality,
        "wide_price_db_bytes": wide_bytes,
        "compact_price_db_bytes": compact_bytes,
        "storage_savings_vs_wide_fraction": round(1 - (compact_bytes / wide_bytes), 6) if wide_bytes else None,
        "storage_savings_vs_wide_percent": round((1 - (compact_bytes / wide_bytes)) * 100, 2) if wide_bytes else None,
    }
    json_path = paths.reports / f"{timestamp}-daily-weather-compact-price-quality.json"
    md_path = paths.reports / f"{timestamp}-daily-weather-compact-price-quality.md"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(_compact_quality_markdown(summary), encoding="utf-8")
    return {"json_path": str(json_path), "markdown_path": str(md_path), "summary": summary}


def ingest_weather_history(
    paths: PipelinePaths,
    *,
    start: date,
    end: date,
    limit_pairs: int,
    workers: int,
    cache_only: bool,
    forecast_mode: str,
) -> dict[str, Any]:
    initialize(paths)
    started = time.monotonic()
    pairs = _load_city_date_pairs(paths, start=start, end=end, limit=limit_pairs)
    if not pairs:
        return {"status": "ok", "city_date_pairs": 0, "forecast_rows": 0, "realized_rows": 0, "elapsed_seconds": 0.0}
    if forecast_mode == "historical-forecast":
        return _ingest_historical_weather_ranges(
            paths,
            pairs,
            start=start,
            end=end,
            limit_pairs=limit_pairs,
            workers=workers,
            cache_only=cache_only,
            started=started,
        )

    with _connect(paths.weather_db) as conn:
        _ensure_weather_schema(conn)
        run_id = _insert_pipeline_run(
            conn,
            "weather-history",
            {
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
                "limit_pairs": limit_pairs,
                "cache_only": cache_only,
                "forecast_mode": forecast_mode,
            },
        )

    def run_pair(pair: dict[str, Any]) -> dict[str, Any]:
        with _connect(paths.weather_db) as status_conn:
            _ensure_weather_schema(status_conn)
            if _weather_pair_complete(status_conn, pair):
                return {"status": "skipped_existing", "pair": pair, "forecast_rows": [], "realized_row": None}
        try:
            forecast_rows = _fetch_forecast_rows(pair, paths=paths, cache_only=cache_only, mode=forecast_mode)
            realized_row = _fetch_realized_weather_row(pair, paths=paths, cache_only=cache_only)
            return {"status": "complete", "pair": pair, "forecast_rows": forecast_rows, "realized_row": realized_row}
        except Exception as exc:  # noqa: BLE001
            return {"status": "failed", "pair": pair, "error": f"{type(exc).__name__}: {str(exc)[:500]}", "forecast_rows": [], "realized_row": None}

    max_workers = max(1, min(workers, len(pairs)))
    if max_workers == 1:
        results = [run_pair(pair) for pair in pairs]
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(run_pair, pair) for pair in pairs]
            results = [future.result() for future in as_completed(futures)]

    forecast_rows_written = 0
    realized_rows_written = 0
    failed = 0
    with _connect(paths.weather_db) as conn:
        _ensure_weather_schema(conn)
        for result in results:
            pair = result["pair"]
            if result["status"] == "skipped_existing":
                continue
            if result["status"] != "complete":
                failed += 1
                _upsert_weather_status(conn, pair, "failed", error=result.get("error"))
                continue
            for row in result["forecast_rows"]:
                _insert_forecast_row(conn, row)
                forecast_rows_written += 1
            if result["realized_row"]:
                _insert_realized_row(conn, result["realized_row"])
                realized_rows_written += 1
            _upsert_weather_status(conn, pair, "complete", error=None)
        _finish_pipeline_run(conn, run_id, row_count=forecast_rows_written + realized_rows_written)
        conn.commit()
    report = write_data_quality_report(paths, start=start, end=end)
    return {
        "status": "ok",
        "city_date_pairs": len(pairs),
        "failed_pairs": failed,
        "forecast_rows_written": forecast_rows_written,
        "realized_rows_written": realized_rows_written,
        "weather_db": str(paths.weather_db),
        "report_json": report["json_path"],
        "elapsed_seconds": round(time.monotonic() - started, 2),
    }


def _ingest_historical_weather_ranges(
    paths: PipelinePaths,
    pairs: list[dict[str, Any]],
    *,
    start: date,
    end: date,
    limit_pairs: int,
    workers: int,
    cache_only: bool,
    started: float,
) -> dict[str, Any]:
    with _connect(paths.weather_db) as conn:
        _ensure_weather_schema(conn)
        run_id = _insert_pipeline_run(
            conn,
            "weather-history-ranges",
            {
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
                "limit_pairs": limit_pairs,
                "cache_only": cache_only,
                "forecast_mode": "historical-forecast",
            },
        )
        pending = [pair for pair in pairs if not _weather_pair_complete(conn, pair)]
    skipped_existing = len(pairs) - len(pending)
    if not pending:
        with _connect(paths.weather_db) as conn:
            _ensure_weather_schema(conn)
            _finish_pipeline_run(conn, run_id, row_count=0)
            conn.commit()
        report = write_data_quality_report(paths, start=start, end=end)
        return {
            "status": "ok",
            "city_date_pairs": len(pairs),
            "skipped_existing_pairs": skipped_existing,
            "failed_pairs": 0,
            "forecast_rows_written": 0,
            "realized_rows_written": 0,
            "weather_db": str(paths.weather_db),
            "report_json": report["json_path"],
            "elapsed_seconds": round(time.monotonic() - started, 2),
        }

    city_groups: dict[tuple[str, str, float, float, str, str], list[dict[str, Any]]] = {}
    for pair in pending:
        key = (
            str(pair["normalized_city"]),
            str(pair.get("country_region") or ""),
            float(pair["latitude"]),
            float(pair["longitude"]),
            str(pair["timezone"]),
            str(pair.get("country_region") or ""),
        )
        city_groups.setdefault(key, []).append(pair)

    def run_city(group_pairs: list[dict[str, Any]]) -> dict[str, Any]:
        group_start = min(date.fromisoformat(str(pair["local_date"])) for pair in group_pairs)
        group_end = max(date.fromisoformat(str(pair["local_date"])) for pair in group_pairs)
        exemplar = group_pairs[0]
        try:
            forecast_payload, forecast_path, forecast_payload_id, forecast_url, forecast_params = _fetch_historical_forecast_payload(
                exemplar,
                group_start,
                group_end,
                paths=paths,
                cache_only=cache_only,
            )
            realized_payload, realized_path, realized_payload_id, realized_url, realized_params = _fetch_realized_payload(
                exemplar,
                group_start,
                group_end,
                paths=paths,
                cache_only=cache_only,
            )
            forecast_rows: list[dict[str, Any]] = []
            realized_rows: list[dict[str, Any]] = []
            for pair in group_pairs:
                local_date = date.fromisoformat(str(pair["local_date"]))
                for decision_time in _decision_times_for_pair(pair):
                    forecast_rows.append(
                        _forecast_row(
                            pair,
                            local_date,
                            decision_time,
                            _forecast_run_time(decision_time),
                            "historical_forecast_archive",
                            None,
                            forecast_payload,
                            forecast_url,
                            forecast_params,
                            forecast_path,
                            forecast_payload_id,
                            flags=["historical_forecast_endpoint_is_asof_approximation"],
                        )
                    )
                realized_rows.append(
                    _realized_row_from_payload(
                        pair,
                        local_date,
                        realized_payload,
                        realized_url,
                        realized_params,
                        realized_path,
                        realized_payload_id,
                    )
                )
            return {"status": "complete", "pairs": group_pairs, "forecast_rows": forecast_rows, "realized_rows": realized_rows}
        except Exception as exc:  # noqa: BLE001
            return {
                "status": "failed",
                "pairs": group_pairs,
                "error": f"{type(exc).__name__}: {str(exc)[:500]}",
                "forecast_rows": [],
                "realized_rows": [],
            }

    groups = list(city_groups.values())
    max_workers = max(1, min(workers, len(groups)))
    if max_workers == 1:
        results = []
        for index, group in enumerate(groups, start=1):
            results.append(run_city(group))
            _progress(max(1, min(5, len(groups))), f"weather_city_groups={index}/{len(groups)} elapsed_seconds={time.monotonic() - started:.1f}")
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(run_city, group) for group in groups]
            results = []
            for index, future in enumerate(as_completed(futures), start=1):
                results.append(future.result())
                _progress(max(1, min(5, len(groups))), f"weather_city_groups={index}/{len(groups)} elapsed_seconds={time.monotonic() - started:.1f}")

    forecast_rows_written = 0
    realized_rows_written = 0
    failed_pairs = 0
    with _connect(paths.weather_db) as conn:
        _ensure_weather_schema(conn)
        for result in results:
            if result["status"] != "complete":
                failed_pairs += len(result["pairs"])
                for pair in result["pairs"]:
                    _upsert_weather_status(conn, pair, "failed", error=result.get("error"))
                continue
            for row in result["forecast_rows"]:
                _insert_forecast_row(conn, row)
                forecast_rows_written += 1
            for row in result["realized_rows"]:
                _insert_realized_row(conn, row)
                realized_rows_written += 1
            for pair in result["pairs"]:
                _upsert_weather_status(conn, pair, "complete", error=None)
        _finish_pipeline_run(conn, run_id, row_count=forecast_rows_written + realized_rows_written)
        conn.commit()
    report = write_data_quality_report(paths, start=start, end=end)
    return {
        "status": "ok",
        "city_date_pairs": len(pairs),
        "skipped_existing_pairs": skipped_existing,
        "failed_pairs": failed_pairs,
        "forecast_rows_written": forecast_rows_written,
        "realized_rows_written": realized_rows_written,
        "weather_db": str(paths.weather_db),
        "report_json": report["json_path"],
        "elapsed_seconds": round(time.monotonic() - started, 2),
    }


def build_decision_artifact(
    paths: PipelinePaths,
    *,
    start: date,
    end: date,
    limit_markets: int,
    max_staleness_minutes: int,
    price_store: str,
) -> dict[str, Any]:
    initialize(paths)
    started = time.monotonic()
    market_rows = _load_market_rows(paths, start=start, end=end, limit=limit_markets)
    rows: list[dict[str, Any]] = []
    price_db = paths.compact_price_db if price_store == "compact" else paths.price_db
    with _connect(price_db) as price_conn, _connect(paths.weather_db) as weather_conn:
        price_conn.row_factory = sqlite3.Row
        weather_conn.row_factory = sqlite3.Row
        if price_store == "compact":
            _ensure_compact_price_schema(price_conn)
            quote_snapshot = _compact_quote_snapshot
        else:
            _ensure_price_schema(price_conn)
            quote_snapshot = _quote_snapshot
        for market in market_rows:
            for decision_time in _decision_times_for_market(market):
                forecast = _forecast_snapshot(weather_conn, market, decision_time)
                realized = _realized_snapshot(weather_conn, market)
                for side, token_id in (("YES", market.get("yes_token_id")), ("NO", market.get("no_token_id"))):
                    if not token_id:
                        continue
                    quote = quote_snapshot(price_conn, str(token_id), decision_time, max_staleness_minutes=max_staleness_minutes)
                    rows.append(
                        _decision_row(
                            market,
                            decision_time,
                            quote,
                            forecast,
                            realized,
                            side=side,
                            token_id=str(token_id),
                            price_store=price_store,
                        )
                    )

    with _connect(paths.decision_db) as conn:
        _ensure_decision_schema(conn)
        run_id = _insert_pipeline_run(
            conn,
            "decision-artifact",
            {
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
                "limit_markets": limit_markets,
                "max_staleness_minutes": max_staleness_minutes,
                "price_store": price_store,
            },
        )
        clear_stats = _clear_decision_snapshots_for_rebuild(conn, start=start, end=end)
        for row in rows:
            _insert_decision_snapshot(conn, row)
        _record_decision_artifact_scope(
            conn,
            start=start,
            end=end,
            price_store=price_store,
            max_staleness_minutes=max_staleness_minutes,
            row_count=len(rows),
        )
        observed_high_stats = _populate_observed_high_snapshots(paths, conn, start=start, end=end)
        _finish_pipeline_run(conn, run_id, row_count=len(rows))
        conn.commit()
    report = write_data_quality_report(paths, start=start, end=end)
    return {
        "status": "ok",
        "decision_rows_written": len(rows),
        "decision_db": str(paths.decision_db),
        "cleared_rows_before_rebuild": clear_stats,
        "observed_high": observed_high_stats,
        "price_store": price_store,
        "report_json": report["json_path"],
        "elapsed_seconds": round(time.monotonic() - started, 2),
    }


def attach_settlement_labels(
    paths: PipelinePaths,
    *,
    start: date,
    end: date,
    labels_csv: Optional[Path],
) -> dict[str, Any]:
    initialize(paths)
    started = time.monotonic()
    source_csv = labels_csv or _latest_settlement_labels_csv(paths)
    required = {
        "universe_row_id",
        "market_date_local",
        "canonical_label",
        "recommended_action",
        "exclude_from_backtest_replay",
    }
    rows: list[dict[str, Any]] = []
    skipped_outside_window = 0
    skipped_missing_required = 0
    with source_csv.open(newline="") as file:
        reader = csv.DictReader(file)
        missing_columns = required.difference(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(f"Canonical settlement labels CSV missing columns: {sorted(missing_columns)}")
        for raw in reader:
            row_date = (raw.get("market_date_local") or "").strip()
            if not row_date or row_date < start.isoformat() or row_date > end.isoformat():
                skipped_outside_window += 1
                continue
            if not raw.get("universe_row_id"):
                skipped_missing_required += 1
                continue
            rows.append(raw)

    imported_at = datetime.now(timezone.utc).isoformat()
    recommended_actions: dict[str, int] = {}
    replay_excluded = 0
    with _connect(paths.decision_db) as conn:
        _ensure_decision_schema(conn)
        run_id = _insert_pipeline_run(
            conn,
            "settlement-labels",
            {
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
                "labels_csv": str(source_csv),
            },
        )
        conn.execute(
            "DELETE FROM canonical_settlement_labels WHERE market_date_local >= ? AND market_date_local <= ?",
            (start.isoformat(), end.isoformat()),
        )
        for raw in rows:
            action = raw.get("recommended_action") or ""
            recommended_actions[action] = recommended_actions.get(action, 0) + 1
            if _to_int(raw.get("exclude_from_backtest_replay")) == 1:
                replay_excluded += 1
            conn.execute(
                """
                INSERT OR REPLACE INTO canonical_settlement_labels(
                    universe_row_id, market_id, slug, normalized_city, country_region, timezone,
                    market_date_local, bucket_label, current_bucket_type, canonical_bucket_type,
                    operator, market_units, settlement_unit, settlement_precision,
                    threshold_low_f, threshold_high_f, threshold_low_display, threshold_high_display,
                    resolved_outcome, canonical_label, canonical_label_source, settlement_source,
                    source_family, source_station, source_url, resolution_date_text,
                    resolution_date_conflict, source_text_summary, open_meteo_source_name,
                    open_meteo_realized_high_f, open_meteo_realized_high_display,
                    open_meteo_current_continuous_label, open_meteo_source_rule_proxy_label,
                    current_continuous_mismatch, source_rule_proxy_mismatch,
                    distance_to_threshold_f, distance_to_threshold_bucket_f,
                    group_constraint_status, recommended_action, exclude_from_backtest_replay,
                    exclude_from_open_meteo_training, audit_flags, raw_label_json,
                    source_csv_path, imported_at_utc
                )
                VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                (
                    raw.get("universe_row_id"),
                    raw.get("market_id"),
                    raw.get("slug"),
                    raw.get("normalized_city"),
                    raw.get("country_region"),
                    raw.get("timezone"),
                    raw.get("market_date_local"),
                    raw.get("bucket_label"),
                    raw.get("current_bucket_type"),
                    raw.get("canonical_bucket_type"),
                    raw.get("operator"),
                    raw.get("market_units"),
                    raw.get("settlement_unit"),
                    raw.get("settlement_precision"),
                    _to_float(raw.get("threshold_low_f")),
                    _to_float(raw.get("threshold_high_f")),
                    raw.get("threshold_low_display"),
                    raw.get("threshold_high_display"),
                    _to_int(raw.get("resolved_outcome")),
                    _to_int(raw.get("canonical_label")),
                    raw.get("canonical_label_source"),
                    raw.get("settlement_source"),
                    raw.get("source_family"),
                    raw.get("source_station"),
                    raw.get("source_url"),
                    raw.get("resolution_date_text"),
                    _to_int(raw.get("resolution_date_conflict")),
                    raw.get("source_text_summary"),
                    raw.get("open_meteo_source_name"),
                    _to_float(raw.get("open_meteo_realized_high_f")),
                    raw.get("open_meteo_realized_high_display"),
                    _to_int(raw.get("open_meteo_current_continuous_label")),
                    _to_int(raw.get("open_meteo_source_rule_proxy_label")),
                    _to_int(raw.get("current_continuous_mismatch")),
                    _to_int(raw.get("source_rule_proxy_mismatch")),
                    _to_float(raw.get("distance_to_threshold_f")),
                    raw.get("distance_to_threshold_bucket_f"),
                    raw.get("group_constraint_status"),
                    raw.get("recommended_action"),
                    _to_int(raw.get("exclude_from_backtest_replay")),
                    _to_int(raw.get("exclude_from_open_meteo_training")),
                    raw.get("audit_flags"),
                    json.dumps(raw, sort_keys=True),
                    str(source_csv),
                    imported_at,
                ),
            )
        _finish_pipeline_run(conn, run_id, row_count=len(rows))
        conn.commit()
        joined_rows = int(conn.execute("SELECT COUNT(*) FROM decision_snapshots_current_with_labels").fetchone()[0])
        decision_rows_with_label = int(
            conn.execute(
                "SELECT COUNT(*) FROM decision_snapshots_current_with_labels WHERE canonical_label IS NOT NULL"
            ).fetchone()[0]
        )
        decision_rows_excluded = int(
            conn.execute(
                "SELECT COUNT(*) FROM decision_snapshots_current_with_labels WHERE exclude_from_backtest_replay = 1"
            ).fetchone()[0]
        )
    return {
        "status": "ok",
        "labels_csv": str(source_csv),
        "decision_db": str(paths.decision_db),
        "rows_imported": len(rows),
        "skipped_outside_window": skipped_outside_window,
        "skipped_missing_required": skipped_missing_required,
        "recommended_actions": dict(sorted(recommended_actions.items())),
        "rows_exclude_from_backtest_replay": replay_excluded,
        "decision_rows_current_view": joined_rows,
        "decision_rows_with_canonical_label": decision_rows_with_label,
        "decision_rows_exclude_from_backtest_replay": decision_rows_excluded,
        "elapsed_seconds": round(time.monotonic() - started, 2),
    }


def write_data_quality_report(paths: PipelinePaths, *, start: date, end: date) -> dict[str, Any]:
    initialize(paths)
    summary = _quality_summary(paths, start=start, end=end)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = paths.reports / f"{timestamp}-daily-weather-data-quality.json"
    md_path = paths.reports / f"{timestamp}-daily-weather-data-quality.md"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(_quality_markdown(summary), encoding="utf-8")
    return {"json_path": str(json_path), "markdown_path": str(md_path), "summary": summary}


def _ensure_market_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS pipeline_runs (
            run_id TEXT PRIMARY KEY,
            command TEXT NOT NULL,
            started_at_utc TEXT NOT NULL,
            finished_at_utc TEXT,
            row_count INTEGER DEFAULT 0,
            settings_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS source_payloads (
            payload_id TEXT PRIMARY KEY,
            source_name TEXT NOT NULL,
            cache_path TEXT NOT NULL,
            payload_sha256 TEXT,
            fetched_at_utc TEXT,
            request_json TEXT
        );
        CREATE TABLE IF NOT EXISTS market_universe (
            universe_row_id TEXT PRIMARY KEY,
            market_id TEXT,
            condition_id TEXT,
            slug TEXT,
            question TEXT,
            description TEXT,
            event_id TEXT,
            event_slug TEXT,
            event_title TEXT,
            city_text TEXT,
            normalized_city TEXT,
            country_region TEXT,
            latitude REAL,
            longitude REAL,
            timezone TEXT,
            market_date_local TEXT,
            created_at_utc TEXT,
            start_at_utc TEXT,
            close_time_utc TEXT,
            resolved_at_utc TEXT,
            relevant_cutoff_time_utc TEXT,
            bucket_label TEXT,
            bucket_type TEXT,
            operator TEXT,
            threshold_low_f REAL,
            threshold_high_f REAL,
            units TEXT,
            yes_token_id TEXT,
            no_token_id TEXT,
            resolved_outcome INTEGER,
            settlement_source TEXT,
            telonex_source_path TEXT,
            telonex_source_row INTEGER,
            telonex_quotes_from TEXT,
            telonex_quotes_to TEXT,
            parser_confidence REAL,
            parse_warnings_json TEXT,
            source_payload_id TEXT,
            first_seen_run_id TEXT,
            updated_at_utc TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_market_universe_date ON market_universe(market_date_local);
        CREATE INDEX IF NOT EXISTS idx_market_universe_city_date ON market_universe(normalized_city, market_date_local);
        CREATE INDEX IF NOT EXISTS idx_market_universe_yes_token ON market_universe(yes_token_id);
        CREATE TABLE IF NOT EXISTS market_tokens (
            token_id TEXT PRIMARY KEY,
            universe_row_id TEXT NOT NULL,
            market_id TEXT,
            side TEXT NOT NULL,
            outcome_label TEXT,
            resolved_outcome INTEGER,
            FOREIGN KEY(universe_row_id) REFERENCES market_universe(universe_row_id)
        );
        CREATE TABLE IF NOT EXISTS market_parse_warnings (
            warning_id TEXT PRIMARY KEY,
            universe_row_id TEXT,
            market_id TEXT,
            warning TEXT NOT NULL,
            source_path TEXT
        );
        """
    )


def _ensure_price_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS pipeline_runs (
            run_id TEXT PRIMARY KEY,
            command TEXT NOT NULL,
            started_at_utc TEXT NOT NULL,
            finished_at_utc TEXT,
            row_count INTEGER DEFAULT 0,
            settings_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS source_payloads (
            payload_id TEXT PRIMARY KEY,
            source_name TEXT NOT NULL,
            cache_path TEXT NOT NULL,
            payload_sha256 TEXT,
            fetched_at_utc TEXT,
            request_json TEXT
        );
        CREATE TABLE IF NOT EXISTS token_quote_history (
            token_id TEXT NOT NULL,
            market_id TEXT,
            universe_row_id TEXT,
            side TEXT,
            timestamp_utc TEXT NOT NULL,
            bid REAL,
            ask REAL,
            mid REAL,
            last_price REAL,
            spread REAL,
            bid_size REAL,
            ask_size REAL,
            liquidity_depth REAL,
            source_payload_id TEXT,
            source_cache_path TEXT NOT NULL,
            fetch_timestamp_utc TEXT NOT NULL,
            data_quality_flags TEXT,
            PRIMARY KEY(token_id, timestamp_utc, source_cache_path)
        );
        CREATE INDEX IF NOT EXISTS idx_quote_token_time ON token_quote_history(token_id, timestamp_utc);
        CREATE INDEX IF NOT EXISTS idx_quote_market_time ON token_quote_history(market_id, timestamp_utc);
        CREATE INDEX IF NOT EXISTS idx_quote_universe_time ON token_quote_history(universe_row_id, timestamp_utc);
        CREATE TABLE IF NOT EXISTS price_fetch_status (
            token_id TEXT NOT NULL,
            market_id TEXT,
            universe_row_id TEXT,
            start_ts INTEGER,
            end_ts INTEGER,
            status TEXT NOT NULL,
            rows_written INTEGER DEFAULT 0,
            error TEXT,
            retryable INTEGER DEFAULT 1,
            updated_at_utc TEXT NOT NULL,
            PRIMARY KEY(token_id, start_ts, end_ts)
        );
        CREATE INDEX IF NOT EXISTS idx_price_fetch_status_status ON price_fetch_status(status);
        CREATE INDEX IF NOT EXISTS idx_price_fetch_status_universe ON price_fetch_status(universe_row_id);
        """
    )
    _ensure_column(conn, "price_fetch_status", "retryable", "INTEGER DEFAULT 1")
    _backfill_retryable(conn, "price_fetch_status")


def _ensure_compact_price_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS pipeline_runs (
            run_id TEXT PRIMARY KEY,
            command TEXT NOT NULL,
            started_at_utc TEXT NOT NULL,
            finished_at_utc TEXT,
            row_count INTEGER DEFAULT 0,
            settings_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS compact_tokens (
            token_ref INTEGER PRIMARY KEY,
            token_id TEXT NOT NULL UNIQUE,
            market_id TEXT,
            universe_row_id TEXT,
            side TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_compact_tokens_universe ON compact_tokens(universe_row_id);
        CREATE TABLE IF NOT EXISTS compact_sources (
            source_ref INTEGER PRIMARY KEY,
            source_cache_path TEXT NOT NULL UNIQUE,
            source_payload_id TEXT,
            payload_sha256 TEXT
        );
        CREATE TABLE IF NOT EXISTS compact_quote_history (
            token_ref INTEGER NOT NULL,
            timestamp_utc TEXT NOT NULL,
            bid REAL,
            ask REAL,
            mid REAL,
            last_price REAL,
            spread REAL,
            bid_size REAL,
            ask_size REAL,
            liquidity_depth REAL,
            source_ref INTEGER,
            duplicate_count INTEGER NOT NULL DEFAULT 1,
            conflict_count INTEGER NOT NULL DEFAULT 0,
            data_quality_flags TEXT,
            PRIMARY KEY(token_ref, timestamp_utc),
            FOREIGN KEY(token_ref) REFERENCES compact_tokens(token_ref),
            FOREIGN KEY(source_ref) REFERENCES compact_sources(source_ref)
        ) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS compact_price_fetch_status (
            token_ref INTEGER NOT NULL,
            start_ts INTEGER NOT NULL,
            end_ts INTEGER NOT NULL,
            status TEXT NOT NULL,
            rows_written INTEGER DEFAULT 0,
            source_rows_seen INTEGER DEFAULT 0,
            duplicate_rows_seen INTEGER DEFAULT 0,
            conflict_rows_seen INTEGER DEFAULT 0,
            error TEXT,
            retryable INTEGER DEFAULT 1,
            updated_at_utc TEXT NOT NULL,
            PRIMARY KEY(token_ref, start_ts, end_ts),
            FOREIGN KEY(token_ref) REFERENCES compact_tokens(token_ref)
        );
        CREATE INDEX IF NOT EXISTS idx_compact_status_status ON compact_price_fetch_status(status);
        """
    )
    _backfill_retryable(conn, "compact_price_fetch_status")


def _ensure_weather_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS pipeline_runs (
            run_id TEXT PRIMARY KEY,
            command TEXT NOT NULL,
            started_at_utc TEXT NOT NULL,
            finished_at_utc TEXT,
            row_count INTEGER DEFAULT 0,
            settings_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS source_payloads (
            payload_id TEXT PRIMARY KEY,
            source_name TEXT NOT NULL,
            cache_path TEXT NOT NULL,
            payload_sha256 TEXT,
            fetched_at_utc TEXT,
            request_json TEXT
        );
        CREATE TABLE IF NOT EXISTS forecast_snapshots (
            forecast_id TEXT PRIMARY KEY,
            normalized_city TEXT NOT NULL,
            country_region TEXT,
            latitude REAL NOT NULL,
            longitude REAL NOT NULL,
            timezone TEXT NOT NULL,
            local_date TEXT NOT NULL,
            decision_time_utc TEXT NOT NULL,
            forecast_run_time_utc TEXT,
            source_name TEXT NOT NULL,
            model_name TEXT,
            temperature_2m_max_f REAL,
            hourly_temperature_2m_max_f REAL,
            apparent_temperature_max_f REAL,
            humidity_mean REAL,
            dew_point_max_f REAL,
            precipitation_sum_in REAL,
            cloud_cover_mean REAL,
            wind_speed_10m_max_mph REAL,
            feature_json TEXT,
            request_url TEXT,
            request_params_json TEXT,
            source_payload_id TEXT,
            raw_cache_path TEXT,
            fetch_timestamp_utc TEXT NOT NULL,
            data_quality_flags TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_forecast_city_date_decision ON forecast_snapshots(normalized_city, local_date, decision_time_utc);
        CREATE TABLE IF NOT EXISTS realized_daily_weather (
            realized_id TEXT PRIMARY KEY,
            normalized_city TEXT NOT NULL,
            country_region TEXT,
            latitude REAL NOT NULL,
            longitude REAL NOT NULL,
            timezone TEXT NOT NULL,
            local_date TEXT NOT NULL,
            temperature_2m_max_f REAL,
            hourly_temperature_2m_max_f REAL,
            apparent_temperature_max_f REAL,
            humidity_mean REAL,
            dew_point_max_f REAL,
            precipitation_sum_in REAL,
            cloud_cover_mean REAL,
            wind_speed_10m_max_mph REAL,
            source_name TEXT NOT NULL,
            feature_json TEXT,
            request_url TEXT,
            request_params_json TEXT,
            source_payload_id TEXT,
            raw_cache_path TEXT,
            fetch_timestamp_utc TEXT NOT NULL,
            data_quality_flags TEXT,
            UNIQUE(normalized_city, local_date, source_name)
        );
        CREATE TABLE IF NOT EXISTS weather_fetch_status (
            normalized_city TEXT NOT NULL,
            local_date TEXT NOT NULL,
            status TEXT NOT NULL,
            error TEXT,
            retryable INTEGER DEFAULT 1,
            updated_at_utc TEXT NOT NULL,
            PRIMARY KEY(normalized_city, local_date)
        );
        """
    )
    _ensure_column(conn, "weather_fetch_status", "retryable", "INTEGER DEFAULT 1")
    _backfill_retryable(conn, "weather_fetch_status")


def _ensure_decision_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS pipeline_runs (
            run_id TEXT PRIMARY KEY,
            command TEXT NOT NULL,
            started_at_utc TEXT NOT NULL,
            finished_at_utc TEXT,
            row_count INTEGER DEFAULT 0,
            settings_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS decision_snapshots (
            decision_id TEXT PRIMARY KEY,
            universe_row_id TEXT NOT NULL,
            market_id TEXT,
            slug TEXT,
            question TEXT,
            normalized_city TEXT,
            country_region TEXT,
            timezone TEXT,
            market_date_local TEXT,
            bucket_label TEXT,
            bucket_type TEXT,
            threshold_low_f REAL,
            threshold_high_f REAL,
            side TEXT,
            yes_token_id TEXT,
            token_id TEXT,
            no_token_id TEXT,
            decision_time_utc TEXT NOT NULL,
            decision_hour_utc INTEGER,
            lead_days INTEGER,
            quote_store TEXT,
            quote_timestamp_utc TEXT,
            bid REAL,
            ask REAL,
            mid REAL,
            last_price REAL,
            spread REAL,
            bid_size REAL,
            ask_size REAL,
            liquidity_depth REAL,
            quote_source_payload_id TEXT,
            quote_source_cache_path TEXT,
            quote_data_quality_flags TEXT,
            price_stale_seconds REAL,
            forecast_run_time_utc TEXT,
            forecast_source_name TEXT,
            forecast_temperature_2m_max_f REAL,
            forecast_features_json TEXT,
            realized_temperature_2m_max_f REAL,
            resolved_outcome INTEGER,
            settlement_source TEXT,
            data_quality_flags TEXT,
            created_at_utc TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_decision_token_time ON decision_snapshots(token_id, decision_time_utc);
        CREATE INDEX IF NOT EXISTS idx_decision_market_time ON decision_snapshots(market_id, decision_time_utc);
        CREATE INDEX IF NOT EXISTS idx_decision_market_date ON decision_snapshots(market_date_local);
        CREATE TABLE IF NOT EXISTS decision_artifact_scope (
            scope_id TEXT PRIMARY KEY CHECK(scope_id = 'current'),
            target_start_date TEXT NOT NULL,
            target_end_date TEXT NOT NULL,
            price_store TEXT NOT NULL,
            max_staleness_minutes INTEGER NOT NULL,
            built_at_utc TEXT NOT NULL,
            row_count INTEGER NOT NULL
        );
        CREATE VIEW IF NOT EXISTS decision_snapshots_current AS
        SELECT d.*
        FROM decision_snapshots d
        JOIN decision_artifact_scope s ON s.scope_id = 'current'
        WHERE d.market_date_local >= s.target_start_date
          AND d.market_date_local <= s.target_end_date;
        CREATE TABLE IF NOT EXISTS observed_high_snapshots (
            observed_high_id TEXT PRIMARY KEY,
            normalized_city TEXT NOT NULL,
            market_date_local TEXT NOT NULL,
            timezone TEXT NOT NULL,
            decision_time_utc TEXT NOT NULL,
            decision_local_hour INTEGER,
            observed_high_so_far_f REAL,
            observed_high_observed_at_utc TEXT,
            observed_high_source TEXT,
            observed_high_sample_count INTEGER NOT NULL DEFAULT 0,
            observed_high_total_sample_count INTEGER NOT NULL DEFAULT 0,
            observed_high_is_final_at_decision INTEGER NOT NULL DEFAULT 0,
            observed_high_minutes_before_decision REAL,
            observed_high_data_quality_flags TEXT,
            observed_high_missing_reason TEXT,
            observed_high_source_payload_id TEXT,
            observed_high_raw_cache_path TEXT,
            created_at_utc TEXT NOT NULL,
            UNIQUE(normalized_city, market_date_local, decision_time_utc)
        );
        CREATE INDEX IF NOT EXISTS idx_observed_high_decision
            ON observed_high_snapshots(normalized_city, market_date_local, decision_time_utc);
        CREATE TABLE IF NOT EXISTS canonical_settlement_labels (
            universe_row_id TEXT PRIMARY KEY,
            market_id TEXT,
            slug TEXT,
            normalized_city TEXT,
            country_region TEXT,
            timezone TEXT,
            market_date_local TEXT,
            bucket_label TEXT,
            current_bucket_type TEXT,
            canonical_bucket_type TEXT,
            operator TEXT,
            market_units TEXT,
            settlement_unit TEXT,
            settlement_precision TEXT,
            threshold_low_f REAL,
            threshold_high_f REAL,
            threshold_low_display TEXT,
            threshold_high_display TEXT,
            resolved_outcome INTEGER,
            canonical_label INTEGER,
            canonical_label_source TEXT,
            settlement_source TEXT,
            source_family TEXT,
            source_station TEXT,
            source_url TEXT,
            resolution_date_text TEXT,
            resolution_date_conflict INTEGER,
            source_text_summary TEXT,
            open_meteo_source_name TEXT,
            open_meteo_realized_high_f REAL,
            open_meteo_realized_high_display TEXT,
            open_meteo_current_continuous_label INTEGER,
            open_meteo_source_rule_proxy_label INTEGER,
            current_continuous_mismatch INTEGER,
            source_rule_proxy_mismatch INTEGER,
            distance_to_threshold_f REAL,
            distance_to_threshold_bucket_f TEXT,
            group_constraint_status TEXT,
            recommended_action TEXT,
            exclude_from_backtest_replay INTEGER,
            exclude_from_open_meteo_training INTEGER,
            audit_flags TEXT,
            raw_label_json TEXT NOT NULL,
            source_csv_path TEXT NOT NULL,
            imported_at_utc TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_canonical_labels_market_date ON canonical_settlement_labels(market_date_local);
        CREATE INDEX IF NOT EXISTS idx_canonical_labels_action ON canonical_settlement_labels(recommended_action);
        CREATE INDEX IF NOT EXISTS idx_canonical_labels_exclude ON canonical_settlement_labels(exclude_from_backtest_replay);
        """
    )
    conn.execute("DROP VIEW IF EXISTS decision_snapshots_current_with_labels")
    conn.execute("DROP VIEW IF EXISTS decision_snapshots_current_with_observed_high")
    conn.execute("DROP VIEW IF EXISTS decision_snapshots_current_enriched")
    conn.execute(
        """
        CREATE VIEW decision_snapshots_current_with_observed_high AS
        SELECT
          d.*,
          o.observed_high_so_far_f,
          o.observed_high_observed_at_utc,
          o.observed_high_source,
          o.decision_local_hour,
          o.observed_high_sample_count,
          o.observed_high_total_sample_count,
          o.observed_high_is_final_at_decision,
          o.observed_high_minutes_before_decision,
          o.observed_high_data_quality_flags,
          o.observed_high_missing_reason,
          o.observed_high_source_payload_id,
          o.observed_high_raw_cache_path
        FROM decision_snapshots_current d
        LEFT JOIN observed_high_snapshots o
          ON o.normalized_city = d.normalized_city
         AND o.market_date_local = d.market_date_local
         AND o.decision_time_utc = d.decision_time_utc
        """
    )
    conn.execute(
        """
        CREATE VIEW decision_snapshots_current_with_labels AS
        SELECT
          d.*,
          l.canonical_label,
          l.canonical_label_source,
          l.canonical_bucket_type,
          l.settlement_unit,
          l.settlement_precision,
          l.source_family AS canonical_source_family,
          l.source_station AS canonical_source_station,
          l.source_url AS canonical_source_url,
          l.resolution_date_text,
          l.resolution_date_conflict,
          l.open_meteo_current_continuous_label,
          l.open_meteo_source_rule_proxy_label,
          l.recommended_action,
          l.exclude_from_backtest_replay,
          l.exclude_from_open_meteo_training,
          l.audit_flags AS settlement_audit_flags
        FROM decision_snapshots_current d
        LEFT JOIN canonical_settlement_labels l ON l.universe_row_id = d.universe_row_id
        """
    )
    conn.execute(
        """
        CREATE VIEW decision_snapshots_current_enriched AS
        SELECT
          d.*,
          o.observed_high_so_far_f,
          o.observed_high_observed_at_utc,
          o.observed_high_source,
          o.decision_local_hour,
          o.observed_high_sample_count,
          o.observed_high_total_sample_count,
          o.observed_high_is_final_at_decision,
          o.observed_high_minutes_before_decision,
          o.observed_high_data_quality_flags,
          o.observed_high_missing_reason,
          o.observed_high_source_payload_id,
          o.observed_high_raw_cache_path,
          l.canonical_label,
          l.canonical_label_source,
          l.canonical_bucket_type,
          l.settlement_unit,
          l.settlement_precision,
          l.source_family AS canonical_source_family,
          l.source_station AS canonical_source_station,
          l.source_url AS canonical_source_url,
          l.resolution_date_text,
          l.resolution_date_conflict,
          l.open_meteo_current_continuous_label,
          l.open_meteo_source_rule_proxy_label,
          l.recommended_action,
          l.exclude_from_backtest_replay,
          l.exclude_from_open_meteo_training,
          l.audit_flags AS settlement_audit_flags
        FROM decision_snapshots_current d
        LEFT JOIN observed_high_snapshots o
          ON o.normalized_city = d.normalized_city
         AND o.market_date_local = d.market_date_local
         AND o.decision_time_utc = d.decision_time_utc
        LEFT JOIN canonical_settlement_labels l ON l.universe_row_id = d.universe_row_id
        """
    )
    _ensure_column(conn, "decision_snapshots", "side", "TEXT")
    _ensure_column(conn, "decision_snapshots", "yes_token_id", "TEXT")
    _ensure_column(conn, "decision_snapshots", "decision_hour_utc", "INTEGER")
    _ensure_column(conn, "decision_snapshots", "lead_days", "INTEGER")
    _ensure_column(conn, "decision_snapshots", "quote_store", "TEXT")
    _ensure_column(conn, "decision_snapshots", "bid_size", "REAL")
    _ensure_column(conn, "decision_snapshots", "ask_size", "REAL")
    _ensure_column(conn, "decision_snapshots", "liquidity_depth", "REAL")
    _ensure_column(conn, "decision_snapshots", "quote_source_payload_id", "TEXT")
    _ensure_column(conn, "decision_snapshots", "quote_source_cache_path", "TEXT")
    _ensure_column(conn, "decision_snapshots", "quote_data_quality_flags", "TEXT")
    _ensure_column(conn, "observed_high_snapshots", "decision_local_hour", "INTEGER")


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, definition: str) -> None:
    existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    if column not in existing:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def _backfill_retryable(conn: sqlite3.Connection, table: str) -> None:
    conn.execute(
        f"""
        UPDATE {table}
        SET retryable = CASE WHEN status = 'complete' THEN 0 ELSE 1 END
        WHERE retryable IS NULL
           OR retryable NOT IN (0, 1)
           OR (status = 'complete' AND retryable != 0)
           OR (status != 'complete' AND retryable != 1)
        """
    )


def _insert_pipeline_run(conn: sqlite3.Connection, command: str, settings: Mapping[str, Any]) -> str:
    run_id = _stable_id("run", command, datetime.now(timezone.utc).isoformat(), settings)
    conn.execute(
        "INSERT INTO pipeline_runs(run_id, command, started_at_utc, settings_json) VALUES (?, ?, ?, ?)",
        (run_id, command, datetime.now(timezone.utc).isoformat(), json.dumps(dict(settings), sort_keys=True)),
    )
    return run_id


def _finish_pipeline_run(conn: sqlite3.Connection, run_id: str, *, row_count: int) -> None:
    conn.execute(
        "UPDATE pipeline_runs SET finished_at_utc = ?, row_count = ? WHERE run_id = ?",
        (datetime.now(timezone.utc).isoformat(), row_count, run_id),
    )


def _download_telonex_markets(paths: PipelinePaths, *, telonex_env: Optional[Path]) -> Path:
    from weather_strategy.telonex import TelonexClient

    client = TelonexClient(cache_dir=paths.telonex_raw, env_path=str(telonex_env or ".env"))
    return client.download_dataset_parquet(exchange="polymarket", dataset="markets")


def _find_market_parquet(paths: PipelinePaths) -> Optional[Path]:
    candidates = (
        paths.telonex_raw / "polymarket" / "datasets" / "markets.parquet",
        paths.work / "cache" / "live_like_backtest" / "telonex" / "polymarket" / "datasets" / "markets.parquet",
        paths.work / "cache" / "telonex_long_backtest" / "telonex" / "polymarket" / "datasets" / "markets.parquet",
        paths.work / "cache" / "long_backtest" / "telonex" / "polymarket" / "datasets" / "markets.parquet",
        paths.work / "cache" / "telonex_datasets" / "polymarket_markets.parquet",
    )
    return next((path for path in candidates if path.exists()), None)


def _load_candidate_market_records(path: Path, *, start: date, end: date) -> list[Mapping[str, Any]]:
    import polars as pl  # type: ignore

    cols = pl.read_parquet(path, n_rows=0).columns
    text_cols = [col for col in ("question", "description", "slug", "event_slug", "event_title") if col in cols]
    if not text_cols:
        raise ValueError(f"Telonex markets parquet has no expected text columns: {path}")
    text_filter = pl.any_horizontal(
        [
            pl.col(col).fill_null("").cast(pl.String).str.to_lowercase().str.contains("highest temperature", literal=True)
            for col in text_cols
        ]
    )
    low_filter = pl.any_horizontal(
        [
            pl.col(col).fill_null("").cast(pl.String).str.to_lowercase().str.contains("lowest temperature", literal=True)
            for col in text_cols
        ]
    )
    scan = pl.scan_parquet(path).with_row_index("_source_row_index").filter(text_filter & ~low_filter)
    if "status" in cols:
        scan = scan.filter(pl.col("status").fill_null("") == "resolved")
    if "end_date_us" in cols:
        min_us = _date_to_epoch_us(start - timedelta(days=10))
        max_us = _date_to_epoch_us(end + timedelta(days=11)) - 1
        scan = scan.filter((pl.col("end_date_us") >= min_us) & (pl.col("end_date_us") <= max_us))
    return [dict(row) for row in scan.sort("end_date_us").collect().to_dicts()]


def _market_record_to_universe_rows(
    record: Mapping[str, Any],
    *,
    source_path: Path,
    source_index: int,
    start: date,
    end: date,
) -> Optional[list[dict[str, Any]]]:
    from weather_strategy.long_backtest import (
        _binary_no_token,
        _market_parse_today,
        _polymarket_yes_payout,
        _telonex_market_record_to_raw,
    )
    from weather_strategy.parser import parse_weather_market

    raw = _telonex_market_record_to_raw(record)
    if raw is None:
        return None
    market = parse_weather_market(raw, today=_market_parse_today(raw))
    if market is None or market.target_date is None or market.city is None:
        return None
    if market.target_date < start or market.target_date > end:
        return []

    no_token_id = _binary_no_token(raw)
    resolved_yes = _polymarket_yes_payout(raw)
    rows = []
    payload_json = json.dumps(_jsonable(record), sort_keys=True, default=str)
    payload_id = _stable_id("telonex-market-record", source_path, source_index, payload_json)
    for bucket in market.buckets:
        warnings = _market_warnings(market, bucket, no_token_id, raw)
        bucket_info = _bucket_info(bucket)
        row_id = _stable_id("market-row", market.id, market.slug, bucket.label, bucket.token_id)
        payout = _bucket_resolved_outcome(bucket, raw, fallback_yes=resolved_yes)
        rows.append(
            {
                "universe_row_id": row_id,
                "market_id": str(record.get("market_id") or market.id or ""),
                "condition_id": str(record.get("market_id") or market.id or ""),
                "slug": market.slug,
                "question": market.question,
                "description": market.resolution_rules,
                "event_id": _none_or_str(record.get("event_id")),
                "event_slug": market.event_slug,
                "event_title": _none_or_str(record.get("event_title")),
                "city_text": market.city.display_name,
                "normalized_city": market.city.display_name,
                "country_region": market.city.state,
                "latitude": market.city.latitude,
                "longitude": market.city.longitude,
                "timezone": market.city.timezone,
                "market_date_local": market.target_date.isoformat(),
                "created_at_utc": _iso_from_epoch_us(record.get("created_at_us")),
                "start_at_utc": _iso_from_epoch_us(record.get("start_date_us") or record.get("prepared_at_us")),
                "close_time_utc": _iso_from_epoch_us(record.get("end_date_us")),
                "resolved_at_utc": _iso_from_epoch_us(record.get("settled_at_us")),
                "relevant_cutoff_time_utc": _iso_from_epoch_us(record.get("end_date_us")),
                "bucket_label": bucket.label,
                "bucket_type": bucket_info["bucket_type"],
                "operator": bucket_info["operator"],
                "threshold_low_f": bucket.lower_f,
                "threshold_high_f": bucket.upper_f,
                "units": bucket.resolution_unit or "F",
                "yes_token_id": bucket.token_id,
                "no_token_id": no_token_id,
                "resolved_outcome": payout,
                "settlement_source": "telonex_polymarket_result_id" if payout is not None else "telonex_unresolved_or_ambiguous",
                "telonex_source_path": str(source_path),
                "telonex_source_row": int(record.get("_source_row_index", source_index) or source_index),
                "telonex_quotes_from": _none_or_str(record.get("quotes_from")),
                "telonex_quotes_to": _none_or_str(record.get("quotes_to")),
                "parser_confidence": _parser_confidence(warnings),
                "parse_warnings_json": json.dumps(warnings, sort_keys=True),
                "source_payload_id": payload_id,
                "source_payload": {
                    "payload_id": payload_id,
                    "source_name": "telonex_markets_parquet",
                    "cache_path": str(source_path),
                    "payload_sha256": _sha256_text(payload_json),
                    "request_json": json.dumps({"row_index": record.get("_source_row_index", source_index)}, sort_keys=True),
                },
                "parse_warnings": warnings,
            }
        )
    return rows


def _insert_market_row(conn: sqlite3.Connection, row: Mapping[str, Any], run_id: str) -> None:
    payload = row["source_payload"]
    conn.execute(
        """
        INSERT OR REPLACE INTO source_payloads(payload_id, source_name, cache_path, payload_sha256, fetched_at_utc, request_json)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            payload["payload_id"],
            payload["source_name"],
            payload["cache_path"],
            payload["payload_sha256"],
            None,
            payload["request_json"],
        ),
    )
    columns = [
        "universe_row_id",
        "market_id",
        "condition_id",
        "slug",
        "question",
        "description",
        "event_id",
        "event_slug",
        "event_title",
        "city_text",
        "normalized_city",
        "country_region",
        "latitude",
        "longitude",
        "timezone",
        "market_date_local",
        "created_at_utc",
        "start_at_utc",
        "close_time_utc",
        "resolved_at_utc",
        "relevant_cutoff_time_utc",
        "bucket_label",
        "bucket_type",
        "operator",
        "threshold_low_f",
        "threshold_high_f",
        "units",
        "yes_token_id",
        "no_token_id",
        "resolved_outcome",
        "settlement_source",
        "telonex_source_path",
        "telonex_source_row",
        "telonex_quotes_from",
        "telonex_quotes_to",
        "parser_confidence",
        "parse_warnings_json",
        "source_payload_id",
    ]
    values = [row.get(column) for column in columns]
    values.extend([run_id, datetime.now(timezone.utc).isoformat()])
    conn.execute(
        f"""
        INSERT OR REPLACE INTO market_universe({", ".join(columns)}, first_seen_run_id, updated_at_utc)
        VALUES ({", ".join(["?"] * (len(columns) + 2))})
        """,
        values,
    )
    if row.get("yes_token_id"):
        conn.execute(
            """
            INSERT OR REPLACE INTO market_tokens(token_id, universe_row_id, market_id, side, outcome_label, resolved_outcome)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (row["yes_token_id"], row["universe_row_id"], row["market_id"], "YES", row["bucket_label"], row["resolved_outcome"]),
        )
    if row.get("no_token_id"):
        no_resolved = 1 - int(row["resolved_outcome"]) if row.get("resolved_outcome") in (0, 1) else None
        conn.execute(
            """
            INSERT OR REPLACE INTO market_tokens(token_id, universe_row_id, market_id, side, outcome_label, resolved_outcome)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (row["no_token_id"], row["universe_row_id"], row["market_id"], "NO", "No", no_resolved),
        )
    for warning in row.get("parse_warnings", []):
        warning_id = _stable_id("warning", row["universe_row_id"], warning)
        conn.execute(
            """
            INSERT OR REPLACE INTO market_parse_warnings(warning_id, universe_row_id, market_id, warning, source_path)
            VALUES (?, ?, ?, ?, ?)
            """,
            (warning_id, row["universe_row_id"], row["market_id"], warning, row["telonex_source_path"]),
        )


def _load_market_rows(paths: PipelinePaths, *, start: date, end: date, limit: int) -> list[dict[str, Any]]:
    with _connect(paths.market_db) as conn:
        conn.row_factory = sqlite3.Row
        query = """
            SELECT * FROM market_universe
            WHERE market_date_local >= ? AND market_date_local <= ?
            ORDER BY market_date_local, normalized_city, market_id, bucket_label
        """
        params: list[Any] = [start.isoformat(), end.isoformat()]
        if limit > 0:
            query += " LIMIT ?"
            params.append(limit)
        return [dict(row) for row in conn.execute(query, params).fetchall()]


def _ingest_price_history_from_cache(
    paths: PipelinePaths,
    token_jobs: list[dict[str, Any]],
    *,
    start: date,
    end: date,
    limit_markets: int,
    max_staleness_minutes: int,
    sides: str,
    progress_every: int,
    candidate_token_jobs: int,
    job_status: str,
    limit_jobs: int,
    started: float,
) -> dict[str, Any]:
    quote_cache_dirs = _quote_cache_dirs(paths)
    with _connect(paths.price_db) as conn:
        _ensure_price_schema(conn)
        run_id = _insert_pipeline_run(
            conn,
            "price-history-cache",
            {
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
                "limit_markets": limit_markets,
                "cache_only": True,
                "max_staleness_minutes": max_staleness_minutes,
                "sides": sides,
                "candidate_token_jobs": candidate_token_jobs,
                "job_status": job_status,
                "limit_jobs": limit_jobs,
            },
        )
        incomplete_jobs = [job for job in token_jobs if not _price_job_complete(conn, job)]
    skipped_existing = len(token_jobs) - len(incomplete_jobs)
    if not incomplete_jobs or not quote_cache_dirs:
        with _connect(paths.price_db) as conn:
            _ensure_price_schema(conn)
            if not quote_cache_dirs:
                for job in incomplete_jobs:
                    _upsert_price_status(conn, job, "cache_miss", rows=0, error="No local Telonex quote cache directories found")
            _finish_pipeline_run(conn, run_id, row_count=0)
            conn.commit()
        report = write_data_quality_report(paths, start=start, end=end)
        return {
            "status": "ok",
            "candidate_token_jobs": candidate_token_jobs,
            "token_jobs": len(token_jobs),
            "complete_jobs": 0,
            "cache_miss_jobs": len(incomplete_jobs) if not quote_cache_dirs else 0,
            "failed_jobs": 0,
            "skipped_existing_jobs": skipped_existing,
            "rows_written": 0,
            "price_db": str(paths.price_db),
            "report_json": report["json_path"],
            "elapsed_seconds": round(time.monotonic() - started, 2),
        }

    jobs_by_token: dict[str, list[dict[str, Any]]] = {}
    for job in incomplete_jobs:
        jobs_by_token.setdefault(str(job["token_id"]), []).append(job)
    min_day = min(datetime.fromtimestamp(int(job["start_ts"]), tz=timezone.utc).date() for job in incomplete_jobs)
    max_day = max(datetime.fromtimestamp(int(job["end_ts"]), tz=timezone.utc).date() for job in incomplete_jobs)
    quote_paths = [
        path
        for directory in quote_cache_dirs
        for path in sorted(directory.glob("*.parquet"))
        if (path_date := _quote_cache_path_date(path)) is not None and min_day <= path_date <= max_day
    ]
    row_counts_by_job = {_job_key(job): 0 for job in incomplete_jobs}
    rows_written = 0
    failed_paths = 0
    with _connect(paths.price_db) as conn:
        _ensure_price_schema(conn)
        for index, path in enumerate(quote_paths, start=1):
            try:
                rows_written += _insert_matching_quote_rows_from_path(conn, path, jobs_by_token, row_counts_by_job)
            except Exception:
                failed_paths += 1
            if progress_every > 0 and index % progress_every == 0:
                _progress(
                    progress_every,
                    f"price_cache_files={index}/{len(quote_paths)} rows_written={rows_written} elapsed_seconds={time.monotonic() - started:.1f}",
                )
            if index % 250 == 0:
                conn.commit()
        complete = 0
        cache_miss = 0
        for job in incomplete_jobs:
            row_count = row_counts_by_job[_job_key(job)]
            if row_count > 0:
                complete += 1
                _upsert_price_status(conn, job, "complete", rows=row_count, error=None)
            else:
                cache_miss += 1
                _upsert_price_status(conn, job, "cache_miss", rows=0, error="No matching cached Telonex quote parquet found")
        _finish_pipeline_run(conn, run_id, row_count=rows_written)
        conn.commit()
    report = write_data_quality_report(paths, start=start, end=end)
    return {
        "status": "ok",
        "candidate_token_jobs": candidate_token_jobs,
        "token_jobs": len(token_jobs),
        "complete_jobs": complete,
        "cache_miss_jobs": cache_miss,
        "failed_paths": failed_paths,
        "failed_jobs": 0,
        "skipped_existing_jobs": skipped_existing,
        "rows_written": rows_written,
        "price_db": str(paths.price_db),
        "report_json": report["json_path"],
        "elapsed_seconds": round(time.monotonic() - started, 2),
    }


def _token_jobs_from_market_rows(rows: Iterable[Mapping[str, Any]], *, max_staleness_minutes: int, sides: str = "both") -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int]] = set()
    for row in rows:
        decisions = _decision_times_for_market(row)
        if not decisions:
            continue
        start_ts = int((min(decisions) - timedelta(minutes=max_staleness_minutes)).timestamp())
        end_ts = int((max(decisions) + timedelta(seconds=1)).timestamp())
        quote_start = _parse_date(row.get("telonex_quotes_from"))
        quote_end = _parse_date(row.get("telonex_quotes_to"))
        if quote_start is not None:
            start_ts = max(start_ts, int(datetime.combine(quote_start, datetime_time(0, 0), tzinfo=timezone.utc).timestamp()))
        if quote_end is not None:
            end_ts = min(end_ts, int(datetime.combine(quote_end + timedelta(days=1), datetime_time(0, 0), tzinfo=timezone.utc).timestamp()) - 1)
        if end_ts <= start_ts:
            continue
        side_items: list[tuple[str, Any]] = []
        if sides in ("yes", "both"):
            side_items.append(("YES", row.get("yes_token_id")))
        if sides in ("no", "both"):
            side_items.append(("NO", row.get("no_token_id")))
        for side, token_id in side_items:
            if not token_id:
                continue
            key = (str(token_id), start_ts, end_ts)
            if key in seen:
                continue
            seen.add(key)
            jobs.append(
                {
                    "token_id": str(token_id),
                    "market_id": row.get("market_id"),
                    "universe_row_id": row.get("universe_row_id"),
                    "side": side,
                    "slug": row.get("slug"),
                    "outcome": "Yes" if side == "YES" else "No",
                    "start_ts": start_ts,
                    "end_ts": end_ts,
                }
            )
    return jobs


def _filter_price_jobs(
    paths: PipelinePaths,
    jobs: list[dict[str, Any]],
    *,
    job_status: str,
    limit_jobs: int,
) -> list[dict[str, Any]]:
    if job_status == "all" and limit_jobs <= 0:
        return jobs
    with _connect(paths.price_db) as conn:
        conn.row_factory = sqlite3.Row
        _ensure_price_schema(conn)
        status_by_key = {
            (str(row["token_id"]), int(row["start_ts"]), int(row["end_ts"])): {
                "status": row["status"],
                "retryable": int(row["retryable"] or 0),
            }
            for row in conn.execute(
                """
                SELECT token_id, start_ts, end_ts, status, retryable
                FROM price_fetch_status
                """
            )
        }
    selected: list[dict[str, Any]] = []
    for job in jobs:
        state = status_by_key.get(_job_key(job))
        status = state["status"] if state else None
        retryable = bool(state and state["retryable"])
        include = False
        if job_status == "all":
            include = True
        elif job_status == "incomplete":
            include = status != "complete"
        elif job_status == "retryable":
            include = retryable or status is None
        else:
            include = status == job_status
        if include:
            selected.append(job)
            if limit_jobs > 0 and len(selected) >= limit_jobs:
                break
    return selected


def _wide_price_status_by_key(paths: PipelinePaths) -> dict[tuple[str, int, int], dict[str, Any]]:
    with _connect(paths.price_db) as conn:
        conn.row_factory = sqlite3.Row
        _ensure_price_schema(conn)
        return {
            (str(row["token_id"]), int(row["start_ts"]), int(row["end_ts"])): dict(row)
            for row in conn.execute(
                """
                SELECT token_id, market_id, universe_row_id, start_ts, end_ts, status, rows_written, error, retryable, updated_at_utc
                FROM price_fetch_status
                """
            )
        }


def _ensure_compact_tokens(conn: sqlite3.Connection, jobs: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    for job in jobs:
        conn.execute(
            """
            INSERT OR IGNORE INTO compact_tokens(token_id, market_id, universe_row_id, side)
            VALUES (?, ?, ?, ?)
            """,
            (job["token_id"], job.get("market_id"), job.get("universe_row_id"), job.get("side")),
        )
    return _compact_token_refs(conn)


def _compact_token_refs(conn: sqlite3.Connection) -> dict[str, int]:
    conn.row_factory = sqlite3.Row
    return {str(row["token_id"]): int(row["token_ref"]) for row in conn.execute("SELECT token_ref, token_id FROM compact_tokens")}


def _sync_compact_status_from_wide(
    conn: sqlite3.Connection,
    jobs: Iterable[Mapping[str, Any]],
    token_refs: Mapping[str, int],
    source_status: Mapping[tuple[str, int, int], Mapping[str, Any]],
) -> None:
    conn.row_factory = sqlite3.Row
    existing_status = {
        (int(row["token_ref"]), int(row["start_ts"]), int(row["end_ts"])): str(row["status"])
        for row in conn.execute(
            "SELECT token_ref, start_ts, end_ts, status FROM compact_price_fetch_status"
        )
    }
    for job in jobs:
        token_ref = token_refs[str(job["token_id"])]
        status_key = (token_ref, int(job["start_ts"]), int(job["end_ts"]))
        current_status = existing_status.get(status_key)
        if current_status == "complete":
            continue
        state = source_status.get(_job_key(job))
        if state is None:
            status = "missing_source_status"
            error = "No matching wide price_fetch_status row"
            retryable = 1
        elif state.get("status") == "complete":
            status = "pending_copy"
            error = None
            retryable = 1
        else:
            status = str(state.get("status") or "unknown")
            error = state.get("error")
            retryable = _status_retryable(status)
        if current_status and current_status not in {"pending_copy", "missing_source_status"} and status != "pending_copy":
            continue
        conn.execute(
            """
            INSERT OR REPLACE INTO compact_price_fetch_status(
                token_ref, start_ts, end_ts, status, rows_written, source_rows_seen,
                duplicate_rows_seen, conflict_rows_seen, error, retryable, updated_at_utc
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                token_ref,
                job["start_ts"],
                job["end_ts"],
                status,
                0,
                0,
                0,
                0,
                error,
                retryable,
                datetime.now(timezone.utc).isoformat(),
            ),
        )


def _filter_compact_jobs(
    paths: PipelinePaths,
    jobs: list[dict[str, Any]],
    *,
    job_status: str,
    limit_jobs: int,
    failure_class: str = "all",
) -> list[dict[str, Any]]:
    with _connect(paths.compact_price_db) as conn:
        conn.row_factory = sqlite3.Row
        _ensure_compact_price_schema(conn)
        status_by_key = {
            (str(row["token_id"]), int(row["start_ts"]), int(row["end_ts"])): {
                "status": row["status"],
                "retryable": int(row["retryable"] or 0),
                "error": row["error"],
            }
            for row in conn.execute(
                """
                SELECT t.token_id, s.start_ts, s.end_ts, s.status, s.retryable, s.error
                FROM compact_price_fetch_status s
                JOIN compact_tokens t ON t.token_ref = s.token_ref
                """
            )
        }
    selected: list[dict[str, Any]] = []
    for job in jobs:
        state = status_by_key.get(_job_key(job))
        status = state["status"] if state else None
        retryable = bool(state and state["retryable"])
        if job_status == "retryable":
            include = retryable or status is None
        else:
            include = status == job_status
        if include and failure_class != "all":
            include = status == "failed" and _compact_failure_class_matches(
                failure_class,
                _compact_failure_class(state.get("error") if state else None),
            )
        if include:
            selected.append(job)
            if limit_jobs > 0 and len(selected) >= limit_jobs:
                break
    return selected


def _compact_failure_class(error: Optional[str]) -> str:
    text = (error or "").lower()
    if "http 404" in text or "http error 404" in text or "404: not found" in text:
        return "http_404"
    if "cached missing file" in text or "missing-response" in text:
        return "local_cache_missing"
    if (
        "nodename nor servname" in text
        or "network is unreachable" in text
        or "no route to host" in text
        or "connection reset" in text
        or "timed out" in text
        or "timeout" in text
        or "handshake operation timed out" in text
        or "temporary failure" in text
    ):
        return "transient"
    if "429" in text or "rate limit" in text or "rate-limit" in text or "too many requests" in text:
        return "rate_limit"
    if "invalid asset" in text or "unknown asset" in text or "token mismatch" in text:
        return "parser_or_token_mismatch"
    return "unknown"


def _compact_failure_class_matches(requested: str, actual: str) -> bool:
    if requested == "all":
        return True
    if requested == "source_unavailable":
        return actual in {"http_404", "local_cache_missing"}
    return requested == actual


def _copy_wide_job_to_compact(
    wide_conn: sqlite3.Connection,
    compact_conn: sqlite3.Connection,
    job: Mapping[str, Any],
    token_ref: int,
    source_ref_cache: dict[str, int],
) -> dict[str, int]:
    start_iso = datetime.fromtimestamp(int(job["start_ts"]), tz=timezone.utc).isoformat()
    end_iso = datetime.fromtimestamp(int(job["end_ts"]), tz=timezone.utc).isoformat()
    rows = wide_conn.execute(
        """
        SELECT timestamp_utc, bid, ask, mid, last_price, spread, bid_size, ask_size,
               liquidity_depth, source_payload_id, source_cache_path, data_quality_flags
        FROM token_quote_history
        WHERE token_id = ? AND timestamp_utc >= ? AND timestamp_utc <= ?
        ORDER BY timestamp_utc
        """,
        (job["token_id"], start_iso, end_iso),
    )
    before = _compact_job_row_count(compact_conn, token_ref, start_iso, end_iso)
    source_rows_seen = 0
    batch: list[Mapping[str, Any]] = []
    for row in rows:
        batch.append(dict(row))
        source_rows_seen += 1
        if len(batch) >= 5000:
            _insert_compact_quote_row_batch(compact_conn, token_ref, batch, source_ref_cache)
            batch.clear()
    if batch:
        _insert_compact_quote_row_batch(compact_conn, token_ref, batch, source_ref_cache)
    after = _compact_job_row_count(compact_conn, token_ref, start_iso, end_iso)
    unique_rows_inserted = max(0, after - before)
    return {
        "source_rows_seen": source_rows_seen,
        "unique_rows_inserted": unique_rows_inserted,
        "duplicate_rows_collapsed": max(0, source_rows_seen - unique_rows_inserted),
        "compact_rows_for_job": after,
    }


def _insert_compact_quote_rows(
    compact_conn: sqlite3.Connection,
    token_ref: int,
    job: Mapping[str, Any],
    rows: Iterable[Mapping[str, Any]],
    source_ref_cache: dict[str, int],
) -> dict[str, int]:
    start_iso = datetime.fromtimestamp(int(job["start_ts"]), tz=timezone.utc).isoformat()
    end_iso = datetime.fromtimestamp(int(job["end_ts"]), tz=timezone.utc).isoformat()
    before = _compact_job_row_count(compact_conn, token_ref, start_iso, end_iso)
    source_rows_seen = 0
    batch: list[Mapping[str, Any]] = []
    for row in rows:
        batch.append(row)
        source_rows_seen += 1
        if len(batch) >= 5000:
            _insert_compact_quote_row_batch(compact_conn, token_ref, batch, source_ref_cache)
            batch.clear()
    if batch:
        _insert_compact_quote_row_batch(compact_conn, token_ref, batch, source_ref_cache)
    after = _compact_job_row_count(compact_conn, token_ref, start_iso, end_iso)
    unique_rows_inserted = max(0, after - before)
    return {
        "source_rows_seen": source_rows_seen,
        "unique_rows_inserted": unique_rows_inserted,
        "duplicate_rows_collapsed": max(0, source_rows_seen - unique_rows_inserted),
        "compact_rows_for_job": after,
    }


def _insert_compact_quote_row_batch(
    conn: sqlite3.Connection,
    token_ref: int,
    rows: Iterable[Mapping[str, Any]],
    source_ref_cache: dict[str, int],
) -> None:
    values = []
    for row in rows:
        source_ref = _compact_source_ref(conn, row, source_ref_cache)
        values.append(
            (
                token_ref,
                row["timestamp_utc"],
                row.get("bid"),
                row.get("ask"),
                row.get("mid"),
                row.get("last_price"),
                row.get("spread"),
                row.get("bid_size"),
                row.get("ask_size"),
                row.get("liquidity_depth"),
                source_ref,
                row.get("data_quality_flags"),
            )
        )
    conn.executemany(
        """
        INSERT OR IGNORE INTO compact_quote_history(
            token_ref, timestamp_utc, bid, ask, mid, last_price, spread,
            bid_size, ask_size, liquidity_depth, source_ref, data_quality_flags
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        values,
    )


def _compact_source_ref(conn: sqlite3.Connection, row: Mapping[str, Any], source_ref_cache: dict[str, int]) -> Optional[int]:
    source_path = row.get("source_cache_path")
    if not source_path:
        return None
    source_path = str(source_path)
    cached = source_ref_cache.get(source_path)
    if cached is not None:
        return cached
    conn.execute(
        """
        INSERT OR IGNORE INTO compact_sources(source_cache_path, source_payload_id, payload_sha256)
        VALUES (?, ?, ?)
        """,
        (source_path, row.get("source_payload_id"), row.get("payload_sha256")),
    )
    source_ref = conn.execute("SELECT source_ref FROM compact_sources WHERE source_cache_path = ?", (source_path,)).fetchone()[0]
    source_ref_cache[source_path] = int(source_ref)
    return int(source_ref)


def _compact_job_row_count(conn: sqlite3.Connection, token_ref: int, start_iso: str, end_iso: str) -> int:
    return int(
        conn.execute(
            """
            SELECT COUNT(*)
            FROM compact_quote_history
            WHERE token_ref = ? AND timestamp_utc >= ? AND timestamp_utc <= ?
            """,
            (token_ref, start_iso, end_iso),
        ).fetchone()[0]
    )


def _upsert_compact_status(
    conn: sqlite3.Connection,
    token_ref: int,
    job: Mapping[str, Any],
    status: str,
    *,
    rows: int,
    source_rows: int,
    duplicates: int,
    conflicts: int,
    error: Optional[str],
) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO compact_price_fetch_status(
            token_ref, start_ts, end_ts, status, rows_written, source_rows_seen,
            duplicate_rows_seen, conflict_rows_seen, error, retryable, updated_at_utc
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            token_ref,
            job["start_ts"],
            job["end_ts"],
            status,
            rows,
            source_rows,
            duplicates,
            conflicts,
            error,
            _status_retryable(status),
            datetime.now(timezone.utc).isoformat(),
        ),
    )


def _compact_price_quality(paths: PipelinePaths) -> dict[str, Any]:
    with _connect(paths.compact_price_db) as conn:
        conn.row_factory = sqlite3.Row
        _ensure_compact_price_schema(conn)
        if paths.market_db.exists():
            conn.execute("ATTACH DATABASE ? AS marketdb", (str(paths.market_db),))
        status_counts = [
            dict(row)
            for row in conn.execute(
                """
                SELECT status, COUNT(*) AS count, COALESCE(SUM(rows_written), 0) AS rows_written,
                       COALESCE(SUM(source_rows_seen), 0) AS source_rows_seen,
                       COALESCE(SUM(duplicate_rows_seen), 0) AS duplicate_rows_seen,
                       COALESCE(SUM(retryable), 0) AS retryable_count
                FROM compact_price_fetch_status
                GROUP BY status
                ORDER BY status
                """
            )
        ]
        row = conn.execute(
            """
            SELECT
              (SELECT COUNT(*) FROM compact_tokens) AS tokens,
              (SELECT COUNT(*) FROM compact_sources) AS sources,
              (SELECT COUNT(*) FROM compact_quote_history) AS quote_rows,
              (SELECT MIN(timestamp_utc) FROM compact_quote_history) AS min_quote_timestamp,
              (SELECT MAX(timestamp_utc) FROM compact_quote_history) AS max_quote_timestamp
            """
        ).fetchone()
        field_presence = dict(
            conn.execute(
                """
                SELECT
                  SUM(CASE WHEN bid IS NOT NULL THEN 1 ELSE 0 END) AS bid_rows,
                  SUM(CASE WHEN ask IS NOT NULL THEN 1 ELSE 0 END) AS ask_rows,
                  SUM(CASE WHEN mid IS NOT NULL THEN 1 ELSE 0 END) AS mid_rows,
                  SUM(CASE WHEN last_price IS NOT NULL THEN 1 ELSE 0 END) AS last_price_rows,
                  SUM(CASE WHEN spread IS NOT NULL THEN 1 ELSE 0 END) AS spread_rows,
                  SUM(CASE WHEN bid_size IS NOT NULL THEN 1 ELSE 0 END) AS bid_size_rows,
                  SUM(CASE WHEN ask_size IS NOT NULL THEN 1 ELSE 0 END) AS ask_size_rows,
                  SUM(CASE WHEN liquidity_depth IS NOT NULL THEN 1 ELSE 0 END) AS liquidity_depth_rows,
                  SUM(CASE WHEN source_ref IS NOT NULL THEN 1 ELSE 0 END) AS source_ref_rows
                FROM compact_quote_history
                """
            ).fetchone()
        )
        source_audit = dict(
            conn.execute(
                """
                SELECT
                  COUNT(*) AS source_refs,
                  SUM(CASE WHEN source_payload_id IS NOT NULL THEN 1 ELSE 0 END) AS source_payload_id_refs,
                  SUM(CASE WHEN payload_sha256 IS NOT NULL THEN 1 ELSE 0 END) AS payload_sha256_refs,
                  SUM(CASE WHEN source_cache_path IS NOT NULL AND source_cache_path != '' THEN 1 ELSE 0 END) AS source_cache_path_refs
                FROM compact_sources
                """
            ).fetchone()
        )
        status_by_side = [
            dict(row)
            for row in conn.execute(
                """
                SELECT t.side, s.status, COUNT(*) AS token_jobs, COALESCE(SUM(s.rows_written), 0) AS rows_written
                FROM compact_price_fetch_status s
                JOIN compact_tokens t ON t.token_ref = s.token_ref
                GROUP BY t.side, s.status
                ORDER BY t.side, s.status
                """
            )
        ]
        status_by_city = []
        status_by_bucket_type = []
        status_by_bucket_side = []
        if paths.market_db.exists():
            status_by_city = [
                dict(row)
                for row in conn.execute(
                    """
                    SELECT m.normalized_city, s.status, COUNT(*) AS token_jobs, COALESCE(SUM(s.rows_written), 0) AS rows_written
                    FROM compact_price_fetch_status s
                    JOIN compact_tokens t ON t.token_ref = s.token_ref
                    JOIN marketdb.market_universe m ON m.universe_row_id = t.universe_row_id
                    GROUP BY m.normalized_city, s.status
                    ORDER BY m.normalized_city, s.status
                    LIMIT 200
                    """
                )
            ]
            status_by_bucket_type = [
                dict(row)
                for row in conn.execute(
                    """
                    SELECT m.bucket_type, s.status, COUNT(*) AS token_jobs, COALESCE(SUM(s.rows_written), 0) AS rows_written
                    FROM compact_price_fetch_status s
                    JOIN compact_tokens t ON t.token_ref = s.token_ref
                    JOIN marketdb.market_universe m ON m.universe_row_id = t.universe_row_id
                    GROUP BY m.bucket_type, s.status
                    ORDER BY m.bucket_type, s.status
                    """
                )
            ]
            status_by_bucket_side = [
                dict(row)
                for row in conn.execute(
                    """
                    SELECT m.bucket_type, t.side, s.status, COUNT(*) AS token_jobs, COALESCE(SUM(s.rows_written), 0) AS rows_written
                    FROM compact_price_fetch_status s
                    JOIN compact_tokens t ON t.token_ref = s.token_ref
                    JOIN marketdb.market_universe m ON m.universe_row_id = t.universe_row_id
                    GROUP BY m.bucket_type, t.side, s.status
                    ORDER BY m.bucket_type, t.side, s.status
                    """
                )
            ]
    size = paths.compact_price_db.stat().st_size if paths.compact_price_db.exists() else 0
    return {
        **dict(row),
        "fetch_status_counts": status_counts,
        "field_presence": field_presence,
        "source_audit": source_audit,
        "fetch_status_by_side": status_by_side,
        "fetch_status_by_city": status_by_city,
        "fetch_status_by_bucket_type": status_by_bucket_type,
        "fetch_status_by_bucket_side": status_by_bucket_side,
        "db_bytes": size,
        "db_gib": round(size / 1024**3, 4),
    }


def _validate_compact_samples(paths: PipelinePaths, jobs: list[Mapping[str, Any]]) -> dict[str, Any]:
    if not jobs:
        return {"sample_jobs": 0, "mismatches": 0, "checks": []}
    checks = []
    mismatches = 0
    with _connect(paths.price_db) as wide_conn, _connect(paths.compact_price_db) as compact_conn:
        wide_conn.row_factory = sqlite3.Row
        compact_conn.row_factory = sqlite3.Row
        token_refs = _compact_token_refs(compact_conn)
        for job in jobs:
            end_iso = datetime.fromtimestamp(int(job["end_ts"]), tz=timezone.utc).isoformat()
            wide = wide_conn.execute(
                """
                SELECT timestamp_utc, bid, ask, mid, last_price
                FROM token_quote_history
                WHERE token_id = ? AND timestamp_utc <= ?
                ORDER BY timestamp_utc DESC
                LIMIT 1
                """,
                (job["token_id"], end_iso),
            ).fetchone()
            compact = compact_conn.execute(
                """
                SELECT timestamp_utc, bid, ask, mid, last_price
                FROM compact_quote_history
                WHERE token_ref = ? AND timestamp_utc <= ?
                ORDER BY timestamp_utc DESC
                LIMIT 1
                """,
                (token_refs[str(job["token_id"])], end_iso),
            ).fetchone()
            wide_tuple = tuple(wide) if wide else None
            compact_tuple = tuple(compact) if compact else None
            ok = wide_tuple == compact_tuple
            if not ok:
                mismatches += 1
            checks.append(
                {
                    "token_id_prefix": str(job["token_id"])[:8],
                    "end_time": end_iso,
                    "matched": ok,
                    "wide": wide_tuple,
                    "compact": compact_tuple,
                }
            )
    return {"sample_jobs": len(jobs), "mismatches": mismatches, "checks": checks}


def _validate_compact_self_samples(paths: PipelinePaths, jobs: list[Mapping[str, Any]]) -> dict[str, Any]:
    if not jobs:
        return {"sample_jobs": 0, "checks": []}
    checks = []
    with _connect(paths.compact_price_db) as compact_conn:
        compact_conn.row_factory = sqlite3.Row
        token_refs = _compact_token_refs(compact_conn)
        for job in jobs:
            token_ref = token_refs[str(job["token_id"])]
            start_iso = datetime.fromtimestamp(int(job["start_ts"]), tz=timezone.utc).isoformat()
            end_iso = datetime.fromtimestamp(int(job["end_ts"]), tz=timezone.utc).isoformat()
            row = compact_conn.execute(
                """
                SELECT COUNT(*) AS rows, MIN(timestamp_utc) AS min_timestamp, MAX(timestamp_utc) AS max_timestamp
                FROM compact_quote_history
                WHERE token_ref = ? AND timestamp_utc >= ? AND timestamp_utc <= ?
                """,
                (token_ref, start_iso, end_iso),
            ).fetchone()
            checks.append(
                {
                    "token_id_prefix": str(job["token_id"])[:8],
                    "start_time": start_iso,
                    "end_time": end_iso,
                    "compact_rows": int(row["rows"] or 0),
                    "min_timestamp": row["min_timestamp"],
                    "max_timestamp": row["max_timestamp"],
                }
            )
    return {"sample_jobs": len(jobs), "checks": checks}


def _price_job_complete(conn: sqlite3.Connection, job: Mapping[str, Any]) -> bool:
    row = conn.execute(
        "SELECT status FROM price_fetch_status WHERE token_id = ? AND start_ts = ? AND end_ts = ?",
        (job["token_id"], job["start_ts"], job["end_ts"]),
    ).fetchone()
    return bool(row and row[0] == "complete")


def _load_quote_rows_for_job(
    job: Mapping[str, Any],
    *,
    client: Any,
    cache_only: bool,
    quote_cache_dirs: list[Path],
) -> list[dict[str, Any]]:
    start = datetime.fromtimestamp(int(job["start_ts"]), tz=timezone.utc)
    end = datetime.fromtimestamp(int(job["end_ts"]), tz=timezone.utc)
    paths: list[Path] = []
    if cache_only:
        for day in _date_range(start.date(), end.date()):
            for directory in quote_cache_dirs:
                paths.extend(sorted(directory.glob(f"{day.isoformat()}-*.parquet")))
    else:
        for day in _date_range(start.date(), end.date()):
            paths.append(
                client.download_parquet(
                    exchange="polymarket",
                    channel="quotes",
                    day=day,
                    params={"asset_id": job["token_id"]},
                )
            )
    return _quote_rows_from_paths(paths, token_id=str(job["token_id"]), start=start, end=end)


def _quote_rows_from_paths(paths: Iterable[Path], *, token_id: str, start: datetime, end: datetime) -> list[dict[str, Any]]:
    from weather_strategy.telonex import _load_parquet_records

    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for path in paths:
        if path.suffix != ".parquet" or not path.exists():
            continue
        try:
            records = _load_parquet_records(path)
        except Exception:
            continue
        payload_id = _stable_id("quote-payload", path, path.stat().st_size, int(path.stat().st_mtime))
        for record in records:
            if not _record_matches_token(record, token_id):
                continue
            timestamp = _record_timestamp(record)
            if timestamp is None:
                continue
            timestamp = _as_utc(timestamp)
            if timestamp < start or timestamp > end:
                continue
            bid = _first_numeric(record, ("best_bid_price", "bid_price", "bestBidPrice", "best_bid", "bestBid", "bid"))
            ask = _first_numeric(record, ("best_ask_price", "ask_price", "bestAskPrice", "best_ask", "bestAsk", "ask"))
            mid = _first_numeric(record, ("mid_price", "midPrice", "midpoint", "mid"))
            last = _first_numeric(record, ("last_price", "lastPrice", "price"))
            if mid is None and bid is not None and ask is not None:
                mid = (bid + ask) / 2
            spread = ask - bid if ask is not None and bid is not None else None
            key = (token_id, timestamp.isoformat())
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "timestamp_utc": timestamp.isoformat(),
                    "bid": bid,
                    "ask": ask,
                    "mid": mid,
                    "last_price": last,
                    "spread": spread,
                    "bid_size": _first_numeric(record, ("best_bid_size", "bid_size", "bestBidSize")),
                    "ask_size": _first_numeric(record, ("best_ask_size", "ask_size", "bestAskSize")),
                    "liquidity_depth": _first_numeric(record, ("liquidity", "depth", "size")),
                    "source_payload_id": payload_id,
                    "source_cache_path": str(path),
                    "payload_sha256": _sha256_file(path),
                    "data_quality_flags": _quote_flags(bid, ask, mid, last),
                }
            )
    rows.sort(key=lambda row: row["timestamp_utc"])
    return rows


def _insert_matching_quote_rows_from_path(
    conn: sqlite3.Connection,
    path: Path,
    jobs_by_token: Mapping[str, list[dict[str, Any]]],
    row_counts_by_job: dict[tuple[str, int, int], int],
) -> int:
    from weather_strategy.telonex import _load_parquet_records

    if path.suffix != ".parquet" or not path.exists():
        return 0
    records = _load_parquet_records(path)
    payload_id = _stable_id("quote-payload", path, path.stat().st_size, int(path.stat().st_mtime))
    payload_sha = _sha256_file(path)
    inserted = 0
    now = datetime.now(timezone.utc).isoformat()
    for record in records:
        token_id = _record_token_id(record)
        if token_id is None or token_id not in jobs_by_token:
            continue
        timestamp = _record_timestamp(record)
        if timestamp is None:
            continue
        timestamp = _as_utc(timestamp)
        quote_row = _quote_record_to_row(record, timestamp, path, payload_id, payload_sha)
        for job in jobs_by_token[token_id]:
            start = datetime.fromtimestamp(int(job["start_ts"]), tz=timezone.utc)
            end = datetime.fromtimestamp(int(job["end_ts"]), tz=timezone.utc)
            if timestamp < start or timestamp > end:
                continue
            _insert_quote_row(conn, job, quote_row, fetch_timestamp_utc=now)
            row_counts_by_job[_job_key(job)] += 1
            inserted += 1
    return inserted


def _quote_record_to_row(
    record: Mapping[str, Any],
    timestamp: datetime,
    path: Path,
    payload_id: str,
    payload_sha: Optional[str],
) -> dict[str, Any]:
    bid = _first_numeric(record, ("best_bid_price", "bid_price", "bestBidPrice", "best_bid", "bestBid", "bid"))
    ask = _first_numeric(record, ("best_ask_price", "ask_price", "bestAskPrice", "best_ask", "bestAsk", "ask"))
    mid = _first_numeric(record, ("mid_price", "midPrice", "midpoint", "mid"))
    last = _first_numeric(record, ("last_price", "lastPrice", "price"))
    if mid is None and bid is not None and ask is not None:
        mid = (bid + ask) / 2
    spread = ask - bid if ask is not None and bid is not None else None
    return {
        "timestamp_utc": timestamp.isoformat(),
        "bid": bid,
        "ask": ask,
        "mid": mid,
        "last_price": last,
        "spread": spread,
        "bid_size": _first_numeric(record, ("best_bid_size", "bid_size", "bestBidSize")),
        "ask_size": _first_numeric(record, ("best_ask_size", "ask_size", "bestAskSize")),
        "liquidity_depth": _first_numeric(record, ("liquidity", "depth", "size")),
        "source_payload_id": payload_id,
        "source_cache_path": str(path),
        "payload_sha256": payload_sha,
        "data_quality_flags": _quote_flags(bid, ask, mid, last),
    }


def _insert_quote_row(
    conn: sqlite3.Connection,
    job: Mapping[str, Any],
    row: Mapping[str, Any],
    *,
    fetch_timestamp_utc: str,
) -> None:
    conn.execute(
        """
        INSERT OR IGNORE INTO source_payloads(payload_id, source_name, cache_path, payload_sha256, fetched_at_utc, request_json)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            row["source_payload_id"],
            "telonex_quotes_parquet",
            row["source_cache_path"],
            row.get("payload_sha256"),
            None,
            json.dumps({"token_id": job["token_id"], "start_ts": job["start_ts"], "end_ts": job["end_ts"]}, sort_keys=True),
        ),
    )
    conn.execute(
        """
        INSERT OR REPLACE INTO token_quote_history(
            token_id, market_id, universe_row_id, side, timestamp_utc, bid, ask, mid, last_price, spread,
            bid_size, ask_size, liquidity_depth, source_payload_id, source_cache_path,
            fetch_timestamp_utc, data_quality_flags
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            job["token_id"],
            job.get("market_id"),
            job.get("universe_row_id"),
            job.get("side"),
            row["timestamp_utc"],
            row.get("bid"),
            row.get("ask"),
            row.get("mid"),
            row.get("last_price"),
            row.get("spread"),
            row.get("bid_size"),
            row.get("ask_size"),
            row.get("liquidity_depth"),
            row.get("source_payload_id"),
            row.get("source_cache_path"),
            fetch_timestamp_utc,
            row.get("data_quality_flags"),
        ),
    )


def _insert_quote_history(conn: sqlite3.Connection, job: Mapping[str, Any], rows: list[Mapping[str, Any]]) -> int:
    inserted = 0
    now = datetime.now(timezone.utc).isoformat()
    for row in rows:
        _insert_quote_row(conn, job, row, fetch_timestamp_utc=now)
        inserted += 1
    return inserted


def _upsert_price_status(conn: sqlite3.Connection, job: Mapping[str, Any], status: str, *, rows: int, error: Optional[str]) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO price_fetch_status(
            token_id, market_id, universe_row_id, start_ts, end_ts, status, rows_written, error, retryable, updated_at_utc
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            job["token_id"],
            job.get("market_id"),
            job.get("universe_row_id"),
            job.get("start_ts"),
            job.get("end_ts"),
            status,
            rows,
            error,
            _status_retryable(status),
            datetime.now(timezone.utc).isoformat(),
        ),
    )


def _status_retryable(status: str) -> int:
    return 0 if status == "complete" else 1


def _load_city_date_pairs(paths: PipelinePaths, *, start: date, end: date, limit: int) -> list[dict[str, Any]]:
    with closing(_connect(paths.market_db)) as conn:
        conn.row_factory = sqlite3.Row
        query = """
            SELECT normalized_city, country_region, latitude, longitude, timezone, market_date_local AS local_date
            FROM market_universe
            WHERE market_date_local >= ? AND market_date_local <= ?
            GROUP BY normalized_city, country_region, latitude, longitude, timezone, market_date_local
            ORDER BY local_date, normalized_city
        """
        params: list[Any] = [start.isoformat(), end.isoformat()]
        if limit > 0:
            query += " LIMIT ?"
            params.append(limit)
        return [dict(row) for row in conn.execute(query, params).fetchall()]


def _fetch_forecast_rows(pair: Mapping[str, Any], *, paths: PipelinePaths, cache_only: bool, mode: str) -> list[dict[str, Any]]:
    rows = []
    local_date = date.fromisoformat(str(pair["local_date"]))
    for decision_time in _decision_times_for_pair(pair):
        run_time = _forecast_run_time(decision_time)
        if mode == "single-runs":
            rows.extend(_fetch_single_run_forecast_rows(pair, local_date, decision_time, run_time, paths=paths, cache_only=cache_only))
        else:
            rows.append(_fetch_historical_forecast_row(pair, local_date, decision_time, paths=paths, cache_only=cache_only))
    return rows


def _fetch_single_run_forecast_rows(
    pair: Mapping[str, Any],
    local_date: date,
    decision_time: datetime,
    run_time: datetime,
    *,
    paths: PipelinePaths,
    cache_only: bool,
) -> list[dict[str, Any]]:
    url = "https://single-runs-api.open-meteo.com/v1/forecast"
    rows = []
    forecast_days = max(2, min(10, (local_date - run_time.date()).days + 2))
    for source_name, model_name in (
        ("single_run_best_match", None),
        ("single_run_gfs_global", "gfs_global"),
        ("single_run_ecmwf_ifs025", "ecmwf_ifs025"),
    ):
        params: dict[str, Any] = {
            "latitude": pair["latitude"],
            "longitude": pair["longitude"],
            "run": run_time.strftime("%Y-%m-%dT%H:%M"),
            "forecast_days": forecast_days,
            "daily": ",".join(
                (
                    "temperature_2m_max",
                    "precipitation_sum",
                    "precipitation_hours",
                    "wind_speed_10m_max",
                    "wind_gusts_10m_max",
                    "shortwave_radiation_sum",
                )
            ),
            "hourly": _hourly_weather_vars(),
            "temperature_unit": "fahrenheit",
            "wind_speed_unit": "mph",
            "precipitation_unit": "inch",
            "timezone": pair["timezone"],
        }
        if model_name:
            params["models"] = model_name
        payload, raw_path, payload_id = _cached_open_meteo_json(paths, "forecast", url, params, cache_only=cache_only)
        rows.append(_forecast_row(pair, local_date, decision_time, run_time, source_name, model_name, payload, url, params, raw_path, payload_id))
    return rows


def _fetch_historical_forecast_row(
    pair: Mapping[str, Any],
    local_date: date,
    decision_time: datetime,
    *,
    paths: PipelinePaths,
    cache_only: bool,
) -> dict[str, Any]:
    payload, raw_path, payload_id, url, params = _fetch_historical_forecast_payload(
        pair,
        local_date,
        local_date,
        paths=paths,
        cache_only=cache_only,
    )
    return _forecast_row(
        pair,
        local_date,
        decision_time,
        _forecast_run_time(decision_time),
        "historical_forecast_archive",
        None,
        payload,
        url,
        params,
        raw_path,
        payload_id,
        flags=["historical_forecast_endpoint_is_asof_approximation"],
    )


def _fetch_realized_weather_row(pair: Mapping[str, Any], *, paths: PipelinePaths, cache_only: bool) -> dict[str, Any]:
    local_date = date.fromisoformat(str(pair["local_date"]))
    payload, raw_path, payload_id, url, params = _fetch_realized_payload(
        pair,
        local_date,
        local_date,
        paths=paths,
        cache_only=cache_only,
    )
    return _realized_row_from_payload(pair, local_date, payload, url, params, raw_path, payload_id)


def _fetch_historical_forecast_payload(
    pair: Mapping[str, Any],
    start_date: date,
    end_date: date,
    *,
    paths: PipelinePaths,
    cache_only: bool,
) -> tuple[dict[str, Any], Path, str, str, dict[str, Any]]:
    url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
    params = {
        "latitude": pair["latitude"],
        "longitude": pair["longitude"],
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "daily": "temperature_2m_max,precipitation_sum,wind_speed_10m_max,wind_gusts_10m_max,shortwave_radiation_sum",
        "hourly": _hourly_weather_vars(),
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "precipitation_unit": "inch",
        "timezone": pair["timezone"],
    }
    payload, raw_path, payload_id = _cached_open_meteo_json(paths, "forecast", url, params, cache_only=cache_only)
    return payload, raw_path, payload_id, url, params


def _fetch_realized_payload(
    pair: Mapping[str, Any],
    start_date: date,
    end_date: date,
    *,
    paths: PipelinePaths,
    cache_only: bool,
) -> tuple[dict[str, Any], Path, str, str, dict[str, Any]]:
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": pair["latitude"],
        "longitude": pair["longitude"],
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "daily": "temperature_2m_max,precipitation_sum,wind_speed_10m_max,wind_gusts_10m_max,shortwave_radiation_sum",
        "hourly": _hourly_weather_vars(),
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "precipitation_unit": "inch",
        "timezone": pair["timezone"],
    }
    payload, raw_path, payload_id = _cached_open_meteo_json(paths, "realized", url, params, cache_only=cache_only)
    return payload, raw_path, payload_id, url, params


def _realized_row_from_payload(
    pair: Mapping[str, Any],
    local_date: date,
    payload: Mapping[str, Any],
    url: str,
    params: Mapping[str, Any],
    raw_path: Path,
    payload_id: str,
) -> dict[str, Any]:
    features = _weather_features(payload, local_date)
    return {
        "realized_id": _stable_id("realized", pair["normalized_city"], local_date.isoformat(), "open_meteo_archive"),
        "normalized_city": pair["normalized_city"],
        "country_region": pair["country_region"],
        "latitude": pair["latitude"],
        "longitude": pair["longitude"],
        "timezone": pair["timezone"],
        "local_date": local_date.isoformat(),
        "temperature_2m_max_f": features.get("temperature_2m_max"),
        "hourly_temperature_2m_max_f": features.get("hourly_temperature_2m_max"),
        "apparent_temperature_max_f": features.get("hourly_apparent_temperature_max"),
        "humidity_mean": features.get("relative_humidity_2m_mean"),
        "dew_point_max_f": features.get("dew_point_2m_max"),
        "precipitation_sum_in": features.get("precipitation_sum"),
        "cloud_cover_mean": features.get("cloud_cover_mean"),
        "wind_speed_10m_max_mph": features.get("wind_speed_10m_max") or features.get("hourly_wind_speed_10m_max"),
        "source_name": "open_meteo_archive_daily",
        "feature_json": json.dumps(features, sort_keys=True),
        "request_url": url,
        "request_params_json": json.dumps(params, sort_keys=True),
        "source_payload_id": payload_id,
        "raw_cache_path": str(raw_path),
        "fetch_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "data_quality_flags": json.dumps(_weather_flags(features), sort_keys=True),
    }


def _forecast_row(
    pair: Mapping[str, Any],
    local_date: date,
    decision_time: datetime,
    run_time: datetime,
    source_name: str,
    model_name: Optional[str],
    payload: Mapping[str, Any],
    url: str,
    params: Mapping[str, Any],
    raw_path: Path,
    payload_id: str,
    flags: Optional[list[str]] = None,
) -> dict[str, Any]:
    features = _weather_features(payload, local_date)
    row_id = _stable_id("forecast", pair["normalized_city"], local_date.isoformat(), decision_time.isoformat(), source_name, model_name)
    combined_flags = list(flags or []) + _weather_flags(features)
    return {
        "forecast_id": row_id,
        "normalized_city": pair["normalized_city"],
        "country_region": pair["country_region"],
        "latitude": pair["latitude"],
        "longitude": pair["longitude"],
        "timezone": pair["timezone"],
        "local_date": local_date.isoformat(),
        "decision_time_utc": decision_time.isoformat(),
        "forecast_run_time_utc": run_time.isoformat(),
        "source_name": source_name,
        "model_name": model_name,
        "temperature_2m_max_f": features.get("temperature_2m_max"),
        "hourly_temperature_2m_max_f": features.get("hourly_temperature_2m_max"),
        "apparent_temperature_max_f": features.get("hourly_apparent_temperature_max"),
        "humidity_mean": features.get("relative_humidity_2m_mean"),
        "dew_point_max_f": features.get("dew_point_2m_max"),
        "precipitation_sum_in": features.get("precipitation_sum"),
        "cloud_cover_mean": features.get("cloud_cover_mean"),
        "wind_speed_10m_max_mph": features.get("wind_speed_10m_max") or features.get("hourly_wind_speed_10m_max"),
        "feature_json": json.dumps(features, sort_keys=True),
        "request_url": url,
        "request_params_json": json.dumps(params, sort_keys=True),
        "source_payload_id": payload_id,
        "raw_cache_path": str(raw_path),
        "fetch_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "data_quality_flags": json.dumps(combined_flags, sort_keys=True),
    }


def _insert_forecast_row(conn: sqlite3.Connection, row: Mapping[str, Any]) -> None:
    _insert_weather_source_payload(conn, row, "open_meteo_forecast")
    columns = [
        "forecast_id",
        "normalized_city",
        "country_region",
        "latitude",
        "longitude",
        "timezone",
        "local_date",
        "decision_time_utc",
        "forecast_run_time_utc",
        "source_name",
        "model_name",
        "temperature_2m_max_f",
        "hourly_temperature_2m_max_f",
        "apparent_temperature_max_f",
        "humidity_mean",
        "dew_point_max_f",
        "precipitation_sum_in",
        "cloud_cover_mean",
        "wind_speed_10m_max_mph",
        "feature_json",
        "request_url",
        "request_params_json",
        "source_payload_id",
        "raw_cache_path",
        "fetch_timestamp_utc",
        "data_quality_flags",
    ]
    conn.execute(
        f"INSERT OR REPLACE INTO forecast_snapshots({', '.join(columns)}) VALUES ({', '.join(['?'] * len(columns))})",
        [row.get(column) for column in columns],
    )


def _insert_realized_row(conn: sqlite3.Connection, row: Mapping[str, Any]) -> None:
    _insert_weather_source_payload(conn, row, "open_meteo_realized")
    columns = [
        "realized_id",
        "normalized_city",
        "country_region",
        "latitude",
        "longitude",
        "timezone",
        "local_date",
        "temperature_2m_max_f",
        "hourly_temperature_2m_max_f",
        "apparent_temperature_max_f",
        "humidity_mean",
        "dew_point_max_f",
        "precipitation_sum_in",
        "cloud_cover_mean",
        "wind_speed_10m_max_mph",
        "source_name",
        "feature_json",
        "request_url",
        "request_params_json",
        "source_payload_id",
        "raw_cache_path",
        "fetch_timestamp_utc",
        "data_quality_flags",
    ]
    conn.execute(
        f"INSERT OR REPLACE INTO realized_daily_weather({', '.join(columns)}) VALUES ({', '.join(['?'] * len(columns))})",
        [row.get(column) for column in columns],
    )


def _insert_weather_source_payload(conn: sqlite3.Connection, row: Mapping[str, Any], source_name: str) -> None:
    path = Path(str(row["raw_cache_path"]))
    conn.execute(
        """
        INSERT OR REPLACE INTO source_payloads(payload_id, source_name, cache_path, payload_sha256, fetched_at_utc, request_json)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            row["source_payload_id"],
            source_name,
            str(path),
            _sha256_file(path) if path.exists() else None,
            row.get("fetch_timestamp_utc"),
            row.get("request_params_json"),
        ),
    )


def _weather_pair_complete(conn: sqlite3.Connection, pair: Mapping[str, Any]) -> bool:
    row = conn.execute(
        "SELECT status FROM weather_fetch_status WHERE normalized_city = ? AND local_date = ?",
        (pair["normalized_city"], pair["local_date"]),
    ).fetchone()
    return bool(row and row[0] == "complete")


def _upsert_weather_status(conn: sqlite3.Connection, pair: Mapping[str, Any], status: str, *, error: Optional[str]) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO weather_fetch_status(normalized_city, local_date, status, error, retryable, updated_at_utc)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            pair["normalized_city"],
            pair["local_date"],
            status,
            error,
            _status_retryable(status),
            datetime.now(timezone.utc).isoformat(),
        ),
    )


def _cached_open_meteo_json(
    paths: PipelinePaths,
    namespace: str,
    url: str,
    params: Mapping[str, Any],
    *,
    cache_only: bool,
) -> tuple[dict[str, Any], Path, str]:
    raw_dir = paths.open_meteo_raw / namespace
    raw_dir.mkdir(parents=True, exist_ok=True)
    key = _stable_id("open-meteo", url, params)
    path = raw_dir / f"{key}.json"
    meta_path = raw_dir / f"{key}.meta.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8")), path, key
    if cache_only:
        raise FileNotFoundError(f"Open-Meteo cache miss for {url} {dict(params)}")
    full_url = f"{url}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(full_url, headers={"User-Agent": "DailyWeatherDataPipeline/0.1"})
    with urllib.request.urlopen(request, timeout=45) as response:  # noqa: S310 - official public weather endpoint
        payload = response.read()
    path.write_bytes(payload)
    meta_path.write_text(
        json.dumps(
            {
                "url": url,
                "params": dict(params),
                "full_url": full_url,
                "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return json.loads(payload.decode("utf-8")), path, key


def _decision_times_for_market(row: Mapping[str, Any]) -> list[datetime]:
    if not row.get("market_date_local") or not row.get("timezone"):
        return []
    target = date.fromisoformat(str(row["market_date_local"]))
    zone = _zone(str(row["timezone"]))
    created = _parse_datetime(row.get("start_at_utc") or row.get("created_at_utc"))
    closed = _parse_datetime(row.get("close_time_utc"))
    candidates = []
    current = target - timedelta(days=3)
    while current <= target + timedelta(days=1):
        for hour in ENTRY_HOURS_UTC:
            decision = datetime.combine(current, datetime_time(hour, 0), tzinfo=timezone.utc)
            local_date = decision.astimezone(zone).date()
            lead_days = (target - local_date).days
            target_day_maintenance = local_date == target
            if 1 <= lead_days <= 2 or target_day_maintenance:
                if created is not None and decision < created:
                    continue
                if closed is not None and decision >= closed:
                    continue
                candidates.append(decision)
        current += timedelta(days=1)
    return sorted(set(candidates))


def _decision_times_for_pair(pair: Mapping[str, Any]) -> list[datetime]:
    target = date.fromisoformat(str(pair["local_date"]))
    zone = _zone(str(pair["timezone"]))
    decisions = []
    current = target - timedelta(days=2)
    while current <= target:
        for hour in ENTRY_HOURS_UTC:
            decision = datetime.combine(current, datetime_time(hour, 0), tzinfo=timezone.utc)
            lead_days = (target - decision.astimezone(zone).date()).days
            if 1 <= lead_days <= 2 or decision.astimezone(zone).date() == target:
                decisions.append(decision)
        current += timedelta(days=1)
    return sorted(set(decisions))


def _quote_snapshot(
    conn: sqlite3.Connection,
    token_id: str,
    decision_time: datetime,
    *,
    max_staleness_minutes: int,
) -> Optional[sqlite3.Row]:
    if not token_id:
        return None
    row = conn.execute(
        """
        SELECT * FROM token_quote_history
        WHERE token_id = ? AND timestamp_utc <= ?
        ORDER BY timestamp_utc DESC
        LIMIT 1
        """,
        (token_id, decision_time.isoformat()),
    ).fetchone()
    if row is None:
        return None
    timestamp = _parse_datetime(row["timestamp_utc"])
    if timestamp is None:
        return None
    if (decision_time - timestamp).total_seconds() > max_staleness_minutes * 60:
        return None
    return row


def _compact_quote_snapshot(
    conn: sqlite3.Connection,
    token_id: str,
    decision_time: datetime,
    *,
    max_staleness_minutes: int,
) -> Optional[sqlite3.Row]:
    if not token_id:
        return None
    row = conn.execute(
        """
        SELECT
          q.timestamp_utc,
          q.bid,
          q.ask,
          q.mid,
          q.last_price,
          q.spread,
          q.bid_size,
          q.ask_size,
          q.liquidity_depth,
          q.data_quality_flags,
          s.source_payload_id,
          s.source_cache_path
        FROM compact_tokens t
        JOIN compact_quote_history q ON q.token_ref = t.token_ref
        LEFT JOIN compact_sources s ON s.source_ref = q.source_ref
        WHERE t.token_id = ? AND q.timestamp_utc <= ?
        ORDER BY q.timestamp_utc DESC
        LIMIT 1
        """,
        (token_id, decision_time.isoformat()),
    ).fetchone()
    if row is None:
        return None
    timestamp = _parse_datetime(row["timestamp_utc"])
    if timestamp is None:
        return None
    if (decision_time - timestamp).total_seconds() > max_staleness_minutes * 60:
        return None
    return row


def _forecast_snapshot(conn: sqlite3.Connection, market: Mapping[str, Any], decision_time: datetime) -> Optional[sqlite3.Row]:
    row = conn.execute(
        """
        SELECT * FROM forecast_snapshots
        WHERE normalized_city = ?
          AND local_date = ?
          AND decision_time_utc = ?
          AND forecast_run_time_utc <= ?
        ORDER BY forecast_run_time_utc DESC, source_name
        LIMIT 1
        """,
        (
            market["normalized_city"],
            market["market_date_local"],
            decision_time.isoformat(),
            decision_time.isoformat(),
        ),
    ).fetchone()
    return row


def _realized_snapshot(conn: sqlite3.Connection, market: Mapping[str, Any]) -> Optional[sqlite3.Row]:
    return conn.execute(
        """
        SELECT * FROM realized_daily_weather
        WHERE normalized_city = ? AND local_date = ?
        ORDER BY source_name
        LIMIT 1
        """,
        (market["normalized_city"], market["market_date_local"]),
    ).fetchone()


def _populate_observed_high_snapshots(
    paths: PipelinePaths,
    decision_conn: sqlite3.Connection,
    *,
    start: date,
    end: date,
) -> dict[str, Any]:
    _ensure_decision_schema(decision_conn)
    previous_row_factory = decision_conn.row_factory
    decision_conn.row_factory = sqlite3.Row
    try:
        decision_rows = decision_conn.execute(
            """
            SELECT normalized_city, market_date_local, timezone, decision_time_utc
            FROM decision_snapshots
            WHERE market_date_local >= ? AND market_date_local <= ?
            GROUP BY normalized_city, market_date_local, timezone, decision_time_utc
            ORDER BY normalized_city, market_date_local, decision_time_utc
            """,
            (start.isoformat(), end.isoformat()),
        ).fetchall()
    finally:
        decision_conn.row_factory = previous_row_factory
    with _connect(paths.weather_db) as weather_conn:
        weather_conn.row_factory = sqlite3.Row
        realized_rows = {
            (row["normalized_city"], row["local_date"]): row
            for row in weather_conn.execute(
                """
                SELECT normalized_city, local_date, timezone, source_name, source_payload_id, raw_cache_path
                FROM realized_daily_weather
                WHERE local_date >= ? AND local_date <= ?
                """,
                (start.isoformat(), end.isoformat()),
            )
        }

    series_cache: dict[tuple[str, str, str], tuple[list[tuple[datetime, float]], Optional[str]]] = {}
    inserted = 0
    rows_with_observed_high = 0
    missing_reasons: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    leakage_violations = 0
    final_leak_violations = 0
    decision_conn.execute(
        "DELETE FROM observed_high_snapshots WHERE market_date_local >= ? AND market_date_local <= ?",
        (start.isoformat(), end.isoformat()),
    )
    for row in decision_rows:
        decision_time = _parse_datetime(row["decision_time_utc"])
        city = row["normalized_city"]
        local_date = row["market_date_local"]
        zone_name = row["timezone"] or "UTC"
        realized = realized_rows.get((city, local_date))
        snapshot = _observed_high_snapshot(
            city=city,
            local_date=local_date,
            zone_name=zone_name,
            decision_time=decision_time,
            realized=realized,
            series_cache=series_cache,
        )
        _insert_observed_high_snapshot(decision_conn, snapshot)
        inserted += 1
        if snapshot["observed_high_so_far_f"] is not None:
            rows_with_observed_high += 1
        reason = snapshot["observed_high_missing_reason"]
        if reason:
            missing_reasons[reason] = missing_reasons.get(reason, 0) + 1
        source = snapshot["observed_high_source"] or "missing"
        source_counts[source] = source_counts.get(source, 0) + 1
        observed_at = _parse_datetime(snapshot["observed_high_observed_at_utc"])
        if decision_time and observed_at and observed_at > decision_time:
            leakage_violations += 1
        flags = set(_parse_jsonish(snapshot["observed_high_data_quality_flags"]))
        if {"final_daily_aggregate_used", "derived_from_final_daily_aggregate"}.intersection(flags):
            final_leak_violations += 1
    return {
        "unique_city_date_decisions": len(decision_rows),
        "rows_written": inserted,
        "rows_with_observed_high": rows_with_observed_high,
        "rows_missing_observed_high": inserted - rows_with_observed_high,
        "missing_reasons": dict(sorted(missing_reasons.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "future_observed_high_violations": leakage_violations,
        "final_daily_aggregate_leak_violations": final_leak_violations,
    }


def _observed_high_snapshot(
    *,
    city: str,
    local_date: str,
    zone_name: str,
    decision_time: Optional[datetime],
    realized: Optional[sqlite3.Row],
    series_cache: dict[tuple[str, str, str], tuple[list[tuple[datetime, float]], Optional[str]]],
) -> dict[str, Any]:
    flags = ["historical_archive_hourly_point_in_time_reconstruction"]
    missing_reason = None
    source_payload_id = realized["source_payload_id"] if realized else None
    raw_cache_path = realized["raw_cache_path"] if realized else None
    observations: list[tuple[datetime, float]] = []
    series_error: Optional[str] = None
    if decision_time is None:
        missing_reason = "invalid_decision_time"
    elif realized is None:
        missing_reason = "missing_realized_weather_row"
    elif not raw_cache_path:
        missing_reason = "missing_realized_raw_cache_path"
    else:
        cache_key = (str(raw_cache_path), local_date, zone_name)
        if cache_key not in series_cache:
            series_cache[cache_key] = _hourly_temperature_series(raw_cache_path, local_date, zone_name)
        observations, series_error = series_cache[cache_key]
        if series_error:
            missing_reason = series_error
        elif not observations:
            missing_reason = "missing_hourly_temperature_observations"
    eligible: list[tuple[datetime, float]] = []
    if not missing_reason and decision_time is not None:
        eligible = [(observed_at, value) for observed_at, value in observations if observed_at <= decision_time]
        if not eligible:
            missing_reason = "no_observations_at_or_before_decision"
    observed_high = None
    observed_at_utc = None
    minutes_before = None
    is_final = 0
    decision_local_hour = None
    if decision_time is not None:
        try:
            decision_local_hour = decision_time.astimezone(ZoneInfo(zone_name)).hour
        except ZoneInfoNotFoundError:
            decision_local_hour = decision_time.hour
    if eligible and decision_time is not None:
        observed_at, observed_high = max(eligible, key=lambda item: (item[1], item[0]))
        observed_at_utc = observed_at.isoformat()
        minutes_before = (decision_time - observed_at).total_seconds() / 60.0
        if len(eligible) == len(observations):
            is_final = 1
            flags.append("full_local_day_observed_by_decision")
    if missing_reason:
        flags.append(f"missing:{missing_reason}")
        if missing_reason in {
            "missing_realized_weather_row",
            "missing_realized_raw_cache_path",
            "missing_hourly_temperature_observations",
            "raw_cache_path_not_found",
            "raw_cache_json_decode_error",
            "raw_cache_missing_hourly_time_or_temperature",
        }:
            flags.append("final_daily_aggregate_not_used")
    return {
        "observed_high_id": _stable_id("observed_high", city, local_date, decision_time.isoformat() if decision_time else ""),
        "normalized_city": city,
        "market_date_local": local_date,
        "timezone": zone_name,
        "decision_time_utc": decision_time.isoformat() if decision_time else "",
        "decision_local_hour": decision_local_hour,
        "observed_high_so_far_f": observed_high,
        "observed_high_observed_at_utc": observed_at_utc,
        "observed_high_source": "open_meteo_archive_hourly" if observations else None,
        "observed_high_sample_count": len(eligible),
        "observed_high_total_sample_count": len(observations),
        "observed_high_is_final_at_decision": is_final,
        "observed_high_minutes_before_decision": minutes_before,
        "observed_high_data_quality_flags": json.dumps(flags, sort_keys=True),
        "observed_high_missing_reason": missing_reason,
        "observed_high_source_payload_id": source_payload_id,
        "observed_high_raw_cache_path": raw_cache_path,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def _hourly_temperature_series(
    raw_cache_path: str,
    local_date: str,
    zone_name: str,
) -> tuple[list[tuple[datetime, float]], Optional[str]]:
    path = Path(raw_cache_path)
    if not path.exists():
        return [], "raw_cache_path_not_found"
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError:
        return [], "raw_cache_json_decode_error"
    hourly = payload.get("hourly") if isinstance(payload.get("hourly"), Mapping) else {}
    times = hourly.get("time") or []
    temps = hourly.get("temperature_2m") or []
    if not isinstance(times, list) or not isinstance(temps, list):
        return [], "raw_cache_missing_hourly_time_or_temperature"
    try:
        zone = ZoneInfo(zone_name)
    except ZoneInfoNotFoundError:
        zone = timezone.utc
    observations: list[tuple[datetime, float]] = []
    for index, raw_time in enumerate(times):
        if index >= len(temps) or not str(raw_time).startswith(local_date):
            continue
        value = _to_float(temps[index])
        if value is None:
            continue
        local_time = datetime.fromisoformat(str(raw_time))
        if local_time.tzinfo is None:
            local_time = local_time.replace(tzinfo=zone)
        observed_at_utc = local_time.astimezone(timezone.utc)
        observations.append((observed_at_utc, value))
    observations.sort(key=lambda item: item[0])
    return observations, None


def _insert_observed_high_snapshot(conn: sqlite3.Connection, row: Mapping[str, Any]) -> None:
    columns = [
        "observed_high_id",
        "normalized_city",
        "market_date_local",
        "timezone",
        "decision_time_utc",
        "decision_local_hour",
        "observed_high_so_far_f",
        "observed_high_observed_at_utc",
        "observed_high_source",
        "observed_high_sample_count",
        "observed_high_total_sample_count",
        "observed_high_is_final_at_decision",
        "observed_high_minutes_before_decision",
        "observed_high_data_quality_flags",
        "observed_high_missing_reason",
        "observed_high_source_payload_id",
        "observed_high_raw_cache_path",
        "created_at_utc",
    ]
    conn.execute(
        f"INSERT OR REPLACE INTO observed_high_snapshots({', '.join(columns)}) VALUES ({', '.join(['?'] * len(columns))})",
        [row.get(column) for column in columns],
    )


def _clear_decision_snapshots_for_rebuild(conn: sqlite3.Connection, *, start: date, end: date) -> dict[str, int]:
    params = (start.isoformat(), end.isoformat())
    before_total = int(conn.execute("SELECT COUNT(*) FROM decision_snapshots").fetchone()[0])
    before_target = int(
        conn.execute(
            "SELECT COUNT(*) FROM decision_snapshots WHERE market_date_local >= ? AND market_date_local <= ?",
            params,
        ).fetchone()[0]
    )
    conn.execute("DELETE FROM decision_snapshots")
    return {
        "total_rows": before_total,
        "target_window_rows": before_target,
        "out_of_target_window_rows": max(0, before_total - before_target),
    }


def _record_decision_artifact_scope(
    conn: sqlite3.Connection,
    *,
    start: date,
    end: date,
    price_store: str,
    max_staleness_minutes: int,
    row_count: int,
) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO decision_artifact_scope(
            scope_id, target_start_date, target_end_date, price_store,
            max_staleness_minutes, built_at_utc, row_count
        )
        VALUES ('current', ?, ?, ?, ?, ?, ?)
        """,
        (
            start.isoformat(),
            end.isoformat(),
            price_store,
            max_staleness_minutes,
            datetime.now(timezone.utc).isoformat(),
            row_count,
        ),
    )


def _row_get(row: Optional[Mapping[str, Any]], key: str, default: Any = None) -> Any:
    if row is None:
        return default
    if isinstance(row, sqlite3.Row):
        return row[key] if key in row.keys() else default
    return row.get(key, default)


def _decision_row(
    market: Mapping[str, Any],
    decision_time: datetime,
    quote: Optional[sqlite3.Row],
    forecast: Optional[sqlite3.Row],
    realized: Optional[sqlite3.Row],
    *,
    side: str,
    token_id: str,
    price_store: str,
) -> dict[str, Any]:
    flags = []
    parser_warnings = _parse_jsonish(market.get("parse_warnings_json"))
    flags.extend(f"parser:{warning}" for warning in parser_warnings if warning)
    quote_start = _parse_date(market.get("telonex_quotes_from"))
    quote_end = _parse_date(market.get("telonex_quotes_to"))
    if quote_start is None or quote_end is None:
        flags.append("missing_telonex_quote_availability")
    else:
        quote_start_dt = datetime.combine(quote_start, datetime_time(0, 0), tzinfo=timezone.utc)
        quote_end_dt = datetime.combine(quote_end + timedelta(days=1), datetime_time(0, 0), tzinfo=timezone.utc)
        if decision_time < quote_start_dt:
            flags.append("decision_before_telonex_quote_availability")
        if decision_time >= quote_end_dt:
            flags.append("decision_after_telonex_quote_availability")
    if quote is None:
        flags.append("missing_quote_snapshot")
    if forecast is None:
        flags.append("missing_forecast_snapshot")
    if realized is None:
        flags.append("missing_realized_label")
    stale = None
    if quote is not None:
        timestamp = _parse_datetime(_row_get(quote, "timestamp_utc"))
        if timestamp:
            stale = (decision_time - timestamp).total_seconds()
            if stale is not None and stale < 0:
                flags.append("future_quote_violation")
    market_local_date = _parse_date(market.get("market_date_local"))
    decision_hour_utc = decision_time.hour
    lead_days = None
    if market_local_date is not None:
        zone_name = market.get("timezone") or "UTC"
        try:
            zone = ZoneInfo(str(zone_name))
        except Exception:
            zone = timezone.utc
        decision_local_date = decision_time.astimezone(zone).date()
        lead_days = (market_local_date - decision_local_date).days
    return {
        "decision_id": _stable_id("decision", market["universe_row_id"], side, token_id, decision_time.isoformat()),
        "universe_row_id": market["universe_row_id"],
        "market_id": market.get("market_id"),
        "slug": market.get("slug"),
        "question": market.get("question"),
        "normalized_city": market.get("normalized_city"),
        "country_region": market.get("country_region"),
        "timezone": market.get("timezone"),
        "market_date_local": market.get("market_date_local"),
        "bucket_label": market.get("bucket_label"),
        "bucket_type": market.get("bucket_type"),
        "threshold_low_f": market.get("threshold_low_f"),
        "threshold_high_f": market.get("threshold_high_f"),
        "side": side,
        "yes_token_id": market.get("yes_token_id"),
        "token_id": token_id,
        "no_token_id": market.get("no_token_id"),
        "decision_time_utc": decision_time.isoformat(),
        "decision_hour_utc": decision_hour_utc,
        "lead_days": lead_days,
        "quote_store": price_store,
        "quote_timestamp_utc": _row_get(quote, "timestamp_utc"),
        "bid": _row_get(quote, "bid"),
        "ask": _row_get(quote, "ask"),
        "mid": _row_get(quote, "mid"),
        "last_price": _row_get(quote, "last_price"),
        "spread": _row_get(quote, "spread"),
        "bid_size": _row_get(quote, "bid_size"),
        "ask_size": _row_get(quote, "ask_size"),
        "liquidity_depth": _row_get(quote, "liquidity_depth"),
        "quote_source_payload_id": _row_get(quote, "source_payload_id"),
        "quote_source_cache_path": _row_get(quote, "source_cache_path"),
        "quote_data_quality_flags": _row_get(quote, "data_quality_flags"),
        "price_stale_seconds": stale,
        "forecast_run_time_utc": forecast["forecast_run_time_utc"] if forecast else None,
        "forecast_source_name": forecast["source_name"] if forecast else None,
        "forecast_temperature_2m_max_f": forecast["temperature_2m_max_f"] if forecast else None,
        "forecast_features_json": forecast["feature_json"] if forecast else None,
        "realized_temperature_2m_max_f": realized["temperature_2m_max_f"] if realized else None,
        "resolved_outcome": market.get("resolved_outcome"),
        "settlement_source": market.get("settlement_source"),
        "data_quality_flags": json.dumps(flags, sort_keys=True),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def _insert_decision_snapshot(conn: sqlite3.Connection, row: Mapping[str, Any]) -> None:
    columns = [
        "decision_id",
        "universe_row_id",
        "market_id",
        "slug",
        "question",
        "normalized_city",
        "country_region",
        "timezone",
        "market_date_local",
        "bucket_label",
        "bucket_type",
        "threshold_low_f",
        "threshold_high_f",
        "side",
        "yes_token_id",
        "token_id",
        "no_token_id",
        "decision_time_utc",
        "decision_hour_utc",
        "lead_days",
        "quote_store",
        "quote_timestamp_utc",
        "bid",
        "ask",
        "mid",
        "last_price",
        "spread",
        "bid_size",
        "ask_size",
        "liquidity_depth",
        "quote_source_payload_id",
        "quote_source_cache_path",
        "quote_data_quality_flags",
        "price_stale_seconds",
        "forecast_run_time_utc",
        "forecast_source_name",
        "forecast_temperature_2m_max_f",
        "forecast_features_json",
        "realized_temperature_2m_max_f",
        "resolved_outcome",
        "settlement_source",
        "data_quality_flags",
        "created_at_utc",
    ]
    conn.execute(
        f"INSERT OR REPLACE INTO decision_snapshots({', '.join(columns)}) VALUES ({', '.join(['?'] * len(columns))})",
        [row.get(column) for column in columns],
    )


def _quality_summary(paths: PipelinePaths, *, start: date, end: date) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_start_date": start.isoformat(),
        "target_end_date": end.isoformat(),
        "paths": _paths_json(paths),
    }
    with _connect(paths.market_db) as conn:
        conn.row_factory = sqlite3.Row
        _ensure_market_schema(conn)
        summary["markets"] = _market_quality(conn, start=start, end=end)
    with _connect(paths.price_db) as conn:
        conn.row_factory = sqlite3.Row
        _ensure_price_schema(conn)
        summary["prices"] = _price_quality(conn)
    if paths.compact_price_db.exists():
        with _connect(paths.compact_price_db) as conn:
            conn.row_factory = sqlite3.Row
            _ensure_compact_price_schema(conn)
            summary["compact_price_status"] = _compact_price_status_quality(conn, paths=paths)
    with _connect(paths.weather_db) as conn:
        conn.row_factory = sqlite3.Row
        _ensure_weather_schema(conn)
        summary["weather"] = _weather_quality(conn)
    with _connect(paths.decision_db) as conn:
        conn.row_factory = sqlite3.Row
        _ensure_decision_schema(conn)
        summary["decision_artifact"] = _decision_quality(conn, start=start, end=end)
    summary["usable_market_count_for_backtesting"] = _usable_market_count(paths, start=start, end=end)
    summary["supported_backtest_window"] = _supported_window_summary(summary)
    summary["known_gaps"] = _known_gaps(summary)
    summary["backtest_ready"] = not summary["known_gaps"] and int(summary.get("usable_market_count_for_backtesting") or 0) > 0
    return summary


def _market_quality(conn: sqlite3.Connection, *, start: date, end: date) -> dict[str, Any]:
    row = conn.execute(
        """
        SELECT
          COUNT(*) AS rows,
          COUNT(DISTINCT market_id) AS markets,
          COUNT(DISTINCT normalized_city || '|' || market_date_local) AS city_date_pairs,
          SUM(CASE WHEN parser_confidence >= 0.9 THEN 1 ELSE 0 END) AS parsed_successfully,
          SUM(CASE WHEN latitude IS NOT NULL AND longitude IS NOT NULL AND timezone IS NOT NULL THEN 1 ELSE 0 END) AS with_coordinates_timezone,
          SUM(CASE WHEN yes_token_id IS NOT NULL AND yes_token_id != '' THEN 1 ELSE 0 END) AS with_yes_token,
          SUM(CASE WHEN no_token_id IS NOT NULL AND no_token_id != '' THEN 1 ELSE 0 END) AS with_no_token,
          SUM(CASE WHEN resolved_outcome IS NOT NULL THEN 1 ELSE 0 END) AS with_resolved_outcome,
          MIN(market_date_local) AS min_market_date,
          MAX(market_date_local) AS max_market_date,
          MIN(NULLIF(telonex_quotes_from, '')) AS min_quotes_from,
          MAX(NULLIF(telonex_quotes_to, '')) AS max_quotes_to
        FROM market_universe
        WHERE market_date_local >= ? AND market_date_local <= ?
        """,
        (start.isoformat(), end.isoformat()),
    ).fetchone()
    warnings = conn.execute(
        """
        SELECT warning, COUNT(*) AS count
        FROM market_parse_warnings
        GROUP BY warning
        ORDER BY count DESC, warning
        LIMIT 25
        """
    ).fetchall()
    ambiguous = conn.execute(
        """
        SELECT normalized_city, COUNT(*) AS count
        FROM market_universe
        WHERE parser_confidence < 0.9
        GROUP BY normalized_city
        ORDER BY count DESC, normalized_city
        LIMIT 25
        """
    ).fetchall()
    result = dict(row)
    decision_rows = conn.execute(
        """
        SELECT *
        FROM market_universe
        WHERE market_date_local >= ? AND market_date_local <= ?
        """,
        (start.isoformat(), end.isoformat()),
    ).fetchall()
    markets_with_live_decision_times = 0
    expected_decision_rows = 0
    for market_row in decision_rows:
        decision_times = _decision_times_for_market(dict(market_row))
        if not decision_times:
            continue
        markets_with_live_decision_times += 1
        side_count = int(bool(market_row["yes_token_id"])) + int(bool(market_row["no_token_id"]))
        expected_decision_rows += len(decision_times) * side_count
    result["markets_with_live_decision_times"] = markets_with_live_decision_times
    result["expected_decision_rows"] = expected_decision_rows
    result["parse_warnings_top"] = [dict(item) for item in warnings]
    result["ambiguous_city_rows_top"] = [dict(item) for item in ambiguous]
    return result


def _price_quality(conn: sqlite3.Connection) -> dict[str, Any]:
    row = conn.execute(
        """
        SELECT
          COALESCE(SUM(CASE WHEN status = 'complete' THEN rows_written ELSE 0 END), 0) AS quote_rows,
          COUNT(DISTINCT CASE WHEN status = 'complete' THEN market_id END) AS markets_with_price_history,
          COUNT(DISTINCT CASE WHEN status = 'complete' THEN token_id END) AS tokens_with_price_history,
          datetime(MIN(CASE WHEN status = 'complete' THEN start_ts END), 'unixepoch') AS min_quote_timestamp,
          datetime(MAX(CASE WHEN status = 'complete' THEN end_ts END), 'unixepoch') AS max_quote_timestamp
        FROM price_fetch_status
        """
    ).fetchone()
    failures = conn.execute(
        """
        SELECT error, COUNT(*) AS count
        FROM price_fetch_status
        WHERE status = 'failed'
        GROUP BY error
        ORDER BY count DESC
        LIMIT 20
        """
    ).fetchall()
    statuses = conn.execute(
        """
        SELECT
          status,
          COUNT(*) AS count,
          SUM(rows_written) AS rows_written,
          SUM(CASE WHEN retryable = 1 THEN 1 ELSE 0 END) AS retryable_count
        FROM price_fetch_status
        GROUP BY status
        ORDER BY status
        """
    ).fetchall()
    result = dict(row)
    result["quote_coverage_basis"] = "price_fetch_status_complete_jobs"
    result["fetch_failures_top"] = [dict(item) for item in failures]
    result["fetch_status_counts"] = [dict(item) for item in statuses]
    return result


def _compact_price_status_quality(conn: sqlite3.Connection, *, paths: PipelinePaths) -> dict[str, Any]:
    row = conn.execute(
        """
        SELECT
          COALESCE(SUM(CASE WHEN status = 'complete' THEN rows_written ELSE 0 END), 0) AS quote_rows_from_status,
          COUNT(DISTINCT CASE WHEN status = 'complete' THEN token_ref END) AS tokens_with_price_history,
          datetime(MIN(CASE WHEN status = 'complete' THEN start_ts END), 'unixepoch') AS min_job_window_start,
          datetime(MAX(CASE WHEN status = 'complete' THEN end_ts END), 'unixepoch') AS max_job_window_end
        FROM compact_price_fetch_status
        """
    ).fetchone()
    statuses = conn.execute(
        """
        SELECT
          status,
          COUNT(*) AS count,
          SUM(rows_written) AS rows_written,
          SUM(source_rows_seen) AS source_rows_seen,
          SUM(duplicate_rows_seen) AS duplicate_rows_seen,
          SUM(CASE WHEN retryable = 1 THEN 1 ELSE 0 END) AS retryable_count
        FROM compact_price_fetch_status
        GROUP BY status
        ORDER BY status
        """
    ).fetchall()
    refs = conn.execute(
        """
        SELECT
          (SELECT COUNT(*) FROM compact_tokens) AS token_refs,
          (SELECT COUNT(*) FROM compact_sources) AS source_refs
        """
    ).fetchone()
    return {
        **dict(row),
        **dict(refs),
        "db_bytes": paths.compact_price_db.stat().st_size if paths.compact_price_db.exists() else 0,
        "quote_coverage_basis": "compact_price_fetch_status_complete_jobs",
        "fetch_status_counts": [dict(item) for item in statuses],
    }


def _weather_quality(conn: sqlite3.Connection) -> dict[str, Any]:
    forecast = conn.execute(
        """
        SELECT
          COUNT(*) AS forecast_rows,
          COUNT(DISTINCT normalized_city || '|' || local_date) AS city_dates_with_forecast,
          MIN(local_date) AS min_forecast_date,
          MAX(local_date) AS max_forecast_date
        FROM forecast_snapshots
        """
    ).fetchone()
    realized = conn.execute(
        """
        SELECT
          COUNT(*) AS realized_rows,
          COUNT(DISTINCT normalized_city || '|' || local_date) AS city_dates_with_realized_labels,
          MIN(local_date) AS min_realized_date,
          MAX(local_date) AS max_realized_date
        FROM realized_daily_weather
        """
    ).fetchone()
    failures = conn.execute(
        """
        SELECT error, COUNT(*) AS count
        FROM weather_fetch_status
        WHERE status = 'failed'
        GROUP BY error
        ORDER BY count DESC
        LIMIT 20
        """
    ).fetchall()
    statuses = conn.execute(
        """
        SELECT
          status,
          COUNT(*) AS count,
          SUM(CASE WHEN retryable = 1 THEN 1 ELSE 0 END) AS retryable_count
        FROM weather_fetch_status
        GROUP BY status
        ORDER BY status
        """
    ).fetchall()
    return {
        **dict(forecast),
        **dict(realized),
        "open_meteo_fetch_failures_top": [dict(item) for item in failures],
        "fetch_status_counts": [dict(item) for item in statuses],
    }


def _decision_quality(conn: sqlite3.Connection, *, start: date, end: date) -> dict[str, Any]:
    params = (start.isoformat(), end.isoformat())
    scope = conn.execute(
        """
        SELECT target_start_date, target_end_date, price_store, max_staleness_minutes, built_at_utc, row_count
        FROM decision_artifact_scope
        WHERE scope_id = 'current'
        """
    ).fetchone()
    all_window_counts = conn.execute(
        """
        SELECT
          COUNT(*) AS total_rows_all_windows,
          SUM(CASE WHEN market_date_local < ? OR market_date_local > ? THEN 1 ELSE 0 END) AS out_of_target_window_rows,
          MIN(market_date_local) AS min_artifact_market_date,
          MAX(market_date_local) AS max_artifact_market_date
        FROM decision_snapshots
        """,
        params,
    ).fetchone()
    row = conn.execute(
        """
        SELECT
          COUNT(*) AS decision_rows,
          COUNT(DISTINCT universe_row_id) AS markets_in_decision_artifact,
          SUM(CASE WHEN side = 'YES' THEN 1 ELSE 0 END) AS yes_decision_rows,
          SUM(CASE WHEN side = 'NO' THEN 1 ELSE 0 END) AS no_decision_rows,
          SUM(CASE WHEN quote_timestamp_utc IS NOT NULL THEN 1 ELSE 0 END) AS rows_with_quote_snapshot,
          SUM(CASE WHEN forecast_run_time_utc IS NOT NULL THEN 1 ELSE 0 END) AS rows_with_forecast_snapshot,
          SUM(CASE WHEN realized_temperature_2m_max_f IS NOT NULL THEN 1 ELSE 0 END) AS rows_with_realized_label,
          SUM(CASE WHEN bid_size IS NOT NULL THEN 1 ELSE 0 END) AS rows_with_bid_size,
          SUM(CASE WHEN ask_size IS NOT NULL THEN 1 ELSE 0 END) AS rows_with_ask_size,
          SUM(CASE WHEN liquidity_depth IS NOT NULL THEN 1 ELSE 0 END) AS rows_with_liquidity_depth,
          SUM(CASE WHEN quote_source_cache_path IS NOT NULL THEN 1 ELSE 0 END) AS rows_with_quote_source_cache_path,
          SUM(CASE WHEN quote_source_payload_id IS NOT NULL THEN 1 ELSE 0 END) AS rows_with_quote_source_payload_id
        FROM decision_snapshots
        WHERE market_date_local >= ? AND market_date_local <= ?
        """,
        params,
    ).fetchone()
    flags = conn.execute(
        """
        SELECT value AS flag, COUNT(*) AS count
        FROM decision_snapshots, json_each(decision_snapshots.data_quality_flags)
        WHERE market_date_local >= ? AND market_date_local <= ?
        GROUP BY value
        ORDER BY count DESC, value
        LIMIT 25
        """,
        params,
    ).fetchall()
    by_city = conn.execute(
        """
        SELECT normalized_city, COUNT(*) AS decision_rows,
               SUM(CASE WHEN quote_timestamp_utc IS NOT NULL THEN 1 ELSE 0 END) AS rows_with_quote_snapshot,
               SUM(CASE WHEN forecast_run_time_utc IS NOT NULL THEN 1 ELSE 0 END) AS rows_with_forecast_snapshot,
               SUM(CASE WHEN realized_temperature_2m_max_f IS NOT NULL THEN 1 ELSE 0 END) AS rows_with_realized_label
        FROM decision_snapshots
        WHERE market_date_local >= ? AND market_date_local <= ?
        GROUP BY normalized_city
        ORDER BY normalized_city
        LIMIT 200
        """,
        params,
    ).fetchall()
    by_bucket_type = conn.execute(
        """
        SELECT bucket_type, COUNT(*) AS decision_rows,
               SUM(CASE WHEN quote_timestamp_utc IS NOT NULL THEN 1 ELSE 0 END) AS rows_with_quote_snapshot,
               SUM(CASE WHEN forecast_run_time_utc IS NOT NULL THEN 1 ELSE 0 END) AS rows_with_forecast_snapshot,
               SUM(CASE WHEN realized_temperature_2m_max_f IS NOT NULL THEN 1 ELSE 0 END) AS rows_with_realized_label
        FROM decision_snapshots
        WHERE market_date_local >= ? AND market_date_local <= ?
        GROUP BY bucket_type
        ORDER BY bucket_type
        """,
        params,
    ).fetchall()
    by_side = conn.execute(
        """
        SELECT side, COUNT(*) AS decision_rows,
               SUM(CASE WHEN quote_timestamp_utc IS NOT NULL THEN 1 ELSE 0 END) AS rows_with_quote_snapshot,
               AVG(spread) AS avg_spread,
               AVG(COALESCE(liquidity_depth, bid_size + ask_size)) AS avg_liquidity_proxy
        FROM decision_snapshots
        WHERE market_date_local >= ? AND market_date_local <= ?
        GROUP BY side
        ORDER BY side
        """,
        params,
    ).fetchall()
    by_lead_days = conn.execute(
        """
        SELECT lead_days, COUNT(*) AS decision_rows,
               SUM(CASE WHEN quote_timestamp_utc IS NOT NULL THEN 1 ELSE 0 END) AS rows_with_quote_snapshot,
               SUM(CASE WHEN forecast_run_time_utc IS NOT NULL THEN 1 ELSE 0 END) AS rows_with_forecast_snapshot
        FROM decision_snapshots
        WHERE market_date_local >= ? AND market_date_local <= ?
        GROUP BY lead_days
        ORDER BY lead_days
        """,
        params,
    ).fetchall()
    by_decision_hour = conn.execute(
        """
        SELECT decision_hour_utc, COUNT(*) AS decision_rows,
               SUM(CASE WHEN quote_timestamp_utc IS NOT NULL THEN 1 ELSE 0 END) AS rows_with_quote_snapshot,
               AVG(price_stale_seconds) AS avg_price_stale_seconds
        FROM decision_snapshots
        WHERE market_date_local >= ? AND market_date_local <= ?
        GROUP BY decision_hour_utc
        ORDER BY decision_hour_utc
        """,
        params,
    ).fetchall()
    by_spread_bucket = conn.execute(
        """
        SELECT
          CASE
            WHEN quote_timestamp_utc IS NULL THEN 'missing_quote'
            WHEN spread IS NULL THEN 'missing_spread'
            WHEN spread <= 0.02 THEN 'spread_0_02'
            WHEN spread <= 0.05 THEN 'spread_0_05'
            WHEN spread <= 0.10 THEN 'spread_0_10'
            ELSE 'spread_gt_10'
          END AS spread_bucket,
          COUNT(*) AS decision_rows,
          SUM(CASE WHEN quote_timestamp_utc IS NOT NULL THEN 1 ELSE 0 END) AS rows_with_quote_snapshot
        FROM decision_snapshots
        WHERE market_date_local >= ? AND market_date_local <= ?
        GROUP BY spread_bucket
        ORDER BY spread_bucket
        """,
        params,
    ).fetchall()
    by_liquidity_bucket = conn.execute(
        """
        SELECT
          CASE
            WHEN quote_timestamp_utc IS NULL THEN 'missing_quote'
            WHEN COALESCE(liquidity_depth, bid_size + ask_size) IS NULL THEN 'missing_liquidity'
            WHEN COALESCE(liquidity_depth, bid_size + ask_size) < 100 THEN 'liquidity_lt_100'
            WHEN COALESCE(liquidity_depth, bid_size + ask_size) < 1000 THEN 'liquidity_100_1000'
            WHEN COALESCE(liquidity_depth, bid_size + ask_size) < 10000 THEN 'liquidity_1000_10000'
            ELSE 'liquidity_ge_10000'
          END AS liquidity_bucket,
          COUNT(*) AS decision_rows,
          SUM(CASE WHEN quote_timestamp_utc IS NOT NULL THEN 1 ELSE 0 END) AS rows_with_quote_snapshot
        FROM decision_snapshots
        WHERE market_date_local >= ? AND market_date_local <= ?
        GROUP BY liquidity_bucket
        ORDER BY liquidity_bucket
        """,
        params,
    ).fetchall()
    quote_sources = conn.execute(
        """
        SELECT quote_store, COUNT(*) AS decision_rows,
               COUNT(DISTINCT quote_source_cache_path) AS source_cache_paths,
               COUNT(DISTINCT quote_source_payload_id) AS source_payload_ids
        FROM decision_snapshots
        WHERE quote_timestamp_utc IS NOT NULL
          AND market_date_local >= ? AND market_date_local <= ?
        GROUP BY quote_store
        ORDER BY quote_store
        """,
        params,
    ).fetchall()
    observed = conn.execute(
        """
        SELECT
          COUNT(*) AS decision_rows,
          COUNT(DISTINCT d.normalized_city || '|' || d.market_date_local || '|' || d.decision_time_utc) AS unique_city_date_decisions,
          SUM(CASE WHEN o.observed_high_so_far_f IS NOT NULL THEN 1 ELSE 0 END) AS rows_with_observed_high_so_far,
          SUM(CASE WHEN o.observed_high_so_far_f IS NULL THEN 1 ELSE 0 END) AS rows_missing_observed_high_so_far,
          SUM(CASE WHEN o.observed_high_raw_cache_path IS NOT NULL THEN 1 ELSE 0 END) AS rows_with_observed_high_source_cache_path,
          SUM(CASE WHEN o.observed_high_source_payload_id IS NOT NULL THEN 1 ELSE 0 END) AS rows_with_observed_high_source_payload_id,
          SUM(CASE WHEN o.observed_high_observed_at_utc > d.decision_time_utc THEN 1 ELSE 0 END) AS future_observed_high_violations,
          SUM(CASE
                WHEN o.observed_high_data_quality_flags LIKE '%final_daily_aggregate_used%'
                  OR o.observed_high_data_quality_flags LIKE '%derived_from_final_daily_aggregate%'
                THEN 1 ELSE 0
              END) AS final_daily_aggregate_leak_violations,
          SUM(CASE WHEN o.observed_high_is_final_at_decision = 1 THEN 1 ELSE 0 END) AS rows_final_high_known_at_decision
        FROM decision_snapshots d
        LEFT JOIN observed_high_snapshots o
          ON o.normalized_city = d.normalized_city
         AND o.market_date_local = d.market_date_local
         AND o.decision_time_utc = d.decision_time_utc
        WHERE d.market_date_local >= ? AND d.market_date_local <= ?
        """,
        params,
    ).fetchone()
    observed_by_city = conn.execute(
        """
        SELECT d.normalized_city, COUNT(*) AS decision_rows,
               SUM(CASE WHEN o.observed_high_so_far_f IS NOT NULL THEN 1 ELSE 0 END) AS rows_with_observed_high_so_far
        FROM decision_snapshots d
        LEFT JOIN observed_high_snapshots o
          ON o.normalized_city = d.normalized_city
         AND o.market_date_local = d.market_date_local
         AND o.decision_time_utc = d.decision_time_utc
        WHERE d.market_date_local >= ? AND d.market_date_local <= ?
        GROUP BY d.normalized_city
        ORDER BY d.normalized_city
        LIMIT 200
        """,
        params,
    ).fetchall()
    observed_by_missing_reason = conn.execute(
        """
        SELECT COALESCE(o.observed_high_missing_reason, 'available') AS missing_reason,
               COUNT(*) AS decision_rows
        FROM decision_snapshots d
        LEFT JOIN observed_high_snapshots o
          ON o.normalized_city = d.normalized_city
         AND o.market_date_local = d.market_date_local
         AND o.decision_time_utc = d.decision_time_utc
        WHERE d.market_date_local >= ? AND d.market_date_local <= ?
        GROUP BY missing_reason
        ORDER BY decision_rows DESC, missing_reason
        """,
        params,
    ).fetchall()
    observed_by_source = conn.execute(
        """
        SELECT COALESCE(o.observed_high_source, 'missing') AS observed_high_source,
               COUNT(*) AS decision_rows,
               COUNT(DISTINCT o.observed_high_raw_cache_path) AS source_cache_paths,
               COUNT(DISTINCT o.observed_high_source_payload_id) AS source_payload_ids
        FROM decision_snapshots d
        LEFT JOIN observed_high_snapshots o
          ON o.normalized_city = d.normalized_city
         AND o.market_date_local = d.market_date_local
         AND o.decision_time_utc = d.decision_time_utc
        WHERE d.market_date_local >= ? AND d.market_date_local <= ?
        GROUP BY observed_high_source
        ORDER BY observed_high_source
        """,
        params,
    ).fetchall()
    observed_by_lead_days = conn.execute(
        """
        SELECT d.lead_days, COUNT(*) AS decision_rows,
               SUM(CASE WHEN o.observed_high_so_far_f IS NOT NULL THEN 1 ELSE 0 END) AS rows_with_observed_high_so_far
        FROM decision_snapshots d
        LEFT JOIN observed_high_snapshots o
          ON o.normalized_city = d.normalized_city
         AND o.market_date_local = d.market_date_local
         AND o.decision_time_utc = d.decision_time_utc
        WHERE d.market_date_local >= ? AND d.market_date_local <= ?
        GROUP BY d.lead_days
        ORDER BY d.lead_days
        """,
        params,
    ).fetchall()
    observed_by_utc_hour = conn.execute(
        """
        SELECT d.decision_hour_utc, COUNT(*) AS decision_rows,
               SUM(CASE WHEN o.observed_high_so_far_f IS NOT NULL THEN 1 ELSE 0 END) AS rows_with_observed_high_so_far
        FROM decision_snapshots d
        LEFT JOIN observed_high_snapshots o
          ON o.normalized_city = d.normalized_city
         AND o.market_date_local = d.market_date_local
         AND o.decision_time_utc = d.decision_time_utc
        WHERE d.market_date_local >= ? AND d.market_date_local <= ?
        GROUP BY d.decision_hour_utc
        ORDER BY d.decision_hour_utc
        """,
        params,
    ).fetchall()
    observed_by_local_hour = conn.execute(
        """
        SELECT o.decision_local_hour, COUNT(*) AS decision_rows,
               SUM(CASE WHEN o.observed_high_so_far_f IS NOT NULL THEN 1 ELSE 0 END) AS rows_with_observed_high_so_far
        FROM decision_snapshots d
        LEFT JOIN observed_high_snapshots o
          ON o.normalized_city = d.normalized_city
         AND o.market_date_local = d.market_date_local
         AND o.decision_time_utc = d.decision_time_utc
        WHERE d.market_date_local >= ? AND d.market_date_local <= ?
        GROUP BY o.decision_local_hour
        ORDER BY o.decision_local_hour
        """,
        params,
    ).fetchall()
    label_summary = conn.execute(
        """
        SELECT
          COUNT(*) AS decision_rows,
          SUM(CASE WHEN l.canonical_label IS NOT NULL THEN 1 ELSE 0 END) AS rows_with_canonical_label,
          SUM(CASE WHEN l.canonical_label IS NULL THEN 1 ELSE 0 END) AS rows_missing_canonical_label,
          SUM(CASE WHEN l.exclude_from_backtest_replay = 1 THEN 1 ELSE 0 END) AS rows_exclude_from_backtest_replay,
          SUM(CASE WHEN l.exclude_from_open_meteo_training = 1 THEN 1 ELSE 0 END) AS rows_exclude_from_open_meteo_training,
          COUNT(DISTINCT l.source_csv_path) AS source_csv_paths
        FROM decision_snapshots d
        LEFT JOIN canonical_settlement_labels l ON l.universe_row_id = d.universe_row_id
        WHERE d.market_date_local >= ? AND d.market_date_local <= ?
        """,
        params,
    ).fetchone()
    label_actions = conn.execute(
        """
        SELECT COALESCE(l.recommended_action, 'missing_label') AS recommended_action,
               COUNT(*) AS decision_rows
        FROM decision_snapshots d
        LEFT JOIN canonical_settlement_labels l ON l.universe_row_id = d.universe_row_id
        WHERE d.market_date_local >= ? AND d.market_date_local <= ?
        GROUP BY recommended_action
        ORDER BY decision_rows DESC, recommended_action
        """,
        params,
    ).fetchall()
    label_sources = conn.execute(
        """
        SELECT COALESCE(l.canonical_label_source, 'missing_label') AS canonical_label_source,
               COUNT(*) AS decision_rows
        FROM decision_snapshots d
        LEFT JOIN canonical_settlement_labels l ON l.universe_row_id = d.universe_row_id
        WHERE d.market_date_local >= ? AND d.market_date_local <= ?
        GROUP BY canonical_label_source
        ORDER BY decision_rows DESC, canonical_label_source
        """,
        params,
    ).fetchall()
    result = dict(row)
    result["artifact_scope"] = dict(scope) if scope else None
    result.update(dict(all_window_counts))
    result["out_of_target_window_rows"] = int(result.get("out_of_target_window_rows") or 0)
    result["quality_flags_top"] = [dict(item) for item in flags]
    result["coverage_by_city"] = [dict(item) for item in by_city]
    result["coverage_by_bucket_type"] = [dict(item) for item in by_bucket_type]
    result["coverage_by_side"] = [dict(item) for item in by_side]
    result["coverage_by_lead_days"] = [dict(item) for item in by_lead_days]
    result["coverage_by_decision_hour_utc"] = [dict(item) for item in by_decision_hour]
    result["coverage_by_spread_bucket"] = [dict(item) for item in by_spread_bucket]
    result["coverage_by_liquidity_bucket"] = [dict(item) for item in by_liquidity_bucket]
    result["quote_source_audit"] = [dict(item) for item in quote_sources]
    result["observed_high"] = {
        **dict(observed),
        "coverage_by_city": [dict(item) for item in observed_by_city],
        "coverage_by_missing_reason": [dict(item) for item in observed_by_missing_reason],
        "coverage_by_source": [dict(item) for item in observed_by_source],
        "coverage_by_lead_days": [dict(item) for item in observed_by_lead_days],
        "coverage_by_decision_hour_utc": [dict(item) for item in observed_by_utc_hour],
        "coverage_by_decision_local_hour": [dict(item) for item in observed_by_local_hour],
    }
    result["settlement_labels"] = {
        **dict(label_summary),
        "coverage_by_recommended_action": [dict(item) for item in label_actions],
        "coverage_by_canonical_label_source": [dict(item) for item in label_sources],
    }
    result["edge_bucket_note"] = "edge buckets are strategy/model outputs and should be added by the backtesting engine, not fabricated in the data pipeline."
    return result


def _usable_market_count(paths: PipelinePaths, *, start: date, end: date) -> int:
    with _connect(paths.market_db) as market_conn, _connect(paths.price_db) as price_conn, _connect(paths.weather_db) as weather_conn:
        market_conn.row_factory = sqlite3.Row
        price_conn.row_factory = sqlite3.Row
        weather_conn.row_factory = sqlite3.Row
        tokens = {
            row["universe_row_id"]
            for row in market_conn.execute(
                "SELECT universe_row_id, yes_token_id, normalized_city, market_date_local FROM market_universe WHERE resolved_outcome IS NOT NULL"
            )
        }
        priced = {
            row["universe_row_id"]
            for row in price_conn.execute(
                """
                SELECT DISTINCT universe_row_id
                FROM price_fetch_status
                WHERE status = 'complete' AND universe_row_id IS NOT NULL
                """
            )
        }
        forecasted = {
            f"{row['normalized_city']}|{row['local_date']}"
            for row in weather_conn.execute("SELECT DISTINCT normalized_city, local_date FROM forecast_snapshots")
        }
        realized = {
            f"{row['normalized_city']}|{row['local_date']}"
            for row in weather_conn.execute("SELECT DISTINCT normalized_city, local_date FROM realized_daily_weather")
        }
        count = 0
        for row in market_conn.execute(
            """
            SELECT universe_row_id, normalized_city, market_date_local
            FROM market_universe
            WHERE resolved_outcome IS NOT NULL
              AND market_date_local >= ?
              AND market_date_local <= ?
            """,
            (start.isoformat(), end.isoformat()),
        ):
            key = f"{row['normalized_city']}|{row['market_date_local']}"
            if row["universe_row_id"] in tokens and row["universe_row_id"] in priced and key in forecasted and key in realized:
                count += 1
        return count


def _known_gaps(summary: Mapping[str, Any]) -> list[str]:
    gaps = []
    markets = summary.get("markets") or {}
    prices = summary.get("prices") or {}
    weather = summary.get("weather") or {}
    if markets.get("min_market_date") != summary.get("target_start_date") or markets.get("max_market_date") != summary.get("target_end_date"):
        gaps.append("market_universe_does_not_span_full_target_window")
    expected_city_dates = int(markets.get("city_date_pairs") or 0)
    if int(weather.get("city_dates_with_forecast") or 0) < expected_city_dates:
        gaps.append("forecast_history_incomplete")
    if int(weather.get("city_dates_with_realized_labels") or 0) < expected_city_dates:
        gaps.append("realized_weather_labels_incomplete")
    decisions = summary.get("decision_artifact") or {}
    if int(decisions.get("decision_rows") or 0) > 0:
        if int(decisions.get("rows_with_quote_snapshot") or 0) < int(decisions.get("decision_rows") or 0):
            gaps.append("price_history_incomplete")
    elif int(prices.get("markets_with_price_history") or 0) < int(markets.get("markets") or 0):
        gaps.append("price_history_incomplete")
    decision_markets_missing = int(decisions.get("markets_in_decision_artifact") or 0) < int(
        markets.get("markets_with_live_decision_times") or 0
    )
    decision_rows_missing = int(decisions.get("decision_rows") or 0) < int(markets.get("expected_decision_rows") or 0)
    if decision_markets_missing or decision_rows_missing:
        gaps.append("decision_artifact_incomplete")
    if int(decisions.get("out_of_target_window_rows") or 0) > 0:
        gaps.append("decision_artifact_contains_rows_outside_target_window")
    observed = decisions.get("observed_high") or {}
    if int(decisions.get("decision_rows") or 0) > 0 and int(observed.get("decision_rows") or 0) != int(decisions.get("decision_rows") or 0):
        gaps.append("observed_high_snapshots_incomplete")
    if int(observed.get("future_observed_high_violations") or 0) > 0:
        gaps.append("observed_high_future_timestamp_violation")
    if int(observed.get("final_daily_aggregate_leak_violations") or 0) > 0:
        gaps.append("observed_high_final_daily_aggregate_leak")
    labels = decisions.get("settlement_labels") or {}
    if int(decisions.get("decision_rows") or 0) > 0 and int(labels.get("rows_with_canonical_label") or 0) != int(
        decisions.get("decision_rows") or 0
    ):
        gaps.append("canonical_settlement_labels_incomplete")
    if markets.get("min_quotes_from") and str(markets.get("min_quotes_from")) > summary.get("target_start_date", ""):
        gaps.append("telonex_quote_availability_starts_after_target_start")
    return gaps


def _supported_window_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    markets = summary.get("markets") or {}
    archive_start = markets.get("min_quotes_from")
    target_end = summary.get("target_end_date")
    market_start = None
    if archive_start:
        try:
            market_start = (date.fromisoformat(str(archive_start)) + timedelta(days=2)).isoformat()
        except ValueError:
            market_start = archive_start
    return {
        "telonex_quote_archive_start": archive_start,
        "recommended_v1_market_start": market_start,
        "recommended_v1_market_end": target_end,
        "basis": "Telonex market metadata min quotes_from plus two-day live lead window; direct probe found 2025-10-09 unavailable and 2025-10-11 available.",
    }


def _quality_markdown(summary: Mapping[str, Any]) -> str:
    markets = summary.get("markets") or {}
    prices = summary.get("prices") or {}
    compact = summary.get("compact_price_status") or {}
    weather = summary.get("weather") or {}
    decisions = summary.get("decision_artifact") or {}
    observed = decisions.get("observed_high") or {}
    labels = decisions.get("settlement_labels") or {}
    lines = [
        "# DailyWeather Data Quality Report",
        "",
        f"Generated: `{summary.get('generated_at_utc')}`",
        f"Target window: `{summary.get('target_start_date')}` through `{summary.get('target_end_date')}`",
        f"Backtest ready: `{summary.get('backtest_ready')}`",
        f"Supported window: `{(summary.get('supported_backtest_window') or {}).get('recommended_v1_market_start')}` through `{(summary.get('supported_backtest_window') or {}).get('recommended_v1_market_end')}`",
        "",
        "## Coverage",
        "",
        f"- Market rows: `{markets.get('rows') or 0}` across `{markets.get('markets') or 0}` market ids and `{markets.get('city_date_pairs') or 0}` city/date pairs.",
        f"- Parsed successfully: `{markets.get('parsed_successfully') or 0}`.",
        f"- Coordinates/timezone present: `{markets.get('with_coordinates_timezone') or 0}`.",
        f"- YES token IDs present: `{markets.get('with_yes_token') or 0}`; NO token IDs present: `{markets.get('with_no_token') or 0}`.",
        f"- Resolved outcomes: `{markets.get('with_resolved_outcome') or 0}`.",
        f"- Legacy wide price-history markets: `{prices.get('markets_with_price_history') or 0}`; quote rows: `{prices.get('quote_rows') or 0}`.",
        f"- Compact price status rows: `{compact.get('quote_rows_from_status') or 0}` quote rows from `{compact.get('tokens_with_price_history') or 0}` complete token refs; DB bytes `{compact.get('db_bytes') or 0}`.",
        f"- Forecast city/dates: `{weather.get('city_dates_with_forecast') or 0}`; realized city/dates: `{weather.get('city_dates_with_realized_labels') or 0}`.",
        f"- Decision rows: `{decisions.get('decision_rows') or 0}`.",
        f"- Decision artifact scope: `{(decisions.get('artifact_scope') or {}).get('target_start_date')}` through `{(decisions.get('artifact_scope') or {}).get('target_end_date')}`; out-of-target rows: `{decisions.get('out_of_target_window_rows') or 0}`.",
        f"- Decision quote snapshots: `{decisions.get('rows_with_quote_snapshot') or 0}`; quote source paths: `{decisions.get('rows_with_quote_source_cache_path') or 0}`.",
        f"- Decision quote size fields: bid size `{decisions.get('rows_with_bid_size') or 0}`, ask size `{decisions.get('rows_with_ask_size') or 0}`, liquidity depth `{decisions.get('rows_with_liquidity_depth') or 0}`.",
        f"- Observed high so far: `{observed.get('rows_with_observed_high_so_far') or 0}` rows; future observed-high violations `{observed.get('future_observed_high_violations') or 0}`; final-aggregate leak flags `{observed.get('final_daily_aggregate_leak_violations') or 0}`.",
        f"- Canonical settlement labels: `{labels.get('rows_with_canonical_label') or 0}` rows; replay exclusions `{labels.get('rows_exclude_from_backtest_replay') or 0}`.",
        f"- Final usable market count for backtesting: `{summary.get('usable_market_count_for_backtesting') or 0}`.",
        "",
        "## Replay Breakdowns",
        "",
        "- JSON report includes coverage by city, bucket type, side, lead days, decision hour UTC, spread bucket, liquidity bucket, source audit, and data-quality flags.",
        "- Edge buckets are strategy/model outputs and should be added by the backtesting engine.",
        "",
        "## Known Gaps",
        "",
    ]
    gaps = summary.get("known_gaps") or []
    lines.extend([f"- `{gap}`" for gap in gaps] or ["- None recorded."])
    lines.extend(
        [
            "",
            "## Artifact Paths",
            "",
            f"- Market universe: `{summary['paths']['market_db']}`",
            f"- Price history: `{summary['paths']['price_db']}`",
            f"- Weather history: `{summary['paths']['weather_db']}`",
            f"- Decision artifact: `{summary['paths']['decision_db']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def _compact_quality_markdown(summary: Mapping[str, Any]) -> str:
    compact = summary.get("compact_prices") or {}
    field_presence = compact.get("field_presence") or {}
    source_audit = compact.get("source_audit") or {}
    lines = [
        "# DailyWeather Compact Price Quality Report",
        "",
        f"Generated: `{summary.get('generated_at_utc')}`",
        f"Target window: `{summary.get('target_start_date')}` through `{summary.get('target_end_date')}`",
        f"Compact DB bytes: `{summary.get('compact_price_db_bytes')}`",
        f"Wide DB bytes: `{summary.get('wide_price_db_bytes')}`",
        f"Storage savings vs wide: `{summary.get('storage_savings_vs_wide_percent')}`%",
        "",
        "## Compact Coverage",
        "",
        f"- Tokens: `{compact.get('tokens') or 0}`.",
        f"- Sources: `{compact.get('sources') or 0}`.",
        f"- Quote rows: `{compact.get('quote_rows') or 0}`.",
        f"- Min quote timestamp: `{compact.get('min_quote_timestamp')}`.",
        f"- Max quote timestamp: `{compact.get('max_quote_timestamp')}`.",
        f"- Source-path rows: `{field_presence.get('source_ref_rows') or 0}`.",
        f"- Bid/ask/mid/last/spread rows: `{field_presence.get('bid_rows') or 0}` / `{field_presence.get('ask_rows') or 0}` / `{field_presence.get('mid_rows') or 0}` / `{field_presence.get('last_price_rows') or 0}` / `{field_presence.get('spread_rows') or 0}`.",
        f"- Size/depth rows: bid size `{field_presence.get('bid_size_rows') or 0}`, ask size `{field_presence.get('ask_size_rows') or 0}`, liquidity depth `{field_presence.get('liquidity_depth_rows') or 0}`.",
        f"- Source refs with cache path/payload id/sha256: `{source_audit.get('source_cache_path_refs') or 0}` / `{source_audit.get('source_payload_id_refs') or 0}` / `{source_audit.get('payload_sha256_refs') or 0}`.",
        "",
        "## Fetch Status",
        "",
    ]
    for item in compact.get("fetch_status_counts") or []:
        lines.append(
            f"- `{item.get('status')}`: `{item.get('count')}` jobs, `{item.get('rows_written')}` rows, `{item.get('retryable_count')}` retryable."
        )
    lines.extend(
        [
            "",
            "## Breakdown Coverage",
            "",
            "- JSON report includes compact fetch status by side, city, bucket type, and bucket type plus side.",
        ]
    )
    return "\n".join(lines) + "\n"


def _paths_json(paths: PipelinePaths) -> dict[str, str]:
    return {
        "telonex_raw": str(paths.telonex_raw),
        "open_meteo_raw": str(paths.open_meteo_raw),
        "market_db": str(paths.market_db),
        "weather_db": str(paths.weather_db),
        "price_db": str(paths.price_db),
        "compact_price_db": str(paths.compact_price_db),
        "decision_db": str(paths.decision_db),
        "logs": str(paths.logs),
        "reports": str(paths.reports),
    }


def _progress(progress_every: int, message: str) -> None:
    if progress_every <= 0:
        return
    # Keep progress on stderr so JSON command output remains parseable.
    print(f"[data-pipeline] {message}", file=sys.stderr, flush=True)


def _market_warnings(market: Any, bucket: Any, no_token_id: Optional[str], raw: Mapping[str, Any]) -> list[str]:
    warnings = []
    if market.city is None:
        warnings.append("missing_city")
    if market.target_date is None:
        warnings.append("missing_market_date")
    if not bucket.token_id:
        warnings.append("missing_yes_token_id")
    if no_token_id is None and _looks_binary_outcome(raw):
        warnings.append("missing_no_token_id")
    if bucket.lower_f is None and bucket.upper_f is None:
        warnings.append("unbounded_bucket")
    if not raw.get("quotes_from") and not raw.get("telonex_quotes_from"):
        warnings.append("missing_telonex_quotes_from")
    if not raw.get("quotes_to") and not raw.get("telonex_quotes_to"):
        warnings.append("missing_telonex_quotes_to")
    return warnings


def _parser_confidence(warnings: list[str]) -> float:
    if not warnings:
        return 1.0
    hard = {"missing_city", "missing_market_date", "missing_yes_token_id", "unbounded_bucket"}
    penalty = sum(0.2 if warning in hard else 0.05 for warning in warnings)
    return round(max(0.0, 1.0 - penalty), 2)


def _bucket_info(bucket: Any) -> dict[str, str]:
    lower = bucket.lower_f
    upper = bucket.upper_f
    if lower is not None and upper is not None:
        width = float(upper) - float(lower)
        if width <= 1.05:
            return {"bucket_type": "exact", "operator": "between_inclusive"}
        return {"bucket_type": "range", "operator": "between_inclusive"}
    if lower is not None:
        return {"bucket_type": "above", "operator": "gte"}
    if upper is not None:
        return {"bucket_type": "below", "operator": "lte"}
    return {"bucket_type": "unknown", "operator": "unknown"}


def _bucket_resolved_outcome(bucket: Any, raw: Mapping[str, Any], *, fallback_yes: Optional[int]) -> Optional[int]:
    prices = _parse_jsonish(raw.get("outcomePrices") or raw.get("outcome_prices"))
    tokens = [str(item) for item in _parse_jsonish(raw.get("clobTokenIds") or raw.get("clob_token_ids"))]
    if bucket.token_id and bucket.token_id in tokens:
        index = tokens.index(bucket.token_id)
        if index < len(prices):
            try:
                value = float(prices[index])
            except (TypeError, ValueError):
                return fallback_yes
            if value <= 0.001:
                return 0
            if value >= 0.999:
                return 1
    return fallback_yes


def _looks_binary_outcome(raw: Mapping[str, Any]) -> bool:
    outcomes = [str(item).lower() for item in _parse_jsonish(raw.get("outcomes"))]
    return "yes" in outcomes and "no" in outcomes


def _parse_jsonish(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        return parsed if isinstance(parsed, list) else []
    return []


def _date_to_epoch_us(value: date) -> int:
    return int(datetime.combine(value, datetime_time(0, 0), tzinfo=timezone.utc).timestamp() * 1_000_000)


def _iso_from_epoch_us(value: Any) -> Optional[str]:
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        return None
    if numeric <= 0:
        return None
    try:
        return datetime.fromtimestamp(numeric / 1_000_000, tz=timezone.utc).isoformat()
    except (OverflowError, OSError, ValueError):
        return None


def _record_matches_token(record: Mapping[str, Any], token_id: str) -> bool:
    value = _record_token_id(record)
    if value is not None:
        return value == token_id
    return False


def _record_token_id(record: Mapping[str, Any]) -> Optional[str]:
    for key in ("token_id", "tokenId", "clobTokenId", "asset_id", "assetId"):
        value = _case_field(record, key)
        if value is not None:
            return str(value).strip()
    return None


def _job_key(job: Mapping[str, Any]) -> tuple[str, int, int]:
    return (str(job["token_id"]), int(job["start_ts"]), int(job["end_ts"]))


def _quote_cache_path_date(path: Path) -> Optional[date]:
    try:
        return date.fromisoformat(path.name[:10])
    except ValueError:
        return None


def _record_timestamp(record: Mapping[str, Any]) -> Optional[datetime]:
    for key in ("timestamp", "ts", "time", "datetime", "created_at", "createdAt", "event_time", "eventTime"):
        parsed = _parse_timestamp_value(_case_field(record, key), key)
        if parsed is not None:
            return parsed
    for key, value in record.items():
        lower = str(key).lower()
        if "timestamp" in lower or lower.endswith("_time"):
            parsed = _parse_timestamp_value(value, str(key))
            if parsed is not None:
                return parsed
    return None


def _parse_timestamp_value(value: Any, key: str) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if hasattr(value, "to_pydatetime"):
        converted = value.to_pydatetime()
        return converted if converted.tzinfo else converted.replace(tzinfo=timezone.utc)
    numeric = _to_float(value)
    if numeric is not None:
        try:
            lower = key.lower()
            if lower.endswith("_ns") or numeric > 1e17:
                return datetime.fromtimestamp(numeric / 1_000_000_000, tz=timezone.utc)
            if lower.endswith("_us") or numeric > 1e14:
                return datetime.fromtimestamp(numeric / 1_000_000, tz=timezone.utc)
            if lower.endswith("_ms") or numeric > 1e11:
                return datetime.fromtimestamp(numeric / 1_000, tz=timezone.utc)
            return datetime.fromtimestamp(numeric, tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _first_numeric(record: Mapping[str, Any], keys: Iterable[str]) -> Optional[float]:
    for key in keys:
        value = _to_float(_case_field(record, key))
        if value is not None:
            return value
    return None


def _case_field(record: Mapping[str, Any], key: str) -> Any:
    if key in record:
        return record[key]
    lower = key.lower()
    for candidate, value in record.items():
        if str(candidate).lower() == lower:
            return value
    return None


def _to_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric) or math.isinf(numeric):
        return None
    return numeric


def _to_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return int(value)
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric) or math.isinf(numeric):
        return None
    return int(numeric)


def _latest_settlement_labels_csv(paths: PipelinePaths) -> Path:
    directory = paths.work / "reports" / "settlement_label_audit"
    candidates = sorted(directory.glob("*-settlement-label-audit-canonical-labels.csv"))
    if not candidates:
        raise FileNotFoundError(f"No canonical settlement label CSV found under {directory}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _quote_flags(bid: Optional[float], ask: Optional[float], mid: Optional[float], last: Optional[float]) -> str:
    flags = []
    if bid is None and ask is None and mid is None and last is None:
        flags.append("no_price_fields")
    if bid is not None and ask is not None and ask < bid:
        flags.append("crossed_quote")
    for name, value in (("bid", bid), ("ask", ask), ("mid", mid), ("last", last)):
        if value is not None and not 0 <= value <= 1:
            flags.append(f"{name}_outside_probability_range")
    return json.dumps(flags, sort_keys=True)


def _weather_features(payload: Mapping[str, Any], target_date: date) -> dict[str, float]:
    features: dict[str, float] = {}
    daily = payload.get("daily") if isinstance(payload.get("daily"), Mapping) else {}
    daily_times = daily.get("time") or []
    daily_index = _date_index(daily_times, target_date)
    if daily_index is not None:
        for key in (
            "temperature_2m_max",
            "precipitation_sum",
            "precipitation_hours",
            "wind_speed_10m_max",
            "wind_gusts_10m_max",
            "shortwave_radiation_sum",
        ):
            value = _value_at_index(daily.get(key), daily_index)
            if value is not None:
                features[key] = value
    hourly = payload.get("hourly") if isinstance(payload.get("hourly"), Mapping) else {}
    hourly_times = hourly.get("time") or []
    indices = [index for index, value in enumerate(hourly_times) if str(value).startswith(target_date.isoformat())]
    if indices:
        _add_feature(features, hourly, indices, "temperature_2m", max, "hourly_temperature_2m_max")
        _add_feature(features, hourly, indices, "apparent_temperature", max, "hourly_apparent_temperature_max")
        _add_feature(features, hourly, indices, "relative_humidity_2m", _mean, "relative_humidity_2m_mean")
        _add_feature(features, hourly, indices, "dew_point_2m", max, "dew_point_2m_max")
        _add_feature(features, hourly, indices, "cloud_cover", _mean, "cloud_cover_mean")
        _add_feature(features, hourly, indices, "wind_speed_10m", max, "hourly_wind_speed_10m_max")
        _add_feature(features, hourly, indices, "precipitation", sum, "hourly_precipitation_sum")
    return features


def _date_index(values: Iterable[Any], target_date: date) -> Optional[int]:
    for index, value in enumerate(values):
        if str(value)[:10] == target_date.isoformat():
            return index
    return None


def _value_at_index(values: Any, index: int) -> Optional[float]:
    if not isinstance(values, list) or index >= len(values):
        return None
    return _to_float(values[index])


def _add_feature(features: dict[str, float], hourly: Mapping[str, Any], indices: list[int], key: str, reducer: Any, output_key: str) -> None:
    values = [_to_float((hourly.get(key) or [None])[index]) for index in indices if index < len(hourly.get(key) or [])]
    clean = [value for value in values if value is not None]
    if clean:
        features[output_key] = float(reducer(clean))


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values)


def _weather_flags(features: Mapping[str, Any]) -> list[str]:
    flags = []
    if features.get("temperature_2m_max") is None and features.get("hourly_temperature_2m_max") is None:
        flags.append("missing_temperature_max")
    return flags


def _hourly_weather_vars() -> str:
    return ",".join(
        (
            "temperature_2m",
            "relative_humidity_2m",
            "dew_point_2m",
            "apparent_temperature",
            "precipitation",
            "cloud_cover",
            "wind_speed_10m",
            "pressure_msl",
            "shortwave_radiation",
        )
    )


def _forecast_run_time(decision_time: datetime, availability_lag_hours: int = 6) -> datetime:
    current = decision_time if decision_time.tzinfo else decision_time.replace(tzinfo=timezone.utc)
    eligible = current.astimezone(timezone.utc) - timedelta(hours=availability_lag_hours)
    run_hour = (eligible.hour // 6) * 6
    return eligible.replace(hour=run_hour, minute=0, second=0, microsecond=0)


def _quote_cache_dirs(paths: PipelinePaths) -> list[Path]:
    candidates = (
        paths.telonex_raw / "polymarket" / "quotes",
        paths.work / "cache" / "live_like_backtest" / "telonex" / "polymarket" / "quotes",
        paths.work / "cache" / "telonex_long_backtest" / "telonex" / "polymarket" / "quotes",
        paths.work / "cache" / "long_backtest" / "telonex" / "polymarket" / "quotes",
    )
    return [path for path in candidates if path.exists()]


def _date_range(start: date, end: date) -> Iterable[date]:
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def _as_utc(value: datetime) -> datetime:
    return value.astimezone(timezone.utc) if value.tzinfo else value.replace(tzinfo=timezone.utc)


def _parse_datetime(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


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


def _zone(name: str) -> ZoneInfo:
    try:
        return ZoneInfo(name)
    except ZoneInfoNotFoundError:
        return ZoneInfo("UTC")


def _stable_id(*parts: Any) -> str:
    payload = json.dumps(_jsonable(parts), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> Optional[str]:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return str(value)


def _none_or_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _parse_date_arg(value: str) -> date:
    return date.fromisoformat(value)


if __name__ == "__main__":
    raise SystemExit(main())
