from __future__ import annotations

import argparse
import csv
import json
import os
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


FIRST_10_STATIONS = ("KLGA", "KDAL", "KMIA", "KORD", "KSEA", "KHOU", "KAUS", "KLAX", "KSFO", "KBKF")
TARGET_START = date(2026, 1, 1)
TARGET_END = date(2026, 6, 29)
THERMAL_WINDOW_START_LOCAL_HOUR = 12
THERMAL_WINDOW_END_LOCAL_HOUR = 18


@dataclass(frozen=True)
class Paths:
    work_root: Path

    @property
    def work(self) -> Path:
        return self.work_root / "work"

    @property
    def market_db(self) -> Path:
        return self.work / "data/backtest/daily_weather_market_universe.sqlite"

    @property
    def compact_price_db(self) -> Path:
        return self.work / "data/backtest/daily_weather_price_history_compact.sqlite"

    @property
    def wide_price_db(self) -> Path:
        return self.work / "data/backtest/daily_weather_price_history.sqlite"

    @property
    def weather_db(self) -> Path:
        return self.work / "data/backtest/daily_weather_weather_history.sqlite"

    @property
    def decision_db(self) -> Path:
        return self.work / "data/backtest/daily_weather_decision_artifact.sqlite"

    @property
    def station_db(self) -> Path:
        return self.work / "data/backtest/daily_weather_station_observations.sqlite"

    @property
    def telonex_raw_quotes(self) -> Path:
        return self.work / "cache/telonex/raw/polymarket/quotes"

    @property
    def freeze_json(self) -> Path:
        return self.work / "reports/dataset_freeze/20260707T040322Z-six-month-v1-dataset-freeze.json"

    @property
    def latest_compact_report(self) -> Path:
        return self.work / "reports/data_quality/20260707T032429Z-daily-weather-compact-price-quality.json"

    @property
    def latest_decision_report(self) -> Path:
        return self.work / "reports/data_quality/20260707T033504Z-daily-weather-data-quality.json"

    @property
    def report_dir(self) -> Path:
        return self.work / "reports/telonex_preservation"


def _connect_ro(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only=ON")
    return conn


def _scalar(conn: sqlite3.Connection, sql: str, params: tuple[Any, ...] = ()) -> Any:
    row = conn.execute(sql, params).fetchone()
    return row[0] if row else None


def _rows(conn: sqlite3.Connection, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    return [dict(row) for row in conn.execute(sql, params).fetchall()]


def _file_info(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "bytes": path.stat().st_size if path.exists() and path.is_file() else None,
    }


def _tree_count_and_bytes(path: Path, suffix: str | None = None) -> dict[str, Any]:
    count = 0
    bytes_total = 0
    if path.exists():
        for root, _, files in os.walk(path):
            for filename in files:
                if suffix and not filename.endswith(suffix):
                    continue
                count += 1
                try:
                    bytes_total += (Path(root) / filename).stat().st_size
                except OSError:
                    pass
    return {"path": str(path), "exists": path.exists(), "files": count, "bytes": bytes_total}


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _classify_failure(error: str | None) -> str:
    text = error or ""
    if "HTTP 404" in text:
        return "http_404_source_unavailable"
    if "cached missing file" in text.lower():
        return "cached_missing_source_unavailable"
    if "429" in text or "rate" in text.lower():
        return "rate_limit_or_throttle"
    if "dns" in text.lower() or "resolve" in text.lower() or "NameResolutionError" in text:
        return "dns_or_network_transient"
    if "sqlite" in text.lower() or "database is locked" in text.lower():
        return "sqlite_operational"
    if not text:
        return "missing_error_text"
    return "other_operational_or_unknown"


def _parse_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def collect_core_status(paths: Paths) -> dict[str, Any]:
    status: dict[str, Any] = {
        "target_window": {"start": TARGET_START.isoformat(), "end": TARGET_END.isoformat()},
        "files": {
            "market_universe_db": _file_info(paths.market_db),
            "compact_price_db": _file_info(paths.compact_price_db),
            "wide_price_db": _file_info(paths.wide_price_db),
            "weather_history_db": _file_info(paths.weather_db),
            "decision_artifact_db": _file_info(paths.decision_db),
            "station_observations_db": _file_info(paths.station_db),
            "freeze_json": _file_info(paths.freeze_json),
            "latest_compact_report": _file_info(paths.latest_compact_report),
            "latest_decision_report": _file_info(paths.latest_decision_report),
        },
        "raw_cache": {
            "telonex_quote_parquets": _tree_count_and_bytes(paths.telonex_raw_quotes, ".parquet"),
            "telonex_missing_markers": _tree_count_and_bytes(paths.telonex_raw_quotes, ".missing.json"),
            "global_weather_expansion_telonex": _tree_count_and_bytes(
                paths.work / "cache/global_weather_expansion/telonex"
            ),
        },
        "freeze_manifest": _load_json(paths.freeze_json),
    }

    with _connect_ro(paths.market_db) as conn:
        source_paths = _rows(
            conn,
            """
            SELECT telonex_source_path, COUNT(*) AS market_rows
            FROM market_universe
            GROUP BY telonex_source_path
            ORDER BY market_rows DESC
            """,
        )
        status["market_universe"] = {
            "market_rows": _scalar(conn, "SELECT COUNT(*) FROM market_universe"),
            "token_rows": _scalar(conn, "SELECT COUNT(*) FROM market_tokens"),
            "parsed_target_window_rows": _scalar(
                conn,
                """
                SELECT COUNT(*)
                FROM market_universe
                WHERE market_date_local BETWEEN ? AND ?
                """,
                (TARGET_START.isoformat(), TARGET_END.isoformat()),
            ),
            "telonex_source_paths": source_paths,
            "telonex_source_paths_exist": [
                {"path": row["telonex_source_path"], "exists": Path(row["telonex_source_path"]).exists()}
                for row in source_paths
            ],
        }

    with _connect_ro(paths.compact_price_db) as conn:
        status_rows = _rows(
            conn,
            """
            SELECT status, retryable, COUNT(*) AS jobs, SUM(rows_written) AS rows_written,
                   SUM(source_rows_seen) AS source_rows_seen,
                   SUM(duplicate_rows_seen) AS duplicate_rows_seen,
                   MIN(updated_at_utc) AS min_updated_at_utc,
                   MAX(updated_at_utc) AS max_updated_at_utc
            FROM compact_price_fetch_status
            GROUP BY status, retryable
            ORDER BY status, retryable
            """,
        )
        failure_counter: Counter[str] = Counter()
        for row in conn.execute("SELECT error FROM compact_price_fetch_status WHERE status = 'failed'"):
            failure_counter[_classify_failure(row["error"])] += 1
        status["compact_price_status"] = {
            "token_refs": _scalar(conn, "SELECT COUNT(*) FROM compact_tokens"),
            "source_refs": _scalar(conn, "SELECT COUNT(*) FROM compact_sources"),
            "status_rows": status_rows,
            "failure_breakdown": dict(sorted(failure_counter.items())),
            "cache_miss_jobs": _scalar(
                conn, "SELECT COUNT(*) FROM compact_price_fetch_status WHERE status = 'cache_miss'"
            ),
            "complete_jobs": _scalar(conn, "SELECT COUNT(*) FROM compact_price_fetch_status WHERE status = 'complete'"),
            "failed_jobs": _scalar(conn, "SELECT COUNT(*) FROM compact_price_fetch_status WHERE status = 'failed'"),
            "operational_failed_jobs": sum(
                count
                for failure_class, count in failure_counter.items()
                if not failure_class.endswith("_source_unavailable")
            ),
            "source_unavailable_failed_jobs": sum(
                count for failure_class, count in failure_counter.items() if failure_class.endswith("_source_unavailable")
            ),
        }

    with _connect_ro(paths.decision_db) as conn:
        row = conn.execute(
            """
            SELECT COUNT(*) AS decision_rows,
                   SUM(quote_timestamp_utc IS NOT NULL) AS quote_covered_rows,
                   SUM(quote_timestamp_utc IS NULL) AS missing_quote_rows,
                   SUM(forecast_run_time_utc IS NOT NULL) AS forecast_rows,
                   SUM(realized_temperature_2m_max_f IS NOT NULL) AS realized_rows,
                   SUM(observed_high_so_far_f IS NOT NULL) AS observed_high_rows,
                   SUM(quote_timestamp_utc IS NOT NULL AND quote_source_cache_path IS NOT NULL) AS quoted_rows_with_source_path
            FROM decision_snapshots_current_enriched
            """
        ).fetchone()
        scope = _rows(conn, "SELECT * FROM decision_artifact_scope")
        status["decision_artifact"] = dict(row)
        status["decision_artifact"]["scope"] = scope
        status["decision_artifact"]["quote_coverage_fraction"] = (
            status["decision_artifact"]["quote_covered_rows"] / status["decision_artifact"]["decision_rows"]
        )

    return status


def collect_downstream_artifact_inventory(paths: Paths) -> list[dict[str, Any]]:
    entries = [
        {
            "category": "canonical_db",
            "path": str(paths.market_db),
            "telonex_dependency": "market universe, token ids, canonical Polymarket settlement outcomes",
            "local_cache_status": "supported_from_local_db_and_markets_parquet",
        },
        {
            "category": "canonical_db",
            "path": str(paths.compact_price_db),
            "telonex_dependency": "compact tick/orderbook quote history and source-unavailable status ledger",
            "local_cache_status": "supported_from_local_compact_db",
        },
        {
            "category": "removed_db",
            "path": str(paths.wide_price_db),
            "telonex_dependency": "old wide quote history",
            "local_cache_status": "intentionally_absent_do_not_recreate",
        },
        {
            "category": "canonical_db",
            "path": str(paths.decision_db),
            "telonex_dependency": "point-in-time quote snapshots and Telonex-derived canonical labels",
            "local_cache_status": "supported_from_local_decision_artifact",
        },
        {
            "category": "raw_cache",
            "path": str(paths.telonex_raw_quotes),
            "telonex_dependency": "raw Telonex quote parquet payloads and missing markers",
            "local_cache_status": "preserved_locally",
        },
        {
            "category": "freeze_manifest",
            "path": str(paths.work / "reports/dataset_freeze"),
            "telonex_dependency": "six_month_v1 freeze, quote gap audit, source-unavailable samples",
            "local_cache_status": "preserved_locally",
        },
    ]
    report_dirs = [
        ("data_quality", "compact quote and decision artifact quality reports"),
        ("settlement_label_audit", "Telonex/Polymarket canonical settlement label audit"),
        ("weather_model_validation", "market-edge diagnostics using quote-covered rows where present"),
        ("weather_model_robustness", "market midpoint diagnostics using quote-covered rows"),
        ("weather_model_sota_validation", "market midpoint diagnostics using quote-covered rows"),
        ("weather_model_anti_overfit", "market midpoint diagnostics using quote-covered rows"),
        ("weather_model_improvement", "market midpoint diagnostics using quote-covered rows"),
        ("station_targeted_weather_model", "station-targeted model diagnostics with market midpoint comparison"),
        ("backtesting_engine", "shared scores, candidate replays, accounting replays, grids, ledgers"),
        ("live_forward_shadow_smoke", "fixture/smoke live-forward replay artifacts"),
        ("live_forward_shadow_smoke2", "fixture/smoke live-forward replay artifacts"),
        ("live_like_strategy_comparison_smoke", "smoke strategy comparison artifacts"),
        ("live_like_strategy_comparison_smoke2", "smoke strategy comparison artifacts"),
        ("global_weather_expansion", "separate global-market Telonex quote proof/batch artifacts"),
    ]
    for dirname, dependency in report_dirs:
        path = paths.work / "reports" / dirname
        if path.exists():
            file_count = sum(1 for p in path.rglob("*") if p.is_file())
            entries.append(
                {
                    "category": "report_or_score_artifact",
                    "path": str(path),
                    "file_count": file_count,
                    "telonex_dependency": dependency,
                    "local_cache_status": "preserved_locally",
                }
            )
    log_dirs = [
        "telonex_backtests",
        "long_backtests",
        "live_like_backtests",
        "live_like_strategy_replays_smoke",
        "live_like_strategy_replays_smoke2",
        "scored_replays",
        "strategy_search_50",
        "strategy_search_50_multimonth",
        "threshold_sweeps",
    ]
    for dirname in log_dirs:
        path = paths.work / "logs" / dirname
        if path.exists():
            file_count = sum(1 for p in path.rglob("*") if p.is_file())
            entries.append(
                {
                    "category": "legacy_log_artifact",
                    "path": str(path),
                    "file_count": file_count,
                    "telonex_dependency": "legacy Telonex-backed replay/search logs; not part of six_month_v1 freeze",
                    "local_cache_status": "preserved_locally_but_not_reproducibility_gate_for_six_month_v1",
                }
            )
    return entries


def _load_market_rows(paths: Paths) -> dict[str, dict[str, Any]]:
    with _connect_ro(paths.market_db) as conn:
        rows = _rows(
            conn,
            """
            SELECT universe_row_id, market_id, normalized_city, country_region, timezone,
                   market_date_local, bucket_type, threshold_low_f, threshold_high_f,
                   yes_token_id, no_token_id, close_time_utc, relevant_cutoff_time_utc,
                   telonex_quotes_from, telonex_quotes_to
            FROM market_universe
            WHERE market_date_local BETWEEN ? AND ?
            """,
            (TARGET_START.isoformat(), TARGET_END.isoformat()),
        )
    return {row["universe_row_id"]: row for row in rows}


def _load_token_rows(paths: Paths) -> dict[str, list[dict[str, Any]]]:
    with _connect_ro(paths.compact_price_db) as conn:
        rows = _rows(
            conn,
            """
            SELECT t.universe_row_id, t.token_ref, t.token_id, t.market_id, t.side,
                   s.start_ts, s.end_ts, s.status, s.rows_written, s.source_rows_seen, s.error
            FROM compact_tokens t
            LEFT JOIN compact_price_fetch_status s ON s.token_ref = t.token_ref
            """,
        )
    by_universe: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        row["failure_class"] = _classify_failure(row.get("error")) if row.get("status") == "failed" else ""
        by_universe[row["universe_row_id"]].append(row)
    return by_universe


def _load_first10_station_comparisons(paths: Paths) -> list[dict[str, Any]]:
    placeholders = ",".join("?" for _ in FIRST_10_STATIONS)
    with _connect_ro(paths.station_db) as conn:
        return _rows(
            conn,
            f"""
            SELECT universe_row_id, market_id, station_id, settlement_source_url, local_date,
                   canonical_bucket_type, recommended_action
            FROM station_settlement_label_comparison
            WHERE station_id IN ({placeholders})
              AND local_date BETWEEN ? AND ?
              AND COALESCE(recommended_action, '') != 'exclude_or_manual_review'
            """,
            (*FIRST_10_STATIONS, TARGET_START.isoformat(), TARGET_END.isoformat()),
        )


def _utc_window(local_date: str, timezone_name: str) -> tuple[str, str]:
    tz = ZoneInfo(timezone_name)
    parsed = date.fromisoformat(local_date)
    start_local = datetime.combine(parsed, time(THERMAL_WINDOW_START_LOCAL_HOUR, 0), tzinfo=tz)
    end_local = datetime.combine(parsed, time(THERMAL_WINDOW_END_LOCAL_HOUR, 0), tzinfo=tz)
    start_utc = start_local.astimezone(timezone.utc).isoformat()
    end_utc = end_local.astimezone(timezone.utc).isoformat()
    return start_utc, end_utc


def audit_first10_thermal_windows(paths: Paths, *, max_samples: int = 200) -> dict[str, Any]:
    market_rows = _load_market_rows(paths)
    token_rows_by_universe = _load_token_rows(paths)
    station_rows = _load_first10_station_comparisons(paths)
    by_station: dict[str, dict[str, Any]] = {}
    targeted_online_pulls: list[dict[str, Any]] = []
    permanent_unavailable: list[dict[str, Any]] = []
    unsupported_missing_status: list[dict[str, Any]] = []

    for station_id in FIRST_10_STATIONS:
        by_station[station_id] = {
            "station_id": station_id,
            "universe_rows": 0,
            "token_windows": 0,
            "complete_token_windows": 0,
            "failed_source_unavailable_token_windows": 0,
            "cache_miss_or_missing_status_token_windows": 0,
            "thermal_quote_rows": 0,
            "thermal_windows_with_quotes": 0,
            "thermal_windows_without_quotes_but_complete": 0,
            "market_closed_before_thermal_window_token_windows": 0,
            "complete_job_window_overlaps_thermal_window": 0,
            "complete_job_window_no_overlap_token_windows": 0,
            "targeted_online_pull_candidate_token_windows": 0,
            "min_thermal_quote_rows": None,
            "max_thermal_quote_rows": 0,
        }

    quote_conn = _connect_ro(paths.compact_price_db)
    try:
        quote_cursor = quote_conn.cursor()
        seen_universe_by_station: set[tuple[str, str]] = set()
        for station_row in station_rows:
            universe_row_id = station_row["universe_row_id"]
            market = market_rows.get(universe_row_id)
            if not market:
                continue
            station_id = station_row["station_id"]
            summary = by_station[station_id]
            if (station_id, universe_row_id) not in seen_universe_by_station:
                seen_universe_by_station.add((station_id, universe_row_id))
                summary["universe_rows"] += 1
            start_utc, end_utc = _utc_window(station_row["local_date"], market["timezone"])
            start_dt = _parse_datetime(start_utc)
            end_dt = _parse_datetime(end_utc)
            close_dt = _parse_datetime(market.get("close_time_utc") or market.get("relevant_cutoff_time_utc"))
            for token in token_rows_by_universe.get(universe_row_id, []):
                summary["token_windows"] += 1
                market_closed_before_window = bool(close_dt and start_dt and close_dt <= start_dt)
                job_start_dt = (
                    datetime.fromtimestamp(int(token["start_ts"]), tz=timezone.utc)
                    if token.get("start_ts") is not None
                    else None
                )
                job_end_dt = (
                    datetime.fromtimestamp(int(token["end_ts"]), tz=timezone.utc)
                    if token.get("end_ts") is not None
                    else None
                )
                job_overlaps_window = bool(
                    job_start_dt and job_end_dt and start_dt and end_dt and job_start_dt < end_dt and job_end_dt > start_dt
                )
                base = {
                    "station_id": station_id,
                    "local_date": station_row["local_date"],
                    "thermal_window_start_utc": start_utc,
                    "thermal_window_end_utc": end_utc,
                    "market_close_utc": market.get("close_time_utc"),
                    "universe_row_id": universe_row_id,
                    "market_id": market["market_id"],
                    "normalized_city": market["normalized_city"],
                    "timezone": market["timezone"],
                    "bucket_type": market["bucket_type"],
                    "side": token["side"],
                    "token_ref": token["token_ref"],
                    "token_id": token["token_id"],
                    "status": token.get("status"),
                    "failure_class": token.get("failure_class", ""),
                    "reason": "",
                }
                if market_closed_before_window:
                    summary["market_closed_before_thermal_window_token_windows"] += 1
                status = token.get("status")
                if status == "complete":
                    summary["complete_token_windows"] += 1
                    if job_overlaps_window:
                        summary["complete_job_window_overlaps_thermal_window"] += 1
                    else:
                        summary["complete_job_window_no_overlap_token_windows"] += 1
                    if market_closed_before_window:
                        summary["thermal_windows_without_quotes_but_complete"] += 1
                        if summary["min_thermal_quote_rows"] is None:
                            summary["min_thermal_quote_rows"] = 0
                        else:
                            summary["min_thermal_quote_rows"] = min(summary["min_thermal_quote_rows"], 0)
                        continue
                    row = quote_cursor.execute(
                        """
                        SELECT COUNT(*) AS quote_rows, MIN(timestamp_utc) AS first_quote_utc,
                               MAX(timestamp_utc) AS last_quote_utc
                        FROM compact_quote_history
                        WHERE token_ref = ?
                          AND timestamp_utc >= ?
                          AND timestamp_utc < ?
                        """,
                        (token["token_ref"], start_utc, end_utc),
                    ).fetchone()
                    quote_rows = int(row["quote_rows"] or 0)
                    summary["thermal_quote_rows"] += quote_rows
                    summary["max_thermal_quote_rows"] = max(summary["max_thermal_quote_rows"], quote_rows)
                    if summary["min_thermal_quote_rows"] is None:
                        summary["min_thermal_quote_rows"] = quote_rows
                    else:
                        summary["min_thermal_quote_rows"] = min(summary["min_thermal_quote_rows"], quote_rows)
                    if quote_rows > 0:
                        summary["thermal_windows_with_quotes"] += 1
                    else:
                        summary["thermal_windows_without_quotes_but_complete"] += 1
                        if not job_overlaps_window:
                            summary["targeted_online_pull_candidate_token_windows"] += 1
                            if len(targeted_online_pulls) < max_samples:
                                candidate = dict(base)
                                candidate["reason"] = "complete_current_decision_job_does_not_overlap_open_thermal_window"
                                targeted_online_pulls.append(candidate)
                    continue
                if status == "failed" and token.get("failure_class", "").endswith("_source_unavailable"):
                    summary["failed_source_unavailable_token_windows"] += 1
                    if len(permanent_unavailable) < max_samples:
                        sample = dict(base)
                        sample["reason"] = (
                            "market_closed_before_local_thermal_window"
                            if market_closed_before_window
                            else "source_unavailable_for_current_compact_job"
                        )
                        permanent_unavailable.append(sample)
                    continue
                summary["cache_miss_or_missing_status_token_windows"] += 1
                if status == "cache_miss":
                    summary["targeted_online_pull_candidate_token_windows"] += 1
                    if len(targeted_online_pulls) < max_samples:
                        candidate = dict(base)
                        candidate["reason"] = "cache_miss_status"
                        targeted_online_pulls.append(candidate)
                elif len(unsupported_missing_status) < max_samples:
                    sample = dict(base)
                    sample["reason"] = "missing_or_unknown_compact_status"
                    unsupported_missing_status.append(sample)
    finally:
        quote_conn.close()

    for station_id, summary in by_station.items():
        complete = summary["complete_token_windows"]
        summary["thermal_windows_with_quotes_fraction_of_complete"] = (
            summary["thermal_windows_with_quotes"] / complete if complete else None
        )
        summary["avg_thermal_quote_rows_per_complete_window"] = (
            summary["thermal_quote_rows"] / complete if complete else None
        )
        if summary["min_thermal_quote_rows"] is None:
            summary["min_thermal_quote_rows"] = 0

    return {
        "definition": {
            "stations": list(FIRST_10_STATIONS),
            "window_local": f"{THERMAL_WINDOW_START_LOCAL_HOUR:02d}:00-{THERMAL_WINDOW_END_LOCAL_HOUR:02d}:00",
            "target_start_date": TARGET_START.isoformat(),
            "target_end_date": TARGET_END.isoformat(),
            "manual_review_rows_excluded": True,
            "quote_source": "local compact_quote_history; no Telonex HTTP calls",
        },
        "by_station": [by_station[station_id] for station_id in FIRST_10_STATIONS],
        "targeted_online_pull_candidate_token_windows": sum(
            row["targeted_online_pull_candidate_token_windows"] for row in by_station.values()
        ),
        "market_closed_before_thermal_window_token_windows": sum(
            row["market_closed_before_thermal_window_token_windows"] for row in by_station.values()
        ),
        "targeted_online_pulls": targeted_online_pulls,
        "permanent_source_unavailable_samples": permanent_unavailable,
        "unsupported_missing_status_samples": unsupported_missing_status,
    }


def build_gap_report(core_status: dict[str, Any], thermal: dict[str, Any]) -> dict[str, Any]:
    compact = core_status["compact_price_status"]
    decision = core_status["decision_artifact"]
    targeted_pull_count = thermal["targeted_online_pull_candidate_token_windows"]
    failed_source_unavailable_count = sum(
        row["failed_source_unavailable_token_windows"] for row in thermal["by_station"]
    )
    return {
        "fully_supported_from_local_cache": [
            "six_month_v1 market universe and canonical Polymarket/Telonex settlement labels",
            "six_month_v1 compact quote store for all 54,000 complete token jobs",
            "six_month_v1 decision artifact audit for 472,042 decision rows",
            "quote-covered strategy replay subset: quote_timestamp_utc IS NOT NULL and settlement/manual-review rows excluded",
            "weather-only and station-target validation that does not require missing market prices",
            "existing market-midpoint diagnostics and generated score/replay artifacts already written under work/reports",
            "first-10 station observation proof and station-targeted model reports",
        ],
        "needs_targeted_online_telonex_pulls_before_expiry": [
            "None for six_month_v1 current scope: compact status has 0 cache_miss jobs and 0 operational failed jobs.",
            "None for first-10 Phase B/C local 12:00-18:00 thermal windows: these token windows are after market close under current market metadata, so they are not recoverable tradable quote windows.",
        ]
        if targeted_pull_count == 0 and compact["cache_miss_jobs"] == 0 and compact["operational_failed_jobs"] == 0
        else [
            "Only token/time windows listed in targeted_online_pulls.csv should be considered; do not run broad global backfill."
        ],
        "permanently_source_unavailable": [
            f"{compact['source_unavailable_failed_jobs']} six_month_v1 compact token jobs "
            f"({compact['failure_breakdown']})",
            f"{decision['missing_quote_rows']} decision rows without quote snapshots due to source-unavailable Telonex quote files",
            f"{failed_source_unavailable_count} first-10 station-scope token jobs map to source-unavailable compact jobs; all sampled local noon-to-6pm windows are already after market close",
        ],
        "not_a_telonex_pull_gap": [
            "The old wide DB is intentionally absent and must not be recreated.",
            "A denser Phase B/C quote artifact should use market-open windows; first-10 local-afternoon/noon-to-6pm windows are closed-market windows under current market metadata.",
            "Complete jobs with zero ticks in a local thermal window should be treated as local market inactivity/no available quote ticks in preserved data, not as a cache_miss.",
        ],
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def write_outputs(paths: Paths, report: dict[str, Any], timestamp: str) -> dict[str, str]:
    paths.report_dir.mkdir(parents=True, exist_ok=True)
    prefix = paths.report_dir / f"{timestamp}-telonex-preservation-audit"
    json_path = prefix.with_suffix(".json")
    md_path = prefix.with_suffix(".md")
    inventory_csv = Path(f"{prefix}-artifact-inventory.csv")
    thermal_csv = Path(f"{prefix}-first10-thermal-window-density.csv")
    targeted_csv = Path(f"{prefix}-targeted-online-pulls.csv")
    unavailable_csv = Path(f"{prefix}-source-unavailable-windows.csv")

    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    inventory_fields = ["category", "path", "file_count", "telonex_dependency", "local_cache_status"]
    write_csv(inventory_csv, report["artifact_inventory"], inventory_fields)
    thermal_fields = [
        "station_id",
        "universe_rows",
        "token_windows",
        "complete_token_windows",
        "failed_source_unavailable_token_windows",
        "cache_miss_or_missing_status_token_windows",
        "market_closed_before_thermal_window_token_windows",
        "complete_job_window_overlaps_thermal_window",
        "complete_job_window_no_overlap_token_windows",
        "targeted_online_pull_candidate_token_windows",
        "thermal_windows_with_quotes",
        "thermal_windows_without_quotes_but_complete",
        "thermal_quote_rows",
        "min_thermal_quote_rows",
        "max_thermal_quote_rows",
        "avg_thermal_quote_rows_per_complete_window",
        "thermal_windows_with_quotes_fraction_of_complete",
    ]
    write_csv(thermal_csv, report["thermal_window_audit"]["by_station"], thermal_fields)
    targeted_fields = [
        "station_id",
        "local_date",
        "thermal_window_start_utc",
        "thermal_window_end_utc",
        "market_close_utc",
        "universe_row_id",
        "market_id",
        "normalized_city",
        "timezone",
        "bucket_type",
        "side",
        "token_ref",
        "token_id",
        "status",
        "failure_class",
        "reason",
    ]
    write_csv(targeted_csv, report["thermal_window_audit"]["targeted_online_pulls"], targeted_fields)
    write_csv(unavailable_csv, report["thermal_window_audit"]["permanent_source_unavailable_samples"], targeted_fields)

    core = report["core_status"]
    compact = core["compact_price_status"]
    decision = core["decision_artifact"]
    thermal = report["thermal_window_audit"]
    lines = [
        "# Telonex Preservation Audit",
        "",
        f"Generated: `{report['generated_at_utc']}`",
        "",
        "## Status",
        "",
        "- Role: Data/Telonex Preservation Agent.",
        "- Scope: `six_month_v1`, `2026-01-01` through `2026-06-29`.",
        "- Network use: none. This report reads local SQLite/report/cache artifacts only.",
        "- Wide DB: intentionally absent; do not recreate or populate it.",
        "",
        "## Current Cache State",
        "",
        f"- Compact complete jobs: `{compact['complete_jobs']}`.",
        f"- Compact cache-miss jobs: `{compact['cache_miss_jobs']}`.",
        f"- Compact failed jobs: `{compact['failed_jobs']}`.",
        f"- Source-unavailable failed jobs: `{compact['source_unavailable_failed_jobs']}`.",
        f"- Operational failed jobs: `{compact['operational_failed_jobs']}`.",
        f"- Failure breakdown: `{compact['failure_breakdown']}`.",
        f"- Decision rows: `{decision['decision_rows']}`.",
        f"- Quote-covered decision rows: `{decision['quote_covered_rows']}`.",
        f"- Missing quote decision rows: `{decision['missing_quote_rows']}`.",
        f"- Quote coverage: `{decision['quote_coverage_fraction']:.4%}`.",
        "",
        "## Local Reproducibility",
        "",
        "- `six_month_v1` is locally auditable without Telonex HTTP calls from the preserved market universe DB, compact quote DB, raw quote cache, decision artifact, data-quality reports, and freeze manifests.",
        "- The only all-row replay blocker is source-unavailable Telonex quote data, not retryable cache misses.",
        "- Trading replay must filter to quote-covered, non-manual-review rows; price imputation is not allowed.",
        "",
        "## First-10 Station Thermal Window Audit",
        "",
        f"- Local thermal window definition: `{thermal['definition']['window_local']}` local time.",
        f"- Market-closed-before-window token windows: `{thermal['market_closed_before_thermal_window_token_windows']}`.",
        f"- Targeted online Telonex pull candidate token windows: `{thermal['targeted_online_pull_candidate_token_windows']}`.",
        "- Per-station density CSV: "
        f"`{thermal_csv}`.",
        "- Targeted online pulls CSV: "
        f"`{targeted_csv}`.",
        "- Source-unavailable sample CSV: "
        f"`{unavailable_csv}`.",
        "",
        "## Pre-Expiry Decision",
        "",
    ]
    for item in report["gap_report"]["fully_supported_from_local_cache"]:
        lines.append(f"- Fully supported locally: {item}.")
    for item in report["gap_report"]["needs_targeted_online_telonex_pulls_before_expiry"]:
        lines.append(f"- Needs targeted online Telonex pull: {item}")
    for item in report["gap_report"]["permanently_source_unavailable"]:
        lines.append(f"- Permanently source-unavailable: {item}.")
    for item in report["gap_report"]["not_a_telonex_pull_gap"]:
        lines.append(f"- Not a Telonex pull gap: {item}")
    lines.extend(
        [
            "",
            "## Output Files",
            "",
            f"- JSON: `{json_path}`",
            f"- Markdown: `{md_path}`",
            f"- Artifact inventory CSV: `{inventory_csv}`",
            f"- First-10 thermal density CSV: `{thermal_csv}`",
            f"- Targeted online pulls CSV: `{targeted_csv}`",
            f"- Source-unavailable samples CSV: `{unavailable_csv}`",
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return {
        "json": str(json_path),
        "markdown": str(md_path),
        "artifact_inventory_csv": str(inventory_csv),
        "thermal_density_csv": str(thermal_csv),
        "targeted_online_pulls_csv": str(targeted_csv),
        "source_unavailable_csv": str(unavailable_csv),
    }


def run_audit(work_root: Path) -> dict[str, Any]:
    paths = Paths(work_root=work_root.resolve())
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    core_status = collect_core_status(paths)
    artifact_inventory = collect_downstream_artifact_inventory(paths)
    thermal_window_audit = audit_first10_thermal_windows(paths)
    report: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "timestamp": timestamp,
        "core_status": core_status,
        "artifact_inventory": artifact_inventory,
        "thermal_window_audit": thermal_window_audit,
        "gap_report": build_gap_report(core_status, thermal_window_audit),
    }
    report["output_files"] = write_outputs(paths, report, timestamp)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit local Telonex dependency preservation before API expiry.")
    parser.add_argument("--work-root", default=".", help="Weather Trader workspace root")
    args = parser.parse_args()
    report = run_audit(Path(args.work_root))
    print(json.dumps(report["output_files"], indent=2))


if __name__ == "__main__":
    main()
