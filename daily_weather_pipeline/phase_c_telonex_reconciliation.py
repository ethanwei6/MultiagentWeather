from __future__ import annotations

import argparse
import csv
import json
import shlex
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from .cli import (
    DEFAULT_SOURCE_REPO,
    PipelinePaths,
    _compact_token_refs,
    _connect,
    _ensure_compact_price_schema,
    _finish_pipeline_run,
    _insert_compact_quote_rows,
    _insert_pipeline_run,
    _load_quote_rows_for_job,
    _quote_cache_dirs,
    _upsert_compact_status,
    bootstrap,
)


DEFAULT_PHASE_C_PULL_LIST = Path(
    "work/reports/phase_c_convergence/20260707T163004Z-phase-c-convergence-measurement-targeted-telonex-pull-list.csv"
)
DEFAULT_PRESERVATION_PULL_LIST = Path(
    "work/reports/telonex_preservation/20260707T162817Z-telonex-preservation-audit-targeted-online-pulls.csv"
)
DEFAULT_OUTPUT_DIR = Path("work/reports/phase_c_telonex_reconciliation")
DEFAULT_PRICE_DB = Path("work/data/backtest/daily_weather_price_history_compact.sqlite")
DEFAULT_UNIVERSE_DB = Path("work/data/backtest/daily_weather_market_universe.sqlite")

CLASS_AFTER_MARKET = "after-market/closed-window"
CLASS_COMPLETE_SPARSE = "already complete but sparse"
CLASS_SOURCE_UNAVAILABLE = "source-unavailable"
CLASS_METADATA_BUG = "metadata/window bug"
CLASS_GENUINE_PULLABLE = "genuinely pullable/useful before API expiry"

MATERIAL_COVERAGE_SECONDS = 60


def run_reconciliation(
    *,
    work_root: Path,
    phase_c_pull_list: Path,
    preservation_pull_list: Path,
    output_dir: Path,
) -> dict[str, Any]:
    root = work_root.resolve()
    price_db = _resolve(root, DEFAULT_PRICE_DB)
    universe_db = _resolve(root, DEFAULT_UNIVERSE_DB)
    phase_c_csv = _resolve(root, phase_c_pull_list)
    preservation_csv = _resolve(root, preservation_pull_list)
    out_dir = _resolve(root, output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_csv(phase_c_csv)
    preservation_rows = _read_csv(preservation_csv) if preservation_csv.exists() else []
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    reconciled = _reconcile_rows(rows, price_db=price_db, universe_db=universe_db)
    genuine_manifest = [row for row in reconciled if row["classification"] == CLASS_GENUINE_PULLABLE]

    summary = _summarize(reconciled, preservation_rows)
    recommendation = _recommendation(summary)
    prefix = out_dir / f"{timestamp}-phase-c-telonex-reconciliation"
    row_csv = Path(f"{prefix}-rows.csv")
    manifest_csv = Path(f"{prefix}-genuine-pull-manifest.csv")
    examples_csv = Path(f"{prefix}-examples.csv")
    json_path = Path(f"{prefix}.json")
    md_path = Path(f"{prefix}.md")

    _write_csv(row_csv, reconciled)
    _write_csv(manifest_csv, genuine_manifest, fieldnames=_manifest_fields())
    _write_csv(examples_csv, _examples(reconciled, per_class=5))

    pull_command = _pull_command(manifest_csv, len(genuine_manifest)) if genuine_manifest else ""
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "role": "DailyWeather Telonex Reconciliation Agent",
        "network_use": "none",
        "inputs": {
            "phase_c_pull_list": str(phase_c_csv),
            "preservation_pull_list": str(preservation_csv),
            "compact_price_db": str(price_db),
            "market_universe_db": str(universe_db),
        },
        "summary": summary,
        "recommendation": recommendation,
        "pull_command_for_later_approval": pull_command,
        "artifacts": {
            "json": str(json_path),
            "markdown": str(md_path),
            "row_reconciliation_csv": str(row_csv),
            "genuine_pull_manifest_csv": str(manifest_csv),
            "examples_csv": str(examples_csv),
        },
    }
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(_render_markdown(report, reconciled, genuine_manifest), encoding="utf-8")
    return report


def run_targeted_pull(
    *,
    work_root: Path,
    source_repo: Path,
    manifest: Path,
    limit_jobs: int,
    workers: int,
    progress_every: int,
    telonex_env: Optional[Path],
    hard_timeout_seconds: float,
    execute_online_pull: bool,
) -> dict[str, Any]:
    root = work_root.resolve()
    manifest_path = _resolve(root, manifest)
    rows = _read_csv(manifest_path)
    jobs = _jobs_from_manifest(rows, limit_jobs=limit_jobs)
    if not execute_online_pull:
        return {
            "status": "dry_run_only",
            "manifest": str(manifest_path),
            "jobs": len(jobs),
            "note": "Pass --execute-online-pull only after explicit approval; this command would call Telonex and mutate the compact DB.",
        }
    if not jobs:
        return {"status": "ok", "manifest": str(manifest_path), "jobs": 0}

    paths = PipelinePaths(root)
    bootstrap(paths, source_repo)
    from weather_strategy.telonex import TelonexClient

    hard_timeout = hard_timeout_seconds if hard_timeout_seconds > 0 else None
    client = TelonexClient(
        cache_dir=paths.telonex_raw,
        env_path=str(telonex_env or ".env"),
        hard_timeout_seconds=hard_timeout,
    )
    quote_cache_dirs = _quote_cache_dirs(paths)
    started = time.monotonic()

    with _connect(paths.compact_price_db) as compact_conn:
        compact_conn.row_factory = sqlite3.Row
        _ensure_compact_price_schema(compact_conn)
        run_id = _insert_pipeline_run(
            compact_conn,
            "phase-c-targeted-telonex-pull",
            {
                "manifest": str(manifest_path),
                "jobs": len(jobs),
                "limit_jobs": limit_jobs,
                "workers": workers,
            },
        )
        compact_conn.commit()

    def fetch(job: dict[str, Any]) -> dict[str, Any]:
        try:
            rows_for_job = _load_quote_rows_for_job(
                job,
                client=client,
                cache_only=False,
                quote_cache_dirs=quote_cache_dirs,
            )
            return {"status": "complete", "job": job, "rows": rows_for_job}
        except Exception as exc:  # noqa: BLE001
            return {"status": "failed", "job": job, "rows": [], "error": f"{type(exc).__name__}: {str(exc)[:500]}"}

    max_workers = max(1, min(workers, len(jobs)))
    results: list[dict[str, Any]] = []
    if max_workers == 1:
        for index, job in enumerate(jobs, start=1):
            results.append(fetch(job))
            if progress_every > 0 and index % progress_every == 0:
                print(
                    f"phase_c_targeted_telonex_pull_jobs={index}/{len(jobs)} elapsed_seconds={time.monotonic() - started:.1f}",
                    file=sys.stderr,
                )
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(fetch, job) for job in jobs]
            for index, future in enumerate(as_completed(futures), start=1):
                results.append(future.result())
                if progress_every > 0 and index % progress_every == 0:
                    print(
                        f"phase_c_targeted_telonex_pull_jobs={index}/{len(jobs)} elapsed_seconds={time.monotonic() - started:.1f}",
                        file=sys.stderr,
                    )

    complete = 0
    failed = 0
    source_rows_seen = 0
    unique_rows_inserted = 0
    duplicates = 0
    with _connect(paths.compact_price_db) as compact_conn:
        compact_conn.row_factory = sqlite3.Row
        _ensure_compact_price_schema(compact_conn)
        token_refs = _compact_token_refs(compact_conn)
        source_ref_cache: dict[str, int] = {}
        for result in results:
            job = result["job"]
            token_ref = token_refs[str(job["token_id"])]
            if result["status"] != "complete":
                failed += 1
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
                continue
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
            complete += 1
            source_rows_seen += metrics["source_rows_seen"]
            unique_rows_inserted += metrics["unique_rows_inserted"]
            duplicates += metrics["duplicate_rows_collapsed"]
        _finish_pipeline_run(compact_conn, run_id, row_count=unique_rows_inserted)
        compact_conn.commit()

    return {
        "status": "ok",
        "manifest": str(manifest_path),
        "jobs": len(jobs),
        "complete_jobs": complete,
        "failed_jobs": failed,
        "source_rows_seen": source_rows_seen,
        "unique_rows_inserted": unique_rows_inserted,
        "duplicate_rows_collapsed": duplicates,
        "elapsed_seconds": round(time.monotonic() - started, 2),
    }


def _reconcile_rows(rows: Sequence[Mapping[str, str]], *, price_db: Path, universe_db: Path) -> list[dict[str, Any]]:
    with sqlite3.connect(f"file:{price_db}?mode=ro", uri=True) as price_conn, sqlite3.connect(
        f"file:{universe_db}?mode=ro", uri=True
    ) as universe_conn:
        price_conn.row_factory = sqlite3.Row
        universe_conn.row_factory = sqlite3.Row
        output = []
        for row in rows:
            output.append(_reconcile_row(dict(row), price_conn=price_conn, universe_conn=universe_conn))
        return output


def _reconcile_row(row: dict[str, Any], *, price_conn: sqlite3.Connection, universe_conn: sqlite3.Connection) -> dict[str, Any]:
    token_ref = _optional_int(row.get("token_ref"))
    event_time = _parse_dt(row.get("event_time_utc"))
    suggested_start = _parse_dt(row.get("suggested_pull_start_utc"))
    suggested_end = _parse_dt(row.get("suggested_pull_end_utc"))
    market = universe_conn.execute(
        """
        SELECT universe_row_id, market_id, normalized_city, timezone, market_date_local,
               start_at_utc, close_time_utc, relevant_cutoff_time_utc,
               telonex_quotes_from, telonex_quotes_to
        FROM market_universe
        WHERE universe_row_id = ?
        """,
        (row.get("universe_row_id"),),
    ).fetchone()
    market_start = _parse_dt(market["start_at_utc"]) if market else None
    market_close = _parse_dt((market["close_time_utc"] or market["relevant_cutoff_time_utc"]) if market else None)

    status_rows = _status_rows(price_conn, token_ref)
    failure_classes = sorted({_failure_class(status.get("error")) for status in status_rows if status.get("status") == "failed"})
    status_summary = _status_summary(status_rows)
    quote_window = _quote_window_counts(price_conn, token_ref, suggested_start, suggested_end)
    all_token_quotes = _all_token_quote_bounds(price_conn, token_ref)

    suggested_open_overlap_seconds = _overlap_seconds(suggested_start, suggested_end, market_start, market_close)
    post_fact_open_overlap_seconds = _overlap_seconds(event_time, suggested_end, event_time, market_close)
    complete_overlap_seconds = _complete_overlap_seconds(status_rows, suggested_start, suggested_end)
    complete_post_fact_open_overlap_seconds = _complete_overlap_seconds(
        status_rows,
        max_dt(event_time, suggested_start, market_start),
        min_dt(suggested_end, market_close),
    )
    complete_status_count = sum(1 for item in status_rows if item.get("status") == "complete")
    cache_miss_status_count = sum(1 for item in status_rows if item.get("status") == "cache_miss")
    failed_status_count = sum(1 for item in status_rows if item.get("status") == "failed")
    event_after_market_close = bool(event_time and market_close and event_time >= market_close)
    suggested_start_after_market_close = bool(suggested_start and market_close and suggested_start >= market_close)

    classification, rationale = _classify(
        market=market,
        token_ref=token_ref,
        event_time=event_time,
        suggested_start=suggested_start,
        suggested_end=suggested_end,
        market_close=market_close,
        complete_status_count=complete_status_count,
        cache_miss_status_count=cache_miss_status_count,
        failed_status_count=failed_status_count,
        failure_classes=failure_classes,
        quote_rows_suggested=quote_window["quote_rows_suggested_window"],
        post_fact_open_overlap_seconds=post_fact_open_overlap_seconds,
        complete_post_fact_open_overlap_seconds=complete_post_fact_open_overlap_seconds,
    )

    pull_start = max_dt(suggested_start, market_start)
    pull_end = min_dt(suggested_end, market_close)
    if classification != CLASS_GENUINE_PULLABLE:
        pull_start = None
        pull_end = None

    enriched = dict(row)
    enriched.update(
        {
            "classification": classification,
            "classification_rationale": rationale,
            "market_start_utc": _iso(market_start),
            "market_close_utc": _iso(market_close),
            "telonex_quotes_from": market["telonex_quotes_from"] if market else "",
            "telonex_quotes_to": market["telonex_quotes_to"] if market else "",
            "compact_status_summary": status_summary,
            "compact_complete_status_count": complete_status_count,
            "compact_cache_miss_status_count": cache_miss_status_count,
            "compact_failed_status_count": failed_status_count,
            "compact_failure_classes": ";".join(failure_classes),
            "compact_complete_window_start_utc": _iso(_status_bound(status_rows, "min")),
            "compact_complete_window_end_utc": _iso(_status_bound(status_rows, "max")),
            "event_after_market_close": int(event_after_market_close),
            "suggested_start_after_market_close": int(suggested_start_after_market_close),
            "suggested_window_open_overlap_seconds": round(suggested_open_overlap_seconds, 3),
            "post_fact_open_overlap_seconds": round(post_fact_open_overlap_seconds, 3),
            "compact_complete_overlap_suggested_seconds": round(complete_overlap_seconds, 3),
            "compact_complete_overlap_post_fact_open_seconds": round(complete_post_fact_open_overlap_seconds, 3),
            "quote_rows_suggested_window": quote_window["quote_rows_suggested_window"],
            "quote_rows_post_fact_open_window": _quote_count(
                price_conn,
                token_ref,
                max_dt(event_time, suggested_start, market_start),
                min_dt(suggested_end, market_close),
            ),
            "first_quote_suggested_window_utc": quote_window["first_quote_suggested_window_utc"],
            "last_quote_suggested_window_utc": quote_window["last_quote_suggested_window_utc"],
            "first_quote_token_utc": all_token_quotes["first_quote_token_utc"],
            "last_quote_token_utc": all_token_quotes["last_quote_token_utc"],
            "pull_start_utc": _iso(pull_start),
            "pull_end_utc": _iso(pull_end),
            "pull_window_seconds": round(_seconds_between(pull_start, pull_end), 3) if pull_start and pull_end else 0,
        }
    )
    return enriched


def _classify(
    *,
    market: Optional[sqlite3.Row],
    token_ref: Optional[int],
    event_time: Optional[datetime],
    suggested_start: Optional[datetime],
    suggested_end: Optional[datetime],
    market_close: Optional[datetime],
    complete_status_count: int,
    cache_miss_status_count: int,
    failed_status_count: int,
    failure_classes: Sequence[str],
    quote_rows_suggested: int,
    post_fact_open_overlap_seconds: float,
    complete_post_fact_open_overlap_seconds: float,
) -> tuple[str, str]:
    if not market or token_ref is None or not event_time or not suggested_start or not suggested_end or not market_close:
        return CLASS_METADATA_BUG, "missing market, token, timestamp, or market close metadata"
    if suggested_end <= suggested_start:
        return CLASS_METADATA_BUG, "suggested pull end is not after suggested pull start"
    if failed_status_count and not complete_status_count and any(item in {"http_404", "local_cache_missing"} for item in failure_classes):
        return CLASS_SOURCE_UNAVAILABLE, "compact status is failed with source-unavailable class"
    if event_time >= market_close or post_fact_open_overlap_seconds <= 0:
        return CLASS_AFTER_MARKET, "fact timestamp has no post-fact open-market overlap under market close metadata"
    if quote_rows_suggested > 0:
        return CLASS_COMPLETE_SPARSE, "local compact quotes already exist in the suggested window but miss the Phase C freshness/density gate"
    if complete_status_count and complete_post_fact_open_overlap_seconds >= MATERIAL_COVERAGE_SECONDS:
        return CLASS_COMPLETE_SPARSE, "compact complete status materially covers the post-fact open segment but no dense quotes exist"
    if cache_miss_status_count or complete_status_count:
        return CLASS_GENUINE_PULLABLE, "post-fact open-market segment is not materially covered by existing compact quote rows/status"
    if failed_status_count:
        return CLASS_METADATA_BUG, "failed compact status is not source-unavailable and should be reviewed before retry"
    return CLASS_METADATA_BUG, "no compact status row explains this token/window"


def _summarize(rows: Sequence[Mapping[str, Any]], preservation_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    classification_counts = Counter(str(row["classification"]) for row in rows)
    city_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        city_counts[str(row.get("city", ""))][str(row["classification"])] += 1
    status_counts = Counter(str(row.get("compact_status_summary", "")) for row in rows)
    event_class_counts = Counter(str(row.get("event_class", "")) for row in rows)
    fact_state_counts = Counter(str(row.get("fact_state", "")) for row in rows)
    source_unavailable = sum(1 for row in rows if row["classification"] == CLASS_SOURCE_UNAVAILABLE)
    metadata_bug = sum(1 for row in rows if row["classification"] == CLASS_METADATA_BUG)
    genuine = [row for row in rows if row["classification"] == CLASS_GENUINE_PULLABLE]
    return {
        "phase_c_rows": len(rows),
        "preservation_targeted_online_pull_rows": len(preservation_rows),
        "classification_counts": dict(sorted(classification_counts.items())),
        "compact_status_summary_counts": dict(sorted(status_counts.items())),
        "event_class_counts": dict(sorted(event_class_counts.items())),
        "fact_state_counts": dict(sorted(fact_state_counts.items())),
        "city_classification_counts": {
            city: dict(sorted(counts.items())) for city, counts in sorted(city_counts.items())
        },
        "rows_with_event_after_market_close": sum(int(row.get("event_after_market_close") or 0) for row in rows),
        "rows_with_suggested_start_after_market_close": sum(
            int(row.get("suggested_start_after_market_close") or 0) for row in rows
        ),
        "rows_with_post_fact_open_overlap": sum(float(row.get("post_fact_open_overlap_seconds") or 0) > 0 for row in rows),
        "rows_with_any_existing_quote_in_suggested_window": sum(
            int(row.get("quote_rows_suggested_window") or 0) > 0 for row in rows
        ),
        "source_unavailable_rows": source_unavailable,
        "metadata_or_window_bug_rows": metadata_bug,
        "genuinely_pullable_rows": len(genuine),
        "genuinely_pullable_unique_tokens": len({row.get("token_id") for row in genuine}),
        "genuinely_pullable_cities": dict(Counter(str(row.get("city", "")) for row in genuine)),
        "genuinely_pullable_total_pull_window_seconds": round(
            sum(float(row.get("pull_window_seconds") or 0) for row in genuine), 3
        ),
    }


def _recommendation(summary: Mapping[str, Any]) -> dict[str, Any]:
    genuine = int(summary.get("genuinely_pullable_rows") or 0)
    if genuine:
        return {
            "decision": "GO_SMALL_CAPPED_DIAGNOSTIC_PULL_ONLY",
            "rationale": (
                f"{genuine} rows have post-fact open-market time that is not materially covered by existing compact quotes. "
                "This is a narrow Phase C preservation/diagnostic pull, not a broad backfill and not strategy evidence."
            ),
            "strict_boundaries": [
                "Use only the generated genuine-pull manifest.",
                "Do not retry source-unavailable 404 or cached-missing rows.",
                "Do not run strategy/PnL tests from the pull result.",
                "Rerun Phase C measurement after any approved quote addition before making infrastructure decisions.",
            ],
        }
    return {
        "decision": "NO_GO_ON_PAID_TELONEX_PULL",
        "rationale": "No rows have a useful post-fact open-market window missing from local compact coverage.",
        "strict_boundaries": [
            "Do not run broad backfill.",
            "Do not retry source-unavailable 404 or cached-missing rows.",
        ],
    }


def _render_markdown(
    report: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    genuine_manifest: Sequence[Mapping[str, Any]],
) -> str:
    summary = report["summary"]
    recommendation = report["recommendation"]
    lines = [
        "# Phase C Telonex Reconciliation",
        "",
        f"Generated: `{report['generated_at_utc']}`",
        "",
        "## Scope",
        "",
        "- Network use: none for this reconciliation report.",
        f"- Phase C pull-list rows reconciled: `{summary['phase_c_rows']}`.",
        f"- Preservation targeted-online-pulls rows: `{summary['preservation_targeted_online_pull_rows']}`.",
        "- Boundaries: no broad backfill, no strategy/PnL tests, no weather-model work, no destructive DB actions.",
        "",
        "## Classification Counts",
        "",
        "| Classification | Rows |",
        "| --- | ---: |",
    ]
    for key, value in summary["classification_counts"].items():
        lines.append(f"| {key} | {value} |")
    lines.extend(
        [
            "",
            "## Key Evidence",
            "",
            f"- Compact status for the 624 rows: `{summary['compact_status_summary_counts']}`.",
            f"- Rows whose fact timestamp is already at/after market close: `{summary['rows_with_event_after_market_close']}`.",
            f"- Rows whose suggested pull start is already at/after market close: `{summary['rows_with_suggested_start_after_market_close']}`.",
            f"- Rows with post-fact open-market overlap: `{summary['rows_with_post_fact_open_overlap']}`.",
            f"- Rows with existing compact quotes inside the suggested window: `{summary['rows_with_any_existing_quote_in_suggested_window']}`.",
            f"- Source-unavailable rows in the 624-row list: `{summary['source_unavailable_rows']}`.",
            f"- Metadata/window bug rows: `{summary['metadata_or_window_bug_rows']}`.",
            "",
            "## Why The Two Reports Differ",
            "",
            "- Phase C called the CSV a targeted pull list because it was ranking sparse event windows from the compact quote table. It did not mean the compact token job was missing or that every row was market-open and pullable.",
            "- The preservation audit reported zero targeted online pulls for its narrower local `12:00-18:00` thermal-window audit and for compact status cleanup. Those local thermal windows are closed-market windows under the current metadata, and compact status has `0` cache-miss and `0` operational failures.",
            "- Reconciliation shows both statements were locally true for their scopes. The only new pullable subset is a small early-morning Phase C event-window subset, not the local-afternoon heat-window thesis.",
            "",
            "## Recommendation",
            "",
            f"- Decision: `{recommendation['decision']}`.",
            f"- Rationale: {recommendation['rationale']}",
        ]
    )
    for item in recommendation["strict_boundaries"]:
        lines.append(f"- Boundary: {item}")
    lines.extend(
        [
            "",
            "## Genuinely Pullable Rows",
            "",
            f"- Rows: `{summary['genuinely_pullable_rows']}`.",
            f"- Unique tokens: `{summary['genuinely_pullable_unique_tokens']}`.",
            f"- Cities: `{summary['genuinely_pullable_cities']}`.",
            f"- Total targeted window seconds: `{summary['genuinely_pullable_total_pull_window_seconds']}`.",
        ]
    )
    if genuine_manifest:
        lines.extend(["", "| Rank | City | Date | Event UTC | Pull Start UTC | Pull End UTC | Reason |", "| ---: | --- | --- | --- | --- | --- | --- |"])
        for row in genuine_manifest:
            lines.append(
                "| {rank} | {city} | {date} | {event} | {start} | {end} | {reason} |".format(
                    rank=row.get("priority_rank", ""),
                    city=row.get("city", ""),
                    date=row.get("market_date", ""),
                    event=row.get("event_time_utc", ""),
                    start=row.get("pull_start_utc", ""),
                    end=row.get("pull_end_utc", ""),
                    reason=row.get("classification_rationale", ""),
                )
            )
        lines.extend(
            [
                "",
                "Do not run this without explicit approval:",
                "",
                "```bash",
                report["pull_command_for_later_approval"],
                "```",
            ]
        )
    lines.extend(
        [
            "",
            "## Examples",
            "",
        ]
    )
    for row in _examples(rows, per_class=3):
        lines.append(
            "- `{classification}`: rank `{rank}`, {city} {date}, event `{event}`, close `{close}`, reason `{reason}`.".format(
                classification=row.get("classification"),
                rank=row.get("priority_rank"),
                city=row.get("city"),
                date=row.get("market_date"),
                event=row.get("event_time_utc"),
                close=row.get("market_close_utc"),
                reason=row.get("classification_rationale"),
            )
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- JSON: `{report['artifacts']['json']}`",
            f"- Row reconciliation CSV: `{report['artifacts']['row_reconciliation_csv']}`",
            f"- Genuine pull manifest CSV: `{report['artifacts']['genuine_pull_manifest_csv']}`",
            f"- Examples CSV: `{report['artifacts']['examples_csv']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def _jobs_from_manifest(rows: Sequence[Mapping[str, str]], *, limit_jobs: int) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int]] = set()
    for row in rows:
        token_id = str(row.get("token_id") or "")
        start = _parse_dt(row.get("pull_start_utc"))
        end = _parse_dt(row.get("pull_end_utc"))
        if not token_id or not start or not end or end <= start:
            continue
        key = (token_id, int(start.timestamp()), int(end.timestamp()))
        if key in seen:
            continue
        seen.add(key)
        jobs.append(
            {
                "token_id": token_id,
                "market_id": row.get("market_id"),
                "universe_row_id": row.get("universe_row_id"),
                "side": row.get("side"),
                "slug": "",
                "outcome": "Yes" if str(row.get("side", "")).upper() == "YES" else "No",
                "start_ts": key[1],
                "end_ts": key[2],
            }
        )
        if limit_jobs > 0 and len(jobs) >= limit_jobs:
            break
    return jobs


def _status_rows(conn: sqlite3.Connection, token_ref: Optional[int]) -> list[dict[str, Any]]:
    if token_ref is None:
        return []
    rows = conn.execute(
        """
        SELECT status, retryable, start_ts, end_ts, rows_written, source_rows_seen, error
        FROM compact_price_fetch_status
        WHERE token_ref = ?
        ORDER BY start_ts, end_ts, status
        """,
        (token_ref,),
    ).fetchall()
    return [dict(row) for row in rows]


def _status_summary(status_rows: Sequence[Mapping[str, Any]]) -> str:
    counts = Counter(str(row.get("status") or "") for row in status_rows)
    return ";".join(f"{key}:{counts[key]}" for key in sorted(counts)) if counts else "missing"


def _status_bound(status_rows: Sequence[Mapping[str, Any]], bound: str) -> Optional[datetime]:
    times: list[datetime] = []
    for row in status_rows:
        if row.get("status") != "complete":
            continue
        value = row.get("start_ts") if bound == "min" else row.get("end_ts")
        if value is not None:
            times.append(datetime.fromtimestamp(int(value), tz=timezone.utc))
    if not times:
        return None
    return min(times) if bound == "min" else max(times)


def _complete_overlap_seconds(
    status_rows: Sequence[Mapping[str, Any]],
    start: Optional[datetime],
    end: Optional[datetime],
) -> float:
    if not start or not end or end <= start:
        return 0.0
    total = 0.0
    for row in status_rows:
        if row.get("status") != "complete":
            continue
        job_start = datetime.fromtimestamp(int(row["start_ts"]), tz=timezone.utc)
        job_end = datetime.fromtimestamp(int(row["end_ts"]), tz=timezone.utc)
        total += _overlap_seconds(start, end, job_start, job_end)
    return total


def _quote_window_counts(
    conn: sqlite3.Connection,
    token_ref: Optional[int],
    start: Optional[datetime],
    end: Optional[datetime],
) -> dict[str, Any]:
    if token_ref is None or not start or not end:
        return {"quote_rows_suggested_window": 0, "first_quote_suggested_window_utc": "", "last_quote_suggested_window_utc": ""}
    row = conn.execute(
        """
        SELECT COUNT(*) AS quote_rows, MIN(timestamp_utc) AS first_quote, MAX(timestamp_utc) AS last_quote
        FROM compact_quote_history
        WHERE token_ref = ?
          AND timestamp_utc >= ?
          AND timestamp_utc <= ?
        """,
        (token_ref, start.isoformat(), end.isoformat()),
    ).fetchone()
    return {
        "quote_rows_suggested_window": int(row["quote_rows"] or 0),
        "first_quote_suggested_window_utc": str(row["first_quote"] or ""),
        "last_quote_suggested_window_utc": str(row["last_quote"] or ""),
    }


def _quote_count(
    conn: sqlite3.Connection,
    token_ref: Optional[int],
    start: Optional[datetime],
    end: Optional[datetime],
) -> int:
    if token_ref is None or not start or not end or end <= start:
        return 0
    return int(
        conn.execute(
            """
            SELECT COUNT(*)
            FROM compact_quote_history
            WHERE token_ref = ?
              AND timestamp_utc >= ?
              AND timestamp_utc <= ?
            """,
            (token_ref, start.isoformat(), end.isoformat()),
        ).fetchone()[0]
        or 0
    )


def _all_token_quote_bounds(conn: sqlite3.Connection, token_ref: Optional[int]) -> dict[str, str]:
    if token_ref is None:
        return {"first_quote_token_utc": "", "last_quote_token_utc": ""}
    row = conn.execute(
        """
        SELECT MIN(timestamp_utc) AS first_quote, MAX(timestamp_utc) AS last_quote
        FROM compact_quote_history
        WHERE token_ref = ?
        """,
        (token_ref,),
    ).fetchone()
    return {"first_quote_token_utc": str(row["first_quote"] or ""), "last_quote_token_utc": str(row["last_quote"] or "")}


def _examples(rows: Sequence[Mapping[str, Any]], *, per_class: int) -> list[dict[str, Any]]:
    counts: Counter[str] = Counter()
    selected: list[dict[str, Any]] = []
    for row in rows:
        classification = str(row.get("classification", ""))
        if counts[classification] >= per_class:
            continue
        selected.append(dict(row))
        counts[classification] += 1
    return selected


def _manifest_fields() -> list[str]:
    return [
        "priority_rank",
        "classification",
        "token_id",
        "token_ref",
        "market_id",
        "universe_row_id",
        "side",
        "city",
        "market_date",
        "event_class",
        "fact_state",
        "event_time_utc",
        "market_close_utc",
        "pull_start_utc",
        "pull_end_utc",
        "pull_window_seconds",
        "compact_status_summary",
        "classification_rationale",
    ]


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Optional[Sequence[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        seen = set()
        for row in rows:
            for key in row.keys():
                if key not in seen:
                    keys.append(key)
                    seen.add(key)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _pull_command(manifest: Path, jobs: int) -> str:
    return (
        "PYTHONPATH=. PYTHONPYCACHEPREFIX=/tmp/codex-pycache "
        "python3 -m daily_weather_pipeline.phase_c_telonex_reconciliation pull-targeted "
        "--work-root . "
        f"--manifest {shlex.quote(str(manifest))} "
        f"--limit-jobs {jobs} "
        "--workers 1 --progress-every 1 --telonex-env .env "
        "--telonex-hard-timeout-seconds 60 --execute-online-pull"
    )


def _failure_class(error: Any) -> str:
    text = str(error or "").lower()
    if "http 404" in text or "http error 404" in text or "404: not found" in text:
        return "http_404"
    if "cached missing file" in text or "missing-response" in text:
        return "local_cache_missing"
    if "rate limit" in text or "429" in text:
        return "rate_limit"
    if "timeout" in text or "connection reset" in text or "network" in text:
        return "transient"
    if not text:
        return ""
    return "unknown"


def _overlap_seconds(
    start_a: Optional[datetime],
    end_a: Optional[datetime],
    start_b: Optional[datetime],
    end_b: Optional[datetime],
) -> float:
    if not start_a or not end_a or not start_b or not end_b:
        return 0.0
    start = max(start_a, start_b)
    end = min(end_a, end_b)
    if end <= start:
        return 0.0
    return (end - start).total_seconds()


def _seconds_between(start: Optional[datetime], end: Optional[datetime]) -> float:
    if not start or not end or end <= start:
        return 0.0
    return (end - start).total_seconds()


def _parse_dt(value: Any) -> Optional[datetime]:
    if not value:
        return None
    text = str(value)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _optional_int(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _iso(value: Optional[datetime]) -> str:
    return value.isoformat() if value else ""


def max_dt(*values: Optional[datetime]) -> Optional[datetime]:
    present = [value for value in values if value is not None]
    return max(present) if present else None


def min_dt(*values: Optional[datetime]) -> Optional[datetime]:
    present = [value for value in values if value is not None]
    return min(present) if present else None


def _resolve(root: Path, path: Path) -> Path:
    expanded = path.expanduser()
    return expanded if expanded.is_absolute() else root / expanded


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Reconcile Phase C sparse-event rows against Telonex preservation status.")
    subparsers = parser.add_subparsers(dest="command")

    report = subparsers.add_parser("report", help="Write the local-only reconciliation report")
    report.add_argument("--work-root", default=".")
    report.add_argument("--phase-c-pull-list", default=str(DEFAULT_PHASE_C_PULL_LIST))
    report.add_argument("--preservation-pull-list", default=str(DEFAULT_PRESERVATION_PULL_LIST))
    report.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))

    pull = subparsers.add_parser("pull-targeted", help="Run an explicitly approved targeted Telonex pull from a manifest")
    pull.add_argument("--work-root", default=".")
    pull.add_argument("--source-repo", default=str(DEFAULT_SOURCE_REPO))
    pull.add_argument("--manifest", required=True)
    pull.add_argument("--limit-jobs", type=int, default=0)
    pull.add_argument("--workers", type=int, default=1)
    pull.add_argument("--progress-every", type=int, default=1)
    pull.add_argument("--telonex-env", default=None)
    pull.add_argument("--telonex-hard-timeout-seconds", type=float, default=60.0)
    pull.add_argument("--execute-online-pull", action="store_true")

    args = parser.parse_args(argv)
    command = args.command or "report"
    if command == "report":
        result = run_reconciliation(
            work_root=Path(args.work_root),
            phase_c_pull_list=Path(args.phase_c_pull_list),
            preservation_pull_list=Path(args.preservation_pull_list),
            output_dir=Path(args.output_dir),
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if command == "pull-targeted":
        result = run_targeted_pull(
            work_root=Path(args.work_root),
            source_repo=Path(args.source_repo).expanduser(),
            manifest=Path(args.manifest),
            limit_jobs=args.limit_jobs,
            workers=args.workers,
            progress_every=args.progress_every,
            telonex_env=Path(args.telonex_env).expanduser() if args.telonex_env else None,
            hard_timeout_seconds=args.telonex_hard_timeout_seconds,
            execute_online_pull=args.execute_online_pull,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    raise AssertionError(command)


if __name__ == "__main__":
    raise SystemExit(main())
