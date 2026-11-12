from __future__ import annotations

import sqlite3
from collections import defaultdict
from contextlib import closing
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Optional
from zoneinfo import ZoneInfo

from .weather_corpus_snapshot import (
    CorpusSnapshotError,
    reject_corpus_database_sidecars,
    resolve_corpus_snapshot,
)


REQUIRED_SOURCE_KINDS = ("forecast", "realized")
ENTRY_HOURS_UTC = (0, 6, 12, 18)


def weather_corpus_coverage(
    *,
    work_root: Path,
    start: date,
    end: date,
    as_of_utc: datetime,
    label_availability_lag_hours: float = 12.0,
    market_db: Optional[Path] = None,
    weather_db: Optional[Path] = None,
    snapshot_id: Optional[str] = None,
    snapshot_source: Optional[str] = None,
) -> dict[str, Any]:
    """Return fail-closed market/weather coverage without mutating any store.

    Coverage is measured for every label-available market city/date and for both
    required data-product kinds.  A fresh city can therefore never hide a stale
    city through a workspace-wide maximum date.
    """

    root = Path(work_root).resolve()
    as_of = _normalize_utc(as_of_utc)
    if (market_db is None) != (weather_db is None):
        raise ValueError("market_db and weather_db overrides must be supplied together")
    explicit_override_supplied = market_db is not None
    if market_db is None or weather_db is None:
        try:
            resolution = resolve_corpus_snapshot(root)
        except CorpusSnapshotError as exc:
            return {
                "coverage_contract_version": "weather_corpus_city_source_date_v2",
                "as_of_utc": as_of.isoformat(),
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
                "label_availability_lag_hours": float(label_availability_lag_hours),
                "required_source_kinds": list(REQUIRED_SOURCE_KINDS),
                "coverage_available": False,
                "operationally_fresh": False,
                "fail_closed_reasons": ["active_corpus_snapshot_invalid"],
                "coverage_error": str(exc),
                "labelable_city_date_pairs": 0,
                "complete_city_date_pairs": 0,
                "pending_city_date_pairs": 0,
                "failed_city_date_pairs": 0,
                "coverage_fraction": 0.0,
                "required_city_source_date_checks": 0,
                "covered_city_source_date_checks": 0,
                "city_coverage": [],
                "city_source_date_coverage": [],
            }
        market_db = resolution.market_db
        weather_db = resolution.weather_db
        snapshot_id = resolution.snapshot_id
        snapshot_source = resolution.source
    elif snapshot_source is None:
        snapshot_source = "explicit_sidecar_free_override"
    market_db = Path(market_db)
    weather_db = Path(weather_db)
    immutable_reads = (snapshot_source or "") == "active_immutable_snapshot"
    reject_sidecars = (snapshot_source or "") in {
        "active_immutable_snapshot",
        "copy_on_write_candidate",
        "explicit_sidecar_free_override",
    }
    allow_zero_length_wal = (snapshot_source or "") == "copy_on_write_candidate"
    base = {
        "coverage_contract_version": "weather_corpus_city_source_date_v2",
        "as_of_utc": as_of.isoformat(),
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "label_availability_lag_hours": float(label_availability_lag_hours),
        "required_source_kinds": list(REQUIRED_SOURCE_KINDS),
        "corpus_snapshot_id": snapshot_id,
        "corpus_snapshot_source": snapshot_source or (
            "explicit_sidecar_free_override"
            if explicit_override_supplied
            else "legacy_unversioned_seed"
        ),
        "market_db": str(market_db),
        "weather_db": str(weather_db),
    }
    missing_stores = [str(path) for path in (market_db, weather_db) if not path.exists()]
    if missing_stores:
        return {
            **base,
            "coverage_available": False,
            "operationally_fresh": False,
            "fail_closed_reasons": ["required_corpus_store_missing"],
            "missing_stores": missing_stores,
            "labelable_city_date_pairs": 0,
            "complete_city_date_pairs": 0,
            "pending_city_date_pairs": 0,
            "failed_city_date_pairs": 0,
            "coverage_fraction": 0.0,
            "required_city_source_date_checks": 0,
            "covered_city_source_date_checks": 0,
            "city_coverage": [],
            "city_source_date_coverage": [],
        }

    try:
        pairs = _load_market_pairs_read_only(
            market_db,
            start=start,
            end=end,
            immutable=immutable_reads,
            reject_sidecars=reject_sidecars,
            sidecar_reason_prefix=snapshot_source or "corpus_coverage",
            allow_zero_length_wal=allow_zero_length_wal,
        )
        forecasts = _load_source_dates_read_only(
            weather_db,
            table="forecast_snapshots",
            start=start,
            end=end,
            immutable=immutable_reads,
            reject_sidecars=reject_sidecars,
            sidecar_reason_prefix=snapshot_source or "corpus_coverage",
            allow_zero_length_wal=allow_zero_length_wal,
        )
        realized = _load_source_dates_read_only(
            weather_db,
            table="realized_daily_weather",
            start=start,
            end=end,
            immutable=immutable_reads,
            reject_sidecars=reject_sidecars,
            sidecar_reason_prefix=snapshot_source or "corpus_coverage",
            allow_zero_length_wal=allow_zero_length_wal,
        )
        statuses = _load_fetch_status_read_only(
            weather_db,
            start=start,
            end=end,
            immutable=immutable_reads,
            reject_sidecars=reject_sidecars,
            sidecar_reason_prefix=snapshot_source or "corpus_coverage",
            allow_zero_length_wal=allow_zero_length_wal,
        )
    except (CorpusSnapshotError, sqlite3.DatabaseError, OSError) as exc:
        return {
            **base,
            "coverage_available": False,
            "operationally_fresh": False,
            "fail_closed_reasons": ["corpus_coverage_read_failed"],
            "coverage_error": f"{type(exc).__name__}: {str(exc)[:500]}",
            "labelable_city_date_pairs": 0,
            "complete_city_date_pairs": 0,
            "pending_city_date_pairs": 0,
            "failed_city_date_pairs": 0,
            "coverage_fraction": 0.0,
            "required_city_source_date_checks": 0,
            "covered_city_source_date_checks": 0,
            "city_coverage": [],
            "city_source_date_coverage": [],
        }

    labelable_pairs = [
        pair
        for pair in pairs
        if _label_available_at_utc(
            str(pair["local_date"]),
            str(pair["timezone"]),
            float(label_availability_lag_hours),
        )
        <= as_of
    ]
    source_date_rows: list[dict[str, Any]] = []
    pair_states: list[dict[str, Any]] = []
    city_pairs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    complete_count = 0
    pending_count = 0
    failed_count = 0
    covered_checks = 0

    for pair in labelable_pairs:
        city = str(pair["normalized_city"])
        target_date = str(pair["local_date"])
        key = (city, target_date)
        forecast_rows_by_source = forecasts.get(key, {})
        forecast_sources = sorted(forecast_rows_by_source)
        realized_sources = sorted(realized.get(key, ()))
        expected_decision_times = _expected_forecast_decision_times_utc(
            target_date,
            str(pair["timezone"]),
        )
        forecast_source_coverage = []
        complete_forecast_sources: list[str] = []
        for source_name in forecast_sources:
            observed_times = set(forecast_rows_by_source[source_name])
            missing_times = sorted(expected_decision_times - observed_times)
            unexpected_times = sorted(observed_times - expected_decision_times)
            source_complete = bool(expected_decision_times) and not missing_times
            if source_complete:
                complete_forecast_sources.append(source_name)
            forecast_source_coverage.append(
                {
                    "source_name": source_name,
                    "covered_decision_times_utc": sorted(observed_times & expected_decision_times),
                    "missing_decision_times_utc": missing_times,
                    "unexpected_decision_times_utc": unexpected_times,
                    "complete_expected_decision_set": source_complete,
                }
            )
        forecast_present = bool(complete_forecast_sources)
        realized_present = bool(realized_sources)
        covered_checks += int(forecast_present) + int(realized_present)
        source_date_rows.extend(
            (
                {
                    "normalized_city": city,
                    "target_date": target_date,
                    "source_kind": "forecast",
                    "source_names": forecast_sources,
                    "complete_source_names": complete_forecast_sources,
                    "expected_decision_times_utc": sorted(expected_decision_times),
                    "source_decision_time_coverage": forecast_source_coverage,
                    "present": forecast_present,
                },
                {
                    "normalized_city": city,
                    "target_date": target_date,
                    "source_kind": "realized",
                    "source_names": realized_sources,
                    "present": realized_present,
                },
            )
        )
        complete = forecast_present and realized_present
        stored_status = statuses.get(key)
        if complete:
            state = "complete"
            complete_count += 1
        elif stored_status == "failed":
            state = "failed"
            failed_count += 1
        else:
            state = "pending"
            pending_count += 1
        pair_state = {
            "normalized_city": city,
            "target_date": target_date,
            "timezone": str(pair["timezone"]),
            "state": state,
            "stored_fetch_status": stored_status,
            "forecast_present": forecast_present,
            "forecast_sources": forecast_sources,
            "forecast_complete_sources": complete_forecast_sources,
            "expected_forecast_decision_times_utc": sorted(expected_decision_times),
            "forecast_source_decision_time_coverage": forecast_source_coverage,
            "realized_present": realized_present,
            "realized_sources": realized_sources,
            "missing_source_kinds": [
                source_kind
                for source_kind, present in (
                    ("forecast", forecast_present),
                    ("realized", realized_present),
                )
                if not present
            ],
        }
        pair_states.append(pair_state)
        city_pairs[city].append(pair_state)

    city_coverage: list[dict[str, Any]] = []
    for city in sorted(city_pairs):
        items = sorted(city_pairs[city], key=lambda item: item["target_date"])
        required_dates = [str(item["target_date"]) for item in items]
        complete_dates = [str(item["target_date"]) for item in items if item["state"] == "complete"]
        latest_required = required_dates[-1] if required_dates else None
        latest_complete = complete_dates[-1] if complete_dates else None
        if latest_required and latest_complete:
            staleness_days: Optional[int] = max(
                0,
                (date.fromisoformat(latest_required) - date.fromisoformat(latest_complete)).days,
            )
        else:
            staleness_days = None
        source_coverage: dict[str, dict[str, Any]] = {}
        for source_kind in REQUIRED_SOURCE_KINDS:
            covered_dates_for_kind = [
                str(item["target_date"])
                for item in items
                if bool(item[f"{source_kind}_present"])
            ]
            source_names = sorted(
                {
                    source
                    for item in items
                    for source in item[f"{source_kind}_sources"]
                }
            )
            source_coverage[source_kind] = {
                "source_names": source_names,
                "covered_target_dates": covered_dates_for_kind,
                "missing_target_dates": [
                    target_date
                    for target_date in required_dates
                    if target_date not in set(covered_dates_for_kind)
                ],
                "latest_covered_target_date": (
                    covered_dates_for_kind[-1] if covered_dates_for_kind else None
                ),
            }
        city_ready = bool(items) and all(item["state"] == "complete" for item in items)
        city_coverage.append(
            {
                "normalized_city": city,
                "required_target_dates": required_dates,
                "complete_target_dates": complete_dates,
                "incomplete_target_dates": [
                    str(item["target_date"])
                    for item in items
                    if item["state"] != "complete"
                ],
                "latest_required_target_date": latest_required,
                "latest_complete_target_date": latest_complete,
                "staleness_days": staleness_days,
                "source_coverage": source_coverage,
                "operationally_fresh": city_ready,
            }
        )

    required_checks = len(labelable_pairs) * len(REQUIRED_SOURCE_KINDS)
    operationally_fresh = bool(labelable_pairs) and complete_count == len(labelable_pairs)
    reasons: list[str] = []
    if not labelable_pairs:
        reasons.append("no_labelable_market_city_date_pairs")
    if pending_count:
        reasons.append("pending_city_source_date_coverage")
    if failed_count:
        reasons.append("failed_city_source_date_coverage")
    if complete_count < len(labelable_pairs):
        reasons.append("incomplete_city_source_date_coverage")

    stale_cities = [
        str(item["normalized_city"])
        for item in city_coverage
        if not bool(item["operationally_fresh"])
    ]
    numeric_staleness = [
        int(item["staleness_days"])
        for item in city_coverage
        if item.get("staleness_days") is not None
    ]
    return {
        **base,
        "coverage_available": True,
        "operationally_fresh": operationally_fresh,
        "fail_closed_reasons": reasons,
        "city_date_pairs_in_window": len(pairs),
        "labelable_city_date_pairs": len(labelable_pairs),
        "complete_city_date_pairs": complete_count,
        "pending_city_date_pairs": pending_count,
        "failed_city_date_pairs": failed_count,
        "coverage_fraction": (
            complete_count / len(labelable_pairs) if labelable_pairs else 0.0
        ),
        "required_city_source_date_checks": required_checks,
        "covered_city_source_date_checks": covered_checks,
        "city_count": len(city_coverage),
        "fresh_city_count": len(city_coverage) - len(stale_cities),
        "stale_city_count": len(stale_cities),
        "stale_or_incomplete_cities": stale_cities,
        "maximum_city_staleness_days": max(numeric_staleness) if numeric_staleness else None,
        "pair_states": pair_states,
        "city_coverage": city_coverage,
        "city_source_date_coverage": source_date_rows,
    }


def coverage_allows_city_date(
    coverage: Mapping[str, Any],
    *,
    normalized_city: str,
    target_date: date | str,
) -> bool:
    """Return a fail-closed decision for one city/date in a coverage report."""

    if not bool(coverage.get("coverage_available")):
        return False
    target = target_date.isoformat() if isinstance(target_date, date) else str(target_date)
    matching = [
        row
        for row in coverage.get("city_source_date_coverage") or []
        if str(row.get("normalized_city")) == normalized_city
        and str(row.get("target_date")) == target
    ]
    return (
        len(matching) == len(REQUIRED_SOURCE_KINDS)
        and {str(row.get("source_kind")) for row in matching} == set(REQUIRED_SOURCE_KINDS)
        and all(bool(row.get("present")) for row in matching)
    )


def _load_market_pairs_read_only(
    market_db: Path,
    *,
    start: date,
    end: date,
    immutable: bool,
    reject_sidecars: bool,
    sidecar_reason_prefix: str,
    allow_zero_length_wal: bool,
) -> list[dict[str, str]]:
    with closing(
        _read_only_connection(
            market_db,
            immutable=immutable,
            reject_sidecars=reject_sidecars,
            sidecar_reason_prefix=sidecar_reason_prefix,
            allow_zero_length_wal=allow_zero_length_wal,
        )
    ) as conn:
        rows = conn.execute(
            """
            SELECT normalized_city, timezone, market_date_local AS local_date
            FROM market_universe
            WHERE market_date_local >= ? AND market_date_local <= ?
              AND normalized_city IS NOT NULL AND normalized_city != ''
              AND timezone IS NOT NULL AND timezone != ''
            GROUP BY normalized_city, timezone, market_date_local
            ORDER BY local_date, normalized_city
            """,
            (start.isoformat(), end.isoformat()),
        ).fetchall()
    return [
        {
            "normalized_city": str(row[0]),
            "timezone": str(row[1]),
            "local_date": str(row[2]),
        }
        for row in rows
    ]


def _load_source_dates_read_only(
    weather_db: Path,
    *,
    table: str,
    start: date,
    end: date,
    immutable: bool,
    reject_sidecars: bool,
    sidecar_reason_prefix: str,
    allow_zero_length_wal: bool,
) -> Any:
    if table not in {"forecast_snapshots", "realized_daily_weather"}:
        raise ValueError(f"Unsupported weather table: {table}")
    with closing(
        _read_only_connection(
            weather_db,
            immutable=immutable,
            reject_sidecars=reject_sidecars,
            sidecar_reason_prefix=sidecar_reason_prefix,
            allow_zero_length_wal=allow_zero_length_wal,
        )
    ) as conn:
        if table == "forecast_snapshots":
            rows = conn.execute(
                """
                SELECT normalized_city, local_date, source_name, decision_time_utc
                FROM forecast_snapshots
                WHERE local_date >= ? AND local_date <= ?
                  AND temperature_2m_max_f IS NOT NULL
                GROUP BY normalized_city, local_date, source_name, decision_time_utc
                ORDER BY local_date, normalized_city, source_name, decision_time_utc
                """,
                (start.isoformat(), end.isoformat()),
            ).fetchall()
            result: dict[tuple[str, str], dict[str, set[str]]] = defaultdict(
                lambda: defaultdict(set)
            )
            for city, local_date, source_name, decision_time_utc in rows:
                result[(str(city), str(local_date))][str(source_name)].add(
                    _canonical_utc_text(str(decision_time_utc))
                )
            return result
        rows = conn.execute(
            """
            SELECT normalized_city, local_date, source_name
            FROM realized_daily_weather
            WHERE local_date >= ? AND local_date <= ?
              AND temperature_2m_max_f IS NOT NULL
            GROUP BY normalized_city, local_date, source_name
            ORDER BY local_date, normalized_city, source_name
            """,
            (start.isoformat(), end.isoformat()),
        ).fetchall()
    realized_result: dict[tuple[str, str], set[str]] = defaultdict(set)
    for city, local_date, source_name in rows:
        realized_result[(str(city), str(local_date))].add(str(source_name))
    return realized_result


def _expected_forecast_decision_times_utc(
    target_date_text: str,
    timezone_name: str,
) -> set[str]:
    target = date.fromisoformat(target_date_text)
    try:
        zone = ZoneInfo(timezone_name)
    except Exception:
        zone = timezone.utc
    decisions: set[str] = set()
    current = target - timedelta(days=2)
    while current <= target:
        for hour in ENTRY_HOURS_UTC:
            decision = datetime.combine(
                current,
                datetime.min.time().replace(hour=hour),
                tzinfo=timezone.utc,
            )
            local_date = decision.astimezone(zone).date()
            lead_days = (target - local_date).days
            if 1 <= lead_days <= 2 or local_date == target:
                decisions.add(decision.isoformat())
        current += timedelta(days=1)
    return decisions


def _canonical_utc_text(value: str) -> str:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat()


def _load_fetch_status_read_only(
    weather_db: Path,
    *,
    start: date,
    end: date,
    immutable: bool,
    reject_sidecars: bool,
    sidecar_reason_prefix: str,
    allow_zero_length_wal: bool,
) -> dict[tuple[str, str], str]:
    with closing(
        _read_only_connection(
            weather_db,
            immutable=immutable,
            reject_sidecars=reject_sidecars,
            sidecar_reason_prefix=sidecar_reason_prefix,
            allow_zero_length_wal=allow_zero_length_wal,
        )
    ) as conn:
        rows = conn.execute(
            """
            SELECT normalized_city, local_date, status
            FROM weather_fetch_status
            WHERE local_date >= ? AND local_date <= ?
            """,
            (start.isoformat(), end.isoformat()),
        ).fetchall()
    return {(str(city), str(local_date)): str(status) for city, local_date, status in rows}


def _read_only_connection(
    path: Path,
    *,
    immutable: bool,
    reject_sidecars: bool,
    sidecar_reason_prefix: str,
    allow_zero_length_wal: bool,
) -> sqlite3.Connection:
    if reject_sidecars:
        reject_corpus_database_sidecars(
            path,
            reason_prefix=sidecar_reason_prefix,
            allow_zero_length_wal=allow_zero_length_wal,
        )
    suffix = "&immutable=1" if immutable else ""
    conn = sqlite3.connect(f"file:{path}?mode=ro{suffix}", uri=True)
    try:
        if reject_sidecars:
            reject_corpus_database_sidecars(
                path,
                reason_prefix=sidecar_reason_prefix,
                allow_zero_length_wal=allow_zero_length_wal,
            )
    except Exception:
        conn.close()
        raise
    return conn


def _label_available_at_utc(
    local_date_text: str,
    timezone_name: str,
    lag_hours: float,
) -> datetime:
    try:
        zone = ZoneInfo(timezone_name)
    except Exception:
        zone = timezone.utc
    local_date = date.fromisoformat(local_date_text)
    local_end = datetime.combine(
        local_date + timedelta(days=1),
        datetime.min.time(),
        tzinfo=zone,
    )
    return local_end.astimezone(timezone.utc) + timedelta(hours=float(lag_hours))


def _normalize_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
