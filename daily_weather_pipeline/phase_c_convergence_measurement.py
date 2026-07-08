from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable, Mapping, Optional, Sequence
from zoneinfo import ZoneInfo


DEFAULT_START_DATE = date(2026, 1, 1)
DEFAULT_END_DATE = date(2026, 6, 29)
DEFAULT_DECISION_DB = Path("work/data/backtest/daily_weather_decision_artifact.sqlite")
DEFAULT_PRICE_DB = Path("work/data/backtest/daily_weather_price_history_compact.sqlite")
DEFAULT_STATION_DB = Path("work/data/backtest/daily_weather_station_observations.sqlite")
DEFAULT_OUTPUT_DIR = Path("work/reports/phase_c_convergence")
FIRST_10_STATIONS = ("KLGA", "KDAL", "KMIA", "KORD", "KSEA", "KHOU", "KAUS", "KLAX", "KSFO", "KBKF")
PRIMARY_QUOTE_FRESHNESS_SECONDS = 600.0
PRIMARY_SPREAD_CAP = 0.05
REPRICE_DISTANCE = 0.05
PULL_WINDOW_BEFORE_MINUTES = 60
PULL_WINDOW_AFTER_MINUTES = 180
DECAY_OFFSETS_MINUTES = (0, 1, 2, 5, 10, 15, 30, 60, 120, 180)
LOCAL_AFTERNOON_START_HOUR = 12
LOCAL_AFTERNOON_END_HOUR = 20
TARGETED_PULL_MAX_ROWS = 1000
TARGETED_PULL_PER_GROUP_LIMIT = 2


@dataclass(frozen=True)
class MarketSideRow:
    universe_row_id: str
    market_id: str
    slug: str
    question: str
    city: str
    region: str
    timezone_name: str
    market_date: str
    bucket_label: str
    canonical_bucket_type: str
    threshold_low_f: Optional[float]
    threshold_high_f: Optional[float]
    side: str
    token_id: str
    yes_token_id: str
    no_token_id: str
    canonical_label: Optional[int]
    settlement_unit: str
    settlement_precision: str
    source_family: str
    station_name: str
    source_url: str
    recommended_action: str
    exclude_from_backtest_replay: int
    token_ref: Optional[int] = None
    compact_fetch_status: str = ""


@dataclass(frozen=True)
class StationPoint:
    observed_at_utc: datetime
    local_time: time
    temperature_f: float
    high_so_far_f: float
    settlement_value: float
    sample_count: int


@dataclass(frozen=True)
class FactEvent:
    row: MarketSideRow
    fact_state: str
    event_class: str
    event_time_utc: datetime
    local_hour: float
    high_so_far_f: float
    settlement_value: float
    p_yes_fact_fv: float
    side_fact_fv: float
    source_rule: str
    sample_count_at_event: int
    canonical_yes_label: Optional[int]
    canonical_side_label: Optional[int]
    fact_matches_canonical_yes: Optional[bool]


@dataclass(frozen=True)
class QuotePoint:
    timestamp_utc: datetime
    bid: Optional[float]
    ask: Optional[float]
    mid: Optional[float]
    last_price: Optional[float]
    spread: Optional[float]
    bid_size: Optional[float]
    ask_size: Optional[float]
    liquidity_depth: Optional[float]
    data_quality_flags: str


def run_phase_c_convergence_measurement(
    *,
    work_root: Path,
    start: date = DEFAULT_START_DATE,
    end: date = DEFAULT_END_DATE,
    station_ids: Sequence[str] = FIRST_10_STATIONS,
    heat_close_local: time = time(17, 30),
    output_dir: Optional[Path] = None,
) -> dict[str, Any]:
    root = work_root.resolve()
    decision_db = root / DEFAULT_DECISION_DB
    price_db = root / DEFAULT_PRICE_DB
    station_db = root / DEFAULT_STATION_DB
    report_dir = (root / (output_dir or DEFAULT_OUTPUT_DIR)).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)

    station_sources = _load_station_sources(station_db, station_ids=station_ids, start=start, end=end)
    token_refs = _load_compact_token_refs(price_db)
    compact_statuses = _load_compact_statuses(price_db)
    market_rows = _load_market_side_rows(
        decision_db,
        station_sources=station_sources,
        token_refs=token_refs,
        compact_statuses=compact_statuses,
        start=start,
        end=end,
    )
    station_points = _load_station_points(
        station_db,
        station_sources=station_sources,
        start=start,
        end=end,
    )
    events = classify_fact_events(market_rows, station_points, heat_close_local=heat_close_local)
    measurement = _measure_events(price_db, events)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    prefix = report_dir / f"{timestamp}-phase-c-convergence-measurement"
    event_csv = prefix.with_name(prefix.name + "-events.csv")
    decay_csv = prefix.with_name(prefix.name + "-decay.csv")
    breakdown_csv = prefix.with_name(prefix.name + "-breakdowns.csv")
    pull_csv = prefix.with_name(prefix.name + "-targeted-telonex-pull-list.csv")
    json_path = prefix.with_suffix(".json")
    md_path = prefix.with_suffix(".md")

    _write_csv(event_csv, measurement["event_rows"])
    _write_csv(decay_csv, measurement["decay_rows"])
    _write_csv(breakdown_csv, measurement["breakdown_rows"])
    _write_csv(pull_csv, measurement["targeted_pull_rows"])

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "role": "Phase C Measurement Agent",
        "scope": {
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "station_ids": list(station_ids),
            "station_source_rows": len(station_sources),
            "heat_close_local": heat_close_local.strftime("%H:%M"),
            "quote_window_minutes": {
                "before": PULL_WINDOW_BEFORE_MINUTES,
                "after": PULL_WINDOW_AFTER_MINUTES,
            },
            "primary_quote_freshness_seconds": PRIMARY_QUOTE_FRESHNESS_SECONDS,
            "primary_spread_cap": PRIMARY_SPREAD_CAP,
            "reprice_distance": REPRICE_DISTANCE,
        },
        "inputs": {
            "decision_db": str(decision_db),
            "compact_price_db": str(price_db),
            "station_observation_db": str(station_db),
            "decision_view": "decision_snapshots_current_enriched",
            "quote_table": "compact_quote_history",
            "station_table": "station_observations",
        },
        "methodology": _methodology(),
        "summary": measurement["summary"],
        "quality_decision": _quality_decision(measurement["summary"]),
        "artifacts": {
            "events_csv": str(event_csv),
            "decay_csv": str(decay_csv),
            "breakdowns_csv": str(breakdown_csv),
            "targeted_telonex_pull_list_csv": str(pull_csv),
            "summary_json": str(json_path),
            "summary_markdown": str(md_path),
        },
        "limitations": _limitations(measurement["summary"]),
        "targeted_pull_policy": (
            "Rows are included only for first-10 station Phase C event-side tokens with missing "
            "or insufficient compact quotes inside the event-local window. This is not a broad backfill list."
        ),
    }
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(_render_markdown(report), encoding="utf-8")
    return report


def classify_fact_events(
    rows: Sequence[MarketSideRow],
    station_points: Mapping[tuple[str, str], Sequence[StationPoint]],
    *,
    heat_close_local: time,
) -> list[FactEvent]:
    events: list[FactEvent] = []
    for row in rows:
        points = station_points.get((row.source_url, row.market_date), ())
        if not points:
            continue
        fact = _classify_bucket_fact(row, points, heat_close_local=heat_close_local)
        if fact is None:
            continue
        state, event_class, point, p_yes, source_rule = fact
        side_fv = p_yes if row.side.upper() == "YES" else 1.0 - p_yes
        canonical_side_label = None
        if row.canonical_label is not None:
            canonical_side_label = row.canonical_label if row.side.upper() == "YES" else 1 - row.canonical_label
        events.append(
            FactEvent(
                row=row,
                fact_state=state,
                event_class=event_class,
                event_time_utc=point.observed_at_utc,
                local_hour=point.local_time.hour + point.local_time.minute / 60.0,
                high_so_far_f=point.high_so_far_f,
                settlement_value=point.settlement_value,
                p_yes_fact_fv=p_yes,
                side_fact_fv=side_fv,
                source_rule=source_rule,
                sample_count_at_event=point.sample_count,
                canonical_yes_label=row.canonical_label,
                canonical_side_label=canonical_side_label,
                fact_matches_canonical_yes=(
                    None if row.canonical_label is None else bool(int(round(p_yes)) == row.canonical_label)
                ),
            )
        )
    return events


def _classify_bucket_fact(
    row: MarketSideRow,
    points: Sequence[StationPoint],
    *,
    heat_close_local: time,
) -> Optional[tuple[str, str, StationPoint, float, str]]:
    bucket_type = row.canonical_bucket_type
    lower = row.threshold_low_f
    upper = row.threshold_high_f
    if bucket_type == "above" and lower is not None:
        crossed = _first_point(points, lambda point: point.settlement_value >= lower - 1e-9)
        if crossed:
            return (
                "crossed_open_ended_threshold",
                "deterministic_crossed",
                crossed,
                1.0,
                "first station high-so-far rounded to settlement precision at or above threshold",
            )
        close = _heat_close_point(points, heat_close_local)
        if close and close.settlement_value < lower - 1e-9:
            return (
                "near_dead_above_after_heat_close",
                "near_heat_window_close",
                close,
                0.0,
                "pre-registered 17:30 local heat-close rule; not a deterministic settlement fact",
            )
    elif bucket_type == "below" and upper is not None:
        crossed = _first_point(points, lambda point: point.settlement_value > upper + 1e-9)
        if crossed:
            return (
                "dead_below_bucket_crossed",
                "deterministic_crossed",
                crossed,
                0.0,
                "first station high-so-far rounded to settlement precision above below-bucket cutoff",
            )
        close = _heat_close_point(points, heat_close_local)
        if close and close.settlement_value <= upper + 1e-9:
            return (
                "near_settled_below_after_heat_close",
                "near_heat_window_close",
                close,
                1.0,
                "pre-registered 17:30 local heat-close rule; not a deterministic settlement fact",
            )
    elif bucket_type in {"exact", "range"} and lower is not None and upper is not None:
        exceeded = _first_point(points, lambda point: point.settlement_value > upper + 1e-9)
        if exceeded:
            return (
                "dead_exact_or_range_bucket_upper_exceeded",
                "deterministic_crossed",
                exceeded,
                0.0,
                "first station high-so-far rounded to settlement precision above bucket upper bound",
            )
        close = _heat_close_point(points, heat_close_local)
        if close:
            if close.settlement_value < lower - 1e-9:
                return (
                    "near_dead_exact_below_after_heat_close",
                    "near_heat_window_close",
                    close,
                    0.0,
                    "pre-registered 17:30 local heat-close rule; not a deterministic settlement fact",
                )
            if lower - 1e-9 <= close.settlement_value <= upper + 1e-9:
                return (
                    "near_settled_exact_after_heat_close",
                    "near_heat_window_close",
                    close,
                    1.0,
                    "pre-registered 17:30 local heat-close rule; not a deterministic settlement fact",
                )
    return None


def measure_quote_path(event: FactEvent, quotes: Sequence[QuotePoint]) -> dict[str, Any]:
    event_time = event.event_time_utc
    post_quotes = [quote for quote in quotes if quote.timestamp_utc >= event_time]
    pre_quotes = [quote for quote in quotes if quote.timestamp_utc < event_time]
    first_after = post_quotes[0] if post_quotes else None
    prev_before = pre_quotes[-1] if pre_quotes else None
    repriced_mid = _first_repriced(post_quotes, event.side_fact_fv, field="mid")
    repriced_ask = _first_repriced(post_quotes, event.side_fact_fv, field="ask")

    row = _event_base_row(event)
    row.update(
        {
            "token_ref": event.row.token_ref,
            "compact_fetch_status": event.row.compact_fetch_status,
            "quote_count_m60_p180": len(quotes),
            "quote_count_pre_60m": len(pre_quotes),
            "quote_count_post_10m": _count_between(post_quotes, event_time, 10),
            "quote_count_post_30m": _count_between(post_quotes, event_time, 30),
            "quote_count_post_60m": _count_between(post_quotes, event_time, 60),
            "quote_count_post_180m": len(post_quotes),
            "prev_quote_seconds_before_fact": _seconds_between(prev_before.timestamp_utc, event_time) if prev_before else None,
            "next_quote_seconds_after_fact": _seconds_between(event_time, first_after.timestamp_utc) if first_after else None,
            "repriced_mid_seconds": _seconds_between(event_time, repriced_mid.timestamp_utc) if repriced_mid else None,
            "repriced_ask_seconds": _seconds_between(event_time, repriced_ask.timestamp_utc) if repriced_ask else None,
            "first_quote_timestamp_utc": _iso(first_after.timestamp_utc) if first_after else "",
            "first_quote_bid": _quote_field(first_after, "bid"),
            "first_quote_ask": _quote_field(first_after, "ask"),
            "first_quote_mid": _quote_field(first_after, "mid"),
            "first_quote_spread": _quote_field(first_after, "spread"),
            "first_quote_bid_size": _quote_field(first_after, "bid_size"),
            "first_quote_ask_size": _quote_field(first_after, "ask_size"),
            "first_quote_liquidity_depth": _quote_field(first_after, "liquidity_depth"),
            "first_quote_mid_distance_from_fv": _distance(_quote_field(first_after, "mid"), event.side_fact_fv),
            "first_quote_ask_distance_from_fv": _distance(_quote_field(first_after, "ask"), event.side_fact_fv),
            "first_quote_bid_distance_from_fv": _distance(_quote_field(first_after, "bid"), event.side_fact_fv),
            "first_quote_primary_phase_c_fresh": (
                first_after is not None
                and _seconds_between(event_time, first_after.timestamp_utc) <= PRIMARY_QUOTE_FRESHNESS_SECONDS
            ),
            "first_quote_spread_lte_5c": (
                first_after is not None and first_after.spread is not None and first_after.spread <= PRIMARY_SPREAD_CAP
            ),
            "has_visible_ask_size": first_after is not None and first_after.ask_size is not None,
            "has_visible_bid_size": first_after is not None and first_after.bid_size is not None,
        }
    )
    row["primary_measurable"] = bool(
        row["first_quote_primary_phase_c_fresh"] and row["first_quote_spread_lte_5c"]
    )
    row["true_side_stale_ask_gap_gte_5c"] = bool(
        event.side_fact_fv == 1.0
        and row["primary_measurable"]
        and row["first_quote_ask_distance_from_fv"] is not None
        and row["first_quote_ask_distance_from_fv"] >= REPRICE_DISTANCE
    )
    row["false_side_stale_ask_above_5c"] = bool(
        event.side_fact_fv == 0.0
        and row["primary_measurable"]
        and row["first_quote_ask"] is not None
        and row["first_quote_ask"] >= REPRICE_DISTANCE
    )
    return row


def _measure_events(price_db: Path, events: Sequence[FactEvent]) -> dict[str, Any]:
    event_rows: list[dict[str, Any]] = []
    decay_rows: list[dict[str, Any]] = []
    targeted_pull_candidate_rows: list[dict[str, Any]] = []
    with sqlite3.connect(f"file:{price_db}?mode=ro", uri=True) as conn:
        conn.row_factory = sqlite3.Row
        for index, event in enumerate(events, start=1):
            quotes = _load_event_quotes(conn, event)
            event_row = measure_quote_path(event, quotes)
            event_row["event_index"] = index
            event_rows.append(event_row)
            decay_rows.extend(_decay_rows(event, quotes, index))
            if _needs_targeted_pull(event_row):
                targeted_pull_candidate_rows.append(_targeted_pull_row(event, event_row))

    targeted_pull_rows = _prioritize_targeted_pulls(targeted_pull_candidate_rows)
    breakdown_rows = _breakdown_rows(event_rows)
    summary = _summary(event_rows, decay_rows, targeted_pull_rows, targeted_pull_candidate_rows)
    return {
        "event_rows": event_rows,
        "decay_rows": decay_rows,
        "breakdown_rows": breakdown_rows,
        "targeted_pull_rows": targeted_pull_rows,
        "summary": summary,
    }


def _load_event_quotes(conn: sqlite3.Connection, event: FactEvent) -> list[QuotePoint]:
    if event.row.token_ref is None:
        return []
    start = event.event_time_utc - timedelta(minutes=PULL_WINDOW_BEFORE_MINUTES)
    end = event.event_time_utc + timedelta(minutes=PULL_WINDOW_AFTER_MINUTES)
    rows = conn.execute(
        """
        SELECT timestamp_utc, bid, ask, mid, last_price, spread, bid_size, ask_size,
               liquidity_depth, data_quality_flags
        FROM compact_quote_history
        WHERE token_ref = ?
          AND timestamp_utc >= ?
          AND timestamp_utc <= ?
        ORDER BY timestamp_utc
        """,
        (event.row.token_ref, _sqlite_ts(start), _sqlite_ts(end)),
    ).fetchall()
    return [
        QuotePoint(
            timestamp_utc=_parse_utc(row["timestamp_utc"]),
            bid=_optional_float(row["bid"]),
            ask=_optional_float(row["ask"]),
            mid=_optional_float(row["mid"]),
            last_price=_optional_float(row["last_price"]),
            spread=_optional_float(row["spread"]),
            bid_size=_optional_float(row["bid_size"]),
            ask_size=_optional_float(row["ask_size"]),
            liquidity_depth=_optional_float(row["liquidity_depth"]),
            data_quality_flags=str(row["data_quality_flags"] or ""),
        )
        for row in rows
    ]


def _load_station_sources(
    station_db: Path,
    *,
    station_ids: Sequence[str],
    start: date,
    end: date,
) -> dict[str, dict[str, str]]:
    placeholders = ",".join("?" for _ in station_ids)
    with sqlite3.connect(f"file:{station_db}?mode=ro", uri=True) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            f"""
            SELECT station_id, settlement_source_url, source_family, timezone_name
            FROM station_daily_highs
            WHERE station_id IN ({placeholders})
              AND local_date BETWEEN ? AND ?
            GROUP BY station_id, settlement_source_url, source_family, timezone_name
            ORDER BY station_id
            """,
            (*station_ids, start.isoformat(), end.isoformat()),
        ).fetchall()
    return {
        str(row["settlement_source_url"]): {
            "station_id": str(row["station_id"]),
            "source_family": str(row["source_family"]),
            "timezone_name": str(row["timezone_name"]),
        }
        for row in rows
    }


def _load_market_side_rows(
    decision_db: Path,
    *,
    station_sources: Mapping[str, Mapping[str, str]],
    token_refs: Mapping[str, int],
    compact_statuses: Mapping[int, str],
    start: date,
    end: date,
) -> list[MarketSideRow]:
    source_urls = tuple(station_sources)
    placeholders = ",".join("?" for _ in source_urls)
    query = f"""
        SELECT universe_row_id, market_id, slug, question, normalized_city, country_region,
               timezone, market_date_local, bucket_label, canonical_bucket_type,
               threshold_low_f, threshold_high_f, side, token_id, yes_token_id, no_token_id,
               canonical_label, settlement_unit, settlement_precision, canonical_source_family,
               canonical_source_station, canonical_source_url, recommended_action,
               exclude_from_backtest_replay
        FROM decision_snapshots_current_enriched
        WHERE canonical_source_url IN ({placeholders})
          AND market_date_local BETWEEN ? AND ?
          AND settlement_unit = 'F'
          AND settlement_precision = 'whole_F'
          AND canonical_bucket_type IN ('above', 'below', 'exact', 'range')
          AND COALESCE(recommended_action, '') != 'exclude_or_manual_review'
          AND COALESCE(exclude_from_backtest_replay, 0) = 0
        GROUP BY universe_row_id, side
        ORDER BY normalized_city, market_date_local, universe_row_id, side
    """
    with sqlite3.connect(f"file:{decision_db}?mode=ro", uri=True) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(query, (*source_urls, start.isoformat(), end.isoformat())).fetchall()
    market_rows: list[MarketSideRow] = []
    for row in rows:
        token_id = str(row["token_id"] or "")
        token_ref = token_refs.get(token_id)
        market_rows.append(
            MarketSideRow(
                universe_row_id=str(row["universe_row_id"] or ""),
                market_id=str(row["market_id"] or ""),
                slug=str(row["slug"] or ""),
                question=str(row["question"] or ""),
                city=str(row["normalized_city"] or ""),
                region=str(row["country_region"] or ""),
                timezone_name=str(row["timezone"] or ""),
                market_date=str(row["market_date_local"] or ""),
                bucket_label=str(row["bucket_label"] or ""),
                canonical_bucket_type=str(row["canonical_bucket_type"] or ""),
                threshold_low_f=_optional_float(row["threshold_low_f"]),
                threshold_high_f=_optional_float(row["threshold_high_f"]),
                side=str(row["side"] or ""),
                token_id=token_id,
                yes_token_id=str(row["yes_token_id"] or ""),
                no_token_id=str(row["no_token_id"] or ""),
                canonical_label=_optional_int(row["canonical_label"]),
                settlement_unit=str(row["settlement_unit"] or ""),
                settlement_precision=str(row["settlement_precision"] or ""),
                source_family=str(row["canonical_source_family"] or ""),
                station_name=str(row["canonical_source_station"] or ""),
                source_url=str(row["canonical_source_url"] or ""),
                recommended_action=str(row["recommended_action"] or ""),
                exclude_from_backtest_replay=int(row["exclude_from_backtest_replay"] or 0),
                token_ref=token_ref,
                compact_fetch_status=compact_statuses.get(token_ref, "missing_token_ref") if token_ref else "missing_token_ref",
            )
        )
    return market_rows


def _load_station_points(
    station_db: Path,
    *,
    station_sources: Mapping[str, Mapping[str, str]],
    start: date,
    end: date,
) -> dict[tuple[str, str], list[StationPoint]]:
    source_urls = tuple(station_sources)
    placeholders = ",".join("?" for _ in source_urls)
    with sqlite3.connect(f"file:{station_db}?mode=ro", uri=True) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            f"""
            SELECT station_id, settlement_source_url, observed_at_utc, local_date, local_time,
                   temperature_f
            FROM station_observations
            WHERE settlement_source_url IN ({placeholders})
              AND local_date BETWEEN ? AND ?
              AND temperature_f IS NOT NULL
            ORDER BY settlement_source_url, local_date, observed_at_utc
            """,
            (*source_urls, start.isoformat(), end.isoformat()),
        ).fetchall()

    output: dict[tuple[str, str], list[StationPoint]] = defaultdict(list)
    running_high: dict[tuple[str, str], float] = {}
    sample_counts: Counter[tuple[str, str]] = Counter()
    for row in rows:
        key = (str(row["settlement_source_url"] or ""), str(row["local_date"] or ""))
        temperature_f = float(row["temperature_f"])
        current_high = max(running_high.get(key, -math.inf), temperature_f)
        running_high[key] = current_high
        sample_counts[key] += 1
        output[key].append(
            StationPoint(
                observed_at_utc=_parse_utc(str(row["observed_at_utc"])),
                local_time=_parse_time(str(row["local_time"])),
                temperature_f=temperature_f,
                high_so_far_f=current_high,
                settlement_value=float(_round_half_up(current_high)),
                sample_count=sample_counts[key],
            )
        )
    return dict(output)


def _load_compact_token_refs(price_db: Path) -> dict[str, int]:
    with sqlite3.connect(f"file:{price_db}?mode=ro", uri=True) as conn:
        rows = conn.execute("SELECT token_ref, token_id FROM compact_tokens").fetchall()
    return {str(token_id): int(token_ref) for token_ref, token_id in rows}


def _load_compact_statuses(price_db: Path) -> dict[int, str]:
    with sqlite3.connect(f"file:{price_db}?mode=ro", uri=True) as conn:
        rows = conn.execute(
            """
            SELECT token_ref,
                   GROUP_CONCAT(status || ':' || count, ';') AS statuses
            FROM (
                SELECT token_ref, status, COUNT(*) AS count
                FROM compact_price_fetch_status
                GROUP BY token_ref, status
            )
            GROUP BY token_ref
            """
        ).fetchall()
    return {int(token_ref): str(statuses or "") for token_ref, statuses in rows}


def _event_base_row(event: FactEvent) -> dict[str, Any]:
    row = event.row
    return {
        "universe_row_id": row.universe_row_id,
        "market_id": row.market_id,
        "slug": row.slug,
        "city": row.city,
        "region": row.region,
        "timezone": row.timezone_name,
        "market_date": row.market_date,
        "bucket_label": row.bucket_label,
        "canonical_bucket_type": row.canonical_bucket_type,
        "threshold_low_f": row.threshold_low_f,
        "threshold_high_f": row.threshold_high_f,
        "side": row.side,
        "token_id": row.token_id,
        "fact_state": event.fact_state,
        "event_class": event.event_class,
        "event_time_utc": _iso(event.event_time_utc),
        "event_local_hour": round(event.local_hour, 4),
        "high_so_far_f": round(event.high_so_far_f, 4),
        "settlement_value": event.settlement_value,
        "p_yes_fact_fv": event.p_yes_fact_fv,
        "side_fact_fv": event.side_fact_fv,
        "source_rule": event.source_rule,
        "sample_count_at_event": event.sample_count_at_event,
        "canonical_yes_label": event.canonical_yes_label,
        "canonical_side_label": event.canonical_side_label,
        "fact_matches_canonical_yes": event.fact_matches_canonical_yes,
        "settlement_unit": row.settlement_unit,
        "settlement_precision": row.settlement_precision,
        "source_family": row.source_family,
        "station_name": row.station_name,
        "source_url": row.source_url,
        "recommended_action": row.recommended_action,
    }


def _decay_rows(event: FactEvent, quotes: Sequence[QuotePoint], event_index: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for offset in DECAY_OFFSETS_MINUTES:
        target = event.event_time_utc + timedelta(minutes=offset)
        quote = _first_quote_at_or_after(quotes, target)
        if quote is not None and quote.timestamp_utc > target + timedelta(minutes=10):
            quote = None
        rows.append(
            {
                "event_index": event_index,
                "universe_row_id": event.row.universe_row_id,
                "market_id": event.row.market_id,
                "city": event.row.city,
                "market_date": event.row.market_date,
                "station_name": event.row.station_name,
                "canonical_bucket_type": event.row.canonical_bucket_type,
                "side": event.row.side,
                "fact_state": event.fact_state,
                "event_class": event.event_class,
                "side_fact_fv": event.side_fact_fv,
                "offset_minutes": offset,
                "quote_timestamp_utc": _iso(quote.timestamp_utc) if quote else "",
                "quote_lag_seconds_from_offset": _seconds_between(target, quote.timestamp_utc) if quote else None,
                "minutes_after_fact": _seconds_between(event.event_time_utc, quote.timestamp_utc) / 60.0 if quote else None,
                "bid": quote.bid if quote else None,
                "ask": quote.ask if quote else None,
                "mid": quote.mid if quote else None,
                "spread": quote.spread if quote else None,
                "bid_size": quote.bid_size if quote else None,
                "ask_size": quote.ask_size if quote else None,
                "mid_distance_from_fv": _distance(quote.mid if quote else None, event.side_fact_fv),
                "ask_distance_from_fv": _distance(quote.ask if quote else None, event.side_fact_fv),
                "bid_distance_from_fv": _distance(quote.bid if quote else None, event.side_fact_fv),
            }
        )
    return rows


def _breakdown_rows(event_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    dimensions = [
        ("overall", ()),
        ("event_class", ("event_class",)),
        ("fact_state", ("fact_state",)),
        ("city", ("city",)),
        ("station", ("station_name",)),
        ("bucket_side", ("canonical_bucket_type", "side")),
        ("city_bucket_side", ("city", "canonical_bucket_type", "side")),
        ("event_class_side", ("event_class", "side")),
    ]
    output: list[dict[str, Any]] = []
    for breakdown_name, fields in dimensions:
        groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
        for row in event_rows:
            groups[tuple(row.get(field, "") for field in fields)].append(row)
        for key, rows in groups.items():
            base = {"breakdown": breakdown_name}
            for field, value in zip(fields, key):
                base[field] = value
            base.update(_row_group_metrics(rows))
            output.append(base)
    return output


def _summary(
    event_rows: Sequence[Mapping[str, Any]],
    decay_rows: Sequence[Mapping[str, Any]],
    targeted_pull_rows: Sequence[Mapping[str, Any]],
    targeted_pull_candidate_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    deterministic = [row for row in event_rows if row.get("event_class") == "deterministic_crossed"]
    near = [row for row in event_rows if row.get("event_class") == "near_heat_window_close"]
    true_side = [row for row in event_rows if row.get("side_fact_fv") == 1.0]
    false_side = [row for row in event_rows if row.get("side_fact_fv") == 0.0]
    primary = [row for row in event_rows if row.get("primary_measurable")]
    afternoon = [
        row
        for row in event_rows
        if LOCAL_AFTERNOON_START_HOUR <= float(row.get("event_local_hour") or 0) <= LOCAL_AFTERNOON_END_HOUR
    ]
    decay_summary = _decay_summary(decay_rows)
    return {
        "event_side_rows": len(event_rows),
        "deterministic_event_side_rows": len(deterministic),
        "near_heat_close_event_side_rows": len(near),
        "true_side_event_rows": len(true_side),
        "false_side_event_rows": len(false_side),
        "events_with_token_ref": sum(row.get("token_ref") is not None for row in event_rows),
        "events_with_any_quote_window": sum((row.get("quote_count_m60_p180") or 0) > 0 for row in event_rows),
        "events_with_next_quote_10m": sum(_lte(row.get("next_quote_seconds_after_fact"), 600) for row in event_rows),
        "events_with_next_quote_30m": sum(_lte(row.get("next_quote_seconds_after_fact"), 1800) for row in event_rows),
        "events_primary_measurable": len(primary),
        "primary_measurable_rate": _rate(len(primary), len(event_rows)),
        "afternoon_event_side_rows": len(afternoon),
        "afternoon_primary_measurable_rate": _rate(
            sum(row.get("primary_measurable") for row in afternoon),
            len(afternoon),
        ),
        "median_next_quote_seconds": _median_numeric(row.get("next_quote_seconds_after_fact") for row in event_rows),
        "median_quote_count_post_60m": _median_numeric(row.get("quote_count_post_60m") for row in event_rows),
        "median_first_quote_spread": _median_numeric(row.get("first_quote_spread") for row in event_rows),
        "median_first_quote_ask_size": _median_numeric(row.get("first_quote_ask_size") for row in event_rows),
        "visible_ask_size_rate": _rate(sum(row.get("has_visible_ask_size") for row in event_rows), len(event_rows)),
        "visible_bid_size_rate": _rate(sum(row.get("has_visible_bid_size") for row in event_rows), len(event_rows)),
        "deterministic_canonical_yes_match_rate": _match_rate(deterministic),
        "near_heat_close_canonical_yes_match_rate": _match_rate(near),
        "repriced_mid_within_5c_10m": sum(_lte(row.get("repriced_mid_seconds"), 600) for row in event_rows),
        "repriced_mid_within_5c_30m": sum(_lte(row.get("repriced_mid_seconds"), 1800) for row in event_rows),
        "repriced_mid_within_5c_60m": sum(_lte(row.get("repriced_mid_seconds"), 3600) for row in event_rows),
        "repriced_mid_within_5c_180m": sum(_lte(row.get("repriced_mid_seconds"), 10800) for row in event_rows),
        "true_side_primary_rows": sum(row.get("primary_measurable") for row in true_side),
        "true_side_stale_ask_gap_gte_5c": sum(row.get("true_side_stale_ask_gap_gte_5c") for row in true_side),
        "true_side_stale_ask_gap_gte_5c_rate_primary": _rate(
            sum(row.get("true_side_stale_ask_gap_gte_5c") for row in true_side),
            sum(row.get("primary_measurable") for row in true_side),
        ),
        "false_side_primary_rows": sum(row.get("primary_measurable") for row in false_side),
        "false_side_stale_ask_above_5c": sum(row.get("false_side_stale_ask_above_5c") for row in false_side),
        "false_side_stale_ask_above_5c_rate_primary": _rate(
            sum(row.get("false_side_stale_ask_above_5c") for row in false_side),
            sum(row.get("primary_measurable") for row in false_side),
        ),
        "targeted_pull_rows": len(targeted_pull_rows),
        "targeted_pull_candidate_rows": len(targeted_pull_candidate_rows),
        "targeted_pull_unique_tokens": len({row.get("token_id") for row in targeted_pull_rows}),
        "targeted_pull_policy": {
            "max_rows": TARGETED_PULL_MAX_ROWS,
            "per_city_fact_state_month_limit": TARGETED_PULL_PER_GROUP_LIMIT,
            "priority_order": [
                "local afternoon event windows",
                "true-side deterministic fair-value rows",
                "rows with no compact quotes in the event window",
                "earlier event dates for quick source verification",
            ],
        },
        "by_event_class": _counter_dict(event_rows, "event_class"),
        "by_fact_state": _counter_dict(event_rows, "fact_state"),
        "by_city": _counter_dict(event_rows, "city"),
        "by_bucket_side": _counter_dict(event_rows, ("canonical_bucket_type", "side")),
        "event_group_metrics": {
            "overall": _row_group_metrics(event_rows),
            "deterministic": _row_group_metrics(deterministic),
            "near_heat_close": _row_group_metrics(near),
            "true_side": _row_group_metrics(true_side),
            "false_side": _row_group_metrics(false_side),
            "afternoon": _row_group_metrics(afternoon),
        },
        "decay_curve": decay_summary,
    }


def _row_group_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "rows": 0,
            "primary_measurable_rate": None,
            "next_quote_10m_rate": None,
            "median_next_quote_seconds": None,
            "median_quote_count_post_60m": None,
            "median_first_quote_mid_distance": None,
            "median_first_quote_ask_distance": None,
            "repriced_mid_30m_rate": None,
            "true_side_stale_ask_gap_gte_5c_rate_primary": None,
            "canonical_yes_match_rate": None,
        }
    true_side_primary = [
        row for row in rows if row.get("side_fact_fv") == 1.0 and row.get("primary_measurable")
    ]
    return {
        "rows": len(rows),
        "primary_measurable_rows": sum(row.get("primary_measurable") for row in rows),
        "primary_measurable_rate": _rate(sum(row.get("primary_measurable") for row in rows), len(rows)),
        "next_quote_10m_rate": _rate(sum(_lte(row.get("next_quote_seconds_after_fact"), 600) for row in rows), len(rows)),
        "next_quote_30m_rate": _rate(sum(_lte(row.get("next_quote_seconds_after_fact"), 1800) for row in rows), len(rows)),
        "median_next_quote_seconds": _median_numeric(row.get("next_quote_seconds_after_fact") for row in rows),
        "median_quote_count_post_60m": _median_numeric(row.get("quote_count_post_60m") for row in rows),
        "median_first_quote_spread": _median_numeric(row.get("first_quote_spread") for row in rows),
        "median_first_quote_mid_distance": _median_numeric(row.get("first_quote_mid_distance_from_fv") for row in rows),
        "median_first_quote_ask_distance": _median_numeric(row.get("first_quote_ask_distance_from_fv") for row in rows),
        "repriced_mid_30m_rate": _rate(sum(_lte(row.get("repriced_mid_seconds"), 1800) for row in rows), len(rows)),
        "true_side_stale_ask_gap_gte_5c_rate_primary": _rate(
            sum(row.get("true_side_stale_ask_gap_gte_5c") for row in true_side_primary),
            len(true_side_primary),
        ),
        "canonical_yes_match_rate": _match_rate(rows),
    }


def _decay_summary(decay_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, Any], list[Mapping[str, Any]]] = defaultdict(list)
    for row in decay_rows:
        grouped[(row.get("event_class"), row.get("offset_minutes"))].append(row)
    output: list[dict[str, Any]] = []
    for (event_class, offset), rows in sorted(grouped.items(), key=lambda item: (str(item[0][0]), int(item[0][1]))):
        observed = [row for row in rows if row.get("quote_timestamp_utc")]
        output.append(
            {
                "event_class": event_class,
                "offset_minutes": offset,
                "rows": len(rows),
                "quote_observed_rows": len(observed),
                "quote_observed_rate": _rate(len(observed), len(rows)),
                "median_mid_distance_from_fv": _median_numeric(row.get("mid_distance_from_fv") for row in observed),
                "mean_mid_distance_from_fv": _mean_numeric(row.get("mid_distance_from_fv") for row in observed),
                "median_ask_distance_from_fv": _median_numeric(row.get("ask_distance_from_fv") for row in observed),
                "mean_ask_distance_from_fv": _mean_numeric(row.get("ask_distance_from_fv") for row in observed),
                "median_spread": _median_numeric(row.get("spread") for row in observed),
                "median_ask_size": _median_numeric(row.get("ask_size") for row in observed),
            }
        )
    return output


def _quality_decision(summary: Mapping[str, Any]) -> dict[str, Any]:
    primary_rate = summary.get("primary_measurable_rate") or 0.0
    stale_rate = summary.get("true_side_stale_ask_gap_gte_5c_rate_primary") or 0.0
    deterministic_match = summary.get("deterministic_canonical_yes_match_rate") or 0.0
    median_quotes_60 = summary.get("median_quote_count_post_60m") or 0.0
    if deterministic_match < 0.95:
        go_no_go = "NO_GO_CLASSIFIER"
        rationale = "Deterministic station facts do not agree with canonical labels at the required audit level."
    elif primary_rate < 0.50 or median_quotes_60 < 3:
        go_no_go = "NO_GO_QUOTE_DENSITY"
        rationale = "Compact quote density is not sufficient for primary Phase C measurement at 10-minute/5c gates."
    elif stale_rate < 0.05:
        go_no_go = "NO_GO_EDGE"
        rationale = "Fresh true-side fact quotes rarely show at least 5c stale ask distance from deterministic fair value."
    else:
        go_no_go = "GO_SHADOW_INFRA_ONLY"
        rationale = (
            "Historical measurement shows enough stale quote states to justify read-only Sleeve C shadow infrastructure, "
            "but not fill capture or live trading claims."
        )
    return {
        "recommendation": go_no_go,
        "rationale": rationale,
        "pre_registered_thresholds": {
            "deterministic_canonical_yes_match_rate_min": 0.95,
            "primary_measurable_rate_min": 0.50,
            "median_quote_count_post_60m_min": 3,
            "true_side_stale_ask_gap_gte_5c_rate_primary_min": 0.05,
        },
    }


def _limitations(summary: Mapping[str, Any]) -> list[str]:
    limitations = [
        "Measurement only: no strategy PnL optimization, no fills, no latency-capture assumption, and no live-trading authorization.",
        "Primary universe is the first-10 whole-F Wunderground/METAR station proof set; non-US, NOAA/weather.gov, and CWA connector rows are out of scope.",
        "Near heat-window-close events use a fixed 17:30 local pre-registered proxy and are labeled near-deterministic, not deterministic settlement facts.",
        "Quote observations are visible historical compact book states; they do not prove order fillability after live latency.",
        "Canonical settlement labels are used only to audit fact-state correctness after the event, not as inputs to event classification.",
    ]
    if (summary.get("targeted_pull_rows") or 0) > 0:
        limitations.append("Some event-side tokens lack dense compact quotes in the event window; see the targeted Telonex pull list.")
    return limitations


def _methodology() -> dict[str, Any]:
    return {
        "candidate_filter": [
            "decision_snapshots_current_enriched rows for the first-10 station source URLs",
            "whole-F Wunderground/METAR settlement rows only",
            "recommended_action != exclude_or_manual_review",
            "exclude_from_backtest_replay != 1",
            "canonical bucket types above, below, exact, or range",
        ],
        "deterministic_fact_rules": [
            "above bucket YES becomes fact-true at first station high-so-far rounded half-up to whole F >= lower threshold",
            "below bucket YES becomes fact-false at first station high-so-far rounded half-up to whole F > upper cutoff",
            "exact/range bucket YES becomes fact-false at first station high-so-far rounded half-up to whole F > upper bound",
        ],
        "near_heat_close_rule": [
            "first observation at or after 17:30 local is the pre-registered heat-window-close proxy",
            "above bucket not crossed by then is labeled near_dead_above_after_heat_close",
            "below bucket still below/equal cutoff by then is labeled near_settled_below_after_heat_close",
            "exact/range bucket below lower or inside range by then is labeled near-dead or near-settled",
        ],
        "quote_measurement": [
            "compact_quote_history only, queried from 60 minutes before to 180 minutes after the fact timestamp",
            "primary measurable event requires first quote after fact within 10 minutes and spread <= 5c",
            "repricing is first post-fact quote with mid or ask within 5c of side deterministic fair value",
            "decay curve uses first quote at or after each offset with a 10-minute tolerance",
        ],
    }


def _render_markdown(report: Mapping[str, Any]) -> str:
    summary = report["summary"]
    quality = report["quality_decision"]
    artifacts = report["artifacts"]
    lines = [
        "# Phase C Convergence Measurement",
        "",
        f"Generated: {report['generated_at_utc']}",
        "",
        "## Recommendation",
        "",
        f"- Decision: `{quality['recommendation']}`",
        f"- Rationale: {quality['rationale']}",
        "",
        "## Scope",
        "",
        f"- Window: `{report['scope']['start_date']}` through `{report['scope']['end_date']}`.",
        f"- Stations: `{', '.join(report['scope']['station_ids'])}`.",
        f"- Heat-close rule: first observation at or after `{report['scope']['heat_close_local']}` local.",
        f"- Primary quote gate: next compact quote within `{int(PRIMARY_QUOTE_FRESHNESS_SECONDS / 60)}` minutes and spread <= `{PRIMARY_SPREAD_CAP:.2f}`.",
        "",
        "## Headline Metrics",
        "",
        f"- Event-side rows: `{summary['event_side_rows']}` (`{summary['deterministic_event_side_rows']}` deterministic, `{summary['near_heat_close_event_side_rows']}` near heat-close).",
        f"- Primary measurable rate: `{_pct(summary['primary_measurable_rate'])}`.",
        f"- Afternoon primary measurable rate: `{_pct(summary['afternoon_primary_measurable_rate'])}`.",
        f"- Median next quote after fact: `{_fmt(summary['median_next_quote_seconds'])}` seconds.",
        f"- Median quotes in first 60 minutes after fact: `{_fmt(summary['median_quote_count_post_60m'])}`.",
        f"- Deterministic fact/canonical-label match rate: `{_pct(summary['deterministic_canonical_yes_match_rate'])}`.",
        f"- Near heat-close/canonical-label match rate: `{_pct(summary['near_heat_close_canonical_yes_match_rate'])}`.",
        f"- True-side primary rows: `{summary['true_side_primary_rows']}`; stale true-side ask gap >=5c: `{summary['true_side_stale_ask_gap_gte_5c']}` (`{_pct(summary['true_side_stale_ask_gap_gte_5c_rate_primary'])}`).",
        f"- Sparse-event pull candidates: `{summary['targeted_pull_candidate_rows']}`; capped targeted pull rows: `{summary['targeted_pull_rows']}` across `{summary['targeted_pull_unique_tokens']}` unique tokens.",
        "",
        "## Decay Curve",
        "",
        "| Event class | Offset min | Quote observed | Median mid distance | Median ask distance | Median spread |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["decay_curve"]:
        lines.append(
            "| {event_class} | {offset_minutes} | {quote_observed_rate} | {mid} | {ask} | {spread} |".format(
                event_class=row["event_class"],
                offset_minutes=row["offset_minutes"],
                quote_observed_rate=_pct(row["quote_observed_rate"]),
                mid=_fmt(row["median_mid_distance_from_fv"]),
                ask=_fmt(row["median_ask_distance_from_fv"]),
                spread=_fmt(row["median_spread"]),
            )
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- Events CSV: `{artifacts['events_csv']}`",
            f"- Decay CSV: `{artifacts['decay_csv']}`",
            f"- Breakdowns CSV: `{artifacts['breakdowns_csv']}`",
            f"- Targeted Telonex pull list: `{artifacts['targeted_telonex_pull_list_csv']}`",
            f"- JSON: `{artifacts['summary_json']}`",
            "",
            "## Limitations",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in report["limitations"])
    lines.append("")
    return "\n".join(lines)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _first_point(points: Sequence[StationPoint], predicate: Any) -> Optional[StationPoint]:
    for point in points:
        if predicate(point):
            return point
    return None


def _heat_close_point(points: Sequence[StationPoint], heat_close_local: time) -> Optional[StationPoint]:
    for point in points:
        if point.local_time >= heat_close_local:
            return point
    return None


def _first_repriced(quotes: Sequence[QuotePoint], fair_value: float, *, field: str) -> Optional[QuotePoint]:
    for quote in quotes:
        value = getattr(quote, field)
        if value is not None and abs(float(value) - fair_value) <= REPRICE_DISTANCE + 1e-12:
            return quote
    return None


def _first_quote_at_or_after(quotes: Sequence[QuotePoint], target: datetime) -> Optional[QuotePoint]:
    for quote in quotes:
        if quote.timestamp_utc >= target:
            return quote
    return None


def _needs_targeted_pull(row: Mapping[str, Any]) -> bool:
    if row.get("token_ref") is None:
        return True
    if (row.get("quote_count_m60_p180") or 0) == 0:
        return True
    next_quote = row.get("next_quote_seconds_after_fact")
    if next_quote is None or float(next_quote) > PRIMARY_QUOTE_FRESHNESS_SECONDS:
        return True
    if (row.get("quote_count_post_60m") or 0) < 3:
        return True
    return False


def _targeted_pull_row(event: FactEvent, row: Mapping[str, Any]) -> dict[str, Any]:
    start = event.event_time_utc - timedelta(minutes=PULL_WINDOW_BEFORE_MINUTES)
    end = event.event_time_utc + timedelta(minutes=PULL_WINDOW_AFTER_MINUTES)
    reasons: list[str] = []
    if row.get("token_ref") is None:
        reasons.append("missing_compact_token_ref")
    if (row.get("quote_count_m60_p180") or 0) == 0:
        reasons.append("no_compact_quotes_in_event_window")
    next_quote = row.get("next_quote_seconds_after_fact")
    if next_quote is None or float(next_quote) > PRIMARY_QUOTE_FRESHNESS_SECONDS:
        reasons.append("no_next_quote_within_10m_after_fact")
    if (row.get("quote_count_post_60m") or 0) < 3:
        reasons.append("fewer_than_3_quotes_in_first_60m_after_fact")
    return {
        "token_id": event.row.token_id,
        "token_ref": event.row.token_ref,
        "market_id": event.row.market_id,
        "universe_row_id": event.row.universe_row_id,
        "side": event.row.side,
        "side_fact_fv": event.side_fact_fv,
        "canonical_bucket_type": event.row.canonical_bucket_type,
        "city": event.row.city,
        "market_date": event.row.market_date,
        "station_name": event.row.station_name,
        "source_url": event.row.source_url,
        "event_class": event.event_class,
        "fact_state": event.fact_state,
        "event_time_utc": _iso(event.event_time_utc),
        "event_local_hour": round(event.local_hour, 4),
        "suggested_pull_start_utc": _iso(start),
        "suggested_pull_end_utc": _iso(end),
        "compact_fetch_status": event.row.compact_fetch_status,
        "reason": ";".join(reasons),
    }


def _prioritize_targeted_pulls(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    per_group_counts: Counter[tuple[str, str, str]] = Counter()
    for raw in sorted(rows, key=_targeted_pull_sort_key):
        row = dict(raw)
        month = str(row.get("market_date", ""))[:7]
        group_key = (str(row.get("city", "")), str(row.get("fact_state", "")), month)
        if per_group_counts[group_key] >= TARGETED_PULL_PER_GROUP_LIMIT:
            continue
        row["priority_rank"] = len(selected) + 1
        row["priority_policy"] = (
            "capped local-afternoon/true-side-first targeted pull list; full sparse-event evidence remains in events CSV"
        )
        selected.append(row)
        per_group_counts[group_key] += 1
        if len(selected) >= TARGETED_PULL_MAX_ROWS:
            break
    return selected


def _targeted_pull_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    local_hour = float(row.get("event_local_hour") or 0.0)
    is_afternoon = LOCAL_AFTERNOON_START_HOUR <= local_hour <= LOCAL_AFTERNOON_END_HOUR
    is_true_side = float(row.get("side_fact_fv") or 0.0) == 1.0
    no_quotes = "no_compact_quotes_in_event_window" in str(row.get("reason", ""))
    complete_status = str(row.get("compact_fetch_status", "")).startswith("complete")
    return (
        0 if is_afternoon else 1,
        0 if is_true_side else 1,
        0 if no_quotes else 1,
        0 if complete_status else 1,
        str(row.get("market_date", "")),
        str(row.get("city", "")),
        str(row.get("fact_state", "")),
        str(row.get("token_id", "")),
    )


def _count_between(quotes: Sequence[QuotePoint], event_time: datetime, minutes: int) -> int:
    end = event_time + timedelta(minutes=minutes)
    return sum(event_time <= quote.timestamp_utc <= end for quote in quotes)


def _distance(value: Optional[float], fair_value: float) -> Optional[float]:
    if value is None:
        return None
    return abs(float(value) - float(fair_value))


def _quote_field(quote: Optional[QuotePoint], field: str) -> Optional[float]:
    if quote is None:
        return None
    return getattr(quote, field)


def _seconds_between(start: datetime, end: datetime) -> float:
    return (end - start).total_seconds()


def _counter_dict(rows: Sequence[Mapping[str, Any]], field: Any) -> dict[str, int]:
    counter: Counter[str] = Counter()
    if isinstance(field, tuple):
        for row in rows:
            counter["|".join(str(row.get(item, "")) for item in field)] += 1
    else:
        for row in rows:
            counter[str(row.get(field, ""))] += 1
    return dict(counter)


def _match_rate(rows: Sequence[Mapping[str, Any]]) -> Optional[float]:
    comparable = [row for row in rows if row.get("fact_matches_canonical_yes") is not None]
    if not comparable:
        return None
    return sum(bool(row.get("fact_matches_canonical_yes")) for row in comparable) / len(comparable)


def _median_numeric(values: Iterable[Any]) -> Optional[float]:
    nums = [float(value) for value in values if value is not None and value != ""]
    if not nums:
        return None
    return float(median(nums))


def _mean_numeric(values: Iterable[Any]) -> Optional[float]:
    nums = [float(value) for value in values if value is not None and value != ""]
    if not nums:
        return None
    return float(mean(nums))


def _rate(numerator: Any, denominator: Any) -> Optional[float]:
    if denominator is None or float(denominator) == 0:
        return None
    return float(numerator or 0) / float(denominator)


def _lte(value: Any, threshold: float) -> bool:
    return value is not None and value != "" and float(value) <= threshold


def _optional_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    return float(value)


def _optional_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    return int(value)


def _round_half_up(value: float) -> int:
    return int(Decimal(str(value)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _parse_time(value: str) -> time:
    return time.fromisoformat(value)


def _sqlite_ts(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def _fmt(value: Any) -> str:
    if value is None or value == "":
        return "NA"
    return f"{float(value):.4f}"


def _pct(value: Any) -> str:
    if value is None or value == "":
        return "NA"
    return f"{100.0 * float(value):.2f}%"


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def _parse_hhmm(value: str) -> time:
    hour, minute = value.split(":", 1)
    return time(int(hour), int(minute))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Measure Phase C station-fact convergence quote behavior.")
    parser.add_argument("--work-root", default=".", type=Path)
    parser.add_argument("--start-date", default=DEFAULT_START_DATE.isoformat())
    parser.add_argument("--end-date", default=DEFAULT_END_DATE.isoformat())
    parser.add_argument("--station-ids", default=",".join(FIRST_10_STATIONS))
    parser.add_argument("--heat-close-local", default="17:30")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), type=Path)
    args = parser.parse_args(argv)

    report = run_phase_c_convergence_measurement(
        work_root=args.work_root,
        start=_parse_date(args.start_date),
        end=_parse_date(args.end_date),
        station_ids=tuple(item.strip() for item in args.station_ids.split(",") if item.strip()),
        heat_close_local=_parse_hhmm(args.heat_close_local),
        output_dir=args.output_dir,
    )
    print(json.dumps({"summary": report["summary"], "artifacts": report["artifacts"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
