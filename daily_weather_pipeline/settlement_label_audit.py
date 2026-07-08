from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Iterable, Optional


SUPPORTED_START = "2025-10-13"
SUPPORTED_END = "2026-06-29"

MONTHS = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "sept": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}

DATE_TEXT_RE = re.compile(
    r"\bon\s+(?P<day>\d{1,2})\s+"
    r"(?P<month>Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\s+'(?P<year>\d{2})\b",
    re.I,
)
URL_RE = re.compile(r"available here:\s*(?P<url>https?://[^\s.]+(?:\.[^\s.]+)*[^\s]*)", re.I)


@dataclass(frozen=True)
class SourceMeta:
    source_family: str
    source_station: str
    source_url: str
    settlement_precision: str
    settlement_unit: str
    resolution_date_text: str
    resolution_date_conflict: int
    source_text_summary: str


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    work_root = Path(args.work_root).resolve()
    output_dir = Path(args.output_dir) if args.output_dir else work_root / "work" / "reports" / "settlement_label_audit"
    output_dir.mkdir(parents=True, exist_ok=True)

    audit = build_audit(work_root=work_root, start=args.start_date, end=args.end_date)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    prefix = f"{timestamp}-settlement-label-audit"

    labels_csv = output_dir / f"{prefix}-canonical-labels.csv"
    actions_csv = output_dir / f"{prefix}-rows-to-relabel-or-exclude.csv"
    groups_csv = output_dir / f"{prefix}-city-date-settlement-constraints.csv"
    summary_json = output_dir / f"{prefix}-summary.json"
    report_md = output_dir / f"{prefix}.md"

    _write_csv(labels_csv, audit["rows"])
    _write_csv(actions_csv, [row for row in audit["rows"] if row["recommended_action"] != "keep_polymarket_label"])
    _write_csv(groups_csv, audit["groups"])
    summary_json.write_text(json.dumps(audit["summary"], indent=2, sort_keys=True) + "\n")
    report_md.write_text(_render_report(audit["summary"], labels_csv, actions_csv, groups_csv, summary_json), encoding="utf-8")

    print(json.dumps({
        "report_md": str(report_md),
        "summary_json": str(summary_json),
        "canonical_labels_csv": str(labels_csv),
        "rows_to_relabel_or_exclude_csv": str(actions_csv),
        "city_date_constraints_csv": str(groups_csv),
    }, indent=2))
    return 0


def build_audit(*, work_root: Path, start: str, end: str) -> dict[str, Any]:
    data_root = work_root / "work" / "data" / "backtest"
    market_conn = _connect_readonly(data_root / "daily_weather_market_universe.sqlite")
    weather_conn = _connect_readonly(data_root / "daily_weather_weather_history.sqlite")
    try:
        market_rows = market_conn.execute(
            """
            SELECT universe_row_id, market_id, condition_id, slug, question, description,
                   event_id, event_slug, event_title, normalized_city, country_region, timezone,
                   market_date_local, bucket_label, bucket_type, operator, threshold_low_f,
                   threshold_high_f, units, yes_token_id, no_token_id, resolved_outcome,
                   settlement_source, telonex_source_path, telonex_source_row,
                   parser_confidence, parse_warnings_json
            FROM market_universe
            WHERE market_date_local >= ?
              AND market_date_local <= ?
              AND normalized_city IS NOT NULL
              AND resolved_outcome IN (0, 1)
            ORDER BY market_date_local, normalized_city, universe_row_id
            """,
            (start, end),
        ).fetchall()
        realized_rows = weather_conn.execute(
            """
            SELECT normalized_city, local_date, temperature_2m_max_f, hourly_temperature_2m_max_f,
                   source_name, raw_cache_path, request_url, data_quality_flags
            FROM realized_daily_weather
            WHERE local_date >= ?
              AND local_date <= ?
              AND temperature_2m_max_f IS NOT NULL
            """,
            (start, end),
        ).fetchall()
    finally:
        market_conn.close()
        weather_conn.close()

    realized = {(row["normalized_city"], row["local_date"]): row for row in realized_rows}
    group_constraints = _infer_group_constraints(market_rows)

    output_rows: list[dict[str, Any]] = []
    for row in market_rows:
        row_dict = dict(row)
        meta = _source_meta(row_dict)
        realized_row = realized.get((row["normalized_city"], row["market_date_local"]))
        open_meteo_high_f = _optional_float(realized_row["temperature_2m_max_f"]) if realized_row is not None else None
        open_meteo_high_display = _display_value(open_meteo_high_f, meta.settlement_unit)
        continuous_label = _continuous_label(open_meteo_high_f, row["threshold_low_f"], row["threshold_high_f"])
        source_rule_label = _source_rule_proxy_label(row_dict, meta, open_meteo_high_f)
        resolved = int(row["resolved_outcome"])
        current_mismatch = _mismatch(continuous_label, resolved)
        source_rule_mismatch = _mismatch(source_rule_label, resolved)
        canonical_bucket_type = _canonical_bucket_type(row_dict, meta)
        threshold_low_display, threshold_high_display = _threshold_display_bounds(row_dict, meta)
        margin_f = _distance_to_threshold_f(open_meteo_high_f, row["threshold_low_f"], row["threshold_high_f"])
        group_key = _group_key(row_dict, meta)
        group = group_constraints[group_key]
        flags = _row_flags(row_dict, meta, group, current_mismatch, source_rule_mismatch)
        action = _recommended_action(flags, current_mismatch)
        output_rows.append(
            {
                "universe_row_id": row["universe_row_id"],
                "market_id": row["market_id"],
                "slug": row["slug"],
                "normalized_city": row["normalized_city"],
                "country_region": row["country_region"],
                "timezone": row["timezone"],
                "market_date_local": row["market_date_local"],
                "question": row["question"],
                "bucket_label": row["bucket_label"],
                "current_bucket_type": row["bucket_type"],
                "canonical_bucket_type": canonical_bucket_type,
                "operator": row["operator"],
                "market_units": row["units"],
                "settlement_unit": meta.settlement_unit,
                "settlement_precision": meta.settlement_precision,
                "threshold_low_f": _round_or_none(row["threshold_low_f"], 4),
                "threshold_high_f": _round_or_none(row["threshold_high_f"], 4),
                "threshold_low_display": _round_or_none(threshold_low_display, 4),
                "threshold_high_display": _round_or_none(threshold_high_display, 4),
                "resolved_outcome": resolved,
                "canonical_label": resolved,
                "canonical_label_source": "polymarket_resolved_outcome_via_telonex_final_token_price",
                "settlement_source": row["settlement_source"],
                "source_family": meta.source_family,
                "source_station": meta.source_station,
                "source_url": meta.source_url,
                "resolution_date_text": meta.resolution_date_text,
                "resolution_date_conflict": meta.resolution_date_conflict,
                "source_text_summary": meta.source_text_summary,
                "open_meteo_source_name": realized_row["source_name"] if realized_row is not None else "",
                "open_meteo_realized_high_f": _round_or_none(open_meteo_high_f, 4),
                "open_meteo_realized_high_display": _round_or_none(open_meteo_high_display, 4),
                "open_meteo_current_continuous_label": continuous_label,
                "open_meteo_source_rule_proxy_label": source_rule_label,
                "current_continuous_mismatch": current_mismatch,
                "source_rule_proxy_mismatch": source_rule_mismatch,
                "distance_to_threshold_f": _round_or_none(margin_f, 4),
                "distance_to_threshold_bucket_f": _margin_bucket(margin_f),
                "group_possible_settlement_values": group["possible_values_compact"],
                "group_possible_count": group["possible_count"],
                "group_constraint_status": group["constraint_status"],
                "group_inferred_settlement_display": group["inferred_value"],
                "recommended_action": action,
                "exclude_from_backtest_replay": int(action.startswith("exclude")),
                "exclude_from_open_meteo_training": int(current_mismatch == 1 or action.startswith("exclude")),
                "audit_flags": "|".join(flags),
                "telonex_source_path": row["telonex_source_path"],
                "telonex_source_row": row["telonex_source_row"],
                "parser_confidence": row["parser_confidence"],
                "parse_warnings_json": row["parse_warnings_json"],
            }
        )

    group_rows = list(group_constraints.values())
    summary = _summary(output_rows, group_rows, start, end)
    return {"rows": output_rows, "groups": group_rows, "summary": summary}


def _infer_group_constraints(rows: Iterable[sqlite3.Row]) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        row_dict = dict(row)
        meta = _source_meta(row_dict)
        grouped[_group_key(row_dict, meta)].append(row_dict)

    output: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for key, items in grouped.items():
        city, local_date, unit, precision = key
        candidates = _candidate_values(unit, precision)
        possible = []
        for value in candidates:
            ok = True
            for item in items:
                meta = _source_meta(item)
                contains = _source_value_in_bucket(item, meta, value)
                if int(item["resolved_outcome"]) == 1 and not contains:
                    ok = False
                    break
                if int(item["resolved_outcome"]) == 0 and contains:
                    ok = False
                    break
            if ok:
                possible.append(value)
        compact = _compact_values(possible, precision)
        status = "exact" if len(possible) == 1 else "ambiguous" if possible else "contradiction"
        output[key] = {
            "normalized_city": city,
            "market_date_local": local_date,
            "settlement_unit": unit,
            "settlement_precision": precision,
            "market_rows": len(items),
            "yes_rows": sum(1 for item in items if int(item["resolved_outcome"]) == 1),
            "possible_count": len(possible),
            "possible_values_compact": compact,
            "constraint_status": status,
            "inferred_value": _format_source_value(possible[0], precision) if len(possible) == 1 else "",
        }
    return output


def _summary(rows: list[dict[str, Any]], groups: list[dict[str, Any]], start: str, end: str) -> dict[str, Any]:
    total = len(rows)
    current_mismatch = sum(int(row["current_continuous_mismatch"]) for row in rows)
    source_rule_mismatch = sum(int(row["source_rule_proxy_mismatch"]) for row in rows if row["open_meteo_source_rule_proxy_label"] != "")
    source_rule_n = sum(1 for row in rows if row["open_meteo_source_rule_proxy_label"] != "")
    fixes = sum(
        1 for row in rows
        if int(row["current_continuous_mismatch"]) == 1 and row["open_meteo_source_rule_proxy_label"] != "" and int(row["source_rule_proxy_mismatch"]) == 0
    )
    breaks = sum(
        1 for row in rows
        if int(row["current_continuous_mismatch"]) == 0 and row["open_meteo_source_rule_proxy_label"] != "" and int(row["source_rule_proxy_mismatch"]) == 1
    )
    current_relabel = [row for row in rows if int(row["current_continuous_mismatch"]) == 1]
    excluded = [row for row in rows if int(row["exclude_from_backtest_replay"]) == 1]
    near_2f = [row for row in rows if row["distance_to_threshold_f"] != "" and float(row["distance_to_threshold_f"]) <= 2.0]
    exact_range = [row for row in rows if row["current_bucket_type"] in {"exact", "range"}]

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_window": {"start": start, "end": end},
        "row_count": total,
        "city_date_group_count": len(groups),
        "open_meteo_continuous_mismatch_count": current_mismatch,
        "open_meteo_continuous_mismatch_rate": _rate(current_mismatch, total),
        "source_rule_proxy_mismatch_count": source_rule_mismatch,
        "source_rule_proxy_mismatch_rate": _rate(source_rule_mismatch, source_rule_n),
        "source_rule_proxy_rows": source_rule_n,
        "continuous_mismatch_fixed_by_source_rule_proxy": fixes,
        "continuous_match_broken_by_source_rule_proxy": breaks,
        "current_open_meteo_rows_to_relabel": len(current_relabel),
        "rows_to_exclude_from_backtest_replay": len(excluded),
        "rows_to_exclude_from_open_meteo_training": sum(int(row["exclude_from_open_meteo_training"]) for row in rows),
        "within_0_2f_threshold_rows": len(near_2f),
        "within_0_2f_threshold_mismatch_count": sum(int(row["current_continuous_mismatch"]) for row in near_2f),
        "within_0_2f_threshold_mismatch_rate": _rate(sum(int(row["current_continuous_mismatch"]) for row in near_2f), len(near_2f)),
        "exact_or_range_rows": len(exact_range),
        "exact_or_range_mismatch_count": sum(int(row["current_continuous_mismatch"]) for row in exact_range),
        "exact_or_range_mismatch_rate": _rate(sum(int(row["current_continuous_mismatch"]) for row in exact_range), len(exact_range)),
        "current_range_reclassified_exact_count": sum(
            1 for row in rows if row["current_bucket_type"] == "range" and row["canonical_bucket_type"] == "exact"
        ),
        "celsius_exact_reclassified_count": sum(
            1
            for row in rows
            if row["market_units"] == "C" and row["current_bucket_type"] == "range" and row["canonical_bucket_type"] == "exact"
        ),
        "by_current_bucket_type": _group_summary(rows, "current_bucket_type"),
        "by_canonical_bucket_type": _group_summary(rows, "canonical_bucket_type"),
        "by_distance_to_threshold_bucket_f": _group_summary(rows, "distance_to_threshold_bucket_f"),
        "by_source_family": _group_summary(rows, "source_family"),
        "by_settlement_precision": _group_summary(rows, "settlement_precision"),
        "by_city_top_mismatch": _top_group_summary(rows, "normalized_city", limit=12),
        "by_recommended_action": dict(Counter(row["recommended_action"] for row in rows)),
        "group_constraint_status": dict(Counter(row["constraint_status"] for row in groups)),
        "notes": [
            "canonical_label is Polymarket/Telonex settlement payout, not Open-Meteo",
            "source_rule_proxy_label is only a diagnostic computed from Open-Meteo and resolution-text precision",
            "exclude_from_backtest_replay is reserved for parser/source-text hazards, not ordinary weather-source disagreement",
        ],
    }


def _group_summary(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(row)
    output = []
    for group, items in grouped.items():
        mismatch = sum(int(item["current_continuous_mismatch"]) for item in items)
        proxy_items = [item for item in items if item["open_meteo_source_rule_proxy_label"] != ""]
        proxy_mismatch = sum(int(item["source_rule_proxy_mismatch"]) for item in proxy_items)
        output.append({
            "group": group,
            "n": len(items),
            "current_mismatch_n": mismatch,
            "current_mismatch_rate": _rate(mismatch, len(items)),
            "source_rule_proxy_n": len(proxy_items),
            "source_rule_proxy_mismatch_n": proxy_mismatch,
            "source_rule_proxy_mismatch_rate": _rate(proxy_mismatch, len(proxy_items)),
        })
    output.sort(key=lambda item: (-float(item["current_mismatch_rate"]), -int(item["n"]), item["group"]))
    return output


def _top_group_summary(rows: list[dict[str, Any]], key: str, *, limit: int) -> list[dict[str, Any]]:
    items = _group_summary(rows, key)
    items = [item for item in items if int(item["n"]) >= 25]
    return items[:limit]


def _source_meta(row: dict[str, Any]) -> SourceMeta:
    description = str(row.get("description") or "")
    lowered = description.lower()
    unit = _settlement_unit(description) or str(row.get("units") or "F")
    if "wunderground" in lowered:
        family = "Wunderground"
    elif "central weather administration" in lowered or "cwa.gov.tw" in lowered:
        family = "Taipei CWA"
    elif "weather.gov" in lowered or "noaa" in lowered:
        family = "NOAA"
    else:
        family = "unknown"

    if "one decimal place" in lowered:
        precision = f"decimal_0.1_{unit}"
    elif "whole degree" in lowered:
        precision = f"whole_{unit}"
    else:
        precision = "unknown"

    station = _source_station(description, family)
    url = _source_url(description)
    resolution_date = _resolution_date(description)
    conflict = int(bool(resolution_date and resolution_date != str(row.get("market_date_local") or "")))
    summary = _source_summary(family, station, unit, precision, url)
    return SourceMeta(
        source_family=family,
        source_station=station,
        source_url=url,
        settlement_precision=precision,
        settlement_unit=unit,
        resolution_date_text=resolution_date,
        resolution_date_conflict=conflict,
        source_text_summary=summary,
    )


def _settlement_unit(description: str) -> Optional[str]:
    lowered = description.lower()
    recorded = re.search(r"recorded .*? in degrees (?P<unit>Fahrenheit|Celsius)\b", description, re.I | re.S)
    if recorded:
        return "C" if recorded.group("unit").lower() == "celsius" else "F"
    measured = re.search(r"measures temperatures (?:in|to) (?:whole degrees )?(?P<unit>Fahrenheit|Celsius)\b", description, re.I)
    if measured:
        return "C" if measured.group("unit").lower() == "celsius" else "F"
    if "degrees celsius" in lowered:
        return "C"
    if "degrees fahrenheit" in lowered:
        return "F"
    # Unit glyphs appear in boilerplate toggle instructions, so use them only as
    # a last resort when the prose did not identify the settlement unit.
    if "°c" in lowered and "°f" not in lowered:
        return "C"
    if "°f" in lowered and "°c" not in lowered:
        return "F"
    return None


def _source_station(description: str, family: str) -> str:
    patterns = [
        r"recorded at the (?P<station>.*?) in degrees",
        r"recorded by NOAA at the (?P<station>.*?) in degrees",
        r"recorded by (?P<station>Taipei's Central Weather Administration) in degrees",
    ]
    for pattern in patterns:
        match = re.search(pattern, description, re.I | re.S)
        if match:
            return " ".join(match.group("station").split())
    return family


def _source_url(description: str) -> str:
    match = URL_RE.search(description)
    if not match:
        return ""
    return match.group("url").rstrip(".,)")


def _resolution_date(description: str) -> str:
    match = DATE_TEXT_RE.search(description)
    if not match:
        return ""
    month = MONTHS[match.group("month").lower()[:3]]
    day = int(match.group("day"))
    year = 2000 + int(match.group("year"))
    return f"{year:04d}-{month:02d}-{day:02d}"


def _source_summary(family: str, station: str, unit: str, precision: str, url: str) -> str:
    parts = [family]
    if station and station != family:
        parts.append(station)
    parts.append(f"{precision} in {unit}")
    if url:
        parts.append(url)
    return "; ".join(parts)


def _group_key(row: dict[str, Any], meta: SourceMeta) -> tuple[str, str, str, str]:
    return (
        str(row.get("normalized_city") or ""),
        str(row.get("market_date_local") or ""),
        meta.settlement_unit,
        meta.settlement_precision,
    )


def _candidate_values(unit: str, precision: str) -> list[int]:
    if precision.startswith("decimal_0.1"):
        if unit == "C":
            return list(range(-500, 651))
        return list(range(-800, 1510))
    if unit == "C":
        return list(range(-50, 66))
    return list(range(-80, 151))


def _source_value_in_bucket(row: dict[str, Any], meta: SourceMeta, value: int) -> bool:
    lower, upper = _threshold_display_bounds(row, meta)
    if meta.settlement_precision.startswith("decimal_0.1"):
        test_value = value / 10.0
    else:
        test_value = float(value)
    if lower is not None and test_value < lower - 1e-9:
        return False
    if upper is not None and test_value > upper + 1e-9:
        return False
    return True


def _threshold_display_bounds(row: dict[str, Any], meta: SourceMeta) -> tuple[Optional[float], Optional[float]]:
    lower_f = _optional_float(row.get("threshold_low_f"))
    upper_f = _optional_float(row.get("threshold_high_f"))
    lower = _display_value(lower_f, meta.settlement_unit)
    upper = _display_value(upper_f, meta.settlement_unit)
    if meta.settlement_precision.startswith("whole_"):
        if lower is not None:
            lower = round(lower)
        if upper is not None:
            upper = round(upper)
        if lower is not None and upper is not None:
            label = str(row.get("bucket_label") or "").lower()
            if meta.settlement_unit == "C" and "between" not in label:
                upper = lower
    elif meta.settlement_precision.startswith("decimal_0.1"):
        if lower is not None:
            lower = math.floor(lower * 10 + 1e-6) / 10.0
        if upper is not None:
            upper = math.floor(upper * 10 + 1e-6) / 10.0
    return lower, upper


def _canonical_bucket_type(row: dict[str, Any], meta: SourceMeta) -> str:
    lower, upper = _threshold_display_bounds(row, meta)
    if lower is None and upper is None:
        return "unknown"
    if lower is None:
        return "below"
    if upper is None:
        return "above"
    if meta.settlement_precision.startswith("whole_") and lower == upper:
        return "exact"
    if meta.settlement_unit == "F" and meta.settlement_precision.startswith("whole_") and upper - lower <= 1:
        return "exact"
    label = str(row.get("bucket_label") or "").lower()
    if "between" not in label and meta.settlement_unit == "C" and upper - lower < 1.01:
        return "exact"
    return "range"


def _source_rule_proxy_label(row: dict[str, Any], meta: SourceMeta, open_meteo_high_f: Optional[float]) -> Any:
    if open_meteo_high_f is None:
        return ""
    if meta.settlement_precision.startswith("whole_"):
        source_value = _round_half_up(_display_value(open_meteo_high_f, meta.settlement_unit))
    elif meta.settlement_precision.startswith("decimal_0.1"):
        source_value = _round_tenth_half_up(_display_value(open_meteo_high_f, meta.settlement_unit))
    else:
        return ""
    return int(_source_value_in_bucket(row, meta, int(round(source_value * 10)) if meta.settlement_precision.startswith("decimal_0.1") else int(source_value)))


def _continuous_label(value_f: Optional[float], lower_f: Any, upper_f: Any) -> Any:
    if value_f is None:
        return ""
    lower = _optional_float(lower_f)
    upper = _optional_float(upper_f)
    if lower is not None and value_f < lower:
        return 0
    if upper is not None and value_f > upper:
        return 0
    return 1


def _row_flags(
    row: dict[str, Any],
    meta: SourceMeta,
    group: dict[str, Any],
    current_mismatch: int,
    source_rule_mismatch: int,
) -> list[str]:
    flags: list[str] = []
    if current_mismatch:
        flags.append("open_meteo_continuous_disagrees_with_polymarket")
    if row.get("bucket_type") == "range" and _canonical_bucket_type(row, meta) == "exact":
        flags.append("celsius_exact_currently_classified_as_range")
    if source_rule_mismatch:
        flags.append("open_meteo_source_rule_proxy_disagrees_with_polymarket")
    if meta.resolution_date_conflict:
        flags.append("question_resolution_date_conflict")
    if meta.source_family == "unknown" or meta.settlement_precision == "unknown":
        flags.append("unsupported_or_unknown_resolution_source_text")
    if group["constraint_status"] == "contradiction":
        flags.append("polymarket_resolution_constraints_contradict")
    elif group["constraint_status"] == "ambiguous":
        flags.append("city_date_settlement_value_not_exactly_inferred")
    return flags


def _recommended_action(flags: list[str], current_mismatch: int) -> str:
    severe = {
        "question_resolution_date_conflict",
        "unsupported_or_unknown_resolution_source_text",
        "polymarket_resolution_constraints_contradict",
    }
    if any(flag in severe for flag in flags):
        return "exclude_or_manual_review"
    if current_mismatch:
        return "relabel_to_polymarket"
    return "keep_polymarket_label"


def _mismatch(label: Any, resolved: int) -> int:
    if label == "":
        return 0
    return int(int(label) != int(resolved))


def _display_value(value_f: Optional[float], unit: str) -> Optional[float]:
    if value_f is None:
        return None
    if unit == "C":
        return (float(value_f) - 32.0) * 5.0 / 9.0
    return float(value_f)


def _round_half_up(value: Optional[float]) -> int:
    if value is None:
        return 0
    return int(Decimal(str(value)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def _round_tenth_half_up(value: Optional[float]) -> float:
    if value is None:
        return 0.0
    return float(Decimal(str(value)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def _format_source_value(value: int, precision: str) -> str:
    if precision.startswith("decimal_0.1"):
        return f"{value / 10.0:.1f}"
    return str(value)


def _compact_values(values: list[int], precision: str) -> str:
    if not values:
        return ""
    values = sorted(values)
    ranges: list[tuple[int, int]] = []
    start = prev = values[0]
    for value in values[1:]:
        if value == prev + 1:
            prev = value
            continue
        ranges.append((start, prev))
        start = prev = value
    ranges.append((start, prev))
    parts = []
    for start, end in ranges:
        if start == end:
            parts.append(_format_source_value(start, precision))
        else:
            parts.append(f"{_format_source_value(start, precision)}..{_format_source_value(end, precision)}")
    return ",".join(parts)


def _distance_to_threshold_f(value_f: Optional[float], lower_f: Any, upper_f: Any) -> Optional[float]:
    if value_f is None:
        return None
    distances = []
    lower = _optional_float(lower_f)
    upper = _optional_float(upper_f)
    if lower is not None:
        distances.append(abs(float(value_f) - lower))
    if upper is not None:
        distances.append(abs(float(value_f) - upper))
    return min(distances) if distances else None


def _margin_bucket(value: Optional[float]) -> str:
    if value is None:
        return "unknown"
    if value <= 0.5:
        return "0-0.5F"
    if value <= 1.0:
        return "0.5-1F"
    if value <= 2.0:
        return "1-2F"
    if value <= 4.0:
        return "2-4F"
    return "4F+"


def _round_or_none(value: Any, digits: int) -> Any:
    numeric = _optional_float(value)
    return "" if numeric is None else round(numeric, digits)


def _optional_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _rate(numerator: int, denominator: int) -> Any:
    if denominator <= 0:
        return None
    return round(numerator / denominator, 6)


def _connect_readonly(path: Path) -> sqlite3.Connection:
    uri = f"file:{path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only=ON")
    return conn


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _render_report(summary: dict[str, Any], labels_csv: Path, actions_csv: Path, groups_csv: Path, summary_json: Path) -> str:
    bucket_lines = _table_lines(summary["by_current_bucket_type"], ["group", "n", "current_mismatch_n", "current_mismatch_rate"], limit=8)
    source_lines = _table_lines(summary["by_source_family"], ["group", "n", "current_mismatch_n", "current_mismatch_rate"], limit=8)
    distance_lines = _table_lines(summary["by_distance_to_threshold_bucket_f"], ["group", "n", "current_mismatch_n", "current_mismatch_rate"], limit=8)
    city_lines = _table_lines(summary["by_city_top_mismatch"], ["group", "n", "current_mismatch_n", "current_mismatch_rate"], limit=12)
    action_lines = "\n".join(f"- `{key}`: `{value}`" for key, value in sorted(summary["by_recommended_action"].items()))
    constraints = "\n".join(f"- `{key}`: `{value}`" for key, value in sorted(summary["group_constraint_status"].items()))
    return "\n".join(
        [
            "# DailyWeather Settlement / Label Audit",
            "",
            f"Generated: {summary['generated_at_utc']}",
            f"Window: `{summary['target_window']['start']}` through `{summary['target_window']['end']}`",
            "",
            "## Decision",
            "",
            "Use Polymarket/Telonex resolved outcomes as the canonical bucket settlement label for training bucket classifiers and replaying PnL. Open-Meteo realized highs are not canonical settlement labels because the market rules name station-specific sources and source-specific display precision.",
            "",
            "Open-Meteo realized highs may remain useful for weather-model regression, but any bucket-level training or PnL replay must either use the Polymarket label or explicitly exclude rows where an Open-Meteo-derived label is being evaluated as the target.",
            "",
            "## Headline Counts",
            "",
            f"- Market/bucket rows audited: `{summary['row_count']}`.",
            f"- City/date/source-unit groups audited: `{summary['city_date_group_count']}`.",
            f"- Current Open-Meteo continuous-label mismatches: `{summary['open_meteo_continuous_mismatch_count']}` (`{summary['open_meteo_continuous_mismatch_rate']}`).",
            f"- Rows within 0-2F of a threshold: `{summary['within_0_2f_threshold_rows']}` with `{summary['within_0_2f_threshold_mismatch_count']}` mismatches (`{summary['within_0_2f_threshold_mismatch_rate']}`).",
            f"- Current exact/range rows: `{summary['exact_or_range_rows']}` with `{summary['exact_or_range_mismatch_count']}` mismatches (`{summary['exact_or_range_mismatch_rate']}`).",
            f"- Celsius exact rows currently classified as range: `{summary['celsius_exact_reclassified_count']}`.",
            f"- All current `range` rows reclassified as exact under source-unit precision: `{summary['current_range_reclassified_exact_count']}`.",
            f"- Rows recommended for Open-Meteo-to-Polymarket relabeling: `{summary['current_open_meteo_rows_to_relabel']}`.",
            f"- Rows excluded/manual-review for replay: `{summary['rows_to_exclude_from_backtest_replay']}`.",
            "",
            "## Why Mismatches Happen",
            "",
            "1. The current label computation tests decimal Open-Meteo Fahrenheit highs directly against parsed thresholds.",
            "2. The settlement rules usually point to Wunderground or NOAA airport/station pages and say to use whole-degree precision in the market's displayed unit; Taipei CWA rows use one-decimal Celsius.",
            "3. Open-Meteo is a gridded/archive weather product, not the named station settlement source. Even when rounding is applied, station/source differences remain.",
            "4. The parser currently classifies Celsius single-degree markets as `range` because `1C` is `1.8F`; these should be canonical `exact` bucket rows under source-unit precision.",
            "5. A small set of rows has source-text hazards such as question/resolution date conflicts or unsupported source text and should not be used for replay without manual review.",
            "",
            "## Current Bucket Type",
            "",
            *bucket_lines,
            "",
            "## Threshold Distance",
            "",
            *distance_lines,
            "",
            "## Settlement Source",
            "",
            *source_lines,
            "",
            "## Highest-Mismatch Cities",
            "",
            *city_lines,
            "",
            "## Recommended Actions",
            "",
            action_lines,
            "",
            "## City-Date Constraint Status",
            "",
            constraints,
            "",
            "## Canonical Policy",
            "",
            "- `canonical_label` is the resolved YES payout from Polymarket/Telonex final token prices.",
            "- Backtest settlement and PnL replay must use `canonical_label`, not an Open-Meteo-derived bucket label.",
            "- Bucket-level model training should use `canonical_label` when the question/date/source text is internally consistent.",
            "- Daily-high regression can use Open-Meteo realized highs only as an Open-Meteo target; do not treat those targets as official market settlement highs.",
            "- Exact/range membership must be evaluated in the settlement source unit and precision from the rules text, not by raw Fahrenheit thresholds alone.",
            "- Rows with `exclude_or_manual_review` should be kept out of live-like replay and promotion diagnostics until manually remapped or verified.",
            "",
            "## Artifacts",
            "",
            f"- Canonical labels: `{labels_csv}`",
            f"- Rows to relabel/exclude: `{actions_csv}`",
            f"- City-date constraints: `{groups_csv}`",
            f"- Machine summary: `{summary_json}`",
            "",
        ]
    )


def _table_lines(rows: list[dict[str, Any]], columns: list[str], *, limit: int) -> list[str]:
    if not rows:
        return ["No rows."]
    output = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows[:limit]:
        output.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return output


def _parse_args(argv: Optional[list[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Open-Meteo labels against Polymarket weather-market settlements.")
    parser.add_argument("--work-root", default=".", help="Weather Trader private workspace root.")
    parser.add_argument("--start-date", default=SUPPORTED_START)
    parser.add_argument("--end-date", default=SUPPORTED_END)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
