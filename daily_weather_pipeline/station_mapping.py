from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import urllib.parse
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional


DEFAULT_START_DATE = date(2026, 1, 1)
DEFAULT_END_DATE = date(2026, 6, 29)
DEFAULT_LABELS_CSV = Path(
    "work/reports/settlement_label_audit/20260701T090014Z-settlement-label-audit-canonical-labels.csv"
)


@dataclass(frozen=True)
class MappingPaths:
    work_root: Path

    @property
    def decision_db(self) -> Path:
        return self.work_root / "work/data/backtest/daily_weather_decision_artifact.sqlite"

    @property
    def labels_csv(self) -> Path:
        return self.work_root / DEFAULT_LABELS_CSV

    @property
    def report_dir(self) -> Path:
        return self.work_root / "work/reports/station_mapping"


CSV_FIELDS = [
    "normalized_city",
    "market_family",
    "source_family",
    "settlement_source_named",
    "likely_station_vendor",
    "station_name",
    "station_id",
    "source_url",
    "settlement_unit",
    "settlement_precision",
    "rounding_rule",
    "timezone",
    "local_market_day_definition",
    "historical_observations_available",
    "live_observations_available",
    "intraday_high_so_far_point_in_time",
    "market_rows",
    "decision_rows",
    "quote_covered_decision_rows",
    "quote_coverage_pct",
    "spread_lte_10c_rows",
    "canonical_bucket_types",
    "recommended_action_counts",
    "exclude_or_manual_review_rows",
    "expected_data_quality_risks",
    "phase_b_suitable",
    "phase_c_suitable",
    "ingestion_status",
    "notes",
]


def build_station_mapping(
    *,
    work_root: Path,
    start: date = DEFAULT_START_DATE,
    end: date = DEFAULT_END_DATE,
    labels_csv: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> dict[str, Any]:
    paths = MappingPaths(work_root.resolve())
    labels_path = labels_csv.resolve() if labels_csv else paths.labels_csv
    report_dir = output_dir.resolve() if output_dir else paths.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)

    decision_stats = _load_decision_stats(paths.decision_db)
    audit_stats = _load_audit_stats(labels_path, start=start, end=end)
    rows = _build_rows(decision_stats=decision_stats, audit_stats=audit_stats)
    recommendations = _rank_recommended_ingestion(rows)
    ready_statuses = {"ready", "ready_with_source_url_filter", "ready_with_manual_review_filters"}
    blocked_statuses = {"blocked_missing_station_id", "needs_custom_connector_probe", "ready_after_noaa_historical_probe"}
    ready = [row["normalized_city"] for row in rows if row["ingestion_status"] in ready_statuses]
    ready_source_rows = _dedupe_preserve_order([
        _city_source_label(row)
        for row in rows
        if row["ingestion_status"] in ready_statuses
    ])
    blocked = _dedupe_preserve_order([
        _city_source_label(row)
        for row in rows
        if row["ingestion_status"] in blocked_statuses
    ])
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    prefix = report_dir / f"{timestamp}-six-month-v1-settlement-station-mapping"
    csv_path = prefix.with_suffix(".csv")
    json_path = prefix.with_suffix(".json")
    md_path = prefix.with_suffix(".md")

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_window": {"start": start.isoformat(), "end": end.isoformat()},
        "inputs": {
            "decision_db": str(paths.decision_db),
            "canonical_labels_csv": str(labels_path),
        },
        "artifact_paths": {
            "csv": str(csv_path),
            "json": str(json_path),
            "markdown": str(md_path),
        },
        "counts": {
            "mapping_rows": len(rows),
            "cities": len({row["normalized_city"] for row in rows}),
            "ready_source_rows": len(ready_source_rows),
            "blocked_or_ambiguous_source_rows": len(blocked),
            "decision_rows": sum(int(row["decision_rows"]) for row in rows),
            "quote_covered_decision_rows": sum(int(row["quote_covered_decision_rows"]) for row in rows),
        },
        "ready_cities_for_historical_station_ingestion": sorted(set(ready)),
        "ready_source_rows_for_historical_station_ingestion": ready_source_rows,
        "blocked_or_ambiguous_source_rows": blocked,
        "recommended_first_city_source_rows": recommendations,
        "rows": rows,
        "notes": [
            "This is a mapping and feasibility artifact only; it does not pull historical station observations.",
            "Settlement source, station, unit, precision, and source URL come from the corrected canonical settlement-label audit joined into the six_month_v1 decision artifact.",
            "Wunderground rows are treated as named settlement pages whose station IDs are ICAO-style airport identifiers; ingestion should still preserve the Wunderground source URL for auditability.",
            "Historical/live availability is a feasibility classification from source family and parsed station ID, not proof from a fresh large data pull.",
        ],
    }

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(_render_markdown(summary), encoding="utf-8")
    return summary


def _load_decision_stats(decision_db: Path) -> dict[tuple[str, str, str, str, str, str], dict[str, Any]]:
    if not decision_db.exists():
        raise FileNotFoundError(decision_db)
    query = """
        SELECT
          normalized_city,
          timezone,
          canonical_source_family,
          canonical_source_station,
          canonical_source_url,
          settlement_unit,
          settlement_precision,
          COUNT(*) AS decision_rows,
          SUM(CASE WHEN quote_timestamp_utc IS NOT NULL THEN 1 ELSE 0 END) AS quote_covered_decision_rows,
          SUM(CASE WHEN spread IS NOT NULL AND spread <= 0.10 THEN 1 ELSE 0 END) AS spread_lte_10c_rows,
          GROUP_CONCAT(DISTINCT canonical_bucket_type) AS canonical_bucket_types
        FROM decision_snapshots_current_enriched
        GROUP BY
          normalized_city,
          timezone,
          canonical_source_family,
          canonical_source_station,
          canonical_source_url,
          settlement_unit,
          settlement_precision
        ORDER BY normalized_city, canonical_source_family, canonical_source_station
    """
    with sqlite3.connect(f"file:{decision_db}?mode=ro", uri=True) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(query).fetchall()
    output: dict[tuple[str, str, str, str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (
            row["normalized_city"] or "",
            row["canonical_source_family"] or "",
            row["canonical_source_station"] or "",
            row["canonical_source_url"] or "",
            row["settlement_unit"] or "",
            row["settlement_precision"] or "",
        )
        decision_rows = int(row["decision_rows"] or 0)
        quote_rows = int(row["quote_covered_decision_rows"] or 0)
        output[key] = {
            "normalized_city": row["normalized_city"] or "",
            "timezone": row["timezone"] or "",
            "source_family": row["canonical_source_family"] or "",
            "source_station": row["canonical_source_station"] or "",
            "source_url": row["canonical_source_url"] or "",
            "settlement_unit": row["settlement_unit"] or "",
            "settlement_precision": row["settlement_precision"] or "",
            "decision_rows": decision_rows,
            "quote_covered_decision_rows": quote_rows,
            "quote_coverage_pct": round(100.0 * quote_rows / decision_rows, 2) if decision_rows else 0.0,
            "spread_lte_10c_rows": int(row["spread_lte_10c_rows"] or 0),
            "canonical_bucket_types": _sorted_csv(row["canonical_bucket_types"] or ""),
        }
    return output


def _load_audit_stats(
    labels_csv: Path,
    *,
    start: date,
    end: date,
) -> dict[tuple[str, str, str, str, str, str], dict[str, Any]]:
    if not labels_csv.exists():
        raise FileNotFoundError(labels_csv)
    output: dict[tuple[str, str, str, str, str, str], dict[str, Any]] = {}
    with labels_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_date = _parse_date(row.get("market_date_local", ""))
            if row_date is None or row_date < start or row_date > end:
                continue
            key = (
                row.get("normalized_city", ""),
                row.get("source_family", ""),
                row.get("source_station", ""),
                row.get("source_url", ""),
                row.get("settlement_unit", ""),
                row.get("settlement_precision", ""),
            )
            stats = output.setdefault(
                key,
                {
                    "market_rows": 0,
                    "action_counts": Counter(),
                    "audit_flag_counts": Counter(),
                    "source_text_summary": row.get("source_text_summary", ""),
                    "settlement_source": row.get("settlement_source", ""),
                    "timezone": row.get("timezone", ""),
                    "first_market_date_local": row.get("market_date_local", ""),
                    "last_market_date_local": row.get("market_date_local", ""),
                },
            )
            stats["market_rows"] += 1
            action = row.get("recommended_action", "") or "unknown"
            stats["action_counts"][action] += 1
            for flag in _split_flags(row.get("audit_flags", "")):
                stats["audit_flag_counts"][flag] += 1
            stats["first_market_date_local"] = min(stats["first_market_date_local"], row.get("market_date_local", ""))
            stats["last_market_date_local"] = max(stats["last_market_date_local"], row.get("market_date_local", ""))
    return output


def _build_rows(
    *,
    decision_stats: Mapping[tuple[str, str, str, str, str, str], Mapping[str, Any]],
    audit_stats: Mapping[tuple[str, str, str, str, str, str], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    all_keys = sorted(set(decision_stats) | set(audit_stats))
    by_city_key_count = Counter(key[0] for key in all_keys)
    rows: list[dict[str, Any]] = []
    for key in all_keys:
        city, family, station, url, unit, precision = key
        decision = dict(decision_stats.get(key, {}))
        audit = dict(audit_stats.get(key, {}))
        station_id = _station_id(family, url)
        source_feasibility = _source_feasibility(family, station_id)
        risks = _risk_notes(
            city=city,
            family=family,
            station=station,
            precision=precision,
            city_source_count=by_city_key_count[city],
            audit_flags=audit.get("audit_flag_counts", Counter()),
        )
        ingestion_status = _ingestion_status(
            family=family,
            station_id=station_id,
            city_source_count=by_city_key_count[city],
            risks=risks,
        )
        rows.append(
            {
                "normalized_city": city,
                "market_family": "daily_high_temperature",
                "source_family": family,
                "settlement_source_named": audit.get("source_text_summary")
                or _source_summary(family, station, unit, precision, url),
                "likely_station_vendor": source_feasibility["vendor"],
                "station_name": _normalized_station_name(city, station, station_id),
                "station_id": station_id,
                "source_url": url,
                "settlement_unit": unit,
                "settlement_precision": precision,
                "rounding_rule": _rounding_rule(unit, precision),
                "timezone": decision.get("timezone") or audit.get("timezone", ""),
                "local_market_day_definition": _market_day_definition(
                    decision.get("timezone") or audit.get("timezone", "")
                ),
                "historical_observations_available": source_feasibility["historical"],
                "live_observations_available": source_feasibility["live"],
                "intraday_high_so_far_point_in_time": source_feasibility["intraday"],
                "market_rows": audit.get("market_rows", 0),
                "decision_rows": decision.get("decision_rows", 0),
                "quote_covered_decision_rows": decision.get("quote_covered_decision_rows", 0),
                "quote_coverage_pct": decision.get("quote_coverage_pct", 0.0),
                "spread_lte_10c_rows": decision.get("spread_lte_10c_rows", 0),
                "canonical_bucket_types": decision.get("canonical_bucket_types", ""),
                "recommended_action_counts": _counter_compact(audit.get("action_counts", Counter())),
                "exclude_or_manual_review_rows": int(
                    audit.get("action_counts", Counter()).get("exclude_or_manual_review", 0)
                ),
                "expected_data_quality_risks": "; ".join(risks),
                "phase_b_suitable": _phase_suitability("B", ingestion_status, risks),
                "phase_c_suitable": _phase_suitability("C", ingestion_status, risks),
                "ingestion_status": ingestion_status,
                "notes": _row_notes(family, city, station_id, by_city_key_count[city], risks),
            }
        )
    return rows


def _station_id(family: str, url: str) -> str:
    parsed = urllib.parse.urlparse(url or "")
    if family == "Wunderground":
        bits = [bit for bit in parsed.path.split("/") if bit]
        return bits[-1].upper() if bits else ""
    if family == "NOAA":
        query = urllib.parse.parse_qs(parsed.query)
        return (query.get("site") or [""])[0].upper()
    if family == "Taipei CWA":
        query = urllib.parse.parse_qs(parsed.query)
        return (query.get("ID") or [""])[0]
    return ""


def _source_feasibility(family: str, station_id: str) -> dict[str, str]:
    if family == "Wunderground":
        return {
            "vendor": "Wunderground history page; ICAO/METAR airport station",
            "historical": "yes_likely_wunderground_history_and_metar_archive",
            "live": "yes_likely_metar_or_wunderground_recent_observations",
            "intraday": "yes_if_timestamped_station_observations_are_ingested_as_of_decision_time",
        }
    if family == "NOAA":
        return {
            "vendor": "NOAA/weather.gov WRH timeseries; ICAO/METAR airport station",
            "historical": "needs_probe_weather_gov_recent_plus_noaa_metar_isd_archive",
            "live": "yes_likely_weather_gov_or_metar_recent_observations",
            "intraday": "yes_if_metar_weather_gov_samples_are_archived_with_observed_at_utc",
        }
    if family == "Taipei CWA":
        return {
            "vendor": "Taiwan Central Weather Administration official station",
            "historical": "needs_custom_cwa_connector_probe",
            "live": "yes_likely_cwa_recent_observations",
            "intraday": "yes_if_cwa_polling_or_archive_keeps_timestamped_samples",
        }
    return {
        "vendor": "unknown",
        "historical": "unknown",
        "live": "unknown",
        "intraday": "unknown",
    }


def _risk_notes(
    *,
    city: str,
    family: str,
    station: str,
    precision: str,
    city_source_count: int,
    audit_flags: Counter[str],
) -> list[str]:
    risks: list[str] = []
    if city_source_count > 1:
        risks.append("city_has_multiple_settlement_source_rows; preserve per-market source_url/station_id")
    if "Buckly" in station:
        risks.append("source_station_typo_alias; use station_id KBKF/source_url as canonical")
    if precision == "decimal_0.1_C":
        risks.append("decimal_celsius_precision_requires_exact_tenth_degree_handling")
    if family == "Taipei CWA":
        risks.append("requires_cwa_specific_connector_and_timezone_validation")
    if family == "NOAA":
        risks.append("weather_gov_timeseries_historical_depth_must_be_verified")
    for flag, count in audit_flags.most_common():
        if flag == "celsius_exact_currently_classified_as_range":
            risks.append(f"{flag}:{count}; use canonical_bucket_type/source_unit_precision")
        elif flag == "city_date_settlement_value_not_exactly_inferred":
            risks.append(f"{flag}:{count}; daily high cannot always be inferred exactly from bucket payouts")
        elif flag in {
            "polymarket_resolution_constraints_contradict",
            "question_resolution_date_conflict",
            "open_meteo_continuous_disagrees_with_polymarket",
            "open_meteo_source_rule_proxy_disagrees_with_polymarket",
        }:
            risks.append(f"{flag}:{count}")
    return _dedupe_preserve_order(risks)


def _ingestion_status(*, family: str, station_id: str, city_source_count: int, risks: Iterable[str]) -> str:
    risk_text = " ".join(risks)
    if not station_id:
        return "blocked_missing_station_id"
    if family == "Taipei CWA":
        return "needs_custom_connector_probe"
    if family == "NOAA":
        return "ready_after_noaa_historical_probe"
    if "question_resolution_date_conflict" in risk_text or "polymarket_resolution_constraints_contradict" in risk_text:
        return "ready_with_manual_review_filters"
    if city_source_count > 1:
        return "ready_with_source_url_filter"
    return "ready"


def _phase_suitability(phase: str, ingestion_status: str, risks: Iterable[str]) -> str:
    risk_text = " ".join(risks)
    if ingestion_status.startswith("blocked"):
        return "no"
    if "question_resolution_date_conflict" in risk_text or "polymarket_resolution_constraints_contradict" in risk_text:
        return "yes_after_manual_review_filters"
    if ingestion_status in {"needs_custom_connector_probe", "ready_after_noaa_historical_probe"}:
        return "yes_after_connector_probe"
    if ingestion_status == "ready_with_source_url_filter":
        return "yes_if_replay_keys_by_source_url"
    if phase == "C" and "city_date_settlement_value_not_exactly_inferred" in risk_text:
        return "yes_for_station_observation_trigger_not_bucket_inference"
    return "yes"


def _rank_recommended_ingestion(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def score(row: Mapping[str, Any]) -> tuple[int, int, float, int]:
        if row["source_family"] == "Wunderground" and str(row["station_id"]).startswith("K"):
            availability_score = 5
        elif row["source_family"] == "Wunderground":
            availability_score = 4
        elif row["source_family"] == "NOAA":
            availability_score = 2
        elif row["source_family"] == "Taipei CWA":
            availability_score = 1
        else:
            availability_score = 0
        return (
            availability_score,
            int(row["quote_covered_decision_rows"]),
            float(row["quote_coverage_pct"]),
            int(row["spread_lte_10c_rows"]),
        )

    candidates = [
        row
        for row in rows
        if row["ingestion_status"]
        in {"ready", "ready_with_source_url_filter", "ready_with_manual_review_filters"}
    ]
    ranked = sorted(candidates, key=score, reverse=True)
    output = []
    seen_cities: set[str] = set()
    for row in ranked:
        if row["normalized_city"] in seen_cities:
            continue
        seen_cities.add(row["normalized_city"])
        output.append(
            {
                "rank": len(output) + 1,
                "city_source": _city_source_label(row),
                "station_id": row["station_id"],
                "source_family": row["source_family"],
                "quote_covered_decision_rows": row["quote_covered_decision_rows"],
                "quote_coverage_pct": row["quote_coverage_pct"],
                "spread_lte_10c_rows": row["spread_lte_10c_rows"],
                "reason": _recommendation_reason(row),
            }
        )
        if len(output) >= 10:
            break
    return output


def _recommendation_reason(row: Mapping[str, Any]) -> str:
    if str(row["station_id"]).startswith("K"):
        suffix = "manual-review settlement rows must still be filtered later" if row["ingestion_status"] == "ready_with_manual_review_filters" else "straightforward first station pull"
        return f"US airport ICAO station with straightforward METAR/ASOS history and strong quote coverage proxy; {suffix}"
    if row["ingestion_status"] == "ready_with_source_url_filter":
        return "large quote-covered slice, but ingestion must key by source URL because city has source switches"
    return "large quote-covered slice and parsed airport station source"


def _render_markdown(summary: Mapping[str, Any]) -> str:
    counts = summary["counts"]
    lines = [
        "# six_month_v1 Settlement Station Mapping",
        "",
        f"Generated: `{summary['generated_at_utc']}`",
        f"Window: `{summary['target_window']['start']}` through `{summary['target_window']['end']}`",
        "",
        "## Scope",
        "",
        "This is a mapping and feasibility report only. It does not pull a historical station dataset.",
        "",
        f"- Cities: `{counts['cities']}`",
        f"- City/source mapping rows: `{counts['mapping_rows']}`",
        f"- Decision rows represented: `{counts['decision_rows']}`",
        f"- Quote-covered decision rows represented: `{counts['quote_covered_decision_rows']}`",
        f"- Station-ingestion-ready source rows: `{counts['ready_source_rows']}`",
        f"- Probe/custom-connector source rows: `{counts['blocked_or_ambiguous_source_rows']}`",
        "",
        "## Recommended First Pull",
        "",
        "| Rank | City/source | Station | Quote rows | Quote coverage | Reason |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in summary["recommended_first_city_source_rows"]:
        lines.append(
            "| {rank} | {city_source} | `{station_id}` | `{quote_covered_decision_rows}` | `{quote_coverage_pct}%` | {reason} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Ready Source Rows",
            "",
            *[f"- {item}" for item in summary["ready_source_rows_for_historical_station_ingestion"]],
            "",
            "## Blocked Or Ambiguous Source Rows",
            "",
        ]
    )
    blocked = summary["blocked_or_ambiguous_source_rows"]
    lines.extend([f"- {item}" for item in blocked] if blocked else ["- None."])
    lines.extend(
        [
            "",
            "## Mapping Rows",
            "",
            "| City | Source | Station | Unit | Precision | Decisions | Quote rows | Status | Risks |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in summary["rows"]:
        risks = str(row["expected_data_quality_risks"]).replace("|", "/")
        station = f"{row['station_name']} ({row['station_id']})"
        lines.append(
            f"| {row['normalized_city']} | {row['source_family']} | {station} | {row['settlement_unit']} | {row['settlement_precision']} | `{row['decision_rows']}` | `{row['quote_covered_decision_rows']}` | `{row['ingestion_status']}` | {risks} |"
        )
    lines.extend(
        [
            "",
            "## Rules For Downstream Agents",
            "",
            "- Key station data by `source_url` and `station_id`, not city alone.",
            "- Preserve the original settlement source URL and source family in replay artifacts.",
            "- Evaluate bucket membership in `settlement_unit` at `settlement_precision`; do not reuse raw Fahrenheit thresholds for Celsius exact markets.",
            "- For Phase B/C observed-high features, use only timestamped station observations with `observed_at_utc <= decision_time_utc`.",
            "- Keep `exclude_or_manual_review` settlement rows out of strategy replay until manually resolved.",
            "",
            "## Artifacts",
            "",
            f"- CSV: `{summary['artifact_paths']['csv']}`",
            f"- JSON: `{summary['artifact_paths']['json']}`",
            f"- Markdown: `{summary['artifact_paths']['markdown']}`",
            "",
        ]
    )
    return "\n".join(lines)


def _rounding_rule(unit: str, precision: str) -> str:
    if precision.startswith("whole_"):
        return f"Use source-displayed {unit}; whole-degree precision; parser proxy uses half-up rounding before bucket comparison."
    if precision == "decimal_0.1_C":
        return "Use source-displayed Celsius to one decimal place; parser proxy uses half-up rounding to nearest tenth."
    return "Unknown precision; require manual source-text review."


def _market_day_definition(timezone_name: str) -> str:
    if not timezone_name:
        return "calendar day in parsed market timezone; timezone missing from row"
    return f"00:00:00 through 23:59:59 local calendar day in {timezone_name}; convert observation timestamps to this local day before computing high."


def _source_summary(family: str, station: str, unit: str, precision: str, url: str) -> str:
    return f"{family}; {station}; {precision} in {unit}; {url}"


def _normalized_station_name(city: str, station: str, station_id: str) -> str:
    if city == "Denver, CO" and station_id == "KBKF":
        return "Buckley Space Force Base Station"
    return station


def _row_notes(family: str, city: str, station_id: str, city_source_count: int, risks: Iterable[str]) -> str:
    notes = []
    if city_source_count > 1:
        notes.append("Do not collapse this city to one station; the six_month_v1 source URL changes by market/date.")
    if family == "Wunderground":
        notes.append("Use Wunderground URL for audit; prefer timestamped METAR/airport observations if they match displayed source.")
    if family == "NOAA":
        notes.append("Probe weather.gov timeseries and NOAA/IEM METAR archive before large historical pull.")
    if family == "Taipei CWA":
        notes.append("Build a CWA-specific connector before replaying this subset.")
    if not risks:
        notes.append("No additional source-specific risk identified from settlement audit.")
    return " ".join(notes)


def _city_source_label(row: Mapping[str, Any]) -> str:
    return f"{row['normalized_city']} / {row['source_family']} / {row['station_id']} / {row['station_name']}"


def _counter_compact(counter: Counter[str]) -> str:
    if not counter:
        return ""
    return ";".join(f"{key}={value}" for key, value in sorted(counter.items()))


def _split_flags(value: str) -> list[str]:
    return [part for part in (value or "").split("|") if part]


def _dedupe_preserve_order(values: Iterable[str]) -> list[str]:
    seen = set()
    output = []
    for value in values:
        if value not in seen:
            output.append(value)
            seen.add(value)
    return output


def _sorted_csv(value: str) -> str:
    return ",".join(sorted(part for part in value.split(",") if part))


def _parse_date(value: str) -> Optional[date]:
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _parse_date_arg(value: str) -> date:
    return date.fromisoformat(value)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build six_month_v1 settlement station mapping artifacts.")
    parser.add_argument("--work-root", default=".")
    parser.add_argument("--start-date", default=DEFAULT_START_DATE.isoformat())
    parser.add_argument("--end-date", default=DEFAULT_END_DATE.isoformat())
    parser.add_argument("--labels-csv", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)
    summary = build_station_mapping(
        work_root=Path(args.work_root),
        start=_parse_date_arg(args.start_date),
        end=_parse_date_arg(args.end_date),
        labels_csv=Path(args.labels_csv) if args.labels_csv else None,
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )
    print(json.dumps({"artifact_paths": summary["artifact_paths"], "counts": summary["counts"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
