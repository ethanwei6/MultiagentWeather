from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sqlite3
import time
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, time as datetime_time, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional
from zoneinfo import ZoneInfo


FIRST_10_STATIONS = ("KLGA", "KDAL", "KMIA", "KORD", "KSEA", "KHOU", "KAUS", "KLAX", "KSFO", "KBKF")
PROBE_SUPPORTED_SOURCE_FAMILIES = frozenset({"Wunderground", "NOAA"})
READY_INGESTION_STATUSES = frozenset(
    {"ready", "ready_with_source_url_filter", "ready_with_manual_review_filters", "ready_after_noaa_historical_probe"}
)
DEFAULT_START_DATE = date(2026, 1, 1)
DEFAULT_END_DATE = date(2026, 6, 29)
DEFAULT_PROOF_START_DATE = date(2026, 1, 1)
DEFAULT_PROOF_END_DATE = date(2026, 1, 3)
DEFAULT_MAPPING_JSON = Path(
    "work/reports/station_mapping/20260707T102258Z-six-month-v1-settlement-station-mapping.json"
)
DEFAULT_LABELS_CSV = Path(
    "work/reports/settlement_label_audit/20260701T090014Z-settlement-label-audit-canonical-labels.csv"
)
IEM_ASOS_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
AVIATIONWEATHER_LIVE_METAR_URL = "https://aviationweather.gov/api/data/metar"
DECISION_HOURS_UTC = (0, 6, 12, 18)
USER_AGENT = "DailyWeather-DataPipeline/0.1 station-observation-proof"
LABEL_AGREEMENT_MIN = 0.95
NEAR_THRESHOLD_DISTANCE_F = 0.5
NEAR_THRESHOLD_LABEL_AGREEMENT_MIN = 0.70
STATION_SOURCE_KEY_SEPARATOR = "||"

COASTAL_INLAND_CLASS_BY_CITY = {
    "Austin, TX": "inland_proxy",
    "Beijing, CN": "inland_proxy",
    "Chicago, IL": "great_lakes_coastal_proxy",
    "Dallas, TX": "inland_proxy",
    "Denver, CO": "inland_high_elevation_proxy",
    "Guangzhou, CN": "coastal_delta_proxy",
    "Houston, TX": "inland_gulf_influenced_proxy",
    "London, GB": "estuary_or_urban_airport_proxy",
    "Los Angeles, CA": "coastal_basin_proxy",
    "Madrid, ES": "inland_proxy",
    "Miami, FL": "coastal_proxy",
    "Moscow, RU": "inland_proxy",
    "New York, NY": "coastal_or_bay_proxy",
    "Paris, FR": "inland_proxy",
    "San Francisco, CA": "coastal_or_bay_proxy",
    "Seattle, WA": "coastal_or_sound_proxy",
    "Seoul, KR": "coastal_airport_proxy",
    "Shanghai, CN": "coastal_delta_proxy",
    "Shenzhen, CN": "coastal_proxy",
    "Singapore, SG": "coastal_island_proxy",
    "Taipei, TW": "near_coast_basin_proxy",
    "Tokyo, JP": "coastal_bay_proxy",
}

PRECISE_T_RE = re.compile(r"(?:^|\s)T(?P<t_sign>[01])(?P<t_tenths>\d{3})(?P<d_sign>[01])(?P<d_tenths>\d{3})(?:\s|$)")
MAIN_TEMP_RE = re.compile(r"(?:^|\s)(?P<temp>M?\d{2})/(?P<dew>M?\d{2}|//)(?:\s|$)")


@dataclass(frozen=True)
class StationConfig:
    station_id: str
    normalized_city: str
    timezone_name: str
    settlement_source_family: str
    settlement_source_url: str
    settlement_station_name: str
    settlement_unit: str
    settlement_precision: str
    mapping_ingestion_status: str = ""
    historical_source_family: str = "IEM_ASOS_METAR"
    live_source_family: str = "NOAA_AVIATIONWEATHER_METAR"


@dataclass(frozen=True)
class StationConfigSelection:
    configs: list[StationConfig]
    skipped_source_rows: list[dict[str, Any]]


@dataclass(frozen=True)
class Observation:
    station_id: str
    raw_station_id: str
    settlement_source_url: str
    observation_source_url: str
    source_family: str
    observed_at_utc: str
    local_date: str
    local_time: str
    timezone_name: str
    raw_temperature: Optional[float]
    source_unit: str
    temperature_f: Optional[float]
    temperature_c: Optional[float]
    station_latitude: Optional[float]
    station_longitude: Optional[float]
    station_elevation_m: Optional[float]
    fetch_timestamp_utc: str
    source_payload_id: str
    quality_flags: tuple[str, ...]
    raw_metar: str
    raw_row_json: str


@dataclass(frozen=True)
class StationMetadata:
    station_id: str
    settlement_source_url: str
    settlement_source_family: str
    historical_source_family: str
    normalized_city: str
    settlement_station_name: str
    timezone_name: str
    settlement_unit: str
    settlement_precision: str
    station_latitude: Optional[float]
    station_longitude: Optional[float]
    station_elevation_m: Optional[float]
    city_latitude: Optional[float]
    city_longitude: Optional[float]
    city_to_station_distance_km: Optional[float]
    coastal_inland_class: str
    urban_airport_notes: str
    metadata_source: str
    source_payload_id: str
    raw_metadata_json: str
    updated_at_utc: str
    quality_flags: tuple[str, ...]


def run_station_observation_pipeline(
    *,
    work_root: Path,
    start: date = DEFAULT_START_DATE,
    end: date = DEFAULT_END_DATE,
    proof_start: date = DEFAULT_PROOF_START_DATE,
    proof_end: date = DEFAULT_PROOF_END_DATE,
    station_ids: Optional[Iterable[str]] = None,
    mapping_json: Optional[Path] = None,
    labels_csv: Optional[Path] = None,
    mode: str = "proof_then_full",
    cache_only: bool = False,
) -> dict[str, Any]:
    root = work_root.resolve()
    mapping_path = (mapping_json or (root / DEFAULT_MAPPING_JSON)).resolve()
    labels_path = (labels_csv or (root / DEFAULT_LABELS_CSV)).resolve()
    station_ids_tuple = tuple(station_ids) if station_ids is not None else None
    selection = load_station_config_selection(mapping_path, station_ids=station_ids_tuple)
    configs = selection.configs

    data_dir = root / "work/data/backtest"
    cache_dir = root / "work/cache/station_observations/iem_asos"
    report_dir = root / "work/reports/station_observations"
    data_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    db_path = data_dir / "daily_weather_station_observations.sqlite"
    decision_db = data_dir / "daily_weather_decision_artifact.sqlite"
    weather_db = data_dir / "daily_weather_weather_history.sqlite"
    city_coordinates = load_city_coordinates(weather_db)

    with sqlite3.connect(db_path) as conn:
        _ensure_schema(conn)

    proof_result: dict[str, Any] = {}
    full_result: dict[str, Any] = {}
    probe_failed_source_rows: list[dict[str, Any]] = []
    full_configs = configs
    probe_passed_source_count = len(configs)
    expanded_to_full = False
    if mode in {"proof", "proof_then_full"}:
        proof_result = ingest_station_observations(
            db_path=db_path,
            cache_dir=cache_dir,
            configs=configs,
            start=proof_start,
            end=proof_end,
            phase="proof",
            cache_only=cache_only,
            city_coordinates=city_coordinates,
        )
        proof_validation = validate_observation_coverage(
            db_path=db_path,
            configs=configs,
            start=proof_start,
            end=proof_end,
        )
        _apply_fetch_failures_to_validation(proof_validation, proof_result)
        passed_keys = {
            (row["station_id"], row["settlement_source_url"])
            for row in proof_validation.get("station_checks", [])
            if row.get("passed")
        }
        probe_failed_source_rows = [
            row for row in proof_validation.get("station_checks", []) if not row.get("passed")
        ]
        if mode == "proof_then_full":
            full_configs = [
                config
                for config in configs
                if (config.station_id, config.settlement_source_url) in passed_keys
            ]
            probe_passed_source_count = len(full_configs)
    else:
        proof_validation = {"passed": True, "skipped": True}

    if mode == "full" or (mode == "proof_then_full" and full_configs):
        expanded_to_full = bool(full_configs)
        full_result = ingest_station_observations(
            db_path=db_path,
            cache_dir=cache_dir,
            configs=full_configs,
            start=start,
            end=end,
            phase="six_month_v1",
            cache_only=cache_only,
            city_coordinates=city_coordinates,
        )
        full_validation = validate_observation_coverage(
            db_path=db_path,
            configs=full_configs,
            start=start,
            end=end,
        )
        _apply_fetch_failures_to_validation(full_validation, full_result)
        full_success_keys = {
            (row["station_id"], row["settlement_source_url"])
            for row in full_validation.get("station_checks", [])
            if row.get("passed")
        }
        full_configs = [
            config
            for config in full_configs
            if (config.station_id, config.settlement_source_url) in full_success_keys
        ]
        expanded_to_full = bool(full_configs)
        with sqlite3.connect(db_path) as conn:
            _ensure_schema(conn)
            daily_result = build_daily_highs(conn, configs=full_configs, start=start, end=end)
            intraday_result = build_intraday_snapshots(
                conn,
                decision_db=decision_db,
                configs=full_configs,
                start=start,
                end=end,
            )
            comparison = compare_station_highs_to_canonical_labels(
                conn,
                configs=full_configs,
                labels_csv=labels_path,
                start=start,
                end=end,
            )
            label_agreement_gate = build_label_agreement_gate(
                comparison,
                configs=full_configs,
                full_validation=full_validation,
            )
    else:
        daily_result = {}
        intraday_result = {}
        comparison = {}
        label_agreement_gate = {}
        full_validation = {}

    with sqlite3.connect(db_path) as conn:
        summary_counts = _summary_counts(conn, configs=full_configs, start=start, end=end)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    scope_slug = "expanded-station-observation-proof" if station_ids_tuple is None else _station_scope_slug(full_configs)
    prefix = report_dir / f"{timestamp}-{scope_slug}"
    artifacts = export_artifacts(
        db_path=db_path,
        report_prefix=prefix,
        configs=full_configs,
        start=start,
        end=end,
        expanded_to_full=expanded_to_full,
    )
    if expanded_to_full:
        artifacts.update(
            export_label_quality_artifacts(
                report_prefix=prefix,
                comparison=comparison,
                label_agreement_gate=label_agreement_gate,
            )
        )
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "station_ids": [config.station_id for config in configs],
        "full_window_station_ids": [config.station_id for config in full_configs],
        "target_window": {"start": start.isoformat(), "end": end.isoformat()},
        "proof_window": {"start": proof_start.isoformat(), "end": proof_end.isoformat()},
        "mode": mode,
        "expanded_to_full": expanded_to_full,
        "source_probe_coverage": {
            "candidate_source_rows": len(configs),
            "probe_passed_source_rows": probe_passed_source_count,
            "probe_failed_source_rows": len(probe_failed_source_rows),
            "skipped_source_rows": len(selection.skipped_source_rows),
            "full_window_source_rows": len(full_configs) if expanded_to_full else 0,
        },
        "skipped_source_rows": selection.skipped_source_rows,
        "probe_failed_source_rows": probe_failed_source_rows,
        "source_selection": _source_selection_summary(configs),
        "db_path": str(db_path),
        "raw_cache_dir": str(cache_dir),
        "inputs": {
            "mapping_json": str(mapping_path),
            "canonical_labels_csv": str(labels_path),
            "decision_db": str(decision_db),
            "weather_db": str(weather_db),
        },
        "artifacts": artifacts,
        "proof_result": proof_result,
        "proof_validation": proof_validation,
        "full_result": full_result,
        "full_validation": full_validation,
        "daily_high_result": daily_result,
        "intraday_snapshot_result": intraday_result,
        "canonical_label_comparison": comparison,
        "label_agreement_gate": label_agreement_gate,
        "summary_counts": summary_counts,
        "ready_for_phase_b_c_replay": _phase_ready(
            comparison,
            proof_validation,
            full_validation,
            expanded_to_full,
            label_agreement_gate,
        ),
        "known_limitations": _known_limitations(
            comparison,
            expanded_to_full,
            label_agreement_gate=label_agreement_gate,
            skipped_source_rows=selection.skipped_source_rows,
            probe_failed_source_rows=probe_failed_source_rows,
        ),
    }
    json_path = prefix.with_suffix(".json")
    md_path = prefix.with_suffix(".md")
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(_render_report(report), encoding="utf-8")
    report["artifacts"]["quality_json"] = str(json_path)
    report["artifacts"]["handoff_markdown"] = str(md_path)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def load_station_config_selection(
    mapping_json: Path,
    *,
    station_ids: Optional[Iterable[str]] = None,
) -> StationConfigSelection:
    wanted = set(station_ids) if station_ids is not None else None
    data = json.loads(mapping_json.read_text(encoding="utf-8"))
    configs: dict[tuple[str, str], StationConfig] = {}
    skipped: list[dict[str, Any]] = []
    for row in data["rows"]:
        station_id = row.get("station_id", "")
        if wanted is not None and station_id not in wanted:
            continue
        source_family = row.get("source_family", "")
        if source_family not in PROBE_SUPPORTED_SOURCE_FAMILIES:
            skipped.append(_skipped_source_row(row, "unsupported_source_family_for_iem_metar_probe"))
            continue
        if row.get("ingestion_status") not in READY_INGESTION_STATUSES:
            skipped.append(_skipped_source_row(row, "mapping_status_not_ready_for_station_probe"))
            continue
        if not _looks_like_icao_station_id(station_id):
            skipped.append(_skipped_source_row(row, "station_id_not_icao_metar_style"))
            continue
        key = (station_id, row.get("source_url", ""))
        configs[key] = StationConfig(
            station_id=station_id,
            normalized_city=row.get("normalized_city", ""),
            timezone_name=row.get("timezone", ""),
            settlement_source_family=source_family,
            settlement_source_url=row.get("source_url", ""),
            settlement_station_name=row.get("station_name", ""),
            settlement_unit=row.get("settlement_unit", ""),
            settlement_precision=row.get("settlement_precision", ""),
            mapping_ingestion_status=row.get("ingestion_status", ""),
        )
    if wanted is None:
        ordered = sorted(
            configs.values(),
            key=lambda item: (
                item.normalized_city,
                item.settlement_source_family,
                item.station_id,
                item.settlement_source_url,
            ),
        )
        return StationConfigSelection(configs=ordered, skipped_source_rows=skipped)
    ordered = sorted(configs.values(), key=lambda item: station_ids_tuple_index(station_ids or (), item.station_id))
    return StationConfigSelection(configs=ordered, skipped_source_rows=skipped)


def load_station_configs(mapping_json: Path, *, station_ids: Iterable[str]) -> list[StationConfig]:
    selection = load_station_config_selection(mapping_json, station_ids=station_ids)
    wanted = set(station_ids)
    missing = sorted(wanted - {config.station_id for config in selection.configs})
    if missing:
        raise ValueError(f"Missing probe-supported mapping rows for station ids: {missing}")
    return selection.configs


def load_city_coordinates(weather_db: Path) -> dict[str, tuple[float, float]]:
    if not weather_db.exists():
        return {}
    query = """
        SELECT normalized_city, AVG(latitude) AS latitude, AVG(longitude) AS longitude
        FROM realized_daily_weather
        GROUP BY normalized_city
    """
    with sqlite3.connect(f"file:{weather_db}?mode=ro", uri=True) as conn:
        rows = conn.execute(query).fetchall()
    output = {}
    for city, latitude, longitude in rows:
        if city and latitude is not None and longitude is not None:
            output[str(city)] = (float(latitude), float(longitude))
    return output


def station_ids_tuple_index(station_ids: Iterable[str], station_id: str) -> int:
    ordered = tuple(station_ids)
    return ordered.index(station_id) if station_id in ordered else len(ordered)


def _skipped_source_row(row: Mapping[str, Any], reason: str) -> dict[str, Any]:
    return {
        "normalized_city": row.get("normalized_city", ""),
        "source_family": row.get("source_family", ""),
        "station_id": row.get("station_id", ""),
        "source_url": row.get("source_url", ""),
        "timezone": row.get("timezone", ""),
        "settlement_unit": row.get("settlement_unit", ""),
        "settlement_precision": row.get("settlement_precision", ""),
        "ingestion_status": row.get("ingestion_status", ""),
        "skip_reason": reason,
    }


def _looks_like_icao_station_id(station_id: str) -> bool:
    return bool(re.fullmatch(r"[A-Z0-9]{4}", station_id or ""))


def ingest_station_observations(
    *,
    db_path: Path,
    cache_dir: Path,
    configs: list[StationConfig],
    start: date,
    end: date,
    phase: str,
    cache_only: bool,
    city_coordinates: Mapping[str, tuple[float, float]],
) -> dict[str, Any]:
    started = time.monotonic()
    fetched_at = datetime.now(timezone.utc).isoformat()
    station_results = []
    totals = Counter()
    with sqlite3.connect(db_path) as conn:
        _ensure_schema(conn)
        for config in configs:
            request_url = _iem_request_url(config, start=start, end=end)
            try:
                raw_payload, cache_path, payload_id, source_status = _get_or_fetch_raw_payload(
                    cache_dir=cache_dir,
                    config=config,
                    request_url=request_url,
                    start=start,
                    end=end,
                    phase=phase,
                    fetched_at=fetched_at,
                    cache_only=cache_only,
                )
            except Exception as exc:
                result = {
                    "station_id": config.station_id,
                    "settlement_source_url": config.settlement_source_url,
                    "source_family": config.settlement_source_family,
                    "raw_rows": 0,
                    "observations_seen": 0,
                    "observations_inserted_or_replaced": 0,
                    "parsed_temperature_rows": 0,
                    "missing_temperature_rows": 0,
                    "raw_cache_path": "",
                    "payload_id": "",
                    "quality_flags": f"fetch_failed:{type(exc).__name__}",
                    "error": str(exc),
                }
                station_results.append(result)
                totals["fetch_failed_source_rows"] += 1
                print(
                    f"station_observations phase={phase} station={config.station_id} source_family={config.settlement_source_family} fetch_failed={type(exc).__name__}"
                )
                continue
            rows, observations = parse_iem_csv_payload(
                raw_payload,
                config=config,
                request_url=request_url,
                fetched_at=fetched_at,
                payload_id=payload_id,
            )
            _insert_payload(
                conn,
                payload_id=payload_id,
                config=config,
                request_url=request_url,
                cache_path=cache_path,
                fetched_at=fetched_at,
                byte_count=len(raw_payload),
                row_count=rows,
                phase=phase,
                start=start,
                end=end,
                quality_flags=source_status,
            )
            inserted = _insert_observations(conn, observations)
            metadata = _station_metadata_from_observations(
                config=config,
                observations=observations,
                city_coordinates=city_coordinates,
                payload_id=payload_id,
                updated_at=fetched_at,
            )
            _upsert_station_metadata(conn, metadata)
            parsed_temperature_rows = sum(1 for obs in observations if obs.temperature_f is not None)
            missing_temperature_rows = len(observations) - parsed_temperature_rows
            result = {
                "station_id": config.station_id,
                "settlement_source_url": config.settlement_source_url,
                "source_family": config.settlement_source_family,
                "raw_rows": rows,
                "observations_seen": len(observations),
                "observations_inserted_or_replaced": inserted,
                "parsed_temperature_rows": parsed_temperature_rows,
                "missing_temperature_rows": missing_temperature_rows,
                "raw_cache_path": str(cache_path),
                "payload_id": payload_id,
                "quality_flags": source_status,
            }
            station_results.append(result)
            totals["raw_rows"] += rows
            totals["observations_seen"] += len(observations)
            totals["parsed_temperature_rows"] += parsed_temperature_rows
            totals["missing_temperature_rows"] += missing_temperature_rows
            print(
                f"station_observations phase={phase} station={config.station_id} source_family={config.settlement_source_family} raw_rows={rows} parsed={parsed_temperature_rows} missing={missing_temperature_rows}"
            )
    return {
        "phase": phase,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "stations": station_results,
        "totals": dict(totals),
        "elapsed_seconds": round(time.monotonic() - started, 2),
    }


def parse_iem_csv_payload(
    payload: bytes,
    *,
    config: StationConfig,
    request_url: str,
    fetched_at: str,
    payload_id: str,
) -> tuple[int, list[Observation]]:
    text = payload.decode("utf-8", errors="replace")
    reader = csv.DictReader(text.splitlines())
    observations: list[Observation] = []
    row_count = 0
    zone = ZoneInfo(config.timezone_name)
    for raw in reader:
        row_count += 1
        valid = (raw.get("valid") or "").strip()
        if not valid:
            continue
        observed_at = _parse_iem_valid_utc(valid)
        local_dt = observed_at.astimezone(zone)
        raw_station = (raw.get("station") or "").strip()
        metar = raw.get("metar") or ""
        temp = parse_temperature_from_iem_row(raw)
        station_latitude = _optional_float(raw.get("lat") or raw.get("latitude"))
        station_longitude = _optional_float(raw.get("lon") or raw.get("longitude"))
        station_elevation_m = _optional_float(raw.get("elevation") or raw.get("elev"))
        quality_flags = list(temp["quality_flags"])
        if raw_station and raw_station not in {config.station_id, config.station_id[1:]}:
            quality_flags.append("raw_station_id_differs_from_config")
        if raw_station == config.station_id[1:]:
            quality_flags.append("iem_uses_three_letter_station_alias")
        if station_latitude is not None and station_longitude is not None:
            quality_flags.append("iem_station_latlon_present")
        if station_elevation_m is not None:
            quality_flags.append("iem_station_elevation_present")
        observations.append(
            Observation(
                station_id=config.station_id,
                raw_station_id=raw_station,
                settlement_source_url=config.settlement_source_url,
                observation_source_url=request_url,
                source_family=config.historical_source_family,
                observed_at_utc=observed_at.isoformat(),
                local_date=local_dt.date().isoformat(),
                local_time=local_dt.time().isoformat(),
                timezone_name=config.timezone_name,
                raw_temperature=temp["raw_temperature"],
                source_unit=temp["source_unit"],
                temperature_f=temp["temperature_f"],
                temperature_c=temp["temperature_c"],
                station_latitude=station_latitude,
                station_longitude=station_longitude,
                station_elevation_m=station_elevation_m,
                fetch_timestamp_utc=fetched_at,
                source_payload_id=payload_id,
                quality_flags=tuple(_dedupe(quality_flags)),
                raw_metar=metar,
                raw_row_json=json.dumps(raw, sort_keys=True),
            )
        )
    return row_count, observations


def parse_temperature_from_iem_row(row: Mapping[str, str]) -> dict[str, Any]:
    metar = row.get("metar") or ""
    precise = _parse_precise_t_group_c(metar)
    if precise is not None:
        return _temperature_result(precise, "C", "parsed_metar_precise_t_group")
    tmpf = _optional_float(row.get("tmpf"))
    if tmpf is not None:
        return _temperature_result(tmpf, "F", "iem_tmpf_used")
    main = _parse_main_metar_temperature_c(metar)
    if main is not None:
        return _temperature_result(main, "C", "parsed_metar_integer_temp")
    return {
        "raw_temperature": None,
        "source_unit": "",
        "temperature_f": None,
        "temperature_c": None,
        "quality_flags": ("missing_temperature",),
    }


def build_daily_highs(
    conn: sqlite3.Connection,
    *,
    configs: list[StationConfig],
    start: date,
    end: date,
) -> dict[str, Any]:
    built_at = datetime.now(timezone.utc).isoformat()
    station_url_pairs = [(config.station_id, config.settlement_source_url) for config in configs]
    deleted = 0
    inserted = 0
    for station_id, source_url in station_url_pairs:
        deleted += conn.execute(
            """
            DELETE FROM station_daily_highs
            WHERE station_id = ? AND settlement_source_url = ? AND local_date >= ? AND local_date <= ?
            """,
            (station_id, source_url, start.isoformat(), end.isoformat()),
        ).rowcount
        rows = conn.execute(
            """
            SELECT
              station_id,
              settlement_source_url,
              source_family,
              local_date,
              timezone_name,
              observed_at_utc,
              temperature_f,
              temperature_c,
              quality_flags
            FROM station_observations
            WHERE station_id = ?
              AND settlement_source_url = ?
              AND local_date >= ?
              AND local_date <= ?
            ORDER BY local_date, observed_at_utc
            """,
            (station_id, source_url, start.isoformat(), end.isoformat()),
        ).fetchall()
        by_day: dict[str, list[sqlite3.Row]] = defaultdict(list)
        for row in rows:
            by_day[row[3]].append(row)
        for local_date, obs_rows in sorted(by_day.items()):
            valid = [row for row in obs_rows if row[6] is not None]
            flags = []
            if not valid:
                flags.append("no_temperature_observations")
                high_f = high_c = high_at = None
            else:
                high_row = max(valid, key=lambda row: (float(row[6]), row[5]))
                high_f = float(high_row[6])
                high_c = float(high_row[7])
                high_at = high_row[5]
                if len(valid) < 12:
                    flags.append("low_daily_sample_count")
            conn.execute(
                """
                INSERT OR REPLACE INTO station_daily_highs(
                  station_id, settlement_source_url, source_family, local_date, timezone_name,
                  high_temperature_f, high_temperature_c, high_observed_at_utc,
                  sample_count, missing_temperature_rows, first_observed_at_utc, last_observed_at_utc,
                  quality_flags, built_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    station_id,
                    source_url,
                    obs_rows[0][2],
                    local_date,
                    obs_rows[0][4],
                    high_f,
                    high_c,
                    high_at,
                    len(valid),
                    len(obs_rows) - len(valid),
                    obs_rows[0][5],
                    obs_rows[-1][5],
                    "|".join(flags) if flags else "ok",
                    built_at,
                ),
            )
            inserted += 1
    conn.commit()
    return {"deleted_rebuilt_rows": deleted, "daily_high_rows_written": inserted}


def build_intraday_snapshots(
    conn: sqlite3.Connection,
    *,
    decision_db: Path,
    configs: list[StationConfig],
    start: date,
    end: date,
) -> dict[str, Any]:
    built_at = datetime.now(timezone.utc).isoformat()
    config_by_url = {config.settlement_source_url: config for config in configs}
    deleted = 0
    inserted = 0
    for config in configs:
        deleted += conn.execute(
            """
            DELETE FROM station_intraday_high_snapshots
            WHERE station_id = ? AND settlement_source_url = ? AND local_date >= ? AND local_date <= ?
            """,
            (config.station_id, config.settlement_source_url, start.isoformat(), end.isoformat()),
        ).rowcount
    decision_rows = _load_decision_times(decision_db, config_by_url=config_by_url, start=start, end=end)
    observations = _load_observation_groups(conn, configs=configs, start=start, end=end)
    for row in decision_rows:
        key = (row["station_id"], row["settlement_source_url"], row["local_date"])
        decision_time = datetime.fromisoformat(row["decision_time_utc"])
        available = []
        for obs in observations.get(key, []):
            observed_at = datetime.fromisoformat(obs["observed_at_utc"])
            if observed_at <= decision_time and obs["temperature_f"] is not None:
                available.append(obs)
        flags = []
        if not available:
            flags.append("no_station_observations_as_of_decision")
            high_f = high_c = high_at = None
        else:
            high_obs = max(available, key=lambda item: (float(item["temperature_f"]), item["observed_at_utc"]))
            high_f = float(high_obs["temperature_f"])
            high_c = float(high_obs["temperature_c"])
            high_at = high_obs["observed_at_utc"]
        conn.execute(
            """
            INSERT OR REPLACE INTO station_intraday_high_snapshots(
              station_id, settlement_source_url, source_family, local_date, timezone_name,
              decision_time_utc, high_so_far_f, high_so_far_c, high_observed_at_utc,
              sample_count, quality_flags, built_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row["station_id"],
                row["settlement_source_url"],
                "IEM_ASOS_METAR",
                row["local_date"],
                row["timezone_name"],
                row["decision_time_utc"],
                high_f,
                high_c,
                high_at,
                len(available),
                "|".join(flags) if flags else "ok",
                built_at,
            ),
        )
        inserted += 1
    conn.commit()
    return {"deleted_rebuilt_rows": deleted, "intraday_snapshot_rows_written": inserted}


def compare_station_highs_to_canonical_labels(
    conn: sqlite3.Connection,
    *,
    configs: list[StationConfig],
    labels_csv: Path,
    start: date,
    end: date,
) -> dict[str, Any]:
    built_at = datetime.now(timezone.utc).isoformat()
    source_urls = {config.settlement_source_url for config in configs}
    config_by_url = {config.settlement_source_url: config for config in configs}
    config_by_station_source = {
        _station_source_key(config.station_id, config.settlement_source_url): config for config in configs
    }
    conn.execute(
        """
        DELETE FROM station_settlement_label_comparison
        WHERE local_date >= ? AND local_date <= ?
        """,
        (start.isoformat(), end.isoformat()),
    )
    daily = {}
    for row in conn.execute(
        """
        SELECT
          d.station_id,
          d.settlement_source_url,
          d.local_date,
          d.high_temperature_f,
          d.high_temperature_c,
          d.quality_flags,
          d.high_observed_at_utc,
          o.source_unit AS station_high_source_unit,
          o.quality_flags AS station_high_quality_flags
        FROM station_daily_highs d
        LEFT JOIN station_observations o
          ON o.station_id = d.station_id
         AND o.settlement_source_url = d.settlement_source_url
         AND o.observed_at_utc = d.high_observed_at_utc
        WHERE d.local_date >= ? AND d.local_date <= ?
        """,
        (start.isoformat(), end.isoformat()),
    ):
        daily[(row["station_id"], row["settlement_source_url"], row["local_date"])] = row
    totals = Counter()
    by_station: dict[str, Counter[str]] = defaultdict(Counter)
    by_station_source: dict[str, Counter[str]] = defaultdict(Counter)
    by_city: dict[str, Counter[str]] = defaultdict(Counter)
    by_station_high_source_unit: dict[str, Counter[str]] = defaultdict(Counter)
    by_settlement_unit: dict[str, Counter[str]] = defaultdict(Counter)
    by_bucket_type: dict[str, Counter[str]] = defaultdict(Counter)
    by_distance: dict[str, Counter[str]] = defaultdict(Counter)
    by_month: dict[str, Counter[str]] = defaultdict(Counter)
    by_mismatch_cause: dict[str, Counter[str]] = defaultdict(Counter)
    with labels_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for label_row in reader:
            row_date = _date_or_none(label_row.get("market_date_local", ""))
            if row_date is None or row_date < start or row_date > end:
                continue
            source_url = label_row.get("source_url", "")
            if source_url not in source_urls:
                continue
            config = config_by_url[source_url]
            station_source_key = _station_source_key(config.station_id, source_url)
            action = label_row.get("recommended_action", "")
            if action == "exclude_or_manual_review":
                totals["manual_review_rows_filtered"] += 1
                by_station[config.station_id]["manual_review_rows_filtered"] += 1
                by_station_source[station_source_key]["manual_review_rows_filtered"] += 1
                continue
            totals["rows_considered"] += 1
            by_station[config.station_id]["rows_considered"] += 1
            by_station_source[station_source_key]["rows_considered"] += 1
            daily_row = daily.get((config.station_id, source_url, label_row["market_date_local"]))
            if daily_row is None or daily_row["high_temperature_f"] is None:
                totals["missing_station_high_rows"] += 1
                by_station[config.station_id]["missing_station_high_rows"] += 1
                by_station_source[station_source_key]["missing_station_high_rows"] += 1
                continue
            high_f = float(daily_row["high_temperature_f"])
            station_high_source_unit = str(daily_row["station_high_source_unit"] or "unknown")
            station_high_quality_flags = str(daily_row["station_high_quality_flags"] or "")
            station_label = _station_label_for_row(label_row, high_f)
            canonical_label = _int_or_none(label_row.get("canonical_label"))
            mismatch = int(station_label is not None and canonical_label is not None and station_label != canonical_label)
            distance = _threshold_distance_f(label_row, high_f)
            distance_bucket = _distance_bucket(distance)
            mismatch_cause = _label_mismatch_cause(
                label_row,
                high_f=high_f,
                station_label=station_label,
                canonical_label=canonical_label,
                station_high_source_unit=station_high_source_unit,
            )
            totals["rows_compared"] += 1
            totals["mismatch_rows"] += mismatch
            _record_label_comparison(by_station[config.station_id], mismatch=mismatch, distance=distance, cause=mismatch_cause)
            _record_label_comparison(
                by_station_source[station_source_key],
                mismatch=mismatch,
                distance=distance,
                cause=mismatch_cause,
            )
            _record_label_comparison(by_city[config.normalized_city], mismatch=mismatch, distance=distance, cause=mismatch_cause)
            _record_label_comparison(
                by_station_high_source_unit[station_high_source_unit],
                mismatch=mismatch,
                distance=distance,
                cause=mismatch_cause,
            )
            _record_label_comparison(
                by_settlement_unit[label_row.get("settlement_unit", "") or "unknown"],
                mismatch=mismatch,
                distance=distance,
                cause=mismatch_cause,
            )
            _record_label_comparison(
                by_bucket_type[label_row.get("canonical_bucket_type", "") or "unknown"],
                mismatch=mismatch,
                distance=distance,
                cause=mismatch_cause,
            )
            _record_label_comparison(by_distance[distance_bucket], mismatch=mismatch, distance=distance, cause=mismatch_cause)
            _record_label_comparison(by_month[label_row["market_date_local"][:7]], mismatch=mismatch, distance=distance, cause=mismatch_cause)
            if mismatch_cause:
                _record_label_comparison(
                    by_mismatch_cause[mismatch_cause],
                    mismatch=mismatch,
                    distance=distance,
                    cause=mismatch_cause,
                )
            if distance is not None and distance <= 2.0:
                totals["near_2f_rows_compared"] += 1
                totals["near_2f_mismatch_rows"] += mismatch
            if distance is not None and distance <= 1.0:
                totals["near_1f_rows_compared"] += 1
                totals["near_1f_mismatch_rows"] += mismatch
            if distance is not None and distance <= 0.5:
                totals["near_0_5f_rows_compared"] += 1
                totals["near_0_5f_mismatch_rows"] += mismatch
            conn.execute(
                """
                INSERT OR REPLACE INTO station_settlement_label_comparison(
                  universe_row_id, market_id, station_id, settlement_source_url, local_date,
                  normalized_city, settlement_source_family,
                  canonical_bucket_type, operator, threshold_low_f, threshold_high_f,
                  station_high_f, station_high_c, station_high_source_unit,
                  station_high_quality_flags, settlement_value, settlement_unit,
                  settlement_precision, station_label, canonical_label, mismatch,
                  threshold_distance_f, threshold_distance_bucket_f, recommended_action,
                  label_mismatch_cause,
                  built_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    label_row.get("universe_row_id"),
                    label_row.get("market_id"),
                    config.station_id,
                    source_url,
                    label_row.get("market_date_local"),
                    config.normalized_city,
                    config.settlement_source_family,
                    label_row.get("canonical_bucket_type"),
                    label_row.get("operator"),
                    _optional_float(label_row.get("threshold_low_f")),
                    _optional_float(label_row.get("threshold_high_f")),
                    high_f,
                    _f_to_c(high_f),
                    station_high_source_unit,
                    station_high_quality_flags,
                    _settlement_value(label_row, high_f),
                    label_row.get("settlement_unit"),
                    label_row.get("settlement_precision"),
                    station_label,
                    canonical_label,
                    mismatch,
                    distance,
                    distance_bucket,
                    action,
                    mismatch_cause,
                    built_at,
                ),
            )
    conn.commit()
    summary = dict(totals)
    summary["mismatch_rate"] = _rate(totals["mismatch_rows"], totals["rows_compared"])
    summary["agreement_rate"] = _agreement_rate(totals["mismatch_rows"], totals["rows_compared"])
    summary["near_2f_mismatch_rate"] = _rate(totals["near_2f_mismatch_rows"], totals["near_2f_rows_compared"])
    summary["near_1f_mismatch_rate"] = _rate(totals["near_1f_mismatch_rows"], totals["near_1f_rows_compared"])
    summary["near_0_5f_mismatch_rate"] = _rate(totals["near_0_5f_mismatch_rows"], totals["near_0_5f_rows_compared"])
    summary["near_0_5f_agreement_rate"] = _agreement_rate(
        totals["near_0_5f_mismatch_rows"],
        totals["near_0_5f_rows_compared"],
    )
    return {
        "summary": summary,
        "by_station": _counter_rates(by_station),
        "by_station_source": _station_source_counter_rates(by_station_source, config_by_station_source),
        "by_city": _counter_rates(by_city),
        "by_station_high_source_unit": _counter_rates(by_station_high_source_unit),
        "by_settlement_unit": _counter_rates(by_settlement_unit),
        "by_bucket_type": _counter_rates(by_bucket_type),
        "by_threshold_distance_bucket_f": _counter_rates(by_distance),
        "by_month": _counter_rates(by_month),
        "by_mismatch_cause": _counter_rates(by_mismatch_cause),
    }


def build_label_agreement_gate(
    comparison: Mapping[str, Any],
    *,
    configs: list[StationConfig],
    full_validation: Mapping[str, Any],
) -> dict[str, Any]:
    summary = (comparison or {}).get("summary") or {}
    by_station_source = (comparison or {}).get("by_station_source") or {}
    coverage_by_key = {
        _station_source_key(row.get("station_id", ""), row.get("settlement_source_url", "")): row
        for row in full_validation.get("station_checks", [])
    }
    whitelist = []
    quarantine = []
    station_source_rows = []
    for config in configs:
        key = _station_source_key(config.station_id, config.settlement_source_url)
        stats = by_station_source.get(key, {})
        coverage = coverage_by_key.get(key, {})
        rows_compared = int(stats.get("rows_compared", 0) or 0)
        mismatch_rows = int(stats.get("mismatch_rows", 0) or 0)
        near_rows = int(stats.get("near_0_5f_rows_compared", 0) or 0)
        near_mismatch_rows = int(stats.get("near_0_5f_mismatch_rows", 0) or 0)
        agreement = _agreement_rate(mismatch_rows, rows_compared)
        near_agreement = _agreement_rate(near_mismatch_rows, near_rows)
        failure_reasons = []
        if not coverage.get("passed", False):
            failure_reasons.append("parse_coverage_failed")
        if int(stats.get("missing_station_high_rows", 0) or 0):
            failure_reasons.append("missing_station_high_rows")
        if rows_compared <= 0:
            failure_reasons.append("no_label_rows_compared")
        elif agreement is None or agreement < LABEL_AGREEMENT_MIN:
            failure_reasons.append("overall_label_agreement_below_95pct")
        if near_rows <= 0:
            failure_reasons.append("no_near_0_5f_label_rows")
        elif near_agreement is None or near_agreement < NEAR_THRESHOLD_LABEL_AGREEMENT_MIN:
            failure_reasons.append("near_0_5f_label_agreement_below_70pct")
        row = {
            "station_id": config.station_id,
            "normalized_city": config.normalized_city,
            "settlement_source_family": config.settlement_source_family,
            "settlement_source_url": config.settlement_source_url,
            "settlement_station_name": config.settlement_station_name,
            "settlement_unit": config.settlement_unit,
            "settlement_precision": config.settlement_precision,
            "parse_coverage_passed": bool(coverage.get("passed", False)),
            "rows_compared": rows_compared,
            "mismatch_rows": mismatch_rows,
            "agreement_rate": agreement,
            "near_0_5f_rows_compared": near_rows,
            "near_0_5f_mismatch_rows": near_mismatch_rows,
            "near_0_5f_agreement_rate": near_agreement,
            "manual_review_rows_filtered": int(stats.get("manual_review_rows_filtered", 0) or 0),
            "dominant_mismatch_cause": _dominant_mismatch_cause(stats),
            "quality_gate_passed": not failure_reasons,
            "failure_reasons": ";".join(failure_reasons),
            "primary_diagnosis": _primary_station_source_diagnosis(stats, failure_reasons),
        }
        station_source_rows.append(row)
        if row["quality_gate_passed"]:
            whitelist.append(row)
        else:
            quarantine.append(row)
    overall_agreement = summary.get("agreement_rate")
    near_0_5f_agreement = summary.get("near_0_5f_agreement_rate")
    overall_reasons = []
    if int(summary.get("missing_station_high_rows", 0) or 0):
        overall_reasons.append("missing_station_high_rows")
    if overall_agreement is None or overall_agreement < LABEL_AGREEMENT_MIN:
        overall_reasons.append("overall_label_agreement_below_95pct")
    if near_0_5f_agreement is None or near_0_5f_agreement < NEAR_THRESHOLD_LABEL_AGREEMENT_MIN:
        overall_reasons.append("near_0_5f_label_agreement_below_70pct")
    if quarantine:
        overall_reasons.append("one_or_more_station_source_rows_quarantined")
    return {
        "passed": not overall_reasons,
        "overall_agreement_min": LABEL_AGREEMENT_MIN,
        "near_threshold_distance_f": NEAR_THRESHOLD_DISTANCE_F,
        "near_0_5f_agreement_min": NEAR_THRESHOLD_LABEL_AGREEMENT_MIN,
        "overall_agreement_rate": overall_agreement,
        "near_0_5f_agreement_rate": near_0_5f_agreement,
        "overall_failure_reasons": overall_reasons,
        "station_source_rows": station_source_rows,
        "whitelist_station_source_rows": whitelist,
        "quarantine_station_source_rows": quarantine,
        "whitelist_station_ids": sorted({row["station_id"] for row in whitelist}),
        "quarantine_station_ids": sorted({row["station_id"] for row in quarantine}),
    }


def export_label_quality_artifacts(
    *,
    report_prefix: Path,
    comparison: Mapping[str, Any],
    label_agreement_gate: Mapping[str, Any],
) -> dict[str, str]:
    whitelist_csv = report_prefix.with_name(report_prefix.name + "-phase-b-c-station-whitelist.csv")
    quarantine_csv = report_prefix.with_name(report_prefix.name + "-station-quarantine.csv")
    diagnostics_csv = report_prefix.with_name(report_prefix.name + "-label-agreement-diagnostics.csv")
    _write_dict_csv(whitelist_csv, label_agreement_gate.get("whitelist_station_source_rows", []))
    _write_dict_csv(quarantine_csv, label_agreement_gate.get("quarantine_station_source_rows", []))
    _write_dict_csv(diagnostics_csv, _label_agreement_diagnostic_rows(comparison))
    return {
        "phase_b_c_station_whitelist_csv": str(whitelist_csv),
        "station_quarantine_csv": str(quarantine_csv),
        "label_agreement_diagnostics_csv": str(diagnostics_csv),
    }


def validate_observation_coverage(
    *,
    db_path: Path,
    configs: list[StationConfig],
    start: date,
    end: date,
) -> dict[str, Any]:
    expected_days = (end - start).days + 1
    station_checks = []
    passed = True
    with sqlite3.connect(db_path) as conn:
        for config in configs:
            row = conn.execute(
                """
                SELECT
                  COUNT(*) AS rows,
                  SUM(CASE WHEN temperature_f IS NOT NULL THEN 1 ELSE 0 END) AS parsed_rows,
                  COUNT(DISTINCT local_date) AS local_dates
                FROM station_observations
                WHERE station_id = ?
                  AND settlement_source_url = ?
                  AND local_date >= ?
                  AND local_date <= ?
                """,
                (config.station_id, config.settlement_source_url, start.isoformat(), end.isoformat()),
            ).fetchone()
            rows = int(row[0] or 0)
            parsed = int(row[1] or 0)
            local_dates = int(row[2] or 0)
            missing_fraction = 1.0 - (parsed / rows) if rows else 1.0
            check = {
                "station_id": config.station_id,
                "normalized_city": config.normalized_city,
                "source_family": config.settlement_source_family,
                "settlement_source_url": config.settlement_source_url,
                "rows": rows,
                "parsed_temperature_rows": parsed,
                "missing_temperature_fraction": round(missing_fraction, 6),
                "local_dates": local_dates,
                "expected_local_dates": expected_days,
                "passed": rows > 0 and parsed > 0 and local_dates == expected_days and missing_fraction <= 0.05,
            }
            station_checks.append(check)
            passed = passed and bool(check["passed"])
    return {"passed": passed, "station_checks": station_checks}


def _apply_fetch_failures_to_validation(validation: dict[str, Any], ingest_result: Mapping[str, Any]) -> None:
    failed = {
        (row.get("station_id", ""), row.get("settlement_source_url", "")): row
        for row in ingest_result.get("stations", [])
        if str(row.get("quality_flags", "")).startswith("fetch_failed:")
    }
    if not failed:
        return
    for check in validation.get("station_checks", []):
        key = (check.get("station_id", ""), check.get("settlement_source_url", ""))
        if key in failed:
            check["passed"] = False
            check["failure_reason"] = failed[key].get("quality_flags", "fetch_failed")
            check["fetch_error"] = failed[key].get("error", "")
    validation["passed"] = all(bool(row.get("passed")) for row in validation.get("station_checks", []))


def export_artifacts(
    *,
    db_path: Path,
    report_prefix: Path,
    configs: list[StationConfig],
    start: date,
    end: date,
    expanded_to_full: bool,
) -> dict[str, str]:
    artifacts = {
        "station_observation_db": str(db_path),
    }
    if not expanded_to_full:
        return artifacts
    station_ids = [config.station_id for config in configs]
    if not station_ids:
        return artifacts
    observations_csv = report_prefix.with_name(report_prefix.name + "-normalized-observations.csv")
    metadata_csv = report_prefix.with_name(report_prefix.name + "-station-metadata.csv")
    daily_csv = report_prefix.with_name(report_prefix.name + "-daily-highs.csv")
    intraday_csv = report_prefix.with_name(report_prefix.name + "-intraday-high-so-far.csv")
    comparison_csv = report_prefix.with_name(report_prefix.name + "-canonical-label-comparison.csv")
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        _export_query(
            conn,
            observations_csv,
            """
            SELECT station_id, raw_station_id, settlement_source_url AS source_url, observation_source_url,
                   source_family, observed_at_utc, local_date, local_time, timezone_name,
                   raw_temperature, source_unit, temperature_f, temperature_c,
                   station_latitude, station_longitude, station_elevation_m,
                   fetch_timestamp_utc, source_payload_id, quality_flags, raw_metar, raw_row_json
            FROM station_observations
            WHERE station_id IN ({placeholders}) AND local_date >= ? AND local_date <= ?
            ORDER BY station_id, observed_at_utc
            """.format(placeholders=",".join("?" for _ in station_ids)),
            (*station_ids, start.isoformat(), end.isoformat()),
        )
        _export_query(
            conn,
            metadata_csv,
            """
            SELECT station_id, settlement_source_url AS source_url, settlement_source_family,
                   historical_source_family, normalized_city, settlement_station_name,
                   timezone_name, settlement_unit, settlement_precision, station_latitude,
                   station_longitude, station_elevation_m, city_latitude, city_longitude,
                   city_to_station_distance_km, coastal_inland_class, urban_airport_notes,
                   metadata_source, source_payload_id, raw_metadata_json, updated_at_utc,
                   quality_flags
            FROM station_metadata
            WHERE station_id IN ({placeholders})
            ORDER BY normalized_city, settlement_source_family, station_id, settlement_source_url
            """.format(placeholders=",".join("?" for _ in station_ids)),
            tuple(station_ids),
        )
        _export_query(
            conn,
            daily_csv,
            """
            SELECT * FROM station_daily_highs
            WHERE station_id IN ({placeholders}) AND local_date >= ? AND local_date <= ?
            ORDER BY station_id, local_date
            """.format(placeholders=",".join("?" for _ in station_ids)),
            (*station_ids, start.isoformat(), end.isoformat()),
        )
        _export_query(
            conn,
            intraday_csv,
            """
            SELECT * FROM station_intraday_high_snapshots
            WHERE station_id IN ({placeholders}) AND local_date >= ? AND local_date <= ?
            ORDER BY station_id, local_date, decision_time_utc
            """.format(placeholders=",".join("?" for _ in station_ids)),
            (*station_ids, start.isoformat(), end.isoformat()),
        )
        _export_query(
            conn,
            comparison_csv,
            """
            SELECT * FROM station_settlement_label_comparison
            WHERE station_id IN ({placeholders}) AND local_date >= ? AND local_date <= ?
            ORDER BY station_id, local_date, universe_row_id
            """.format(placeholders=",".join("?" for _ in station_ids)),
            (*station_ids, start.isoformat(), end.isoformat()),
        )
    artifacts.update(
        {
            "normalized_observations_csv": str(observations_csv),
            "station_metadata_csv": str(metadata_csv),
            "daily_station_highs_csv": str(daily_csv),
            "intraday_high_so_far_csv": str(intraday_csv),
            "canonical_label_comparison_csv": str(comparison_csv),
        }
    )
    return artifacts


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS station_source_payloads (
            payload_id TEXT PRIMARY KEY,
            station_id TEXT NOT NULL,
            settlement_source_url TEXT NOT NULL,
            source_family TEXT NOT NULL,
            request_url TEXT NOT NULL,
            cache_path TEXT NOT NULL,
            fetched_at_utc TEXT NOT NULL,
            payload_sha256 TEXT NOT NULL,
            byte_count INTEGER NOT NULL,
            row_count INTEGER NOT NULL,
            phase TEXT NOT NULL,
            start_date TEXT NOT NULL,
            end_date TEXT NOT NULL,
            quality_flags TEXT
        );
        CREATE TABLE IF NOT EXISTS station_observations (
            station_id TEXT NOT NULL,
            raw_station_id TEXT,
            settlement_source_url TEXT NOT NULL,
            observation_source_url TEXT NOT NULL,
            source_family TEXT NOT NULL,
            observed_at_utc TEXT NOT NULL,
            local_date TEXT NOT NULL,
            local_time TEXT NOT NULL,
            timezone_name TEXT NOT NULL,
            raw_temperature REAL,
            source_unit TEXT,
            temperature_f REAL,
            temperature_c REAL,
            station_latitude REAL,
            station_longitude REAL,
            station_elevation_m REAL,
            fetch_timestamp_utc TEXT NOT NULL,
            source_payload_id TEXT NOT NULL,
            quality_flags TEXT,
            raw_metar TEXT,
            raw_row_json TEXT,
            PRIMARY KEY (station_id, settlement_source_url, source_family, observed_at_utc)
        );
        CREATE INDEX IF NOT EXISTS idx_station_obs_station_date
          ON station_observations(station_id, settlement_source_url, local_date, observed_at_utc);
        CREATE TABLE IF NOT EXISTS station_metadata (
            station_id TEXT NOT NULL,
            settlement_source_url TEXT NOT NULL,
            settlement_source_family TEXT NOT NULL,
            historical_source_family TEXT NOT NULL,
            normalized_city TEXT NOT NULL,
            settlement_station_name TEXT,
            timezone_name TEXT NOT NULL,
            settlement_unit TEXT,
            settlement_precision TEXT,
            station_latitude REAL,
            station_longitude REAL,
            station_elevation_m REAL,
            city_latitude REAL,
            city_longitude REAL,
            city_to_station_distance_km REAL,
            coastal_inland_class TEXT,
            urban_airport_notes TEXT,
            metadata_source TEXT,
            source_payload_id TEXT,
            raw_metadata_json TEXT,
            updated_at_utc TEXT NOT NULL,
            quality_flags TEXT,
            PRIMARY KEY (station_id, settlement_source_url)
        );
        CREATE TABLE IF NOT EXISTS station_daily_highs (
            station_id TEXT NOT NULL,
            settlement_source_url TEXT NOT NULL,
            source_family TEXT NOT NULL,
            local_date TEXT NOT NULL,
            timezone_name TEXT NOT NULL,
            high_temperature_f REAL,
            high_temperature_c REAL,
            high_observed_at_utc TEXT,
            sample_count INTEGER NOT NULL,
            missing_temperature_rows INTEGER NOT NULL,
            first_observed_at_utc TEXT,
            last_observed_at_utc TEXT,
            quality_flags TEXT,
            built_at_utc TEXT NOT NULL,
            PRIMARY KEY (station_id, settlement_source_url, local_date)
        );
        CREATE TABLE IF NOT EXISTS station_intraday_high_snapshots (
            station_id TEXT NOT NULL,
            settlement_source_url TEXT NOT NULL,
            source_family TEXT NOT NULL,
            local_date TEXT NOT NULL,
            timezone_name TEXT NOT NULL,
            decision_time_utc TEXT NOT NULL,
            high_so_far_f REAL,
            high_so_far_c REAL,
            high_observed_at_utc TEXT,
            sample_count INTEGER NOT NULL,
            quality_flags TEXT,
            built_at_utc TEXT NOT NULL,
            PRIMARY KEY (station_id, settlement_source_url, local_date, decision_time_utc)
        );
        CREATE TABLE IF NOT EXISTS station_settlement_label_comparison (
            universe_row_id TEXT PRIMARY KEY,
            market_id TEXT,
            station_id TEXT NOT NULL,
            settlement_source_url TEXT NOT NULL,
            local_date TEXT NOT NULL,
            normalized_city TEXT,
            settlement_source_family TEXT,
            canonical_bucket_type TEXT,
            operator TEXT,
            threshold_low_f REAL,
            threshold_high_f REAL,
            station_high_f REAL,
            station_high_c REAL,
            station_high_source_unit TEXT,
            station_high_quality_flags TEXT,
            settlement_value REAL,
            settlement_unit TEXT,
            settlement_precision TEXT,
            station_label INTEGER,
            canonical_label INTEGER,
            mismatch INTEGER,
            threshold_distance_f REAL,
            threshold_distance_bucket_f TEXT,
            recommended_action TEXT,
            label_mismatch_cause TEXT,
            built_at_utc TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_station_comparison_station_date
          ON station_settlement_label_comparison(station_id, local_date);
        """
    )
    _ensure_columns(
        conn,
        "station_observations",
        {
            "station_latitude": "REAL",
            "station_longitude": "REAL",
            "station_elevation_m": "REAL",
        },
    )
    _ensure_columns(
        conn,
        "station_settlement_label_comparison",
        {
            "normalized_city": "TEXT",
            "settlement_source_family": "TEXT",
            "station_high_source_unit": "TEXT",
            "station_high_quality_flags": "TEXT",
            "label_mismatch_cause": "TEXT",
        },
    )
    conn.row_factory = sqlite3.Row


def _get_or_fetch_raw_payload(
    *,
    cache_dir: Path,
    config: StationConfig,
    request_url: str,
    start: date,
    end: date,
    phase: str,
    fetched_at: str,
    cache_only: bool,
) -> tuple[bytes, Path, str, str]:
    station_dir = cache_dir / config.station_id
    station_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(station_dir.glob(f"{config.station_id}-{phase}-{start.isoformat()}-{end.isoformat()}-*.csv"))
    if cache_only and existing:
        cache_path = existing[-1]
        payload = cache_path.read_bytes()
        return payload, cache_path, _payload_id(request_url, payload, config.settlement_source_url), "cache_hit"
    if cache_only:
        raise FileNotFoundError(f"No cached station payload for {config.station_id} {phase} {start} {end}")
    request = urllib.request.Request(request_url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = response.read()
    digest = hashlib.sha256(payload).hexdigest()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    cache_path = station_dir / f"{config.station_id}-{phase}-{start.isoformat()}-{end.isoformat()}-{timestamp}-{digest[:12]}.csv"
    cache_path.write_bytes(payload)
    sidecar = cache_path.with_suffix(".request.json")
    sidecar.write_text(
        json.dumps(
            {
                "station_id": config.station_id,
                "request_url": request_url,
                "fetched_at_utc": fetched_at,
                "source_family": config.historical_source_family,
                "settlement_source_url": config.settlement_source_url,
                "payload_sha256": digest,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return payload, cache_path, _payload_id(request_url, payload, config.settlement_source_url), "downloaded"


def _iem_request_url(config: StationConfig, *, start: date, end: date) -> str:
    zone = ZoneInfo(config.timezone_name)
    start_utc = datetime.combine(start, datetime_time.min, tzinfo=zone).astimezone(timezone.utc)
    end_exclusive_utc = datetime.combine(end + timedelta(days=1), datetime_time.min, tzinfo=zone).astimezone(timezone.utc)
    request_start = start_utc.date()
    request_end_exclusive = end_exclusive_utc.date() + timedelta(days=1)
    params = [
        ("station", config.station_id),
        ("data", "tmpf"),
        ("data", "metar"),
        ("year1", str(request_start.year)),
        ("month1", str(request_start.month)),
        ("day1", str(request_start.day)),
        ("year2", str(request_end_exclusive.year)),
        ("month2", str(request_end_exclusive.month)),
        ("day2", str(request_end_exclusive.day)),
        ("tz", "Etc/UTC"),
        ("format", "onlycomma"),
        ("latlon", "yes"),
        ("elev", "yes"),
        ("missing", "empty"),
        ("trace", "T"),
        ("direct", "no"),
    ]
    return IEM_ASOS_URL + "?" + urllib.parse.urlencode(params)


def _insert_payload(
    conn: sqlite3.Connection,
    *,
    payload_id: str,
    config: StationConfig,
    request_url: str,
    cache_path: Path,
    fetched_at: str,
    byte_count: int,
    row_count: int,
    phase: str,
    start: date,
    end: date,
    quality_flags: str,
) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO station_source_payloads(
          payload_id, station_id, settlement_source_url, source_family, request_url,
          cache_path, fetched_at_utc, payload_sha256, byte_count, row_count,
          phase, start_date, end_date, quality_flags
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            payload_id,
            config.station_id,
            config.settlement_source_url,
            config.historical_source_family,
            request_url,
            str(cache_path),
            fetched_at,
            payload_id.split(":", 1)[-1],
            byte_count,
            row_count,
            phase,
            start.isoformat(),
            end.isoformat(),
            quality_flags,
        ),
    )


def _insert_observations(conn: sqlite3.Connection, observations: list[Observation]) -> int:
    conn.executemany(
        """
        INSERT OR REPLACE INTO station_observations(
          station_id, raw_station_id, settlement_source_url, observation_source_url, source_family,
          observed_at_utc, local_date, local_time, timezone_name, raw_temperature, source_unit,
          temperature_f, temperature_c, station_latitude, station_longitude, station_elevation_m,
          fetch_timestamp_utc, source_payload_id, quality_flags,
          raw_metar, raw_row_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                obs.station_id,
                obs.raw_station_id,
                obs.settlement_source_url,
                obs.observation_source_url,
                obs.source_family,
                obs.observed_at_utc,
                obs.local_date,
                obs.local_time,
                obs.timezone_name,
                obs.raw_temperature,
                obs.source_unit,
                obs.temperature_f,
                obs.temperature_c,
                obs.station_latitude,
                obs.station_longitude,
                obs.station_elevation_m,
                obs.fetch_timestamp_utc,
                obs.source_payload_id,
                "|".join(obs.quality_flags) if obs.quality_flags else "ok",
                obs.raw_metar,
                obs.raw_row_json,
            )
            for obs in observations
        ],
    )
    conn.commit()
    return len(observations)


def _upsert_station_metadata(conn: sqlite3.Connection, metadata: StationMetadata) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO station_metadata(
          station_id, settlement_source_url, settlement_source_family, historical_source_family,
          normalized_city, settlement_station_name, timezone_name, settlement_unit,
          settlement_precision, station_latitude, station_longitude, station_elevation_m,
          city_latitude, city_longitude, city_to_station_distance_km, coastal_inland_class,
          urban_airport_notes, metadata_source, source_payload_id, raw_metadata_json,
          updated_at_utc, quality_flags
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            metadata.station_id,
            metadata.settlement_source_url,
            metadata.settlement_source_family,
            metadata.historical_source_family,
            metadata.normalized_city,
            metadata.settlement_station_name,
            metadata.timezone_name,
            metadata.settlement_unit,
            metadata.settlement_precision,
            metadata.station_latitude,
            metadata.station_longitude,
            metadata.station_elevation_m,
            metadata.city_latitude,
            metadata.city_longitude,
            metadata.city_to_station_distance_km,
            metadata.coastal_inland_class,
            metadata.urban_airport_notes,
            metadata.metadata_source,
            metadata.source_payload_id,
            metadata.raw_metadata_json,
            metadata.updated_at_utc,
            "|".join(metadata.quality_flags) if metadata.quality_flags else "ok",
        ),
    )
    conn.commit()


def _station_metadata_from_observations(
    *,
    config: StationConfig,
    observations: list[Observation],
    city_coordinates: Mapping[str, tuple[float, float]],
    payload_id: str,
    updated_at: str,
) -> StationMetadata:
    metadata_observation = next(
        (
            obs
            for obs in observations
            if obs.station_latitude is not None
            or obs.station_longitude is not None
            or obs.station_elevation_m is not None
        ),
        None,
    )
    station_latitude = metadata_observation.station_latitude if metadata_observation else None
    station_longitude = metadata_observation.station_longitude if metadata_observation else None
    station_elevation_m = metadata_observation.station_elevation_m if metadata_observation else None
    city_latitude = city_longitude = None
    city_coord = city_coordinates.get(config.normalized_city)
    if city_coord is not None:
        city_latitude, city_longitude = city_coord
    distance = _haversine_km(city_latitude, city_longitude, station_latitude, station_longitude)
    coastal_class, urban_notes = _station_context(
        city=config.normalized_city,
        station_id=config.station_id,
        station_name=config.settlement_station_name,
        city_to_station_distance_km=distance,
    )
    flags = []
    if metadata_observation is None:
        flags.append("no_iem_station_metadata_rows")
    else:
        flags.append("metadata_from_iem_asos_payload")
    if station_latitude is None or station_longitude is None:
        flags.append("station_latlon_missing")
    if station_elevation_m is None:
        flags.append("station_elevation_missing")
    if city_latitude is None or city_longitude is None:
        flags.append("city_coordinate_proxy_missing")
    else:
        flags.append("city_coordinate_proxy_from_open_meteo_realized_weather")
    if config.normalized_city in COASTAL_INLAND_CLASS_BY_CITY:
        flags.append("coastal_inland_class_static_proxy")
    else:
        flags.append("coastal_inland_class_unknown")
    raw_metadata = {
        "station_id": config.station_id,
        "normalized_city": config.normalized_city,
        "station_latitude": station_latitude,
        "station_longitude": station_longitude,
        "station_elevation_m": station_elevation_m,
        "city_latitude": city_latitude,
        "city_longitude": city_longitude,
        "city_to_station_distance_km": distance,
        "source_payload_id": payload_id,
        "source_family": config.historical_source_family,
        "settlement_source_family": config.settlement_source_family,
        "settlement_source_url": config.settlement_source_url,
    }
    return StationMetadata(
        station_id=config.station_id,
        settlement_source_url=config.settlement_source_url,
        settlement_source_family=config.settlement_source_family,
        historical_source_family=config.historical_source_family,
        normalized_city=config.normalized_city,
        settlement_station_name=config.settlement_station_name,
        timezone_name=config.timezone_name,
        settlement_unit=config.settlement_unit,
        settlement_precision=config.settlement_precision,
        station_latitude=station_latitude,
        station_longitude=station_longitude,
        station_elevation_m=station_elevation_m,
        city_latitude=city_latitude,
        city_longitude=city_longitude,
        city_to_station_distance_km=distance,
        coastal_inland_class=coastal_class,
        urban_airport_notes=urban_notes,
        metadata_source="IEM_ASOS_METAR_PAYLOAD" if metadata_observation is not None else "mapping_only",
        source_payload_id=payload_id,
        raw_metadata_json=json.dumps(raw_metadata, sort_keys=True),
        updated_at_utc=updated_at,
        quality_flags=tuple(_dedupe(flags)),
    )


def _station_context(
    *,
    city: str,
    station_id: str,
    station_name: str,
    city_to_station_distance_km: Optional[float],
) -> tuple[str, str]:
    coastal_class = COASTAL_INLAND_CLASS_BY_CITY.get(city, "unknown")
    notes = []
    if _looks_like_icao_station_id(station_id):
        notes.append("icao_metar_airport_station")
    if station_name:
        notes.append(f"station_name={station_name}")
    if city_to_station_distance_km is None:
        notes.append("city_to_station_distance_proxy_unavailable")
    elif city_to_station_distance_km <= 15:
        notes.append("city_core_or_close_airport_proxy")
    elif city_to_station_distance_km <= 45:
        notes.append("metro_airport_proxy")
    else:
        notes.append("outer_airport_or_regional_station_proxy")
    return coastal_class, "; ".join(notes)


def _haversine_km(
    lat1: Optional[float],
    lon1: Optional[float],
    lat2: Optional[float],
    lon2: Optional[float],
) -> Optional[float]:
    if lat1 is None or lon1 is None or lat2 is None or lon2 is None:
        return None
    radius_km = 6371.0088
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2.0) ** 2
    return round(radius_km * 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a)), 3)


def _ensure_columns(conn: sqlite3.Connection, table: str, columns: Mapping[str, str]) -> None:
    existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    for name, column_type in columns.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {column_type}")


def _load_decision_times(
    decision_db: Path,
    *,
    config_by_url: Mapping[str, StationConfig],
    start: date,
    end: date,
) -> list[dict[str, Any]]:
    urls = list(config_by_url)
    if not urls:
        return []
    placeholders = ",".join("?" for _ in urls)
    query = f"""
        SELECT DISTINCT canonical_source_url, market_date_local, timezone, decision_time_utc
        FROM decision_snapshots_current_enriched
        WHERE canonical_source_url IN ({placeholders})
          AND market_date_local >= ?
          AND market_date_local <= ?
        ORDER BY canonical_source_url, market_date_local, decision_time_utc
    """
    with sqlite3.connect(f"file:{decision_db}?mode=ro", uri=True) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(query, (*urls, start.isoformat(), end.isoformat())).fetchall()
    output = []
    for row in rows:
        config = config_by_url[row["canonical_source_url"]]
        output.append(
            {
                "station_id": config.station_id,
                "settlement_source_url": config.settlement_source_url,
                "local_date": row["market_date_local"],
                "timezone_name": row["timezone"],
                "decision_time_utc": row["decision_time_utc"],
            }
        )
    return output


def _load_observation_groups(
    conn: sqlite3.Connection,
    *,
    configs: list[StationConfig],
    start: date,
    end: date,
) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    output: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for config in configs:
        rows = conn.execute(
            """
            SELECT station_id, settlement_source_url, local_date, observed_at_utc, temperature_f, temperature_c
            FROM station_observations
            WHERE station_id = ?
              AND settlement_source_url = ?
              AND local_date >= ?
              AND local_date <= ?
            ORDER BY observed_at_utc
            """,
            (config.station_id, config.settlement_source_url, start.isoformat(), end.isoformat()),
        ).fetchall()
        for row in rows:
            output[(row[0], row[1], row[2])].append(
                {
                    "station_id": row[0],
                    "settlement_source_url": row[1],
                    "local_date": row[2],
                    "observed_at_utc": row[3],
                    "temperature_f": row[4],
                    "temperature_c": row[5],
                }
            )
    return output


def _source_selection_summary(configs: list[StationConfig]) -> list[dict[str, str]]:
    return [
        {
            "station_id": config.station_id,
            "city": config.normalized_city,
            "settlement_source": f"{config.settlement_source_family} {config.settlement_source_url}",
            "historical_observation_source": "Iowa Environmental Mesonet ASOS/AWOS/METAR archive CSV",
            "live_observation_source": f"NOAA AviationWeather current METAR API/cache ({AVIATIONWEATHER_LIVE_METAR_URL}?ids={config.station_id}&format=json)",
            "timezone": config.timezone_name,
            "settlement_unit": config.settlement_unit,
            "settlement_precision": config.settlement_precision,
            "mapping_ingestion_status": config.mapping_ingestion_status,
        }
        for config in configs
    ]


def _station_scope_slug(configs: list[StationConfig]) -> str:
    if len(configs) == len(FIRST_10_STATIONS) and {config.station_id for config in configs} == set(FIRST_10_STATIONS):
        return "first10-station-observation-proof"
    if not configs:
        return "station-observation-proof-empty"
    return f"{len(configs)}-source-row-station-observation-proof"


def _summary_counts(
    conn: sqlite3.Connection,
    *,
    configs: list[StationConfig],
    start: date,
    end: date,
) -> dict[str, Any]:
    station_ids = [config.station_id for config in configs]
    if not station_ids:
        return {
            "normalized_observation_rows": 0,
            "observation_rows_with_temperature": 0,
            "stations_with_observations": 0,
            "station_metadata_rows": 0,
            "daily_high_rows": 0,
            "daily_high_rows_with_temperature": 0,
            "intraday_snapshot_rows": 0,
            "intraday_snapshot_rows_with_high_so_far": 0,
            "intraday_future_observation_violations": 0,
        }
    placeholders = ",".join("?" for _ in station_ids)
    params = (*station_ids, start.isoformat(), end.isoformat())
    obs = conn.execute(
        f"""
        SELECT COUNT(*), SUM(CASE WHEN temperature_f IS NOT NULL THEN 1 ELSE 0 END), COUNT(DISTINCT station_id)
        FROM station_observations
        WHERE station_id IN ({placeholders}) AND local_date >= ? AND local_date <= ?
        """,
        params,
    ).fetchone()
    metadata = conn.execute(
        f"""
        SELECT COUNT(*)
        FROM station_metadata
        WHERE station_id IN ({placeholders})
        """,
        tuple(station_ids),
    ).fetchone()
    daily = conn.execute(
        f"""
        SELECT COUNT(*), SUM(CASE WHEN high_temperature_f IS NOT NULL THEN 1 ELSE 0 END)
        FROM station_daily_highs
        WHERE station_id IN ({placeholders}) AND local_date >= ? AND local_date <= ?
        """,
        params,
    ).fetchone()
    intraday = conn.execute(
        f"""
        SELECT COUNT(*), SUM(CASE WHEN high_so_far_f IS NOT NULL THEN 1 ELSE 0 END)
        FROM station_intraday_high_snapshots
        WHERE station_id IN ({placeholders}) AND local_date >= ? AND local_date <= ?
        """,
        params,
    ).fetchone()
    future_intraday = conn.execute(
        f"""
        SELECT COUNT(*)
        FROM station_intraday_high_snapshots
        WHERE station_id IN ({placeholders})
          AND local_date >= ?
          AND local_date <= ?
          AND high_observed_at_utc IS NOT NULL
          AND high_observed_at_utc > decision_time_utc
        """,
        params,
    ).fetchone()
    return {
        "normalized_observation_rows": int(obs[0] or 0),
        "observation_rows_with_temperature": int(obs[1] or 0),
        "stations_with_observations": int(obs[2] or 0),
        "station_metadata_rows": int(metadata[0] or 0),
        "daily_high_rows": int(daily[0] or 0),
        "daily_high_rows_with_temperature": int(daily[1] or 0),
        "intraday_snapshot_rows": int(intraday[0] or 0),
        "intraday_snapshot_rows_with_high_so_far": int(intraday[1] or 0),
        "intraday_future_observation_violations": int(future_intraday[0] or 0),
    }


def _render_report(report: Mapping[str, Any]) -> str:
    comparison = report.get("canonical_label_comparison") or {}
    summary = comparison.get("summary") or {}
    label_gate = report.get("label_agreement_gate") or {}
    source_probe_coverage = report.get("source_probe_coverage") or {}
    lines = [
        "# Expanded Settlement Station Observation Proof",
        "",
        f"Generated: `{report['generated_at_utc']}`",
        f"Window: `{report['target_window']['start']}` through `{report['target_window']['end']}`",
        f"Expanded to full six_month_v1: `{report['expanded_to_full']}`",
        f"Ready for Phase B/C replay: `{report['ready_for_phase_b_c_replay']}`",
        f"Label agreement gate passed: `{label_gate.get('passed')}`",
        "",
        "## Sources",
        "",
        "- Historical proof/full source: Iowa Environmental Mesonet ASOS/AWOS/METAR CSV, with raw METAR retained and Wunderground settlement URL preserved separately.",
        "- Live source recommendation: NOAA AviationWeather current METAR API/cache for timestamped current station observations.",
        "- Direct Wunderground scraping was not used for bulk ingestion in this proof; it remains the settlement source URL for audit alignment.",
        "",
        "## Source Probe Coverage",
        "",
    ]
    for key, value in source_probe_coverage.items():
        lines.append(f"- `{key}`: `{value}`")
    failed = report.get("probe_failed_source_rows") or []
    skipped = report.get("skipped_source_rows") or []
    if failed:
        lines.extend(
            [
                "",
                "### Failed Probes",
                "",
                "| City | Source | Station | Rows | Parsed | Local dates | Missing temp fraction | Source URL |",
                "| --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for row in failed:
            lines.append(
                f"| {row.get('normalized_city', '')} | {row.get('source_family', '')} | `{row.get('station_id', '')}` | `{row.get('rows', 0)}` | `{row.get('parsed_temperature_rows', 0)}` | `{row.get('local_dates', 0)}/{row.get('expected_local_dates', 0)}` | `{row.get('missing_temperature_fraction')}` | {row.get('settlement_source_url', '')} |"
            )
    if skipped:
        lines.extend(
            [
                "",
                "### Skipped Source Rows",
                "",
                "| City | Source | Station | Status | Reason | Source URL |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
        )
        for row in skipped:
            lines.append(
                f"| {row.get('normalized_city', '')} | {row.get('source_family', '')} | `{row.get('station_id', '')}` | `{row.get('ingestion_status', '')}` | `{row.get('skip_reason', '')}` | {row.get('source_url', '')} |"
            )
    lines.extend(
        [
            "",
        "## Counts",
        "",
        ]
    )
    for key, value in report.get("summary_counts", {}).items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Proof Validation",
            "",
            f"- Passed: `{report['proof_validation'].get('passed')}`",
            "",
            "| City | Source | Station | Rows | Parsed temp rows | Missing temp fraction | Local dates | Passed |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in report["proof_validation"].get("station_checks", []):
        lines.append(
            f"| {row.get('normalized_city', '')} | {row.get('source_family', '')} | `{row['station_id']}` | `{row['rows']}` | `{row['parsed_temperature_rows']}` | `{row['missing_temperature_fraction']}` | `{row['local_dates']}/{row['expected_local_dates']}` | `{row['passed']}` |"
        )
    if report.get("full_validation"):
        lines.extend(
            [
                "",
                "## Full Window Validation",
                "",
                f"- Passed: `{report['full_validation'].get('passed')}`",
                "",
                "| City | Source | Station | Rows | Parsed temp rows | Missing temp fraction | Local dates | Passed |",
                "| --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for row in report["full_validation"].get("station_checks", []):
            lines.append(
                f"| {row.get('normalized_city', '')} | {row.get('source_family', '')} | `{row['station_id']}` | `{row['rows']}` | `{row['parsed_temperature_rows']}` | `{row['missing_temperature_fraction']}` | `{row['local_dates']}/{row['expected_local_dates']}` | `{row['passed']}` |"
            )
    if summary:
        lines.extend(
            [
                "",
                "## Canonical Label Comparison",
                "",
                f"- Rows compared: `{summary.get('rows_compared', 0)}`",
                f"- Manual-review rows filtered: `{summary.get('manual_review_rows_filtered', 0)}`",
                f"- Missing station-high rows: `{summary.get('missing_station_high_rows', 0)}`",
                f"- Mismatch rows: `{summary.get('mismatch_rows', 0)}`",
                f"- Mismatch rate: `{summary.get('mismatch_rate')}`",
                f"- Agreement rate: `{summary.get('agreement_rate')}`",
                f"- Near 2F rows/mismatch/rate: `{summary.get('near_2f_rows_compared', 0)}` / `{summary.get('near_2f_mismatch_rows', 0)}` / `{summary.get('near_2f_mismatch_rate')}`",
                f"- Near 1F rows/mismatch/rate: `{summary.get('near_1f_rows_compared', 0)}` / `{summary.get('near_1f_mismatch_rows', 0)}` / `{summary.get('near_1f_mismatch_rate')}`",
                f"- Near 0.5F rows/mismatch/rate: `{summary.get('near_0_5f_rows_compared', 0)}` / `{summary.get('near_0_5f_mismatch_rows', 0)}` / `{summary.get('near_0_5f_mismatch_rate')}`",
                f"- Near 0.5F agreement rate: `{summary.get('near_0_5f_agreement_rate')}`",
                "",
                "### By Station",
                "",
                "| Station | Rows | Mismatches | Agreement | Near 0.5F agreement | Filtered manual-review |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
        )
        for station, row in sorted((comparison.get("by_station") or {}).items()):
            lines.append(
                f"| {station} | `{row.get('rows_compared', 0)}` | `{row.get('mismatch_rows', 0)}` | `{row.get('agreement_rate')}` | `{row.get('near_0_5f_agreement_rate')}` | `{row.get('manual_review_rows_filtered', 0)}` |"
            )
    if label_gate:
        lines.extend(
            [
                "",
                "## Label Agreement Gate",
                "",
                f"- Passed: `{label_gate.get('passed')}`",
                f"- Required overall agreement: `>= {label_gate.get('overall_agreement_min')}`",
                f"- Observed overall agreement: `{label_gate.get('overall_agreement_rate')}`",
                f"- Required agreement within `{label_gate.get('near_threshold_distance_f')}`F of thresholds: `>= {label_gate.get('near_0_5f_agreement_min')}`",
                f"- Observed agreement within 0.5F: `{label_gate.get('near_0_5f_agreement_rate')}`",
                f"- Whitelist station/source rows: `{len(label_gate.get('whitelist_station_source_rows', []))}`",
                f"- Quarantine station/source rows: `{len(label_gate.get('quarantine_station_source_rows', []))}`",
                f"- Overall failure reasons: `{';'.join(label_gate.get('overall_failure_reasons', []))}`",
                "",
                "### Phase B/C Whitelist",
                "",
                "| City | Station | Source | Rows | Agreement | Near 0.5F agreement |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
        )
        for row in label_gate.get("whitelist_station_source_rows", []):
            lines.append(
                f"| {row.get('normalized_city', '')} | `{row.get('station_id', '')}` | {row.get('settlement_source_family', '')} | `{row.get('rows_compared', 0)}` | `{row.get('agreement_rate')}` | `{row.get('near_0_5f_agreement_rate')}` |"
            )
        lines.extend(
            [
                "",
                "### Quarantine",
                "",
                "| City | Station | Source | Rows | Agreement | Near 0.5F agreement | Reasons | Diagnosis |",
                "| --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for row in label_gate.get("quarantine_station_source_rows", []):
            lines.append(
                f"| {row.get('normalized_city', '')} | `{row.get('station_id', '')}` | {row.get('settlement_source_family', '')} | `{row.get('rows_compared', 0)}` | `{row.get('agreement_rate')}` | `{row.get('near_0_5f_agreement_rate')}` | `{row.get('failure_reasons', '')}` | `{row.get('primary_diagnosis', '')}` |"
            )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
        ]
    )
    for key, value in report.get("artifacts", {}).items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Handoff Rules",
            "",
            "- Use `station_intraday_high_snapshots` only as a point-in-time feature; each row is capped at `observed_at_utc <= decision_time_utc`.",
            "- Use `station_daily_highs` as settlement-station daily target data, not as live decision-time data.",
            "- Use `station_metadata` for station lat/lon/elevation, coarse coastal/inland class, urban/airport notes, timezone, and city-to-station distance proxy.",
            "- Keep `exclude_or_manual_review` rows filtered out of strategy-ready replay outputs.",
            "- Preserve `settlement_source_url`, `source_payload_id`, and raw METAR/cache paths in downstream replay audit trails.",
            "",
            "## Known Limitations",
            "",
        ]
    )
    for item in report.get("known_limitations", []):
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def _phase_ready(
    comparison: Mapping[str, Any],
    proof_validation: Mapping[str, Any],
    full_validation: Mapping[str, Any],
    expanded_to_full: bool,
    label_agreement_gate: Mapping[str, Any],
) -> bool:
    if not expanded_to_full or not proof_validation.get("passed") or not full_validation.get("passed"):
        return False
    summary = (comparison or {}).get("summary") or {}
    if summary.get("missing_station_high_rows", 0) != 0:
        return False
    if not label_agreement_gate or not label_agreement_gate.get("passed"):
        return False
    return True


def _known_limitations(
    comparison: Mapping[str, Any],
    expanded_to_full: bool,
    *,
    label_agreement_gate: Mapping[str, Any],
    skipped_source_rows: list[dict[str, Any]],
    probe_failed_source_rows: list[dict[str, Any]],
) -> list[str]:
    limitations = [
        "IEM ASOS/METAR is an as-is archive with limited quality control; raw METAR strings and cache payloads are retained for audit.",
        "Wunderground settlement pages and weather.gov pages were not scraped for bulk historical data; source URLs remain preserved as settlement provenance while IEM METAR rows provide timestamped observations where probes pass.",
        "Coastal/inland class and urban-airport notes are coarse metadata proxies for modeling slices, not authoritative GIS labels.",
    ]
    if not expanded_to_full:
        limitations.append("Full six_month_v1 ingestion did not run because proof validation did not pass or mode was proof-only.")
    if skipped_source_rows:
        limitations.append("Some mapped source rows were skipped before IEM probing because the current connector supports Wunderground/NOAA ICAO-style METAR rows, not custom CWA station pages.")
    if probe_failed_source_rows:
        limitations.append("Some supported source rows failed the proof probe and were not expanded into the full six-month window.")
    summary = (comparison or {}).get("summary") or {}
    if summary.get("mismatch_rate") is not None and not label_agreement_gate.get("passed", False):
        limitations.append("Station-derived labels do not pass the station/source label-agreement gate; use only the whitelist artifact for Phase B/C research and keep quarantined stations out of replay/model promotion.")
    return limitations


def _station_label_for_row(row: Mapping[str, str], high_f: float) -> Optional[int]:
    settlement_value = _settlement_value(row, high_f)
    if settlement_value is None:
        return None
    lower, upper = _settlement_thresholds_for_row(row)
    if lower is not None and settlement_value < lower - 1e-9:
        return 0
    if upper is not None and settlement_value > upper + 1e-9:
        return 0
    return 1


def _settlement_value(row: Mapping[str, str], high_f: float) -> Optional[float]:
    unit = row.get("settlement_unit") or "F"
    precision = row.get("settlement_precision") or "whole_F"
    value = high_f if unit == "F" else _f_to_c(high_f)
    if precision.startswith("whole_"):
        return float(_round_half_up(value))
    if precision.startswith("decimal_0.1"):
        return float(_round_tenth_half_up(value))
    return value


def _settlement_thresholds_for_row(row: Mapping[str, str]) -> tuple[Optional[float], Optional[float]]:
    unit = row.get("settlement_unit") or "F"
    lower = _optional_float(row.get("threshold_low_display"))
    upper = _optional_float(row.get("threshold_high_display"))
    if lower is None:
        lower_f = _optional_float(row.get("threshold_low_f"))
        if lower_f is not None:
            lower = lower_f if unit == "F" else _f_to_c(lower_f)
    if upper is None:
        upper_f = _optional_float(row.get("threshold_high_f"))
        if upper_f is not None:
            upper = upper_f if unit == "F" else _f_to_c(upper_f)
    return lower, upper


def _label_mismatch_cause(
    row: Mapping[str, str],
    *,
    high_f: float,
    station_label: Optional[int],
    canonical_label: Optional[int],
    station_high_source_unit: str,
) -> str:
    if station_label is None or canonical_label is None or station_label == canonical_label:
        return ""
    distance = _threshold_distance_f(row, high_f)
    settlement_unit = row.get("settlement_unit") or "F"
    if distance is not None and distance <= NEAR_THRESHOLD_DISTANCE_F:
        if station_high_source_unit and station_high_source_unit != settlement_unit:
            return "near_threshold_source_unit_rounding_or_display_difference"
        return "near_threshold_rounding_or_display_difference"
    if station_high_source_unit and station_high_source_unit != settlement_unit:
        return "source_unit_rounding_or_station_display_difference"
    return "polymarket_vs_station_settlement_difference"


def _threshold_distance_f(row: Mapping[str, str], high_f: float) -> Optional[float]:
    thresholds = [_optional_float(row.get("threshold_low_f")), _optional_float(row.get("threshold_high_f"))]
    thresholds = [value for value in thresholds if value is not None]
    if not thresholds:
        return None
    return min(abs(high_f - value) for value in thresholds)


def _distance_bucket(distance: Optional[float]) -> str:
    if distance is None:
        return "unknown"
    if distance <= 0.5:
        return "0-0.5F"
    if distance <= 1.0:
        return "0.5-1F"
    if distance <= 2.0:
        return "1-2F"
    if distance <= 4.0:
        return "2-4F"
    return "4F+"


def _counter_rates(counters: Mapping[str, Counter[str]]) -> dict[str, dict[str, Any]]:
    output = {}
    for key, counter in counters.items():
        row = dict(counter)
        row["mismatch_rate"] = _rate(counter.get("mismatch_rows", 0), counter.get("rows_compared", 0))
        row["agreement_rate"] = _agreement_rate(counter.get("mismatch_rows", 0), counter.get("rows_compared", 0))
        row["near_0_5f_mismatch_rate"] = _rate(
            counter.get("near_0_5f_mismatch_rows", 0),
            counter.get("near_0_5f_rows_compared", 0),
        )
        row["near_0_5f_agreement_rate"] = _agreement_rate(
            counter.get("near_0_5f_mismatch_rows", 0),
            counter.get("near_0_5f_rows_compared", 0),
        )
        output[key] = row
    return output


def _station_source_counter_rates(
    counters: Mapping[str, Counter[str]],
    config_by_key: Mapping[str, StationConfig],
) -> dict[str, dict[str, Any]]:
    output = _counter_rates(counters)
    for key, row in output.items():
        config = config_by_key.get(key)
        if config is None:
            continue
        row.update(
            {
                "station_id": config.station_id,
                "normalized_city": config.normalized_city,
                "settlement_source_family": config.settlement_source_family,
                "settlement_source_url": config.settlement_source_url,
                "settlement_station_name": config.settlement_station_name,
                "settlement_unit": config.settlement_unit,
                "settlement_precision": config.settlement_precision,
                "dominant_mismatch_cause": _dominant_mismatch_cause(row),
            }
        )
    return output


def _record_label_comparison(
    counter: Counter[str],
    *,
    mismatch: int,
    distance: Optional[float],
    cause: str,
) -> None:
    counter["rows_compared"] += 1
    counter["mismatch_rows"] += mismatch
    if distance is not None and distance <= NEAR_THRESHOLD_DISTANCE_F:
        counter["near_0_5f_rows_compared"] += 1
        counter["near_0_5f_mismatch_rows"] += mismatch
    if mismatch and cause:
        counter[f"mismatch_cause__{cause}"] += 1


def _station_source_key(station_id: str, settlement_source_url: str) -> str:
    return f"{station_id}{STATION_SOURCE_KEY_SEPARATOR}{settlement_source_url}"


def _dominant_mismatch_cause(stats: Mapping[str, Any]) -> str:
    causes = [
        (key.removeprefix("mismatch_cause__"), int(value))
        for key, value in stats.items()
        if key.startswith("mismatch_cause__") and int(value or 0) > 0
    ]
    if not causes:
        return ""
    causes.sort(key=lambda item: (-item[1], item[0]))
    return causes[0][0]


def _primary_station_source_diagnosis(stats: Mapping[str, Any], failure_reasons: list[str]) -> str:
    if not failure_reasons:
        return "passes_station_source_label_agreement_gate"
    dominant_cause = _dominant_mismatch_cause(stats)
    if dominant_cause:
        return dominant_cause
    if "parse_coverage_failed" in failure_reasons:
        return "parse_coverage_failure"
    if "no_label_rows_compared" in failure_reasons:
        return "no_label_rows_available_for_station_source"
    return "label_agreement_gate_failure"


def _label_agreement_diagnostic_rows(comparison: Mapping[str, Any]) -> list[dict[str, Any]]:
    sections = [
        ("station", comparison.get("by_station") or {}),
        ("station_source", comparison.get("by_station_source") or {}),
        ("city", comparison.get("by_city") or {}),
        ("station_high_source_unit", comparison.get("by_station_high_source_unit") or {}),
        ("settlement_unit", comparison.get("by_settlement_unit") or {}),
        ("threshold_distance_bucket_f", comparison.get("by_threshold_distance_bucket_f") or {}),
        ("bucket_type", comparison.get("by_bucket_type") or {}),
        ("month", comparison.get("by_month") or {}),
        ("mismatch_cause", comparison.get("by_mismatch_cause") or {}),
    ]
    rows = []
    for dimension, values in sections:
        for value, stats in sorted(values.items()):
            row = {
                "dimension": dimension,
                "value": value.replace(STATION_SOURCE_KEY_SEPARATOR, " "),
                "rows_compared": stats.get("rows_compared", 0),
                "mismatch_rows": stats.get("mismatch_rows", 0),
                "mismatch_rate": stats.get("mismatch_rate"),
                "agreement_rate": stats.get("agreement_rate"),
                "near_0_5f_rows_compared": stats.get("near_0_5f_rows_compared", 0),
                "near_0_5f_mismatch_rows": stats.get("near_0_5f_mismatch_rows", 0),
                "near_0_5f_mismatch_rate": stats.get("near_0_5f_mismatch_rate"),
                "near_0_5f_agreement_rate": stats.get("near_0_5f_agreement_rate"),
                "manual_review_rows_filtered": stats.get("manual_review_rows_filtered", 0),
                "missing_station_high_rows": stats.get("missing_station_high_rows", 0),
            }
            if dimension == "station_source":
                row.update(
                    {
                        "station_id": stats.get("station_id", ""),
                        "normalized_city": stats.get("normalized_city", ""),
                        "settlement_source_family": stats.get("settlement_source_family", ""),
                        "settlement_source_url": stats.get("settlement_source_url", ""),
                        "settlement_unit": stats.get("settlement_unit", ""),
                        "settlement_precision": stats.get("settlement_precision", ""),
                        "dominant_mismatch_cause": stats.get("dominant_mismatch_cause", ""),
                    }
                )
            rows.append(row)
    return rows


def _write_dict_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = list(rows)
    with path.open("w", newline="", encoding="utf-8") as f:
        if not rows:
            f.write("")
            return
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def _export_query(conn: sqlite3.Connection, path: Path, query: str, params: tuple[Any, ...]) -> None:
    rows = conn.execute(query, params).fetchall()
    with path.open("w", newline="", encoding="utf-8") as f:
        if not rows:
            f.write("")
            return
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def _parse_iem_valid_utc(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)


def _parse_precise_t_group_c(metar: str) -> Optional[float]:
    match = PRECISE_T_RE.search(metar)
    if not match:
        return None
    sign = -1 if match.group("t_sign") == "1" else 1
    return sign * (int(match.group("t_tenths")) / 10.0)


def _parse_main_metar_temperature_c(metar: str) -> Optional[float]:
    match = MAIN_TEMP_RE.search(metar)
    if not match:
        return None
    return float(_metar_signed_int(match.group("temp")))


def _metar_signed_int(value: str) -> int:
    if value.startswith("M"):
        return -int(value[1:])
    return int(value)


def _temperature_result(raw_temperature: float, source_unit: str, flag: str) -> dict[str, Any]:
    if source_unit == "F":
        temp_f = raw_temperature
        temp_c = _f_to_c(raw_temperature)
    else:
        temp_c = raw_temperature
        temp_f = _c_to_f(raw_temperature)
    return {
        "raw_temperature": raw_temperature,
        "source_unit": source_unit,
        "temperature_f": round(temp_f, 6),
        "temperature_c": round(temp_c, 6),
        "quality_flags": (flag,),
    }


def _c_to_f(value: float) -> float:
    return value * 9.0 / 5.0 + 32.0


def _f_to_c(value: float) -> float:
    return (value - 32.0) * 5.0 / 9.0


def _optional_float(value: Any) -> Optional[float]:
    if value in (None, "", "M"):
        return None
    try:
        if isinstance(value, float) and math.isnan(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    return int(float(value))


def _round_half_up(value: float) -> int:
    return int(Decimal(str(value)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def _round_tenth_half_up(value: float) -> float:
    return float(Decimal(str(value)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def _payload_id(request_url: str, payload: bytes, settlement_source_url: str = "") -> str:
    return (
        "sha256:"
        + hashlib.sha256(
            request_url.encode("utf-8") + b"\0" + settlement_source_url.encode("utf-8") + b"\0" + payload
        ).hexdigest()
    )


def _date_or_none(value: str) -> Optional[date]:
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _rate(num: int, den: int) -> Optional[float]:
    if not den:
        return None
    return round(num / den, 8)


def _agreement_rate(mismatch_rows: int, rows_compared: int) -> Optional[float]:
    rate = _rate(mismatch_rows, rows_compared)
    if rate is None:
        return None
    return round(1.0 - rate, 8)


def _dedupe(values: Iterable[str]) -> list[str]:
    output = []
    seen = set()
    for value in values:
        if value and value not in seen:
            output.append(value)
            seen.add(value)
    return output


def _parse_station_ids(value: str) -> tuple[str, ...]:
    if value.strip().lower() in {"", "auto", "all", "probe-supported"}:
        return ()
    return tuple(part.strip().upper() for part in value.split(",") if part.strip())


def _parse_date_arg(value: str) -> date:
    return date.fromisoformat(value)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build settlement-station observation artifacts for probe-supported METAR stations.")
    parser.add_argument("--work-root", default=".")
    parser.add_argument("--start-date", default=DEFAULT_START_DATE.isoformat())
    parser.add_argument("--end-date", default=DEFAULT_END_DATE.isoformat())
    parser.add_argument("--proof-start-date", default=DEFAULT_PROOF_START_DATE.isoformat())
    parser.add_argument("--proof-end-date", default=DEFAULT_PROOF_END_DATE.isoformat())
    parser.add_argument("--station-ids", default="auto", help="Comma-separated station ids, or auto/all for all probe-supported mapping rows.")
    parser.add_argument("--mapping-json", default=None)
    parser.add_argument("--labels-csv", default=None)
    parser.add_argument("--mode", choices=("proof", "proof_then_full", "full"), default="proof_then_full")
    parser.add_argument("--cache-only", action="store_true")
    args = parser.parse_args(argv)
    report = run_station_observation_pipeline(
        work_root=Path(args.work_root),
        start=_parse_date_arg(args.start_date),
        end=_parse_date_arg(args.end_date),
        proof_start=_parse_date_arg(args.proof_start_date),
        proof_end=_parse_date_arg(args.proof_end_date),
        station_ids=_parse_station_ids(args.station_ids) or None,
        mapping_json=Path(args.mapping_json) if args.mapping_json else None,
        labels_csv=Path(args.labels_csv) if args.labels_csv else None,
        mode=args.mode,
        cache_only=args.cache_only,
    )
    print(
        json.dumps(
            {
                "expanded_to_full": report["expanded_to_full"],
                "ready_for_phase_b_c_replay": report["ready_for_phase_b_c_replay"],
                "artifacts": report["artifacts"],
                "summary_counts": report["summary_counts"],
                "canonical_label_comparison": report.get("canonical_label_comparison", {}).get("summary", {}),
                "label_agreement_gate": {
                    "passed": report.get("label_agreement_gate", {}).get("passed"),
                    "overall_agreement_rate": report.get("label_agreement_gate", {}).get("overall_agreement_rate"),
                    "near_0_5f_agreement_rate": report.get("label_agreement_gate", {}).get("near_0_5f_agreement_rate"),
                    "whitelist_station_ids": report.get("label_agreement_gate", {}).get("whitelist_station_ids", []),
                    "quarantine_station_ids": report.get("label_agreement_gate", {}).get("quarantine_station_ids", []),
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
