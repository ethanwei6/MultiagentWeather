"""Research-only contracts for rebuilding the physical weather model.

Nothing in this module places orders, mutates a ledger, fetches network data, or
fits a model.  It exists to reject station/source drift and future information
before any result-bearing physical-weather study is defined.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import parse_qs, urlparse
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


CONTRACT_VERSION = "dailyweather.physical_weather_source_contract.v1"
GFS_BUCKET = "noaa-gfs-bdp-pds"
GFS_MODEL = "gfs_0p25"
GFS_MINIMUM_RUN_AGE = timedelta(hours=6)
GFS_CYCLE_HOURS = (0, 6, 12, 18)
GFS_MAX_FORECAST_HOUR = 120
GFS_GRID_DEGREES = 0.25
GFS_VARIABLE = "TMP:2 m above ground"
GFS_INTERPOLATION = "bilinear_four_surrounding_0p25_degree_nodes"
GFS_ELEVATION_HANDLING = "no_lapse_rate_adjustment_residual_calibration_only"
GFS_DECODER = "eccodes"
CANONICAL_LABEL_SOURCE = "polymarket_canonical_payout"
PHYSICAL_TARGET_SOURCE = "NOAA_ISD"
RULE_PROVENANCE_INCOMPLETE = "RULE_PROVENANCE_INCOMPLETE"
STATION_INFO_RAW_SHA256 = "ac9fb2c3524e89a32b290453db2b42e5c78d69d6831a4388357f3dcd5ad81381"


class PhysicalWeatherContractError(ValueError):
    """Raised when a physical-weather input violates the frozen contract."""


@dataclass(frozen=True)
class SettlementBinding:
    city: str
    station_id: str
    station_latitude: float
    station_longitude: float
    station_elevation_m: float
    timezone_name: str
    source_family: str
    source_url: str
    unit: str


# Exact bindings parsed from locally preserved Gamma market payloads for the
# July/August recorded universe available on 2026-08-17 (through target dates
# 2026-08-02/03).  A later rule change must fail until this contract is reviewed.
RECORDED_SETTLEMENT_BINDINGS: tuple[SettlementBinding, ...] = (
    SettlementBinding("Austin, TX", "KAUS", 30.1831, -97.68063, 148, "America/Chicago", "Wunderground", "https://www.wunderground.com/history/daily/us/tx/austin/KAUS", "F"),
    SettlementBinding("Beijing, CN", "ZBAA", 40.082, 116.603, 31, "Asia/Shanghai", "Wunderground", "https://www.wunderground.com/history/daily/cn/beijing/ZBAA", "C"),
    SettlementBinding("Chicago, IL", "KORD", 41.96017, -87.93161, 202, "America/Chicago", "Wunderground", "https://www.wunderground.com/history/daily/us/il/chicago/KORD", "F"),
    SettlementBinding("Dallas, TX", "KDAL", 32.83836, -96.83584, 148, "America/Chicago", "Wunderground", "https://www.wunderground.com/history/daily/us/tx/dallas/KDAL", "F"),
    SettlementBinding("Denver, CO", "KBKF", 39.713, -104.758, 1703, "America/Denver", "Wunderground", "https://www.wunderground.com/history/daily/us/co/aurora/KBKF", "F"),
    SettlementBinding("Guangzhou, CN", "ZGGG", 23.392, 113.307, 11, "Asia/Shanghai", "Wunderground", "https://www.wunderground.com/history/daily/cn/guangzhou/ZGGG", "C"),
    SettlementBinding("Houston, TX", "KHOU", 29.64582, -95.28214, 13, "America/Chicago", "Wunderground", "https://www.wunderground.com/history/daily/us/tx/houston/KHOU", "F"),
    SettlementBinding("London, GB", "EGLC", 51.505, 0.055, 10, "Europe/London", "Wunderground", "https://www.wunderground.com/history/daily/gb/london/EGLC", "C"),
    SettlementBinding("Los Angeles, CA", "KLAX", 33.93817, -118.3866, 30, "America/Los_Angeles", "Wunderground", "https://www.wunderground.com/history/daily/us/ca/los-angeles/KLAX", "F"),
    SettlementBinding("Madrid, ES", "LEMD", 40.466, -3.555, 589, "Europe/Madrid", "Wunderground", "https://www.wunderground.com/history/daily/es/madrid/LEMD", "C"),
    SettlementBinding("Miami, FL", "KMIA", 25.78806, -80.31692, 1, "America/New_York", "Wunderground", "https://www.wunderground.com/history/daily/us/fl/miami/KMIA", "F"),
    SettlementBinding("Moscow, RU", "UUWW", 55.592, 37.261, 195, "Europe/Moscow", "NOAA", "https://www.weather.gov/wrh/timeseries?site=UUWW", "C"),
    SettlementBinding("New York, NY", "KLGA", 40.77945, -73.88027, 9, "America/New_York", "Wunderground", "https://www.wunderground.com/history/daily/us/ny/new-york-city/KLGA", "F"),
    SettlementBinding("Paris, FR", "LFPB", 48.967, 2.428, 50, "Europe/Paris", "Wunderground", "https://www.wunderground.com/history/daily/fr/bonneuil-en-france/LFPB", "C"),
    SettlementBinding("San Francisco, CA", "KSFO", 37.61961, -122.36561, 2, "America/Los_Angeles", "Wunderground", "https://www.wunderground.com/history/daily/us/ca/san-francisco/KSFO", "F"),
    SettlementBinding("Seattle, WA", "KSEA", 47.44467, -122.31442, 115, "America/Los_Angeles", "Wunderground", "https://www.wunderground.com/history/daily/us/wa/seatac/KSEA", "F"),
    SettlementBinding("Seoul, KR", "RKSI", 37.469, 126.451, 7, "Asia/Seoul", "Wunderground", "https://www.wunderground.com/history/daily/kr/incheon/RKSI", "C"),
    SettlementBinding("Shanghai, CN", "ZSPD", 31.146, 121.8, 4, "Asia/Shanghai", "Wunderground", "https://www.wunderground.com/history/daily/cn/shanghai/ZSPD", "C"),
    SettlementBinding("Shenzhen, CN", "ZGSZ", 22.639, 113.803, 18, "Asia/Shanghai", "Wunderground", "https://www.wunderground.com/history/daily/cn/shenzhen/ZGSZ", "C"),
    SettlementBinding("Singapore, SG", "WSSS", 1.368, 103.982, 17, "Asia/Singapore", "Wunderground", "https://www.wunderground.com/history/daily/sg/singapore/WSSS", "C"),
    SettlementBinding("Taipei, TW", "RCSS", 25.069, 121.552, 8, "Asia/Taipei", "Wunderground", "https://www.wunderground.com/history/daily/tw/taipei/RCSS", "C"),
    SettlementBinding("Tokyo, JP", "RJTT", 35.553, 139.781, 5, "Asia/Tokyo", "Wunderground", "https://www.wunderground.com/history/daily/jp/tokyo/RJTT", "C"),
)

QUESTION_CITY_ALIASES: Mapping[str, tuple[str, ...]] = {
    "New York, NY": ("New York", "New York City"),
    "Seoul, KR": ("Seoul", "Seoul (Incheon)"),
}


@dataclass(frozen=True)
class Bucket:
    lower_inclusive: int | None
    upper_inclusive: int | None


@dataclass(frozen=True)
class MarketRule:
    event_id: str
    market_id: str
    condition_id: str
    target_date: date
    binding: SettlementBinding
    bucket: Bucket
    question: str
    payload_sha256: str
    retrieved_at_utc: datetime
    display_precision: str = "whole_degree"
    local_date_semantics: str = "settlement_station_local_calendar_date"
    revision_cutoff: str = "first_datapoint_for_following_date"
    rounding_rule: str = "not_specified_use_canonical_payout"


@dataclass(frozen=True)
class RegistryMismatch:
    city: str
    runtime_station_id: str
    settlement_station_id: str


@dataclass(frozen=True)
class GfsGridNode:
    latitude: float
    longitude: float
    interpolation_weight: float
    temperature_k: float


@dataclass(frozen=True)
class GfsInputProvenance:
    station_id: str
    station_latitude: float
    station_longitude: float
    station_metadata_source_url: str
    station_metadata_sha256: str
    run_time_utc: datetime
    forecast_hour: int
    valid_time_utc: datetime
    object_key: str
    object_last_modified_utc: datetime
    object_etag: str
    object_size_bytes: int
    index_key: str
    index_last_modified_utc: datetime
    index_sha256: str
    byte_range_start: int
    byte_range_end_inclusive: int
    raw_range_sha256: str
    variable_descriptor: str
    grid_nodes: tuple[GfsGridNode, ...]
    decoded_temperature_k: float
    decoder_name: str
    decoder_version: str
    decoder_binary_sha256: str
    decoded_message_sha256: str
    fetched_at_utc: datetime
    feature_assembled_at_utc: datetime
    quote_time_utc: datetime
    decision_time_utc: datetime


@dataclass(frozen=True)
class PhysicalTargetProvenance:
    station_id: str
    local_date: date
    timezone_name: str
    source_dataset: str
    source_url: str
    raw_payload_sha256: str
    first_available_at_utc: datetime
    computed_at_utc: datetime
    maximum_temperature_c: float


@dataclass(frozen=True)
class CanonicalMarketLabel:
    market_id: str
    condition_id: str
    yes_outcome: int
    source: str
    raw_payload_sha256: str
    label_available_at_utc: datetime


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def recorded_inventory_sha256() -> str:
    return _canonical_sha256([binding.__dict__ for binding in RECORDED_SETTLEMENT_BINDINGS])


def _bindings_by_city() -> dict[str, SettlementBinding]:
    return {binding.city: binding for binding in RECORDED_SETTLEMENT_BINDINGS}


def _require_utc(value: datetime, field: str) -> datetime:
    if value.tzinfo is None:
        raise PhysicalWeatherContractError(f"{field} must be timezone-aware")
    return value.astimezone(timezone.utc)


def _require_sha256(value: str, field: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise PhysicalWeatherContractError(f"{field} is not a lowercase SHA-256")


def _source_url_from_payload(payload: Mapping[str, Any], description: str) -> str:
    explicit = str(payload.get("resolutionSource") or "").strip().rstrip(".")
    if explicit:
        return explicit
    urls = [item.rstrip(".,") for item in re.findall(r"https?://[^\s\"')]+", description)]
    candidates = [item for item in urls if "weather.gov/wrh/timeseries" in item]
    if len(candidates) != 1:
        raise PhysicalWeatherContractError("Gamma rule does not identify one settlement URL")
    return candidates[0]


def _source_and_station(source_url: str) -> tuple[str, str]:
    parsed = urlparse(source_url)
    host = parsed.netloc.lower()
    if host in {"wunderground.com", "www.wunderground.com"}:
        station = parsed.path.rstrip("/").rsplit("/", 1)[-1].upper()
        source = "Wunderground"
    elif host.endswith("weather.gov"):
        sites = parse_qs(parsed.query).get("site", [])
        station = sites[0].upper() if len(sites) == 1 else ""
        source = "NOAA"
    else:
        raise PhysicalWeatherContractError("Gamma rule uses an unsupported settlement source")
    if not re.fullmatch(r"[A-Z0-9]{4}", station):
        raise PhysicalWeatherContractError("Gamma rule does not identify an ICAO-style station")
    return source, station


def _parse_bucket(question: str, expected_unit: str) -> Bucket:
    unit_match = re.search(r"°([FC])", question)
    if unit_match is None or unit_match.group(1) != expected_unit:
        raise PhysicalWeatherContractError("question unit conflicts with settlement rule")
    normalized = question.replace("−", "-")
    range_match = re.search(r"between\s+(-?\d+)\s*-\s*(-?\d+)°[FC]", normalized, re.I)
    if range_match:
        lower, upper = int(range_match.group(1)), int(range_match.group(2))
        if lower > upper:
            raise PhysicalWeatherContractError("bucket range is reversed")
        return Bucket(lower, upper)
    threshold_match = re.search(r"be\s+(-?\d+)°[FC]\s+or\s+(below|lower|higher|above)", normalized, re.I)
    if threshold_match:
        value = int(threshold_match.group(1))
        direction = threshold_match.group(2).lower()
        return Bucket(None, value) if direction in {"below", "lower"} else Bucket(value, None)
    exact_match = re.search(r"be\s+(-?\d+)°[FC](?:\s|\?|$)", normalized, re.I)
    if exact_match:
        value = int(exact_match.group(1))
        return Bucket(value, value)
    raise PhysicalWeatherContractError("question bucket is not a supported integer partition")


def parse_gamma_market_rule(
    payload: Mapping[str, Any],
    *,
    event_id: str,
    retrieved_at_utc: datetime,
) -> MarketRule:
    """Bind one exact Gamma market payload to its settlement station and bucket."""

    try:
        city, event_date_text = event_id.rsplit("|", 1)
        event_date = date.fromisoformat(event_date_text)
    except ValueError as exc:
        raise PhysicalWeatherContractError("event_id must end in an ISO local date") from exc
    expected = _bindings_by_city().get(city)
    if expected is None:
        raise PhysicalWeatherContractError("event city is outside the frozen recorded universe")
    market_id = str(payload.get("id") or "")
    condition_id = str(payload.get("conditionId") or "")
    question = str(payload.get("question") or "")
    description = str(payload.get("description") or "")
    if not market_id or not condition_id or not question or not description:
        raise PhysicalWeatherContractError("Gamma payload is missing rule identity or text")
    target_text = str(payload.get("endDateIso") or payload.get("endDate") or "")[:10]
    try:
        target_date = date.fromisoformat(target_text)
    except ValueError as exc:
        raise PhysicalWeatherContractError("Gamma payload target date is not ISO-8601") from exc
    if target_date != event_date:
        raise PhysicalWeatherContractError("Gamma target date conflicts with event_id")
    city_names = QUESTION_CITY_ALIASES.get(city, (city.rsplit(",", 1)[0],))
    if not any(
        re.search(rf"highest temperature in {re.escape(city_name)}\s+be", question, re.I)
        for city_name in city_names
    ):
        raise PhysicalWeatherContractError("Gamma question city conflicts with event_id")

    source_url = _source_url_from_payload(payload, description)
    source_family, station_id = _source_and_station(source_url)
    fahrenheit = bool(re.search(r"whole degrees Fahrenheit", description, re.I))
    celsius = bool(re.search(r"whole degrees Celsius", description, re.I))
    if fahrenheit == celsius:
        raise PhysicalWeatherContractError("Gamma rule does not state one whole-degree unit")
    unit = "F" if fahrenheit else "C"
    if "first data point for the following date" not in description.lower() and "first datapoint for the following date" not in description.lower():
        raise PhysicalWeatherContractError("Gamma rule does not pin the revision cutoff")
    parsed_binding = SettlementBinding(
        city=city,
        station_id=station_id,
        station_latitude=expected.station_latitude,
        station_longitude=expected.station_longitude,
        station_elevation_m=expected.station_elevation_m,
        timezone_name=expected.timezone_name,
        source_family=source_family,
        source_url=source_url,
        unit=unit,
    )
    if parsed_binding != expected:
        raise PhysicalWeatherContractError(
            f"Gamma settlement binding changed for {city}: {parsed_binding!r}"
        )
    retrieved = _require_utc(retrieved_at_utc, "retrieved_at_utc")
    return MarketRule(
        event_id=event_id,
        market_id=market_id,
        condition_id=condition_id,
        target_date=target_date,
        binding=parsed_binding,
        bucket=_parse_bucket(question, unit),
        question=question,
        payload_sha256=_canonical_sha256(payload),
        retrieved_at_utc=retrieved,
    )


def validate_rule_available_at_decision(rule: MarketRule, decision_time_utc: datetime) -> None:
    if rule.retrieved_at_utc > _require_utc(decision_time_utc, "decision_time_utc"):
        raise PhysicalWeatherContractError(
            f"market rule was retrieved after the decision: {RULE_PROVENANCE_INCOMPLETE}"
        )


def validate_event_partition(rules: Iterable[MarketRule]) -> SettlementBinding:
    """Require one source binding and an exhaustive, non-overlapping integer partition."""

    rows = list(rules)
    if not rows:
        raise PhysicalWeatherContractError("event has no market rules")
    event_ids = {row.event_id for row in rows}
    bindings = {row.binding for row in rows}
    market_ids = {row.market_id for row in rows}
    condition_ids = {row.condition_id for row in rows}
    if len(event_ids) != 1 or len(bindings) != 1:
        raise PhysicalWeatherContractError("event buckets do not share one event/source binding")
    if len(market_ids) != len(rows) or len(condition_ids) != len(rows):
        raise PhysicalWeatherContractError("event contains duplicate market or condition identities")
    ordered = sorted(
        (row.bucket for row in rows),
        key=lambda item: float("-inf") if item.lower_inclusive is None else item.lower_inclusive,
    )
    if ordered[0].lower_inclusive is not None or ordered[-1].upper_inclusive is not None:
        raise PhysicalWeatherContractError("event integer buckets are not exhaustive")
    for left, right in zip(ordered, ordered[1:]):
        if left.upper_inclusive is None or right.lower_inclusive is None:
            raise PhysicalWeatherContractError("event has overlapping unbounded buckets")
        if left.upper_inclusive + 1 != right.lower_inclusive:
            raise PhysicalWeatherContractError("event integer buckets have a gap or overlap")
    return rows[0].binding


def runtime_registry_mismatches(registry_entries: Sequence[Sequence[Any]]) -> tuple[RegistryMismatch, ...]:
    expected = _bindings_by_city()
    seen: set[str] = set()
    mismatches: list[RegistryMismatch] = []
    for entry in registry_entries:
        if len(entry) < 6:
            raise PhysicalWeatherContractError("runtime registry entry is malformed")
        city = f"{entry[0]}, {entry[1]}"
        if city not in expected or city in seen:
            raise PhysicalWeatherContractError("runtime registry city set is malformed")
        seen.add(city)
        runtime_station = str(entry[5]).upper()
        if runtime_station != expected[city].station_id:
            mismatches.append(RegistryMismatch(city, runtime_station, expected[city].station_id))
    if seen != set(expected):
        raise PhysicalWeatherContractError("runtime registry does not cover the 22-city universe")
    return tuple(sorted(mismatches, key=lambda item: item.city))


def assert_runtime_registry_station_aligned(registry_entries: Sequence[Sequence[Any]]) -> None:
    mismatches = runtime_registry_mismatches(registry_entries)
    if mismatches:
        details = ", ".join(
            f"{row.city}:{row.runtime_station_id}->{row.settlement_station_id}"
            for row in mismatches
        )
        raise PhysicalWeatherContractError(f"runtime settlement-station mismatch: {details}")


def validate_station_info_probe(payload: Mapping[str, Any]) -> None:
    """Validate the frozen official 22-station coordinate response."""

    if int(payload.get("http_status", 0)) != 200:
        raise PhysicalWeatherContractError("station-info probe did not return HTTP 200")
    parsed_url = urlparse(str(payload.get("request_url") or ""))
    if parsed_url.netloc != "aviationweather.gov" or parsed_url.path != "/api/data/stationinfo":
        raise PhysicalWeatherContractError("station-info probe is not from AviationWeather")
    rows = payload.get("response_body")
    if not isinstance(rows, list):
        raise PhysicalWeatherContractError("station-info response body is malformed")
    raw = json.dumps(rows, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    actual_sha256 = hashlib.sha256(raw).hexdigest()
    if actual_sha256 != payload.get("raw_response_sha256") or actual_sha256 != STATION_INFO_RAW_SHA256:
        raise PhysicalWeatherContractError("station-info raw response hash changed")
    expected = {row.station_id: row for row in RECORDED_SETTLEMENT_BINDINGS}
    actual: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise PhysicalWeatherContractError("station-info row is malformed")
        station_id = str(row.get("icaoId") or "")
        if station_id in actual:
            raise PhysicalWeatherContractError("station-info response has duplicate station IDs")
        actual[station_id] = row
    if set(actual) != set(expected):
        raise PhysicalWeatherContractError("station-info response does not cover the 22 stations")
    for station_id, binding in expected.items():
        row = actual[station_id]
        if "METAR" not in row.get("siteType", ()):
            raise PhysicalWeatherContractError(f"{station_id} is not a METAR station")
        if (float(row["lat"]), float(row["lon"]), float(row["elev"])) != (
            binding.station_latitude,
            binding.station_longitude,
            binding.station_elevation_m,
        ):
            raise PhysicalWeatherContractError(f"{station_id} official coordinates changed")


def gfs_t_minus_six_cycle(decision_time_utc: datetime) -> datetime:
    decision = _require_utc(decision_time_utc, "decision_time_utc")
    cutoff = decision - GFS_MINIMUM_RUN_AGE
    cycle_hour = max(hour for hour in GFS_CYCLE_HOURS if hour <= cutoff.hour)
    return cutoff.replace(hour=cycle_hour, minute=0, second=0, microsecond=0)


def required_gfs_forecast_hours(
    *,
    run_time_utc: datetime,
    target_date: date,
    timezone_name: str,
) -> tuple[int, ...]:
    """Return every UTC-valid hourly GFS field in one station-local day."""

    run = _require_utc(run_time_utc, "run_time_utc")
    try:
        local_timezone = ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError as exc:
        raise PhysicalWeatherContractError("station timezone is not in the IANA database") from exc
    hours = tuple(
        forecast_hour
        for forecast_hour in range(GFS_MAX_FORECAST_HOUR + 1)
        if (run + timedelta(hours=forecast_hour)).astimezone(local_timezone).date() == target_date
    )
    if len(hours) not in {23, 24, 25}:
        raise PhysicalWeatherContractError("target local day is outside the fixed GFS horizon")
    return hours


def _expected_gfs_variable_descriptor(forecast_hour: int) -> str:
    if forecast_hour == 0:
        return f"{GFS_VARIABLE}:anl"
    return f"{GFS_VARIABLE}:{forecast_hour} hour fcst"


def validate_gfs_input(provenance: GfsInputProvenance, binding: SettlementBinding) -> None:
    """Enforce identical direct-NODD source identity and a causal timing chain."""

    if provenance.station_id != binding.station_id:
        raise PhysicalWeatherContractError("GFS input is not at the settlement station")
    if (provenance.station_latitude, provenance.station_longitude) != (
        binding.station_latitude,
        binding.station_longitude,
    ):
        raise PhysicalWeatherContractError("GFS coordinates are not the official settlement-station coordinates")
    parsed_metadata_url = urlparse(provenance.station_metadata_source_url)
    if parsed_metadata_url.netloc != "aviationweather.gov" or parsed_metadata_url.path != "/api/data/stationinfo":
        raise PhysicalWeatherContractError("station metadata is not from AviationWeather station info")
    _require_sha256(provenance.station_metadata_sha256, "station_metadata_sha256")
    if provenance.station_metadata_sha256 != STATION_INFO_RAW_SHA256:
        raise PhysicalWeatherContractError("station metadata payload is not the frozen official response")
    _require_sha256(provenance.index_sha256, "index_sha256")
    _require_sha256(provenance.raw_range_sha256, "raw_range_sha256")
    decision = _require_utc(provenance.decision_time_utc, "decision_time_utc")
    run = _require_utc(provenance.run_time_utc, "run_time_utc")
    if run != gfs_t_minus_six_cycle(decision):
        raise PhysicalWeatherContractError("GFS run is not the frozen t-6h cycle")
    expected_prefix = f"gfs.{run:%Y%m%d}/{run:%H}/atmos/gfs.t{run:%H}z.pgrb2.0p25.f"
    if not provenance.object_key.startswith(expected_prefix):
        raise PhysicalWeatherContractError("GFS object key conflicts with run identity")
    expected_suffix = f"{provenance.forecast_hour:03d}"
    if not provenance.object_key.endswith(expected_suffix) or provenance.index_key != provenance.object_key + ".idx":
        raise PhysicalWeatherContractError("GFS object/index key conflicts with forecast hour")
    if provenance.object_size_bytes <= 0 or not provenance.object_etag:
        raise PhysicalWeatherContractError("GFS object identity is incomplete")
    if (
        provenance.byte_range_start < 0
        or provenance.byte_range_end_inclusive < provenance.byte_range_start
        or provenance.byte_range_end_inclusive >= provenance.object_size_bytes
    ):
        raise PhysicalWeatherContractError("GFS byte range is invalid")
    valid = _require_utc(provenance.valid_time_utc, "valid_time_utc")
    if provenance.forecast_hour < 0 or valid != run + timedelta(hours=provenance.forecast_hour):
        raise PhysicalWeatherContractError("GFS valid time conflicts with run/forecast hour")
    if provenance.variable_descriptor != _expected_gfs_variable_descriptor(provenance.forecast_hour):
        raise PhysicalWeatherContractError("GFS variable descriptor is not the frozen 2 m TMP field")
    if provenance.decoder_name != GFS_DECODER or not provenance.decoder_version:
        raise PhysicalWeatherContractError("GRIB decoder identity is not frozen")
    _require_sha256(provenance.decoder_binary_sha256, "decoder_binary_sha256")
    _require_sha256(provenance.decoded_message_sha256, "decoded_message_sha256")
    nodes = provenance.grid_nodes
    if len(nodes) != 4 or len({(row.latitude, row.longitude) for row in nodes}) != 4:
        raise PhysicalWeatherContractError("GFS interpolation requires four unique grid nodes")
    for node in nodes:
        if (
            abs(node.latitude / GFS_GRID_DEGREES - round(node.latitude / GFS_GRID_DEGREES)) > 1e-9
            or abs(node.longitude / GFS_GRID_DEGREES - round(node.longitude / GFS_GRID_DEGREES)) > 1e-9
            or node.interpolation_weight < 0
            or not math.isfinite(node.temperature_k)
        ):
            raise PhysicalWeatherContractError("GFS grid-node provenance is invalid")
    if abs(sum(row.interpolation_weight for row in nodes) - 1.0) > 1e-9:
        raise PhysicalWeatherContractError("GFS bilinear weights do not sum to one")
    if not (
        min(row.latitude for row in nodes) <= binding.station_latitude <= max(row.latitude for row in nodes)
        and min(row.longitude for row in nodes) <= binding.station_longitude <= max(row.longitude for row in nodes)
    ):
        raise PhysicalWeatherContractError("GFS grid nodes do not bound the settlement station")
    interpolated = sum(row.interpolation_weight * row.temperature_k for row in nodes)
    if not math.isfinite(provenance.decoded_temperature_k) or abs(interpolated - provenance.decoded_temperature_k) > 1e-9:
        raise PhysicalWeatherContractError("decoded GFS temperature conflicts with bilinear nodes")
    times = (
        _require_utc(provenance.object_last_modified_utc, "object_last_modified_utc"),
        _require_utc(provenance.index_last_modified_utc, "index_last_modified_utc"),
        _require_utc(provenance.fetched_at_utc, "fetched_at_utc"),
        _require_utc(provenance.feature_assembled_at_utc, "feature_assembled_at_utc"),
        _require_utc(provenance.quote_time_utc, "quote_time_utc"),
        decision,
    )
    object_modified, index_modified, fetched, assembled, quote, _ = times
    if object_modified > fetched or index_modified > fetched or not (fetched <= assembled <= quote <= decision):
        raise PhysicalWeatherContractError("GFS input timing is not causal")


def validate_gfs_local_day_inputs(
    inputs: Sequence[GfsInputProvenance],
    *,
    binding: SettlementBinding,
    target_date: date,
) -> float:
    """Validate the full hourly station-local day and return its raw GFS max K.

    The v1 construction uses bilinear 2 m instantaneous temperature, no lapse-
    rate adjustment, and no mixed-cycle or missing-hour fallback.
    """

    if not inputs:
        raise PhysicalWeatherContractError("GFS local-day input set is empty")
    run_times = {_require_utc(row.run_time_utc, "run_time_utc") for row in inputs}
    decision_times = {_require_utc(row.decision_time_utc, "decision_time_utc") for row in inputs}
    if len(run_times) != 1 or len(decision_times) != 1:
        raise PhysicalWeatherContractError("GFS local day mixes model cycles or decisions")
    run = next(iter(run_times))
    expected_hours = required_gfs_forecast_hours(
        run_time_utc=run,
        target_date=target_date,
        timezone_name=binding.timezone_name,
    )
    actual_hours = tuple(sorted(row.forecast_hour for row in inputs))
    if actual_hours != expected_hours:
        raise PhysicalWeatherContractError("GFS local day is missing or duplicates an hourly input")
    for row in inputs:
        validate_gfs_input(row, binding)
    return max(row.decoded_temperature_k for row in inputs)


def validate_gfs_range_probe(payload: Mapping[str, Any]) -> None:
    """Validate a bounded direct-NODD object/index/range feasibility proof."""

    if payload.get("bucket") != GFS_BUCKET or payload.get("model") != GFS_MODEL:
        raise PhysicalWeatherContractError("GFS probe source identity changed")
    observed = _require_utc(datetime.fromisoformat(str(payload["observed_at_utc"]).replace("Z", "+00:00")), "observed_at_utc")
    run = _require_utc(datetime.fromisoformat(str(payload["run_time_utc"]).replace("Z", "+00:00")), "run_time_utc")
    if run != gfs_t_minus_six_cycle(observed):
        raise PhysicalWeatherContractError("GFS probe is not the t-6h cycle")
    for field in ("index_sha256", "tmp_raw_range_sha256", "tmax_raw_range_sha256"):
        _require_sha256(str(payload.get(field) or ""), field)
    if int(payload.get("range_http_status", 0)) != 206:
        raise PhysicalWeatherContractError("GFS server did not honor the byte range")
    for prefix in ("tmp", "tmax"):
        start = int(payload[f"{prefix}_range_start"])
        end = int(payload[f"{prefix}_range_end_inclusive"])
        length = int(payload[f"{prefix}_range_content_length"])
        if end - start + 1 != length:
            raise PhysicalWeatherContractError(f"{prefix} GFS byte range length conflicts")
        if end >= int(payload["object_content_length"]):
            raise PhysicalWeatherContractError(f"{prefix} GFS byte range exceeds the object")
    if str(payload.get("tmp_index_line") or "").split(":", 2)[-1] != "d=2026081700:TMP:2 m above ground:24 hour fcst:":
        raise PhysicalWeatherContractError("GFS probe TMP descriptor changed")
    for field in ("object_last_modified_utc", "index_last_modified_utc"):
        modified = _require_utc(datetime.fromisoformat(str(payload[field]).replace("Z", "+00:00")), field)
        if modified > observed:
            raise PhysicalWeatherContractError("GFS probe observed an unavailable object")


def validate_physical_target(
    target: PhysicalTargetProvenance,
    *,
    binding: SettlementBinding,
    model_fit_cutoff_utc: datetime,
) -> None:
    if target.station_id != binding.station_id or target.timezone_name != binding.timezone_name:
        raise PhysicalWeatherContractError("physical target is not settlement-station local")
    if target.source_dataset != PHYSICAL_TARGET_SOURCE:
        raise PhysicalWeatherContractError("physical target must come from NOAA ISD")
    _require_sha256(target.raw_payload_sha256, "raw_payload_sha256")
    available = _require_utc(target.first_available_at_utc, "first_available_at_utc")
    computed = _require_utc(target.computed_at_utc, "computed_at_utc")
    cutoff = _require_utc(model_fit_cutoff_utc, "model_fit_cutoff_utc")
    if not (available <= computed <= cutoff):
        raise PhysicalWeatherContractError("physical target was not available by model fit")


def validate_scoring_label(label: CanonicalMarketLabel, *, evaluation_cutoff_utc: datetime) -> None:
    """Only canonical Polymarket payouts may be used as evaluation truth."""

    if not isinstance(label, CanonicalMarketLabel):
        raise PhysicalWeatherContractError("physical observations are not canonical scoring labels")
    if label.source != CANONICAL_LABEL_SOURCE or label.yes_outcome not in {0, 1}:
        raise PhysicalWeatherContractError("scoring label is not a canonical binary payout")
    _require_sha256(label.raw_payload_sha256, "raw_payload_sha256")
    if _require_utc(label.label_available_at_utc, "label_available_at_utc") > _require_utc(evaluation_cutoff_utc, "evaluation_cutoff_utc"):
        raise PhysicalWeatherContractError("canonical label was unavailable at evaluation cutoff")
