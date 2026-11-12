from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import re
import statistics
import time
import urllib.parse
import urllib.request
from collections import Counter
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Optional, Sequence
from zoneinfo import ZoneInfo

from .backtesting_engine import (
    FEE_MODEL_POLYMARKET_WEATHER_TAKER,
    OLD_NO_TAIL_LIVE_FORWARD_SHADOW_PREREGISTRATION,
    OLD_NO_TAIL_LIVE_FORWARD_SHADOW_PROFILE_KEY,
    DecisionSnapshot,
    FeeModelConfig,
    ForecastSnapshot,
    ObservedHighSnapshot,
    QuoteSnapshot,
    ReplayFrame,
    ReplayGeneratedScoreCache,
    SharedScoreContract,
    WeatherModelRunner,
    calculate_fill_fee_usd,
    run_old_no_tail_live_forward_shadow_logging,
)
from .live_forward_shadow_scoring import (
    LiveForwardShadowScoringAdapter,
    default_live_forward_shadow_scoring_adapter,
    shadow_scoring_parity_report,
)
from .station_observations import (
    StationConfig,
    _get_or_fetch_raw_payload,
    _iem_request_url,
    parse_iem_csv_payload,
)
from .venue_execution import VenueExecutionConstraints, assess_venue_order


SCHEMA_VERSION = "dailyweather.live_forward_shadow.intraday_recorder.v1"
SETTLEMENT_REFRESH_SCHEMA_VERSION = "dailyweather.live_forward_shadow.settlement_refresh.v1"
USER_AGENT = "DailyWeather-LiveForwardShadowRecorder/0.1"
GAMMA_BASE_URL = "https://gamma-api.polymarket.com"
CLOB_BASE_URL = "https://clob.polymarket.com"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_GFS_URL = "https://api.open-meteo.com/v1/gfs"
OPEN_METEO_ECMWF_URL = "https://api.open-meteo.com/v1/ecmwf"
OPEN_METEO_ENSEMBLE_URL = "https://ensemble-api.open-meteo.com/v1/ensemble"
DEFAULT_OUTPUT_DIR = Path("work/logs/live_forward_shadow_intraday")
DEFAULT_REPORT_DIR = Path("work/reports/live_forward_shadow_intraday")
HTTP_TIMEOUT_SECONDS = 12.0
BOOK_FETCH_WORKERS = 8
FORWARD_SHADOW_GATE_MIN_RESOLVED_CLUSTERS = 30
FORWARD_SHADOW_PRODUCTION_MIN_RESOLVED_CLUSTERS = 100
FORWARD_SHADOW_PRODUCTION_MIN_VALID_DAYS = 30
FORWARD_ECONOMIC_BOOTSTRAP_CONFIDENCE = 0.95
FORWARD_ECONOMIC_BOOTSTRAP_RESAMPLES = 10_000
FORWARD_ECONOMIC_BOOTSTRAP_SEED = 20_260_710
DEFAULT_RESOLVED_RECHECK_WINDOW_HOURS = 48.0
WILSON_Z_95 = 1.959963984540054
WILSON_Z_80 = 1.2815515655446004


@dataclass(frozen=True)
class CityConfig:
    name: str
    region: str
    latitude: float
    longitude: float
    timezone: str
    metar_station: Optional[str] = None
    aliases: tuple[str, ...] = ()

    @property
    def display_name(self) -> str:
        return f"{self.name}, {self.region}"


@dataclass(frozen=True)
class Bucket:
    label: str
    lower_f: Optional[float]
    upper_f: Optional[float]
    yes_token_id: Optional[str]
    no_token_id: Optional[str]
    yes_price: Optional[float] = None
    no_price: Optional[float] = None
    resolution_unit: Optional[str] = None
    resolution_precision: Optional[float] = None
    source_bucket_type: Optional[str] = None

    @property
    def bucket_type(self) -> str:
        if self.source_bucket_type:
            return self.source_bucket_type
        if self.lower_f is not None and self.upper_f is not None:
            width = self.upper_f - self.lower_f
            if 0 <= width <= 1.25:
                return "exact"
            return "range"
        if self.lower_f is not None:
            return "above"
        if self.upper_f is not None:
            return "below"
        return "unknown"


@dataclass(frozen=True)
class WeatherMarket:
    market_id: str
    condition_id: str
    slug: str
    event_slug: str
    question: str
    close_time: Optional[datetime]
    target_date: date
    city: CityConfig
    bucket: Bucket
    raw: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _TemperatureBucketParse:
    lower_f: Optional[float]
    upper_f: Optional[float]
    label: str
    source_unit: Optional[str]
    source_precision: Optional[float]
    source_bucket_type: str


@dataclass(frozen=True)
class BookSnapshot:
    token_id: str
    fetched_at_utc: datetime
    payload_id: str
    raw_cache_path: str
    best_bid: Optional[float]
    best_ask: Optional[float]
    bid_size: Optional[float]
    ask_size: Optional[float]
    mid: Optional[float]
    spread: Optional[float]
    top_bids: tuple[Mapping[str, float], ...]
    top_asks: tuple[Mapping[str, float], ...]
    bid_levels: tuple[Mapping[str, float], ...] = ()
    ask_levels: tuple[Mapping[str, float], ...] = ()
    full_book_complete: Optional[bool] = None
    min_order_size_shares: Optional[float] = None
    tick_size: Optional[float] = None
    neg_risk: Optional[bool] = None
    fee_rate: Optional[float] = None
    fee_rate_source: Optional[str] = None
    raw_status: str = "ok"
    error: Optional[str] = None
    received_at_utc: Optional[datetime] = None
    venue_timestamp_raw: Optional[str] = None
    source_sha256: Optional[str] = None
    market_condition_id: Optional[str] = None

    @property
    def visible_top_depth(self) -> Optional[float]:
        if self.bid_size is None and self.ask_size is None:
            return None
        return float(self.bid_size or 0.0) + float(self.ask_size or 0.0)


@dataclass(frozen=True)
class ForecastSource:
    source_name: str
    fetched_at_utc: datetime
    temperature_2m_max_f: Optional[float]
    samples_f: tuple[float, ...]
    features: Mapping[str, Any]
    raw_cache_path: str
    payload_id: str
    request_url: str
    status: str = "ok"
    error: Optional[str] = None


@dataclass(frozen=True)
class StationObservationSummary:
    station_id: Optional[str]
    status: str
    fetched_at_utc: datetime
    local_date: Optional[str] = None
    latest_observed_at_utc: Optional[datetime] = None
    latest_temperature_f: Optional[float] = None
    high_so_far_f: Optional[float] = None
    high_so_far_observed_at_utc: Optional[datetime] = None
    observation_count: int = 0
    raw_cache_path: Optional[str] = None
    payload_id: Optional[str] = None
    source_family: str = "IEM_ASOS_METAR"
    error: Optional[str] = None


@dataclass(frozen=True)
class LiveShadowArtifacts:
    json_path: Path
    markdown_path: Path
    events_path: Path
    raw_dir: Path
    summary: Mapping[str, Any]


@dataclass(frozen=True)
class FrozenS01MarketDiscovery:
    """Exhaustive, deterministic venue discovery result for frozen standalone S01."""

    markets: tuple[WeatherMarket, ...]
    diagnostics: Mapping[str, Any]

    @property
    def gate_valid(self) -> bool:
        return self.diagnostics.get("gate_valid") is True


@dataclass(frozen=True)
class FrozenS01InputTiming:
    """Pure timing proof for the model-input -> venue-quote -> decision boundary."""

    gate_valid: bool
    feature_time_utc: Optional[datetime]
    earliest_quote_time_utc: Optional[datetime]
    latest_quote_time_utc: Optional[datetime]
    decision_time_utc: datetime
    reasons: tuple[str, ...]


class _ListFrameDataset:
    def __init__(self, frames: Sequence[ReplayFrame]) -> None:
        self._frames = list(frames)

    def iter_frames(self, limit: int = 0) -> Iterator[ReplayFrame]:
        count = 0
        for frame in self._frames:
            if limit and count >= limit:
                return
            count += 1
            yield frame


DEFAULT_CITIES: tuple[CityConfig, ...] = (
    CityConfig("New York", "NY", 40.7128, -74.0060, "America/New_York", "KNYC", ("NYC", "New York City")),
    CityConfig("Los Angeles", "CA", 34.0522, -118.2437, "America/Los_Angeles", "KCQT", ("LA",)),
    CityConfig("Chicago", "IL", 41.8781, -87.6298, "America/Chicago", "KMDW"),
    CityConfig("Miami", "FL", 25.7617, -80.1918, "America/New_York", "KMIA"),
    CityConfig("Austin", "TX", 30.2672, -97.7431, "America/Chicago", "KAUS"),
    CityConfig("Dallas", "TX", 32.7767, -96.7970, "America/Chicago", "KDAL"),
    CityConfig("Houston", "TX", 29.7604, -95.3698, "America/Chicago", "KHOU"),
    CityConfig("Phoenix", "AZ", 33.4484, -112.0740, "America/Phoenix", "KPHX"),
    CityConfig("Philadelphia", "PA", 39.9526, -75.1652, "America/New_York", "KPHL"),
    CityConfig("Washington", "DC", 38.9072, -77.0369, "America/New_York", "KDCA", ("DC", "Washington DC", "D.C.")),
    CityConfig("Boston", "MA", 42.3601, -71.0589, "America/New_York", "KBOS"),
    CityConfig("San Francisco", "CA", 37.7749, -122.4194, "America/Los_Angeles", "KSFO", ("SF",)),
    CityConfig("Seattle", "WA", 47.6062, -122.3321, "America/Los_Angeles", "KSEA"),
    CityConfig("Denver", "CO", 39.7392, -104.9903, "America/Denver", "KDEN"),
    CityConfig("Las Vegas", "NV", 36.1699, -115.1398, "America/Los_Angeles", "KLAS", ("Vegas",)),
    CityConfig("Seoul", "KR", 37.5665, 126.9780, "Asia/Seoul", "RKSS"),
    CityConfig("Tokyo", "JP", 35.6762, 139.6503, "Asia/Tokyo", "RJTT"),
    CityConfig("Beijing", "CN", 39.9042, 116.4074, "Asia/Shanghai", "ZBAA"),
    CityConfig("Shanghai", "CN", 31.2304, 121.4737, "Asia/Shanghai", "ZSSS"),
    CityConfig("Guangzhou", "CN", 23.1291, 113.2644, "Asia/Shanghai", "ZGGG"),
    CityConfig("Shenzhen", "CN", 22.5431, 114.0579, "Asia/Shanghai", "ZGSZ"),
    CityConfig("Taipei", "TW", 25.0330, 121.5654, "Asia/Taipei", "RCSS"),
    CityConfig("Singapore", "SG", 1.3521, 103.8198, "Asia/Singapore", "WSSS"),
    CityConfig("Moscow", "RU", 55.7558, 37.6173, "Europe/Moscow", "UUEE"),
    CityConfig("Paris", "FR", 48.8566, 2.3522, "Europe/Paris", "LFPG"),
    CityConfig("London", "GB", 51.5072, -0.1276, "Europe/London", "EGLL"),
    CityConfig("Berlin", "DE", 52.52, 13.405, "Europe/Berlin", "EDDB"),
    CityConfig("Rome", "IT", 41.9028, 12.4964, "Europe/Rome", "LIRF"),
    CityConfig("Madrid", "ES", 40.4168, -3.7038, "Europe/Madrid", "LEMD"),
    CityConfig("Sydney", "AU", -33.8688, 151.2093, "Australia/Sydney", "YSSY"),
)


FROZEN_CITY_REGISTRY_SHA256 = (
    "48c59c254690bc4d31e5310d2dfd48a996e3bb312017d11d1f15c6b4cdc7670b"
)

FROZEN_S01_CITY_DISPLAY_NAMES: tuple[str, ...] = (
    "Austin, TX",
    "Beijing, CN",
    "Chicago, IL",
    "Dallas, TX",
    "Denver, CO",
    "Guangzhou, CN",
    "Houston, TX",
    "London, GB",
    "Los Angeles, CA",
    "Madrid, ES",
    "Miami, FL",
    "Moscow, RU",
    "New York, NY",
    "Paris, FR",
    "San Francisco, CA",
    "Seattle, WA",
    "Seoul, KR",
    "Shanghai, CN",
    "Shenzhen, CN",
    "Singapore, SG",
    "Taipei, TW",
    "Tokyo, JP",
)
_FROZEN_S01_CITY_DISPLAY_NAME_SET = frozenset(FROZEN_S01_CITY_DISPLAY_NAMES)
FROZEN_S01_CITIES: tuple[CityConfig, ...] = (
    CityConfig("Austin", "TX", 30.2672, -97.7431, "America/Chicago", "KAUS"),
    CityConfig("Beijing", "CN", 39.9042, 116.4074, "Asia/Shanghai", "ZBAA"),
    CityConfig("Chicago", "IL", 41.8781, -87.6298, "America/Chicago", "KMDW"),
    CityConfig("Dallas", "TX", 32.7767, -96.7970, "America/Chicago", "KDAL"),
    CityConfig("Denver", "CO", 39.7392, -104.9903, "America/Denver", "KDEN"),
    CityConfig("Guangzhou", "CN", 23.1291, 113.2644, "Asia/Shanghai", "ZGGG"),
    CityConfig("Houston", "TX", 29.7604, -95.3698, "America/Chicago", "KHOU"),
    CityConfig("London", "GB", 51.5072, -0.1276, "Europe/London", "EGLL"),
    CityConfig("Los Angeles", "CA", 34.0522, -118.2437, "America/Los_Angeles", "KCQT", ("LA",)),
    CityConfig("Madrid", "ES", 40.4168, -3.7038, "Europe/Madrid", "LEMD"),
    CityConfig("Miami", "FL", 25.7617, -80.1918, "America/New_York", "KMIA"),
    CityConfig("Moscow", "RU", 55.7558, 37.6173, "Europe/Moscow", "UUEE"),
    CityConfig("New York", "NY", 40.7128, -74.0060, "America/New_York", "KNYC", ("NYC", "New York City")),
    CityConfig("Paris", "FR", 48.8566, 2.3522, "Europe/Paris", "LFPG"),
    CityConfig("San Francisco", "CA", 37.7749, -122.4194, "America/Los_Angeles", "KSFO", ("SF",)),
    CityConfig("Seattle", "WA", 47.6062, -122.3321, "America/Los_Angeles", "KSEA"),
    CityConfig("Seoul", "KR", 37.5665, 126.9780, "Asia/Seoul", "RKSS"),
    CityConfig("Shanghai", "CN", 31.2304, 121.4737, "Asia/Shanghai", "ZSSS"),
    CityConfig("Shenzhen", "CN", 22.5431, 114.0579, "Asia/Shanghai", "ZGSZ"),
    CityConfig("Singapore", "SG", 1.3521, 103.8198, "Asia/Singapore", "WSSS"),
    CityConfig("Taipei", "TW", 25.0330, 121.5654, "Asia/Taipei", "RCSS"),
    CityConfig("Tokyo", "JP", 35.6762, 139.6503, "Asia/Tokyo", "RJTT"),
)
_FROZEN_S01_CITY_REGISTRY_ENTRIES = [
    [
        city.name,
        city.region,
        city.latitude,
        city.longitude,
        city.timezone,
        city.metar_station,
        list(city.aliases),
    ]
    for city in FROZEN_S01_CITIES
]
if (
    len(FROZEN_S01_CITIES) != 22
    or hashlib.sha256(
        json.dumps(
            _FROZEN_S01_CITY_REGISTRY_ENTRIES,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    != FROZEN_CITY_REGISTRY_SHA256
):  # pragma: no cover - import-time contract assertion
    raise RuntimeError("Frozen S01 city configuration must resolve to exactly 22 cities")

FROZEN_S01_LIVE_FORECAST_SOURCE = "open_meteo_best_match"


MONTHS = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}
_MONTH_PATTERN = "|".join(sorted(MONTHS, key=len, reverse=True))
_DATE_RE = re.compile(
    rf"\b(?P<month>{_MONTH_PATTERN})\.?(?:\s+|-)(?P<day>\d{{1,2}})(?:st|nd|rd|th)?(?:[,\s-]+(?P<year>\d{{4}}))?\b",
    re.I,
)


def record_live_forward_shadow(
    *,
    work_root: Path,
    output_dir: Optional[Path] = None,
    report_dir: Optional[Path] = None,
    iterations: int = 1,
    interval_seconds: int = 60,
    discovery_pages: int = 2,
    discovery_request_limit: int = 50,
    market_limit: int = 100,
    starting_cash_usd: float = 100.0,
    cache_only: bool = False,
    model_runner: Optional[WeatherModelRunner] = None,
    resolved_recheck_window_hours: float = DEFAULT_RESOLVED_RECHECK_WINDOW_HOURS,
) -> LiveShadowArtifacts:
    root = work_root.resolve()
    out_dir = (output_dir or root / DEFAULT_OUTPUT_DIR).resolve()
    reports = (report_dir or root / DEFAULT_REPORT_DIR).resolve()
    raw_dir = out_dir / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    requested_iterations = iterations
    if iterations <= 0:
        iterations = 1

    started_at = datetime.now(timezone.utc)
    run_id = started_at.strftime("%Y%m%dT%H%M%SZ")
    frames: list[ReplayFrame] = []
    live_snapshots: list[dict[str, Any]] = []
    poll_summaries: list[dict[str, Any]] = []
    discovery_errors: list[str] = []
    score_cache = ReplayGeneratedScoreCache()
    scoring_adapter = (
        LiveForwardShadowScoringAdapter(model_runner=model_runner, score_cache=score_cache)
        if model_runner is not None
        else default_live_forward_shadow_scoring_adapter(work_root=root, score_cache=score_cache)
    )
    scoring_diagnostics = {
        "firewall_violation_count": 0,
        "missing_model_output_count": 0,
    }

    for index in range(iterations):
        poll_time = datetime.now(timezone.utc)
        markets: list[WeatherMarket] = []
        try:
            markets = discover_active_weather_markets(
                pages=discovery_pages,
                request_limit=discovery_request_limit,
                market_limit=market_limit,
                as_of=poll_time,
            )
        except Exception as exc:
            discovery_errors.append(f"{type(exc).__name__}: {str(exc)[:300]}")
        poll_frames, poll_snapshots = collect_poll_frames(
            markets,
            collected_at_utc=poll_time,
            raw_dir=raw_dir / run_id / f"poll-{index + 1:03d}",
            cache_only=cache_only,
            scoring_adapter=scoring_adapter,
            scoring_diagnostics=scoring_diagnostics,
        )
        frames.extend(poll_frames)
        live_snapshots.extend(poll_snapshots)
        poll_summaries.append(
            {
                "poll_index": index + 1,
                "collected_at_utc": poll_time.isoformat(),
                "active_weather_markets": len(markets),
                "shadow_decision_frames": len(poll_frames),
                "live_snapshots": len(poll_snapshots),
            }
        )
        if index < iterations - 1:
            time.sleep(max(0, interval_seconds))

    shadow_payload = run_old_no_tail_live_forward_shadow_logging(
        _ListFrameDataset(frames),
        starting_cash_usd=starting_cash_usd,
        limit=len(frames),
    )
    events = _enrich_shadow_events(
        shadow_payload.pop("shadow_events", []),
        live_snapshots,
    )
    score_cache_path = reports / f"{run_id}-old-no-tail-live-forward-shadow-shared-scores.jsonl"
    score_cache.write_jsonl(score_cache_path)
    score_cache_summary = score_cache.report_summary(artifact_path=score_cache_path)
    runner_metadata = scoring_adapter.runner_metadata()
    shadow_scoring_parity = shadow_scoring_parity_report(
        events=list(events),
        score_cache_summary=score_cache_summary,
        runner_metadata=runner_metadata,
        firewall_violation_count=int(scoring_diagnostics.get("firewall_violation_count") or 0),
        missing_model_output_count=int(scoring_diagnostics.get("missing_model_output_count") or 0),
    )
    summary = _summary_payload(
        run_id=run_id,
        started_at=started_at,
        completed_at=datetime.now(timezone.utc),
        requested_iterations=requested_iterations,
        effective_iterations=iterations,
        interval_seconds=interval_seconds,
        discovery_errors=discovery_errors,
        poll_summaries=poll_summaries,
        live_snapshots=live_snapshots,
        shadow_payload=shadow_payload,
        events=events,
        raw_dir=raw_dir / run_id,
        score_cache_summary=score_cache_summary,
        shadow_scoring_parity=shadow_scoring_parity,
    )
    settlement_refresh = refresh_pending_would_trade_settlements(
        report_dir=reports,
        raw_dir=raw_dir / run_id / "settlement_refresh",
        run_id=run_id,
        cache_only=cache_only,
        resolved_recheck_window_hours=resolved_recheck_window_hours,
    )
    events_path = reports / f"{run_id}-old-no-tail-live-forward-shadow-intraday-events.jsonl"
    json_path = reports / f"{run_id}-old-no-tail-live-forward-shadow-intraday.json"
    markdown_path = reports / f"{run_id}-old-no-tail-live-forward-shadow-intraday.md"
    with events_path.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(_json_ready(event), sort_keys=True) + "\n")
    summary = {
        **summary,
        "artifacts": {
            "summary_json": str(json_path),
            "daily_markdown_report": str(markdown_path),
            "events_jsonl": str(events_path),
            "shared_scores_jsonl": str(score_cache_path),
            "raw_payload_dir": str(raw_dir / run_id),
            "settlement_refresh_json": str((settlement_refresh.get("artifacts") or {}).get("summary_json")),
            "settlement_refresh_markdown": str((settlement_refresh.get("artifacts") or {}).get("markdown_report")),
        },
        "settlement_refresh": settlement_refresh,
    }
    summary = {
        **summary,
        "forward_collection_progress": _forward_collection_progress(
            reports,
            current_summary=summary,
            current_settlement_refresh=settlement_refresh,
        ),
    }
    json_path.write_text(json.dumps(_json_ready(summary), indent=2, sort_keys=True), encoding="utf-8")
    markdown_path.write_text(_render_markdown(summary), encoding="utf-8")
    return LiveShadowArtifacts(
        json_path=json_path,
        markdown_path=markdown_path,
        events_path=events_path,
        raw_dir=raw_dir / run_id,
        summary=summary,
    )


def discover_active_weather_markets(
    *,
    pages: int,
    request_limit: int,
    market_limit: int,
    as_of: datetime,
) -> list[WeatherMarket]:
    request_limit = max(1, min(100, int(request_limit)))
    raw_markets: list[Mapping[str, Any]] = []
    for page in range(max(1, pages)):
        payload = _get_json(
            f"{GAMMA_BASE_URL}/events",
            {
                "active": "true",
                "closed": "false",
                "limit": request_limit,
                "offset": page * request_limit,
                "order": "volume24hr",
                "ascending": "false",
            },
        )
        raw_markets.extend(_iter_event_markets(payload if isinstance(payload, list) else payload.get("events", [])))
    for query in _temperature_search_queries(as_of.date()):
        payload = _get_json(
            f"{GAMMA_BASE_URL}/public-search",
            {
                "q": query,
                "limit_per_type": min(50, request_limit),
                "page": 1,
                "search_profiles": "false",
                "search_tags": "false",
            },
        )
        if isinstance(payload, Mapping):
            raw_markets.extend(_iter_event_markets(payload.get("events") or []))

    parsed: list[WeatherMarket] = []
    seen: set[str] = set()
    for raw in raw_markets:
        if raw.get("closed") is True:
            continue
        if not looks_like_temperature_market(raw):
            continue
        market = parse_weather_market(raw, today=as_of.date())
        if market is None or market.bucket.no_token_id is None:
            continue
        key = market.market_id or market.slug or market.question
        if key in seen:
            continue
        seen.add(key)
        parsed.append(market)
    ordered = sort_markets_for_old_no_tail_priority(parsed, as_of=as_of)
    return ordered[:market_limit] if market_limit else ordered


def discover_frozen_s01_active_weather_markets(
    *,
    request_limit: int,
    as_of: datetime,
    max_pages: int = 100,
) -> FrozenS01MarketDiscovery:
    """Fetch every active Gamma keyset page and retain the frozen 22-city universe.

    This path deliberately has no market-count truncation. A request failure,
    malformed or repeated cursor/page, pagination guard exhaustion, malformed
    tested-city temperature contract, missing YES/NO token, or empty parsed
    universe makes the discovery gate invalid.
    """

    requested_limit = int(request_limit)
    # Gamma currently applies an effective 100-event keyset page even when a
    # larger documented limit is requested.  A page size is not a universe
    # cap: every cursor is followed to the documented short final page.
    request_limit = 100
    max_pages = max(1, int(max_pages))
    raw_markets: list[Mapping[str, Any]] = []
    page_signatures: set[str] = set()
    seen_cursors: set[str] = set()
    errors: list[str] = []
    pages_fetched = 0
    pagination_complete = False
    raw_event_count = 0
    after_cursor: Optional[str] = None

    for page in range(max_pages):
        params: dict[str, Any] = {
            "active": "true",
            "closed": "false",
            "limit": request_limit,
            "tag_slug": "weather",
        }
        if after_cursor is not None:
            params["after_cursor"] = after_cursor
        try:
            payload = _get_json(
                f"{GAMMA_BASE_URL}/events/keyset",
                params,
            )
        except Exception as exc:
            errors.append(f"page_{page}:{type(exc).__name__}:{str(exc)[:200]}")
            break
        if not isinstance(payload, Mapping):
            errors.append(f"page_{page}:invalid_keyset_payload")
            break
        # The production keyset schema names the page ``events``.  Accept the
        # older ``data`` spelling only as a compatibility fallback, never as
        # an excuse to treat a malformed page as complete.
        events = payload.get("events")
        if events is None:
            events = payload.get("data")
        if not isinstance(events, list):
            errors.append(f"page_{page}:invalid_events_payload")
            break
        pages_fetched += 1
        raw_event_count += len(events)
        signature = _canonical_payload_sha256(events)
        if events and signature in page_signatures:
            errors.append(f"page_{page}:repeated_page")
            break
        page_signatures.add(signature)
        raw_markets.extend(_iter_event_markets(events))
        if "next_cursor" not in payload:
            if len(events) < request_limit:
                pagination_complete = True
            else:
                errors.append(f"page_{page}:missing_next_cursor")
            break
        next_cursor = payload.get("next_cursor")
        if next_cursor is None:
            pagination_complete = True
            break
        if not isinstance(next_cursor, str) or not next_cursor.strip():
            errors.append(f"page_{page}:malformed_next_cursor")
            break
        next_cursor = next_cursor.strip()
        if next_cursor == after_cursor or next_cursor in seen_cursors:
            errors.append(f"page_{page}:repeated_next_cursor")
            break
        if not events:
            errors.append(f"page_{page}:empty_page_with_next_cursor")
            break
        seen_cursors.add(next_cursor)
        after_cursor = next_cursor
    else:
        errors.append("pagination_guard_exhausted")

    parsed: list[WeatherMarket] = []
    seen: set[str] = set()
    duplicate_count = 0
    excluded_city_market_count = 0
    non_executable_market_count = 0
    ambiguous_execution_status_count = 0
    malformed_tested_city_market_count = 0
    missing_token_pair_count = 0
    tested_city_temperature_candidate_count = 0
    for raw in raw_markets:
        if not looks_like_temperature_market(raw):
            continue
        raw_text = " ".join(
            str(raw.get(key) or "")
            for key in (
                "question",
                "title",
                "description",
                "slug",
                "eventSlug",
                "eventTitle",
                "eventDescription",
            )
        )
        city = find_city(raw_text)
        if city is None:
            continue
        if city.display_name not in _FROZEN_S01_CITY_DISPLAY_NAME_SET:
            excluded_city_market_count += 1
            continue
        tested_city_temperature_candidate_count += 1
        active = raw.get("active")
        closed = raw.get("closed")
        accepting_orders = raw.get("acceptingOrders")
        if accepting_orders is None:
            accepting_orders = raw.get("accepting_orders")
        if not all(
            isinstance(value, bool) for value in (active, closed, accepting_orders)
        ):
            ambiguous_execution_status_count += 1
            continue
        if active is not True or closed is not False or accepting_orders is not True:
            non_executable_market_count += 1
            continue
        market = parse_weather_market(raw, today=as_of.date())
        if market is None:
            malformed_tested_city_market_count += 1
            continue
        if not market.bucket.yes_token_id or not market.bucket.no_token_id:
            missing_token_pair_count += 1
            continue
        key = market.market_id or market.slug or market.question
        if key in seen:
            duplicate_count += 1
            continue
        seen.add(key)
        parsed.append(market)

    ordered = tuple(sort_markets_for_old_no_tail_priority(parsed, as_of=as_of))
    city_market_counts = {
        name: sum(1 for market in ordered if market.city.display_name == name)
        for name in FROZEN_S01_CITY_DISPLAY_NAMES
    }
    gate_reasons = list(errors)
    if not pagination_complete:
        gate_reasons.append("pagination_incomplete")
    if ambiguous_execution_status_count:
        gate_reasons.append("tested_city_market_execution_status_ambiguous")
    if malformed_tested_city_market_count:
        gate_reasons.append("malformed_tested_city_temperature_market")
    if missing_token_pair_count:
        gate_reasons.append("tested_city_market_missing_yes_no_token_pair")
    if not ordered:
        gate_reasons.append("no_frozen_s01_markets_discovered")
    diagnostics = {
        "schema_version": "dailyweather.s01_market_discovery.v1",
        "gate_valid": not gate_reasons,
        "gate_reasons": tuple(dict.fromkeys(gate_reasons)),
        "pagination_complete": pagination_complete,
        "pagination_mode": "gamma_events_keyset",
        "endpoint": f"{GAMMA_BASE_URL}/events/keyset",
        "server_filter_tag_slug": "weather",
        "pages_fetched": pages_fetched,
        "requested_limit": requested_limit,
        "request_limit": request_limit,
        "followed_cursor_count": len(seen_cursors),
        "raw_event_count": raw_event_count,
        "raw_market_count": len(raw_markets),
        "tested_city_temperature_candidate_count": tested_city_temperature_candidate_count,
        "parsed_market_count": len(ordered),
        "duplicate_market_count": duplicate_count,
        "excluded_city_market_count": excluded_city_market_count,
        "non_executable_market_count": non_executable_market_count,
        "ambiguous_execution_status_count": ambiguous_execution_status_count,
        "malformed_tested_city_market_count": malformed_tested_city_market_count,
        "missing_token_pair_count": missing_token_pair_count,
        "tested_city_count": len(FROZEN_S01_CITIES),
        "city_market_counts": city_market_counts,
        "cities_with_active_markets": tuple(
            name for name, count in city_market_counts.items() if count
        ),
        "cities_without_active_markets": tuple(
            name for name, count in city_market_counts.items() if not count
        ),
    }
    return FrozenS01MarketDiscovery(markets=ordered, diagnostics=diagnostics)


def sort_markets_for_old_no_tail_priority(
    markets: Sequence[WeatherMarket],
    *,
    as_of: datetime,
) -> list[WeatherMarket]:
    def sort_key(market: WeatherMarket) -> tuple[int, int, str, str]:
        lead_days = _market_lead_days(market, as_of)
        if lead_days in (1, 2):
            priority = 0
        elif lead_days is not None and lead_days > 2:
            priority = 1
        elif lead_days == 0:
            priority = 2
        else:
            priority = 3
        return (priority, abs((lead_days if lead_days is not None else 99) - 1), market.city.display_name, market.slug)

    return sorted(markets, key=sort_key)


def collect_poll_frames(
    markets: Sequence[WeatherMarket],
    *,
    collected_at_utc: datetime,
    raw_dir: Path,
    scoring_adapter: LiveForwardShadowScoringAdapter,
    scoring_diagnostics: dict[str, int],
    cache_only: bool = False,
) -> tuple[list[ReplayFrame], list[dict[str, Any]]]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    frames: list[ReplayFrame] = []
    snapshots: list[dict[str, Any]] = []
    forecast_cache: dict[tuple[str, str], tuple[list[ForecastSource], StationObservationSummary]] = {}
    book_cache = fetch_books_for_markets(
        markets,
        raw_dir=raw_dir / "clob_books",
        fetched_at=collected_at_utc,
        cache_only=cache_only,
    )
    for market in markets:
        key = (market.city.display_name, market.target_date.isoformat())
        if key not in forecast_cache:
            forecast_cache[key] = (
                fetch_open_meteo_forecasts(
                    market.city,
                    market.target_date,
                    raw_dir=raw_dir / "open_meteo",
                    fetched_at=collected_at_utc,
                    cache_only=cache_only,
                ),
                fetch_station_observations(
                    market.city,
                    market.target_date,
                    raw_dir=raw_dir / "station_observations",
                    fetched_at=collected_at_utc,
                    cache_only=cache_only,
                ),
            )
        forecasts, station_summary = forecast_cache[key]
        yes_book = book_cache.get(market.bucket.yes_token_id or "")
        no_book = book_cache.get(market.bucket.no_token_id or "")
        frame = build_replay_frame(
            market,
            no_book=no_book,
            yes_book=yes_book,
            forecasts=forecasts,
            station_summary=station_summary,
            collected_at_utc=collected_at_utc,
            scoring_adapter=scoring_adapter,
            scoring_diagnostics=scoring_diagnostics,
        )
        frames.append(frame)
        snapshots.append(
            {
                "decision_id": frame.decision.decision_id,
                "market_metadata": live_market_metadata(market),
                "parsed_contract": parsed_contract(market),
                "observed_books": {
                    "YES": book_to_payload(yes_book),
                    "NO": book_to_payload(no_book),
                },
                "assumed_executable_price": {
                    "side": "NO",
                    "basis": "observed_NO_best_ask_snapshot_not_a_fill_and_not_truth",
                    "price": no_book.best_ask if no_book else None,
                    "visible_size_at_price": no_book.ask_size if no_book else None,
                    "token_id": market.bucket.no_token_id,
                },
                "open_meteo_forecasts": [forecast_to_payload(item) for item in forecasts],
                "station_observations": station_summary_to_payload(station_summary),
            }
        )
    return frames, snapshots


def fetch_books_for_markets(
    markets: Sequence[WeatherMarket],
    *,
    raw_dir: Path,
    fetched_at: datetime,
    cache_only: bool,
) -> dict[str, Optional[BookSnapshot]]:
    tokens = sorted(
        {
            token
            for market in markets
            for token in (market.bucket.yes_token_id, market.bucket.no_token_id)
            if token
        }
    )
    if not tokens:
        return {}
    workers = max(1, min(BOOK_FETCH_WORKERS, len(tokens)))
    results: dict[str, Optional[BookSnapshot]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                fetch_order_book,
                token,
                raw_dir=raw_dir,
                fetched_at=fetched_at,
                cache_only=cache_only,
            ): token
            for token in tokens
        }
        for future in concurrent.futures.as_completed(futures):
            token = futures[future]
            try:
                results[token] = future.result()
            except Exception as exc:
                results[token] = BookSnapshot(
                    token_id=token,
                    fetched_at_utc=fetched_at,
                    payload_id="",
                    raw_cache_path=str(raw_dir / f"{_safe_name(token)}-{fetched_at.strftime('%Y%m%dT%H%M%SZ')}.json"),
                    best_bid=None,
                    best_ask=None,
                    bid_size=None,
                    ask_size=None,
                    mid=None,
                    spread=None,
                    top_bids=(),
                    top_asks=(),
                    raw_status="error",
                    error=f"{type(exc).__name__}: {str(exc)[:300]}",
                )
    return {token: results[token] for token in tokens}


def build_replay_frame(
    market: WeatherMarket,
    *,
    no_book: Optional[BookSnapshot],
    yes_book: Optional[BookSnapshot],
    forecasts: Sequence[ForecastSource],
    station_summary: StationObservationSummary,
    collected_at_utc: datetime,
    scoring_adapter: LiveForwardShadowScoringAdapter,
    scoring_diagnostics: dict[str, int],
    forecast_snapshot_builder: Optional[
        Callable[[Sequence[ForecastSource]], Optional[ForecastSnapshot]]
    ] = None,
    decision_id_override: Optional[str] = None,
    side: str = "NO",
) -> ReplayFrame:
    side_upper = str(side).upper()
    if side_upper not in {"YES", "NO"}:
        raise ValueError(f"unsupported replay-frame side: {side}")
    selected_book = yes_book if side_upper == "YES" else no_book
    selected_token_id = (
        market.bucket.yes_token_id
        if side_upper == "YES"
        else market.bucket.no_token_id
    ) or ""
    quote = quote_snapshot_from_book(selected_book)
    forecast_snapshot = (forecast_snapshot_builder or forecast_snapshot_from_sources)(
        forecasts
    )
    observed = observed_high_snapshot_from_station(station_summary)
    decision_id = decision_id_override or _stable_id(
        "live-shadow",
        market.market_id,
        selected_token_id,
        collected_at_utc.isoformat(timespec="seconds"),
    )
    decision = DecisionSnapshot(
        decision_id=decision_id,
        universe_row_id=f"live:{market.market_id or market.slug}:{selected_token_id}",
        market_id=market.market_id,
        slug=market.slug,
        question=market.question,
        normalized_city=market.city.display_name,
        country_region=market.city.region,
        timezone=market.city.timezone,
        market_date_local=market.target_date.isoformat(),
        bucket_label=f"{side_upper}: {market.bucket.label}",
        bucket_type=market.bucket.bucket_type,
        threshold_low_f=market.bucket.lower_f,
        threshold_high_f=market.bucket.upper_f,
        side=side_upper,
        yes_token_id=market.bucket.yes_token_id,
        no_token_id=market.bucket.no_token_id,
        token_id=selected_token_id,
        decision_time_utc=collected_at_utc,
        decision_hour_utc=collected_at_utc.hour,
        lead_days=(market.target_date - collected_at_utc.astimezone(ZoneInfo(market.city.timezone)).date()).days,
        quote_store="live_polymarket_clob_public_book_snapshot",
        quote=quote,
        forecast=forecast_snapshot,
        data_quality_flags=live_data_quality_flags(
            market=market,
            quote=quote,
            forecasts=forecasts,
            station_summary=station_summary,
        ),
        parser_confidence=0.95,
        observed_high=observed,
        observed_high_source_payload_id=station_summary.payload_id,
        observed_high_raw_cache_path=station_summary.raw_cache_path,
        observed_high_data_quality_flags=(station_summary.status,),
        observed_high_missing_reason=None if observed is not None else station_summary.status,
        decision_local_hour=_local_hour(collected_at_utc, market.city.timezone),
        settlement_source="pending_live_market",
    )
    score_result = scoring_adapter.score_decision(
        decision,
        settlement_unit=market.bucket.resolution_unit,
        settlement_precision=(
            str(market.bucket.resolution_precision)
            if market.bucket.resolution_precision is not None
            else None
        ),
    )
    if score_result.firewall_violation:
        scoring_diagnostics["firewall_violation_count"] = (
            int(scoring_diagnostics.get("firewall_violation_count") or 0) + 1
        )
    if score_result.missing_model_output:
        scoring_diagnostics["missing_model_output_count"] = (
            int(scoring_diagnostics.get("missing_model_output_count") or 0) + 1
        )
    return ReplayFrame(decision=decision, canonical_label=None, skip_reasons=(), score=score_result.score)


def quote_snapshot_from_book(book: Optional[BookSnapshot]) -> Optional[QuoteSnapshot]:
    if book is None or book.raw_status != "ok":
        return None
    return QuoteSnapshot(
        timestamp_utc=book.fetched_at_utc,
        bid=book.best_bid,
        ask=book.best_ask,
        mid=book.mid,
        last_price=None,
        spread=book.spread,
        bid_size=book.bid_size,
        ask_size=book.ask_size,
        liquidity_depth=book.visible_top_depth,
        source_payload_id=book.payload_id,
        source_cache_path=book.raw_cache_path,
        data_quality_flags=("live_public_clob_book",),
    )


def forecast_snapshot_from_sources(forecasts: Sequence[ForecastSource]) -> Optional[ForecastSnapshot]:
    usable = [item for item in forecasts if item.temperature_2m_max_f is not None]
    if not usable:
        return None
    values = [float(item.temperature_2m_max_f) for item in usable]
    return ForecastSnapshot(
        run_time_utc=max(item.fetched_at_utc for item in usable),
        source_name="live_open_meteo_multi_source_mean",
        temperature_2m_max_f=sum(values) / len(values),
        features={
            "source_count": len(usable),
            "sources": [item.source_name for item in usable],
            "source_temperature_2m_max_f": {
                item.source_name: item.temperature_2m_max_f for item in usable
            },
        },
    )


def frozen_s01_forecast_snapshot_from_sources(
    forecasts: Sequence[ForecastSource],
) -> Optional[ForecastSnapshot]:
    """Build the frozen S01 scoring input from one live best-match forecast.

    The historical endpoint used by the immutable replay is unavailable for
    future dates. The live endpoint identity is therefore retained explicitly;
    it is never relabelled as the historical archive. No cross-source mean is
    permitted on this path.
    """

    matches = [
        item
        for item in forecasts
        if item.source_name == FROZEN_S01_LIVE_FORECAST_SOURCE
        and item.status == "ok"
        and item.temperature_2m_max_f is not None
    ]
    if len(matches) != 1:
        return None
    source = matches[0]
    return ForecastSnapshot(
        run_time_utc=source.fetched_at_utc,
        source_name=source.source_name,
        temperature_2m_max_f=float(source.temperature_2m_max_f),
        features={
            **dict(source.features),
            "live_forecast_source_difference_from_frozen_history": True,
            "frozen_historical_source_name": "historical_forecast_archive",
        },
    )


def validate_frozen_s01_input_timing(
    *,
    forecasts: Sequence[ForecastSource],
    station_summary: StationObservationSummary,
    books: Sequence[Optional[BookSnapshot]],
    decision_time_utc: datetime,
) -> FrozenS01InputTiming:
    """Fail closed unless feature_time <= every venue quote <= decision time."""

    decision_time = decision_time_utc.astimezone(timezone.utc)
    selected = [
        item
        for item in forecasts
        if item.source_name == FROZEN_S01_LIVE_FORECAST_SOURCE
        and item.status == "ok"
        and item.temperature_2m_max_f is not None
    ]
    feature_times = [item.fetched_at_utc.astimezone(timezone.utc) for item in selected]
    if station_summary.status not in {"error", "missing"}:
        feature_times.append(station_summary.fetched_at_utc.astimezone(timezone.utc))
    valid_books = [book for book in books if book is not None and book.raw_status == "ok"]
    quote_times = [book.fetched_at_utc.astimezone(timezone.utc) for book in valid_books]
    feature_time = max(feature_times) if feature_times else None
    earliest_quote = min(quote_times) if quote_times else None
    latest_quote = max(quote_times) if quote_times else None
    reasons: list[str] = []
    if len(selected) != 1:
        reasons.append("exactly_one_live_best_match_forecast_required")
    if feature_time is None:
        reasons.append("feature_time_missing")
    if len(valid_books) != len(books) or not quote_times:
        reasons.append("venue_book_missing_or_invalid")
    if feature_time is not None and earliest_quote is not None and feature_time > earliest_quote:
        reasons.append("feature_time_after_venue_quote")
    if latest_quote is not None and latest_quote > decision_time:
        reasons.append("venue_quote_after_decision")
    return FrozenS01InputTiming(
        gate_valid=not reasons,
        feature_time_utc=feature_time,
        earliest_quote_time_utc=earliest_quote,
        latest_quote_time_utc=latest_quote,
        decision_time_utc=decision_time,
        reasons=tuple(reasons),
    )


def observed_high_snapshot_from_station(
    station_summary: StationObservationSummary,
) -> Optional[ObservedHighSnapshot]:
    if station_summary.high_so_far_f is None or station_summary.high_so_far_observed_at_utc is None:
        return None
    return ObservedHighSnapshot(
        observed_high_f=station_summary.high_so_far_f,
        observed_high_timestamp_utc=station_summary.high_so_far_observed_at_utc,
        observation_source=station_summary.source_family,
        observation_final=False,
        source_family=station_summary.source_family,
        source_station=station_summary.station_id,
        deterministic_state="live_intraday_high_so_far",
    )


def shared_score_from_forecasts(
    market: WeatherMarket,
    *,
    forecasts: Sequence[ForecastSource],
    station_summary: StationObservationSummary,
    no_book: Optional[BookSnapshot],
    yes_book: Optional[BookSnapshot],
    scored_at: datetime,
) -> Optional[SharedScoreContract]:
    """Legacy live forecast adapter; preserved only for diagnostic compatibility."""
    source_probabilities: dict[str, float] = {}
    for forecast in forecasts:
        if forecast.status != "ok" or not forecast.samples_f:
            continue
        probability = _bucket_probability(
            samples=forecast.samples_f,
            lower_f=market.bucket.lower_f,
            upper_f=market.bucket.upper_f,
            sigma_floor=2.5,
        )
        source_probabilities[forecast.source_name] = probability
    if not source_probabilities:
        return None
    values = [max(0.0, min(1.0, value)) for value in source_probabilities.values()]
    p_yes = sum(values) / len(values)
    p_no = 1.0 - p_yes
    no_ask = no_book.best_ask if no_book else None
    model_agreement = sum(1 for value in values if value <= 0.14) / len(values)
    if no_ask is not None:
        model_agreement = min(
            model_agreement,
            sum(1 for value in values if (1.0 - value) >= no_ask) / len(values),
        )
    stdev = statistics.pstdev(values) if len(values) > 1 else 0.0
    return SharedScoreContract(
        p_yes=round(p_yes, 6),
        side="NO",
        p_side=round(p_no, 6),
        model_agreement=round(model_agreement, 6),
        model_count=len(values),
        probability_stdev=round(stdev, 6),
        per_source_probabilities={key: round(value, 6) for key, value in source_probabilities.items()},
        observed_high=observed_high_snapshot_from_station(station_summary),
        parser_confidence=0.95,
        settlement_unit=market.bucket.resolution_unit,
        settlement_precision=(
            str(market.bucket.resolution_precision) if market.bucket.resolution_precision is not None else None
        ),
        scored_at_utc=scored_at,
        score_model_version="live_open_meteo_multisource_normal_v1",
        score_family="old_no_tail_shadow_live_forecast_adapter",
        metadata={
            "legacy_non_gate_valid_helper": True,
            "gate_valid_for_shadow_deployment": False,
            "read_only_shadow": True,
            "real_orders_allowed": False,
            "observed_no_ask": no_book.best_ask if no_book else None,
            "observed_no_bid": no_book.best_bid if no_book else None,
            "observed_yes_ask": yes_book.best_ask if yes_book else None,
            "station_observation_status": station_summary.status,
        },
    )


def live_data_quality_flags(
    *,
    market: WeatherMarket,
    quote: Optional[QuoteSnapshot],
    forecasts: Sequence[ForecastSource],
    station_summary: StationObservationSummary,
) -> tuple[str, ...]:
    flags: list[str] = ["live_forward_shadow_read_only"]
    if quote is None:
        flags.append("missing_no_book")
    if not any(item.temperature_2m_max_f is not None for item in forecasts):
        flags.append("missing_open_meteo_forecast")
    if station_summary.high_so_far_f is None:
        flags.append("missing_station_high_so_far")
    if market.bucket.resolution_unit:
        flags.append(f"resolution_unit_{market.bucket.resolution_unit.lower()}")
    return tuple(flags)


def fetch_order_book(
    token_id: Optional[str],
    *,
    raw_dir: Path,
    fetched_at: datetime,
    cache_only: bool,
) -> Optional[BookSnapshot]:
    if not token_id:
        return None
    raw_dir.mkdir(parents=True, exist_ok=True)
    params = {"token_id": token_id}
    url = f"{CLOB_BASE_URL}/book?{urllib.parse.urlencode(params)}"
    cache_path = raw_dir / f"{_safe_name(token_id)}-{fetched_at.strftime('%Y%m%dT%H%M%SZ')}.json"
    try:
        payload = _read_cached_or_fetch_json(url, cache_path=cache_path, cache_only=cache_only)
        received_at = datetime.now(timezone.utc)
        payload_id = _payload_id(url, payload)
        source_sha256 = _canonical_payload_sha256(payload)
        venue_timestamp_raw = payload.get("timestamp") if isinstance(payload, Mapping) else None
        venue_timestamp = parse_polymarket_book_timestamp(venue_timestamp_raw)
        bids = _parse_levels(payload.get("bids") or [])
        asks = _parse_levels(payload.get("asks") or [])
        best_bid = max(bids, key=lambda item: item[0]) if bids else None
        best_ask = min(asks, key=lambda item: item[0]) if asks else None
        bid = best_bid[0] if best_bid else None
        ask = best_ask[0] if best_ask else None
        return BookSnapshot(
            token_id=token_id,
            fetched_at_utc=venue_timestamp or fetched_at,
            payload_id=payload_id,
            raw_cache_path=str(cache_path),
            best_bid=bid,
            best_ask=ask,
            bid_size=best_bid[1] if best_bid else None,
            ask_size=best_ask[1] if best_ask else None,
            mid=((bid + ask) / 2) if bid is not None and ask is not None else bid if ask is None else ask,
            spread=(ask - bid) if bid is not None and ask is not None else None,
            top_bids=tuple({"price": price, "size": size} for price, size in sorted(bids, reverse=True)[:5]),
            top_asks=tuple({"price": price, "size": size} for price, size in sorted(asks)[:5]),
            bid_levels=tuple(
                {"price": price, "size": size}
                for price, size in sorted(bids, reverse=True)
            ),
            ask_levels=tuple(
                {"price": price, "size": size}
                for price, size in sorted(asks)
            ),
            full_book_complete=True,
            min_order_size_shares=_optional_float(payload.get("min_order_size")),
            tick_size=_optional_float(payload.get("tick_size")),
            neg_risk=_optional_bool(payload.get("neg_risk")),
            fee_rate=_optional_float(payload.get("fee_rate")),
            fee_rate_source=(
                "clob_order_book_payload"
                if _optional_float(payload.get("fee_rate")) is not None
                else None
            ),
            raw_status="ok" if venue_timestamp is not None else "invalid",
            error=(
                None
                if venue_timestamp is not None
                else "missing_or_invalid_polymarket_book_timestamp"
            ),
            received_at_utc=received_at,
            venue_timestamp_raw=(
                str(venue_timestamp_raw) if venue_timestamp_raw is not None else None
            ),
            source_sha256=source_sha256,
        )
    except Exception as exc:
        return BookSnapshot(
            token_id=token_id,
            fetched_at_utc=fetched_at,
            payload_id="",
            raw_cache_path=str(cache_path),
            best_bid=None,
            best_ask=None,
            bid_size=None,
            ask_size=None,
            mid=None,
            spread=None,
            top_bids=(),
            top_asks=(),
            raw_status="error",
            error=f"{type(exc).__name__}: {str(exc)[:300]}",
        )


def fetch_open_meteo_forecasts(
    city: CityConfig,
    target_date: date,
    *,
    raw_dir: Path,
    fetched_at: datetime,
    cache_only: bool,
) -> list[ForecastSource]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    requests = _open_meteo_requests(city)
    outputs: list[ForecastSource] = []
    for source_name, url, params in requests:
        request_url = f"{url}?{urllib.parse.urlencode(params)}"
        cache_path = raw_dir / f"{_safe_name(city.display_name)}-{target_date.isoformat()}-{source_name}-{fetched_at.strftime('%Y%m%dT%H%M%SZ')}.json"
        try:
            payload = _read_cached_or_fetch_json(request_url, cache_path=cache_path, cache_only=cache_only)
            received_at = datetime.now(timezone.utc)
            samples = tuple(extract_daily_temperature_samples(payload, target_date))
            features = {
                **extract_weather_features(payload, target_date),
                "latitude": city.latitude,
                "longitude": city.longitude,
                "model_name": source_name.removeprefix("open_meteo_"),
                "input_source_payload_id": _payload_id(request_url, payload),
                "input_source_sha256": _canonical_payload_sha256(payload),
                "input_source_cache_path": str(cache_path),
                "input_source_request_url": request_url,
                "feature_time_utc": received_at.isoformat(),
            }
            outputs.append(
                ForecastSource(
                    source_name=source_name,
                    fetched_at_utc=received_at,
                    temperature_2m_max_f=(sum(samples) / len(samples)) if samples else None,
                    samples_f=samples,
                    features=features,
                    raw_cache_path=str(cache_path),
                    payload_id=_payload_id(request_url, payload),
                    request_url=request_url,
                )
            )
        except Exception as exc:
            outputs.append(
                ForecastSource(
                    source_name=source_name,
                    fetched_at_utc=fetched_at,
                    temperature_2m_max_f=None,
                    samples_f=(),
                    features={},
                    raw_cache_path=str(cache_path),
                    payload_id="",
                    request_url=request_url,
                    status="error",
                    error=f"{type(exc).__name__}: {str(exc)[:300]}",
                )
            )
    return outputs


def fetch_station_observations(
    city: CityConfig,
    target_date: date,
    *,
    raw_dir: Path,
    fetched_at: datetime,
    cache_only: bool,
) -> StationObservationSummary:
    station_id = city.metar_station
    if not station_id:
        return StationObservationSummary(
            station_id=None,
            status="no_station_configured",
            fetched_at_utc=fetched_at,
        )
    raw_dir.mkdir(parents=True, exist_ok=True)
    config = StationConfig(
        station_id=station_id,
        normalized_city=city.display_name,
        timezone_name=city.timezone,
        settlement_source_family="live_shadow_metar",
        settlement_source_url=f"metar:{station_id}",
        settlement_station_name=station_id,
        settlement_unit="F",
        settlement_precision="whole_degree",
    )
    try:
        request_url = _iem_request_url(config, start=target_date, end=target_date)
        payload, cache_path, payload_id, source_status = _get_or_fetch_raw_payload(
            cache_dir=raw_dir,
            config=config,
            request_url=request_url,
            start=target_date,
            end=target_date,
            phase="live_forward_shadow",
            fetched_at=fetched_at.isoformat(),
            cache_only=cache_only,
        )
        received_at = datetime.now(timezone.utc)
        _mirror_recent_station_payload(raw_dir=raw_dir, station_id=station_id, cache_path=cache_path)
        return _station_summary_from_iem_payload(
            city=city,
            target_date=target_date,
            fetched_at=received_at,
            config=config,
            request_url=request_url,
            payload=payload,
            cache_path=cache_path,
            payload_id=payload_id,
            source_status=source_status,
        )
    except Exception as exc:
        fallback = _latest_recent_station_payload(raw_dir=raw_dir, station_id=station_id, target_date=target_date)
        if fallback is not None:
            cache_path, payload = fallback
            try:
                received_at = datetime.now(timezone.utc)
                return _station_summary_from_iem_payload(
                    city=city,
                    target_date=target_date,
                    fetched_at=received_at,
                    config=config,
                    request_url=request_url,
                    payload=payload,
                    cache_path=cache_path,
                    payload_id=_stable_id(
                        "station-recent-cache",
                        request_url,
                        str(cache_path),
                        str(cache_path.stat().st_mtime_ns),
                    ),
                    source_status=f"recent_cache_fallback_after_{type(exc).__name__}",
                    source_error=f"{type(exc).__name__}: {str(exc)[:300]}",
                )
            except Exception as fallback_exc:
                return StationObservationSummary(
                    station_id=station_id,
                    status="error",
                    fetched_at_utc=fetched_at,
                    local_date=target_date.isoformat(),
                    error=(
                        f"{type(exc).__name__}: {str(exc)[:160]}; "
                        f"fallback_parse_failed={type(fallback_exc).__name__}: {str(fallback_exc)[:120]}"
                    ),
                )
        return StationObservationSummary(
            station_id=station_id,
            status="error",
            fetched_at_utc=fetched_at,
            local_date=target_date.isoformat(),
            error=f"{type(exc).__name__}: {str(exc)[:300]}",
        )


def _station_summary_from_iem_payload(
    *,
    city: CityConfig,
    target_date: date,
    fetched_at: datetime,
    config: StationConfig,
    request_url: str,
    payload: bytes,
    cache_path: Path,
    payload_id: str,
    source_status: str,
    source_error: Optional[str] = None,
) -> StationObservationSummary:
    station_id = city.metar_station
    _, observations = parse_iem_csv_payload(
        payload,
        config=config,
        request_url=request_url,
        fetched_at=fetched_at.isoformat(),
        payload_id=payload_id,
    )
    usable = [
        item
        for item in observations
        if item.temperature_f is not None
        and item.local_date == target_date.isoformat()
        and _parse_utc(item.observed_at_utc) <= fetched_at
    ]
    if not usable:
        return StationObservationSummary(
            station_id=station_id,
            status=f"{source_status}:no_temperature_observations_as_of_poll",
            fetched_at_utc=fetched_at,
            local_date=target_date.isoformat(),
            observation_count=0,
            raw_cache_path=str(cache_path),
            payload_id=payload_id,
            error=source_error,
        )
    latest = max(usable, key=lambda item: item.observed_at_utc)
    high = max(usable, key=lambda item: (float(item.temperature_f or -999.0), item.observed_at_utc))
    return StationObservationSummary(
        station_id=station_id,
        status=source_status,
        fetched_at_utc=fetched_at,
        local_date=target_date.isoformat(),
        latest_observed_at_utc=_parse_utc(latest.observed_at_utc),
        latest_temperature_f=latest.temperature_f,
        high_so_far_f=high.temperature_f,
        high_so_far_observed_at_utc=_parse_utc(high.observed_at_utc),
        observation_count=len(usable),
        raw_cache_path=str(cache_path),
        payload_id=payload_id,
        error=source_error,
    )


def _station_recent_cache_dir(raw_dir: Path) -> Path:
    if len(raw_dir.parents) >= 3 and raw_dir.parents[2].name == "raw":
        return raw_dir.parents[2] / "station_recent"
    return raw_dir / "_station_recent"


def _mirror_recent_station_payload(*, raw_dir: Path, station_id: str, cache_path: Path) -> None:
    if not cache_path.exists():
        return
    recent_dir = _station_recent_cache_dir(raw_dir) / station_id
    recent_dir.mkdir(parents=True, exist_ok=True)
    target = recent_dir / cache_path.name
    if not target.exists():
        target.write_bytes(cache_path.read_bytes())
    sidecar = cache_path.with_suffix(".request.json")
    if sidecar.exists():
        sidecar_target = target.with_suffix(".request.json")
        if not sidecar_target.exists():
            sidecar_target.write_text(sidecar.read_text(encoding="utf-8"), encoding="utf-8")


def _latest_recent_station_payload(
    *,
    raw_dir: Path,
    station_id: str,
    target_date: date,
) -> Optional[tuple[Path, bytes]]:
    pattern = f"{station_id}-live_forward_shadow-{target_date.isoformat()}-{target_date.isoformat()}-*.csv"
    candidates: list[Path] = []
    candidates.extend((_station_recent_cache_dir(raw_dir) / station_id).glob(pattern))
    if len(raw_dir.parents) >= 3 and raw_dir.parents[2].name == "raw":
        candidates.extend(raw_dir.parents[2].glob(f"*/poll-*/station_observations/{station_id}/{pattern}"))
    existing = [path for path in candidates if path.exists()]
    if not existing:
        return None
    latest = max(existing, key=lambda path: path.stat().st_mtime_ns)
    return latest, latest.read_bytes()


def parse_weather_market(raw: Mapping[str, Any], *, today: date) -> Optional[WeatherMarket]:
    question = str(raw.get("question") or raw.get("title") or "")
    description = str(raw.get("description") or raw.get("rules") or raw.get("resolutionSource") or "")
    event_context = " ".join(
        str(raw.get(key) or "")
        for key in ("slug", "eventTitle", "eventSubtitle", "eventDescription", "eventSlug")
    )
    combined = f"{question}\n{description}\n{event_context}"
    city = find_city(combined)
    target_date = parse_market_date(combined, today=today)
    if city is None or target_date is None:
        return None
    outcomes = parse_jsonish_list(raw.get("outcomes"))
    token_ids = parse_jsonish_list(raw.get("clobTokenIds") or raw.get("clob_token_ids"))
    prices = [_optional_float(item) for item in parse_jsonish_list(raw.get("outcomePrices") or raw.get("outcome_prices"))]
    bucket = parse_binary_temperature_bucket(question, outcomes=outcomes, token_ids=token_ids, prices=prices)
    if bucket is None:
        return None
    bucket = _apply_resolution_precision(bucket, combined)
    return WeatherMarket(
        market_id=str(raw.get("id") or raw.get("conditionId") or raw.get("condition_id") or raw.get("slug") or ""),
        condition_id=str(raw.get("conditionId") or raw.get("condition_id") or ""),
        slug=str(raw.get("slug") or ""),
        event_slug=str(raw.get("eventSlug") or raw.get("event_slug") or ""),
        question=question,
        close_time=parse_close_time(raw.get("endDate") or raw.get("end_date") or raw.get("closeTime")),
        target_date=target_date,
        city=city,
        bucket=bucket,
        raw=raw,
    )


def parse_binary_temperature_bucket(
    question: str,
    *,
    outcomes: Sequence[Any],
    token_ids: Sequence[Any],
    prices: Sequence[Optional[float]],
) -> Optional[Bucket]:
    outcome_labels = [str(outcome).strip().lower() for outcome in outcomes]
    if "yes" not in outcome_labels or "no" not in outcome_labels:
        return None
    yes_index = outcome_labels.index("yes")
    no_index = outcome_labels.index("no")
    parsed = _parse_temperature_bucket_details(question)
    if parsed is None:
        return None
    yes_token = str(token_ids[yes_index]) if yes_index < len(token_ids) and token_ids[yes_index] else None
    no_token = str(token_ids[no_index]) if no_index < len(token_ids) and token_ids[no_index] else None
    return Bucket(
        label=parsed.label,
        lower_f=parsed.lower_f,
        upper_f=parsed.upper_f,
        yes_token_id=yes_token,
        no_token_id=no_token,
        yes_price=prices[yes_index] if yes_index < len(prices) else None,
        no_price=prices[no_index] if no_index < len(prices) else None,
        resolution_unit=parsed.source_unit,
        resolution_precision=parsed.source_precision,
        source_bucket_type=parsed.source_bucket_type,
    )


def parse_temperature_bucket(label: str) -> Optional[tuple[Optional[float], Optional[float], str]]:
    parsed = _parse_temperature_bucket_details(label)
    if parsed is None:
        return None
    return parsed.lower_f, parsed.upper_f, parsed.label


def _parse_temperature_bucket_details(label: str) -> Optional[_TemperatureBucketParse]:
    unit = _detect_unit(label)
    explicit_unit = _detect_explicit_unit(label)
    source_precision = 1.0 if explicit_unit else None
    clean = _clean_temperature_label(label)
    lower = clean.lower()
    range_match = re.search(
        r"(?:between\s+)?(?P<low>-?\d+(?:\.\d+)?)\s*(?:-|to|through|thru|and)\s*(?P<high>-?\d+(?:\.\d+)?)",
        lower,
    )
    if range_match:
        low = _to_fahrenheit(float(range_match.group("low")), unit)
        high = _to_fahrenheit(float(range_match.group("high")), unit)
        bucket_type = "exact" if unit == "F" and 0 <= high - low <= 1.0 else "range"
        return _TemperatureBucketParse(low, high, f"{low:g}-{high:g}F", explicit_unit, source_precision, bucket_type)
    under_match = re.search(r"(?:under|below|less than|lower than|<)\s*(?P<value>-?\d+(?:\.\d+)?)", lower)
    if under_match:
        value = _to_fahrenheit(float(under_match.group("value")) - 0.01, unit)
        return _TemperatureBucketParse(None, value, f"below {value:g}F", explicit_unit, source_precision, "below")
    at_most_match = re.search(r"(?:at most|no more than|<=)\s*(?P<value>-?\d+(?:\.\d+)?)", lower)
    if at_most_match:
        value = _to_fahrenheit(float(at_most_match.group("value")), unit)
        return _TemperatureBucketParse(None, value, f"at most {value:g}F", explicit_unit, source_precision, "below")
    above_match = re.search(r"(?P<value>-?\d+(?:\.\d+)?)\s*(?:or above|or higher|and above|or more|\+)", lower)
    if above_match:
        value = _to_fahrenheit(float(above_match.group("value")), unit)
        return _TemperatureBucketParse(value, None, f"{value:g}F or above", explicit_unit, source_precision, "above")
    over_match = re.search(r"(?:over|above|greater than|higher than|>)\s*(?P<value>-?\d+(?:\.\d+)?)", lower)
    if over_match:
        value = _to_fahrenheit(float(over_match.group("value")) + 0.01, unit)
        return _TemperatureBucketParse(value, None, f"above {value:g}F", explicit_unit, source_precision, "above")
    predicate_match = re.search(
        r"\bbe\s+(?P<value>-?\d+(?:\.\d+)?)\s*(?P<qualifier>or below|or lower|or less|or above|or higher|or more)?\b",
        lower,
        re.I,
    )
    if predicate_match:
        source_value = float(predicate_match.group("value"))
        value = _to_fahrenheit(source_value, unit)
        qualifier = (predicate_match.group("qualifier") or "").lower()
        if "below" in qualifier or "lower" in qualifier or "less" in qualifier:
            return _TemperatureBucketParse(None, value, f"{value:g}F or below", explicit_unit, source_precision, "below")
        if "above" in qualifier or "higher" in qualifier or "more" in qualifier:
            return _TemperatureBucketParse(value, None, f"{value:g}F or above", explicit_unit, source_precision, "above")
        upper = _exact_bucket_upper_f(source_value, unit)
        return _TemperatureBucketParse(value, upper, f"{value:g}F", explicit_unit, source_precision, "exact")
    return None


def looks_like_temperature_market(raw: Mapping[str, Any]) -> bool:
    text = " ".join(
        str(raw.get(key) or "")
        for key in ("question", "title", "description", "slug", "eventSlug", "eventTitle", "eventDescription")
    ).lower()
    if "lowest temperature" in text or "lowest temp" in text or "low temperature" in text:
        return False
    return "highest temperature" in text or "highest temp" in text or "high temp" in text


def parse_market_date(text: str, *, today: date) -> Optional[date]:
    matches = list(_DATE_RE.finditer(text))
    if not matches:
        return None
    match = next((item for item in matches if item.group("year")), matches[0])
    month = MONTHS[match.group("month").lower().rstrip(".")]
    day = int(match.group("day"))
    year = int(match.group("year")) if match.group("year") else today.year
    parsed = date(year, month, day)
    if match.group("year") is None and parsed < today:
        parsed = date(year + 1, month, day)
    return parsed


def parse_close_time(value: Any) -> Optional[datetime]:
    if not value:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def find_city(text: str) -> Optional[CityConfig]:
    lowered = text.lower()
    for city in DEFAULT_CITIES:
        for candidate in (city.name, city.display_name, *city.aliases):
            candidate_lower = candidate.lower()
            if len(candidate_lower) <= 3:
                if re.search(rf"\b{re.escape(candidate_lower)}\b", lowered):
                    return city
            elif candidate_lower in lowered:
                return city
    return None


def extract_daily_temperature_samples(payload: Mapping[str, Any], target_date: date) -> list[float]:
    daily = payload.get("daily") if isinstance(payload, Mapping) else {}
    if not isinstance(daily, Mapping):
        return []
    times = list(daily.get("time") or [])
    try:
        index = [date.fromisoformat(str(item)) for item in times].index(target_date)
    except ValueError:
        return []
    samples: list[float] = []
    for key, values in daily.items():
        if key == "time" or not str(key).startswith("temperature_2m_max"):
            continue
        samples.extend(_value_at_index(values, index))
    return [value for value in samples if value is not None]


def extract_weather_features(payload: Mapping[str, Any], target_date: date) -> dict[str, Any]:
    daily = payload.get("daily") if isinstance(payload, Mapping) else {}
    features: dict[str, Any] = {}
    if isinstance(daily, Mapping):
        times = list(daily.get("time") or [])
        try:
            index = [date.fromisoformat(str(item)) for item in times].index(target_date)
        except ValueError:
            index = -1
        if index >= 0:
            for key in (
                "precipitation_sum",
                "precipitation_hours",
                "wind_speed_10m_max",
                "wind_gusts_10m_max",
                "shortwave_radiation_sum",
            ):
                value = _single_value_at_index(daily.get(key), index)
                if value is not None:
                    features[key] = value
    hourly = payload.get("hourly") if isinstance(payload, Mapping) else {}
    if isinstance(hourly, Mapping):
        times = list(hourly.get("time") or [])
        indices = [index for index, item in enumerate(times) if str(item).startswith(target_date.isoformat())]
        if indices:
            reducers: tuple[tuple[str, str, Callable[[Sequence[float]], float]], ...] = (
                ("temperature_2m", "hourly_temperature_2m_max", max),
                ("apparent_temperature", "hourly_apparent_temperature_max", max),
                (
                    "relative_humidity_2m",
                    "relative_humidity_2m_mean",
                    lambda values: sum(values) / len(values),
                ),
                ("dew_point_2m", "dew_point_2m_max", max),
                (
                    "cloud_cover",
                    "cloud_cover_mean",
                    lambda values: sum(values) / len(values),
                ),
                ("wind_speed_10m", "hourly_wind_speed_10m_max", max),
                ("precipitation", "hourly_precipitation_sum", sum),
            )
            for source_key, output_key, reducer in reducers:
                values = [
                    _single_value_at_index(hourly.get(source_key), index)
                    for index in indices
                ]
                cleaned = [float(value) for value in values if value is not None]
                if cleaned:
                    features[output_key] = float(reducer(cleaned))
    return features


def _open_meteo_requests(city: CityConfig) -> tuple[tuple[str, str, dict[str, object]], ...]:
    base = {
        "latitude": city.latitude,
        "longitude": city.longitude,
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
        "hourly": ",".join(
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
        ),
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "precipitation_unit": "inch",
        "timezone": city.timezone,
    }
    return (
        ("open_meteo_best_match", OPEN_METEO_FORECAST_URL, dict(base)),
        ("open_meteo_gfs_global", OPEN_METEO_GFS_URL, {**base, "models": "gfs_global"}),
        ("open_meteo_ecmwf", OPEN_METEO_ECMWF_URL, dict(base)),
        (
            "open_meteo_ensemble_gfs",
            OPEN_METEO_ENSEMBLE_URL,
            {
                "latitude": city.latitude,
                "longitude": city.longitude,
                "models": "gfs_seamless",
                "daily": "temperature_2m_max",
                "temperature_unit": "fahrenheit",
                "timezone": city.timezone,
            },
        ),
    )


def _bucket_probability(
    *,
    samples: Sequence[float],
    lower_f: Optional[float],
    upper_f: Optional[float],
    sigma_floor: float,
) -> float:
    if not samples:
        return 0.0
    return sum(_normal_interval_probability(sample, sigma_floor, lower_f, upper_f) for sample in samples) / len(samples)


def _normal_interval_probability(
    mean: float,
    sigma: float,
    lower: Optional[float],
    upper: Optional[float],
) -> float:
    sigma = max(0.1, sigma)
    low_cdf = 0.0 if lower is None else _normal_cdf((lower - mean) / sigma)
    high_cdf = 1.0 if upper is None else _normal_cdf((upper - mean) / sigma)
    return max(0.0, min(1.0, high_cdf - low_cdf))


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _enrich_shadow_events(
    events: Sequence[Mapping[str, Any]],
    live_snapshots: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    by_decision = {str(item.get("decision_id")): item for item in live_snapshots}
    enriched = []
    for event in events:
        payload = dict(event)
        market = payload.get("market_discovery") or {}
        decision_id = str(market.get("decision_id") or "")
        if decision_id in by_decision:
            live = by_decision[decision_id]
            payload["live_market_metadata"] = live.get("market_metadata")
            payload["parsed_contract"] = live.get("parsed_contract")
            payload["observed_order_books"] = live.get("observed_books")
            payload["assumed_executable_price"] = live.get("assumed_executable_price")
            payload["open_meteo_forecast_sources"] = live.get("open_meteo_forecasts")
            payload["live_station_observations"] = live.get("station_observations")
        enriched.append(payload)
    return enriched


def _summary_payload(
    *,
    run_id: str,
    started_at: datetime,
    completed_at: datetime,
    requested_iterations: int,
    effective_iterations: int,
    interval_seconds: int,
    discovery_errors: Sequence[str],
    poll_summaries: Sequence[Mapping[str, Any]],
    live_snapshots: Sequence[Mapping[str, Any]],
    shadow_payload: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    raw_dir: Path,
    score_cache_summary: Mapping[str, Any],
    shadow_scoring_parity: Mapping[str, Any],
) -> Mapping[str, Any]:
    no_trade_reasons = Counter()
    would_trade = 0
    would_trade_cluster_keys: set[str] = set()
    quote_status = Counter()
    forecast_status = Counter()
    station_status = Counter()
    event_diagnostics = _event_diagnostics(events)
    for event in events:
        decision = event.get("decision") or {}
        if decision.get("would_trade"):
            would_trade += 1
            cluster_key = _would_trade_cluster_key(_settlement_identifiers_from_event(event))
            if cluster_key:
                would_trade_cluster_keys.add(cluster_key)
        else:
            for reason in decision.get("no_trade_reasons") or ("unknown",):
                no_trade_reasons[str(reason)] += 1
        quote_status[str((event.get("quote") or {}).get("status") or "unknown")] += 1
        forecast_status[str((event.get("open_meteo_forecast") or {}).get("status") or "unknown")] += 1
        station = event.get("live_station_observations") or {}
        station_status[str(station.get("status") or "unknown")] += 1
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "generated_at_utc": completed_at.isoformat(),
        "started_at_utc": started_at.isoformat(),
        "completed_at_utc": completed_at.isoformat(),
        "mode": "read_only_live_forward_shadow_intraday_recorder",
        "profile_key": OLD_NO_TAIL_LIVE_FORWARD_SHADOW_PROFILE_KEY,
        "read_only": True,
        "real_order_placement_allowed": False,
        "real_orders_placed": 0,
        "fills_simulated_as_truth": False,
        "assumed_executable_price_policy": "recorded_separately_from_observed_book; no fill truth claim",
        "historical_pnl_numbers_are_expected_performance": False,
        "pre_registered_plan": OLD_NO_TAIL_LIVE_FORWARD_SHADOW_PREREGISTRATION,
        "requested_iterations": requested_iterations,
        "effective_iterations": effective_iterations,
        "interval_seconds": interval_seconds,
        "polls": list(poll_summaries),
        "discovery_errors": list(discovery_errors),
        "raw_payload_dir": str(raw_dir),
        "live_snapshot_count": len(live_snapshots),
        "shadow_summary": dict(shadow_payload),
        "shadow_scoring_parity": dict(shadow_scoring_parity),
        "replay_generated_score_cache": dict(score_cache_summary),
        "decision_counts": {
            "shadow_decisions": len(events),
            "would_trade_signal_clusters": len(would_trade_cluster_keys),
            "would_trade_signals": would_trade,
            "raw_would_trade_poll_detections": would_trade,
            "no_trade_decisions": len(events) - would_trade,
        },
        "no_trade_reason_counts": dict(sorted(no_trade_reasons.items())),
        "quote_status_counts": dict(sorted(quote_status.items())),
        "forecast_status_counts": dict(sorted(forecast_status.items())),
        "station_status_counts": dict(sorted(station_status.items())),
        **event_diagnostics,
        "shadow_events_preview": list(events[:20]),
    }


def _event_diagnostics(events: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    settlement_status = Counter()
    would_trade_settlement_status = Counter()
    would_trade_cluster_settlement_status: dict[str, str] = {}
    lead_days = Counter()
    close_counts = Counter()
    quote_counts = Counter()
    spreads: list[float] = []
    visible_depths: list[float] = []
    bid_sizes: list[float] = []
    ask_sizes: list[float] = []
    would_trade_count = 0
    for event in events:
        decision_payload = event.get("decision") or {}
        market = event.get("market_discovery") or {}
        quote = event.get("quote") or {}
        settlement = event.get("settlement") or {}
        live_market = event.get("live_market_metadata") or {}
        lead = _optional_int(market.get("lead_days"))
        if lead is None:
            lead_days["missing"] += 1
        else:
            lead_days[str(lead)] += 1
            if lead in (1, 2):
                lead_days["priority_1_2"] += 1
            elif lead == 0:
                lead_days["same_day"] += 1
            else:
                lead_days["non_priority"] += 1
        settlement_key = str(settlement.get("status") or "unknown")
        settlement_status[settlement_key] += 1
        if decision_payload.get("would_trade"):
            would_trade_count += 1
            would_trade_settlement_status[settlement_key] += 1
            cluster_key = _would_trade_cluster_key(_settlement_identifiers_from_event(event))
            if cluster_key:
                would_trade_cluster_settlement_status[cluster_key] = _more_final_settlement_status(
                    would_trade_cluster_settlement_status.get(cluster_key),
                    settlement_key,
                )
        quote_status = str(quote.get("status") or "unknown")
        quote_counts[quote_status] += 1
        bid = _optional_float(quote.get("bid"))
        ask = _optional_float(quote.get("ask"))
        if quote_status == "logged":
            if bid is not None and ask is not None:
                quote_counts["bid_ask_available"] += 1
            if bid is None:
                quote_counts["missing_bid"] += 1
            if ask is None:
                quote_counts["missing_ask"] += 1
        spread = _optional_float(quote.get("spread"))
        if spread is not None:
            spreads.append(spread)
        depth = quote.get("top_of_book_depth") if isinstance(quote.get("top_of_book_depth"), Mapping) else {}
        bid_size = _optional_float(depth.get("bid_size") if isinstance(depth, Mapping) else None)
        ask_size = _optional_float(depth.get("ask_size") if isinstance(depth, Mapping) else None)
        visible_depth = _optional_float(depth.get("liquidity_depth") if isinstance(depth, Mapping) else None)
        if bid_size is not None:
            bid_sizes.append(bid_size)
        if ask_size is not None:
            ask_sizes.append(ask_size)
        if visible_depth is not None:
            visible_depths.append(visible_depth)
        close_time = parse_close_time(live_market.get("close_time_utc"))
        decision_time = _parse_optional_utc(market.get("decision_time_utc"))
        if close_time is None:
            close_counts["missing_close_time"] += 1
        elif decision_time is not None and close_time <= decision_time:
            close_counts["market_closed_at_or_before_poll"] += 1
            if lead in (1, 2):
                close_counts["priority_lead_day_market_close_blocks_entry"] += 1
            elif lead == 0:
                close_counts["same_day_market_close_blocks_intraday_family"] += 1
        else:
            close_counts["market_open_after_poll"] += 1
    resolved_would_trades = sum(
        count
        for status, count in would_trade_settlement_status.items()
        if status == "resolved"
    )
    cluster_status_counts = Counter(would_trade_cluster_settlement_status.values())
    current_run_gate = _evaluate_forward_shadow_gate(
        _gate_clusters_from_current_events(events)
    )
    resolved_would_trade_clusters = int(current_run_gate["resolved_cluster_count"])
    return {
        "settlement_status_counts": dict(sorted(settlement_status.items())),
        "would_trade_settlement_status_counts": dict(sorted(would_trade_settlement_status.items())),
        "would_trade_cluster_settlement_status_counts": dict(sorted(cluster_status_counts.items())),
        "lead_day_counts": dict(sorted(lead_days.items())),
        "quote_quality": {
            "counts": dict(sorted(quote_counts.items())),
            "spread": _numeric_summary(spreads),
            "visible_top_depth": _numeric_summary(visible_depths),
            "bid_size": _numeric_summary(bid_sizes),
            "ask_size": _numeric_summary(ask_sizes),
        },
        "close_time_diagnostics": dict(sorted(close_counts.items())),
        "live_vs_replay_behavior": {
            "status": current_run_gate["status"],
            "reason": current_run_gate["reason"],
            "current_run_would_trade_signals": would_trade_count,
            "current_run_resolved_would_trade_signals": resolved_would_trades,
            "gate_count_basis": "deduped_true_opportunity_clusters",
            "current_run_would_trade_signal_clusters": len(would_trade_cluster_settlement_status),
            "current_run_resolved_would_trade_clusters": resolved_would_trade_clusters,
            "current_run_forward_shadow_gate": current_run_gate,
            "pre_registered_expected_trades_per_day": dict(
                OLD_NO_TAIL_LIVE_FORWARD_SHADOW_PREREGISTRATION["expected_trades_per_utc_day"]
            ),
            "pre_registered_hit_rate_confidence_bands": dict(
                OLD_NO_TAIL_LIVE_FORWARD_SHADOW_PREREGISTRATION["hit_rate_confidence_bands"]
            ),
            "hit_rate_confidence_bands_are_secondary_diagnostics_only": True,
            "pre_registered_primary_economic_gate": {
                "statistic": "fee_adjusted_conservative_executable_net_pnl_per_deduped_resolved_opportunity_cluster",
                "uncertainty": "deterministic_95pct_nonparametric_cluster_bootstrap_percentile",
                "checkpoint_resolved_clusters": FORWARD_SHADOW_GATE_MIN_RESOLVED_CLUSTERS,
                "production_minimum_resolved_clusters": FORWARD_SHADOW_PRODUCTION_MIN_RESOLVED_CLUSTERS,
                "production_minimum_valid_calendar_days": FORWARD_SHADOW_PRODUCTION_MIN_VALID_DAYS,
                "grants_live_permission": False,
            },
            "historical_pnl_numbers_used_as_expected_performance": False,
        },
    }


def refresh_pending_would_trade_settlements(
    *,
    report_dir: Path,
    raw_dir: Path,
    run_id: str,
    cache_only: bool,
    generated_at: Optional[datetime] = None,
    resolved_recheck_window_hours: float = DEFAULT_RESOLVED_RECHECK_WINDOW_HOURS,
) -> Mapping[str, Any]:
    generated = generated_at or datetime.now(timezone.utc)
    report_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    updates: list[Mapping[str, Any]] = []
    gamma_event_cache: dict[
        str,
        tuple[Optional[Mapping[str, Any]], Optional[str], Optional[str], Optional[str], Optional[str]],
    ] = {}
    pending_clusters = _pending_would_trade_clusters(
        report_dir,
        generated_at=generated,
        resolved_recheck_window_hours=resolved_recheck_window_hours,
    )
    for record in pending_clusters:
        event = record["event"]
        identifiers = record.get("identifiers") if isinstance(record.get("identifiers"), Mapping) else {}
        if not identifiers:
            identifiers = _settlement_identifiers_from_event(event)
        event_slug = identifiers.get("event_slug")
        if not event_slug:
            updates.append(
                _settlement_refresh_update(
                    record=record,
                    identifiers=identifiers,
                    fetched_at=generated,
                    settlement={
                        "status": "refresh_error",
                        "later_update_required": True,
                        "outcome_logged": False,
                        "source": "polymarket_gamma_event_slug",
                        "reason": "missing_event_slug",
                    },
                )
            )
            continue
        if event_slug not in gamma_event_cache:
            try:
                payload, cache_path, payload_id, request_url = _fetch_gamma_event_by_slug(
                    event_slug,
                    raw_dir=raw_dir,
                    fetched_at=generated,
                    cache_only=cache_only,
                )
                gamma_event_cache[event_slug] = (payload, str(cache_path), payload_id, request_url, None)
            except Exception as exc:
                gamma_event_cache[event_slug] = (
                    None,
                    None,
                    None,
                    f"{GAMMA_BASE_URL}/events/slug/{urllib.parse.quote(event_slug)}",
                    f"{type(exc).__name__}: {str(exc)[:300]}",
                )
        gamma_event, raw_cache_path, payload_id, request_url, error = gamma_event_cache[event_slug]
        if gamma_event is None:
            updates.append(
                _settlement_refresh_update(
                    record=record,
                    identifiers=identifiers,
                    fetched_at=generated,
                    settlement={
                        "status": "refresh_error",
                        "later_update_required": True,
                        "outcome_logged": False,
                        "source": "polymarket_gamma_event_slug",
                        "request_url": request_url,
                        "error": error,
                    },
                )
            )
            continue
        gamma_market = _find_gamma_market(gamma_event, identifiers)
        if gamma_market is None:
            updates.append(
                _settlement_refresh_update(
                    record=record,
                    identifiers=identifiers,
                    fetched_at=generated,
                    settlement={
                        "status": "refresh_error",
                        "later_update_required": True,
                        "outcome_logged": False,
                        "source": "polymarket_gamma_event_slug",
                        "request_url": request_url,
                        "raw_cache_path": raw_cache_path,
                        "payload_id": payload_id,
                        "reason": "market_not_found_in_gamma_event",
                    },
                )
            )
            continue
        settlement = _settlement_from_gamma_market(
            gamma_market,
            identifiers=identifiers,
            request_url=request_url,
            raw_cache_path=raw_cache_path,
            payload_id=payload_id,
        )
        settlement = _settlement_after_resolved_recheck(
            record=record,
            settlement=settlement,
            refreshed_at=generated,
        )
        updates.append(
            _settlement_refresh_update(
                record=record,
                identifiers=identifiers,
                fetched_at=generated,
                settlement=settlement,
            )
        )
    status_counts = Counter(str((update.get("settlement") or {}).get("status") or "unknown") for update in updates)
    raw_pending_events_scanned = sum(int(record.get("raw_signal_count") or 1) for record in pending_clusters)
    anomalies = [
        update.get("settlement_recheck_anomaly")
        for update in updates
        if isinstance(update.get("settlement_recheck_anomaly"), Mapping)
    ]
    json_path = report_dir / f"{run_id}-old-no-tail-live-forward-shadow-settlement-refresh.json"
    markdown_path = report_dir / f"{run_id}-old-no-tail-live-forward-shadow-settlement-refresh.md"
    anomaly_events_path = report_dir / f"{run_id}-old-no-tail-live-forward-shadow-settlement-refresh-anomalies.jsonl"
    summary: Mapping[str, Any] = {
        "schema_version": SETTLEMENT_REFRESH_SCHEMA_VERSION,
        "run_id": run_id,
        "generated_at_utc": generated.isoformat(),
        "mode": "read_only_live_forward_shadow_settlement_refresh",
        "profile_key": OLD_NO_TAIL_LIVE_FORWARD_SHADOW_PROFILE_KEY,
        "read_only": True,
        "real_order_placement_allowed": False,
        "real_orders_placed": 0,
        "fills_simulated_as_truth": False,
        "historical_pnl_numbers_are_expected_performance": False,
        "resolved_recheck_window_hours": resolved_recheck_window_hours,
        "resolved_recheck_cluster_count": sum(1 for record in pending_clusters if record.get("resolved_recheck")),
        "settlement_recheck_anomaly_count": len(anomalies),
        "settlement_recheck_anomalies": anomalies,
        "source_event_logs_scanned": sorted(
            {
                str(path)
                for record in pending_clusters
                for path in (record.get("source_event_logs") or (record.get("events_path"),))
                if path
            }
        ),
        "pending_would_trade_events_scanned": raw_pending_events_scanned,
        "raw_pending_would_trade_events_scanned": raw_pending_events_scanned,
        "pending_would_trade_clusters_scanned": len(pending_clusters),
        "gamma_events_fetched": sum(
            1
            for payload, _, _, _, error in gamma_event_cache.values()
            if payload is not None and error is None
        ),
        "settlement_status_counts": dict(sorted(status_counts.items())),
        "settlement_cluster_status_counts": dict(sorted(status_counts.items())),
        "resolved_would_trade_signals": int(status_counts.get("resolved") or 0),
        "resolved_would_trade_clusters": int(status_counts.get("resolved") or 0),
        "updates": updates,
        "artifacts": {
            "summary_json": str(json_path),
            "markdown_report": str(markdown_path),
            "anomaly_events_jsonl": str(anomaly_events_path),
            "raw_payload_dir": str(raw_dir),
        },
    }
    with anomaly_events_path.open("w", encoding="utf-8") as handle:
        for anomaly in anomalies:
            event = {"event_type": "settlement_recheck_anomaly", **anomaly}
            handle.write(json.dumps(_json_ready(event), sort_keys=True) + "\n")
    json_path.write_text(json.dumps(_json_ready(summary), indent=2, sort_keys=True), encoding="utf-8")
    markdown_path.write_text(_render_settlement_refresh_markdown(summary), encoding="utf-8")
    return summary


def _pending_would_trade_events(report_dir: Path) -> list[Mapping[str, Any]]:
    records: dict[str, Mapping[str, Any]] = {}
    for events_path in sorted(report_dir.glob("*-old-no-tail-live-forward-shadow-intraday-events.jsonl")):
        try:
            lines = events_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            decision = event.get("decision") if isinstance(event, Mapping) else {}
            if not isinstance(decision, Mapping) or not decision.get("would_trade"):
                continue
            decision_id = str(((event.get("market_discovery") or {}).get("decision_id")) or "")
            if not decision_id:
                continue
            records[decision_id] = {
                "event": event,
                "events_path": events_path,
                "line_number": line_number,
            }
    return [records[key] for key in sorted(records)]


def _pending_would_trade_clusters(
    report_dir: Path,
    *,
    generated_at: Optional[datetime] = None,
    resolved_recheck_window_hours: float = DEFAULT_RESOLVED_RECHECK_WINDOW_HOURS,
) -> list[Mapping[str, Any]]:
    now = generated_at or datetime.now(timezone.utc)
    latest_cluster_states = _latest_settlement_refresh_cluster_states(report_dir)
    clusters: dict[str, dict[str, Any]] = {}
    for record in _pending_would_trade_events(report_dir):
        event = record["event"]
        identifiers = _settlement_identifiers_from_event(event)
        cluster_key = _would_trade_cluster_key(identifiers)
        if not cluster_key:
            decision_id = str(identifiers.get("decision_id") or "")
            cluster_key = f"decision:{decision_id}" if decision_id else ""
        if not cluster_key:
            continue
        latest_state = latest_cluster_states.get(cluster_key)
        event_settlement = event.get("settlement") if isinstance(event.get("settlement"), Mapping) else {}
        effective_state = latest_state if latest_state else event_settlement
        if isinstance(effective_state, Mapping) and effective_state.get("status") == "resolved":
            resolved_at = _resolved_at_from_settlement_state(effective_state)
            if not _within_resolved_recheck_window(
                resolved_at,
                now=now,
                window_hours=resolved_recheck_window_hours,
            ):
                continue
            previous_settlement = {
                "status": effective_state.get("status"),
                "canonical_label": effective_state.get("canonical_label"),
                "payout_for_shadow_side": effective_state.get("payout_for_shadow_side"),
                "source": effective_state.get("source"),
                "reason": effective_state.get("reason"),
                "manual_review_required": effective_state.get("manual_review_required"),
                "later_update_required": effective_state.get("later_update_required"),
                "resolved_at_utc": _iso_or_none(resolved_at),
                "generated_at_utc": effective_state.get("generated_at_utc"),
            }
        else:
            resolved_at = None
            previous_settlement = None
        decision_id = str(identifiers.get("decision_id") or "")
        cluster = clusters.setdefault(
            cluster_key,
            {
                "cluster_key": cluster_key,
                "event": event,
                "identifiers": identifiers,
                "events_path": record.get("events_path"),
                "line_number": record.get("line_number"),
                "first_seen_events_jsonl": str(record.get("events_path")),
                "first_seen_line_number": record.get("line_number"),
                "last_seen_events_jsonl": str(record.get("events_path")),
                "last_seen_line_number": record.get("line_number"),
                "source_event_logs": [],
                "decision_ids": [],
                "raw_signal_count": 0,
                "resolved_recheck": bool(previous_settlement),
                "previous_settlement": previous_settlement,
                "previous_resolved_at_utc": _iso_or_none(resolved_at),
            },
        )
        cluster["event"] = event
        cluster["identifiers"] = identifiers
        cluster["events_path"] = record.get("events_path")
        cluster["line_number"] = record.get("line_number")
        cluster["last_seen_events_jsonl"] = str(record.get("events_path"))
        cluster["last_seen_line_number"] = record.get("line_number")
        if previous_settlement:
            cluster["resolved_recheck"] = True
            cluster["previous_settlement"] = previous_settlement
            cluster["previous_resolved_at_utc"] = _iso_or_none(resolved_at)
        cluster["raw_signal_count"] = int(cluster.get("raw_signal_count") or 0) + 1
        source_logs = cluster.setdefault("source_event_logs", [])
        source_path = str(record.get("events_path"))
        if source_path and source_path not in source_logs:
            source_logs.append(source_path)
        if decision_id:
            decision_ids = cluster.setdefault("decision_ids", [])
            if decision_id not in decision_ids:
                decision_ids.append(decision_id)
    return [clusters[key] for key in sorted(clusters)]


def _resolved_at_from_settlement_state(state: Mapping[str, Any]) -> Optional[datetime]:
    resolved_at = _parse_optional_utc(state.get("resolved_at_utc") or state.get("resolved_at"))
    if resolved_at is not None:
        return resolved_at
    return _parse_optional_utc(state.get("generated_at_utc"))


def _within_resolved_recheck_window(
    resolved_at: Optional[datetime],
    *,
    now: datetime,
    window_hours: float,
) -> bool:
    if resolved_at is None or window_hours <= 0:
        return False
    if resolved_at.tzinfo is None:
        resolved_at = resolved_at.replace(tzinfo=timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    age = now.astimezone(timezone.utc) - resolved_at.astimezone(timezone.utc)
    if age < timedelta(0):
        return True
    return age <= timedelta(hours=window_hours)


def _iso_or_none(value: Optional[datetime]) -> Optional[str]:
    return value.isoformat() if value is not None else None


def _settlement_identifiers_from_event(event: Mapping[str, Any]) -> Mapping[str, Any]:
    live_market = event.get("live_market_metadata") if isinstance(event.get("live_market_metadata"), Mapping) else {}
    market = event.get("market_discovery") if isinstance(event.get("market_discovery"), Mapping) else {}
    parsed = event.get("parsed_contract") if isinstance(event.get("parsed_contract"), Mapping) else {}
    decision = event.get("decision") if isinstance(event.get("decision"), Mapping) else {}
    signal = decision.get("would_trade_signal") if isinstance(decision.get("would_trade_signal"), Mapping) else {}
    market_slug = str(live_market.get("slug") or market.get("slug") or "")
    event_slug = str(live_market.get("event_slug") or "")
    if not event_slug and market_slug:
        event_slug = _infer_event_slug_from_market_slug(market_slug)
    return {
        "decision_id": str(market.get("decision_id") or signal.get("decision_id") or ""),
        "event_slug": event_slug,
        "market_slug": market_slug,
        "market_id": str(live_market.get("market_id") or market.get("market_id") or ""),
        "condition_id": str(live_market.get("condition_id") or market.get("condition_id") or ""),
        "question": str(live_market.get("question") or market.get("question") or ""),
        "city": str(parsed.get("city") or market.get("normalized_city") or live_market.get("city") or ""),
        "market_date_local": str(parsed.get("date") or market.get("market_date_local") or live_market.get("target_date") or ""),
        "bucket_label": str(parsed.get("bucket_label") or market.get("bucket_label") or ""),
        "side": str(signal.get("side") or parsed.get("side") or market.get("side") or "NO").upper(),
        "token_id": str(signal.get("token_id") or market.get("token_id") or parsed.get("no_token_id") or ""),
        "yes_token_id": str(parsed.get("yes_token_id") or market.get("yes_token_id") or ""),
        "no_token_id": str(parsed.get("no_token_id") or market.get("no_token_id") or ""),
    }


def _infer_event_slug_from_market_slug(market_slug: str) -> str:
    return re.sub(
        r"-(?:-?\d+(?:c|f)(?:orbelow|orhigher|orabove)?|-?\d+(?:c|f)?(?:or-below|or-higher|or-above))$",
        "",
        market_slug,
    )


def _fetch_gamma_event_by_slug(
    event_slug: str,
    *,
    raw_dir: Path,
    fetched_at: datetime,
    cache_only: bool,
) -> tuple[Mapping[str, Any], Path, str, str]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    request_url = f"{GAMMA_BASE_URL}/events/slug/{urllib.parse.quote(event_slug)}"
    cache_path = raw_dir / f"gamma-event-{_safe_name(event_slug)}-{fetched_at.strftime('%Y%m%dT%H%M%SZ')}.json"
    payload = _read_cached_or_fetch_json(request_url, cache_path=cache_path, cache_only=cache_only)
    if not isinstance(payload, Mapping):
        raise ValueError("Gamma event response was not a JSON object")
    return payload, cache_path, _payload_id(request_url, payload), request_url


def _find_gamma_market(
    gamma_event: Mapping[str, Any],
    identifiers: Mapping[str, Any],
) -> Optional[Mapping[str, Any]]:
    markets = gamma_event.get("markets")
    if not isinstance(markets, list):
        return None
    condition_id = str(identifiers.get("condition_id") or "")
    market_id = str(identifiers.get("market_id") or "")
    market_slug = str(identifiers.get("market_slug") or "")
    no_token_id = str(identifiers.get("no_token_id") or "")
    yes_token_id = str(identifiers.get("yes_token_id") or "")
    for market in markets:
        if not isinstance(market, Mapping):
            continue
        token_ids = {str(token) for token in parse_jsonish_list(market.get("clobTokenIds") or market.get("clob_token_ids"))}
        if condition_id and str(market.get("conditionId") or market.get("condition_id") or "") == condition_id:
            return market
        if market_id and str(market.get("id") or "") == market_id:
            return market
        if market_slug and str(market.get("slug") or "") == market_slug:
            return market
        if no_token_id and no_token_id in token_ids:
            return market
        if yes_token_id and yes_token_id in token_ids:
            return market
    return None


def _settlement_from_gamma_market(
    gamma_market: Mapping[str, Any],
    *,
    identifiers: Mapping[str, Any],
    request_url: Optional[str],
    raw_cache_path: Optional[str],
    payload_id: Optional[str],
) -> Mapping[str, Any]:
    outcomes = [str(item) for item in parse_jsonish_list(gamma_market.get("outcomes"))]
    prices = [_optional_float(item) for item in parse_jsonish_list(gamma_market.get("outcomePrices") or gamma_market.get("outcome_prices"))]
    closed = _truthy(gamma_market.get("closed"))
    uma_finality_status, uma_finality_reason = _gamma_uma_finality(gamma_market)
    base = {
        "source": "polymarket_gamma_event_slug",
        "request_url": request_url,
        "raw_cache_path": raw_cache_path,
        "payload_id": payload_id,
        "gamma_market_id": str(gamma_market.get("id") or ""),
        "gamma_market_slug": str(gamma_market.get("slug") or ""),
        "condition_id": str(gamma_market.get("conditionId") or gamma_market.get("condition_id") or ""),
        "closed": closed,
        "active": _truthy(gamma_market.get("active")),
        "outcomes": outcomes,
        "outcome_prices": prices,
        "uma_resolution_statuses": parse_jsonish_list(gamma_market.get("umaResolutionStatuses") or gamma_market.get("uma_resolution_statuses")),
        "uma_finality_status": uma_finality_status,
    }
    if not closed:
        return {
            **base,
            "status": "pending",
            "later_update_required": True,
            "outcome_logged": False,
            "reason": "gamma_market_not_closed",
        }
    if uma_finality_status == "disputed":
        return {
            **base,
            "status": "excluded_or_manual_review",
            "later_update_required": True,
            "outcome_logged": False,
            "manual_review_required": True,
            "reason": uma_finality_reason,
        }
    if uma_finality_status == "unfinalized":
        return {
            **base,
            "status": "unresolved_or_ambiguous",
            "later_update_required": True,
            "outcome_logged": False,
            "manual_review_required": True,
            "reason": uma_finality_reason,
        }
    winner_index = _binary_winner_index(prices)
    if winner_index is None or winner_index >= len(outcomes):
        return {
            **base,
            "status": "unresolved_or_ambiguous",
            "later_update_required": True,
            "outcome_logged": False,
            "manual_review_required": True,
            "reason": "gamma_market_closed_without_single_binary_outcome_price",
        }
    canonical_label = _normalise_outcome_label(outcomes[winner_index])
    side = str(identifiers.get("side") or "NO").upper()
    return {
        **base,
        "status": "resolved",
        "later_update_required": False,
        "outcome_logged": True,
        "canonical_label": canonical_label,
        "payout_for_shadow_side": 1 if canonical_label == side else 0,
        "side": side,
        "payout_is_shadow_settlement_only": True,
    }


def _settlement_after_resolved_recheck(
    *,
    record: Mapping[str, Any],
    settlement: Mapping[str, Any],
    refreshed_at: datetime,
) -> Mapping[str, Any]:
    if not record.get("resolved_recheck"):
        return settlement
    previous = record.get("previous_settlement") if isinstance(record.get("previous_settlement"), Mapping) else {}
    previous_label = str(previous.get("canonical_label") or "")
    previous_payout = previous.get("payout_for_shadow_side")
    current_status = str(settlement.get("status") or "unknown")
    current_label = str(settlement.get("canonical_label") or "")
    current_payout = settlement.get("payout_for_shadow_side")
    reason: Optional[str] = None
    if current_status != "resolved":
        reason = "resolved_cluster_recheck_no_longer_final"
    elif previous_label and current_label and previous_label != current_label:
        reason = "resolved_cluster_recheck_winning_outcome_changed"
    elif (
        previous_payout in (0, 1, False, True)
        and current_payout in (0, 1, False, True)
        and int(previous_payout) != int(current_payout)
    ):
        reason = "resolved_cluster_recheck_shadow_payout_changed"
    if reason is None:
        return {
            **settlement,
            "resolved_at_utc": (
                previous.get("resolved_at_utc")
                or previous.get("resolved_at")
                or settlement.get("resolved_at_utc")
                or refreshed_at.isoformat()
            ),
            "post_resolution_recheck": True,
            "post_resolution_recheck_result": "still_final_same_winner",
            "previous_canonical_label": previous.get("canonical_label"),
            "previous_payout_for_shadow_side": previous.get("payout_for_shadow_side"),
        }
    anomaly = {
        "severity": "critical",
        "anomaly_type": reason,
        "cluster_key": record.get("cluster_key"),
        "decision_ids": list(record.get("decision_ids") or []),
        "detected_at_utc": refreshed_at.isoformat(),
        "previous_status": previous.get("status"),
        "previous_canonical_label": previous.get("canonical_label"),
        "previous_payout_for_shadow_side": previous.get("payout_for_shadow_side"),
        "previous_resolved_at_utc": previous.get("resolved_at_utc") or previous.get("resolved_at"),
        "current_status": current_status,
        "current_canonical_label": settlement.get("canonical_label"),
        "current_payout_for_shadow_side": settlement.get("payout_for_shadow_side"),
        "current_uma_finality_status": settlement.get("uma_finality_status"),
        "current_reason": settlement.get("reason"),
        "gate_action": "transition_to_excluded_or_manual_review",
    }
    return {
        **settlement,
        "status": "excluded_or_manual_review",
        "later_update_required": True,
        "outcome_logged": False,
        "manual_review_required": True,
        "reason": reason,
        "post_resolution_recheck": True,
        "post_resolution_recheck_result": "anomaly_excluded_from_gate",
        "settlement_recheck_anomaly": anomaly,
        "previous_canonical_label": previous.get("canonical_label"),
        "previous_payout_for_shadow_side": previous.get("payout_for_shadow_side"),
        "previous_resolved_at_utc": previous.get("resolved_at_utc") or previous.get("resolved_at"),
        "resolved_at_utc": previous.get("resolved_at_utc") or previous.get("resolved_at"),
    }


def _gamma_uma_finality(gamma_market: Mapping[str, Any]) -> tuple[str, str]:
    statuses = [
        str(item).strip().lower()
        for item in parse_jsonish_list(
            gamma_market.get("umaResolutionStatuses")
            or gamma_market.get("uma_resolution_statuses")
            or gamma_market.get("umaResolutionStatus")
            or gamma_market.get("uma_resolution_status")
        )
        if str(item).strip()
    ]
    status_text = " ".join(statuses)
    disputed_flags = (
        gamma_market.get("disputed"),
        gamma_market.get("isDisputed"),
        gamma_market.get("is_disputed"),
        gamma_market.get("umaDisputed"),
        gamma_market.get("uma_disputed"),
    )
    if any(_truthy(flag) for flag in disputed_flags):
        return "disputed", "gamma_uma_dispute_flag_present"
    if any(token in status_text for token in ("dispute", "challenged", "challenge")):
        return "disputed", "gamma_uma_resolution_disputed"
    if statuses:
        final_markers = ("resolved", "finalized", "finalised", "final", "settled")
        if all(any(marker in status for marker in final_markers) for status in statuses):
            return "final", "gamma_uma_resolution_final"
        return "unfinalized", "gamma_uma_resolution_status_not_final"
    return "unknown_not_reported", "gamma_uma_resolution_status_not_reported"


def _settlement_refresh_update(
    *,
    record: Mapping[str, Any],
    identifiers: Mapping[str, Any],
    fetched_at: datetime,
    settlement: Mapping[str, Any],
) -> Mapping[str, Any]:
    event = record["event"]
    decision = event.get("decision") if isinstance(event.get("decision"), Mapping) else {}
    signal = decision.get("would_trade_signal") if isinstance(decision.get("would_trade_signal"), Mapping) else {}
    quote = event.get("quote") if isinstance(event.get("quote"), Mapping) else {}
    decision_ids = [str(item) for item in (record.get("decision_ids") or []) if str(item)]
    if not decision_ids and identifiers.get("decision_id"):
        decision_ids = [str(identifiers.get("decision_id"))]
    cluster_key = str(record.get("cluster_key") or _would_trade_cluster_key(identifiers) or "")
    settlement_payload = dict(settlement)
    if settlement_payload.get("status") == "resolved" and not (
        settlement_payload.get("resolved_at_utc") or settlement_payload.get("resolved_at")
    ):
        settlement_payload["resolved_at_utc"] = fetched_at.isoformat()
    anomaly = settlement_payload.get("settlement_recheck_anomaly")
    return {
        "decision_id": identifiers.get("decision_id"),
        "decision_ids": decision_ids,
        "cluster_key": cluster_key,
        "raw_signal_count": int(record.get("raw_signal_count") or 1),
        "first_seen_events_jsonl": record.get("first_seen_events_jsonl") or str(record.get("events_path")),
        "first_seen_line_number": record.get("first_seen_line_number") or record.get("line_number"),
        "last_seen_events_jsonl": record.get("last_seen_events_jsonl") or str(record.get("events_path")),
        "last_seen_line_number": record.get("last_seen_line_number") or record.get("line_number"),
        "refreshed_at_utc": fetched_at.isoformat(),
        "source_events_jsonl": str(record.get("events_path")),
        "source_line_number": record.get("line_number"),
        "market": dict(identifiers),
        "original_shadow_signal": {
            "would_trade": bool(decision.get("would_trade")),
            "decision_reason": decision.get("decision_reason"),
            "requested_notional_usd": signal.get("requested_notional_usd") if isinstance(signal, Mapping) else None,
            "limit_price": signal.get("limit_price") if isinstance(signal, Mapping) else None,
            "token_id": signal.get("token_id") if isinstance(signal, Mapping) else None,
            "side": signal.get("side") if isinstance(signal, Mapping) else identifiers.get("side"),
        },
        "observed_book_at_shadow_decision": {
            "bid": quote.get("bid"),
            "ask": quote.get("ask"),
            "spread": quote.get("spread"),
            "top_of_book_depth": quote.get("top_of_book_depth"),
            "timestamp_utc": quote.get("timestamp_utc"),
            "source_payload_id": quote.get("source_payload_id"),
        },
        "settlement": settlement_payload,
        **({"settlement_recheck_anomaly": anomaly} if isinstance(anomaly, Mapping) else {}),
    }


def _binary_winner_index(prices: Sequence[Optional[float]]) -> Optional[int]:
    if not prices:
        return None
    winners = [
        index
        for index, price in enumerate(prices)
        if price is not None and price >= 0.99
    ]
    if len(winners) != 1:
        return None
    winner = winners[0]
    for index, price in enumerate(prices):
        if index == winner:
            continue
        if price is None or price > 0.01:
            return None
    return winner


def _normalise_outcome_label(value: str) -> str:
    lowered = value.strip().lower()
    if lowered == "yes":
        return "YES"
    if lowered == "no":
        return "NO"
    return value.strip()


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def wilson_interval(successes: int, trials: int, *, confidence: float) -> Mapping[str, Any]:
    z_by_confidence = {
        0.95: WILSON_Z_95,
        0.80: WILSON_Z_80,
    }
    z = z_by_confidence.get(round(float(confidence), 2))
    if z is None:
        raise ValueError("unsupported Wilson confidence; expected 0.95 or 0.80")
    lower, upper = _wilson_bounds(successes, trials, z=z)
    return {
        "confidence": confidence,
        "method": f"wilson_{int(round(confidence * 100))}pct",
        "successes": int(successes),
        "trials": int(trials),
        "hit_rate": _rate(successes, trials),
        "lower": lower,
        "upper": upper,
    }


def _wilson_bounds(successes: int, trials: int, *, z: float) -> tuple[Optional[float], Optional[float]]:
    successes = int(successes)
    trials = int(trials)
    if trials <= 0:
        return None, None
    successes = max(0, min(successes, trials))
    phat = successes / trials
    denominator = 1.0 + (z * z / trials)
    center = phat + (z * z / (2.0 * trials))
    margin = z * math.sqrt((phat * (1.0 - phat) + (z * z / (4.0 * trials))) / trials)
    return (
        max(0.0, (center - margin) / denominator),
        min(1.0, (center + margin) / denominator),
    )


def _rate(successes: int, trials: int) -> Optional[float]:
    return None if trials <= 0 else successes / trials


def _evaluate_forward_shadow_gate(
    clusters: Sequence[Mapping[str, Any]],
    *,
    forward_log_days: int = 0,
) -> Mapping[str, Any]:
    resolved = [
        cluster
        for cluster in clusters
        if _gate_valid_resolved_cluster(cluster)
    ]
    trials = len(resolved)
    successes = sum(1 for cluster in resolved if int(cluster.get("payout_for_shadow_side") or 0) == 1)
    records = [_economic_record_for_cluster(cluster) for cluster in resolved]
    valid_records = [record for record in records if record["economically_evaluable"]]
    invalid_records = [record for record in records if not record["economically_evaluable"]]
    filled_records = [record for record in valid_records if record["execution_status"] == "venue_executable"]
    pnl_values = [float(record["net_pnl_usd"]) for record in valid_records]
    uncertainty = _deterministic_cluster_bootstrap_mean_interval(pnl_values)
    filled_successes = sum(1 for record in filled_records if int(record["payout_for_shadow_side"]) == 1)
    filled_trials = len(filled_records)
    wilson_95 = wilson_interval(filled_successes, filled_trials, confidence=0.95)
    wilson_80 = wilson_interval(filled_successes, filled_trials, confidence=0.80)
    total_filled_shares = sum(float(record["shares"]) for record in filled_records)
    total_executable_cost = sum(float(record["entry_cost_including_fees_usd"]) for record in filled_records)
    break_even_hit_rate = (
        total_executable_cost / total_filled_shares
        if total_filled_shares > 0.0
        else None
    )
    uncertainty_lower = uncertainty.get("lower_mean_net_pnl_usd")
    uncertainty_upper = uncertainty.get("upper_mean_net_pnl_usd")
    if trials < FORWARD_SHADOW_GATE_MIN_RESOLVED_CLUSTERS:
        status = "not_enough_resolved_clusters"
        reason = (
            "requires at least "
            f"{FORWARD_SHADOW_GATE_MIN_RESOLVED_CLUSTERS} resolved deduped would-trade clusters"
        )
    elif invalid_records:
        status = "invalid_economic_inputs"
        reason = "one_or_more_resolved_clusters_lack_fail_closed_venue_execution_economics"
    elif uncertainty_upper is not None and float(uncertainty_upper) < 0.0:
        status = "failed"
        reason = "deterministic_95pct_cluster_bootstrap_upper_mean_net_pnl_below_zero"
    elif uncertainty_lower is not None and float(uncertainty_lower) > 0.0:
        status = "supportive"
        reason = "deterministic_95pct_cluster_bootstrap_lower_mean_net_pnl_above_zero"
    else:
        status = "inconclusive"
        reason = "deterministic_95pct_cluster_bootstrap_mean_net_pnl_interval_includes_zero"
    production_count_met = trials >= FORWARD_SHADOW_PRODUCTION_MIN_RESOLVED_CLUSTERS
    production_days_met = int(forward_log_days) >= FORWARD_SHADOW_PRODUCTION_MIN_VALID_DAYS
    production_economics_met = status == "supportive"
    production_evidence_met = production_count_met and production_days_met and production_economics_met
    hit_rate_vs_break_even = "not_evaluable"
    if break_even_hit_rate is not None and wilson_95["lower"] is not None and wilson_95["upper"] is not None:
        if float(wilson_95["lower"]) > break_even_hit_rate:
            hit_rate_vs_break_even = "wilson_95_lower_above_price_weighted_break_even"
        elif float(wilson_95["upper"]) < break_even_hit_rate:
            hit_rate_vs_break_even = "wilson_95_upper_below_price_weighted_break_even"
        else:
            hit_rate_vs_break_even = "wilson_95_interval_crosses_price_weighted_break_even"
    return {
        "schema_version": "dailyweather.live_forward_shadow.forward_economic_gate.v2",
        "status": status,
        "reason": reason,
        "minimum_resolved_cluster_count": FORWARD_SHADOW_GATE_MIN_RESOLVED_CLUSTERS,
        "gate_count_basis": "deduped_true_opportunity_clusters",
        "primary_statistic": "mean_fee_adjusted_conservative_executable_net_pnl_usd_per_resolved_opportunity_cluster",
        "resolved_cluster_count": trials,
        "winning_cluster_count": successes,
        "losing_cluster_count": max(0, trials - successes),
        "economically_evaluable_cluster_count": len(valid_records),
        "invalid_economic_cluster_count": len(invalid_records),
        "economic_input_status_counts": dict(sorted(Counter(str(record["execution_status"]) for record in records).items())),
        "invalid_economic_cluster_reasons": dict(
            sorted(Counter(str(record["reason"]) for record in invalid_records).items())
        ),
        "venue_executable_cluster_count": filled_trials,
        "no_fill_cluster_count": sum(1 for record in valid_records if record["execution_status"] == "no_fill"),
        "ineligible_cluster_count": sum(1 for record in valid_records if record["execution_status"] == "ineligible"),
        "total_venue_executable_shares": total_filled_shares,
        "total_entry_cost_including_fees_usd": total_executable_cost,
        "total_fee_usd": sum(float(record["fee_usd"]) for record in filled_records),
        "total_net_pnl_usd": sum(pnl_values),
        "mean_net_pnl_usd_per_resolved_opportunity_cluster": (
            sum(pnl_values) / len(pnl_values) if pnl_values else None
        ),
        "economic_uncertainty": uncertainty,
        "fee_policy": {
            "required_fee_model": FEE_MODEL_POLYMARKET_WEATHER_TAKER,
            "formula": "shares * fee_rate * price * (1 - price)",
            "recorded_weather_fee_rate_required_per_cluster": True,
            "missing_or_invalid_fee_rate_fails_closed": True,
        },
        "conservative_price_policy": (
            "for BUY signals use the highest valid recorded entry, worst-price, and latency/delayed price; "
            "explicit no-fill and venue-ineligible opportunities contribute zero PnL rather than being dropped"
        ),
        "secondary_hit_rate_diagnostics_only": True,
        "hit_rate": _rate(filled_successes, filled_trials),
        "all_resolved_opportunity_outcome_hit_rate": _rate(successes, trials),
        "price_weighted_break_even_hit_rate": break_even_hit_rate,
        "hit_rate_wilson_95_vs_price_weighted_break_even": hit_rate_vs_break_even,
        "wilson_95": dict(wilson_95),
        "wilson_80": dict(wilson_80),
        "checkpoint": {
            "minimum_resolved_clusters": FORWARD_SHADOW_GATE_MIN_RESOLVED_CLUSTERS,
            "reached": trials >= FORWARD_SHADOW_GATE_MIN_RESOLVED_CLUSTERS,
            "purpose": "first_economic_evidence_review_only",
            "grants_live_permission": False,
        },
        "production_candidate_requirements": {
            "minimum_resolved_clusters": FORWARD_SHADOW_PRODUCTION_MIN_RESOLVED_CLUSTERS,
            "minimum_valid_calendar_days": FORWARD_SHADOW_PRODUCTION_MIN_VALID_DAYS,
            "observed_valid_calendar_days": int(forward_log_days),
            "resolved_cluster_count_met": production_count_met,
            "valid_calendar_days_met": production_days_met,
            "economic_uncertainty_gate_met": production_economics_met,
            "evidence_requirements_met": production_evidence_met,
            "status": (
                "evidence_requirements_met_manual_review_still_required"
                if production_evidence_met
                else "not_yet_met"
            ),
            "live_permission_granted": False,
        },
        "live_permission_granted": False,
        "economic_records_preview": valid_records[:10],
    }


def _deterministic_cluster_bootstrap_mean_interval(
    values: Sequence[float],
    *,
    confidence: float = FORWARD_ECONOMIC_BOOTSTRAP_CONFIDENCE,
    resamples: int = FORWARD_ECONOMIC_BOOTSTRAP_RESAMPLES,
    seed: int = FORWARD_ECONOMIC_BOOTSTRAP_SEED,
) -> Mapping[str, Any]:
    """Return a reproducible percentile interval for the mean cluster PnL.

    Sampling uses a fixed 32-bit linear-congruential generator rather than the
    process-global PRNG, so identical cluster PnLs produce identical bounds.
    The opportunity cluster is the resampling unit; no poll-level duplicates
    enter the uncertainty calculation.
    """

    cleaned = [float(value) for value in values if math.isfinite(float(value))]
    if not cleaned:
        return {
            "method": "deterministic_nonparametric_cluster_bootstrap_percentile",
            "confidence": confidence,
            "resamples": resamples,
            "seed": seed,
            "cluster_count": 0,
            "lower_mean_net_pnl_usd": None,
            "upper_mean_net_pnl_usd": None,
        }
    if not 0.0 < confidence < 1.0:
        raise ValueError("bootstrap confidence must be between zero and one")
    if resamples <= 0:
        raise ValueError("bootstrap resamples must be positive")
    sample_count = len(cleaned)
    state = int(seed) & 0xFFFFFFFF
    means: list[float] = []
    for _ in range(int(resamples)):
        sample_total = 0.0
        for _ in range(sample_count):
            state = (1_664_525 * state + 1_013_904_223) & 0xFFFFFFFF
            sample_total += cleaned[state % sample_count]
        means.append(sample_total / sample_count)
    means.sort()
    alpha = (1.0 - confidence) / 2.0
    return {
        "method": "deterministic_nonparametric_cluster_bootstrap_percentile",
        "resampling_unit": "deduped_resolved_opportunity_cluster",
        "prng": "fixed_seed_32bit_lcg_1664525_1013904223",
        "confidence": confidence,
        "resamples": int(resamples),
        "seed": int(seed),
        "cluster_count": sample_count,
        "lower_mean_net_pnl_usd": _sorted_percentile(means, alpha),
        "upper_mean_net_pnl_usd": _sorted_percentile(means, 1.0 - alpha),
    }


def _sorted_percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("cannot calculate a percentile of an empty sequence")
    position = max(0.0, min(1.0, float(quantile))) * (len(values) - 1)
    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))
    if lower_index == upper_index:
        return float(values[lower_index])
    weight = position - lower_index
    return float(values[lower_index]) * (1.0 - weight) + float(values[upper_index]) * weight


def _economic_record_for_cluster(cluster: Mapping[str, Any]) -> Mapping[str, Any]:
    observation = cluster.get("economic_observation")
    if not isinstance(observation, Mapping):
        observation = cluster
    cluster_key = str(cluster.get("cluster_key") or "")
    payout = cluster.get("payout_for_shadow_side")
    status = _normalise_shadow_execution_status(observation.get("execution_status"))
    side = str(observation.get("side") or "").upper()
    if side not in {"YES", "NO"}:
        return _invalid_economic_record(cluster_key, "missing_or_invalid_recorded_shadow_side")
    order_type = str(observation.get("order_type") or "").upper()
    order_action = str(observation.get("order_action") or "").upper()
    if not order_type:
        return _invalid_economic_record(cluster_key, "missing_recorded_order_type")
    if order_action != "BUY":
        return _invalid_economic_record(cluster_key, "missing_or_unsupported_recorded_order_action")
    venue_assessment = observation.get("venue_execution_assessment")
    if not isinstance(venue_assessment, Mapping):
        return _invalid_economic_record(cluster_key, "missing_shared_venue_execution_assessment")
    assessment_error = _venue_assessment_error(
        observation,
        venue_assessment,
        execution_status=status,
    )
    if assessment_error is not None:
        return _invalid_economic_record(cluster_key, assessment_error)
    if status in {"no_fill", "ineligible"}:
        return {
            "cluster_key": cluster_key,
            "economically_evaluable": True,
            "execution_status": status,
            "reason": str(observation.get("execution_reason") or status),
            "payout_for_shadow_side": int(bool(payout)),
            "side": side,
            "order_type": order_type,
            "order_action": order_action,
            "shares": 0.0,
            "entry_price": None,
            "conservative_entry_price": None,
            "fee_rate": _optional_float(observation.get("fee_rate")),
            "fee_usd": 0.0,
            "entry_cost_including_fees_usd": 0.0,
            "payout_usd": 0.0,
            "net_pnl_usd": 0.0,
            "venue_execution_assessment": dict(venue_assessment),
        }
    if status != "venue_executable":
        return _invalid_economic_record(cluster_key, status or "unknown_execution_status")
    entry_price = _optional_float(observation.get("entry_price"))
    conservative_price = _optional_float(observation.get("conservative_entry_price"))
    if entry_price is None or not 0.0 < entry_price < 1.0:
        return _invalid_economic_record(cluster_key, "missing_or_invalid_recorded_entry_price")
    if conservative_price is None:
        conservative_price = entry_price
    if not 0.0 < conservative_price < 1.0:
        return _invalid_economic_record(cluster_key, "missing_or_invalid_conservative_entry_price")
    conservative_price = max(entry_price, conservative_price)
    shares = _optional_float(observation.get("shares"))
    notional = _optional_float(observation.get("notional_usd"))
    if shares is None and notional is not None and notional > 0.0:
        shares = notional / conservative_price
    if shares is None or shares <= 0.0:
        return _invalid_economic_record(cluster_key, "missing_or_invalid_venue_executable_size")
    minimum_shares = _optional_float(observation.get("min_order_size_shares"))
    if minimum_shares is None or minimum_shares <= 0.0:
        return _invalid_economic_record(cluster_key, "missing_or_invalid_venue_minimum_shares")
    if shares + 1e-12 < minimum_shares:
        minimum_rule = str(venue_assessment.get("minimum_size_rule_status") or "")
        if not (
            order_type == "FOK"
            and order_action == "BUY"
            and minimum_rule
            == "not_applied_v2_market_buy_amount_is_usdc_and_subminimum_match_observed"
        ):
            return _invalid_economic_record(
                cluster_key,
                "recorded_subminimum_size_lacks_shared_fok_buy_exemption",
            )
    available_size = _optional_float(observation.get("available_size_at_price"))
    if available_size is None or available_size < 0.0:
        return _invalid_economic_record(cluster_key, "missing_or_invalid_recorded_venue_depth")
    if available_size + 1e-12 < shares:
        return _invalid_economic_record(cluster_key, "recorded_size_exceeds_depth_without_no_fill_status")
    fee_rate = _optional_float(observation.get("fee_rate"))
    if fee_rate is None:
        return _invalid_economic_record(cluster_key, "missing_recorded_weather_fee_rate")
    if fee_rate < 0.0:
        return _invalid_economic_record(cluster_key, "invalid_negative_weather_fee_rate")
    tick_size = _optional_float(observation.get("tick_size"))
    if tick_size is None or tick_size <= 0.0:
        return _invalid_economic_record(cluster_key, "missing_or_invalid_recorded_tick_size")
    execution_proof_assessment = str(
        venue_assessment.get("execution_basis") or ""
    ) in {
        "top_level_proves_full_fill",
        "full_order_book_fok_buy_usdc_walk",
    }
    if execution_proof_assessment:
        if venue_assessment.get("all_or_none_fill") is not True:
            return _invalid_economic_record(
                cluster_key,
                "full_book_execution_missing_all_or_none_fill_proof",
            )
        fee_usd = _optional_float(venue_assessment.get("fee_usd"))
        entry_cost = _optional_float(venue_assessment.get("maximum_loss_usd"))
        if fee_usd is None or entry_cost is None:
            return _invalid_economic_record(
                cluster_key,
                "full_book_execution_missing_fee_or_maximum_loss",
            )
    else:
        fee_usd = calculate_fill_fee_usd(
            shares=shares,
            price=conservative_price,
            fee_config=FeeModelConfig(
                fee_model=FEE_MODEL_POLYMARKET_WEATHER_TAKER,
                fee_rate=fee_rate,
            ),
        )
        entry_cost = shares * conservative_price + fee_usd
    payout_usd = shares * int(bool(payout))
    return {
        "cluster_key": cluster_key,
        "economically_evaluable": True,
        "execution_status": "venue_executable",
        "reason": str(observation.get("execution_reason") or "recorded_venue_executable_shadow_entry"),
        "payout_for_shadow_side": int(bool(payout)),
        "side": side,
        "order_type": order_type,
        "order_action": order_action,
        "shares": shares,
        "min_order_size_shares": minimum_shares,
        "tick_size": tick_size,
        "available_size_at_price": available_size,
        "entry_price": entry_price,
        "conservative_entry_price": conservative_price,
        "conservative_price_applied": conservative_price > entry_price + 1e-12,
        "fee_rate": fee_rate,
        "fee_usd": fee_usd,
        "entry_cost_including_fees_usd": entry_cost,
        "payout_usd": payout_usd,
        "net_pnl_usd": payout_usd - entry_cost,
        "venue_execution_assessment": dict(venue_assessment),
        "execution_basis": venue_assessment.get("execution_basis"),
        "maximum_loss_semantics": venue_assessment.get("maximum_loss_semantics"),
    }


def _venue_assessment_error(
    observation: Mapping[str, Any],
    assessment: Mapping[str, Any],
    *,
    execution_status: str,
) -> Optional[str]:
    if str(assessment.get("order_type") or "").upper() != str(observation.get("order_type") or "").upper():
        return "shared_venue_assessment_order_type_mismatch"
    if str(assessment.get("side") or "").upper() != str(observation.get("order_action") or "").upper():
        return "shared_venue_assessment_order_action_mismatch"
    required_fields = {
        "min_order_size_shares": observation.get("min_order_size_shares"),
        "tick_size": observation.get("tick_size"),
        "fee_rate": observation.get("fee_rate"),
        "available_size_at_price": observation.get("available_size_at_price"),
    }
    for field_name, value in required_fields.items():
        parsed = _optional_float(value)
        if parsed is None:
            return f"missing_recorded_{field_name}"
        if field_name == "fee_rate":
            if parsed < 0.0:
                return "invalid_recorded_fee_rate"
        elif field_name == "available_size_at_price":
            if parsed < 0.0:
                return "invalid_recorded_available_size_at_price"
        elif parsed <= 0.0:
            return f"invalid_recorded_{field_name}"
    execution_proof_assessment = str(
        assessment.get("execution_basis") or ""
    ) in {
        "top_level_proves_full_fill",
        "full_order_book_fok_buy_usdc_walk",
    }
    numeric_matches = (
        ("intended_shares", observation.get("shares")),
        (
            "limit_price",
            observation.get("limit_price")
            if execution_proof_assessment
            else observation.get("conservative_entry_price"),
        ),
        ("available_depth_shares", observation.get("available_size_at_price")),
    )
    if execution_proof_assessment:
        numeric_matches = (
            *numeric_matches,
            ("weighted_average_price", observation.get("conservative_entry_price")),
        )
    for assessment_field, observation_value in numeric_matches:
        assessed = _optional_float(assessment.get(assessment_field))
        observed = _optional_float(observation_value)
        if assessed is None or observed is None or abs(assessed - observed) > 1e-8:
            return f"shared_venue_assessment_{assessment_field}_mismatch"
    reasons = {str(reason) for reason in assessment.get("reasons") or []}
    fatal_reasons = {
        "missing_venue_metadata",
        "missing_venue_min_order_size",
        "invalid_venue_min_order_size",
        "missing_venue_tick_size",
        "invalid_venue_tick_size",
        "missing_venue_fee_rate",
        "invalid_venue_fee_rate",
        "missing_venue_depth",
        "invalid_venue_depth",
        "invalid_intended_shares",
        "invalid_limit_price",
        "order_price_off_tick",
        "unverified_venue_minimum_size_semantics",
        "fak_partial_execution_unimplemented",
        "missing_full_order_book_levels",
        "full_order_book_completeness_unproven",
        "invalid_full_book_level",
        "full_book_level_price_off_tick",
        "invalid_intended_notional_usd",
        "missing_current_market_fee_source",
        "missing_fee_rate_source",
        "non_current_market_fee_source",
        "official_weather_fee_schedule_rate_mismatch",
        "fee_source_not_current_market_info",
        "missing_top_level_provenance",
        "invalid_execution_quote_timestamp",
        "missing_top_level_fill_proof_fields",
        "invalid_top_level_ask_size",
        "invalid_top_level_ask_price",
        "top_level_ask_price_off_tick",
    }
    fatal = sorted(reasons.intersection(fatal_reasons))
    if fatal:
        return f"shared_venue_assessment_failed:{','.join(fatal)}"
    if execution_status == "venue_executable":
        if assessment.get("eligible") is not True:
            return "shared_venue_assessment_not_eligible_for_recorded_execution"
        if reasons:
            return "eligible_shared_venue_assessment_contains_block_reasons"
    elif execution_status == "no_fill":
        derived_full_book_no_fill = (
            assessment.get("full_book_enforcement_requested") is True
            and reasons.issubset(
                {
                    "insufficient_full_book_depth_within_limit",
                    "no_full_book_ask_within_limit",
                    "top_ask_above_worst_price_limit",
                }
            )
        )
        if (
            observation.get("execution_status_explicit") is not True
            and not derived_full_book_no_fill
        ):
            return "no_fill_status_was_not_explicitly_recorded"
        if (
            assessment.get("eligible") is not True
            and reasons != {"insufficient_venue_depth"}
            and not derived_full_book_no_fill
        ):
            return "no_fill_lacks_valid_shared_venue_assessment"
    elif execution_status == "ineligible":
        if observation.get("execution_status_explicit") is not True:
            return "ineligible_status_was_not_explicitly_recorded"
        if assessment.get("eligible") is True or not reasons:
            return "ineligible_status_conflicts_with_shared_venue_assessment"
        allowed_ineligible_reasons = {
            "insufficient_venue_depth",
            "venue_order_notional_mismatch",
            "venue_notional_cap_exceeded",
            "venue_maximum_loss_cap_exceeded",
            "insufficient_full_book_depth_within_limit",
            "no_full_book_ask_within_limit",
            "top_ask_above_worst_price_limit",
        }
        if not reasons.issubset(allowed_ineligible_reasons):
            return "ineligible_status_has_unapproved_shared_venue_reason"
    return None


def _invalid_economic_record(cluster_key: str, reason: str) -> Mapping[str, Any]:
    return {
        "cluster_key": cluster_key,
        "economically_evaluable": False,
        "execution_status": "invalid_economic_inputs",
        "reason": reason,
        "net_pnl_usd": None,
    }


def _normalise_shadow_execution_status(value: Any) -> str:
    status = str(value or "").strip().lower().replace("-", "_")
    if status in {
        "venue_executable",
        "executable",
        "assumed_executable",
        "filled",
        "hypothetical_fill",
    }:
        return "venue_executable"
    if status in {
        "no_fill",
        "unfilled",
        "fok_unfilled",
        "latency_no_fill",
        "insufficient_depth_no_fill",
    }:
        return "no_fill"
    if status in {
        "ineligible",
        "venue_ineligible",
        "below_minimum",
        "below_min_order_size",
        "invalid_tick",
    }:
        return "ineligible"
    return status or "unknown"


def _economic_observation_from_event(event: Mapping[str, Any]) -> Mapping[str, Any]:
    decision = event.get("decision") if isinstance(event.get("decision"), Mapping) else {}
    signal = decision.get("would_trade_signal") if isinstance(decision.get("would_trade_signal"), Mapping) else {}
    signal_metadata = signal.get("metadata") if isinstance(signal.get("metadata"), Mapping) else {}
    assumed = event.get("assumed_executable_price") if isinstance(event.get("assumed_executable_price"), Mapping) else {}
    explicit = event.get("execution_economics") if isinstance(event.get("execution_economics"), Mapping) else {}
    if not explicit and isinstance(decision.get("execution_economics"), Mapping):
        explicit = decision.get("execution_economics") or {}
    quote = event.get("quote") if isinstance(event.get("quote"), Mapping) else {}
    market = event.get("live_market_metadata") if isinstance(event.get("live_market_metadata"), Mapping) else {}
    discovery = event.get("market_discovery") if isinstance(event.get("market_discovery"), Mapping) else {}
    side = str(
        explicit.get("side")
        or assumed.get("side")
        or signal.get("side")
        or discovery.get("side")
        or ""
    ).upper()
    order_type = str(
        explicit.get("order_type")
        or signal.get("order_type")
        or signal_metadata.get("order_type")
        or ""
    ).upper()
    order_action = str(
        explicit.get("order_action")
        or explicit.get("action")
        or signal.get("action")
        or ""
    ).upper()
    observed_books = event.get("observed_order_books") if isinstance(event.get("observed_order_books"), Mapping) else {}
    side_book = observed_books.get(side) if isinstance(observed_books.get(side), Mapping) else {}
    entry_price = _first_optional_float(
        (explicit, assumed, signal, quote),
        ("entry_price", "price", "limit_price", "ask"),
    )
    candidate_prices = [entry_price]
    for source in (explicit, assumed, signal, signal_metadata, quote):
        for key in (
            "conservative_entry_price",
            "worst_executable_price",
            "worse_entry_price",
            "worse_price",
            "latency_entry_price",
            "delayed_entry_price",
        ):
            candidate_prices.append(_optional_float(source.get(key)))
    valid_prices = [price for price in candidate_prices if price is not None and 0.0 < price < 1.0]
    conservative_price = max(valid_prices) if valid_prices else None
    shares = _first_optional_float(
        (explicit, signal, signal_metadata),
        ("venue_executable_shares", "executed_shares", "requested_shares", "shares"),
    )
    notional = _first_optional_float(
        (explicit, signal, signal_metadata),
        ("venue_executable_notional_usd", "executed_notional_usd", "requested_notional_usd", "notional_usd"),
    )
    size_price = conservative_price or entry_price
    if shares is None and notional is not None and size_price is not None and size_price > 0.0:
        shares = notional / size_price
    minimum_shares = _first_optional_float(
        (explicit, assumed, market, side_book),
        ("min_order_size_shares", "min_order_size", "minimum_order_size"),
    )
    tick_size = _first_optional_float(
        (explicit, assumed, market, side_book),
        ("tick_size", "minimum_tick_size"),
    )
    available_size = _first_optional_float(
        (explicit, assumed, side_book, quote.get("top_of_book_depth") or {}),
        ("available_size_at_price", "visible_size_at_price", "ask_size"),
    )
    fee_rate = _first_optional_float(
        (explicit, assumed, market, side_book, signal_metadata),
        ("weather_taker_fee_rate", "fee_rate"),
    )
    fee_rate_source = str(
        explicit.get("fee_rate_source")
        or market.get("fee_rate_source")
        or side_book.get("fee_rate_source")
        or signal_metadata.get("fee_rate_source")
        or ""
    ) or None
    raw_ask_levels = explicit.get("ask_levels")
    if not isinstance(raw_ask_levels, (list, tuple)):
        raw_ask_levels = side_book.get("ask_levels")
    ask_levels = (
        tuple(dict(item) for item in raw_ask_levels if isinstance(item, Mapping))
        if isinstance(raw_ask_levels, (list, tuple))
        else ()
    )
    full_book_complete = _optional_bool(
        explicit.get("full_book_complete")
        if explicit.get("full_book_complete") is not None
        else side_book.get("full_book_complete")
    )
    require_full_book = bool(
        _optional_bool(
            explicit.get("require_full_book_execution")
            if explicit.get("require_full_book_execution") is not None
            else signal_metadata.get("require_full_book_execution")
        )
    )
    best_ask_price = _first_optional_float(
        (explicit, side_book, quote),
        ("best_ask", "ask", "entry_price"),
    )
    top_level_provenance = str(
        explicit.get("book_provenance")
        or side_book.get("payload_id")
        or side_book.get("raw_cache_path")
        or quote.get("source_payload_id")
        or quote.get("source_cache_path")
        or quote.get("payload_id")
        or ""
    ) or None
    quote_timestamp_raw = (
        explicit.get("quote_timestamp_utc")
        or side_book.get("fetched_at_utc")
        or quote.get("timestamp_utc")
    )
    decision_timestamp_raw = (
        explicit.get("decision_time_utc")
        or decision.get("decision_time_utc")
        or discovery.get("decision_time_utc")
        or event.get("collected_at_utc")
    )
    quote_timestamp = _parse_optional_utc(quote_timestamp_raw)
    decision_timestamp = _parse_optional_utc(decision_timestamp_raw)
    recorded_max_quote_age_seconds = _first_optional_float(
        (explicit, signal_metadata, event.get("execution_policy") or {}),
        ("max_quote_age_seconds", "stale_quote_max_age_seconds"),
    )
    quote_age_seconds = (
        (decision_timestamp - quote_timestamp).total_seconds()
        if quote_timestamp is not None and decision_timestamp is not None
        else None
    )
    quote_timestamp_valid = bool(
        quote_age_seconds is not None
        and quote_age_seconds >= 0.0
        and (
            recorded_max_quote_age_seconds is None
            or (
                recorded_max_quote_age_seconds >= 0.0
                and quote_age_seconds <= recorded_max_quote_age_seconds
            )
        )
    )
    worst_price_limit = conservative_price
    explicit_status = (
        explicit.get("execution_status")
        or explicit.get("status")
        or assumed.get("execution_status")
    )
    status = _normalise_shadow_execution_status(explicit_status)
    status_was_explicit = explicit_status not in (None, "")
    reason = explicit.get("execution_reason") or explicit.get("reason")
    assessment = assess_venue_order(
        constraints=VenueExecutionConstraints(
            min_order_size_shares=minimum_shares,
            tick_size=tick_size,
            fee_rate=fee_rate,
            fee_rate_source=fee_rate_source,
        ),
        intended_shares=shares,
        limit_price=conservative_price,
        available_depth_shares=available_size,
        intended_notional_usd=notional,
        max_notional_usd=_first_optional_float(
            (explicit, signal_metadata),
            ("max_notional_usd", "max_order_notional_usd"),
        ),
        max_loss_usd=_first_optional_float(
            (explicit, signal_metadata),
            ("max_loss_usd", "max_order_loss_usd"),
        ),
        order_type=order_type,
        side=order_action,
        ask_levels=ask_levels or None,
        full_book_complete=full_book_complete,
        require_full_book=require_full_book,
        best_ask_price=best_ask_price,
        best_ask_size=available_size,
        top_level_provenance=top_level_provenance,
        quote_timestamp_valid=quote_timestamp_valid,
        require_current_market_fee=require_full_book,
    )
    if status == "unknown" and assessment.eligible:
        status = "venue_executable"
        reason = "derived_from_shared_order_type_aware_venue_assessment"
    elif (
        status == "unknown"
        and require_full_book
        and set(assessment.reasons).issubset(
            {
                "insufficient_full_book_depth_within_limit",
                "no_full_book_ask_within_limit",
                "top_ask_above_worst_price_limit",
            }
        )
    ):
        status = "no_fill"
        reason = "derived_full_book_fok_no_fill"
    if assessment.execution_basis in {
        "top_level_proves_full_fill",
        "full_order_book_fok_buy_usdc_walk",
    }:
        shares = assessment.cumulative_shares
        conservative_price = assessment.weighted_average_price or conservative_price
        available_size = assessment.available_depth_shares
    return {
        "execution_status": status,
        "execution_status_explicit": status_was_explicit,
        "execution_reason": reason,
        "side": side or None,
        "order_type": order_type or None,
        "order_action": order_action or None,
        "entry_price": entry_price,
        "conservative_entry_price": conservative_price,
        "limit_price": worst_price_limit,
        "shares": shares,
        "notional_usd": notional,
        "min_order_size_shares": minimum_shares,
        "tick_size": tick_size,
        "available_size_at_price": available_size,
        "fee_rate": fee_rate,
        "fee_rate_source": fee_rate_source,
        "ask_levels": [dict(item) for item in ask_levels],
        "full_book_complete": full_book_complete,
        "require_full_book_execution": require_full_book,
        "top_level_provenance": top_level_provenance,
        "quote_timestamp_valid": quote_timestamp_valid,
        "decision_time_utc": (
            decision_timestamp.isoformat() if decision_timestamp is not None else None
        ),
        "quote_age_seconds": quote_age_seconds,
        "recorded_max_quote_age_seconds": recorded_max_quote_age_seconds,
        "venue_execution_assessment": assessment.as_report_dict(),
    }


def _first_optional_float(
    sources: Sequence[Mapping[str, Any]],
    keys: Sequence[str],
) -> Optional[float]:
    for source in sources:
        if not isinstance(source, Mapping):
            continue
        for key in keys:
            value = _optional_float(source.get(key))
            if value is not None:
                return value
    return None


def _gate_valid_resolved_cluster(cluster: Mapping[str, Any]) -> bool:
    if str(cluster.get("latest_status") or cluster.get("status") or "") != "resolved":
        return False
    payout = cluster.get("payout_for_shadow_side")
    return payout in (0, 1, False, True)


def _gate_clusters_from_current_events(events: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    clusters: dict[str, dict[str, Any]] = {}
    for event in events:
        decision = event.get("decision") if isinstance(event, Mapping) else {}
        if not isinstance(decision, Mapping) or not decision.get("would_trade"):
            continue
        identifiers = _settlement_identifiers_from_event(event)
        cluster_key = _would_trade_cluster_key(identifiers)
        if not cluster_key:
            continue
        settlement = event.get("settlement") if isinstance(event.get("settlement"), Mapping) else {}
        cluster = clusters.setdefault(
            cluster_key,
            {
                "cluster_key": cluster_key,
                "raw_signal_count": 0,
                "latest_status": "pending",
                "economic_observation": dict(_economic_observation_from_event(event)),
            },
        )
        cluster["raw_signal_count"] = int(cluster["raw_signal_count"]) + 1
        status = str(settlement.get("status") or "pending")
        cluster["latest_status"] = _more_final_settlement_status(
            str(cluster.get("latest_status") or "pending"),
            status,
        )
        if status == "resolved":
            cluster["payout_for_shadow_side"] = settlement.get("payout_for_shadow_side")
    return list(clusters.values())


def _forward_collection_progress(
    report_dir: Path,
    *,
    current_summary: Mapping[str, Any],
    current_settlement_refresh: Optional[Mapping[str, Any]] = None,
) -> Mapping[str, Any]:
    summaries: list[Mapping[str, Any]] = []
    for path in sorted(report_dir.glob("*-old-no-tail-live-forward-shadow-intraday.json")):
        try:
            summaries.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    summaries.append(current_summary)
    total_decisions = 0
    total_would_trade = 0
    days: set[str] = set()
    gate_valid_days: set[str] = set()
    artifacts = 0
    for summary in summaries:
        artifacts += 1
        generated = str(summary.get("generated_at_utc") or summary.get("started_at_utc") or "")
        if generated:
            days.add(generated[:10])
            if _gate_artifact_validity(summary)[0]:
                gate_valid_days.add(generated[:10])
        decisions = summary.get("decision_counts") or {}
        total_decisions += int(decisions.get("shadow_decisions") or 0)
        total_would_trade += int(decisions.get("would_trade_signals") or 0)
    latest_settlements = _latest_settlement_refresh_states(
        report_dir,
        current_settlement_refresh=current_settlement_refresh,
    )
    latest_cluster_settlements = _latest_settlement_refresh_cluster_states(
        report_dir,
        current_settlement_refresh=current_settlement_refresh,
    )
    refresh_status_counts = Counter(str(state.get("status") or "unknown") for state in latest_settlements.values())
    total_resolved_would_trade = int(refresh_status_counts.get("resolved") or 0)
    total_unresolved_would_trade = max(0, total_would_trade - total_resolved_would_trade)
    unique_signal_summary = _unique_would_trade_signal_summary(
        report_dir,
        latest_cluster_settlements=latest_cluster_settlements,
        event_log_validity=_gate_event_log_validity(report_dir, current_summary=current_summary),
    )
    forward_gate = _evaluate_forward_shadow_gate(
        unique_signal_summary["gate_valid_resolved_clusters"],
        forward_log_days=len(gate_valid_days),
    )
    gate_resolved_clusters = int(forward_gate["resolved_cluster_count"])
    return {
        "artifact_count_in_report_dir_including_current": artifacts,
        "days_with_forward_logs": len(days),
        "logged_utc_dates": sorted(days),
        "gate_valid_forward_log_days": len(gate_valid_days),
        "gate_valid_logged_utc_dates": sorted(gate_valid_days),
        "shadow_decisions": total_decisions,
        "would_trade_signals": total_would_trade,
        "resolved_would_trade_signals": total_resolved_would_trade,
        "unresolved_would_trade_signals": total_unresolved_would_trade,
        "latest_settlement_refresh_status_counts": dict(sorted(refresh_status_counts.items())),
        "unique_market_side_signal_clusters": unique_signal_summary["unique_market_side_signal_clusters"],
        "unique_market_side_signal_cluster_status_counts": unique_signal_summary[
            "unique_market_side_signal_cluster_status_counts"
        ],
        "unique_resolved_market_side_signal_clusters": unique_signal_summary[
            "unique_resolved_market_side_signal_clusters"
        ],
        "unique_unresolved_market_side_signal_clusters": unique_signal_summary[
            "unique_unresolved_market_side_signal_clusters"
        ],
        "gate_count_basis": "deduped_true_opportunity_clusters",
        "gate_resolved_would_trade_clusters": gate_resolved_clusters,
        "gate_unresolved_would_trade_clusters": unique_signal_summary[
            "unique_unresolved_market_side_signal_clusters"
        ],
        "gate_winning_would_trade_clusters": forward_gate["winning_cluster_count"],
        "gate_losing_would_trade_clusters": forward_gate["losing_cluster_count"],
        "forward_shadow_gate": forward_gate,
        "gate_excluded_raw_detection_count_by_reason": unique_signal_summary[
            "gate_excluded_raw_detection_count_by_reason"
        ],
        "gate_excluded_cluster_count_by_reason": unique_signal_summary[
            "gate_excluded_cluster_count_by_reason"
        ],
        "gate_excluded_artifact_count_by_reason": unique_signal_summary[
            "gate_excluded_artifact_count_by_reason"
        ],
        "raw_signals_are_poll_level_not_independent_contracts": True,
        "target_resolved_would_trades": 30,
        "target_resolved_would_trade_clusters": 30,
        "checkpoint_resolved_would_trade_clusters": FORWARD_SHADOW_GATE_MIN_RESOLVED_CLUSTERS,
        "checkpoint_complete": gate_resolved_clusters >= FORWARD_SHADOW_GATE_MIN_RESOLVED_CLUSTERS,
        "production_target_resolved_would_trade_clusters": FORWARD_SHADOW_PRODUCTION_MIN_RESOLVED_CLUSTERS,
        "target_forward_log_days_min": FORWARD_SHADOW_PRODUCTION_MIN_VALID_DAYS,
        "collection_complete": bool(
            (forward_gate.get("production_candidate_requirements") or {}).get("evidence_requirements_met")
        ),
        "live_permission_granted": False,
    }


def _unique_would_trade_signal_summary(
    report_dir: Path,
    *,
    latest_cluster_settlements: Mapping[str, Mapping[str, Any]],
    event_log_validity: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any]:
    clusters: dict[str, dict[str, Any]] = {}
    cluster_status: dict[str, str] = {}
    excluded_raw_counts: Counter[str] = Counter()
    excluded_cluster_keys_by_reason: dict[str, set[str]] = {}
    for events_path in sorted(report_dir.glob("*-old-no-tail-live-forward-shadow-intraday-events.jsonl")):
        try:
            lines = events_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        validity = event_log_validity.get(str(events_path), {"gate_valid": False, "reason": "legacy_pre_fix_missing_shadow_scoring_parity"})
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            decision = event.get("decision") if isinstance(event, Mapping) else {}
            if not isinstance(decision, Mapping) or not decision.get("would_trade"):
                continue
            identifiers = _settlement_identifiers_from_event(event)
            decision_id = str(identifiers.get("decision_id") or "")
            cluster_key = _would_trade_cluster_key(identifiers)
            if not cluster_key:
                continue
            if not validity.get("gate_valid"):
                reason = str(validity.get("reason") or "non_gate_valid_scoring_artifact")
                excluded_raw_counts[reason] += 1
                excluded_cluster_keys_by_reason.setdefault(reason, set()).add(cluster_key)
                continue
            cluster = clusters.setdefault(
                cluster_key,
                {
                    "cluster_key": cluster_key,
                    "market": {
                        key: identifiers.get(key)
                        for key in (
                            "event_slug",
                            "market_slug",
                            "market_id",
                            "condition_id",
                            "city",
                            "market_date_local",
                            "bucket_label",
                            "side",
                            "token_id",
                        )
                    },
                    "raw_signal_count": 0,
                    "decision_ids": [],
                    "first_seen_events_jsonl": str(events_path),
                    "first_seen_line_number": line_number,
                    "last_seen_events_jsonl": str(events_path),
                    "last_seen_line_number": line_number,
                    "economic_observation": dict(_economic_observation_from_event(event)),
                },
            )
            cluster["raw_signal_count"] = int(cluster["raw_signal_count"]) + 1
            cluster["last_seen_events_jsonl"] = str(events_path)
            cluster["last_seen_line_number"] = line_number
            if decision_id:
                cluster["decision_ids"].append(decision_id)
            settlement = event.get("settlement") if isinstance(event.get("settlement"), Mapping) else {}
            raw_status = str(settlement.get("status") or "pending")
            cluster_status[cluster_key] = _more_final_settlement_status(
                cluster_status.get(cluster_key),
                raw_status,
            )
            if raw_status == "resolved":
                cluster["payout_for_shadow_side"] = settlement.get("payout_for_shadow_side")

    for cluster_key, state in latest_cluster_settlements.items():
        if cluster_key not in clusters:
            continue
        status = str(state.get("status") or "unknown")
        previous = cluster_status.get(cluster_key)
        cluster_status[cluster_key] = _more_final_settlement_status(previous, status)
        clusters[cluster_key]["latest_settlement"] = dict(state)
        if status == "resolved":
            clusters[cluster_key]["payout_for_shadow_side"] = state.get("payout_for_shadow_side")
    status_counts = Counter()
    gate_valid_resolved_clusters: list[Mapping[str, Any]] = []
    excluded_cluster_counts: Counter[str] = Counter()
    for cluster_key, cluster in clusters.items():
        status = cluster_status.get(cluster_key, "pending")
        cluster["latest_status"] = status
        status_counts[status] += 1
        reason = _gate_cluster_exclusion_reason(cluster)
        if reason is None:
            gate_valid_resolved_clusters.append(cluster)
        else:
            excluded_cluster_counts[reason] += 1
    resolved = int(status_counts.get("resolved") or 0)
    artifact_excluded_counts = Counter(
        str(item.get("reason") or "unknown")
        for item in event_log_validity.values()
        if item.get("gate_valid") is False and item.get("events_jsonl")
    )
    return {
        "unique_market_side_signal_clusters": len(clusters),
        "unique_market_side_signal_cluster_status_counts": dict(sorted(status_counts.items())),
        "unique_resolved_market_side_signal_clusters": resolved,
        "unique_unresolved_market_side_signal_clusters": max(0, len(clusters) - resolved),
        "gate_valid_resolved_clusters": gate_valid_resolved_clusters,
        "gate_excluded_raw_detection_count_by_reason": dict(sorted(excluded_raw_counts.items())),
        "gate_excluded_cluster_count_by_reason": {
            **dict(sorted(excluded_cluster_counts.items())),
            **{
                reason: len(keys)
                for reason, keys in sorted(excluded_cluster_keys_by_reason.items())
                if reason not in excluded_cluster_counts
            },
        },
        "gate_excluded_artifact_count_by_reason": dict(sorted(artifact_excluded_counts.items())),
        "clusters_preview": list(clusters.values())[:20],
    }


def _gate_event_log_validity(
    report_dir: Path,
    *,
    current_summary: Mapping[str, Any],
) -> Mapping[str, Mapping[str, Any]]:
    validity: dict[str, Mapping[str, Any]] = {}
    summaries: list[tuple[Path, Mapping[str, Any]]] = []
    for path in sorted(report_dir.glob("*-old-no-tail-live-forward-shadow-intraday.json")):
        try:
            summaries.append((path, json.loads(path.read_text(encoding="utf-8"))))
        except (OSError, json.JSONDecodeError):
            continue
    summaries.append((Path(str(current_summary.get("artifacts", {}).get("summary_json", ""))), current_summary))
    for summary_path, summary in summaries:
        events_path = _events_path_for_summary(summary_path, summary)
        if not events_path:
            continue
        is_valid, reason = _gate_artifact_validity(summary)
        validity[str(events_path)] = {
            "events_jsonl": str(events_path),
            "gate_valid": is_valid,
            "reason": reason,
        }
    return validity


def _events_path_for_summary(summary_path: Path, summary: Mapping[str, Any]) -> Optional[Path]:
    artifacts = summary.get("artifacts") if isinstance(summary.get("artifacts"), Mapping) else {}
    events_jsonl = artifacts.get("events_jsonl") if isinstance(artifacts, Mapping) else None
    if events_jsonl:
        return Path(str(events_jsonl))
    if summary_path.name.endswith("-old-no-tail-live-forward-shadow-intraday.json"):
        return summary_path.with_name(
            summary_path.name.replace(
                "-old-no-tail-live-forward-shadow-intraday.json",
                "-old-no-tail-live-forward-shadow-intraday-events.jsonl",
            )
        )
    return None


def _gate_artifact_validity(summary: Mapping[str, Any]) -> tuple[bool, str]:
    scoring = summary.get("shadow_scoring_parity") if isinstance(summary, Mapping) else {}
    if not isinstance(scoring, Mapping):
        return False, "legacy_pre_fix_missing_shadow_scoring_parity"
    if scoring.get("gate_valid_for_shadow_deployment") is True:
        return True, "gate_valid"
    reason = str(scoring.get("gate_validity_reason") or "non_gate_valid_scoring_artifact")
    if "legacy_live_open_meteo_adapter_scores_present" in reason:
        return False, "legacy_pre_fix_score_family_artifact"
    return False, "non_gate_valid_scoring_artifact"


def _gate_cluster_exclusion_reason(cluster: Mapping[str, Any]) -> Optional[str]:
    status = str(cluster.get("latest_status") or "unknown")
    if status == "resolved":
        return None if _gate_valid_resolved_cluster(cluster) else "resolved_missing_shadow_payout"
    if status == "pending":
        return "pending_settlement"
    if status == "unresolved_or_ambiguous":
        return "ambiguous_or_unfinalized_settlement"
    if status == "excluded_or_manual_review":
        return "manual_review_or_disputed_settlement"
    if status == "refresh_error":
        return "settlement_refresh_error"
    return "unresolved_settlement"


def _would_trade_cluster_key(identifiers: Mapping[str, Any]) -> str:
    side = str(identifiers.get("side") or "").upper()
    token_id = str(identifiers.get("token_id") or "")
    market_slug = str(identifiers.get("market_slug") or "")
    event_slug = str(identifiers.get("event_slug") or "")
    event_or_market_slug = event_slug or market_slug
    target_date = str(identifiers.get("market_date_local") or "")
    bucket = str(identifiers.get("bucket_label") or "")
    if not bucket and market_slug:
        bucket = _infer_bucket_label_from_market_slug(market_slug)
    if not event_or_market_slug:
        event_or_market_slug = str(
            identifiers.get("condition_id") or identifiers.get("market_id") or ""
        )
    if not any((event_or_market_slug, token_id, side, bucket, target_date)):
        return ""
    return "|".join(
        (
            f"event_or_market_slug={_cluster_key_value(event_or_market_slug)}",
            f"market_slug={_cluster_key_value(market_slug)}",
            f"token_id={_cluster_key_value(token_id)}",
            f"side={_cluster_key_value(side)}",
            f"bucket={_cluster_key_value(bucket)}",
            f"target_date={_cluster_key_value(target_date)}",
        )
    )


def _cluster_key_value(value: Any) -> str:
    normalized = re.sub(r"\s+", " ", str(value or "").strip()).lower()
    return normalized.replace("|", "%7C")


def _infer_bucket_label_from_market_slug(market_slug: str) -> str:
    match = re.search(r"(-?\d+(?:\.\d+)?)(c|f)(?:-|$)", market_slug, re.I)
    if not match:
        return ""
    value = float(match.group(1))
    unit = match.group(2).upper()
    fahrenheit = _to_fahrenheit(value, unit)
    return f"{fahrenheit:g}F"


def _more_final_settlement_status(previous: Optional[str], current: str) -> str:
    priority = {
        "excluded_or_manual_review": 5,
        "unresolved_or_ambiguous": 4,
        "refresh_error": 3,
        "resolved": 2,
        "pending": 0,
        "unknown": 0,
    }
    if previous is None:
        return current
    return current if priority.get(current, 0) >= priority.get(previous, 0) else previous


def _latest_settlement_refresh_states(
    report_dir: Path,
    *,
    current_settlement_refresh: Optional[Mapping[str, Any]] = None,
) -> Mapping[str, Mapping[str, Any]]:
    refreshes: list[Mapping[str, Any]] = []
    for path in sorted(report_dir.glob("*-old-no-tail-live-forward-shadow-settlement-refresh.json")):
        try:
            refreshes.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    if current_settlement_refresh:
        refreshes.append(current_settlement_refresh)
    states: dict[str, Mapping[str, Any]] = {}
    state_times: dict[str, str] = {}
    for refresh in refreshes:
        generated_at = str(refresh.get("generated_at_utc") or "")
        for update in refresh.get("updates") or []:
            if not isinstance(update, Mapping):
                continue
            settlement = update.get("settlement") if isinstance(update.get("settlement"), Mapping) else {}
            market = update.get("market") if isinstance(update.get("market"), Mapping) else {}
            cluster_key = str(update.get("cluster_key") or _would_trade_cluster_key(market) or "")
            decision_ids = [str(item) for item in (update.get("decision_ids") or []) if str(item)]
            if not decision_ids and update.get("decision_id"):
                decision_ids = [str(update.get("decision_id"))]
            for decision_id in decision_ids:
                if decision_id in state_times and generated_at < state_times[decision_id]:
                    continue
                states[decision_id] = {
                    "generated_at_utc": generated_at,
                    "status": settlement.get("status") or "unknown",
                    "canonical_label": settlement.get("canonical_label"),
                    "payout_for_shadow_side": settlement.get("payout_for_shadow_side"),
                    "resolved_at_utc": settlement.get("resolved_at_utc") or settlement.get("resolved_at"),
                    "source": settlement.get("source"),
                    "reason": settlement.get("reason"),
                    "manual_review_required": settlement.get("manual_review_required"),
                    "later_update_required": settlement.get("later_update_required"),
                    "cluster_key": cluster_key,
                    "settlement_recheck_anomaly": update.get("settlement_recheck_anomaly")
                    or settlement.get("settlement_recheck_anomaly"),
                }
                state_times[decision_id] = generated_at
    return states


def _latest_settlement_refresh_cluster_states(
    report_dir: Path,
    *,
    current_settlement_refresh: Optional[Mapping[str, Any]] = None,
) -> Mapping[str, Mapping[str, Any]]:
    refreshes: list[Mapping[str, Any]] = []
    for path in sorted(report_dir.glob("*-old-no-tail-live-forward-shadow-settlement-refresh.json")):
        try:
            refreshes.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    if current_settlement_refresh:
        refreshes.append(current_settlement_refresh)
    states: dict[str, Mapping[str, Any]] = {}
    state_times: dict[str, str] = {}
    for refresh in refreshes:
        generated_at = str(refresh.get("generated_at_utc") or "")
        for update in refresh.get("updates") or []:
            if not isinstance(update, Mapping):
                continue
            market = update.get("market") if isinstance(update.get("market"), Mapping) else {}
            cluster_key = str(update.get("cluster_key") or _would_trade_cluster_key(market) or "")
            if not cluster_key:
                continue
            if cluster_key in state_times and generated_at < state_times[cluster_key]:
                continue
            settlement = update.get("settlement") if isinstance(update.get("settlement"), Mapping) else {}
            states[cluster_key] = {
                "generated_at_utc": generated_at,
                "status": settlement.get("status") or "unknown",
                "canonical_label": settlement.get("canonical_label"),
                "payout_for_shadow_side": settlement.get("payout_for_shadow_side"),
                "resolved_at_utc": settlement.get("resolved_at_utc") or settlement.get("resolved_at"),
                "source": settlement.get("source"),
                "reason": settlement.get("reason"),
                "manual_review_required": settlement.get("manual_review_required"),
                "later_update_required": settlement.get("later_update_required"),
                "settlement_recheck_anomaly": update.get("settlement_recheck_anomaly")
                or settlement.get("settlement_recheck_anomaly"),
                "decision_ids": list(update.get("decision_ids") or []),
                "raw_signal_count": update.get("raw_signal_count"),
            }
            state_times[cluster_key] = generated_at
    return states


def _render_settlement_refresh_markdown(summary: Mapping[str, Any]) -> str:
    counts = summary.get("settlement_status_counts") or {}
    artifacts = summary.get("artifacts") or {}
    lines = [
        "# Old NO-tail Live-Forward Settlement Refresh",
        "",
        f"- Generated: `{summary.get('generated_at_utc')}`",
        f"- Profile: `{summary.get('profile_key')}`",
        "- Mode: read-only settlement reconciliation for prior would-trade shadow signals.",
        "- No real orders were placed, no fills were simulated as truth, and payouts are shadow-settlement fields only.",
        f"- Resolved-cluster recheck window hours: `{summary.get('resolved_recheck_window_hours')}`",
        f"- Resolved clusters rechecked: `{summary.get('resolved_recheck_cluster_count', 0)}`",
        f"- Settlement recheck anomalies: `{summary.get('settlement_recheck_anomaly_count', 0)}`",
        f"- Pending would-trade poll detections scanned: `{summary.get('raw_pending_would_trade_events_scanned', summary.get('pending_would_trade_events_scanned', 0))}`",
        f"- Pending deduped would-trade clusters scanned: `{summary.get('pending_would_trade_clusters_scanned', 0)}`",
        f"- Gamma events fetched: `{summary.get('gamma_events_fetched', 0)}`",
        "",
        "## Artifacts",
        "",
        f"- Summary JSON: `{artifacts.get('summary_json')}`",
        f"- Raw payload directory: `{artifacts.get('raw_payload_dir')}`",
        "",
        "## Deduped Cluster Status Counts",
        "",
    ]
    if counts:
        for key, count in sorted(counts.items()):
            lines.append(f"- `{key}`: `{count}`")
    else:
        lines.append("- No prior pending would-trade signals found.")
    anomalies = summary.get("settlement_recheck_anomalies") or []
    if anomalies:
        lines.extend(["", "## Settlement Recheck Anomalies", ""])
        for anomaly in anomalies[:50]:
            lines.append(
                "- CRITICAL "
                f"`{anomaly.get('anomaly_type')}` "
                f"cluster `{anomaly.get('cluster_key')}` "
                f"previous `{anomaly.get('previous_status')}`/`{anomaly.get('previous_canonical_label')}` "
                f"current `{anomaly.get('current_status')}`/`{anomaly.get('current_canonical_label')}` "
                "-> `excluded_or_manual_review`"
            )
    updates = summary.get("updates") or []
    if updates:
        lines.extend(["", "## Refreshed Clusters", ""])
        for update in updates[:50]:
            market = update.get("market") or {}
            settlement = update.get("settlement") or {}
            lines.append(
                "- "
                f"`{update.get('cluster_key') or update.get('decision_id')}` "
                f"{market.get('city')} {market.get('market_date_local')} {market.get('bucket_label')} "
                f"raw detections `{update.get('raw_signal_count', 1)}` "
                f"-> `{settlement.get('status')}` "
                f"label `{settlement.get('canonical_label')}` "
                f"shadow payout `{settlement.get('payout_for_shadow_side')}`"
            )
    return "\n".join(lines) + "\n"


def _render_markdown(summary: Mapping[str, Any]) -> str:
    artifacts = summary.get("artifacts") or {}
    decisions = summary.get("decision_counts") or {}
    lead_days = summary.get("lead_day_counts") or {}
    settlements = summary.get("settlement_status_counts") or {}
    would_trade_settlements = summary.get("would_trade_settlement_status_counts") or {}
    would_trade_cluster_settlements = summary.get("would_trade_cluster_settlement_status_counts") or {}
    settlement_refresh = summary.get("settlement_refresh") or {}
    quote_quality = summary.get("quote_quality") or {}
    close_times = summary.get("close_time_diagnostics") or {}
    progress = summary.get("forward_collection_progress") or {}
    forward_gate = progress.get("forward_shadow_gate") if isinstance(progress.get("forward_shadow_gate"), Mapping) else {}
    live_vs_replay = summary.get("live_vs_replay_behavior") or {}
    scoring = summary.get("shadow_scoring_parity") or {}
    lines = [
        "# Old NO-tail Live-Forward Shadow Intraday Report",
        "",
        f"- Generated: `{summary.get('generated_at_utc')}`",
        f"- Profile: `{summary.get('profile_key')}`",
        "- Mode: read-only live-forward shadow logging. No real orders were placed and no fills were simulated as truth.",
        "- Historical PnL numbers are not used as expected performance.",
        f"- Polls: `{summary.get('effective_iterations')}` at `{summary.get('interval_seconds')}` seconds",
        f"- Shadow decisions: `{decisions.get('shadow_decisions', 0)}`",
        f"- Would-trade poll detections: `{decisions.get('raw_would_trade_poll_detections', decisions.get('would_trade_signals', 0))}`",
        f"- Would-trade clusters: `{decisions.get('would_trade_signal_clusters', 0)}`",
        f"- No-trade decisions: `{decisions.get('no_trade_decisions', 0)}`",
        f"- Lead-day 1-2 decisions: `{lead_days.get('priority_1_2', 0)}`",
        "",
        "## Artifacts",
        "",
        f"- Summary JSON: `{artifacts.get('summary_json')}`",
        f"- Events JSONL: `{artifacts.get('events_jsonl')}`",
        f"- Raw payload directory: `{artifacts.get('raw_payload_dir')}`",
        f"- Settlement refresh JSON: `{artifacts.get('settlement_refresh_json')}`",
        f"- Settlement refresh Markdown: `{artifacts.get('settlement_refresh_markdown')}`",
        "",
        "## Pre-Registration",
        "",
        "- Expected would-trades/day: point `1.0`, sanity band `0.0-4.0`.",
        "- Primary evidence: fee-adjusted conservative executable net PnL per deduped resolved opportunity cluster.",
        "- Uncertainty: deterministic 95% nonparametric cluster-bootstrap percentile interval with 10,000 fixed-seed resamples.",
        "- Hit rate and Wilson intervals are secondary diagnostics compared with price-weighted break-even.",
        "- Thirty resolved clusters is a review checkpoint only; production-candidate metadata requires at least 100 resolved clusters and 30 valid calendar days and never grants live permission by itself.",
        "",
        "## Shadow Scoring Parity",
        "",
        f"- Gate valid for shadow deployment: `{scoring.get('gate_valid_for_shadow_deployment')}`",
        f"- Reason: `{scoring.get('gate_validity_reason')}`",
        f"- Scoring path: `{scoring.get('scoring_path')}`",
        f"- Legacy live Open-Meteo adapter score count: `{scoring.get('legacy_live_open_meteo_adapter_score_count', 0)}`",
        f"- Score cache records: `{(scoring.get('score_cache') or {}).get('record_count', 0)}`",
        f"- Firewall violations: `{scoring.get('firewall_violation_count', 0)}`",
        f"- Pre-fix artifacts excluded from gate: `{scoring.get('pre_fix_artifacts_excluded_from_gate')}`",
        "",
        "## No-Trade Reasons",
        "",
    ]
    reasons = summary.get("no_trade_reason_counts") or {}
    if reasons:
        for reason, count in sorted(reasons.items(), key=lambda item: (-int(item[1]), item[0])):
            lines.append(f"- `{reason}`: `{count}`")
    else:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "## Lead Days",
            "",
        ]
    )
    if lead_days:
        for key, count in sorted(lead_days.items()):
            lines.append(f"- `{key}`: `{count}`")
    else:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "## Settlement Tracking",
            "",
        ]
    )
    if settlements:
        for key, count in sorted(settlements.items()):
            lines.append(f"- All decisions `{key}`: `{count}`")
    else:
        lines.append("- No settlement rows.")
    if would_trade_settlements:
        for key, count in sorted(would_trade_settlements.items()):
            lines.append(f"- Would-trade poll detections `{key}`: `{count}`")
    else:
        lines.append("- Would-trade poll-detection settlement rows: `0`")
    if would_trade_cluster_settlements:
        for key, count in sorted(would_trade_cluster_settlements.items()):
            lines.append(f"- Would-trade clusters `{key}`: `{count}`")
    else:
        lines.append("- Would-trade cluster settlement rows: `0`")
    refresh_counts = settlement_refresh.get("settlement_status_counts") or {}
    lines.extend(["", "## Settlement Refresh", ""])
    lines.append(
        f"- Prior pending poll detections scanned: `{settlement_refresh.get('raw_pending_would_trade_events_scanned', settlement_refresh.get('pending_would_trade_events_scanned', 0))}`"
    )
    lines.append(f"- Prior pending clusters scanned: `{settlement_refresh.get('pending_would_trade_clusters_scanned', 0)}`")
    if refresh_counts:
        for key, count in sorted(refresh_counts.items()):
            lines.append(f"- Refreshed clusters `{key}`: `{count}`")
    else:
        lines.append("- Refreshed rows: `0`")
    lines.extend(
        [
            "",
            "## Quote Quality",
            "",
        ]
    )
    counts = quote_quality.get("counts") or {}
    for key, count in sorted(counts.items()):
        lines.append(f"- `{key}`: `{count}`")
    visible = quote_quality.get("visible_top_depth") or {}
    spread = quote_quality.get("spread") or {}
    lines.append(f"- Visible top depth median: `{visible.get('median')}` across `{visible.get('count', 0)}` rows")
    lines.append(f"- Spread median: `{spread.get('median')}` across `{spread.get('count', 0)}` rows")
    lines.extend(
        [
            "",
            "## Market Close Diagnostics",
            "",
        ]
    )
    if close_times:
        for key, count in sorted(close_times.items()):
            lines.append(f"- `{key}`: `{count}`")
    else:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "## Forward Progress",
            "",
            f"- Days with logs: `{progress.get('days_with_forward_logs', 0)}`",
            f"- Cumulative raw would-trade poll detections: `{progress.get('would_trade_signals', 0)}`",
            f"- Cumulative resolved raw poll detections: `{progress.get('resolved_would_trade_signals', 0)}`",
            f"- Cumulative unresolved raw poll detections: `{progress.get('unresolved_would_trade_signals', 0)}`",
            f"- Unique market-side signal clusters: `{progress.get('unique_market_side_signal_clusters', 0)}`",
            f"- Unique resolved market-side signal clusters: `{progress.get('unique_resolved_market_side_signal_clusters', 0)}`",
            f"- Unique unresolved market-side signal clusters: `{progress.get('unique_unresolved_market_side_signal_clusters', 0)}`",
            f"- Live gate basis: `{progress.get('gate_count_basis', 'deduped_true_opportunity_clusters')}`",
            f"- Gate resolved clusters: `{progress.get('gate_resolved_would_trade_clusters', 0)}` / `{progress.get('target_resolved_would_trade_clusters', progress.get('target_resolved_would_trades', 30))}`",
            f"- Gate status: `{forward_gate.get('status')}`",
            f"- Gate reason: `{forward_gate.get('reason')}`",
            f"- Conservative fee-adjusted net PnL: `{forward_gate.get('total_net_pnl_usd')}`",
            f"- Mean net PnL per resolved opportunity: `{forward_gate.get('mean_net_pnl_usd_per_resolved_opportunity_cluster')}`",
            f"- Deterministic 95% cluster-bootstrap mean-PnL interval: `{(forward_gate.get('economic_uncertainty') or {}).get('lower_mean_net_pnl_usd')}` to `{(forward_gate.get('economic_uncertainty') or {}).get('upper_mean_net_pnl_usd')}`",
            f"- Venue-executable / no-fill / ineligible / invalid clusters: `{forward_gate.get('venue_executable_cluster_count', 0)}` / `{forward_gate.get('no_fill_cluster_count', 0)}` / `{forward_gate.get('ineligible_cluster_count', 0)}` / `{forward_gate.get('invalid_economic_cluster_count', 0)}`",
            f"- Secondary filled hit rate: `{forward_gate.get('hit_rate')}`",
            f"- Price-weighted break-even hit rate: `{forward_gate.get('price_weighted_break_even_hit_rate')}`",
            f"- Wilson 95%: `{(forward_gate.get('wilson_95') or {}).get('lower')}` to `{(forward_gate.get('wilson_95') or {}).get('upper')}`",
            f"- Wilson 80%: `{(forward_gate.get('wilson_80') or {}).get('lower')}` to `{(forward_gate.get('wilson_80') or {}).get('upper')}`",
            f"- Production evidence metadata: `{(forward_gate.get('production_candidate_requirements') or {}).get('status')}`; live permission: `False`.",
            "- Raw signal counts are poll-level observations, not independent contract outcomes.",
            f"- Collection complete: `{progress.get('collection_complete', False)}`",
            "",
            "## Live vs Replay Behavior",
            "",
            f"- Status: `{live_vs_replay.get('status')}`",
            f"- Reason: `{live_vs_replay.get('reason')}`",
        ]
    )
    return "\n".join(lines) + "\n"






def live_market_metadata(market: WeatherMarket) -> Mapping[str, Any]:
    return {
        "market_id": market.market_id,
        "condition_id": market.condition_id,
        "slug": market.slug,
        "event_slug": market.event_slug,
        "question": market.question,
        "close_time_utc": market.close_time.isoformat() if market.close_time else None,
        "target_date": market.target_date.isoformat(),
        "city": market.city.display_name,
        "timezone": market.city.timezone,
        "metar_station": market.city.metar_station,
    }


def parsed_contract(market: WeatherMarket) -> Mapping[str, Any]:
    return {
        "city": market.city.display_name,
        "date": market.target_date.isoformat(),
        "bucket_label": market.bucket.label,
        "bucket_type": market.bucket.bucket_type,
        "threshold_low_f": market.bucket.lower_f,
        "threshold_high_f": market.bucket.upper_f,
        "side": "NO",
        "yes_token_id": market.bucket.yes_token_id,
        "no_token_id": market.bucket.no_token_id,
        "resolution_unit": market.bucket.resolution_unit,
        "resolution_precision": market.bucket.resolution_precision,
    }


def book_to_payload(book: Optional[BookSnapshot]) -> Mapping[str, Any]:
    if book is None:
        return {"status": "missing"}
    return {
        "status": book.raw_status,
        "error": book.error,
        "token_id": book.token_id,
        "fetched_at_utc": book.fetched_at_utc.isoformat(),
        "best_bid": book.best_bid,
        "best_ask": book.best_ask,
        "bid_size": book.bid_size,
        "ask_size": book.ask_size,
        "mid": book.mid,
        "spread": book.spread,
        "visible_top_depth": book.visible_top_depth,
        "top_bids": list(book.top_bids),
        "top_asks": list(book.top_asks),
        "bid_levels": list(book.bid_levels),
        "ask_levels": list(book.ask_levels),
        "full_book_complete": book.full_book_complete,
        "min_order_size_shares": book.min_order_size_shares,
        "tick_size": book.tick_size,
        "neg_risk": book.neg_risk,
        "fee_rate": book.fee_rate,
        "fee_rate_source": book.fee_rate_source,
        "payload_id": book.payload_id,
        "source_sha256": book.source_sha256,
        "venue_timestamp_raw": book.venue_timestamp_raw,
        "received_at_utc": (
            book.received_at_utc.isoformat() if book.received_at_utc else None
        ),
        "raw_cache_path": book.raw_cache_path,
    }


def forecast_to_payload(forecast: ForecastSource) -> Mapping[str, Any]:
    return {
        "source_name": forecast.source_name,
        "status": forecast.status,
        "error": forecast.error,
        "fetched_at_utc": forecast.fetched_at_utc.isoformat(),
        "temperature_2m_max_f": forecast.temperature_2m_max_f,
        "sample_count": len(forecast.samples_f),
        "samples_f": list(forecast.samples_f[:20]),
        "features": dict(forecast.features),
        "payload_id": forecast.payload_id,
        "raw_cache_path": forecast.raw_cache_path,
    }


def station_summary_to_payload(summary: StationObservationSummary) -> Mapping[str, Any]:
    return {
        "station_id": summary.station_id,
        "status": summary.status,
        "error": summary.error,
        "fetched_at_utc": summary.fetched_at_utc.isoformat(),
        "local_date": summary.local_date,
        "latest_observed_at_utc": summary.latest_observed_at_utc.isoformat() if summary.latest_observed_at_utc else None,
        "latest_temperature_f": summary.latest_temperature_f,
        "high_so_far_f": summary.high_so_far_f,
        "high_so_far_observed_at_utc": (
            summary.high_so_far_observed_at_utc.isoformat() if summary.high_so_far_observed_at_utc else None
        ),
        "observation_count": summary.observation_count,
        "raw_cache_path": summary.raw_cache_path,
        "payload_id": summary.payload_id,
        "source_family": summary.source_family,
    }


def _get_json(url: str, params: Optional[Mapping[str, Any]] = None) -> Any:
    full_url = url if not params else f"{url}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(full_url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
        return json.loads(response.read().decode("utf-8"))


def _read_cached_or_fetch_json(url: str, *, cache_path: Path, cache_only: bool) -> Any:
    if cache_only:
        if not cache_path.exists():
            raise FileNotFoundError(f"cache miss for {cache_path}")
        return json.loads(cache_path.read_text(encoding="utf-8"))
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
        payload = json.loads(response.read().decode("utf-8"))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    sidecar = cache_path.with_suffix(".request.json")
    sidecar.write_text(json.dumps({"url": url, "fetched_at_utc": datetime.now(timezone.utc).isoformat()}, indent=2), encoding="utf-8")
    return payload


def _iter_event_markets(events: Iterable[Any]) -> Iterator[Mapping[str, Any]]:
    for event in events:
        if not isinstance(event, Mapping):
            continue
        markets = event.get("markets")
        if isinstance(markets, list):
            for market in markets:
                if isinstance(market, Mapping):
                    merged = dict(market)
                    merged.setdefault("eventSlug", event.get("slug"))
                    merged.setdefault("eventTitle", event.get("title"))
                    merged.setdefault("eventSubtitle", event.get("subtitle"))
                    merged.setdefault("eventDescription", event.get("description"))
                    yield merged
        else:
            yield event


def _temperature_search_queries(today: date) -> tuple[str, ...]:
    dates = (today, today + timedelta(days=1), today + timedelta(days=2))
    date_queries = tuple(f"{value.strftime('%B')} {value.day} highest temperature" for value in dates)
    return (*date_queries, "highest temperature", "temperature weather")


def parse_jsonish_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        parsed = json.loads(stripped)
        return parsed if isinstance(parsed, list) else []
    return []


def _parse_levels(levels: Iterable[Any]) -> list[tuple[float, float]]:
    parsed: list[tuple[float, float]] = []
    for level in levels:
        if isinstance(level, Mapping):
            price = level.get("price")
            size = level.get("size")
        elif isinstance(level, (list, tuple)) and len(level) >= 2:
            price, size = level[0], level[1]
        else:
            continue
        try:
            parsed.append((float(price), float(size)))
        except (TypeError, ValueError):
            continue
    return parsed


def _payload_id(url: str, payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return _stable_id(url, blob.hex())


def _canonical_payload_sha256(payload: Any) -> str:
    blob = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def parse_polymarket_book_timestamp(value: Any) -> Optional[datetime]:
    """Parse the venue timestamp supplied by the Polymarket CLOB book API."""

    if value in (None, "") or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) or (
        isinstance(value, str) and re.fullmatch(r"[+-]?\d+(?:\.\d+)?", value.strip())
    ):
        try:
            seconds = float(value)
            while abs(seconds) >= 100_000_000_000:
                seconds /= 1000.0
            parsed = datetime.fromtimestamp(seconds, tz=timezone.utc)
        except (OverflowError, OSError, TypeError, ValueError):
            return None
    else:
        parsed = _parse_optional_utc(value)
        if parsed is None:
            return None
        parsed = parsed.astimezone(timezone.utc)
    if not datetime(2000, 1, 1, tzinfo=timezone.utc) <= parsed < datetime(
        2100, 1, 1, tzinfo=timezone.utc
    ):
        return None
    return parsed


def _stable_id(*parts: str) -> str:
    import hashlib

    blob = "\n".join(parts).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:24]


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")[:120] or "item"


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _parse_optional_utc(value: Any) -> Optional[datetime]:
    if value in (None, ""):
        return None
    try:
        return _parse_utc(str(value))
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_bool(value: Any) -> Optional[bool]:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _optional_int(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _numeric_summary(values: Sequence[float]) -> Mapping[str, Optional[float]]:
    cleaned = sorted(float(value) for value in values if value is not None)
    if not cleaned:
        return {"count": 0, "min": None, "median": None, "mean": None, "max": None}
    return {
        "count": len(cleaned),
        "min": cleaned[0],
        "median": statistics.median(cleaned),
        "mean": sum(cleaned) / len(cleaned),
        "max": cleaned[-1],
    }


def _market_lead_days(market: WeatherMarket, as_of: datetime) -> Optional[int]:
    try:
        local_date = as_of.astimezone(ZoneInfo(market.city.timezone)).date()
    except Exception:
        return None
    return (market.target_date - local_date).days


def _single_value_at_index(values: Any, index: int) -> Optional[float]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return None
    if index >= len(values):
        return None
    return _optional_float(values[index])


def _value_at_index(values: Any, index: int) -> list[Optional[float]]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return []
    if index >= len(values):
        return []
    value = values[index]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_optional_float(item) for item in value]
    return [_optional_float(value)]


def _detect_unit(label: str) -> str:
    return _detect_explicit_unit(label) or "F"


def _detect_explicit_unit(label: str) -> Optional[str]:
    lowered = label.lower()
    spelled_units: set[str] = set()
    if re.search(r"\b(?:degrees?|deg\.?)\s+celsius\b|\bcelsius\b", lowered):
        spelled_units.add("C")
    if re.search(r"\b(?:degrees?|deg\.?)\s+fahrenheit\b|\bfahrenheit\b", lowered):
        spelled_units.add("F")
    if len(spelled_units) == 1:
        return next(iter(spelled_units))

    token_units = {
        match.group("unit").upper()
        for match in re.finditer(
            r"(?<![A-Za-z])-?\d+(?:\.\d+)?\s*(?:°\s*|deg(?:rees?)?\.?\s*)?(?P<unit>[cf])\b",
            label,
            re.I,
        )
    }
    if len(token_units) == 1:
        return next(iter(token_units))
    return None


def _clean_temperature_label(label: str) -> str:
    clean = label.replace("°", " ")
    clean = re.sub(r"(?<=\d)\s*deg(?:rees?)?\.?\s*[cf]\b", "", clean, flags=re.I)
    clean = re.sub(r"(?<=\d)\s*degrees?\s+(?:celsius|fahrenheit)\b", "", clean, flags=re.I)
    clean = re.sub(r"(?<=\d)\s*[cf]\b", "", clean, flags=re.I)
    return " ".join(clean.split())


def _to_fahrenheit(value: float, unit: str) -> float:
    return value * 9 / 5 + 32 if unit.upper() == "C" else value


def _exact_bucket_upper_f(source_value: float, unit: str) -> float:
    if unit.upper() == "C":
        return _to_fahrenheit(source_value + 0.999, unit)
    return source_value + 0.999


def _apply_resolution_precision(bucket: Bucket, text: str) -> Bucket:
    unit = _detect_settlement_unit(text)
    precision = _detect_settlement_precision(text)
    updates: dict[str, Any] = {}
    if unit:
        updates["resolution_unit"] = unit
    if precision is not None:
        updates["resolution_precision"] = precision
    if not updates:
        return bucket
    return replace(bucket, **updates)


def _detect_settlement_unit(text: str) -> Optional[str]:
    lowered = text.lower()
    recorded = re.search(r"recorded .*? in degrees (?P<unit>Fahrenheit|Celsius)\b", text, re.I | re.S)
    if recorded:
        return "C" if recorded.group("unit").lower() == "celsius" else "F"
    measured = re.search(
        r"measures temperatures (?:in|to) (?:whole degrees )?(?P<unit>Fahrenheit|Celsius)\b",
        text,
        re.I,
    )
    if measured:
        return "C" if measured.group("unit").lower() == "celsius" else "F"
    if "degrees celsius" in lowered and "degrees fahrenheit" not in lowered:
        return "C"
    if "degrees fahrenheit" in lowered and "degrees celsius" not in lowered:
        return "F"
    if "°c" in lowered and "°f" not in lowered:
        return "C"
    if "°f" in lowered and "°c" not in lowered:
        return "F"
    return None


def _detect_settlement_precision(text: str) -> Optional[float]:
    lowered = text.lower()
    if "one decimal place" in lowered:
        return 0.1
    if re.search(r"\bwhole degrees?\b", lowered):
        return 1.0
    return None


def _local_hour(moment: datetime, timezone_name: str) -> float:
    local = moment.astimezone(ZoneInfo(timezone_name))
    return local.hour + local.minute / 60.0 + local.second / 3600.0


def _json_ready(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def main(argv: Optional[Sequence[str]] = None) -> int:
    global HTTP_TIMEOUT_SECONDS

    parser = argparse.ArgumentParser(description="Record read-only old NO-tail live-forward intraday shadow logs.")
    parser.add_argument("--work-root", default=".")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--report-dir", default=None)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--interval-seconds", type=int, default=60)
    parser.add_argument("--discovery-pages", type=int, default=2)
    parser.add_argument("--discovery-request-limit", type=int, default=50)
    parser.add_argument("--market-limit", type=int, default=100)
    parser.add_argument("--starting-cash-usd", type=float, default=100.0)
    parser.add_argument("--http-timeout-seconds", type=float, default=HTTP_TIMEOUT_SECONDS)
    parser.add_argument(
        "--resolved-recheck-window-hours",
        type=float,
        default=DEFAULT_RESOLVED_RECHECK_WINDOW_HOURS,
    )
    parser.add_argument("--cache-only", action="store_true")
    args = parser.parse_args(argv)
    HTTP_TIMEOUT_SECONDS = max(1.0, float(args.http_timeout_seconds))

    artifacts = record_live_forward_shadow(
        work_root=Path(args.work_root),
        output_dir=Path(args.output_dir) if args.output_dir else None,
        report_dir=Path(args.report_dir) if args.report_dir else None,
        iterations=args.iterations,
        interval_seconds=args.interval_seconds,
        discovery_pages=args.discovery_pages,
        discovery_request_limit=args.discovery_request_limit,
        market_limit=args.market_limit,
        starting_cash_usd=args.starting_cash_usd,
        cache_only=args.cache_only,
        resolved_recheck_window_hours=args.resolved_recheck_window_hours,
    )
    print(
        json.dumps(
            {
                "summary_json": str(artifacts.json_path),
                "markdown": str(artifacts.markdown_path),
                "events_jsonl": str(artifacts.events_path),
                "raw_dir": str(artifacts.raw_dir),
                "settlement_refresh_json": str(
                    ((artifacts.summary.get("settlement_refresh") or {}).get("artifacts") or {}).get("summary_json")
                ),
                "settlement_refresh_markdown": str(
                    ((artifacts.summary.get("settlement_refresh") or {}).get("artifacts") or {}).get("markdown_report")
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
