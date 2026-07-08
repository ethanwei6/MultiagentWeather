from __future__ import annotations

import argparse
import concurrent.futures
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
from typing import Any, Iterable, Iterator, Mapping, Optional, Sequence
from zoneinfo import ZoneInfo

from .backtesting_engine import (
    OLD_NO_TAIL_LIVE_FORWARD_SHADOW_PREREGISTRATION,
    OLD_NO_TAIL_LIVE_FORWARD_SHADOW_PROFILE_KEY,
    DecisionSnapshot,
    ForecastSnapshot,
    ObservedHighSnapshot,
    QuoteSnapshot,
    ReplayFrame,
    SharedScoreContract,
    run_old_no_tail_live_forward_shadow_logging,
)
from .station_observations import (
    StationConfig,
    _get_or_fetch_raw_payload,
    _iem_request_url,
    parse_iem_csv_payload,
)


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

    @property
    def bucket_type(self) -> str:
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
    raw_status: str = "ok"
    error: Optional[str] = None

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
    CityConfig("Washington", "DC", 38.9072, -77.0369, "America/New_York", "KDCA", ("Washington DC", "D.C.")),
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
    )
    settlement_refresh = refresh_pending_would_trade_settlements(
        report_dir=reports,
        raw_dir=raw_dir / run_id / "settlement_refresh",
        run_id=run_id,
        cache_only=cache_only,
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
                "order": "volume_24hr",
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
    return results


def build_replay_frame(
    market: WeatherMarket,
    *,
    no_book: Optional[BookSnapshot],
    yes_book: Optional[BookSnapshot],
    forecasts: Sequence[ForecastSource],
    station_summary: StationObservationSummary,
    collected_at_utc: datetime,
) -> ReplayFrame:
    quote = quote_snapshot_from_book(no_book)
    forecast_snapshot = forecast_snapshot_from_sources(forecasts)
    observed = observed_high_snapshot_from_station(station_summary)
    score = shared_score_from_forecasts(
        market,
        forecasts=forecasts,
        station_summary=station_summary,
        no_book=no_book,
        yes_book=yes_book,
        scored_at=collected_at_utc,
    )
    decision_id = _stable_id(
        "live-shadow",
        market.market_id,
        market.bucket.no_token_id or "",
        collected_at_utc.isoformat(timespec="seconds"),
    )
    decision = DecisionSnapshot(
        decision_id=decision_id,
        universe_row_id=f"live:{market.market_id or market.slug}:{market.bucket.no_token_id or ''}",
        market_id=market.market_id,
        slug=market.slug,
        question=market.question,
        normalized_city=market.city.display_name,
        country_region=market.city.region,
        timezone=market.city.timezone,
        market_date_local=market.target_date.isoformat(),
        bucket_label=f"NO: {market.bucket.label}",
        bucket_type=market.bucket.bucket_type,
        threshold_low_f=market.bucket.lower_f,
        threshold_high_f=market.bucket.upper_f,
        side="NO",
        yes_token_id=market.bucket.yes_token_id,
        no_token_id=market.bucket.no_token_id,
        token_id=market.bucket.no_token_id or "",
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
    return ReplayFrame(decision=decision, canonical_label=None, skip_reasons=(), score=score)


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
        payload_id = _payload_id(url, payload)
        bids = _parse_levels(payload.get("bids") or [])
        asks = _parse_levels(payload.get("asks") or [])
        best_bid = max(bids, key=lambda item: item[0]) if bids else None
        best_ask = min(asks, key=lambda item: item[0]) if asks else None
        bid = best_bid[0] if best_bid else None
        ask = best_ask[0] if best_ask else None
        return BookSnapshot(
            token_id=token_id,
            fetched_at_utc=fetched_at,
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
            samples = tuple(extract_daily_temperature_samples(payload, target_date))
            features = extract_weather_features(payload, target_date)
            outputs.append(
                ForecastSource(
                    source_name=source_name,
                    fetched_at_utc=fetched_at,
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
        )
    except Exception as exc:
        return StationObservationSummary(
            station_id=station_id,
            status="error",
            fetched_at_utc=fetched_at,
            local_date=target_date.isoformat(),
            error=f"{type(exc).__name__}: {str(exc)[:300]}",
        )


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
    parsed = parse_temperature_bucket(question)
    if parsed is None:
        return None
    lower_f, upper_f, label = parsed
    yes_token = str(token_ids[yes_index]) if yes_index < len(token_ids) and token_ids[yes_index] else None
    no_token = str(token_ids[no_index]) if no_index < len(token_ids) and token_ids[no_index] else None
    return Bucket(
        label=label,
        lower_f=lower_f,
        upper_f=upper_f,
        yes_token_id=yes_token,
        no_token_id=no_token,
        yes_price=prices[yes_index] if yes_index < len(prices) else None,
        no_price=prices[no_index] if no_index < len(prices) else None,
    )


def parse_temperature_bucket(label: str) -> Optional[tuple[Optional[float], Optional[float], str]]:
    unit = _detect_unit(label)
    clean = " ".join(label.replace("deg", "").replace("F", "").replace("f", "").replace("C", "").replace("c", "").split())
    lower = clean.lower()
    range_match = re.search(r"(?P<low>-?\d+(?:\.\d+)?)\s*(?:-|to|through|thru)\s*(?P<high>-?\d+(?:\.\d+)?)", lower)
    if range_match:
        low = _to_fahrenheit(float(range_match.group("low")), unit)
        high = _to_fahrenheit(float(range_match.group("high")), unit)
        return low, high, f"{low:g}-{high:g}F"
    under_match = re.search(r"(?:under|below|less than|lower than|<)\s*(?P<value>-?\d+(?:\.\d+)?)", lower)
    if under_match:
        value = _to_fahrenheit(float(under_match.group("value")) - 0.01, unit)
        return None, value, f"below {value:g}F"
    at_most_match = re.search(r"(?:at most|no more than|<=)\s*(?P<value>-?\d+(?:\.\d+)?)", lower)
    if at_most_match:
        value = _to_fahrenheit(float(at_most_match.group("value")), unit)
        return None, value, f"at most {value:g}F"
    above_match = re.search(r"(?P<value>-?\d+(?:\.\d+)?)\s*(?:or above|or higher|and above|or more|\+)", lower)
    if above_match:
        value = _to_fahrenheit(float(above_match.group("value")), unit)
        return value, None, f"{value:g}F or above"
    over_match = re.search(r"(?:over|above|greater than|higher than|>)\s*(?P<value>-?\d+(?:\.\d+)?)", lower)
    if over_match:
        value = _to_fahrenheit(float(over_match.group("value")) + 0.01, unit)
        return value, None, f"above {value:g}F"
    predicate_match = re.search(
        r"\bbe\s+(?P<value>-?\d+(?:\.\d+)?)\s*(?:deg\s*)?(?P<unit>[cf])?\s*(?P<qualifier>or below|or lower|or less|or above|or higher|or more)?\b",
        label,
        re.I,
    )
    if predicate_match:
        predicate_unit = (predicate_match.group("unit") or unit).upper()
        value = _to_fahrenheit(float(predicate_match.group("value")), predicate_unit)
        qualifier = (predicate_match.group("qualifier") or "").lower()
        if "below" in qualifier or "lower" in qualifier or "less" in qualifier:
            return None, value, f"{value:g}F or below"
        if "above" in qualifier or "higher" in qualifier or "more" in qualifier:
            return value, None, f"{value:g}F or above"
        return value, value + 0.999, f"{value:g}F"
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
            for key in ("temperature_2m", "apparent_temperature", "relative_humidity_2m", "dew_point_2m"):
                values = [_single_value_at_index(hourly.get(key), index) for index in indices]
                cleaned = [float(value) for value in values if value is not None]
                if cleaned:
                    features[f"hourly_{key}_max"] = max(cleaned)
                    features[f"hourly_{key}_mean"] = sum(cleaned) / len(cleaned)
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
    resolved_would_trade_clusters = int(cluster_status_counts.get("resolved") or 0)
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
            "status": "not_evaluable_yet",
            "reason": "requires at least 30 resolved deduped would-trade clusters or 7-14 days of forward logs",
            "current_run_would_trade_signals": would_trade_count,
            "current_run_resolved_would_trade_signals": resolved_would_trades,
            "gate_count_basis": "deduped_true_opportunity_clusters",
            "current_run_would_trade_signal_clusters": len(would_trade_cluster_settlement_status),
            "current_run_resolved_would_trade_clusters": resolved_would_trade_clusters,
            "pre_registered_expected_trades_per_day": dict(
                OLD_NO_TAIL_LIVE_FORWARD_SHADOW_PREREGISTRATION["expected_trades_per_utc_day"]
            ),
            "pre_registered_hit_rate_confidence_bands": dict(
                OLD_NO_TAIL_LIVE_FORWARD_SHADOW_PREREGISTRATION["hit_rate_confidence_bands"]
            ),
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
) -> Mapping[str, Any]:
    generated = generated_at or datetime.now(timezone.utc)
    report_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    updates: list[Mapping[str, Any]] = []
    gamma_event_cache: dict[str, tuple[Optional[Mapping[str, Any]], Optional[str], Optional[str], Optional[str], Optional[str]]] = {}
    pending_clusters = _pending_would_trade_clusters(report_dir)
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
        updates.append(
            _settlement_refresh_update(
                record=record,
                identifiers=identifiers,
                fetched_at=generated,
                settlement=_settlement_from_gamma_market(
                    gamma_market,
                    identifiers=identifiers,
                    request_url=request_url,
                    raw_cache_path=raw_cache_path,
                    payload_id=payload_id,
                ),
            )
        )
    status_counts = Counter(str((update.get("settlement") or {}).get("status") or "unknown") for update in updates)
    raw_pending_events_scanned = sum(int(record.get("raw_signal_count") or 1) for record in pending_clusters)
    json_path = report_dir / f"{run_id}-old-no-tail-live-forward-shadow-settlement-refresh.json"
    markdown_path = report_dir / f"{run_id}-old-no-tail-live-forward-shadow-settlement-refresh.md"
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
        "gamma_events_fetched": sum(1 for payload, _, _, _, error in gamma_event_cache.values() if payload is not None and error is None),
        "settlement_status_counts": dict(sorted(status_counts.items())),
        "settlement_cluster_status_counts": dict(sorted(status_counts.items())),
        "resolved_would_trade_signals": int(status_counts.get("resolved") or 0),
        "resolved_would_trade_clusters": int(status_counts.get("resolved") or 0),
        "updates": updates,
        "artifacts": {
            "summary_json": str(json_path),
            "markdown_report": str(markdown_path),
            "raw_payload_dir": str(raw_dir),
        },
    }
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
            settlement = event.get("settlement") if isinstance(event, Mapping) else {}
            if isinstance(settlement, Mapping) and settlement.get("status") == "resolved":
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


def _pending_would_trade_clusters(report_dir: Path) -> list[Mapping[str, Any]]:
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
        if latest_state and latest_state.get("status") == "resolved":
            continue
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
            },
        )
        cluster["event"] = event
        cluster["identifiers"] = identifiers
        cluster["events_path"] = record.get("events_path")
        cluster["line_number"] = record.get("line_number")
        cluster["last_seen_events_jsonl"] = str(record.get("events_path"))
        cluster["last_seen_line_number"] = record.get("line_number")
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
    }
    if not closed:
        return {
            **base,
            "status": "pending",
            "later_update_required": True,
            "outcome_logged": False,
            "reason": "gamma_market_not_closed",
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
        "settlement": dict(settlement),
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
    artifacts = 0
    for summary in summaries:
        artifacts += 1
        generated = str(summary.get("generated_at_utc") or summary.get("started_at_utc") or "")
        if generated:
            days.add(generated[:10])
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
    )
    gate_resolved_clusters = int(unique_signal_summary["unique_resolved_market_side_signal_clusters"])
    return {
        "artifact_count_in_report_dir_including_current": artifacts,
        "days_with_forward_logs": len(days),
        "logged_utc_dates": sorted(days),
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
        "raw_signals_are_poll_level_not_independent_contracts": True,
        "target_resolved_would_trades": 30,
        "target_resolved_would_trade_clusters": 30,
        "target_forward_log_days_min": 7,
        "target_forward_log_days_max": 14,
        "collection_complete": gate_resolved_clusters >= 30 or len(days) >= 14,
    }


def _unique_would_trade_signal_summary(
    report_dir: Path,
    *,
    latest_cluster_settlements: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any]:
    clusters: dict[str, dict[str, Any]] = {}
    cluster_status: dict[str, str] = {}
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
            identifiers = _settlement_identifiers_from_event(event)
            decision_id = str(identifiers.get("decision_id") or "")
            cluster_key = _would_trade_cluster_key(identifiers)
            if not cluster_key:
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

    for cluster_key, state in latest_cluster_settlements.items():
        status = str(state.get("status") or "unknown")
        previous = cluster_status.get(cluster_key)
        cluster_status[cluster_key] = _more_final_settlement_status(previous, status)
        clusters.setdefault(cluster_key, {"cluster_key": cluster_key})["latest_settlement"] = dict(state)
    status_counts = Counter()
    for cluster_key, cluster in clusters.items():
        status = cluster_status.get(cluster_key, "pending")
        cluster["latest_status"] = status
        status_counts[status] += 1
    resolved = int(status_counts.get("resolved") or 0)
    return {
        "unique_market_side_signal_clusters": len(clusters),
        "unique_market_side_signal_cluster_status_counts": dict(sorted(status_counts.items())),
        "unique_resolved_market_side_signal_clusters": resolved,
        "unique_unresolved_market_side_signal_clusters": max(0, len(clusters) - resolved),
        "clusters_preview": list(clusters.values())[:20],
    }


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
        "resolved": 4,
        "excluded_or_manual_review": 3,
        "unresolved_or_ambiguous": 2,
        "refresh_error": 1,
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
                    "source": settlement.get("source"),
                    "cluster_key": cluster_key,
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
                "source": settlement.get("source"),
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
    live_vs_replay = summary.get("live_vs_replay_behavior") or {}
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
        "- Hit-rate confidence bands: Wilson 95% primary, Wilson 80% secondary after at least 30 resolved would-trades.",
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


def update_migration_brief_with_artifacts(
    brief_path: Path,
    *,
    artifacts: LiveShadowArtifacts,
) -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    summary = artifacts.summary
    decisions = summary.get("decision_counts") or {}
    quote_quality = summary.get("quote_quality") or {}
    progress = summary.get("forward_collection_progress") or {}
    settlement_refresh = summary.get("settlement_refresh") or {}
    close_times = summary.get("close_time_diagnostics") or {}
    reasons = summary.get("no_trade_reason_counts") or {}
    refresh_artifacts = settlement_refresh.get("artifacts") or {}
    refresh_counts = settlement_refresh.get("settlement_status_counts") or {}
    block = [
        "",
        f"- {timestamp} - Live Forward Shadow / Intraday Recorder Agent",
        "  - Change: started read-only live-forward shadow collection for the frozen old NO-tail family only (`old_high_confidence_no_tail`, `no14`, flat `$2`, window-bankroll, `bprice70`).",
        f"  - Files/artifacts changed: `{artifacts.json_path}`, `{artifacts.markdown_path}`, `{artifacts.events_path}`, `{artifacts.raw_dir}`, `{refresh_artifacts.get('summary_json')}`, `{refresh_artifacts.get('markdown_report')}`.",
        (
            "  - Metrics: "
            f"`{decisions.get('shadow_decisions', 0)}` shadow decisions, "
            f"`{decisions.get('raw_would_trade_poll_detections', decisions.get('would_trade_signals', 0))}` raw would-trade poll detections, "
            f"`{decisions.get('would_trade_signal_clusters', 0)}` deduped would-trade clusters, "
            f"`{decisions.get('no_trade_decisions', 0)}` no-trade decisions; "
            f"top no-trade reasons `{_brief_top_counts(reasons)}`."
        ),
        (
            "  - Settlement refresh: "
            f"`{settlement_refresh.get('raw_pending_would_trade_events_scanned', settlement_refresh.get('pending_would_trade_events_scanned', 0))}` prior pending raw poll detections scanned; "
            f"`{settlement_refresh.get('pending_would_trade_clusters_scanned', 0)}` deduped clusters scanned; "
            f"cluster status counts `{dict(refresh_counts)}`. "
            "Refresh artifacts are append-only and do not mutate earlier intraday event logs."
        ),
        (
            "  - Quote/close diagnostics: "
            f"quote counts `{dict((quote_quality.get('counts') or {}))}`, "
            f"visible top-depth median `{(quote_quality.get('visible_top_depth') or {}).get('median')}`, "
            f"spread median `{(quote_quality.get('spread') or {}).get('median')}`, "
            f"close-time counts `{dict(close_times)}`."
        ),
        (
            "  - Forward progress: "
            f"`{progress.get('days_with_forward_logs', 0)}` UTC log days, "
            f"`{progress.get('would_trade_signals', 0)}` cumulative raw would-trade poll detections, "
            f"`{progress.get('resolved_would_trade_signals', 0)}` raw resolved, "
            f"`{progress.get('unresolved_would_trade_signals', 0)}` raw unresolved; "
            f"`{progress.get('unique_market_side_signal_clusters', 0)}` unique market-side signal clusters, "
            f"`{progress.get('unique_resolved_market_side_signal_clusters', 0)}` unique resolved, "
            f"`{progress.get('unique_unresolved_market_side_signal_clusters', 0)}` unique unresolved; "
            "live-vs-replay behavior remains not evaluable until the pre-registered deduped cluster/time gate is met."
        ),
        "  - Validation: recorder artifacts mark `real_order_placement_allowed=false`, `real_orders_placed=0`, and keep assumed executable price separate from observed CLOB book state. No fills are simulated as truth and no historical PnL is treated as expected performance.",
        "  - Follow-up risks: continue collecting forward rows at 1-5 minute cadence where feasible; settlement outcomes remain pending until canonical Polymarket outcomes are available.",
    ]
    text = brief_path.read_text(encoding="utf-8")
    marker = "## Project Change Log"
    if marker in text:
        prefix, suffix = text.split(marker, 1)
        updated = prefix + marker + "\n" + "\n".join(block) + suffix
    else:
        updated = text.rstrip() + "\n\n## Project Change Log\n" + "\n".join(block) + "\n"
    brief_path.write_text(updated, encoding="utf-8")


def _brief_top_counts(counts: Mapping[str, Any], *, limit: int = 5) -> str:
    if not counts:
        return "none"
    rows = sorted(
        ((str(key), int(value or 0)) for key, value in counts.items()),
        key=lambda item: (-item[1], item[0]),
    )
    return ", ".join(f"{key}={value}" for key, value in rows[:limit])


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
        "payload_id": book.payload_id,
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
    if re.search(r"(?:deg\s*)?c\b", label, re.I):
        return "C"
    return "F"


def _to_fahrenheit(value: float, unit: str) -> float:
    return value * 9 / 5 + 32 if unit.upper() == "C" else value


def _apply_resolution_precision(bucket: Bucket, text: str) -> Bucket:
    lowered = text.lower()
    if "whole degrees celsius" in lowered or "whole degree celsius" in lowered:
        return replace(bucket, resolution_unit="C", resolution_precision=1.0)
    if "whole degrees fahrenheit" in lowered or "whole degree fahrenheit" in lowered:
        return replace(bucket, resolution_unit="F", resolution_precision=1.0)
    return bucket


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
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--update-brief", action="store_true")
    parser.add_argument("--brief-path", default="DailyWeather_PRIVATE_MIGRATION_BRIEF.md")
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
    )
    if args.update_brief:
        update_migration_brief_with_artifacts(Path(args.brief_path), artifacts=artifacts)
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
