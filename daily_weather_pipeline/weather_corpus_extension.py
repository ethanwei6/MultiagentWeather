from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from contextlib import closing
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional
from zoneinfo import ZoneInfo

from .cli import (
    DEFAULT_SOURCE_REPO,
    PipelinePaths,
    _connect,
    _decision_times_for_pair,
    _fetch_historical_forecast_payload,
    _fetch_realized_payload,
    _finish_pipeline_run,
    _forecast_row,
    _insert_forecast_row,
    _insert_market_row,
    _insert_pipeline_run,
    _insert_realized_row,
    _load_city_date_pairs,
    _realized_row_from_payload,
    _stable_id,
    _upsert_weather_status,
    initialize_weather_corpus,
)
from .live_forward_shadow_recorder import (
    DEFAULT_CITIES,
    GAMMA_BASE_URL,
    HTTP_TIMEOUT_SECONDS,
    USER_AGENT,
    CityConfig,
    WeatherMarket,
    _iter_event_markets,
    parse_weather_market,
)
from .weather_model_runner import ObservedHighGuardedShrinkageWeatherModelRunner
from .weather_model_validation import resolve_training_corpus_end_date
from .weather_corpus_coverage import weather_corpus_coverage
from .weather_corpus_snapshot import (
    active_pointer_path,
    create_corpus_candidate,
    discard_corpus_candidate,
    exclusive_corpus_writer_lock,
    normalize_corpus_candidate_databases,
    promote_corpus_candidate,
    resolve_corpus_snapshot,
)


DEFAULT_LOOKBACK_DAYS = 4
LABEL_AVAILABILITY_LAG_HOURS = 12.0


def run_weather_corpus_extension(
    *,
    paths: PipelinePaths,
    start: date,
    end: date,
    as_of_utc: Optional[datetime] = None,
    cache_only: bool = False,
    market_discovery_mode: str = "cached-and-online",
    gamma_city_limit: int = 0,
    weather_pair_limit: int = 0,
    report_dir: Optional[Path] = None,
    source_repo: Path = DEFAULT_SOURCE_REPO,
) -> dict[str, Any]:
    """Extend market/weather corpus rows needed by the live weather-model runner.

    This path intentionally avoids Telonex and quote history. It only writes market
    definitions from public Polymarket Gamma payloads and weather rows from
    Open-Meteo archive/historical-forecast endpoints.
    """

    as_of = _normalize_utc(as_of_utc or datetime.now(timezone.utc))
    if market_discovery_mode not in {"cached", "online", "cached-and-online"}:
        raise ValueError(f"Unknown market_discovery_mode: {market_discovery_mode}")
    with exclusive_corpus_writer_lock(paths.root) as lock:
        if not active_pointer_path(paths.root).exists():
            initialize_weather_corpus(paths)
        candidate = create_corpus_candidate(paths)
        try:
            report = _run_weather_corpus_candidate(
                candidate=candidate,
                source_repo=source_repo,
                start=start,
                end=end,
                as_of=as_of,
                cache_only=cache_only,
                market_discovery_mode=market_discovery_mode,
                gamma_city_limit=gamma_city_limit,
                weather_pair_limit=weather_pair_limit,
            )
            report["corpus_writer_lock"] = {
                "status": "acquired",
                "owner_token": lock.get("owner_token"),
                "acquired_at_utc": lock.get("acquired_at_utc"),
            }
            if bool(report.get("operational_corpus_ready")):
                coverage = report.get("operational_corpus_coverage") or {}
                promotion = promote_corpus_candidate(
                    candidate,
                    validation_summary={
                        "operationally_fresh": True,
                        "coverage_contract_version": coverage.get(
                            "coverage_contract_version"
                        ),
                        "start_date": start.isoformat(),
                        "end_date": end.isoformat(),
                        "as_of_utc": as_of.isoformat(),
                        "labelable_city_date_pairs": coverage.get(
                            "labelable_city_date_pairs"
                        ),
                        "complete_city_date_pairs": coverage.get(
                            "complete_city_date_pairs"
                        ),
                        "required_city_source_date_checks": coverage.get(
                            "required_city_source_date_checks"
                        ),
                        "covered_city_source_date_checks": coverage.get(
                            "covered_city_source_date_checks"
                        ),
                    },
                )
                report["snapshot_promotion"] = dict(promotion)
                active = resolve_corpus_snapshot(paths.root)
                active_coverage = weather_corpus_coverage(
                    work_root=paths.root,
                    start=start,
                    end=end,
                    as_of_utc=as_of,
                    label_availability_lag_hours=LABEL_AVAILABILITY_LAG_HOURS,
                    market_db=active.market_db,
                    weather_db=active.weather_db,
                    snapshot_id=active.snapshot_id,
                    snapshot_source=active.source,
                )
                report["operational_corpus_coverage"] = active_coverage
                report["training_corpus_summary"] = dict(
                    _training_summary(paths=paths, as_of_utc=as_of, source_repo=source_repo)
                )
            else:
                validation_reasons = list(
                    (report.get("operational_corpus_coverage") or {}).get(
                        "fail_closed_reasons"
                    )
                    or []
                )
                if int((report.get("market_extension") or {}).get("error_count") or 0):
                    validation_reasons.append("market_extension_errors_present")
                if str((report.get("weather_extension") or {}).get("status") or "") != "ok":
                    validation_reasons.append("weather_extension_incomplete")
                report["snapshot_promotion"] = {
                    "status": "validation_failed_prior_snapshot_retained",
                    "previous_snapshot_id": candidate.base_resolution.snapshot_id,
                    "pointer_changed": False,
                    "fail_closed_reasons": list(dict.fromkeys(validation_reasons)),
                }
                discard_corpus_candidate(candidate)
            data_quality = _write_corpus_data_quality_report(
                paths,
                report.get("operational_corpus_coverage") or {},
            )
            report["data_quality_report_json"] = data_quality.get("json_path")
            artifacts = _write_extension_report(paths, report, report_dir=report_dir)
            report["artifacts"] = artifacts
            Path(artifacts["json"]).write_text(
                json.dumps(report, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            Path(artifacts["markdown"]).write_text(
                _render_extension_markdown(report),
                encoding="utf-8",
            )
            return report
        except Exception:
            discard_corpus_candidate(candidate)
            raise


def _run_weather_corpus_candidate(
    *,
    source_repo: Path,
    candidate: Any,
    start: date,
    end: date,
    as_of: datetime,
    cache_only: bool,
    market_discovery_mode: str,
    gamma_city_limit: int,
    weather_pair_limit: int,
) -> dict[str, Any]:
    """Build and validate one private copy-on-write corpus candidate."""

    paths = candidate.paths

    market_result = extend_market_universe_from_gamma(
        paths=paths,
        start=start,
        end=end,
        as_of_utc=as_of,
        cache_only=cache_only,
        mode=market_discovery_mode,
        gamma_city_limit=gamma_city_limit,
    )
    weather_result = extend_weather_history_for_labelable_pairs(
        paths=paths,
        start=start,
        end=end,
        as_of_utc=as_of,
        cache_only=cache_only,
        pair_limit=weather_pair_limit,
    )
    normalize_corpus_candidate_databases(candidate)
    coverage = weather_corpus_coverage(
        work_root=paths.root,
        start=start,
        end=end,
        as_of_utc=as_of,
        label_availability_lag_hours=LABEL_AVAILABILITY_LAG_HOURS,
        market_db=paths.market_db,
        weather_db=paths.weather_db,
        snapshot_source="copy_on_write_candidate",
    )
    operationally_fresh = (
        bool(coverage.get("operationally_fresh"))
        and int(market_result.get("error_count") or 0) == 0
        and str(weather_result.get("status") or "") == "ok"
    )
    if operationally_fresh:
        training_summary = _training_summary(
            source_repo=source_repo,
            paths=paths,
            as_of_utc=as_of,
            use_explicit_corpus_paths=True,
        )
    else:
        training_summary = {
            "training_corpus_end_date": None,
            "training_row_count": 0,
            "training_corpus_staleness_warning": True,
            "training_corpus_operationally_fresh": False,
            "training_corpus_coverage_available": coverage.get("coverage_available"),
            "training_corpus_snapshot_id": None,
            "training_corpus_snapshot_source": "rejected_copy_on_write_candidate",
            "warning_flags": ("candidate_corpus_validation_incomplete",),
        }
    report = {
        "status": "ok" if operationally_fresh else "incomplete",
        "role": "DailyWeather Data/Weather Corpus Agent",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "as_of_utc": as_of.isoformat(),
        "scope": {
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "label_availability_lag_hours": LABEL_AVAILABILITY_LAG_HOURS,
            "market_discovery_mode": market_discovery_mode,
            "cache_only": cache_only,
            "gamma_city_limit": gamma_city_limit,
            "weather_pair_limit": weather_pair_limit,
        },
        "row_builder_contract": {
            "market_rows": "market_universe rows keyed by normalized_city and market_date_local, with bucket thresholds and timezone",
            "forecast_rows": "forecast_snapshots rows keyed by normalized_city, local_date, decision_time_utc",
            "realized_rows": "realized_daily_weather rows keyed by normalized_city and local_date",
            "training_join": "_build_validation_rows joins market rows + forecast rows + realized rows by city/date; quote history is not required",
            "leakage_firewall": "runner still filters labels by local target day end plus lag <= decision_time_utc",
        },
        "market_extension": market_result,
        "weather_extension": weather_result,
        "training_corpus_summary": dict(training_summary),
        "operational_corpus_coverage": coverage,
        "operational_corpus_ready": operationally_fresh,
        "missing_data_product": _missing_data_product(
            training_summary,
            market_result,
            weather_result,
            coverage,
        ),
    }
    if not operationally_fresh and report["missing_data_product"] is None:
        report["missing_data_product"] = {
            "product": "validated atomic corpus candidate",
            "reason": "Candidate validation did not pass every market, weather, and coverage gate; the prior active snapshot will be retained.",
        }
    return report


def extend_market_universe_from_gamma(
    *,
    paths: PipelinePaths,
    start: date,
    end: date,
    as_of_utc: datetime,
    cache_only: bool,
    mode: str,
    gamma_city_limit: int = 0,
) -> dict[str, Any]:
    rows: dict[str, Mapping[str, Any]] = {}
    source_counts: Counter[str] = Counter()
    errors: list[dict[str, str]] = []
    if mode in {"cached", "cached-and-online"}:
        for (
            payload,
            source_path,
            payload_id,
            source_file_sha256,
        ) in iter_cached_gamma_event_payloads(paths, start=start, end=end):
            for row in market_rows_from_gamma_event_payload(
                payload,
                source_path=source_path,
                payload_id=payload_id,
                source_file_sha256=source_file_sha256,
                as_of_utc=as_of_utc,
                start=start,
                end=end,
            ):
                rows[str(row["universe_row_id"])] = row
                source_counts["cached_shadow_gamma_event"] += 1
    if mode in {"online", "cached-and-online"} and not cache_only:
        for (
            payload,
            source_path,
            payload_id,
            source_file_sha256,
            error,
        ) in fetch_gamma_event_payloads(
            paths,
            start=start,
            end=end,
            as_of_utc=as_of_utc,
            gamma_city_limit=gamma_city_limit,
        ):
            if error:
                errors.append(error)
                continue
            if source_file_sha256 is None:
                errors.append(
                    {
                        "path": str(source_path),
                        "error": "gamma_event_source_file_sha256_missing",
                    }
                )
                continue
            for row in market_rows_from_gamma_event_payload(
                payload,
                source_path=source_path,
                payload_id=payload_id,
                source_file_sha256=source_file_sha256,
                as_of_utc=as_of_utc,
                start=start,
                end=end,
            ):
                rows[str(row["universe_row_id"])] = row
                source_counts["online_gamma_event"] += 1

    with closing(_connect(paths.market_db)) as conn:
        run_id = _insert_pipeline_run(
            conn,
            "weather-corpus-market-extension",
            {
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
                "as_of_utc": as_of_utc.isoformat(),
                "mode": mode,
                "cache_only": cache_only,
                "gamma_city_limit": gamma_city_limit,
            },
        )
        for row in rows.values():
            _insert_market_row(conn, row, run_id)
        orphan_source_payload_rows_pruned = conn.execute(
            """
            DELETE FROM source_payloads
            WHERE source_name = 'polymarket_gamma_public_event'
              AND NOT EXISTS (
                    SELECT 1
                    FROM market_universe
                    WHERE market_universe.source_payload_id = source_payloads.payload_id
              )
            """
        ).rowcount
        _finish_pipeline_run(conn, run_id, row_count=len(rows))
        conn.commit()

    by_date: Counter[str] = Counter(str(row["market_date_local"]) for row in rows.values())
    return {
        "status": "ok",
        "market_rows_written": len(rows),
        "market_rows_by_date": dict(sorted(by_date.items())),
        "source_counts": dict(source_counts),
        "orphan_source_payload_rows_pruned": int(
            max(0, orphan_source_payload_rows_pruned)
        ),
        "errors_sample": errors[:20],
        "error_count": len(errors),
    }


def extend_weather_history_for_labelable_pairs(
    *,
    paths: PipelinePaths,
    start: date,
    end: date,
    as_of_utc: datetime,
    cache_only: bool,
    pair_limit: int = 0,
) -> dict[str, Any]:
    pairs = _load_city_date_pairs(paths, start=start, end=end, limit=0)
    labelable_pairs = [pair for pair in pairs if pair_label_available_at_utc(pair) <= as_of_utc]
    states_before = _classify_weather_pairs(paths, labelable_pairs)
    counts_before = Counter(state for _pair, state in states_before)
    selectable_pairs = [pair for pair, state in states_before if state != "complete"]
    selected_pairs = list(selectable_pairs)
    if pair_limit > 0:
        selected_pairs = selected_pairs[: int(pair_limit)]
    forecast_rows_written = 0
    realized_rows_written = 0
    failed_pairs_this_run = 0
    with closing(_connect(paths.weather_db)) as conn:
        run_id = _insert_pipeline_run(
            conn,
            "weather-corpus-weather-extension",
            {
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
                "as_of_utc": as_of_utc.isoformat(),
                "cache_only": cache_only,
                "pair_limit": pair_limit,
                "label_availability_lag_hours": LABEL_AVAILABILITY_LAG_HOURS,
            },
        )
        for pair in selected_pairs:
            local_date = date.fromisoformat(str(pair["local_date"]))
            try:
                forecast_payload, forecast_path, forecast_payload_id, forecast_url, forecast_params = _fetch_historical_forecast_payload(
                    pair,
                    local_date,
                    local_date,
                    paths=paths,
                    cache_only=cache_only,
                )
                realized_payload, realized_path, realized_payload_id, realized_url, realized_params = _fetch_realized_payload(
                    pair,
                    local_date,
                    local_date,
                    paths=paths,
                    cache_only=cache_only,
                )
                for decision_time in _decision_times_for_pair(pair):
                    row = _forecast_row(
                        pair,
                        local_date,
                        decision_time,
                        _forecast_run_time_for_extension(decision_time),
                        "historical_forecast_archive",
                        None,
                        forecast_payload,
                        forecast_url,
                        forecast_params,
                        forecast_path,
                        forecast_payload_id,
                        flags=["historical_forecast_endpoint_is_asof_approximation"],
                    )
                    _insert_forecast_row(conn, row)
                    forecast_rows_written += 1
                realized_row = _realized_row_from_payload(
                    pair,
                    local_date,
                    realized_payload,
                    realized_url,
                    realized_params,
                    realized_path,
                    realized_payload_id,
                )
                _insert_realized_row(conn, realized_row)
                realized_rows_written += 1
                _upsert_weather_status(conn, pair, "complete", error=None)
            except Exception as exc:  # noqa: BLE001 - persisted per-pair status for resumability
                failed_pairs_this_run += 1
                _upsert_weather_status(conn, pair, "failed", error=f"{type(exc).__name__}: {str(exc)[:500]}")
        _finish_pipeline_run(conn, run_id, row_count=forecast_rows_written + realized_rows_written)
        conn.commit()
    states_after = _classify_weather_pairs(paths, labelable_pairs)
    counts_after = Counter(state for _pair, state in states_after)
    remaining_unselected = max(0, len(selectable_pairs) - len(selected_pairs))
    return {
        "status": (
            "ok"
            if counts_after.get("pending", 0) == 0 and counts_after.get("failed", 0) == 0
            else "incomplete"
        ),
        "city_date_pairs_in_window": len(pairs),
        "labelable_city_date_pairs": len(labelable_pairs),
        "not_yet_labelable_city_date_pairs": len(pairs) - len(labelable_pairs),
        "pair_limit": int(pair_limit),
        "pair_counts_before_selection": {
            "complete": int(counts_before.get("complete", 0)),
            "pending": int(counts_before.get("pending", 0)),
            "failed": int(counts_before.get("failed", 0)),
            "skipped_existing": int(counts_before.get("complete", 0)),
        },
        "complete_city_date_pairs_before_selection": int(counts_before.get("complete", 0)),
        "pending_city_date_pairs_before_selection": int(counts_before.get("pending", 0)),
        "failed_city_date_pairs_before_selection": int(counts_before.get("failed", 0)),
        "skipped_existing_pairs": int(counts_before.get("complete", 0)),
        "selectable_city_date_pairs": len(selectable_pairs),
        "selected_city_date_pairs": len(selected_pairs),
        "remaining_unselected_city_date_pairs": remaining_unselected,
        "failed_pairs_this_run": failed_pairs_this_run,
        "failed_pairs": failed_pairs_this_run,
        "pair_counts_after_update": {
            "complete": int(counts_after.get("complete", 0)),
            "pending": int(counts_after.get("pending", 0)),
            "failed": int(counts_after.get("failed", 0)),
        },
        "complete_city_date_pairs": int(counts_after.get("complete", 0)),
        "pending_city_date_pairs": int(counts_after.get("pending", 0)),
        "failed_city_date_pairs": int(counts_after.get("failed", 0)),
        "forecast_rows_written": forecast_rows_written,
        "realized_rows_written": realized_rows_written,
        "labelable_dates": sorted({str(pair["local_date"]) for pair in labelable_pairs}),
    }


def iter_cached_gamma_event_payloads(
    paths: PipelinePaths,
    *,
    start: date,
    end: date,
) -> Iterable[tuple[Mapping[str, Any], Path, str, str]]:
    candidates: list[Path] = []
    raw_root = paths.work / "logs" / "live_forward_shadow_intraday" / "raw"
    if raw_root.exists():
        candidates.extend(raw_root.glob("**/settlement_refresh/gamma-event-*.json"))
    corpus_cache = paths.work / "cache" / "polymarket_gamma" / "weather_corpus"
    if corpus_cache.exists():
        candidates.extend(corpus_cache.glob("gamma-event-*.json"))
    for path in sorted(candidates):
        if path.name.endswith(".request.json"):
            continue
        try:
            cache_bytes = path.read_bytes()
            payload = json.loads(cache_bytes.decode("utf-8"))
        except Exception:
            continue
        if not isinstance(payload, Mapping):
            continue
        if not _event_intersects_window(payload, start=start, end=end):
            continue
        payload_id, source_file_sha256 = _gamma_event_file_identity(
            paths,
            source_path=path,
            cache_bytes=cache_bytes,
        )
        yield payload, path, payload_id, source_file_sha256


def fetch_gamma_event_payloads(
    paths: PipelinePaths,
    *,
    start: date,
    end: date,
    as_of_utc: datetime,
    gamma_city_limit: int = 0,
) -> Iterable[
    tuple[
        Mapping[str, Any],
        Path,
        str,
        Optional[str],
        Optional[dict[str, str]],
    ]
]:
    raw_dir = paths.work / "cache" / "polymarket_gamma" / "weather_corpus"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for slug in candidate_gamma_event_slugs(start=start, end=end, gamma_city_limit=gamma_city_limit):
        cache_path = raw_dir / f"gamma-event-{slug}.json"
        request_path = raw_dir / f"gamma-event-{slug}.request.json"
        url = f"{GAMMA_BASE_URL}/events/slug/{urllib.parse.quote(slug)}"
        try:
            if cache_path.exists():
                cache_bytes = cache_path.read_bytes()
                payload = json.loads(cache_bytes.decode("utf-8"))
            else:
                request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
                with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_SECONDS) as response:  # noqa: S310 - public Gamma endpoint
                    payload = json.loads(response.read().decode("utf-8"))
                cache_bytes = json.dumps(payload, indent=2, sort_keys=True).encode(
                    "utf-8"
                )
                cache_path.write_bytes(cache_bytes)
                request_path.write_text(
                    json.dumps({"url": url, "fetched_at_utc": as_of_utc.isoformat()}, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
            if isinstance(payload, Mapping):
                payload_id, source_file_sha256 = _gamma_event_file_identity(
                    paths,
                    source_path=cache_path,
                    cache_bytes=cache_bytes,
                )
                yield (
                    payload,
                    cache_path,
                    payload_id,
                    source_file_sha256,
                    None,
                )
        except urllib.error.HTTPError as exc:
            if exc.code != 404:
                yield (
                    {},
                    cache_path,
                    "",
                    None,
                    {"slug": slug, "error": f"HTTPError:{exc.code}"},
                )
        except Exception as exc:  # noqa: BLE001
            yield (
                {},
                cache_path,
                "",
                None,
                {"slug": slug, "error": f"{type(exc).__name__}: {str(exc)[:200]}"},
            )


def _gamma_event_file_identity(
    paths: PipelinePaths,
    *,
    source_path: Path,
    cache_bytes: bytes,
) -> tuple[str, str]:
    source_file_sha256 = hashlib.sha256(cache_bytes).hexdigest()
    source_identity = _normalized_gamma_source_path(paths.root, source_path)
    payload_id = _stable_id(
        "gamma-event-cache-file-v1",
        source_identity,
        source_file_sha256,
    )
    return payload_id, source_file_sha256


def _normalized_gamma_source_path(root: Path, source_path: Path) -> str:
    workspace = Path(root).resolve()
    resolved = Path(source_path).resolve()
    try:
        return resolved.relative_to(workspace).as_posix()
    except ValueError:
        return resolved.as_posix()


def candidate_gamma_event_slugs(
    *,
    start: date,
    end: date,
    gamma_city_limit: int = 0,
) -> list[str]:
    cities = list(DEFAULT_CITIES)
    if gamma_city_limit > 0:
        cities = cities[:gamma_city_limit]
    slugs: list[str] = []
    current = start
    while current <= end:
        for city in cities:
            for city_slug in _city_slug_candidates(city):
                slugs.append(
                    f"highest-temperature-in-{city_slug}-on-{current.strftime('%B').lower()}-{current.day}-{current.year}"
                )
        current += timedelta(days=1)
    return sorted(dict.fromkeys(slugs))


def market_rows_from_gamma_event_payload(
    payload: Mapping[str, Any],
    *,
    source_path: Path,
    payload_id: str,
    source_file_sha256: str,
    as_of_utc: datetime,
    start: date,
    end: date,
) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    for raw in _iter_event_markets((payload,)):
        market = parse_weather_market(raw, today=as_of_utc.date())
        if market is None or market.target_date < start or market.target_date > end:
            continue
        rows.append(
            gamma_market_to_universe_row(
                market,
                raw=raw,
                source_path=source_path,
                payload_id=payload_id,
                source_file_sha256=source_file_sha256,
            )
        )
    return rows


def gamma_market_to_universe_row(
    market: WeatherMarket,
    *,
    raw: Mapping[str, Any],
    source_path: Path,
    payload_id: str,
    source_file_sha256: Optional[str] = None,
) -> Mapping[str, Any]:
    row_id = _stable_id(
        "market-row",
        "gamma",
        market.market_id,
        market.slug,
        market.bucket.label,
        market.bucket.yes_token_id,
    )
    warnings = _gamma_market_warnings(market)
    source_json = json.dumps(_jsonable(raw), sort_keys=True, default=str)
    request_source = "gamma_event_payload"
    if source_file_sha256 is None:
        if source_path.is_file():
            source_file_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
        else:
            # Backward-compatible support for direct row-conversion callers that
            # do not own an event cache file. File-backed ingestion always passes
            # the full physical cache-file digest explicitly.
            source_file_sha256 = hashlib.sha256(
                source_json.encode("utf-8")
            ).hexdigest()
            request_source = "gamma_market_inline_compatibility"
    if (
        len(source_file_sha256) != 64
        or source_file_sha256 != source_file_sha256.lower()
        or any(character not in "0123456789abcdef" for character in source_file_sha256)
    ):
        raise ValueError("source_file_sha256 must be a lowercase SHA-256 digest")
    return {
        "universe_row_id": row_id,
        "market_id": market.market_id,
        "condition_id": market.condition_id,
        "slug": market.slug,
        "question": market.question,
        "description": str(raw.get("description") or raw.get("rules") or raw.get("resolutionSource") or ""),
        "event_id": _none_or_str(raw.get("eventId") or raw.get("event_id")),
        "event_slug": market.event_slug,
        "event_title": _none_or_str(raw.get("eventTitle") or raw.get("event_title")),
        "city_text": market.city.display_name,
        "normalized_city": market.city.display_name,
        "country_region": market.city.region,
        "latitude": market.city.latitude,
        "longitude": market.city.longitude,
        "timezone": market.city.timezone,
        "market_date_local": market.target_date.isoformat(),
        "created_at_utc": _none_or_str(raw.get("createdAt") or raw.get("creationDate")),
        "start_at_utc": _none_or_str(raw.get("startDate") or raw.get("startTime")),
        "close_time_utc": market.close_time.isoformat() if market.close_time else _none_or_str(raw.get("endDate")),
        "resolved_at_utc": _none_or_str(raw.get("resolvedAt") or raw.get("closedTime")),
        "relevant_cutoff_time_utc": market.close_time.isoformat() if market.close_time else _none_or_str(raw.get("endDate")),
        "bucket_label": market.bucket.label,
        "bucket_type": market.bucket.bucket_type,
        "operator": _operator_for_bucket(market.bucket.bucket_type),
        "threshold_low_f": market.bucket.lower_f,
        "threshold_high_f": market.bucket.upper_f,
        "units": market.bucket.resolution_unit or "F",
        "yes_token_id": market.bucket.yes_token_id,
        "no_token_id": market.bucket.no_token_id,
        "resolved_outcome": None,
        "settlement_source": "polymarket_gamma_public_market_definition_no_canonical_label",
        "telonex_source_path": str(source_path),
        "telonex_source_row": None,
        "telonex_quotes_from": None,
        "telonex_quotes_to": None,
        "parser_confidence": 0.95 if not warnings else 0.80,
        "parse_warnings_json": json.dumps(warnings, sort_keys=True),
        "source_payload_id": payload_id,
        "source_payload": {
            "payload_id": payload_id,
            "source_name": "polymarket_gamma_public_event",
            "cache_path": str(source_path),
            "payload_sha256": source_file_sha256,
            "request_json": json.dumps({"source": request_source}, sort_keys=True),
        },
        "parse_warnings": warnings,
    }


def pair_label_available_at_utc(pair: Mapping[str, Any], lag_hours: float = LABEL_AVAILABILITY_LAG_HOURS) -> datetime:
    try:
        zone = ZoneInfo(str(pair["timezone"]))
    except Exception:
        zone = timezone.utc
    local_date = date.fromisoformat(str(pair["local_date"]))
    local_end = datetime.combine(local_date + timedelta(days=1), datetime.min.time(), tzinfo=zone)
    return local_end.astimezone(timezone.utc) + timedelta(hours=float(lag_hours))


def _pair_has_training_weather(paths: PipelinePaths, pair: Mapping[str, Any]) -> bool:
    expected_decision_times = {
        decision_time.isoformat()
        for decision_time in _decision_times_for_pair(pair)
    }
    with closing(_connect(paths.weather_db)) as conn:
        forecast_rows = conn.execute(
            """
            SELECT source_name, decision_time_utc
            FROM forecast_snapshots
            WHERE normalized_city = ? AND local_date = ? AND temperature_2m_max_f IS NOT NULL
            """,
            (pair["normalized_city"], pair["local_date"]),
        ).fetchall()
        realized_count = int(
            conn.execute(
                """
                SELECT COUNT(*)
                FROM realized_daily_weather
                WHERE normalized_city = ? AND local_date = ? AND temperature_2m_max_f IS NOT NULL
                """,
                (pair["normalized_city"], pair["local_date"]),
            ).fetchone()[0]
        )
    forecast_times_by_source: dict[str, set[str]] = {}
    for source_name, decision_time_utc in forecast_rows:
        try:
            parsed = datetime.fromisoformat(str(decision_time_utc).replace("Z", "+00:00"))
        except ValueError:
            continue
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        canonical_time = parsed.astimezone(timezone.utc).isoformat()
        forecast_times_by_source.setdefault(str(source_name), set()).add(canonical_time)
    complete_forecast_set = bool(expected_decision_times) and any(
        expected_decision_times.issubset(observed_times)
        for observed_times in forecast_times_by_source.values()
    )
    return complete_forecast_set and realized_count > 0


def _classify_weather_pairs(
    paths: PipelinePaths,
    pairs: Iterable[Mapping[str, Any]],
) -> list[tuple[Mapping[str, Any], str]]:
    pair_list = list(pairs)
    if not pair_list:
        return []
    with closing(_connect(paths.weather_db)) as conn:
        statuses = {
            (str(row[0]), str(row[1])): str(row[2])
            for row in conn.execute(
                "SELECT normalized_city, local_date, status FROM weather_fetch_status"
            ).fetchall()
        }
    classified: list[tuple[Mapping[str, Any], str]] = []
    for pair in pair_list:
        key = (str(pair["normalized_city"]), str(pair["local_date"]))
        if _pair_has_training_weather(paths, pair):
            state = "complete"
        elif statuses.get(key) == "failed":
            state = "failed"
        else:
            state = "pending"
        classified.append((pair, state))
    return classified


def _training_summary(
    *,
    source_repo: Path = DEFAULT_SOURCE_REPO,
    paths: Any,
    as_of_utc: datetime,
    use_explicit_corpus_paths: bool = False,
) -> Mapping[str, Any]:
    runner_kwargs: dict[str, Any] = {
        "work_root": paths.root,
        "source_repo": source_repo,
        "end": resolve_training_corpus_end_date(None, now_utc=as_of_utc),
    }
    if use_explicit_corpus_paths:
        runner_kwargs.update(
            {
                "corpus_market_db": paths.market_db,
                "corpus_weather_db": paths.weather_db,
                "corpus_override_provenance": "copy_on_write_candidate",
            }
        )
    runner = ObservedHighGuardedShrinkageWeatherModelRunner(
        **runner_kwargs,
    )
    return runner.training_corpus_summary(as_of_utc=as_of_utc)


def _missing_data_product(
    training_summary: Mapping[str, Any],
    market_result: Mapping[str, Any],
    weather_result: Mapping[str, Any],
    coverage: Mapping[str, Any],
) -> Optional[dict[str, Any]]:
    if bool(coverage.get("operationally_fresh")):
        return None
    if not bool(coverage.get("coverage_available")):
        return {
            "product": "market/weather corpus coverage metadata",
            "reason": "Coverage could not be read; corpus freshness fails closed.",
            "fail_closed_reasons": list(coverage.get("fail_closed_reasons") or []),
        }
    if int(coverage.get("labelable_city_date_pairs") or 0) == 0:
        return {
            "product": "label-available Polymarket city/date definitions",
            "reason": "No label-available market city/date pairs were present in the requested operational window.",
        }
    if int(coverage.get("failed_city_date_pairs") or 0) > 0:
        return {
            "product": "Open-Meteo forecast/archive weather rows",
            "reason": "At least one label-available city/date has a failed weather update; freshness fails closed until it is complete.",
            "stale_or_incomplete_cities": list(coverage.get("stale_or_incomplete_cities") or []),
        }
    if int(coverage.get("pending_city_date_pairs") or 0) > 0:
        return {
            "product": "Open-Meteo forecast/archive weather rows",
            "reason": "At least one label-available city/date is missing a required forecast or realized source row; freshness fails closed.",
            "stale_or_incomplete_cities": list(coverage.get("stale_or_incomplete_cities") or []),
        }
    return {
        "product": "joined validation rows",
        "reason": "Per-city/source/date operational coverage is incomplete; inspect the coverage metadata instead of the global corpus maximum date.",
        "training_corpus_end_date_diagnostic_only": training_summary.get("training_corpus_end_date"),
        "market_rows_written_this_run": int(market_result.get("market_rows_written") or 0),
        "selected_weather_pairs_this_run": int(weather_result.get("selected_city_date_pairs") or 0),
    }


def _write_corpus_data_quality_report(
    paths: PipelinePaths,
    coverage: Mapping[str, Any],
) -> dict[str, str]:
    """Write corpus-only quality artifacts without opening replay stores."""

    paths.reports.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = paths.reports / f"{timestamp}-weather-corpus-data-quality.json"
    md_path = paths.reports / f"{timestamp}-weather-corpus-data-quality.md"
    json_path.write_text(json.dumps(dict(coverage), indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# Weather Corpus Data Quality",
        "",
        f"- Operationally fresh: `{coverage.get('operationally_fresh')}`",
        f"- Labelable city/date pairs: `{coverage.get('labelable_city_date_pairs')}`",
        f"- Complete city/date pairs: `{coverage.get('complete_city_date_pairs')}`",
        f"- Pending city/date pairs: `{coverage.get('pending_city_date_pairs')}`",
        f"- Failed city/date pairs: `{coverage.get('failed_city_date_pairs')}`",
        f"- Fresh cities: `{coverage.get('fresh_city_count')}`",
        f"- Stale/incomplete cities: `{coverage.get('stale_city_count')}`",
        "",
        "Global maximum dates are diagnostic only. Operational readiness is the conjunction of every required city/source/date check.",
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return {"json_path": str(json_path), "markdown_path": str(md_path)}


def _write_extension_report(paths: PipelinePaths, report: Mapping[str, Any], *, report_dir: Optional[Path]) -> dict[str, str]:
    target_dir = (report_dir or (paths.work / "reports" / "strategy_review"))
    if not target_dir.is_absolute():
        target_dir = paths.root / target_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    prefix = target_dir / f"{stamp}-wi5-weather-corpus-extension"
    return {"json": str(prefix.with_suffix(".json")), "markdown": str(prefix.with_suffix(".md"))}


def _render_extension_markdown(report: Mapping[str, Any]) -> str:
    summary = report.get("training_corpus_summary") or {}
    coverage = report.get("operational_corpus_coverage") or {}
    market = report.get("market_extension") or {}
    weather = report.get("weather_extension") or {}
    promotion = report.get("snapshot_promotion") or {}
    missing = report.get("missing_data_product")
    lines = [
        "# WI-5 Weather Corpus Extension",
        "",
        f"- Generated at UTC: `{report.get('generated_at_utc')}`",
        f"- As of UTC: `{report.get('as_of_utc')}`",
        f"- Training corpus end date: `{summary.get('training_corpus_end_date')}`",
        f"- Training row count: `{summary.get('training_row_count')}`",
        f"- Staleness days: `{summary.get('training_corpus_staleness_days')}`",
        f"- Staleness warning: `{summary.get('training_corpus_staleness_warning')}`",
        f"- Per-city/source/date corpus ready: `{coverage.get('operationally_fresh')}`",
        f"- Complete operational pairs: `{coverage.get('complete_city_date_pairs')}` / `{coverage.get('labelable_city_date_pairs')}`",
        f"- Stale/incomplete cities: `{coverage.get('stale_or_incomplete_cities')}`",
        f"- Market rows written: `{market.get('market_rows_written')}`",
        f"- Labelable city/date pairs: `{weather.get('labelable_city_date_pairs')}`",
        f"- Forecast rows written: `{weather.get('forecast_rows_written')}`",
        f"- Realized rows written: `{weather.get('realized_rows_written')}`",
        f"- Failed weather pairs: `{weather.get('failed_pairs')}`",
        f"- Snapshot promotion status: `{promotion.get('status')}`",
        f"- Active snapshot ID: `{promotion.get('snapshot_id') or promotion.get('previous_snapshot_id')}`",
        f"- Active pointer changed: `{promotion.get('pointer_changed')}`",
        "",
        "## Builder Contract",
        "",
    ]
    for key, value in (report.get("row_builder_contract") or {}).items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Missing Data Product", ""])
    if missing:
        lines.append(f"- Product: `{missing.get('product')}`")
        lines.append(f"- Reason: {missing.get('reason')}")
    else:
        lines.append("- None. Every required city/source/date pair in the operational window is complete.")
    lines.append("")
    return "\n".join(lines)


def _forecast_run_time_for_extension(decision_time: datetime) -> datetime:
    eligible = _normalize_utc(decision_time) - timedelta(hours=6)
    run_hour = (eligible.hour // 6) * 6
    return eligible.replace(hour=run_hour, minute=0, second=0, microsecond=0)


def _event_intersects_window(payload: Mapping[str, Any], *, start: date, end: date) -> bool:
    slug = str(payload.get("slug") or "")
    title = str(payload.get("title") or "")
    text = f"{slug} {title}"
    for current in _date_range(start, end):
        if current.strftime("%B").lower() in text.lower() and str(current.day) in text:
            return True
        if current.isoformat() in text:
            return True
        slug_date = f"{current.strftime('%B').lower()}-{current.day}-{current.year}"
        if slug_date in slug:
            return True
    return False


def _city_slug_candidates(city: CityConfig) -> tuple[str, ...]:
    names = [city.name, *(alias for alias in city.aliases if len(alias) > 3)]
    if city.name == "New York":
        names.append("New York City")
    if city.name == "Washington":
        names.extend(["Washington DC", "Washington D.C."])
    output = []
    for name in names:
        slug = "".join(ch.lower() if ch.isalnum() else "-" for ch in name)
        slug = "-".join(part for part in slug.split("-") if part)
        if slug:
            output.append(slug)
    return tuple(dict.fromkeys(output))


def _gamma_market_warnings(market: WeatherMarket) -> list[str]:
    warnings = []
    if not market.bucket.yes_token_id:
        warnings.append("missing_yes_token_id")
    if not market.bucket.no_token_id:
        warnings.append("missing_no_token_id")
    if market.bucket.lower_f is None and market.bucket.upper_f is None:
        warnings.append("missing_temperature_threshold")
    if not market.city.timezone:
        warnings.append("missing_timezone")
    return warnings


def _operator_for_bucket(bucket_type: str) -> str:
    if bucket_type in {"exact", "range"}:
        return "between"
    if bucket_type == "above":
        return ">="
    if bucket_type == "below":
        return "<="
    return "unknown"


def _date_range(start: date, end: date) -> Iterable[date]:
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def _normalize_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _none_or_str(value: Any) -> Optional[str]:
    if value in (None, ""):
        return None
    return str(value)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
