"""Offline pre-trade risk checks for forward-shadow research and replay."""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from daily_weather_pipeline.venue_execution import (
    VenueExecutionAssessment,
    VenueExecutionConstraints,
    assess_venue_order,
)


SCHEMA_VERSION = "dailyweather.live_trading_plumbing.v1"
SAFE_MODES = ("read_only", "shadow", "paper")
REAL_MODE = "real"
KILL_SWITCH_ACTIVE_VALUES = {"1", "true", "yes", "on", "halt", "killed", "disabled"}
AUTHORITATIVE_LIFECYCLE_AUTHORITY = "dailyweather.append_only_lifecycle_store.v1"
AUTHORITATIVE_LIFECYCLE_MAX_AGE_SECONDS = 5.0
AUTHORITATIVE_LIFECYCLE_MAX_FUTURE_SKEW_SECONDS = 1.0


@dataclass(frozen=True)
class LiveSafetyConfig:
    mode: str = "read_only"
    real_orders_disabled_by_default: bool = True
    require_real_orders_flag: bool = True
    require_real_orders_confirmation_flag: bool = True
    max_daily_realized_loss_usd: float = 5.0
    max_daily_deployed_notional_usd: float = 20.0
    max_open_exposure_usd: float = 25.0
    max_order_notional_usd: float = 2.0
    per_token_side_exposure_cap_usd: float = 5.0
    per_city_date_exposure_cap_usd: float = 5.0
    per_region_day_exposure_cap_usd: float = 10.0
    one_primary_exposure_per_city_date: bool = True
    kill_switch_path: Optional[str] = None
    kill_switch_env_var: str = "DAILYWEATHER_LIVE_KILL_SWITCH"
    stale_quote_max_age_seconds: int = 90
    max_spread: float = 0.10
    min_liquidity_depth_usd: float = 2.0
    require_depth_to_cover_order: bool = True
    enforce_venue_constraints: bool = False
    max_order_loss_usd: Optional[float] = None
    require_full_book_execution: bool = False
    ledger_path: Optional[str] = None
    require_existing_ledger: bool = False
    allowed_market_statuses: tuple[str, ...] = ("open", "active", "market_open_after_poll")
    blocked_resolution_statuses: tuple[str, ...] = (
        "closed",
        "resolved",
        "settled",
        "cancelled",
        "canceled",
        "halted",
    )

    def normalized_mode(self) -> str:
        return self.mode.strip().lower()

    def real_order_mode_enabled(self) -> bool:
        return self.normalized_mode() == REAL_MODE

    def as_report_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "mode": self.normalized_mode(),
            "read_only_default": self.normalized_mode() == "read_only",
            "real_orders_disabled_by_default": self.real_orders_disabled_by_default,
            "real_order_mode_enabled": self.real_order_mode_enabled(),
            "real_orders_require_flag": self.require_real_orders_flag,
            "real_orders_require_confirmation_flag": self.require_real_orders_confirmation_flag,
            "max_daily_realized_loss_usd": self.max_daily_realized_loss_usd,
            "max_daily_deployed_notional_usd": self.max_daily_deployed_notional_usd,
            "max_open_exposure_usd": self.max_open_exposure_usd,
            "max_order_notional_usd": self.max_order_notional_usd,
            "per_token_side_exposure_cap_usd": self.per_token_side_exposure_cap_usd,
            "per_city_date_exposure_cap_usd": self.per_city_date_exposure_cap_usd,
            "per_region_day_exposure_cap_usd": self.per_region_day_exposure_cap_usd,
            "one_primary_exposure_per_city_date": self.one_primary_exposure_per_city_date,
            "kill_switch_path": self.kill_switch_path,
            "kill_switch_env_var": self.kill_switch_env_var,
            "stale_quote_max_age_seconds": self.stale_quote_max_age_seconds,
            "max_spread": self.max_spread,
            "min_liquidity_depth_usd": self.min_liquidity_depth_usd,
            "require_depth_to_cover_order": self.require_depth_to_cover_order,
            "enforce_venue_constraints": self.enforce_venue_constraints,
            "max_order_loss_usd": self.max_order_loss_usd,
            "require_full_book_execution": self.require_full_book_execution,
            "ledger_path": self.ledger_path,
            "require_existing_ledger": self.require_existing_ledger,
            "allowed_market_statuses": list(self.allowed_market_statuses),
            "blocked_resolution_statuses": list(self.blocked_resolution_statuses),
        }

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "LiveSafetyConfig":
        fields = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        values = {key: value for key, value in raw.items() if key in fields}
        if "allowed_market_statuses" in values:
            values["allowed_market_statuses"] = tuple(values["allowed_market_statuses"])
        if "blocked_resolution_statuses" in values:
            values["blocked_resolution_statuses"] = tuple(values["blocked_resolution_statuses"])
        return cls(**values)


@dataclass(frozen=True)
class RuntimeFlags:
    real_orders_requested: bool = False
    confirm_real_orders: bool = False
    dry_run: bool = True

    def as_report_dict(self) -> dict[str, Any]:
        return {
            "real_orders_requested": self.real_orders_requested,
            "confirm_real_orders": self.confirm_real_orders,
            "dry_run": self.dry_run,
        }


@dataclass(frozen=True)
class ProposedOrder:
    decision_id: str
    strategy_name: str
    token_id: str
    side: str
    action: str
    notional_usd: float
    limit_price: Optional[float]
    city: str
    market_date_local: str
    region: str
    quote_timestamp_utc: Optional[datetime]
    bid: Optional[float]
    ask: Optional[float]
    spread: Optional[float]
    available_depth_usd: Optional[float]
    market_status: str = "open"
    resolution_status: str = "unresolved"
    primary_exposure: bool = True
    source: str = "manual"
    source_path: Optional[str] = None
    order_type: str = "FOK"
    intended_shares: Optional[float] = None
    venue_min_order_size_shares: Optional[float] = None
    venue_tick_size: Optional[float] = None
    venue_fee_rate: Optional[float] = None
    venue_fee_rate_source: Optional[str] = None
    available_depth_shares: Optional[float] = None
    ask_levels: tuple[Mapping[str, Any], ...] = ()
    full_book_complete: Optional[bool] = None
    book_provenance: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def token_side_key(self) -> str:
        return f"{self.token_id}|{self.side.upper()}"

    @property
    def city_date_key(self) -> str:
        return f"{self.city}|{self.market_date_local}"

    @property
    def region_day_key(self) -> str:
        return f"{self.region}|{self.market_date_local}"

    def as_report_dict(self) -> dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "strategy_name": self.strategy_name,
            "token_id": self.token_id,
            "side": self.side.upper(),
            "action": self.action.upper(),
            "notional_usd": self.notional_usd,
            "limit_price": self.limit_price,
            "city": self.city,
            "market_date_local": self.market_date_local,
            "region": self.region,
            "quote_timestamp_utc": _format_dt(self.quote_timestamp_utc),
            "bid": self.bid,
            "ask": self.ask,
            "spread": self.spread,
            "available_depth_usd": self.available_depth_usd,
            "market_status": self.market_status,
            "resolution_status": self.resolution_status,
            "primary_exposure": self.primary_exposure,
            "source": self.source,
            "source_path": self.source_path,
            "order_type": self.order_type,
            "intended_shares": self.intended_shares,
            "venue_min_order_size_shares": self.venue_min_order_size_shares,
            "venue_tick_size": self.venue_tick_size,
            "venue_fee_rate": self.venue_fee_rate,
            "venue_fee_rate_source": self.venue_fee_rate_source,
            "available_depth_shares": self.available_depth_shares,
            "ask_levels": [dict(item) for item in self.ask_levels],
            "full_book_complete": self.full_book_complete,
            "book_provenance": self.book_provenance,
            "metadata": _json_ready(self.metadata),
        }


@dataclass(frozen=True)
class OpenPosition:
    token_id: str
    side: str
    shares: float
    cost_usd: float
    city: str
    market_date_local: str
    region: str
    primary_exposure: bool = True
    last_mark_price: Optional[float] = None

    @property
    def token_side_key(self) -> str:
        return f"{self.token_id}|{self.side.upper()}"

    @property
    def city_date_key(self) -> str:
        return f"{self.city}|{self.market_date_local}"

    @property
    def region_day_key(self) -> str:
        return f"{self.region}|{self.market_date_local}"

    @property
    def average_entry_price(self) -> Optional[float]:
        if self.shares <= 0:
            return None
        return self.cost_usd / self.shares

    def as_report_dict(self) -> dict[str, Any]:
        return {
            "token_id": self.token_id,
            "side": self.side.upper(),
            "shares": self.shares,
            "cost_usd": self.cost_usd,
            "average_entry_price": self.average_entry_price,
            "city": self.city,
            "market_date_local": self.market_date_local,
            "region": self.region,
            "primary_exposure": self.primary_exposure,
            "last_mark_price": self.last_mark_price,
        }


@dataclass(frozen=True)
class PortfolioSnapshot:
    cash_balance_usd: float
    realized_pnl_today_usd: float = 0.0
    daily_deployed_notional_usd: float = 0.0
    open_positions: tuple[OpenPosition, ...] = ()
    unresolved_reconciliation_count: int = 0
    settlement_unresolved_count: int = 0
    ledger_cash_mismatch_usd: float = 0.0

    @property
    def open_exposure_usd(self) -> float:
        return sum(position.cost_usd for position in self.open_positions if position.shares > 0)

    def exposure_by_token_side(self) -> dict[str, float]:
        return _sum_positions(self.open_positions, lambda item: item.token_side_key)

    def exposure_by_city_date(self) -> dict[str, float]:
        return _sum_positions(self.open_positions, lambda item: item.city_date_key)

    def exposure_by_region_day(self) -> dict[str, float]:
        return _sum_positions(self.open_positions, lambda item: item.region_day_key)

    def primary_city_date_count(self, city_date_key: str) -> int:
        return sum(
            1
            for position in self.open_positions
            if position.primary_exposure and position.city_date_key == city_date_key and position.shares > 0
        )

    def as_report_dict(self) -> dict[str, Any]:
        return {
            "cash_balance_usd": self.cash_balance_usd,
            "realized_pnl_today_usd": self.realized_pnl_today_usd,
            "daily_deployed_notional_usd": self.daily_deployed_notional_usd,
            "open_exposure_usd": self.open_exposure_usd,
            "open_positions": [position.as_report_dict() for position in self.open_positions],
            "exposure_by_token_side": self.exposure_by_token_side(),
            "exposure_by_city_date": self.exposure_by_city_date(),
            "exposure_by_region_day": self.exposure_by_region_day(),
            "unresolved_reconciliation_count": self.unresolved_reconciliation_count,
            "settlement_unresolved_count": self.settlement_unresolved_count,
            "ledger_cash_mismatch_usd": self.ledger_cash_mismatch_usd,
        }


@dataclass(frozen=True)
class RiskDecision:
    allowed: bool
    block_reasons: tuple[str, ...]
    cap_dimensions_hit: tuple[str, ...]
    remaining_capacity: Mapping[str, float]
    audit_metadata: Mapping[str, Any]

    def as_report_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "block_reasons": list(self.block_reasons),
            "cap_dimensions_hit": list(self.cap_dimensions_hit),
            "remaining_capacity": dict(self.remaining_capacity),
            "audit_metadata": _json_ready(self.audit_metadata),
        }


@dataclass(frozen=True)
class GuardedExecutionResult:
    status: str
    pre_trade_gate_checked: bool
    risk_decision: RiskDecision
    dry_run: bool = True
    read_only: bool = True
    real_order_placement_allowed: bool = False
    real_orders_placed: Optional[int] = 0
    order_adapter_invoked: bool = False
    order_adapter_name: Optional[str] = None
    audit_metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def accepted_by_pre_trade_gate(self) -> bool:
        return self.risk_decision.allowed

    def as_report_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "status": self.status,
            "pre_trade_gate_checked": self.pre_trade_gate_checked,
            "accepted_by_pre_trade_gate": self.accepted_by_pre_trade_gate,
            "dry_run": self.dry_run,
            "read_only": self.read_only,
            "real_order_placement_allowed": self.real_order_placement_allowed,
            "real_orders_placed": self.real_orders_placed,
            "order_submission_code_path_invoked": self.order_adapter_invoked,
            "order_adapter_invoked": self.order_adapter_invoked,
            "order_adapter_name": self.order_adapter_name,
            "risk_decision": self.risk_decision.as_report_dict(),
            "audit_metadata": _json_ready(self.audit_metadata),
        }


class GuardedDryRunOrderExecutor:
    """Current executor boundary.

    This class is dry-run only. It may receive an adapter object for fault tests,
    but it never invokes that adapter and therefore cannot place real orders.
    """

    def __init__(
        self,
        *,
        config: LiveSafetyConfig = LiveSafetyConfig(),
        checker: Optional[Any] = None,
        order_adapter: Optional[Any] = None,
    ) -> None:
        self.config = config
        self.checker = checker or PreTradeRiskChecker(config)
        self.order_adapter = order_adapter

    def submit_order(
        self,
        order: ProposedOrder,
        portfolio: PortfolioSnapshot,
        *,
        runtime_flags: RuntimeFlags = RuntimeFlags(),
        now: Optional[datetime] = None,
        lifecycle_halt_status: Optional[Mapping[str, Any]] = None,
    ) -> GuardedExecutionResult:
        requested_flags = runtime_flags
        effective_flags = replace(runtime_flags, dry_run=True)
        checker_kwargs: dict[str, Any] = {
            "runtime_flags": effective_flags,
            "now": now,
        }
        if lifecycle_halt_status is not None:
            checker_kwargs["lifecycle_halt_status"] = lifecycle_halt_status
        decision = self.checker.check(order, portfolio, **checker_kwargs)
        status = (
            "blocked_by_pre_trade_gate"
            if not decision.allowed
            else "accepted_by_pre_trade_gate_dry_run_not_submitted"
        )
        return GuardedExecutionResult(
            status=status,
            pre_trade_gate_checked=True,
            risk_decision=decision,
            dry_run=True,
            read_only=True,
            real_order_placement_allowed=False,
            real_orders_placed=0,
            order_adapter_invoked=False,
            order_adapter_name=_adapter_name(self.order_adapter),
            audit_metadata={
                "requested_runtime_flags": requested_flags.as_report_dict(),
                "effective_runtime_flags": effective_flags.as_report_dict(),
                "dry_run_forced": not requested_flags.dry_run,
                "order_adapter_present": self.order_adapter is not None,
                "guard_policy": "pre_trade_gate_required_and_order_adapter_disabled_in_dry_run",
                "current_status": "dry_run_only_cannot_place_real_orders",
            },
        )


class PreTradeRiskChecker:
    def __init__(self, config: LiveSafetyConfig) -> None:
        self.config = config

    def check(
        self,
        order: ProposedOrder,
        portfolio: PortfolioSnapshot,
        *,
        runtime_flags: RuntimeFlags = RuntimeFlags(),
        now: Optional[datetime] = None,
        lifecycle_halt_status: Optional[Mapping[str, Any]] = None,
        require_authoritative_lifecycle_status: bool = False,
    ) -> RiskDecision:
        current_time = _ensure_utc(now or datetime.now(timezone.utc))
        reasons: list[str] = []
        cap_hits: list[str] = []
        mode = self.config.normalized_mode()
        action = str(order.action or "").strip().upper()
        matching_position = next(
            (
                position
                for position in portfolio.open_positions
                if position.token_side_key == order.token_side_key and position.shares > 0
            ),
            None,
        )
        sell_shares = _proposed_order_shares(order) if action == "SELL" else None
        sell_cost_reduction_usd = 0.0

        if require_authoritative_lifecycle_status:
            reasons.extend(
                _authoritative_lifecycle_status_reasons(lifecycle_halt_status)
            )
        elif lifecycle_halt_status is not None:
            if not isinstance(lifecycle_halt_status, Mapping):
                reasons.append("lifecycle_halt_status_invalid")
            else:
                halt_required = lifecycle_halt_status.get("halt_required")
                if not isinstance(halt_required, bool):
                    reasons.append("lifecycle_halt_status_invalid")
                elif halt_required:
                    reasons.append("lifecycle_state_halt")

        if action not in {"BUY", "SELL"}:
            reasons.append("unsupported_order_action")
        elif action == "SELL":
            if matching_position is None:
                reasons.append("sell_position_missing")
            elif sell_shares is None or sell_shares <= 0:
                reasons.append("invalid_sell_share_quantity")
            elif sell_shares > matching_position.shares + 1e-9:
                reasons.append("sell_exceeds_open_position")
            else:
                sell_cost_reduction_usd = (
                    matching_position.cost_usd * sell_shares / matching_position.shares
                )
                if (
                    matching_position.city_date_key != order.city_date_key
                    or matching_position.region_day_key != order.region_day_key
                ):
                    reasons.append("sell_position_metadata_mismatch")

        if mode not in SAFE_MODES and mode != REAL_MODE:
            reasons.append("unsupported_live_safety_mode")
        if runtime_flags.real_orders_requested and not self.config.real_order_mode_enabled():
            reasons.append("real_order_mode_not_enabled")
        if self.config.real_order_mode_enabled() or runtime_flags.real_orders_requested:
            if self.config.require_real_orders_flag and not runtime_flags.real_orders_requested:
                reasons.append("real_orders_flag_missing")
            if self.config.require_real_orders_confirmation_flag and not runtime_flags.confirm_real_orders:
                reasons.append("real_orders_confirmation_missing")
            if runtime_flags.dry_run:
                reasons.append("dry_run_cannot_place_real_orders")
            if not self.config.enforce_venue_constraints:
                reasons.append("venue_constraints_not_enforced_for_real_orders")
            if self.config.max_order_loss_usd is None:
                reasons.append("missing_max_order_loss_cap")
            if not self.config.require_full_book_execution:
                reasons.append("full_book_enforcement_not_required_for_real_orders")

        kill_switch = _kill_switch_state(self.config)
        if kill_switch["active"]:
            reasons.append("kill_switch_active")

        if order.notional_usd <= 0:
            reasons.append("invalid_order_notional")
        if order.notional_usd > self.config.max_order_notional_usd:
            reasons.append("max_order_notional_exceeded")
            cap_hits.append("max_order_notional_usd")

        venue_assessment: Optional[VenueExecutionAssessment] = None
        risk_amount_usd = max(0.0, order.notional_usd) if action == "BUY" else 0.0
        if self.config.enforce_venue_constraints:
            assessment_price = order.limit_price if order.limit_price is not None else order.ask
            intended_shares = order.intended_shares
            if (
                intended_shares is None
                and order.notional_usd > 0
                and assessment_price is not None
                and assessment_price > 0
            ):
                intended_shares = order.notional_usd / assessment_price
            venue_assessment = assess_venue_order(
                constraints=VenueExecutionConstraints(
                    min_order_size_shares=order.venue_min_order_size_shares,
                    tick_size=order.venue_tick_size,
                    fee_rate=order.venue_fee_rate,
                    fee_rate_source=order.venue_fee_rate_source,
                ),
                intended_shares=intended_shares,
                limit_price=assessment_price,
                available_depth_shares=order.available_depth_shares,
                intended_notional_usd=order.notional_usd,
                max_notional_usd=self.config.max_order_notional_usd,
                max_loss_usd=self.config.max_order_loss_usd,
                order_type=order.order_type,
                side=order.action,
                ask_levels=(order.ask_levels or None),
                full_book_complete=order.full_book_complete,
                require_full_book=self.config.require_full_book_execution,
                best_ask_price=order.ask,
                best_ask_size=order.available_depth_shares,
                top_level_provenance=order.book_provenance,
                quote_timestamp_valid=(
                    order.quote_timestamp_utc is not None
                    and _ensure_utc(order.quote_timestamp_utc) <= current_time
                ),
                require_current_market_fee=self.config.require_full_book_execution,
            )
            reasons.extend(venue_assessment.reasons)
            if self.config.max_order_loss_usd is None:
                reasons.append("missing_max_order_loss_cap")
            elif (
                venue_assessment.maximum_loss_usd is not None
                and venue_assessment.maximum_loss_usd > self.config.max_order_loss_usd + 1e-9
            ):
                cap_hits.append("max_order_loss_usd")
            if action == "BUY" and venue_assessment.maximum_loss_usd is not None:
                risk_amount_usd = venue_assessment.maximum_loss_usd

        if (
            action == "BUY"
            and portfolio.realized_pnl_today_usd <= -abs(self.config.max_daily_realized_loss_usd)
        ):
            reasons.append("max_daily_realized_loss_exceeded")
            cap_hits.append("max_daily_realized_loss_usd")

        exposure_delta_usd = (
            risk_amount_usd if action == "BUY" else -sell_cost_reduction_usd
        )
        open_exposure_after = max(0.0, portfolio.open_exposure_usd + exposure_delta_usd)
        if action == "BUY" and open_exposure_after > self.config.max_open_exposure_usd:
            reasons.append("max_open_exposure_exceeded")
            cap_hits.append("max_open_exposure_usd")

        daily_deployed_after = portfolio.daily_deployed_notional_usd + (
            risk_amount_usd if action == "BUY" else 0.0
        )
        if (
            action == "BUY"
            and daily_deployed_after > self.config.max_daily_deployed_notional_usd
        ):
            reasons.append("max_daily_deployed_notional_exceeded")
            cap_hits.append("max_daily_deployed_notional_usd")

        token_side_after = max(
            0.0,
            portfolio.exposure_by_token_side().get(order.token_side_key, 0.0)
            + exposure_delta_usd,
        )
        if action == "BUY" and token_side_after > self.config.per_token_side_exposure_cap_usd:
            reasons.append("per_token_side_exposure_cap_exceeded")
            cap_hits.append("per_token_side_exposure_cap_usd")

        city_date_after = max(
            0.0,
            portfolio.exposure_by_city_date().get(order.city_date_key, 0.0)
            + exposure_delta_usd,
        )
        if action == "BUY" and city_date_after > self.config.per_city_date_exposure_cap_usd:
            reasons.append("per_city_date_exposure_cap_exceeded")
            cap_hits.append("per_city_date_exposure_cap_usd")

        region_day_after = max(
            0.0,
            portfolio.exposure_by_region_day().get(order.region_day_key, 0.0)
            + exposure_delta_usd,
        )
        if action == "BUY" and region_day_after > self.config.per_region_day_exposure_cap_usd:
            reasons.append("per_region_day_exposure_cap_exceeded")
            cap_hits.append("per_region_day_exposure_cap_usd")

        if (
            self.config.one_primary_exposure_per_city_date
            and action == "BUY"
            and order.primary_exposure
            and portfolio.primary_city_date_count(order.city_date_key) > 0
        ):
            reasons.append("primary_city_date_exposure_exists")
            cap_hits.append("one_primary_exposure_per_city_date")

        quote_age_seconds = None
        if order.quote_timestamp_utc is None:
            reasons.append("stale_quote")
        else:
            quote_age_seconds = max(0.0, (current_time - _ensure_utc(order.quote_timestamp_utc)).total_seconds())
            if quote_age_seconds > self.config.stale_quote_max_age_seconds:
                reasons.append("stale_quote")

        if order.spread is None:
            reasons.append("missing_spread")
        elif order.spread > self.config.max_spread:
            reasons.append("spread_too_wide")

        required_depth = self.config.min_liquidity_depth_usd
        if self.config.require_depth_to_cover_order:
            required_depth = max(required_depth, order.notional_usd)
        depth_available_usd = order.available_depth_usd
        if (
            venue_assessment is not None
            and venue_assessment.execution_basis
            in {
                "top_level_proves_full_fill",
                "full_order_book_fok_buy_usdc_walk",
            }
        ):
            depth_available_usd = venue_assessment.cumulative_cost_usd
        if depth_available_usd is None or depth_available_usd < required_depth:
            reasons.append("insufficient_liquidity_depth")

        market_status = (order.market_status or "").strip().lower()
        if market_status not in {item.lower() for item in self.config.allowed_market_statuses}:
            reasons.append("market_not_open")
        resolution_status = (order.resolution_status or "").strip().lower()
        if resolution_status in {item.lower() for item in self.config.blocked_resolution_statuses}:
            reasons.append("market_resolution_status_blocked")

        if portfolio.unresolved_reconciliation_count > 0:
            reasons.append("unresolved_reconciliation_halt")
        if portfolio.settlement_unresolved_count > 0:
            reasons.append("settlement_unresolved_halt")
        if abs(portfolio.ledger_cash_mismatch_usd) > 1e-6:
            reasons.append("ledger_cash_mismatch_halt")

        remaining_capacity = {
            "max_order_notional_usd": self.config.max_order_notional_usd - order.notional_usd,
            "max_daily_realized_loss_usd": self.config.max_daily_realized_loss_usd + portfolio.realized_pnl_today_usd,
            "max_daily_deployed_notional_usd": self.config.max_daily_deployed_notional_usd - daily_deployed_after,
            "max_open_exposure_usd": self.config.max_open_exposure_usd - open_exposure_after,
            "per_token_side_exposure_cap_usd": self.config.per_token_side_exposure_cap_usd - token_side_after,
            "per_city_date_exposure_cap_usd": self.config.per_city_date_exposure_cap_usd - city_date_after,
            "per_region_day_exposure_cap_usd": self.config.per_region_day_exposure_cap_usd - region_day_after,
            "liquidity_depth_usd": (depth_available_usd or 0.0) - required_depth,
        }
        if self.config.max_order_loss_usd is not None:
            remaining_capacity["max_order_loss_usd"] = (
                self.config.max_order_loss_usd - risk_amount_usd
            )
        remaining_capacity = {key: round(float(value), 8) for key, value in remaining_capacity.items()}

        audit = {
            "checked_at_utc": current_time.isoformat(),
            "config_mode": mode,
            "runtime_flags": runtime_flags.as_report_dict(),
            "kill_switch": kill_switch,
            "quote_age_seconds": quote_age_seconds,
            "required_liquidity_depth_usd": required_depth,
            "depth_available_usd": depth_available_usd,
            "venue_execution_assessment": (
                venue_assessment.as_report_dict() if venue_assessment else None
            ),
            "risk_amount_usd": risk_amount_usd,
            "exposure_delta_usd": exposure_delta_usd,
            "sell_shares": sell_shares,
            "sell_cost_reduction_usd": sell_cost_reduction_usd,
            "lifecycle_halt_status": _json_ready(lifecycle_halt_status),
            "authoritative_lifecycle_status_required": (
                require_authoritative_lifecycle_status
            ),
            "order": order.as_report_dict(),
            "portfolio_summary": portfolio.as_report_dict(),
        }
        unique_reasons = tuple(dict.fromkeys(reasons))
        return RiskDecision(
            allowed=not unique_reasons,
            block_reasons=unique_reasons,
            cap_dimensions_hit=tuple(dict.fromkeys(cap_hits)),
            remaining_capacity=remaining_capacity,
            audit_metadata=audit,
        )


def proposed_order_from_shadow_event(
    event: Mapping[str, Any],
    *,
    source_path: Optional[Path] = None,
    line_number: Optional[int] = None,
) -> Optional[ProposedOrder]:
    decision = event.get("decision") if isinstance(event.get("decision"), Mapping) else {}
    if not decision or not decision.get("would_trade"):
        return None
    signal = decision.get("would_trade_signal") if isinstance(decision.get("would_trade_signal"), Mapping) else {}
    discovery = event.get("market_discovery") if isinstance(event.get("market_discovery"), Mapping) else {}
    parsed = event.get("parsed_contract") if isinstance(event.get("parsed_contract"), Mapping) else {}
    quote = event.get("quote") if isinstance(event.get("quote"), Mapping) else {}
    settlement = event.get("settlement") if isinstance(event.get("settlement"), Mapping) else {}
    top_depth = quote.get("top_of_book_depth") if isinstance(quote.get("top_of_book_depth"), Mapping) else {}
    live_market = (
        event.get("live_market_metadata")
        if isinstance(event.get("live_market_metadata"), Mapping)
        else {}
    )
    side = str(signal.get("side") or discovery.get("side") or parsed.get("side") or "NO").upper()
    observed_books = (
        event.get("observed_order_books")
        if isinstance(event.get("observed_order_books"), Mapping)
        else {}
    )
    side_book = observed_books.get(side) if isinstance(observed_books.get(side), Mapping) else {}
    token_id = str(signal.get("token_id") or discovery.get("token_id") or parsed.get("no_token_id") or "")
    ask = _optional_float(quote.get("ask"))
    available_depth = _depth_usd(top_depth=top_depth, side=side, ask=ask, bid=_optional_float(quote.get("bid")))
    city = str(parsed.get("city") or discovery.get("normalized_city") or "unknown_city")
    market_date = str(parsed.get("date") or discovery.get("market_date_local") or "unknown_date")
    metadata = {
        "source_line_number": line_number,
        "decision_reason": decision.get("decision_reason"),
        "bucket_label": parsed.get("bucket_label") or discovery.get("bucket_label"),
        "question": (event.get("live_market_metadata") or {}).get("question")
        if isinstance(event.get("live_market_metadata"), Mapping)
        else None,
    }
    fallback_source = str(source_path) if source_path is not None else "in_memory_shadow_event"
    fallback_line = line_number if line_number is not None else "unknown"
    return ProposedOrder(
        decision_id=str(signal.get("decision_id") or discovery.get("decision_id") or f"{fallback_source}:{fallback_line}"),
        strategy_name=str(event.get("profile_key") or "old_high_confidence_no_tail"),
        token_id=token_id,
        side=side,
        action="BUY",
        notional_usd=float(signal.get("requested_notional_usd") or 0.0),
        limit_price=_optional_float(signal.get("limit_price")),
        city=city,
        market_date_local=market_date,
        region=_region_from_city(city),
        quote_timestamp_utc=_parse_optional_dt(quote.get("timestamp_utc")),
        bid=_optional_float(quote.get("bid")),
        ask=ask,
        spread=_optional_float(quote.get("spread")),
        available_depth_usd=available_depth,
        market_status=str((event.get("close_time_status") or "open")),
        resolution_status=str(settlement.get("status") or "unresolved"),
        primary_exposure=True,
        source="live_forward_shadow_recorder",
        source_path=str(source_path) if source_path is not None else None,
        order_type="FOK",
        intended_shares=_optional_float(
            signal.get("requested_shares") or signal.get("shares")
        ),
        venue_min_order_size_shares=_first_optional_float(
            (live_market, side_book),
            ("min_order_size_shares", "min_order_size", "minimum_order_size"),
        ),
        venue_tick_size=_first_optional_float(
            (live_market, side_book),
            ("tick_size", "minimum_tick_size"),
        ),
        venue_fee_rate=_first_optional_float(
            (live_market, signal.get("metadata") or {}),
            ("weather_taker_fee_rate", "fee_rate"),
        ),
        venue_fee_rate_source=str(
            live_market.get("fee_rate_source")
            or side_book.get("fee_rate_source")
            or (signal.get("metadata") or {}).get("fee_rate_source")
            or ""
        )
        or None,
        available_depth_shares=_optional_float(top_depth.get("ask_size")),
        ask_levels=_mapping_levels(side_book.get("ask_levels")),
        full_book_complete=_optional_bool(side_book.get("full_book_complete")),
        book_provenance=str(
            side_book.get("payload_id")
            or side_book.get("raw_cache_path")
            or ""
        )
        or None,
        metadata=metadata,
    )


def _depth_usd(*, top_depth: Mapping[str, Any], side: str, ask: Optional[float], bid: Optional[float]) -> Optional[float]:
    price = ask if side.upper() == "NO" else ask
    ask_size = _optional_float(top_depth.get("ask_size"))
    liquidity = _optional_float(top_depth.get("liquidity_depth"))
    if price is not None and ask_size is not None:
        return price * ask_size
    if price is not None and liquidity is not None:
        return price * liquidity
    if bid is not None and liquidity is not None:
        return bid * liquidity
    return None


def _adapter_name(adapter: Optional[Any]) -> Optional[str]:
    if adapter is None:
        return None
    return f"{adapter.__class__.__module__}.{adapter.__class__.__name__}"


def _proposed_order_shares(order: ProposedOrder) -> Optional[float]:
    if order.intended_shares is not None:
        return float(order.intended_shares)
    if order.limit_price is None or order.limit_price <= 0:
        return None
    return float(order.notional_usd / order.limit_price)


def _sum_positions(items: Iterable[OpenPosition], key_fn: Any) -> dict[str, float]:
    totals: dict[str, float] = {}
    for item in items:
        if item.shares <= 0:
            continue
        key = str(key_fn(item))
        totals[key] = totals.get(key, 0.0) + float(item.cost_usd)
    return totals


def _kill_switch_state(config: LiveSafetyConfig) -> dict[str, Any]:
    env_value = os.environ.get(config.kill_switch_env_var, "")
    env_active = env_value.strip().lower() in KILL_SWITCH_ACTIVE_VALUES
    path_active = False
    path = config.kill_switch_path
    if path:
        path_active = Path(path).exists()
    return {
        "active": bool(env_active or path_active),
        "env_var": config.kill_switch_env_var,
        "env_active": env_active,
        "path": path,
        "path_active": path_active,
    }


def _region_from_city(city: str) -> str:
    upper = city.upper()
    if any(country in upper for country in (", CN", ", JP", ", KR", ", SG", ", TW", ", HK", ", TH", ", IN")):
        return "asia"
    if any(country in upper for country in (", US", ", CA", ", MX", ", BR", ", AR")):
        return "americas"
    if any(country in upper for country in (", GB", ", ES", ", FR", ", DE", ", IT", ", NL", ", SE")):
        return "europe"
    return "unknown"


def _optional_float(value: Any) -> Optional[float]:
    if value is None or value == "":
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


def _mapping_levels(value: Any) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(dict(item) for item in value if isinstance(item, Mapping))


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


def _parse_optional_dt(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return _ensure_utc(value)
    text = str(value)
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return _ensure_utc(datetime.fromisoformat(text))
    except ValueError:
        return None


def _authoritative_lifecycle_status_reasons(
    status: Optional[Mapping[str, Any]],
) -> list[str]:
    if status is None:
        return ["authoritative_lifecycle_status_required"]
    if not isinstance(status, Mapping):
        return ["authoritative_lifecycle_status_invalid"]
    invalid = False
    if status.get("authority") != AUTHORITATIVE_LIFECYCLE_AUTHORITY:
        invalid = True
    store_uuid = status.get("store_uuid")
    try:
        if not isinstance(store_uuid, str) or str(uuid.UUID(store_uuid)) != store_uuid:
            invalid = True
    except (ValueError, AttributeError):
        invalid = True
    genesis_sha256 = status.get("genesis_sha256")
    if (
        not isinstance(genesis_sha256, str)
        or len(genesis_sha256) != 64
        or any(character not in "0123456789abcdef" for character in genesis_sha256)
    ):
        invalid = True
    if status.get("store_initialized") is not True:
        invalid = True
    store_path = status.get("store_path")
    if not isinstance(store_path, str) or not store_path or not Path(store_path).is_absolute():
        invalid = True
    snapshot_sha256 = status.get("snapshot_sha256")
    if (
        not isinstance(snapshot_sha256, str)
        or len(snapshot_sha256) != 64
        or any(character not in "0123456789abcdef" for character in snapshot_sha256)
    ):
        invalid = True
    max_sequence = status.get("max_event_sequence")
    if isinstance(max_sequence, bool) or not isinstance(max_sequence, int) or max_sequence < 0:
        invalid = True
    intent_count = status.get("intent_count")
    if isinstance(intent_count, bool) or not isinstance(intent_count, int) or intent_count < 0:
        invalid = True
    observed_at = _parse_optional_dt(status.get("observed_at_utc"))
    halt_required = status.get("halt_required")
    if observed_at is None or not isinstance(halt_required, bool):
        invalid = True
    if invalid:
        return ["authoritative_lifecycle_status_invalid"]

    age_seconds = (datetime.now(timezone.utc) - observed_at).total_seconds()
    reasons: list[str] = []
    if (
        age_seconds > AUTHORITATIVE_LIFECYCLE_MAX_AGE_SECONDS
        or age_seconds < -AUTHORITATIVE_LIFECYCLE_MAX_FUTURE_SKEW_SECONDS
    ):
        reasons.append("authoritative_lifecycle_status_stale")
    if halt_required:
        reasons.append("lifecycle_state_halt")
    return reasons


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _format_dt(value: Optional[datetime]) -> Optional[str]:
    return None if value is None else _ensure_utc(value).isoformat()


def _json_ready(value: Any) -> Any:
    if isinstance(value, datetime):
        return _format_dt(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_ready(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value
