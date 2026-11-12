from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Mapping, Optional, Sequence


VENUE_EXECUTION_POLICY_VERSION = "dailyweather.venue_execution.v3"
VENUE_FEE_ROUNDING_DECIMALS = 5
EXPLICIT_VENUE_TICK_BASIS = "explicit_venue_tick_size"
HISTORICAL_REPLAY_EXECUTION_CONTEXT = "historical_replay"
HISTORICAL_RESTING_ASK_TICK_BASIS = (
    "observed_resting_ask_is_tick_valid_by_venue_acceptance"
)
OFFICIAL_WEATHER_FEE_SCHEDULE_SOURCE = (
    "official_documented_weather_category_fee_schedule_v1_rate_0_05"
)
OFFICIAL_WEATHER_FEE_SCHEDULE_RATE = 0.05


@dataclass(frozen=True)
class VenueExecutionConstraints:
    """Venue-owned fields captured with a decision/book snapshot.

    ``min_order_size_shares`` preserves the current recorder field name, but
    its enforcement semantics are intentionally order-type aware. A real V2
    FOK BUY has matched below this value, while the V2 API defines BUY market
    ``amount`` in USDC rather than shares.
    """

    min_order_size_shares: Optional[float]
    tick_size: Optional[float]
    fee_rate: Optional[float]
    fee_rate_source: Optional[str] = None

    @property
    def metadata_complete(self) -> bool:
        return (
            self.min_order_size_shares is not None
            and self.tick_size is not None
            and self.fee_rate is not None
        )


@dataclass(frozen=True)
class VenueExecutionAssessment:
    eligible: bool
    intended_shares: Optional[float]
    limit_price: Optional[float]
    gross_notional_usd: Optional[float]
    fee_usd: Optional[float]
    maximum_loss_usd: Optional[float]
    available_depth_shares: Optional[float]
    order_type: str
    side: str
    minimum_size_rule_status: str
    reasons: tuple[str, ...]
    full_book_enforcement_requested: bool = False
    full_book_complete: Optional[bool] = None
    full_book_level_count: int = 0
    levels_consumed: int = 0
    cumulative_shares: Optional[float] = None
    cumulative_cost_usd: Optional[float] = None
    weighted_average_price: Optional[float] = None
    worst_fill_price: Optional[float] = None
    unfilled_notional_usd: Optional[float] = None
    execution_basis: str = "top_of_book_screen_only"
    all_or_none_fill: bool = False
    book_walk_legs: tuple[Mapping[str, float], ...] = ()
    maximum_loss_semantics: str = (
        "gross_requested_amount_plus_modeled_fees_upper_bound_not_exact_wallet_debit"
    )
    fee_rate: Optional[float] = None
    fee_rate_source: Optional[str] = None
    top_level_provenance: Optional[str] = None
    quote_timestamp_valid: Optional[bool] = None
    tick_validation_basis: Optional[str] = None
    execution_context: str = "live_or_unspecified"
    historical_tick_policy_requested: bool = False
    historical_tick_policy_eligible: bool = False
    direct_polymarket_resting_ask_provenance: Optional[bool] = None
    raw_compact_decision_passthrough_verified: Optional[bool] = None
    quote_non_crossed: Optional[bool] = None
    price_transformation_applied: Optional[bool] = None
    current_market_fee_required: bool = False
    policy_version: str = VENUE_EXECUTION_POLICY_VERSION

    def as_report_dict(self) -> dict[str, object]:
        return {
            "policy_version": self.policy_version,
            "eligible": self.eligible,
            "intended_shares": self.intended_shares,
            "limit_price": self.limit_price,
            "gross_notional_usd": self.gross_notional_usd,
            "fee_usd": self.fee_usd,
            "maximum_loss_usd": self.maximum_loss_usd,
            "available_depth_shares": self.available_depth_shares,
            "order_type": self.order_type,
            "side": self.side,
            "minimum_size_rule_status": self.minimum_size_rule_status,
            "full_book_enforcement_requested": self.full_book_enforcement_requested,
            "full_book_complete": self.full_book_complete,
            "full_book_level_count": self.full_book_level_count,
            "levels_consumed": self.levels_consumed,
            "cumulative_shares": self.cumulative_shares,
            "cumulative_cost_usd": self.cumulative_cost_usd,
            "weighted_average_price": self.weighted_average_price,
            "worst_fill_price": self.worst_fill_price,
            "unfilled_notional_usd": self.unfilled_notional_usd,
            "execution_basis": self.execution_basis,
            "all_or_none_fill": self.all_or_none_fill,
            "book_walk_legs": [dict(item) for item in self.book_walk_legs],
            "maximum_loss_semantics": self.maximum_loss_semantics,
            "fee_rate": self.fee_rate,
            "fee_rate_source": self.fee_rate_source,
            "top_level_provenance": self.top_level_provenance,
            "quote_timestamp_valid": self.quote_timestamp_valid,
            "tick_validation_basis": self.tick_validation_basis,
            "execution_context": self.execution_context,
            "historical_tick_policy_requested": self.historical_tick_policy_requested,
            "historical_tick_policy_eligible": self.historical_tick_policy_eligible,
            "direct_polymarket_resting_ask_provenance": (
                self.direct_polymarket_resting_ask_provenance
            ),
            "raw_compact_decision_passthrough_verified": (
                self.raw_compact_decision_passthrough_verified
            ),
            "quote_non_crossed": self.quote_non_crossed,
            "price_transformation_applied": self.price_transformation_applied,
            "current_market_fee_required": self.current_market_fee_required,
            "reasons": list(self.reasons),
        }


def _historical_resting_ask_tick_reasons(
    *,
    execution_context: str,
    require_full_book: bool,
    immediate_market_buy: bool,
    limit_price: Optional[float],
    best_ask_price: Optional[float],
    top_level_provenance: Optional[str],
    quote_timestamp_valid: Optional[bool],
    direct_polymarket_resting_ask_provenance: Optional[bool],
    raw_compact_decision_passthrough_verified: Optional[bool],
    quote_non_crossed: Optional[bool],
    price_transformation_applied: Optional[bool],
) -> tuple[str, ...]:
    """Return blockers for the narrowly audited historical tick exception.

    A resting ask accepted by Polymarket is evidence that the *captured ask*
    was venue-tick-valid at that historical instant. It is not evidence of the
    current tick, any transformed price, or any deeper book level. Callers
    therefore have to prove every part of the audited raw-to-decision chain.
    """

    reasons: list[str] = []
    if execution_context != HISTORICAL_REPLAY_EXECUTION_CONTEXT:
        reasons.append("historical_tick_policy_not_replay_context")
    if not require_full_book:
        reasons.append("historical_tick_policy_requires_top_level_fok_proof")
    if not immediate_market_buy:
        reasons.append("historical_tick_policy_requires_fok_buy")
    if not _prices_exactly_equal(limit_price, best_ask_price):
        reasons.append(
            "historical_tick_policy_requires_limit_equal_observed_best_ask"
        )
    if not str(top_level_provenance or "").strip():
        reasons.append("historical_tick_policy_missing_top_level_provenance")
    if direct_polymarket_resting_ask_provenance is not True:
        reasons.append(
            "historical_tick_policy_requires_direct_polymarket_resting_ask_provenance"
        )
    if raw_compact_decision_passthrough_verified is not True:
        reasons.append(
            "historical_tick_policy_requires_raw_compact_decision_passthrough"
        )
    if quote_timestamp_valid is not True:
        reasons.append("historical_tick_policy_requires_causal_quote")
    if quote_non_crossed is not True:
        reasons.append("historical_tick_policy_requires_non_crossed_quote")
    if price_transformation_applied is not False:
        reasons.append("historical_tick_policy_requires_no_price_transformation")
    return tuple(reasons)


def _prices_exactly_equal(
    left: Optional[float],
    right: Optional[float],
) -> bool:
    if left is None or right is None:
        return False
    try:
        return Decimal(str(left)) == Decimal(str(right))
    except Exception:
        return False


def assess_venue_order(
    *,
    constraints: Optional[VenueExecutionConstraints],
    intended_shares: Optional[float],
    limit_price: Optional[float],
    available_depth_shares: Optional[float],
    intended_notional_usd: Optional[float] = None,
    max_notional_usd: Optional[float] = None,
    max_loss_usd: Optional[float] = None,
    order_type: str = "FOK",
    side: str = "BUY",
    ask_levels: Optional[Sequence[Mapping[str, Any]]] = None,
    full_book_complete: Optional[bool] = None,
    require_full_book: bool = False,
    best_ask_price: Optional[float] = None,
    best_ask_size: Optional[float] = None,
    top_level_provenance: Optional[str] = None,
    quote_timestamp_valid: Optional[bool] = None,
    require_current_market_fee: bool = False,
    execution_context: str = "live_or_unspecified",
    allow_historical_resting_ask_tick_policy: bool = False,
    direct_polymarket_resting_ask_provenance: Optional[bool] = None,
    raw_compact_decision_passthrough_verified: Optional[bool] = None,
    quote_non_crossed: Optional[bool] = None,
    price_transformation_applied: Optional[bool] = None,
) -> VenueExecutionAssessment:
    """Evaluate an order without resizing it.

    This function never resizes. V2 FOK/FAK BUYs are dollar-amount orders, and
    a recorded matched V2 BUY proves that treating the book's
    ``min_order_size`` as a five-share taker floor would be false. The field is
    therefore recorded but not applied to FOK BUYs. FAK remains blocked until
    multi-level partial-fill simulation exists; other order/side combinations
    fail closed until their minimum-size semantics are proven.
    """

    reasons: list[str] = []
    min_order_size = constraints.min_order_size_shares if constraints else None
    tick_size = constraints.tick_size if constraints else None
    fee_rate = constraints.fee_rate if constraints else None
    fee_rate_source = constraints.fee_rate_source if constraints else None
    order_type_upper = str(order_type or "").upper()
    side_upper = str(side or "").upper()
    immediate_market_buy = order_type_upper == "FOK" and side_upper == "BUY"
    normalized_execution_context = str(execution_context or "").strip().lower()
    historical_tick_policy_eligible = False
    tick_validation_basis = (
        EXPLICIT_VENUE_TICK_BASIS
        if tick_size is not None and tick_size > 0
        else None
    )
    minimum_size_rule_status = (
        "not_applied_v2_market_buy_amount_is_usdc_and_subminimum_match_observed"
        if immediate_market_buy
        else (
            "fak_partial_execution_unimplemented"
            if order_type_upper == "FAK"
            else "unverified_for_order_type_and_side"
        )
    )

    if constraints is None:
        reasons.append("missing_venue_metadata")
    if min_order_size is None:
        reasons.append("missing_venue_min_order_size")
    elif min_order_size <= 0:
        reasons.append("invalid_venue_min_order_size")
    if tick_size is None:
        if allow_historical_resting_ask_tick_policy:
            historical_tick_reasons = _historical_resting_ask_tick_reasons(
                execution_context=normalized_execution_context,
                require_full_book=require_full_book,
                immediate_market_buy=immediate_market_buy,
                limit_price=limit_price,
                best_ask_price=best_ask_price,
                top_level_provenance=top_level_provenance,
                quote_timestamp_valid=quote_timestamp_valid,
                direct_polymarket_resting_ask_provenance=(
                    direct_polymarket_resting_ask_provenance
                ),
                raw_compact_decision_passthrough_verified=(
                    raw_compact_decision_passthrough_verified
                ),
                quote_non_crossed=quote_non_crossed,
                price_transformation_applied=price_transformation_applied,
            )
            if historical_tick_reasons:
                reasons.append("missing_venue_tick_size")
                reasons.extend(historical_tick_reasons)
            else:
                historical_tick_policy_eligible = True
                tick_validation_basis = HISTORICAL_RESTING_ASK_TICK_BASIS
        else:
            reasons.append("missing_venue_tick_size")
    elif tick_size <= 0:
        reasons.append("invalid_venue_tick_size")
    if fee_rate is None:
        reasons.append("missing_venue_fee_rate")
    elif fee_rate < 0:
        reasons.append("invalid_venue_fee_rate")
    if require_full_book:
        normalized_fee_source = str(fee_rate_source or "").strip().lower()
        documented_replay_schedule = (
            normalized_fee_source == OFFICIAL_WEATHER_FEE_SCHEDULE_SOURCE
        )
        if not normalized_fee_source:
            reasons.append("missing_fee_rate_source")
        elif documented_replay_schedule:
            if (
                fee_rate is not None
                and abs(fee_rate - OFFICIAL_WEATHER_FEE_SCHEDULE_RATE) > 1e-12
            ):
                reasons.append("official_weather_fee_schedule_rate_mismatch")
        elif "default" in normalized_fee_source or "category" in normalized_fee_source:
            reasons.append("non_current_market_fee_source")
        if (
            require_current_market_fee
            and "current_market_info" not in normalized_fee_source
        ):
            reasons.append("fee_source_not_current_market_info")
        if not str(top_level_provenance or "").strip():
            reasons.append("missing_top_level_provenance")
        if quote_timestamp_valid is not True:
            reasons.append("invalid_execution_quote_timestamp")
        if best_ask_price is None or best_ask_size is None:
            reasons.append("missing_top_level_fill_proof_fields")
        elif best_ask_size <= 0:
            reasons.append("invalid_top_level_ask_size")
    if not require_full_book:
        if available_depth_shares is None:
            reasons.append("missing_venue_depth")
        elif available_depth_shares < 0:
            reasons.append("invalid_venue_depth")

    if require_full_book:
        if intended_notional_usd is None or intended_notional_usd <= 0:
            reasons.append("invalid_intended_notional_usd")
    elif intended_shares is None or intended_shares <= 0:
        reasons.append("invalid_intended_shares")
    if order_type_upper == "FAK":
        reasons.append("fak_partial_execution_unimplemented")
    elif not immediate_market_buy:
        reasons.append("unverified_venue_minimum_size_semantics")

    if limit_price is None or not 0 < limit_price < 1:
        reasons.append("invalid_limit_price")
    elif tick_size is not None and tick_size > 0 and not price_is_on_tick(limit_price, tick_size):
        reasons.append("order_price_off_tick")

    if require_full_book:
        return _assess_required_fok_buy(
            constraints=constraints,
            intended_notional_usd=intended_notional_usd,
            limit_price=limit_price,
            ask_levels=ask_levels,
            full_book_complete=full_book_complete,
            max_notional_usd=max_notional_usd,
            max_loss_usd=max_loss_usd,
            order_type=order_type_upper,
            side=side_upper,
            minimum_size_rule_status=minimum_size_rule_status,
            base_reasons=reasons,
            best_ask_price=best_ask_price,
            best_ask_size=best_ask_size,
            top_level_provenance=top_level_provenance,
            quote_timestamp_valid=quote_timestamp_valid,
            require_current_market_fee=require_current_market_fee,
            tick_validation_basis=tick_validation_basis,
            execution_context=normalized_execution_context,
            historical_tick_policy_requested=(
                allow_historical_resting_ask_tick_policy
            ),
            historical_tick_policy_eligible=historical_tick_policy_eligible,
            direct_polymarket_resting_ask_provenance=(
                direct_polymarket_resting_ask_provenance
            ),
            raw_compact_decision_passthrough_verified=(
                raw_compact_decision_passthrough_verified
            ),
            quote_non_crossed=quote_non_crossed,
            price_transformation_applied=price_transformation_applied,
        )

    if (
        intended_shares is not None
        and intended_shares > 0
        and available_depth_shares is not None
        and available_depth_shares >= 0
        and available_depth_shares + 1e-9 < intended_shares
    ):
        reasons.append("insufficient_venue_depth")

    gross_notional: Optional[float] = None
    fee_usd: Optional[float] = None
    maximum_loss: Optional[float] = None
    if (
        intended_shares is not None
        and intended_shares > 0
        and limit_price is not None
        and 0 < limit_price < 1
    ):
        gross_notional = float(Decimal(str(intended_shares)) * Decimal(str(limit_price)))
        if intended_notional_usd is not None and abs(intended_notional_usd - gross_notional) > 1e-8:
            reasons.append("venue_order_notional_mismatch")
        if max_notional_usd is not None and gross_notional > max_notional_usd + 1e-9:
            reasons.append("venue_notional_cap_exceeded")
        if fee_rate is not None and fee_rate >= 0:
            fee_usd = weather_taker_fee_usd(
                shares=intended_shares,
                price=limit_price,
                fee_rate=fee_rate,
            )
            maximum_loss = gross_notional + fee_usd
            if max_loss_usd is not None and maximum_loss > max_loss_usd + 1e-9:
                reasons.append("venue_maximum_loss_cap_exceeded")

    unique_reasons = tuple(dict.fromkeys(reasons))
    return VenueExecutionAssessment(
        eligible=not unique_reasons,
        intended_shares=intended_shares,
        limit_price=limit_price,
        gross_notional_usd=gross_notional,
        fee_usd=fee_usd,
        maximum_loss_usd=maximum_loss,
        available_depth_shares=available_depth_shares,
        order_type=order_type_upper,
        side=side_upper,
        minimum_size_rule_status=minimum_size_rule_status,
        reasons=unique_reasons,
        fee_rate=fee_rate,
        fee_rate_source=fee_rate_source,
        top_level_provenance=top_level_provenance,
        quote_timestamp_valid=quote_timestamp_valid,
        tick_validation_basis=tick_validation_basis,
        execution_context=normalized_execution_context,
        historical_tick_policy_requested=allow_historical_resting_ask_tick_policy,
        historical_tick_policy_eligible=historical_tick_policy_eligible,
        direct_polymarket_resting_ask_provenance=(
            direct_polymarket_resting_ask_provenance
        ),
        raw_compact_decision_passthrough_verified=(
            raw_compact_decision_passthrough_verified
        ),
        quote_non_crossed=quote_non_crossed,
        price_transformation_applied=price_transformation_applied,
        current_market_fee_required=require_current_market_fee,
    )


def _assess_required_fok_buy(
    *,
    constraints: Optional[VenueExecutionConstraints],
    intended_notional_usd: Optional[float],
    limit_price: Optional[float],
    ask_levels: Optional[Sequence[Mapping[str, Any]]],
    full_book_complete: Optional[bool],
    max_notional_usd: Optional[float],
    max_loss_usd: Optional[float],
    order_type: str,
    side: str,
    minimum_size_rule_status: str,
    base_reasons: Sequence[str],
    best_ask_price: Optional[float],
    best_ask_size: Optional[float],
    top_level_provenance: Optional[str],
    quote_timestamp_valid: Optional[bool],
    require_current_market_fee: bool,
    tick_validation_basis: Optional[str],
    execution_context: str,
    historical_tick_policy_requested: bool,
    historical_tick_policy_eligible: bool,
    direct_polymarket_resting_ask_provenance: Optional[bool],
    raw_compact_decision_passthrough_verified: Optional[bool],
    quote_non_crossed: Optional[bool],
    price_transformation_applied: Optional[bool],
) -> VenueExecutionAssessment:
    reasons = list(base_reasons)
    tick_size = constraints.tick_size if constraints else None
    top_price_valid = best_ask_price is not None and 0 < best_ask_price < 1
    top_size_valid = best_ask_size is not None and best_ask_size > 0
    if best_ask_price is not None:
        if not top_price_valid:
            reasons.append("invalid_top_level_ask_price")
        elif tick_size is not None and tick_size > 0 and not price_is_on_tick(
            best_ask_price,
            tick_size,
        ):
            reasons.append("top_level_ask_price_off_tick")

    target = (
        Decimal(str(intended_notional_usd))
        if intended_notional_usd is not None and intended_notional_usd > 0
        else Decimal("0")
    )
    limit = (
        Decimal(str(limit_price))
        if limit_price is not None and 0 < limit_price < 1
        else Decimal("0")
    )
    if top_price_valid and top_size_valid and target > 0 and limit > 0:
        top_price = Decimal(str(best_ask_price))
        top_size = Decimal(str(best_ask_size))
        if top_price > limit:
            reasons.append("top_ask_above_worst_price_limit")
            return _top_level_fok_assessment(
                constraints=constraints,
                target=target,
                top_price=top_price,
                top_size=top_size,
                limit_price=limit_price,
                max_notional_usd=max_notional_usd,
                max_loss_usd=max_loss_usd,
                order_type=order_type,
                side=side,
                minimum_size_rule_status=minimum_size_rule_status,
                reasons=reasons,
                top_level_provenance=top_level_provenance,
                quote_timestamp_valid=quote_timestamp_valid,
                require_current_market_fee=require_current_market_fee,
                tick_validation_basis=tick_validation_basis,
                execution_context=execution_context,
                historical_tick_policy_requested=historical_tick_policy_requested,
                historical_tick_policy_eligible=historical_tick_policy_eligible,
                direct_polymarket_resting_ask_provenance=(
                    direct_polymarket_resting_ask_provenance
                ),
                raw_compact_decision_passthrough_verified=(
                    raw_compact_decision_passthrough_verified
                ),
                quote_non_crossed=quote_non_crossed,
                price_transformation_applied=price_transformation_applied,
                price_within_limit=False,
            )
        if top_price * top_size >= target:
            return _top_level_fok_assessment(
                constraints=constraints,
                target=target,
                top_price=top_price,
                top_size=top_size,
                limit_price=limit_price,
                max_notional_usd=max_notional_usd,
                max_loss_usd=max_loss_usd,
                order_type=order_type,
                side=side,
                minimum_size_rule_status=minimum_size_rule_status,
                reasons=reasons,
                top_level_provenance=top_level_provenance,
                quote_timestamp_valid=quote_timestamp_valid,
                require_current_market_fee=require_current_market_fee,
                tick_validation_basis=tick_validation_basis,
                execution_context=execution_context,
                historical_tick_policy_requested=historical_tick_policy_requested,
                historical_tick_policy_eligible=historical_tick_policy_eligible,
                direct_polymarket_resting_ask_provenance=(
                    direct_polymarket_resting_ask_provenance
                ),
                raw_compact_decision_passthrough_verified=(
                    raw_compact_decision_passthrough_verified
                ),
                quote_non_crossed=quote_non_crossed,
                price_transformation_applied=price_transformation_applied,
                price_within_limit=True,
            )

    if historical_tick_policy_eligible:
        reasons.append("historical_tick_policy_only_valid_for_observed_best_ask_top_fill")
    return _assess_full_book_fok_buy(
        constraints=constraints,
        intended_notional_usd=intended_notional_usd,
        limit_price=limit_price,
        ask_levels=ask_levels,
        full_book_complete=full_book_complete,
        max_notional_usd=max_notional_usd,
        max_loss_usd=max_loss_usd,
        order_type=order_type,
        side=side,
        minimum_size_rule_status=minimum_size_rule_status,
        base_reasons=reasons,
        top_level_provenance=top_level_provenance,
        quote_timestamp_valid=quote_timestamp_valid,
        require_current_market_fee=require_current_market_fee,
        tick_validation_basis=tick_validation_basis,
        execution_context=execution_context,
        historical_tick_policy_requested=historical_tick_policy_requested,
        historical_tick_policy_eligible=historical_tick_policy_eligible,
        direct_polymarket_resting_ask_provenance=(
            direct_polymarket_resting_ask_provenance
        ),
        raw_compact_decision_passthrough_verified=(
            raw_compact_decision_passthrough_verified
        ),
        quote_non_crossed=quote_non_crossed,
        price_transformation_applied=price_transformation_applied,
    )


def _top_level_fok_assessment(
    *,
    constraints: Optional[VenueExecutionConstraints],
    target: Decimal,
    top_price: Decimal,
    top_size: Decimal,
    limit_price: Optional[float],
    max_notional_usd: Optional[float],
    max_loss_usd: Optional[float],
    order_type: str,
    side: str,
    minimum_size_rule_status: str,
    reasons: Sequence[str],
    top_level_provenance: Optional[str],
    quote_timestamp_valid: Optional[bool],
    require_current_market_fee: bool,
    tick_validation_basis: Optional[str],
    execution_context: str,
    historical_tick_policy_requested: bool,
    historical_tick_policy_eligible: bool,
    direct_polymarket_resting_ask_provenance: Optional[bool],
    raw_compact_decision_passthrough_verified: Optional[bool],
    quote_non_crossed: Optional[bool],
    price_transformation_applied: Optional[bool],
    price_within_limit: bool,
) -> VenueExecutionAssessment:
    assessment_reasons = list(reasons)
    if max_notional_usd is not None and target > Decimal(str(max_notional_usd)):
        assessment_reasons.append("venue_notional_cap_exceeded")
    consumed_shares = target / top_price if price_within_limit else Decimal("0")
    fee_rate = constraints.fee_rate if constraints else None
    fee_usd: Optional[float] = None
    maximum_loss: Optional[float] = None
    legs: tuple[Mapping[str, float], ...] = ()
    if price_within_limit and consumed_shares > 0 and fee_rate is not None and fee_rate >= 0:
        fee_usd = weather_taker_fee_usd(
            shares=float(consumed_shares),
            price=float(top_price),
            fee_rate=fee_rate,
        )
        maximum_loss = float(target) + fee_usd
        legs = (
            {
                "price": float(top_price),
                "available_shares": float(top_size),
                "consumed_shares": float(consumed_shares),
                "consumed_cost_usd": float(target),
                "fee_usd": fee_usd,
            },
        )
    if (
        maximum_loss is not None
        and max_loss_usd is not None
        and maximum_loss > max_loss_usd + 1e-9
    ):
        assessment_reasons.append("venue_maximum_loss_cap_exceeded")
    unique_reasons = tuple(dict.fromkeys(assessment_reasons))
    fill_proven = price_within_limit and consumed_shares > 0
    return VenueExecutionAssessment(
        eligible=fill_proven and not unique_reasons,
        intended_shares=float(consumed_shares) if fill_proven else None,
        limit_price=limit_price,
        gross_notional_usd=float(target) if target > 0 else None,
        fee_usd=fee_usd,
        maximum_loss_usd=maximum_loss,
        available_depth_shares=float(top_size),
        order_type=order_type,
        side=side,
        minimum_size_rule_status=minimum_size_rule_status,
        reasons=unique_reasons,
        full_book_enforcement_requested=True,
        full_book_complete=None,
        full_book_level_count=0,
        levels_consumed=1 if fill_proven else 0,
        cumulative_shares=float(consumed_shares),
        cumulative_cost_usd=float(target) if fill_proven else 0.0,
        weighted_average_price=float(top_price) if fill_proven else None,
        worst_fill_price=float(top_price) if fill_proven else None,
        unfilled_notional_usd=0.0 if fill_proven else float(target),
        execution_basis=(
            (
                HISTORICAL_RESTING_ASK_TICK_BASIS
                if tick_validation_basis == HISTORICAL_RESTING_ASK_TICK_BASIS
                else "top_level_proves_full_fill"
            )
            if fill_proven
            else "top_level_proves_no_fill_above_limit"
        ),
        all_or_none_fill=fill_proven and not unique_reasons,
        book_walk_legs=legs,
        fee_rate=fee_rate,
        fee_rate_source=(constraints.fee_rate_source if constraints else None),
        top_level_provenance=top_level_provenance,
        quote_timestamp_valid=quote_timestamp_valid,
        tick_validation_basis=tick_validation_basis,
        execution_context=execution_context,
        historical_tick_policy_requested=historical_tick_policy_requested,
        historical_tick_policy_eligible=historical_tick_policy_eligible,
        direct_polymarket_resting_ask_provenance=(
            direct_polymarket_resting_ask_provenance
        ),
        raw_compact_decision_passthrough_verified=(
            raw_compact_decision_passthrough_verified
        ),
        quote_non_crossed=quote_non_crossed,
        price_transformation_applied=price_transformation_applied,
        current_market_fee_required=require_current_market_fee,
    )


def _assess_full_book_fok_buy(
    *,
    constraints: Optional[VenueExecutionConstraints],
    intended_notional_usd: Optional[float],
    limit_price: Optional[float],
    ask_levels: Optional[Sequence[Mapping[str, Any]]],
    full_book_complete: Optional[bool],
    max_notional_usd: Optional[float],
    max_loss_usd: Optional[float],
    order_type: str,
    side: str,
    minimum_size_rule_status: str,
    base_reasons: Sequence[str],
    top_level_provenance: Optional[str],
    quote_timestamp_valid: Optional[bool],
    require_current_market_fee: bool,
    tick_validation_basis: Optional[str],
    execution_context: str,
    historical_tick_policy_requested: bool,
    historical_tick_policy_eligible: bool,
    direct_polymarket_resting_ask_provenance: Optional[bool],
    raw_compact_decision_passthrough_verified: Optional[bool],
    quote_non_crossed: Optional[bool],
    price_transformation_applied: Optional[bool],
) -> VenueExecutionAssessment:
    reasons = list(base_reasons)
    raw_levels = list(ask_levels) if ask_levels is not None else []
    if not raw_levels:
        reasons.append("missing_full_order_book_levels")
    if full_book_complete is not True:
        reasons.append("full_order_book_completeness_unproven")

    tick_size = constraints.tick_size if constraints else None
    fee_rate = constraints.fee_rate if constraints else None
    normalized_levels: list[tuple[Decimal, Decimal, int]] = []
    for index, level in enumerate(raw_levels):
        if not isinstance(level, Mapping):
            reasons.append("invalid_full_book_level")
            continue
        try:
            price = Decimal(str(level.get("price")))
            size = Decimal(str(level.get("size")))
        except Exception:
            reasons.append("invalid_full_book_level")
            continue
        if not Decimal("0") < price < Decimal("1") or size <= 0:
            reasons.append("invalid_full_book_level")
            continue
        if tick_size is not None and tick_size > 0 and not price_is_on_tick(
            float(price),
            tick_size,
        ):
            reasons.append("full_book_level_price_off_tick")
        normalized_levels.append((price, size, index))
    normalized_levels.sort(key=lambda item: (item[0], item[2]))

    target = (
        Decimal(str(intended_notional_usd))
        if intended_notional_usd is not None and intended_notional_usd > 0
        else Decimal("0")
    )
    if max_notional_usd is not None and target > Decimal(str(max_notional_usd)):
        reasons.append("venue_notional_cap_exceeded")

    limit = (
        Decimal(str(limit_price))
        if limit_price is not None and 0 < limit_price < 1
        else Decimal("0")
    )
    remaining = target
    cumulative_cost = Decimal("0")
    cumulative_shares = Decimal("0")
    worst_fill_price: Optional[Decimal] = None
    fee_sum = Decimal("0")
    legs: list[Mapping[str, float]] = []
    total_available_shares = Decimal("0")
    eligible_level_count = 0
    for price, available_shares, _index in normalized_levels:
        if limit <= 0 or price > limit:
            continue
        eligible_level_count += 1
        total_available_shares += available_shares
        if remaining <= 0:
            continue
        available_cost = price * available_shares
        consumed_cost = min(remaining, available_cost)
        consumed_shares = consumed_cost / price
        cumulative_cost += consumed_cost
        cumulative_shares += consumed_shares
        remaining -= consumed_cost
        worst_fill_price = price
        leg_fee_usd = 0.0
        if fee_rate is not None and fee_rate >= 0:
            leg_fee_unrounded = (
                consumed_shares
                * Decimal(str(fee_rate))
                * price
                * (Decimal("1") - price)
            )
            leg_fee_usd = _round_decimal_usd(leg_fee_unrounded)
            fee_sum += Decimal(str(leg_fee_usd))
        legs.append(
            {
                "price": float(price),
                "available_shares": float(available_shares),
                "consumed_shares": float(consumed_shares),
                "consumed_cost_usd": float(consumed_cost),
                "fee_usd": leg_fee_usd,
            }
        )

    if eligible_level_count == 0 and target > 0:
        reasons.append("no_full_book_ask_within_limit")
    if remaining > Decimal("0.000000001"):
        reasons.append("insufficient_full_book_depth_within_limit")

    weighted_price = (
        cumulative_cost / cumulative_shares
        if cumulative_shares > 0
        else None
    )
    full_book_filled = target > 0 and remaining <= Decimal("0.000000001")
    fee_usd = (
        float(fee_sum)
        if full_book_filled and fee_rate is not None and fee_rate >= 0
        else None
    )
    maximum_loss = (
        float(target) + float(fee_usd)
        if full_book_filled and fee_usd is not None
        else None
    )
    if (
        maximum_loss is not None
        and max_loss_usd is not None
        and maximum_loss > max_loss_usd + 1e-9
    ):
        reasons.append("venue_maximum_loss_cap_exceeded")

    unique_reasons = tuple(dict.fromkeys(reasons))
    return VenueExecutionAssessment(
        eligible=not unique_reasons,
        intended_shares=float(cumulative_shares) if cumulative_shares > 0 else None,
        limit_price=limit_price,
        gross_notional_usd=float(target) if target > 0 else None,
        fee_usd=fee_usd,
        maximum_loss_usd=maximum_loss,
        available_depth_shares=(
            float(total_available_shares) if normalized_levels else None
        ),
        order_type=order_type,
        side=side,
        minimum_size_rule_status=minimum_size_rule_status,
        reasons=unique_reasons,
        full_book_enforcement_requested=True,
        full_book_complete=full_book_complete,
        full_book_level_count=len(raw_levels),
        levels_consumed=len(legs),
        cumulative_shares=(
            float(cumulative_shares) if cumulative_shares > 0 else 0.0
        ),
        cumulative_cost_usd=float(cumulative_cost),
        weighted_average_price=(
            float(weighted_price) if weighted_price is not None else None
        ),
        worst_fill_price=(
            float(worst_fill_price) if worst_fill_price is not None else None
        ),
        unfilled_notional_usd=max(0.0, float(remaining)),
        execution_basis="full_order_book_fok_buy_usdc_walk",
        all_or_none_fill=full_book_filled and not unique_reasons,
        book_walk_legs=tuple(legs),
        fee_rate=fee_rate,
        fee_rate_source=(constraints.fee_rate_source if constraints else None),
        top_level_provenance=top_level_provenance,
        quote_timestamp_valid=quote_timestamp_valid,
        tick_validation_basis=tick_validation_basis,
        execution_context=execution_context,
        historical_tick_policy_requested=historical_tick_policy_requested,
        historical_tick_policy_eligible=historical_tick_policy_eligible,
        direct_polymarket_resting_ask_provenance=(
            direct_polymarket_resting_ask_provenance
        ),
        raw_compact_decision_passthrough_verified=(
            raw_compact_decision_passthrough_verified
        ),
        quote_non_crossed=quote_non_crossed,
        price_transformation_applied=price_transformation_applied,
        current_market_fee_required=require_current_market_fee,
    )


def _round_decimal_usd(
    value: Decimal,
    decimals: int = VENUE_FEE_ROUNDING_DECIMALS,
) -> float:
    quantum = Decimal("1").scaleb(-decimals)
    return float(value.quantize(quantum, rounding=ROUND_HALF_UP))


def price_is_on_tick(price: float, tick_size: float) -> bool:
    if tick_size <= 0:
        return False
    price_decimal = Decimal(str(price))
    tick_decimal = Decimal(str(tick_size))
    units = price_decimal / tick_decimal
    return units == units.to_integral_value()


def weather_taker_fee_usd(
    *,
    shares: float,
    price: float,
    fee_rate: float,
    decimals: int = VENUE_FEE_ROUNDING_DECIMALS,
) -> float:
    fee = (
        Decimal(str(shares))
        * Decimal(str(fee_rate))
        * Decimal(str(price))
        * (Decimal("1") - Decimal(str(price)))
    )
    quantum = Decimal("1").scaleb(-decimals)
    return float(fee.quantize(quantum, rounding=ROUND_HALF_UP))
