from __future__ import annotations

"""Deterministic scheduling, signal, and allocation rules for historical S01 research."""

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_CEILING, ROUND_DOWN, ROUND_HALF_UP
from fractions import Fraction
from types import MappingProxyType
from typing import Any, Mapping, Optional, Protocol, Sequence



ENGINE_SCHEMA_VERSION = "dailyweather.s01_parity_engine.v1"
FROZEN_STRATEGY_ID = "S01"
FROZEN_VARIANT_ID = "s01_fv_both_all_ld12"
EXPECTED_SIGNAL_COUNT = 2_089
EXPECTED_DECISION_WINDOW_COUNT = 548
EXPECTED_MAXIMUM_SIMULTANEOUS_SIGNALS = 16

DECISION_WINDOW_HOURS = (0, 6, 12, 18)
MAXIMUM_ENTRY_LAG_SECONDS = Decimal("900")
TARGET_EVENT_RISK_FRACTION = Decimal("0.02")
VENUE_MINIMUM_EVENT_RISK_CEILING_FRACTION = Decimal("0.08")
TOTAL_RESERVED_RISK_FRACTION = Decimal("1.00")
WINDOW_ALLOCATION_POLICY_ID = (
    "target_0p020__min_accommodation__floor_0p000__window_equal_daily_sleeves"
)
WINDOW_ALLOCATION_POLICY = "equal_daily_sleeves"
CASH_FLOOR_FRACTION = Decimal("0")
DEFAULT_SHARE_INCREMENT = Decimal("0.01")
FEE_QUANTUM_USD = Decimal("0.00001")
MARKET_BUY_MAKER_QUANTUM_USD = Decimal("0.01")
OTHER_OUTCOME = "__OTHER__"
EPSILON = Decimal("0.000000000001")


class S01ParityError(RuntimeError):
    pass


@dataclass(frozen=True)
class FrozenCity:
    canonical: str
    name: str
    region: str
    aliases: tuple[str, ...] = ()


FROZEN_CITIES: tuple[FrozenCity, ...] = (
    FrozenCity("Austin, TX", "Austin", "TX"),
    FrozenCity("Beijing, CN", "Beijing", "CN"),
    FrozenCity("Chicago, IL", "Chicago", "IL"),
    FrozenCity("Dallas, TX", "Dallas", "TX"),
    FrozenCity("Denver, CO", "Denver", "CO"),
    FrozenCity("Guangzhou, CN", "Guangzhou", "CN"),
    FrozenCity("Houston, TX", "Houston", "TX"),
    FrozenCity("London, GB", "London", "GB"),
    FrozenCity("Los Angeles, CA", "Los Angeles", "CA", ("LA",)),
    FrozenCity("Madrid, ES", "Madrid", "ES"),
    FrozenCity("Miami, FL", "Miami", "FL"),
    FrozenCity("Moscow, RU", "Moscow", "RU"),
    FrozenCity("New York, NY", "New York", "NY", ("NYC", "New York City")),
    FrozenCity("Paris, FR", "Paris", "FR"),
    FrozenCity("San Francisco, CA", "San Francisco", "CA", ("SF",)),
    FrozenCity("Seattle, WA", "Seattle", "WA"),
    FrozenCity("Seoul, KR", "Seoul", "KR"),
    FrozenCity("Shanghai, CN", "Shanghai", "CN"),
    FrozenCity("Shenzhen, CN", "Shenzhen", "CN"),
    FrozenCity("Singapore, SG", "Singapore", "SG"),
    FrozenCity("Taipei, TW", "Taipei", "TW"),
    FrozenCity("Tokyo, JP", "Tokyo", "JP"),
)


def _city_lookup_key(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).strip()).casefold()


def _build_city_aliases() -> Mapping[str, FrozenCity]:
    output: dict[str, FrozenCity] = {}
    for city in FROZEN_CITIES:
        for value in (city.canonical, city.name, *city.aliases):
            key = _city_lookup_key(value)
            previous = output.get(key)
            if previous is not None and previous != city:
                raise S01ParityError(f"ambiguous frozen city alias: {value}")
            output[key] = city
    return MappingProxyType(output)


FROZEN_CITY_BY_ALIAS = _build_city_aliases()


def canonicalize_city(value: str) -> Optional[FrozenCity]:
    return FROZEN_CITY_BY_ALIAS.get(_city_lookup_key(value))




class OpportunityIdentity(Protocol):
    event_id: str
    token_id: str
    side: str
    decision_id: str
    decision_time_utc: datetime


def canonical_opportunity_key(candidate: OpportunityIdentity) -> str:
    return json.dumps(
        [
            str(candidate.event_id),
            str(candidate.token_id),
            str(candidate.side).upper(),
            str(candidate.decision_id),
        ],
        separators=(",", ":"),
        ensure_ascii=True,
    )


def canonical_opportunity_order(candidate: OpportunityIdentity) -> tuple[Any, ...]:
    signal_decision_time = getattr(candidate, "signal_decision_time_utc", None)
    decision_time = _aware_utc(
        candidate.decision_time_utc
        if signal_decision_time is None
        else signal_decision_time
    )
    return (
        decision_time,
        str(candidate.event_id),
        str(candidate.token_id),
        str(candidate.side).upper(),
        str(candidate.decision_id),
    )


def _aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise S01ParityError("UTC datetime must be timezone-aware")
    return value.astimezone(timezone.utc)


def _iso_z(value: datetime) -> str:
    return _aware_utc(value).isoformat(timespec="seconds").replace("+00:00", "Z")


@dataclass(frozen=True)
class DecisionWindow:
    observed_at_utc: datetime
    scheduled_at_utc: datetime
    lag_seconds: Decimal
    entry_allowed: bool
    window_key: str
    next_scheduled_at_utc: datetime


def classify_utc_decision_window(observed_at_utc: datetime) -> DecisionWindow:
    observed = _aware_utc(observed_at_utc)
    scheduled_hour = max(hour for hour in DECISION_WINDOW_HOURS if hour <= observed.hour)
    scheduled = observed.replace(
        hour=scheduled_hour, minute=0, second=0, microsecond=0
    )
    lag_microseconds = (
        (observed - scheduled).days * 86_400_000_000
        + (observed - scheduled).seconds * 1_000_000
        + (observed - scheduled).microseconds
    )
    lag = Decimal(lag_microseconds) / Decimal(1_000_000)
    next_scheduled = scheduled + timedelta(hours=6)
    return DecisionWindow(
        observed_at_utc=observed,
        scheduled_at_utc=scheduled,
        lag_seconds=lag,
        entry_allowed=lag <= MAXIMUM_ENTRY_LAG_SECONDS,
        window_key=f"s01-window:{scheduled.strftime('%Y%m%dT%H0000Z')}",
        next_scheduled_at_utc=next_scheduled,
    )


def next_utc_decision_windows(after_utc: datetime, count: int) -> tuple[datetime, ...]:
    if count < 0:
        raise ValueError("count must be non-negative")
    observed = _aware_utc(after_utc)
    first = classify_utc_decision_window(observed).next_scheduled_at_utc
    return tuple(first + timedelta(hours=6 * index) for index in range(count))


def equal_daily_sleeve_entitlement_usd(
    frozen_daily_spendable_cash_usd: Decimal,
    scheduled_at_utc: datetime,
) -> Decimal:
    frozen = _decimal(frozen_daily_spendable_cash_usd)
    scheduled = _aware_utc(scheduled_at_utc)
    if (
        frozen < 0
        or scheduled.hour not in DECISION_WINDOW_HOURS
        or scheduled.minute
        or scheduled.second
        or scheduled.microsecond
    ):
        raise S01ParityError("equal daily sleeve inputs are invalid")
    return frozen * Decimal(DECISION_WINDOW_HOURS.index(scheduled.hour) + 1) / Decimal("4")


def equal_daily_sleeve_available_cash_usd(
    *,
    frozen_daily_spendable_cash_usd: Decimal,
    confirmed_daily_spend_usd: Decimal,
    actual_available_cash_usd: Decimal,
    scheduled_at_utc: datetime,
) -> Decimal:
    spent = _decimal(confirmed_daily_spend_usd)
    actual = _decimal(actual_available_cash_usd)
    if spent < 0 or actual < 0:
        raise S01ParityError("equal daily sleeve cash values cannot be negative")
    entitlement = equal_daily_sleeve_entitlement_usd(
        frozen_daily_spendable_cash_usd,
        scheduled_at_utc,
    )
    return min(actual, max(Decimal("0"), entitlement - spent))


def window_intent_key(window: DecisionWindow, candidate: OpportunityIdentity) -> str:
    digest = hashlib.sha256(canonical_opportunity_key(candidate).encode("utf-8"))
    return f"s01-intent:{window.window_key}:{digest.hexdigest()}"


class EligibilityInput(OpportunityIdentity, Protocol):
    bucket_type: str
    lead_days: Optional[int]
    quote_time_utc: datetime
    bid: Optional[Any]
    ask: Optional[Any]
    model_probability: Optional[Any]


@dataclass(frozen=True)
class SignalDecision:
    eligible: bool
    reason: str
    model_probability: Optional[Decimal]
    gross_edge: Optional[Decimal]


def _decimal(value: Any) -> Decimal:
    if isinstance(value, Decimal):
        output = value
    else:
        output = Decimal(str(value))
    if not output.is_finite():
        raise S01ParityError("non-finite numeric input")
    return output


def _optional_decimal(value: Any) -> Optional[Decimal]:
    return None if value is None else _decimal(value)


def frozen_s01_eligibility(candidate: EligibilityInput) -> SignalDecision:
    probability = _optional_decimal(candidate.model_probability)
    signal_ask = getattr(candidate, "signal_ask", None)
    ask = _optional_decimal(
        candidate.ask if signal_ask is None else signal_ask
    )
    if probability is None or ask is None:
        return SignalDecision(False, "missing_probability_or_ask", probability, None)
    computed_edge = probability - ask
    supplied_edge = _optional_decimal(getattr(candidate, "gross_edge", None))
    if supplied_edge is not None and abs(supplied_edge - computed_edge) > EPSILON:
        return SignalDecision(
            False, "gross_edge_identity_mismatch", probability, supplied_edge
        )
    edge = supplied_edge if supplied_edge is not None else computed_edge
    if candidate.lead_days not in {1, 2}:
        return SignalDecision(False, "lead_day_filter", probability, edge)
    signal_decision_time = getattr(candidate, "signal_decision_time_utc", None)
    decision_time = _aware_utc(
        candidate.decision_time_utc
        if signal_decision_time is None
        else signal_decision_time
    )
    signal_quote_time = getattr(candidate, "signal_quote_time_utc", None)
    quote_time = _aware_utc(
        candidate.quote_time_utc
        if signal_quote_time is None
        else signal_quote_time
    )
    if quote_time > decision_time:
        return SignalDecision(False, "quote_after_decision", probability, edge)
    if (decision_time - quote_time).total_seconds() > 60.0:
        return SignalDecision(False, "quote_age_above_60s", probability, edge)
    signal_bid = getattr(candidate, "signal_bid", None)
    bid = _optional_decimal(candidate.bid if signal_bid is None else signal_bid)
    if bid is None:
        return SignalDecision(False, "missing_two_sided_quote", probability, edge)
    spread = ask - bid
    if spread < 0:
        return SignalDecision(False, "crossed_book", probability, edge)
    if spread > Decimal("0.10") + EPSILON:
        return SignalDecision(False, "spread_above_0.10", probability, edge)
    if probability < Decimal("0.70"):
        return SignalDecision(False, "fair_value_below_0.70", probability, edge)
    if edge < Decimal("0.08"):
        return SignalDecision(False, "gross_edge_below_0.08", probability, edge)
    if str(candidate.bucket_type).lower() in {"exact", "range"} and (
        probability < Decimal("0.98")
        or ask < Decimal("0.70")
        or edge < Decimal("0.10")
    ):
        return SignalDecision(False, "bounded_bucket_gate", probability, edge)
    return SignalDecision(True, "signal", probability, edge)


@dataclass(frozen=True)
class PositionLeg:
    event_id: str
    market_id: str
    token_id: str
    side: str
    shares: Decimal
    gross_notional_usd: Decimal
    fee_usd: Decimal
    cash_outlay_usd: Decimal

    def __post_init__(self) -> None:
        side = str(self.side).upper()
        if side not in {"YES", "NO"}:
            raise S01ParityError(f"unsupported position side: {self.side}")
        object.__setattr__(self, "side", side)
        for field in (
            "shares",
            "gross_notional_usd",
            "fee_usd",
            "cash_outlay_usd",
        ):
            value = _decimal(getattr(self, field))
            if value < 0:
                raise S01ParityError(f"negative position field: {field}")
            object.__setattr__(self, field, value)
        if self.cash_outlay_usd != self.gross_notional_usd + self.fee_usd:
            raise S01ParityError("position cash outlay does not equal notional plus fee")


@dataclass(frozen=True)
class OpenEvent:
    event_id: str
    outcome_market_ids: tuple[str, ...]
    positions: tuple[PositionLeg, ...] = ()

    def __post_init__(self) -> None:
        outcomes = tuple(sorted(set(str(item) for item in self.outcome_market_ids)))
        if not outcomes or OTHER_OUTCOME in outcomes:
            raise S01ParityError("event outcomes must exclude the reserved __OTHER__ value")
        if any(position.event_id != self.event_id for position in self.positions):
            raise S01ParityError("position event id mismatch")
        if any(position.market_id not in outcomes for position in self.positions):
            raise S01ParityError("position market missing from event outcomes")
        object.__setattr__(self, "outcome_market_ids", outcomes)


def committed_cash(positions: Sequence[PositionLeg]) -> Decimal:
    return sum((item.cash_outlay_usd for item in positions), Decimal("0"))


def scenario_pnl(
    positions: Sequence[PositionLeg], outcome_market_id: str
) -> Decimal:
    total = Decimal("0")
    for position in positions:
        market_wins = position.market_id == outcome_market_id
        leg_wins = market_wins if position.side == "YES" else not market_wins
        payout = position.shares if leg_wins else Decimal("0")
        total += payout - position.cash_outlay_usd
    return total


def joint_event_maximum_loss(
    positions: Sequence[PositionLeg], outcome_market_ids: Sequence[str]
) -> Decimal:
    if not positions:
        return Decimal("0")
    outcomes = tuple(dict.fromkeys([*(str(item) for item in outcome_market_ids), OTHER_OUTCOME]))
    if len(outcomes) == 1:
        raise S01ParityError("event outcome universe is empty")
    return max(
        Decimal("0"),
        max(-scenario_pnl(positions, outcome) for outcome in outcomes),
    )


def event_committed_cash(event: OpenEvent) -> Decimal:
    return committed_cash(event.positions)


def event_joint_maximum_loss(event: OpenEvent) -> Decimal:
    return joint_event_maximum_loss(event.positions, event.outcome_market_ids)


@dataclass(frozen=True)
class PlanningState:
    equity_usd: Decimal
    available_cash_usd: Decimal
    open_events: tuple[OpenEvent, ...] = ()

    def __post_init__(self) -> None:
        equity = _decimal(self.equity_usd)
        cash = _decimal(self.available_cash_usd)
        if equity < 0 or cash < 0:
            raise S01ParityError("equity and available cash must be non-negative")
        if len({item.event_id for item in self.open_events}) != len(self.open_events):
            raise S01ParityError("duplicate open event")
        object.__setattr__(self, "equity_usd", equity)
        object.__setattr__(self, "available_cash_usd", cash)
        if self.committed_cash_usd + cash > equity + EPSILON:
            raise S01ParityError("committed cash plus available cash exceeds equity")
        if self.total_reserved_risk_usd > equity + EPSILON:
            raise S01ParityError("initial total reserved risk exceeds equity")

    @property
    def committed_cash_usd(self) -> Decimal:
        return sum((event_committed_cash(item) for item in self.open_events), Decimal("0"))

    @property
    def total_reserved_risk_usd(self) -> Decimal:
        return sum((event_joint_maximum_loss(item) for item in self.open_events), Decimal("0"))

    def event(self, event_id: str) -> Optional[OpenEvent]:
        return next((item for item in self.open_events if item.event_id == event_id), None)

    def with_position(
        self, candidate: "ExecutionCandidate", position: PositionLeg
    ) -> "PlanningState":
        existing = self.event(candidate.event_id)
        outcomes = tuple(sorted(set(candidate.outcome_market_ids)))
        if existing is not None and existing.outcome_market_ids != outcomes:
            raise S01ParityError("event outcome universe changed within batch")
        replacement = OpenEvent(
            event_id=candidate.event_id,
            outcome_market_ids=outcomes,
            positions=(*(existing.positions if existing else ()), position),
        )
        events = tuple(
            replacement if item.event_id == candidate.event_id else item
            for item in self.open_events
        )
        if existing is None:
            events = (*events, replacement)
        events = tuple(sorted(events, key=lambda item: item.event_id))
        return PlanningState(
            equity_usd=self.equity_usd,
            available_cash_usd=self.available_cash_usd - position.cash_outlay_usd,
            open_events=events,
        )


@dataclass(frozen=True)
class ExecutionCandidate:
    decision_id: str
    event_id: str
    market_id: str
    token_id: str
    side: str
    city: str
    bucket_type: str
    lead_days: Optional[int]
    decision_time_utc: datetime
    quote_time_utc: datetime
    bid: Optional[Any]
    ask: Optional[Any]
    model_probability: Optional[Any]
    fee_rate: Any
    minimum_order_size_shares: Any
    available_depth_shares: Any
    outcome_market_ids: tuple[str, ...]
    gross_edge: Optional[Any] = None
    share_increment: Any = DEFAULT_SHARE_INCREMENT
    signal_decision_time_utc: Optional[datetime] = None
    signal_quote_time_utc: Optional[datetime] = None
    signal_bid: Optional[Any] = None
    signal_ask: Optional[Any] = None

    def __post_init__(self) -> None:
        side = str(self.side).upper()
        if side not in {"YES", "NO"}:
            raise S01ParityError(f"unsupported candidate side: {self.side}")
        object.__setattr__(self, "side", side)
        outcomes = tuple(sorted(set(str(item) for item in self.outcome_market_ids)))
        if not outcomes or OTHER_OUTCOME in outcomes or self.market_id not in outcomes:
            raise S01ParityError("candidate has incomplete event outcome universe")
        object.__setattr__(self, "outcome_market_ids", outcomes)
        for field in (
            "bid",
            "ask",
            "model_probability",
            "fee_rate",
            "minimum_order_size_shares",
            "available_depth_shares",
            "gross_edge",
            "share_increment",
            "signal_bid",
            "signal_ask",
        ):
            object.__setattr__(self, field, _optional_decimal(getattr(self, field)))
        if self.fee_rate is None or self.fee_rate < 0 or self.fee_rate > 1:
            raise S01ParityError("fee rate must be in [0,1]")
        if self.minimum_order_size_shares is None or self.minimum_order_size_shares <= 0:
            raise S01ParityError("venue minimum must be positive")
        if self.available_depth_shares is None or self.available_depth_shares < 0:
            raise S01ParityError("available depth must be non-negative")
        if self.share_increment is None or self.share_increment <= 0:
            raise S01ParityError("share increment must be positive")
        if self.ask is not None:
            object.__setattr__(
                self,
                "share_increment",
                marketable_buy_share_increment(
                    self.ask,
                    self.share_increment,
                ),
            )
        _aware_utc(self.decision_time_utc)
        _aware_utc(self.quote_time_utc)
        if self.signal_decision_time_utc is not None:
            _aware_utc(self.signal_decision_time_utc)
        if self.signal_quote_time_utc is not None:
            _aware_utc(self.signal_quote_time_utc)


def fill_fee_usd(
    *, shares: Decimal, price: Decimal, fee_rate: Decimal
) -> Decimal:
    return (
        shares * fee_rate * price * (Decimal("1") - price)
    ).quantize(FEE_QUANTUM_USD, rounding=ROUND_HALF_UP)


def marketable_buy_share_increment(
    price: Decimal,
    base_increment: Decimal = DEFAULT_SHARE_INCREMENT,
) -> Decimal:
    """Smallest share step whose BUY maker amount is an exact cent."""

    price_value = _decimal(price)
    increment_value = _decimal(base_increment)
    if (
        price_value <= 0
        or price_value >= 1
        or increment_value <= 0
    ):
        raise S01ParityError("marketable BUY precision inputs are invalid")
    units = Fraction(
        price_value * increment_value / MARKET_BUY_MAKER_QUANTUM_USD
    ).denominator
    return increment_value * units


def position_for(candidate: ExecutionCandidate, shares: Decimal) -> PositionLeg:
    quantity = _decimal(shares)
    price = _optional_decimal(candidate.ask)
    if quantity <= 0 or price is None or not Decimal("0") < price < Decimal("1"):
        raise S01ParityError("position requires positive shares and price in (0,1)")
    notional = quantity * price
    fee = fill_fee_usd(shares=quantity, price=price, fee_rate=candidate.fee_rate)
    return PositionLeg(
        event_id=candidate.event_id,
        market_id=candidate.market_id,
        token_id=candidate.token_id,
        side=candidate.side,
        shares=quantity,
        gross_notional_usd=notional,
        fee_usd=fee,
        cash_outlay_usd=notional + fee,
    )


@dataclass(frozen=True)
class PlannedOrder:
    canonical_key: str
    shares: Decimal
    sizing_mode: str
    gross_notional_usd: Decimal
    fee_usd: Decimal
    cash_outlay_usd: Decimal
    event_risk_before_usd: Decimal
    event_risk_after_usd: Decimal
    total_risk_after_usd: Decimal
    committed_cash_after_usd: Decimal


@dataclass(frozen=True)
class BatchDecision:
    canonical_key: str
    accepted: bool
    reason: str
    desired_shares: Optional[Decimal]
    order: Optional[PlannedOrder]


@dataclass(frozen=True)
class BatchPlan:
    decisions: tuple[BatchDecision, ...]
    orders: tuple[PlannedOrder, ...]
    final_state: PlanningState


def _floor_increment(value: Decimal, increment: Decimal) -> Decimal:
    return (value / increment).to_integral_value(rounding=ROUND_DOWN) * increment


def _ceil_increment(value: Decimal, increment: Decimal) -> Decimal:
    return (value / increment).to_integral_value(rounding=ROUND_CEILING) * increment


def _deduplicate_candidates(
    candidates: Sequence[ExecutionCandidate],
) -> tuple[ExecutionCandidate, ...]:
    grouped: dict[str, list[ExecutionCandidate]] = {}
    for candidate in candidates:
        grouped.setdefault(canonical_opportunity_key(candidate), []).append(candidate)
    output: list[ExecutionCandidate] = []
    for key, members in grouped.items():
        first = members[0]
        if any(item != first for item in members[1:]):
            raise S01ParityError(f"conflicting duplicate canonical opportunity: {key}")
        output.append(first)
    return tuple(sorted(output, key=canonical_opportunity_order))


def _maximum_target_shares(
    candidate: ExecutionCandidate,
    state: PlanningState,
    *,
    event_risk_cap: Decimal,
    total_risk_cap: Decimal,
) -> Decimal:
    existing = state.event(candidate.event_id)
    positions = existing.positions if existing else ()
    outcomes = existing.outcome_market_ids if existing else candidate.outcome_market_ids
    event_before = joint_event_maximum_loss(positions, outcomes)
    other_risk = state.total_reserved_risk_usd - event_before
    permitted_event_risk = min(event_risk_cap, total_risk_cap - other_risk)
    if permitted_event_risk < 0:
        return Decimal("0")
    increment = candidate.share_increment
    depth_units = int(candidate.available_depth_shares // increment)
    if depth_units <= 0:
        return Decimal("0")

    def cash_fits(units: int) -> bool:
        if units <= 0:
            return True
        return position_for(candidate, increment * units).cash_outlay_usd <= state.available_cash_usd + EPSILON

    low, high = 0, depth_units
    while low < high:
        middle = (low + high + 1) // 2
        if cash_fits(middle):
            low = middle
        else:
            high = middle - 1
    cash_and_depth_units = low
    if cash_and_depth_units <= 0:
        return Decimal("0")

    scenario_losses = {
        outcome: -scenario_pnl(positions, outcome)
        for outcome in (*outcomes, OTHER_OUTCOME)
    }

    def increasing_scenarios_fit(units: int) -> bool:
        if units <= 0:
            return all(loss <= permitted_event_risk + EPSILON for loss in scenario_losses.values())
        position = position_for(candidate, increment * units)
        for outcome, existing_loss in scenario_losses.items():
            market_wins = candidate.market_id == outcome
            leg_wins = market_wins if candidate.side == "YES" else not market_wins
            if leg_wins:
                continue
            if existing_loss + position.cash_outlay_usd > permitted_event_risk + EPSILON:
                return False
        return True

    low, high = 0, cash_and_depth_units
    while low < high:
        middle = (low + high + 1) // 2
        if increasing_scenarios_fit(middle):
            low = middle
        else:
            high = middle - 1
    quantity = increment * low
    if quantity <= 0:
        return Decimal("0")
    proposed = (*positions, position_for(candidate, quantity))
    if joint_event_maximum_loss(proposed, outcomes) > permitted_event_risk + EPSILON:
        return Decimal("0")
    return quantity


def _assess_exact_shares(
    candidate: ExecutionCandidate,
    state: PlanningState,
    shares: Decimal,
    *,
    event_risk_cap: Decimal,
    total_risk_cap: Decimal,
) -> tuple[Optional[PositionLeg], str]:
    if shares != _floor_increment(shares, candidate.share_increment):
        return None, "share_quantity_not_marketable_buy_representable"
    if candidate.available_depth_shares + EPSILON < shares:
        return None, "insufficient_depth_for_exact_minimum"
    position = position_for(candidate, shares)
    if position.cash_outlay_usd > state.available_cash_usd + EPSILON:
        return None, "insufficient_available_cash"
    if state.committed_cash_usd + position.cash_outlay_usd > state.equity_usd + EPSILON:
        return None, "cash_solvency_cap"
    existing = state.event(candidate.event_id)
    positions = existing.positions if existing else ()
    outcomes = existing.outcome_market_ids if existing else candidate.outcome_market_ids
    if existing is not None and outcomes != candidate.outcome_market_ids:
        return None, "event_outcome_universe_mismatch"
    before = joint_event_maximum_loss(positions, outcomes)
    after = joint_event_maximum_loss((*positions, position), outcomes)
    total_after = state.total_reserved_risk_usd - before + after
    if after > event_risk_cap + EPSILON:
        return None, "event_risk_cap"
    if total_after > total_risk_cap + EPSILON:
        return None, "total_reserved_risk_cap"
    return position, "eligible"


def plan_complete_batch(
    candidates: Sequence[ExecutionCandidate],
    initial_state: PlanningState,
    *,
    window: Optional[DecisionWindow] = None,
) -> BatchPlan:
    """Plan every unique candidate in frozen order, without mutating external state."""

    state = initial_state
    decisions: list[BatchDecision] = []
    orders: list[PlannedOrder] = []
    for candidate in _deduplicate_candidates(candidates):
        key = canonical_opportunity_key(candidate)
        if window is not None and not window.entry_allowed:
            decisions.append(BatchDecision(key, False, "decision_window_late", None, None))
            continue
        city = canonicalize_city(candidate.city)
        if city is None:
            decisions.append(BatchDecision(key, False, "city_not_in_frozen_allowlist", None, None))
            continue
        signal = frozen_s01_eligibility(candidate)
        if not signal.eligible:
            decisions.append(BatchDecision(key, False, signal.reason, None, None))
            continue
        existing = state.event(candidate.event_id)
        if existing is not None and existing.outcome_market_ids != candidate.outcome_market_ids:
            decisions.append(BatchDecision(key, False, "event_outcome_universe_mismatch", None, None))
            continue
        target_cap = state.equity_usd * TARGET_EVENT_RISK_FRACTION
        minimum_cap = (
            state.equity_usd * VENUE_MINIMUM_EVENT_RISK_CEILING_FRACTION
        )
        total_cap = state.equity_usd * TOTAL_RESERVED_RISK_FRACTION
        target_shares = _maximum_target_shares(
            candidate,
            state,
            event_risk_cap=target_cap,
            total_risk_cap=total_cap,
        )
        sizing_mode = "two_percent_proportional_target"
        desired_shares = target_shares
        if target_shares < candidate.minimum_order_size_shares:
            desired_shares = _ceil_increment(
                candidate.minimum_order_size_shares,
                candidate.share_increment,
            )
            sizing_mode = "exact_venue_minimum_accommodation"
            position, reason = _assess_exact_shares(
                candidate,
                state,
                desired_shares,
                event_risk_cap=minimum_cap,
                total_risk_cap=total_cap,
            )
            if position is None:
                decisions.append(
                    BatchDecision(
                        key,
                        False,
                        f"venue_minimum_accommodation:{reason}",
                        desired_shares,
                        None,
                    )
                )
                continue
        else:
            position, reason = _assess_exact_shares(
                candidate,
                state,
                desired_shares,
                event_risk_cap=target_cap,
                total_risk_cap=total_cap,
            )
            if position is None:
                decisions.append(BatchDecision(key, False, reason, desired_shares, None))
                continue
        existing = state.event(candidate.event_id)
        existing_positions = existing.positions if existing else ()
        outcomes = existing.outcome_market_ids if existing else candidate.outcome_market_ids
        event_before = joint_event_maximum_loss(existing_positions, outcomes)
        event_after = joint_event_maximum_loss((*existing_positions, position), outcomes)
        next_state = state.with_position(candidate, position)
        order = PlannedOrder(
            canonical_key=key,
            shares=position.shares,
            sizing_mode=sizing_mode,
            gross_notional_usd=position.gross_notional_usd,
            fee_usd=position.fee_usd,
            cash_outlay_usd=position.cash_outlay_usd,
            event_risk_before_usd=event_before,
            event_risk_after_usd=event_after,
            total_risk_after_usd=next_state.total_reserved_risk_usd,
            committed_cash_after_usd=next_state.committed_cash_usd,
        )
        orders.append(order)
        decisions.append(BatchDecision(key, True, "planned", desired_shares, order))
        state = next_state
    return BatchPlan(tuple(decisions), tuple(orders), state)
