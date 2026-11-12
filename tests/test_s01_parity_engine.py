from __future__ import annotations

import random
import unittest
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from daily_weather_pipeline.s01_parity_engine import (
    DECISION_WINDOW_HOURS,
    EXPECTED_DECISION_WINDOW_COUNT,
    EXPECTED_MAXIMUM_SIMULTANEOUS_SIGNALS,
    EXPECTED_SIGNAL_COUNT,
    FROZEN_CITIES,
    FROZEN_CITY_BY_ALIAS,
    BatchPlan,
    ExecutionCandidate,
    OpenEvent,
    PlanningState,
    PositionLeg,
    canonical_opportunity_key,
    canonical_opportunity_order,
    canonicalize_city,
    classify_utc_decision_window,
    committed_cash,
    frozen_s01_eligibility,
    joint_event_maximum_loss,
    marketable_buy_share_increment,
    next_utc_decision_windows,
    plan_complete_batch,
    window_intent_key,
)


UTC = timezone.utc
D = Decimal
DECISION_TIME = datetime(2026, 7, 22, 6, tzinfo=UTC)


def candidate(index: int = 0, **overrides: object) -> ExecutionCandidate:
    values: dict[str, object] = {
        "decision_id": f"decision-{index:02d}",
        "event_id": f"Austin, TX|2026-07-{23 + index:02d}",
        "market_id": f"market-{index:02d}",
        "token_id": f"token-{index:02d}",
        "side": "YES",
        "city": "Austin",
        "bucket_type": "above",
        "lead_days": 1,
        "decision_time_utc": DECISION_TIME,
        "quote_time_utc": DECISION_TIME - timedelta(seconds=10),
        "bid": D("0.19"),
        "ask": D("0.20"),
        "model_probability": D("0.90"),
        "fee_rate": D("0"),
        "minimum_order_size_shares": D("5"),
        "available_depth_shares": D("100"),
        "outcome_market_ids": (f"market-{index:02d}",),
    }
    values.update(overrides)
    return ExecutionCandidate(**values)


def leg(
    *,
    event_id: str,
    market_id: str,
    token_id: str,
    side: str,
    shares: str,
    cash: str,
) -> PositionLeg:
    return PositionLeg(
        event_id=event_id,
        market_id=market_id,
        token_id=token_id,
        side=side,
        shares=D(shares),
        gross_notional_usd=D(cash),
        fee_usd=D("0"),
        cash_outlay_usd=D(cash),
    )


class S01ParityEngineTests(unittest.TestCase):
    def test_historical_engine_constants_and_city_registry(self) -> None:
        self.assertEqual(
            (
                EXPECTED_SIGNAL_COUNT,
                EXPECTED_DECISION_WINDOW_COUNT,
                EXPECTED_MAXIMUM_SIMULTANEOUS_SIGNALS,
                DECISION_WINDOW_HOURS,
            ),
            (2089, 548, 16, (0, 6, 12, 18)),
        )
        self.assertEqual(len(FROZEN_CITIES), 22)
        self.assertEqual(canonicalize_city("NYC").canonical, "New York, NY")
        self.assertEqual(canonicalize_city(" new york city ").canonical, "New York, NY")
        self.assertEqual(canonicalize_city("LA").canonical, "Los Angeles, CA")
        self.assertEqual(canonicalize_city("SF").canonical, "San Francisco, CA")
        self.assertIsNone(canonicalize_city("Phoenix, AZ"))
        with self.assertRaises(TypeError):
            FROZEN_CITY_BY_ALIAS["phoenix"] = FROZEN_CITIES[0]  # type: ignore[index]

    def test_canonical_key_and_order_are_the_frozen_fields(self) -> None:
        row = candidate(
            1,
            event_id="event-a",
            token_id="token-b",
            side="no",
            decision_id="decision-c",
        )
        self.assertEqual(
            canonical_opportunity_key(row),
            '["event-a","token-b","NO","decision-c"]',
        )
        self.assertEqual(
            canonical_opportunity_order(row),
            (DECISION_TIME, "event-a", "token-b", "NO", "decision-c"),
        )

    def test_window_boundary_lateness_next_runs_and_restart_keys(self) -> None:
        exact = classify_utc_decision_window(DECISION_TIME)
        last_valid = classify_utc_decision_window(
            DECISION_TIME + timedelta(seconds=900)
        )
        late = classify_utc_decision_window(
            DECISION_TIME + timedelta(seconds=900, microseconds=1)
        )
        restarted = classify_utc_decision_window(
            DECISION_TIME + timedelta(minutes=7)
        )
        self.assertTrue(exact.entry_allowed)
        self.assertTrue(last_valid.entry_allowed)
        self.assertFalse(late.entry_allowed)
        self.assertEqual(exact.window_key, restarted.window_key)
        self.assertEqual(exact.window_key, "s01-window:20260722T060000Z")
        self.assertEqual(exact.next_scheduled_at_utc.hour, 12)
        self.assertEqual(
            [item.hour for item in next_utc_decision_windows(DECISION_TIME, 8)],
            [12, 18, 0, 6, 12, 18, 0, 6],
        )
        self.assertEqual(
            window_intent_key(exact, candidate()),
            window_intent_key(restarted, candidate()),
        )

    def test_frozen_signal_uses_raw_probability_and_edge_identity(self) -> None:
        accepted = frozen_s01_eligibility(
            candidate(
                bucket_type="exact",
                bid=D("0.79"),
                ask=D("0.80"),
                model_probability=D("0.99"),
                gross_edge=D("0.19"),
            )
        )
        self.assertTrue(accepted.eligible)
        self.assertEqual(accepted.model_probability, D("0.99"))
        self.assertEqual(accepted.gross_edge, D("0.19"))
        mismatch = frozen_s01_eligibility(
            candidate(gross_edge=D("0.69"))
        )
        self.assertEqual(mismatch.reason, "gross_edge_identity_mismatch")
        stale = frozen_s01_eligibility(
            candidate(quote_time_utc=DECISION_TIME - timedelta(seconds=61))
        )
        self.assertEqual(stale.reason, "quote_age_above_60s")

    def test_fresh_execution_price_does_not_rewrite_frozen_signal_membership(self) -> None:
        execution_time = DECISION_TIME + timedelta(minutes=5)
        row = candidate(
            decision_time_utc=execution_time,
            quote_time_utc=execution_time - timedelta(seconds=2),
            bid=D("0.29"),
            ask=D("0.30"),
            gross_edge=D("0.70"),
            signal_decision_time_utc=DECISION_TIME,
            signal_quote_time_utc=DECISION_TIME - timedelta(seconds=10),
            signal_bid=D("0.19"),
            signal_ask=D("0.20"),
        )

        signal = frozen_s01_eligibility(row)
        plan = plan_complete_batch(
            (row,),
            PlanningState(equity_usd=D("100"), available_cash_usd=D("100")),
        )

        self.assertTrue(signal.eligible)
        self.assertTrue(plan.decisions[0].accepted)
        self.assertEqual(plan.decisions[0].order.gross_notional_usd, D("1.9800"))

    def test_joint_risk_nets_two_mutually_exclusive_no_legs_but_not_cash(self) -> None:
        positions = (
            leg(
                event_id="event",
                market_id="A",
                token_id="no-a",
                side="NO",
                shares="5",
                cash="3.5",
            ),
            leg(
                event_id="event",
                market_id="B",
                token_id="no-b",
                side="NO",
                shares="5",
                cash="3.5",
            ),
        )
        self.assertEqual(committed_cash(positions), D("7.0"))
        self.assertEqual(joint_event_maximum_loss(positions, ("A", "B")), D("2.0"))

    def test_other_outcome_is_always_included_for_yes_risk(self) -> None:
        positions = (
            leg(
                event_id="event",
                market_id="A",
                token_id="yes-a",
                side="YES",
                shares="5",
                cash="2",
            ),
        )
        self.assertEqual(joint_event_maximum_loss(positions, ("A",)), D("2"))

    def test_proportional_order_stays_at_two_percent_and_is_not_raised_to_eight(self) -> None:
        initial = PlanningState(D("100"), D("100"))
        plan = plan_complete_batch((candidate(),), initial)
        self.assertEqual(len(plan.orders), 1)
        order = plan.orders[0]
        self.assertEqual(order.shares, D("10"))
        self.assertEqual(order.event_risk_after_usd, D("2"))
        self.assertEqual(order.sizing_mode, "two_percent_proportional_target")

    def test_market_buy_share_increment_keeps_maker_amount_to_cents(self) -> None:
        self.assertEqual(
            marketable_buy_share_increment(D("0.74")),
            D("0.50"),
        )
        plan = plan_complete_batch(
            (
                candidate(
                    ask=D("0.74"),
                    bid=D("0.73"),
                    model_probability=D("0.95"),
                ),
            ),
            PlanningState(D("250"), D("250")),
        )
        order = plan.orders[0]
        self.assertEqual(order.shares, D("6.50"))
        self.assertEqual(order.gross_notional_usd, D("4.8100"))
        self.assertEqual(
            order.gross_notional_usd,
            order.gross_notional_usd.quantize(D("0.01")),
        )

    def test_below_minimum_uses_exact_minimum_only_within_eight_percent(self) -> None:
        initial = PlanningState(D("100"), D("100"))
        accommodated = plan_complete_batch(
            (
                candidate(
                    ask=D("0.50"),
                    bid=D("0.49"),
                    model_probability=D("0.90"),
                    fee_rate=D("0.10"),
                ),
            ),
            initial,
        )
        order = accommodated.orders[0]
        self.assertEqual(order.shares, D("5"))
        self.assertEqual(order.fee_usd, D("0.12500"))
        self.assertEqual(order.cash_outlay_usd, D("2.62500"))
        self.assertEqual(order.sizing_mode, "exact_venue_minimum_accommodation")

        blocked = plan_complete_batch(
            (
                candidate(
                    ask=D("0.90"),
                    bid=D("0.89"),
                    model_probability=D("0.99"),
                    minimum_order_size_shares=D("10"),
                ),
            ),
            initial,
        )
        self.assertFalse(blocked.decisions[0].accepted)
        self.assertEqual(
            blocked.decisions[0].reason,
            "venue_minimum_accommodation:event_risk_cap",
        )

    def test_exact_minimum_fails_closed_when_depth_is_short(self) -> None:
        plan = plan_complete_batch(
            (candidate(available_depth_shares=D("4")),),
            PlanningState(D("100"), D("100")),
        )
        self.assertEqual(len(plan.orders), 0)
        self.assertEqual(
            plan.decisions[0].reason,
            "venue_minimum_accommodation:insufficient_depth_for_exact_minimum",
        )

    def test_cash_is_not_released_by_joint_risk_netting(self) -> None:
        positions = (
            leg(
                event_id="hedged",
                market_id="A",
                token_id="yes-a",
                side="YES",
                shares="100",
                cash="50",
            ),
            leg(
                event_id="hedged",
                market_id="A",
                token_id="no-a",
                side="NO",
                shares="100",
                cash="50",
            ),
        )
        event = OpenEvent("hedged", ("A",), positions)
        initial = PlanningState(D("100"), D("0"), (event,))
        self.assertEqual(initial.total_reserved_risk_usd, D("0"))
        self.assertEqual(initial.committed_cash_usd, D("100"))
        plan = plan_complete_batch((candidate(),), initial)
        self.assertEqual(len(plan.orders), 0)
        self.assertEqual(
            plan.decisions[0].reason,
            "venue_minimum_accommodation:insufficient_available_cash",
        )

    def test_second_no_leg_uses_joint_event_math(self) -> None:
        first = leg(
            event_id="Austin, TX|2026-07-23",
            market_id="A",
            token_id="no-a",
            side="NO",
            shares="5",
            cash="3.5",
        )
        event = OpenEvent(first.event_id, ("A", "B"), (first,))
        initial = PlanningState(D("100"), D("96.5"), (event,))
        second = candidate(
            event_id=first.event_id,
            market_id="B",
            token_id="no-b",
            side="NO",
            bid=D("0.69"),
            ask=D("0.70"),
            model_probability=D("0.99"),
            outcome_market_ids=("A", "B"),
        )
        plan = plan_complete_batch((second,), initial)
        self.assertEqual(plan.orders[0].shares, D("5"))
        self.assertEqual(plan.orders[0].event_risk_before_usd, D("3.5"))
        self.assertEqual(plan.orders[0].event_risk_after_usd, D("2.0"))
        self.assertEqual(plan.final_state.committed_cash_usd, D("7.0"))

    def test_complete_sixteen_candidate_batch_is_ordered_and_input_immutable(self) -> None:
        candidates = [candidate(index) for index in range(16)]
        random.Random(20260722).shuffle(candidates)
        before = tuple(candidates)
        initial = PlanningState(D("100"), D("100"))
        plan: BatchPlan = plan_complete_batch(candidates, initial)
        self.assertEqual(tuple(candidates), before)
        self.assertEqual(initial, PlanningState(D("100"), D("100")))
        self.assertEqual(len(plan.decisions), 16)
        self.assertEqual(len(plan.orders), 16)
        self.assertEqual(
            [item.canonical_key for item in plan.orders],
            [
                canonical_opportunity_key(item)
                for item in sorted(candidates, key=canonical_opportunity_order)
            ],
        )
        self.assertEqual(plan.final_state.committed_cash_usd, D("32"))
        self.assertEqual(plan.final_state.total_reserved_risk_usd, D("32"))

    def test_late_batch_is_planning_only_and_rejects_every_entry(self) -> None:
        window = classify_utc_decision_window(
            DECISION_TIME + timedelta(minutes=16)
        )
        initial = PlanningState(D("100"), D("100"))
        plan = plan_complete_batch((candidate(0), candidate(1)), initial, window=window)
        self.assertEqual(plan.orders, ())
        self.assertEqual(plan.final_state, initial)
        self.assertEqual(
            [item.reason for item in plan.decisions],
            ["decision_window_late", "decision_window_late"],
        )


if __name__ == "__main__":
    unittest.main()
