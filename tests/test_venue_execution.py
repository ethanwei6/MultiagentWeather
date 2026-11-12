from __future__ import annotations

import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone

from daily_weather_pipeline.backtesting_engine import (
    DecisionSnapshot,
    FillPolicy,
    ForecastSnapshot,
    OrderAction,
    OrderIntent,
    QuoteSnapshot,
    SkipReason,
    _quote_with_worse_fill_tick,
    simulate_fixed_worse_price_economic_sensitivity,
    simulate_order_fill,
)
from daily_weather_pipeline.shadow_risk import (
    LiveSafetyConfig,
    PortfolioSnapshot,
    PreTradeRiskChecker,
    ProposedOrder,
)
from daily_weather_pipeline.venue_execution import (
    HISTORICAL_RESTING_ASK_TICK_BASIS,
    OFFICIAL_WEATHER_FEE_SCHEDULE_SOURCE,
    VenueExecutionConstraints,
)


NOW = datetime(2026, 7, 13, 12, 0, tzinfo=timezone.utc)


class VenueExecutionSemanticsTests(unittest.TestCase):
    def test_documented_weather_fee_schedule_allowed_replay_blocked_live(self) -> None:
        replay = simulate_order_fill(
            _backtest_order(),
            replace(
                _decision(),
                venue_execution_constraints=replace(
                    _venue_constraints(),
                    fee_rate=0.05,
                    fee_rate_source=OFFICIAL_WEATHER_FEE_SCHEDULE_SOURCE,
                ),
            ),
            fill_policy=FillPolicy(
                enforce_venue_constraints=True,
                require_full_book_execution=True,
                max_order_notional_usd=2.0,
                max_order_loss_usd=3.0,
            ),
        )
        live = PreTradeRiskChecker(
            replace(_live_config(), require_full_book_execution=True)
        ).check(
            replace(
                _live_order(),
                venue_fee_rate_source=OFFICIAL_WEATHER_FEE_SCHEDULE_SOURCE,
            ),
            PortfolioSnapshot(cash_balance_usd=100.0),
            now=NOW,
        )

        self.assertIsNotNone(replay.fill)
        self.assertNotIn(
            "non_current_market_fee_source",
            (replay.venue_execution_assessment or {})["reasons"],
        )
        self.assertFalse(live.allowed)
        self.assertIn("fee_source_not_current_market_info", live.block_reasons)

    def test_full_book_fok_buy_has_cross_path_all_or_none_parity(self) -> None:
        levels = (
            {"price": 0.82, "size": 2.0},
            {"price": 0.83, "size": 100.0},
            {"price": 0.80, "size": 1.0},
        )
        decision = replace(
            _decision(),
            quote=replace(
                _decision().quote,
                bid=0.79,
                ask=0.80,
                ask_size=1.0,
                ask_levels=levels,
                full_book_complete=True,
            ),
        )
        replay = simulate_order_fill(
            replace(_backtest_order(), limit_price=0.82),
            decision,
            fill_policy=FillPolicy(
                enforce_venue_constraints=True,
                require_full_book_execution=True,
                max_order_notional_usd=2.0,
                max_order_loss_usd=3.0,
            ),
        )
        live_order = replace(
            _live_order(),
            bid=0.79,
            ask=0.80,
            limit_price=0.82,
            available_depth_usd=0.80,
            available_depth_shares=1.0,
            ask_levels=levels,
            full_book_complete=True,
            book_provenance="fixture-current-book",
        )
        live = PreTradeRiskChecker(
            replace(_live_config(), require_full_book_execution=True)
        ).check(
            live_order,
            PortfolioSnapshot(cash_balance_usd=100.0),
            now=NOW,
        )

        self.assertIsNotNone(replay.fill)
        self.assertTrue(live.allowed)
        replay_assessment = replay.venue_execution_assessment or {}
        live_assessment = live.audit_metadata["venue_execution_assessment"]
        for field in (
            "cumulative_shares",
            "cumulative_cost_usd",
            "weighted_average_price",
            "worst_fill_price",
            "fee_usd",
            "maximum_loss_usd",
            "levels_consumed",
        ):
            self.assertEqual(live_assessment[field], replay_assessment[field])
        self.assertEqual(replay_assessment["levels_consumed"], 2)
        self.assertAlmostEqual(replay_assessment["cumulative_cost_usd"], 2.0)
        self.assertAlmostEqual(replay_assessment["fee_usd"], 0.0188)
        self.assertAlmostEqual(replay_assessment["maximum_loss_usd"], 2.0188)
        self.assertEqual(replay_assessment["worst_fill_price"], 0.82)
        self.assertTrue(replay_assessment["all_or_none_fill"])
        self.assertEqual(
            replay_assessment["execution_basis"],
            "full_order_book_fok_buy_usdc_walk",
        )
        self.assertAlmostEqual(replay.fill.price, replay_assessment["weighted_average_price"])
        self.assertAlmostEqual(replay.fill.shares, replay_assessment["cumulative_shares"])
        self.assertAlmostEqual(replay.fill.fee_usd, 0.0188)
        self.assertEqual(live.audit_metadata["depth_available_usd"], 2.0)

    def test_top_level_can_prove_entire_fok_buy_without_full_levels(self) -> None:
        policy = FillPolicy(
            enforce_venue_constraints=True,
            require_full_book_execution=True,
            max_order_notional_usd=2.0,
            max_order_loss_usd=3.0,
        )
        replay = simulate_order_fill(_backtest_order(), _decision(), fill_policy=policy)
        live = PreTradeRiskChecker(
            replace(_live_config(), require_full_book_execution=True)
        ).check(
            replace(_live_order(), book_provenance="fixture-current-book"),
            PortfolioSnapshot(cash_balance_usd=100.0),
            now=NOW,
        )

        self.assertIsNotNone(replay.fill)
        self.assertTrue(live.allowed)
        replay_assessment = replay.venue_execution_assessment or {}
        live_assessment = live.audit_metadata["venue_execution_assessment"]
        self.assertEqual(
            replay_assessment["execution_basis"],
            "top_level_proves_full_fill",
        )
        self.assertEqual(
            live_assessment["execution_basis"],
            "top_level_proves_full_fill",
        )
        self.assertEqual(replay_assessment["full_book_level_count"], 0)
        self.assertTrue(replay_assessment["all_or_none_fill"])
        self.assertAlmostEqual(replay_assessment["cumulative_cost_usd"], 2.0)
        self.assertAlmostEqual(replay_assessment["fee_usd"], 0.018)
        self.assertAlmostEqual(replay_assessment["maximum_loss_usd"], 2.018)

    def test_audited_historical_resting_ask_can_replace_missing_tick_in_replay_only(
        self,
    ) -> None:
        audited_quote = replace(
            _decision().quote,
            source_payload_id="telonex-polymarket-direct-resting-quote",
            source_cache_path="raw/polymarket/quotes/2026-07-13.jsonl.gz",
            direct_polymarket_resting_ask_provenance=True,
            raw_compact_decision_passthrough_verified=True,
            price_transformation_applied=False,
        )
        replay = simulate_order_fill(
            _backtest_order(),
            replace(
                _decision(),
                quote=audited_quote,
                venue_execution_constraints=replace(
                    _venue_constraints(),
                    tick_size=None,
                ),
            ),
            fill_policy=FillPolicy(
                enforce_venue_constraints=True,
                require_full_book_execution=True,
                allow_historical_resting_ask_tick_policy=True,
                max_order_notional_usd=2.0,
                max_order_loss_usd=3.0,
            ),
        )
        live = PreTradeRiskChecker(
            replace(_live_config(), require_full_book_execution=True)
        ).check(
            replace(
                _live_order(),
                venue_tick_size=None,
                book_provenance="current-polymarket-book",
            ),
            PortfolioSnapshot(cash_balance_usd=100.0),
            now=NOW,
        )

        replay_assessment = replay.venue_execution_assessment or {}
        live_assessment = live.audit_metadata["venue_execution_assessment"]
        self.assertIsNotNone(replay.fill)
        self.assertEqual(
            replay_assessment["execution_basis"],
            HISTORICAL_RESTING_ASK_TICK_BASIS,
        )
        self.assertEqual(
            replay_assessment["tick_validation_basis"],
            HISTORICAL_RESTING_ASK_TICK_BASIS,
        )
        self.assertEqual(replay_assessment["execution_context"], "historical_replay")
        self.assertTrue(replay_assessment["historical_tick_policy_eligible"])
        self.assertNotIn("missing_venue_tick_size", replay_assessment["reasons"])

        self.assertFalse(live.allowed)
        self.assertIn("missing_venue_tick_size", live.block_reasons)
        self.assertIsNone(live_assessment["tick_validation_basis"])
        self.assertNotEqual(
            live_assessment["execution_basis"],
            HISTORICAL_RESTING_ASK_TICK_BASIS,
        )
        self.assertEqual(live_assessment["execution_context"], "live_or_unspecified")

    def test_historical_tick_policy_fails_closed_when_any_audit_proof_is_missing(
        self,
    ) -> None:
        audited_quote = replace(
            _decision().quote,
            source_payload_id="telonex-polymarket-direct-resting-quote",
            direct_polymarket_resting_ask_provenance=True,
            raw_compact_decision_passthrough_verified=True,
            price_transformation_applied=False,
        )
        base_decision = replace(
            _decision(),
            quote=audited_quote,
            venue_execution_constraints=replace(_venue_constraints(), tick_size=None),
        )
        policy = FillPolicy(
            enforce_venue_constraints=True,
            require_full_book_execution=True,
            allow_historical_resting_ask_tick_policy=True,
            max_order_notional_usd=2.0,
            max_order_loss_usd=3.0,
        )
        cases = (
            (
                "non_fok",
                replace(_backtest_order(), order_type="GTC"),
                base_decision,
                "historical_tick_policy_requires_fok_buy",
            ),
            (
                "limit_not_observed_ask",
                replace(_backtest_order(), limit_price=0.83),
                base_decision,
                "historical_tick_policy_requires_limit_equal_observed_best_ask",
            ),
            (
                "not_direct_polymarket",
                _backtest_order(),
                replace(
                    base_decision,
                    quote=replace(
                        audited_quote,
                        direct_polymarket_resting_ask_provenance=False,
                    ),
                ),
                "historical_tick_policy_requires_direct_polymarket_resting_ask_provenance",
            ),
            (
                "passthrough_unverified",
                _backtest_order(),
                replace(
                    base_decision,
                    quote=replace(
                        audited_quote,
                        raw_compact_decision_passthrough_verified=False,
                    ),
                ),
                "historical_tick_policy_requires_raw_compact_decision_passthrough",
            ),
            (
                "crossed",
                _backtest_order(),
                replace(base_decision, quote=replace(audited_quote, bid=0.83)),
                "historical_tick_policy_requires_non_crossed_quote",
            ),
            (
                "transformed",
                _backtest_order(),
                replace(
                    base_decision,
                    quote=replace(audited_quote, price_transformation_applied=True),
                ),
                "historical_tick_policy_requires_no_price_transformation",
            ),
            (
                "future_quote",
                _backtest_order(),
                replace(
                    base_decision,
                    quote=replace(audited_quote, timestamp_utc=NOW + timedelta(seconds=1)),
                ),
                "historical_tick_policy_requires_causal_quote",
            ),
        )

        for label, order, decision, expected_reason in cases:
            with self.subTest(label=label):
                result = simulate_order_fill(order, decision, fill_policy=policy)
                assessment = result.venue_execution_assessment or {}
                self.assertIsNone(result.fill)
                self.assertIn("missing_venue_tick_size", assessment["reasons"])
                self.assertIn(expected_reason, assessment["reasons"])
                self.assertFalse(assessment["historical_tick_policy_eligible"])

    def test_historical_tick_basis_cannot_validate_deeper_book_levels(self) -> None:
        quote = replace(
            _decision().quote,
            ask_size=1.0,
            ask_levels=(
                {"price": 0.82, "size": 1.0},
                {"price": 0.82, "size": 10.0},
            ),
            full_book_complete=True,
            direct_polymarket_resting_ask_provenance=True,
            raw_compact_decision_passthrough_verified=True,
            price_transformation_applied=False,
        )
        result = simulate_order_fill(
            _backtest_order(),
            replace(
                _decision(),
                quote=quote,
                venue_execution_constraints=replace(_venue_constraints(), tick_size=None),
            ),
            fill_policy=FillPolicy(
                enforce_venue_constraints=True,
                require_full_book_execution=True,
                allow_historical_resting_ask_tick_policy=True,
                max_order_notional_usd=2.0,
                max_order_loss_usd=3.0,
            ),
        )

        self.assertIsNone(result.fill)
        self.assertIn(
            "historical_tick_policy_only_valid_for_observed_best_ask_top_fill",
            (result.venue_execution_assessment or {})["reasons"],
        )

    def test_fixed_point_01_sensitivity_is_economic_only_after_strict_base_proof(
        self,
    ) -> None:
        quote = replace(
            _decision().quote,
            source_payload_id="telonex-polymarket-direct-resting-quote",
            direct_polymarket_resting_ask_provenance=True,
            raw_compact_decision_passthrough_verified=True,
            price_transformation_applied=False,
        )
        decision = replace(
            _decision(),
            quote=quote,
            venue_execution_constraints=replace(_venue_constraints(), tick_size=None),
        )
        policy = FillPolicy(
            enforce_venue_constraints=True,
            require_full_book_execution=True,
            allow_historical_resting_ask_tick_policy=True,
            max_order_notional_usd=2.0,
            max_order_loss_usd=3.0,
        )
        base = simulate_order_fill(_backtest_order(), decision, fill_policy=policy)
        transformed_reexecution = simulate_order_fill(
            _backtest_order(),
            replace(
                decision,
                quote=_quote_with_worse_fill_tick(
                    quote,
                    action=OrderAction.BUY,
                    tick=0.01,
                ),
            ),
            fill_policy=policy,
        )
        economic = simulate_fixed_worse_price_economic_sensitivity(
            _backtest_order(),
            decision,
            price_delta_usd=0.01,
            fill_policy=policy,
            base_fill_result=base,
        )

        self.assertIsNotNone(base.fill)
        self.assertIsNone(transformed_reexecution.fill)
        transformed_reasons = (
            transformed_reexecution.venue_execution_assessment or {}
        )["reasons"]
        self.assertIn(
            "historical_tick_policy_requires_limit_equal_observed_best_ask",
            transformed_reasons,
        )
        self.assertIn(
            "historical_tick_policy_requires_no_price_transformation",
            transformed_reasons,
        )

        self.assertIsNotNone(economic.fill)
        self.assertEqual(economic.status, "economic_sensitivity_filled")
        self.assertAlmostEqual(economic.fill.price, 0.83)
        self.assertAlmostEqual(economic.fill.shares, 2.0 / 0.83)
        self.assertAlmostEqual(economic.fill.notional_usd, 2.0)
        self.assertEqual(
            economic.price_source,
            "economic_counterfactual_not_venue_execution",
        )
        audit = economic.economic_sensitivity_assessment or {}
        self.assertTrue(audit["economic_sensitivity_only"])
        self.assertFalse(audit["counterfactual_price_venue_validated"])
        self.assertFalse(audit["counterfactual_price_tick_validated"])
        self.assertFalse(audit["counterfactual_price_limit_validated"])
        self.assertFalse(audit["counterfactual_order_submitted_or_reexecuted"])
        self.assertEqual(
            (economic.venue_execution_assessment or {})["execution_basis"],
            HISTORICAL_RESTING_ASK_TICK_BASIS,
        )

    def test_full_levels_required_when_top_level_cannot_cover_fok_buy(self) -> None:
        policy = FillPolicy(
            enforce_venue_constraints=True,
            require_full_book_execution=True,
            max_order_notional_usd=2.0,
            max_order_loss_usd=3.0,
        )
        replay = simulate_order_fill(
            _backtest_order(),
            replace(
                _decision(),
                quote=replace(_decision().quote, ask_size=1.0),
            ),
            fill_policy=policy,
        )
        live = PreTradeRiskChecker(
            replace(_live_config(), require_full_book_execution=True)
        ).check(
            replace(
                _live_order(),
                available_depth_shares=1.0,
                book_provenance="fixture-current-book",
            ),
            PortfolioSnapshot(cash_balance_usd=100.0),
            now=NOW,
        )

        self.assertIsNone(replay.fill)
        replay_reasons = set((replay.venue_execution_assessment or {})["reasons"])
        self.assertIn("missing_full_order_book_levels", replay_reasons)
        self.assertIn("full_order_book_completeness_unproven", replay_reasons)
        self.assertFalse(live.allowed)
        self.assertIn("missing_full_order_book_levels", live.block_reasons)
        self.assertIn("full_order_book_completeness_unproven", live.block_reasons)

    def test_full_book_uses_captured_market_fee_rate_per_leg(self) -> None:
        levels = (
            {"price": 0.80, "size": 1.0},
            {"price": 0.82, "size": 2.0},
        )
        decision = replace(
            _decision(),
            venue_execution_constraints=replace(
                _venue_constraints(),
                fee_rate=0.10,
            ),
            quote=replace(
                _decision().quote,
                bid=0.79,
                ask=0.80,
                ask_size=1.0,
                ask_levels=levels,
                full_book_complete=True,
            ),
        )
        result = simulate_order_fill(
            replace(_backtest_order(), limit_price=0.82),
            decision,
            fill_policy=FillPolicy(
                enforce_venue_constraints=True,
                require_full_book_execution=True,
                max_order_notional_usd=2.0,
                max_order_loss_usd=3.0,
            ),
        )

        assessment = result.venue_execution_assessment or {}
        self.assertIsNotNone(result.fill)
        self.assertAlmostEqual(assessment["fee_usd"], 0.0376)
        self.assertAlmostEqual(assessment["maximum_loss_usd"], 2.0376)
        self.assertEqual(
            assessment["maximum_loss_semantics"],
            "gross_requested_amount_plus_modeled_fees_upper_bound_not_exact_wallet_debit",
        )

    def test_full_book_fok_buy_stresses_are_exact_no_fills(self) -> None:
        base_quote = replace(
            _decision().quote,
            bid=0.79,
            ask=0.80,
            ask_size=1.0,
            full_book_complete=True,
        )
        policy = FillPolicy(
            enforce_venue_constraints=True,
            require_full_book_execution=True,
            max_order_notional_usd=2.0,
            max_order_loss_usd=3.0,
        )
        shallow = simulate_order_fill(
            _backtest_order(),
            replace(
                _decision(),
                quote=replace(
                    base_quote,
                    ask_levels=({"price": 0.80, "size": 1.0},),
                ),
            ),
            fill_policy=policy,
        )
        capped_out = simulate_order_fill(
            _backtest_order(),
            replace(
                _decision(),
                quote=replace(
                    base_quote,
                    ask_levels=({"price": 0.83, "size": 100.0},),
                ),
            ),
            fill_policy=policy,
        )
        off_tick_level = simulate_order_fill(
            _backtest_order(),
            replace(
                _decision(),
                quote=replace(
                    base_quote,
                    ask_levels=({"price": 0.815, "size": 100.0},),
                ),
            ),
            fill_policy=policy,
        )

        self.assertIsNone(shallow.fill)
        self.assertIn(
            "insufficient_full_book_depth_within_limit",
            (shallow.venue_execution_assessment or {})["reasons"],
        )
        self.assertEqual(
            (shallow.venue_execution_assessment or {})["cumulative_cost_usd"],
            0.8,
        )
        self.assertIsNone(capped_out.fill)
        self.assertIn(
            "no_full_book_ask_within_limit",
            (capped_out.venue_execution_assessment or {})["reasons"],
        )
        self.assertIsNone(off_tick_level.fill)
        self.assertIn(
            "full_book_level_price_off_tick",
            (off_tick_level.venue_execution_assessment or {})["reasons"],
        )

    def test_real_matched_fok_buy_refutes_blanket_five_share_floor(self) -> None:
        # Sanitized fields copied from the local legacy V2 execution ledger.
        receipt = {
            "success": True,
            "status": "matched",
            "requested_notional_usd": 1.364175,
            "requested_shares": 1.935,
            "makingAmount": 1.359997,
            "takingAmount": 1.915489,
            "orderID_present": True,
            "transaction_hash_present": True,
        }
        saved_book_min_order_size = 5.0

        self.assertTrue(receipt["success"])
        self.assertEqual(receipt["status"], "matched")
        self.assertTrue(receipt["orderID_present"])
        self.assertTrue(receipt["transaction_hash_present"])
        self.assertLess(receipt["requested_shares"], saved_book_min_order_size)
        self.assertLess(receipt["takingAmount"], saved_book_min_order_size)

    def test_backtest_and_live_fok_buy_share_max_loss_without_resizing(self) -> None:
        decision = _decision()
        order = _backtest_order()
        fill_policy = FillPolicy(
            enforce_venue_constraints=True,
            max_order_notional_usd=2.0,
            max_order_loss_usd=3.0,
            policy_name="v2_fok_buy_order_type_aware_venue_rules_v1",
        )

        replay = simulate_order_fill(order, decision, fill_policy=fill_policy)
        live = PreTradeRiskChecker(_live_config()).check(
            _live_order(),
            PortfolioSnapshot(cash_balance_usd=100.0),
            now=NOW,
        )

        self.assertIsNotNone(replay.fill)
        self.assertTrue(live.allowed)
        replay_venue = replay.venue_execution_assessment or {}
        live_venue = live.audit_metadata["venue_execution_assessment"]
        expected_shares = 2.0 / 0.82
        self.assertAlmostEqual(replay_venue["intended_shares"], expected_shares)
        self.assertAlmostEqual(live_venue["intended_shares"], expected_shares)
        self.assertLess(replay_venue["intended_shares"], 5.0)
        self.assertEqual(
            replay_venue["minimum_size_rule_status"],
            "not_applied_v2_market_buy_amount_is_usdc_and_subminimum_match_observed",
        )
        self.assertEqual(
            live_venue["minimum_size_rule_status"],
            replay_venue["minimum_size_rule_status"],
        )
        self.assertAlmostEqual(
            live_venue["maximum_loss_usd"],
            replay_venue["maximum_loss_usd"],
        )
        self.assertAlmostEqual(
            live.audit_metadata["risk_amount_usd"],
            replay_venue["maximum_loss_usd"],
        )

    def test_venue_enforcement_fails_closed_without_metadata(self) -> None:
        replay = simulate_order_fill(
            _backtest_order(),
            replace(_decision(), venue_execution_constraints=None),
            fill_policy=FillPolicy(
                enforce_venue_constraints=True,
                max_order_notional_usd=2.0,
                max_order_loss_usd=3.0,
            ),
        )
        live = PreTradeRiskChecker(_live_config()).check(
            replace(
                _live_order(),
                venue_min_order_size_shares=None,
                venue_tick_size=None,
                venue_fee_rate=None,
                available_depth_shares=None,
            ),
            PortfolioSnapshot(cash_balance_usd=100.0),
            now=NOW,
        )

        self.assertIsNone(replay.fill)
        self.assertEqual(replay.skip_reason, SkipReason.MISSING_VENUE_METADATA)
        self.assertFalse(live.allowed)
        self.assertIn("missing_venue_min_order_size", live.block_reasons)
        self.assertIn("missing_venue_tick_size", live.block_reasons)
        self.assertIn("missing_venue_fee_rate", live.block_reasons)
        self.assertIn("missing_venue_depth", live.block_reasons)

    def test_off_tick_depth_and_max_loss_each_fail_closed_without_bump(self) -> None:
        base_policy = FillPolicy(
            enforce_venue_constraints=True,
            max_order_notional_usd=2.0,
            max_order_loss_usd=3.0,
        )
        off_tick = simulate_order_fill(
            replace(_backtest_order(), limit_price=0.825),
            _decision(),
            fill_policy=base_policy,
        )
        shallow = simulate_order_fill(
            _backtest_order(),
            replace(_decision(), quote=replace(_decision().quote, ask_size=1.0)),
            fill_policy=base_policy,
        )
        over_loss_cap = simulate_order_fill(
            _backtest_order(),
            _decision(),
            fill_policy=replace(base_policy, max_order_loss_usd=2.0),
        )

        self.assertEqual(off_tick.skip_reason, SkipReason.ORDER_PRICE_OFF_TICK)
        self.assertEqual(shallow.skip_reason, SkipReason.INSUFFICIENT_VISIBLE_SIZE)
        self.assertEqual(
            over_loss_cap.skip_reason,
            SkipReason.VENUE_MAXIMUM_LOSS_CAP_EXCEEDED,
        )
        for result in (off_tick, shallow, over_loss_cap):
            assessment = result.venue_execution_assessment or {}
            self.assertAlmostEqual(
                assessment["intended_shares"],
                2.0 / assessment["limit_price"],
            )
            self.assertLess(assessment["intended_shares"], 5.0)

    def test_non_market_buy_minimum_semantics_remain_fail_closed(self) -> None:
        result = simulate_order_fill(
            replace(_backtest_order(), order_type="GTC"),
            _decision(),
            fill_policy=FillPolicy(
                enforce_venue_constraints=True,
                max_order_notional_usd=2.0,
                max_order_loss_usd=3.0,
            ),
        )

        self.assertIsNone(result.fill)
        self.assertEqual(
            result.skip_reason,
            SkipReason.UNVERIFIED_VENUE_MINIMUM_SIZE_SEMANTICS,
        )

    def test_fok_never_partial_fills_even_when_generic_policy_allows_partial(self) -> None:
        result = simulate_order_fill(
            _backtest_order(),
            replace(_decision(), quote=replace(_decision().quote, ask_size=1.0)),
            fill_policy=FillPolicy(allow_partial=True),
        )

        self.assertIsNone(result.fill)
        self.assertEqual(result.status, "insufficient_visible_size")
        self.assertEqual(result.filled_shares, 0.0)

    def test_fak_partial_execution_fails_closed_until_book_walk_exists(self) -> None:
        result = simulate_order_fill(
            replace(_backtest_order(), order_type="FAK"),
            _decision(),
            fill_policy=FillPolicy(allow_partial=True),
        )

        self.assertIsNone(result.fill)
        self.assertEqual(result.status, "fak_partial_execution_unimplemented")
        self.assertEqual(
            result.skip_reason,
            SkipReason.FAK_PARTIAL_EXECUTION_UNIMPLEMENTED,
        )

    def test_historical_fill_policy_remains_unchanged_by_default(self) -> None:
        result = simulate_order_fill(
            _backtest_order(),
            replace(_decision(), venue_execution_constraints=None),
        )

        self.assertIsNotNone(result.fill)
        self.assertIsNone(result.venue_execution_assessment)


def _venue_constraints() -> VenueExecutionConstraints:
    return VenueExecutionConstraints(
        min_order_size_shares=5.0,
        tick_size=0.01,
        fee_rate=0.05,
        fee_rate_source="fixture_current_market_info",
    )


def _decision() -> DecisionSnapshot:
    quote = QuoteSnapshot(
        timestamp_utc=NOW - timedelta(seconds=2),
        bid=0.81,
        ask=0.82,
        mid=0.815,
        last_price=0.82,
        spread=0.01,
        bid_size=10.0,
        ask_size=10.0,
        liquidity_depth=20.0,
        source_payload_id="fixture-book",
        source_cache_path="fixture.json",
    )
    return DecisionSnapshot(
        decision_id="venue-fixture",
        universe_row_id="venue-fixture",
        market_id="market-fixture",
        slug="fixture",
        question="Fixture?",
        normalized_city="London, GB",
        country_region="GB",
        timezone="Europe/London",
        market_date_local="2026-07-14",
        bucket_label="99F",
        bucket_type="exact",
        threshold_low_f=99.0,
        threshold_high_f=99.0,
        side="NO",
        yes_token_id="yes-token",
        no_token_id="no-token",
        token_id="no-token",
        decision_time_utc=NOW,
        decision_hour_utc=12,
        lead_days=1,
        quote_store="fixture",
        quote=quote,
        forecast=ForecastSnapshot(
            run_time_utc=NOW - timedelta(hours=6),
            source_name="fixture",
            temperature_2m_max_f=80.0,
        ),
        venue_execution_constraints=_venue_constraints(),
    )


def _backtest_order() -> OrderIntent:
    return OrderIntent(
        strategy_name="old_high_confidence_no_tail",
        decision_id="venue-fixture",
        token_id="no-token",
        side="NO",
        action=OrderAction.BUY,
        requested_notional_usd=2.0,
        limit_price=0.82,
        order_type="FOK",
    )


def _live_config() -> LiveSafetyConfig:
    return replace(
        LiveSafetyConfig(),
        enforce_venue_constraints=True,
        max_order_notional_usd=2.0,
        max_order_loss_usd=3.0,
    )


def _live_order() -> ProposedOrder:
    return ProposedOrder(
        decision_id="venue-fixture",
        strategy_name="old_high_confidence_no_tail",
        token_id="no-token",
        side="NO",
        action="BUY",
        notional_usd=2.0,
        limit_price=0.82,
        city="London, GB",
        market_date_local="2026-07-14",
        region="europe",
        quote_timestamp_utc=NOW - timedelta(seconds=2),
        bid=0.81,
        ask=0.82,
        spread=0.01,
        available_depth_usd=8.2,
        order_type="FOK",
        venue_min_order_size_shares=5.0,
        venue_tick_size=0.01,
        venue_fee_rate=0.05,
        venue_fee_rate_source="fixture_current_market_info",
        available_depth_shares=10.0,
        book_provenance="fixture-current-book",
    )


if __name__ == "__main__":
    unittest.main()
