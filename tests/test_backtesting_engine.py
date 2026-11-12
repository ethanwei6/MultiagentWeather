from __future__ import annotations

import csv
import gzip
import json
import sqlite3
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from daily_weather_pipeline.backtesting_engine import (
    ANTI_OVERFIT_VALIDATION_SCHEMA,
    Gen2ValidationVariantSpec,
    Gen2ValidationVariantState,
    _score_frame_for_throughput,
    _build_candidate_weather_model_runner,
    BACKTEST_REPORT_SCHEMA,
    ConservativeFVEdgeConfig,
    ConservativeFVEdgeStrategy,
    CompactQuoteHistoryLookup,
    DEFAULT_STRATEGY_REPLAY_FEE_MODEL,
    DEFAULT_STRATEGY_REPLAY_FEE_RATE,
    DeterministicFixtureWeatherModelRunner,
    DictScoreStore,
    DefaultLeakageValidator,
    evaluate_exit_hold_primitive,
    ExposureIntent,
    ExitHoldAction,
    HeatWindowSnapshot,
    CanonicalLabel,
    CanonicalLabelStore,
    calculate_fill_fee_usd,
    DecisionSnapshot,
    FeeModelConfig,
    Fill,
    FillPolicy,
    ForecastSnapshot,
    HighConfidenceNOTailConfig,
    HighConfidenceNOTailStrategy,
    NoopStrategyPlugin,
    ObservedHighSnapshot,
    OrderAction,
    OrderIntent,
    PointInTimePolicy,
    PortfolioCaps,
    PortfolioCapValidator,
    PortfolioState,
    PositionSnapshot,
    QuoteSnapshot,
    ReplayGeneratedScoreCache,
    ReplayDataset,
    ReplayEngine,
    ReplayLedger,
    simulate_order_fill,
    SharedScoreContract,
    SkipReason,
    SQLiteDecisionFrameLoader,
    SQLiteDecisionArtifactLoader,
    StrategyFamily,
    StrategyDecision,
    StrategyInput,
    UniversalGateConfig,
    ValidationOnlyEngine,
    VALIDATION_ONLY_REPORT_SCHEMA,
    WeatherModelDecisionContext,
    WeatherModelPrediction,
    _weather_runner_slug,
    FEE_MODEL_POLYMARKET_WEATHER_TAKER,
    FEE_MODEL_ZERO,
    OFFICIAL_TRADING_REPLAY_SQL_PREDICATE,
    OLD_NO_TAIL_LIVE_FORWARD_SHADOW_PROFILE_KEY,
    evaluate_universal_gates,
    old_no_tail_live_forward_shadow_spec,
    run_candidate_replay,
    run_old_no_tail_live_forward_shadow_logging,
    run_conservative_fv_edge_candidate_replay,
    run_conservative_fv_edge_throughput_diagnostic,
    run_gen2_strategy_validation,
    write_conservative_fv_edge_accounting_replay_report,
    write_gen2_reproducibility_audit_report,
    write_old_no_tail_live_forward_shadow_report,
    write_framework_contract_report,
    _quote_with_worse_fill_tick,
    _write_gen2_strategy_validation_csvs,
)
from daily_weather_pipeline.venue_execution import OFFICIAL_WEATHER_FEE_SCHEDULE_SOURCE, VenueExecutionConstraints
from daily_weather_pipeline.cli import _ensure_decision_schema


class BacktestingEngineTests(unittest.TestCase):
    def test_canonical_label_store_uses_polymarket_labels_and_excludes_manual_review(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "canonical-labels.csv"
            _write_label_csv(
                path,
                [
                    {
                        "universe_row_id": "row-1",
                        "market_id": "market-1",
                        "canonical_label": "1",
                        "recommended_action": "relabel_to_polymarket",
                        "exclude_from_backtest_replay": "0",
                    },
                    {
                        "universe_row_id": "row-2",
                        "market_id": "market-2",
                        "canonical_label": "0",
                        "recommended_action": "exclude_or_manual_review",
                        "exclude_from_backtest_replay": "1",
                    },
                ],
            )
            store = CanonicalLabelStore.from_csv(path)

        label = store.get_for_decision(_decision("row-1", market_id="market-1"))
        excluded = store.get_for_decision(_decision("row-2", market_id="market-2"))
        self.assertIsNotNone(label)
        self.assertEqual(label.payout_for_side("YES"), 1)
        self.assertEqual(label.payout_for_side("NO"), 0)
        self.assertTrue(label.is_polymarket_canonical)
        self.assertIsNotNone(excluded)
        self.assertFalse(excluded.is_replay_allowed)

    def test_point_in_time_policy_flags_future_quote_and_excluded_labels(self) -> None:
        decision = _decision(
            "row-1",
            quote_time=datetime(2026, 1, 2, 12, 1, tzinfo=timezone.utc),
            forecast=None,
        )
        label = _label("row-1", recommended_action="exclude_or_manual_review", exclude=True)
        reasons = PointInTimePolicy().skip_reasons(decision, label)
        self.assertIn(SkipReason.FUTURE_QUOTE, reasons)
        self.assertIn(SkipReason.MISSING_FORECAST_SNAPSHOT, reasons)
        self.assertIn(SkipReason.CANONICAL_LABEL_EXCLUDED, reasons)

    def test_sqlite_loader_reads_scoped_current_decision_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "decision.sqlite"
            with sqlite3.connect(path) as conn:
                _ensure_decision_schema(conn)
                conn.execute(
                    """
                    INSERT INTO decision_artifact_scope(
                        scope_id, target_start_date, target_end_date, price_store,
                        max_staleness_minutes, built_at_utc, row_count
                    )
                    VALUES ('current', '2026-01-01', '2026-06-29', 'compact', 90, 'now', 1)
                    """
                )
                conn.execute(
                    """
                    INSERT INTO decision_snapshots(
                        decision_id, universe_row_id, market_id, slug, normalized_city,
                        market_date_local, bucket_type, side, token_id, decision_time_utc,
                        decision_hour_utc, quote_store, quote_timestamp_utc, bid, ask,
                        forecast_run_time_utc, forecast_source_name, forecast_features_json,
                        data_quality_flags, quote_data_quality_flags, created_at_utc
                    )
                    VALUES (
                        'decision-1', 'row-1', 'market-1', 'slug-1', 'London, GB',
                        '2026-01-02', 'exact', 'YES', 'token-yes',
                        '2026-01-01T12:00:00+00:00', 12, 'compact',
                        '2026-01-01T11:59:00+00:00', 0.40, 0.44,
                        '2026-01-01T06:00:00+00:00', 'open_meteo_historical_forecast',
                        '{"temperature": 42}', '["ok"]', '["quote_ok"]', 'now'
                    )
                    """
                )
                conn.commit()

            rows = list(SQLiteDecisionArtifactLoader(path).iter_decisions(limit=1))

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].decision_id, "decision-1")
        self.assertEqual(rows[0].quote.bid, 0.40)
        self.assertEqual(rows[0].forecast.features["temperature"], 42)
        self.assertEqual(rows[0].data_quality_flags, ("ok",))

    def test_sqlite_frame_loader_applies_official_predicate_and_preserves_source_audit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "decision.sqlite"
            with sqlite3.connect(path) as conn:
                _create_enriched_table(conn)
                _insert_enriched_row(conn, "good-row", quote_timestamp="2026-01-01T11:59:00+00:00")
                _insert_enriched_row(
                    conn,
                    "excluded-row",
                    quote_timestamp="2026-01-01T11:59:00+00:00",
                    exclude_from_backtest_replay=1,
                    recommended_action="exclude_or_manual_review",
                )
                _insert_enriched_row(conn, "missing-quote-row", quote_timestamp=None)
                conn.commit()

            frames = list(
                SQLiteDecisionFrameLoader(path, official_trading_replay_only=True).iter_frames()
            )

        self.assertEqual(OFFICIAL_TRADING_REPLAY_SQL_PREDICATE.count("quote_timestamp_utc"), 1)
        self.assertEqual([frame.decision.universe_row_id for frame in frames], ["good-row"])
        self.assertEqual(frames[0].decision.quote.source_cache_path, "/tmp/quote.parquet")
        self.assertEqual(frames[0].decision.quote.source_payload_id, "quote-payload")
        self.assertEqual(frames[0].decision.quote.data_quality_flags, ("quote_ok",))
        self.assertEqual(frames[0].canonical_label.audit_flags, ("settlement_ok",))

    def test_sqlite_frame_loader_without_official_predicate_reports_missing_quote_and_label_exclusion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "decision.sqlite"
            with sqlite3.connect(path) as conn:
                _create_enriched_table(conn)
                _insert_enriched_row(
                    conn,
                    "excluded-row",
                    quote_timestamp="2026-01-01T11:59:00+00:00",
                    exclude_from_backtest_replay=1,
                    recommended_action="exclude_or_manual_review",
                )
                _insert_enriched_row(conn, "missing-quote-row", quote_timestamp=None)
                conn.commit()

            frames = list(
                SQLiteDecisionFrameLoader(path, official_trading_replay_only=False).iter_frames()
            )

        reasons = {frame.decision.universe_row_id: frame.skip_reasons for frame in frames}
        self.assertIn(SkipReason.CANONICAL_LABEL_EXCLUDED, reasons["excluded-row"])
        self.assertIn(SkipReason.MISSING_QUOTE_SNAPSHOT, reasons["missing-quote-row"])

    def test_no_lookahead_validator_flags_future_forecast_score_and_observed_high(self) -> None:
        decision = _decision(
            "row-1",
            forecast=ForecastSnapshot(
                run_time_utc=datetime(2026, 1, 2, 13, tzinfo=timezone.utc),
                source_name="future",
                temperature_2m_max_f=72.0,
            ),
        )
        score = _score(
            observed_high=ObservedHighSnapshot(
                observed_high_f=73.0,
                observed_high_timestamp_utc=datetime(2026, 1, 2, 12, 30, tzinfo=timezone.utc),
                observation_source="future_station",
            )
        )
        frame = _frame(decision=decision, score=score)
        violations = DefaultLeakageValidator().validate(frame)
        self.assertEqual(
            sorted(violation.code for violation in violations),
            ["future_forecast", "future_observed_high"],
        )

    def test_validation_only_aggregate_report_is_deterministic_for_same_fixtures(self) -> None:
        decision = _decision("row-1")
        dataset = ReplayDataset(
            _ListDecisionLoader([decision]),
            _DictLabelStore({"row-1": _label("row-1")}),
            score_store=DictScoreStore({"row-1": _score_with_observed_high_and_heat_window()}),
            point_in_time_policy=PointInTimePolicy(require_shared_score=True),
        )
        first = ValidationOnlyEngine(dataset).validate(collect_gate_results=False).report
        second = ValidationOnlyEngine(dataset).validate(collect_gate_results=False).report
        for key in (
            "score_contract_coverage",
            "artifact_coverage",
            "universal_gate_summary",
            "canonical_label_action_counts",
            "frame_skip_reason_counts",
            "leakage_validation",
            "schema_gaps",
        ):
            self.assertEqual(first[key], second[key])

    def test_replay_engine_does_not_expose_labels_or_outcomes_to_strategy_input(self) -> None:
        seen: list[StrategyInput] = []

        class AssertingStrategy:
            name = "asserting_strategy"

            def evaluate(self, snapshot: StrategyInput) -> StrategyDecision:
                seen.append(snapshot)
                for forbidden in (
                    "canonical_label",
                    "resolved_outcome",
                    "realized_temperature_2m_max_f",
                ):
                    if hasattr(snapshot, forbidden):
                        raise AssertionError(f"StrategyInput leaked {forbidden}")
                return StrategyDecision(
                    orders=(
                        OrderIntent(
                            strategy_name=self.name,
                            decision_id=snapshot.decision_id,
                            token_id=snapshot.token_id,
                            side=snapshot.side,
                            action=OrderAction.BUY,
                            requested_notional_usd=1.0,
                            limit_price=snapshot.quote.ask,
                            reason="synthetic test intent",
                        ),
                    )
                )

        decision = _decision("row-1")
        dataset = ReplayDataset(_ListDecisionLoader([decision]), _DictLabelStore({"row-1": _label("row-1")}))
        run = ReplayEngine(dataset).replay(AssertingStrategy())
        self.assertEqual(len(seen), 1)
        self.assertEqual(len(run.order_intents), 1)
        self.assertEqual(run.report["mode"], "framework_only_no_pnl")

    def test_noop_strategy_records_no_action_without_accounting_pnl(self) -> None:
        decision = _decision("row-1")
        dataset = ReplayDataset(_ListDecisionLoader([decision]), _DictLabelStore({"row-1": _label("row-1")}))
        run = ReplayEngine(dataset).replay(NoopStrategyPlugin())
        self.assertEqual(run.order_intents, ())
        self.assertEqual(run.report["skip_reason_counts"][SkipReason.STRATEGY_NO_ACTION.value], 1)

    def test_replay_ledger_settlement_uses_canonical_label_by_side(self) -> None:
        label = _label("row-1", canonical_label=1)
        ledger = ReplayLedger(starting_cash_usd=10.0)
        ledger.apply_fill(
            Fill(
                token_id="token-yes",
                side="YES",
                action=OrderAction.BUY,
                timestamp_utc=datetime(2026, 1, 2, 12, tzinfo=timezone.utc),
                price=0.40,
                shares=2.0,
            )
        )
        settlement = ledger.settle_position("token-yes", "YES", label)
        self.assertEqual(settlement.payout_per_share, 1)
        self.assertAlmostEqual(ledger.cash_usd, 11.2)

        no_ledger = ReplayLedger(starting_cash_usd=10.0)
        no_ledger.apply_fill(
            Fill(
                token_id="token-no",
                side="NO",
                action=OrderAction.BUY,
                timestamp_utc=datetime(2026, 1, 2, 12, tzinfo=timezone.utc),
                price=0.30,
                shares=2.0,
            )
        )
        no_settlement = no_ledger.settle_position("token-no", "NO", label)
        self.assertEqual(no_settlement.payout_per_share, 0)
        self.assertAlmostEqual(no_ledger.cash_usd, 9.4)

    def test_polymarket_weather_taker_fee_formula_rounds_to_five_decimals(self) -> None:
        config = FeeModelConfig(
            fee_model=FEE_MODEL_POLYMARKET_WEATHER_TAKER,
            fee_rate=0.05,
        )
        self.assertAlmostEqual(
            calculate_fill_fee_usd(shares=2.0, price=0.50, fee_config=config),
            0.025,
        )
        self.assertAlmostEqual(
            calculate_fill_fee_usd(shares=(1.0 / 0.70), price=0.70, fee_config=config),
            0.015,
        )

    def test_zero_fee_model_remains_exactly_zero(self) -> None:
        self.assertEqual(
            calculate_fill_fee_usd(
                shares=2.0,
                price=0.50,
                fee_config=FeeModelConfig(fee_model=FEE_MODEL_ZERO),
            ),
            0.0,
        )

    def test_report_schema_keeps_label_policy_and_future_sections_explicit(self) -> None:
        self.assertIn("label_policy", BACKTEST_REPORT_SCHEMA["required_sections"])
        self.assertIn("accounting_summary", BACKTEST_REPORT_SCHEMA["required_sections"])
        self.assertIn("walk_forward_validation", BACKTEST_REPORT_SCHEMA["required_sections"])
        self.assertIn("plateau_analysis", BACKTEST_REPORT_SCHEMA["required_sections"])
        self.assertIn(
            "rolling_origin_or_walk_forward_splits",
            ANTI_OVERFIT_VALIDATION_SCHEMA["required_controls"],
        )
        self.assertEqual(
            BACKTEST_REPORT_SCHEMA["label_policy"]["exclude_rule"],
            "recommended_action=exclude_or_manual_review",
        )

    def test_shared_score_contract_validates_side_probability_consistency(self) -> None:
        score = _score(side="NO", p_yes=0.20, p_side=0.80)
        self.assertTrue(score.is_valid)
        bad_score = _score(side="NO", p_yes=0.20, p_side=0.20)
        self.assertIn("p_side_inconsistent_with_side", bad_score.score_errors)

    def test_conservative_fv_edge_strategy_places_buy_at_ask_for_open_bucket_score(self) -> None:
        decision = _decision("row-1", bucket_type="above")
        snapshot = _frame(decision=decision, score=_score(p_yes=0.86, p_side=0.86)).strategy_input()
        strategy = ConservativeFVEdgeStrategy()
        result = strategy.evaluate(snapshot)
        self.assertEqual(len(result.orders), 1)
        order = result.orders[0]
        self.assertEqual(order.action, OrderAction.BUY)
        self.assertEqual(order.limit_price, decision.quote.ask)
        self.assertEqual(order.requested_notional_usd, ConservativeFVEdgeConfig().order_notional_usd)
        self.assertEqual(order.metadata["quote_source_cache_path"], decision.quote.source_cache_path)

    def test_conservative_fv_edge_strategy_reports_gate_skip_reasons(self) -> None:
        decision = _decision("row-1", bucket_type="above")
        snapshot = _frame(decision=decision, score=_score(p_yes=0.46, p_side=0.46)).strategy_input()
        result = ConservativeFVEdgeStrategy().evaluate(snapshot)
        self.assertEqual(result.orders, ())
        self.assertIn(
            SkipReason.FAIR_VALUE_BELOW_MINIMUM.value,
            result.metadata["candidate_skip_reasons"],
        )
        self.assertIn(
            SkipReason.NET_EDGE_BELOW_MINIMUM.value,
            result.metadata["candidate_skip_reasons"],
        )

    def test_conservative_fv_edge_replay_uses_ask_accounting_and_quote_audit(self) -> None:
        decision = _decision("row-1", bucket_type="above")
        frame = _frame(decision=decision, score=_score(p_yes=0.86, p_side=0.86))
        run = run_conservative_fv_edge_candidate_replay(
            _ListFrameDataset([frame]),
            limit=1,
            execute_orders=True,
        )
        self.assertEqual(run.report["decision_counts"]["fills"], 1)
        self.assertEqual(run.fills[0].price, decision.quote.ask)
        self.assertEqual(run.report["execution_assumptions"]["buy_price_source"], "ask")
        self.assertEqual(run.report["execution_assumptions"]["sell_price_source"], "bid")
        self.assertEqual(run.report["quote_source_audit"]["frames_with_quote_source_cache_path"], 1)
        self.assertEqual(run.report["coverage"]["lead_days"], {"1": 1})
        self.assertEqual(run.report["profitability_claim"], "exact_fixed_replay_accounting_only")
        self.assertEqual(
            run.report["profitability_claim_scope"],
            "realized_accounting_for_this_exact_fixed_parameter_replay_only",
        )

    def test_fixed_accounting_replay_reports_trade_pnl_and_slices(self) -> None:
        decision = _decision("row-1", bucket_type="above")
        frame = _frame(
            decision=decision,
            score=_score(p_yes=0.86, p_side=0.86),
            label=_label("row-1", canonical_label=1),
        )
        run = run_conservative_fv_edge_candidate_replay(
            _ListFrameDataset([frame]),
            limit=1,
            execute_orders=True,
        )
        accounting = run.report["fixed_strategy_accounting"]
        self.assertTrue(accounting["accounting_complete"])
        self.assertEqual(accounting["fill_counts"]["attempted_fills"], 1)
        self.assertEqual(accounting["fill_counts"]["actual_fills"], 1)
        self.assertEqual(accounting["pnl"]["win_count"], 1)
        self.assertEqual(accounting["pnl"]["loss_count"], 0)
        self.assertAlmostEqual(accounting["pnl"]["total_cost_usd"], 1.0)
        expected_payout = 1.0 / decision.quote.ask
        self.assertAlmostEqual(accounting["pnl"]["total_payout_usd"], expected_payout)
        self.assertAlmostEqual(accounting["pnl"]["realized_pnl_usd"], expected_payout - 1.0)
        self.assertIn("London, GB", accounting["pnl_by_city"])
        self.assertIn("12", accounting["pnl_by_decision_hour_utc"])
        self.assertIn("2026-01", accounting["pnl_by_month"])
        self.assertIn("2026-W01", accounting["pnl_by_week"])
        self.assertEqual(accounting["concentration"]["top_city_by_absolute_pnl"]["key"], "London, GB")
        self.assertEqual(
            accounting["concentration"]["top_trade_by_absolute_pnl"]["decision_id"],
            decision.decision_id,
        )
        self.assertEqual(accounting["best_trades"][0]["decision_id"], decision.decision_id)
        self.assertTrue(accounting["market_midpoint_diagnostic"]["diagnostic_only_not_optimization"])

    def test_fixed_accounting_replay_includes_weather_taker_fees_in_cash_pnl_and_trades(self) -> None:
        decision = _decision("row-1", bucket_type="above")
        frame = _frame(
            decision=decision,
            score=_score(p_yes=0.86, p_side=0.86),
            label=_label("row-1", canonical_label=1),
        )
        run = run_conservative_fv_edge_candidate_replay(
            _ListFrameDataset([frame]),
            limit=1,
            execute_orders=True,
            fee_config=FeeModelConfig(
                fee_model=FEE_MODEL_POLYMARKET_WEATHER_TAKER,
                fee_rate=0.05,
            ),
        )
        accounting = run.report["fixed_strategy_accounting"]
        fill = run.fills[0]
        expected_fee = calculate_fill_fee_usd(
            shares=1.0 / decision.quote.ask,
            price=decision.quote.ask,
            fee_config=FeeModelConfig(
                fee_model=FEE_MODEL_POLYMARKET_WEATHER_TAKER,
                fee_rate=0.05,
            ),
        )
        expected_payout = 1.0 / decision.quote.ask
        expected_net_cost = 1.0 + expected_fee
        expected_net_pnl = expected_payout - expected_net_cost
        self.assertEqual(fill.fee_usd, expected_fee)
        self.assertAlmostEqual(accounting["pnl"]["gross_no_fee_total_cost_usd"], 1.0)
        self.assertAlmostEqual(accounting["pnl"]["total_fees_usd"], expected_fee)
        self.assertAlmostEqual(accounting["pnl"]["total_cost_usd"], expected_net_cost)
        self.assertAlmostEqual(
            accounting["pnl"]["gross_no_fee_realized_pnl_usd"],
            expected_payout - 1.0,
        )
        self.assertAlmostEqual(accounting["pnl"]["realized_pnl_usd"], expected_net_pnl)
        self.assertAlmostEqual(
            accounting["pnl"]["net_fee_adjusted_roi_on_deployed_capital"],
            expected_net_pnl / expected_net_cost,
        )
        self.assertAlmostEqual(
            run.report["accounting_summary"]["ending_cash_usd"],
            100.0 + expected_net_pnl,
        )
        trade = accounting["all_trades"][0]
        self.assertEqual(trade["fee_usd"], expected_fee)
        self.assertEqual(trade["fee_model"], FEE_MODEL_POLYMARKET_WEATHER_TAKER)
        self.assertAlmostEqual(trade["cost_usd"], expected_net_cost)
        self.assertAlmostEqual(trade["net_fee_adjusted_realized_pnl_usd"], expected_net_pnl)

    def test_accounting_report_defaults_to_weather_taker_fee_and_persists_full_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            decision_dir = root / "work" / "data" / "backtest"
            decision_dir.mkdir(parents=True)
            decision_db = decision_dir / "daily_weather_decision_artifact.sqlite"
            with sqlite3.connect(decision_db) as conn:
                _create_enriched_table(conn)
                _insert_enriched_row(
                    conn,
                    "good-row",
                    quote_timestamp="2026-01-01T11:59:00+00:00",
                )
                conn.execute(
                    "UPDATE decision_snapshots_current_enriched SET bucket_type = 'above', canonical_bucket_type = 'above'"
                )
                conn.commit()
            result = write_conservative_fv_edge_accounting_replay_report(
                work_root=root,
                output_dir=root / "work" / "reports" / "backtesting_engine",
                limit=1,
                weather_runner="fixture",
            )
            payload = json.loads(Path(result["json_path"]).read_text(encoding="utf-8"))
            ledger_path = Path(result["trade_ledger_path"])
            self.assertEqual(
                payload["fixed_strategy_accounting"]["execution_policy"]["fee_model"]["fee_model"],
                DEFAULT_STRATEGY_REPLAY_FEE_MODEL,
            )
            self.assertEqual(
                payload["fixed_strategy_accounting"]["execution_policy"]["fee_model"]["fee_rate"],
                DEFAULT_STRATEGY_REPLAY_FEE_RATE,
            )
            self.assertTrue(payload["full_trade_ledger"]["persisted"])
            self.assertTrue(payload["full_trade_ledger"]["all_fills_included"])
            self.assertTrue(ledger_path.exists())
            with ledger_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["fee_model"], FEE_MODEL_POLYMARKET_WEATHER_TAKER)

    def test_fixed_accounting_replay_reports_skipped_fill_without_fake_pnl(self) -> None:
        decision = _decision(
            "row-1",
            bucket_type="above",
            quote=_quote(ask=0.44, ask_size=1.0),
        )
        frame = _frame(decision=decision, score=_score(p_yes=0.86, p_side=0.86))
        run = run_conservative_fv_edge_candidate_replay(
            _ListFrameDataset([frame]),
            limit=1,
            execute_orders=True,
        )
        accounting = run.report["fixed_strategy_accounting"]
        self.assertEqual(run.report["decision_counts"]["order_intents"], 1)
        self.assertEqual(run.report["decision_counts"]["fills"], 0)
        self.assertEqual(accounting["fill_counts"]["attempted_fills"], 1)
        self.assertEqual(accounting["fill_counts"]["skipped_fills"], 1)
        self.assertEqual(
            accounting["fill_counts"]["fill_status_counts"],
            {"insufficient_visible_size": 1},
        )
        self.assertEqual(accounting["pnl"]["total_cost_usd"], 0)
        self.assertEqual(accounting["pnl"]["realized_pnl_usd"], 0)

    def test_conservative_fv_edge_replay_skips_missing_shared_score_without_fills(self) -> None:
        decision = _decision("row-1", bucket_type="above")
        frame = _frame(decision=decision, score=None)
        frame = type(frame)(
            decision=frame.decision,
            canonical_label=frame.canonical_label,
            score=None,
            skip_reasons=PointInTimePolicy(require_shared_score=True).skip_reasons(
                frame.decision,
                frame.canonical_label,
                None,
            ),
        )
        run = run_conservative_fv_edge_candidate_replay(_ListFrameDataset([frame]), limit=1)
        self.assertEqual(run.report["decision_counts"]["fills"], 0)
        self.assertEqual(run.report["skip_reason_counts"], {SkipReason.MISSING_SHARED_SCORE.value: 1})
        self.assertEqual(run.report["candidate_replay_blockers"], [SkipReason.MISSING_SHARED_SCORE.value])
        self.assertFalse(run.report["anti_overfit_readiness"]["passes_anti_overfit_reporting_gates"])

    def test_weather_model_context_firewall_omits_prices_labels_and_future_weather(self) -> None:
        future_observed = ObservedHighSnapshot(
            observed_high_f=99.0,
            observed_high_timestamp_utc=datetime(2026, 1, 2, 12, 30, tzinfo=timezone.utc),
            observation_source="future_station",
        )
        decision = replace(
            _decision("row-1", bucket_type="above"),
            observed_high=future_observed,
            resolved_outcome=1,
            realized_temperature_2m_max_f=99.0,
        )
        context = WeatherModelDecisionContext.from_decision(decision)
        payload = context.to_hash_payload()
        for forbidden in (
            "bid",
            "ask",
            "canonical_label",
            "resolved_outcome",
            "realized_temperature_2m_max_f",
        ):
            self.assertNotIn(forbidden, payload)
        self.assertIsNone(context.observed_high_so_far_f)
        self.assertEqual(context.observed_high_omitted_reason, "future_observed_high_not_model_visible")

        leaky_decision = replace(
            decision,
            forecast=ForecastSnapshot(
                run_time_utc=datetime(2026, 1, 2, 6, tzinfo=timezone.utc),
                source_name="synthetic",
                temperature_2m_max_f=72.0,
                features={"actual_high_f": 75.0},
            ),
        )
        with self.assertRaises(ValueError):
            DeterministicFixtureWeatherModelRunner().predict(
                WeatherModelDecisionContext.from_decision(leaky_decision)
            )

    def test_fixture_runner_and_score_cache_are_deterministic_and_hash_guarded(self) -> None:
        decision = _decision("row-1", bucket_type="above")
        context = WeatherModelDecisionContext.from_decision(decision)
        runner = DeterministicFixtureWeatherModelRunner()
        first = runner.predict(context)
        second = runner.predict(context)
        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        self.assertEqual(first.p_yes, second.p_yes)
        self.assertEqual(context.input_hash, WeatherModelDecisionContext.from_decision(decision).input_hash)

        score = first.to_shared_score(
            side=decision.side,
            observed_high=None,
            parser_confidence=decision.parser_confidence,
            settlement_unit="F",
            settlement_precision="whole_F",
        )
        cache = ReplayGeneratedScoreCache()
        record = cache.add(context=context, prediction=first, score=score)
        self.assertTrue(cache.is_reusable(record, context=context, runner_metadata=runner.metadata()))
        changed_runner = DeterministicFixtureWeatherModelRunner(probability_stdev=0.02)
        self.assertFalse(
            cache.is_reusable(record, context=context, runner_metadata=changed_runner.metadata())
        )

    def test_score_cache_can_stream_compressed_without_retaining_records(self) -> None:
        decision = _decision("row-1", bucket_type="above")
        context = WeatherModelDecisionContext.from_decision(decision)
        runner = DeterministicFixtureWeatherModelRunner()
        prediction = runner.predict(context)
        self.assertIsNotNone(prediction)
        assert prediction is not None
        score = prediction.to_shared_score(
            side=decision.side,
            observed_high=None,
            parser_confidence=decision.parser_confidence,
            settlement_unit="F",
            settlement_precision="whole_F",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scores.jsonl.gz"
            cache = ReplayGeneratedScoreCache(
                stream_path=path,
                compress=True,
                keep_records=False,
            )
            record = cache.add(context=context, prediction=prediction, score=score)
            self.assertEqual(cache.records, [])
            self.assertTrue(cache.is_reusable(record, context=context, runner_metadata=runner.metadata()))
            cache.write_jsonl(path)
            summary = cache.report_summary(artifact_path=path)
            self.assertEqual(summary["record_count"], 1)
            self.assertEqual(summary["artifact_compression"], "gzip")
            self.assertTrue(summary["streaming_write"])
            self.assertFalse(summary["records_retained_in_memory"])
            self.assertEqual(summary["input_hashes_recorded"], 1)
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                rows = [json.loads(line) for line in handle]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["decision_id"], context.decision_id)
            self.assertEqual(rows[0]["input_hash"], context.input_hash)

    def test_candidate_replay_uses_fixture_runner_for_intents_without_execution(self) -> None:
        decision = _decision("row-1", bucket_type="above")
        base_frame = _frame(decision=decision, score=_score(p_yes=0.86, p_side=0.86))
        frame = type(base_frame)(
            decision=base_frame.decision,
            canonical_label=base_frame.canonical_label,
            score=None,
            skip_reasons=PointInTimePolicy(require_shared_score=True).skip_reasons(
                base_frame.decision,
                base_frame.canonical_label,
                None,
            ),
        )
        cache = ReplayGeneratedScoreCache()
        run = run_conservative_fv_edge_candidate_replay(
            _ListFrameDataset([frame]),
            limit=1,
            model_runner=DeterministicFixtureWeatherModelRunner(),
            score_cache=cache,
            execute_orders=False,
        )
        self.assertEqual(run.report["decision_counts"]["order_intents"], 1)
        self.assertEqual(run.report["decision_counts"]["fills"], 0)
        self.assertEqual(run.fills, ())
        self.assertEqual(run.report["weather_model_runner"]["model_outputs_generated"], 1)
        self.assertEqual(run.report["replay_generated_score_cache"]["record_count"], 1)
        self.assertFalse(run.report["accounting_summary"]["economic_strategy_results_performed"])
        self.assertEqual(run.report["quote_source_audit"]["frames_with_quote_source_cache_path"], 1)

    def test_candidate_replay_reports_missing_model_output_skip(self) -> None:
        class NullWeatherModelRunner:
            def metadata(self):
                return {
                    "runner_name": "NullWeatherModelRunner",
                    "model_version": "null",
                    "model_code_hash": "null-code",
                    "model_config_hash": "null-config",
                    "fixture_only": "true",
                }

            def predict(self, context: WeatherModelDecisionContext):
                return None

        decision = _decision("row-1", bucket_type="above")
        base_frame = _frame(decision=decision, score=_score())
        frame = type(base_frame)(
            decision=base_frame.decision,
            canonical_label=base_frame.canonical_label,
            score=None,
            skip_reasons=PointInTimePolicy(require_shared_score=True).skip_reasons(
                base_frame.decision,
                base_frame.canonical_label,
                None,
            ),
        )
        run = run_conservative_fv_edge_candidate_replay(
            _ListFrameDataset([frame]),
            limit=1,
            model_runner=NullWeatherModelRunner(),
        )
        self.assertEqual(run.report["decision_counts"]["order_intents"], 0)
        self.assertEqual(run.report["weather_model_runner"]["missing_model_outputs"], 1)
        self.assertIn(SkipReason.MISSING_MODEL_OUTPUT.value, run.report["skip_reason_counts"])
        self.assertIn(SkipReason.MISSING_SHARED_SCORE.value, run.report["skip_reason_counts"])

    def test_candidate_replay_reports_research_runner_without_fixture_flag(self) -> None:
        class ResearchOnlyWeatherModelRunner:
            def metadata(self):
                return {
                    "runner_name": "ResearchOnlyWeatherModelRunner",
                    "model_version": "research-only-test",
                    "model_code_hash": "research-code",
                    "model_config_hash": "research-config",
                    "fixture_only": "false",
                    "promotion_status": "research_only_not_promoted",
                }

            def predict(self, context: WeatherModelDecisionContext):
                return WeatherModelPrediction(
                    p_yes=0.99,
                    model_agreement=1.0,
                    model_count=4,
                    probability_stdev=0.01,
                    per_source_probabilities={"research_stub": 0.99},
                    remaining_exceed_probability=0.99,
                    observed_high_aware_confidence=None,
                    model_version="research-only-test",
                    model_code_hash="research-code",
                    model_config_hash="research-config",
                    feature_provenance=context.feature_provenance,
                    input_source_paths=context.input_source_paths,
                    forecast_as_of_utc=context.forecast_run_time_utc,
                    prediction_timestamp_utc=datetime.now(timezone.utc),
                    prediction_as_of_utc=context.decision_time_utc,
                    metadata={"input_hash": context.input_hash, "research_only": True},
                )

        decision = _decision("row-1", bucket_type="above")
        base_frame = _frame(decision=decision, score=_score())
        frame = type(base_frame)(
            decision=base_frame.decision,
            canonical_label=base_frame.canonical_label,
            score=None,
            skip_reasons=PointInTimePolicy(require_shared_score=True).skip_reasons(
                base_frame.decision,
                base_frame.canonical_label,
                None,
            ),
        )
        run = run_conservative_fv_edge_candidate_replay(
            _ListFrameDataset([frame]),
            limit=1,
            model_runner=ResearchOnlyWeatherModelRunner(),
            execute_orders=False,
        )
        runner_report = run.report["weather_model_runner"]
        self.assertFalse(runner_report["fixture_runner_used"])
        self.assertTrue(runner_report["real_weather_model_runner_plugged_in"])
        self.assertTrue(runner_report["research_only_not_promoted"])
        self.assertEqual(run.report["decision_counts"]["order_intents"], 1)
        self.assertEqual(run.report["decision_counts"]["fills"], 0)
        self.assertIn(
            "research_only_weather_model_not_promoted",
            run.report["profitability_claim_blockers"],
        )

    def test_candidate_weather_runner_slug_preserves_fixture_default_and_research_alias(self) -> None:
        self.assertEqual(_weather_runner_slug("fixture"), "fixture")
        self.assertEqual(_weather_runner_slug("observed-high-guarded"), "observed-high-guarded")
        self.assertEqual(
            _weather_runner_slug("observed_high_guarded_shrinkage"),
            "observed-high-guarded",
        )

    def test_conservative_fv_edge_throughput_diagnostic_reports_funnel_and_frontier(self) -> None:
        decision = _decision("row-1", bucket_type="above")
        frame = _frame(
            decision=decision,
            score=_score(p_yes=0.86, p_side=0.86),
            label=_label("row-1", canonical_label=1),
        )
        report = run_conservative_fv_edge_throughput_diagnostic(
            _ListFrameDataset([frame]),
            limit=1,
            fee_config=FeeModelConfig(
                fee_model=FEE_MODEL_POLYMARKET_WEATHER_TAKER,
                fee_rate=0.05,
            ),
        )
        self.assertTrue(report["diagnostic_only"])
        self.assertEqual(report["total_official_rows"], 1)
        self.assertEqual(report["weather_model_runner"]["model_outputs_generated"], 0)
        self.assertEqual(report["weather_model_runner"]["missing_model_outputs"], 0)
        self.assertEqual(report["gate_funnel"]["cumulative_counts"]["rows_with_model_output"], 1)
        self.assertEqual(report["variants"]["baseline"]["fills"], 1)
        self.assertIn("loosen_min_fair_value", {row["variant_key"] for row in report["one_gate_loosen_tests"]})
        self.assertIn("frontier_combined_moderate", report["trade_frequency_frontier"])
        self.assertEqual(report["execution_realism"]["order_latency_model"], "not_modeled")
        self.assertFalse(report["boundary"]["parameter_optimization_performed"] if "boundary" in report else False)

    def test_universal_gates_require_same_day_observed_high_fields(self) -> None:
        frame = _frame(score=_score(observed_high=None))
        result = evaluate_universal_gates(frame, family=StrategyFamily.SAME_DAY_OBSERVED_HIGH)
        self.assertFalse(result.passed)
        self.assertIn(SkipReason.MISSING_OBSERVED_HIGH_SNAPSHOT, result.skip_reasons)

    def test_universal_gates_accept_late_day_heat_window_fixture(self) -> None:
        frame = _frame(score=_score_with_observed_high_and_heat_window())
        result = evaluate_universal_gates(
            frame,
            family=StrategyFamily.LATE_DAY_HEAT_WINDOW,
            config=UniversalGateConfig(bounded_probability_stdev_max=0.20),
        )
        self.assertTrue(result.passed)

    def test_universal_gates_flag_market_and_model_gate_failures(self) -> None:
        decision = _decision("row-1", quote=_quote(ask=0.99, spread=0.20, ask_size=1.0))
        score = _score(model_count=1, model_agreement=0.50)
        frame = _frame(decision=decision, score=score)
        result = evaluate_universal_gates(
            frame,
            family=StrategyFamily.FORECAST_ONLY,
            config=UniversalGateConfig(nominal_order_usd_for_size_check=10.0),
        )
        self.assertIn(SkipReason.SPREAD_TOO_WIDE, result.skip_reasons)
        self.assertIn(SkipReason.ASK_ABOVE_MAXIMUM, result.skip_reasons)
        self.assertIn(SkipReason.INSUFFICIENT_VISIBLE_SIZE, result.skip_reasons)
        self.assertIn(SkipReason.MODEL_COUNT_LOW, result.skip_reasons)
        self.assertIn(SkipReason.MODEL_AGREEMENT_LOW, result.skip_reasons)

    def test_portfolio_cap_validator_flags_city_date_and_primary_exposure_caps(self) -> None:
        caps = PortfolioCaps(bankroll_usd=100.0, per_city_date_total_fraction=0.15)
        validator = PortfolioCapValidator(caps)
        existing = [
            ExposureIntent(
                token_id="token-a",
                side="YES",
                city="London, GB",
                market_date_local="2026-01-02",
                region="Europe",
                settlement_source_family="Wunderground",
                decision_hour_utc=12,
                notional_usd=10.0,
            )
        ]
        proposed = ExposureIntent(
            token_id="token-b",
            side="NO",
            city="London, GB",
            market_date_local="2026-01-02",
            region="Europe",
            settlement_source_family="Wunderground",
            decision_hour_utc=12,
            notional_usd=10.0,
        )
        result = validator.validate(existing, proposed)
        self.assertFalse(result.passed)
        self.assertIn(SkipReason.CITY_DATE_CAP_EXCEEDED, result.skip_reasons)
        self.assertIn(SkipReason.PRIMARY_CITY_DATE_EXPOSURE_EXISTS, result.skip_reasons)

    def test_portfolio_cap_validator_flags_token_side_and_primary_across_hours(self) -> None:
        caps = PortfolioCaps(
            bankroll_usd=100.0,
            per_token_side_total_fraction=0.05,
            max_primary_exposures_per_city_date_per_run=1,
        )
        validator = PortfolioCapValidator(caps)
        existing = [
            ExposureIntent(
                token_id="token-a",
                side="YES",
                city="London, GB",
                market_date_local="2026-01-02",
                region="Europe",
                settlement_source_family="Wunderground",
                decision_hour_utc=10,
                notional_usd=4.0,
            )
        ]
        proposed = ExposureIntent(
            token_id="token-a",
            side="YES",
            city="London, GB",
            market_date_local="2026-01-02",
            region="Europe",
            settlement_source_family="Wunderground",
            decision_hour_utc=12,
            notional_usd=2.0,
        )
        result = validator.validate(existing, proposed)
        self.assertFalse(result.passed)
        self.assertIn(SkipReason.TOKEN_SIDE_CAP_EXCEEDED, result.skip_reasons)
        self.assertIn(SkipReason.PRIMARY_CITY_DATE_EXPOSURE_EXISTS, result.skip_reasons)

    def test_portfolio_state_releases_expired_exposures_before_new_cap_check(self) -> None:
        caps = PortfolioCaps(
            bankroll_usd=100.0,
            per_settlement_source_total_fraction=0.015,
        )
        state = PortfolioState(caps)
        first = ExposureIntent(
            token_id="token-a",
            side="NO",
            city="London, GB",
            market_date_local="2026-01-02",
            region="Europe",
            settlement_source_family="Wunderground",
            decision_hour_utc=12,
            notional_usd=1.0,
            release_time_utc=datetime(2026, 1, 3, 12, tzinfo=timezone.utc),
        )
        second = ExposureIntent(
            token_id="token-b",
            side="NO",
            city="London, GB",
            market_date_local="2026-01-03",
            region="Europe",
            settlement_source_family="Wunderground",
            decision_hour_utc=12,
            notional_usd=1.0,
            release_time_utc=datetime(2026, 1, 4, 12, tzinfo=timezone.utc),
        )
        self.assertTrue(state.reserve(first).passed)
        self.assertFalse(state.check(second).passed)

        released = state.release_expired(datetime(2026, 1, 3, 12, 1, tzinfo=timezone.utc))

        self.assertEqual(released, 1)
        self.assertTrue(state.reserve(second).passed)

    def test_portfolio_state_blocks_joint_total_reserved_bankroll_breach(self) -> None:
        caps = PortfolioCaps(
            bankroll_usd=10.0,
            total_reserved_bankroll_fraction=1.0,
            window_bankroll_fraction=2.0,
            per_position_window_fraction=1.0,
            per_city_date_total_fraction=2.0,
            per_region_total_fraction=2.0,
            per_token_side_total_fraction=2.0,
            per_settlement_source_total_fraction=2.0,
            max_primary_exposures_per_city_date_per_run=10,
        )
        state = PortfolioState(caps)
        release_time = datetime(2026, 1, 3, 12, tzinfo=timezone.utc)

        orders = [
            ExposureIntent(
                token_id=f"token-{index}",
                side="YES",
                city=f"City {index}",
                market_date_local="2026-01-02",
                region="Region",
                settlement_source_family="Wunderground",
                decision_hour_utc=12,
                notional_usd=4.0,
                release_time_utc=release_time,
            )
            for index in range(3)
        ]

        self.assertTrue(state.reserve(orders[0]).passed)
        self.assertTrue(state.reserve(orders[1]).passed)
        third_result = state.reserve(orders[2])

        self.assertFalse(third_result.passed)
        self.assertEqual(third_result.skip_reasons, (SkipReason.TOTAL_RESERVED_CAP_EXCEEDED,))
        self.assertEqual(len(state.exposures), 2)
        self.assertAlmostEqual(state.summary()["peak_concurrent_reserved_capital_usd"], 8.0)
        self.assertAlmostEqual(state.summary()["minimum_cash_balance_usd"], 2.0)
        self.assertEqual(
            state.summary()["rejected_orders_by_cap_reason"],
            {SkipReason.TOTAL_RESERVED_CAP_EXCEEDED.value: 1},
        )
        self.assertEqual(state.summary()["insolvency_attempts"], 1)

    def test_portfolio_state_holds_cash_until_release_time(self) -> None:
        caps = PortfolioCaps(
            bankroll_usd=10.0,
            total_reserved_bankroll_fraction=1.0,
            window_bankroll_fraction=2.0,
            per_position_window_fraction=1.0,
            per_city_date_total_fraction=2.0,
            per_region_total_fraction=2.0,
            per_token_side_total_fraction=2.0,
            per_settlement_source_total_fraction=2.0,
            max_primary_exposures_per_city_date_per_run=10,
        )
        state = PortfolioState(caps)
        first = ExposureIntent(
            token_id="token-a",
            side="NO",
            city="London, GB",
            market_date_local="2026-01-02",
            region="Europe",
            settlement_source_family="Wunderground",
            decision_hour_utc=12,
            notional_usd=7.0,
            release_time_utc=datetime(2026, 1, 3, 12, tzinfo=timezone.utc),
        )
        second = replace(
            first,
            token_id="token-b",
            city="Paris, FR",
            notional_usd=4.0,
            release_time_utc=datetime(2026, 1, 4, 12, tzinfo=timezone.utc),
        )

        self.assertTrue(state.reserve(first).passed)
        self.assertFalse(state.reserve(second).passed)
        self.assertEqual(state.release_expired(datetime(2026, 1, 3, 11, 59, tzinfo=timezone.utc)), 0)
        self.assertFalse(state.reserve(second).passed)
        self.assertEqual(state.release_expired(datetime(2026, 1, 3, 12, 1, tzinfo=timezone.utc)), 1)
        self.assertTrue(state.reserve(second).passed)
        self.assertAlmostEqual(state.summary()["minimum_cash_balance_usd"], 3.0)

    def test_candidate_replay_batch_orders_are_checked_against_updated_cash_state(self) -> None:
        class BatchStrategy:
            name = "batch_strategy"

            def evaluate(self, snapshot: StrategyInput) -> StrategyDecision:
                return StrategyDecision(
                    orders=tuple(
                        OrderIntent(
                            strategy_name=self.name,
                            decision_id=snapshot.decision_id,
                            token_id=f"token-{index}",
                            side=snapshot.side,
                            action=OrderAction.BUY,
                            requested_notional_usd=4.0,
                            limit_price=snapshot.quote.ask,
                            reason="synthetic batch order",
                        )
                        for index in range(3)
                    )
                )

        decision = _decision("row-batch")
        frame = _frame(
            decision=decision,
            score=_score_with_observed_high_and_heat_window(),
            label=_label("row-batch"),
        )
        caps = PortfolioCaps(
            bankroll_usd=10.0,
            total_reserved_bankroll_fraction=1.0,
            window_bankroll_fraction=2.0,
            per_position_window_fraction=1.0,
            per_city_date_total_fraction=2.0,
            per_region_total_fraction=2.0,
            per_token_side_total_fraction=2.0,
            per_settlement_source_total_fraction=2.0,
            max_primary_exposures_per_city_date_per_run=10,
        )

        run = run_candidate_replay(
            _ListFrameDataset([frame]),
            BatchStrategy(),
            strategy_config=ConservativeFVEdgeConfig(),
            starting_cash_usd=10.0,
            execute_orders=False,
            portfolio_caps=caps,
        )

        self.assertEqual(len(run.order_intents), 2)
        portfolio = run.report["portfolio_state"]
        self.assertAlmostEqual(portfolio["peak_concurrent_reserved_capital_usd"], 8.0)
        self.assertAlmostEqual(portfolio["minimum_cash_balance_usd"], 2.0)
        self.assertEqual(
            portfolio["rejected_orders_by_cap_reason"],
            {SkipReason.TOTAL_RESERVED_CAP_EXCEEDED.value: 1},
        )
        self.assertEqual(portfolio["insolvency_attempts"], 1)

    def test_gen2_validation_is_deterministic_for_identical_synthetic_inputs(self) -> None:
        decision = replace(
            _decision(
                "row-gen2-1",
                bucket_type="above",
                quote=_quote(bid=0.80, ask=0.82, spread=0.02, ask_size=100.0),
            ),
            side="NO",
            token_id="token-no",
        )
        frame = _frame(
            decision=decision,
            score=_score(side="NO", p_yes=0.04, p_side=0.96, model_agreement=1.0),
            label=_label(decision.universe_row_id, canonical_label=0),
        )
        dataset = _ListFrameDataset([frame])

        first_payload, _ = run_gen2_strategy_validation(
            dataset,
            fee_config=FeeModelConfig(fee_model=FEE_MODEL_POLYMARKET_WEATHER_TAKER, fee_rate=0.05),
        )
        second_payload, _ = run_gen2_strategy_validation(
            dataset,
            fee_config=FeeModelConfig(fee_model=FEE_MODEL_POLYMARKET_WEATHER_TAKER, fee_rate=0.05),
        )

        self.assertEqual(first_payload["ranked_variants"], second_payload["ranked_variants"])
        self.assertEqual(first_payload["criteria_results"], second_payload["criteria_results"])
        self.assertEqual(first_payload["variants"], second_payload["variants"])
        self.assertEqual(first_payload["overall_verdict"], second_payload["overall_verdict"])

    def test_gen2_fixed_2500_row_slice_writes_byte_identical_ledgers(self) -> None:
        decision_time = datetime(2026, 1, 2, 12, tzinfo=timezone.utc)
        frames = []
        for idx in range(2500):
            quote = _quote(
                timestamp_utc=decision_time - timedelta(minutes=1),
                bid=0.80,
                ask=0.82,
                spread=0.02,
                ask_size=100.0,
            )
            forecast = ForecastSnapshot(
                run_time_utc=decision_time - timedelta(hours=6),
                source_name="synthetic_forecast",
                temperature_2m_max_f=70.0,
                features={"row_index": float(idx)},
            )
            decision = replace(
                _decision(
                    f"row-fixed-slice-{idx}",
                    market_id=f"market-fixed-slice-{idx}",
                    bucket_type="above",
                    quote=quote,
                    forecast=forecast,
                ),
                side="NO",
                token_id=f"token-no-{idx}",
                yes_token_id=f"token-yes-{idx}",
                no_token_id=f"token-no-{idx}",
                decision_time_utc=decision_time,
                decision_hour_utc=12,
            )
            frames.append(
                _frame(
                    decision=decision,
                    score=_score(
                        side="NO",
                        p_yes=0.04,
                        p_side=0.96,
                        model_agreement=1.0,
                        observed_high=None,
                    ),
                    label=_label(decision.universe_row_id, canonical_label=0),
                )
            )

        def ledger_and_ranked_bytes(output_dir: Path) -> tuple[bytes, bytes]:
            payload, states = run_gen2_strategy_validation(
                _ListFrameDataset(frames),
                fee_config=FeeModelConfig(
                    fee_model=FEE_MODEL_POLYMARKET_WEATHER_TAKER,
                    fee_rate=0.05,
                ),
            )
            paths = _write_gen2_strategy_validation_csvs(
                states=states,
                payload=payload,
                output_dir=output_dir,
                timestamp="fixed-slice",
            )
            return (
                Path(paths["trade_ledger"]).read_bytes(),
                Path(paths["ranked_grid_csv"]).read_bytes(),
            )

        with tempfile.TemporaryDirectory() as first_dir, tempfile.TemporaryDirectory() as second_dir:
            first_ledger, first_ranked = ledger_and_ranked_bytes(Path(first_dir))
            second_ledger, second_ranked = ledger_and_ranked_bytes(Path(second_dir))

        self.assertEqual(first_ledger, second_ledger)
        self.assertEqual(first_ranked, second_ranked)

    def test_gen2_reproducibility_audit_reemits_only_after_matching_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ranked = root / "ranked.csv"
            ledger = root / "ledger.csv"
            ranked.write_text("variant_key,net_fee_adjusted_pnl_usd\nvariant-a,1.0\n", encoding="utf-8")
            ledger.write_text("variant_key,stress_scenario,realized_pnl_usd\nvariant-a,base,1.0\n", encoding="utf-8")
            payload = {
                "mode": "gen2_strategy_validation_fee_aware_walk_forward",
                "dataset_id": "six_month_v1",
                "official_trading_replay_sql_predicate": OFFICIAL_TRADING_REPLAY_SQL_PREDICATE,
                "row_limit": 0,
                "total_official_rows": 2,
                "fee_model": {"fee_model": FEE_MODEL_POLYMARKET_WEATHER_TAKER, "fee_rate": 0.05},
                "weather_model_runner": {"metadata": {"model_version": "fixture"}},
                "pre_registered_pass_fail_criteria": {"fixture": True},
                "strategies_tested": ["fixture"],
                "staging_protocol": {"stage": "fixture"},
                "determinism_controls": {"ranking_tie_breakers": ["variant_key_asc"]},
                "code_hashes": {"backtesting_engine_py_sha256": "hash"},
                "replay_generated_score_cache": {"record_count": 0},
                "variants": {
                    "variant-a": {
                        "label": "Variant A",
                        "stage": "fixture",
                        "strategy_family": "fixture",
                        "parameters": {"x": 1},
                        "config": {"y": 2},
                        "portfolio_caps": {"bankroll_usd": 100.0},
                    }
                },
                "overall_verdict": {
                    "status": "suspended_pending_reproducibility",
                    "replay_gate_pass_variants": ["variant-a"],
                    "research_only_variants": [],
                    "rejected_variants": [],
                },
                "csv_artifacts": {
                    "ranked_grid_csv": str(ranked),
                    "trade_ledger": str(ledger),
                },
            }
            run_paths = []
            for idx in range(2):
                run_path = root / f"run-{idx}.json"
                run_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
                run_paths.append(run_path)

            result = write_gen2_reproducibility_audit_report(
                run_json_paths=run_paths,
                output_dir=root,
                required_runs=2,
            )
            audit = json.loads(Path(result["json_path"]).read_text(encoding="utf-8"))

        self.assertEqual(
            audit["reproducibility"]["independent_runs_reproduced_every_cited_result"],
            2,
        )
        self.assertTrue(audit["comparisons"]["ranked_grid_csv_byte_identical"])
        self.assertTrue(audit["comparisons"]["full_trade_ledger_csv_byte_identical"])
        self.assertEqual(audit["re_emitted_overall_verdict"]["status"], "live-candidate")

    def test_fill_simulation_uses_sell_bid_and_partial_fill_policy(self) -> None:
        decision = _decision(
            "row-1",
            quote=_quote(bid=0.40, ask=0.44, spread=0.04, bid_size=5.0, ask_size=120.0),
        )
        order = OrderIntent(
            strategy_name="fixture",
            decision_id=decision.decision_id,
            token_id=decision.token_id,
            side=decision.side,
            action=OrderAction.SELL,
            requested_shares=10.0,
        )
        blocked = simulate_order_fill(order, decision, fill_policy=FillPolicy(allow_partial=False))
        self.assertIsNone(blocked.fill)
        self.assertEqual(blocked.skip_reason, SkipReason.INSUFFICIENT_VISIBLE_SIZE)

        partial = simulate_order_fill(order, decision, fill_policy=FillPolicy(allow_partial=True))
        self.assertIsNotNone(partial.fill)
        self.assertEqual(partial.status, "partial_filled")
        self.assertEqual(partial.fill.price, decision.quote.bid)
        self.assertEqual(partial.fill.shares, 5.0)
        self.assertEqual(partial.price_source, "bid")

    def test_worse_fill_tick_stresses_buy_ask_and_sell_bid(self) -> None:
        quote = _quote(bid=0.40, ask=0.44, spread=0.04)
        buy_stress = _quote_with_worse_fill_tick(quote, action=OrderAction.BUY, tick=0.01)
        sell_stress = _quote_with_worse_fill_tick(quote, action=OrderAction.SELL, tick=0.01)
        self.assertAlmostEqual(buy_stress.ask, 0.45)
        self.assertAlmostEqual(buy_stress.bid, 0.40)
        self.assertAlmostEqual(buy_stress.spread, 0.05)
        self.assertAlmostEqual(sell_stress.bid, 0.39)
        self.assertAlmostEqual(sell_stress.ask, 0.44)
        self.assertAlmostEqual(sell_stress.spread, 0.05)

    def test_simulate_order_fill_can_record_delayed_fill_timestamp(self) -> None:
        decision = _decision("row-1")
        order = OrderIntent(
            strategy_name="fixture",
            decision_id=decision.decision_id,
            token_id=decision.token_id,
            side=decision.side,
            action=OrderAction.BUY,
            requested_notional_usd=1.0,
        )
        delayed_time = datetime(2026, 1, 2, 12, 0, 3, tzinfo=timezone.utc)
        result = simulate_order_fill(order, decision, fill_timestamp_utc=delayed_time)
        self.assertIsNotNone(result.fill)
        self.assertEqual(result.fill.timestamp_utc, delayed_time)

    def test_compact_quote_lookup_returns_first_quote_after_delay(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "compact.sqlite"
            with sqlite3.connect(path) as conn:
                conn.execute(
                    "CREATE TABLE compact_tokens (token_ref INTEGER PRIMARY KEY, token_id TEXT, market_id TEXT, universe_row_id TEXT, side TEXT)"
                )
                conn.execute(
                    "CREATE TABLE compact_sources (source_ref INTEGER PRIMARY KEY, source_payload_id TEXT, source_cache_path TEXT)"
                )
                conn.execute(
                    """
                    CREATE TABLE compact_quote_history (
                        token_ref INTEGER,
                        timestamp_utc TEXT,
                        bid REAL,
                        ask REAL,
                        mid REAL,
                        last_price REAL,
                        spread REAL,
                        bid_size REAL,
                        ask_size REAL,
                        liquidity_depth REAL,
                        source_ref INTEGER,
                        data_quality_flags TEXT
                    )
                    """
                )
                conn.execute(
                    "INSERT INTO compact_tokens(token_ref, token_id, market_id, universe_row_id, side) VALUES (1, 'token-yes', 'market-1', 'row-1', 'YES')"
                )
                conn.execute(
                    "INSERT INTO compact_sources(source_ref, source_payload_id, source_cache_path) VALUES (7, 'payload-delayed', '/tmp/delayed.parquet')"
                )
                conn.executemany(
                    """
                    INSERT INTO compact_quote_history(
                        token_ref, timestamp_utc, bid, ask, mid, last_price, spread,
                        bid_size, ask_size, liquidity_depth, source_ref, data_quality_flags
                    )
                    VALUES (1, ?, 0.40, ?, 0.43, 0.43, 0.04, 100.0, 100.0, 200.0, 7, '["ok"]')
                    """,
                    [
                        ("2026-01-02T12:00:01+00:00", 0.44),
                        ("2026-01-02T12:00:03+00:00", 0.45),
                    ],
                )
                conn.commit()
            with CompactQuoteHistoryLookup(path) as lookup:
                quote = lookup.quote_after(
                    token_id="token-yes",
                    earliest_time_utc=datetime(2026, 1, 2, 12, 0, 2, tzinfo=timezone.utc),
                )
                missing = lookup.quote_after(
                    token_id="token-yes",
                    earliest_time_utc=datetime(2026, 1, 2, 12, 0, 2, tzinfo=timezone.utc),
                    max_wait_seconds=0,
                )
        self.assertIsNotNone(quote)
        self.assertEqual(quote.timestamp_utc, datetime(2026, 1, 2, 12, 0, 3, tzinfo=timezone.utc))
        self.assertEqual(quote.ask, 0.45)
        self.assertEqual(quote.source_payload_id, "payload-delayed")
        self.assertIsNone(missing)

    def test_high_confidence_no_tail_strategy_is_no_only_and_supports_kelly_sizing(self) -> None:
        decision = replace(
            _decision("row-1", bucket_type="above", quote=_quote(bid=0.82, ask=0.84, spread=0.02)),
            side="NO",
            token_id="token-no",
        )
        score = _score(side="NO", p_yes=0.08, p_side=0.92)
        snapshot = _frame(decision=decision, score=score).strategy_input()
        strategy = HighConfidenceNOTailStrategy(
            HighConfidenceNOTailConfig(
                sizing_mode="fractional_kelly",
                sizing_bankroll_usd=100.0,
                kelly_fraction=0.50,
                max_order_notional_usd=10.0,
            )
        )
        result = strategy.evaluate(snapshot)
        self.assertEqual(len(result.orders), 1)
        self.assertEqual(result.orders[0].side, "NO")
        self.assertGreater(result.orders[0].requested_notional_usd, 0.0)
        yes_snapshot = _frame(decision=replace(decision, side="YES", token_id="token-yes"), score=score).strategy_input()
        yes_result = strategy.evaluate(yes_snapshot)
        self.assertEqual(yes_result.orders, ())
        self.assertIn(SkipReason.UNSUPPORTED_SIDE.value, yes_result.metadata["candidate_skip_reasons"])

    def test_old_no_tail_live_forward_shadow_profile_freezes_flat_two_semantics(self) -> None:
        spec = old_no_tail_live_forward_shadow_spec(starting_cash_usd=100.0)
        self.assertEqual(spec.key, OLD_NO_TAIL_LIVE_FORWARD_SHADOW_PROFILE_KEY)
        self.assertEqual(spec.strategy_family, "old_high_confidence_no_tail")
        self.assertEqual(spec.parameters["no_counter_event_cap"], 0.14)
        self.assertEqual(spec.parameters["flat_order_size_usd"], 2.0)
        self.assertEqual(spec.parameters["bankroll_scope"], "window_bankroll")
        self.assertEqual(spec.parameters["bounded_min_price"], 0.70)
        self.assertTrue(spec.parameters["read_only_shadow"])
        self.assertFalse(spec.parameters["real_orders_allowed"])
        self.assertEqual(spec.config_report["sizing_mode"], "flat")
        self.assertEqual(spec.config_report["order_notional_usd"], 2.0)

    def test_old_no_tail_live_forward_shadow_logs_would_trade_without_fills_or_pnl(self) -> None:
        decision = replace(
            _decision("row-1", bucket_type="above", quote=_quote(bid=0.82, ask=0.84, spread=0.02)),
            side="NO",
            token_id="token-no",
        )
        frame = _frame(
            decision=decision,
            score=_score(side="NO", p_yes=0.05, p_side=0.95, model_agreement=1.0),
            label=_label("row-1", canonical_label=0),
        )
        payload = run_old_no_tail_live_forward_shadow_logging(
            _ListFrameDataset([frame]),
            starting_cash_usd=100.0,
            limit=10,
        )
        self.assertEqual(payload["mode"], "read_only_live_forward_shadow_logging")
        self.assertTrue(payload["read_only"])
        self.assertFalse(payload["real_order_placement_allowed"])
        self.assertEqual(payload["real_orders_placed"], 0)
        self.assertFalse(payload["fills_simulated"])
        self.assertFalse(payload["ledger_accounting_performed"])
        self.assertFalse(payload["historical_pnl_numbers_are_expected_performance"])
        self.assertEqual(payload["decision_counts"]["would_trade_signals"], 1)
        self.assertNotIn("pnl_usd", payload)
        event = payload["shadow_events"][0]
        self.assertTrue(event["decision"]["would_trade"])
        self.assertEqual(event["decision"]["would_trade_signal"]["requested_notional_usd"], 2.0)
        self.assertEqual(event["decision"]["would_trade_signal"]["side"], "NO")
        self.assertEqual(payload["pre_trade_gate"]["checked_would_trade_signals"], 1)
        self.assertTrue(payload["pre_trade_gate"]["mandatory_before_future_order_adapter"])
        self.assertFalse(payload["pre_trade_gate"]["order_submission_code_path_invoked"])
        self.assertEqual(payload["pre_trade_gate"]["real_orders_placed"], 0)
        self.assertTrue(event["pre_trade_gate"]["pre_trade_gate_checked"])
        self.assertFalse(event["pre_trade_gate"]["order_submission_code_path_invoked"])
        self.assertEqual(event["pre_trade_gate"]["real_orders_placed"], 0)
        self.assertEqual(event["quote"]["top_of_book_depth"]["ask_size"], 120.0)
        self.assertEqual(event["open_meteo_forecast"]["source_name"], "open_meteo_historical_forecast")
        self.assertEqual(event["model_output"]["p_no"], 0.95)
        self.assertEqual(event["settlement"]["status"], "resolved")
        self.assertEqual(event["settlement"]["payout_for_shadow_side"], 1)

    def test_old_no_tail_live_forward_shadow_logs_no_trade_reasons(self) -> None:
        decision = replace(
            _decision("row-1", bucket_type="above", quote=_quote(bid=0.82, ask=0.84, spread=0.02)),
            side="NO",
            token_id="token-no",
        )
        frame = _frame(
            decision=decision,
            score=_score(side="NO", p_yes=0.20, p_side=0.80, model_agreement=1.0),
        )
        payload = run_old_no_tail_live_forward_shadow_logging(_ListFrameDataset([frame]))
        event = payload["shadow_events"][0]
        self.assertFalse(event["decision"]["would_trade"])
        self.assertIn(
            SkipReason.COUNTER_EVENT_PROBABILITY_HIGH.value,
            event["decision"]["no_trade_reasons"],
        )
        self.assertIn(
            SkipReason.FAIR_VALUE_BELOW_MINIMUM.value,
            event["decision"]["no_trade_reasons"],
        )
        self.assertEqual(payload["decision_counts"]["no_trade_decisions"], 1)

    def test_exit_hold_primitive_exits_at_bid_when_hold_probability_breaks(self) -> None:
        decision = _decision("row-1")
        position = PositionSnapshot(
            token_id=decision.token_id,
            side=decision.side,
            shares=1.0,
            cost_usd=decision.quote.ask,
            last_quote_bid=decision.quote.bid,
            last_quote_ask=decision.quote.ask,
            decision_time_utc=decision.decision_time_utc,
        )
        exit_decision = evaluate_exit_hold_primitive(
            position=position,
            score=_score(p_yes=0.40, p_side=0.40),
            quote=decision.quote,
            hold_min_probability=0.50,
        )
        self.assertEqual(exit_decision.action, ExitHoldAction.EXIT)
        self.assertEqual(exit_decision.limit_price, decision.quote.bid)

    def test_validation_only_engine_reports_score_and_gate_readiness_without_pnl(self) -> None:
        decision = _decision("row-1")
        dataset = ReplayDataset(
            _ListDecisionLoader([decision]),
            _DictLabelStore({"row-1": _label("row-1")}),
            score_store=DictScoreStore({"row-1": _score_with_observed_high_and_heat_window()}),
            point_in_time_policy=PointInTimePolicy(require_shared_score=True),
        )
        run = ValidationOnlyEngine(dataset).validate(families=(StrategyFamily.FORECAST_ONLY,))
        self.assertEqual(run.report["mode"], "validation_only_no_trading_replay_no_pnl")
        self.assertEqual(run.report["score_contract_coverage"]["frames_with_shared_score"], 1)
        self.assertIn("portfolio_cap_readiness", run.report)
        self.assertIn("anti_overfit_readiness", run.report)
        self.assertIn("schema_gaps", run.report)
        self.assertNotIn("pnl_usd", run.report)

    def test_framework_contract_report_writes_validation_only_schema_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = write_framework_contract_report(Path(directory))
            json_path = Path(result["json_path"])
            md_path = Path(result["markdown_path"])
            payload = json_path.read_text(encoding="utf-8")
        self.assertTrue(json_path.name.endswith("-backtesting-framework-contracts.json"))
        self.assertIn("validation_only_report_schema", payload)
        self.assertIn("anti_overfit_validation_schema", payload)
        self.assertIn("no_pnl_statement", VALIDATION_ONLY_REPORT_SCHEMA["required_sections"])
        self.assertIn("anti_overfit_readiness", VALIDATION_ONLY_REPORT_SCHEMA["required_sections"])
        self.assertTrue(md_path.name.endswith("-backtesting-framework-contracts.md"))

    def test_gen2_terminal_no_fill_is_not_committed_before_next_candidate(self) -> None:
        class AlwaysFOKStrategy:
            name = "always_fok"

            def evaluate(self, snapshot: StrategyInput) -> StrategyDecision:
                return StrategyDecision(
                    orders=(
                        OrderIntent(
                            strategy_name=self.name,
                            decision_id=snapshot.decision_id,
                            token_id=snapshot.token_id,
                            side=snapshot.side,
                            action=OrderAction.BUY,
                            requested_notional_usd=2.0,
                            limit_price=snapshot.quote.ask,
                            order_type="FOK",
                            reason="portfolio no-fill lifecycle fixture",
                        ),
                    )
                )

        class DelayedQuoteLookup:
            def __init__(self, quotes: dict[str, QuoteSnapshot]) -> None:
                self.quotes = quotes

            def quote_after(
                self,
                *,
                token_id: str,
                earliest_time_utc: datetime,
                max_wait_seconds: int,
            ) -> QuoteSnapshot | None:
                quote = self.quotes.get(token_id)
                return (
                    replace(quote, timestamp_utc=earliest_time_utc)
                    if quote is not None
                    else None
                )

        caps = PortfolioCaps(
            bankroll_usd=2.0,
            total_reserved_bankroll_fraction=1.0,
            window_bankroll_fraction=2.0,
            per_position_window_fraction=1.0,
            per_city_date_total_fraction=2.0,
            per_region_total_fraction=2.0,
            per_token_side_total_fraction=2.0,
            per_settlement_source_total_fraction=2.0,
            max_primary_exposures_per_city_date_per_run=10,
        )
        spec = Gen2ValidationVariantSpec(
            key="terminal-no-fill-fixture",
            label="Terminal no-fill fixture",
            stage="test",
            strategy_family="test",
            strategy=AlwaysFOKStrategy(),
            config_report={},
            portfolio_caps=caps,
            parameters={},
        )

        for expected_status in (
            "insufficient_visible_size",
            "venue_execution_ineligible",
        ):
            with self.subTest(expected_status=expected_status):
                venue_constrained = expected_status == "venue_execution_ineligible"
                fee_config = (
                    FeeModelConfig(
                        fee_model=FEE_MODEL_POLYMARKET_WEATHER_TAKER,
                        fee_rate=0.05,
                    )
                    if venue_constrained
                    else FeeModelConfig()
                )
                fill_policy = FillPolicy(
                    enforce_venue_constraints=venue_constrained,
                    require_full_book_execution=venue_constrained,
                    allow_historical_resting_ask_tick_policy=venue_constrained,
                )
                state = Gen2ValidationVariantState(
                    spec=spec,
                    starting_cash_usd=2.0,
                    fee_config=fee_config,
                    fill_policy=fill_policy,
                )
                first_time = datetime(2026, 1, 2, 12, tzinfo=timezone.utc)
                second_time = first_time + timedelta(minutes=1)

                def executable_quote(
                    *,
                    timestamp: datetime,
                    ask_size: float,
                    execution_proven: bool,
                ) -> QuoteSnapshot:
                    return replace(
                        _quote(
                            timestamp_utc=timestamp - timedelta(seconds=1),
                            bid=0.78,
                            ask=0.80,
                            spread=0.02,
                            ask_size=ask_size,
                        ),
                        direct_polymarket_resting_ask_provenance=True,
                        raw_compact_decision_passthrough_verified=execution_proven,
                        price_transformation_applied=False,
                    )

                first_quote = executable_quote(
                    timestamp=first_time,
                    ask_size=10.0 if venue_constrained else 1.0,
                    execution_proven=not venue_constrained,
                )
                second_quote = executable_quote(
                    timestamp=second_time,
                    ask_size=10.0,
                    execution_proven=True,
                )
                constraints = (
                    VenueExecutionConstraints(
                        min_order_size_shares=5.0,
                        tick_size=None,
                        fee_rate=0.05,
                        fee_rate_source=OFFICIAL_WEATHER_FEE_SCHEDULE_SOURCE,
                    )
                    if venue_constrained
                    else None
                )

                def replay_frame(
                    *,
                    row_id: str,
                    token_id: str,
                    decision_time: datetime,
                    quote: QuoteSnapshot,
                ):
                    decision = replace(
                        _decision(row_id, bucket_type="above", quote=quote),
                        decision_id=f"decision-{row_id}",
                        universe_row_id=row_id,
                        token_id=token_id,
                        no_token_id=token_id,
                        side="NO",
                        decision_time_utc=decision_time,
                        venue_execution_constraints=constraints,
                    )
                    return _frame(
                        decision=decision,
                        score=_score(
                            side="NO",
                            p_yes=0.04,
                            p_side=0.96,
                            model_agreement=1.0,
                        ),
                        label=_label(row_id, canonical_label=0),
                    )

                first = replay_frame(
                    row_id=f"{expected_status}-first",
                    token_id=f"{expected_status}-token-a",
                    decision_time=first_time,
                    quote=first_quote,
                )
                second = replay_frame(
                    row_id=f"{expected_status}-second",
                    token_id=f"{expected_status}-token-b",
                    decision_time=second_time,
                    quote=second_quote,
                )
                delayed_lookup = DelayedQuoteLookup(
                    {
                        first.decision.token_id: first_quote,
                        second.decision.token_id: second_quote,
                    }
                )
                process_kwargs = {
                    "delayed_quote_lookup": delayed_lookup,
                    "delay_seconds": 2,
                    "delayed_quote_max_wait_seconds": 60,
                    "worse_fill_tick": 0.01,
                }

                state.process_frame(first, **process_kwargs)

                self.assertEqual(state.fill_status_counts[expected_status], 1)
                for portfolio in (
                    state.portfolio_state,
                    state.delayed_portfolio_state,
                    state.worse_fill_portfolio_state,
                ):
                    after_no_fill = portfolio.summary()
                    self.assertEqual(after_no_fill["reserved_exposure_count"], 0)
                    self.assertEqual(after_no_fill["committed_reservations"], 0)
                    self.assertEqual(
                        after_no_fill["no_fill_reservations_not_committed"],
                        1,
                    )
                    self.assertAlmostEqual(after_no_fill["minimum_cash_balance_usd"], 2.0)

                state.process_frame(second, **process_kwargs)

                self.assertEqual(len(state.order_intents), 2)
                self.assertEqual(
                    len(state.ledger.fills),
                    1,
                    dict(state.fill_status_counts),
                )
                self.assertEqual(state.ledger.fills[0].token_id, second.decision.token_id)
                self.assertEqual(state.cap_block_counts, {})
                for portfolio in (
                    state.portfolio_state,
                    state.delayed_portfolio_state,
                    state.worse_fill_portfolio_state,
                ):
                    summary = portfolio.summary()
                    self.assertEqual(summary["reserve_attempts"], 2)
                    self.assertEqual(summary["accepted_reservations"], 2)
                    self.assertEqual(summary["committed_reservations"], 1)
                    self.assertEqual(
                        summary["no_fill_reservations_not_committed"],
                        1,
                    )
                    self.assertEqual(summary["reserved_exposure_count"], 1)
                    self.assertAlmostEqual(summary["reserved_notional_usd"], 2.0)
                    self.assertAlmostEqual(summary["peak_cash_usage_usd"], 2.0)
                    self.assertAlmostEqual(summary["minimum_cash_balance_usd"], 0.0)
                    self.assertEqual(portfolio.exposures[0].token_id, second.decision.token_id)
                self.assertIsNot(state.portfolio_state, state.delayed_portfolio_state)
                self.assertIsNot(state.portfolio_state, state.worse_fill_portfolio_state)

    def test_conservative_reservation_cap_preserves_allocated_cash_and_is_not_insolvency(self) -> None:
        caps = PortfolioCaps(
            bankroll_usd=100.0,
            total_reserved_bankroll_fraction=0.80,
            window_bankroll_fraction=1.0,
            per_position_window_fraction=1.0,
            per_city_date_total_fraction=1.0,
            per_region_total_fraction=1.0,
            per_token_side_total_fraction=1.0,
            per_settlement_source_total_fraction=1.0,
            max_primary_exposures_per_city_date_per_run=100,
        )
        state = PortfolioState(caps)

        accepted = ExposureIntent(
            token_id="accepted-token",
            side="NO",
            city="city-a",
            market_date_local="2026-07-14",
            region="region-a",
            settlement_source_family="source-a",
            decision_hour_utc=12,
            notional_usd=80.0,
        )
        rejected_by_reserve_cap = replace(
            accepted,
            token_id="cap-rejected-token",
            city="city-b",
            notional_usd=1.0,
        )

        self.assertTrue(state.reserve(accepted).passed)
        result = state.evaluate_reservation(rejected_by_reserve_cap)

        self.assertFalse(result.passed)
        self.assertIn(SkipReason.TOTAL_RESERVED_CAP_EXCEEDED, result.skip_reasons)
        self.assertAlmostEqual(
            result.exposures_after["allocated_cash_unreserved_usd"],
            19.0,
        )
        self.assertAlmostEqual(
            result.exposures_after["reservation_capacity_remaining_usd"],
            -1.0,
        )
        summary = state.summary()
        self.assertAlmostEqual(summary["allocated_cash_unreserved_usd"], 20.0)
        self.assertAlmostEqual(
            summary["minimum_allocated_cash_unreserved_usd"],
            20.0,
        )
        self.assertAlmostEqual(summary["reservation_capacity_remaining_usd"], 0.0)
        self.assertEqual(summary["insolvency_attempts"], 0)

    def test_generated_score_clears_only_stale_score_skip_reasons(self) -> None:
        decision = _decision("generated-score-clears-stale-skip", bucket_type="above")
        frame = replace(
            _frame(decision=decision),
            score=None,
            skip_reasons=(
                SkipReason.MISSING_SHARED_SCORE,
                SkipReason.NOT_IN_FROZEN_DECISIVE_CANDIDATE_SET,
            ),
        )

        scored, generated, missing, firewall_violation = _score_frame_for_throughput(
            frame,
            model_runner=DeterministicFixtureWeatherModelRunner(),
            score_cache=ReplayGeneratedScoreCache(),
            point_policy=PointInTimePolicy(require_shared_score=True),
        )

        self.assertIsNotNone(scored.score)
        self.assertTrue(generated)
        self.assertFalse(missing)
        self.assertFalse(firewall_violation)
        self.assertNotIn(SkipReason.MISSING_SHARED_SCORE, scored.skip_reasons)
        self.assertIn(
            SkipReason.NOT_IN_FROZEN_DECISIVE_CANDIDATE_SET,
            scored.skip_reasons,
        )

    def test_decision_loader_preserves_explicit_false_over_conflicting_fallback_fields(
        self,
    ) -> None:
        base_row = {
            "decision_id": "decision-loader-presence",
            "universe_row_id": "row-loader-presence",
            "token_id": "token-no",
            "side": "NO",
            "decision_time_utc": "2026-01-02T12:00:00+00:00",
            "quote_timestamp_utc": "2026-01-02T11:59:00+00:00",
            "bid": 0.81,
            "ask": 0.82,
            "bid_size": 10.0,
            "ask_size": 10.0,
            "quote_source_payload_id": "direct-polymarket-quote",
            "quote_source_cache_path": "raw/polymarket/quotes.jsonl.gz",
            "quote_direct_polymarket_resting_ask_provenance": False,
            "direct_polymarket_resting_ask_provenance": True,
            "quote_raw_compact_decision_passthrough_verified": False,
            "raw_compact_decision_passthrough_verified": True,
            "quote_price_transformation_applied": False,
            "price_transformation_applied": True,
        }

        preferred_false = DecisionSnapshot.from_sqlite_row(base_row)
        preferred_true = DecisionSnapshot.from_sqlite_row(
            {
                **base_row,
                "quote_direct_polymarket_resting_ask_provenance": True,
                "direct_polymarket_resting_ask_provenance": False,
                "quote_raw_compact_decision_passthrough_verified": True,
                "raw_compact_decision_passthrough_verified": False,
                "quote_price_transformation_applied": True,
                "price_transformation_applied": False,
            }
        )

        self.assertIsNotNone(preferred_false.quote)
        self.assertFalse(
            preferred_false.quote.direct_polymarket_resting_ask_provenance
        )
        self.assertFalse(
            preferred_false.quote.raw_compact_decision_passthrough_verified
        )
        self.assertIs(preferred_false.quote.price_transformation_applied, False)
        self.assertTrue(
            preferred_true.quote.direct_polymarket_resting_ask_provenance
        )
        self.assertTrue(
            preferred_true.quote.raw_compact_decision_passthrough_verified
        )
        self.assertIs(preferred_true.quote.price_transformation_applied, True)

    def test_point_in_time_policy_uses_explicit_replay_quote_age(self) -> None:
        decision = _decision(
            "row-quote-age",
            quote_time=datetime(2026, 1, 2, 11, 58, 59, tzinfo=timezone.utc),
        )
        label = _label("row-quote-age")

        strict = PointInTimePolicy(max_quote_staleness_seconds=60).skip_reasons(
            decision,
            label,
        )
        sensitivity = PointInTimePolicy(
            max_quote_staleness_seconds=120
        ).skip_reasons(decision, label)

        self.assertIn(SkipReason.STALE_QUOTE, strict)
        self.assertNotIn(SkipReason.STALE_QUOTE, sensitivity)

    def test_immutable_sqlite_loader_rejects_sidecars_and_leaves_none(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frozen-decision.sqlite"
            with sqlite3.connect(path) as conn:
                _ensure_decision_schema(conn)
                conn.execute(
                    """
                    INSERT INTO decision_artifact_scope(
                        scope_id, target_start_date, target_end_date, price_store,
                        max_staleness_minutes, built_at_utc, row_count
                    )
                    VALUES ('current', '2026-01-01', '2026-01-02', 'compact', 90, 'now', 0)
                    """
                )
                conn.commit()
            wal_path = Path(f"{path}-wal")
            shm_path = Path(f"{path}-shm")
            wal_path.write_bytes(b"ambiguous-uncheckpointed-state")

            with self.assertRaisesRegex(RuntimeError, "WAL/SHM sidecars"):
                list(
                    SQLiteDecisionArtifactLoader(
                        path,
                        immutable=True,
                    ).iter_decisions(limit=1)
                )

            wal_path.unlink()
            rows = list(
                SQLiteDecisionArtifactLoader(path, immutable=True).iter_decisions(
                    limit=1
                )
            )
            self.assertEqual(rows, [])
            self.assertFalse(wal_path.exists())
            self.assertFalse(shm_path.exists())

    def test_observed_high_runner_pins_verified_corpus_snapshot(self) -> None:
        snapshot_id = "a" * 64
        resolution = SimpleNamespace(snapshot_id=snapshot_id)
        runner = MagicMock()
        with patch(
            "daily_weather_pipeline.weather_model_runner.ObservedHighGuardedShrinkageWeatherModelRunner",
            return_value=runner,
        ), patch(
            "daily_weather_pipeline.weather_corpus_snapshot.resolve_corpus_snapshot",
            return_value=resolution,
        ) as resolver:
            built = _build_candidate_weather_model_runner(
                "observed-high-guarded",
                work_root=Path("."),
                corpus_snapshot_id=snapshot_id,
            )

        self.assertIs(built, runner)
        self.assertIs(runner._pinned_corpus_resolution, resolution)
        resolver.assert_called_once()

        with patch(
            "daily_weather_pipeline.weather_model_runner.ObservedHighGuardedShrinkageWeatherModelRunner",
            return_value=MagicMock(),
        ), patch(
            "daily_weather_pipeline.weather_corpus_snapshot.resolve_corpus_snapshot",
            return_value=SimpleNamespace(snapshot_id="b" * 64),
        ):
            with self.assertRaisesRegex(RuntimeError, "does not match"):
                _build_candidate_weather_model_runner(
                    "observed-high-guarded",
                    work_root=Path("."),
                    corpus_snapshot_id=snapshot_id,
                )


class _ListDecisionLoader:
    def __init__(self, decisions: list[DecisionSnapshot]) -> None:
        self.decisions = decisions

    def iter_decisions(self, *, start_date=None, end_date=None, limit=0):
        yield from self.decisions[: limit or None]


class _ListFrameDataset:
    def __init__(self, frames) -> None:
        self.frames = list(frames)

    def iter_frames(self, *, start_date=None, end_date=None, limit=0):
        yield from self.frames[: limit or None]


class _DictLabelStore:
    def __init__(self, labels: dict[str, CanonicalLabel]) -> None:
        self.labels = labels

    def get_for_decision(self, decision: DecisionSnapshot):
        return self.labels.get(decision.universe_row_id)


def _create_enriched_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE decision_snapshots_current_enriched (
            decision_id TEXT,
            universe_row_id TEXT,
            market_id TEXT,
            slug TEXT,
            question TEXT,
            normalized_city TEXT,
            country_region TEXT,
            timezone TEXT,
            market_date_local TEXT,
            bucket_label TEXT,
            bucket_type TEXT,
            threshold_low_f REAL,
            threshold_high_f REAL,
            side TEXT,
            yes_token_id TEXT,
            no_token_id TEXT,
            token_id TEXT,
            decision_time_utc TEXT,
            decision_hour_utc INTEGER,
            lead_days INTEGER,
            quote_store TEXT,
            quote_timestamp_utc TEXT,
            bid REAL,
            ask REAL,
            mid REAL,
            last_price REAL,
            spread REAL,
            bid_size REAL,
            ask_size REAL,
            liquidity_depth REAL,
            quote_source_payload_id TEXT,
            quote_source_cache_path TEXT,
            quote_data_quality_flags TEXT,
            price_stale_seconds REAL,
            forecast_run_time_utc TEXT,
            forecast_source_name TEXT,
            forecast_temperature_2m_max_f REAL,
            forecast_features_json TEXT,
            data_quality_flags TEXT,
            parser_confidence REAL,
            created_at_utc TEXT,
            observed_high_so_far_f REAL,
            observed_high_observed_at_utc TEXT,
            observed_high_source TEXT,
            decision_local_hour REAL,
            observed_high_is_final_at_decision INTEGER,
            observed_high_data_quality_flags TEXT,
            observed_high_missing_reason TEXT,
            observed_high_source_payload_id TEXT,
            observed_high_raw_cache_path TEXT,
            canonical_label INTEGER,
            canonical_label_source TEXT,
            canonical_bucket_type TEXT,
            settlement_unit TEXT,
            settlement_precision TEXT,
            canonical_source_family TEXT,
            canonical_source_station TEXT,
            canonical_source_url TEXT,
            resolution_date_text TEXT,
            resolution_date_conflict INTEGER,
            open_meteo_current_continuous_label INTEGER,
            open_meteo_source_rule_proxy_label INTEGER,
            recommended_action TEXT,
            exclude_from_backtest_replay INTEGER,
            exclude_from_open_meteo_training INTEGER,
            settlement_audit_flags TEXT
        )
        """
    )


def _insert_enriched_row(
    conn: sqlite3.Connection,
    universe_row_id: str,
    *,
    quote_timestamp: str | None,
    exclude_from_backtest_replay: int = 0,
    recommended_action: str = "keep_polymarket_label",
) -> None:
    conn.execute(
        """
        INSERT INTO decision_snapshots_current_enriched (
            decision_id, universe_row_id, market_id, slug, question, normalized_city,
            country_region, timezone, market_date_local, bucket_label, bucket_type,
            threshold_low_f, threshold_high_f, side, yes_token_id, no_token_id,
            token_id, decision_time_utc, decision_hour_utc, lead_days, quote_store,
            quote_timestamp_utc, bid, ask, mid, last_price, spread, bid_size,
            ask_size, liquidity_depth, quote_source_payload_id, quote_source_cache_path,
            quote_data_quality_flags, price_stale_seconds, forecast_run_time_utc,
            forecast_source_name, forecast_temperature_2m_max_f, forecast_features_json,
            data_quality_flags, parser_confidence, created_at_utc, observed_high_so_far_f,
            observed_high_observed_at_utc, observed_high_source, decision_local_hour,
            observed_high_is_final_at_decision, observed_high_data_quality_flags,
            observed_high_missing_reason, observed_high_source_payload_id,
            observed_high_raw_cache_path, canonical_label, canonical_label_source,
            canonical_bucket_type, settlement_unit, settlement_precision,
            canonical_source_family, canonical_source_station, canonical_source_url,
            resolution_date_text, resolution_date_conflict,
            open_meteo_current_continuous_label, open_meteo_source_rule_proxy_label,
            recommended_action, exclude_from_backtest_replay,
            exclude_from_open_meteo_training, settlement_audit_flags
        )
        VALUES (
            ?, ?, 'market-1', 'slug-1', 'Synthetic weather market?', 'London, GB',
            'GB', 'Europe/London', '2026-01-02', '72F', 'exact', 72.0, 72.0,
            'YES', 'token-yes', 'token-no', 'token-yes',
            '2026-01-01T12:00:00+00:00', 12, 1, 'compact', ?, 0.40, 0.44,
            0.42, 0.42, 0.04, 100.0, 120.0, 220.0, 'quote-payload',
            '/tmp/quote.parquet', '["quote_ok"]', 60.0,
            '2026-01-01T06:00:00+00:00', 'open_meteo_historical_forecast',
            72.0, '{"forecast_feature": 1}', '["ok"]', 1.0, 'now', 71.0,
            '2026-01-01T11:30:00+00:00', 'synthetic_station', 12.0, 0,
            '["observed_ok"]', NULL, 'observed-payload', '/tmp/observed.json',
            1, 'polymarket_resolved_outcome_via_telonex_final_token_price',
            'exact', 'F', 'whole_F', 'Wunderground', 'station',
            'https://example.test', '2026-01-02', 0, 0, 0, ?, ?, 0,
            'settlement_ok'
        )
        """,
        (
            f"decision-{universe_row_id}",
            universe_row_id,
            quote_timestamp,
            recommended_action,
            exclude_from_backtest_replay,
        ),
    )


def _decision(
    universe_row_id: str,
    *,
    market_id: str = "market-1",
    bucket_type: str = "exact",
    quote_time: datetime = datetime(2026, 1, 2, 11, 59, tzinfo=timezone.utc),
    quote: QuoteSnapshot | None = None,
    forecast: ForecastSnapshot | None = ForecastSnapshot(
        run_time_utc=datetime(2026, 1, 2, 6, tzinfo=timezone.utc),
        source_name="open_meteo_historical_forecast",
        temperature_2m_max_f=72.0,
        features={"forecast_feature": 1.0},
    ),
) -> DecisionSnapshot:
    quote = quote or _quote(timestamp_utc=quote_time)
    return DecisionSnapshot(
        decision_id=f"decision-{universe_row_id}",
        universe_row_id=universe_row_id,
        market_id=market_id,
        slug="slug-1",
        question="Synthetic weather market?",
        normalized_city="London, GB",
        country_region="GB",
        timezone="Europe/London",
        market_date_local="2026-01-02",
        bucket_label="72F",
        bucket_type=bucket_type,
        threshold_low_f=72.0,
        threshold_high_f=72.0,
        side="YES",
        yes_token_id="token-yes",
        no_token_id="token-no",
        token_id="token-yes",
        decision_time_utc=datetime(2026, 1, 2, 12, tzinfo=timezone.utc),
        decision_hour_utc=12,
        lead_days=1,
        quote_store="compact",
        quote=quote,
        forecast=forecast,
        parser_confidence=1.0,
    )


def _quote(
    *,
    timestamp_utc: datetime = datetime(2026, 1, 2, 11, 59, tzinfo=timezone.utc),
    bid: float = 0.40,
    ask: float = 0.44,
    spread: float = 0.04,
    bid_size: float = 100.0,
    ask_size: float = 120.0,
) -> QuoteSnapshot:
    return QuoteSnapshot(
        timestamp_utc=timestamp_utc,
        bid=bid,
        ask=ask,
        mid=(bid + ask) / 2.0,
        last_price=(bid + ask) / 2.0,
        spread=spread,
        bid_size=bid_size,
        ask_size=ask_size,
        liquidity_depth=100.0 + ask_size,
        source_payload_id="payload-1",
        source_cache_path="/tmp/raw.parquet",
        data_quality_flags=(),
    )


def _score(
    *,
    side: str = "YES",
    p_yes: float = 0.80,
    p_side: float = 0.80,
    model_count: int = 4,
    model_agreement: float = 1.0,
    observed_high: ObservedHighSnapshot | None = ObservedHighSnapshot(
        observed_high_f=71.0,
        observed_high_timestamp_utc=datetime(2026, 1, 2, 11, 30, tzinfo=timezone.utc),
        observation_source="synthetic_station",
        source_match_confidence=0.95,
    ),
    heat_window: HeatWindowSnapshot | None = None,
) -> SharedScoreContract:
    return SharedScoreContract(
        p_yes=p_yes,
        side=side,
        p_side=p_side,
        model_agreement=model_agreement,
        model_count=model_count,
        probability_stdev=0.05,
        per_source_probabilities={"synthetic_source": p_yes},
        remaining_exceed_probability=0.10,
        observed_high_aware_confidence=0.90 if observed_high else None,
        observed_high=observed_high,
        heat_window=heat_window,
        parser_confidence=1.0,
        settlement_unit="F",
        settlement_precision="whole_F",
        scored_at_utc=datetime(2026, 1, 2, 11, 55, tzinfo=timezone.utc),
        score_model_version="synthetic",
    )


def _score_with_observed_high_and_heat_window() -> SharedScoreContract:
    return _score(
        heat_window=HeatWindowSnapshot(
            local_hour=16.0,
            forecast_peak_hour_local=14.0,
            remaining_hourly_forecast_max_f=72.0,
            heat_window_exhaustion_score=0.90,
            solar_radiation_state="low",
            cloud_cover_state="cloudy",
            wind_state="breezy",
            season="winter",
        )
    )


def _frame(
    *,
    decision: DecisionSnapshot | None = None,
    score: SharedScoreContract | None = None,
    label: CanonicalLabel | None = None,
):
    from daily_weather_pipeline.backtesting_engine import ReplayFrame

    decision = decision or _decision("row-1")
    label = label or _label(decision.universe_row_id)
    score = score if score is not None else _score()
    return ReplayFrame(
        decision=decision,
        canonical_label=label,
        score=score,
        skip_reasons=PointInTimePolicy().skip_reasons(decision, label, score),
    )


def _label(
    universe_row_id: str,
    *,
    canonical_label: int = 1,
    recommended_action: str = "keep_polymarket_label",
    exclude: bool = False,
) -> CanonicalLabel:
    return CanonicalLabel(
        universe_row_id=universe_row_id,
        market_id="market-1",
        canonical_label=canonical_label,
        canonical_label_source="polymarket_resolved_outcome_via_telonex_final_token_price",
        recommended_action=recommended_action,
        exclude_from_backtest_replay=exclude,
        canonical_bucket_type="exact",
        source_family="Wunderground",
        source_station="station",
        source_url="https://example.test",
        settlement_unit="F",
        settlement_precision="whole_F",
        resolution_date_text="2026-01-02",
        resolution_date_conflict=False,
        open_meteo_current_continuous_label=0,
        open_meteo_source_rule_proxy_label=0,
        audit_flags=(),
    )


def _write_label_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "universe_row_id",
        "market_id",
        "canonical_label",
        "canonical_label_source",
        "recommended_action",
        "exclude_from_backtest_replay",
        "canonical_bucket_type",
        "source_family",
        "source_station",
        "source_url",
        "settlement_unit",
        "settlement_precision",
        "resolution_date_text",
        "resolution_date_conflict",
        "open_meteo_current_continuous_label",
        "open_meteo_source_rule_proxy_label",
        "audit_flags",
    ]
    defaults = {
        "canonical_label_source": "polymarket_resolved_outcome_via_telonex_final_token_price",
        "canonical_bucket_type": "exact",
        "source_family": "Wunderground",
        "source_station": "station",
        "source_url": "https://example.test",
        "settlement_unit": "F",
        "settlement_precision": "whole_F",
        "resolution_date_text": "2026-01-02",
        "resolution_date_conflict": "0",
        "open_meteo_current_continuous_label": "0",
        "open_meteo_source_rule_proxy_label": "0",
        "audit_flags": "",
    }
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({**defaults, **row})


if __name__ == "__main__":
    unittest.main()
