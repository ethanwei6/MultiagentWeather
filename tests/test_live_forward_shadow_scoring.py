from __future__ import annotations

import unittest
from datetime import date, datetime, timezone
from pathlib import Path

from daily_weather_pipeline.backtesting_engine import (
    DecisionSnapshot,
    ForecastSnapshot,
    ObservedHighSnapshot,
    QuoteSnapshot,
    ReplayGeneratedScoreCache,
    WeatherModelDecisionContext,
    validate_weather_model_context_firewall,
)
from daily_weather_pipeline.live_forward_shadow_scoring import (
    LEGACY_LIVE_OPEN_METEO_SCORE_FAMILY,
    LiveForwardShadowScoringAdapter,
    RUNNER_SCORE_FAMILY,
    default_live_forward_shadow_scoring_adapter,
    model_visible_observed_high,
    shadow_scoring_parity_report,
)
from daily_weather_pipeline.weather_model_runner import (
    ObservedHighGuardedShrinkageWeatherModelRunner,
)


class LiveForwardShadowScoringTests(unittest.TestCase):
    def test_default_live_adapter_uses_dynamic_training_corpus_ceiling(self) -> None:
        adapter = default_live_forward_shadow_scoring_adapter(
            work_root=Path("."),
            score_cache=ReplayGeneratedScoreCache(),
        )

        self.assertGreaterEqual(
            adapter.model_runner.end,
            date(2026, 7, 9),
        )

    def test_live_adapter_matches_replay_runner_shared_score_fields(self) -> None:
        runner = _runner(probability=0.08)
        decision = _decision()

        context = WeatherModelDecisionContext.from_decision(decision)
        validate_weather_model_context_firewall(context)
        replay_prediction = runner.predict(context)
        self.assertIsNotNone(replay_prediction)
        assert replay_prediction is not None
        replay_score = replay_prediction.to_shared_score(
            side=decision.side,
            observed_high=model_visible_observed_high(decision),
            parser_confidence=decision.parser_confidence,
            settlement_unit="F",
            settlement_precision="1.0",
        )

        cache = ReplayGeneratedScoreCache()
        live_result = LiveForwardShadowScoringAdapter(
            model_runner=runner,
            score_cache=cache,
        ).score_decision(
            decision,
            settlement_unit="F",
            settlement_precision="1.0",
        )

        self.assertIsNone(live_result.firewall_violation)
        self.assertFalse(live_result.missing_model_output)
        self.assertIsNotNone(live_result.score)
        live_score = live_result.score
        assert live_score is not None
        self.assertEqual(live_score.score_family, RUNNER_SCORE_FAMILY)
        self.assertEqual(live_score.p_yes, replay_score.p_yes)
        self.assertEqual(live_score.p_side, replay_score.p_side)
        self.assertEqual(live_score.model_agreement, replay_score.model_agreement)
        self.assertEqual(live_score.model_count, replay_score.model_count)
        self.assertEqual(live_score.per_source_probabilities, replay_score.per_source_probabilities)
        self.assertEqual(live_score.score_model_version, replay_score.score_model_version)
        self.assertEqual(live_score.metadata["model_code_hash"], replay_score.metadata["model_code_hash"])
        self.assertEqual(live_score.metadata["model_config_hash"], replay_score.metadata["model_config_hash"])
        self.assertEqual(live_score.metadata["runner_metadata"]["runner_name"], "ObservedHighGuardedShrinkageWeatherModelRunner")
        self.assertEqual(cache.report_summary()["record_count"], 1)

        input_payload = live_result.context.to_hash_payload()
        for forbidden in ("bid", "ask", "mid", "canonical_label", "resolved_outcome"):
            self.assertNotIn(forbidden, input_payload)

    def test_legacy_score_family_is_not_gate_valid(self) -> None:
        report = shadow_scoring_parity_report(
            events=[
                {
                    "model_output": {
                        "status": "logged",
                        "score_family": LEGACY_LIVE_OPEN_METEO_SCORE_FAMILY,
                        "metadata": {},
                    }
                }
            ],
            score_cache_summary={
                "record_count": 0,
                "model_versions": [],
                "model_code_hashes": [],
                "model_config_hashes": [],
            },
            runner_metadata={
                "runner_name": "ObservedHighGuardedShrinkageWeatherModelRunner",
                "model_version": "observed_high_guarded_shrinkage_research_runner_v1",
            },
        )

        self.assertFalse(report["gate_valid_for_shadow_deployment"])
        self.assertEqual(report["legacy_live_open_meteo_adapter_score_count"], 1)
        self.assertIn("legacy_live_open_meteo_adapter_scores_present", report["gate_validity_reason"])
        self.assertTrue(report["pre_fix_artifacts_excluded_from_gate"])

    def test_parity_report_fails_closed_on_stale_training_corpus(self) -> None:
        report = shadow_scoring_parity_report(
            events=[_parity_event(_fresh_coverage_metadata(end_date="2026-07-01"))],
            score_cache_summary=_score_cache_summary(),
            runner_metadata=_runner_metadata(),
            generated_at_utc=datetime(2026, 7, 9, 12, tzinfo=timezone.utc),
        )

        self.assertFalse(report["gate_valid_for_shadow_deployment"])
        self.assertEqual(report["training_corpus_end_date"], "2026-07-01")
        self.assertEqual(report["training_row_count"], 123)
        self.assertEqual(report["training_corpus_configured_end_date"], "2026-07-09")
        self.assertEqual(report["training_corpus_staleness_days"], 8)
        self.assertTrue(report["training_corpus_staleness_warning"])
        self.assertIn("training_corpus_stale_gt_3d", report["warning_flags"])
        self.assertIn("training_corpus_stale_by_global_date", report["gate_validity_reason"])

    def test_parity_report_is_valid_with_fresh_relevant_city_date_coverage(self) -> None:
        report = shadow_scoring_parity_report(
            events=[_parity_event(_fresh_coverage_metadata())],
            score_cache_summary=_score_cache_summary(),
            runner_metadata=_runner_metadata(),
            generated_at_utc=datetime(2026, 7, 9, 12, tzinfo=timezone.utc),
        )

        self.assertTrue(report["gate_valid_for_shadow_deployment"])
        self.assertEqual(report["gate_validity_reason"], "gate_valid")
        self.assertEqual(report["training_corpus_coverage_available_count"], 1)
        self.assertEqual(report["training_corpus_operationally_fresh_count"], 1)
        self.assertEqual(report["training_corpus_relevant_city_fresh_count"], 1)
        self.assertTrue(
            report["training_corpus_relevant_coverage"][0][
                "relevant_city_date_coverage_fresh"
            ]
        )

    def test_parity_report_fails_closed_when_coverage_is_unavailable(self) -> None:
        metadata = _fresh_coverage_metadata()
        metadata.update(
            {
                "training_corpus_coverage_available": False,
                "training_corpus_operationally_fresh": False,
                "training_corpus_city_coverage": None,
            }
        )
        report = shadow_scoring_parity_report(
            events=[_parity_event(metadata)],
            score_cache_summary=_score_cache_summary(),
            runner_metadata=_runner_metadata(),
            generated_at_utc=datetime(2026, 7, 9, 12, tzinfo=timezone.utc),
        )

        self.assertFalse(report["gate_valid_for_shadow_deployment"])
        self.assertIn("training_corpus_coverage_unavailable", report["gate_validity_reason"])
        self.assertIn(
            "training_corpus_relevant_city_date_coverage_missing",
            report["gate_validity_reason"],
        )

    def test_parity_report_fails_closed_on_incomplete_relevant_city_date(self) -> None:
        metadata = _fresh_coverage_metadata()
        metadata["training_corpus_operationally_fresh"] = False
        metadata["training_corpus_city_coverage"] = {
            **metadata["training_corpus_city_coverage"],
            "operationally_fresh": False,
            "complete_target_dates": ["2026-07-06"],
            "incomplete_target_dates": ["2026-07-07"],
            "latest_complete_target_date": "2026-07-06",
            "source_coverage": {
                "forecast": {
                    "missing_target_dates": ["2026-07-07"],
                },
                "realized": {
                    "missing_target_dates": [],
                },
            },
        }
        report = shadow_scoring_parity_report(
            events=[_parity_event(metadata)],
            score_cache_summary=_score_cache_summary(),
            runner_metadata=_runner_metadata(),
            generated_at_utc=datetime(2026, 7, 9, 12, tzinfo=timezone.utc),
        )

        self.assertFalse(report["gate_valid_for_shadow_deployment"])
        self.assertIn(
            "training_corpus_coverage_stale_or_incomplete",
            report["gate_validity_reason"],
        )
        self.assertIn(
            "training_corpus_relevant_city_date_coverage_stale_or_incomplete",
            report["gate_validity_reason"],
        )

def _fresh_coverage_metadata(*, end_date: str = "2026-07-07") -> dict[str, object]:
    return {
        "runner_metadata": {
            "runner_name": "ObservedHighGuardedShrinkageWeatherModelRunner"
        },
        "model_code_hash": "code",
        "model_config_hash": "config",
        "training_corpus_end_date": end_date,
        "training_corpus_configured_end_date": "2026-07-09",
        "training_row_count": 123,
        "training_corpus_loaded_row_count": 150,
        "training_corpus_staleness_warning": False,
        "training_corpus_coverage_available": True,
        "training_corpus_operationally_fresh": True,
        "training_corpus_city_coverage": {
            "normalized_city": "Test City, TC",
            "required_target_dates": [end_date],
            "complete_target_dates": [end_date],
            "incomplete_target_dates": [],
            "latest_required_target_date": end_date,
            "latest_complete_target_date": end_date,
            "source_coverage": {
                "forecast": {"missing_target_dates": []},
                "realized": {"missing_target_dates": []},
            },
            "operationally_fresh": True,
        },
    }


def _parity_event(metadata: dict[str, object]) -> dict[str, object]:
    return {
        "market_discovery": {
            "normalized_city": "Test City, TC",
            "market_date_local": "2026-07-10",
        },
        "model_output": {
            "status": "logged",
            "score_family": RUNNER_SCORE_FAMILY,
            "metadata": metadata,
        },
    }


def _score_cache_summary() -> dict[str, object]:
    return {
        "record_count": 1,
        "model_versions": ["observed_high_guarded_shrinkage_research_runner_v1"],
        "model_code_hashes": ["code"],
        "model_config_hashes": ["config"],
    }


def _runner_metadata() -> dict[str, str]:
    return {
        "runner_name": "ObservedHighGuardedShrinkageWeatherModelRunner",
        "model_version": "observed_high_guarded_shrinkage_research_runner_v1",
    }


def _runner(probability: float) -> ObservedHighGuardedShrinkageWeatherModelRunner:
    def baseline_provider(_context: WeatherModelDecisionContext):
        return probability, {
            "open_meteo.kernel_tight": probability,
            "open_meteo.normal_parametric": min(0.99, probability + 0.02),
        }

    return ObservedHighGuardedShrinkageWeatherModelRunner(
        historical_rows=[],
        baseline_probability_provider=baseline_provider,
    )


def _decision() -> DecisionSnapshot:
    decision_time = datetime(2026, 7, 8, 12, tzinfo=timezone.utc)
    return DecisionSnapshot(
        decision_id="decision-1",
        universe_row_id="live:market-1:no-token",
        market_id="market-1",
        slug="test-city-high-july-9-100f",
        question="Will the high temperature in Test City be 100F or above on July 9?",
        normalized_city="Test City, TC",
        country_region="TC",
        timezone="UTC",
        market_date_local="2026-07-09",
        bucket_label="NO: 100F or above",
        bucket_type="above",
        threshold_low_f=100.0,
        threshold_high_f=None,
        side="NO",
        yes_token_id="yes-token",
        no_token_id="no-token",
        token_id="no-token",
        decision_time_utc=decision_time,
        decision_hour_utc=12,
        lead_days=1,
        quote_store="live_polymarket_clob_public_book_snapshot",
        quote=QuoteSnapshot(
            timestamp_utc=decision_time,
            bid=0.69,
            ask=0.70,
            mid=0.695,
            last_price=None,
            spread=0.01,
            bid_size=20.0,
            ask_size=15.0,
            liquidity_depth=35.0,
            source_payload_id="book-payload",
            source_cache_path="/tmp/book.json",
        ),
        forecast=ForecastSnapshot(
            run_time_utc=datetime(2026, 7, 8, 6, tzinfo=timezone.utc),
            source_name="live_open_meteo_multi_source_mean",
            temperature_2m_max_f=82.0,
            features={
                "source_count": 2,
                "latitude": 0.0,
                "longitude": 0.0,
            },
        ),
        data_quality_flags=("live_forward_shadow_read_only",),
        parser_confidence=0.95,
        observed_high=ObservedHighSnapshot(
            observed_high_f=79.0,
            observed_high_timestamp_utc=decision_time,
            observation_source="IEM_ASOS_METAR",
            observation_final=False,
            source_family="IEM_ASOS_METAR",
            source_station="KTCY",
        ),
        observed_high_source_payload_id="station-payload",
        observed_high_raw_cache_path="/tmp/station.csv",
        observed_high_data_quality_flags=("downloaded",),
        decision_local_hour=12.0,
        settlement_source="pending_live_market",
    )


if __name__ == "__main__":
    unittest.main()
