from __future__ import annotations

import json
import unittest
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from daily_weather_pipeline.live_forward_shadow_recorder import _FROZEN_S01_CITY_REGISTRY_ENTRIES
from daily_weather_pipeline.physical_weather_contract import (
    CANONICAL_LABEL_SOURCE,
    CanonicalMarketLabel,
    GFS_ELEVATION_HANDLING,
    GFS_INTERPOLATION,
    GfsGridNode,
    GfsInputProvenance,
    MarketRule,
    PhysicalTargetProvenance,
    PhysicalWeatherContractError,
    RECORDED_SETTLEMENT_BINDINGS,
    STATION_INFO_RAW_SHA256,
    assert_runtime_registry_station_aligned,
    gfs_t_minus_six_cycle,
    parse_gamma_market_rule,
    recorded_inventory_sha256,
    required_gfs_forecast_hours,
    runtime_registry_mismatches,
    validate_event_partition,
    validate_gfs_input,
    validate_gfs_local_day_inputs,
    validate_gfs_range_probe,
    validate_physical_target,
    validate_rule_available_at_decision,
    validate_scoring_label,
    validate_station_info_probe,
)


UTC = timezone.utc
SHA = "a" * 64


def _austin_payload() -> dict[str, object]:
    return {
        "id": "fixture-austin-temperature",
        "conditionId": "0xcondition",
        "question": "Will the highest temperature in Austin be between 90-91°F on July 25?",
        "endDateIso": "2026-07-25",
        "resolutionSource": "https://www.wunderground.com/history/daily/us/tx/austin/KAUS",
        "description": (
            "This market will resolve to the temperature range that contains the highest temperature "
            "recorded at the Austin-Bergstrom International Airport Station in degrees Fahrenheit on 25 Jul '26.\n\n"
            "The resolution source is https://www.wunderground.com/history/daily/us/tx/austin/KAUS.\n\n"
            "This market can not resolve until the first data point for the following date has been published.\n\n"
            "The resolution source for this market measures temperatures to whole degrees Fahrenheit."
        ),
    }


def _moscow_payload() -> dict[str, object]:
    return {
        "id": "moscow-market",
        "conditionId": "0xmoscow",
        "question": "Will the highest temperature in Moscow be 21°C on July 24?",
        "endDateIso": "2026-07-24",
        "resolutionSource": "",
        "description": (
            "This market will resolve to the temperature range that contains the highest temperature "
            "recorded by NOAA at Vnukovo in degrees Celsius on 24 Jul '26.\n\n"
            "The resolution source is https://www.weather.gov/wrh/timeseries?site=UUWW.\n\n"
            "This market can not resolve until the first datapoint for the following date has been published.\n\n"
            "The resolution source for this market measures temperatures to whole degrees Celsius."
        ),
    }


class PhysicalWeatherContractTests(unittest.TestCase):
    def test_recorded_inventory_is_frozen_to_22_exact_bindings(self) -> None:
        self.assertEqual(len(RECORDED_SETTLEMENT_BINDINGS), 22)
        self.assertEqual(len({row.city for row in RECORDED_SETTLEMENT_BINDINGS}), 22)
        self.assertEqual(len({row.station_id for row in RECORDED_SETTLEMENT_BINDINGS}), 22)
        self.assertEqual(
            recorded_inventory_sha256(),
            "ac7b9c6a929a686e65ef0dceb09199e7bfc5a37e92748049320f5b35fdd7a0a2",
        )

    def test_official_station_info_fixture_pins_all_coordinates(self) -> None:
        fixture = json.loads((Path(__file__).with_name("noaa_stationinfo_probe_fixture.json")).read_text(encoding="utf-8"))
        validate_station_info_probe(fixture)
        self.assertEqual(fixture["raw_response_sha256"], STATION_INFO_RAW_SHA256)

    def test_gamma_rule_parser_pins_source_station_unit_date_and_bucket(self) -> None:
        retrieved = datetime(2026, 7, 21, 12, tzinfo=UTC)
        austin = parse_gamma_market_rule(
            _austin_payload(),
            event_id="Austin, TX|2026-07-25",
            retrieved_at_utc=retrieved,
        )
        self.assertEqual(austin.binding.station_id, "KAUS")
        self.assertEqual(austin.binding.unit, "F")
        self.assertEqual((austin.bucket.lower_inclusive, austin.bucket.upper_inclusive), (90, 91))
        self.assertEqual(austin.rounding_rule, "not_specified_use_canonical_payout")
        moscow = parse_gamma_market_rule(
            _moscow_payload(),
            event_id="Moscow, RU|2026-07-24",
            retrieved_at_utc=retrieved,
        )
        self.assertEqual(moscow.binding.station_id, "UUWW")
        self.assertEqual(moscow.binding.source_family, "NOAA")
        self.assertEqual((moscow.bucket.lower_inclusive, moscow.bucket.upper_inclusive), (21, 21))

    def test_gamma_rule_change_and_post_decision_retrieval_fail_closed(self) -> None:
        changed = _austin_payload()
        changed["resolutionSource"] = "https://www.wunderground.com/history/daily/us/tx/austin/KATT"
        with self.assertRaisesRegex(PhysicalWeatherContractError, "binding changed"):
            parse_gamma_market_rule(
                changed,
                event_id="Austin, TX|2026-07-25",
                retrieved_at_utc=datetime(2026, 7, 21, tzinfo=UTC),
            )
        rule = parse_gamma_market_rule(
            _austin_payload(),
            event_id="Austin, TX|2026-07-25",
            retrieved_at_utc=datetime(2026, 7, 25, 12, tzinfo=UTC),
        )
        with self.assertRaisesRegex(PhysicalWeatherContractError, "after the decision"):
            validate_rule_available_at_decision(rule, datetime(2026, 7, 25, 11, tzinfo=UTC))

    def test_event_bucket_partition_must_be_exhaustive_and_gap_free(self) -> None:
        base = parse_gamma_market_rule(
            _moscow_payload(),
            event_id="Moscow, RU|2026-07-24",
            retrieved_at_utc=datetime(2026, 7, 21, tzinfo=UTC),
        )
        lower = replace(base, market_id="1", condition_id="1", bucket=replace(base.bucket, lower_inclusive=None, upper_inclusive=20))
        exact = replace(base, market_id="2", condition_id="2")
        upper = replace(base, market_id="3", condition_id="3", bucket=replace(base.bucket, lower_inclusive=22, upper_inclusive=None))
        self.assertEqual(validate_event_partition((lower, exact, upper)), base.binding)
        with self.assertRaisesRegex(PhysicalWeatherContractError, "gap or overlap"):
            validate_event_partition((lower, replace(upper, bucket=replace(upper.bucket, lower_inclusive=23))))

    def test_historical_recorder_registry_flags_nine_settlement_station_mismatches(self) -> None:
        entries = _FROZEN_S01_CITY_REGISTRY_ENTRIES
        mismatches = runtime_registry_mismatches(entries)
        self.assertEqual(
            {(row.city, row.runtime_station_id, row.settlement_station_id) for row in mismatches},
            {
                ("Chicago, IL", "KMDW", "KORD"),
                ("Denver, CO", "KDEN", "KBKF"),
                ("London, GB", "EGLL", "EGLC"),
                ("Los Angeles, CA", "KCQT", "KLAX"),
                ("Moscow, RU", "UUEE", "UUWW"),
                ("New York, NY", "KNYC", "KLGA"),
                ("Paris, FR", "LFPG", "LFPB"),
                ("Seoul, KR", "RKSS", "RKSI"),
                ("Shanghai, CN", "ZSSS", "ZSPD"),
            },
        )
        with self.assertRaisesRegex(PhysicalWeatherContractError, "runtime settlement-station mismatch"):
            assert_runtime_registry_station_aligned(entries)

    def test_gfs_t_minus_six_cycle_and_range_probe_are_frozen(self) -> None:
        self.assertEqual(
            gfs_t_minus_six_cycle(datetime(2026, 8, 17, 10, 7, tzinfo=UTC)),
            datetime(2026, 8, 17, 0, tzinfo=UTC),
        )
        self.assertEqual(
            gfs_t_minus_six_cycle(datetime(2026, 8, 17, 5, 59, tzinfo=UTC)),
            datetime(2026, 8, 16, 18, tzinfo=UTC),
        )
        fixture = json.loads((Path(__file__).with_name("noaa_gfs_range_probe_fixture.json")).read_text(encoding="utf-8"))
        validate_gfs_range_probe(fixture)
        fixture["object_last_modified_utc"] = "2026-08-17T10:07:33Z"
        with self.assertRaisesRegex(PhysicalWeatherContractError, "unavailable object"):
            validate_gfs_range_probe(fixture)

    def test_required_forecast_hours_follow_station_local_dst_boundaries(self) -> None:
        normal = required_gfs_forecast_hours(
            run_time_utc=datetime(2026, 7, 1, tzinfo=UTC),
            target_date=date(2026, 7, 2),
            timezone_name="America/New_York",
        )
        spring = required_gfs_forecast_hours(
            run_time_utc=datetime(2026, 3, 7, tzinfo=UTC),
            target_date=date(2026, 3, 8),
            timezone_name="America/New_York",
        )
        autumn = required_gfs_forecast_hours(
            run_time_utc=datetime(2026, 10, 31, tzinfo=UTC),
            target_date=date(2026, 11, 1),
            timezone_name="America/New_York",
        )
        self.assertEqual((len(normal), len(spring), len(autumn)), (24, 23, 25))
        self.assertEqual(GFS_INTERPOLATION, "bilinear_four_surrounding_0p25_degree_nodes")
        self.assertEqual(GFS_ELEVATION_HANDLING, "no_lapse_rate_adjustment_residual_calibration_only")

    def test_gfs_input_requires_direct_causal_station_provenance(self) -> None:
        binding = RECORDED_SETTLEMENT_BINDINGS[0]
        fixture = json.loads((Path(__file__).with_name("noaa_stationinfo_probe_fixture.json")).read_text(encoding="utf-8"))
        west = (binding.station_longitude - (-97.75)) / 0.25
        north = (binding.station_latitude - 30.0) / 0.25
        nodes = (
            GfsGridNode(30.0, -97.75, (1.0 - west) * (1.0 - north), 300.0),
            GfsGridNode(30.0, -97.5, west * (1.0 - north), 300.0),
            GfsGridNode(30.25, -97.75, (1.0 - west) * north, 300.0),
            GfsGridNode(30.25, -97.5, west * north, 300.0),
        )
        provenance = GfsInputProvenance(
            station_id="KAUS",
            station_latitude=binding.station_latitude,
            station_longitude=binding.station_longitude,
            station_metadata_source_url=fixture["request_url"],
            station_metadata_sha256=STATION_INFO_RAW_SHA256,
            run_time_utc=datetime(2026, 8, 17, 0, tzinfo=UTC),
            forecast_hour=24,
            valid_time_utc=datetime(2026, 8, 18, 0, tzinfo=UTC),
            object_key="gfs.20260817/00/atmos/gfs.t00z.pgrb2.0p25.f024",
            object_last_modified_utc=datetime(2026, 8, 17, 3, 43, 9, tzinfo=UTC),
            object_etag="18c37c5b149717b2cdff37ca08ed4c7e",
            object_size_bytes=553047280,
            index_key="gfs.20260817/00/atmos/gfs.t00z.pgrb2.0p25.f024.idx",
            index_last_modified_utc=datetime(2026, 8, 17, 3, 43, 35, tzinfo=UTC),
            index_sha256=SHA,
            byte_range_start=428008300,
            byte_range_end_inclusive=428523065,
            raw_range_sha256=SHA,
            variable_descriptor="TMP:2 m above ground:24 hour fcst",
            grid_nodes=nodes,
            decoded_temperature_k=300.0,
            decoder_name="eccodes",
            decoder_version="fixture-only",
            decoder_binary_sha256=SHA,
            decoded_message_sha256=SHA,
            fetched_at_utc=datetime(2026, 8, 17, 10, 0, tzinfo=UTC),
            feature_assembled_at_utc=datetime(2026, 8, 17, 10, 1, tzinfo=UTC),
            quote_time_utc=datetime(2026, 8, 17, 10, 2, tzinfo=UTC),
            decision_time_utc=datetime(2026, 8, 17, 10, 3, tzinfo=UTC),
        )
        validate_gfs_input(provenance, binding)
        with self.assertRaisesRegex(PhysicalWeatherContractError, "timing is not causal"):
            validate_gfs_input(
                replace(provenance, object_last_modified_utc=datetime(2026, 8, 17, 10, 2, tzinfo=UTC)),
                binding,
            )

        hours = required_gfs_forecast_hours(
            run_time_utc=provenance.run_time_utc,
            target_date=date(2026, 8, 18),
            timezone_name=binding.timezone_name,
        )
        local_day_inputs = tuple(
            replace(
                provenance,
                forecast_hour=hour,
                valid_time_utc=provenance.run_time_utc + timedelta(hours=hour),
                object_key=f"gfs.20260817/00/atmos/gfs.t00z.pgrb2.0p25.f{hour:03d}",
                index_key=f"gfs.20260817/00/atmos/gfs.t00z.pgrb2.0p25.f{hour:03d}.idx",
                variable_descriptor=f"TMP:2 m above ground:{hour} hour fcst",
            )
            for hour in hours
        )
        self.assertEqual(
            validate_gfs_local_day_inputs(local_day_inputs, binding=binding, target_date=date(2026, 8, 18)),
            300.0,
        )
        with self.assertRaisesRegex(PhysicalWeatherContractError, "missing or duplicates"):
            validate_gfs_local_day_inputs(local_day_inputs[:-1], binding=binding, target_date=date(2026, 8, 18))

    def test_isd_physical_targets_are_causal_but_not_scoring_truth(self) -> None:
        binding = RECORDED_SETTLEMENT_BINDINGS[0]
        target = PhysicalTargetProvenance(
            station_id="KAUS",
            local_date=date(2025, 7, 25),
            timezone_name="America/Chicago",
            source_dataset="NOAA_ISD",
            source_url="https://www.ncei.noaa.gov/products/land-based-station/integrated-surface-database",
            raw_payload_sha256=SHA,
            first_available_at_utc=datetime(2025, 7, 27, tzinfo=UTC),
            computed_at_utc=datetime(2025, 7, 27, 1, tzinfo=UTC),
            maximum_temperature_c=36.0,
        )
        validate_physical_target(
            target,
            binding=binding,
            model_fit_cutoff_utc=datetime(2026, 7, 1, tzinfo=UTC),
        )
        with self.assertRaisesRegex(PhysicalWeatherContractError, "not canonical scoring labels"):
            validate_scoring_label(target, evaluation_cutoff_utc=datetime(2026, 7, 1, tzinfo=UTC))  # type: ignore[arg-type]
        label = CanonicalMarketLabel(
            market_id="3009081",
            condition_id="0xcondition",
            yes_outcome=1,
            source=CANONICAL_LABEL_SOURCE,
            raw_payload_sha256=SHA,
            label_available_at_utc=datetime(2026, 7, 26, tzinfo=UTC),
        )
        validate_scoring_label(label, evaluation_cutoff_utc=datetime(2026, 7, 27, tzinfo=UTC))


if __name__ == "__main__":
    unittest.main()
