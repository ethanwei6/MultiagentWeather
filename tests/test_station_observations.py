from __future__ import annotations

import sqlite3
import tempfile
import unittest
from datetime import date
from pathlib import Path

from daily_weather_pipeline.station_observations import (
    StationConfig,
    _ensure_schema,
    _insert_observations,
    _phase_ready,
    _station_label_for_row,
    parse_iem_csv_payload,
    parse_temperature_from_iem_row,
    validate_observation_coverage,
)


class StationObservationTests(unittest.TestCase):
    def test_parses_precise_metar_temperature_before_tmpf(self) -> None:
        row = {
            "tmpf": "31.00",
            "metar": "KLGA 010051Z 23010KT 10SM FEW050 M01/M12 A2976 RMK AO2 T10061117",
        }

        parsed = parse_temperature_from_iem_row(row)

        self.assertEqual(parsed["source_unit"], "C")
        self.assertAlmostEqual(parsed["raw_temperature"], -0.6)
        self.assertAlmostEqual(parsed["temperature_f"], 30.92)
        self.assertIn("parsed_metar_precise_t_group", parsed["quality_flags"])

    def test_parses_integer_metar_temperature_when_tmpf_missing(self) -> None:
        row = {
            "tmpf": "",
            "metar": "KLGA 010020Z AUTO 23010G16KT 10SM FEW055 SCT120 M01/M12 A2976 RMK MADISHF",
        }

        parsed = parse_temperature_from_iem_row(row)

        self.assertEqual(parsed["source_unit"], "C")
        self.assertAlmostEqual(parsed["raw_temperature"], -1.0)
        self.assertAlmostEqual(parsed["temperature_f"], 30.2)
        self.assertIn("parsed_metar_integer_temp", parsed["quality_flags"])

    def test_iem_payload_uses_utc_timestamp_and_local_date(self) -> None:
        config = StationConfig(
            station_id="KLGA",
            normalized_city="New York, NY",
            timezone_name="America/New_York",
            settlement_source_family="Wunderground",
            settlement_source_url="https://www.wunderground.com/history/daily/us/ny/new-york-city/KLGA",
            settlement_station_name="LaGuardia Airport Station",
            settlement_unit="F",
            settlement_precision="whole_F",
        )
        payload = "\n".join(
            [
                "station,valid,lon,lat,elevation,tmpf,metar",
                "LGA,2026-01-01 04:55,-73.8803,40.7794,9.00,,KLGA 010455Z AUTO 23010KT 10SM CLR M01/M12 A2976 RMK T10101120",
                "LGA,2026-01-01 05:00,-73.8803,40.7794,9.00,,KLGA 010500Z AUTO 23010KT 10SM CLR M02/M12 A2976 RMK T10201120",
            ]
        ).encode()

        _, observations = parse_iem_csv_payload(
            payload,
            config=config,
            request_url="https://example.test",
            fetched_at="2026-07-07T00:00:00+00:00",
            payload_id="sha256:test",
        )

        self.assertEqual(observations[0].local_date, "2025-12-31")
        self.assertEqual(observations[1].local_date, "2026-01-01")
        self.assertAlmostEqual(observations[0].station_latitude or 0.0, 40.7794)
        self.assertAlmostEqual(observations[0].station_longitude or 0.0, -73.8803)
        self.assertAlmostEqual(observations[0].station_elevation_m or 0.0, 9.0)
        self.assertIn("iem_uses_three_letter_station_alias", observations[0].quality_flags)

    def test_proof_validation_requires_all_expected_local_dates(self) -> None:
        config = StationConfig(
            station_id="KLGA",
            normalized_city="New York, NY",
            timezone_name="America/New_York",
            settlement_source_family="Wunderground",
            settlement_source_url="https://www.wunderground.com/history/daily/us/ny/new-york-city/KLGA",
            settlement_station_name="LaGuardia Airport Station",
            settlement_unit="F",
            settlement_precision="whole_F",
        )
        payload = "\n".join(
            [
                "station,valid,tmpf,metar",
                "LGA,2026-01-01 05:00,,KLGA 010500Z AUTO 23010KT 10SM CLR M02/M12 A2976 RMK T10201120",
                "LGA,2026-01-02 05:00,,KLGA 020500Z AUTO 23010KT 10SM CLR M01/M12 A2976 RMK T10101120",
            ]
        ).encode()
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "station.sqlite"
            with sqlite3.connect(db_path) as conn:
                _ensure_schema(conn)
                _, observations = parse_iem_csv_payload(
                    payload,
                    config=config,
                    request_url="https://example.test",
                    fetched_at="2026-07-07T00:00:00+00:00",
                    payload_id="sha256:test",
                )
                _insert_observations(conn, observations)

            result = validate_observation_coverage(
                db_path=db_path,
                configs=[config],
                start=date(2026, 1, 1),
                end=date(2026, 1, 2),
            )

        self.assertTrue(result["passed"])

    def test_station_label_uses_half_up_whole_f_rounding(self) -> None:
        row = {
            "settlement_unit": "F",
            "settlement_precision": "whole_F",
            "threshold_low_f": "75",
            "threshold_high_f": "75",
        }

        self.assertEqual(_station_label_for_row(row, 74.49), 0)
        self.assertEqual(_station_label_for_row(row, 74.5), 1)
        self.assertEqual(_station_label_for_row(row, 75.49), 1)
        self.assertEqual(_station_label_for_row(row, 75.5), 0)

    def test_station_label_compares_celsius_thresholds_in_display_unit(self) -> None:
        row = {
            "settlement_unit": "C",
            "settlement_precision": "whole_C",
            "threshold_low_f": "42.8",
            "threshold_high_f": "44.5982",
            "threshold_low_display": "6.0",
            "threshold_high_display": "6.0",
        }

        self.assertEqual(_station_label_for_row(row, 42.8), 1)
        self.assertEqual(_station_label_for_row(row, 44.6), 0)

    def test_phase_ready_requires_label_agreement_gate(self) -> None:
        comparison = {"summary": {"missing_station_high_rows": 0}}
        proof_validation = {"passed": True}
        full_validation = {"passed": True}

        self.assertFalse(_phase_ready(comparison, proof_validation, full_validation, True, {"passed": False}))
        self.assertTrue(_phase_ready(comparison, proof_validation, full_validation, True, {"passed": True}))


if __name__ == "__main__":
    unittest.main()
