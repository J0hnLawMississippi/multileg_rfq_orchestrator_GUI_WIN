import unittest
from datetime import datetime, timezone

import numpy as np

from db_option_pricer_win import (
    MaturityIVData,
    StrikeIVInterpolator,
    StructurePricer,
    parse_legs,
    time_to_expiry_years,
)


class StructurePricerLegTimesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.current_spot = 100_000.0
        self.ref_time = datetime(2026, 1, 1, tzinfo=timezone.utc)

        self.iv_data = {
            "20MAR26": MaturityIVData(
                maturity="20MAR26",
                expiry_ts=0,
                forward=101_000.0,
                strikes=np.array([90_000.0, 110_000.0], dtype=np.float64),
                ivs=np.array([0.50, 0.55], dtype=np.float64),
                mark_prices_btc=np.array([0.0, 0.0], dtype=np.float64),
            ),
            "25DEC26": MaturityIVData(
                maturity="25DEC26",
                expiry_ts=0,
                forward=103_000.0,
                strikes=np.array([90_000.0, 110_000.0], dtype=np.float64),
                ivs=np.array([0.45, 0.50], dtype=np.float64),
                mark_prices_btc=np.array([0.0, 0.0], dtype=np.float64),
            ),
        }
        self.iv_interps = {
            mat: StrikeIVInterpolator(data) for mat, data in self.iv_data.items()
        }
        self.pricer = StructurePricer(
            self.iv_interps,
            self.iv_data,
            self.current_spot,
            r=0.0,
        )

    def test_price_structure_preserves_per_leg_expiry_times_in_input_order(self) -> None:
        legs = parse_legs(
            [
                "L 1 25DEC26-100000-C",
                "S 1 20MAR26-100000-P",
                "L 2 25DEC26-100000-P",
            ]
        )

        result = self.pricer.price_structure(
            legs,
            target_spot=self.current_spot,
            ref_time=self.ref_time,
        )

        expected = np.array(
            [
                time_to_expiry_years("25DEC26", self.ref_time),
                time_to_expiry_years("20MAR26", self.ref_time),
                time_to_expiry_years("25DEC26", self.ref_time),
            ],
            dtype=np.float64,
        )

        np.testing.assert_allclose(result.leg_times_to_expiry_years, expected)
        self.assertEqual(result.leg_times_to_expiry_years.shape, (3,))

    def test_deprecated_scalar_time_to_expiry_years_maps_to_first_leg(self) -> None:
        legs = parse_legs(
            [
                "L 1 25DEC26-100000-C",
                "S 1 20MAR26-100000-P",
            ]
        )
        result = self.pricer.price_structure(
            legs,
            target_spot=self.current_spot,
            ref_time=self.ref_time,
        )

        with self.assertWarns(DeprecationWarning):
            deprecated_scalar = result.time_to_expiry_years

        self.assertAlmostEqual(
            deprecated_scalar,
            result.leg_times_to_expiry_years[0],
        )


if __name__ == "__main__":
    unittest.main()
