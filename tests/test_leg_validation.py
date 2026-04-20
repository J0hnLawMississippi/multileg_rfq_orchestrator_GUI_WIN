import unittest

from db_option_pricer_win import parse_leg


class ParseLegValidationTests(unittest.TestCase):
    def test_rejects_zero_size(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "Invalid leg size '0'.*greater than 0"
        ):
            parse_leg("L 0 20MAR26-70000-C")

    def test_rejects_negative_size(self) -> None:
        with self.assertRaisesRegex(ValueError, "Expected canonical format"):
            parse_leg("L -1 20MAR26-70000-C")

    def test_rejects_malformed_size(self) -> None:
        with self.assertRaisesRegex(ValueError, "Expected canonical format"):
            parse_leg("L 1..2 20MAR26-70000-C")

    def test_accepts_fractional_size(self) -> None:
        leg = parse_leg("L 0.25 20MAR26-70000-C")
        self.assertEqual(leg.size, 0.25)

    def test_error_text_consistency_for_bad_size_values(self) -> None:
        for size in ("0", "0.0"):
            with self.assertRaises(ValueError) as ctx:
                parse_leg(f"L {size} 20MAR26-70000-C")
            self.assertIn("size must be a finite number greater than 0.", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
