from datetime import datetime, timezone

import pytest

from db_option_pricer_win import maturity_to_datetime, time_to_expiry_years


def test_maturity_to_datetime_parses_day_month_year():
    dt = maturity_to_datetime("20MAR26")
    assert dt == datetime(2026, 3, 20, 8, 0, 0, tzinfo=timezone.utc)


def test_maturity_to_datetime_invalid_month():
    with pytest.raises(ValueError):
        maturity_to_datetime("20XYZ26")


def test_time_to_expiry_floors_at_small_positive():
    ref = datetime(2026, 3, 21, tzinfo=timezone.utc)
    assert time_to_expiry_years("20MAR26", ref) > 0
