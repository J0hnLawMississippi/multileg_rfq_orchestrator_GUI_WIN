import pytest

from db_option_pricer_win import Direction, OptionType, parse_leg, parse_legs


def test_parse_leg_happy_path():
    leg = parse_leg("L 0.5 20MAR26-70000-C")
    assert leg.direction == Direction.LONG
    assert leg.option_type == OptionType.CALL
    assert leg.instrument_name == "BTC-20MAR26-70000-C"


def test_parse_leg_invalid_format():
    with pytest.raises(ValueError):
        parse_leg("bad spec")


def test_parse_legs_multiple():
    legs = parse_legs(["L 0.5 20MAR26-70000-C", "S 1 20MAR26-65000-P"])
    assert len(legs) == 2
    assert legs[1].direction == Direction.SHORT
