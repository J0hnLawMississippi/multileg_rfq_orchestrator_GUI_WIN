from datetime import datetime, timezone

from db_option_pricer_win import OptionLeg, OptionType, Direction, StructurePricer


def _leg(strike=70000.0, option_type=OptionType.CALL):
    return OptionLeg(Direction.LONG, 1.0, "20MAR26", strike, option_type, f"BTC-20MAR26-{int(strike)}-{option_type.value}")


def test_pricer_monotonicity_for_long_call(iv_interp_20mar26, maturity_data_20mar26):
    pr = StructurePricer({"20MAR26": iv_interp_20mar26}, {"20MAR26": maturity_data_20mar26}, 70000.0)
    leg = _leg()
    ref = datetime(2026, 1, 1, tzinfo=timezone.utc)
    low = pr.price_structure([leg], 65000.0, ref_time=ref).total_usd
    high = pr.price_structure([leg], 75000.0, ref_time=ref).total_usd
    assert high > low


def test_pricer_vol_shock_changes_output(iv_interp_20mar26, maturity_data_20mar26):
    pr = StructurePricer({"20MAR26": iv_interp_20mar26}, {"20MAR26": maturity_data_20mar26}, 70000.0)
    leg = _leg()
    ref = datetime(2026, 1, 1, tzinfo=timezone.utc)
    base = pr.price_structure([leg], 70000.0, ref_time=ref, vol_shift=0.0).total_usd
    shocked = pr.price_structure([leg], 70000.0, ref_time=ref, vol_shift=-0.10).total_usd
    assert base != shocked
