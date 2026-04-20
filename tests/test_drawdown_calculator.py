from db_option_pricer_win import OptionLeg, OptionType, Direction, VectorizedDrawdownCalculator


def test_drawdown_shock_changes_path(iv_interp_20mar26, maturity_data_20mar26):
    leg = OptionLeg(Direction.LONG, 1.0, "20MAR26", 70000.0, OptionType.CALL, "BTC-20MAR26-70000-C")
    dd = VectorizedDrawdownCalculator(
        iv_interpolators={"20MAR26": iv_interp_20mar26},
        maturity_data={"20MAR26": maturity_data_20mar26},
        current_spot=70000.0,
        legs=[leg],
        original_spot=70000.0,
        spot_grid_points=30,
        vol_shock=-0.2,
    )
    flat, shock = dd.compute(days_forward=2)
    assert len(flat.spot_range) == 30
    assert flat.max_drawdown_usd != shock.max_drawdown_usd
