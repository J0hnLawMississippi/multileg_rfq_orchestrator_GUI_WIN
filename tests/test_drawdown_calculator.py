from db_option_pricer_win import (
    Direction,
    MaturityIVData,
    OptionLeg,
    OptionType,
    StrikeIVInterpolator,
    VectorizedDrawdownCalculator,
)


def test_drawdown_shock_changes_path(iv_interp_20mar26, maturity_data_20mar26):
    # Use a future maturity label so this assertion remains valid over time.
    maturity = "20MAR30"
    maturity_data = MaturityIVData(
        maturity=maturity,
        expiry_ts=maturity_data_20mar26.expiry_ts,
        forward=maturity_data_20mar26.forward,
        strikes=maturity_data_20mar26.strikes,
        ivs=maturity_data_20mar26.ivs,
        mark_prices_btc=maturity_data_20mar26.mark_prices_btc,
    )
    iv_interp = StrikeIVInterpolator(maturity_data)
    leg = OptionLeg(
        Direction.LONG,
        1.0,
        maturity,
        70000.0,
        OptionType.CALL,
        "BTC-20MAR30-70000-C",
    )
    dd = VectorizedDrawdownCalculator(
        iv_interpolators={maturity: iv_interp},
        maturity_data={maturity: maturity_data},
        current_spot=70000.0,
        legs=[leg],
        original_spot=70000.0,
        spot_grid_points=30,
        vol_shock=-0.2,
    )
    flat, shock = dd.compute(days_forward=2)
    assert len(flat.spot_range) == 30
    assert flat.max_drawdown_usd != shock.max_drawdown_usd
