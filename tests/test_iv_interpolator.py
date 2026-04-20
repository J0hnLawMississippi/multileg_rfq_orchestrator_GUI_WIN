import numpy as np
import pytest

from db_option_pricer_win import MaturityIVData, StrikeIVInterpolator


def test_interpolator_boundaries(iv_interp_20mar26):
    vals = iv_interp_20mar26.iv_at_strikes(np.array([50000.0, 70000.0, 90000.0]))
    assert vals[0] == pytest.approx(0.72)
    assert 0.01 <= vals[1] <= 5.0
    assert vals[2] == pytest.approx(0.70)


def test_interpolator_requires_two_points():
    with pytest.raises(ValueError):
        StrikeIVInterpolator(
            MaturityIVData(
                maturity="20MAR26",
                expiry_ts=0,
                forward=70000.0,
                strikes=np.array([70000.0]),
                ivs=np.array([0.65]),
                mark_prices_btc=np.array([0.05]),
            )
        )
