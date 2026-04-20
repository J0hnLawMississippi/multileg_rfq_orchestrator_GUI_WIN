from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from db_option_pricer_win import MaturityIVData, StrikeIVInterpolator


@pytest.fixture
def fixture_dir() -> Path:
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def load_fixture(fixture_dir):
    def _load(name: str):
        return json.loads((fixture_dir / name).read_text())

    return _load


@pytest.fixture
def maturity_data_20mar26() -> MaturityIVData:
    return MaturityIVData(
        maturity="20MAR26",
        expiry_ts=1773993600000,
        forward=70000.0,
        strikes=np.array([60000.0, 65000.0, 70000.0, 75000.0, 80000.0]),
        ivs=np.array([0.72, 0.68, 0.64, 0.66, 0.70]),
        mark_prices_btc=np.array([0.12, 0.08, 0.05, 0.03, 0.02]),
    )


@pytest.fixture
def iv_interp_20mar26(maturity_data_20mar26):
    return StrikeIVInterpolator(maturity_data_20mar26)
