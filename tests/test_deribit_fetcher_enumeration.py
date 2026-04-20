import pytest

from db_option_pricer_win import parse_leg
from deribit_fetcher_win import enumerate_smile_instruments


class StubHandler:
    async def get(self, method, params):
        assert method == "public/get_instruments"
        return {
            "result": [
                {"instrument_name": f"BTC-20MAR26-{k}-{cp}"}
                for k in (65000, 70000, 75000)
                for cp in ("C", "P")
            ]
        }


@pytest.mark.asyncio
async def test_enumerate_smile_instruments_selects_strike_window():
    legs = [parse_leg("L 1 20MAR26-70000-C")]
    instruments, windows = await enumerate_smile_instruments(legs, StubHandler(), half_width=1)
    assert "BTC-20MAR26-70000-C" in instruments
    assert len(windows["20MAR26"]) == 3
