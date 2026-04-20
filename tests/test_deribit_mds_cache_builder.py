import pytest

from deribit_fetcher_win import DeribitMarketDataService
from tests.helpers.ws_builders import deribit_index_push, deribit_ticker_push


@pytest.mark.asyncio
async def test_mds_builds_cache_and_readiness():
    inst = frozenset(["BTC-20MAR26-70000-C", "BTC-20MAR26-70000-P"])
    mds = DeribitMarketDataService(
        instruments=inst,
        maturity_strike_windows={"20MAR26": [70000.0]},
        min_strikes_ready=1,
    )
    mds._on_message(deribit_index_push())
    mds._on_message(deribit_ticker_push("BTC-20MAR26-70000-C"))
    assert mds.is_ready
    snap = mds.get_snapshot("BTC-20MAR26-70000-C")
    assert snap is not None
    assert snap.mark_iv > 0
