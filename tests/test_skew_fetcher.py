import pytest

from skew_fetcher_win import SkewFetcher


class FakeSkew(SkewFetcher):
    async def _group_options_by_maturity(self, currency="BTC"):
        return {"20MAR26": (["BTC-20MAR26-75000-C"], ["BTC-20MAR26-65000-P"])}

    async def _calculate_skew(self, calls, puts):
        assert calls and puts
        return 4.2


@pytest.mark.asyncio
async def test_fetch_skew_for_maturity_happy_path():
    val = await FakeSkew(handler=None).fetch_skew_for_maturity("20MAR26")
    assert val == 4.2


@pytest.mark.asyncio
async def test_fetch_skew_for_unknown_maturity_raises():
    with pytest.raises(ValueError):
        await FakeSkew(handler=None).fetch_skew_for_maturity("01JAN99")
