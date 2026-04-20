import pytest

from dvol_fetcher_win import _get_dvol, _get_index_price, fetch_dvol_and_price


class FakeClient:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return None

    async def get(self, method, params=None):
        if method == "public/get_volatility_index_data":
            return {"result": {"data": [[1, 0, 55, 45, 50], [2, 0, 56, 44, 51]]}}
        if method == "public/get_index_price":
            return {"result": {"index_price": 70000}}
        raise AssertionError(method)


@pytest.mark.asyncio
async def test_get_dvol_parses_window():
    out = await _get_dvol(FakeClient(), "BTC")
    assert out["dvol"] == 51.0
    assert out["dvol_high_24h"] == 56.0
    assert out["dvol_low_24h"] == 44.0


@pytest.mark.asyncio
async def test_get_index_price_parses_value():
    px = await _get_index_price(FakeClient(), "BTC")
    assert px == 70000.0


@pytest.mark.asyncio
async def test_fetch_dvol_and_price_offline(monkeypatch):
    import dvol_fetcher_win as mod

    monkeypatch.setattr(mod, "_DeribitPublicClient", lambda: FakeClient())
    data = await fetch_dvol_and_price("BTC")
    assert set(data) == {"dvol", "dvol_high_24h", "dvol_low_24h", "index_price"}
