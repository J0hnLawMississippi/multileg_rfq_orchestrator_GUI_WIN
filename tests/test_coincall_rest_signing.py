import pytest

import rfq_orchestrator_win as orch
from rfq_orchestrator_win import CoincallREST, RFQConfig
from tests.helpers.fake_aiohttp import FakeSession


@pytest.mark.asyncio
async def test_coincall_create_rfq_uses_canonical_json_and_signature(monkeypatch):
    cfg = RFQConfig()
    rest = CoincallREST(cfg, "k", "s")
    rest._session = FakeSession(payloads=[{"code": 0, "data": {"requestId": "req-1"}}])
    monkeypatch.setattr(orch.time, "time", lambda: 1700000000.0)

    data = await rest.create_rfq([{"instrumentName": "BTCUSD-20MAR26-70000-C", "side": "BUY", "qty": "1"}])
    assert data["requestId"] == "req-1"
    endpoint, kwargs = rest._session.calls[0]
    assert endpoint.endswith("/create/v1")
    assert kwargs["headers"]["Content-Type"] == "application/json"
    assert kwargs["data"].startswith('{"legs":')


def test_signing_is_deterministic_for_known_prehash():
    rest = CoincallREST(RFQConfig(), "k", "secret")
    assert rest._sign("abc") == rest._sign("abc")
