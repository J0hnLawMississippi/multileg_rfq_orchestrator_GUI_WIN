from __future__ import annotations

import json


class FakeResponse:
    def __init__(self, payload: dict | str):
        self._payload = payload

    async def text(self) -> str:
        if isinstance(self._payload, str):
            return self._payload
        return json.dumps(self._payload)


class FakeRequestContext:
    def __init__(self, payload: dict | str):
        self._resp = FakeResponse(payload)

    async def __aenter__(self):
        return self._resp

    async def __aexit__(self, *exc):
        return False


class FakeSession:
    def __init__(self, payloads: list[dict | str] | None = None):
        self.payloads = payloads or [{"code": 0, "data": {}}]
        self.calls: list[tuple[str, dict]] = []

    def post(self, endpoint: str, **kwargs):
        self.calls.append((endpoint, kwargs))
        payload = self.payloads.pop(0) if self.payloads else {"code": 0, "data": {}}
        return FakeRequestContext(payload)

    async def close(self):
        return None
