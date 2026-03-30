#!/usr/bin/env python3
"""
dvol_fetcher.py
---------------
Self-contained async fetcher for Deribit DVOL and BTC index price.

Extracted from /home/claw/deribit/db_vol_data.py — only the public REST
calls needed for the GUI info bar are included (no ccxt, no RV, no rich).

Public API:
    fetch_dvol_and_price(currency="BTC") -> dict
        Returns: {dvol, dvol_high_24h, dvol_low_24h, index_price}
"""

import asyncio
import time
from typing import Dict, Any, Optional

import aiohttp
from aiolimiter import AsyncLimiter

_DERIBIT_BASE = "https://www.deribit.com/api/v2/"
_RATE_LIMIT = 20  # requests per second


class _DeribitPublicClient:
    """Minimal aiohttp client for Deribit public JSON-RPC endpoints."""

    def __init__(self, rate_limit: int = _RATE_LIMIT) -> None:
        self._limiter = AsyncLimiter(rate_limit, 1.0)
        self._session: Optional[aiohttp.ClientSession] = None

    async def open(self) -> None:
        connector = aiohttp.TCPConnector(
            limit=50,
            limit_per_host=20,
            keepalive_timeout=60,
        )
        self._session = aiohttp.ClientSession(connector=connector)

    async def close(self) -> None:
        if self._session:
            await self._session.close()
            self._session = None

    async def __aenter__(self) -> "_DeribitPublicClient":
        await self.open()
        return self

    async def __aexit__(self, *_) -> None:
        await self.close()

    async def get(
        self, method: str, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        if self._session is None:
            raise RuntimeError("Client not opened")
        url = _DERIBIT_BASE + method
        async with self._limiter:
            async with self._session.get(url, params=params or {}) as resp:
                resp.raise_for_status()
                return await resp.json()


async def _get_dvol(client: _DeribitPublicClient, currency: str) -> Dict[str, float]:
    now_ms = int(time.time() * 1000)
    start_ms = now_ms - (24 * 60 * 60 * 1000)  # 24 hours back

    resp = await client.get(
        "public/get_volatility_index_data",
        params={
            "currency": currency,
            "start_timestamp": start_ms,
            "end_timestamp": now_ms,
            "resolution": "3600",  # 1-hour candles → up to 24 returned
        },
    )

    data = resp.get("result", {}).get("data", [])
    if not data:
        raise ValueError(f"No DVOL data returned for {currency}")

    # Candle format: [timestamp, open, high, low, close]
    latest = data[-1]
    window = data[-24:] if len(data) >= 24 else data
    return {
        "dvol": float(latest[4]),
        "dvol_high_24h": float(max(d[2] for d in window)),
        "dvol_low_24h": float(min(d[3] for d in window)),
    }


async def _get_index_price(client: _DeribitPublicClient, currency: str) -> float:
    resp = await client.get(
        "public/get_index_price",
        params={"index_name": f"{currency.lower()}_usd"},
    )
    return float(resp["result"]["index_price"])


async def fetch_dvol_and_price(currency: str = "BTC") -> Dict[str, Any]:
    """
    Fetch DVOL + index price concurrently from Deribit public REST.

    Returns dict with keys:
        dvol           (float) - current DVOL index value
        dvol_high_24h  (float) - 24-hour DVOL high
        dvol_low_24h   (float) - 24-hour DVOL low
        index_price    (float) - current BTC spot index price
    """
    async with _DeribitPublicClient() as client:
        dvol_data, index_price = await asyncio.gather(
            _get_dvol(client, currency),
            _get_index_price(client, currency),
        )

    return {**dvol_data, "index_price": index_price}


if __name__ == "__main__":
    import json

    async def _main():
        print("Fetching DVOL and BTC index price from Deribit...")
        data = await fetch_dvol_and_price("BTC")
        print(json.dumps(data, indent=2))

    asyncio.run(_main())
