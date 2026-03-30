#!/usr/bin/env python3
"""
skew_fetcher.py
---------------
Self-contained async 25-delta skew calculator for BTC options.

Extracted from /home/claw/deribit/scaledVega_delta25skew_new.py.
Uses the local deribit_api.py (already in /home/claw/coincall/).

Skew definition (call-put convention):
    skew = call_iv - put_iv  (percentage points, at the 25-delta strike)

Public API:
    SkewFetcher(handler)                         — create fetcher with open handler
    await fetcher.fetch_skew_for_maturity(mat)   — returns float or raises

    fetch_skew_once(maturity_label)              — convenience: opens its own handler
"""

import asyncio
import re
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

from deribit_api import (
    CachedDeribitApiHandler,
    ConnectionPoolConfig,
    ResultParser,
    ListParser,
)


# ---------------------------------------------------------------------------
# Utility helpers (from scaledVega_delta25skew_new.py)
# ---------------------------------------------------------------------------

def _extract_strike(ticker: str) -> int:
    """Parse strike price from option ticker string, e.g. BTC-20MAR26-70000-C -> 70000."""
    return int(ticker.split("-")[-2])


def _extract_maturity(ticker: str) -> str:
    """Parse maturity label from option ticker, e.g. BTC-20MAR26-70000-C -> 20MAR26."""
    return ticker.split("-")[1]


def _days_to_expiration(maturity_str: str) -> int:
    """Convert maturity string like '20MAR26' to integer days from today."""
    match = re.match(r"(\d{1,2})([A-Z]{3})(\d{2})", maturity_str)
    if not match:
        raise ValueError(f"Invalid maturity format: {maturity_str!r}")
    day, month_str, year = match.groups()
    month_map = {
        "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4,
        "MAY": 5, "JUN": 6, "JUL": 7, "AUG": 8,
        "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
    }
    month = month_map.get(month_str)
    if not month:
        raise ValueError(f"Invalid month: {month_str!r}")
    expiry_date = datetime(2000 + int(year), month, int(day))
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    return (expiry_date - today).days


# ---------------------------------------------------------------------------
# SkewFetcher — reuses an externally managed CachedDeribitApiHandler
# ---------------------------------------------------------------------------

class SkewFetcher:
    """
    Fetches 25-delta skew for a given BTC option maturity.

    Expects a live (entered) CachedDeribitApiHandler.  The caller is
    responsible for the handler lifecycle.
    """

    def __init__(self, handler: CachedDeribitApiHandler) -> None:
        self._handler = handler

    # ------------------------------------------------------------------
    # Low-level data access
    # ------------------------------------------------------------------

    async def _get_instruments(self, currency: str = "BTC") -> List[Dict[str, Any]]:
        parser = ListParser(
            self._handler,
            "public/get_instruments",
            params={"currency": currency, "kind": "option"},
        )
        return await parser.get()

    async def _get_order_book(self, instrument_name: str) -> Dict[str, Any]:
        parser = ResultParser(
            self._handler,
            "public/get_order_book",
            params={"instrument_name": instrument_name, "depth": 5},
        )
        return await parser.get()

    async def _get_option_data(self, instrument_name: str) -> Dict[str, Any]:
        book = await self._get_order_book(instrument_name)
        return {
            "instrument_name": instrument_name,
            "underlying_price": book.get("underlying_price"),
            "mark_iv": book.get("mark_iv"),
            "delta": book.get("greeks", {}).get("delta"),
        }

    async def _get_multiple_option_data(
        self, instruments: List[str]
    ) -> List[Dict[str, Any]]:
        tasks = [self._get_option_data(name) for name in instruments]
        return await asyncio.gather(*tasks)

    async def _group_options_by_maturity(
        self, currency: str = "BTC"
    ) -> Dict[str, Tuple[List[str], List[str]]]:
        """Return {maturity: ([calls], [puts])} sorted by DTE."""
        instruments = await self._get_instruments(currency)
        grouped: Dict[str, Tuple[List[str], List[str]]] = defaultdict(
            lambda: ([], [])
        )
        for inst in instruments:
            name = inst["instrument_name"]
            maturity = _extract_maturity(name)
            if name.endswith("-C"):
                grouped[maturity][0].append(name)
            elif name.endswith("-P"):
                grouped[maturity][1].append(name)
        sorted_maturities = sorted(
            grouped.keys(), key=lambda m: _days_to_expiration(m)
        )
        return {m: grouped[m] for m in sorted_maturities}

    # ------------------------------------------------------------------
    # Core 25-delta skew calculation (mirrors calculate_25_delta_skew)
    # ------------------------------------------------------------------

    async def _calculate_skew(
        self, calls: List[str], puts: List[str]
    ) -> float:
        """
        Calculate 25-delta skew (call IV - put IV) for the given call/put lists.
        Returns the skew value in percentage points.
        Raises on any error.
        """
        if not calls or not puts:
            raise ValueError("Empty calls or puts list")

        # Get underlying price from the first call's order book
        first_call_data = await self._get_option_data(calls[0])
        underlying = first_call_data["underlying_price"]
        if underlying is None or underlying <= 0:
            raise ValueError("Could not determine underlying price")

        # Filter to ~25-delta range
        call_lower, call_upper = underlying, underlying * 1.35
        put_lower, put_upper = underlying * 0.65, underlying

        filtered_calls = [
            t for t in calls if call_lower <= _extract_strike(t) <= call_upper
        ]
        filtered_puts = [
            t for t in puts if put_lower <= _extract_strike(t) <= put_upper
        ]

        if not filtered_calls:
            raise ValueError("No calls found in 25-delta range")
        if not filtered_puts:
            raise ValueError("No puts found in 25-delta range")

        call_data_list, put_data_list = await asyncio.gather(
            self._get_multiple_option_data(filtered_calls),
            self._get_multiple_option_data(filtered_puts),
        )

        # Option with delta closest to +0.25
        closest_call = min(
            call_data_list,
            key=lambda x: abs((x["delta"] or 1.0) - 0.25),
        )
        # Option with delta closest to -0.25
        closest_put = min(
            put_data_list,
            key=lambda x: abs(abs(x["delta"] or 0.0) - 0.25),
        )

        call_iv = closest_call.get("mark_iv")
        put_iv = closest_put.get("mark_iv")
        if call_iv is None or put_iv is None:
            raise ValueError("mark_iv missing for closest-delta options")

        return float(call_iv) - float(put_iv)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def fetch_skew_for_maturity(self, maturity_label: str) -> float:
        """
        Fetch 25-delta skew for the given maturity label (e.g. '20MAR26').

        Fetches all BTC options, finds calls and puts for the requested
        maturity, then computes skew = call_iv - put_iv at the 25-delta strike.

        Returns:
            float — skew in percentage points (positive = calls pricier)

        Raises:
            ValueError — if maturity not found or data unavailable
            Any aiohttp / network exception on connectivity failure
        """
        grouped = await self._group_options_by_maturity("BTC")

        if maturity_label not in grouped:
            available = list(grouped.keys())
            raise ValueError(
                f"Maturity {maturity_label!r} not found on Deribit. "
                f"Available: {available[:8]}"
            )

        calls, puts = grouped[maturity_label]
        return await self._calculate_skew(calls, puts)


# ---------------------------------------------------------------------------
# Convenience wrapper — creates its own handler session
# ---------------------------------------------------------------------------

async def fetch_skew_once(maturity_label: str, currency: str = "BTC") -> float:
    """
    One-shot skew fetch that manages its own Deribit handler session.

    Useful for testing.  For repeated calls from the GUI, prefer creating a
    single SkewFetcher with a shared CachedDeribitApiHandler.
    """
    pool_config = ConnectionPoolConfig(limit=50, limit_per_host=20, keepalive_timeout=60)
    async with CachedDeribitApiHandler(
        testnet=False,
        cache_ttl_sec=300,
        pool_config=pool_config,
        rate_limit=20,
    ) as handler:
        fetcher = SkewFetcher(handler)
        return await fetcher.fetch_skew_for_maturity(maturity_label)


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    async def _main():
        mat = sys.argv[1] if len(sys.argv) > 1 else None
        if mat is None:
            # Default: fetch the nearest maturity's skew
            pool_config = ConnectionPoolConfig(limit=50, limit_per_host=20)
            async with CachedDeribitApiHandler(
                testnet=False, cache_ttl_sec=300, pool_config=pool_config, rate_limit=20
            ) as handler:
                fetcher = SkewFetcher(handler)
                grouped = await fetcher._group_options_by_maturity("BTC")
                mat = next(iter(grouped))
                print(f"Using nearest maturity: {mat}")

        print(f"Fetching 25-delta skew for {mat}...")
        skew = await fetch_skew_once(mat)
        print(f"Skew ({mat}): {skew:+.2f}%  (call_iv - put_iv)")

    asyncio.run(_main())
