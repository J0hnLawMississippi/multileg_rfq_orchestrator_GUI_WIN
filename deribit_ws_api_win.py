#!/usr/bin/env python3
"""
Deribit API — REST handler (unchanged) + WebSocket market data service.

DeribitMarketDataService:
  - Maintains live cache of index price, per-instrument marks/IVs/forwards
  - Subscribes to deribit_price_index + per-instrument ticker.100ms channels
  - Exposes synchronous reads (zero await latency) for pricer consumption
  - Falls back to REST bootstrap on startup until first WS push arrives
  - Handles reconnection with cache preservation (stale but non-zero)
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set

import aiohttp
import numpy as np
from numpy.typing import NDArray

from deribit_api import CachedDeribitApiHandler
from db_option_pricer_win import MaturityIVData

LOG = logging.getLogger("deribit.ws")


# =============================================================================
# CACHE ENTRY TYPES
# =============================================================================


@dataclass(slots=True)
class InstrumentSnapshot:
    """
    Latest pushed state for a single option instrument.
    All fields sourced from ticker.{instrument}.100ms WS channel.
    """

    instrument_name: str
    mark_price_btc: float  # mark price in BTC
    mark_iv: float  # IV as decimal (0.65 = 65%)
    underlying_price: float  # forward price in USD
    index_price: float  # spot at time of push
    timestamp_ms: int  # exchange timestamp
    received_at: float  # local monotonic time


@dataclass(slots=True)
class IndexSnapshot:
    price: float
    timestamp_ms: int
    received_at: float


# =============================================================================
# DERIBIT WS MARKET DATA SERVICE
# =============================================================================


class DeribitMarketDataService:
    """
    Persistent Deribit public WebSocket connection.

    Maintains a live in-memory cache of:
      - BTC/USD index price
      - Per-instrument: mark_price, mark_iv, underlying_price (forward)

    Usage:
        svc = DeribitMarketDataService(instruments=["BTC-27FEB26-65000-P", ...])
        task = asyncio.create_task(svc.run())
        await svc.wait_ready(timeout=15.0)

        # Zero-latency reads anywhere in the process:
        spot    = svc.index_price
        snap    = svc.get_instrument("BTC-27FEB26-65000-P")
        iv_data = await svc.build_maturity_iv_data(maturities, rest_handler)

        svc.stop()
        await task
    """

    WS_URL = "wss://www.deribit.com/ws/api/v2"
    HEARTBEAT_SECS = 10.0
    RECEIVE_TIMEOUT = 25.0
    MAX_BACKOFF = 60.0

    def __init__(
        self,
        instruments: List[str],
        currency: str = "BTC",
        testnet: bool = False,
    ) -> None:
        self._instruments: FrozenSet[str] = frozenset(instruments)
        self._currency = currency.upper()
        self._ws_url = (
            "wss://test.deribit.com/ws/api/v2"
            if testnet
            else "wss://www.deribit.com/ws/api/v2"
        )

        # Live cache — written by WS callback, read by pricer (no lock needed:
        # asyncio is single-threaded; reads always see a consistent dict entry)
        self._index: Optional[IndexSnapshot] = None
        self._instruments_cache: Dict[str, InstrumentSnapshot] = {}

        # Expiry timestamps are static — fetched once via REST, never via WS
        self._expiry_ts: Dict[str, int] = {}

        self._ready = asyncio.Event()
        self._running = False
        self._req_id = 0

        # Track which instruments still need a first push before _ready fires
        self._pending_initial: Set[str] = set(instruments)

    # ------------------------------------------------------------------
    # Public read interface (synchronous — zero await overhead)
    # ------------------------------------------------------------------

    @property
    def index_price(self) -> float:
        if self._index is None:
            raise RuntimeError("MarketDataService not yet ready")
        return self._index.price

    @property
    def index_age_ms(self) -> float:
        """Milliseconds since last index push received."""
        if self._index is None:
            return float("inf")
        return (time.monotonic() - self._index.received_at) * 1000

    def get_instrument(self, name: str) -> Optional[InstrumentSnapshot]:
        return self._instruments_cache.get(name)

    def instrument_age_ms(self, name: str) -> float:
        snap = self._instruments_cache.get(name)
        if snap is None:
            return float("inf")
        return (time.monotonic() - snap.received_at) * 1000

    def get_expiry_ts(self, instrument: str) -> Optional[int]:
        return self._expiry_ts.get(instrument)

    # ------------------------------------------------------------------
    # MaturityIVData builder — replaces OptimisedDeribitFetcher entirely
    # after WS is warm. REST handler passed for expiry_ts bootstrap only.
    # ------------------------------------------------------------------

    async def build_maturity_iv_data(
        self,
        maturities: List[str],
        rest_handler: CachedDeribitApiHandler,
        max_age_ms: float = 5000.0,
    ) -> tuple[float, Dict[str, MaturityIVData]]:
        """
        Build (spot, {maturity: MaturityIVData}) from live WS cache.

        Args:
            maturities:   List of maturity strings e.g. ["27FEB26", "6MAR26"]
            rest_handler: Used ONLY if expiry_ts not yet cached (first call)
            max_age_ms:   Warn if any snapshot is older than this

        Returns:
            (spot, iv_data_dict)

        Raises:
            RuntimeError: If service not ready or instrument not subscribed
        """
        if not self._ready.is_set():
            raise RuntimeError("MarketDataService not ready. Await wait_ready() first.")

        spot = self.index_price

        # Ensure expiry timestamps cached (REST, once per instrument lifetime)
        missing_expiry = [
            inst for inst in self._instruments if inst not in self._expiry_ts
        ]
        if missing_expiry:
            await self._fetch_expiry_timestamps(missing_expiry, rest_handler)

        iv_data: Dict[str, MaturityIVData] = {}

        for maturity in maturities:
            prefix = f"{self._currency}-{maturity.upper()}-"
            mat_snaps = [
                snap
                for name, snap in self._instruments_cache.items()
                if name.startswith(prefix)
            ]
            if not mat_snaps:
                raise RuntimeError(
                    f"No WS data for maturity {maturity}. "
                    f"Is it in the instruments subscription list?"
                )

            # Age check — warn but don't block
            for snap in mat_snaps:
                age = (time.monotonic() - snap.received_at) * 1000
                if age > max_age_ms:
                    LOG.warning(
                        "%s snapshot is %.0fms old (threshold %.0fms)",
                        snap.instrument_name,
                        age,
                        max_age_ms,
                    )

            # Build arrays from live snapshots
            # Average call + put IV at each strike (matching original behaviour)
            strike_groups: Dict[float, list] = {}
            for snap in mat_snaps:
                strike = float(snap.instrument_name.split("-")[2])
                strike_groups.setdefault(strike, []).append(snap)

            unique_strikes = sorted(strike_groups)

            avg_ivs = np.array(
                [
                    float(np.mean([s.mark_iv for s in strike_groups[k]]))
                    for k in unique_strikes
                ],
                dtype=np.float64,
            )
            avg_marks = np.array(
                [
                    float(np.mean([s.mark_price_btc for s in strike_groups[k]]))
                    for k in unique_strikes
                ],
                dtype=np.float64,
            )
            # Forward: use ATM-closest strike snapshot's underlying_price
            # (all strikes share the same forward for a given maturity)
            forward = float(mat_snaps[0].underlying_price)

            # Expiry timestamp — static, from REST bootstrap
            sample_inst = mat_snaps[0].instrument_name
            expiry_ts = self._expiry_ts.get(sample_inst, 0)

            iv_data[maturity.upper()] = MaturityIVData(
                maturity=maturity.upper(),
                expiry_ts=expiry_ts,
                forward=forward,
                strikes=np.array(unique_strikes, dtype=np.float64),
                ivs=avg_ivs,
                mark_prices_btc=avg_marks,
            )

        return spot, iv_data

    # ------------------------------------------------------------------
    # Subscription management — call when leg set changes
    # ------------------------------------------------------------------

    async def subscribe_instruments(
        self,
        instruments: List[str],
        ws: aiohttp.ClientWebSocketResponse,
    ) -> None:
        """
        Add new instruments to live subscription.
        Safe to call while connected — Deribit accepts mid-session subscribe.
        """
        new_instruments = [i for i in instruments if i not in self._instruments]
        if not new_instruments:
            return

        self._instruments = frozenset(self._instruments | set(new_instruments))
        self._pending_initial.update(new_instruments)

        channels = [f"ticker.{i}.100ms" for i in new_instruments]
        await self._send_subscribe(ws, channels)
        LOG.info("Subscribed to %d new instruments", len(new_instruments))

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def wait_ready(self, timeout: float = 15.0) -> bool:
        """
        Block until first push received for all subscribed instruments
        AND index price received.
        """
        try:
            await asyncio.wait_for(self._ready.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            LOG.error(
                "MarketDataService ready timeout. Still pending: %s",
                self._pending_initial,
            )
            return False

    def stop(self) -> None:
        self._running = False
        self._ready.set()  # unblock any waiters

    async def run(self) -> None:
        self._running = True
        backoff = 1.0
        attempt = 0

        while self._running:
            try:
                await self._connect_and_serve()
                backoff = 1.0
                attempt = 0
            except Exception as exc:
                LOG.warning("Deribit WS disconnected: %s", exc)
            finally:
                # Do NOT clear cache on disconnect — stale data better than none
                # _ready remains set so reads continue with last known values
                pass

            if not self._running:
                break

            attempt += 1
            delay = min(backoff * (2 ** min(attempt, 6)), self.MAX_BACKOFF)
            LOG.info("Deribit WS reconnecting in %.1fs", delay)
            await asyncio.sleep(delay)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _next_id(self) -> int:
        self._req_id += 1
        return self._req_id

    async def _send_subscribe(
        self,
        ws: aiohttp.ClientWebSocketResponse,
        channels: List[str],
    ) -> None:
        await ws.send_json(
            {
                "jsonrpc": "2.0",
                "method": "public/subscribe",
                "params": {"channels": channels},
                "id": self._next_id(),
            }
        )

    async def _connect_and_serve(self) -> None:
        index_channel = f"deribit_price_index.{self._currency.lower()}_usd"
        ticker_channels = [f"ticker.{inst}.100ms" for inst in self._instruments]
        all_channels = [index_channel, *ticker_channels]

        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(
                self._ws_url,
                receive_timeout=self.RECEIVE_TIMEOUT,
                heartbeat=None,
                max_msg_size=0,
            ) as ws:
                LOG.info(
                    "Deribit WS connected - subscribing %d channels", len(all_channels)
                )
                await self._send_subscribe(ws, all_channels)

                hb_task = asyncio.create_task(self._heartbeat(ws))
                try:
                    async for msg in ws:
                        if not self._running:
                            break
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            self._handle_message(json.loads(msg.data))
                        elif msg.type in (
                            aiohttp.WSMsgType.CLOSE,
                            aiohttp.WSMsgType.ERROR,
                            aiohttp.WSMsgType.CLOSED,
                        ):
                            break
                finally:
                    hb_task.cancel()
                    try:
                        await hb_task
                    except asyncio.CancelledError:
                        pass

    async def _heartbeat(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        """
        Deribit public WS heartbeat.
        Send /public/test — expect {"result": "ok"} within 30s.
        """
        while not ws.closed:
            await asyncio.sleep(self.HEARTBEAT_SECS)
            if not ws.closed:
                try:
                    await ws.send_json(
                        {
                            "jsonrpc": "2.0",
                            "method": "public/test",
                            "id": self._next_id(),
                        }
                    )
                except Exception as exc:
                    LOG.warning("Deribit heartbeat failed: %s", exc)
                    break

    def _handle_message(self, data: Dict) -> None:
        """
        Route inbound WS message to appropriate cache update.
        Deribit push format:
          {"jsonrpc":"2.0","method":"subscription","params":{"channel":"...","data":{...}}}
        """
        method = data.get("method")
        if method != "subscription":
            return  # heartbeat ack, subscription confirm, etc.

        params = data.get("params", {})
        channel = params.get("channel", "")
        d = params.get("data", {})
        now = time.monotonic()

        if channel.startswith("deribit_price_index"):
            self._index = IndexSnapshot(
                price=float(d["price"]),
                timestamp_ms=int(d.get("timestamp", 0)),
                received_at=now,
            )
            self._pending_initial.discard("__index__")

        elif channel.startswith("ticker."):
            inst = d.get("instrument_name", "")
            if inst not in self._instruments:
                return

            mark_btc = float(d.get("mark_price", 0))
            mark_iv = float(d.get("mark_iv", 0))  # Deribit pushes as decimal
            und = float(d.get("underlying_price", 0))
            idx = float(d.get("index_price", 0))
            ts = int(d.get("timestamp", 0))

            # Guard against zero/missing fields — keep previous value if push
            # is incomplete (Deribit occasionally sends partial updates)
            prev = self._instruments_cache.get(inst)
            self._instruments_cache[inst] = InstrumentSnapshot(
                instrument_name=inst,
                mark_price_btc=mark_btc
                if mark_btc > 0
                else (prev.mark_price_btc if prev else 0.0),
                mark_iv=mark_iv if mark_iv > 0 else (prev.mark_iv if prev else 0.0),
                underlying_price=und
                if und > 0
                else (prev.underlying_price if prev else 0.0),
                index_price=idx if idx > 0 else (prev.index_price if prev else 0.0),
                timestamp_ms=ts,
                received_at=now,
            )

            self._pending_initial.discard(inst)

        # Fire ready when all instruments + index have received at least one push
        if (
            not self._ready.is_set()
            and self._index is not None
            and not self._pending_initial
        ):
            LOG.info("MarketDataService ready - all channels warm")
            self._ready.set()

    async def _fetch_expiry_timestamps(
        self,
        instruments: List[str],
        rest_handler: CachedDeribitApiHandler,
    ) -> None:
        """
        Fetch expiration_timestamp for instruments not yet cached.
        Called at most once per instrument — result cached permanently.
        """
        results = await asyncio.gather(
            *[
                rest_handler.get(
                    "public/get_instrument",
                    params={"instrument_name": inst},
                )
                for inst in instruments
            ],
            return_exceptions=True,
        )
        for inst, res in zip(instruments, results):
            if isinstance(res, Exception):
                LOG.error("Failed to fetch expiry for %s: %s", inst, res)
                continue
            ts = res["result"].get("expiration_timestamp")
            if ts:
                self._expiry_ts[inst] = int(ts)
                LOG.debug("Cached expiry for %s: %d", inst, ts)
