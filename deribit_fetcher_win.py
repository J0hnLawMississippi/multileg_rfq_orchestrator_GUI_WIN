#!/usr/bin/env python3
"""
Deribit market data — two operational modes:

Mode 1 (startup / fallback): OptimisedDeribitFetcher
  REST-based, parallelised, used for initial bootstrap and
  expiry timestamp fetch (once per instrument lifetime).

Mode 2 (steady-state): DeribitMarketDataService
  Persistent WebSocket, subscribes to:
    - deribit_price_index.btc_usd
    - ticker.{instrument}.raw  for ±8 STRIKES around each leg strike
      (both C and P at each strike = up to 17 strikes × 2 = 34 instruments)
  Zero REST latency for evaluate_structure() once warm.
  Cache preserved across reconnects.

Instrument enumeration (startup, once):
  GET public/get_instruments — filter by maturity, extract unique strikes
  sorted ascending, slice ±8 STRIKE POSITIONS around each leg strike,
  then select all instruments (C+P) at those strikes. Deduplicates per maturity.

Readiness condition:
  Per maturity: ≥ (2*half_width + 1) unique strikes have received at least
  one valid push (mark_iv > 0, underlying_price > 0).
  Instruments at illiquid far-OTM strikes that never push do NOT block readiness
  provided minimum strike coverage is met.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import aiohttp
import numpy as np
from numpy.typing import NDArray

from deribit_ws_api_win import CachedDeribitApiHandler
from db_option_pricer_win import MaturityIVData, OptionLeg

LOG = logging.getLogger("deribit.fetcher")

SMILE_HALF_WIDTH: int = 8
# Minimum unique strikes with valid pushes required per maturity before ready
_MIN_STRIKES_READY: int = max(4, SMILE_HALF_WIDTH)


# =============================================================================
# CACHE ENTRY TYPES
# =============================================================================


@dataclass(slots=True)
class InstrumentSnapshot:
    """Latest pushed state for a single option instrument."""

    instrument_name: str
    mark_price_btc: float
    mark_iv: float  # decimal (0.65 = 65%)
    underlying_price: float  # forward in USD
    index_price: float  # spot at time of push
    timestamp_ms: int
    received_at: float  # local monotonic


@dataclass(slots=True)
class IndexSnapshot:
    price: float
    timestamp_ms: int
    received_at: float


# =============================================================================
# INSTRUMENT ENUMERATOR  (startup, REST, one-time)
# =============================================================================


async def enumerate_smile_instruments(
    legs: List[OptionLeg],
    handler: CachedDeribitApiHandler,
    currency: str = "BTC",
    half_width: int = SMILE_HALF_WIDTH,
) -> Tuple[FrozenSet[str], Dict[str, NDArray[np.float64]]]:
    """
    For each maturity, fetch all available Deribit instruments, extract
    unique strikes sorted ascending, find the ±half_width strike window
    around each leg strike, then select ALL instruments (C + P) at those
    strikes.  Deduplicates across legs sharing a maturity.

    Returns:
        instruments : frozenset of instrument_name strings (Deribit format)
        maturity_strike_windows : {maturity: sorted array of selected strikes}
                                   used downstream to verify smile coverage

    REST calls: 1 (get_instruments for all BTC options, cached).
    """
    currency = currency.upper()

    resp = await handler.get(
        "public/get_instruments",
        params={"currency": currency, "kind": "option", "expired": "false"},
    )
    all_instruments: List[Dict] = resp["result"]

    # Group by maturity
    maturity_all: Dict[str, List[Dict]] = {}
    for inst in all_instruments:
        parts = inst["instrument_name"].split("-")
        # Deribit format: BTC-DDMMMYY-STRIKE-C/P  → parts[1] is maturity
        if len(parts) == 4:
            maturity_all.setdefault(parts[1], []).append(inst)

    # Group leg strikes by maturity
    maturity_leg_strikes: Dict[str, Set[float]] = {}
    for leg in legs:
        maturity_leg_strikes.setdefault(leg.maturity.upper(), set()).add(leg.strike)

    selected_instruments: Set[str] = set()
    maturity_strike_windows: Dict[str, NDArray[np.float64]] = {}

    for maturity, leg_strikes in maturity_leg_strikes.items():
        mat_insts = maturity_all.get(maturity, [])
        if not mat_insts:
            raise ValueError(
                f"No Deribit instruments found for maturity {maturity}. "
                f"Check maturity string format (expected DDMMMYY e.g. 6MAR26)."
            )

        # Extract unique strikes available at this maturity
        available_strikes_set: Set[float] = {
            float(i["instrument_name"].split("-")[2]) for i in mat_insts
        }
        available_strikes = np.array(sorted(available_strikes_set), dtype=np.float64)

        # For each leg strike, find ±half_width strike positions
        selected_strikes: Set[float] = set()
        for leg_strike in leg_strikes:
            nearest_idx = int(np.argmin(np.abs(available_strikes - leg_strike)))
            lo = max(0, nearest_idx - half_width)
            hi = min(len(available_strikes), nearest_idx + half_width + 1)
            selected_strikes.update(available_strikes[lo:hi].tolist())

        selected_strikes_arr = np.array(sorted(selected_strikes), dtype=np.float64)
        maturity_strike_windows[maturity] = selected_strikes_arr

        # Select ALL instruments (C and P) at the selected strikes
        selected_strike_set = set(selected_strikes_arr.tolist())
        for inst in mat_insts:
            strike = float(inst["instrument_name"].split("-")[2])
            if strike in selected_strike_set:
                selected_instruments.add(inst["instrument_name"])

        LOG.info(
            "%s: %d leg strike(s) -> %d unique strikes selected (+/-%d) "
            "-> %d instruments (C+P)",
            maturity,
            len(leg_strikes),
            len(selected_strikes_arr),
            half_width,
            sum(1 for s in selected_instruments if s.split("-")[1] == maturity),
        )

    LOG.info(
        "Total instruments to subscribe: %d across %d maturities",
        len(selected_instruments),
        len(maturity_leg_strikes),
    )
    return frozenset(selected_instruments), maturity_strike_windows


# =============================================================================
# REST FETCHER  (fallback / bootstrap)
# =============================================================================


async def _fetch_maturity_rest(
    handler: CachedDeribitApiHandler,
    maturity: str,
    currency: str,
    book_summary_list: List[Dict],
) -> MaturityIVData:
    """
    Build MaturityIVData for one maturity from a pre-fetched book_summary.
    Makes exactly 2 parallel REST calls: ticker + get_instrument.
    """
    prefix = f"{currency.upper()}-{maturity.upper()}-"
    filtered = [
        s
        for s in book_summary_list
        if s["instrument_name"].startswith(prefix)
        and s.get("mark_iv")
        and float(s["mark_iv"]) > 0
    ]
    if not filtered:
        raise ValueError(f"No IV data for maturity {maturity}")

    strike_groups: Dict[float, List[Tuple[float, float]]] = {}
    for s in filtered:
        k = float(s["instrument_name"].split("-")[2])
        iv = float(s["mark_iv"]) / 100.0
        mp = float(s.get("mark_price") or 0.0)
        strike_groups.setdefault(k, []).append((iv, mp))

    unique_strikes = sorted(strike_groups)
    avg_ivs = [float(np.mean([v[0] for v in strike_groups[k]])) for k in unique_strikes]
    avg_marks = [
        float(np.mean([v[1] for v in strike_groups[k]])) for k in unique_strikes
    ]

    sample = filtered[0]["instrument_name"]
    ticker_resp, inst_resp = await asyncio.gather(
        handler.get("public/ticker", params={"instrument_name": sample}),
        handler.get("public/get_instrument", params={"instrument_name": sample}),
    )

    underlying = ticker_resp["result"].get("underlying_price")
    if not underlying or float(underlying) <= 0:
        raise ValueError(f"Missing underlying_price for {maturity}")

    return MaturityIVData(
        maturity=maturity.upper(),
        expiry_ts=int(inst_resp["result"]["expiration_timestamp"]),
        forward=float(underlying),
        strikes=np.array(unique_strikes, dtype=np.float64),
        ivs=np.array(avg_ivs, dtype=np.float64),
        mark_prices_btc=np.array(avg_marks, dtype=np.float64),
    )


class OptimisedDeribitFetcher:
    """
    REST-based fetcher. Used during startup before WS is warm
    and as explicit fallback if WS data is stale.
    """

    def __init__(self, handler: CachedDeribitApiHandler) -> None:
        self._handler = handler

    async def fetch_all(
        self,
        maturities: List[str],
        currency: str = "BTC",
    ) -> Tuple[float, Dict[str, MaturityIVData]]:
        """Returns (spot, {maturity: MaturityIVData})."""
        currency = currency.upper()

        index_resp, book_resp = await asyncio.gather(
            self._handler.get(
                "public/get_index_price",
                params={"index_name": f"{currency.lower()}_usd"},
            ),
            self._handler.get(
                "public/get_book_summary_by_currency",
                params={"currency": currency, "kind": "option"},
            ),
        )

        spot = float(index_resp["result"]["index_price"])
        book_summary_list = book_resp["result"]

        iv_data_list = await asyncio.gather(
            *[
                _fetch_maturity_rest(self._handler, m, currency, book_summary_list)
                for m in maturities
            ]
        )

        return spot, {d.maturity: d for d in iv_data_list}


# =============================================================================
# DERIBIT WEBSOCKET MARKET DATA SERVICE
# =============================================================================


class DeribitMarketDataService:
    """
    Persistent Deribit public WebSocket connection.

    Readiness model (replaces all-instruments-must-push):
      Per maturity: ready when ≥ _MIN_STRIKES_READY unique strikes have
      received a valid push (mark_iv > 0, underlying_price > 0).
      Far-OTM instruments that never push are tolerated provided minimum
      smile coverage is available for cubic spline interpolation.

    maturity_strike_windows: the set of strikes we EXPECT data for,
      passed in from enumerate_smile_instruments. Used to track per-maturity
      coverage without hardcoding instrument names.
    """

    WS_URL = "wss://www.deribit.com/ws/api/v2"
    WS_URL_TESTNET = "wss://test.deribit.com/ws/api/v2"
    HEARTBEAT_SECS = 10.0
    RECEIVE_TIMEOUT = 25.0
    MAX_BACKOFF = 60.0

    def __init__(
        self,
        instruments: FrozenSet[str],
        maturity_strike_windows: Dict[str, NDArray[np.float64]],
        currency: str = "BTC",
        testnet: bool = False,
        max_age_ms: float = 5000.0,
        min_strikes_ready: int = _MIN_STRIKES_READY,
    ) -> None:
        self._instruments = instruments
        self._strike_windows = maturity_strike_windows
        self._currency = currency.upper()
        self._ws_url = self.WS_URL_TESTNET if testnet else self.WS_URL
        self._max_age_ms = max_age_ms
        self._min_strikes_ready = min_strikes_ready

        self._index: Optional[IndexSnapshot] = None
        self._snaps: Dict[str, InstrumentSnapshot] = {}
        self._expiry: Dict[str, int] = {}

        # Per-maturity: set of strikes that have received ≥1 valid push
        self._warmed_strikes: Dict[str, Set[float]] = {
            m: set() for m in maturity_strike_windows
        }

        self._ready = asyncio.Event()
        self._running = False
        self._req_id = 0
        self._ws_ref: Optional[aiohttp.ClientWebSocketResponse] = None
        self._stop_event = asyncio.Event()

    # ------------------------------------------------------------------
    # Public synchronous reads
    # ------------------------------------------------------------------

    @property
    def is_ready(self) -> bool:
        return self._ready.is_set()

    @property
    def index_price(self) -> float:
        if self._index is None:
            raise RuntimeError("MarketDataService not yet ready")
        return self._index.price

    def index_age_ms(self) -> float:
        if self._index is None:
            return float("inf")
        return (time.monotonic() - self._index.received_at) * 1000.0

    def get_snapshot(self, instrument: str) -> Optional[InstrumentSnapshot]:
        return self._snaps.get(instrument)

    def snapshot_age_ms(self, instrument: str) -> float:
        snap = self._snaps.get(instrument)
        if snap is None:
            return float("inf")
        return (time.monotonic() - snap.received_at) * 1000.0

    def coverage_report(self) -> Dict[str, Dict]:
        """
        Returns per-maturity coverage summary for diagnostics.
        {maturity: {warmed: int, window: int, ready: bool}}
        """
        return {
            mat: {
                "warmed_strikes": len(self._warmed_strikes.get(mat, set())),
                "window_strikes": len(window),
                "min_required": self._min_strikes_ready,
                "ready": len(self._warmed_strikes.get(mat, set()))
                >= self._min_strikes_ready,
            }
            for mat, window in self._strike_windows.items()
        }

    # ------------------------------------------------------------------
    # MaturityIVData builder
    # ------------------------------------------------------------------

    async def build_maturity_iv_data(
        self,
        maturities: List[str],
        rest_handler: CachedDeribitApiHandler,
    ) -> Tuple[float, Dict[str, MaturityIVData]]:
        """
        Build (spot, {maturity: MaturityIVData}) from live WS cache.
        Only instruments with valid pushes (mark_iv > 0) are included.
        Raises RuntimeError if a maturity has fewer than _MIN_STRIKES_READY
        valid strikes.
        """
        if not self._ready.is_set():
            raise RuntimeError("MarketDataService not ready - call wait_ready() first")

        missing_expiry = [
            inst for inst in self._instruments if inst not in self._expiry
        ]
        if missing_expiry:
            await self._bootstrap_expiry(missing_expiry, rest_handler)

        spot = self.index_price
        iv_data: Dict[str, MaturityIVData] = {}

        for maturity in maturities:
            maturity_upper = maturity.upper()
            prefix = f"{self._currency}-{maturity_upper}-"

            # Only use snapshots with valid IV and forward
            mat_snaps = [
                snap
                for name, snap in self._snaps.items()
                if name.startswith(prefix)
                and snap.mark_iv > 0
                and snap.underlying_price > 0
            ]

            if not mat_snaps:
                raise RuntimeError(f"No valid WS snapshots for maturity {maturity}.")

            # Age warnings
            for snap in mat_snaps:
                age = (time.monotonic() - snap.received_at) * 1000.0
                if age > self._max_age_ms:
                    LOG.warning(
                        "%s is %.0fms old (threshold %.0fms)",
                        snap.instrument_name,
                        age,
                        self._max_age_ms,
                    )

            # Aggregate C + P at each strike
            strike_groups: Dict[float, List[InstrumentSnapshot]] = {}
            for snap in mat_snaps:
                k = float(snap.instrument_name.split("-")[2])
                strike_groups.setdefault(k, []).append(snap)

            n_valid_strikes = len(strike_groups)
            if n_valid_strikes < self._min_strikes_ready:
                raise RuntimeError(
                    f"{maturity}: only {n_valid_strikes} valid strikes in WS cache "
                    f"(minimum {self._min_strikes_ready} required for spline). "
                    f"Coverage: {self.coverage_report().get(maturity_upper)}"
                )

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

            forward = float(mat_snaps[0].underlying_price)
            sample_inst = mat_snaps[0].instrument_name
            expiry_ts = self._expiry.get(sample_inst, 0)
            if expiry_ts == 0:
                LOG.warning("expiry_ts missing for %s", sample_inst)

            iv_data[maturity_upper] = MaturityIVData(
                maturity=maturity_upper,
                expiry_ts=expiry_ts,
                forward=forward,
                strikes=np.array(unique_strikes, dtype=np.float64),
                ivs=avg_ivs,
                mark_prices_btc=avg_marks,
            )

            LOG.info(
                "%s: built MaturityIVData from %d valid strikes "
                "(%d instruments in cache)",
                maturity_upper,
                n_valid_strikes,
                len(mat_snaps),
            )

        return spot, iv_data

    # ------------------------------------------------------------------
    # Subscription management
    # ------------------------------------------------------------------

    async def add_instruments(self, new_instruments: List[str]) -> None:
        truly_new = [i for i in new_instruments if i not in self._instruments]
        if not truly_new:
            return
        self._instruments = frozenset(self._instruments | set(truly_new))
        if self._ws_ref and not self._ws_ref.closed:
            channels = [f"ticker.{i}.100ms" for i in truly_new]
            await self._send(self._ws_ref, "public/subscribe", {"channels": channels})
            LOG.info("Mid-session subscribed to %d new instruments", len(truly_new))
        else:
            LOG.warning(
                "add_instruments called while WS disconnected - "
                "will subscribe on reconnect"
            )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def wait_ready(self, timeout: float = 20.0) -> bool:
        try:
            await asyncio.wait_for(self._ready.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            report = self.coverage_report()
            LOG.error(
                "wait_ready timeout. Coverage report: %s",
                {
                    m: f"{v['warmed_strikes']}/{v['window_strikes']} strikes warmed"
                    for m, v in report.items()
                },
            )
            # Partial readiness: fire ready if ALL maturities meet minimum
            # (some far-OTM instruments simply never push)
            if self._index is not None and all(v["ready"] for v in report.values()):
                LOG.warning(
                    "Partial readiness accepted - some far-OTM instruments "
                    "did not push within timeout but minimum strike coverage met"
                )
                self._ready.set()
                return True
            return False

    def stop(self) -> None:
        self._running = False
        self._ready.set()
        self._stop_event.set()  # ← signals _connect_and_serve to exit

    """
    def stop(self) -> None:
        self._running = False
        self._ready.set()
    """

    async def run(self) -> None:
        self._running = True
        backoff = 1.0
        attempt = 0

        while self._running:
            self._stop_event.clear()  # ← reset before each connection attempt

            try:
                await self._connect_and_serve()
                backoff = 1.0
                attempt = 0
            except Exception as exc:
                LOG.warning("Deribit WS error: %s", exc)
            finally:
                self._ws_ref = None

            if not self._running:
                break

            attempt += 1
            delay = min(backoff * (2 ** min(attempt, 6)), self.MAX_BACKOFF)
            LOG.info("Deribit WS reconnecting in %.1fs (attempt %d)", delay, attempt)
            await asyncio.sleep(delay)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _next_id(self) -> int:
        self._req_id += 1
        return self._req_id

    async def _send(
        self,
        ws: aiohttp.ClientWebSocketResponse,
        method: str,
        params: Dict,
    ) -> None:
        await ws.send_json(
            {
                "jsonrpc": "2.0",
                "method": method,
                "params": params,
                "id": self._next_id(),
            }
        )

    async def _connect_and_serve(self) -> None:
        index_ch = f"deribit_price_index.{self._currency.lower()}_usd"
        ticker_chs = [f"ticker.{i}.100ms" for i in self._instruments]
        channels = [index_ch, *ticker_chs]

        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(
                self._ws_url,
                receive_timeout=self.RECEIVE_TIMEOUT,
                heartbeat=None,
                max_msg_size=0,
            ) as ws:
                self._ws_ref = ws
                LOG.info(
                    "Deribit WS connected - subscribing %d channels "
                    "(%d instruments across %d maturities)",
                    len(channels),
                    len(self._instruments),
                    len(self._strike_windows),
                )
                await self._send(ws, "public/subscribe", {"channels": channels})

                # Reconnect with warm cache: re-set ready immediately
                if self._snaps and self._index:
                    report = self.coverage_report()
                    if all(v["ready"] for v in report.values()):
                        self._ready.set()
                        LOG.info("Reconnected with warm cache - ready immediately")

                hb = asyncio.create_task(self._heartbeat(ws))
                try:
                    # Race the message stream against the stop signal
                    read_task = asyncio.create_task(self._read_loop(ws))
                    await asyncio.wait(
                        [read_task, asyncio.create_task(self._stop_event.wait())],
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    read_task.cancel()
                    try:
                        await read_task
                    except asyncio.CancelledError:
                        pass
                finally:
                    hb.cancel()
                    try:
                        await hb
                    except asyncio.CancelledError:
                        pass
                    await ws.close()  # ← explicit close, immediate exit
                """    
                hb = asyncio.create_task(self._heartbeat(ws))
                try:
                    async for msg in ws:
                        if not self._running:
                            break
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            self._on_message(json.loads(msg.data))
                        elif msg.type in (
                            aiohttp.WSMsgType.CLOSE,
                            aiohttp.WSMsgType.ERROR,
                            aiohttp.WSMsgType.CLOSED,
                        ):
                            LOG.warning(
                                "WS closed: type=%s data=%s",
                                msg.type, msg.data,
                            )
                            break
                finally:
                    hb.cancel()
                    try:
                        await hb
                    except asyncio.CancelledError:
                        pass
                """

    async def _read_loop(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                self._on_message(json.loads(msg.data))
            elif msg.type in (
                aiohttp.WSMsgType.CLOSE,
                aiohttp.WSMsgType.ERROR,
                aiohttp.WSMsgType.CLOSED,
            ):
                LOG.warning("WS closed: type=%s data=%s", msg.type, msg.data)
                break

    async def _heartbeat(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        while not ws.closed:
            await asyncio.sleep(self.HEARTBEAT_SECS)
            if not ws.closed:
                try:
                    await self._send(ws, "public/test", {})
                except Exception as exc:
                    LOG.warning("Heartbeat failed: %s", exc)
                    break

    def _on_message(self, data: Dict) -> None:
        if data.get("method") != "subscription":
            LOG.debug("Non-subscription message: %s", json.dumps(data)[:300])
            return

        params = data.get("params", {})
        channel = params.get("channel", "")
        d = params.get("data", {})
        now = time.monotonic()

        LOG.debug("WS push: channel=%r keys=%s", channel, list(d.keys()))

        if channel.startswith("deribit_price_index"):
            self._index = IndexSnapshot(
                price=float(d["price"]),
                timestamp_ms=int(d.get("timestamp", 0)),
                received_at=now,
            )
            LOG.debug("Index updated: %.2f", self._index.price)

        elif channel.startswith("ticker."):
            inst = d.get("instrument_name", "")
            LOG.debug(
                "Ticker push: inst=%r in_set=%s mark_iv=%s underlying=%s",
                inst,
                inst in self._instruments,
                d.get("mark_iv"),
                d.get("underlying_price"),
            )

            if inst not in self._instruments:
                LOG.debug("Ignoring unsubscribed instrument: %r", inst)
                return

            prev = self._snaps.get(inst)

            def _coerce(key: str, prev_val: float) -> float:
                try:
                    fv = float(d.get(key) or 0)
                    return fv if fv > 0 else prev_val
                except (TypeError, ValueError):
                    return prev_val

            p0 = prev.mark_price_btc if prev else 0.0
            p1 = prev.mark_iv if prev else 0.0
            p2 = prev.underlying_price if prev else 0.0
            p3 = prev.index_price if prev else 0.0

            mark_iv_raw = _coerce("mark_iv", p1 * 100.0)  # prev stored as decimal
            # 100ms channel pushes mark_iv as percentage — convert to decimal
            mark_iv = mark_iv_raw / 100.0 if mark_iv_raw > 1.0 else mark_iv_raw
            underlying_price = _coerce("underlying_price", p2)

            LOG.debug(
                "Coerced: inst=%r mark_iv=%.4f (raw=%.4f) underlying=%.2f",
                inst,
                mark_iv,
                mark_iv_raw,
                underlying_price,
            )

            self._snaps[inst] = InstrumentSnapshot(
                instrument_name=inst,
                mark_price_btc=_coerce("mark_price", p0),
                mark_iv=mark_iv,
                underlying_price=underlying_price,
                index_price=_coerce("index_price", p3),
                timestamp_ms=int(d.get("timestamp", 0)),
                received_at=now,
            )

            # Only count strike as warmed if push has valid IV + forward
            if mark_iv > 0 and underlying_price > 0:
                parts = inst.split("-")
                maturity = parts[1]
                strike = float(parts[2])
                if maturity in self._warmed_strikes:
                    self._warmed_strikes[maturity].add(strike)
                    LOG.debug(
                        "Strike warmed: %s @ %.0f (total warmed: %d/%d)",
                        maturity,
                        strike,
                        len(self._warmed_strikes[maturity]),
                        self._min_strikes_ready,
                    )
                else:
                    LOG.debug(
                        "Maturity %r not in warmed_strikes tracking set: %s",
                        maturity,
                        list(self._warmed_strikes.keys()),
                    )

        else:
            LOG.debug("Unhandled channel: %r", channel)

        self._check_ready()

    def _check_ready(self) -> None:
        """
        Fire _ready when:
          - index has received ≥1 push
          - every tracked maturity has ≥ _min_strikes_ready warmed strikes
        Called after every inbound message — cheap set-size comparison.
        """
        if self._ready.is_set():
            return
        if self._index is None:
            return
        if not self._warmed_strikes:
            return
        if all(
            len(warmed) >= self._min_strikes_ready
            for warmed in self._warmed_strikes.values()
        ):
            total_warmed = sum(len(w) for w in self._warmed_strikes.values())
            LOG.info(
                "MarketDataService ready - %d total strikes warmed across %d maturities",
                total_warmed,
                len(self._warmed_strikes),
            )
            self._ready.set()

    async def _bootstrap_expiry(
        self,
        instruments: List[str],
        rest_handler: CachedDeribitApiHandler,
    ) -> None:
        """
        Fetch expiration_timestamp once per maturity using a single
        representative instrument. All instruments in the same maturity
        share an identical expiry timestamp.
        """
        # One representative instrument per maturity
        maturity_sample: Dict[str, str] = {}
        for inst in instruments:
            parts = inst.split("-")  # BTC-DDMMMYY-STRIKE-C/P
            maturity = parts[1]
            if maturity not in maturity_sample:
                maturity_sample[maturity] = inst

        samples = list(maturity_sample.values())
        LOG.info(
            "Bootstrapping expiry timestamps: %d REST calls for %d maturities "
            "(%d instruments share these timestamps)",
            len(samples),
            len(samples),
            len(instruments),
        )

        results = await asyncio.gather(
            *[
                rest_handler.get(
                    "public/get_instrument",
                    params={"instrument_name": inst},
                )
                for inst in samples
            ],
            return_exceptions=True,
        )

        for inst, res in zip(samples, results):
            if isinstance(res, Exception):
                LOG.error("Expiry fetch failed for %s: %s", inst, res)
                continue
            ts = res["result"].get("expiration_timestamp")
            maturity = inst.split("-")[1]
            if ts:
                # Apply this timestamp to ALL instruments in the same maturity
                for i in instruments:
                    if i.split("-")[1] == maturity:
                        self._expiry[i] = int(ts)
                LOG.debug(
                    "Expiry cached for maturity %s (%d instruments): ts=%d",
                    maturity,
                    sum(1 for i in instruments if i.split("-")[1] == maturity),
                    int(ts),
                )
