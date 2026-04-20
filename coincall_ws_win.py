#!/usr/bin/env python3
"""
Persistent Coincall WebSocket client.

Single connection, started once at process startup.
RFQ flows register/unregister per-request queues on the shared connection.

Message dispatch (from API docs):
  dt=129  rfqTaker      — our RFQ lifecycle: ACTIVE/CANCELLED/FILLED/TRADED_AWAY
  dt=131  quoteReceived — incoming maker quotes with quoteSide field

Subscriptions:
  rfqTaker      — our RFQ state changes
  quoteReceived — maker quotes directed at us

Heartbeat: send {"action":"heartbeat"} every 10s, expect rc=1 response.
Connection closes server-side if no message for 30s.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Set

import aiohttp

LOG = logging.getLogger("coincall.ws")


# =============================================================================
# MESSAGE TYPE CONSTANTS  (from API docs)
# =============================================================================


class CCMsgType(int, Enum):
    RFQ_TAKER = 129
    QUOTE_RCVD = 131


class RFQState(str, Enum):
    ACTIVE = "ACTIVE"
    CANCELLED = "CANCELLED"
    FILLED = "FILLED"
    TRADED_AWAY = "TRADED_AWAY"


class QuoteState(str, Enum):
    OPEN = "OPEN"
    CANCELLED = "CANCELLED"
    FILLED = "FILLED"


# =============================================================================
# TYPED EVENT CONTAINERS
# =============================================================================


@dataclass(slots=True, frozen=True)
class RFQStateUpdate:
    """dt=129: our RFQ changed state."""

    request_id: str
    state: RFQState
    legs: tuple  # raw leg dicts — immutable snapshot


@dataclass(slots=True, frozen=True)
class QuoteReceived:
    """dt=131: a maker submitted or updated a quote for our RFQ."""

    request_id: str
    quote_id: str
    state: QuoteState
    quote_side: str  # "BUY" | "SELL"
    legs: tuple  # raw leg dicts: instrumentName/price/quantity/side


WSEvent = RFQStateUpdate | QuoteReceived


# =============================================================================
# PERSISTENT WEBSOCKET CLIENT
# =============================================================================


class CoincallWSClient:
    """
    Persistent authenticated WebSocket connection to Coincall options WS.

    Lifecycle:
        client   = CoincallWSClient(ws_url, api_key, api_secret)
        run_task = asyncio.create_task(client.run())
        await client.wait_ready(timeout=15.0)

        # Per-RFQ usage:
        queue = client.register(request_id)
        try:
            # consume queue ...
        finally:
            client.unregister(request_id)

        client.stop()
        await run_task
    """

    _SUBSCRIPTIONS = ("rfqTaker", "quoteReceived")
    _HEARTBEAT_SECS = 10.0  # API docs: send before 30s silence
    _RECEIVE_TIMEOUT = 25.0  # local read timeout — triggers reconnect
    _MAX_BACKOFF = 60.0

    def __init__(self, ws_url: str, api_key: str, api_secret: str) -> None:
        self._ws_url = ws_url
        self._api_key = api_key
        self._api_secret = api_secret

        self._running = False
        self._ready = asyncio.Event()

        # request_id → Queue[WSEvent]
        self._queues: Dict[str, asyncio.Queue[WSEvent]] = {}
        self._stop_event = asyncio.Event()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def is_ready(self) -> bool:
        return self._ready.is_set()

    async def wait_ready(self, timeout: float = 15.0) -> bool:
        try:
            await asyncio.wait_for(self._ready.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            LOG.error("Coincall WS failed to connect within %.0fs", timeout)
            return False

    def register(self, request_id: str) -> asyncio.Queue[WSEvent]:
        """
        Register a typed event queue for request_id.
        Must be called BEFORE the RFQ is submitted so no events are missed.
        """
        q: asyncio.Queue[WSEvent] = asyncio.Queue(maxsize=256)
        self._queues[str(request_id)] = q
        LOG.debug("Queue registered: %s", request_id)
        return q

    def unregister(self, request_id: str) -> None:
        self._queues.pop(str(request_id), None)
        LOG.debug("Queue unregistered: %s", request_id)

    def stop(self) -> None:
        self._running = False
        self._ready.set()
        self._stop_event.set()

    """
    def stop(self) -> None:
        self._running = False
        self._ready.set()   # unblock any wait_ready callers
    """
    # ------------------------------------------------------------------
    # Connection loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        self._running = True
        backoff = 1.0
        attempt = 0

        async with aiohttp.ClientSession() as session:
            while self._running:
                self._stop_event.clear()  # ← reset before each connection attempt
                try:
                    await self._connect_and_serve(session)
                    backoff = 1.0
                    attempt = 0
                except Exception as exc:
                    LOG.warning("Coincall WS error: %s", exc)
                finally:
                    self._ready.clear()

                if not self._running:
                    break

                attempt += 1
                delay = min(backoff * (2 ** min(attempt, 6)), self._MAX_BACKOFF)
                LOG.info(
                    "Coincall WS reconnecting in %.1fs (attempt %d)", delay, attempt
                )
                await asyncio.sleep(delay)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _signed_url(self) -> str:
        ts = str(int(time.time() * 1000))
        prehash = f"GET/users/self/verify?apiKey={self._api_key}&ts={ts}"
        sig = (
            hmac.new(
                self._api_secret.encode(),
                prehash.encode(),
                hashlib.sha256,
            )
            .hexdigest()
            .upper()
        )
        return (
            f"{self._ws_url}"
            f"?code=10&uuid={self._api_key}&ts={ts}"
            f"&sign={sig}&apiKey={self._api_key}"
        )

    async def _connect_and_serve(self, session: aiohttp.ClientSession) -> None:
        url = self._signed_url()
        async with session.ws_connect(
            url,
            receive_timeout=self._RECEIVE_TIMEOUT,
            heartbeat=None,  # manual heartbeat per API spec
            max_msg_size=0,
        ) as ws:
            LOG.info("Coincall WS connected")

            # Subscribe to both channels concurrently
            await asyncio.gather(
                *[
                    ws.send_json({"action": "subscribe", "dataType": dt})
                    for dt in self._SUBSCRIPTIONS
                ]
            )
            self._ready.set()

            hb = asyncio.create_task(self._heartbeat(ws))
            try:
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
                await ws.close()
            """
            hb = asyncio.create_task(self._heartbeat(ws))
            try:
                async for msg in ws:
                    if not self._running:
                        break
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        self._dispatch(json.loads(msg.data))
                    elif msg.type in (
                        aiohttp.WSMsgType.CLOSE,
                        aiohttp.WSMsgType.ERROR,
                        aiohttp.WSMsgType.CLOSED,
                    ):
                        LOG.warning(
                            "Coincall WS closed: type=%s data=%s",
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
                self._dispatch(json.loads(msg.data))
            elif msg.type in (
                aiohttp.WSMsgType.CLOSE,
                aiohttp.WSMsgType.ERROR,
                aiohttp.WSMsgType.CLOSED,
            ):
                LOG.warning(
                    "Coincall WS closed: type=%s data=%s",
                    msg.type,
                    msg.data,
                )
                break

    async def _heartbeat(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        """
        Send heartbeat every 10s.  API responds with rc=1 — discarded in _dispatch.
        If send fails the connection is dead; break to trigger reconnect.
        """
        while not ws.closed:
            await asyncio.sleep(self._HEARTBEAT_SECS)
            if not ws.closed:
                try:
                    await ws.send_json({"action": "heartbeat"})
                    LOG.debug("Heartbeat sent")
                except Exception as exc:
                    LOG.warning("Heartbeat send failed: %s", exc)
                    break

    def _dispatch(self, raw: Dict) -> None:
        """
        Parse and route inbound message to the correct registered queue.

        rc=1 → heartbeat ack, discard.
        dt=129 → RFQStateUpdate
        dt=131 → QuoteReceived
        All other dt values discarded silently.
        """
        if raw.get("rc") == 1:
            return

        dt = raw.get("dt")
        d = raw.get("d")
        if not d:
            return

        request_id = str(d.get("requestId", ""))
        queue = self._queues.get(request_id)
        if not queue:
            return  # not our RFQ

        event: Optional[WSEvent] = None

        if dt == CCMsgType.RFQ_TAKER:
            state_raw = d.get("state", "")
            try:
                state = RFQState(state_raw)
            except ValueError:
                LOG.warning("Unknown RFQ state: %s", state_raw)
                return
            event = RFQStateUpdate(
                request_id=request_id,
                state=state,
                legs=tuple(d.get("legs", [])),
            )

        elif dt == CCMsgType.QUOTE_RCVD:
            state_raw = d.get("state", "")
            try:
                quote_state = QuoteState(state_raw)
            except ValueError:
                LOG.warning("Unknown quote state: %s", state_raw)
                return
            event = QuoteReceived(
                request_id=request_id,
                quote_id=str(d.get("quoteId", "")),
                state=quote_state,
                quote_side=d.get("quoteSide", ""),
                legs=tuple(d.get("legs", [])),
            )

        if event is not None:
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                LOG.error(
                    "Event queue full for request_id=%s - dropping %s",
                    request_id,
                    type(event).__name__,
                )
