#!/usr/bin/env python3
"""
RFQ Orchestrator — full WS market data integration.

Startup sequence:
  1. REST: enumerate smile instruments (±8 strikes per leg, per maturity)
  2. Launch DeribitMarketDataService WS  (index + ticker.100ms / raw subscriptions)
  3. Launch CoincallWSClient WS          (rfqTaker + quoteReceived)
  4. Await both ready concurrently
  5. evaluate_structure() reads from WS cache — zero REST latency
  6. execute_rfq_flow() uses already-connected Coincall WS — zero setup latency

Data flow:
  Deribit WS push → InstrumentSnapshot cache
                  → build_maturity_iv_data() → StrikeIVInterpolator
                  → StructurePricer / VectorizedDrawdownCalculator
                  → QuoteValidator (deribit_marks_usd from WS snapshots)

Coincall WS push → typed WSEvent queue per request_id
                 → _process_quotes() typed dispatch
                 → CoincallREST.execute_trade() on valid quote
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import hmac
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Final, FrozenSet, List, Optional, Tuple
from urllib.parse import urlencode

import aiohttp
from dotenv import load_dotenv

from deribit_ws_api_win import CachedDeribitApiHandler
from deribit_fetcher_win import (
    DeribitMarketDataService,
    OptimisedDeribitFetcher,
    enumerate_smile_instruments,
    SMILE_HALF_WIDTH,
)
from coincall_ws_win import (
    CoincallWSClient,
    QuoteReceived,
    QuoteState,
    RFQState,
    RFQStateUpdate,
    WSEvent,
)
from db_option_pricer_win import (
    Direction,
    OptionLeg,
    StructurePricer,
    StrikeIVInterpolator,
    VectorizedDrawdownCalculator,
    parse_legs,
)

LOG = logging.getLogger("rfq.orchestrator")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


# =============================================================================
# CONFIG
# =============================================================================


class ThresholdType(str, Enum):
    CREDIT = "CREDIT"
    DEBIT = "DEBIT"


@dataclass
class RFQConfig:
    target_spot: float = 70000.0
    threshold_type: ThresholdType = ThresholdType.CREDIT
    threshold_value: float = 250.0
    max_dd_usd_flat: float = -5000.0
    max_dd_usd_shocked: float = -8000.0
    drawdown_days: int = 7
    vol_shock: float = -0.10
    spot_range_pct: float = 0.50
    spot_grid_points: int = 300
    max_slippage_percent: float = 2.5
    price_deviation_threshold: float = 0.05
    max_leg_price_usd: float = 2000.0
    rfq_timeout_seconds: float = 30.0
    coincall_base_url: str = "https://api.coincall.com"
    coincall_ws_url: str = "wss://ws.coincall.com/options"
    rate_limit_per_second: int = 20
    # WS data freshness: warn if snapshot older than this before validating
    max_snapshot_age_ms: float = 5000.0

    @property
    def expected_quote_side(self) -> str:
        return "BUY" if self.threshold_type == ThresholdType.CREDIT else "SELL"

    def validate(self) -> List[str]:
        errors: List[str] = []
        if self.threshold_value < 0:
            errors.append("threshold_value must be non-negative")
        if self.max_dd_usd_flat > 0:
            errors.append("max_dd_usd_flat should be negative")
        if self.max_dd_usd_shocked > 0:
            errors.append("max_dd_usd_shocked should be negative")
        if self.drawdown_days < 1:
            errors.append("drawdown_days must be >= 1")
        return errors


# =============================================================================
# SYMBOL CONVERSION
# =============================================================================


def deribit_to_coincall(name: str) -> str:
    parts = name.split("-")
    if parts[0] == "BTC":
        parts[0] = "BTCUSD"
    return "-".join(parts)


def coincall_to_deribit(name: str) -> str:
    parts = name.split("-")
    if parts[0] == "BTCUSD":
        parts[0] = "BTC"
    return "-".join(parts)


# =============================================================================
# RATE LIMITER
# =============================================================================


class AsyncTokenBucket:
    def __init__(self, rate: int, period: float = 1.0) -> None:
        self._rate = rate
        self._period = period
        self._tokens = float(rate)
        self._updated = time.monotonic()
        self._lock = asyncio.Lock()

    async def __aenter__(self) -> "AsyncTokenBucket":
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._updated
            self._tokens = min(
                self._rate,
                self._tokens + elapsed * (self._rate / self._period),
            )
            self._updated = now
            if self._tokens < 1.0:
                wait = (1.0 - self._tokens) * (self._period / self._rate)
                await asyncio.sleep(wait)
            self._tokens -= 1.0
        return self

    async def __aexit__(self, *_: Any) -> None:
        pass


# =============================================================================
# COINCALL REST CLIENT
# =============================================================================


class CoincallREST:
    """
    Minimal Coincall REST client.

    Endpoints:
      POST /open/option/blocktrade/request/create/v1   JSON body
      POST /open/option/blocktrade/request/accept/v1   form-encoded body
      POST /open/option/blocktrade/request/cancel/v1   form-encoded body

    Signature:
      JSON endpoints:  prehash uses JSON-serialised param values
      Form endpoints:  prehash uses raw string param values
                       (matching the actual form body sent)
    """

    def __init__(self, config: RFQConfig, api_key: str, api_secret: str) -> None:
        self._config = config
        self._api_key = api_key
        self._api_secret = api_secret.encode()
        self._bucket = AsyncTokenBucket(config.rate_limit_per_second)
        self._session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self) -> "CoincallREST":
        self._session = aiohttp.ClientSession(
            base_url=self._config.coincall_base_url,
            timeout=aiohttp.ClientTimeout(total=15, connect=5),
            connector=aiohttp.TCPConnector(limit=10, limit_per_host=5),
        )
        return self

    async def __aexit__(self, *_: Any) -> None:
        if self._session:
            await self._session.close()

    # ------------------------------------------------------------------
    # Signature helpers
    # ------------------------------------------------------------------

    def _auth_suffix(self, ts: str) -> str:
        return f"uuid={self._api_key}&ts={ts}&x-req-ts-diff=5000"

    def _sign(self, prehash: str) -> str:
        return (
            hmac.new(
                self._api_secret,
                prehash.encode(),
                hashlib.sha256,
            )
            .hexdigest()
            .upper()
        )

    def _base_headers(self, ts: str, sig: str) -> Dict[str, str]:
        return {
            "X-CC-APIKEY": self._api_key,
            "sign": sig,
            "ts": ts,
            "X-REQ-TS-DIFF": "5000",
        }

    def _json_prehash(self, endpoint: str, payload: Dict, ts: str) -> str:
        parts = [
            f"{k}={json.dumps(v, separators=(',', ':'), sort_keys=True) if isinstance(v, (dict, list)) else v}"
            for k in sorted(payload)
            for v in [payload[k]]
        ]
        return f"POST{endpoint}?{'&'.join(parts)}&{self._auth_suffix(ts)}"

    def _form_prehash(self, endpoint: str, payload: Dict, ts: str) -> str:
        parts = [f"{k}={v}" for k, v in sorted(payload.items())]
        return f"POST{endpoint}?{'&'.join(parts)}&{self._auth_suffix(ts)}"

    # ------------------------------------------------------------------
    # Endpoints
    # ------------------------------------------------------------------

    async def create_rfq(self, legs: List[Dict]) -> Optional[Dict]:
        """POST /open/option/blocktrade/request/create/v1  (JSON body)."""
        # If all legs SELL, flip to BUY per exchange convention
        if all(lg.get("side", "").upper() == "SELL" for lg in legs):
            legs = [
                {
                    "instrumentName": lg["instrumentName"],
                    "side": "BUY",
                    "qty": lg["qty"],
                }
                for lg in legs
            ]

        endpoint = "/open/option/blocktrade/request/create/v1"
        payload = {"legs": legs}
        ts = str(int(time.time() * 1000))
        sig = self._sign(self._json_prehash(endpoint, payload, ts))
        headers = {**self._base_headers(ts, sig), "Content-Type": "application/json"}
        body = json.dumps(payload, separators=(",", ":"), sort_keys=True)

        async with self._bucket:
            async with self._session.post(endpoint, headers=headers, data=body) as r:
                resp = await _parse_response(r)

        if resp.get("code") == 0:
            return resp.get("data")
        LOG.error("create_rfq failed: %s", resp)
        return None

    async def execute_trade(self, request_id: str, quote_id: str) -> Optional[Dict]:
        """POST /open/option/blocktrade/request/accept/v1  (form-encoded)."""
        endpoint = "/open/option/blocktrade/request/accept/v1"
        payload = {"requestId": request_id, "quoteId": quote_id}
        ts = str(int(time.time() * 1000))
        sig = self._sign(self._form_prehash(endpoint, payload, ts))
        headers = {
            **self._base_headers(ts, sig),
            "Content-Type": "application/x-www-form-urlencoded",
        }

        async with self._bucket:
            async with self._session.post(
                endpoint, headers=headers, data=urlencode(payload)
            ) as r:
                resp = await _parse_response(r)

        if resp.get("code") == 0:
            return resp.get("data")
        LOG.error("execute_trade failed: %s", resp)
        return None

    async def cancel_rfq(self, request_id: str) -> bool:
        """POST /open/option/blocktrade/request/cancel/v1  (form-encoded)."""
        endpoint = "/open/option/blocktrade/request/cancel/v1"
        payload = {"requestId": request_id}
        ts = str(int(time.time() * 1000))
        sig = self._sign(self._form_prehash(endpoint, payload, ts))
        headers = {
            **self._base_headers(ts, sig),
            "Content-Type": "application/x-www-form-urlencoded",
        }

        async with self._bucket:
            async with self._session.post(
                endpoint, headers=headers, data=urlencode(payload)
            ) as r:
                resp = await _parse_response(r)

        if resp.get("code") == 0:
            LOG.info("RFQ cancelled: %s", request_id)
            return True
        LOG.warning("cancel_rfq failed: %s", resp)
        return False


async def _parse_response(r: aiohttp.ClientResponse) -> Dict:
    text = await r.text()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"code": -1, "msg": f"non-JSON: {text[:200]}"}


# =============================================================================
# QUOTE VALIDATOR  (WS-sourced marks via InstrumentSnapshot)
# =============================================================================


class QuoteValidator:
    """
    Validates incoming QuoteReceived events against:
      - quoteSide filter
      - per-leg price deviation from Deribit WS mark
      - net structure value threshold
      - total slippage vs Deribit mark
      - max leg price cap (debit legs)

    deribit_marks_usd: Deribit instrument name → mark in USD (unsigned, per unit)
    Sourced directly from InstrumentSnapshot.mark_price_btc × spot at quote time.
    """

    def __init__(
        self,
        config: RFQConfig,
        deribit_marks_usd: Dict[str, float],
    ) -> None:
        self._config = config
        self._marks = deribit_marks_usd

    def validate(self, quote: QuoteReceived) -> Tuple[bool, str, float]:
        """
        Returns (is_valid, reason, net_value_usd).
        net_value_usd > 0 → credit received, < 0 → debit paid.
        """
        if quote.quote_side != self._config.expected_quote_side:
            return False, f"Wrong quoteSide={quote.quote_side!r}", 0.0

        if not quote.legs:
            return False, "No legs in quote", 0.0

        rejections: List[str] = []
        total_quote = 0.0
        total_mark = 0.0
        dev = self._config.price_deviation_threshold

        for leg in quote.legs:
            cc_inst = leg.get("instrumentName", "")
            db_inst = coincall_to_deribit(cc_inst)
            price = float(leg.get("price", 0))
            qty = float(leg.get("quantity", 0))
            side = leg.get("side", "").upper()

            # Maker BUY = we receive → credit sign +1
            # Maker SELL = we pay   → debit  sign -1
            our_sign = 1.0 if side == "BUY" else -1.0
            mark = self._marks.get(db_inst, 0.0)

            if our_sign < 0 and price > self._config.max_leg_price_usd:
                rejections.append(
                    f"{cc_inst}: ask {price:.2f} > cap {self._config.max_leg_price_usd:.2f}"
                )

            if mark > 0:
                if our_sign > 0 and price < mark * (1.0 - dev):
                    rejections.append(
                        f"{cc_inst}: bid {price:.2f} < floor {mark * (1.0 - dev):.2f}"
                    )
                elif our_sign < 0 and price > mark * (1.0 + dev):
                    rejections.append(
                        f"{cc_inst}: ask {price:.2f} > ceiling {mark * (1.0 + dev):.2f}"
                    )

            total_quote += price * qty * our_sign
            total_mark += mark * qty * our_sign

        cfg = self._config
        if cfg.threshold_type == ThresholdType.CREDIT:
            if total_quote < cfg.threshold_value:
                rejections.append(
                    f"Credit {total_quote:.2f} < required {cfg.threshold_value:.2f}"
                )
        else:
            if total_quote < -cfg.threshold_value:
                rejections.append(
                    f"Debit {-total_quote:.2f} > max {cfg.threshold_value:.2f}"
                )

        if total_mark != 0.0:
            slip_limit = abs(total_mark) * (cfg.max_slippage_percent / 100.0)
            if (total_quote - total_mark) < -slip_limit:
                rejections.append(
                    f"Slippage {total_quote - total_mark:.2f} "
                    f"exceeds limit {-slip_limit:.2f}"
                )

        ok = not rejections
        return ok, "; ".join(rejections) if rejections else "OK", total_quote


# =============================================================================
# STRUCTURE EVALUATION  (WS cache path)
# =============================================================================


@dataclass
class StructureEvaluation:
    legs: List[OptionLeg]
    current_spot: float
    target_spot: float
    total_btc: float
    total_usd: float
    per_leg_btc: List[float]
    per_leg_usd: List[float]
    per_leg_iv: List[float]
    per_leg_forward: List[float]
    max_dd_flat: float
    worst_spot_flat: float
    max_dd_shocked: float
    worst_spot_shocked: float
    threshold_passed: bool
    threshold_reason: str
    dd_flat_passed: bool
    dd_shocked_passed: bool
    all_passed: bool
    # Unsigned USD mark per instrument — fed to QuoteValidator
    deribit_marks_usd: Dict[str, float] = field(default_factory=dict)


async def evaluate_structure(
    leg_specs: List[str],
    config: RFQConfig,
    mds: DeribitMarketDataService,
    rest_handler: CachedDeribitApiHandler,
) -> StructureEvaluation:
    """
    Price structure and evaluate thresholds.

    Market data sourced from DeribitMarketDataService WS cache (~0ms).
    REST handler used only for expiry_ts bootstrap on first call.

    Falls back to OptimisedDeribitFetcher REST if WS not ready.
    """
    try:
        parsed_legs = parse_legs(leg_specs)
    except ValueError as exc:
        raise ValueError(
            "Leg validation failed. Please fix the leg format and values "
            f"(size > 0, strike > 0, valid DDMMMYY maturity): {exc}"
        ) from exc
    maturities = list({lg.maturity for lg in parsed_legs})

    if mds.is_ready:
        current_spot, iv_data = await mds.build_maturity_iv_data(
            maturities, rest_handler
        )
        LOG.info("Evaluation using WS cache (spot=%.2f)", current_spot)
    else:
        LOG.warning("WS not ready - falling back to REST fetcher")
        fetcher = OptimisedDeribitFetcher(rest_handler)
        current_spot, iv_data = await fetcher.fetch_all(maturities)

    for mat, d in iv_data.items():
        LOG.info(
            "  %s: fwd=%.2f basis=%+.2f strikes=%d",
            mat,
            d.forward,
            d.forward - current_spot,
            len(d.strikes),
        )

    iv_interps = {m: StrikeIVInterpolator(d) for m, d in iv_data.items()}
    pricer = StructurePricer(iv_interps, iv_data, current_spot, r=0.0)
    result = pricer.price_structure(parsed_legs, config.target_spot)

    n = len(parsed_legs)
    per_leg_btc = [
        float(result.leg_prices_btc[i] / parsed_legs[i].size) for i in range(n)
    ]
    per_leg_usd = [
        float(result.leg_prices_usd[i] / parsed_legs[i].size) for i in range(n)
    ]

    # Build mark map from WS snapshots where available, fall back to pricer
    deribit_marks_usd: Dict[str, float] = {}
    spot_for_marks = current_spot
    for i, leg in enumerate(parsed_legs):
        snap = mds.get_snapshot(leg.instrument_name)
        if snap is not None and snap.mark_price_btc > 0:
            # Convert BTC mark to USD using WS index_price for consistency
            deribit_marks_usd[leg.instrument_name] = (
                snap.mark_price_btc * snap.index_price
            )
        else:
            deribit_marks_usd[leg.instrument_name] = abs(per_leg_usd[i])

    dd_calc = VectorizedDrawdownCalculator(
        iv_interpolators=iv_interps,
        maturity_data=iv_data,
        current_spot=config.target_spot,
        legs=parsed_legs,
        original_spot=current_spot,
        r=0.0,
        spot_range_pct=config.spot_range_pct,
        spot_grid_points=config.spot_grid_points,
        vol_shock=config.vol_shock,
    )
    dd_flat, dd_shock = dd_calc.compute(days_forward=config.drawdown_days)

    total_usd = result.total_usd
    if config.threshold_type == ThresholdType.CREDIT:
        effective = -total_usd
        passed = effective >= config.threshold_value
        reason = (
            f"Credit {effective:.2f} >= {config.threshold_value:.2f}"
            if passed
            else f"Credit {effective:.2f} < {config.threshold_value:.2f}"
        )
    else:
        effective = total_usd
        passed = effective <= config.threshold_value
        reason = (
            f"Debit {effective:.2f} <= {config.threshold_value:.2f}"
            if passed
            else f"Debit {effective:.2f} > {config.threshold_value:.2f}"
        )

    dd_flat_ok = dd_flat.max_drawdown_usd >= config.max_dd_usd_flat
    dd_shocked_ok = dd_shock.max_drawdown_usd >= config.max_dd_usd_shocked

    return StructureEvaluation(
        legs=parsed_legs,
        current_spot=current_spot,
        target_spot=config.target_spot,
        total_btc=result.total_btc,
        total_usd=total_usd,
        per_leg_btc=per_leg_btc,
        per_leg_usd=per_leg_usd,
        per_leg_iv=[float(v) * 100 for v in result.leg_ivs],
        per_leg_forward=[float(v) for v in result.leg_forwards],
        max_dd_flat=dd_flat.max_drawdown_usd,
        worst_spot_flat=dd_flat.worst_spot_usd,
        max_dd_shocked=dd_shock.max_drawdown_usd,
        worst_spot_shocked=dd_shock.worst_spot_usd,
        threshold_passed=passed,
        threshold_reason=reason,
        dd_flat_passed=dd_flat_ok,
        dd_shocked_passed=dd_shocked_ok,
        all_passed=passed and dd_flat_ok and dd_shocked_ok,
        deribit_marks_usd=deribit_marks_usd,
    )


# =============================================================================
# QUOTE PROCESSING LOOP
# =============================================================================


async def _process_quotes(
    rest: CoincallREST,
    request_id: str,
    queue: asyncio.Queue[WSEvent],
    validator: QuoteValidator,
    config: RFQConfig,
    mds: DeribitMarketDataService,
    parsed_legs: List[OptionLeg],
    current_spot: float,
) -> bool:
    """
    Drain event queue until valid quote is executed, RFQ terminates, or timeout.

    On each incoming quote, refreshes deribit_marks_usd from live WS snapshots
    before validation — ensures marks are current at the moment of the quote,
    not from when evaluate_structure() ran.
    """
    deadline = time.monotonic() + config.rfq_timeout_seconds
    LOG.info("Awaiting quotes (timeout=%.0fs)...", config.rfq_timeout_seconds)

    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            LOG.info("Quote timeout - cancelling RFQ")
            await rest.cancel_rfq(request_id)
            return False

        try:
            event: WSEvent = await asyncio.wait_for(queue.get(), timeout=remaining)
        except asyncio.TimeoutError:
            LOG.info("Queue timeout - cancelling RFQ")
            await rest.cancel_rfq(request_id)
            return False

        # --- RFQ lifecycle ---
        if isinstance(event, RFQStateUpdate):
            LOG.info("RFQ state -> %s", event.state.value)
            if event.state == RFQState.FILLED:
                return True
            if event.state in (RFQState.CANCELLED, RFQState.TRADED_AWAY):
                return False
            continue  # ACTIVE — keep waiting

        # --- Incoming quote ---
        if isinstance(event, QuoteReceived):
            if event.state != QuoteState.OPEN:
                LOG.debug("Non-OPEN quote %s: %s", event.quote_id, event.state.value)
                continue

            # Refresh marks from WS at the moment the quote arrives
            # This is the key improvement: validation against current market
            live_marks = _build_live_marks(mds, parsed_legs, current_spot, config)
            validator = QuoteValidator(config, live_marks)

            ok, reason, net_val = validator.validate(event)
            if not ok:
                LOG.info("Quote %s rejected: %s", event.quote_id, reason)
                continue

            LOG.info(
                "Valid quote %s: net=%.2f USD — executing",
                event.quote_id,
                net_val,
            )
            result = await rest.execute_trade(request_id, event.quote_id)
            if result:
                LOG.info(
                    "EXECUTED: requestId=%s quoteId=%s",
                    request_id,
                    event.quote_id,
                )
                return True

            LOG.warning(
                "Execution rejected for quote %s — continuing",
                event.quote_id,
            )


def _build_live_marks(
    mds: DeribitMarketDataService,
    parsed_legs: List[OptionLeg],
    current_spot: float,
    config: RFQConfig,
) -> Dict[str, float]:
    """
    Build unsigned USD mark per instrument from live WS snapshots.
    Falls back to zero if snapshot missing (validator will skip dev check).
    Logs a warning if any snapshot exceeds max_snapshot_age_ms.
    """
    marks: Dict[str, float] = {}
    for leg in parsed_legs:
        snap = mds.get_snapshot(leg.instrument_name)
        if snap is not None and snap.mark_price_btc > 0:
            age_ms = mds.snapshot_age_ms(leg.instrument_name)
            if age_ms > config.max_snapshot_age_ms:
                LOG.warning(
                    "Stale snapshot for %s: %.0fms old",
                    leg.instrument_name,
                    age_ms,
                )
            marks[leg.instrument_name] = snap.mark_price_btc * snap.index_price
        else:
            LOG.warning(
                "No WS snapshot for %s — mark validation skipped for this leg",
                leg.instrument_name,
            )
            marks[leg.instrument_name] = 0.0
    return marks


# =============================================================================
# RFQ FLOW
# =============================================================================


async def execute_rfq_flow(
    evaluation: StructureEvaluation,
    config: RFQConfig,
    ws_client: CoincallWSClient,
    rest: CoincallREST,
    mds: DeribitMarketDataService,
) -> bool:
    """
    Submit RFQ and process quotes against live WS marks.
    WS already connected — zero setup latency.
    """
    cc_legs = [
        {
            "instrumentName": deribit_to_coincall(leg.instrument_name),
            "side": "SELL" if leg.direction == Direction.SHORT else "BUY",
            "qty": str(leg.size),
        }
        for leg in evaluation.legs
    ]

    LOG.info("Submitting RFQ:")
    for cl in cc_legs:
        LOG.info("  %s %s %s", cl["side"], cl["qty"], cl["instrumentName"])

    rfq_resp = await rest.create_rfq(cc_legs)
    if not rfq_resp:
        LOG.error("RFQ creation failed")
        return False

    request_id = str(rfq_resp.get("requestId"))
    LOG.info("RFQ created: %s", request_id)

    # Register queue BEFORE any quotes can arrive
    queue = ws_client.register(request_id)
    validator = QuoteValidator(config, evaluation.deribit_marks_usd)

    try:
        return await _process_quotes(
            rest=rest,
            request_id=request_id,
            queue=queue,
            validator=validator,
            config=config,
            mds=mds,
            parsed_legs=evaluation.legs,
            current_spot=evaluation.current_spot,
        )
    finally:
        ws_client.unregister(request_id)


# =============================================================================
# DISPLAY
# =============================================================================


def print_evaluation(ev: StructureEvaluation, cfg: RFQConfig) -> None:
    W = 70
    print(f"\n{'=' * W}")
    print("STRUCTURE EVALUATION")
    print(f"{'=' * W}")
    print(f"  Spot (live WS):  {ev.current_spot:>12,.2f}")
    print(f"  Target spot:     {ev.target_spot:>12,.2f}")
    print()
    print(f"  {'Leg':<32} {'IV':>7} {'Forward':>10} {'BTC':>10} {'USD':>12}")
    print(f"  {'-' * 32} {'-' * 7} {'-' * 10} {'-' * 10} {'-' * 12}")
    for i, leg in enumerate(ev.legs):
        label = f"{leg.direction.value} {leg.size} {leg.instrument_name}"
        print(
            f"  {label:<32} {ev.per_leg_iv[i]:>6.1f}%"
            f" {ev.per_leg_forward[i]:>10.0f}"
            f" {ev.per_leg_btc[i]:>10.4f}"
            f" {ev.per_leg_usd[i]:>12.2f}"
        )
    print(
        f"  {'TOTAL':<32} {'':>7} {'':>10} {ev.total_btc:>10.4f} {ev.total_usd:>12.2f}"
    )
    print(f"\n  THRESHOLD  [{cfg.threshold_type.value}]")
    print(f"  {ev.threshold_reason}")
    print(f"  {'[PASS]' if ev.threshold_passed else '[FAIL]'}")
    print(f"\n  DRAWDOWN  [{cfg.drawdown_days}d, +-{cfg.spot_range_pct * 100:.0f}%]")
    print(
        f"  Vol flat:   {ev.max_dd_flat:>11,.2f} USD @ {ev.worst_spot_flat:>9,.0f}"
        f"  {'[OK]' if ev.dd_flat_passed else '[X]'}  (limit {cfg.max_dd_usd_flat:,.2f})"
    )
    print(
        f"  Vol {cfg.vol_shock * 100:+.0f}%:  {ev.max_dd_shocked:>11,.2f} USD"
        f" @ {ev.worst_spot_shocked:>9,.0f}"
        f"  {'[OK]' if ev.dd_shocked_passed else '[X]'}  (limit {cfg.max_dd_usd_shocked:,.2f})"
    )
    print(f"\n  {'-' * 50}")
    if ev.all_passed:
        print("  [OK] ALL CHECKS PASSED -- RFQ will be sent")
    else:
        print("  [FAIL] CHECKS FAILED -- no RFQ")
        if not ev.threshold_passed:
            print(f"    * Threshold: {ev.threshold_reason}")
        if not ev.dd_flat_passed:
            print(
                f"    * DD flat:    {ev.max_dd_flat:,.2f} < limit {cfg.max_dd_usd_flat:,.2f}"
            )
        if not ev.dd_shocked_passed:
            print(
                f"    * DD shocked: {ev.max_dd_shocked:,.2f} < limit {cfg.max_dd_usd_shocked:,.2f}"
            )
    print(f"  {'-' * 50}")


# =============================================================================
# ENTRY POINT
# =============================================================================

# Credential env-var mapping — single source of truth
_ACCOUNT_CREDENTIALS: Final[Dict[str, Tuple[str, str]]] = {
    "main": ("COINCALL_API_KEY", "COINCALL_API_SECRET"),
    "sub": ("COINCALL_SUB_API_KEY", "COINCALL_SUB_API_SECRET"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RFQ Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Accounts:
  main      Uses COINCALL_API_KEY + COINCALL_API_SECRET
  sub       Uses COINCALL_SUB_API_KEY + COINCALL_SUB_API_SECRET

Examples:
  python rfq_orchestrator_win.py --account sub
  python rfq_orchestrator_win.py --account main
        """,
    )
    parser.add_argument(
        "--account",
        choices=["main", "sub"],
        required=True,
        help="Account to trade on: main or sub (required)",
    )
    return parser.parse_args()


async def main() -> None:
    load_dotenv()

    config = RFQConfig(
        target_spot=74000.0,
        threshold_type=ThresholdType.DEBIT,
        threshold_value=550.0,
        max_dd_usd_flat=-2000.0,
        max_dd_usd_shocked=-1800.0,
        drawdown_days=13,
        vol_shock=-0.10,
        spot_range_pct=0.50,
        spot_grid_points=300,
        rfq_timeout_seconds=1200.0,
        max_slippage_percent=7.5,
        price_deviation_threshold=0.15,
        max_leg_price_usd=7000.0,
        max_snapshot_age_ms=5000.0,
    )

    errors = config.validate()
    if errors:
        LOG.error("Config errors: %s", errors)
        sys.exit(1)

    leg_specs = [
        "L 0.6 20MAR26-70000-P",
        "L 0.6 20MAR26-78000-C",
        "L 0.5 24APR26-70000-P",
        "L 0.5 24APR26-78000-C",
        "S 0.5 29MAY26-70000-P",
        "S 0.5 29MAY26-78000-C",
    ]

    args = parse_args()
    key_var, secret_var = _ACCOUNT_CREDENTIALS[args.account]
    api_key = os.environ.get(key_var)
    api_secret = os.environ.get(secret_var)
    if not api_key or not api_secret:
        LOG.error(
            f"Missing credentials for account '{args.account}'. "
            f"Set {key_var} and {secret_var} in environment or .env file."
        )
        sys.exit(1)
    LOG.info(f"Account: {args.account.upper()} ({key_var})")

    async with CachedDeribitApiHandler(cache_ttl_sec=3600) as deribit_handler:
        # Step 1: enumerate smile instruments
        try:
            parsed_legs = parse_legs(leg_specs)
        except ValueError as exc:
            LOG.error("Leg validation failed at startup: %s", exc)
            sys.exit(1)
        LOG.info("Enumerating smile instruments (+/-%d strikes)...", SMILE_HALF_WIDTH)
        instruments, strike_windows = await enumerate_smile_instruments(
            legs=parsed_legs,
            handler=deribit_handler,
        )

        # Step 2: start both WS connections
        mds = DeribitMarketDataService(
            instruments=instruments,
            maturity_strike_windows=strike_windows,  # ← new parameter
            max_age_ms=config.max_snapshot_age_ms,
        )

        ws_cc = CoincallWSClient(config.coincall_ws_url, api_key, api_secret)

        mds_task = asyncio.create_task(mds.run(), name="deribit-mds")
        wscc_task = asyncio.create_task(ws_cc.run(), name="coincall-ws")

        LOG.info("Waiting for both WS connections to be ready...")
        mds_ready, cc_ready = await asyncio.gather(
            mds.wait_ready(timeout=20.0),
            ws_cc.wait_ready(timeout=15.0),
        )

        if not mds_ready:
            LOG.error("Deribit market data WS failed to become ready")
            mds.stop()
            ws_cc.stop()
            await asyncio.gather(mds_task, wscc_task, return_exceptions=True)
            sys.exit(1)

        if not cc_ready:
            LOG.error("Coincall WS failed to connect")
            mds.stop()
            ws_cc.stop()
            await asyncio.gather(mds_task, wscc_task, return_exceptions=True)
            sys.exit(1)

        LOG.info("Both WS connections ready")

        # Bootstrap expiry timestamps immediately at startup (1 REST call per maturity)
        # so first evaluate_structure() call has zero additional REST overhead
        if instruments:
            await mds._bootstrap_expiry(list(instruments), deribit_handler)
            LOG.info("Expiry timestamps bootstrapped")

        # ------------------------------------------------------------------
        # Step 3: evaluate + execute within persistent REST session
        # ------------------------------------------------------------------
        async with CoincallREST(config, api_key, api_secret) as rest:
            LOG.info("Evaluating structure...")
            evaluation = await evaluate_structure(
                leg_specs, config, mds, deribit_handler
            )
            print_evaluation(evaluation, config)

            if not evaluation.all_passed:
                LOG.info("Thresholds not met. Exiting.")
            else:
                loop = asyncio.get_event_loop()
                confirm = await loop.run_in_executor(
                    None, lambda: input("\nProceed with RFQ? [y/N]: ").strip().lower()
                )
                if confirm != "y":
                    LOG.info("Aborted.")
                else:
                    success = await execute_rfq_flow(
                        evaluation, config, ws_cc, rest, mds
                    )
                    LOG.info(
                        "[OK] Trade executed" if success else "[FAIL] No trade executed"
                    )

        # ------------------------------------------------------------------
        # Shutdown
        # ------------------------------------------------------------------
        mds.stop()
        ws_cc.stop()
        await asyncio.gather(mds_task, wscc_task, return_exceptions=True)


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
