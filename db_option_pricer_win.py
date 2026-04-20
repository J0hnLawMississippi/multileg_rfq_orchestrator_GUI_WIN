#!/usr/bin/env python3
"""
Production-grade vectorized crypto option pricer.

Black-76 model (r=0 default) with:
  - Multi-leg option structure support
  - Per-maturity forward pricing
  - Fixed IV per strike (matching Deribit PB behavior)
  - Max drawdown with optional parallel vol shock
  - Deribit API integration for live IV surface + forwards

Design: no horizontal IV shift. Each strike retains its observed IV.
Index shift only changes the forward. Vol shock is a separate uniform
additive shift to all IVs (matching PB's vol slider behavior).
"""

import asyncio
import math
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm
from scipy.interpolate import CubicSpline


# =============================================================================
# ENUMS & DATA STRUCTURES
# =============================================================================


class OptionType(Enum):
    CALL = "C"
    PUT = "P"


class Direction(Enum):
    LONG = "L"
    SHORT = "S"


@dataclass(frozen=True, slots=True)
class OptionLeg:
    """Single option leg specification."""

    direction: Direction
    size: float
    maturity: str
    strike: float
    option_type: OptionType
    instrument_name: str

    @property
    def sign(self) -> float:
        return 1.0 if self.direction == Direction.LONG else -1.0


@dataclass
class MaturityIVData:
    """IV data for an entire maturity including forward."""

    maturity: str
    expiry_ts: int
    forward: float
    strikes: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    ivs: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    mark_prices_btc: NDArray[np.float64] = field(default_factory=lambda: np.array([]))


@dataclass
class StructurePriceResult:
    """Result of pricing a multi-leg structure."""

    legs: List[OptionLeg]
    spot: float
    shifted_spot: float
    leg_prices_btc: NDArray[np.float64]
    leg_prices_usd: NDArray[np.float64]
    leg_ivs: NDArray[np.float64]
    leg_forwards: NDArray[np.float64]
    total_btc: float
    total_usd: float
    time_to_expiry_years: float
    vol_shift: float = 0.0


@dataclass
class DrawdownResult:
    """Max drawdown analysis result."""

    spot_range: NDArray[np.float64]
    pnl_usd: NDArray[np.float64]
    pnl_btc: NDArray[np.float64]
    max_drawdown_usd: float
    max_drawdown_btc: float
    worst_spot_usd: float
    worst_spot_btc: float
    current_value_usd: float
    current_value_btc: float
    days_forward: int
    entry_prices_btc: NDArray[np.float64]
    vol_shift: float = 0.0


# =============================================================================
# INPUT PARSER
# =============================================================================

_LEG_PATTERN = re.compile(
    r"^([LS])\s+"
    r"((?:0|[1-9]\d*)(?:\.\d+)?)\s+"
    r"(\d{1,2}[A-Z]{3}\d{2})-((?:0|[1-9]\d*))-([CP])$",
    re.IGNORECASE,
)


def parse_leg(spec: str) -> OptionLeg:
    spec_trimmed = spec.strip()
    m = _LEG_PATTERN.match(spec_trimmed)
    if not m:
        raise ValueError(
            f"Invalid leg spec: '{spec}'. "
            "Expected canonical format: "
            "L|S <size> <DDMMMYY>-<strike>-<C|P> "
            "(e.g. 'L 0.5 20MAR26-70000-C'). "
            "Use plain decimals only (no signs, exponent notation, or extra dots)."
        )
    direction = Direction(m.group(1).upper())
    size_text = m.group(2)
    size = float(size_text)
    if not math.isfinite(size) or size <= 0.0:
        raise ValueError(
            f"Invalid leg size '{size_text}' in '{spec_trimmed}': "
            "size must be a finite number greater than 0."
        )

    maturity = m.group(3).upper()
    strike_text = m.group(4)
    strike = float(strike_text)
    if not math.isfinite(strike) or strike <= 0.0:
        raise ValueError(
            f"Invalid strike '{strike_text}' in '{spec_trimmed}': "
            "strike must be a finite number greater than 0."
        )

    try:
        maturity_to_datetime(maturity)
    except ValueError as exc:
        raise ValueError(
            f"Invalid maturity '{maturity}' in '{spec_trimmed}': {exc}. "
            "Expected a real calendar date in DDMMMYY format (e.g. 20MAR26)."
        ) from exc

    opt_type = OptionType(m.group(5).upper())
    instrument_name = f"BTC-{maturity}-{int(strike)}-{opt_type.value}"
    return OptionLeg(direction, size, maturity, strike, opt_type, instrument_name)


def parse_legs(specs: List[str]) -> List[OptionLeg]:
    return [parse_leg(s) for s in specs]


# =============================================================================
# MATURITY UTILITIES
# =============================================================================

_MONTH_MAP = {
    "JAN": 1,
    "FEB": 2,
    "MAR": 3,
    "APR": 4,
    "MAY": 5,
    "JUN": 6,
    "JUL": 7,
    "AUG": 8,
    "SEP": 9,
    "OCT": 10,
    "NOV": 11,
    "DEC": 12,
}


def maturity_to_datetime(maturity: str) -> datetime:
    day = int(maturity[: len(maturity) - 5])
    mon_str = maturity[len(maturity) - 5 : len(maturity) - 2]
    year = 2000 + int(maturity[-2:])
    month = _MONTH_MAP.get(mon_str)
    if month is None:
        raise ValueError(f"Unknown month: {mon_str}")
    return datetime(year, month, day, 8, 0, 0, tzinfo=timezone.utc)


def time_to_expiry_years(maturity: str, ref_time: Optional[datetime] = None) -> float:
    expiry = maturity_to_datetime(maturity)
    ref = ref_time or datetime.now(timezone.utc)
    return max((expiry - ref).total_seconds() / (365.25 * 86400), 1e-10)


# =============================================================================
# VECTORIZED BLACK-76
# =============================================================================
# =============================================================================
# EXPIRY-AWARE PRICING — changes to Black76 class
# =============================================================================


class Black76:
    """Vectorized Black-76 with expiry handling."""

    @staticmethod
    def price(
        F: NDArray[np.float64],
        K: NDArray[np.float64],
        sigma: NDArray[np.float64],
        T: NDArray[np.float64],
        is_call: NDArray[np.bool_],
        r: float = 0.0,
    ) -> NDArray[np.float64]:
        F = np.asarray(F, dtype=np.float64)
        K = np.asarray(K, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        T = np.asarray(T, dtype=np.float64)
        is_call = np.asarray(is_call, dtype=np.bool_)

        # Separate expired from live
        expired = T <= 0
        live = ~expired

        result = np.empty_like(F)

        # Expired legs: settle at intrinsic
        if np.any(expired):
            call_intrinsic = np.maximum(F[expired] - K[expired], 0.0)
            put_intrinsic = np.maximum(K[expired] - F[expired], 0.0)
            result[expired] = np.where(is_call[expired], call_intrinsic, put_intrinsic)

        # Live legs: standard Black-76
        if np.any(live):
            T_live = T[live]
            F_live = F[live]
            K_live = K[live]
            sigma_live = sigma[live]
            is_call_live = is_call[live]

            sqrt_T = np.sqrt(T_live)
            d1 = (np.log(F_live / K_live) + 0.5 * sigma_live**2 * T_live) / (
                sigma_live * sqrt_T
            )
            d2 = d1 - sigma_live * sqrt_T
            df = np.exp(-r * T_live)

            call_price = df * (F_live * norm.cdf(d1) - K_live * norm.cdf(d2))
            put_price = df * (K_live * norm.cdf(-d2) - F_live * norm.cdf(-d1))
            result[live] = np.where(is_call_live, call_price, put_price)

        return result

    @staticmethod
    def delta(
        F: NDArray[np.float64],
        K: NDArray[np.float64],
        sigma: NDArray[np.float64],
        T: NDArray[np.float64],
        is_call: NDArray[np.bool_],
        r: float = 0.0,
    ) -> NDArray[np.float64]:
        F = np.asarray(F, dtype=np.float64)
        K = np.asarray(K, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        T = np.asarray(T, dtype=np.float64)
        is_call = np.asarray(is_call, dtype=np.bool_)

        expired = T <= 0
        live = ~expired

        result = np.empty_like(F)

        # Expired: binary delta (1 if ITM, 0 if OTM)
        if np.any(expired):
            call_itm = (F[expired] > K[expired]).astype(np.float64)
            put_itm = (F[expired] < K[expired]).astype(np.float64) * -1.0
            result[expired] = np.where(is_call[expired], call_itm, put_itm)

        if np.any(live):
            T_live = T[live]
            sqrt_T = np.sqrt(T_live)
            d1 = (np.log(F[live] / K[live]) + 0.5 * sigma[live] ** 2 * T_live) / (
                sigma[live] * sqrt_T
            )
            df = np.exp(-r * T_live)
            call_delta = df * norm.cdf(d1)
            put_delta = df * (norm.cdf(d1) - 1.0)
            result[live] = np.where(is_call[live], call_delta, put_delta)

        return result


# =============================================================================
# STRIKE IV INTERPOLATOR — FIXED SMILE, NO SHIFT
# =============================================================================


class StrikeIVInterpolator:
    """
    Interpolates IV for arbitrary strikes on a fixed smile.

    No horizontal shifting. Each strike gets the IV observed at that
    strike on the original smile. This matches Deribit PB behavior:
    index shift changes the forward, IVs stay pinned to strikes.
    """

    def __init__(self, iv_data: MaturityIVData):
        self._strikes = iv_data.strikes.copy()
        self._ivs = iv_data.ivs.copy()
        self._maturity = iv_data.maturity

        if len(self._strikes) < 2:
            raise ValueError(
                f"Need >=2 strike/IV points, got {len(self._strikes)} for {iv_data.maturity}"
            )

        sort_idx = np.argsort(self._strikes)
        self._strikes = self._strikes[sort_idx]
        self._ivs = self._ivs[sort_idx]

        self._spline = CubicSpline(
            self._strikes, self._ivs, bc_type="natural", extrapolate=False
        )

        self._strike_min = float(self._strikes[0])
        self._strike_max = float(self._strikes[-1])

        # Boundary values for flat extrapolation
        self._iv_left = float(self._ivs[0])
        self._iv_right = float(self._ivs[-1])

    def iv_at_strikes(self, strikes: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Get IVs at given strikes. Flat extrapolation beyond observed range.

        Args:
            strikes: Array of strikes to evaluate

        Returns:
            Array of interpolated IVs (decimal)
        """
        strikes = np.asarray(strikes, dtype=np.float64)
        result = np.empty_like(strikes)

        mask_left = strikes < self._strike_min
        mask_right = strikes > self._strike_max
        mask_interior = ~(mask_left | mask_right)

        if np.any(mask_interior):
            result[mask_interior] = self._spline(strikes[mask_interior])
        if np.any(mask_left):
            result[mask_left] = self._iv_left
        if np.any(mask_right):
            result[mask_right] = self._iv_right

        return np.clip(result, 0.01, 5.0)


# =============================================================================
# IV SURFACE FETCHER
# =============================================================================


class DeribitIVFetcher:
    def __init__(self, handler):
        self._handler = handler

    async def get_index_price(self, currency: str = "BTC") -> float:
        resp = await self._handler.get(
            "public/get_index_price", params={"index_name": f"{currency.lower()}_usd"}
        )
        return float(resp["result"]["index_price"])

    async def get_forward_price(self, maturity: str, currency: str = "BTC") -> float:
        """Get forward price for a maturity via the future ticker."""
        """
        future_name = f"{currency.upper()}-{maturity.upper()}"
        try:
            resp = await self._handler.get(
                "public/ticker",
                params={"instrument_name": future_name}
            )
            result = resp["result"]
            fwd = result.get("underlying_price")
            if fwd and fwd > 0:
                return float(fwd)
            return float(result["mark_price"])
        except Exception:
            pass
        """
        # Fallback: read from any option ticker for that maturity
        resp = await self._handler.get(
            "public/get_book_summary_by_currency",
            params={"currency": currency, "kind": "option"},
        )
        prefix = f"{currency.upper()}-{maturity.upper()}-"
        for s in resp["result"]:
            if s["instrument_name"].startswith(prefix):
                ticker = await self._handler.get(
                    "public/ticker", params={"instrument_name": s["instrument_name"]}
                )
                fwd = ticker["result"].get("underlying_price")
                if fwd and fwd > 0:
                    return float(fwd)

        raise ValueError(f"Could not determine forward for {maturity}")

    async def get_maturity_iv_data(
        self, maturity: str, currency: str = "BTC"
    ) -> MaturityIVData:
        resp = await self._handler.get(
            "public/get_book_summary_by_currency",
            params={"currency": currency, "kind": "option"},
        )
        summaries = resp["result"]
        maturity_upper = maturity.upper()
        prefix = f"{currency.upper()}-{maturity_upper}-"

        filtered = [
            s
            for s in summaries
            if s["instrument_name"].startswith(prefix)
            and s.get("mark_iv") is not None
            and s["mark_iv"] > 0
        ]

        if not filtered:
            raise ValueError(f"No IV data for maturity {maturity}")

        names = [s["instrument_name"] for s in filtered]
        strikes_raw = [float(n.split("-")[2]) for n in names]
        ivs_raw = [s["mark_iv"] / 100.0 for s in filtered]
        marks_raw = [s.get("mark_price", 0.0) or 0.0 for s in filtered]

        strike_iv_map: Dict[float, List[float]] = {}
        strike_mark_map: Dict[float, List[float]] = {}
        for k, v, mp in zip(strikes_raw, ivs_raw, marks_raw):
            strike_iv_map.setdefault(k, []).append(v)
            strike_mark_map.setdefault(k, []).append(mp)

        unique_strikes = sorted(strike_iv_map.keys())
        avg_ivs = [np.mean(strike_iv_map[k]) for k in unique_strikes]
        avg_marks = [np.mean(strike_mark_map[k]) for k in unique_strikes]

        sample = filtered[0]["instrument_name"]
        inst_resp = await self._handler.get(
            "public/get_instrument", params={"instrument_name": sample}
        )
        expiry_ts = inst_resp["result"]["expiration_timestamp"]

        forward = await self.get_forward_price(maturity, currency)

        return MaturityIVData(
            maturity=maturity_upper,
            expiry_ts=expiry_ts,
            forward=forward,
            strikes=np.array(unique_strikes, dtype=np.float64),
            ivs=np.array(avg_ivs, dtype=np.float64),
            mark_prices_btc=np.array(avg_marks, dtype=np.float64),
        )

    async def get_multi_maturity_iv(
        self, maturities: List[str], currency: str = "BTC"
    ) -> Dict[str, MaturityIVData]:
        tasks = [self.get_maturity_iv_data(m, currency) for m in maturities]
        results = await asyncio.gather(*tasks)
        return {r.maturity: r for r in results}


# =============================================================================
# STRUCTURE PRICER
# =============================================================================


class StructurePricer:
    """
    Prices multi-leg option structures.

    Model: Black-76 with per-maturity forwards.
    IV: fixed per strike from observed smile (no shifting).
    Vol shock: uniform additive shift to all IVs.
    """

    def __init__(
        self,
        iv_interpolators: Dict[str, StrikeIVInterpolator],
        maturity_data: Dict[str, MaturityIVData],
        current_spot: float,
        r: float = 0.0,
    ):
        self._iv_interps = iv_interpolators
        self._maturity_data = maturity_data
        self._spot = current_spot
        self._r = r

    @property
    def spot(self) -> float:
        return self._spot

    def _shifted_forward(self, maturity: str, target_spot: float) -> float:
        """Forward shifts by same absolute amount as spot."""
        original_forward = self._maturity_data[maturity].forward
        spot_delta = target_spot - self._spot
        return original_forward + spot_delta

    def price_structure(
        self,
        legs: List[OptionLeg],
        target_spot: float,
        ref_time: Optional[datetime] = None,
        days_offset: int = 0,
        vol_shift: float = 0.0,
    ) -> StructurePriceResult:
        """
        Price structure at target spot with optional vol shock.

        Args:
            legs: Option legs
            target_spot: Target index price
            ref_time: Reference time (default: now)
            days_offset: Days forward from ref_time
            vol_shift: Additive IV shift in decimal (e.g. -0.10 = -10 vol points)

        Returns:
            StructurePriceResult
        """
        ref = ref_time or datetime.now(timezone.utc)
        if days_offset:
            ref = ref + timedelta(days=days_offset)

        n = len(legs)

        # Per-leg forward
        F = np.array(
            [self._shifted_forward(leg.maturity, target_spot) for leg in legs],
            dtype=np.float64,
        )

        K = np.array([leg.strike for leg in legs], dtype=np.float64)
        is_call = np.array(
            [leg.option_type == OptionType.CALL for leg in legs], dtype=np.bool_
        )
        signs = np.array([leg.sign for leg in legs], dtype=np.float64)
        sizes = np.array([leg.size for leg in legs], dtype=np.float64)

        # IVs: fixed per strike, no shift — then apply uniform vol shock
        sigmas = np.empty(n, dtype=np.float64)
        T = np.empty(n, dtype=np.float64)

        maturity_indices: Dict[str, List[int]] = {}
        for i, leg in enumerate(legs):
            maturity_indices.setdefault(leg.maturity, []).append(i)
            T[i] = time_to_expiry_years(leg.maturity, ref)

        for mat, indices in maturity_indices.items():
            idx_arr = np.array(indices)
            mat_strikes = K[idx_arr]
            mat_ivs = self._iv_interps[mat].iv_at_strikes(mat_strikes)
            sigmas[idx_arr] = mat_ivs

        # Apply uniform vol shock
        sigmas_shocked = np.clip(sigmas + vol_shift, 0.01, 5.0)

        prices_usd = Black76.price(F, K, sigmas_shocked, T, is_call, self._r)
        prices_btc = prices_usd / F

        signed_btc = signs * sizes * prices_btc
        signed_usd = signed_btc * target_spot

        return StructurePriceResult(
            legs=legs,
            spot=self._spot,
            shifted_spot=target_spot,
            leg_prices_btc=signed_btc,
            leg_prices_usd=signed_usd,
            leg_ivs=sigmas_shocked,
            leg_forwards=F,
            total_btc=float(np.sum(signed_btc)),
            total_usd=float(np.sum(signed_usd)),
            time_to_expiry_years=float(T[0]),
            vol_shift=vol_shift,
        )


# =============================================================================
# VECTORIZED DRAWDOWN CALCULATOR
# =============================================================================


class VectorizedDrawdownCalculator:
    """
    Fully vectorized drawdown across spot grid × legs.

    Computes two parallel scenarios:
      1. Vol unchanged (vol_shift=0)
      2. Vol shocked (vol_shift=configurable, default -10%)

    IV per strike is FIXED from observed smile.
    Forward shifts proportionally with spot.
    """

    def __init__(
        self,
        iv_interpolators: Dict[str, StrikeIVInterpolator],
        maturity_data: Dict[str, MaturityIVData],
        current_spot: float,
        legs: List[OptionLeg],
        original_spot: float,  # <-- ADD THIS
        r: float = 0.0,
        spot_range_pct: float = 0.50,
        spot_grid_points: int = 500,
        vol_shock: float = -0.10,
    ):
        self._iv_interps = iv_interpolators
        self._maturity_data = maturity_data
        self._spot = current_spot
        self._legs = legs
        self._original_spot = original_spot  # <-- STORE IT
        self._r = r
        self._range_pct = spot_range_pct
        self._grid_n = spot_grid_points
        self._vol_shock = vol_shock

    def _build_forward_grid(
        self, spot_grid: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        orig_fwd = np.array(
            [self._maturity_data[lg.maturity].forward for lg in self._legs],
            dtype=np.float64,
        )
        spot_deltas = spot_grid - self._original_spot  # <-- delta from ORIGINAL spot
        return orig_fwd[np.newaxis, :] + spot_deltas[:, np.newaxis]

    def _build_iv_matrix(self) -> NDArray[np.float64]:
        """
        Fixed IV per leg from observed smile. Shape (L,).
        Same IV used at every spot grid point.
        """
        n_legs = len(self._legs)
        K = np.array([lg.strike for lg in self._legs], dtype=np.float64)
        sigmas = np.empty(n_legs, dtype=np.float64)

        maturity_indices: Dict[str, List[int]] = {}
        for i, lg in enumerate(self._legs):
            maturity_indices.setdefault(lg.maturity, []).append(i)

        for mat, indices in maturity_indices.items():
            idx_arr = np.array(indices)
            sigmas[idx_arr] = self._iv_interps[mat].iv_at_strikes(K[idx_arr])

        return sigmas

    def _price_grid(
        self,
        F_grid: NDArray[np.float64],
        spot_grid: NDArray[np.float64],  # <-- ADD
        K: NDArray[np.float64],
        sigmas: NDArray[np.float64],
        T: NDArray[np.float64],
        is_call: NDArray[np.bool_],
        signs: NDArray[np.float64],
        sizes: NDArray[np.float64],
        vol_shift: float = 0.0,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        n_grid, n_legs = F_grid.shape
        sigmas_shocked = np.clip(sigmas + vol_shift, 0.01, 5.0)

        K_2d = np.broadcast_to(K, (n_grid, n_legs))
        sigma_2d = np.broadcast_to(sigmas_shocked, (n_grid, n_legs))
        T_2d = np.broadcast_to(T, (n_grid, n_legs))
        is_call_2d = np.broadcast_to(is_call, (n_grid, n_legs))

        prices_usd = Black76.price(F_grid, K_2d, sigma_2d, T_2d, is_call_2d, self._r)
        prices_btc = prices_usd / F_grid

        weight = signs * sizes
        total_btc = (weight[np.newaxis, :] * prices_btc).sum(axis=1)
        total_usd = total_btc * spot_grid  # <-- FIX

        return total_btc, total_usd

    def compute(self, days_forward: int = 1) -> Tuple[DrawdownResult, DrawdownResult]:
        """
        Compute max drawdown under two scenarios:
          1. Vol unchanged
          2. Vol shocked by self._vol_shock

        Returns:
            (drawdown_no_shock, drawdown_with_shock)
        """
        n_legs = len(self._legs)
        spot = self._spot

        # Spot grid
        low = spot * (1.0 - self._range_pct)
        high = spot * (1.0 + self._range_pct)
        spot_grid = np.linspace(low, high, self._grid_n)

        # Leg parameters
        K = np.array([lg.strike for lg in self._legs], dtype=np.float64)
        is_call = np.array(
            [lg.option_type == OptionType.CALL for lg in self._legs], dtype=np.bool_
        )
        signs = np.array([lg.sign for lg in self._legs], dtype=np.float64)
        sizes = np.array([lg.size for lg in self._legs], dtype=np.float64)

        now = datetime.now(timezone.utc)
        future = now + timedelta(days=days_forward)

        T_future = np.array(
            [time_to_expiry_years(lg.maturity, future) for lg in self._legs],
            dtype=np.float64,
        )
        T_now = np.array(
            [time_to_expiry_years(lg.maturity, now) for lg in self._legs],
            dtype=np.float64,
        )

        # Fixed IVs from observed smile
        base_sigmas = self._build_iv_matrix()

        # Forward grids
        F_grid = self._build_forward_grid(spot_grid)

        # Entry forwards (at current spot)
        F_entry = np.array(
            [
                self._maturity_data[lg.maturity].forward
                + (self._spot - self._original_spot)
                for lg in self._legs
            ],
            dtype=np.float64,
        )

        # Entry prices (no vol shock, current time)
        entry_usd = Black76.price(F_entry, K, base_sigmas, T_now, is_call, self._r)
        entry_btc = entry_usd / F_entry
        entry_total_btc = float(np.sum(signs * sizes * entry_btc))
        entry_total_usd = entry_total_btc * self._spot  # <-- BTC × entry spot
        entry_leg_btc = signs * sizes * entry_btc

        # --- Scenario 1: Vol unchanged ---
        grid_btc_flat, grid_usd_flat = self._price_grid(
            F_grid,
            spot_grid,
            K,
            base_sigmas,
            T_future,
            is_call,
            signs,
            sizes,
            vol_shift=0.0,
        )
        pnl_usd_flat = grid_usd_flat - entry_total_usd
        pnl_btc_flat = grid_btc_flat - entry_total_btc

        idx_worst_usd_flat = int(np.argmin(pnl_usd_flat))
        idx_worst_btc_flat = int(np.argmin(pnl_btc_flat))

        dd_flat = DrawdownResult(
            spot_range=spot_grid,
            pnl_usd=pnl_usd_flat,
            pnl_btc=pnl_btc_flat,
            max_drawdown_usd=float(pnl_usd_flat[idx_worst_usd_flat]),
            max_drawdown_btc=float(pnl_btc_flat[idx_worst_btc_flat]),
            worst_spot_usd=float(spot_grid[idx_worst_usd_flat]),
            worst_spot_btc=float(spot_grid[idx_worst_btc_flat]),
            current_value_usd=entry_total_usd,
            current_value_btc=entry_total_btc,
            days_forward=days_forward,
            entry_prices_btc=entry_leg_btc,
            vol_shift=0.0,
        )

        # --- Scenario 2: Vol shocked ---
        grid_btc_shock, grid_usd_shock = self._price_grid(
            F_grid,
            spot_grid,
            K,
            base_sigmas,
            T_future,
            is_call,
            signs,
            sizes,
            vol_shift=self._vol_shock,
        )

        pnl_usd_shock = grid_usd_shock - entry_total_usd
        pnl_btc_shock = grid_btc_shock - entry_total_btc

        idx_worst_usd_shock = int(np.argmin(pnl_usd_shock))
        idx_worst_btc_shock = int(np.argmin(pnl_btc_shock))

        dd_shock = DrawdownResult(
            spot_range=spot_grid,
            pnl_usd=pnl_usd_shock,
            pnl_btc=pnl_btc_shock,
            max_drawdown_usd=float(pnl_usd_shock[idx_worst_usd_shock]),
            max_drawdown_btc=float(pnl_btc_shock[idx_worst_btc_shock]),
            worst_spot_usd=float(spot_grid[idx_worst_usd_shock]),
            worst_spot_btc=float(spot_grid[idx_worst_btc_shock]),
            current_value_usd=entry_total_usd,
            current_value_btc=entry_total_btc,
            days_forward=days_forward,
            entry_prices_btc=entry_leg_btc,
            vol_shift=self._vol_shock,
        )

        return dd_flat, dd_shock


# =============================================================================
# ORCHESTRATOR
# =============================================================================


class CryptoOptionAnalyzer:
    def __init__(
        self,
        currency: str = "BTC",
        r: float = 0.0,
        spot_range_pct: float = 0.50,
        spot_grid_points: int = 500,
        vol_shock: float = -0.10,
        testnet: bool = False,
    ):
        self._currency = currency
        self._r = r
        self._range_pct = spot_range_pct
        self._grid_n = spot_grid_points
        self._vol_shock = vol_shock
        self._testnet = testnet

    async def analyze(
        self,
        legs: List[str],
        target_spot: float,
        drawdown_days: int = 1,
        verify_with_pb: bool = False,
    ) -> Dict[str, Any]:

        parsed_legs = parse_legs(legs)
        maturities = list({lg.maturity for lg in parsed_legs})

        print(f"Parsed {len(parsed_legs)} legs across {len(maturities)} maturities")

        # Fetch from Deribit
        # Import here to avoid circular dependency
        from deribit_api import CachedDeribitApiHandler

        async with CachedDeribitApiHandler(
            testnet=self._testnet, cache_ttl_sec=120
        ) as handler:
            fetcher = DeribitIVFetcher(handler)
            current_spot = await fetcher.get_index_price(self._currency)
            print(f"Current {self._currency} index: {current_spot:.2f}")
            iv_data = await fetcher.get_multi_maturity_iv(maturities, self._currency)

        # Print forward info
        for mat, data in iv_data.items():
            basis = data.forward - current_spot
            print(f"  {mat}: forward={data.forward:.2f} (basis={basis:+.2f})")

        # Build interpolators
        iv_interps = {mat: StrikeIVInterpolator(data) for mat, data in iv_data.items()}

        # Pricer
        pricer = StructurePricer(iv_interps, iv_data, current_spot, self._r)

        # Price at current spot
        current_result = pricer.price_structure(parsed_legs, current_spot)

        # Price at target spot (no vol shock)
        shifted_result = pricer.price_structure(parsed_legs, target_spot)

        # Price at target spot with vol shock
        shifted_shocked = pricer.price_structure(
            parsed_legs, target_spot, vol_shift=self._vol_shock
        )

        # Display
        print(f"\n{'=' * 70}")
        print(f"Structure Value at Current Spot ({current_spot:.0f})")
        print(f"{'=' * 70}")
        for i, leg in enumerate(parsed_legs):
            unit_btc = current_result.leg_prices_btc[i] / leg.size
            unit_usd = current_result.leg_prices_usd[i] / leg.size
            print(
                f"  {leg.direction.value} {leg.size} {leg.instrument_name}: "
                f"F={current_result.leg_forwards[i]:.2f} "
                f"IV={current_result.leg_ivs[i] * 100:.1f}% "
                f"BTC={unit_btc:.4f} "
                f"USD={unit_usd:.2f}"
            )
        print(
            f"  TOTAL: {current_result.total_btc:.4f} BTC / {current_result.total_usd:.2f} USD"
        )

        print(f"\n{'=' * 70}")
        print(f"Structure Value at Target Spot ({target_spot:.0f}) - Vol Unchanged")
        print(f"{'=' * 70}")
        for i, leg in enumerate(parsed_legs):
            unit_btc = shifted_result.leg_prices_btc[i] / leg.size
            unit_usd = shifted_result.leg_prices_usd[i] / leg.size
            print(
                f"  {leg.direction.value} {leg.size} {leg.instrument_name}: "
                f"F={shifted_result.leg_forwards[i]:.2f} "
                f"IV={shifted_result.leg_ivs[i] * 100:.1f}% "
                f"BTC={unit_btc:.4f} "
                f"USD={unit_usd:.2f}"
            )
        print(
            f"  TOTAL: {shifted_result.total_btc:.4f} BTC / {shifted_result.total_usd:.2f} USD"
        )

        print(f"\n{'=' * 70}")
        print(
            f"Structure Value at Target Spot ({target_spot:.0f}) - Vol Shock {self._vol_shock * 100:+.0f}%"
        )
        print(f"{'=' * 70}")
        for i, leg in enumerate(parsed_legs):
            unit_btc = shifted_shocked.leg_prices_btc[i] / leg.size
            unit_usd = shifted_shocked.leg_prices_usd[i] / leg.size
            print(
                f"  {leg.direction.value} {leg.size} {leg.instrument_name}: "
                f"F={shifted_shocked.leg_forwards[i]:.2f} "
                f"IV={shifted_shocked.leg_ivs[i] * 100:.1f}% "
                f"BTC={unit_btc:.4f} "
                f"USD={unit_usd:.2f}"
            )
        print(
            f"  TOTAL: {shifted_shocked.total_btc:.4f} BTC / {shifted_shocked.total_usd:.2f} USD"
        )

        # Max drawdown
        dd_calc = VectorizedDrawdownCalculator(
            iv_interpolators=iv_interps,
            maturity_data=iv_data,
            current_spot=target_spot,
            legs=parsed_legs,
            original_spot=current_spot,  # <-- ADD THIS
            r=self._r,
            spot_range_pct=self._range_pct,
            spot_grid_points=self._grid_n,
            vol_shock=self._vol_shock,
        )

        dd_flat, dd_shock = dd_calc.compute(days_forward=drawdown_days)

        print(f"\n{'=' * 70}")
        print(
            f"Max Drawdown ({drawdown_days}d from entry at {target_spot:.0f}, +/-{self._range_pct * 100:.0f}% range)"
        )
        print(f"{'=' * 70}")

        # Warn about legs expiring within drawdown window
        from datetime import timedelta

        eval_date = datetime.now(timezone.utc) + timedelta(days=drawdown_days)
        for leg in parsed_legs:
            expiry = maturity_to_datetime(leg.maturity)
            tte_days = (expiry - datetime.now(timezone.utc)).total_seconds() / 86400
            if tte_days <= drawdown_days:
                print(
                    f"  [WARN] {leg.instrument_name} expires in {tte_days:.1f}d "
                    f"(within {drawdown_days}d window) - will settle at intrinsic"
                )

        print(
            f"  Entry value: {dd_flat.current_value_btc:.4f} BTC / {dd_flat.current_value_usd:.2f} USD"
        )
        print(f"")
        print(f"  Vol Unchanged:")
        print(
            f"    Max DD (USD): {dd_flat.max_drawdown_usd:.2f} at spot {dd_flat.worst_spot_usd:.0f}"
        )
        print(
            f"    Max DD (BTC): {dd_flat.max_drawdown_btc:.4f} at spot {dd_flat.worst_spot_btc:.0f}"
        )
        print(f"")
        print(f"  Vol Shock ({self._vol_shock * 100:+.0f}%):")
        print(
            f"    Max DD (USD): {dd_shock.max_drawdown_usd:.2f} at spot {dd_shock.worst_spot_usd:.0f}"
        )
        print(
            f"    Max DD (BTC): {dd_shock.max_drawdown_btc:.4f} at spot {dd_shock.worst_spot_btc:.0f}"
        )

        # Build output dict
        output = {
            "legs": [
                {
                    "spec": f"{lg.direction.value} {lg.size} {lg.instrument_name}",
                    "strike": lg.strike,
                    "type": lg.option_type.value,
                    "direction": lg.direction.value,
                    "size": lg.size,
                }
                for lg in parsed_legs
            ],
            "current_spot": current_spot,
            "target_spot": target_spot,
            "forwards": {mat: data.forward for mat, data in iv_data.items()},
            "current_value": {
                "btc": current_result.total_btc,
                "usd": current_result.total_usd,
                "per_leg_btc": current_result.leg_prices_btc.tolist(),
                "per_leg_usd": current_result.leg_prices_usd.tolist(),
                "per_leg_iv": (current_result.leg_ivs * 100).tolist(),
                "per_leg_forward": current_result.leg_forwards.tolist(),
            },
            "shifted_value": {
                "btc": shifted_result.total_btc,
                "usd": shifted_result.total_usd,
                "per_leg_btc": shifted_result.leg_prices_btc.tolist(),
                "per_leg_usd": shifted_result.leg_prices_usd.tolist(),
                "per_leg_iv": (shifted_result.leg_ivs * 100).tolist(),
                "per_leg_forward": shifted_result.leg_forwards.tolist(),
            },
            "shifted_value_vol_shocked": {
                "btc": shifted_shocked.total_btc,
                "usd": shifted_shocked.total_usd,
                "per_leg_btc": shifted_shocked.leg_prices_btc.tolist(),
                "per_leg_usd": shifted_shocked.leg_prices_usd.tolist(),
                "per_leg_iv": (shifted_shocked.leg_ivs * 100).tolist(),
                "vol_shock": self._vol_shock,
            },
            "drawdown_flat": {
                "days_forward": drawdown_days,
                "max_dd_usd": dd_flat.max_drawdown_usd,
                "max_dd_btc": dd_flat.max_drawdown_btc,
                "worst_spot_usd": dd_flat.worst_spot_usd,
                "worst_spot_btc": dd_flat.worst_spot_btc,
            },
            "drawdown_vol_shocked": {
                "days_forward": drawdown_days,
                "vol_shock": self._vol_shock,
                "max_dd_usd": dd_shock.max_drawdown_usd,
                "max_dd_btc": dd_shock.max_drawdown_btc,
                "worst_spot_usd": dd_shock.worst_spot_usd,
                "worst_spot_btc": dd_shock.worst_spot_btc,
            },
            "iv_surfaces": {
                mat: {
                    "strikes": data.strikes.tolist(),
                    "ivs": (data.ivs * 100).tolist(),
                    "forward": data.forward,
                }
                for mat, data in iv_data.items()
            },
        }

        # In CryptoOptionAnalyzer.analyze(), replace the entire verify_with_pb block:
        if verify_with_pb:
            print(f"\n{'=' * 70}")
            print("Verifying against pb.deribit.com")
            print(f"{'=' * 70}")
            from pb_verification import DeribitPBVerifier

            verifier = DeribitPBVerifier(headless=False, slow_mo=50)
            try:
                pb_result = await verifier.verify(
                    legs=parsed_legs,
                    target_spot=target_spot,
                    current_spot=current_spot,
                    days_forward=drawdown_days,
                    vol_shock_pct=int(self._vol_shock * 100),
                    pnl_range_pct=self._range_pct,
                    screenshot=True,
                )

                # --- Price comparison ---
                our_lookup = {
                    leg.instrument_name: i for i, leg in enumerate(parsed_legs)
                }

                print(
                    f"\n  {'Instrument':<28} {'PB Mark':>10} {'PB IV':>8} "
                    f"{'Our Mark':>10} {'Our IV':>8} {'dMark%':>8}"
                )
                print(
                    f"  {'-' * 28} {'-' * 10} {'-' * 8} {'-' * 10} {'-' * 8} {'-' * 8}"
                )

                for row in pb_result.rows:
                    idx = our_lookup.get(row.instrument)
                    if idx is not None:
                        our_mark = abs(
                            shifted_result.leg_prices_btc[idx] / parsed_legs[idx].size
                        )
                        our_iv = shifted_result.leg_ivs[idx] * 100
                    else:
                        our_mark = float("nan")
                        our_iv = float("nan")

                    mark_diff_pct = (
                        (our_mark - row.mark_price_btc) / row.mark_price_btc * 100
                        if row.mark_price_btc != 0
                        else float("nan")
                    )

                    print(
                        f"  {row.instrument:<28} "
                        f"{row.mark_price_btc:>10.4f} "
                        f"{row.iv_pct:>7.2f}% "
                        f"{our_mark:>10.4f} "
                        f"{our_iv:>7.2f}% "
                        f"{mark_diff_pct:>+7.2f}%"
                    )

                print(
                    f"\n  PB Total:  {pb_result.total_value_btc:.4f} BTC / "
                    f"{pb_result.total_value_usd:.2f} USD"
                )
                print(
                    f"  Our Total: {shifted_result.total_btc:.4f} BTC / "
                    f"{shifted_result.total_usd:.2f} USD"
                )

                # Forward comparison
                print(f"\n  Forward comparison:")
                for row in pb_result.rows:
                    idx = our_lookup.get(row.instrument)
                    if idx is not None:
                        our_fwd = shifted_result.leg_forwards[idx]
                        print(
                            f"  {row.instrument:<28} "
                            f"PB_F={row.forward:>10.2f}  "
                            f"Our_F={our_fwd:>10.2f}  "
                            f"d={our_fwd - row.forward:>+8.2f}"
                        )

                # --- Drawdown comparison ---
                dd_pb = pb_result.drawdown
                if dd_pb:
                    print(
                        f"\n  {'Scenario':<30} {'PB Max DD':>14} {'Our Max DD':>14} "
                        f"{'PB Spot':>10} {'Our Spot':>10}"
                    )
                    print(f"  {'-' * 30} {'-' * 14} {'-' * 14} {'-' * 10} {'-' * 10}")

                    vol_unch = dd_pb.get("vol_unchanged", {})
                    vol_shock = dd_pb.get("vol_shocked", {})

                    if "max_drawdown_usd" in vol_unch:
                        print(
                            f"  {f'+{drawdown_days}d vol unch':<30} "
                            f"{vol_unch['max_drawdown_usd']:>14,.2f} "
                            f"{dd_flat.max_drawdown_usd:>14,.2f} "
                            f"{vol_unch.get('worst_spot', 0):>10,.0f} "
                            f"{dd_flat.worst_spot_usd:>10,.0f}"
                        )
                    if "max_drawdown_usd" in vol_shock:
                        print(
                            f"  {f'+{drawdown_days}d vol {int(self._vol_shock * 100):+d}%':<30} "
                            f"{vol_shock['max_drawdown_usd']:>14,.2f} "
                            f"{dd_shock.max_drawdown_usd:>14,.2f} "
                            f"{vol_shock.get('worst_spot', 0):>10,.0f} "
                            f"{dd_shock.worst_spot_usd:>10,.0f}"
                        )

                # --- PnL grid comparison: both scenarios ---
                for scenario, pb_grid, label in [
                    ("flat", pb_result.pnl_grid_flat, f"+{drawdown_days}d vol unch"),
                    (
                        "shocked",
                        pb_result.pnl_grid_shocked,
                        f"+{drawdown_days}d vol {int(self._vol_shock * 100):+d}%",
                    ),
                ]:
                    if not pb_grid:
                        continue

                    vs = self._vol_shock if scenario == "shocked" else 0.0

                    print(f"\n  {'=' * 60}")
                    print(f"  PnL GRID: {label} (every 1000 pts)")
                    print(f"  {'=' * 60}")
                    print(
                        f"  {'Spot':>10}  {'PB PnL':>14}  {'Our PnL':>14}  {'Delta':>10}"
                    )
                    print(f"  {'-' * 10}  {'-' * 14}  {'-' * 14}  {'-' * 10}")

                    entry_usd = shifted_result.total_usd
                    for pt in pb_grid:
                        grid_spot = pt["index_price"]
                        our_at_spot = pricer.price_structure(
                            parsed_legs,
                            grid_spot,
                            days_offset=drawdown_days,
                            vol_shift=vs,
                        )
                        our_pnl = our_at_spot.total_usd - entry_usd
                        pb_pnl = pt["today_pnl_usd"]
                        delta = our_pnl - pb_pnl

                        print(
                            f"  {grid_spot:>10,.0f}  "
                            f"{pb_pnl:>14,.2f}  "
                            f"{our_pnl:>14,.2f}  "
                            f"{delta:>+10,.2f}"
                        )

                output["pb_verification"] = {
                    "rows": [
                        {
                            "instrument": r.instrument,
                            "amount": r.amount,
                            "mark_price_btc": r.mark_price_btc,
                            "iv_pct": r.iv_pct,
                            "index": r.index,
                            "forward": r.forward,
                        }
                        for r in pb_result.rows
                    ],
                    "total_btc": pb_result.total_value_btc,
                    "total_usd": pb_result.total_value_usd,
                    "drawdown": pb_result.drawdown,
                    "pnl_grid_flat": pb_result.pnl_grid_flat,
                    "pnl_grid_shocked": pb_result.pnl_grid_shocked,
                }

            except Exception as e:
                output["pb_verification"] = {"error": str(e)}
                print(f"  PB verification failed: {e}")

        return output


# =============================================================================
# CLI ENTRY POINT
# =============================================================================


async def main():

    analyzer = CryptoOptionAnalyzer(
        currency="BTC",
        r=0.0,
        spot_range_pct=0.50,
        spot_grid_points=500,
        vol_shock=-0.10,
        testnet=False,
    )

    print("\n\n" + "=" * 70)
    print("MULTI-LEG STRUCTURE")
    print("=" * 70)

    result = await analyzer.analyze(
        legs=[
            # "S 1.8 20FEB26-75000-C",
            # "S 1.8 20FEB26-65000-P",
            # "S 0.22 20FEB26-64000-P",
            # "L 0.11 20FEB26-64000-C",
            "L 0.6 27FEB26-64000-P",
            "L 0.6 27FEB26-72000-C",
            "L 0.5 27MAR26-64000-P",
            "L 0.5 27MAR26-72000-C",
            "S 0.5 24APR26-64000-P",
            "S 0.5 24APR26-72000-C",
        ],
        target_spot=68000,
        drawdown_days=9,
        verify_with_pb=True,
    )


if __name__ == "__main__":
    asyncio.run(main())
