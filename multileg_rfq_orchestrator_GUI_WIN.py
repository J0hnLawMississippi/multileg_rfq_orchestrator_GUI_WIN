#!/usr/bin/env python3
"""
Multileg RFQ Orchestrator GUI
==============================
PySide6 + matplotlib + qasync front-end for structure analysis and RFQ submission.

Layout (horizontal QSplitter 33%/67%):
  Left panel  — up to 6 leg rows (cascading dropdowns: expiry→strike→type),
                target spot input, max-drawdown output, Evaluate / Submit RFQ buttons.
  Right panel — interactive matplotlib plot (structure PnL),
                parameter bar below (time-shift slider, vol-shift slider).

Data flow:
  Startup  : GET /open/option/getInstruments/BTC  → populate leg dropdowns
  Evaluate : enumerate_smile_instruments (Deribit REST) → DeribitMarketDataService (WS)
             → continuous live re-pricing loop (≤2 Hz plot refresh)
  Submit   : Y/N modal → CoincallREST.create_rfq → _process_quotes → execute_trade

All existing files are imported read-only; this file is standalone.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Standard library — must come before any local imports so that
# logging.basicConfig() fires before rfq_orchestrator's module-level
# basicConfig call (making that second call a no-op).
# ---------------------------------------------------------------------------
import argparse
import asyncio
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Configure root logger FIRST so rfq_orchestrator's basicConfig is a no-op
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
LOG = logging.getLogger("gui")

# ---------------------------------------------------------------------------
# Third-party — GUI stack
# ---------------------------------------------------------------------------
import aiohttp
import numpy as np
from dotenv import load_dotenv

try:
    from PySide6.QtCore import (
        Qt,
        QTimer,
        Signal,
        QObject,
        Slot,
        QMetaObject,
        Q_ARG,
    )
    from PySide6.QtGui import QColor, QPalette, QFont
    from PySide6.QtWidgets import (
        QApplication,
        QMainWindow,
        QWidget,
        QSplitter,
        QVBoxLayout,
        QHBoxLayout,
        QGridLayout,
        QFormLayout,
        QLabel,
        QLineEdit,
        QComboBox,
        QDoubleSpinBox,
        QSlider,
        QPushButton,
        QGroupBox,
        QDialog,
        QDialogButtonBox,
        QScrollArea,
        QFrame,
        QStatusBar,
        QSizePolicy,
        QSpacerItem,
        QTabWidget,
        QPlainTextEdit,
    )
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.figure import Figure
    import matplotlib.pyplot as plt
    import qasync
except ImportError as e:
    print(
        f"Missing dependency: {e}\n"
        "Install with:  pip install pyside6 qasync 'matplotlib>=3.6'"
    )
    sys.exit(1)

# ---------------------------------------------------------------------------
# Local modules — read-only imports
# ---------------------------------------------------------------------------
from db_option_pricer_win import (
    Black76,
    Direction,
    DrawdownResult,
    MaturityIVData,
    OptionLeg,
    OptionType,
    StructurePriceResult,
    StructurePricer,
    StrikeIVInterpolator,
    VectorizedDrawdownCalculator,
    parse_legs,
    maturity_to_datetime,
    time_to_expiry_years,
)
from deribit_fetcher_win import (
    DeribitMarketDataService,
    InstrumentSnapshot,
    OptimisedDeribitFetcher,
    enumerate_smile_instruments,
    SMILE_HALF_WIDTH,
)
from deribit_ws_api_win import CachedDeribitApiHandler
from coincall_ws_win import (
    CoincallWSClient,
    QuoteReceived,
    QuoteState,
    RFQState,
    RFQStateUpdate,
    WSEvent,
)
from rfq_orchestrator_win import (
    CoincallREST,
    RFQConfig,
    StructureEvaluation,
    ThresholdType,
    _ACCOUNT_CREDENTIALS,
    evaluate_structure,
    execute_rfq_flow,
)
from dvol_fetcher_win import fetch_dvol_and_price
from skew_fetcher_win import SkewFetcher
from deribit_api import (
    CachedDeribitApiHandler as SkewDeribitHandler,
    ConnectionPoolConfig,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
COINCALL_REST_BASE = "https://api.coincall.com"
COINCALL_WS_URL = "wss://ws.coincall.com/options"
PLOT_REFRESH_MS = 500  # max 2 Hz redraws
MAX_LEGS = 6
DEMO_SCREENSHOT_DEFAULT_LEGS = [
    "L 1.0 29MAY26-80000-C",
    "S 1.0 26JUN26-82000-C",
]


# =============================================================================
# COLOUR PALETTE  (Fusion dark-ish)
# =============================================================================

COLOUR_STRUCTURE = "#2196F3"  # blue  — structure line
COLOUR_ZERO = "#888888"  # grey  — zero line
COLOUR_SPOT_LIVE = "#E91E63"  # pink  — live spot
COLOUR_SPOT_TGT = "#FFEB3B"  # yellow — target spot
BG_PLOT = "#1E1E1E"
BG_AXES = "#252525"
FG_TEXT = "#DDDDDD"

# Skew colour thresholds (call-put convention: skew = call_iv - put_iv)
SKEW_THRESHOLD_LOW = -5.0  # outer boundary
SKEW_THRESHOLD_ZERO = 0.0  # mid boundary
SKEW_THRESHOLD_HIGH = 5.0  # outer boundary
SKEW_COLOUR_GREEN = "#4CAF50"  # most favourable
SKEW_COLOUR_YELLOW = "#FFC107"  # mildly favourable
SKEW_COLOUR_ORANGE = "#FF9800"  # mildly unfavourable
SKEW_COLOUR_RED = "#F44336"  # most unfavourable
SKEW_COLOUR_NEUTRAL = "#888888"  # no data / loading


# =============================================================================
# VOL INFO BAR  (full-width strip above the main splitter)
# =============================================================================


class VolInfoBar(QFrame):
    """
    Narrow full-width bar displaying live BTC DVOL and index price fetched
    from Deribit.  Updated via update_data() slot (driven by GUIOrchestrator).
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)
        self.setFixedHeight(28)
        self.setStyleSheet(
            "VolInfoBar { background: #1a1a2e; border-bottom: 1px solid #333; }"
        )
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 2, 10, 2)
        layout.setSpacing(20)

        def _lbl(text: str, bold: bool = False) -> QLabel:
            lbl = QLabel(text)
            lbl.setStyleSheet(
                f"color: #aaa; font-size: 11px;"
                + ("font-weight: bold;" if bold else "")
            )
            return lbl

        self._btc_price_lbl = _lbl("BTC LIVE: —")
        self._dvol_lbl = _lbl("DVOL: —")
        self._dvol_hl_lbl = _lbl("24h H/L: — / —")
        self._updated_lbl = _lbl("")

        layout.addWidget(_lbl("Deribit  ", bold=True))
        layout.addWidget(self._btc_price_lbl)
        layout.addWidget(self._dvol_lbl)
        layout.addWidget(self._dvol_hl_lbl)
        layout.addStretch()
        layout.addWidget(self._updated_lbl)

    @Slot(float, float, float, float)
    def update_data(
        self,
        dvol: float,
        dvol_high: float,
        dvol_low: float,
        index_price: float,
    ) -> None:
        self._btc_price_lbl.setText(f"BTC LIVE: ${index_price:,.0f}")
        self._dvol_lbl.setText(f"DVOL: {dvol:.1f}")
        self._dvol_hl_lbl.setText(f"24h H/L: {dvol_high:.1f} / {dvol_low:.1f}")
        ts = datetime.now(timezone.utc).strftime("%H:%M UTC")
        self._updated_lbl.setText(f"updated {ts}")

    @Slot(float)
    def update_live_spot(self, price: float) -> None:
        """Update the spot price label from the live WS stream (called at ~500ms cadence)."""
        self._btc_price_lbl.setText(f"BTC LIVE: ${price:,.0f}")

    def set_loading(self) -> None:
        self._btc_price_lbl.setText("BTC LIVE: …")
        self._dvol_lbl.setText("DVOL: …")
        self._dvol_hl_lbl.setText("24h H/L: …")
        self._updated_lbl.setText("")

    def set_error(self, msg: str = "fetch error") -> None:
        self._dvol_lbl.setText(f"DVOL: {msg}")
        self._updated_lbl.setText("")


# =============================================================================
# SECTION 1 — DATA LAYER: Coincall instrument loader (public REST, startup)
# =============================================================================


class CoincallInstrumentLoader:
    """
    Fetches all live BTC option instruments from Coincall once at startup.

    Endpoint: GET https://api.coincall.com/open/option/getInstruments/BTC
    No authentication required.

    Organises result as:
        expiry_labels : list[str]   e.g. ["20MAR26", "25APR26", ...]
        strikes_by_expiry : dict[expiry_label → sorted list[str]]
            e.g. {"20MAR26": ["70000", "72000", ...], ...}
    Both C and P exist at each strike; the type dropdown is always ["C","P"].
    """

    ENDPOINT = "/open/option/getInstruments/BTC"

    def __init__(self) -> None:
        self.expiry_labels: List[str] = []
        self.strikes_by_expiry: Dict[str, List[str]] = {}
        self._raw: List[Dict] = []

    async def load(self) -> None:
        url = COINCALL_REST_BASE + self.ENDPOINT
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url, timeout=aiohttp.ClientTimeout(total=10)
                ) as r:
                    data = await r.json()
        except Exception as exc:
            LOG.error("Failed to fetch Coincall instruments: %s", exc)
            return

        instruments = data.get("data", [])
        if not instruments:
            LOG.error("Empty instrument list from Coincall")
            return

        self._raw = instruments
        self._organise()
        LOG.info(
            "Loaded %d Coincall instruments across %d expiries",
            len(instruments),
            len(self.expiry_labels),
        )

    def _organise(self) -> None:
        """
        Parse symbolName = "BTCUSD-DDMMMYY-STRIKE-C/P".
        Groups by expiry, collects unique strikes per expiry.
        Sorts expiries chronologically using expirationTimestamp.
        """
        expiry_ts: Dict[str, int] = {}
        strikes: Dict[str, Set[float]] = {}

        for inst in self._raw:
            name = inst.get("symbolName", "")
            parts = name.split("-")
            if len(parts) != 4:
                continue
            expiry = parts[1]
            try:
                strike = float(parts[2])
            except ValueError:
                continue
            ts = inst.get("expirationTimestamp", 0)
            expiry_ts.setdefault(expiry, ts)
            strikes.setdefault(expiry, set()).add(strike)

        sorted_expiries = sorted(expiry_ts.keys(), key=lambda e: expiry_ts[e])
        self.expiry_labels = sorted_expiries
        self.strikes_by_expiry = {
            e: [str(int(k)) for k in sorted(strikes[e])] for e in sorted_expiries
        }


# =============================================================================
# SECTION 2 — GUI: Leg row widget
# =============================================================================


class LegRow(QWidget):
    """
    One leg row: [Dir▼] [Qty] [Expiry▼] [Strike▼] [Type▼] [skew label]
    All dropdowns populated from CoincallInstrumentLoader data.
    """

    changed = Signal()
    # Emitted when expiry, type, or direction changes — carries self so the
    # orchestrator can dispatch a skew fetch for this specific row.
    skew_needed = Signal(object)

    def __init__(
        self,
        loader: CoincallInstrumentLoader,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._loader = loader
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(4)

        # Direction
        self._dir = QComboBox()
        self._dir.addItems(["L", "S"])
        self._dir.setFixedWidth(40)
        self._dir.setToolTip("L = Long (Buy), S = Short (Sell)")

        # Quantity
        self._qty = QDoubleSpinBox()
        self._qty.setRange(0.01, 100.0)
        self._qty.setSingleStep(0.01)
        self._qty.setDecimals(2)
        self._qty.setValue(0.10)
        self._qty.setFixedWidth(64)
        self._qty.setToolTip("Quantity (BTC contracts)")

        # Expiry
        self._expiry = QComboBox()
        self._expiry.setMinimumWidth(80)
        self._expiry.setToolTip("Expiry date")

        # Strike
        self._strike = QComboBox()
        self._strike.setMinimumWidth(72)
        self._strike.setToolTip("Strike price")

        # Type
        self._type = QComboBox()
        self._type.addItems(["C", "P"])
        self._type.setFixedWidth(40)
        self._type.setToolTip("C = Call, P = Put")

        layout.addWidget(self._dir)
        layout.addWidget(self._qty)
        layout.addWidget(self._expiry)
        layout.addWidget(self._strike)
        layout.addWidget(self._type)

        # Skew label — shown to the right of the Type combo
        self._skew_lbl = QLabel("skew: —")
        self._skew_lbl.setFixedWidth(90)
        self._skew_lbl.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self._skew_lbl.setStyleSheet(f"color: {SKEW_COLOUR_NEUTRAL}; font-size: 11px;")
        self._skew_lbl.setToolTip("25-delta skew for this maturity (call IV − put IV)")
        layout.addWidget(self._skew_lbl)

        # Populate expiry from loader (may be empty if not loaded yet)
        self._populate_expiry()

        # Connections
        self._expiry.currentIndexChanged.connect(self._on_expiry_changed)
        self._dir.currentIndexChanged.connect(self._emit_changed)
        self._qty.valueChanged.connect(self._emit_changed)
        self._strike.currentIndexChanged.connect(self._emit_changed)
        self._type.currentIndexChanged.connect(self._emit_changed)

        # Emit skew_needed when the fields that affect skew colour change
        self._expiry.currentIndexChanged.connect(self._on_skew_trigger)
        self._dir.currentIndexChanged.connect(self._on_skew_trigger)
        self._type.currentIndexChanged.connect(self._on_skew_trigger)

    def _populate_expiry(self) -> None:
        self._expiry.blockSignals(True)
        self._expiry.clear()
        if self._loader.expiry_labels:
            self._expiry.addItems(self._loader.expiry_labels)
        else:
            self._expiry.addItem("Loading…")
        self._expiry.blockSignals(False)
        self._refresh_strikes()

    def refresh_from_loader(self) -> None:
        """Called after instrument loader completes."""
        current_expiry = self._expiry.currentText()
        self._expiry.blockSignals(True)
        self._expiry.clear()
        self._expiry.addItems(self._loader.expiry_labels)
        # Restore previous selection if still valid
        idx = self._expiry.findText(current_expiry)
        if idx >= 0:
            self._expiry.setCurrentIndex(idx)
        self._expiry.blockSignals(False)
        self._refresh_strikes()

    @Slot(int)
    def _on_expiry_changed(self, _index: int = 0) -> None:
        self._refresh_strikes()
        self.changed.emit()

    @Slot(int)
    @Slot(float)
    def _emit_changed(self, *_args: object) -> None:
        self.changed.emit()

    def _refresh_strikes(self) -> None:
        expiry = self._expiry.currentText()
        strikes = self._loader.strikes_by_expiry.get(expiry, [])
        current_strike = self._strike.currentText()
        self._strike.blockSignals(True)
        self._strike.clear()
        self._strike.addItems(strikes)
        idx = self._strike.findText(current_strike)
        if idx >= 0:
            self._strike.setCurrentIndex(idx)
        self._strike.blockSignals(False)

    def to_spec_string(self) -> Optional[str]:
        """
        Returns a spec string parseable by db_option_pricer.parse_leg(),
        e.g. "L 0.60 20MAR26-70000-P", or None if data not ready.
        """
        expiry = self._expiry.currentText()
        strike = self._strike.currentText()
        if not expiry or not strike or expiry == "Loading…":
            return None
        direction = self._dir.currentText()
        qty = self._qty.value()
        opt_type = self._type.currentText()
        return f"{direction} {qty:.2f} {expiry}-{strike}-{opt_type}"

    def set_from_spec(self, spec: str) -> None:
        """
        Populate row from a spec string "L 0.6 20MAR26-70000-P".
        Silently ignores invalid specs.
        """
        m = re.match(
            r"^([LS])\s+([\d.]+)\s+(\d{1,2}[A-Z]{3}\d{2})-(\d+)-([CP])$",
            spec.strip(),
            re.IGNORECASE,
        )
        if not m:
            return
        direction, qty_s, expiry, strike, opt_type = m.groups()
        self._dir.setCurrentText(direction.upper())
        self._qty.setValue(float(qty_s))
        idx_e = self._expiry.findText(expiry.upper())
        if idx_e >= 0:
            self._expiry.setCurrentIndex(idx_e)
            self._refresh_strikes()
        idx_k = self._strike.findText(str(int(float(strike))))
        if idx_k >= 0:
            self._strike.setCurrentIndex(idx_k)
        self._type.setCurrentText(opt_type.upper())

    # ------------------------------------------------------------------
    # Skew label helpers
    # ------------------------------------------------------------------

    def get_maturity(self) -> str:
        """Return the currently selected expiry label (e.g. '20MAR26')."""
        return self._expiry.currentText()

    def get_direction(self) -> str:
        """Return 'L' or 'S'."""
        return self._dir.currentText()

    def get_option_type(self) -> str:
        """Return 'C' or 'P'."""
        return self._type.currentText()

    @Slot(int)
    def _on_skew_trigger(self, _index: int = 0) -> None:
        """Emit skew_needed whenever expiry, direction, or type changes."""
        if self._expiry.currentText() and self._expiry.currentText() != "Loading…":
            self.skew_needed.emit(self)

    def set_skew_loading(self) -> None:
        """Show loading indicator while skew fetch is in progress."""
        self._skew_lbl.setText("skew: …")
        self._skew_lbl.setStyleSheet(f"color: {SKEW_COLOUR_NEUTRAL}; font-size: 11px;")

    def set_skew(self, skew: Optional[float]) -> None:
        """
        Display skew value with 4-colour coding based on direction and option type.

        Colour logic (call-put convention: skew = call_iv - put_iv):

          Long Call / Short Put  → lower skew is better (puts pricier = headwind)
            green  : skew < -5.0          most favourable
            yellow : -5.0 ≤ skew < 0.0   mildly favourable
            orange : 0.0 ≤ skew < 5.0    mildly unfavourable
            red    : skew ≥ 5.0           most unfavourable

          Long Put / Short Call  → higher skew is better (calls pricier = headwind)
            green  : skew > 5.0           most favourable
            yellow : 0.0 < skew ≤ 5.0    mildly favourable
            orange : -5.0 ≤ skew ≤ 0.0   mildly unfavourable
            red    : skew < -5.0          most unfavourable
        """
        if skew is None:
            self._skew_lbl.setText("skew: —")
            self._skew_lbl.setStyleSheet(
                f"color: {SKEW_COLOUR_NEUTRAL}; font-size: 11px;"
            )
            return

        direction = self.get_direction()  # "L" or "S"
        opt_type = self.get_option_type()  # "C" or "P"

        # Long Call or Short Put → low skew is favourable
        low_is_good = (direction == "L" and opt_type == "C") or (
            direction == "S" and opt_type == "P"
        )

        if low_is_good:
            if skew < SKEW_THRESHOLD_LOW:
                colour = SKEW_COLOUR_GREEN
            elif skew < SKEW_THRESHOLD_ZERO:
                colour = SKEW_COLOUR_YELLOW
            elif skew < SKEW_THRESHOLD_HIGH:
                colour = SKEW_COLOUR_ORANGE
            else:
                colour = SKEW_COLOUR_RED
        else:
            # Long Put or Short Call → high skew is favourable
            if skew > SKEW_THRESHOLD_HIGH:
                colour = SKEW_COLOUR_GREEN
            elif skew > SKEW_THRESHOLD_ZERO:
                colour = SKEW_COLOUR_YELLOW
            elif skew > SKEW_THRESHOLD_LOW:
                colour = SKEW_COLOUR_ORANGE
            else:
                colour = SKEW_COLOUR_RED

        sign = "+" if skew >= 0 else ""
        self._skew_lbl.setText(f"skew: {sign}{skew:.1f}")
        self._skew_lbl.setStyleSheet(
            f"color: {colour}; font-size: 11px; font-weight: bold;"
        )


# =============================================================================
# SECTION 2b — GUI: PricingTabWidget (Current Spot / Target Spot pricing panel)
# =============================================================================


class PricingTabWidget(QWidget):
    """
    Two-tab pricing display embedded in the left panel.

    Tab 0 — Current Spot:
        Compact grid table: one header row + one row per leg + Total + Max DD footer.

    Tab 1 — Target Spot:
        Two compact grid tables (flat vol / vol-shocked), each with their own
        Total and Max DD footer rows.  Section headers separate the two tables.

    A QLineEdit for the target-spot price appears as a corner widget on the
    tab bar — visible only when the Target Spot tab is active.

    Public API:
        set_leg_names(names)
        update_current(result, dd_flat_usd)
        update_target(flat_result, shock_result, dd_flat_usd, dd_shock_usd, vol_shift)
        get_target_spot() -> Optional[float]
        clear()
    """

    # ── Style constants ────────────────────────────────────────────────
    _HDR_STYLE = "color: #666; font-size: 10px;"
    _NAME_STYLE = "color: #88aacc; font-size: 10px;"
    _CELL_STYLE = "color: #cccccc; font-size: 10px;"
    _TOTAL_STYLE = "color: #e0e0e0; font-size: 10px; font-weight: bold;"
    _DD_STYLE = "color: #ff9966; font-size: 10px; font-weight: bold;"
    _SEC_STYLE = "color: #555; font-size: 9px;"

    # Column indices
    _C_NAME, _C_FWD, _C_IV, _C_PRICE = 0, 1, 2, 3

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._leg_names: List[str] = []
        self._build_ui()

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._tabs = QTabWidget()
        self._tabs.setDocumentMode(True)

        # Corner widget: target-spot input (shown only on tab 1)
        corner = QWidget()
        corner_layout = QHBoxLayout(corner)
        corner_layout.setContentsMargins(4, 2, 6, 2)
        corner_layout.setSpacing(4)
        self._tgt_label = QLabel("$:")
        self._tgt_label.setStyleSheet("color: #aaa; font-size: 11px;")
        self._tgt_input = QLineEdit("")
        self._tgt_input.setFixedWidth(80)
        self._tgt_input.setPlaceholderText("e.g. 80000")
        self._tgt_input.setToolTip("Target spot price used for structure pricing")
        self._tgt_input.setStyleSheet("font-size: 11px;")
        corner_layout.addWidget(self._tgt_label)
        corner_layout.addWidget(self._tgt_input)
        self._corner_widget = corner
        self._corner_widget.setVisible(False)
        self._tabs.setCornerWidget(self._corner_widget, Qt.TopRightCorner)

        # Tab 0 — Current Spot (scroll-wrapped so it never clips with many legs)
        self._cur_scroll = QScrollArea()
        self._cur_scroll.setWidgetResizable(True)
        self._cur_scroll.setFrameShape(QFrame.NoFrame)
        self._cur_inner = QWidget()
        self._cur_layout = QVBoxLayout(self._cur_inner)
        self._cur_layout.setContentsMargins(6, 6, 6, 6)
        self._cur_layout.setSpacing(4)
        self._cur_scroll.setWidget(self._cur_inner)
        self._tabs.addTab(self._cur_scroll, "Current Spot")

        # Tab 1 — Target Spot
        self._tgt_scroll = QScrollArea()
        self._tgt_scroll.setWidgetResizable(True)
        self._tgt_scroll.setFrameShape(QFrame.NoFrame)
        self._tgt_inner = QWidget()
        self._tgt_layout = QVBoxLayout(self._tgt_inner)
        self._tgt_layout.setContentsMargins(6, 6, 6, 6)
        self._tgt_layout.setSpacing(6)
        self._tgt_scroll.setWidget(self._tgt_inner)
        self._tabs.addTab(self._tgt_scroll, "Target Spot")

        self._tabs.currentChanged.connect(self._on_tab_changed)
        root.addWidget(self._tabs)

        self._rebuild_leg_widgets()

    def _on_tab_changed(self, index: int) -> None:
        self._corner_widget.setVisible(index == 1)

    # ------------------------------------------------------------------
    # Layout helpers
    # ------------------------------------------------------------------

    def _lbl(
        self, text: str, style: str, align: Qt.AlignmentFlag = Qt.AlignRight
    ) -> QLabel:
        w = QLabel(text)
        w.setStyleSheet(style)
        w.setAlignment(align)
        return w

    def _clear_layout(self, layout: QVBoxLayout) -> None:
        """Recursively remove and delete all items from a QVBoxLayout."""
        while layout.count():
            item = layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
            else:
                sub = item.layout()
                if sub is not None:
                    self._clear_layout(sub)  # type: ignore[arg-type]

    def _build_grid(
        self,
        parent_layout: QVBoxLayout,
        name_labels: List[QLabel],
        fwd_labels: List[QLabel],
        iv_labels: List[QLabel],
        price_labels: List[QLabel],
        total_label: QLabel,
        dd_label: QLabel,
    ) -> None:
        """
        Build one compact grid table and append it to parent_layout.

        Layout (QGridLayout):
          row 0  : column headers  (Contract | Forward | IV | Price USD)
          rows 1…n : one leg per row
          row n+1 : separator line spanning all columns
          row n+2 : Total row   (label spans cols 0-2, value in col 3)
          row n+3 : Max DD row  (label spans cols 0-2, value in col 3)
        """
        container = QWidget()
        grid = QGridLayout(container)
        grid.setContentsMargins(0, 0, 0, 2)
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(1)

        # Column stretch: contract column stretches, others fixed
        grid.setColumnStretch(self._C_NAME, 1)
        grid.setColumnMinimumWidth(self._C_FWD, 64)
        grid.setColumnMinimumWidth(self._C_IV, 42)
        grid.setColumnMinimumWidth(self._C_PRICE, 68)

        # ── Header row ────────────────────────────────────────────────
        r = 0
        grid.addWidget(
            self._lbl("Contract", self._HDR_STYLE, Qt.AlignLeft), r, self._C_NAME
        )
        grid.addWidget(self._lbl("Forward", self._HDR_STYLE), r, self._C_FWD)
        grid.addWidget(self._lbl("IV", self._HDR_STYLE), r, self._C_IV)
        grid.addWidget(self._lbl("Price USD", self._HDR_STYLE), r, self._C_PRICE)
        r += 1

        # ── Leg data rows ─────────────────────────────────────────────
        for name_lbl, fwd_lbl, iv_lbl, price_lbl in zip(
            name_labels, fwd_labels, iv_labels, price_labels
        ):
            grid.addWidget(name_lbl, r, self._C_NAME)
            grid.addWidget(fwd_lbl, r, self._C_FWD)
            grid.addWidget(iv_lbl, r, self._C_IV)
            grid.addWidget(price_lbl, r, self._C_PRICE)
            r += 1

        # ── Separator ─────────────────────────────────────────────────
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color: #444;")
        sep.setFixedHeight(1)
        grid.addWidget(sep, r, 0, 1, 4)
        r += 1

        # ── Total row ─────────────────────────────────────────────────
        grid.addWidget(
            self._lbl("Total", self._TOTAL_STYLE, Qt.AlignLeft), r, self._C_NAME, 1, 3
        )
        total_label.setAlignment(Qt.AlignRight)
        grid.addWidget(total_label, r, self._C_PRICE)
        r += 1

        # ── Max DD row ────────────────────────────────────────────────
        grid.addWidget(
            self._lbl("Max DD", self._DD_STYLE, Qt.AlignLeft), r, self._C_NAME, 1, 3
        )
        dd_label.setAlignment(Qt.AlignRight)
        grid.addWidget(dd_label, r, self._C_PRICE)

        parent_layout.addWidget(container)

    def _rebuild_leg_widgets(self) -> None:
        """
        Recreate all per-leg label widgets for both tabs from self._leg_names.
        Called whenever the leg list changes (set_leg_names).
        """
        # ── Current Spot tab ──────────────────────────────────────────
        self._clear_layout(self._cur_layout)

        self._cur_name: List[QLabel] = []
        self._cur_fwd: List[QLabel] = []
        self._cur_iv: List[QLabel] = []
        self._cur_price: List[QLabel] = []

        for name in self._leg_names:
            self._cur_name.append(self._lbl(name, self._NAME_STYLE, Qt.AlignLeft))
            self._cur_fwd.append(self._lbl("—", self._CELL_STYLE))
            self._cur_iv.append(self._lbl("—", self._CELL_STYLE))
            self._cur_price.append(self._lbl("—", self._CELL_STYLE))

        self._cur_total = self._lbl("—", self._TOTAL_STYLE)
        self._cur_dd = self._lbl("—", self._DD_STYLE)

        self._build_grid(
            self._cur_layout,
            self._cur_name,
            self._cur_fwd,
            self._cur_iv,
            self._cur_price,
            self._cur_total,
            self._cur_dd,
        )
        self._cur_layout.addStretch()

        # ── Target Spot tab ───────────────────────────────────────────
        self._clear_layout(self._tgt_layout)

        self._tgt_flat_name: List[QLabel] = []
        self._tgt_flat_fwd: List[QLabel] = []
        self._tgt_flat_iv: List[QLabel] = []
        self._tgt_flat_price: List[QLabel] = []

        self._tgt_shock_name: List[QLabel] = []
        self._tgt_shock_fwd: List[QLabel] = []
        self._tgt_shock_iv: List[QLabel] = []
        self._tgt_shock_price: List[QLabel] = []

        for name in self._leg_names:
            self._tgt_flat_name.append(self._lbl(name, self._NAME_STYLE, Qt.AlignLeft))
            self._tgt_flat_fwd.append(self._lbl("—", self._CELL_STYLE))
            self._tgt_flat_iv.append(self._lbl("—", self._CELL_STYLE))
            self._tgt_flat_price.append(self._lbl("—", self._CELL_STYLE))

            self._tgt_shock_name.append(self._lbl(name, self._NAME_STYLE, Qt.AlignLeft))
            self._tgt_shock_fwd.append(self._lbl("—", self._CELL_STYLE))
            self._tgt_shock_iv.append(self._lbl("—", self._CELL_STYLE))
            self._tgt_shock_price.append(self._lbl("—", self._CELL_STYLE))

        self._tgt_flat_total = self._lbl("—", self._TOTAL_STYLE)
        self._tgt_flat_dd = self._lbl("—", self._DD_STYLE)
        self._tgt_shock_total = self._lbl("—", self._TOTAL_STYLE)
        self._tgt_shock_dd = self._lbl("—", self._DD_STYLE)

        # Section header — flat vol
        self._tgt_flat_hdr = self._lbl("── Flat vol ──", self._SEC_STYLE, Qt.AlignLeft)
        self._tgt_layout.addWidget(self._tgt_flat_hdr)
        self._build_grid(
            self._tgt_layout,
            self._tgt_flat_name,
            self._tgt_flat_fwd,
            self._tgt_flat_iv,
            self._tgt_flat_price,
            self._tgt_flat_total,
            self._tgt_flat_dd,
        )

        # Section header — vol shocked (text updated dynamically)
        self._tgt_shock_hdr = self._lbl(
            "── Vol Shock ──", self._SEC_STYLE, Qt.AlignLeft
        )
        self._tgt_layout.addWidget(self._tgt_shock_hdr)
        self._build_grid(
            self._tgt_layout,
            self._tgt_shock_name,
            self._tgt_shock_fwd,
            self._tgt_shock_iv,
            self._tgt_shock_price,
            self._tgt_shock_total,
            self._tgt_shock_dd,
        )

        self._tgt_layout.addStretch()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_leg_names(self, names: List[str]) -> None:
        """Rebuild widgets if the leg list has changed."""
        if names != self._leg_names:
            self._leg_names = list(names)
            self._rebuild_leg_widgets()

    def get_target_spot(self) -> Optional[float]:
        try:
            val = float(self._tgt_input.text().replace(",", ""))
            return val if val > 0 else None
        except ValueError:
            return None

    def update_current(
        self,
        result: "StructurePriceResult",  # type: ignore[name-defined]
        dd_flat_usd: float,
    ) -> None:
        """Populate the Current Spot tab."""
        for i in range(len(self._cur_fwd)):
            try:
                self._cur_fwd[i].setText(f"${result.leg_forwards[i]:,.0f}")
                self._cur_iv[i].setText(f"{result.leg_ivs[i] * 100:.1f}%")
                self._cur_price[i].setText(f"${result.leg_prices_usd[i]:,.0f}")
            except (IndexError, Exception):
                pass
        self._cur_total.setText(f"${result.total_usd:,.0f}")
        self._cur_dd.setText(f"${dd_flat_usd:,.0f}")

    def update_target(
        self,
        flat_result: "StructurePriceResult",  # type: ignore[name-defined]
        shock_result: "StructurePriceResult",  # type: ignore[name-defined]
        dd_flat_usd: float,
        dd_shock_usd: float,
        vol_shift: float,
    ) -> None:
        """Populate the Target Spot tab (flat and vol-shocked sections)."""
        self._tgt_shock_hdr.setText(f"── Vol Shock ({vol_shift:+.0%}) ──")

        for i in range(len(self._tgt_flat_fwd)):
            try:
                self._tgt_flat_fwd[i].setText(f"${flat_result.leg_forwards[i]:,.0f}")
                self._tgt_flat_iv[i].setText(f"{flat_result.leg_ivs[i] * 100:.1f}%")
                self._tgt_flat_price[i].setText(
                    f"${flat_result.leg_prices_usd[i]:,.0f}"
                )

                self._tgt_shock_fwd[i].setText(f"${shock_result.leg_forwards[i]:,.0f}")
                self._tgt_shock_iv[i].setText(f"{shock_result.leg_ivs[i] * 100:.1f}%")
                self._tgt_shock_price[i].setText(
                    f"${shock_result.leg_prices_usd[i]:,.0f}"
                )
            except (IndexError, Exception):
                pass

        self._tgt_flat_total.setText(f"${flat_result.total_usd:,.0f}")
        self._tgt_flat_dd.setText(f"${dd_flat_usd:,.0f}")
        self._tgt_shock_total.setText(f"${shock_result.total_usd:,.0f}")
        self._tgt_shock_dd.setText(f"${dd_shock_usd:,.0f}")

    def clear(self) -> None:
        """Reset all data labels to '—'."""
        for w in (
            self._cur_fwd
            + self._cur_iv
            + self._cur_price
            + [self._cur_total, self._cur_dd]
            + self._tgt_flat_fwd
            + self._tgt_flat_iv
            + self._tgt_flat_price
            + [self._tgt_flat_total, self._tgt_flat_dd]
            + self._tgt_shock_fwd
            + self._tgt_shock_iv
            + self._tgt_shock_price
            + [self._tgt_shock_total, self._tgt_shock_dd]
        ):
            w.setText("—")


# =============================================================================
# SECTION 3 — GUI: Leg panel (left 1/3)
# =============================================================================


class LegPanel(QWidget):
    """
    Left panel: account selector, up to 6 leg rows, inputs, outputs, buttons.
    """

    evaluate_requested = Signal(
        list, object, str
    )  # leg_specs, target_spot (float|None), account
    submit_requested = Signal()
    # Bubbles up from LegRow so GUIOrchestrator can dispatch a skew fetch
    skew_needed = Signal(object)  # carries the LegRow instance

    def __init__(
        self,
        loader: CoincallInstrumentLoader,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._loader = loader
        self._leg_rows: List[LegRow] = []
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        # ── Account selector ────────────────────────────────────────────
        acct_layout = QHBoxLayout()
        acct_layout.addWidget(QLabel("Account:"))
        self._account_combo = QComboBox()
        self._account_combo.addItems(["Sub", "Main"])
        self._account_combo.setToolTip(
            "Main → COINCALL_API_KEY / Sub → COINCALL_SUB_API_KEY"
        )
        self._account_combo.setStyleSheet(
            "QComboBox { background-color: #8b0000; color: white; font-weight: bold;"
            " border: 2px solid #ff4444; border-radius: 3px; padding: 2px 6px; }"
            "QComboBox:hover { background-color: #a00000; }"
            "QComboBox QAbstractItemView { background-color: #8b0000; color: white;"
            " selection-background-color: #cc0000; }"
        )
        acct_layout.addWidget(self._account_combo)
        acct_layout.addStretch()
        root.addLayout(acct_layout)

        # ── Leg group ───────────────────────────────────────────────────
        leg_group = QGroupBox("Legs")
        leg_vbox = QVBoxLayout(leg_group)
        leg_vbox.setSpacing(2)

        # Header row
        hdr = QHBoxLayout()
        for txt, w in [
            ("Dir", 40),
            ("Qty", 64),
            ("Expiry", 80),
            ("Strike", 72),
            ("Type", 40),
            ("25Δ Skew", 90),
        ]:
            lbl = QLabel(txt)
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setFixedWidth(w)
            hdr.addWidget(lbl)
        leg_vbox.addLayout(hdr)

        # Legs container (scrollable for safety)
        self._leg_container = QVBoxLayout()
        self._leg_container.setSpacing(2)
        leg_vbox.addLayout(self._leg_container)

        # Add/remove buttons
        btn_row = QHBoxLayout()
        self._btn_add = QPushButton("+ Add Leg")
        self._btn_add.clicked.connect(self._add_leg)
        self._btn_remove = QPushButton("− Remove")
        self._btn_remove.clicked.connect(self._remove_leg)
        btn_row.addWidget(self._btn_add)
        btn_row.addWidget(self._btn_remove)
        leg_vbox.addLayout(btn_row)
        root.addWidget(leg_group)

        # Start with one leg
        self._add_leg()

        # ── Pricing tab widget ───────────────────────────────────────────
        pricing_group = QGroupBox("Pricing")
        pricing_vbox = QVBoxLayout(pricing_group)
        pricing_vbox.setContentsMargins(4, 4, 4, 4)
        pricing_vbox.setSpacing(0)
        self._pricing = PricingTabWidget()
        pricing_vbox.addWidget(self._pricing)
        root.addWidget(pricing_group)

        rfq_group = QGroupBox("RFQ Controls")
        rfq_form = QFormLayout(rfq_group)
        rfq_form.setContentsMargins(8, 8, 8, 8)
        rfq_form.setSpacing(6)

        self._threshold_type = QComboBox()
        self._threshold_type.addItems(
            [ThresholdType.DEBIT.value, ThresholdType.CREDIT.value]
        )
        self._threshold_type.setCurrentText(ThresholdType.DEBIT.value)
        rfq_form.addRow("Threshold Type", self._threshold_type)

        self._threshold_value = QDoubleSpinBox()
        self._threshold_value.setRange(0.0, 1_000_000.0)
        self._threshold_value.setDecimals(2)
        self._threshold_value.setSingleStep(10.0)
        self._threshold_value.setValue(550.0)
        self._threshold_value.setPrefix("$")
        rfq_form.addRow("Threshold Value", self._threshold_value)

        self._max_dd_flat = QDoubleSpinBox()
        self._max_dd_flat.setRange(-10_000_000.0, 0.0)
        self._max_dd_flat.setDecimals(2)
        self._max_dd_flat.setSingleStep(100.0)
        self._max_dd_flat.setValue(-2000.0)
        self._max_dd_flat.setPrefix("$")
        rfq_form.addRow("Max DD Flat", self._max_dd_flat)

        self._max_dd_shocked = QDoubleSpinBox()
        self._max_dd_shocked.setRange(-10_000_000.0, 0.0)
        self._max_dd_shocked.setDecimals(2)
        self._max_dd_shocked.setSingleStep(100.0)
        self._max_dd_shocked.setValue(-1800.0)
        self._max_dd_shocked.setPrefix("$")
        rfq_form.addRow("Max DD Shocked", self._max_dd_shocked)

        root.addWidget(rfq_group)

        # ── Action buttons ──────────────────────────────────────────────
        self._btn_evaluate = QPushButton("Evaluate")
        self._btn_evaluate.setFixedHeight(32)
        self._btn_evaluate.clicked.connect(self._on_evaluate)
        root.addWidget(self._btn_evaluate)

        self._btn_submit = QPushButton("Submit RFQ ▶")
        self._btn_submit.setFixedHeight(32)
        self._btn_submit.setEnabled(False)
        self._btn_submit.clicked.connect(self.submit_requested.emit)
        root.addWidget(self._btn_submit)

        flow_group = QGroupBox("RFQ Flow")
        flow_layout = QVBoxLayout(flow_group)
        flow_layout.setContentsMargins(4, 4, 4, 4)
        self._rfq_flow = QPlainTextEdit()
        self._rfq_flow.setReadOnly(True)
        self._rfq_flow.setMaximumBlockCount(500)
        self._rfq_flow.setFixedHeight(140)
        flow_layout.addWidget(self._rfq_flow)
        root.addWidget(flow_group)

        root.addStretch()

    # ------------------------------------------------------------------
    # Leg management
    # ------------------------------------------------------------------

    def _add_leg(self) -> None:
        if len(self._leg_rows) >= MAX_LEGS:
            return
        row = LegRow(self._loader)
        row.changed.connect(self._on_leg_changed)
        row.skew_needed.connect(self.skew_needed.emit)
        self._leg_rows.append(row)
        self._leg_container.addWidget(row)
        self._update_add_remove_state()

    def _remove_leg(self) -> None:
        if len(self._leg_rows) <= 1:
            return
        row = self._leg_rows.pop()
        self._leg_container.removeWidget(row)
        row.deleteLater()
        self._update_add_remove_state()

    def _update_add_remove_state(self) -> None:
        self._btn_add.setEnabled(len(self._leg_rows) < MAX_LEGS)
        self._btn_remove.setEnabled(len(self._leg_rows) > 1)

    def _on_leg_changed(self) -> None:
        # Any leg change invalidates current evaluation
        self._btn_submit.setEnabled(False)

    def refresh_leg_dropdowns(self) -> None:
        """Called after instrument loader completes."""
        for row in self._leg_rows:
            row.refresh_from_loader()

    # ------------------------------------------------------------------
    # Data access
    # ------------------------------------------------------------------

    def get_leg_specs(self) -> List[str]:
        specs = []
        for row in self._leg_rows:
            s = row.to_spec_string()
            if s:
                specs.append(s)
        return specs

    def get_target_spot(self) -> Optional[float]:
        return self._pricing.get_target_spot()

    def get_account(self) -> str:
        return self._account_combo.currentText().lower()

    def get_threshold_type(self) -> ThresholdType:
        return ThresholdType(self._threshold_type.currentText())

    def get_threshold_value(self) -> float:
        return float(self._threshold_value.value())

    def get_max_dd_flat(self) -> float:
        return float(self._max_dd_flat.value())

    def get_max_dd_shocked(self) -> float:
        return float(self._max_dd_shocked.value())

    def get_leg_rows(self) -> List["LegRow"]:
        """Return a copy of the current leg rows list."""
        return list(self._leg_rows)

    # ------------------------------------------------------------------
    # Output setters (called from GUIOrchestrator via Qt signals)
    # ------------------------------------------------------------------

    @Slot(str)
    def set_dd_struct(self, text: str) -> None:
        # Forward to pricing widget's current-spot tab footer
        self._pricing._cur_dd.setText(text)

    def update_pricing(
        self,
        cur_result: object,
        tgt_flat_result: object,
        tgt_shock_result: object,
        dd_flat_usd: float,
        dd_shock_usd: float,
        vol_shift: float,
    ) -> None:
        """Push fresh pricing data into both tabs."""
        self._pricing.update_current(cur_result, dd_flat_usd)  # type: ignore[arg-type]
        self._pricing.update_target(  # type: ignore[arg-type]
            tgt_flat_result, tgt_shock_result, dd_flat_usd, dd_shock_usd, vol_shift
        )

    def enable_submit(self, enabled: bool) -> None:
        self._btn_submit.setEnabled(enabled)

    @Slot(str)
    def append_rfq_flow(self, line: str) -> None:
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        self._rfq_flow.appendPlainText(f"{ts} {line}")

    def clear_rfq_flow(self) -> None:
        self._rfq_flow.clear()

    # ------------------------------------------------------------------
    # Evaluate trigger
    # ------------------------------------------------------------------

    def _on_evaluate(self) -> None:
        specs = self.get_leg_specs()
        if not specs:
            LOG.warning("No valid leg specs to evaluate")
            return
        target = self.get_target_spot()  # None → evaluate at current live spot
        # Push contract names into the pricing widget before emitting
        names = []
        for spec in specs:
            # spec format: "L 0.10 20MAR26-80000-C" → "BTC-20MAR26-80000-C"
            m = re.match(
                r"^[LS]\s+[\d.]+\s+((\d{1,2}[A-Z]{3}\d{2})-(\d+)-([CP]))$",
                spec.strip(),
                re.IGNORECASE,
            )
            names.append(f"BTC-{m.group(1).upper()}" if m else spec)
        self._pricing.set_leg_names(names)
        account = self.get_account()
        self.evaluate_requested.emit(specs, target, account)


# =============================================================================
# SECTION 4 — GUI: Parameter bar (below plot)
# =============================================================================


class ParameterBar(QWidget):
    """
    Time shift slider and vol shift slider.
    """

    params_changed = Signal(float, float)  # time_days, vol_shift

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._build_ui()
        self._suppress = False

    def _build_ui(self) -> None:
        layout = QGridLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(4)

        # ── Time shift ──────────────────────────────────────────────────
        layout.addWidget(QLabel("Time shift (days):"), 0, 0)

        self._time_slider = QSlider(Qt.Horizontal)
        self._time_slider.setRange(0, 300)  # 0–30.0 days, step 0.1
        self._time_slider.setValue(70)  # default 7.0 days
        self._time_slider.setToolTip("Days forward for theta decay")
        layout.addWidget(self._time_slider, 0, 1)

        self._time_spin = QDoubleSpinBox()
        self._time_spin.setRange(0.0, 30.0)
        self._time_spin.setSingleStep(0.1)
        self._time_spin.setDecimals(1)
        self._time_spin.setValue(7.0)
        self._time_spin.setFixedWidth(64)
        layout.addWidget(self._time_spin, 0, 2)

        # ── Vol shift ───────────────────────────────────────────────────
        layout.addWidget(QLabel("Vol shift:"), 1, 0)

        self._vol_slider = QSlider(Qt.Horizontal)
        self._vol_slider.setRange(-500, 500)  # -50% to +50%, step 0.1%
        self._vol_slider.setValue(-100)  # default -0.10
        self._vol_slider.setToolTip(
            "Uniform additive IV shift (e.g. −0.10 = −10 vol pts)"
        )
        layout.addWidget(self._vol_slider, 1, 1)

        self._vol_spin = QDoubleSpinBox()
        self._vol_spin.setRange(-0.50, 0.50)
        self._vol_spin.setSingleStep(0.01)
        self._vol_spin.setDecimals(2)
        self._vol_spin.setValue(-0.10)
        self._vol_spin.setFixedWidth(64)
        layout.addWidget(self._vol_spin, 1, 2)

        # ── Connections ─────────────────────────────────────────────────
        self._time_slider.valueChanged.connect(self._on_time_slider)
        self._time_spin.valueChanged.connect(self._on_time_spin)
        self._vol_slider.valueChanged.connect(self._on_vol_slider)
        self._vol_spin.valueChanged.connect(self._on_vol_spin)

    # ── Slider ↔ spinbox sync ────────────────────────────────────────
    @Slot(int)
    def _on_time_slider(self, val: int) -> None:
        if self._suppress:
            return
        self._suppress = True
        self._time_spin.setValue(val / 10.0)
        self._suppress = False
        self._emit_params()

    @Slot(float)
    def _on_time_spin(self, val: float) -> None:
        if self._suppress:
            return
        self._suppress = True
        self._time_slider.setValue(int(round(val * 10)))
        self._suppress = False
        self._emit_params()

    @Slot(int)
    def _on_vol_slider(self, val: int) -> None:
        if self._suppress:
            return
        self._suppress = True
        self._vol_spin.setValue(val / 1000.0)
        self._suppress = False
        self._emit_params()

    @Slot(float)
    def _on_vol_spin(self, val: float) -> None:
        if self._suppress:
            return
        self._suppress = True
        self._vol_slider.setValue(int(round(val * 1000)))
        self._suppress = False
        self._emit_params()

    def _emit_params(self, *_: object) -> None:
        self.params_changed.emit(
            self._time_spin.value(),
            self._vol_spin.value(),
        )

    # ── Accessors ───────────────────────────────────────────────────
    def time_days(self) -> float:
        return self._time_spin.value()

    def vol_shift(self) -> float:
        return self._vol_spin.value()


# =============================================================================
# SECTION 5 — GUI: Plot widget
# =============================================================================


class PlotWidget(QWidget):
    """
    Embeds a matplotlib Figure.  All drawing happens via update_plot().
    Actual canvas.draw() is rate-limited to ≤2 Hz via a QTimer dirty-flag.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._dirty = False

        self._fig = Figure(facecolor=BG_PLOT, tight_layout=True)
        self._ax = self._fig.add_subplot(111, facecolor=BG_AXES)
        self._canvas = FigureCanvasQTAgg(self._fig)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._canvas)

        self._style_axes()

        # Rate-limit timer
        self._timer = QTimer(self)
        self._timer.setInterval(PLOT_REFRESH_MS)
        self._timer.timeout.connect(self._maybe_redraw)
        self._timer.start()

        # Last data store
        self._spot_grid: Optional[np.ndarray] = None
        self._struct_pnl: Optional[np.ndarray] = None
        self._struct_shocked: Optional[np.ndarray] = None
        self._current_spot: float = 0.0
        self._target_spot: float = 0.0
        self._time_days: float = 7.0
        self._vol_shift: float = -0.10

        # Hover annotation (created once, toggled visible)
        self._hover_annot = self._ax.annotate(
            "",
            xy=(0, 0),
            xytext=(12, 12),
            textcoords="offset points",
            bbox=dict(
                boxstyle="round,pad=0.4", fc="#1a1a2e", ec="#555", lw=0.8, alpha=0.92
            ),
            fontsize=8,
            color=FG_TEXT,
            zorder=10,
        )
        self._hover_annot.set_visible(False)
        self._hover_vline = self._ax.axvline(
            x=0, color="#ffffff", lw=0.6, linestyle=":", alpha=0.4, zorder=9
        )
        self._hover_vline.set_visible(False)

        # Connect mouse motion
        self._canvas.mpl_connect("motion_notify_event", self._on_mouse_move)
        self._canvas.mpl_connect("axes_leave_event", self._on_axes_leave)

    def _style_axes(self) -> None:
        ax = self._ax
        ax.tick_params(colors=FG_TEXT, labelsize=8)
        ax.xaxis.label.set_color(FG_TEXT)
        ax.yaxis.label.set_color(FG_TEXT)
        ax.title.set_color(FG_TEXT)
        for spine in ax.spines.values():
            spine.set_edgecolor("#555555")
        ax.grid(True, color="#333333", linewidth=0.5, linestyle="--")
        ax.set_xlabel("Spot (USD)", color=FG_TEXT, fontsize=9)
        ax.set_ylabel("PnL (USD)", color=FG_TEXT, fontsize=9)

    def update_plot(
        self,
        spot_grid: np.ndarray,
        struct_pnl: np.ndarray,
        struct_shocked: np.ndarray,
        current_spot: float,
        target_spot: float,
        time_days: float,
        vol_shift: float,
    ) -> None:
        """Store data and mark dirty. Actual draw happens in timer callback."""
        self._spot_grid = spot_grid
        self._struct_pnl = struct_pnl
        self._struct_shocked = struct_shocked
        self._current_spot = current_spot
        self._target_spot = target_spot
        self._time_days = time_days
        self._vol_shift = vol_shift
        self._dirty = True

    @Slot()
    def _maybe_redraw(self) -> None:
        if not self._dirty or self._spot_grid is None:
            return
        self._dirty = False
        self._draw()

    def _draw(self) -> None:
        ax = self._ax
        ax.cla()
        self._style_axes()

        sg = self._spot_grid
        sp = self._struct_pnl
        ss = self._struct_shocked

        vol_label = f"vol{self._vol_shift:+.0%}"
        time_label = f"{self._time_days:.1f}d"

        # Structure lines (flat and vol-shocked)
        ax.plot(
            sg,
            sp,
            color=COLOUR_STRUCTURE,
            lw=1.8,
            label=f"Structure ({time_label}, flat vol)",
            zorder=3,
        )
        ax.plot(
            sg,
            ss,
            color=COLOUR_STRUCTURE,
            lw=1.2,
            linestyle="--",
            alpha=0.7,
            label=f"Structure ({time_label}, {vol_label})",
            zorder=3,
        )

        # Zero line
        ax.axhline(0, color=COLOUR_ZERO, lw=0.8, linestyle="-", zorder=2)

        # Live spot
        if self._current_spot > 0:
            ax.axvline(
                self._current_spot,
                color=COLOUR_SPOT_LIVE,
                lw=1.0,
                linestyle=":",
                label=f"Live spot {self._current_spot:,.0f}",
                zorder=2,
            )

        # Target spot
        if self._target_spot > 0:
            ax.axvline(
                self._target_spot,
                color=COLOUR_SPOT_TGT,
                lw=1.0,
                linestyle="--",
                label=f"Target spot {self._target_spot:,.0f}",
                zorder=2,
            )

        # Worst-spot marker on structure (flat)
        if sp is not None and len(sp):
            worst_idx = int(np.argmin(sp))
            ax.plot(
                sg[worst_idx],
                sp[worst_idx],
                "v",
                color=COLOUR_STRUCTURE,
                ms=7,
                zorder=5,
                label=f"Worst struct {sp[worst_idx]:,.0f}",
            )

        ax.legend(
            fontsize=7,
            facecolor=BG_AXES,
            edgecolor="#555",
            labelcolor=FG_TEXT,
            loc="best",
        )
        ax.set_title(
            f"PnL across spot grid  —  {time_label} forward",
            color=FG_TEXT,
            fontsize=9,
        )

        # Format x-axis as k
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v / 1000:.0f}k"))

        # Re-create hover annotation and vline after ax.cla() cleared them
        self._hover_annot = ax.annotate(
            "",
            xy=(0, 0),
            xytext=(12, 12),
            textcoords="offset points",
            bbox=dict(
                boxstyle="round,pad=0.4", fc="#1a1a2e", ec="#555", lw=0.8, alpha=0.92
            ),
            fontsize=8,
            color=FG_TEXT,
            zorder=10,
        )
        self._hover_annot.set_visible(False)
        self._hover_vline = ax.axvline(
            x=0, color="#ffffff", lw=0.6, linestyle=":", alpha=0.4, zorder=9
        )
        self._hover_vline.set_visible(False)

        self._canvas.draw_idle()

    def _on_mouse_move(self, event: object) -> None:
        """Show interpolated PnL values at cursor position."""
        # event is a matplotlib MouseEvent
        if (
            self._spot_grid is None
            or self._struct_pnl is None
            or self._struct_shocked is None
        ):
            return
        # Check cursor is inside the axes
        if not getattr(event, "inaxes", None) or event.inaxes is not self._ax:  # type: ignore[union-attr]
            self._hover_annot.set_visible(False)
            self._hover_vline.set_visible(False)
            self._canvas.draw_idle()
            return

        xdata = event.xdata  # type: ignore[union-attr]
        if xdata is None:
            return

        sg = self._spot_grid
        sp = self._struct_pnl
        ss = self._struct_shocked

        # Clamp to grid range
        if xdata < sg[0] or xdata > sg[-1]:
            self._hover_annot.set_visible(False)
            self._hover_vline.set_visible(False)
            self._canvas.draw_idle()
            return

        # Interpolate both curves at cursor x
        pnl_flat = float(np.interp(xdata, sg, sp))
        pnl_shocked = float(np.interp(xdata, sg, ss))

        vol_label = f"vol{self._vol_shift:+.0%}"

        # Build annotation text
        sign_f = "+" if pnl_flat >= 0 else ""
        sign_s = "+" if pnl_shocked >= 0 else ""
        text = (
            f"Spot: ${xdata:,.0f}\n"
            f"Flat:  {sign_f}${pnl_flat:,.0f}\n"
            f"{vol_label}: {sign_s}${pnl_shocked:,.0f}"
        )

        # Position annotation — flip to left side near right edge
        ax = self._ax
        xlim = ax.get_xlim()
        x_frac = (xdata - xlim[0]) / (xlim[1] - xlim[0]) if xlim[1] != xlim[0] else 0.5
        xytext = (-80, 12) if x_frac > 0.75 else (12, 12)

        self._hover_annot.set_text(text)
        self._hover_annot.xy = (xdata, pnl_flat)
        # xyann is the offset-points position tuple (supported since mpl 1.4)
        try:
            self._hover_annot.xyann = xytext  # type: ignore[attr-defined]
        except AttributeError:
            pass
        self._hover_annot.set_visible(True)

        self._hover_vline.set_xdata([xdata, xdata])
        self._hover_vline.set_visible(True)

        self._canvas.draw_idle()

    def _on_axes_leave(self, event: object) -> None:
        """Hide hover annotation when cursor leaves the axes."""
        self._hover_annot.set_visible(False)
        self._hover_vline.set_visible(False)
        self._canvas.draw_idle()


# =============================================================================
# SECTION 6 — GUI: RFQ confirmation dialog
# =============================================================================


class RFQConfirmDialog(QDialog):
    """
    Informational Y/N confirmation before submitting RFQ.
    Displays: leg summary, evaluated price, structure drawdown figure.
    """

    def __init__(
        self,
        leg_specs: List[str],
        evaluation: StructureEvaluation,
        config: RFQConfig,
        spot_value: float,
        spot_label: str,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Confirm RFQ Submission")
        self.setModal(True)
        self.setMinimumWidth(380)

        layout = QVBoxLayout(self)

        # Summary
        layout.addWidget(QLabel(f"<b>Legs ({len(leg_specs)}):</b>"))
        for spec in leg_specs:
            layout.addWidget(QLabel(f"  {spec}"))

        layout.addWidget(QLabel(""))

        eval_price = evaluation.total_usd
        if eval_price > 0:
            price_sign = "debit"
        elif eval_price < 0:
            price_sign = "credit"
        else:
            price_sign = "flat"
        price_lbl = QLabel(
            f"<b>Evaluated price:</b> ${abs(eval_price):,.2f} {price_sign}"
        )
        layout.addWidget(price_lbl)

        layout.addWidget(QLabel(f"<b>{spot_label}:</b> ${spot_value:,.0f}"))
        layout.addWidget(
            QLabel(
                f"<b>Threshold:</b> {config.threshold_type.value} ${config.threshold_value:,.2f}"
            )
        )
        layout.addWidget(QLabel(f"<b>{evaluation.threshold_reason}</b>"))
        layout.addWidget(
            QLabel(
                f"<b>Max DD flat:</b> ${evaluation.max_dd_flat:,.2f} "
                f"(limit ${config.max_dd_usd_flat:,.2f})"
            )
        )
        layout.addWidget(
            QLabel(
                f"<b>Max DD shocked:</b> ${evaluation.max_dd_shocked:,.2f} "
                f"(limit ${config.max_dd_usd_shocked:,.2f})"
            )
        )

        layout.addWidget(QLabel(""))
        note = QLabel(
            "<i>Submission uses the same threshold and drawdown gates as rfq_orchestrator.py.</i>"
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        buttons = QDialogButtonBox(QDialogButtonBox.Cancel | QDialogButtonBox.Ok)
        buttons.button(QDialogButtonBox.Ok).setText("Submit ▶")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


# =============================================================================
# SECTION 7 — GUI: Main window
# =============================================================================


class MainWindow(QMainWindow):
    # Signals emitted by async tasks → GUI thread
    sig_status = Signal(str)
    sig_plot_update = Signal(
        object,
        object,
        object,  # arrays
        float,
        float,  # current_spot, target_spot
    )
    sig_enable_submit = Signal(bool)
    sig_instruments_loaded = Signal()
    # DVOL / vol info bar: (dvol, dvol_high, dvol_low, index_price)
    sig_vol_update = Signal(float, float, float, float)
    # Live spot from WS (updates top bar at ~500ms cadence when Evaluate is running)
    sig_live_spot = Signal(float)
    # Pricing panel update: (cur_result, tgt_flat_result, tgt_shock_result, dd_flat_usd, dd_shock_usd, vol_shift)
    sig_pricing_update = Signal(object, object, object, float, float, float)
    sig_rfq_flow = Signal(str)

    def __init__(self, loader: CoincallInstrumentLoader) -> None:
        super().__init__()
        self._loader = loader
        self.setWindowTitle("Multileg RFQ Orchestrator")
        self.resize(1400, 800)
        self._build_ui()
        self._connect_signals()

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Vol info bar — full-width strip above the splitter
        self._vol_bar = VolInfoBar()
        self._vol_bar.set_loading()
        main_layout.addWidget(self._vol_bar, stretch=0)

        # Splitter: left 1/3, right 2/3
        splitter = QSplitter(Qt.Horizontal)

        self._leg_panel = LegPanel(self._loader)
        splitter.addWidget(self._leg_panel)

        # Right: plot + parameter bar
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(4, 4, 4, 4)
        right_layout.setSpacing(4)

        self._plot_widget = PlotWidget()
        right_layout.addWidget(self._plot_widget, stretch=1)

        self._param_bar = ParameterBar()
        right_layout.addWidget(self._param_bar, stretch=0)

        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)

        main_layout.addWidget(splitter, stretch=1)

        # Status bar
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self._status_bar.showMessage("Ready — load instruments at startup")

    def _connect_signals(self) -> None:
        self.sig_status.connect(self._status_bar.showMessage)
        self.sig_enable_submit.connect(self._leg_panel.enable_submit)
        self.sig_instruments_loaded.connect(self._leg_panel.refresh_leg_dropdowns)
        self.sig_vol_update.connect(self._vol_bar.update_data)
        self.sig_live_spot.connect(self._vol_bar.update_live_spot)
        self.sig_pricing_update.connect(self._on_pricing_update)
        self.sig_rfq_flow.connect(self._leg_panel.append_rfq_flow)

        self._param_bar.params_changed.connect(self._on_params_changed)

    def apply_demo_leg_specs(self, leg_specs: List[str]) -> None:
        """
        Populate the legs panel with exactly the provided specs.
        Intended for deterministic screenshot mode.
        """
        rows = self._leg_panel.get_leg_rows()
        while len(rows) < len(leg_specs):
            self._leg_panel._add_leg()
            rows = self._leg_panel.get_leg_rows()
        while len(rows) > len(leg_specs):
            self._leg_panel._remove_leg()
            rows = self._leg_panel.get_leg_rows()

        names: List[str] = []
        for row, spec in zip(rows, leg_specs):
            row.set_from_spec(spec)
            m = re.match(
                r"^[LS]\s+[\d.]+\s+((\d{1,2}[A-Z]{3}\d{2})-(\d+)-([CP]))$",
                spec.strip(),
                re.IGNORECASE,
            )
            names.append(f"BTC-{m.group(1).upper()}" if m else spec)

        self._leg_panel._pricing.set_leg_names(names)

    @Slot(float, float)
    def _on_params_changed(
        self,
        time_days: float,
        vol_shift: float,
    ) -> None:
        """Relay slider changes to orchestrator."""
        orch = getattr(self, "_orchestrator", None)
        if orch:
            orch.on_params_changed(time_days, vol_shift)

    def set_orchestrator(self, orch: "GUIOrchestrator") -> None:
        self._orchestrator = orch
        self._leg_panel.evaluate_requested.connect(orch.on_evaluate_requested)
        self._leg_panel.submit_requested.connect(orch.on_submit_requested)
        self._leg_panel.skew_needed.connect(orch.on_skew_needed)

    # Called from orchestrator to push plot data into the GUI thread
    @Slot(object, object, object, float, float)
    def on_plot_update(
        self,
        spot_grid: object,
        struct_pnl: object,
        struct_shocked: object,
        current_spot: float,
        target_spot: float,
    ) -> None:
        self._plot_widget.update_plot(
            spot_grid=spot_grid,
            struct_pnl=struct_pnl,
            struct_shocked=struct_shocked,
            current_spot=current_spot,
            target_spot=target_spot,
            time_days=self._param_bar.time_days(),
            vol_shift=self._param_bar.vol_shift(),
        )

    @Slot(object, object, object, float, float, float)
    def _on_pricing_update(
        self,
        cur_result: object,
        tgt_flat_result: object,
        tgt_shock_result: object,
        dd_flat_usd: float,
        dd_shock_usd: float,
        vol_shift: float,
    ) -> None:
        self._leg_panel.update_pricing(
            cur_result,
            tgt_flat_result,
            tgt_shock_result,
            dd_flat_usd,
            dd_shock_usd,
            vol_shift,
        )


# =============================================================================
# SECTION 8 — Orchestrator: bridges async data layer ↔ Qt GUI
# =============================================================================


class GUIOrchestrator(QObject):
    """
    Owns all asyncio tasks. Lives in the main (Qt+asyncio) thread.
    Communicates to MainWindow exclusively via Qt signals.

    Lifecycle:
        startup()               — load instruments, start Deribit/Coincall WS
        on_evaluate_requested() — parse legs, start live re-pricing loop
        on_submit_requested()   — Y/N modal → RFQ flow
        shutdown()              — cancel all tasks, clean up
    """

    def __init__(self, window: MainWindow) -> None:
        super().__init__()
        self._win = window
        window.set_orchestrator(self)

        # Shared state
        self._loader = window._loader

        # Deribit WS (pricing)
        self._deribit_handler: Optional[CachedDeribitApiHandler] = None
        self._mds: Optional[DeribitMarketDataService] = None
        self._mds_task: Optional[asyncio.Task] = None

        # Live loop
        self._live_task: Optional[asyncio.Task] = None
        self._legs: List[str] = []
        self._target_spot: float = 80000.0
        self._latest_live_spot: Optional[float] = None
        self._evaluated_at_target: bool = (
            False  # True iff user set a target spot before evaluating
        )
        self._time_days: float = 7.0
        self._vol_shift: float = -0.10

        # RFQ config (editable via param bar in a future enhancement)
        self._rfq_config: Optional[RFQConfig] = None

        # Vol info bar background task
        self._vol_info_task: Optional[asyncio.Task] = None

        # Skew fetcher — separate long-lived Deribit REST handler
        self._skew_handler: Optional[SkewDeribitHandler] = None
        self._skew_fetcher: Optional[SkewFetcher] = None
        # Cache: maturity_label -> (skew_value, timestamp)
        self._skew_cache: Dict[str, Tuple[float, float]] = {}
        self._skew_refresh_task: Optional[asyncio.Task] = None
        # Guards against duplicate in-flight fetches for the same maturity
        self._skew_inflight: Set[str] = set()

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------

    async def startup(self) -> None:
        LOG.info("Loading Coincall instruments…")
        self._win.sig_status.emit("Loading instruments from Coincall…")
        await self._loader.load()
        self._win.sig_instruments_loaded.emit()
        self._win.sig_status.emit(
            f"Instruments loaded — {len(self._loader.expiry_labels)} expiries available"
        )

        # Start DVOL info bar refresh loop
        self._vol_info_task = asyncio.create_task(
            self._vol_info_loop(), name="vol-info"
        )

        # Open the skew Deribit handler (instruments cached for 5 min)
        pool_cfg = ConnectionPoolConfig(
            limit=50, limit_per_host=20, keepalive_timeout=60
        )
        self._skew_handler = SkewDeribitHandler(
            testnet=False,
            cache_ttl_sec=300,
            pool_config=pool_cfg,
            rate_limit=20,
        )
        await self._skew_handler.__aenter__()
        self._skew_fetcher = SkewFetcher(self._skew_handler)

        # Start periodic skew refresh loop (runs every 60 s)
        self._skew_refresh_task = asyncio.create_task(
            self._skew_refresh_loop(), name="skew-refresh"
        )

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------

    @Slot(list, float, str)
    def on_evaluate_requested(
        self,
        leg_specs: List[str],
        target_spot: Optional[float],
        account: str,
    ) -> None:
        """Called from Qt signal — schedule async task."""
        asyncio.ensure_future(self._evaluate(leg_specs, target_spot, account))

    async def _evaluate(
        self,
        leg_specs: List[str],
        target_spot: Optional[float],
        account: str,
    ) -> None:
        self._win.sig_status.emit("Evaluating — setting up market data…")
        self._win.sig_enable_submit.emit(False)
        self._win._leg_panel.clear_rfq_flow()

        self._legs = leg_specs
        self._evaluated_at_target = target_spot is not None
        if target_spot is not None:
            self._target_spot = target_spot

        # Stop any running live loop
        await self._stop_live_tasks()

        # Parse legs
        try:
            parsed_legs = parse_legs(leg_specs)
        except ValueError as exc:
            self._win.sig_status.emit(
                "Leg validation error. Fix leg format/values "
                f"(size > 0, strike > 0, valid DDMMMYY maturity): {exc}"
            )
            return

        # Start Deribit handler + MDS
        try:
            self._win.sig_status.emit("Enumerating smile instruments (Deribit REST)…")
            self._deribit_handler = CachedDeribitApiHandler(cache_ttl_sec=3600)
            await self._deribit_handler.__aenter__()

            instruments, strike_windows = await enumerate_smile_instruments(
                legs=parsed_legs,
                handler=self._deribit_handler,
            )

            self._mds = DeribitMarketDataService(
                instruments=instruments,
                maturity_strike_windows=strike_windows,
                max_age_ms=5000.0,
            )
            self._mds_task = asyncio.create_task(self._mds.run(), name="deribit-mds")

            self._win.sig_status.emit("Waiting for Deribit WS to warm up…")
            ready = await self._mds.wait_ready(timeout=20.0)
            if not ready:
                self._win.sig_status.emit("ERROR: Deribit WS failed to become ready")
                return

        except Exception as exc:
            self._win.sig_status.emit(f"Deribit setup error: {exc}")
            LOG.exception("Deribit setup failed")
            return

        # Seed config for the live loop with a valid positive spot. In
        # current-spot mode this value is informational only until submit-time,
        # when _submit_rfq() rebuilds the config from the latest live spot.
        initial_config_spot = (
            target_spot if target_spot is not None else self._target_spot
        )
        self._rfq_config = self._build_rfq_config(initial_config_spot)

        # Start live update loop
        self._live_task = asyncio.create_task(
            self._live_update_loop(parsed_legs, target_spot),
            name="live-update",
        )

        self._win.sig_status.emit("Live pricing active")
        self._win.sig_enable_submit.emit(True)

    # ------------------------------------------------------------------
    # Live pricing loop
    # ------------------------------------------------------------------

    async def _live_update_loop(
        self,
        parsed_legs: List[OptionLeg],
        target_spot: Optional[float],
    ) -> None:
        """
        Continuously re-prices the structure from the Deribit WS cache.
        Sleeps 500ms between iterations to match the plot refresh rate.
        """
        while True:
            try:
                await self._reprice(parsed_legs, target_spot)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                LOG.warning("Live loop error: %s", exc)
            await asyncio.sleep(0.5)

    async def _reprice(
        self,
        parsed_legs: List[OptionLeg],
        target_spot: Optional[float],
    ) -> None:
        if not self._mds or not self._mds.is_ready or not self._deribit_handler:
            return

        if self._rfq_config is None:
            return

        # Build IV data from WS cache
        maturities = list({lg.maturity for lg in parsed_legs})
        try:
            current_spot, iv_data = await self._mds.build_maturity_iv_data(
                maturities, self._deribit_handler
            )
        except RuntimeError as exc:
            LOG.debug("WS cache not ready: %s", exc)
            return

        self._latest_live_spot = current_spot

        iv_interps = {m: StrikeIVInterpolator(d) for m, d in iv_data.items()}
        pricer = StructurePricer(iv_interps, iv_data, current_spot, r=0.0)

        # Determine the spot to use as the scenario/drawdown centre.
        # When the user set a target spot before evaluating, centre on that.
        # When the user evaluated at current (live) spot, centre on live spot.
        dd_centre = target_spot if self._evaluated_at_target else current_spot

        # Drawdown calc with current slider values
        days = max(1, int(round(self._time_days))) if self._time_days >= 1 else 1
        dd_calc = VectorizedDrawdownCalculator(
            iv_interpolators=iv_interps,
            maturity_data=iv_data,
            current_spot=dd_centre,
            legs=parsed_legs,
            original_spot=current_spot,
            r=0.0,
            spot_range_pct=0.50,
            spot_grid_points=300,
            vol_shock=self._vol_shift,
        )
        dd_flat, dd_shock = dd_calc.compute(days_forward=days)

        spot_grid = dd_flat.spot_range
        struct_pnl = dd_flat.pnl_usd
        struct_shocked = dd_shock.pnl_usd

        # Pricing at current spot (flat vol) — for Current Spot tab
        cur_result = pricer.price_structure(parsed_legs, current_spot, vol_shift=0.0)

        # Pricing at the evaluation spot — flat vol and vol-shocked — for Target Spot tab.
        # When not evaluating at a target spot, both use current (live) spot.
        eval_spot = dd_centre
        tgt_flat_result = pricer.price_structure(parsed_legs, eval_spot, vol_shift=0.0)
        tgt_shock_result = pricer.price_structure(
            parsed_legs, eval_spot, vol_shift=self._vol_shift
        )

        # Update live spot in top bar (WS cadence ~500ms)
        self._win.sig_live_spot.emit(current_spot)

        # Push plot data.
        # Pass 0.0 for target_spot when not in target-spot mode so the yellow
        # axvline is suppressed by PlotWidget._draw()'s existing '> 0' guard.
        plot_target_spot = target_spot if self._evaluated_at_target else 0.0
        self._win.on_plot_update(
            spot_grid,
            struct_pnl,
            struct_shocked,
            current_spot,
            plot_target_spot,
        )

        # Push pricing panel data
        self._win.sig_pricing_update.emit(
            cur_result,
            tgt_flat_result,
            tgt_shock_result,
            dd_flat.max_drawdown_usd,
            dd_shock.max_drawdown_usd,
            self._vol_shift,
        )

        self._win.sig_status.emit(
            f"Live | struct={tgt_flat_result.total_usd:+,.0f} USD | "
            f"max_dd={dd_flat.max_drawdown_usd:,.0f}"
        )

    # ------------------------------------------------------------------
    # Parameter changes from sliders
    # ------------------------------------------------------------------

    def on_params_changed(
        self,
        time_days: float,
        vol_shift: float,
    ) -> None:
        """Called from Qt signal — update state only. Session WS is always running."""
        self._time_days = time_days
        self._vol_shift = vol_shift
        if self._rfq_config:
            self._rfq_config.vol_shock = vol_shift

    # ------------------------------------------------------------------
    # RFQ submission
    # ------------------------------------------------------------------

    @Slot()
    def on_submit_requested(self) -> None:
        asyncio.ensure_future(self._submit_rfq())

    async def _submit_rfq(self) -> None:
        if not self._mds or not self._deribit_handler or not self._legs:
            self._win.sig_status.emit("No evaluation available — press Evaluate first")
            return

        submit_spot = (
            self._target_spot if self._evaluated_at_target else self._latest_live_spot
        )
        if submit_spot is None or submit_spot <= 0:
            msg = "Submit blocked: live spot unavailable; wait for pricing to refresh"
            self._win.sig_rfq_flow.emit(msg)
            self._win.sig_status.emit(msg)
            return

        self._rfq_config = self._build_rfq_config(submit_spot)
        errors = self._rfq_config.validate()
        if errors:
            msg = "; ".join(errors)
            self._win.sig_rfq_flow.emit(f"Config error: {msg}")
            self._win.sig_status.emit(f"Config error: {msg}")
            return

        self._win.sig_status.emit("Running authoritative RFQ evaluation…")
        try:
            ev = await evaluate_structure(
                self._legs,
                self._rfq_config,
                self._mds,
                self._deribit_handler,
            )
        except ValueError as exc:
            msg = (
                "Evaluation blocked by leg validation error. "
                f"Please correct the legs and re-evaluate: {exc}"
            )
            self._win.sig_rfq_flow.emit(msg)
            self._win.sig_status.emit(msg)
            return
        except Exception as exc:
            self._win.sig_rfq_flow.emit(f"Evaluation failed: {exc}")
            self._win.sig_status.emit(f"Evaluation failed: {exc}")
            LOG.exception("Submit-time evaluation failed")
            return

        self._log_evaluation_result(ev, self._rfq_config)
        if not ev.all_passed:
            self._win.sig_status.emit("Checks failed — RFQ blocked")
            return

        # Show confirmation dialog (in Qt main thread).
        # Display the spot at which the structure was actually evaluated so the
        # user can verify they are submitting under the right assumptions.
        if self._evaluated_at_target:
            spot_label = "Target spot"
            spot_value = ev.target_spot
        else:
            spot_label = "Current spot"
            spot_value = ev.current_spot
        dlg = RFQConfirmDialog(
            leg_specs=self._legs,
            evaluation=ev,
            config=self._rfq_config,
            spot_value=spot_value,
            spot_label=spot_label,
            parent=self._win,
        )
        if dlg.exec() != QDialog.Accepted:
            self._win.sig_status.emit("RFQ submission cancelled by user")
            return

        await self._run_rfq_flow(ev)

    async def _run_rfq_flow(self, ev: StructureEvaluation) -> None:
        if not self._rfq_config or not self._mds:
            return

        # Resolve credentials at submit time from the current combo selection.
        # This is the authoritative moment — whatever account is shown in the
        # UI right now is exactly the account that will execute the trade.
        account = self._win._leg_panel.get_account()
        key_var, secret_var = _ACCOUNT_CREDENTIALS[account]
        api_key = os.environ.get(key_var)
        api_secret = os.environ.get(secret_var)
        if not api_key or not api_secret:
            self._win.sig_status.emit(
                f"ERROR: missing credentials — set {key_var} and {secret_var} in .env"
            )
            return

        LOG.info("Submitting RFQ on %s account (%s)", account.upper(), key_var)

        # Open a fresh Coincall WS connection for RFQ events (rfqTaker +
        # quoteReceived).  This is the same ephemeral pattern used by
        # rfq_orchestrator.py — create, use, tear down.
        self._win.sig_status.emit(
            f"Connecting Coincall RFQ WebSocket [{account.upper()} account]…"
        )
        cc_ws = CoincallWSClient(COINCALL_WS_URL, api_key, api_secret)
        cc_ws_task = asyncio.create_task(cc_ws.run(), name="cc-rfq-ws")

        try:
            ready = await cc_ws.wait_ready(timeout=15.0)
            if not ready:
                self._win.sig_status.emit("ERROR: Coincall RFQ WS failed to connect")
                return

            self._win.sig_status.emit("Submitting RFQ…")
            async with CoincallREST(self._rfq_config, api_key, api_secret) as rest:
                success = await execute_rfq_flow(
                    evaluation=ev,
                    config=self._rfq_config,
                    ws_client=cc_ws,
                    rest=rest,
                    mds=self._mds,
                )
                if success:
                    self._win.sig_rfq_flow.emit("Executed: trade completed")
                    self._win.sig_status.emit("RFQ FILLED — trade executed")
                else:
                    self._win.sig_rfq_flow.emit("RFQ ended: no trade executed")
                    self._win.sig_status.emit("RFQ ended — no trade executed")
        except Exception as exc:
            self._win.sig_status.emit(f"RFQ error: {exc}")
            LOG.exception("RFQ flow error")
        finally:
            cc_ws.stop()
            cc_ws_task.cancel()
            try:
                await cc_ws_task
            except asyncio.CancelledError:
                pass

    def _build_rfq_config(self, target_spot: float) -> RFQConfig:
        panel = self._win._leg_panel
        return RFQConfig(
            target_spot=target_spot,
            threshold_type=panel.get_threshold_type(),
            threshold_value=panel.get_threshold_value(),
            max_dd_usd_flat=panel.get_max_dd_flat(),
            max_dd_usd_shocked=panel.get_max_dd_shocked(),
            drawdown_days=max(1, int(round(self._time_days))),
            vol_shock=self._vol_shift,
            spot_range_pct=0.50,
            spot_grid_points=300,
            rfq_timeout_seconds=30.0,
            max_slippage_percent=7.5,
            price_deviation_threshold=0.15,
            max_leg_price_usd=7000.0,
            max_snapshot_age_ms=5000.0,
        )

    def _log_evaluation_result(self, ev: StructureEvaluation, cfg: RFQConfig) -> None:
        if ev.total_usd > 0:
            total_text = f"Debit {ev.total_usd:,.2f}"
        elif ev.total_usd < 0:
            total_text = f"Credit {abs(ev.total_usd):,.2f}"
        else:
            total_text = "Flat 0.00"

        flat_text = "pass" if ev.dd_flat_passed else "fail"
        shock_text = "pass" if ev.dd_shocked_passed else "fail"
        self._win.sig_rfq_flow.emit(
            f"Evaluation: {total_text} | {ev.threshold_reason} | DD flat {flat_text} | DD shock {shock_text}"
        )
        if not ev.threshold_passed:
            self._win.sig_rfq_flow.emit(f"Blocked: {ev.threshold_reason}")
        if not ev.dd_flat_passed:
            self._win.sig_rfq_flow.emit(
                f"Blocked: DD flat {ev.max_dd_flat:,.2f} < limit {cfg.max_dd_usd_flat:,.2f}"
            )
        if not ev.dd_shocked_passed:
            self._win.sig_rfq_flow.emit(
                f"Blocked: DD shocked {ev.max_dd_shocked:,.2f} < limit {cfg.max_dd_usd_shocked:,.2f}"
            )

    # ------------------------------------------------------------------
    # Vol info bar background loop (60 s refresh)
    # ------------------------------------------------------------------

    async def _vol_info_loop(self) -> None:
        """Fetch DVOL + BTC price from Deribit every 60 seconds."""
        while True:
            try:
                data = await fetch_dvol_and_price("BTC")
                self._win.sig_vol_update.emit(
                    data["dvol"],
                    data["dvol_high_24h"],
                    data["dvol_low_24h"],
                    data["index_price"],
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                LOG.warning("DVOL fetch failed: %s", exc)
                # Show error state on the bar but keep running
                try:
                    self._win._vol_bar.set_error("fetch error")
                except Exception:
                    pass
            await asyncio.sleep(60)

    # ------------------------------------------------------------------
    # Per-leg 25-delta skew fetching
    # ------------------------------------------------------------------

    @Slot(object)
    def on_skew_needed(self, leg_row: "LegRow") -> None:
        """
        Called from Qt signal chain (LegPanel.skew_needed) when the user
        changes expiry, direction, or option type on a leg row.
        Schedules an async skew fetch for that row.
        """
        asyncio.ensure_future(self._fetch_skew_for_row(leg_row))

    async def _fetch_skew_for_row(self, leg_row: "LegRow") -> None:
        """
        Fetch the 25-delta skew for the maturity of *leg_row* and update
        its skew label.  Uses an in-memory cache (TTL = 60 s).
        """
        if self._skew_fetcher is None:
            return

        maturity = leg_row.get_maturity()
        if not maturity or maturity == "Loading…":
            return

        now = time.monotonic()

        # Cache hit?
        cached = self._skew_cache.get(maturity)
        if cached is not None:
            skew_val, cached_at = cached
            if now - cached_at < 60.0:
                leg_row.set_skew(skew_val)
                return

        # Deduplicate in-flight requests for the same maturity
        if maturity in self._skew_inflight:
            leg_row.set_skew_loading()
            return

        self._skew_inflight.add(maturity)
        leg_row.set_skew_loading()

        try:
            skew_val = await self._skew_fetcher.fetch_skew_for_maturity(maturity)
            self._skew_cache[maturity] = (skew_val, time.monotonic())
            # Update *all* visible rows that currently show this maturity
            for row in self._win._leg_panel.get_leg_rows():
                if row.get_maturity() == maturity:
                    row.set_skew(skew_val)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            LOG.warning("Skew fetch failed for %s: %s", maturity, exc)
            leg_row.set_skew(None)
        finally:
            self._skew_inflight.discard(maturity)

    async def _skew_refresh_loop(self) -> None:
        """
        Every 60 seconds re-fetch skew for every maturity currently displayed
        across all leg rows and update their labels.
        """
        while True:
            await asyncio.sleep(60)
            if self._skew_fetcher is None:
                continue
            # Collect unique maturities currently shown
            try:
                rows = self._win._leg_panel.get_leg_rows()
            except Exception:
                continue
            seen: Set[str] = set()
            for row in rows:
                mat = row.get_maturity()
                if mat and mat != "Loading…" and mat not in seen:
                    seen.add(mat)
                    asyncio.ensure_future(self._fetch_skew_for_row(row))

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------

    async def _stop_live_tasks(self) -> None:
        """Cancel the live pricing loop and Deribit WS."""
        if self._live_task:
            self._live_task.cancel()
            try:
                await self._live_task
            except asyncio.CancelledError:
                pass
            self._live_task = None

        if self._mds:
            self._mds.stop()
        if self._mds_task:
            self._mds_task.cancel()
            try:
                await self._mds_task
            except asyncio.CancelledError:
                pass
            self._mds_task = None

        if self._deribit_handler:
            try:
                await self._deribit_handler.__aexit__(None, None, None)
            except Exception:
                pass
            self._deribit_handler = None

    async def shutdown(self) -> None:
        await self._stop_live_tasks()

        # Cancel vol info bar loop
        if self._vol_info_task:
            self._vol_info_task.cancel()
            try:
                await self._vol_info_task
            except asyncio.CancelledError:
                pass
            self._vol_info_task = None

        # Cancel skew refresh loop
        if self._skew_refresh_task:
            self._skew_refresh_task.cancel()
            try:
                await self._skew_refresh_task
            except asyncio.CancelledError:
                pass
            self._skew_refresh_task = None

        # Close skew Deribit handler
        if self._skew_handler:
            try:
                await self._skew_handler.__aexit__(None, None, None)
            except Exception:
                pass
            self._skew_handler = None
            self._skew_fetcher = None

        LOG.info("GUIOrchestrator shutdown complete")


# =============================================================================
# SECTION 9 — Dark Fusion palette
# =============================================================================


def _apply_dark_palette(app: QApplication) -> None:
    app.setStyle("Fusion")
    pal = QPalette()
    pal.setColor(QPalette.Window, QColor(30, 30, 30))
    pal.setColor(QPalette.WindowText, QColor(220, 220, 220))
    pal.setColor(QPalette.Base, QColor(40, 40, 40))
    pal.setColor(QPalette.AlternateBase, QColor(50, 50, 50))
    pal.setColor(QPalette.ToolTipBase, QColor(50, 50, 50))
    pal.setColor(QPalette.ToolTipText, QColor(220, 220, 220))
    pal.setColor(QPalette.Text, QColor(220, 220, 220))
    pal.setColor(QPalette.Button, QColor(55, 55, 55))
    pal.setColor(QPalette.ButtonText, QColor(220, 220, 220))
    pal.setColor(QPalette.BrightText, Qt.red)
    pal.setColor(QPalette.Highlight, QColor(42, 130, 218))
    pal.setColor(QPalette.HighlightedText, Qt.black)
    pal.setColor(QPalette.Disabled, QPalette.Text, QColor(128, 128, 128))
    pal.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(128, 128, 128))
    app.setPalette(pal)


# =============================================================================
# SECTION 10 — Entry point
# =============================================================================


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Multileg RFQ Orchestrator GUI")
    parser.add_argument(
        "--demo-screenshot",
        action="store_true",
        help=(
            "Run deterministic, offline-safe UI mode for CI screenshots "
            "(no network dependencies)."
        ),
    )
    parser.add_argument(
        "--screenshot-path",
        type=str,
        default="",
        help=(
            "Optional absolute or relative file path. If provided in "
            "--demo-screenshot mode, the GUI will save a direct window grab "
            "to this location."
        ),
    )
    parser.add_argument(
        "--auto-exit-after-ready",
        action="store_true",
        help=(
            "If set in --demo-screenshot mode, the app exits automatically "
            "after producing demo state and any requested screenshot."
        ),
    )
    return parser


def parse_cli_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    return build_arg_parser().parse_args(argv)


def _seed_loader_for_demo(loader: CoincallInstrumentLoader) -> None:
    """Inject deterministic expiry/strike data for demo screenshot mode."""
    loader.expiry_labels = ["29MAY26", "26JUN26"]
    loader.strikes_by_expiry = {
        "29MAY26": ["80000"],
        "26JUN26": ["82000"],
    }


def _build_mock_price_result(
    leg_specs: List[str],
    spot: float,
    total_usd: float,
    vol_shift: float,
) -> StructurePriceResult:
    legs = parse_legs(leg_specs)
    leg_prices_usd = np.array([2150.0, -1725.0], dtype=float)
    leg_prices_btc = leg_prices_usd / spot
    return StructurePriceResult(
        legs=legs,
        spot=spot,
        shifted_spot=spot,
        leg_prices_btc=leg_prices_btc,
        leg_prices_usd=leg_prices_usd,
        leg_ivs=np.array([0.58, 0.54], dtype=float),
        leg_forwards=np.array([81200.0, 81950.0], dtype=float),
        leg_times_to_expiry_years=np.array([0.11, 0.18], dtype=float),
        total_btc=float(total_usd / spot),
        total_usd=total_usd,
        vol_shift=vol_shift,
    )


def apply_demo_screenshot_state(
    window: MainWindow,
    screenshot_path: str = "",
    auto_exit_after_ready: bool = False,
) -> None:
    """
    Populate GUI with deterministic content and keep it static for screenshot
    capture in CI. No network/API calls are required.
    """
    window.apply_demo_leg_specs(DEMO_SCREENSHOT_DEFAULT_LEGS)
    window._vol_bar.update_data(
        dvol=61.2,
        dvol_high=64.0,
        dvol_low=58.4,
        index_price=81234.0,
    )
    window._param_bar._time_spin.setValue(7.0)
    window._param_bar._vol_spin.setValue(-0.10)
    window._param_bar._time_slider.setValue(70)
    window._param_bar._vol_slider.setValue(-100)
    window._status_bar.showMessage("Demo screenshot mode — deterministic mock data")

    cur = _build_mock_price_result(
        leg_specs=DEMO_SCREENSHOT_DEFAULT_LEGS,
        spot=81234.0,
        total_usd=425.0,
        vol_shift=0.0,
    )
    tgt_flat = _build_mock_price_result(
        leg_specs=DEMO_SCREENSHOT_DEFAULT_LEGS,
        spot=81500.0,
        total_usd=540.0,
        vol_shift=0.0,
    )
    tgt_shock = _build_mock_price_result(
        leg_specs=DEMO_SCREENSHOT_DEFAULT_LEGS,
        spot=81500.0,
        total_usd=465.0,
        vol_shift=-0.10,
    )
    window._leg_panel.update_pricing(
        cur_result=cur,
        tgt_flat_result=tgt_flat,
        tgt_shock_result=tgt_shock,
        dd_flat_usd=-1200.0,
        dd_shock_usd=-1425.0,
        vol_shift=-0.10,
    )

    spot_grid = np.linspace(70000.0, 92000.0, 300)
    base = (spot_grid - 81500.0) / 1000.0
    struct_pnl = 520.0 - 18.0 * np.square(base) + 42.0 * np.sin(base / 1.8)
    struct_shocked = struct_pnl - 95.0
    window.on_plot_update(
        spot_grid=spot_grid,
        struct_pnl=struct_pnl,
        struct_shocked=struct_shocked,
        current_spot=81234.0,
        target_spot=81500.0,
    )

    def _ready_log() -> None:
        print("DEMO_SCREENSHOT_READY", flush=True)

    QTimer.singleShot(1200, _ready_log)

    def _capture_window_and_maybe_exit() -> None:
        if screenshot_path:
            out_path = os.path.abspath(screenshot_path)
            out_dir = os.path.dirname(out_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            ok = window.grab().save(out_path, "PNG")
            if ok:
                print(f"DEMO_SCREENSHOT_SAVED {out_path}", flush=True)
            else:
                print(f"DEMO_SCREENSHOT_SAVE_FAILED {out_path}", flush=True)
        if auto_exit_after_ready:
            QApplication.instance().quit()

    QTimer.singleShot(1700, _capture_window_and_maybe_exit)


def main() -> None:
    # Windows requires SelectorEventLoop; ProactorEventLoop (the default on
    # Windows ≥ 3.8) is incompatible with aiohttp WebSockets and qasync.
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    args = parse_cli_args()
    load_dotenv()

    if args.demo_screenshot:
        # CI Windows runners frequently lack GPU acceleration in the current
        # desktop/session context. Force software OpenGL for deterministic
        # Matplotlib/Qt rendering during screenshot mode.
        QApplication.setAttribute(Qt.AA_UseSoftwareOpenGL, True)

    app = QApplication(sys.argv)
    _apply_dark_palette(app)

    loop = qasync.QEventLoop(app)
    asyncio.set_event_loop(loop)

    loader = CoincallInstrumentLoader()
    if args.demo_screenshot:
        _seed_loader_for_demo(loader)
    window = MainWindow(loader)
    orch: Optional[GUIOrchestrator] = None
    if not args.demo_screenshot:
        orch = GUIOrchestrator(window)

    window.show()

    async def _startup_and_run() -> None:
        if args.demo_screenshot:
            apply_demo_screenshot_state(
                window,
                screenshot_path=args.screenshot_path,
                auto_exit_after_ready=args.auto_exit_after_ready,
            )
        elif orch is not None:
            await orch.startup()
        # Keep running until the window is closed
        close_event = asyncio.Event()

        def _on_close() -> None:
            close_event.set()

        app.aboutToQuit.connect(_on_close)
        await close_event.wait()
        if orch is not None:
            await orch.shutdown()

    with loop:
        loop.run_until_complete(_startup_and_run())


if __name__ == "__main__":
    main()
