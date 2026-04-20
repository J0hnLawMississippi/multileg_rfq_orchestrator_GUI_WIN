import pytest

PySide6 = pytest.importorskip("PySide6")

from rfq_orchestrator_win import StructureEvaluation
from db_option_pricer_win import Direction, OptionLeg, OptionType
from multileg_rfq_orchestrator_GUI_WIN import GUIOrchestrator


class _Signal:
    def __init__(self):
        self.values = []

    def emit(self, val):
        self.values.append(val)


class _LegPanel:
    def get_threshold_type(self):
        from rfq_orchestrator_win import ThresholdType
        return ThresholdType.CREDIT

    def get_threshold_value(self):
        return 100.0

    def get_max_dd_flat(self):
        return -500.0

    def get_max_dd_shocked(self):
        return -700.0


class _Window:
    def __init__(self):
        self._loader = object()
        self._leg_panel = _LegPanel()
        self.sig_rfq_flow = _Signal()

    def set_orchestrator(self, orch):
        self._orch = orch


def test_gui_orchestrator_smoke_state_transition_message():
    win = _Window()
    orch = GUIOrchestrator(win)
    cfg = orch._build_rfq_config(70000.0)
    assert cfg.target_spot == 70000.0

    leg = OptionLeg(Direction.LONG, 1.0, "20MAR26", 70000.0, OptionType.CALL, "BTC-20MAR26-70000-C")
    ev = StructureEvaluation([leg],70000,70000,0,-120,[0],[0],[60],[70000],-100,65000,-150,64000,False,"Credit low",False,True,False,{})
    orch._log_evaluation_result(ev, cfg)
    assert any("Blocked:" in m for m in win.sig_rfq_flow.values)
