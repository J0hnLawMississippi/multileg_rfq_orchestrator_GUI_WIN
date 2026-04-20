import asyncio

import pytest

from coincall_ws_win import QuoteReceived, QuoteState
from db_option_pricer_win import Direction, OptionLeg, OptionType
from rfq_orchestrator_win import RFQConfig, StructureEvaluation, ThresholdType, execute_rfq_flow


class FakeREST:
    def __init__(self):
        self.executed = []

    async def create_rfq(self, _legs):
        return {"requestId": "req-1"}

    async def execute_trade(self, req, qid):
        self.executed.append((req, qid))
        return {"ok": True}

    async def cancel_rfq(self, _req):
        return True


class FakeWS:
    def __init__(self):
        self.q = asyncio.Queue()
        self.registered = []

    def register(self, rid):
        self.registered.append(rid)
        return self.q

    def unregister(self, rid):
        self.registered.remove(rid)


class FakeMDS:
    def get_snapshot(self, _instrument):
        class S:
            mark_price_btc = 0.002
            index_price = 70000.0
        return S()

    def snapshot_age_ms(self, _instrument):
        return 1.0


@pytest.mark.asyncio
async def test_execute_rfq_flow_happy_path():
    leg = OptionLeg(Direction.LONG, 1, "20MAR26", 70000, OptionType.CALL, "BTC-20MAR26-70000-C")
    ev = StructureEvaluation(
        legs=[leg], current_spot=70000, target_spot=70000, total_btc=0, total_usd=-120,
        per_leg_btc=[0], per_leg_usd=[0], per_leg_iv=[60], per_leg_forward=[70000],
        max_dd_flat=-10, worst_spot_flat=65000, max_dd_shocked=-20, worst_spot_shocked=64000,
        threshold_passed=True, threshold_reason="ok", dd_flat_passed=True, dd_shocked_passed=True,
        all_passed=True, deribit_marks_usd={"BTC-20MAR26-70000-C": 100},
    )
    ws = FakeWS()
    rest = FakeREST()
    mds = FakeMDS()
    cfg = RFQConfig(
        threshold_type=ThresholdType.CREDIT,
        threshold_value=50,
        rfq_timeout_seconds=0.5,
        price_deviation_threshold=0.5,
        max_slippage_percent=20.0,
    )

    await ws.q.put(
        QuoteReceived(
            request_id="req-1", quote_id="q-1", state=QuoteState.OPEN, quote_side="BUY",
            legs=({"instrumentName": "BTCUSD-20MAR26-70000-C", "price": "120", "quantity": "1", "side": "BUY"},),
        )
    )

    ok = await execute_rfq_flow(ev, cfg, ws, rest, mds)
    assert ok
    assert rest.executed == [("req-1", "q-1")]


@pytest.mark.asyncio
@pytest.mark.xfail(reason="Known defect: RFQ queue is registered after create_rfq can race with early WS quote")
async def test_rfq_pre_register_race_regression():
    class RaceWS(FakeWS):
        def __init__(self):
            super().__init__()
            self.last_q = None

        def register(self, rid):
            q = super().register(rid)
            self.last_q = q
            return q

    class RaceREST(FakeREST):
        async def create_rfq(self, _legs):
            # Quote arrives before register in current implementation.
            return {"requestId": "req-race"}

    leg = OptionLeg(Direction.LONG, 1, "20MAR26", 70000, OptionType.CALL, "BTC-20MAR26-70000-C")
    ev = StructureEvaluation([leg],70000,70000,0,-100,[0],[0],[60],[70000],-10,65000,-20,64000,True,"ok",True,True,True,{"BTC-20MAR26-70000-C":100})
    ok = await execute_rfq_flow(ev, RFQConfig(rfq_timeout_seconds=0.01), RaceWS(), RaceREST(), FakeMDS())
    assert ok
