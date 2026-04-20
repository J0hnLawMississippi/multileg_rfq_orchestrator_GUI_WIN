from coincall_ws_win import QuoteReceived, QuoteState
from rfq_orchestrator_win import QuoteValidator, RFQConfig, ThresholdType


def _quote(side="BUY", price="120", quote_side="BUY"):
    return QuoteReceived(
        request_id="req",
        quote_id="q1",
        state=QuoteState.OPEN,
        quote_side=quote_side,
        legs=(
            {
                "instrumentName": "BTCUSD-20MAR26-70000-C",
                "price": price,
                "quantity": "1",
                "side": side,
            },
        ),
    )


def test_quote_validator_happy_credit():
    cfg = RFQConfig(threshold_type=ThresholdType.CREDIT, threshold_value=100, price_deviation_threshold=0.5)
    qv = QuoteValidator(cfg, {"BTC-20MAR26-70000-C": 100})
    ok, reason, net = qv.validate(_quote())
    assert ok
    assert reason == "OK"
    assert net > 0


def test_quote_validator_wrong_quote_side_rejected():
    cfg = RFQConfig(threshold_type=ThresholdType.CREDIT)
    qv = QuoteValidator(cfg, {"BTC-20MAR26-70000-C": 100})
    ok, reason, _ = qv.validate(_quote(quote_side="SELL"))
    assert not ok
    assert "Wrong quoteSide" in reason
