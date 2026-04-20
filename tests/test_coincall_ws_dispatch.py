from coincall_ws_win import CCMsgType, CoincallWSClient, QuoteReceived, RFQStateUpdate
from tests.helpers.ws_builders import quote_received_message, rfq_taker_message


def test_dispatch_routes_to_registered_queue():
    ws = CoincallWSClient("wss://example", "k", "s")
    q = ws.register("req-1")
    ws._dispatch(rfq_taker_message("req-1", state="ACTIVE"))
    ev = q.get_nowait()
    assert isinstance(ev, RFQStateUpdate)


def test_dispatch_ignores_unregistered_request():
    ws = CoincallWSClient("wss://example", "k", "s")
    ws._dispatch(quote_received_message("missing"))
    assert ws._queues == {}


def test_dispatch_parses_quote_received():
    ws = CoincallWSClient("wss://example", "k", "s")
    q = ws.register("req-1")
    msg = quote_received_message("req-1", quote_id="q-9", state="OPEN", quote_side="BUY")
    msg["dt"] = CCMsgType.QUOTE_RCVD
    ws._dispatch(msg)
    ev = q.get_nowait()
    assert isinstance(ev, QuoteReceived)
    assert ev.quote_id == "q-9"
