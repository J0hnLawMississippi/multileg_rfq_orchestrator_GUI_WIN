from __future__ import annotations


def rfq_taker_message(request_id: str, state: str = "ACTIVE") -> dict:
    return {
        "dt": 129,
        "d": {"requestId": request_id, "state": state, "legs": []},
    }


def quote_received_message(
    request_id: str,
    quote_id: str = "q-1",
    state: str = "OPEN",
    quote_side: str = "BUY",
    price: float = 100.0,
    instrument: str = "BTCUSD-20MAR26-70000-C",
    side: str = "BUY",
    qty: str = "1",
) -> dict:
    return {
        "dt": 131,
        "d": {
            "requestId": request_id,
            "quoteId": quote_id,
            "state": state,
            "quoteSide": quote_side,
            "legs": [
                {
                    "instrumentName": instrument,
                    "side": side,
                    "quantity": qty,
                    "price": str(price),
                }
            ],
        },
    }


def deribit_index_push(price: float = 70000.0, timestamp: int = 1710000000000) -> dict:
    return {
        "method": "subscription",
        "params": {
            "channel": "deribit_price_index.btc_usd",
            "data": {"price": price, "timestamp": timestamp},
        },
    }


def deribit_ticker_push(
    instrument: str = "BTC-20MAR26-70000-C",
    mark_iv_pct: float = 65.0,
    mark_price: float = 0.01,
    underlying: float = 70100.0,
    index: float = 70000.0,
    timestamp: int = 1710000000001,
) -> dict:
    return {
        "method": "subscription",
        "params": {
            "channel": f"ticker.{instrument}.100ms",
            "data": {
                "instrument_name": instrument,
                "mark_price": mark_price,
                "mark_iv": mark_iv_pct,
                "underlying_price": underlying,
                "index_price": index,
                "timestamp": timestamp,
            },
        },
    }
