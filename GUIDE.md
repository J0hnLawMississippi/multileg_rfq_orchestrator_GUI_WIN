# RFQ Orchestrator — User Guide

## Overview

`rfq_orchestrator_win.py` prices an options structure against live Deribit market
data, checks it against configurable risk thresholds, and — if you confirm —
submits a block-trade RFQ on Coincall and automatically accepts the first quote
that passes validation.

---

## Installation

Run `install_win.bat` once. It checks Python (3.10+ required, 3.11+ recommended),
upgrades pip, and installs all dependencies.

---

## Credentials

Create a file named `.env` in the same folder as the script:

```
COINCALL_SUB_API_KEY=your_sub_account_api_key
COINCALL_SUB_API_SECRET=your_sub_account_api_secret
COINCALL_API_KEY=your_main_account_api_key
COINCALL_API_SECRET=your_main_account_api_secret
```

Only the keys for the account you intend to trade on are required.

---

## Running

```
python rfq_orchestrator_win.py --account sub
python rfq_orchestrator_win.py --account main
```

`--account sub` (default) uses `COINCALL_SUB_API_KEY` / `COINCALL_SUB_API_SECRET`.  
`--account main` uses `COINCALL_API_KEY` / `COINCALL_API_SECRET`.

---

## Configuration

All configuration is set by editing the `RFQConfig(...)` block and the
`leg_specs` list inside the `main()` function at the bottom of the script.

---

### Structure to trade — `leg_specs`

`leg_specs` is a Python list of strings, one string per option leg.

**Format of each leg string:**

```
"<direction> <size> <DDMMMYY>-<strike>-<type>"
```

| Field | Values | Meaning |
|-------|--------|---------|
| `direction` | `L` | Long — we buy this leg |
| | `S` | Short — we sell this leg |
| `size` | decimal number | Quantity in BTC (e.g. `0.5`, `1.0`) |
| `DDMMMYY` | e.g. `20MAR26` | Expiry date in Deribit format |
| `strike` | integer | Strike price in USD |
| `type` | `C` | Call option |
| | `P` | Put option |

**Example — a 6-leg strangle structure:**

```python
leg_specs = [
    "L 0.6 20MAR26-70000-P",   # Long 0.6 BTC Mar-26 70000 put
    "L 0.6 20MAR26-78000-C",   # Long 0.6 BTC Mar-26 78000 call
    "L 0.5 24APR26-70000-P",   # Long 0.5 BTC Apr-26 70000 put
    "L 0.5 24APR26-78000-C",   # Long 0.5 BTC Apr-26 78000 call
    "S 0.5 29MAY26-70000-P",   # Short 0.5 BTC May-26 70000 put
    "S 0.5 29MAY26-78000-C",   # Short 0.5 BTC May-26 78000 call
]
```

The expiry date must exactly match a Deribit expiry (e.g. `28MAR25`, `25APR25`).
Check available expiries on Deribit before configuring.

---

### Pricing parameters

| Variable | Type | Meaning |
|----------|------|---------|
| `target_spot` | float (USD) | The BTC spot price at which the structure is priced. Set this to your desired entry spot. Does not need to equal the live price — it shifts the forward used for pricing. |

---

### Threshold — when to send the RFQ

The threshold controls whether the program proceeds to send the RFQ at all,
based on the structure's theoretical value at `target_spot`.

| Variable | Type | Meaning |
|----------|------|---------|
| `threshold_type` | `ThresholdType.CREDIT` or `ThresholdType.DEBIT` | `CREDIT` — the structure generates net premium received (e.g. short strangle). `DEBIT` — the structure costs net premium (e.g. long strangle). |
| `threshold_value` | float (USD) | Minimum acceptable net value. For CREDIT: structure must generate at least this many USD in premium. For DEBIT: structure must cost at most this many USD. Always positive. |

**Examples:**
- `threshold_type=CREDIT, threshold_value=250` — only proceed if the structure
  generates at least $250 credit at `target_spot`.
- `threshold_type=DEBIT, threshold_value=550` — only proceed if the structure
  costs at most $550 debit at `target_spot`.

---

### Drawdown limits — risk checks before sending the RFQ

The program computes the worst-case P&L of the structure over `drawdown_days`
across a grid of spot prices (`target_spot` +/- `spot_range_pct`), under two
vol scenarios. Both checks must pass for the RFQ to be sent.

| Variable | Type | Meaning |
|----------|------|---------|
| `max_dd_usd_flat` | float, negative (USD) | Maximum tolerated loss (flat vol scenario). E.g. `-2000` means the structure must not lose more than $2000 at any spot within the range. |
| `max_dd_shocked` | float, negative (USD) | Maximum tolerated loss under the vol shock scenario. Usually tighter than `max_dd_usd_flat`. |
| `drawdown_days` | int | Number of days forward to project the P&L grid. |
| `vol_shock` | float (decimal) | Additive vol shift applied to all IVs in the shocked scenario. `-0.10` means subtract 10 vol points from every strike. |
| `spot_range_pct` | float (decimal) | Half-width of the spot grid. `0.50` = evaluate from `target_spot * 0.50` to `target_spot * 1.50`. |
| `spot_grid_points` | int | Number of spot points in the grid. 300 is sufficient for most structures. |

---

### Quote validation — when to accept an incoming quote

After the RFQ is sent, Coincall market makers submit quotes. Each incoming quote
is validated against live Deribit marks before being accepted.

| Variable | Type | Meaning |
|----------|------|---------|
| `max_slippage_percent` | float (%) | Maximum allowed total slippage of the quoted structure vs Deribit marks. E.g. `7.5` = reject if the quote is worse than 7.5% below the Deribit mark value. |
| `price_deviation_threshold` | float (decimal) | Per-leg price band. E.g. `0.15` = reject any individual leg quoted more than 15% away from its Deribit mark price (in either direction). |
| `max_leg_price_usd` | float (USD) | For debit legs (legs we pay): reject if any single leg's quoted price exceeds this cap. Protects against absurdly priced quotes on any one leg. |
| `rfq_timeout_seconds` | float (seconds) | How long to wait for a valid quote before cancelling the RFQ. |

---

### Infrastructure parameters (normally leave unchanged)

| Variable | Type | Meaning |
|----------|------|---------|
| `coincall_base_url` | str | Coincall REST API base URL. |
| `coincall_ws_url` | str | Coincall WebSocket URL. |
| `rate_limit_per_second` | int | Max REST requests per second to Coincall. |
| `max_snapshot_age_ms` | float (ms) | If a Deribit WebSocket snapshot is older than this when a quote arrives, a warning is logged. The quote is still validated but marked as potentially stale. |

---

## Execution flow

1. **Startup** — enumerates Deribit option instruments around your leg strikes
   via REST (one call).
2. **WebSocket warmup** — connects to Deribit (IV surface) and Coincall
   (RFQ/quote feed) simultaneously. Waits up to 20s for both to be ready.
3. **Evaluation** — prices the structure at `target_spot` using live Deribit
   WebSocket data. Runs drawdown scenarios. Prints results.
4. **Gate** — if all checks pass, prompts `Proceed with RFQ? [y/N]`.
   Type `y` and press Enter to continue, anything else to abort.
5. **RFQ** — submits the block-trade RFQ to Coincall. Waits for quotes.
6. **Quote validation** — each incoming quote is checked against live Deribit
   marks. The first quote that passes all checks is automatically accepted.
7. **Shutdown** — both WebSocket connections are cleanly closed.

---

## Typical workflow for a new trade

1. Identify the structure (legs, strikes, expiries, sizes).
2. Check expiry dates exist on Deribit.
3. Edit `main()` in `rfq_orchestrator_win.py`:
   - Set `leg_specs`.
   - Set `target_spot` to your desired pricing spot.
   - Set `threshold_type` and `threshold_value`.
   - Set `max_dd_usd_flat`, `max_dd_shocked`, `drawdown_days`.
   - Adjust `max_slippage_percent`, `price_deviation_threshold`,
     `max_leg_price_usd`, `rfq_timeout_seconds` as needed.
4. Run: `python rfq_orchestrator_win.py --account sub`
5. Review the printed evaluation. If thresholds pass, type `y` to send the RFQ.
