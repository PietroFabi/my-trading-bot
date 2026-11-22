"""
Step 8: simulate one intraday trade from 1-minute bars, the way the live bot can actually trade it.

Entry
  open : market order queued before the open -> filled at the first 09:30 bar's open.
  orb  : the live morning job stays alive only until ~09:36, so the breakout is decided ONCE: the 09:35 bar must
         close beyond the 09:30-09:34 range in the trade's direction (and first-5-minute relative volume must be
         at least ORB_MIN_RVOL5). Entry at that close (09:36). Otherwise no trade.
Stop (optional, k x ATR from the entry, an OTO order live from the entry bar itself)
  priced in whole cents; filled at the stop price, or at the bar's open if the bar opened beyond it. A stop and a horizon inside the same
  bar count as the stop first. Both choices are conservative.
Exit
  at the close. Horizons (as in src/evaluator.py) = close of the last bar that started before open + m minutes,
  clamped to the close on half days.
Sizing
  notional = equity x R / ATR%, capped at 4% of equity, whole shares from the 08:55 reference price (as live,
  which sizes before the open); apply_day_cap scales a day down to 20% of equity.
Costs
  5 bps per side, so net = gross - 0.10%.
"""
import datetime as dt
import math

import pandas as pd

HORIZONS = {"1_min": 1, "10_min": 10, "30_min": 30, "1_hour": 60, "5_hour": 300, "EOD": None}
COST_PER_SIDE = 0.0005
RISK_PER_ATR = 0.0006
MAX_STOCK = 0.04
MAX_DAY = 0.20
OPENING_RANGE_MINUTES = 5
ORB_MIN_RVOL5 = 1.0


def _shares(notional: float, price: float) -> int:
    return int(math.floor(notional / price + 1e-9))


def position_size(atr_pct: float, ref_price: float, equity: float) -> int:
    notional = min(equity * RISK_PER_ATR / (atr_pct / 100.0), MAX_STOCK * equity)
    return _shares(notional, ref_price)


def stop_price(entry: float, sign: int, k: float, atr_pct: float) -> float:
    """k x ATR against the trade, in whole cents (as an Alpaca stop order must be)."""
    return round(entry * (1 - sign * k * atr_pct / 100.0), 2)


def _price_at(bars: pd.DataFrame, t: dt.datetime):
    before = bars[bars.index < t]
    return float(before["close"].iloc[-1]) if len(before) else None


def _orb_entry(bars: pd.DataFrame, session, sign: int):
    """(entry time, entry price) if the 09:35 bar closes beyond the opening range in the trade's direction."""
    range_end = session.open + dt.timedelta(minutes=OPENING_RANGE_MINUTES)
    opening = bars[bars.index < range_end]
    decision = bars[bars.index == range_end]
    if opening.empty or decision.empty:
        return None
    close = float(decision["close"].iloc[0])
    broke = close > opening["high"].max() if sign > 0 else close < opening["low"].min()
    return (range_end + dt.timedelta(minutes=1), close) if broke else None


def _stop_hit(bars: pd.DataFrame, entry_time: dt.datetime, stop: float, sign: int):
    """(bar start, fill price) of the first bar from the entry on that reaches the stop."""
    live = bars[bars.index >= entry_time]
    if sign > 0:
        hit = live[live["low"] <= stop]
    else:
        hit = live[live["high"] >= stop]
    if hit.empty:
        return None
    bar = hit.iloc[0]
    gapped = bar["open"] <= stop if sign > 0 else bar["open"] >= stop
    return hit.index[0], float(bar["open"]) if gapped else stop


def _no_trade(reason: str, side: str, qty: int, ref_price: float) -> dict:
    empty = {h: None for h in HORIZONS}
    return {"entered": False, "side": side, "entry_time": None, "entry": None, "exit": None, "exit_reason": reason,
            "qty": qty, "ref_price": ref_price, "ret_gross": dict(empty), "ret_net": dict(empty), "pnl_net": dict(empty)}


def simulate_trade(row: pd.Series, bars: pd.DataFrame, session, entry_mode: str, stop_k, equity: float) -> dict:
    """One trade for a gated row. `bars` = that stock's 1-minute bars (index = timestamp; open/high/low/close)."""
    side = row["side"] if "side" in row.index and isinstance(row["side"], str) else ("long" if row["gate_long"] else "short")
    sign = 1 if side == "long" else -1
    ref_price = float(row["ref_price"])
    if bars.empty:
        return _no_trade("no_bars", side, 0, ref_price)
    bars = bars.sort_index()
    qty = position_size(float(row["atr_pct"]), ref_price, equity)
    if qty < 1:
        return _no_trade("size", side, 0, ref_price)

    if entry_mode == "open":
        first = bars[bars.index >= session.open]
        if first.empty or first.index[0] != session.open:
            return _no_trade("no_bars", side, qty, ref_price)
        entry_time, entry = session.open, float(first["open"].iloc[0])
    elif entry_mode == "orb":
        rvol5 = row.get("rvol5")
        if rvol5 is None or pd.isna(rvol5) or rvol5 < ORB_MIN_RVOL5:
            return _no_trade("rvol", side, qty, ref_price)
        got = _orb_entry(bars, session, sign)
        if got is None:
            return _no_trade("no_breakout", side, qty, ref_price)
        entry_time, entry = got
    else:
        raise ValueError(f"unknown entry mode {entry_mode!r}")

    stop = _stop_hit(bars, entry_time, stop_price(entry, sign, stop_k, float(row["atr_pct"])), sign) \
        if stop_k is not None else None

    ret_gross, ret_net, pnl_net = {}, {}, {}
    for name, minutes in HORIZONS.items():
        t = session.close if minutes is None else min(session.open + dt.timedelta(minutes=minutes), session.close)
        if t <= entry_time:
            ret_gross[name] = ret_net[name] = pnl_net[name] = None
            continue
        price = stop[1] if stop is not None and stop[0] < t else _price_at(bars, t)
        if price is None:
            ret_gross[name] = ret_net[name] = pnl_net[name] = None
            continue
        ret_gross[name] = sign * (price / entry - 1)
        ret_net[name] = ret_gross[name] - 2 * COST_PER_SIDE
        pnl_net[name] = qty * entry * ret_net[name]

    exit_price = stop[1] if stop is not None else _price_at(bars, session.close)
    return {"entered": True, "side": side, "entry_time": entry_time, "entry": entry, "exit": exit_price,
            "exit_reason": "stop" if stop is not None else "close", "qty": qty, "ref_price": ref_price,
            "ret_gross": ret_gross, "ret_net": ret_net, "pnl_net": pnl_net}


def apply_day_cap(trades: list, equity: float, cap: float = MAX_DAY) -> list:
    """Scale every planned trade's quantity down so the day's planned notional is at most cap x equity (as live,
    which sizes at plan time from reference prices: trades that later do not enter still used up the budget)."""
    def notional(t):
        return t["qty"] * (t.get("ref_price") or t["entry"])
    total = sum(notional(t) for t in trades if t["qty"] > 0)
    if total <= cap * equity:
        return [dict(t) for t in trades]
    factor = cap * equity / total
    out = []
    for t in trades:
        t = {**t, "qty": _shares(t["qty"] * factor, 1.0)}
        if t.get("entered"):
            t["pnl_net"] = {h: None if r is None else t["qty"] * t["entry"] * r for h, r in t["ret_net"].items()}
        out.append(t)
    return out
