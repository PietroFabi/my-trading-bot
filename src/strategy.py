"""
The daily selection rule, shared by the backtest and the live morning job (spec section 7):
a long-gated stock is a candidate when its prediction is above +threshold, a short-gated stock when it is below
-threshold and the stock is shortable and easy to borrow; strongest |prediction| first, at most 3 per GICS sector,
at most 10 a day. The gates fix the direction: a long-gated stock with a negative prediction is skipped, never shorted.
"""
import hashlib
import json
import math

import numpy as np
import pandas as pd

MAX_TRADES = 10
MAX_PER_SECTOR = 3
RISK_PER_ATR = 0.0006   # R: the loss if the stock moves 1 ATR against us, as a share of equity
MAX_STOCK = 0.04        # at most 4% of equity per stock
MAX_DAY = 0.20          # at most 20% of equity per day
LIVE_ENTRIES = {"open"}  # the frozen strategy enters at the open; the 09:35 breakout path is not built


def select_candidates(day: pd.DataFrame, threshold: float, max_trades: int = MAX_TRADES,
                      max_per_sector: int = MAX_PER_SECTOR) -> pd.DataFrame:
    """`day` has symbol, sector, gate_long, gate_short, pred and optionally shortable. Returns the chosen rows,
    strongest first, with a `side` column."""
    if day.empty:
        return day.assign(side=pd.Series(dtype=str))
    shortable = day["shortable"].astype(bool) if "shortable" in day else True
    longs = day[day["gate_long"].astype(bool) & (day["pred"] > threshold)].assign(side="long")
    shorts = day[day["gate_short"].astype(bool) & (day["pred"] < -threshold) & shortable].assign(side="short")
    pool = pd.concat([longs, shorts])
    pool = pool.loc[pool["pred"].abs().sort_values(ascending=False, kind="stable").index]
    chosen, per_sector = [], {}
    for idx, r in pool.iterrows():
        if len(chosen) == max_trades:
            break
        if per_sector.get(r["sector"], 0) >= max_per_sector:
            continue
        per_sector[r["sector"]] = per_sector.get(r["sector"], 0) + 1
        chosen.append(idx)
    return pool.loc[chosen]


def fade_candidates(day: pd.DataFrame, max_trades: int = MAX_TRADES, max_per_sector: int = MAX_PER_SECTOR) -> pd.DataFrame:
    """
    The pre-registered gap-fade rule (found in exploration, tested on the hold-out): trade AGAINST each gated
    stock's gap. Long-gated (gap up) -> short, short-gated (gap down) -> long; score = -gap_atr, strongest
    |gap_atr| first; no model, no threshold; same daily, sector and borrow limits.
    """
    flipped = day.assign(gate_long=day["gate_short"], gate_short=day["gate_long"], pred=-day["gap_atr"])
    return select_candidates(flipped, 0.0, max_trades, max_per_sector)


# --- Sizing and stops (shared with backtest/simulate.py) ------------------------------------------------

def _shares(notional: float, price: float) -> int:
    return int(math.floor(notional / price + 1e-9))


def position_size(atr_pct: float, ref_price: float, equity: float) -> int:
    """Whole shares worth equity x R / ATR%, capped at 4% of equity (sized from the 08:55 reference price)."""
    notional = min(equity * RISK_PER_ATR / (atr_pct / 100.0), MAX_STOCK * equity)
    return _shares(notional, ref_price)


def stop_price(entry: float, sign: int, k: float, atr_pct: float) -> float:
    """k x ATR against the trade, in whole cents (as an Alpaca stop order must be)."""
    return round(entry * (1 - sign * k * atr_pct / 100.0), 2)


def apply_day_cap(trades: list, equity: float, cap: float = MAX_DAY) -> list:
    """Scale every planned trade's quantity down so the day's planned notional is at most cap x equity (sized at
    plan time from reference prices: trades that later do not enter still used up the budget)."""
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


# --- The frozen model, live ---------------------------------------------------------------------------

class FrozenModelMismatch(Exception):
    """The frozen model file is not the one recorded in the pre-registration."""


def model_sha256(path: str) -> str:
    """sha256 with line endings normalised to LF (same as backtest/freeze.py)."""
    with open(path, "rb") as f:
        return hashlib.sha256(f.read().replace(b"\r\n", b"\n")).hexdigest()


def load_frozen(model_path: str, prereg_path: str, live: str) -> dict:
    """The strategy that goes live, after checking the model file against the pre-registered hash."""
    with open(prereg_path) as f:
        expected = json.load(f)["frozen_model_sha256"]
    actual = model_sha256(model_path)
    if actual != expected:
        raise FrozenModelMismatch(f"{model_path} has sha256 {actual[:12]}, pre-registered {expected[:12]}")
    with open(model_path) as f:
        body = json.load(f)
    return {"strategy": live, "spec": body["strategies"][live], "sha256": actual, "model": body}


def _clean(x):
    """A JSON-safe plain Python value (NaN -> None)."""
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        x = x.item()
    if isinstance(x, float) and math.isnan(x):
        return None
    return x


def score_table(table: pd.DataFrame, spec: dict) -> pd.Series:
    """The frozen score for every row: the ridge prediction (market-relative return) or -gap_atr for the fade."""
    import model
    if spec["kind"] == "ridge":
        ridge = model.RidgeModel.from_dict(spec["ridge"])
        return pd.Series(ridge.predict(model.feature_matrix(table)), index=table.index)
    if spec["kind"] == "gap_fade":
        return -table["gap_atr"].astype(float)
    raise ValueError(f"unknown strategy kind {spec['kind']!r}")


TARGET_FEATURES = ["gap_pct", "excess_gap_pct", "gap_atr", "rvol", "trend", "dist_52w_high", "news_today", "baseline",
                   "surprise", "news_volume", "n_fresh"]


def choose_trades(table: pd.DataFrame, frozen: dict, equity: float, is_shortable) -> tuple:
    """
    The morning decision. `table` = features.morning_feature_table (indexed by symbol). Returns
    (table with `score` and `decision` columns, [target dicts]). `is_shortable(symbol)` is asked only for stocks
    that would be shorted.
    """
    spec = frozen["spec"]
    if spec["entry"] not in LIVE_ENTRIES:
        raise NotImplementedError(f"live entry mode {spec['entry']!r} is not built (the frozen strategy uses 'open')")
    scored = table.copy()
    scored["score"] = score_table(scored, spec)
    scored["decision"] = "not gated"
    gated = scored[scored["gate_long"].astype(bool) | scored["gate_short"].astype(bool)].copy()
    gated["pred"] = gated["score"]
    fade = spec["kind"] == "gap_fade"
    threshold = 0.0 if fade else float(spec["threshold_return"])
    if fade:
        signal = pd.Series(True, index=gated.index)
        would_short = gated["gate_long"].astype(bool)
    else:
        signal = (gated["gate_long"].astype(bool) & (gated["pred"] > threshold)) | \
                 (gated["gate_short"].astype(bool) & (gated["pred"] < -threshold))
        would_short = gated["gate_short"].astype(bool) & signal
    gated["shortable"] = [bool(is_shortable(s)) if would_short[s] else True for s in gated.index]
    scored.loc[gated.index[~signal], "decision"] = "below threshold"
    scored.loc[gated.index[signal & ~gated["shortable"]], "decision"] = "hard to borrow"
    scored.loc[gated.index[signal & gated["shortable"]], "decision"] = "daily or sector limit"

    pool = gated.rename_axis("symbol").reset_index()
    chosen = fade_candidates(pool) if fade else select_candidates(pool, threshold)
    planned = []
    for r in chosen.to_dict("records"):
        sign = 1 if r["side"] == "long" else -1
        planned.append({
            "ticker": r["symbol"], "side": r["side"], "sector": r["sector"], "score": _clean(r["pred"]),
            "threshold": threshold, "ref_price": _clean(r["ref_price"]), "prev_close": _clean(r["prev_close"]),
            "atr_pct": _clean(r["atr_pct"]), "qty": position_size(r["atr_pct"], r["ref_price"], equity),
            "stop_price": stop_price(r["ref_price"], sign, spec["stop_k"], r["atr_pct"]) if spec.get("stop_k") else None,
            "features": {f: _clean(r.get(f)) for f in TARGET_FEATURES},
        })
    targets = []
    for t in apply_day_cap(planned, equity):
        if t["qty"] < 1:
            scored.loc[t["ticker"], "decision"] = "position too small"
            continue
        scored.loc[t["ticker"], "decision"] = "selected"
        targets.append(t)
    return scored, targets
