"""
The daily selection rule, shared by the backtest and the live morning job (spec section 7):
a long-gated stock is a candidate when its prediction is above +threshold, a short-gated stock when it is below
-threshold and the stock is shortable and easy to borrow; strongest |prediction| first, at most 3 per GICS sector,
at most 10 a day. The gates fix the direction: a long-gated stock with a negative prediction is skipped, never shorted.
"""
import pandas as pd

MAX_TRADES = 10
MAX_PER_SECTOR = 3


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
