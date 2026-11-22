"""
Evaluator for v2 (long/short) sessions. src/evaluator.py dispatches here for plans with "strategy": "v2"; v1 plans
keep their own code path untouched.

Per trade (real Alpaca fills, Alpaca SIP 1-minute bars as in the backtest):
  entry      the entry's real fill (market order at the open)
  exit       the stop-loss leg's fill if it triggered (exit_reason "stop"), else the closing order's fill
             (exit_reason "close"), else the official close with exit_status "missed"
  horizons   price = close of the last bar that started before open + m minutes (clamped to the close); once the
             stop has filled, every later horizon uses the stop price
  P&L        side-aware (a short gains when the price falls); net figures deduct 5 bps per side

Every existing results.json field keeps its meaning (portfolio_curve stays gross); v2 only adds fields.
After each session: the would-have-traded log (data/candidates/<date>.csv: the realised market-relative return of
every gated stock, traded or not) and data/confirmation.json (progress against the pre-registered criteria).
"""
import datetime as dt
import glob
import json
import math
import os
import re
from types import SimpleNamespace

import pandas as pd

import broker
import model
from storage import exit_order_id, save_json

CURVE_KEYS = ["Open", "1_min", "10_min", "30_min", "1_hour", "5_hour", "EOD"]
TIMEFRAMES = {"1_min": 1, "10_min": 10, "30_min": 30, "1_hour": 60, "5_hour": 300}
COST_PER_SIDE = 0.0005
MAX_EDGE_GAP = dt.timedelta(minutes=5)


class NotReady(Exception):
    """Orders or bars are not final yet; try again later (the free plan's bars are 15 minutes delayed)."""


def default_bars(symbols, start, end):
    import marketdata
    return marketdata.minute_bars(symbols, start, end, fallback=True)


def _price_at(bars: pd.DataFrame, t: dt.datetime) -> float:
    return float(bars[bars.index < t]["close"].iloc[-1])


def _bars_of(frame: pd.DataFrame, symbol: str, session_open, session_close) -> pd.DataFrame:
    if frame is None or frame.empty or symbol not in frame.index.get_level_values(0):
        raise NotReady(f"{symbol}: no 1-minute bars yet")
    bars = frame.xs(symbol, level=0)
    bars = bars[(bars.index >= session_open) & (bars.index < session_close)]
    if bars.empty or bars.index[0] - session_open > MAX_EDGE_GAP or \
            session_close - bars.index[-1] > MAX_EDGE_GAP + dt.timedelta(minutes=1):
        raise NotReady(f"{symbol}: 1-minute bars incomplete for the session")
    return bars


def _round(d: dict, n: int = 2) -> dict:
    return {k: round(v, n) for k, v in d.items()}


def _evaluate_trade(target: dict, plan: dict, client, frame) -> tuple | None:
    """(trade, gross dollar P&L per horizon, net dollar P&L per horizon), or None if the entry never filled."""
    ticker, sign = target["ticker"], broker.side_sign(target)
    session_open = dt.datetime.fromisoformat(plan["market_open"])
    session_close = dt.datetime.fromisoformat(plan["market_close"])

    entry_order = client.get_order_by_id(target["order_id"])
    if broker.status_of(entry_order) not in broker.FINAL_STATUSES:
        raise NotReady(f"{ticker}: entry order is still '{broker.status_of(entry_order)}'")
    qty = float(entry_order.filled_qty or 0)
    if qty == 0:
        return None
    entry = float(entry_order.filled_avg_price)

    stop_fill = None
    leg_id = broker._stop_leg_id(client, target)
    if leg_id:
        leg = client.get_order_by_id(leg_id)
        if broker.status_of(leg) == "filled":
            stop_fill = (float(leg.filled_avg_price), getattr(leg, "filled_at", None) or session_open)
    closing = broker.existing_order(exit_order_id(plan["session_date"], ticker), client)
    if closing is not None and broker.status_of(closing) not in broker.FINAL_STATUSES:
        raise NotReady(f"{ticker}: exit order is still '{broker.status_of(closing)}'")

    bars = _bars_of(frame, ticker, session_open, session_close)
    if stop_fill is not None:
        exit_price, exit_status, exit_reason = stop_fill[0], "filled", "stop"
    elif closing is not None and float(closing.filled_qty or 0) > 0:
        exit_price, exit_status, exit_reason = float(closing.filled_avg_price), "filled", "close"
    else:
        exit_price, exit_status, exit_reason = _price_at(bars, session_close), "missed", "close"

    prices = {}
    for label, m in TIMEFRAMES.items():
        t = min(session_open + dt.timedelta(minutes=m), session_close)
        prices[label] = stop_fill[0] if stop_fill is not None and stop_fill[1] < t else _price_at(bars, t)
    prices["EOD"] = exit_price

    pnl = {k: sign * qty * (p - entry) for k, p in prices.items()}
    pnl_net = {k: pnl[k] - COST_PER_SIDE * qty * (entry + p) for k, p in prices.items()}
    invested = qty * entry
    features = target.get("features") or {}
    trade = {
        "ticker": ticker,
        "sentiment_score": features.get("news_today"),     # FinBERT tone of the fresh news (-1..+1)
        "gap_percent": features.get("gap_pct"),
        "headlines": target.get("headlines", []),
        "qty": qty,
        "allocation_usd": round(invested, 2),
        "open_price": round(entry, 2),
        "exit_price": round(exit_price, 2),
        "exit_status": exit_status,
        "performance_pct": {k: round(sign * (p - entry) / entry * 100, 2) for k, p in prices.items()},
        "pnl_usd": _round(pnl),
        # v2 additions
        "side": target.get("side", "long"),
        "score": target.get("score"),
        "features": features,
        "stop_price": target.get("stop_price"),
        "exit_reason": exit_reason,
        "performance_pct_net": {k: round(v / invested * 100, 2) for k, v in pnl_net.items()},
        "pnl_usd_net": _round(pnl_net),
    }
    return trade, pnl, pnl_net


def evaluate_plan_v2(plan: dict, starting_balance: float, client, bars_fn=None) -> dict:
    bars_fn = bars_fn or default_bars
    session_open = dt.datetime.fromisoformat(plan["market_open"])
    session_close = dt.datetime.fromisoformat(plan["market_close"])
    ordered = [t for t in plan["targets"] if t.get("order_id")]
    frame = bars_fn([t["ticker"] for t in ordered], session_open, session_close - dt.timedelta(minutes=1)) \
        if ordered else None

    trades, rejected = [], [
        {"ticker": c["ticker"], "gap_percent": c.get("gap_percent"), "sentiment_score": c.get("sentiment_score"),
         "headlines": [], "reason": c["decision"],
         "side": c.get("gate"), "score": c.get("score")}
        for c in plan.get("candidates", []) if c.get("decision") != "selected"]
    curve = {k: starting_balance for k in CURVE_KEYS}
    curve_net = dict(curve)
    by_side = {s: {"trades": 0, "invested_usd": 0.0, "pnl_usd": {k: 0.0 for k in CURVE_KEYS[1:]},
                   "pnl_usd_net": {k: 0.0 for k in CURVE_KEYS[1:]}} for s in ("long", "short")}
    for target in plan["targets"]:
        evidence = {"ticker": target["ticker"], "gap_percent": (target.get("features") or {}).get("gap_pct"),
                    "sentiment_score": (target.get("features") or {}).get("news_today"),
                    "headlines": target.get("headlines", []), "side": target.get("side"), "score": target.get("score")}
        if not target.get("order_id"):
            rejected.append({**evidence, "reason": target.get("skip_reason") or plan.get("skipped")
                             or "order could not be placed"})
            continue
        scored = _evaluate_trade(target, plan, client, frame)
        if scored is None:
            rejected.append({**evidence, "reason": "entry order was not filled"})
            continue
        trade, pnl, pnl_net = scored
        side = by_side[trade["side"]]
        side["trades"] += 1
        side["invested_usd"] += trade["allocation_usd"]
        for k in pnl:
            curve[k] += pnl[k]
            curve_net[k] += pnl_net[k]
            side["pnl_usd"][k] += pnl[k]
            side["pnl_usd_net"][k] += pnl_net[k]
        trades.append(trade)

    for side in by_side.values():
        inv = side["invested_usd"]
        side["return_pct"] = {k: round(v / inv * 100, 2) if inv else 0.0 for k, v in side["pnl_usd"].items()}
        side["pnl_usd"], side["pnl_usd_net"] = _round(side["pnl_usd"]), _round(side["pnl_usd_net"])
        side["invested_usd"] = round(inv, 2)
    threshold = plan.get("score_threshold")
    curve, curve_net = _round(curve), _round(curve_net)
    return {
        "date": plan["session_date"],
        "starting_balance": round(starting_balance, 2),
        "ending_balance": curve["EOD"],
        "portfolio_curve": curve,
        "invested_usd": round(sum(t["allocation_usd"] for t in trades), 2),
        "sentiment_threshold": None,
        "trades": trades,
        "rejected": rejected,
        # v2 additions
        "strategy": "v2",
        "strategy_name": plan.get("strategy_name"),
        "model_sha256": plan.get("model_sha256"),
        "score_thresholds": {"long": threshold, "short": -threshold} if threshold is not None else None,
        "costs_bps": round(COST_PER_SIDE * 10_000),
        "portfolio_curve_net": curve_net,
        "ending_balance_net": curve_net["EOD"],
        "by_side": by_side,
    }


# --- After each session: the would-have-traded log and the confirmation progress -----------------------

def _truthy(v) -> bool:
    return str(v).strip().lower() in ("true", "1", "yes")


def log_candidates(plan: dict, entry: dict, bars_fn, scores_dir: str, candidates_dir: str):
    """The realised market-relative return (open -> close, stock - beta x sector ETF) of every gated stock."""
    path = os.path.join(scores_dir, f"{plan['session_date']}.csv")
    if not os.path.exists(path):
        print(f"::warning::No score log {path}; the would-have-traded log for {plan['session_date']} is skipped")
        return None
    scores = pd.read_csv(path)
    gated = scores[scores["gate_long"].map(_truthy) | scores["gate_short"].map(_truthy)].copy()
    if gated.empty:
        return None
    session_open = dt.datetime.fromisoformat(plan["market_open"])
    session_close = dt.datetime.fromisoformat(plan["market_close"])
    frame = (bars_fn or default_bars)(sorted(set(gated["symbol"]) | set(gated["sector_etf"])), session_open,
                                      session_close - dt.timedelta(minutes=1))
    session = SimpleNamespace(open=session_open)

    def bars(sym):
        if frame is None or frame.empty or sym not in frame.index.get_level_values(0):
            return None
        b = frame.xs(sym, level=0)
        return b[(b.index >= session_open) & (b.index < session_close)]

    traded = {t["ticker"] for t in entry["trades"]}
    gated["gate"] = ["long" if _truthy(x) else "short" for x in gated["gate_long"]]
    gated["target"] = [model.market_relative_return(bars(r.symbol), bars(r.sector_etf), float(r.beta), session)
                       for r in gated.itertuples()]
    gated["traded"] = gated["symbol"].isin(traded)
    gated.insert(0, "session_date", plan["session_date"])
    cols = ["session_date", "symbol", "gate", "sector", "sector_etf", "beta", "score", "decision", "gap_atr",
            "news_today", "surprise", "target", "traded"]
    os.makedirs(candidates_dir, exist_ok=True)
    out = os.path.join(candidates_dir, f"{plan['session_date']}.csv")
    gated[[c for c in cols if c in gated.columns]].to_csv(out, index=False, float_format="%.6g")
    return out


def _finite(x):
    return None if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))) else x


def update_confirmation(results: list, candidates_dir: str, confirmation_file: str, prereg_file: str) -> dict:
    """Live confirmation so far: the IC over every gated stock and the mean net return per trade at the close,
    against the pre-registered criteria."""
    prereg = {}
    if os.path.exists(prereg_file):
        with open(prereg_file) as f:
            prereg = json.load(f)
    live = prereg.get("live_confirmation", {})
    nums = [int(x) for x in re.findall(r"(\d+)\s+(?:sessions|trades)", live.get("length", ""))]
    req_sessions, req_trades = (nums + [60, 150])[:2] if len(nums) >= 2 else (60, 150)
    t_req = float((re.findall(r"t > ([\d.]+)", live.get("primary", "")) or [2.0])[0])

    v2 = [e for e in results if e.get("strategy") == "v2"]
    trades = [t for e in v2 for t in e["trades"]]
    frames = [pd.read_csv(p) for p in sorted(glob.glob(os.path.join(candidates_dir, "*.csv")))]
    cands = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["score", "target", "session_date"])
    cands = cands.dropna(subset=["score", "target"])
    ic, ic_t = model.information_coefficient(cands["score"], cands["target"], cands["session_date"]) \
        if len(cands) else (float("nan"), float("nan"))
    net = pd.Series([t["performance_pct_net"]["EOD"] / 100 for t in trades], dtype=float)
    net_t = float(net.mean() / (net.std(ddof=1) / math.sqrt(len(net)))) if len(net) > 1 and net.std(ddof=1) > 0 else None
    complete = len(v2) >= req_sessions and len(trades) >= req_trades
    primary = bool(ic > 0 and ic_t > t_req) if complete else None
    secondary = bool(net.mean() > 0) if complete and len(net) else None
    conf = {
        "updated_after": v2[-1]["date"] if v2 else None,
        "first_session": v2[0]["date"] if v2 else None,
        "strategy_name": v2[-1].get("strategy_name") if v2 else None,
        "model_sha256": v2[-1].get("model_sha256") if v2 else None,
        "sessions": len(v2), "trades": len(trades),
        "required": {"sessions": req_sessions, "trades": req_trades, "ic_t": t_req},
        "progress": {"sessions": round(min(1.0, len(v2) / req_sessions), 3),
                     "trades": round(min(1.0, len(trades) / req_trades), 3)},
        "ic": _finite(round(ic, 4) if ic == ic else None), "ic_t": _finite(round(ic_t, 2) if ic_t == ic_t else None),
        "candidate_rows": int(len(cands)), "candidate_sessions": int(cands["session_date"].nunique()) if len(cands) else 0,
        "mean_net_per_trade_pct": _finite(round(float(net.mean()) * 100, 4) if len(net) else None),
        "mean_net_t": _finite(round(net_t, 2) if net_t is not None else None),
        "primary_passed": primary, "secondary_passed": secondary,
        "status": "in progress" if not complete else ("passed" if primary else "failed"),
    }
    save_json(confirmation_file, conf)
    return conf
