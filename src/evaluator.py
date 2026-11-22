"""
Evaluator: turn each finished session's real Alpaca fills into an entry in results.json.

- Entry price  = the buy's real fill (market order executed at the open).
- Exit price   = the sell's real fill (market order just before the close). If the exit was
                 missed, the official close is used and the trade is flagged.
- Curve points = the real share count valued at Yahoo 1-minute prices.

The session date always comes from the plan file, never from the clock, so a
run that GitHub starts hours late still records the right day.
"""
import datetime as dt
import os

import yfinance as yf
from alpaca.trading.enums import OrderSide, QueryOrderStatus
from alpaca.trading.requests import GetOrdersRequest

import broker
import evaluator_v2
from market_session import now_ny
from storage import (CANDIDATES_DIR, CONFIRMATION_FILE, PREREG_FILE, RESULTS_FILE, PLANS_DIR, SCORE_BAND_FILE, SCORES_DIR,
                     all_plan_paths, exit_order_id, late_exit_order_id, load_json, plan_path, save_json)

INITIAL_CAPITAL = broker.FUND_SIZE_USD  # starting value of the fund shown on the site

# Minutes after the open. On half days, points past the close use the closing price.
TIMEFRAMES = {
    "1_min": 1,
    "10_min": 10,
    "30_min": 30,
    "1_hour": 60,
    "5_hour": 300,
}
# Keys the portfolio site reads from portfolio_curve. Do not rename.
CURVE_KEYS = ["Open", "1_min", "10_min", "30_min", "1_hour", "5_hour", "EOD"]

# Wait this long after the close before expecting closing fills and complete 1-minute bars.
SETTLE_AFTER_CLOSE = dt.timedelta(minutes=10)
# The first/last bar must be this close to the open/close, or the data has a hole.
MAX_EDGE_GAP = dt.timedelta(minutes=5)


class NotReady(Exception):
    """The session's orders have not reached a final state yet; try again later."""


def _minute_bars(ticker: str, session_date: dt.date, session_open: dt.datetime, session_close: dt.datetime):
    """Regular-hours 1-minute bars for one ticker on one explicit date."""
    data = yf.download(
        [ticker],
        start=session_date,
        end=session_date + dt.timedelta(days=1),
        interval="1m",
        progress=False,
    )
    if data.empty:
        raise RuntimeError(f"{ticker}: no 1-minute data for {session_date}")
    bars = data.xs(ticker, axis=1, level="Ticker") if data.columns.nlevels > 1 else data
    bars = bars.dropna(subset=["Open", "Close"])
    # Keep regular hours only. A bar's timestamp is its start: 09:30 covers 09:30-09:31.
    bars = bars[(bars.index >= session_open) & (bars.index < session_close)]
    if bars.empty:
        raise RuntimeError(f"{ticker}: no regular-hours bars for {session_date}")
    if bars.index[0] - session_open > MAX_EDGE_GAP or session_close - bars.index[-1] > MAX_EDGE_GAP + dt.timedelta(minutes=1):
        raise RuntimeError(f"{ticker}: incomplete session data ({bars.index[0]:%H:%M}-{bars.index[-1]:%H:%M})")
    return bars


def _price_at(bars, t: dt.datetime) -> float:
    """Price at clock time t = close of the last bar that started before t."""
    return float(bars[bars.index < t]["Close"].iloc[-1])


def _evaluate_trade(target: dict, plan: dict, client):
    """Returns (trade_dict, {curve_key: dollar_pnl}) or None if the buy never filled."""
    ticker = target["ticker"]
    session_date = dt.date.fromisoformat(plan["session_date"])
    session_open = dt.datetime.fromisoformat(plan["market_open"])
    session_close = dt.datetime.fromisoformat(plan["market_close"])

    buy = client.get_order_by_id(target["order_id"])
    if broker.status_of(buy) not in broker.FINAL_STATUSES:
        raise NotReady(f"{ticker}: buy order is still '{broker.status_of(buy)}'")
    qty = float(buy.filled_qty or 0)
    if qty == 0:
        print(f"  -> {ticker}: buy ended as '{broker.status_of(buy)}' with no fill. Not a trade.")
        return None
    entry = float(buy.filled_avg_price)

    sell = broker.existing_order(exit_order_id(plan["session_date"], ticker), client)
    if sell is not None and broker.status_of(sell) not in broker.FINAL_STATUSES:
        raise NotReady(f"{ticker}: exit order is still '{broker.status_of(sell)}'")

    bars = _minute_bars(ticker, session_date, session_open, session_close)
    if sell is not None and float(sell.filled_qty or 0) > 0:
        exit_price, exit_status = float(sell.filled_avg_price), "filled"
    else:
        # No closing fill: value the position at the official close and say so.
        exit_price, exit_status = _price_at(bars, session_close), "missed"

    prices = {label: _price_at(bars, min(session_open + dt.timedelta(minutes=m), session_close))
              for label, m in TIMEFRAMES.items()}
    prices["EOD"] = exit_price

    pnl = {k: qty * (p - entry) for k, p in prices.items()}
    trade = {
        "ticker": ticker,
        "sentiment_score": target["sentiment_score"],
        "gap_percent": target.get("gap_percent"),
        "headlines": target.get("headlines", []),  # the evidence FinBERT scored
        "qty": qty,
        "allocation_usd": round(qty * entry, 2),
        "open_price": round(entry, 2),   # real opening fill
        "exit_price": round(exit_price, 2),
        "exit_status": exit_status,
        "performance_pct": {k: round((p - entry) / entry * 100, 2) for k, p in prices.items()},
        "pnl_usd": {k: round(v, 2) for k, v in pnl.items()},
    }
    return trade, pnl


def _evaluate_plan(plan: dict, starting_balance: float, client) -> dict:
    trades, curve = [], {k: starting_balance for k in CURVE_KEYS}
    # Every stock the model looked at but did not end up holding, with the reason.
    rejected = [
        {**{k: c.get(k) for k in ("ticker", "gap_percent", "sentiment_score", "headlines")},
         "reason": "sentiment below threshold"}
        for c in plan.get("candidates", []) if c.get("decision") == "rejected"
    ]
    for target in plan["targets"]:
        evidence = {k: target.get(k) for k in ("ticker", "gap_percent", "sentiment_score", "headlines")}
        if not target.get("order_id"):
            rejected.append({**evidence, "reason": target.get("skip_reason", "order could not be placed")})
            continue
        print(f"\nEvaluating {target['ticker']} (Sentiment: {target['sentiment_score']})...")
        scored = _evaluate_trade(target, plan, client)
        if scored is None:
            rejected.append({**evidence, "reason": "buy order was not filled"})
            continue
        trade, pnl = scored
        for key, dollars in pnl.items():
            curve[key] += dollars
            print(f"  -> {key}: {trade['performance_pct'][key]:+.2f}% (${dollars:+.2f})")
        if trade["exit_status"] == "missed":
            print(f"  -> ⚠️  no closing fill for {trade['ticker']}; valued at the official close")
        trades.append(trade)

    curve = {k: round(v, 2) for k, v in curve.items()}
    return {
        "date": plan["session_date"],
        "starting_balance": round(starting_balance, 2),
        "ending_balance": curve["EOD"],
        "portfolio_curve": curve,
        "invested_usd": round(sum(t["allocation_usd"] for t in trades), 2),
        "sentiment_threshold": plan.get("sentiment_threshold"),
        "trades": trades,
        "rejected": rejected,
    }


def _late_exit_price(ticker: str, session_date: str, session_close: dt.datetime, client, side: str = "long"):
    """
    Real price of the sell that closed a missed exit later (normally the next morning's
    safety net), or None while it has not filled. Tagged sells are found by id; sells from
    before tagging existed are found as the first filled SELL of that stock after the close.
    """
    tagged = broker.existing_order(late_exit_order_id(session_date, ticker), client)
    if tagged is not None:
        return float(tagged.filled_avg_price) if broker.status_of(tagged) == "filled" else None
    orders = client.get_orders(GetOrdersRequest(
        status=QueryOrderStatus.CLOSED, symbols=[ticker], side=OrderSide.BUY if side == "short" else OrderSide.SELL,
        after=session_close, until=session_close + dt.timedelta(days=5), direction="asc", limit=20))
    for o in sorted(orders, key=lambda o: o.submitted_at):
        if broker.status_of(o) == "filled" and o.client_order_id != exit_order_id(session_date, ticker):
            return float(o.filled_avg_price)
    return None


def revise_missed_exits(results: list, plans_dir=PLANS_DIR, client=None) -> bool:
    """
    Replace "missed" exits (valued at the official close) with the real price the shares were
    later sold at, mark them "late", and shift every later session's balances by the same
    dollar difference so the running fund value stays consistent. Returns True if anything changed.
    """
    client = client or broker.trading_client
    changed = False
    for i, entry in enumerate(results):
        missed = [t for t in entry.get("trades", []) if t.get("exit_status") == "missed"]
        plan = load_json(plan_path(entry["date"], plans_dir), default=None) if missed else None
        if not plan:
            continue
        close = dt.datetime.fromisoformat(plan["market_close"])
        delta = delta_net = 0.0
        for t in missed:
            sign = -1 if t.get("side") == "short" else 1
            price = _late_exit_price(t["ticker"], entry["date"], close, client, t.get("side", "long"))
            if price is None:
                continue
            d = sign * t["qty"] * (price - t["exit_price"])
            d_net = d - 0.0005 * t["qty"] * (price - t["exit_price"])   # the exit's 5 bps cost moves with its price
            print(f"Revising {entry['date']} {t['ticker']}: exit ${t['exit_price']:,.2f} (close estimate) "
                  f"-> ${price:,.2f} (real late sell), {d:+,.2f} USD")
            t["exit_price"] = round(price, 2)
            t["exit_status"] = "late"
            t["performance_pct"]["EOD"] = round(sign * (price - t["open_price"]) / t["open_price"] * 100, 2)
            if "pnl_usd" in t:
                t["pnl_usd"]["EOD"] = round(t["pnl_usd"]["EOD"] + d, 2)
            if "pnl_usd_net" in t:  # v2 trade: keep the net figures and the side totals consistent
                t["pnl_usd_net"]["EOD"] = round(t["pnl_usd_net"]["EOD"] + d_net, 2)
                t["performance_pct_net"]["EOD"] = round(t["pnl_usd_net"]["EOD"] / t["allocation_usd"] * 100, 2)
                side = entry.get("by_side", {}).get(t.get("side", "long"))
                if side:
                    side["pnl_usd"]["EOD"] = round(side["pnl_usd"]["EOD"] + d, 2)
                    side["pnl_usd_net"]["EOD"] = round(side["pnl_usd_net"]["EOD"] + d_net, 2)
                    if side["invested_usd"]:
                        side["return_pct"]["EOD"] = round(side["pnl_usd"]["EOD"] / side["invested_usd"] * 100, 2)
            delta += d
            delta_net += d_net
        if delta:
            entry["portfolio_curve"]["EOD"] = round(entry["portfolio_curve"]["EOD"] + delta, 2)
            entry["ending_balance"] = entry["portfolio_curve"]["EOD"]
            if "portfolio_curve_net" in entry:
                entry["portfolio_curve_net"]["EOD"] = round(entry["portfolio_curve_net"]["EOD"] + delta_net, 2)
                entry["ending_balance_net"] = entry["portfolio_curve_net"]["EOD"]
            for later in results[i + 1:]:
                later["starting_balance"] = round(later["starting_balance"] + delta, 2)
                later["ending_balance"] = round(later["ending_balance"] + delta, 2)
                later["portfolio_curve"] = {k: round(v + delta, 2) for k, v in later["portfolio_curve"].items()}
                if "portfolio_curve_net" in later:
                    later["portfolio_curve_net"] = {k: round(v + delta, 2) for k, v in later["portfolio_curve_net"].items()}
                    later["ending_balance_net"] = later["portfolio_curve_net"]["EOD"]
            changed = True
    return changed


def evaluate_pending(now=None, plans_dir=PLANS_DIR, results_file=RESULTS_FILE, client=None, bars_fn=None,
                     scores_dir=SCORES_DIR, candidates_dir=CANDIDATES_DIR, confirmation_file=CONFIRMATION_FILE,
                     prereg_file=PREREG_FILE, score_band_file=SCORE_BAND_FILE) -> int:
    """
    Evaluate every finished session that is not in results.json yet, oldest first.
    Stops at the first session that is not ready, so balances always carry
    forward in date order. Returns the number of sessions written.
    """
    now = now or now_ny()
    client = client or broker.trading_client
    results = load_json(results_file, default=[])
    if not isinstance(results, list):
        raise RuntimeError("results.json is not a list; refusing to overwrite it")
    done = {r.get("date") for r in results if isinstance(r, dict)}

    # First correct earlier sessions whose exit was missed and has since been sold for real,
    # so new sessions carry forward the corrected balance.
    if revise_missed_exits(results, plans_dir, client):
        save_json(results_file, results)

    written = 0
    for path in all_plan_paths(plans_dir):
        plan = load_json(path, default=None)
        if not plan or plan.get("session_date") in done:
            continue
        session_close = dt.datetime.fromisoformat(plan["market_close"])
        print(f"--- EVALUATING SESSION {plan['session_date']} (now {now:%Y-%m-%d %H:%M %Z}) ---")
        if now < session_close + SETTLE_AFTER_CLOSE:
            print(f"Session has not finished settling (closes {session_close:%H:%M %Z}). Try later.")
            break

        v2 = plan.get("strategy") == "v2"
        if v2:
            # v2 is a new segment: it starts from the reset paper account, then carries its own balance forward
            prev_v2 = results and results[-1].get("strategy") == "v2"
            starting_balance = float(results[-1]["ending_balance"]) if prev_v2 else INITIAL_CAPITAL
        else:
            starting_balance = float(results[-1]["ending_balance"]) if results else INITIAL_CAPITAL
        try:
            entry = evaluator_v2.evaluate_plan_v2(plan, starting_balance, client, bars_fn) if v2 \
                else _evaluate_plan(plan, starting_balance, client)
        except (NotReady, evaluator_v2.NotReady) as e:
            print(f"Not ready yet: {e}. Try later.")
            break
        results.append(entry)
        save_json(results_file, results)
        if v2:
            try:
                evaluator_v2.log_candidates(plan, entry, bars_fn, scores_dir, candidates_dir)
            except Exception as e:
                print(f"::warning::Could not write the would-have-traded log for {plan['session_date']}: {e}")
            conf = evaluator_v2.update_confirmation(results, candidates_dir, confirmation_file, prereg_file)
            print(f"Confirmation: {conf['sessions']} sessions, {conf['trades']} trades, IC {conf['ic']} (t {conf['ic_t']})")
            if dt.date.fromisoformat(plan["session_date"]).weekday() == 4 and os.path.exists(score_band_file):
                import alerts  # weekly drift check, on Fridays
                note = alerts.score_drift_notice(candidates_dir, load_json(score_band_file, default=None))
                if note:
                    print(f"::notice::{note}")
        written += 1
        print(f"\nRecorded {plan['session_date']}: ending balance ${entry['ending_balance']:,.2f}")

    if written == 0:
        print("No new sessions to record.")
    return written


if __name__ == "__main__":
    evaluate_pending()
