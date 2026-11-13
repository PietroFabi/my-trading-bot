"""
Evaluator: turn each finished session's real Alpaca fills into an entry in results.json.

- Entry price  = the buy's real fill (market-on-open auction).
- Exit price   = the sell's real fill (market-on-close auction). If the exit was
                 missed, the official close is used and the trade is flagged.
- Curve points = the real share count valued at Yahoo 1-minute prices.

The session date always comes from the plan file, never from the clock, so a
run that GitHub starts hours late still records the right day.
"""
import datetime as dt

import yfinance as yf

import broker
from market_session import now_ny
from storage import RESULTS_FILE, PLANS_DIR, all_plan_paths, exit_order_id, load_json, save_json

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


def evaluate_pending(now=None, plans_dir=PLANS_DIR, results_file=RESULTS_FILE, client=None) -> int:
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

        starting_balance = float(results[-1]["ending_balance"]) if results else INITIAL_CAPITAL
        try:
            entry = _evaluate_plan(plan, starting_balance, client)
        except NotReady as e:
            print(f"Not ready yet: {e}. Try later.")
            break
        results.append(entry)
        save_json(results_file, results)
        written += 1
        print(f"\nRecorded {plan['session_date']}: ending balance ${entry['ending_balance']:,.2f}")

    if written == 0:
        print("No new sessions to record.")
    return written


if __name__ == "__main__":
    evaluate_pending()
