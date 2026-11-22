import csv
import datetime as dt
import json
import os
import tempfile
import unittest
from zoneinfo import ZoneInfo

import pandas as pd

import tests  # noqa: F401
import broker
import evaluator
import evaluator_v2 as ev2
from tests.test_broker_v2 import FakeClient

NY = ZoneInfo("America/New_York")
FRI, MON = dt.date(2026, 11, 20), dt.date(2026, 11, 23)


def at(day, hh, mm, ss=0):
    return dt.datetime.combine(day, dt.time(hh, mm, ss), NY)


def bars_for(day, prices, close=dt.time(16), changes=None):
    """{symbol: flat price}; changes = {symbol: {minute: close price from that minute on}}."""
    rows = []
    minutes = int((dt.datetime.combine(day, close) - dt.datetime.combine(day, dt.time(9, 30))).total_seconds() // 60)
    for sym, p in prices.items():
        level = p
        for m in range(minutes):
            level = (changes or {}).get(sym, {}).get(m, level)
            rows.append({"symbol": sym, "timestamp": at(day, 9, 30) + dt.timedelta(minutes=m), "open": level,
                         "high": level, "low": level, "close": level, "volume": 100.0})
    frame = pd.DataFrame(rows).set_index(["symbol", "timestamp"])
    return lambda symbols, start, end: frame[frame.index.get_level_values(0).isin(symbols)]


def target(ticker, side="long", qty=10, stop=98.5):
    return {"ticker": ticker, "side": side, "qty": qty, "ref_price": 100.0, "stop_price": stop, "score": 0.004,
            "sector": "Tech", "features": {"gap_pct": 1.5 if side == "long" else -1.5, "news_today": 0.6},
            "headlines": [{"headline": f"{ticker} news", "url": "u", "source": "fiscal.ai",
                           "created_at": "2026-11-20T07:00:00-05:00", "score": 0.6}]}


def v2_plan(day, *targets, close=dt.time(16), candidates=()):
    return {"session_date": day.isoformat(), "market_open": at(day, 9, 30).isoformat(),
            "market_close": dt.datetime.combine(day, close, NY).isoformat(), "strategy": "v2", "strategy_name": "A",
            "model_sha256": "a" * 64, "score_threshold": 0.003, "stop_k": 0.75, "equity": 100_000.0,
            "candidates": list(candidates), "targets": list(targets)}


def trade_day(plan, entry_px=100.0, exit_px=None, stop_fill=None):
    """Enter every target at entry_px; then either the stop fills (price, time) or the exit job closes at exit_px."""
    c = FakeClient()
    broker.place_entries(plan, client=c)
    for t in plan["targets"]:
        sign = 1 if t["side"] == "long" else -1
        c.fill(c.orders[t["order_id"]], t["qty"], price=entry_px, position_sign=sign)
        if stop_fill:
            leg = c.orders[t["stop_order_id"]]
            c.fill(leg, t["qty"], price=stop_fill[0], position_sign=-sign)
            leg.filled_at = stop_fill[1]
    if exit_px is not None:
        from tests.test_broker_v2 import ExitTest
        ExitTest.close_fills(c)
        original = c.fill

        def fill_at_exit(order, qty, price=100.0, position_sign=None):
            original(order, qty, price=exit_px, position_sign=position_sign)
        c.fill = fill_at_exit
        broker.place_exit_orders(plan, client=c, sleep=lambda s: None)
        c.fill = original
    return c


class TradeTest(unittest.TestCase):
    def evaluate(self, plan, client, prices, close=dt.time(16), changes=None, start=100_000.0):
        return ev2.evaluate_plan_v2(plan, start, client, bars_for(FRI, prices, close, changes))

    def test_a_long_up_one_percent_keeps_every_v1_field_and_adds_net_and_side(self):
        p = v2_plan(FRI, target("LLL"))
        c = trade_day(p, exit_px=101.0)
        e = self.evaluate(p, c, {"LLL": 101.0})
        t = e["trades"][0]
        for key in ("ticker", "sentiment_score", "gap_percent", "headlines", "qty", "allocation_usd", "open_price",
                    "exit_price", "exit_status", "performance_pct", "pnl_usd"):
            self.assertIn(key, t)
        self.assertEqual((t["side"], t["exit_status"], t["exit_reason"]), ("long", "filled", "close"))
        self.assertEqual((t["open_price"], t["exit_price"], t["allocation_usd"]), (100.0, 101.0, 1000.0))
        self.assertEqual((t["performance_pct"]["EOD"], t["pnl_usd"]["EOD"]), (1.0, 10.0))
        self.assertAlmostEqual(t["pnl_usd_net"]["EOD"], 10 - 0.0005 * 10 * (100 + 101), places=2)
        self.assertEqual(t["performance_pct_net"]["EOD"], 0.9)
        self.assertEqual((t["sentiment_score"], t["gap_percent"]), (0.6, 1.5))
        self.assertEqual(t["headlines"][0]["headline"], "LLL news")

    def test_a_short_whose_price_falls_two_percent_makes_money(self):
        p = v2_plan(FRI, target("SSS", side="short", stop=101.5))
        c = trade_day(p, exit_px=98.0)
        t = self.evaluate(p, c, {"SSS": 98.0})["trades"][0]
        self.assertEqual((t["performance_pct"]["EOD"], t["pnl_usd"]["EOD"]), (2.0, 20.0))

    def test_a_stop_at_10_05_fixes_every_later_horizon_at_the_stop_price(self):
        p = v2_plan(FRI, target("LLL"))
        c = trade_day(p, stop_fill=(98.5, at(FRI, 10, 5, 30)))
        t = self.evaluate(p, c, {"LLL": 100.0}, changes={"LLL": {36: 103.0}})["trades"][0]
        self.assertEqual((t["exit_reason"], t["exit_status"], t["exit_price"]), ("stop", "filled", 98.5))
        self.assertEqual(t["performance_pct"]["30_min"], 0.0)                 # 10:00, before the stop
        for h in ("1_hour", "5_hour", "EOD"):
            self.assertEqual(t["performance_pct"][h], -1.5)

    def test_by_side_adds_up_to_the_portfolio(self):
        p = v2_plan(FRI, target("LLL"), target("SSS", side="short", stop=101.5))
        c = trade_day(p, exit_px=101.0)
        e = self.evaluate(p, c, {"LLL": 101.0, "SSS": 101.0})
        for k in evaluator.CURVE_KEYS[1:]:
            total = e["by_side"]["long"]["pnl_usd"][k] + e["by_side"]["short"]["pnl_usd"][k]
            self.assertAlmostEqual(total, e["portfolio_curve"][k] - e["starting_balance"], places=2)
        self.assertAlmostEqual(e["by_side"]["long"]["invested_usd"] + e["by_side"]["short"]["invested_usd"],
                               e["invested_usd"], places=2)
        self.assertEqual((e["strategy"], e["costs_bps"]), ("v2", 5))
        self.assertEqual(e["score_thresholds"], {"long": 0.003, "short": -0.003})
        self.assertLess(e["portfolio_curve_net"]["EOD"], e["portfolio_curve"]["EOD"])

    def test_on_a_half_day_horizons_past_the_close_use_the_closing_price(self):
        p = v2_plan(FRI, target("LLL"), close=dt.time(13))
        c = trade_day(p, exit_px=102.0)
        t = self.evaluate(p, c, {"LLL": 100.0}, close=dt.time(13), changes={"LLL": {209: 102.0}})["trades"][0]
        self.assertEqual(t["performance_pct"]["5_hour"], 2.0)

    def test_rejected_candidates_keep_their_reason(self):
        p = v2_plan(FRI, target("LLL"), candidates=[{"ticker": "LLL", "gate": "long", "score": 0.004, "decision": "selected"},
                                                    {"ticker": "WEAK", "gate": "short", "score": -0.001,
                                                     "decision": "below threshold", "gap_percent": -0.4,
                                                     "sentiment_score": -0.3}])
        c = trade_day(p, exit_px=101.0)
        e = self.evaluate(p, c, {"LLL": 101.0})
        self.assertEqual([(r["ticker"], r["reason"], r["gap_percent"], r["sentiment_score"]) for r in e["rejected"]],
                         [("WEAK", "below threshold", -0.4, -0.3)])

    def test_the_entry_is_valid_json_without_nan(self):
        p = v2_plan(FRI, target("LLL"))
        c = trade_day(p, exit_px=101.0)
        json.dumps(self.evaluate(p, c, {"LLL": 101.0}), allow_nan=False)


class LateExitTest(unittest.TestCase):
    def test_a_missed_short_exit_is_revised_with_the_right_sign(self):
        p = v2_plan(FRI, target("SSS", side="short", stop=101.5))
        c = trade_day(p)                                                         # no exit: missed
        e = ev2.evaluate_plan_v2(p, 100_000.0, c, bars_for(FRI, {"SSS": 98.0}))
        t = e["trades"][0]
        self.assertEqual((t["exit_status"], t["pnl_usd"]["EOD"]), ("missed", 20.0))
        late = c.submit_order(broker.MarketOrderRequest(symbol="SSS", qty=10, side=broker.OrderSide.BUY,
                                                        time_in_force=broker.TimeInForce.DAY,
                                                        client_order_id=f"{FRI}-SSS-late-exit"))
        c.fill(late, 10, price=97.0)
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, f"{FRI}.json"), "w") as f:
                json.dump(p, f)
            results = [e]
            self.assertTrue(evaluator.revise_missed_exits(results, plans_dir=d, client=c))
        t = results[0]["trades"][0]
        self.assertEqual((t["exit_status"], t["exit_price"], t["pnl_usd"]["EOD"], t["performance_pct"]["EOD"]),
                         ("late", 97.0, 30.0, 3.0))
        self.assertEqual(results[0]["portfolio_curve"]["EOD"], 100_030.0)
        self.assertEqual(results[0]["by_side"]["short"]["pnl_usd"]["EOD"], 30.0)


class PendingTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        d = self.dir.name
        self.plans, self.scores, self.cands = (os.path.join(d, x) for x in ("plans", "scores", "candidates"))
        for x in (self.plans, self.scores):
            os.makedirs(x)
        self.results, self.conf = os.path.join(d, "results.json"), os.path.join(d, "confirmation.json")
        self.prereg = os.path.join(d, "preregistration.json")
        with open(self.prereg, "w") as f:
            json.dump({"live_confirmation": {"length": "the later of 60 sessions and 150 trades"}}, f)
        with open(self.results, "w") as f:
            json.dump([{"date": "2026-11-19", "starting_balance": 100_000.0, "ending_balance": 99_000.0,
                        "portfolio_curve": {k: 99_000.0 for k in evaluator.CURVE_KEYS}, "trades": [], "rejected": []}], f)

    def tearDown(self):
        self.dir.cleanup()

    def write_day(self, day, plan):
        with open(os.path.join(self.plans, f"{day}.json"), "w") as f:
            json.dump(plan, f)
        rows = [("LLL", 1, 0, 0.004, "selected", 1.0), ("AAA", 1, 0, 0.002, "below threshold", 1.0),
                ("BBB", 0, 1, -0.001, "below threshold", 1.5), ("CCC", 0, 1, -0.002, "below threshold", 1.0)]
        with open(os.path.join(self.scores, f"{day}.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["symbol", "sector", "sector_etf", "beta", "gate_long", "gate_short", "score", "decision",
                        "gap_atr", "news_today", "surprise", "liquid"])
            w.writerows([(s, "Tech", "XLK", b, gl, gs, sc, dec, 0.5, 0.3, 0.1, True) for s, gl, gs, sc, dec, b in rows])

    def run_pending(self, client, bars, now):
        return evaluator.evaluate_pending(now=now, plans_dir=self.plans, results_file=self.results, client=client,
                                          bars_fn=bars, scores_dir=self.scores, candidates_dir=self.cands,
                                          confirmation_file=self.conf, prereg_file=self.prereg)

    def test_v2_starts_a_new_segment_at_100k_then_carries_forward_and_logs_every_candidate(self):
        prices = {"LLL": 101.0, "AAA": 102.0, "BBB": 50.5, "CCC": 99.0, "XLK": 50.5}
        p1 = v2_plan(FRI, target("LLL"))
        c = trade_day(p1, exit_px=101.0)
        self.write_day(FRI, p1)
        self.assertEqual(self.run_pending(c, bars_for(FRI, prices), at(FRI, 17, 0)), 1)
        with open(self.results) as f:
            results = json.load(f)
        self.assertEqual((results[-1]["strategy"], results[-1]["starting_balance"]), ("v2", 100_000.0))

        with open(os.path.join(self.cands, f"{FRI}.csv")) as f:
            cands = {r["symbol"]: r for r in csv.DictReader(f)}
        self.assertEqual(set(cands), {"LLL", "AAA", "BBB", "CCC"})
        # every flat-open stock: open 100-ish -> close as given, ETF 50.5 flat -> 0 return
        self.assertAlmostEqual(float(cands["AAA"]["target"]), 0.0)                # flat bars: open == close
        self.assertEqual((cands["LLL"]["traded"], cands["AAA"]["traded"]), ("True", "False"))

        with open(self.conf) as f:
            conf = json.load(f)
        self.assertEqual((conf["sessions"], conf["trades"]), (1, 1))
        self.assertEqual(conf["required"], {"sessions": 60, "trades": 150, "ic_t": 2.0})
        self.assertEqual(conf["status"], "in progress")

        p2 = v2_plan(MON, target("LLL"))
        c2 = trade_day(p2, exit_px=101.0)
        c2.orders.update(c.orders)
        self.write_day(MON, p2)
        bars_mon = bars_for(MON, prices)
        self.assertEqual(self.run_pending(c2, bars_mon, at(MON, 17, 0)), 1)
        with open(self.results) as f:
            results = json.load(f)
        self.assertEqual(results[-1]["starting_balance"], results[-2]["ending_balance"])

    def test_v1_plans_still_go_through_the_v1_evaluator(self):
        called = []
        original = evaluator._evaluate_plan
        evaluator._evaluate_plan = lambda plan, start, client: called.append(plan["session_date"]) or {
            "date": plan["session_date"], "starting_balance": start, "ending_balance": start,
            "portfolio_curve": {k: start for k in evaluator.CURVE_KEYS}, "trades": [], "rejected": []}
        try:
            with open(os.path.join(self.plans, f"{FRI}.json"), "w") as f:
                json.dump({"session_date": FRI.isoformat(), "market_open": at(FRI, 9, 30).isoformat(),
                           "market_close": at(FRI, 16, 0).isoformat(), "targets": []}, f)
            self.run_pending(FakeClient(), None, at(FRI, 17, 0))
        finally:
            evaluator._evaluate_plan = original
        self.assertEqual(called, [FRI.isoformat()])
        self.assertFalse(os.path.exists(self.conf))


if __name__ == "__main__":
    unittest.main()
