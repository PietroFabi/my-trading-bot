import itertools
import os
import subprocess
import sys
import unittest
from types import SimpleNamespace

from alpaca.common.exceptions import APIError

import tests  # noqa: F401
import broker

DAY = "2026-11-20"


def not_found():
    return APIError('{"code": 40410000, "message": "not found"}',
                    http_error=SimpleNamespace(response=SimpleNamespace(status_code=404)))


class FakeClient:
    """Just enough of Alpaca's TradingClient: OTO legs are 'held' until the parent fills."""

    def __init__(self, legs_in_response=True):
        self.orders, self.by_cid, self.positions = {}, {}, {}
        self.submitted, self.cancelled, self.closed = [], [], []
        self.legs_in_response, self.on_cancel = legs_in_response, {}
        self._ids = itertools.count(1)

    # orders
    def submit_order(self, order_data):
        self.submitted.append(order_data)
        oid = f"o{next(self._ids)}"
        parent = SimpleNamespace(id=oid, client_order_id=order_data.client_order_id, symbol=order_data.symbol,
                                 qty=order_data.qty, side=order_data.side, status="accepted", filled_qty="0",
                                 filled_avg_price=None, order_class=order_data.order_class, legs=None, stop_price=None)
        self.orders[oid] = parent
        if order_data.client_order_id:
            self.by_cid[order_data.client_order_id] = parent
        if order_data.stop_loss is not None:
            lid = f"o{next(self._ids)}"
            leg = SimpleNamespace(id=lid, client_order_id=None, symbol=order_data.symbol, qty=order_data.qty,
                                  side="sell" if order_data.side.value == "buy" else "buy", status="held",
                                  filled_qty="0", filled_avg_price=None, stop_price=order_data.stop_loss.stop_price,
                                  legs=None)
            self.orders[lid] = leg
            parent._legs = [leg]
            parent.legs = [leg] if self.legs_in_response else None
        return parent

    def get_order_by_client_id(self, cid):
        if cid not in self.by_cid:
            raise not_found()
        return self.by_cid[cid]

    def get_order_by_id(self, oid, filter=None):
        o = self.orders[oid]
        if filter is not None and getattr(filter, "nested", False):
            return SimpleNamespace(**{**vars(o), "legs": getattr(o, "_legs", None)})
        return o

    def cancel_order_by_id(self, oid):
        self.cancelled.append(oid)
        hook = self.on_cancel.get(oid)
        if hook:
            hook()
        elif self.orders[oid].status not in broker.FINAL_STATUSES:
            self.orders[oid].status = "canceled"

    def get_orders(self, filter=None):
        wanted = set(filter.symbols or []) if filter is not None else None
        return [o for o in self.orders.values() if o.status not in broker.FINAL_STATUSES
                and (wanted is None or o.symbol in wanted)]

    # positions
    def get_open_position(self, symbol):
        if not self.positions.get(symbol):
            raise not_found()
        q = self.positions[symbol]
        return SimpleNamespace(symbol=symbol, qty=str(q), side="long" if q > 0 else "short")

    def get_all_positions(self):
        return [self.get_open_position(s) for s, q in self.positions.items() if q]

    def close_position(self, symbol):
        self.closed.append(symbol)

    # test helpers
    def fill(self, order, qty, price=100.0, position_sign=None):
        order.status, order.filled_qty, order.filled_avg_price = "filled", str(qty), str(price)
        for leg in getattr(order, "_legs", []):
            if leg.status == "held":
                leg.status = "new"
        if position_sign is not None:
            self.positions[order.symbol] = self.positions.get(order.symbol, 0) + position_sign * qty

    def leg_of(self, parent):
        return parent._legs[0]


def target(ticker, side="long", qty=10, stop=98.5):
    return {"ticker": ticker, "side": side, "qty": qty, "ref_price": 100.0, "stop_price": stop}


def plan(*targets):
    return {"session_date": DAY, "strategy": "v2", "targets": list(targets)}


class EntryTest(unittest.TestCase):
    def test_long_and_short_entries_carry_their_stop_as_an_oto_leg(self):
        c = FakeClient()
        p = plan(target("LLL"), target("SSS", side="short", stop=101.5))
        broker.place_entries(p, client=c)
        long_req, short_req = c.submitted
        self.assertEqual((long_req.side.value, long_req.order_class.value, long_req.stop_loss.stop_price), ("buy", "oto", 98.5))
        self.assertEqual((short_req.side.value, short_req.stop_loss.stop_price), ("sell", 101.5))
        self.assertEqual(long_req.client_order_id, f"{DAY}-LLL")
        self.assertEqual(long_req.time_in_force.value, "day")
        self.assertTrue(p["targets"][0]["order_id"])
        self.assertEqual(p["targets"][0]["stop_order_id"], c.leg_of(c.orders[p["targets"][0]["order_id"]]).id)

    def test_no_stop_means_a_plain_market_order(self):
        c = FakeClient()
        broker.place_entries(plan(target("LLL", stop=None)), client=c)
        self.assertIsNone(c.submitted[0].stop_loss)
        self.assertNotEqual(getattr(c.submitted[0].order_class, "value", None), "oto")

    def test_a_rerun_finds_the_existing_entry_instead_of_trading_again(self):
        c = FakeClient()
        broker.place_entries(plan(target("LLL")), client=c)
        broker.place_entries(plan(target("LLL")), client=c)
        self.assertEqual(len(c.submitted), 1)

    def test_a_stock_whose_leftover_is_being_closed_is_skipped(self):
        c = FakeClient()
        p = plan(target("LLL"))
        broker.place_entries(p, client=c, skip_tickers={"LLL"})
        self.assertEqual((c.submitted, p["targets"][0]["skip_reason"]), ([], "leftover position being closed"))


class ExitTest(unittest.TestCase):
    def entered(self, *targets, legs_in_response=True, fill=True):
        c = FakeClient(legs_in_response)
        p = plan(*targets)
        broker.place_entries(p, client=c)
        for t in p["targets"]:
            if fill:
                c.fill(c.orders[t["order_id"]], t["qty"], position_sign=1 if t["side"] == "long" else -1)
        return c, p

    def exits_sent(self, c):
        return [(o.side.value, o.qty, o.client_order_id) for o in c.submitted if o.client_order_id.endswith("-exit")]

    def test_untouched_stop_is_cancelled_then_the_position_is_closed(self):
        c, p = self.entered(target("LLL"), target("SSS", side="short"))
        self.close_fills(c)
        self.assertTrue(broker.place_exit_orders(p, client=c, sleep=lambda s: None))
        for t in p["targets"]:
            self.assertIn(t["stop_order_id"], c.cancelled)
        self.assertEqual(self.exits_sent(c), [("sell", 10, f"{DAY}-LLL-exit"), ("buy", 10, f"{DAY}-SSS-exit")])

    def test_a_stop_that_already_filled_means_no_exit_order(self):
        c, p = self.entered(target("LLL"))
        leg = c.orders[p["targets"][0]["stop_order_id"]]
        c.fill(leg, 10, price=98.5)
        c.positions["LLL"] = 0
        self.assertTrue(broker.place_exit_orders(p, client=c, sleep=lambda s: None))
        self.assertEqual(self.exits_sent(c), [])

    def test_a_stop_filling_during_the_cancel_never_flips_the_position(self):
        c, p = self.entered(target("SSS", side="short"))
        leg = c.orders[p["targets"][0]["stop_order_id"]]

        def stop_fills_first():
            c.fill(leg, 10, price=101.5)
            c.positions["SSS"] = 0
        c.on_cancel[leg.id] = stop_fills_first
        self.assertTrue(broker.place_exit_orders(p, client=c, sleep=lambda s: None))
        self.assertEqual(self.exits_sent(c), [])

    def test_a_filled_stop_is_trusted_even_if_the_position_still_looks_open(self):
        # Alpaca's position endpoint can lag the fill: only the stop check prevents a flip here.
        c, p = self.entered(target("SSS", side="short"))
        c.fill(c.orders[p["targets"][0]["stop_order_id"]], 10, price=101.5)      # stale: positions still -10
        self.close_fills(c)
        broker.place_exit_orders(p, client=c, sleep=lambda s: None)
        self.assertEqual(self.exits_sent(c), [])

    def test_a_position_closed_elsewhere_is_not_closed_again(self):
        # The stop is cancelled normally, but the shares are gone (e.g. closed by hand): only the re-read prevents a flip.
        c, p = self.entered(target("LLL"))
        c.positions["LLL"] = 0
        self.close_fills(c)
        broker.place_exit_orders(p, client=c, sleep=lambda s: None)
        self.assertEqual(self.exits_sent(c), [])

    def test_a_partial_entry_fill_exits_only_the_filled_shares(self):
        c, p = self.entered(target("LLL", qty=10), fill=False)
        c.fill(c.orders[p["targets"][0]["order_id"]], 4, position_sign=1)
        c.orders[p["targets"][0]["order_id"]].status = "partially_filled"
        self.close_fills(c)
        broker.place_exit_orders(p, client=c, sleep=lambda s: None)
        self.assertEqual(self.exits_sent(c), [("sell", 4, f"{DAY}-LLL-exit")])

    def test_the_stop_leg_is_found_by_a_nested_query_when_the_entry_response_had_no_legs(self):
        c, p = self.entered(target("LLL"), legs_in_response=False)
        self.assertIsNone(p["targets"][0].get("stop_order_id"))
        self.close_fills(c)
        broker.place_exit_orders(p, client=c, sleep=lambda s: None)
        self.assertIn(c.leg_of(c.orders[p["targets"][0]["order_id"]]).id, c.cancelled)
        self.assertEqual(self.exits_sent(c), [("sell", 10, f"{DAY}-LLL-exit")])

    def test_a_rerun_finds_the_existing_exit(self):
        c, p = self.entered(target("LLL"))
        self.close_fills(c)
        broker.place_exit_orders(p, client=c, sleep=lambda s: None)
        broker.place_exit_orders(p, client=c, sleep=lambda s: None)
        self.assertEqual(len(self.exits_sent(c)), 1)

    def test_a_v1_plan_without_sides_still_exits_as_long(self):
        c = FakeClient()
        entry = c.submit_order(broker.MarketOrderRequest(symbol="OLD", qty=5, side=broker.OrderSide.BUY,
                                                         time_in_force=broker.TimeInForce.DAY,
                                                         client_order_id=f"{DAY}-OLD"))
        c.fill(entry, 5, position_sign=1)
        p = {"session_date": DAY, "targets": [{"ticker": "OLD", "order_id": entry.id, "qty": 5}]}
        self.close_fills(c)
        self.assertTrue(broker.place_exit_orders(p, client=c, sleep=lambda s: None))
        self.assertEqual(self.exits_sent(c), [("sell", 5, f"{DAY}-OLD-exit")])

    def test_unsold_positions_respect_the_side(self):
        c, p = self.entered(target("LLL"), target("SSS", side="short"))
        self.assertEqual(broker.unsold_positions(p, client=c), ["LLL", "SSS"])
        c.positions["SSS"] = 0
        self.assertEqual(broker.unsold_positions(p, client=c), ["LLL"])

    @staticmethod
    def close_fills(c):
        """Exit orders fill as soon as they are submitted."""
        original = c.submit_order

        def submit(order_data):
            o = original(order_data)
            if order_data.client_order_id and order_data.client_order_id.endswith("-exit"):
                sign = -1 if order_data.side.value == "sell" else 1
                c.fill(o, order_data.qty, position_sign=sign)
            return o
        c.submit_order = submit


class LeftoverTest(unittest.TestCase):
    def test_a_leftover_short_has_its_orders_cancelled_then_is_bought_back_with_a_late_exit_id(self):
        import json
        import tempfile
        c = FakeClient()
        p = plan(target("SSS", side="short"))
        broker.place_entries(p, client=c)
        c.fill(c.orders[p["targets"][0]["order_id"]], 10, position_sign=-1)
        stop_leg = p["targets"][0]["stop_order_id"]
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, f"{DAY}.json"), "w") as f:
                json.dump(p, f)
            closed, failed = broker.close_leftover_positions(client=c, plans_dir=d)
        self.assertEqual((closed, failed), (["SSS"], []))
        self.assertIn(stop_leg, c.cancelled)
        last = c.submitted[-1]
        self.assertEqual((last.side.value, last.qty, last.client_order_id), ("buy", 10, f"{DAY}-SSS-late-exit"))
        self.assertLess(c.cancelled.index(stop_leg), len(c.cancelled))


class LightImportTest(unittest.TestCase):
    def test_the_exit_job_loads_no_heavy_libraries(self):
        # pandas/numpy are allowed: alpaca-py itself imports them (they are its own pip dependencies).
        heavy = ("yfinance", "torch", "transformers", "features", "sentiment", "model", "strategy", "news", "marketdata")
        src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
        code = ("import sys; sys.path.insert(0, %r); import exits; "
                "print(sorted(m for m in %r if m in sys.modules))" % (src, heavy))
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
        self.assertEqual(out, "[]")


if __name__ == "__main__":
    unittest.main()
