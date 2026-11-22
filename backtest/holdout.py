"""
Step 10b: the one-time hold-out (2026-10-09 -> 2026-11-20), evaluated for both pre-registered hypotheses.

Refuses to run unless data/model/frozen_model.json and data/preregistration.json are committed and unchanged, and
the model file's sha256 matches the pre-registration. Refuses a second run once data/backtest/holdout.json exists.
Run from the repo root:  python -m backtest.holdout
"""
import datetime as dt
import json
import os
import pickle
import subprocess

import numpy as np
import pandas as pd

import backtest  # noqa: F401  (src/ on the path)
import model
import strategy
from backtest import explore as ex
from backtest.freeze import HOLDOUT_END, MODEL_FILE, PREREG_FILE, sha256_file
from storage import json_safe

OUT = os.path.join("data", "backtest", "holdout.json")
T_MIN = 2.24


class NotPreregistered(Exception):
    pass


class AlreadyRun(Exception):
    pass


def _git(repo: str, *args) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True)


def check_preregistered(repo: str, model_path: str, prereg_path: str, out_path: str) -> None:
    if os.path.exists(out_path):
        raise AlreadyRun(f"{out_path} exists: the hold-out has already been evaluated")
    for path in (model_path, prereg_path):
        rel = os.path.relpath(path, repo)
        if _git(repo, "ls-files", "--error-unmatch", rel).returncode != 0:
            raise NotPreregistered(f"{rel} is not committed")
        if _git(repo, "status", "--porcelain", "--", rel).stdout.strip():
            raise NotPreregistered(f"{rel} has changes since its commit")
    with open(prereg_path) as f:
        expected = json.load(f)["frozen_model_sha256"]
    if sha256_file(model_path) != expected:
        raise NotPreregistered("the frozen model does not match the hash in the pre-registration")


def passes(r: dict, t_min: float = T_MIN) -> bool:
    return r["ic"] > 0 and r["ic_t"] > t_min and r["eod_net_mean"] > 0


def decide(a: dict, fade: dict, t_min: float = T_MIN) -> str:
    if passes(a, t_min):
        return "A"
    if passes(fade, t_min):
        return "fade"
    return "A"


def _evaluate(name: str, rows, trades: list, days: list, score) -> dict:
    ic, ic_t = model.information_coefficient(score, rows.target, rows.session_date)
    net = ex.trade_stats([t["ret_net"]["EOD"] for t in trades if t["entered"]])
    return {"name": name, "ic": ic, "ic_t": ic_t, "eod_net_mean": net["mean"], "eod_net": net,
            "eod_gross": ex.trade_stats([t["ret_gross"]["EOD"] for t in trades if t["entered"]]),
            "by_side_and_horizon": ex.side_horizon_table(trades, days),
            "curve_net": ex.curve_stats(ex.daily_returns(trades, days)),
            "trades_per_day": sum(t["entered"] for t in trades) / len(days)}


def evaluate(first: dt.date | None = None, last: dt.date | None = None) -> dict:
    """Defaults to the hold-out; other dates only for a smoke test on exploration sessions."""
    from market_session import Session
    with open(MODEL_FILE) as f:
        frozen = json.load(f)
    with open(ex.DATASET, "rb") as f:
        d = pickle.load(f)
    sessions = {s["date"]: Session(s["date"], s["open"], s["close"]) for s in d["sessions"]}
    first, last = first or ex.HOLDOUT_START, last or dt.date.fromisoformat(HOLDOUT_END)
    f = d["features"]
    f = f[(f.session_date >= first) & (f.session_date <= last)]
    gated = f[f.gate_long | f.gate_short]
    gated = model.add_model_columns(ex.add_targets(gated, d["outcome_bars"], sessions))
    flags = ex.shortable_flags()
    gated["shortable"] = [flags.get(s, True) for s in gated.symbol]
    days = sorted(f.session_date.unique())

    a = frozen["strategies"]["A"]
    if a["exclude_price_move"]:
        raise NotImplementedError("the frozen A uses news without price-move headlines; recompute them here first")
    ridge = model.RidgeModel.from_dict(a["ridge"])
    gated["pred"] = ridge.predict(model.feature_matrix(gated))
    a_trades = ex.simulate_rule(gated, d["outcome_bars"], sessions, a["entry"], a["stop_k"], None, {},
                                select=lambda rows: strategy.select_candidates(rows, a["threshold_return"]))
    fade_trades = ex.simulate_rule(gated, d["outcome_bars"], sessions, "open", None, None, {},
                                   select=strategy.fade_candidates)
    scored = gated[gated.target.notna()]
    res_a = _evaluate("A", scored, a_trades, days, scored.pred)
    res_fade = _evaluate("fade", scored, fade_trades, days, -scored.gap_atr)

    import marketdata
    spy = marketdata.daily_bars(["SPY"], first - dt.timedelta(days=10), last, cache_dir=os.path.join(ex.CACHE, "bars"),
                                fallback=False).xs("SPY", level=0)
    spy.index = [pd.Timestamp(i).date() for i in spy.index]
    oc = (spy["close"] / spy["open"] - 1).reindex(days)
    prev_close = spy["close"][spy.index < first].iloc[-1]
    return {
        "evaluated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "holdout": {"first": str(days[0]), "last": str(days[-1]), "sessions": len(days), "gated_rows": len(gated),
                    "rows_with_target": len(scored)},
        "pass_rule": f"IC > 0 with t > {T_MIN} AND mean net return per trade at the close > 0",
        "A": {**res_a, "passes": passes(res_a)},
        "fade": {**res_fade, "passes": passes(res_fade)},
        "goes_live": decide(res_a, res_fade),
        "benchmarks": {"spy_open_to_close": {**ex.curve_stats(oc.fillna(0)), "mean_daily": float(np.nanmean(oc))},
                       "spy_buy_and_hold": float(spy["close"].reindex(days).iloc[-1] / prev_close - 1)},
    }


if __name__ == "__main__":
    check_preregistered(".", MODEL_FILE, PREREG_FILE, OUT)
    result = evaluate()
    with open(OUT, "w") as f:
        json.dump(json_safe(result), f, indent=2, default=str, allow_nan=False)
    print(json.dumps({k: result[k] for k in ("holdout", "goes_live")}, indent=2, default=str))
    for name in ("A", "fade"):
        r = result[name]
        print(f"{name}: IC {r['ic']:.4f} (t {r['ic_t']:.2f}); net/trade {r['eod_net_mean']:.4%} over "
              f"{r['eod_net']['n']} trades (t {r['eod_net']['t']:.2f}); passes: {r['passes']}")
