"""
Step 10a: freeze. Writes data/model/frozen_model.json (everything the live bot needs to trade, nothing to tune) and
data/preregistration.json (the hypotheses, the one-time hold-out test, the decision rule and the live confirmation
criteria, plus the sha256 of the model file). The user commits both before the hold-out is opened.

Two hypotheses (user decision, option B):
  A     the pre-registered strategy: ridge model, momentum gates, best rule from exploration (no edge seen there;
        frozen anyway, as agreed before seeing data).
  fade  one fixed, untuned rule FOUND during exploration: trade against each gated stock's gap.
Run from the repo root:  python -m backtest.freeze
"""
import datetime as dt
import hashlib
import json
import os
import pickle

import backtest  # noqa: F401  (src/ on the path)
import strategy
from backtest import simulate as sim

MODEL_FILE = os.path.join("data", "model", "frozen_model.json")
PREREG_FILE = os.path.join("data", "preregistration.json")
EXPLORATION_FILE = os.path.join("data", "backtest", "exploration.json")
HOLDOUT_END = "2026-11-20"
HOLDOUT_T_MIN = 2.24              # two hypotheses on one hold-out: Bonferroni on the usual t > 2
LIVE_T_MIN = 2.0                  # live confirmation tests one strategy


def build_model(exploration: dict) -> dict:
    import sentiment
    fm, s2 = exploration["final_model"], exploration["stage2_choice"]
    return {
        "version": 2,
        "data_cutoff": exploration["window"]["last"],
        "finbert": {"model": sentiment.MODEL_ID, "revision": sentiment.MODEL_REVISION},
        "scan_before_open_min": 35,
        "selection": {"max_trades": strategy.MAX_TRADES, "max_per_sector": strategy.MAX_PER_SECTOR,
                      "shorts_need": "shortable and easy to borrow"},
        "sizing": {"risk_per_atr": sim.RISK_PER_ATR, "max_stock": sim.MAX_STOCK, "max_day": sim.MAX_DAY},
        "costs": {"per_side": sim.COST_PER_SIDE, "borrow_intraday": 0.0},
        "strategies": {
            "A": {"kind": "ridge", "features": fm["features"], "exclude_price_move": fm["exclude_price_move"],
                  "ridge": {k: fm[k] for k in ("lam", "clip_y", "clip_z", "mean", "std", "coef", "intercept")},
                  "threshold_multiple": s2["threshold"], "in_sample_pred_std": fm["in_sample_pred_std"],
                  "threshold_return": fm["threshold_return"], "entry": s2["entry"], "stop_k": s2["stop"],
                  "orb_min_rvol5": sim.ORB_MIN_RVOL5, "exit": "close"},
            "fade": {"kind": "gap_fade", "score": "-gap_atr", "side": "opposite of the gate", "threshold": None,
                     "entry": "open", "stop_k": None, "exit": "close"},
        },
    }


def build_preregistration(exploration: dict, fade_in_sample: dict, model_sha256: str, frozen_on: str) -> dict:
    w = exploration["window"]
    return {
        "frozen_on": frozen_on,
        "frozen_model_file": MODEL_FILE.replace(os.sep, "/"),
        "frozen_model_sha256": model_sha256,
        "hypotheses": [
            {"name": "A", "role": "primary (pre-registered before any data was seen)",
             "rule": "ridge model on the gated stocks (gap and news agree), trade in the gate's direction when the "
                     "prediction passes the frozen threshold; entry/stop/threshold chosen in exploration",
             "exploration_out_of_fold": {"ic": exploration["stage1_choice"]["ic"],
                                         "ic_t": exploration["stage1_choice"]["t"],
                                         "trades": exploration["stage2_choice"]["n"],
                                         "mean_net_per_trade": exploration["stage2_choice"]["mean"],
                                         "t": exploration["stage2_choice"]["t"]}},
            {"name": "fade", "role": "secondary, FOUND during exploration (gated gaps reversed intraday); fixed "
                                     "before the hold-out was read, nothing tuned",
             "rule": "the same gated stocks, traded AGAINST the gap (gap up -> short, gap down -> long), strongest "
                     "|gap_atr| first, max 10 a day and 3 per sector, entry at the open, no stop, exit at the close",
             "exploration_in_sample": fade_in_sample},
        ],
        "variants_tried": exploration["variants_tried"]["total"] + 1,
        "holdout": {
            "first": w["holdout_start"], "last": HOLDOUT_END, "evaluated": "once, by backtest/holdout.py",
            "pass": {"ic_min": 0.0, "t_min": HOLDOUT_T_MIN, "mean_net_per_trade_min": 0.0,
                     "definition": "IC > 0 with t > 2.24 AND mean net return per trade at the close > 0"},
        },
        "decision_rule": "If A passes the hold-out, A goes live. Otherwise, if the fade passes, the fade goes live. "
                         "If neither passes, A goes live (agreed before seeing data: freeze the best candidate anyway).",
        "live_confirmation": {
            "length": "the later of 60 sessions and 150 trades",
            "primary": f"IC > 0 with t > {LIVE_T_MIN} over the confirmation period",
            "secondary": "mean net return per trade at the close > 0",
            "model": "completely frozen (no retraining)",
            "claimed": "only side x entry mode, exit at the close; other horizons are exploratory",
        },
        "costs": "5 bps per side in every net figure; borrow fee 0 for intraday shorts (assumption)",
    }


def sha256_file(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def write_freeze(exploration: dict, fade_in_sample: dict, model_path: str = MODEL_FILE,
                 prereg_path: str = PREREG_FILE, frozen_on: str | None = None) -> dict:
    for p in (model_path, prereg_path):
        os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(model_path, "w", newline="\n") as f:
        json.dump(build_model(exploration), f, indent=2, sort_keys=True)
        f.write("\n")
    prereg = build_preregistration(exploration, fade_in_sample, sha256_file(model_path),
                                   frozen_on or dt.date.today().isoformat())
    with open(prereg_path, "w", newline="\n") as f:
        json.dump(prereg, f, indent=2)
        f.write("\n")
    return prereg


def fade_in_sample() -> dict:
    """The fade on the out-of-fold exploration sessions where the pattern was noticed (in-sample, descriptive)."""
    import model
    from backtest import explore as ex
    from market_session import Session
    with open(ex.DATASET, "rb") as f:
        d = pickle.load(f)
    with open(os.path.join(ex.CACHE, "exploration_trades.pkl"), "rb") as f:
        oof = pickle.load(f)["oof"]
    sessions = {s["date"]: Session(s["date"], s["open"], s["close"]) for s in d["sessions"]}
    trades = ex.simulate_rule(oof, d["outcome_bars"], sessions, "open", None, None, {}, select=strategy.fade_candidates)
    days = sorted(oof.session_date.unique())
    ic, t = model.information_coefficient(-oof.gap_atr, oof.target, oof.session_date)
    net = ex.trade_stats([x["ret_net"]["EOD"] for x in trades if x["entered"]])
    return {"sessions": len(days), "ic": ic, "ic_t": t, "trades": net["n"], "mean_net_per_trade": net["mean"],
            "t": net["t"], "hit_rate": net["hit_rate"], **{k: v for k, v in ex.curve_stats(ex.daily_returns(trades, days)).items()
                                                            if k in ("sharpe", "max_drawdown")},
            "warning": "in-sample: the pattern was noticed on these same sessions"}


if __name__ == "__main__":
    with open(EXPLORATION_FILE) as f:
        exploration = json.load(f)
    p = write_freeze(exploration, fade_in_sample())
    print(json.dumps(p, indent=2))
