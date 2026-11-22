"""
Step 9: fit and judge the model on the exploration window only (sessions before HOLDOUT_START), out of fold.

Walk-forward by calendar month with an expanding window (May and June are training-only, so July, August,
September and 1-8 October are predicted by models that never saw them). Two selection stages, to limit snooping:
  1. model level, by out-of-fold information coefficient: lambda x {include, exclude} price-move headlines (8)
  2. trading rules, by out-of-fold mean net return per trade at the close: entry x stop x threshold (24)
An equal-weight z-score composite (gap_atr, news_today, surprise) is scored alongside as a benchmark; it never trades.

Writes data/backtest/exploration.json and docs/backtest/exploration-report.md.
Run from the repo root:  python -m backtest.explore
"""
import datetime as dt
import json
import os
import pickle
import time

import numpy as np
import pandas as pd

import backtest  # noqa: F401  (src/ on the path)
import features
import model
import strategy
from backtest import simulate as sim

HOLDOUT_START = dt.date(2026, 10, 9)
FIRST_TEST = 2                      # months used only for training (May has a single session, then June)
LAMBDAS = [0.1, 1.0, 10.0, 100.0]
EXCLUDE_PRICE_MOVE = [False, True]
ENTRIES = ["open", "orb"]
STOPS = [0.75, 1.0, 1.5, None]
THRESHOLDS = [1.5, 2.0, 2.5]
EQUITY = 100_000.0
MIN_TRADES = 30                     # a rule needs at least this many out-of-fold trades to be chosen
COMPOSITE_FEATURES = ["gap_atr", "news_today", "surprise"]

CACHE = ".cache"
DATASET = os.path.join(CACHE, "dataset.pkl")
OUT_JSON = os.path.join("data", "backtest", "exploration.json")
OUT_REPORT = os.path.join("docs", "backtest", "exploration-report.md")


# --- Pure helpers (tested) -------------------------------------------------------------------------

def exploration_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """The only door into the data: hold-out sessions never pass."""
    out = frame[frame["session_date"] < HOLDOUT_START]
    assert out["session_date"].max() < HOLDOUT_START if len(out) else True
    return out


def _open_to_close(bars: pd.DataFrame, session) -> float:
    if bars is None or bars.empty or bars.index[0] != session.open:
        return float("nan")
    return float(bars["close"].iloc[-1] / bars["open"].iloc[0] - 1)


def market_relative_return(stock_bars: pd.DataFrame, etf_bars: pd.DataFrame, beta: float, session) -> float:
    """Open -> close return of the stock minus beta x the same for its sector ETF."""
    return _open_to_close(stock_bars, session) - beta * _open_to_close(etf_bars, session)


def trade_stats(returns) -> dict:
    r = pd.Series([x for x in returns if x is not None and not pd.isna(x)], dtype=float)
    n = len(r)
    sd = r.std(ddof=1) if n > 1 else float("nan")
    return {"n": n, "mean": float(r.mean()) if n else float("nan"),
            "t": float(r.mean() / (sd / np.sqrt(n))) if n > 1 and sd > 0 else float("nan"),
            "hit_rate": float((r > 0).mean()) if n else float("nan")}


def curve_stats(daily: pd.Series) -> dict:
    """Annualised Sharpe and maximum drawdown of a daily return series (compounded)."""
    daily = daily.astype(float)
    equity = (1 + daily).cumprod()
    sd = daily.std(ddof=1)
    return {"sharpe": float(daily.mean() / sd * np.sqrt(252)) if sd > 0 else float("nan"),
            "max_drawdown": float((1 - equity / equity.cummax()).max()),
            "total_return": float(equity.iloc[-1] - 1) if len(equity) else 0.0, "days": int(len(daily))}


class CompositeModel:
    """Benchmark: mean of the training-standardised features, no fitting of weights."""

    def __init__(self, lam=None):
        pass

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self.mean_, self.std_ = np.nanmean(X, axis=0), np.nanstd(X, axis=0)
        self.std_[self.std_ == 0] = 1.0
        return self

    def predict(self, X):
        Z = np.nan_to_num((np.asarray(X, dtype=float) - self.mean_) / self.std_)
        return np.clip(Z, -4, 4).mean(axis=1)


# --- Data preparation ------------------------------------------------------------------------------

def _bars(outcome_bars: dict, day, symbol):
    frame = outcome_bars.get(day)
    if frame is None or symbol not in frame.index.get_level_values(0):
        return None
    return frame.xs(symbol, level=0)


def add_targets(gated: pd.DataFrame, outcome_bars: dict, sessions: dict) -> pd.DataFrame:
    gated = gated.copy()
    gated["target"] = [market_relative_return(_bars(outcome_bars, r.session_date, r.symbol),
                                              _bars(outcome_bars, r.session_date, r.sector_etf), r.beta,
                                              sessions[r.session_date])
                       for r in gated.itertuples()]
    return gated


def exclude_price_move_variant(frame: pd.DataFrame, articles: list, calendar: dict, dataset_sessions: list,
                               scan_before_open: dt.timedelta) -> pd.DataFrame:
    """The same morning tables with price-move headlines left out of every news feature (gates recomputed)."""
    price_cols = [c for c in frame.columns if c not in features.NEWS_COLUMNS + ["gate_long", "gate_short",
                                                                              "session_date", "rvol5"]]
    out = []
    for s in dataset_sessions:
        day = frame[frame.session_date == s["date"]]
        if day.empty:
            continue
        session = calendar[s["date"]]
        before = [calendar[d] for d in s["before"]]
        news = features.news_features(articles, session, before, session.open - scan_before_open,
                                      exclude_price_move=True)
        table = features.build_feature_table(day.set_index("symbol")[[c for c in price_cols if c != "symbol"]], news)
        table["rvol5"] = day.set_index("symbol")["rvol5"]
        table.insert(0, "session_date", s["date"])
        out.append(table.reset_index())
    return pd.concat(out, ignore_index=True)


def shortable_flags(cache_path: str = os.path.join(CACHE, "assets.json")) -> dict:
    """Today's Alpaca shortable & easy-to-borrow flags (no history exists; S&P 500 names rarely change status)."""
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)
    from alpaca.trading.enums import AssetClass
    from alpaca.trading.requests import GetAssetsRequest
    from backtest.build_dataset import _trading_client
    assets = _trading_client().get_all_assets(GetAssetsRequest(asset_class=AssetClass.US_EQUITY))
    flags = {a.symbol: bool(a.shortable and a.easy_to_borrow) for a in assets}
    with open(cache_path, "w") as f:
        json.dump(flags, f)
    return flags


# --- Stage 2: trading rules ------------------------------------------------------------------------

def simulate_rule(oof: pd.DataFrame, outcome_bars: dict, sessions: dict, entry: str, stop, c, memo: dict,
                  select=None) -> list:
    """Every trade the rule would have made, day by day (selection -> simulation -> 20% day cap)."""
    trades = []
    for day, rows in oof.groupby("session_date"):
        chosen = select(rows) if select else strategy.select_candidates(rows, c * rows["train_pred_std"].iloc[0])
        planned = []
        for r in chosen.itertuples():
            key = (day, r.symbol, r.side, entry, stop)
            if key not in memo:
                memo[key] = sim.simulate_trade(pd.Series(r._asdict()), _bars(outcome_bars, day, r.symbol)
                                               if _bars(outcome_bars, day, r.symbol) is not None else pd.DataFrame(),
                                               sessions[day], entry, stop, EQUITY)
            planned.append(dict(memo[key], session_date=day, symbol=r.symbol, sector=r.sector,
                                pred=getattr(r, "pred", None), target=r.target))
        trades.extend(sim.apply_day_cap(planned, EQUITY))
    return trades


def daily_returns(trades: list, days: list, key: str = "pnl_net", horizon: str = "EOD", side=None) -> pd.Series:
    pnl = {d: 0.0 for d in days}
    for t in trades:
        if not t["entered"] or (side and t["side"] != side):
            continue
        if key == "pnl_net":
            v = t["pnl_net"][horizon]
        else:
            r = t["ret_gross"][horizon]
            v = None if r is None else t["qty"] * t["entry"] * r
        if v is not None:
            pnl[t["session_date"]] += v
    return pd.Series(pnl).sort_index() / EQUITY


def side_horizon_table(trades: list, days: list) -> dict:
    out = {}
    for side in ("long", "short", "all"):
        chosen = [t for t in trades if t["entered"] and (side == "all" or t["side"] == side)]
        out[side] = {}
        for h in sim.HORIZONS:
            out[side][h] = {
                "net": trade_stats([t["ret_net"][h] for t in chosen]),
                "gross": trade_stats([t["ret_gross"][h] for t in chosen]),
                "curve_net": curve_stats(daily_returns(trades, days, "pnl_net", h, None if side == "all" else side)),
                "curve_gross": curve_stats(daily_returns(trades, days, "gross", h, None if side == "all" else side)),
            }
    return out


# --- Orchestration ---------------------------------------------------------------------------------

def run(log=print) -> dict:
    t0 = time.time()
    with open(DATASET, "rb") as f:
        d = pickle.load(f)
    from market_session import Session
    sessions = {s["date"]: Session(s["date"], s["open"], s["close"]) for s in d["sessions"]}
    explore_sessions = [s for s in d["sessions"] if s["date"] < HOLDOUT_START]
    frame = exploration_rows(d["features"])
    days = sorted(frame.session_date.unique())
    folds = model.month_folds(days, first_test=FIRST_TEST)
    oof_days = [x for x in days if x >= folds[0][0]]
    log(f"exploration: {len(days)} sessions {days[0]}..{days[-1]}; out-of-fold {len(oof_days)} sessions; "
        f"folds {[(str(a), str(b)) for a, b in folds]}")

    # Variants of the news features
    import market_session as ms
    from backtest.build_dataset import _trading_client
    with open(os.path.join(CACHE, "news_scored.pkl"), "rb") as f:
        articles = pickle.load(f)
    all_before = {b for s in explore_sessions for b in s["before"]}
    calendar = {s.date: s for s in ms.get_sessions(_trading_client(), min(all_before), HOLDOUT_START)}
    variants = {False: frame,
                True: exclude_price_move_variant(frame, articles, calendar, explore_sessions,
                                                 dt.timedelta(minutes=d["meta"]["scan_before_open_min"]))}
    flags = shortable_flags()
    gated = {}
    for ex, f_ in variants.items():
        g = f_[f_.gate_long | f_.gate_short]
        g = model.add_model_columns(add_targets(g, d["outcome_bars"], sessions))
        g["shortable"] = [flags.get(s, True) for s in g.symbol]
        gated[ex] = g
        log(f"variant exclude_price_move={ex}: {len(g)} gated rows, {int(g.target.isna().sum())} without a target "
            f"({time.time() - t0:.0f}s)")

    # Stage 1: model level, by out-of-fold IC
    stage1 = []
    for ex in EXCLUDE_PRICE_MOVE:
        for lam in LAMBDAS:
            oof = model.walk_forward(gated[ex], model.FEATURES, "target", lam, folds)
            ic, t = model.information_coefficient(oof.pred, oof.target, oof.session_date)
            stage1.append({"exclude_price_move": ex, "lam": lam, "ic": ic, "t": t, "rows": int(oof.target.notna().sum())})
        comp = model.walk_forward(gated[ex], COMPOSITE_FEATURES, "target", None, folds, model_factory=CompositeModel)
        ic, t = model.information_coefficient(comp.pred, comp.target, comp.session_date)
        stage1.append({"exclude_price_move": ex, "lam": None, "benchmark": "composite", "ic": ic, "t": t,
                       "rows": int(comp.target.notna().sum())})
    best1 = max((r for r in stage1 if r["lam"] is not None), key=lambda r: r["ic"])
    log("stage 1: " + "; ".join(f"ex={r['exclude_price_move']} lam={r['lam']}: IC {r['ic']:.4f} (t {r['t']:.2f})"
                                for r in stage1))
    ex, lam = best1["exclude_price_move"], best1["lam"]
    oof = model.walk_forward(gated[ex], model.FEATURES, "target", lam, folds)
    comp = model.walk_forward(gated[ex], COMPOSITE_FEATURES, "target", None, folds, model_factory=CompositeModel)

    # Stage 2: trading rules, by out-of-fold mean net return per trade at the close
    memo, stage2, all_trades = {}, [], {}
    for entry in ENTRIES:
        for stop in STOPS:
            for c in THRESHOLDS:
                trades = simulate_rule(oof, d["outcome_bars"], sessions, entry, stop, c, memo)
                st = trade_stats([t["ret_net"]["EOD"] for t in trades if t["entered"]])
                stage2.append({"entry": entry, "stop": stop, "threshold": c, **st,
                               "planned": len(trades), "zero_trade_days": len(oof_days) - len(
                                   {t["session_date"] for t in trades if t["entered"]})})
                all_trades[(entry, stop, c)] = trades
        log(f"stage 2: {entry} done ({time.time() - t0:.0f}s)")
    eligible = [r for r in stage2 if r["n"] >= MIN_TRADES]
    best2 = max(eligible or stage2, key=lambda r: r["mean"])
    chosen = all_trades[(best2["entry"], best2["stop"], best2["threshold"])]

    # The four claimed candidates (side x entry mode, exit at the close) under the chosen stop and threshold
    claimed = {}
    for entry in ENTRIES:
        trades = all_trades[(entry, best2["stop"], best2["threshold"])]
        for side in ("long", "short"):
            claimed[f"{side}_{entry}"] = {
                **trade_stats([t["ret_net"]["EOD"] for t in trades if t["entered"] and t["side"] == side]),
                **curve_stats(daily_returns(trades, oof_days, side=side))}

    # Benchmarks: SPY open-to-close each out-of-fold day, and buy-and-hold over the same span
    import marketdata
    spy = marketdata.daily_bars(["SPY"], oof_days[0] - dt.timedelta(days=10), oof_days[-1],
                                cache_dir=os.path.join(CACHE, "bars"), fallback=False).xs("SPY", level=0)
    spy.index = [pd.Timestamp(i).date() for i in spy.index]
    spy_oc = (spy["close"] / spy["open"] - 1).reindex(oof_days)
    prev_close = spy["close"][spy.index < oof_days[0]].iloc[-1]
    benchmarks = {"spy_open_to_close": {**curve_stats(spy_oc.fillna(0)), "mean_daily": float(spy_oc.mean())},
                  "spy_buy_and_hold": float(spy["close"].reindex(oof_days).iloc[-1] / prev_close - 1)}

    # The model as it would be frozen: fit on every exploration row with a target
    train = gated[ex][gated[ex].target.notna()]
    final = model.RidgeModel(lam).fit(train[model.FEATURES].to_numpy(float), train.target.to_numpy(float))
    final_std = float(np.std(final.predict(train[model.FEATURES].to_numpy(float))))

    ic_by_month = {}
    for name, frame_ in (("model", oof), ("composite", comp)):
        months = pd.to_datetime(frame_.session_date).dt.to_period("M").astype(str)
        ic_by_month[name] = {m: dict(zip(("ic", "t"), model.information_coefficient(g.pred, g.target, g.session_date)))
                             for m, g in frame_.groupby(months.values)}

    # Descriptive diagnostics (no selection uses them): does the gated direction itself predict the target?
    g_oof = oof[oof.target.notna()]
    diagnostics = {"mean_target_by_gate": {}, "univariate_ic": {}}
    for side in ("gate_long", "gate_short"):
        sub = g_oof[g_oof[side]].target
        diagnostics["mean_target_by_gate"][side] = {"n": int(len(sub)), "mean": float(sub.mean()),
                                                   "median": float(sub.median()),
                                                   "t": float(sub.mean() / (sub.std() / np.sqrt(len(sub))))}
    for feat in model.FEATURES:
        diagnostics["univariate_ic"][feat] = dict(zip(("ic", "t"), model.information_coefficient(
            g_oof[feat], g_oof.target, g_oof.session_date)))

    entered = [t for t in chosen if t["entered"]]
    result = {
        "diagnostics": diagnostics,
        "built_at": dt.datetime.now().isoformat(timespec="seconds"),
        "window": {"first": str(days[0]), "last": str(days[-1]), "sessions": len(days),
                   "oof_first": str(oof_days[0]), "oof_sessions": len(oof_days),
                   "folds": [[str(a), str(b)] for a, b in folds], "holdout_start": str(HOLDOUT_START)},
        "variants_tried": {"stage1": 8, "stage2": 24, "total": 32},
        "stage1": stage1, "stage1_choice": best1,
        "stage2": stage2, "stage2_choice": best2, "min_trades": MIN_TRADES,
        "chosen_rule_by_side_and_horizon": side_horizon_table(chosen, oof_days),
        "claimed_candidates": claimed,
        "ic_by_month": ic_by_month,
        "benchmarks": benchmarks,
        "final_model": {"features": model.FEATURES, "exclude_price_move": ex, **final.to_dict(),
                        "in_sample_pred_std": final_std, "threshold_return": best2["threshold"] * final_std,
                        "training_rows": len(train)},
        "share_at_4pct_cap": float(np.mean([t["qty"] * t["ref_price"] >= 0.97 * sim.MAX_STOCK * EQUITY
                                            for t in entered])) if entered else 0.0,
        "trades_per_day": float(len(entered) / len(oof_days)),
        "runtime_seconds": round(time.time() - t0),
    }
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(result, f, indent=2, default=str)
    with open(os.path.join(CACHE, "exploration_trades.pkl"), "wb") as f:
        pickle.dump({"chosen": chosen, "oof": oof, "comp": comp}, f)
    log(f"done in {time.time() - t0:.0f}s -> {OUT_JSON}")
    return result


if __name__ == "__main__":
    run()
