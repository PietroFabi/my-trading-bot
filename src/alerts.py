"""
Notices (never failures: the alert policy fails a run only when you must act).

Coverage, every morning, from the feature table:
  daily      share of S&P 500 members with daily data (previous close and ATR). Normally ~99.6% (99.2-100% over
             123 replayed sessions), so below 70% means a data outage.
  premarket  share with a pre-market price by the scan. Normally 61-92% (most stocks simply have not traded yet),
             so the notice is set well below that, at 40%: below it the SIP minute feed (or its Yahoo fallback) failed.
  news       share with fresh news. Normally 17-46%; zero means the news feed failed.
Score drift, weekly (the afternoon job on Fridays): the last 5 sessions' mean and spread of the live scores over
the gated stocks, against the 95% band of the same 5-session windows over the 123 replayed sessions
(data/backtest/score_band.json; the hold-out alone, one unusual month, would flag most normal weeks).
"""
import glob
import os

import numpy as np
import pandas as pd

MIN_DAILY = 0.70
MIN_PREMARKET = 0.40
DRIFT_WINDOW = 5


def coverage(table: pd.DataFrame) -> dict:
    if table.empty:
        return {"daily": 0.0, "premarket": 0.0, "news": 0.0, "members": 0}
    daily = (table["prev_close"].notna() & table["atr_pct"].notna()).mean()
    return {"daily": round(float(daily), 3), "premarket": round(float(table["ref_price"].notna().mean()), 3),
            "news": round(float((table["n_fresh"] > 0).mean()), 3), "members": int(len(table))}


def coverage_notices(table: pd.DataFrame) -> list:
    c, notes = coverage(table), []
    if c["daily"] < MIN_DAILY:
        notes.append(f"Daily price data for only {c['daily']:.0%} of the {c['members']} S&P 500 members (normally "
                     f"~99%). Stocks without it cannot be traded today; check the Alpaca/Yahoo data feeds.")
    if c["premarket"] < MIN_PREMARKET:
        notes.append(f"Pre-market prices for only {c['premarket']:.0%} of members by the scan (normally 61-92%). "
                     f"The minute-bar feed may be failing.")
    if c["news"] == 0:
        notes.append("No fresh news for any member this morning (normally 17-46%). The news feed may be failing; "
                     "no stock can pass the gates.")
    return notes


def _daily_stats(frame: pd.DataFrame) -> pd.DataFrame:
    g = frame.dropna(subset=["score"]).groupby("session_date")["score"]
    return pd.DataFrame({"mean": g.mean(), "std": g.std()}).dropna().sort_index()


def score_band(frame: pd.DataFrame, window: int = DRIFT_WINDOW) -> dict:
    """95% band of the rolling `window`-session averages of the daily score mean and spread (from the hold-out)."""
    daily = _daily_stats(frame)
    rolled = daily.rolling(window).mean().dropna()
    return {"window": window, "windows": int(len(rolled)), "sessions": int(len(daily)),
            "mean": [float(np.percentile(rolled["mean"], 2.5)), float(np.percentile(rolled["mean"], 97.5))],
            "std": [float(np.percentile(rolled["std"], 2.5)), float(np.percentile(rolled["std"], 97.5))]}


def score_drift_notice(candidates_dir: str, band: dict):
    """A notice if the last `window` live sessions' scores left the hold-out band, else None."""
    window = band["window"]
    paths = sorted(glob.glob(os.path.join(candidates_dir, "*.csv")))[-window:]
    if len(paths) < window:
        return None
    stats = _daily_stats(pd.concat([pd.read_csv(p) for p in paths], ignore_index=True))
    if len(stats) < window:
        return None
    mean, std = float(stats["mean"].mean()), float(stats["std"].mean())
    problems = []
    if not band["mean"][0] <= mean <= band["mean"][1]:
        problems.append(f"mean {mean:+.5f} outside {band['mean'][0]:+.5f}..{band['mean'][1]:+.5f}")
    if not band["std"][0] <= std <= band["std"][1]:
        problems.append(f"spread {std:.5f} outside {band['std'][0]:.5f}..{band['std'][1]:.5f}")
    if not problems:
        return None
    return (f"Live scores over the last {window} sessions drifted from the hold-out ({'; '.join(problems)}). "
            f"The model is frozen and keeps trading; this is a prompt to check the input data.")
