"""Paths and JSON helpers shared by the jobs (no heavy imports, so the exit job stays light)."""
import glob
import json
import math
import numbers
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
PLANS_DIR = os.path.join(DATA_DIR, "plans")      # one morning plan per session: plans/2026-11-16.json
RESULTS_FILE = os.path.join(DATA_DIR, "results.json")  # read by the portfolio site; keep the path
DRYRUN_DIR = os.path.join(DATA_DIR, "dryrun")    # dry-run plans and score logs (never committed by the workflow)
SCORES_DIR = os.path.join(DATA_DIR, "scores")    # every scanned stock's features, gates, score and decision per session
SENTIMENT_CACHE = os.path.join(DATA_DIR, "sentiment", "finbert_cache.json")  # rolling FinBERT scores (no text)
MODEL_FILE = os.path.join(DATA_DIR, "model", "frozen_model.json")
PREREG_FILE = os.path.join(DATA_DIR, "preregistration.json")
HOLDOUT_FILE = os.path.join(DATA_DIR, "backtest", "holdout.json")  # its goes_live field names the live strategy
CANDIDATES_DIR = os.path.join(DATA_DIR, "candidates")  # realised outcome of every gated stock (would-have-traded log)
CONFIRMATION_FILE = os.path.join(DATA_DIR, "confirmation.json")  # live progress vs the pre-registered criteria
SCORE_BAND_FILE = os.path.join(DATA_DIR, "backtest", "score_band.json")  # hold-out score band for the drift notice


def json_safe(obj):
    """
    A copy of obj with every non-finite number (NaN, +/-inf, including numpy floats) replaced by None.

    For research outputs where a statistic can be undefined, e.g. a t-statistic over 0 or 1 trades.
    Python's json writes those as bare NaN by default, which is not valid JSON: browsers' JSON.parse
    rejects the whole file. Write the result with allow_nan=False so a NaN can never slip back in.
    """
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, numbers.Real) and not isinstance(obj, numbers.Integral) and not math.isfinite(obj):
        return None
    return obj


def load_json(path, default):
    if not os.path.exists(path):
        return default
    with open(path, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return default


def save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # allow_nan=False: NaN is not valid JSON and would break the site's response.json().
    with open(path, "w") as f:
        json.dump(data, f, indent=4, allow_nan=False)
        f.write("\n")


def plan_path(session_date: str, plans_dir: str = PLANS_DIR) -> str:
    return os.path.join(plans_dir, f"{session_date}.json")


def all_plan_paths(plans_dir: str = PLANS_DIR) -> list:
    """Plan files, oldest session first (ISO dates sort correctly as text)."""
    return sorted(glob.glob(os.path.join(plans_dir, "*.json")))


def buy_order_id(session_date: str, ticker: str) -> str:
    return f"{session_date}-{ticker}"


def exit_order_id(session_date: str, ticker: str) -> str:
    return f"{session_date}-{ticker}-exit"


def late_exit_order_id(session_date: str, ticker: str) -> str:
    """Sell placed by the next morning's safety net when a session's own exit failed."""
    return f"{session_date}-{ticker}-late-exit"


def origin_session(ticker: str, plans_dir: str = PLANS_DIR):
    """The most recent session whose plan actually ordered this ticker, or None."""
    for path in reversed(all_plan_paths(plans_dir)):
        plan = load_json(path, default=None) or {}
        if any(t.get("ticker") == ticker and t.get("order_id") for t in plan.get("targets", [])):
            return plan["session_date"]
    return None
