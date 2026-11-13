"""Paths and JSON helpers shared by the jobs (no heavy imports, so the midday job stays light)."""
import glob
import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
PLANS_DIR = os.path.join(DATA_DIR, "plans")      # one morning plan per session: plans/2026-11-16.json
RESULTS_FILE = os.path.join(DATA_DIR, "results.json")  # read by the portfolio site; keep the path


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
