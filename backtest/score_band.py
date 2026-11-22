"""
The band for the weekly score-drift notice: the frozen live strategy's scores over the gated stocks of all 123
replayed sessions (features and scores only, no outcomes), summarised as the 95% range of 5-session averages of
the daily mean and spread. Writes data/backtest/score_band.json.

Why all sessions and not only the hold-out: the hold-out is 31 sessions of one unusual market (SPY -11%); its band
would have flagged 74% of the normal exploration weeks. The score is a fixed function of the features, so using the
exploration sessions here tells nothing about performance.
Run from the repo root:  python -m backtest.score_band
"""
import datetime as dt
import json
import pickle

import backtest  # noqa: F401  (src/ on the path)
import alerts
import strategy
from backtest import explore as ex
from backtest.freeze import HOLDOUT_END
from storage import HOLDOUT_FILE, MODEL_FILE, PREREG_FILE, SCORE_BAND_FILE


def build() -> dict:
    with open(HOLDOUT_FILE) as f:
        live = json.load(f)["goes_live"]
    frozen = strategy.load_frozen(MODEL_FILE, PREREG_FILE, live=live)
    with open(ex.DATASET, "rb") as f:
        frame = pickle.load(f)["features"]
    last = dt.date.fromisoformat(HOLDOUT_END)
    replayed = frame[frame.session_date <= last]
    gated = replayed[replayed.gate_long | replayed.gate_short].copy()
    gated["score"] = strategy.score_table(gated.set_index("symbol"), frozen["spec"]).to_numpy()
    band = alerts.score_band(gated[["session_date", "score"]])
    band.update({"strategy": live, "model_sha256": frozen["sha256"], "source": "gated stocks of all replayed sessions, scores only",
                 "first": str(gated.session_date.min()), "last": str(gated.session_date.max())})
    with open(SCORE_BAND_FILE, "w", newline="\n") as f:
        json.dump(band, f, indent=2)
        f.write("\n")
    return band


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
