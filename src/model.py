"""
The strategy model: ridge regression predicting a gated stock's market-relative return (stock - beta x sector ETF)
from the open to the close, plus the walk-forward machinery and the information coefficient used to judge it.

Shared by the backtest (fit, cross-validate) and the live bot (predict with the frozen coefficients).
numpy only: the closed form is ten lines and easy to explain.
"""
import numpy as np
import pandas as pd

# Frozen feature order. log_rvol = log(premarket relative volume), capped: tiny denominators give values like 13,000.
FEATURES = ["gap_atr", "log_rvol", "trend", "dist_52w_high", "news_today", "baseline", "surprise", "news_volume"]
RVOL_CAP = 100.0


def add_model_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["log_rvol"] = np.log(out["rvol"].astype(float).clip(1 / RVOL_CAP, RVOL_CAP))
    return out


def feature_matrix(frame: pd.DataFrame) -> np.ndarray:
    return add_model_columns(frame)[FEATURES].to_numpy(dtype=float)


class RidgeModel:
    """
    Features are standardised with the TRAINING rows' mean and std (stored, so prediction never refits them),
    missing values count as the training mean, z-scores are clipped at +/- clip_z, and the training target is
    clipped at its clip_y quantiles so a few earnings days cannot set the weights. The intercept is not penalised.
    """

    def __init__(self, lam: float, clip_y=(0.01, 0.99), clip_z: float = 4.0):
        self.lam, self.clip_y, self.clip_z = float(lam), clip_y, clip_z

    def _z(self, X) -> np.ndarray:
        Z = (np.asarray(X, dtype=float) - self.mean_) / self.std_
        return np.clip(np.nan_to_num(Z, nan=0.0), -self.clip_z, self.clip_z)

    def fit(self, X, y) -> "RidgeModel":
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        self.mean_ = np.nanmean(X, axis=0)
        std = np.nanstd(X, axis=0)
        self.std_ = np.where(std > 0, std, 1.0)
        if self.clip_y is not None:
            y = np.clip(y, *np.quantile(y, self.clip_y))
        Z = self._z(X)
        self.intercept_ = float(y.mean())
        self.coef_ = np.linalg.solve(Z.T @ Z + self.lam * np.eye(Z.shape[1]), Z.T @ (y - self.intercept_))
        return self

    def predict(self, X) -> np.ndarray:
        return self.intercept_ + self._z(X) @ self.coef_

    def to_dict(self) -> dict:
        return {"lam": self.lam, "clip_y": list(self.clip_y) if self.clip_y else None, "clip_z": self.clip_z,
                "mean": self.mean_.tolist(), "std": self.std_.tolist(), "coef": self.coef_.tolist(),
                "intercept": self.intercept_}

    @classmethod
    def from_dict(cls, d: dict) -> "RidgeModel":
        m = cls(d["lam"], tuple(d["clip_y"]) if d["clip_y"] else None, d["clip_z"])
        m.mean_, m.std_, m.coef_ = (np.array(d[k], dtype=float) for k in ("mean", "std", "coef"))
        m.intercept_ = float(d["intercept"])
        return m


def month_folds(dates, first_test: int = 1) -> list:
    """[(first date, last date)] per calendar month, oldest first, skipping the first `first_test` months
    (they are only ever training data)."""
    days = sorted(set(pd.Series(dates)))
    months = {}
    for d in days:
        months.setdefault((d.year, d.month), []).append(d)
    return [(ds[0], ds[-1]) for ds in list(months.values())[first_test:]]


def walk_forward(frame: pd.DataFrame, features: list, target: str, lam: float, folds: list,
                 model_factory=RidgeModel, on_fit=None) -> pd.DataFrame:
    """
    Expanding-window walk-forward: for each fold, fit on every row dated before the fold's first day (with a known
    target) and predict the fold. Returns the fold rows with `pred`, `train_pred_std` (spread of the model's
    predictions on its own training rows, for thresholds) and `fold`.
    """
    out = []
    for i, (start, end) in enumerate(folds):
        train = frame[(frame.session_date < start) & frame[target].notna()]
        test = frame[(frame.session_date >= start) & (frame.session_date <= end)].copy()
        if train.empty or test.empty:
            continue
        m = model_factory(lam).fit(train[features].to_numpy(dtype=float), train[target].to_numpy(dtype=float))
        if on_fit is not None:
            on_fit(list(train.session_date))
        test["pred"] = m.predict(test[features].to_numpy(dtype=float))
        test["train_pred_std"] = float(np.std(m.predict(train[features].to_numpy(dtype=float))))
        test["fold"] = i
        out.append(test)
    return pd.concat(out) if out else frame.iloc[0:0].assign(pred=[], train_pred_std=[], fold=[])


def daily_ic(pred, realised, dates) -> pd.Series:
    """Spearman rank correlation of predicted vs realised, per day (days with < 3 usable rows are skipped)."""
    df = pd.DataFrame({"p": np.asarray(pred, dtype=float), "r": np.asarray(realised, dtype=float),
                       "d": list(dates)}).dropna()
    out = {}
    for d, g in df.groupby("d"):
        if len(g) >= 3:
            c = g["p"].rank().corr(g["r"].rank())
            if not np.isnan(c):
                out[d] = c
    return pd.Series(out, dtype=float)


def information_coefficient(pred, realised, dates) -> tuple:
    """(mean daily IC, t-stat over days)."""
    ics = daily_ic(pred, realised, dates)
    if len(ics) < 2:
        return (float(ics.mean()) if len(ics) else float("nan"), float("nan"))
    sd = ics.std(ddof=1)
    return float(ics.mean()), float(ics.mean() / (sd / np.sqrt(len(ics)))) if sd > 0 else float("inf")
