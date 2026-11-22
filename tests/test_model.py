import datetime as dt
import unittest

import numpy as np
import pandas as pd

import tests  # noqa: F401
import model


def data(n=400, seed=0, p=3):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p)) * [1.0, 2.0, 0.5][:p] + [0.5, -1.0, 3.0][:p]
    y = X @ np.array([0.3, -0.2, 0.1][:p]) + rng.normal(scale=0.5, size=n)
    return X, y


class RidgeTest(unittest.TestCase):
    def test_matches_least_squares_on_the_augmented_system(self):
        X, y = data()
        lam = 10.0
        m = model.RidgeModel(lam, clip_y=None, clip_z=np.inf).fit(X, y)
        Z = (X - X.mean(0)) / X.std(0)
        A = np.vstack([Z, np.sqrt(lam) * np.eye(3)])
        b = np.concatenate([y - y.mean(), np.zeros(3)])
        np.testing.assert_allclose(m.coef_, np.linalg.lstsq(A, b, rcond=None)[0], atol=1e-10)
        self.assertAlmostEqual(m.intercept_, y.mean())

    def test_tiny_penalty_matches_ordinary_least_squares(self):
        X, y = data()
        m = model.RidgeModel(1e-9, clip_y=None, clip_z=np.inf).fit(X, y)
        A = np.column_stack([np.ones(len(X)), X])
        ols = A @ np.linalg.lstsq(A, y, rcond=None)[0]
        np.testing.assert_allclose(m.predict(X), ols, atol=1e-6)

    def test_standardisation_uses_training_rows_only(self):
        X, y = data()
        m = model.RidgeModel(1.0).fit(X[:200], y[:200])
        np.testing.assert_allclose(m.mean_, X[:200].mean(0))
        np.testing.assert_allclose(m.std_, X[:200].std(0))
        m.predict(X[200:] + 100)                               # predicting never refits the scaler
        np.testing.assert_allclose(m.mean_, X[:200].mean(0))

    def test_missing_feature_values_count_as_the_training_mean(self):
        X, y = data()
        m = model.RidgeModel(1.0).fit(X, y)
        row = X[:1].copy()
        row[0, 1] = np.nan
        filled = X[:1].copy()
        filled[0, 1] = m.mean_[1]
        self.assertAlmostEqual(m.predict(row)[0], m.predict(filled)[0])

    def test_one_extreme_training_return_is_clipped_to_the_99th_percentile(self):
        X, y = data()
        wild = y.copy()
        wild[0] = 50.0                                          # a +5,000% "earnings day"
        clipped = np.clip(wild, np.quantile(wild, 0.01), np.quantile(wild, 0.99))
        np.testing.assert_allclose(model.RidgeModel(1.0).fit(X, wild).coef_,
                                   model.RidgeModel(1.0, clip_y=None).fit(X, clipped).coef_)
        self.assertLess(clipped[0], 5.0)

    def test_an_extreme_feature_value_counts_as_four_standard_deviations(self):
        X, y = data()
        m = model.RidgeModel(1.0).fit(X, y)
        wild, edge = X[:1].copy(), X[:1].copy()
        wild[0, 0] = m.mean_[0] + 50 * m.std_[0]
        edge[0, 0] = m.mean_[0] + 4 * m.std_[0]
        self.assertAlmostEqual(m.predict(wild)[0], m.predict(edge)[0])

    def test_round_trip_through_a_dict_gives_identical_predictions(self):
        X, y = data()
        m = model.RidgeModel(10.0).fit(X, y)
        again = model.RidgeModel.from_dict(m.to_dict())
        np.testing.assert_allclose(again.predict(X), m.predict(X))


def frame(days=60, per_day=30, seed=1):
    rng = np.random.default_rng(seed)
    dates = [d.date() for d in pd.bdate_range("2026-06-01", periods=days)]
    rows = []
    for d in dates:
        for i in range(per_day):
            x = rng.normal(size=2)
            rows.append({"session_date": d, "a": x[0], "b": x[1], "y": 0.01 * x[0] + 0.01 * rng.normal()})
    return pd.DataFrame(rows)


class WalkForwardTest(unittest.TestCase):
    def test_never_trains_on_a_date_at_or_after_the_test_fold_start(self):
        f = frame()
        trained_on = []
        folds = model.month_folds(f.session_date, first_test=1)
        model.walk_forward(f, ["a", "b"], "y", 1.0, folds, on_fit=trained_on.append)
        self.assertEqual(len(trained_on), len(folds))
        for (start, _end), dates in zip(folds, trained_on):
            self.assertLess(max(dates), start)

    def test_predictions_cover_exactly_the_test_folds(self):
        f = frame()
        folds = model.month_folds(f.session_date, first_test=1)
        oof = model.walk_forward(f, ["a", "b"], "y", 1.0, folds)
        self.assertEqual(oof.session_date.min(), folds[0][0])
        self.assertEqual(len(oof), int((f.session_date >= folds[0][0]).sum()))
        self.assertIn("train_pred_std", oof.columns)

    def test_month_folds_leave_the_first_month_for_training_only(self):
        dates = pd.Series([dt.date(2026, 5, 29), dt.date(2026, 6, 1), dt.date(2026, 6, 30), dt.date(2026, 7, 1),
                           dt.date(2026, 7, 31), dt.date(2026, 8, 3)])
        self.assertEqual(model.month_folds(dates, first_test=2),
                         [(dt.date(2026, 7, 1), dt.date(2026, 7, 31)), (dt.date(2026, 8, 3), dt.date(2026, 8, 3))])


class InformationCoefficientTest(unittest.TestCase):
    def test_perfect_predictions_score_one(self):
        f = frame()
        ic, t = model.information_coefficient(f.y, f.y, f.session_date)
        self.assertAlmostEqual(ic, 1.0)

    def test_shuffled_predictions_score_about_zero(self):
        f = frame(days=200)
        shuffled = f.y.sample(frac=1.0, random_state=3).to_numpy()
        ic, t = model.information_coefficient(shuffled, f.y, f.session_date)
        self.assertLess(abs(ic), 0.03)
        self.assertLess(abs(t), 3)

    def test_ic_is_a_daily_rank_correlation_averaged_over_days(self):
        dates = [dt.date(2026, 6, 1)] * 3 + [dt.date(2026, 6, 2)] * 3
        pred = [1, 2, 3, 1, 2, 3]
        real = [10, 20, 30, 30, 20, 10]                         # day 1: +1, day 2: -1
        self.assertAlmostEqual(model.information_coefficient(pred, real, dates)[0], 0.0)


class FeatureTest(unittest.TestCase):
    def test_feature_matrix_uses_log_relative_volume_in_the_frozen_order(self):
        row = pd.DataFrame([{"gap_atr": 1.0, "rvol": np.e, "trend": 0.1, "dist_52w_high": -0.2, "news_today": 0.5,
                             "baseline": 0.3, "surprise": 0.1, "news_volume": 0.2}])
        X = model.feature_matrix(row)
        self.assertEqual(model.FEATURES, ["gap_atr", "log_rvol", "trend", "dist_52w_high", "news_today",
                                          "baseline", "surprise", "news_volume"])
        np.testing.assert_allclose(X[0], [1.0, 1.0, 0.1, -0.2, 0.5, 0.3, 0.1, 0.2])

    def test_extreme_relative_volume_is_capped_before_the_log(self):
        row = pd.DataFrame([{"gap_atr": 0, "rvol": 13076.0, "trend": 0, "dist_52w_high": 0, "news_today": 0,
                             "baseline": 0, "surprise": 0, "news_volume": 0}])
        self.assertAlmostEqual(model.feature_matrix(row)[0, 1], np.log(model.RVOL_CAP))


if __name__ == "__main__":
    unittest.main()
