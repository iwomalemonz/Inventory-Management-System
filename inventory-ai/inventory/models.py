"""The two machine-learning models.

1. Naive Bayes (CategoricalNB) - classifier.
   Question: "Will this product have a HIGH-demand day?" (top 30% of its days)
   Output: probability, used to nudge the forecast up or down.

2. Random Forest (RandomForestRegressor) - regressor.
   Question: "How many units of this product will sell on this date?"

Both only use features known in advance: product, weekday, day of month,
month, season, weekend. Accuracy is measured on the LAST 20% of days
(trained on the earlier 80%), so the numbers reflect real forecasting.
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score, roc_auc_score
from sklearn.naive_bayes import CategoricalNB

from . import config
from .calendar_features import calendar_frame
from .data_prep import build_products, load_daily

NB_FEATURES = ["product_id", "dow", "dom_bucket", "month", "season_id", "is_weekend"]
RF_FEATURES = NB_FEATURES + ["dom"]
NB_CATEGORIES = [64, 7, 4, 12, 4, 2]   # upper bounds so unseen values never crash


class DemandModels:
    def __init__(self, products, nb, rf, base_rate, metrics):
        self.products = products
        self.nb = nb
        self.rf = rf
        self.base_rate = np.asarray(base_rate)   # share of "high" days per product
        self.metrics = metrics

    # ----------------------------------------------------------------- build
    @staticmethod
    def _design_matrix(daily: pd.DataFrame, product_ids: dict) -> pd.DataFrame:
        cal = calendar_frame(daily["date"])
        cal.insert(0, "product_id", daily["product"].map(product_ids).values)
        return cal

    @classmethod
    def train(cls, verbose: bool = True) -> "DemandModels":
        daily = load_daily()
        products = build_products(daily)
        ids = {p["name"]: p["id"] for p in products}
        daily = daily[daily["product"].isin(ids)].reset_index(drop=True)
        X = cls._design_matrix(daily, ids)
        units = daily["units"].to_numpy()

        # chronological split: earlier days train, later days test
        days = np.sort(daily["date"].unique())
        cutoff = days[int(len(days) * (1 - config.TEST_FRACTION))]
        is_train = (daily["date"] < cutoff).to_numpy()

        # "high demand" threshold per product, learned from training days only
        thresholds = (
            daily[is_train].groupby("product")["units"].quantile(config.HIGH_DEMAND_QUANTILE)
        )
        high = (daily["units"] > daily["product"].map(thresholds)).astype(int).to_numpy()

        # ---- evaluate on held-out days
        nb = CategoricalNB(min_categories=NB_CATEGORIES, alpha=1.0)
        nb.fit(X.loc[is_train, NB_FEATURES], high[is_train])
        p_test = nb.predict_proba(X.loc[~is_train, NB_FEATURES])[:, 1]
        y_test = high[~is_train]
        nb_pred = (p_test >= 0.5).astype(int)
        majority = max(y_test.mean(), 1 - y_test.mean())

        rf = RandomForestRegressor(
            n_estimators=200, min_samples_leaf=25, random_state=0, n_jobs=-1
        )
        rf.fit(X.loc[is_train, RF_FEATURES], units[is_train])
        rf_pred = rf.predict(X.loc[~is_train, RF_FEATURES])
        # per-product average is the fair baseline to beat
        prod_mean = daily[is_train].groupby("product")["units"].mean()
        baseline = daily.loc[~is_train, "product"].map(prod_mean).to_numpy()

        metrics = {
            "train_days": int(is_train.sum() / len(products)),
            "test_days": int((~is_train).sum() / len(products)),
            "naive_bayes": {
                "accuracy": round(float(accuracy_score(y_test, nb_pred)), 3),
                "majority_baseline": round(float(majority), 3),
                "auc": round(float(roc_auc_score(y_test, p_test)), 3),
            },
            "random_forest": {
                "mae": round(float(mean_absolute_error(units[~is_train], rf_pred)), 1),
                "baseline_mae": round(float(mean_absolute_error(units[~is_train], baseline)), 1),
                "r2": round(float(r2_score(units[~is_train], rf_pred)), 3),
            },
        }

        # ---- refit on ALL days for the final models
        nb.fit(X[NB_FEATURES], high)
        rf.fit(X[RF_FEATURES], units)
        base_rate = [float(high[(daily["product"] == p["name"]).to_numpy()].mean()) for p in products]

        if verbose:
            print("Naive Bayes :", metrics["naive_bayes"])
            print("Random Forest:", metrics["random_forest"])
        return cls(products, nb, rf, base_rate, metrics)

    # ------------------------------------------------------------ persistence
    def save(self):
        config.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"products": self.products, "nb": self.nb, "rf": self.rf,
             "base_rate": self.base_rate, "metrics": self.metrics},
            config.MODEL_PATH,
        )

    @classmethod
    def load_or_train(cls, force: bool = False) -> "DemandModels":
        if config.MODEL_PATH.exists() and not force:
            try:
                blob = joblib.load(config.MODEL_PATH)
                return cls(blob["products"], blob["nb"], blob["rf"], blob["base_rate"], blob["metrics"])
            except Exception as exc:  # corrupt or from another sklearn version
                print(f"Saved models could not be loaded ({exc}); retraining.")
        print("Training models (first run only)...")
        models = cls.train()
        models.save()
        return models

    # ------------------------------------------------------------- prediction
    def predict(self, dates):
        """Returns (units, p_high), both shaped (n_dates, n_products)."""
        cal = calendar_frame(dates)
        n, P = len(cal), len(self.products)
        X = pd.concat([cal] * P, ignore_index=True)
        X.insert(0, "product_id", np.repeat(np.arange(P), n))
        units = self.rf.predict(X[RF_FEATURES]).reshape(P, n).T
        p_high = self.nb.predict_proba(X[NB_FEATURES])[:, 1].reshape(P, n).T
        return units, p_high
