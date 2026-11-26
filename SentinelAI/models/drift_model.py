"""
Module B: Time-Series Drift Prediction using Random Forest Regressor.

Predicts end-of-burn-in parametric value (Value_168h) strictly from early
measurements (Value_0h and Value_24h), computes predicted drift and drift rate,
and flags components exceeding a configurable safety slope.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

DRIFT_INPUT_COLUMNS: List[str] = ["Value_0h", "Value_24h"]
BURN_IN_HOURS: float = 168.0
DEFAULT_SAFETY_SLOPE: float = 0.025  # parametric units per hour (~4.20 units over 168h)


def derive_early_trajectory_features(X: pd.DataFrame | np.ndarray) -> np.ndarray:
    """
    Transform the two input features (Value_0h, Value_24h) into
    [Value_0h, Value_24h, delta_0_24h, early_rate_24h] without using any
    future measurements or external features.
    """
    arr = np.asarray(X, dtype=float)
    v0 = arr[:, 0]
    v24 = arr[:, 1]
    delta = v24 - v0
    rate_24 = delta / 24.0
    return np.column_stack([v0, v24, delta, rate_24])


@dataclass
class DriftPredictionOutput:
    """Structured Module B inference result for a dataset."""

    predicted_value_168h: np.ndarray
    predicted_drift: np.ndarray
    drift_rate: np.ndarray
    drift_flag: List[bool]
    safety_slope: float


class TimeSeriesDriftPredictor:
    """Random Forest Regressor predicting Value_168h from Value_0h and Value_24h."""

    def __init__(self, n_estimators: int = 250, random_state: int = 42):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.pipeline = Pipeline(
            steps=[
                (
                    "trajectory_features",
                    FunctionTransformer(derive_early_trajectory_features, validate=False),
                ),
                (
                    "rf_regressor",
                    RandomForestRegressor(
                        n_estimators=self.n_estimators,
                        max_depth=12,
                        min_samples_leaf=3,
                        random_state=self.random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        self.is_fitted: bool = False
        self.metrics: Dict[str, float | str | int] = {}
        self.feature_importances: Dict[str, float] = {}

    def fit(self, df: pd.DataFrame) -> "TimeSeriesDriftPredictor":
        """
        Train the Random Forest Regressor on rows with valid Value_168h using a
        reproducible batch-grouped (or component-level) train/test split.
        """
        valid_df = df.dropna(subset=["Value_0h", "Value_24h", "Value_168h"]).copy()
        if len(valid_df) < 20:
            raise ValueError(
                "At least 20 complete records with Value_0h, Value_24h, and Value_168h are required to train the drift model."
            )

        X = valid_df[DRIFT_INPUT_COLUMNS]
        y = valid_df["Value_168h"].astype(float)

        unique_batches = valid_df["batch_id"].nunique() if "batch_id" in valid_df.columns else 0
        if unique_batches >= 5:
            splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=self.random_state)
            train_idx, test_idx = next(splitter.split(X, y, groups=valid_df["batch_id"]))
            split_strategy = "batch-grouped (80/20)"
        else:
            indices = np.arange(len(valid_df))
            train_idx, test_idx = train_test_split(
                indices, test_size=0.2, random_state=self.random_state
            )
            split_strategy = "component-level (80/20)"

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Fit and evaluate strictly on held-out test split to prevent data leakage
        self.pipeline.fit(X_train, y_train)
        y_pred_test = self.pipeline.predict(X_test)

        mae = float(mean_absolute_error(y_test, y_pred_test))
        rmse = float(root_mean_squared_error(y_test, y_pred_test))
        r2 = float(r2_score(y_test, y_pred_test))

        rf: RandomForestRegressor = self.pipeline.named_steps["rf_regressor"]
        derived_names = ["Value_0h", "Value_24h", "Early_Delta_24h", "Early_Slope_24h"]
        self.feature_importances = {
            name: round(float(imp * 100.0), 2)
            for name, imp in zip(derived_names, rf.feature_importances_)
        }

        self.metrics = {
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "r2": round(r2, 4),
            "train_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
            "split_strategy": split_strategy,
        }
        self.is_fitted = True
        return self

    def predict(
        self, df: pd.DataFrame, safety_slope: float = DEFAULT_SAFETY_SLOPE
    ) -> DriftPredictionOutput:
        """
        Predict Value_168h from Value_0h and Value_24h, calculate drift and drift rate,
        and flag components whose absolute drift rate exceeds safety_slope.
        """
        if not self.is_fitted:
            raise RuntimeError("TimeSeriesDriftPredictor must be fitted before calling predict().")

        X = df[DRIFT_INPUT_COLUMNS]
        pred_168 = self.pipeline.predict(X).astype(float)
        val_0 = df["Value_0h"].astype(float).to_numpy()
        pred_drift = pred_168 - val_0
        drift_rate = pred_drift / BURN_IN_HOURS
        flags = [bool(abs(rate) > safety_slope) for rate in drift_rate]

        return DriftPredictionOutput(
            predicted_value_168h=np.round(pred_168, 3),
            predicted_drift=np.round(pred_drift, 3),
            drift_rate=np.round(drift_rate, 5),
            drift_flag=flags,
            safety_slope=float(safety_slope),
        )
