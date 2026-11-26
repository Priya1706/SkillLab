"""
Module A: Dynamic Anomaly Detection using Isolation Forest.

Identifies abnormal components relative to the overall population, including
subtle multivariate anomalies that remain within static datasheet limits.
Provides calibrated 0-100 anomaly scores, NORMAL / WARNING / ANOMALOUS labels,
and real single-feature ablation attributions for explainability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from utils.preprocessing import build_anomaly_features


@dataclass
class AnomalyPredictionOutput:
    """Structured Module A inference result for a dataset."""

    anomaly_scores: np.ndarray
    decision_values: np.ndarray
    anomaly_statuses: List[str]
    feature_contributions: List[Dict[str, float]]


class DynamicAnomalyDetector:
    """Isolation Forest-based dynamic anomaly detector for burn-in screening."""

    def __init__(self, contamination: float = 0.14, random_state: int = 42):
        self.contamination = float(np.clip(contamination, 0.02, 0.40))
        self.random_state = random_state
        self.pipeline = Pipeline(
            steps=[
                ("scaler", RobustScaler()),
                (
                    "isolation_forest",
                    IsolationForest(
                        n_estimators=250,
                        contamination=self.contamination,
                        max_samples="auto",
                        random_state=self.random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        self.feature_names: List[str] = []
        self.training_medians: Dict[str, float] = {}
        self.anomalous_cutoff: float = 0.0
        self.warning_cutoff: float = 0.045
        self.min_decision: float = -0.25
        self.max_decision: float = 0.25
        self.is_fitted: bool = False
        self.training_summary: Dict[str, float] = {}

    def fit(self, df: pd.DataFrame) -> "DynamicAnomalyDetector":
        """Fit the scaling + Isolation Forest pipeline on burn-in features."""
        X = build_anomaly_features(df)
        self.feature_names = list(X.columns)
        self.training_medians = {col: float(X[col].median()) for col in self.feature_names}

        self.pipeline.fit(X)
        decisions = self.pipeline.decision_function(X)

        # IsolationForest sets decision_function = 0 at the contamination boundary
        self.anomalous_cutoff = 0.0
        # Warning zone covers borderline components just above the anomalous boundary
        positive_decisions = decisions[decisions >= self.anomalous_cutoff]
        if len(positive_decisions) > 10:
            # Roughly the lowest 12% of positive-side components form the warning band
            band = float(np.percentile(positive_decisions, 12))
            self.warning_cutoff = float(np.clip(band, 0.02, 0.08))
        else:
            self.warning_cutoff = 0.045

        self.min_decision = float(np.percentile(decisions, 1))
        self.max_decision = float(np.percentile(decisions, 99))
        if self.min_decision >= self.anomalous_cutoff:
            self.min_decision = self.anomalous_cutoff - 0.15
        if self.max_decision <= self.warning_cutoff:
            self.max_decision = self.warning_cutoff + 0.15

        self.is_fitted = True

        statuses = self._classify_decisions(decisions)
        n_total = len(decisions)
        self.training_summary = {
            "n_samples": float(n_total),
            "contamination": float(self.contamination),
            "anomalous_rate": float(sum(s == "ANOMALOUS" for s in statuses) / max(n_total, 1)),
            "warning_rate": float(sum(s == "WARNING" for s in statuses) / max(n_total, 1)),
            "normal_rate": float(sum(s == "NORMAL" for s in statuses) / max(n_total, 1)),
            "mean_decision_normal": float(
                np.mean([d for d, s in zip(decisions, statuses) if s == "NORMAL"])
            )
            if any(s == "NORMAL" for s in statuses)
            else 0.0,
            "mean_decision_anomalous": float(
                np.mean([d for d, s in zip(decisions, statuses) if s == "ANOMALOUS"])
            )
            if any(s == "ANOMALOUS" for s in statuses)
            else 0.0,
        }
        return self

    def _decisions_to_scores(self, decisions: np.ndarray) -> np.ndarray:
        """
        Map Isolation Forest decision_function values monotonically onto [1, 99]:
        - >= warning_cutoff  -> [1, 44.9]   (NORMAL)
        - [anomalous_cutoff, warning_cutoff) -> [45, 69.9] (WARNING)
        - < anomalous_cutoff -> [70, 99]    (ANOMALOUS)
        """
        scores = np.empty_like(decisions, dtype=float)
        for i, d in enumerate(decisions):
            if d >= self.warning_cutoff:
                span = max(self.max_decision - self.warning_cutoff, 1e-6)
                ratio = np.clip((d - self.warning_cutoff) / span, 0.0, 1.0)
                scores[i] = 44.0 - ratio * 40.0
            elif d >= self.anomalous_cutoff:
                span = max(self.warning_cutoff - self.anomalous_cutoff, 1e-6)
                ratio = np.clip((d - self.anomalous_cutoff) / span, 0.0, 1.0)
                scores[i] = 69.5 - ratio * 24.5
            else:
                span = max(self.anomalous_cutoff - self.min_decision, 1e-6)
                ratio = np.clip((self.anomalous_cutoff - d) / span, 0.0, 1.5)
                scores[i] = min(99.5, 70.0 + ratio * 29.0)
        return np.round(np.clip(scores, 1.0, 99.5), 2)

    def _classify_decisions(self, decisions: np.ndarray) -> List[str]:
        statuses: List[str] = []
        for d in decisions:
            if d < self.anomalous_cutoff:
                statuses.append("ANOMALOUS")
            elif d < self.warning_cutoff:
                statuses.append("WARNING")
            else:
                statuses.append("NORMAL")
        return statuses

    def compute_feature_contributions(
        self, X: pd.DataFrame, base_decisions: np.ndarray
    ) -> List[Dict[str, float]]:
        """
        Compute genuine per-component feature contributions via single-feature
        median ablation on the fitted Isolation Forest pipeline.
        Positive improvement when replacing feature j with the population median
        indicates feature j contributed to pulling the component toward anomalous.
        """
        n_rows = len(X)
        n_feats = len(self.feature_names)
        impacts = np.zeros((n_rows, n_feats), dtype=float)

        for j, col in enumerate(self.feature_names):
            X_ablated = X.copy()
            X_ablated[col] = self.training_medians[col]
            ablated_decisions = self.pipeline.decision_function(X_ablated)
            # How much the normality score improved when feature col was neutralized
            impacts[:, j] = np.maximum(0.0, ablated_decisions - base_decisions)

        contributions_list: List[Dict[str, float]] = []
        # Fallback standardized distance if a normal sample has zero positive ablation deltas
        col_stds = {
            col: max(float(X[col].std(ddof=0)), 1e-6) for col in self.feature_names
        }
        for i in range(n_rows):
            row_impacts = impacts[i]
            total = float(row_impacts.sum())
            if total > 1e-9:
                shares = row_impacts / total
            else:
                abs_z = np.array(
                    [
                        abs(float(X.iloc[i][col]) - self.training_medians[col]) / col_stds[col]
                        for col in self.feature_names
                    ]
                )
                z_sum = float(abs_z.sum())
                shares = abs_z / z_sum if z_sum > 1e-9 else np.ones(n_feats) / n_feats

            contrib_dict = {
                col: round(float(shares[j] * 100.0), 2)
                for j, col in enumerate(self.feature_names)
            }
            contributions_list.append(contrib_dict)

        return contributions_list

    def predict(self, df: pd.DataFrame) -> AnomalyPredictionOutput:
        """Score and classify components, returning scores, labels, and feature attributions."""
        if not self.is_fitted:
            raise RuntimeError("DynamicAnomalyDetector must be fitted before calling predict().")

        X = build_anomaly_features(df)
        decisions = self.pipeline.decision_function(X)
        scores = self._decisions_to_scores(decisions)
        statuses = self._classify_decisions(decisions)
        contributions = self.compute_feature_contributions(X, decisions)

        return AnomalyPredictionOutput(
            anomaly_scores=scores,
            decision_values=decisions,
            anomaly_statuses=statuses,
            feature_contributions=contributions,
        )
