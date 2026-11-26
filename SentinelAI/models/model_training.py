"""
Model training and persistence orchestrator for SentinelAI.

Trains Module A (Isolation Forest) and Module B (Random Forest Regressor)
independently, evaluates performance metrics, and persists trained artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple
import joblib
import pandas as pd

from models.anomaly_model import DynamicAnomalyDetector
from models.drift_model import TimeSeriesDriftPredictor

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = Path(__file__).resolve().parent / "saved"
ANOMALY_MODEL_PATH = MODEL_DIR / "anomaly_detector.joblib"
DRIFT_MODEL_PATH = MODEL_DIR / "drift_predictor.joblib"
METADATA_PATH = MODEL_DIR / "model_metadata.json"


def train_and_save_models(
    df: pd.DataFrame,
    contamination: float = 0.14,
    has_ground_truth_168h: bool = True,
    existing_drift_model: Optional[TimeSeriesDriftPredictor] = None,
) -> Tuple[DynamicAnomalyDetector, TimeSeriesDriftPredictor]:
    """
    Train Module A on the provided dataset and train Module B if Value_168h ground truth
    is available (otherwise reuse existing fitted drift model if provided).
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    anomaly_detector = DynamicAnomalyDetector(contamination=contamination, random_state=42)
    anomaly_detector.fit(df)
    joblib.dump(anomaly_detector, ANOMALY_MODEL_PATH)

    if has_ground_truth_168h:
        drift_predictor = TimeSeriesDriftPredictor(n_estimators=250, random_state=42)
        drift_predictor.fit(df)
        joblib.dump(drift_predictor, DRIFT_MODEL_PATH)
    elif existing_drift_model is not None and existing_drift_model.is_fitted:
        drift_predictor = existing_drift_model
    elif DRIFT_MODEL_PATH.exists():
        drift_predictor = joblib.load(DRIFT_MODEL_PATH)
    else:
        raise ValueError(
            "Cannot train drift prediction model: Value_168h ground truth is missing and no pre-trained drift model exists."
        )

    metadata = {
        "anomaly_summary": anomaly_detector.training_summary,
        "drift_metrics": drift_predictor.metrics,
        "drift_feature_importances": drift_predictor.feature_importances,
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return anomaly_detector, drift_predictor


def load_trained_models() -> Optional[Tuple[DynamicAnomalyDetector, TimeSeriesDriftPredictor]]:
    """Load persisted Module A and Module B models if both exist on disk."""
    if ANOMALY_MODEL_PATH.exists() and DRIFT_MODEL_PATH.exists():
        anomaly_detector: DynamicAnomalyDetector = joblib.load(ANOMALY_MODEL_PATH)
        drift_predictor: TimeSeriesDriftPredictor = joblib.load(DRIFT_MODEL_PATH)
        if anomaly_detector.is_fitted and drift_predictor.is_fitted:
            return anomaly_detector, drift_predictor
    return None
