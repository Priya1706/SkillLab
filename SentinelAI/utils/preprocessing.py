"""
Data validation, cleaning, and feature preprocessing utilities for SentinelAI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from data.generate_dataset import STATIC_DATASHEET_LIMITS

REQUIRED_COLUMNS: List[str] = [
    "component_id",
    "batch_id",
    "iddq_ua",
    "leakage_current_na",
    "propagation_delay_ns",
    "temperature_c",
    "Value_0h",
    "Value_24h",
]

NUMERIC_FEATURE_COLUMNS: List[str] = [
    "iddq_ua",
    "leakage_current_na",
    "propagation_delay_ns",
    "temperature_c",
    "Value_0h",
    "Value_24h",
]

OPTIONAL_NUMERIC_COLUMNS: List[str] = ["Value_96h", "Value_168h"]

FEATURE_DISPLAY_NAMES: Dict[str, str] = {
    "iddq_ua": "Standby Current (Iddq, uA)",
    "leakage_current_na": "Leakage Current (nA)",
    "propagation_delay_ns": "Propagation Delay (ns)",
    "temperature_c": "Junction Temperature (C)",
    "Value_0h": "Initial Value (0h)",
    "Value_24h": "Burn-In Value (24h)",
    "early_delta_24h": "Early 24h Parametric Shift",
}


class DatasetValidationError(ValueError):
    """Raised when an uploaded or loaded CSV fails schema or quality checks."""


@dataclass
class PreprocessedDataset:
    """Validated and cleaned dataset alongside metadata flags."""

    df: pd.DataFrame
    has_ground_truth_168h: bool
    has_ground_truth_96h: bool
    has_synthetic_labels: bool
    imputed_cells: int
    warnings: List[str]


def _normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace and map common case-insensitive column aliases."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    alias_map = {
        "componentid": "component_id",
        "component": "component_id",
        "batchid": "batch_id",
        "batch": "batch_id",
        "iddq": "iddq_ua",
        "standby_current": "iddq_ua",
        "leakage": "leakage_current_na",
        "leakage_current": "leakage_current_na",
        "propagation_delay": "propagation_delay_ns",
        "delay": "propagation_delay_ns",
        "temperature": "temperature_c",
        "temp": "temperature_c",
        "value_0h": "Value_0h",
        "value_24h": "Value_24h",
        "value_96h": "Value_96h",
        "value_168h": "Value_168h",
    }

    rename_dict = {}
    existing_lower = {c.lower(): c for c in df.columns}
    for alias_lower, canonical in alias_map.items():
        if canonical not in df.columns and alias_lower in existing_lower:
            rename_dict[existing_lower[alias_lower]] = canonical
    if rename_dict:
        df = df.rename(columns=rename_dict)
    return df


def evaluate_static_limits(df: pd.DataFrame) -> pd.Series:
    """Evaluate traditional static datasheet pass/fail limits per component."""
    cond = (
        (df["iddq_ua"] <= STATIC_DATASHEET_LIMITS["iddq_ua_max"])
        & (df["leakage_current_na"] <= STATIC_DATASHEET_LIMITS["leakage_current_na_max"])
        & (df["propagation_delay_ns"] <= STATIC_DATASHEET_LIMITS["propagation_delay_ns_max"])
        & (df["temperature_c"] <= STATIC_DATASHEET_LIMITS["temperature_c_max"])
        & (df["Value_0h"] >= STATIC_DATASHEET_LIMITS["value_min"])
        & (df["Value_0h"] <= STATIC_DATASHEET_LIMITS["value_max"])
        & (df["Value_24h"] >= STATIC_DATASHEET_LIMITS["value_min"])
        & (df["Value_24h"] <= STATIC_DATASHEET_LIMITS["value_max"])
    )
    if "Value_96h" in df.columns and df["Value_96h"].notna().any():
        v96_ok = df["Value_96h"].isna() | (
            (df["Value_96h"] >= STATIC_DATASHEET_LIMITS["value_min"])
            & (df["Value_96h"] <= STATIC_DATASHEET_LIMITS["value_max"])
        )
        cond = cond & v96_ok
    if "Value_168h" in df.columns and df["Value_168h"].notna().any():
        v168_ok = df["Value_168h"].isna() | (
            (df["Value_168h"] >= STATIC_DATASHEET_LIMITS["value_min"])
            & (df["Value_168h"] <= STATIC_DATASHEET_LIMITS["value_max"])
        )
        cond = cond & v168_ok
    return cond


def load_and_validate_csv(file_or_path) -> PreprocessedDataset:
    """
    Load a CSV file or file-like stream, validate schema and data integrity,
    coerce numeric types, impute minor missing values, and return clean data.
    """
    try:
        raw_df = pd.read_csv(file_or_path)
    except pd.errors.EmptyDataError as exc:
        raise DatasetValidationError("The uploaded CSV file is empty.") from exc
    except pd.errors.ParserError as exc:
        raise DatasetValidationError(f"Malformed CSV format: {exc}") from exc
    except Exception as exc:
        raise DatasetValidationError(f"Unable to read CSV file: {exc}") from exc

    if raw_df.empty or len(raw_df) == 0:
        raise DatasetValidationError("The dataset contains 0 component rows.")

    df = _normalize_column_names(raw_df)

    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise DatasetValidationError(
            f"Missing required column(s): {', '.join(missing_cols)}. "
            f"Required columns are: {', '.join(REQUIRED_COLUMNS)}."
        )

    warnings: List[str] = []

    # Clean ID columns
    df["component_id"] = df["component_id"].astype(str).str.strip()
    df["batch_id"] = df["batch_id"].astype(str).str.strip()
    if (df["component_id"] == "").any() or df["component_id"].isna().any():
        raise DatasetValidationError("Column 'component_id' contains empty values.")
    if (df["batch_id"] == "").any() or df["batch_id"].isna().any():
        df["batch_id"] = df["batch_id"].replace("", "BATCH-UNKNOWN").fillna("BATCH-UNKNOWN")
        warnings.append("Empty batch_id values were assigned to 'BATCH-UNKNOWN'.")

    # Deduplicate component_id if needed
    if df["component_id"].duplicated().any():
        dup_count = int(df["component_id"].duplicated().sum())
        df = df.drop_duplicates(subset=["component_id"], keep="first").reset_index(drop=True)
        warnings.append(f"Removed {dup_count} duplicate component_id row(s).")

    if "device_type" not in df.columns:
        df["device_type"] = "GENERIC-IC"

    # Validate & coerce required numeric columns
    imputed_cells = 0
    for col in NUMERIC_FEATURE_COLUMNS:
        coerced = pd.to_numeric(df[col], errors="coerce")
        if coerced.isna().all():
            raise DatasetValidationError(
                f"Column '{col}' contains no valid numeric values."
            )
        na_count = int(coerced.isna().sum())
        if na_count > 0:
            if na_count > len(df) * 0.5:
                raise DatasetValidationError(
                    f"Column '{col}' has too many missing/non-numeric entries ({na_count}/{len(df)})."
                )
            median_val = float(coerced.median())
            coerced = coerced.fillna(median_val)
            imputed_cells += na_count
            warnings.append(
                f"Imputed {na_count} missing/invalid value(s) in '{col}' using median ({median_val:.3f})."
            )
        df[col] = coerced.astype(float)

    # Handle optional time-series columns (Value_96h, Value_168h)
    for col in OPTIONAL_NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)

    has_ground_truth_96h = "Value_96h" in df.columns and df["Value_96h"].notna().sum() >= max(5, int(0.8 * len(df)))
    has_ground_truth_168h = "Value_168h" in df.columns and df["Value_168h"].notna().sum() >= max(10, int(0.8 * len(df)))

    if not has_ground_truth_168h and "Value_168h" in df.columns and df["Value_168h"].notna().any():
        warnings.append(
            "Value_168h has sparse coverage (<80%); evaluation metrics will only use available ground truth."
        )

    if "static_pass" in df.columns:
        df["static_pass"] = df["static_pass"].astype(str).str.lower().isin(["true", "1", "yes", "pass"])
    else:
        df["static_pass"] = evaluate_static_limits(df)

    has_synthetic_labels = "synthetic_defect_type" in df.columns
    if not has_synthetic_labels:
        df["synthetic_defect_type"] = "Unknown"

    # Derived early shift feature (strictly from 0h and 24h, no future leakage)
    df["early_delta_24h"] = df["Value_24h"] - df["Value_0h"]

    return PreprocessedDataset(
        df=df.reset_index(drop=True),
        has_ground_truth_168h=bool(has_ground_truth_168h),
        has_ground_truth_96h=bool(has_ground_truth_96h),
        has_synthetic_labels=bool(has_synthetic_labels),
        imputed_cells=imputed_cells,
        warnings=warnings,
    )


def build_anomaly_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build feature matrix for Module A (Isolation Forest).
    Uses only electrical measurements and early 0h->24h shift (independent of Module B).
    """
    features = pd.DataFrame(index=df.index)
    for col in NUMERIC_FEATURE_COLUMNS:
        features[col] = df[col].astype(float)
    features["early_delta_24h"] = (df["Value_24h"] - df["Value_0h"]).astype(float)
    return features


def compute_batch_statistics(df: pd.DataFrame) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    """
    Compute global and per-batch mean/std/median for Explainable AI comparisons.
    """
    cols = NUMERIC_FEATURE_COLUMNS + ["early_delta_24h"]
    global_stats: Dict[str, Dict[str, float]] = {}
    for col in cols:
        series = df[col].astype(float)
        std_val = float(series.std(ddof=0))
        global_stats[col] = {
            "mean": float(series.mean()),
            "std": std_val if std_val > 1e-9 else 1e-9,
            "median": float(series.median()),
            "q25": float(series.quantile(0.25)),
            "q75": float(series.quantile(0.75)),
        }

    batch_stats: Dict[str, Dict[str, float]] = {}
    for batch_id, group in df.groupby("batch_id"):
        b_dict: Dict[str, float] = {"count": float(len(group))}
        for col in cols:
            s = group[col].astype(float)
            std_val = float(s.std(ddof=0))
            b_dict[f"{col}_mean"] = float(s.mean())
            b_dict[f"{col}_std"] = std_val if std_val > 1e-9 else global_stats[col]["std"]
            b_dict[f"{col}_median"] = float(s.median())
        batch_stats[str(batch_id)] = b_dict

    return global_stats, batch_stats
