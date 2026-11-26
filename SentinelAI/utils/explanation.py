"""
Explainable AI (XAI) module for SentinelAI.

Translates Module A (Isolation Forest feature ablation attributions + batch/global
statistical deviations) and Module B (Random Forest drift prediction + safety slope
comparison) into transparent, inspector-ready explanations without causal claims.
"""

from __future__ import annotations

from typing import Any, Dict, List
import pandas as pd

from data.generate_dataset import STATIC_DATASHEET_LIMITS
from utils.preprocessing import FEATURE_DISPLAY_NAMES

LIMIT_LOOKUP = {
    "iddq_ua": ("<=", STATIC_DATASHEET_LIMITS["iddq_ua_max"], "uA"),
    "leakage_current_na": ("<=", STATIC_DATASHEET_LIMITS["leakage_current_na_max"], "nA"),
    "propagation_delay_ns": ("<=", STATIC_DATASHEET_LIMITS["propagation_delay_ns_max"], "ns"),
    "temperature_c": ("<=", STATIC_DATASHEET_LIMITS["temperature_c_max"], "C"),
    "Value_0h": ("<= ", STATIC_DATASHEET_LIMITS["value_max"], "units"),
    "Value_24h": ("<= ", STATIC_DATASHEET_LIMITS["value_max"], "units"),
    "early_delta_24h": ("N/A", None, "units"),
}


def generate_component_explanation(
    row: pd.Series,
    feature_contrib: Dict[str, float],
    drift_feature_importances: Dict[str, float],
    global_stats: Dict[str, Dict[str, float]],
    batch_stats: Dict[str, Dict[str, float]],
    safety_slope: float,
) -> Dict[str, Any]:
    """
    Build a structured, plain-language explanation strictly derived from
    observed telemetry, batch statistics, and actual model outputs.
    """
    batch_id = str(row["batch_id"])
    b_stat = batch_stats.get(batch_id, {})

    anomaly_status = str(row["anomaly_status"])
    anomaly_score = float(row["anomaly_score"])
    risk_status = str(row["risk_status"])
    drift_flag = bool(row["drift_flag"])
    pred_168 = float(row["predicted_value_168h"])
    pred_drift = float(row["predicted_drift"])
    drift_rate = float(row["drift_rate"])
    static_pass = bool(row["static_pass"])

    # Rank contributing parameters from Module A ablation output
    sorted_contribs = sorted(
        feature_contrib.items(), key=lambda item: item[1], reverse=True
    )

    parameter_comparisons: List[Dict[str, Any]] = []
    notable_deviations: List[str] = []

    for feat, share_pct in sorted_contribs:
        val = float(row[feat])
        g_mean = global_stats[feat]["mean"]
        g_std = global_stats[feat]["std"]
        b_mean = float(b_stat.get(f"{feat}_mean", g_mean))
        b_std = float(b_stat.get(f"{feat}_std", g_std))

        batch_z = (val - b_mean) / max(b_std, 1e-6)
        global_z = (val - g_mean) / max(g_std, 1e-6)
        pct_diff_batch = ((val - b_mean) / abs(b_mean) * 100.0) if abs(b_mean) > 1e-6 else 0.0

        op, limit_val, unit = LIMIT_LOOKUP.get(feat, ("N/A", None, ""))
        within_limit = True
        if limit_val is not None:
            within_limit = val <= limit_val

        display_name = FEATURE_DISPLAY_NAMES.get(feat, feat)
        parameter_comparisons.append(
            {
                "feature": feat,
                "display_name": display_name,
                "value": round(val, 3),
                "unit": unit,
                "batch_mean": round(b_mean, 3),
                "global_mean": round(g_mean, 3),
                "batch_z_score": round(batch_z, 2),
                "global_z_score": round(global_z, 2),
                "pct_diff_batch": round(pct_diff_batch, 1),
                "contribution_pct": round(share_pct, 1),
                "static_limit": limit_val,
                "within_static_limit": within_limit,
            }
        )

        if share_pct >= 15.0 or abs(batch_z) >= 1.8:
            direction = "above" if batch_z >= 0 else "below"
            notable_deviations.append(
                f"{display_name} ({val:.2f} {unit}) is {abs(batch_z):.1f} std. dev. {direction} "
                f"batch {batch_id} mean ({b_mean:.2f} {unit}), contributing {share_pct:.1f}% to the anomaly score"
            )

    top_three = parameter_comparisons[:3]
    top_names = [f"{p['display_name']} ({p['contribution_pct']:.1f}%)" for p in top_three]

    # Drift assessment text
    abs_rate = abs(drift_rate)
    slope_ratio = abs_rate / max(safety_slope, 1e-9)
    if drift_flag:
        drift_statement = (
            f"Module B predicts a 168h value of {pred_168:.2f} (net drift {pred_drift:+.2f}, "
            f"rate {drift_rate:+.4f}/hr), which exceeds the configured safety slope of "
            f"+/-{safety_slope:.4f}/hr ({slope_ratio:.2f}x threshold)."
        )
    else:
        drift_statement = (
            f"Module B predicts a 168h value of {pred_168:.2f} (net drift {pred_drift:+.2f}, "
            f"rate {drift_rate:+.4f}/hr), remaining within the safety slope of "
            f"+/-{safety_slope:.4f}/hr ({slope_ratio:.2f}x threshold)."
        )

    # Compose clear QA Inspector summary without making causal claims
    reasons: List[str] = []
    if anomaly_status == "ANOMALOUS":
        reasons.append(
            f"Classified as ANOMALOUS by Isolation Forest (anomaly score {anomaly_score:.1f}/100)."
        )
    elif anomaly_status == "WARNING":
        reasons.append(
            f"Classified as WARNING by Isolation Forest (borderline anomaly score {anomaly_score:.1f}/100)."
        )
    else:
        reasons.append(
            f"Within normal Isolation Forest operating envelope (anomaly score {anomaly_score:.1f}/100)."
        )

    if drift_flag:
        reasons.append("Predicted parametric drift rate exceeds the safety threshold.")

    if static_pass and (anomaly_status != "NORMAL" or drift_flag):
        reasons.append(
            "LATENT RISK ALERT: All individual measurements satisfy static datasheet pass/fail limits, "
            "yet multivariate telemetry and/or early drift trajectory indicate abnormal behavior relative to batch peers."
        )
    elif not static_pass:
        reasons.append("One or more parameters also exceed traditional static datasheet limits.")

    if notable_deviations:
        reasons.append("Primary statistical deviations associated with this flag: " + "; ".join(notable_deviations[:3]) + ".")
    else:
        reasons.append(
            "Top weighted model inputs for this component: " + ", ".join(top_names) + "."
        )

    reasons.append(drift_statement)

    summary_text = " ".join(reasons)

    return {
        "component_id": str(row["component_id"]),
        "batch_id": batch_id,
        "risk_status": risk_status,
        "anomaly_status": anomaly_status,
        "anomaly_score": round(anomaly_score, 2),
        "drift_flag": drift_flag,
        "static_pass": static_pass,
        "summary": summary_text,
        "top_contributors": top_three,
        "parameter_comparisons": parameter_comparisons,
        "drift_details": {
            "value_0h": round(float(row["Value_0h"]), 3),
            "value_24h": round(float(row["Value_24h"]), 3),
            "predicted_value_168h": round(pred_168, 3),
            "actual_value_168h": (
                round(float(row["Value_168h"]), 3)
                if pd.notna(row.get("Value_168h"))
                else None
            ),
            "predicted_drift": round(pred_drift, 3),
            "drift_rate": round(drift_rate, 5),
            "safety_slope": round(safety_slope, 5),
            "exceeds_safety_slope": drift_flag,
            "slope_utilization_pct": round(slope_ratio * 100.0, 1),
            "model_feature_importances": drift_feature_importances,
        },
        "disclaimer": (
            "Model attributions reflect statistical associations and Isolation Forest / Random Forest "
            "sensitivity for this component relative to batch peers; they do not establish physical root-cause causality."
        ),
    }
