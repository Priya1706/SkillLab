"""
End-to-end prediction, SQLite persistence, and CSV/PDF reporting utilities for SentinelAI.
"""

from __future__ import annotations

from datetime import datetime, timezone
import io
import json
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional, Tuple
from fpdf import FPDF
import numpy as np
import pandas as pd

from data.generate_dataset import generate_synthetic_dataset
from models.anomaly_model import DynamicAnomalyDetector
from models.drift_model import DEFAULT_SAFETY_SLOPE, TimeSeriesDriftPredictor
from models.model_training import load_trained_models, train_and_save_models
from utils.explanation import generate_component_explanation
from utils.preprocessing import (
    PreprocessedDataset,
    compute_batch_statistics,
    load_and_validate_csv,
)

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "database.db"
SAMPLE_CSV_PATH = BASE_DIR / "data" / "sample_components.csv"


def get_db_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path = DB_PATH) -> None:
    """Create SQLite tables if they do not exist."""
    with get_db_connection(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS analysis_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_name TEXT NOT NULL,
                dataset_source TEXT NOT NULL,
                csv_path TEXT,
                created_at TEXT NOT NULL,
                total_components INTEGER NOT NULL,
                normal_count INTEGER NOT NULL,
                warning_count INTEGER NOT NULL,
                anomalous_count INTEGER NOT NULL,
                high_risk_count INTEGER NOT NULL,
                latent_caught_count INTEGER NOT NULL DEFAULT 0,
                safety_slope REAL NOT NULL,
                contamination REAL NOT NULL DEFAULT 0.14,
                has_ground_truth_168h INTEGER NOT NULL DEFAULT 1,
                drift_mae REAL,
                drift_rmse REAL,
                drift_r2 REAL,
                split_strategy TEXT,
                drift_importances_json TEXT,
                warnings_json TEXT,
                is_demo INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS component_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                component_id TEXT NOT NULL,
                batch_id TEXT NOT NULL,
                device_type TEXT,
                iddq_ua REAL NOT NULL,
                leakage_current_na REAL NOT NULL,
                propagation_delay_ns REAL NOT NULL,
                temperature_c REAL NOT NULL,
                value_0h REAL NOT NULL,
                value_24h REAL NOT NULL,
                value_96h REAL,
                value_168h REAL,
                predicted_value_168h REAL NOT NULL,
                predicted_drift REAL NOT NULL,
                drift_rate REAL NOT NULL,
                drift_flag INTEGER NOT NULL,
                anomaly_score REAL NOT NULL,
                anomaly_status TEXT NOT NULL,
                risk_status TEXT NOT NULL,
                static_pass INTEGER NOT NULL,
                synthetic_defect_type TEXT,
                explanation_summary TEXT NOT NULL,
                explanation_json TEXT NOT NULL,
                FOREIGN KEY (run_id) REFERENCES analysis_runs(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_comp_run_id ON component_results(run_id);
            CREATE INDEX IF NOT EXISTS idx_comp_lookup ON component_results(run_id, component_id);
            CREATE INDEX IF NOT EXISTS idx_comp_batch ON component_results(run_id, batch_id);
            """
        )


def classify_risk_status(
    anomaly_status: str, drift_flag: bool, drift_rate: float, safety_slope: float
) -> str:
    """Combine Module A and Module B findings into overall screening triage status."""
    severe_drift = abs(drift_rate) > (1.5 * safety_slope)
    if anomaly_status == "ANOMALOUS" or (anomaly_status == "WARNING" and drift_flag) or severe_drift:
        return "HIGH RISK"
    if anomaly_status == "WARNING" or drift_flag:
        return "MEDIUM RISK"
    return "LOW RISK"


def run_screening_pipeline(
    csv_path: str | Path,
    dataset_source: str,
    run_name: str,
    safety_slope: float = DEFAULT_SAFETY_SLOPE,
    contamination: float = 0.14,
    retrain: bool = True,
    is_demo: bool = False,
    db_path: Path = DB_PATH,
) -> int:
    """
    Validate dataset, train/load ML models, run Module A & Module B inference,
    generate explanations, and persist results to SQLite. Returns the new run_id.
    """
    init_db(db_path)
    preprocessed: PreprocessedDataset = load_and_validate_csv(csv_path)
    df = preprocessed.df

    loaded = None if retrain else load_trained_models()
    if loaded is not None:
        anomaly_model, drift_model = loaded
        if preprocessed.has_ground_truth_168h and retrain:
            anomaly_model, drift_model = train_and_save_models(
                df,
                contamination=contamination,
                has_ground_truth_168h=True,
            )
    else:
        existing_drift = None
        if not preprocessed.has_ground_truth_168h:
            maybe_loaded = load_trained_models()
            if maybe_loaded is not None:
                existing_drift = maybe_loaded[1]
        anomaly_model, drift_model = train_and_save_models(
            df,
            contamination=contamination,
            has_ground_truth_168h=preprocessed.has_ground_truth_168h,
            existing_drift_model=existing_drift,
        )

    # Run independent inference
    anomaly_out = anomaly_model.predict(df)
    drift_out = drift_model.predict(df, safety_slope=safety_slope)

    df["anomaly_score"] = anomaly_out.anomaly_scores
    df["anomaly_status"] = anomaly_out.anomaly_statuses
    df["predicted_value_168h"] = drift_out.predicted_value_168h
    df["predicted_drift"] = drift_out.predicted_drift
    df["drift_rate"] = drift_out.drift_rate
    df["drift_flag"] = drift_out.drift_flag
    df["risk_status"] = [
        classify_risk_status(a_stat, d_flag, float(d_rate), safety_slope)
        for a_stat, d_flag, d_rate in zip(
            df["anomaly_status"], df["drift_flag"], df["drift_rate"]
        )
    ]

    global_stats, batch_stats = compute_batch_statistics(df)

    explanations_summary: List[str] = []
    explanations_json: List[str] = []
    for idx in range(len(df)):
        row = df.iloc[idx]
        exp = generate_component_explanation(
            row=row,
            feature_contrib=anomaly_out.feature_contributions[idx],
            drift_feature_importances=drift_model.feature_importances,
            global_stats=global_stats,
            batch_stats=batch_stats,
            safety_slope=safety_slope,
        )
        explanations_summary.append(exp["summary"])
        explanations_json.append(json.dumps(exp))

    df["explanation_summary"] = explanations_summary
    df["explanation_json"] = explanations_json

    total_count = int(len(df))
    normal_count = int((df["anomaly_status"] == "NORMAL").sum())
    warning_count = int((df["anomaly_status"] == "WARNING").sum())
    anomalous_count = int((df["anomaly_status"] == "ANOMALOUS").sum())
    high_risk_count = int((df["risk_status"] == "HIGH RISK").sum())
    latent_caught_count = int(
        (df["static_pass"] & (df["risk_status"].isin(["MEDIUM RISK", "HIGH RISK"]))).sum()
    )

    metrics = drift_model.metrics or {}
    created_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    with get_db_connection(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO analysis_runs (
                run_name, dataset_source, csv_path, created_at,
                total_components, normal_count, warning_count, anomalous_count,
                high_risk_count, latent_caught_count, safety_slope, contamination,
                has_ground_truth_168h, drift_mae, drift_rmse, drift_r2,
                split_strategy, drift_importances_json, warnings_json, is_demo
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_name,
                dataset_source,
                str(csv_path),
                created_at,
                total_count,
                normal_count,
                warning_count,
                anomalous_count,
                high_risk_count,
                latent_caught_count,
                float(safety_slope),
                float(contamination),
                1 if preprocessed.has_ground_truth_168h else 0,
                metrics.get("mae"),
                metrics.get("rmse"),
                metrics.get("r2"),
                metrics.get("split_strategy", "N/A"),
                json.dumps(drift_model.feature_importances),
                json.dumps(preprocessed.warnings),
                1 if is_demo else 0,
            ),
        )
        run_id = int(cursor.lastrowid)

        rows_to_insert = []
        for _, r in df.iterrows():
            v96 = float(r["Value_96h"]) if "Value_96h" in r and pd.notna(r["Value_96h"]) else None
            v168 = float(r["Value_168h"]) if "Value_168h" in r and pd.notna(r["Value_168h"]) else None
            rows_to_insert.append(
                (
                    run_id,
                    str(r["component_id"]),
                    str(r["batch_id"]),
                    str(r.get("device_type", "GENERIC-IC")),
                    float(r["iddq_ua"]),
                    float(r["leakage_current_na"]),
                    float(r["propagation_delay_ns"]),
                    float(r["temperature_c"]),
                    float(r["Value_0h"]),
                    float(r["Value_24h"]),
                    v96,
                    v168,
                    float(r["predicted_value_168h"]),
                    float(r["predicted_drift"]),
                    float(r["drift_rate"]),
                    1 if bool(r["drift_flag"]) else 0,
                    float(r["anomaly_score"]),
                    str(r["anomaly_status"]),
                    str(r["risk_status"]),
                    1 if bool(r["static_pass"]) else 0,
                    str(r.get("synthetic_defect_type", "Unknown")),
                    str(r["explanation_summary"]),
                    str(r["explanation_json"]),
                )
            )

        conn.executemany(
            """
            INSERT INTO component_results (
                run_id, component_id, batch_id, device_type,
                iddq_ua, leakage_current_na, propagation_delay_ns, temperature_c,
                value_0h, value_24h, value_96h, value_168h,
                predicted_value_168h, predicted_drift, drift_rate, drift_flag,
                anomaly_score, anomaly_status, risk_status, static_pass,
                synthetic_defect_type, explanation_summary, explanation_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows_to_insert,
        )

    return run_id


def ensure_initialized(db_path: Path = DB_PATH) -> int:
    """
    Ensure synthetic dataset exists, models are trained, and SQLite has an active run.
    Returns the latest run_id.
    """
    init_db(db_path)
    latest = get_latest_run(db_path)
    if latest is not None and load_trained_models() is not None:
        return int(latest["id"])

    if not SAMPLE_CSV_PATH.exists():
        generate_synthetic_dataset(n_components=1200, output_path=SAMPLE_CSV_PATH)

    return run_screening_pipeline(
        csv_path=SAMPLE_CSV_PATH,
        dataset_source="Synthetic Demo Dataset (sample_components.csv)",
        run_name="Initial Synthetic Burn-In Screening",
        safety_slope=DEFAULT_SAFETY_SLOPE,
        contamination=0.14,
        retrain=True,
        is_demo=True,
        db_path=db_path,
    )


def reset_demo_dataset(
    safety_slope: float = DEFAULT_SAFETY_SLOPE,
    contamination: float = 0.14,
    db_path: Path = DB_PATH,
) -> int:
    """Regenerate synthetic demo dataset, retrain both ML models, and store new run."""
    generate_synthetic_dataset(n_components=1200, output_path=SAMPLE_CSV_PATH)
    return run_screening_pipeline(
        csv_path=SAMPLE_CSV_PATH,
        dataset_source="Synthetic Demo Dataset (sample_components.csv)",
        run_name="Demo Reset Screening Run",
        safety_slope=safety_slope,
        contamination=contamination,
        retrain=True,
        is_demo=True,
        db_path=db_path,
    )


def get_latest_run(db_path: Path = DB_PATH) -> Optional[Dict[str, Any]]:
    init_db(db_path)
    with get_db_connection(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM analysis_runs ORDER BY id DESC LIMIT 1"
        ).fetchone()
    if row is None:
        return None
    d = dict(row)
    d["drift_importances"] = json.loads(d.get("drift_importances_json") or "{}")
    d["warnings"] = json.loads(d.get("warnings_json") or "[]")
    return d


def get_run_history(limit: int = 15, db_path: Path = DB_PATH) -> List[Dict[str, Any]]:
    init_db(db_path)
    with get_db_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM analysis_runs ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def get_run_components_df(run_id: int, db_path: Path = DB_PATH) -> pd.DataFrame:
    with get_db_connection(db_path) as conn:
        df = pd.read_sql_query(
            "SELECT * FROM component_results WHERE run_id = ? ORDER BY id ASC",
            conn,
            params=(run_id,),
        )
    return df


def get_component_detail(
    run_id: int, component_id: str, db_path: Path = DB_PATH
) -> Optional[Dict[str, Any]]:
    with get_db_connection(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM component_results WHERE run_id = ? AND component_id = ?",
            (run_id, component_id),
        ).fetchone()
    if row is None:
        return None
    d = dict(row)
    d["explanation"] = json.loads(d["explanation_json"])
    return d


def get_batch_summary(run_id: int, db_path: Path = DB_PATH) -> List[Dict[str, Any]]:
    df = get_run_components_df(run_id, db_path)
    if df.empty:
        return []
    summaries = []
    for batch_id, grp in df.groupby("batch_id"):
        total = int(len(grp))
        normal = int((grp["anomaly_status"] == "NORMAL").sum())
        warning = int((grp["anomaly_status"] == "WARNING").sum())
        anomalous = int((grp["anomaly_status"] == "ANOMALOUS").sum())
        high_risk = int((grp["risk_status"] == "HIGH RISK").sum())
        drift_flagged = int((grp["drift_flag"] == 1).sum())
        latent = int(((grp["static_pass"] == 1) & (grp["risk_status"] != "LOW RISK")).sum())
        yield_pct = round(normal / total * 100.0, 1) if total else 0.0
        summaries.append(
            {
                "batch_id": str(batch_id),
                "total": total,
                "normal": normal,
                "warning": warning,
                "anomalous": anomalous,
                "high_risk": high_risk,
                "drift_flagged": drift_flagged,
                "latent_caught": latent,
                "avg_anomaly_score": round(float(grp["anomaly_score"].mean()), 2),
                "avg_leakage_na": round(float(grp["leakage_current_na"].mean()), 2),
                "avg_drift": round(float(grp["predicted_drift"].mean()), 3),
                "yield_pct": yield_pct,
            }
        )
    summaries.sort(key=lambda x: x["high_risk"], reverse=True)
    return summaries


def generate_pdf_report(
    run_info: Dict[str, Any],
    batch_summary: List[Dict[str, Any]],
    flagged_df: pd.DataFrame,
    batch_filter: Optional[str] = None,
) -> bytes:
    """Create a multi-section PDF Quality Assurance Summary Report using fpdf2."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=14)
    pdf.add_page()

    # Header
    pdf.set_fill_color(11, 25, 44)
    pdf.rect(0, 0, 210, 28, style="F")
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_xy(12, 7)
    pdf.cell(0, 8, "SentinelAI - Burn-In Quality Assurance Report", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_x(12)
    scope = f"Batch Scope: {batch_filter}" if batch_filter else "Batch Scope: All Batches"
    pdf.cell(
        0,
        6,
        f"Dataset: {run_info['dataset_source']} | Run #{run_info['id']} | {run_info['created_at']} | {scope}",
    )

    pdf.set_text_color(30, 41, 59)
    pdf.set_y(34)

    # Disclaimer
    pdf.set_fill_color(239, 246, 255)
    pdf.set_font("Helvetica", "I", 8.5)
    pdf.multi_cell(
        0,
        5,
        "Note: Default demonstration data is synthetic burn-in telemetry created for screening simulation "
        "and is not actual ISRO flight test data.",
        fill=True,
    )
    pdf.ln(4)

    # Executive Summary
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 7, "1. Executive Screening Summary", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    summary_lines = [
        f"Total Components Screened: {run_info['total_components']}",
        f"Normal: {run_info['normal_count']} | Warning: {run_info['warning_count']} | Anomalous: {run_info['anomalous_count']}",
        f"High-Risk Triage: {run_info['high_risk_count']} | Latent Defects Passing Static Limits: {run_info['latent_caught_count']}",
        f"Configured Safety Drift Slope: +/-{run_info['safety_slope']:.4f} units/hr",
        (
            f"Drift Model Evaluation (Module B): MAE = {run_info.get('drift_mae', 'N/A')}, "
            f"RMSE = {run_info.get('drift_rmse', 'N/A')}, R2 = {run_info.get('drift_r2', 'N/A')}"
        ),
    ]
    for line in summary_lines:
        pdf.cell(0, 6, f"  - {line}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # Batch Summary Table
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 7, "2. Batch-Level Quality Summary", new_x="LMARGIN", new_y="NEXT")
    headers = ["Batch ID", "Total", "Normal", "Warn", "Anom", "HighRisk", "Latent", "Yield%"]
    widths = [30, 18, 20, 18, 18, 24, 22, 22]

    pdf.set_fill_color(30, 64, 175)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 8.5)
    for h, w in zip(headers, widths):
        pdf.cell(w, 7, h, border=1, fill=True, align="C")
    pdf.ln()

    pdf.set_text_color(30, 41, 59)
    pdf.set_font("Helvetica", "", 8.5)
    for b in batch_summary:
        vals = [
            b["batch_id"],
            str(b["total"]),
            str(b["normal"]),
            str(b["warning"]),
            str(b["anomalous"]),
            str(b["high_risk"]),
            str(b["latent_caught"]),
            f"{b['yield_pct']}%",
        ]
        for v, w in zip(vals, widths):
            pdf.cell(w, 6, v, border=1, align="C")
        pdf.ln()

    pdf.ln(5)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 7, "3. Top Flagged / High-Risk Components (Top 20)", new_x="LMARGIN", new_y="NEXT")

    if flagged_df.empty:
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 6, "No components flagged under current filter.", new_x="LMARGIN", new_y="NEXT")
    else:
        top_flagged = flagged_df.sort_values(by="anomaly_score", ascending=False).head(20)
        f_headers = ["Comp ID", "Batch", "Leak(nA)", "Pred168h", "DriftRate", "Score", "Risk", "Static"]
        f_widths = [24, 22, 20, 22, 24, 18, 26, 18]
        pdf.set_fill_color(15, 23, 42)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", 8)
        for h, w in zip(f_headers, f_widths):
            pdf.cell(w, 7, h, border=1, fill=True, align="C")
        pdf.ln()

        pdf.set_text_color(30, 41, 59)
        pdf.set_font("Helvetica", "", 8)
        for _, r in top_flagged.iterrows():
            vals = [
                str(r["component_id"]),
                str(r["batch_id"]),
                f"{r['leakage_current_na']:.1f}",
                f"{r['predicted_value_168h']:.2f}",
                f"{r['drift_rate']:+.4f}",
                f"{r['anomaly_score']:.1f}",
                str(r["risk_status"]),
                "PASS" if r["static_pass"] else "FAIL",
            ]
            for v, w in zip(vals, f_widths):
                pdf.cell(w, 6, v, border=1, align="C")
            pdf.ln()

    raw_bytes = pdf.output()
    return bytes(raw_bytes)
