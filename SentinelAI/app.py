"""
SentinelAI – AI-Driven Anomaly Detection in Component Burn-In & Screening
Flask web application entrypoint and HTTP/API route handlers.
"""

from __future__ import annotations

from datetime import datetime
import io
import json
from pathlib import Path
from typing import Any, Dict, List
from flask import (
    Flask,
    Response,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    url_for,
)
import numpy as np
import pandas as pd
from werkzeug.utils import secure_filename

from models.drift_model import DEFAULT_SAFETY_SLOPE
from utils.prediction import (
    BASE_DIR,
    SAMPLE_CSV_PATH,
    ensure_initialized,
    generate_pdf_report,
    get_batch_summary,
    get_component_detail,
    get_latest_run,
    get_run_components_df,
    get_run_history,
    reset_demo_dataset,
    run_screening_pipeline,
)
from utils.preprocessing import DatasetValidationError

UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["SECRET_KEY"] = "sentinelai-burnin-qa-secret-key"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max CSV upload
app.config["UPLOAD_FOLDER"] = str(UPLOAD_DIR)


def _get_active_context() -> Dict[str, Any]:
    """Ensure DB/models are initialized and fetch latest run metadata for templates."""
    ensure_initialized()
    latest = get_latest_run()
    return {
        "latest_run": latest,
        "safety_slope": latest["safety_slope"] if latest else DEFAULT_SAFETY_SLOPE,
        "contamination": latest["contamination"] if latest else 0.14,
    }


@app.context_processor
def inject_global_context() -> Dict[str, Any]:
    """Provide active run badge info to base layout across all pages."""
    try:
        latest = get_latest_run()
    except Exception:
        latest = None
    return {"global_run": latest}


@app.route("/")
def index():
    ctx = _get_active_context()
    history = get_run_history(limit=6)
    return render_template("index.html", **ctx, run_history=history)


@app.route("/dashboard")
def dashboard():
    ctx = _get_active_context()
    latest = ctx["latest_run"]
    df = get_run_components_df(latest["id"])
    batch_summary = get_batch_summary(latest["id"])

    # Latent defects caught (passed traditional static limits yet flagged by SentinelAI)
    latent_df = df[(df["static_pass"] == 1) & (df["risk_status"] != "LOW RISK")].sort_values(
        by="anomaly_score", ascending=False
    )
    latent_samples = latent_df.head(8).to_dict(orient="records")

    return render_template(
        "dashboard.html",
        **ctx,
        batch_summary=batch_summary,
        latent_samples=latent_samples,
    )


@app.route("/api/dashboard-data")
def api_dashboard_data():
    ctx = _get_active_context()
    latest = ctx["latest_run"]
    df = get_run_components_df(latest["id"])
    batch_summary = get_batch_summary(latest["id"])

    # 1. Anomaly distribution
    anomaly_counts = {
        "NORMAL": int((df["anomaly_status"] == "NORMAL").sum()),
        "WARNING": int((df["anomaly_status"] == "WARNING").sum()),
        "ANOMALOUS": int((df["anomaly_status"] == "ANOMALOUS").sum()),
    }

    # 2. Leakage current distribution grouped by risk status
    leakage_by_risk = {}
    for status in ["LOW RISK", "MEDIUM RISK", "HIGH RISK"]:
        vals = df.loc[df["risk_status"] == status, "leakage_current_na"].round(2).tolist()
        leakage_by_risk[status] = vals

    # 3. Component drift trends (representative trajectories: normal, warning, anomalous/high drift)
    drift_traces: List[Dict[str, Any]] = []
    time_points = [0, 24, 96, 168]
    for status, color, label in [
        ("LOW RISK", "#059669", "Normal Sample"),
        ("MEDIUM RISK", "#D97706", "Warning / Drift Sample"),
        ("HIGH RISK", "#DC2626", "High-Risk / Anomalous Sample"),
    ]:
        sub = df[df["risk_status"] == status]
        if not sub.empty:
            # Pick up to 4 representative components from each tier
            if status == "LOW RISK":
                picked = sub.sort_values("anomaly_score", ascending=True).head(4)
            else:
                picked = sub.sort_values("predicted_drift", ascending=False).head(4)
            for _, r in picked.iterrows():
                y_vals = [
                    round(float(r["value_0h"]), 3),
                    round(float(r["value_24h"]), 3),
                    round(float(r["value_96h"]), 3) if pd.notna(r["value_96h"]) else None,
                    round(float(r["value_168h"]), 3)
                    if pd.notna(r["value_168h"])
                    else round(float(r["predicted_value_168h"]), 3),
                ]
                drift_traces.append(
                    {
                        "component_id": r["component_id"],
                        "risk_status": status,
                        "group_label": label,
                        "color": color,
                        "x": time_points,
                        "y": y_vals,
                        "predicted_168h": round(float(r["predicted_value_168h"]), 3),
                    }
                )

    # Compute mean trajectory per risk group as bold summary curves
    mean_trajectories = []
    for status, color in [
        ("LOW RISK", "#059669"),
        ("MEDIUM RISK", "#D97706"),
        ("HIGH RISK", "#DC2626"),
    ]:
        sub = df[df["risk_status"] == status]
        if not sub.empty:
            v96_mean = (
                round(float(sub["value_96h"].dropna().mean()), 3)
                if sub["value_96h"].notna().any()
                else None
            )
            v168_col = sub["value_168h"] if sub["value_168h"].notna().any() else sub["predicted_value_168h"]
            mean_trajectories.append(
                {
                    "name": f"{status} Mean (n={len(sub)})",
                    "color": color,
                    "x": time_points,
                    "y": [
                        round(float(sub["value_0h"].mean()), 3),
                        round(float(sub["value_24h"].mean()), 3),
                        v96_mean,
                        round(float(v168_col.mean()), 3),
                    ],
                }
            )

    # 4. Actual vs Predicted Value_168h
    actual_vs_pred: Dict[str, Any] = {"available": False, "points": []}
    valid_168 = df.dropna(subset=["value_168h"])
    if not valid_168.empty:
        # Sample up to 600 points evenly across risk tiers for crisp rendering
        if len(valid_168) > 600:
            valid_168 = valid_168.sample(n=600, random_state=42)
        actual_vs_pred = {
            "available": True,
            "actual": valid_168["value_168h"].round(3).tolist(),
            "predicted": valid_168["predicted_value_168h"].round(3).tolist(),
            "component_ids": valid_168["component_id"].tolist(),
            "risk_status": valid_168["risk_status"].tolist(),
            "min_val": round(
                float(min(valid_168["value_168h"].min(), valid_168["predicted_value_168h"].min())) - 1.0,
                2,
            ),
            "max_val": round(
                float(max(valid_168["value_168h"].max(), valid_168["predicted_value_168h"].max())) + 1.0,
                2,
            ),
        }

    # 5. Risk distribution across batches
    batch_sorted = sorted(batch_summary, key=lambda b: b["batch_id"])
    batch_risk = {
        "batches": [b["batch_id"] for b in batch_sorted],
        "normal": [b["normal"] for b in batch_sorted],
        "warning": [b["warning"] for b in batch_sorted],
        "anomalous": [b["anomalous"] for b in batch_sorted],
        "high_risk": [b["high_risk"] for b in batch_sorted],
    }

    return jsonify(
        {
            "run": latest,
            "anomaly_counts": anomaly_counts,
            "leakage_by_risk": leakage_by_risk,
            "drift_traces": drift_traces,
            "mean_trajectories": mean_trajectories,
            "actual_vs_pred": actual_vs_pred,
            "batch_risk": batch_risk,
        }
    )


@app.route("/components")
def components():
    ctx = _get_active_context()
    latest = ctx["latest_run"]
    df = get_run_components_df(latest["id"])
    batches = sorted(df["batch_id"].unique().tolist())
    return render_template("components.html", **ctx, batches=batches)


@app.route("/api/components")
def api_components():
    ctx = _get_active_context()
    latest = ctx["latest_run"]
    df = get_run_components_df(latest["id"])

    search = request.args.get("search", "").strip().lower()
    batch_filter = request.args.get("batch_id", "").strip()
    anomaly_filter = request.args.get("anomaly_status", "").strip()
    risk_filter = request.args.get("risk_status", "").strip()
    static_filter = request.args.get("static_pass", "").strip()
    sort_by = request.args.get("sort_by", "anomaly_score").strip()
    sort_dir = request.args.get("sort_dir", "desc").strip().lower()
    page = max(1, request.args.get("page", 1, type=int))
    per_page = min(100, max(10, request.args.get("per_page", 25, type=int)))

    if search:
        mask = (
            df["component_id"].str.lower().str.contains(search, na=False)
            | df["batch_id"].str.lower().str.contains(search, na=False)
            | df["device_type"].fillna("").str.lower().str.contains(search, na=False)
        )
        df = df[mask]
    if batch_filter:
        df = df[df["batch_id"] == batch_filter]
    if anomaly_filter:
        df = df[df["anomaly_status"] == anomaly_filter]
    if risk_filter:
        df = df[df["risk_status"] == risk_filter]
    if static_filter in ("1", "0"):
        df = df[df["static_pass"] == int(static_filter)]

    allowed_sort = {
        "component_id",
        "batch_id",
        "leakage_current_na",
        "value_0h",
        "predicted_value_168h",
        "drift_rate",
        "anomaly_score",
        "risk_status",
        "anomaly_status",
    }
    if sort_by not in allowed_sort:
        sort_by = "anomaly_score"
    ascending = sort_dir == "asc"
    df = df.sort_values(by=sort_by, ascending=ascending, kind="mergesort")

    total_filtered = int(len(df))
    total_pages = max(1, (total_filtered + per_page - 1) // per_page)
    page = min(page, total_pages)
    start = (page - 1) * per_page
    end = start + per_page

    page_df = df.iloc[start:end]
    records = []
    for _, r in page_df.iterrows():
        records.append(
            {
                "component_id": r["component_id"],
                "batch_id": r["batch_id"],
                "device_type": r["device_type"],
                "leakage_current_na": round(float(r["leakage_current_na"]), 2),
                "iddq_ua": round(float(r["iddq_ua"]), 2),
                "value_0h": round(float(r["value_0h"]), 2),
                "value_24h": round(float(r["value_24h"]), 2),
                "predicted_value_168h": round(float(r["predicted_value_168h"]), 2),
                "predicted_drift": round(float(r["predicted_drift"]), 2),
                "drift_rate": round(float(r["drift_rate"]), 5),
                "drift_flag": bool(r["drift_flag"]),
                "anomaly_score": round(float(r["anomaly_score"]), 2),
                "anomaly_status": r["anomaly_status"],
                "risk_status": r["risk_status"],
                "static_pass": bool(r["static_pass"]),
            }
        )

    return jsonify(
        {
            "total": total_filtered,
            "page": page,
            "per_page": per_page,
            "total_pages": total_pages,
            "items": records,
        }
    )


@app.route("/analysis")
@app.route("/analysis/<component_id>")
def analysis(component_id: str | None = None):
    ctx = _get_active_context()
    latest = ctx["latest_run"]
    df = get_run_components_df(latest["id"])

    if df.empty:
        flash("No component records found in the active screening run.", "warning")
        return redirect(url_for("index"))

    # Default to the highest-risk component so inspectors immediately see actionable XAI
    if not component_id:
        top_row = df.sort_values(by="anomaly_score", ascending=False).iloc[0]
        component_id = str(top_row["component_id"])

    detail = get_component_detail(latest["id"], component_id)
    if detail is None:
        flash(f"Component '{component_id}' was not found in the active dataset.", "danger")
        top_row = df.sort_values(by="anomaly_score", ascending=False).iloc[0]
        detail = get_component_detail(latest["id"], str(top_row["component_id"]))

    # Provide quick-switch list (top flagged + sample normal)
    flagged_list = (
        df[df["risk_status"] != "LOW RISK"]
        .sort_values(by="anomaly_score", ascending=False)
        .head(40)[["component_id", "batch_id", "anomaly_score", "risk_status"]]
        .to_dict(orient="records")
    )
    all_ids = df["component_id"].tolist()

    return render_template(
        "analysis.html",
        **ctx,
        detail=detail,
        flagged_list=flagged_list,
        all_ids=all_ids,
    )


@app.route("/reports")
def reports():
    ctx = _get_active_context()
    latest = ctx["latest_run"]
    batch_summary = get_batch_summary(latest["id"])
    history = get_run_history(limit=15)
    batches = [b["batch_id"] for b in sorted(batch_summary, key=lambda x: x["batch_id"])]
    return render_template(
        "reports.html",
        **ctx,
        batch_summary=batch_summary,
        run_history=history,
        batches=batches,
    )


@app.route("/upload", methods=["POST"])
def upload_dataset():
    if "file" not in request.files:
        flash("No file part provided in upload request.", "danger")
        return redirect(url_for("index"))

    file = request.files["file"]
    if not file or file.filename == "":
        flash("Please select a CSV file before uploading.", "warning")
        return redirect(url_for("index"))

    filename = secure_filename(file.filename)
    if not filename.lower().endswith(".csv"):
        flash("Invalid file type. Only .csv files are supported.", "danger")
        return redirect(url_for("index"))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_path = UPLOAD_DIR / f"{timestamp}_{filename}"
    file.save(saved_path)

    safety_slope = request.form.get("safety_slope", type=float) or DEFAULT_SAFETY_SLOPE
    contamination = request.form.get("contamination", type=float) or 0.14
    safety_slope = float(np.clip(safety_slope, 0.001, 1.0))
    contamination = float(np.clip(contamination, 0.02, 0.40))

    try:
        run_id = run_screening_pipeline(
            csv_path=saved_path,
            dataset_source=f"Uploaded CSV ({filename})",
            run_name=f"Upload Screening: {filename}",
            safety_slope=safety_slope,
            contamination=contamination,
            retrain=True,
            is_demo=False,
        )
        run_info = get_latest_run()
        flash(
            f"Dataset '{filename}' processed successfully (Run #{run_id}): "
            f"{run_info['total_components']} components screened, "
            f"{run_info['anomalous_count']} anomalous, {run_info['high_risk_count']} high-risk.",
            "success",
        )
        return redirect(url_for("dashboard"))
    except DatasetValidationError as exc:
        if saved_path.exists():
            saved_path.unlink(missing_ok=True)
        flash(f"CSV Validation Error: {exc}", "danger")
        return redirect(url_for("index"))
    except Exception as exc:
        flash(f"Unexpected error while processing dataset: {exc}", "danger")
        return redirect(url_for("index"))


@app.route("/run-analysis", methods=["POST"])
def run_analysis():
    latest = get_latest_run()
    csv_path = Path(latest["csv_path"]) if latest and latest.get("csv_path") else SAMPLE_CSV_PATH
    if not csv_path.exists():
        csv_path = SAMPLE_CSV_PATH

    safety_slope = request.form.get("safety_slope", type=float)
    contamination = request.form.get("contamination", type=float)
    if safety_slope is None:
        safety_slope = latest["safety_slope"] if latest else DEFAULT_SAFETY_SLOPE
    if contamination is None:
        contamination = latest["contamination"] if latest else 0.14

    safety_slope = float(np.clip(safety_slope, 0.001, 1.0))
    contamination = float(np.clip(contamination, 0.02, 0.40))

    try:
        run_id = run_screening_pipeline(
            csv_path=csv_path,
            dataset_source=latest["dataset_source"] if latest else "Synthetic Demo Dataset",
            run_name=f"Re-Analysis (Slope +/-{safety_slope:.4f}/hr)",
            safety_slope=safety_slope,
            contamination=contamination,
            retrain=True,
            is_demo=bool(latest["is_demo"]) if latest else True,
        )
        flash(
            f"Analysis completed (Run #{run_id}) with safety slope +/-{safety_slope:.4f}/hr "
            f"and anomaly contamination {contamination:.2f}.",
            "success",
        )
    except Exception as exc:
        flash(f"Failed to run analysis: {exc}", "danger")

    next_page = request.form.get("next") or request.referrer or url_for("dashboard")
    return redirect(next_page)


@app.route("/reset-demo", methods=["POST"])
def reset_demo():
    try:
        run_id = reset_demo_dataset(safety_slope=DEFAULT_SAFETY_SLOPE, contamination=0.14)
        flash(
            f"Demo dataset regenerated (1,200 synthetic components) and both ML models retrained (Run #{run_id}).",
            "success",
        )
    except Exception as exc:
        flash(f"Failed to reset demo dataset: {exc}", "danger")
    return redirect(url_for("dashboard"))


@app.route("/export/csv")
def export_csv():
    ctx = _get_active_context()
    latest = ctx["latest_run"]
    df = get_run_components_df(latest["id"])

    batch_filter = request.args.get("batch_id", "").strip()
    risk_filter = request.args.get("risk_status", "").strip()
    if batch_filter:
        df = df[df["batch_id"] == batch_filter]
    if risk_filter:
        df = df[df["risk_status"] == risk_filter]

    export_cols = [
        "component_id",
        "batch_id",
        "device_type",
        "iddq_ua",
        "leakage_current_na",
        "propagation_delay_ns",
        "temperature_c",
        "value_0h",
        "value_24h",
        "value_96h",
        "value_168h",
        "predicted_value_168h",
        "predicted_drift",
        "drift_rate",
        "drift_flag",
        "anomaly_score",
        "anomaly_status",
        "risk_status",
        "static_pass",
        "explanation_summary",
    ]
    export_df = df[[c for c in export_cols if c in df.columns]].copy()
    csv_data = export_df.to_csv(index=False)

    suffix = f"_{batch_filter}" if batch_filter else ""
    filename = f"sentinelai_screening_run{latest['id']}{suffix}.csv"
    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@app.route("/export/pdf")
def export_pdf():
    ctx = _get_active_context()
    latest = ctx["latest_run"]
    df = get_run_components_df(latest["id"])
    batch_summary = get_batch_summary(latest["id"])

    batch_filter = request.args.get("batch_id", "").strip() or None
    if batch_filter:
        df = df[df["batch_id"] == batch_filter]
        batch_summary = [b for b in batch_summary if b["batch_id"] == batch_filter]

    flagged_df = df[df["risk_status"] != "LOW RISK"]
    pdf_bytes = generate_pdf_report(
        run_info=latest,
        batch_summary=batch_summary,
        flagged_df=flagged_df,
        batch_filter=batch_filter,
    )
    suffix = f"_{batch_filter}" if batch_filter else ""
    filename = f"sentinelai_qa_report_run{latest['id']}{suffix}.pdf"
    return send_file(
        io.BytesIO(pdf_bytes),
        mimetype="application/pdf",
        as_attachment=True,
        download_name=filename,
    )


@app.errorhandler(413)
def handle_large_file(error):
    flash("Uploaded file exceeds the 16 MB size limit.", "danger")
    return redirect(url_for("index"))


@app.errorhandler(404)
def handle_not_found(error):
    return render_template("index.html", **_get_active_context(), run_history=get_run_history(limit=6)), 404


if __name__ == "__main__":
    ensure_initialized()
    app.run(host="127.0.0.1", port=5000, debug=False)
