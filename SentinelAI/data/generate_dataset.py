"""
Synthetic Burn-In & Screening Dataset Generator for SentinelAI.

IMPORTANT NOTICE:
This module generates 100% synthetic demonstration data simulating electronic
components undergoing 168-hour High-Temperature Operating Life (HTOL) / Burn-In
screening. It is NOT actual ISRO flight or qualification test data.
"""

from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pandas as pd

# Static datasheet pass/fail limits used for traditional screening comparison
STATIC_DATASHEET_LIMITS = {
    "iddq_ua_max": 45.0,
    "leakage_current_na_max": 250.0,
    "propagation_delay_ns_max": 16.0,
    "temperature_c_max": 135.0,
    "value_min": 85.0,
    "value_max": 130.0,
}


def generate_synthetic_dataset(
    n_components: int = 1200,
    n_batches: int = 12,
    seed: int = 42,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Generate a reproducible synthetic burn-in screening dataset.

    Categories simulated:
    - Normal (~76%): Stable electrical parameters and saturating low drift.
    - High_Leakage (~6%): Elevated leakage & standby current (some within static limits).
    - Gradual_Drift (~7%): Normal 0h baseline, steadily accelerating parametric drift.
    - Sudden_Shift (~5%): Abrupt step-shift in parametric output & delay.
    - Latent_Defect (~6%): Passes ALL static datasheet limits, yet displays
      abnormal multivariate behavior and elevated drift trajectory.
    """
    if n_components < 1000:
        raise ValueError("n_components must be at least 1,000 per specification.")

    rng = np.random.default_rng(seed)
    device_families = ["RH-CMOS-74HC", "SPACE-LDO-3V3", "RAD-SRAM-4M", "AS-MUX-16CH"]

    # Assign batches roughly evenly, with slight realistic batch-level variation
    batch_ids = [f"BATCH-{i + 1:02d}" for i in range(n_batches)]
    batch_assignments = [batch_ids[i % n_batches] for i in range(n_components)]
    rng.shuffle(batch_assignments)

    # Slight wafer/lot-level baseline offsets per batch
    batch_v0_offset = {b: rng.normal(0.0, 0.35) for b in batch_ids}
    batch_leak_offset = {b: rng.normal(0.0, 3.0) for b in batch_ids}

    # Inject slightly higher defect concentration in specific batches (e.g., BATCH-04, BATCH-09)
    # so batch-level quality reports look realistic and actionable.
    categories = ["Normal", "High_Leakage", "Gradual_Drift", "Sudden_Shift", "Latent_Defect"]
    base_probs = np.array([0.76, 0.06, 0.07, 0.05, 0.06])
    troubled_probs = np.array([0.52, 0.12, 0.14, 0.10, 0.12])

    records = []
    for idx in range(n_components):
        comp_id = f"CMP-{idx + 1:04d}"
        batch_id = batch_assignments[idx]
        device_type = device_families[idx % len(device_families)]
        probs = troubled_probs if batch_id in {"BATCH-04", "BATCH-09"} else base_probs
        defect_type = rng.choice(categories, p=probs)

        v0_base = 100.0 + batch_v0_offset[batch_id] + rng.normal(0.0, 1.5)
        leak_base = 62.0 + batch_leak_offset[batch_id]

        if defect_type == "Normal":
            iddq = float(np.clip(rng.normal(16.0, 2.0), 8.0, 26.0))
            leakage = float(np.clip(rng.normal(leak_base, 9.0), 25.0, 105.0))
            prop_delay = float(np.clip(rng.normal(10.2, 0.5), 8.2, 12.4))
            temp = float(np.clip(rng.normal(125.0, 0.7), 122.0, 128.0))

            # Early 24h shift is small and stabilizes logarithmically by 168h
            delta_24 = float(np.clip(rng.normal(0.35, 0.18), 0.02, 0.90))
            val_0 = v0_base
            val_24 = val_0 + delta_24
            val_96 = val_0 + delta_24 * 2.15 + rng.normal(0.0, 0.18)
            val_168 = val_0 + delta_24 * 2.75 + rng.normal(0.0, 0.24)

        elif defect_type == "High_Leakage":
            # Mix of out-of-spec and borderline high leakage components
            iddq = float(np.clip(rng.normal(38.0, 6.5), 24.0, 58.0))
            leakage = float(np.clip(rng.normal(235.0, 38.0), 165.0, 340.0))
            prop_delay = float(np.clip(rng.normal(11.8, 0.9), 9.5, 15.0))
            temp = float(np.clip(rng.normal(129.2, 1.4), 125.5, 134.5))

            delta_24 = float(np.clip(rng.normal(1.25, 0.35), 0.60, 2.30))
            val_0 = v0_base + rng.normal(0.6, 0.8)
            val_24 = val_0 + delta_24
            val_96 = val_0 + delta_24 * 2.90 + rng.normal(0.0, 0.35)
            val_168 = val_0 + delta_24 * 3.85 + rng.normal(0.0, 0.45)

        elif defect_type == "Gradual_Drift":
            # Electrical baseline looks mostly normal at 0h, steady accelerating drift
            iddq = float(np.clip(rng.normal(21.5, 2.6), 14.0, 31.0))
            leakage = float(np.clip(rng.normal(102.0, 15.0), 65.0, 155.0))
            prop_delay = float(np.clip(rng.normal(11.4, 0.7), 9.5, 13.8))
            temp = float(np.clip(rng.normal(126.5, 0.9), 124.0, 129.5))

            delta_24 = float(np.clip(rng.normal(2.25, 0.40), 1.40, 3.40))
            val_0 = v0_base
            val_24 = val_0 + delta_24
            val_96 = val_0 + delta_24 * 3.40 + rng.normal(0.0, 0.45)
            val_168 = val_0 + delta_24 * 5.10 + rng.normal(0.0, 0.60)

        elif defect_type == "Sudden_Shift":
            # Abrupt jump in delay, Iddq, and early parametric shift
            iddq = float(np.clip(rng.normal(31.0, 4.5), 21.0, 44.0))
            leakage = float(np.clip(rng.normal(148.0, 25.0), 95.0, 235.0))
            prop_delay = float(np.clip(rng.normal(14.3, 1.1), 12.2, 17.8))
            temp = float(np.clip(rng.normal(128.0, 1.3), 125.0, 132.5))

            delta_24 = float(np.clip(rng.normal(3.30, 0.60), 2.10, 4.90))
            val_0 = v0_base
            val_24 = val_0 + delta_24
            val_96 = val_0 + delta_24 * 3.15 + rng.normal(0.2, 0.65)
            val_168 = val_0 + delta_24 * 3.95 + rng.normal(0.3, 0.85)

        else:  # Latent_Defect — guaranteed to pass ALL static datasheet limits!
            iddq = float(np.clip(rng.normal(29.5, 2.8), 22.0, 39.0))           # < 45 uA limit
            leakage = float(np.clip(rng.normal(158.0, 18.0), 115.0, 215.0))   # < 250 nA limit
            prop_delay = float(np.clip(rng.normal(12.9, 0.6), 11.4, 14.8))    # < 16 ns limit
            temp = float(np.clip(rng.normal(127.8, 0.8), 125.5, 131.0))       # < 135 C limit

            delta_24 = float(np.clip(rng.normal(1.65, 0.25), 1.15, 2.25))
            val_0 = float(np.clip(v0_base, 94.0, 105.0))
            val_24 = val_0 + delta_24
            val_96 = val_0 + delta_24 * 3.20 + rng.normal(0.0, 0.30)
            val_168 = float(np.clip(val_0 + delta_24 * 4.55 + rng.normal(0.0, 0.40), 104.0, 126.0))

        # Evaluate traditional static pass/fail limits
        static_pass = (
            iddq <= STATIC_DATASHEET_LIMITS["iddq_ua_max"]
            and leakage <= STATIC_DATASHEET_LIMITS["leakage_current_na_max"]
            and prop_delay <= STATIC_DATASHEET_LIMITS["propagation_delay_ns_max"]
            and temp <= STATIC_DATASHEET_LIMITS["temperature_c_max"]
            and STATIC_DATASHEET_LIMITS["value_min"] <= val_0 <= STATIC_DATASHEET_LIMITS["value_max"]
            and STATIC_DATASHEET_LIMITS["value_min"] <= val_24 <= STATIC_DATASHEET_LIMITS["value_max"]
            and STATIC_DATASHEET_LIMITS["value_min"] <= val_96 <= STATIC_DATASHEET_LIMITS["value_max"]
            and STATIC_DATASHEET_LIMITS["value_min"] <= val_168 <= STATIC_DATASHEET_LIMITS["value_max"]
        )

        records.append(
            {
                "component_id": comp_id,
                "batch_id": batch_id,
                "device_type": device_type,
                "iddq_ua": round(iddq, 3),
                "leakage_current_na": round(leakage, 3),
                "propagation_delay_ns": round(prop_delay, 3),
                "temperature_c": round(temp, 2),
                "Value_0h": round(val_0, 3),
                "Value_24h": round(val_24, 3),
                "Value_96h": round(val_96, 3),
                "Value_168h": round(val_168, 3),
                "static_pass": bool(static_pass),
                "synthetic_defect_type": defect_type,
            }
        )

    df = pd.DataFrame(records)

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)

    return df


if __name__ == "__main__":
    default_csv = Path(__file__).resolve().parent / "sample_components.csv"
    df_out = generate_synthetic_dataset(n_components=1200, output_path=default_csv)
    print(f"Generated {len(df_out)} synthetic component records at: {default_csv}")
    print("Synthetic defect distribution:")
    print(df_out["synthetic_defect_type"].value_counts().to_string())
    latent = df_out[df_out["synthetic_defect_type"] == "Latent_Defect"]
    print(f"Latent defects passing static limits: {latent['static_pass'].sum()}/{len(latent)}")
