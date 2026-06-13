import os
import json
import yaml
import joblib
import numpy as np
import pandas as pd

from datetime import datetime
from scipy.stats import ks_2samp
from sklearn.metrics import accuracy_score, f1_score


print("=" * 65)
print("MONITORING - POWER CONSUMPTION CLASSIFIER")
print("=" * 65)

# ── Load config ───────────────────────────────────────────────
with open("params.yaml") as f:
    params = yaml.safe_load(f)

raw_path         = params["data"]["raw_path"]
new_data_path    = params["data"]["new_data_path"]
test_path        = params["data"]["processed_test_path"]
metrics_path     = params["outputs"]["metrics_path"]
bundle_path      = params["outputs"]["model_bundle"]
target_col       = params["preprocessing"]["target_column"]
acc_threshold    = params["monitoring"]["accuracy_threshold"]
f1_threshold     = params["monitoring"]["f1_threshold"]
ks_threshold     = params["monitoring"]["ks_p_value"]

os.makedirs("monitoring/reports", exist_ok=True)
os.makedirs("monitoring/logs",    exist_ok=True)
os.makedirs("monitoring/alerts",  exist_ok=True)

# ── 1. Model performance check ────────────────────────────────
print("\n[1] Model performance check")

with open("artifacts/preprocessing/feature_columns.json") as f:
    feature_cols = json.load(f)

test_df = pd.read_csv(test_path)
X_test  = test_df[feature_cols].values
y_test  = test_df[target_col].values

bundle = joblib.load(bundle_path)
model  = bundle["model"]
y_pred = model.predict(X_test)

accuracy = float(accuracy_score(y_test, y_pred))
f1       = float(f1_score(y_test, y_pred, average="weighted"))

print(f"  Accuracy : {accuracy:.4f}  (threshold: {acc_threshold})")
print(f"  F1-score : {f1:.4f}  (threshold: {f1_threshold})")

perf_ok = accuracy >= acc_threshold and f1 >= f1_threshold

# ── 2. Data drift detection (KS test) ────────────────────────
print("\n[2] Data drift detection (Kolmogorov-Smirnov test)")

base_df = pd.read_csv(raw_path)

# Parse and engineer the same features as preprocess.py
base_df["DateTime"] = pd.to_datetime(base_df["DateTime"], dayfirst=True, errors="coerce")
base_df = base_df.dropna(subset=["DateTime"])
base_df["Hour"]      = base_df["DateTime"].dt.hour
base_df["DayOfWeek"] = base_df["DateTime"].dt.dayofweek
base_df["Month"]     = base_df["DateTime"].dt.month
base_df["IsWeekend"] = (base_df["DayOfWeek"] >= 5).astype(int)
base_df["HourSin"]   = np.sin(2 * np.pi * base_df["Hour"] / 24)
base_df["HourCos"]   = np.cos(2 * np.pi * base_df["Hour"] / 24)
base_df = base_df.drop(columns=["DateTime", "Hour",
                                  "Zone 1 Power Consumption",
                                  "Zone 2  Power Consumption",
                                  "Zone 3  Power Consumption"], errors="ignore")

drift_results  = {}
drift_detected = False
alerts         = []

if not os.path.exists(new_data_path):
    print("  No new data file found — drift check skipped")
    alerts.append("No new data file found. Drift check was skipped.")
else:
    new_df = pd.read_csv(new_data_path)
    new_df["DateTime"] = pd.to_datetime(new_df["DateTime"], dayfirst=True, errors="coerce")
    new_df = new_df.dropna(subset=["DateTime"])
    new_df["Hour"]      = new_df["DateTime"].dt.hour
    new_df["DayOfWeek"] = new_df["DateTime"].dt.dayofweek
    new_df["Month"]     = new_df["DateTime"].dt.month
    new_df["IsWeekend"] = (new_df["DayOfWeek"] >= 5).astype(int)
    new_df["HourSin"]   = np.sin(2 * np.pi * new_df["Hour"] / 24)
    new_df["HourCos"]   = np.cos(2 * np.pi * new_df["Hour"] / 24)
    new_df = new_df.drop(columns=["DateTime", "Hour",
                                    "Zone 1 Power Consumption",
                                    "Zone 2  Power Consumption",
                                    "Zone 3  Power Consumption"], errors="ignore")

    numeric_cols = base_df.select_dtypes(include="number").columns.tolist()

    for col in numeric_cols:
        if col not in new_df.columns:
            continue
        stat, p_value = ks_2samp(base_df[col].dropna(), new_df[col].dropna())
        has_drift = p_value < ks_threshold
        drift_results[col] = {
            "ks_statistic": float(stat),
            "p_value":      float(p_value),
            "drift_detected": bool(has_drift),
        }
        if has_drift:
            drift_detected = True

    drifted_cols = [c for c, v in drift_results.items() if v["drift_detected"]]
    if drift_detected:
        alerts.append(f"Data drift detected in: {', '.join(drifted_cols)}")
        print(f"  Drift detected in: {drifted_cols}")
    else:
        print("  No significant drift detected")

# ── 3. Performance alerts ─────────────────────────────────────
if not perf_ok:
    alerts.append(
        f"Model performance below threshold — "
        f"accuracy={accuracy:.4f} (>={acc_threshold}), "
        f"f1={f1:.4f} (>={f1_threshold})"
    )

retraining_required = drift_detected or not perf_ok

status = "WARNING" if retraining_required else "OK"
print(f"\n  Status: [{status}]")
if alerts:
    for a in alerts:
        print(f"  ⚠  {a}")
else:
    print("  No alerts — model is healthy")

# ── 4. Save reports ───────────────────────────────────────────
timestamp = datetime.now().isoformat(timespec="seconds")

drift_report = {
    "timestamp":           timestamp,
    "model":               bundle["model_name"],
    "performance": {
        "accuracy":        accuracy,
        "f1_score":        f1,
        "accuracy_threshold": acc_threshold,
        "f1_threshold":    f1_threshold,
        "performance_ok":  perf_ok,
    },
    "drift_detection": {
        "method":          "Kolmogorov-Smirnov test",
        "ks_threshold":    ks_threshold,
        "drift_detected":  drift_detected,
        "feature_results": drift_results,
    },
    "alerts":              alerts,
    "retraining_required": retraining_required,
}

with open("monitoring/reports/drift_report.json", "w") as f:
    json.dump(drift_report, f, indent=4)

with open("monitoring/alerts/alerts.json", "w") as f:
    json.dump({"timestamp": timestamp, "alerts": alerts}, f, indent=4)

with open("monitoring/logs/monitoring.log", "a") as f:
    f.write(json.dumps({
        "timestamp": timestamp,
        "status":    status,
        "accuracy":  accuracy,
        "f1_score":  f1,
        "drift":     drift_detected,
        "alerts":    len(alerts),
    }) + "\n")

print(f"\nSaved monitoring reports.")
print("=" * 65)
