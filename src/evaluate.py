import os
import json
import yaml
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report,
)


print("=" * 65)
print("EVALUATION - POWER CONSUMPTION CLASSIFIER")
print("=" * 65)

# ── Load config ───────────────────────────────────────────────
with open("params.yaml") as f:
    params = yaml.safe_load(f)

test_path    = params["data"]["processed_test_path"]
target_col   = params["preprocessing"]["target_column"]
bundle_path  = params["outputs"]["model_bundle"]
metrics_path = params["outputs"]["metrics_path"]

os.makedirs("artifacts/metrics", exist_ok=True)

# ── Load test data and model bundle ──────────────────────────
with open("artifacts/preprocessing/feature_columns.json") as f:
    feature_cols = json.load(f)

test_df = pd.read_csv(test_path)
X_test  = test_df[feature_cols].values
y_test  = test_df[target_col].values

bundle       = joblib.load(bundle_path)
model        = bundle["model"]
class_labels = bundle["class_labels"]

# ── Predictions ───────────────────────────────────────────────
y_pred = model.predict(X_test)

# ── Metrics ───────────────────────────────────────────────────
accuracy  = accuracy_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred, average="weighted")
precision = precision_score(y_test, y_pred, average="weighted")
recall    = recall_score(y_test, y_pred, average="weighted")
report    = classification_report(
    y_test, y_pred, target_names=class_labels, output_dict=True
)

print(f"Accuracy  : {accuracy:.4f}")
print(f"F1-score  : {f1:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print("\nPer-class breakdown:")
for label in class_labels:
    m = report.get(label, {})
    print(f"  {label:8s} — precision: {m.get('precision', 0):.3f}  "
          f"recall: {m.get('recall', 0):.3f}  "
          f"f1: {m.get('f1-score', 0):.3f}")

# ── Save metrics JSON ─────────────────────────────────────────
metrics_out = {
    "timestamp":  datetime.now().isoformat(timespec="seconds"),
    "model":      bundle["model_name"],
    "n_test":     int(len(y_test)),
    "metrics": {
        "accuracy":  float(accuracy),
        "f1_score":  float(f1),
        "precision": float(precision),
        "recall":    float(recall),
    },
    "per_class": {
        label: {
            "precision": float(report[label]["precision"]),
            "recall":    float(report[label]["recall"]),
            "f1_score":  float(report[label]["f1-score"]),
        }
        for label in class_labels if label in report
    },
}

with open(metrics_path, "w") as f:
    json.dump(metrics_out, f, indent=4)
print(f"\nSaved metrics → {metrics_path}")

# ── Plot: predicted vs actual ─────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))

ax.scatter(y_test, y_pred, alpha=0.4, s=20, color="steelblue")

# Diagonal perfect-prediction line
mn = min(y_test.min(), y_pred.min())
mx = max(y_test.max(), y_pred.max())
ax.plot([mn, mx], [mn, mx], linestyle="--", linewidth=1.5, color="steelblue")

ax.set_xlabel("Actual Class")
ax.set_ylabel("Predicted Class")
ax.set_title(
    f"Predicted vs Actual\n"
    f"Accuracy = {accuracy:.4f}, F1 = {f1:.4f}"
)
ax.set_xticks(range(len(class_labels)))
ax.set_yticks(range(len(class_labels)))
ax.set_xticklabels(class_labels)
ax.set_yticklabels(class_labels)
ax.grid(True, linestyle="--", alpha=0.4)

plt.tight_layout()
eval_plot = "artifacts/metrics/evaluation_results.png"
plt.savefig(eval_plot, dpi=120, bbox_inches="tight")
plt.close()
print(f"Saved evaluation plot → {eval_plot}")

print("\n" + "=" * 65)
print("EVALUATION COMPLETE")
print("=" * 65)