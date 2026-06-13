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
    classification_report, ConfusionMatrixDisplay, confusion_matrix,
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

# ── Plots: confusion matrix + feature importance ──────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
disp.plot(ax=axes[0], colorbar=False)
axes[0].set_title(f"Confusion Matrix\n{bundle['model_name']} — Accuracy {accuracy:.4f}")

# Feature importance (RF) or bar of ones (KNN placeholder)
if bundle.get("feature_importances"):
    importances = bundle["feature_importances"]
    feats  = list(importances.keys())
    values = list(importances.values())
    order  = np.argsort(values)
    axes[1].barh([feats[i] for i in order], [values[i] for i in order],
                 color="steelblue")
    axes[1].set_xlabel("Importance")
    axes[1].set_title("Feature Importances (Random Forest)")
    axes[1].grid(axis="x", linestyle="--", alpha=0.4)
else:
    axes[1].text(0.5, 0.5, "Feature importance\nnot available for KNN",
                 ha="center", va="center", transform=axes[1].transAxes)
    axes[1].set_title("Feature Importances")

plt.tight_layout()
eval_plot = "artifacts/metrics/evaluation_results.png"
plt.savefig(eval_plot, dpi=120, bbox_inches="tight")
plt.close()
print(f"Saved evaluation plot → {eval_plot}")

print("\n" + "=" * 65)
print("EVALUATION COMPLETE")
print("=" * 65)
