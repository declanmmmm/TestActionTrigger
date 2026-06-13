import os
import json
import yaml
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


print("=" * 65)
print("MODEL TRAINING - Random Forest vs KNN")
print("=" * 65)

# ── Load config ───────────────────────────────────────────────
with open("params.yaml") as f:
    params = yaml.safe_load(f)

train_path    = params["data"]["processed_train_path"]
target_col    = params["preprocessing"]["target_column"]
bundle_path   = params["outputs"]["model_bundle"]
plot_path     = params["outputs"]["comparison_plot"]
rf_params     = params["models"]["random_forest"]
knn_params    = params["models"]["knn"]

os.makedirs("artifacts/models", exist_ok=True)
os.makedirs("artifacts/metadata", exist_ok=True)

# ── Load training data ────────────────────────────────────────
with open("artifacts/preprocessing/feature_columns.json") as f:
    feature_cols = json.load(f)

train_df = pd.read_csv(train_path)
X_train  = train_df[feature_cols].values
y_train  = train_df[target_col].values

print(f"Training samples: {len(X_train)}  |  Features: {len(feature_cols)}")

# ── Random Forest ─────────────────────────────────────────────
print("\nTraining Random Forest...")
rf = RandomForestClassifier(
    n_estimators = rf_params["n_estimators"],
    max_depth    = rf_params["max_depth"],
    random_state = rf_params["random_state"],
    n_jobs       = -1,
)
rf.fit(X_train, y_train)
rf_train_pred = rf.predict(X_train)
rf_train_acc  = accuracy_score(y_train, rf_train_pred)
print(f"  Train accuracy: {rf_train_acc:.4f}")

# ── KNN ───────────────────────────────────────────────────────
print("\nTraining KNN...")
knn = KNeighborsClassifier(
    n_neighbors = knn_params["n_neighbors"],
    metric      = knn_params["metric"],
)
knn.fit(X_train, y_train)
knn_train_pred = knn.predict(X_train)
knn_train_acc  = accuracy_score(y_train, knn_train_pred)
print(f"  Train accuracy: {knn_train_acc:.4f}")

# ── Select best model ─────────────────────────────────────────
print("\nComparing models on training accuracy...")
if rf_train_acc >= knn_train_acc:
    best_model, best_name = rf, "Random Forest"
else:
    best_model, best_name = knn, "KNN"

print(f"  Selected: {best_name}")

# ── Feature importances (RF only) ─────────────────────────────
feature_importances = None
if best_name == "Random Forest":
    feature_importances = {
        col: float(imp)
        for col, imp in zip(feature_cols, rf.feature_importances_)
    }

# ── Accuracy comparison plot ──────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(
    ["Random Forest", "KNN"],
    [rf_train_acc, knn_train_acc],
    color=["steelblue", "darkorange"],
    width=0.5,
)
ax.set_ylabel("Training Accuracy")
ax.set_title("Model Comparison — Power Consumption Classification")
ax.set_ylim(0, 1.05)
for bar, acc in zip(bars, [rf_train_acc, knn_train_acc]):
    ax.text(bar.get_x() + bar.get_width() / 2, acc + 0.01,
            f"{acc:.4f}", ha="center", fontsize=11)
plt.tight_layout()
plt.savefig(plot_path, dpi=120)
plt.close()
print(f"Saved comparison plot → {plot_path}")

# ── Save deployment bundle ────────────────────────────────────
le = joblib.load("artifacts/preprocessing/label_encoder.pkl")

bundle = {
    "model":              best_model,
    "model_name":         best_name,
    "feature_columns":    feature_cols,
    "class_labels":       list(le.classes_),
    "all_models": {
        "Random Forest": {"train_accuracy": float(rf_train_acc)},
        "KNN":           {"train_accuracy": float(knn_train_acc)},
    },
    "hyperparameters": {
        "random_forest": rf_params,
        "knn":           knn_params,
    },
    "feature_importances": feature_importances,
    "training_metadata": {
        "n_train":    int(len(X_train)),
        "n_features": len(feature_cols),
        "trained_at": datetime.now().isoformat(timespec="seconds"),
    },
}

joblib.dump(bundle, bundle_path)
print(f"Saved model bundle → {bundle_path}")

print("\n" + "=" * 65)
print("TRAINING COMPLETE")
print("=" * 65)
