import os
import json
import yaml
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder


print("=" * 65)
print("PREPROCESSING - POWER CONSUMPTION DATA")
print("=" * 65)

# ── Load config ───────────────────────────────────────────────
with open("params.yaml") as f:
    params = yaml.safe_load(f)

raw_path       = params["data"]["raw_path"]
new_data_path  = params["data"]["new_data_path"]
train_out      = params["data"]["processed_train_path"]
test_out       = params["data"]["processed_test_path"]
target_col     = params["preprocessing"]["target_column"]
test_size      = params["preprocessing"]["test_size"]
random_state   = params["preprocessing"]["random_state"]
bins           = params["preprocessing"]["power_bins"]
labels         = params["preprocessing"]["power_labels"]
min_new_rows   = params["retraining"]["min_new_rows"]

os.makedirs("artifacts/preprocessing", exist_ok=True)
os.makedirs("artifacts", exist_ok=True)

# ── Merge new data if enough rows exist ───────────────────────
df = pd.read_csv(raw_path)
print(f"Base dataset loaded: {len(df)} rows")

if os.path.exists(new_data_path):
    new_df = pd.read_csv(new_data_path)
    if len(new_df) >= min_new_rows:
        df = pd.concat([df, new_df], ignore_index=True).drop_duplicates()
        print(f"Merged new data — total rows: {len(df)}")
    else:
        print(f"New data skipped — only {len(new_df)} rows "
              f"(minimum {min_new_rows} required)")
else:
    print("No new data file found — using base dataset only")

# ── Parse datetime → time features ───────────────────────────
df["DateTime"] = pd.to_datetime(df["DateTime"], dayfirst=True, errors="coerce")
df = df.dropna(subset=["DateTime"])

df["Hour"]      = df["DateTime"].dt.hour
df["DayOfWeek"] = df["DateTime"].dt.dayofweek
df["Month"]     = df["DateTime"].dt.month
df["IsWeekend"] = (df["DayOfWeek"] >= 5).astype(int)

# Cyclic encoding for hour (so 23 and 0 are treated as close)
df["HourSin"]   = np.sin(2 * np.pi * df["Hour"] / 24)
df["HourCos"]   = np.cos(2 * np.pi * df["Hour"] / 24)

df = df.drop(columns=["DateTime", "Hour"])

# ── Target: bin Zone 1 power into Low / Medium / High ─────────
df[target_col] = pd.cut(
    df["Zone 1 Power Consumption"],
    bins=bins,
    labels=labels
)

df = df.drop(columns=[
    "Zone 1 Power Consumption",
    "Zone 2  Power Consumption",
    "Zone 3  Power Consumption",
])

df = df.dropna().drop_duplicates()
print(f"After cleaning: {len(df)} rows")

# ── Outlier removal (IQR) ─────────────────────────────────────
numeric_cols = df.select_dtypes(include="number").columns
before = len(df)
for col in numeric_cols:
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    df = df[(df[col] >= q1 - 1.5 * iqr) & (df[col] <= q3 + 1.5 * iqr)]
print(f"After outlier removal: {len(df)} rows (removed {before - len(df)})")

# ── Features / target split ───────────────────────────────────
feature_cols = [c for c in df.columns if c != target_col]
X = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
y = df[target_col]

le = LabelEncoder()
y_encoded = le.fit_transform(y)

joblib.dump(le, "artifacts/preprocessing/label_encoder.pkl")
print(f"Classes: {list(le.classes_)} → {list(range(len(le.classes_)))}")

# Save feature schema for alignment on new data
with open("artifacts/preprocessing/feature_columns.json", "w") as f:
    json.dump(feature_cols, f, indent=4)
print(f"Saved feature schema ({len(feature_cols)} features)")

# ── Train / test split (stratified) ──────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X.values, y_encoded,
    test_size=test_size,
    random_state=random_state,
    stratify=y_encoded,
)

# ── Scaling ───────────────────────────────────────────────────
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

joblib.dump(scaler, "artifacts/preprocessing/scaler.pkl")
print("Saved scaler.pkl")

# ── Save processed CSVs (for evaluate.py and monitoring) ─────
train_df_out = pd.DataFrame(X_train_scaled, columns=feature_cols)
train_df_out[target_col] = y_train
train_df_out.to_csv(train_out, index=False)

test_df_out = pd.DataFrame(X_test_scaled, columns=feature_cols)
test_df_out[target_col] = y_test
test_df_out.to_csv(test_out, index=False)

print(f"\nTrain: {len(y_train)} samples | Test: {len(y_test)} samples")
print("Preprocessing complete!")
