import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Power Consumption Classifier",
    page_icon="⚡",
    layout="wide",
)

# ── Load model bundle ─────────────────────────────────────────
@st.cache_resource
def load_bundle():
    return joblib.load("artifacts/models/model_bundle.joblib")

bundle       = load_bundle()
model        = bundle["model"]
feature_cols = bundle["feature_columns"]
class_labels = bundle["class_labels"]   # ['High', 'Low', 'Medium']

# ── Session state for history ─────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

# ── Header ────────────────────────────────────────────────────
st.title("⚡ Power Consumption Classifier")
st.markdown(
    f"Predicts whether Zone 1 power consumption is **Low**, **Medium**, or **High** "
    f"using a **{bundle['model_name']}** model."
)

tab1, tab2, tab3 = st.tabs(["Predict", "Model Info", "History"])

# ═══════════════════════════════════════
# TAB 1 — Predict
# ═══════════════════════════════════════
with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input Features")

        temperature = st.number_input("Temperature (°C)", value=18.0, step=0.5)
        humidity    = st.slider("Humidity (%)", 0, 100, 65)
        wind_speed  = st.number_input("Wind Speed (m/s)", value=0.5, min_value=0.0, step=0.1)
        gen_diff    = st.number_input("General Diffuse Flows", value=0.1, min_value=0.0, step=0.01)
        diff_flows  = st.number_input("Diffuse Flows", value=0.1, min_value=0.0, step=0.01)
        hour        = st.slider("Hour of Day", 0, 23, 12)
        day_of_week = st.selectbox("Day of Week", ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
        month       = st.slider("Month", 1, 12, 6)

        dow_index  = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"].index(day_of_week)
        is_weekend = int(dow_index >= 5)
        hour_sin   = np.sin(2 * np.pi * hour / 24)
        hour_cos   = np.cos(2 * np.pi * hour / 24)

    with col2:
        st.subheader("Prediction")

        if st.button("Classify Power Consumption", use_container_width=True):
            input_data = pd.DataFrame([[
                temperature, humidity, wind_speed,
                gen_diff, diff_flows,
                dow_index, month, is_weekend,
                hour_sin, hour_cos,
            ]], columns=feature_cols)

            pred_idx  = model.predict(input_data)[0]
            pred_label = class_labels[pred_idx]

            colour_map = {"Low": "#28a745", "Medium": "#ffc107", "High": "#dc3545"}
            width_map  = {"Low": 33, "Medium": 66, "High": 100}
            colour = colour_map.get(pred_label, "#6c757d")
            width  = width_map.get(pred_label, 50)

            st.success(f"Predicted Category: **{pred_label}**")

            st.markdown(
                f"""
                <div style="margin-top:12px">
                  <p style="margin-bottom:4px;font-weight:600">Consumption Level</p>
                  <div style="width:100%;height:26px;background:#e0e0e0;
                              border-radius:13px;overflow:hidden">
                    <div style="width:{width}%;height:100%;
                                background:{colour};border-radius:13px"></div>
                  </div>
                  <p style="color:{colour};font-size:22px;
                             font-weight:700;margin-top:8px">{pred_label}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.session_state.history.append({
                "Temperature": temperature,
                "Humidity":    humidity,
                "Hour":        hour,
                "Day":         day_of_week,
                "Month":       month,
                "Prediction":  pred_label,
            })

# ═══════════════════════════════════════
# TAB 2 — Model Info
# ═══════════════════════════════════════
with tab2:
    st.subheader("Model Details")

    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Model Type",    bundle["model_name"])
        st.metric("Training Samples", bundle["training_metadata"]["n_train"])
        st.metric("Features",      bundle["training_metadata"]["n_features"])

    with col_b:
        st.metric("Classes", ", ".join(class_labels))
        st.write("**Hyperparameters**")
        key = "random_forest" if bundle["model_name"] == "Random Forest" else "knn"
        st.json(bundle["hyperparameters"][key])

    if bundle.get("feature_importances"):
        st.subheader("Feature Importances")
        imp_df = (
            pd.DataFrame
            .from_dict(bundle["feature_importances"], orient="index", columns=["Importance"])
            .sort_values("Importance", ascending=False)
        )
        st.bar_chart(imp_df)

    st.subheader("All Models Compared")
    st.json(bundle["all_models"])

# ═══════════════════════════════════════
# TAB 3 — History
# ═══════════════════════════════════════
with tab3:
    st.subheader("Prediction History")

    if st.session_state.history:
        hist_df = pd.DataFrame(st.session_state.history)
        st.dataframe(hist_df, use_container_width=True)

        colour_map = {"Low": "#28a745", "Medium": "#ffc107", "High": "#dc3545"}
        counts = hist_df["Prediction"].value_counts()
        st.bar_chart(counts)
    else:
        st.info("No predictions yet — make one in the Predict tab.")

st.caption("COS40007 AI Engineering — Power Consumption Classifier")
