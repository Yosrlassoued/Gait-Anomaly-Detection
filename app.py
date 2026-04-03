import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io

# ── your column names (copy from your notebook) ──────────────────────────
COLUMNS = [
    "elapsed_time", "left_stride_interval", "right_stride_interval",
    "left_swing_interval", "right_swing_interval", "left_swing_percent",
    "right_swing_percent", "left_stance_interval", "right_stance_interval",
    "left_stance_percent", "right_stance_percent", "double_support_interval",
    "step_time",
]

# ── load model ────────────────────────────────────────────────────────────
artifact     = joblib.load("model.pkl")
pipeline     = artifact["pipeline"]
feature_cols = artifact["features"]

# ── feature extractor (copy from your notebook) ───────────────────────────
def extract_features(df):
    features = {}
    cols = [c for c in COLUMNS if c != "elapsed_time"]
    for col in cols:
        s = df[col].dropna()
        features[f"{col}_mean"]   = s.mean()
        features[f"{col}_std"]    = s.std()
        features[f"{col}_cv"]     = s.std() / s.mean() if s.mean() != 0 else 0
        features[f"{col}_median"] = s.median()
        features[f"{col}_iqr"]    = s.quantile(0.75) - s.quantile(0.25)
    return pd.Series(features)

# ── app ───────────────────────────────────────────────────────────────────
st.title(" Gait Anomaly Detection")
st.markdown("Upload a patient's gait file to detect signs of neurodegenerative disease.")

uploaded = st.file_uploader("Upload a .ts gait file", type=["ts"])

if uploaded:
    # read and parse
    content = uploaded.read().decode("utf-8")
    df = pd.read_csv(io.StringIO(content), sep="\t", header=None, names=COLUMNS)
    df = df.apply(pd.to_numeric, errors="coerce").dropna()

    # extract features and predict
    feats = extract_features(df)
    X_new = np.array([[feats.get(c, 0) for c in feature_cols]])
    pred  = pipeline.predict(X_new)[0]
    proba = pipeline.predict_proba(X_new)[0]

    # show result
    if pred == 1:
        st.error(f"⚠️ Impaired gait detected — {proba[1]*100:.1f}% confidence")
    else:
        st.success(f"✅ Healthy gait — {proba[0]*100:.1f}% confidence")

    # show gait signal chart
    st.subheader("Gait Signal")
    st.line_chart(df[["left_stride_interval", "right_stride_interval"]])

    # show key features
    st.subheader("Key Features")
    st.dataframe(pd.DataFrame({
        "Feature": ["Stride variability (CV)", "Double support mean", "Step time mean"],
        "Value":   [
            f"{feats['left_stride_interval_cv']:.4f}",
            f"{feats['double_support_interval_mean']:.3f}s",
            f"{feats['step_time_mean']:.3f}s",
        ]
    }))