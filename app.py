
import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="centered"
)

@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model(
        "fraud_detection_mlp.keras",
        compile=False
    )
    scaler = joblib.load("fraud_scaler.joblib")
    threshold = float(joblib.load("fraud_threshold.joblib"))
    return model, scaler, threshold

st.title("💳 Credit Card Fraud Detection")
st.write(
    "Enter transaction details to estimate the synthetic fraud-risk score."
)

try:
    model, scaler, threshold = load_artifacts()
except Exception as e:
    st.error(f"Could not load model files: {e}")
    st.stop()

with st.form("fraud_form"):
    amount = st.number_input(
        "Transaction Amount (₹)",
        min_value=0.01,
        value=12500.0,
        step=100.0
    )

    hour = st.slider(
        "Transaction Hour",
        min_value=0,
        max_value=23,
        value=2
    )

    international = st.selectbox(
        "International Transaction?",
        ["No", "Yes"]
    )

    card_present = st.selectbox(
        "Was the physical card present?",
        ["No", "Yes"]
    )

    previous_transactions = st.number_input(
        "Transactions in Previous 24 Hours",
        min_value=0,
        max_value=100,
        value=8,
        step=1
    )

    submitted = st.form_submit_button(
        "Analyze Transaction",
        type="primary"
    )

if submitted:
    numeric_features = [
        "transaction_amount",
        "transaction_hour",
        "transactions_last_24h"
    ]

    sample = pd.DataFrame([{
        "transaction_amount": float(amount),
        "transaction_hour": int(hour),
        "international_transaction": int(international == "Yes"),
        "card_present": int(card_present == "Yes"),
        "transactions_last_24h": int(previous_transactions)
    }])

    sample[numeric_features] = scaler.transform(
        sample[numeric_features]
    )

    probability = float(
        model.predict(
            sample.astype("float32"),
            verbose=0
        ).ravel()[0]
    )

    st.subheader("Prediction Results")
    st.metric(
        "Estimated Fraud Risk",
        f"{probability:.2%}"
    )

    if probability >= threshold:
        st.warning("⚠️ Flag for review")
    else:
        st.success("✅ Not flagged")

    st.caption(
        f"Decision threshold: {threshold:.3f}"
    )

st.divider()
st.caption(
    "Demo only: this model uses synthetic data and is not "
    "validated for real-world fraud detection."
)
