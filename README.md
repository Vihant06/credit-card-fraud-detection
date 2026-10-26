---
title: Credit Card Fraud Detection
emoji: 💳
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.23.1
app_file: app.py
pinned: false
---

# Credit Card Fraud Detection — Gradio Demo

This Space provides a Gradio interface for a TensorFlow/Keras MLP trained on a
synthetic credit-card transaction dataset.

## Required files

Upload all of these files to the root of the Space repository:

- `app.py`
- `requirements.txt`
- `fraud_detection_mlp.keras`
- `fraud_scaler.joblib`
- `fraud_threshold.joblib`

The `.keras` file is the trained TensorFlow model. The scaler and threshold
must be the artifacts produced during training in Google Colab. Do not retrain
or recreate the scaler in the Space.

## Deploy

1. In Google Colab, download the three model artifacts created by your training notebook:
   - `fraud_detection_mlp.keras`
   - `fraud_scaler.joblib`
   - `fraud_threshold.joblib`
2. Create a new Hugging Face Space and choose **Gradio** as the SDK, or open your intended Space.
3. Upload the five required files above to the root of the repository.
4. Wait for the Space to build, then test the five inputs.

## Limitations

This model was trained on synthetic data with labels generated from predefined
risk relationships. Its scores are only a software demonstration and are not
validated estimates of real-world fraud. Do not use it to authorize, block, or
investigate real transactions.
