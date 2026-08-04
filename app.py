import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Healthcare Provider Fraud Detection",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 Healthcare Provider Fraud Detection")

st.write("""
Upload a CSV containing the provider-level features used for model training.
The application predicts whether each provider is potentially fraudulent.
""")

model = joblib.load("fraud_model.pkl")

uploaded_file = st.file_uploader(
    "Upload Provider Feature CSV",
    type=["csv"]
)

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data")
    st.dataframe(data.head())

    # Remove Provider column if present
    provider = None
    if "Provider" in data.columns:
        provider = data["Provider"]
        X = data.drop(columns=["Provider"])
    else:
        X = data

    probability = model.predict_proba(X)[:, 1]
    prediction = model.predict(X)

    prediction = ["Yes" if p == 1 else "No" for p in prediction]

    result = pd.DataFrame()

    if provider is not None:
        result["Provider"] = provider

    result["Probability"] = probability
    result["Predicted Class"] = prediction

    st.subheader("Prediction Results")
    st.dataframe(result)

    csv = result.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Prediction CSV",
        data=csv,
        file_name="Predictions.csv",
        mime="text/csv"
    )

else:
    st.info("Please upload a CSV file.")
