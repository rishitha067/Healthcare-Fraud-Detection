import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Healthcare Provider Fraud Detection")

st.title("🏥 Healthcare Provider Fraud Detection")

model = joblib.load("fraud_model.pkl")

st.write("Upload the three unseen dataset files.")

provider_file = st.file_uploader(
    "Upload Unseen Provider File",
    type="csv"
)

ip_file = st.file_uploader(
    "Upload Unseen Inpatient File",
    type="csv"
)

op_file = st.file_uploader(
    "Upload Unseen Outpatient File",
    type="csv"
)

if provider_file and ip_file and op_file:

    unseen = pd.read_csv(provider_file)
    inpatient = pd.read_csv(ip_file)
    outpatient = pd.read_csv(op_file)

    inpatient["AdmissionDt"] = pd.to_datetime(inpatient["AdmissionDt"])
    inpatient["DischargeDt"] = pd.to_datetime(inpatient["DischargeDt"])

    inpatient["HospitalStay"] = (
        inpatient["DischargeDt"] -
        inpatient["AdmissionDt"]
    ).dt.days

    ip_claims = inpatient.groupby(
        "Provider"
    ).size().reset_index(name="IP_Claims")

    op_claims = outpatient.groupby(
        "Provider"
    ).size().reset_index(name="OP_Claims")

    ip_avg_claim = inpatient.groupby(
        "Provider"
    )["InscClaimAmtReimbursed"].mean().reset_index()

    ip_avg_claim.rename(
        columns={
            "InscClaimAmtReimbursed":"IP_AvgClaim"
        },
        inplace=True
    )

    op_avg_claim = outpatient.groupby(
        "Provider"
    )["InscClaimAmtReimbursed"].mean().reset_index()

    op_avg_claim.rename(
        columns={
            "InscClaimAmtReimbursed":"OP_AvgClaim"
        },
        inplace=True
    )

    ip_avg_deductible = inpatient.groupby(
        "Provider"
    )["DeductibleAmtPaid"].mean().reset_index()

    ip_avg_deductible.rename(
        columns={
            "DeductibleAmtPaid":"IP_AvgDeductible"
        },
        inplace=True
    )

    avg_stay = inpatient.groupby(
        "Provider"
    )["HospitalStay"].mean().reset_index()

    avg_stay.rename(
        columns={
            "HospitalStay":"AvgHospitalStay"
        },
        inplace=True
    )

    final_test = unseen.copy()

    final_test = final_test.merge(
        ip_claims,
        on="Provider",
        how="left"
    )

    final_test = final_test.merge(
        op_claims,
        on="Provider",
        how="left"
    )

    final_test = final_test.merge(
        ip_avg_claim,
        on="Provider",
        how="left"
    )

    final_test = final_test.merge(
        op_avg_claim,
        on="Provider",
        how="left"
    )

    final_test = final_test.merge(
        ip_avg_deductible,
        on="Provider",
        how="left"
    )

    final_test = final_test.merge(
        avg_stay,
        on="Provider",
        how="left"
    )

    final_test.fillna(0, inplace=True)

    X = final_test.drop("Provider", axis=1)

    probability = model.predict_proba(X)[:,1]

    prediction = model.predict(X)

    prediction = [
        "Yes" if x==1 else "No"
        for x in prediction
    ]

    result = pd.DataFrame({
        "Provider":final_test["Provider"],
        "Probability":probability,
        "Predicted Class":prediction
    })

    st.success("Prediction Completed")

    st.dataframe(result)

    st.download_button(
        "Download Submission",
        result.to_csv(index=False),
        "Submission.csv",
        "text/csv"
    )
