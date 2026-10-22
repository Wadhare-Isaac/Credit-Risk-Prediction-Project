
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from utils.preprocess import preprocess_input


# -------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------

st.set_page_config(
    page_title="Credit Risk Prediction",
    page_icon="💳",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "notebook" / "models"


# -------------------------------------------------
# LOAD MODEL AND PREPROCESSING ARTIFACTS
# -------------------------------------------------

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_DIR / "xgboost_credit_model.pkl")
    features = joblib.load(MODEL_DIR / "feature_columns.pkl")
    income_bins = joblib.load(MODEL_DIR / "income_bins.pkl")
    return model, features, income_bins


model, feature_columns, income_bins = load_artifacts()


# -------------------------------------------------
# APPLICATION HEADER
# -------------------------------------------------

st.title("💳 Credit Risk Prediction System")

st.write(
    "An explainable machine learning demonstration "
    "for estimating loan default risk using XGBoost."
)

st.info(
    "Portfolio demonstration only. Predictions are not "
    "suitable for real lending or credit approval decisions."
)

st.divider()


# -------------------------------------------------
# APPLICANT INFORMATION
# -------------------------------------------------

st.subheader("Applicant Information")

with st.form("credit_risk_form"):

    col1, col2 = st.columns(2)

    with col1:
        income = st.number_input(
            "Annual Income",
            min_value=1,
            value=50000,
            step=1000
        )

        loan_amount = st.number_input(
            "Requested Loan Amount",
            min_value=1,
            value=15000,
            step=500
        )

        employment_length = st.number_input(
            "Employment Length (Years)",
            min_value=0.0,
            max_value=50.0,
            value=5.0,
            step=1.0
        )

        home_ownership = st.selectbox(
            "Home Ownership",
            ["RENT", "MORTGAGE", "OWN", "OTHER"]
        )

    with col2:
        loan_intent = st.selectbox(
            "Loan Purpose",
            [
                "EDUCATION",
                "MEDICAL",
                "PERSONAL",
                "VENTURE",
                "HOMEIMPROVEMENT",
                "DEBTCONSOLIDATION"
            ]
        )

        previous_default = st.selectbox(
            "Previous Default on File",
            ["N", "Y"],
            format_func=lambda x: "Yes" if x == "Y" else "No"
        )

        credit_history = st.number_input(
            "Credit History Length (Years)",
            min_value=0,
            max_value=100,
            value=4
        )

    submitted = st.form_submit_button(
        "Predict Default Risk",
        type="primary",
        use_container_width=True
    )


# -------------------------------------------------
# PREDICTION
# -------------------------------------------------

if submitted:
    if income <= 0:
        st.error("Annual income must be greater than zero.")
        st.stop()

    if loan_amount <= 0:
        st.error("Loan amount must be greater than zero.")
        st.stop()

    if employment_length < 0:
        st.error("Employment length cannot be negative.")
        st.stop()

    if credit_history < 0:
        st.error("Credit history length cannot be negative.")
        st.stop()

    applicant = {
        "person_income": income,
        "person_home_ownership": home_ownership,
        "person_emp_length": employment_length,
        "loan_intent": loan_intent,
        "loan_amnt": loan_amount,
        "cb_person_default_on_file": previous_default,
        "cb_person_cred_hist_length": credit_history
    }

    X_new = preprocess_input(
        applicant,
        feature_columns,
        income_bins
    )

    probability = float(model.predict_proba(X_new)[0, 1])

    threshold = 0.40
    prediction = int(probability >= threshold)

    st.divider()
    st.subheader("Prediction Results")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Estimated Default Probability",
            f"{probability:.2%}"
        )

    with col2:
        st.metric(
            "Classification Threshold",
            f"{threshold:.0%}"
        )

    st.progress(probability)

    if prediction == 1:
        st.warning(
            "Higher Risk: The estimated default probability "
            "meets or exceeds the model threshold."
        )
    else:
        st.success(
            "Lower Risk: The estimated default probability "
            "is below the model threshold."
        )

    st.caption(
        "The classification reflects a selected experimental "
        "threshold, not an approval or rejection recommendation."
    )

    with st.expander("View processed model inputs"):
        st.dataframe(X_new.T, use_container_width=True)


# -------------------------------------------------
# FOOTER
# -------------------------------------------------

st.divider()

st.caption(
    "Credit Risk Prediction | XGBoost | "
    "Python | Scikit-learn | Streamlit"
)
