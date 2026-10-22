import numpy as np
import pandas as pd


def preprocess_input(data, feature_columns, income_bins):
    """
    Convert applicant data into the exact feature structure
    expected by the trained XGBoost model.
    """

    df = pd.DataFrame([data]).copy()

    # 1. Loan amount relative to annual income
    df["debt_to_income"] = (
        df["loan_amnt"] / df["person_income"]
    )

    # 2. Income categories
    df["income_group"] = pd.cut(
        df["person_income"],
        bins=[
            -np.inf,
            income_bins[1],
            income_bins[2],
            income_bins[3],
            np.inf
        ],
        labels=["low", "mid", "high", "very_high"]
    )

    # 3. Employment categories
    df["emp_length_group"] = pd.cut(
        df["person_emp_length"],
        bins=[0, 2, 5, 10, 50],
        labels=["new", "junior", "mid", "senior"]
    )

    # 4. Credit experience ratio
    df["credit_exp_ratio"] = (
        df["cb_person_cred_hist_length"]
        / (df["person_emp_length"] + 1)
    )

    # 5. Define the original categories used during training
    categories = {
        "person_home_ownership": [
            "MORTGAGE", "OTHER", "OWN", "RENT"
        ],
        "loan_intent": [
            "DEBTCONSOLIDATION",
            "EDUCATION",
            "HOMEIMPROVEMENT",
            "MEDICAL",
            "PERSONAL",
            "VENTURE"
        ],
        "cb_person_default_on_file": ["N", "Y"],
        "income_group": [
            "low", "mid", "high", "very_high"
        ],
        "emp_length_group": [
            "new", "junior", "mid", "senior"
        ]
    }

    # 6. Enforce consistent categorical encoding
    for col, values in categories.items():
        df[col] = pd.Categorical(
            df[col],
            categories=values
        )

    df = pd.get_dummies(df, drop_first=True)

    # 7. Match model feature columns
    df = df.reindex(
        columns=feature_columns,
        fill_value=0
    )

    return df