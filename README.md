# Credit Risk Prediction Using Machine Learning

## Project Overview

Financial institutions face significant losses when borrowers fail to repay loans. Accurately identifying high-risk applicants before approval can reduce default rates, improve lending decisions, and increase profitability.

This project develops a machine learning solution to predict whether a borrower is likely to default on a loan using demographic, financial, and credit-related information.

The project follows a complete end-to-end data science workflow, including:

- Data cleaning and preprocessing
- Exploratory Data Analysis (EDA)
- Feature engineering
- Model development
- Threshold tuning
- Model explainability using SHAP
- Business recommendations

---

##  Business Problem

Loan defaults represent a major financial risk for lenders.

The objective of this project is to build a predictive model capable of identifying borrowers with a high probability of default before a loan is approved.

### Target Variable

| Variable | Description |
|-----------|------------|
| loan_status | 0 = Non-default, 1 = Default |

---

## Dataset Overview

### Dataset Size

| Metric | Value |
|---------|--------|
| Rows | 32,581 |
| Features | 12 |

### Key Features

| Feature | Description |
|-----------|------------|
| person_income | Annual borrower income |
| person_home_ownership | Housing status |
| person_emp_length | Employment duration |
| loan_amnt | Requested loan amount |
| loan_intent | Purpose of loan |
| loan_grade | Loan risk grade |
| cb_person_default_on_file | Previous default history |
| cb_person_cred_hist_length | Credit history length |

---

# Exploratory Data Analysis

Several borrower characteristics exhibited strong relationships with default risk.

## Home Ownership

Default rates varied significantly across ownership categories:

| Ownership Status | Risk Level |
|-----------------|------------|
| RENT | Highest |
| MORTGAGE | Moderate |
| OWN | Lowest |

### Insight

Home ownership appears to act as a proxy for financial stability.

---

## Loan Intent

Default risk differed substantially across loan purposes.

### Highest Risk

- Debt Consolidation
- Medical Loans

### Lowest Risk

- Venture
- Education

### Insight

Loan purpose provides valuable information about borrower financial circumstances.

---

## Loan Grade

Default rates increased dramatically as loan grade deteriorated.

| Grade | Default Risk |
|---------|-------------|
| A | Lowest |
| B | Low |
| C | Moderate |
| D | High |
| E | Very High |
| F | Extremely High |
| G | Highest |
 
![Target distribution](images/Target%20distribution%20plot.png)

![Correlation heatmap](images/Correlation%20heatmap.png)
---

# Feature Engineering

To improve predictive performance, several new features were created.

### Debt-to-Income Ratio

Measures loan burden relative to borrower income.

```python
debt_to_income = loan_amnt / person_income
```

### Income Groups

Borrowers segmented into:

- Low
- Mid
- High
- Very High

### Employment Groups

Borrowers segmented into:

- New
- Junior
- Mid
- Senior

### Credit Experience Ratio

```python
credit_exp_ratio = credit_history_length / (employment_length + 1)
```

This feature captures the relationship between credit history and employment history.

---

# Models Evaluated

Three models were developed and compared.

## 1. Logistic Regression

Used as a baseline model.

### Strengths

- Simple
- Interpretable

### Weaknesses

- Lower overall performance
- Struggled with minority class prediction

---

## 2. Random Forest

Improved predictive power through ensemble learning.

### Results

- ROC-AUC ≈ 0.87
- Better separation between default and non-default borrowers

---

## 3. XGBoost (Best Model)

XGBoost delivered the strongest performance across all evaluation metrics.

### Advantages

- Strong predictive accuracy
- Handles non-linear relationships
- Robust against overfitting
- Excellent performance on tabular data

---

# Threshold Tuning

Instead of relying solely on the default threshold of 0.5, multiple thresholds were evaluated.

| Threshold | Recall | Precision |
|------------|----------|-----------|
| 0.5 | Lower | Higher |
| 0.4 | Balanced | Balanced |
| 0.3 | Highest | Lower |

### Selected Threshold

✅ 0.4

Reason:

Provides the best balance between:

- Identifying risky borrowers
- Avoiding excessive false alarms

---

# Final Model Performance

## XGBoost (Threshold = 0.4)

### Classification Metrics

| Metric | Score |
|----------|--------|
| Accuracy | 0.80 |
| Recall | 0.82 |
| Precision | 0.53 |
| ROC-AUC | 0.86 |

### Interpretation

The model successfully identifies approximately 82% of defaulters while maintaining reasonable precision.

This makes it suitable as a decision-support tool in lending environments.

---

# Feature Importance

The most influential predictors were:

1. Previous Default History
2. Home Ownership Status
3. Debt-to-Income Ratio
4. Income Level
5. Loan Purpose

These factors align closely with established credit risk assessment practices.

![Feature importance](images/Feature%20importance-xgboost.png)

---

# SHAP Analysis

SHAP (SHapley Additive Explanations) was used to explain individual predictions.

### Key Findings

#### Increased Default Risk

- High Debt-to-Income Ratio
- Previous Defaults
- Renting
- Lower Income

#### Reduced Default Risk

- Home Ownership
- Higher Income
- Education Loans

### Why SHAP Matters

SHAP improves transparency by showing how individual features influence model predictions.

This is particularly important in financial decision-making environments.

![sharp analysis](images/sharp%20analysis.png)

---

# Business Recommendations

Based on the analysis:

### 1. Monitor High Debt-to-Income Borrowers

Implement stricter approval criteria for borrowers with high debt burdens.

### 2. Flag Previous Defaulters

Previous default history should remain a major component of risk assessment.

### 3. Incorporate Housing Stability

Home ownership status can provide valuable risk signals.

### 4. Use Purpose-Specific Lending Policies

Debt consolidation and medical loans may require additional scrutiny.

### 5. Deploy the Model as Decision Support

The model should support loan officers rather than fully automate decisions.

---

# Project Limitations

Several limitations should be considered:

- Historical data may not reflect future economic conditions
- Important variables such as credit scores were unavailable
- Dataset contains class imbalance
- Economic factors were not included
- Results may not generalize to all lending markets

---

# Future Work

Potential improvements include:

- Streamlit deployment
- Hyperparameter optimization
- LightGBM and CatBoost comparison
- Additional feature engineering
- Automated model monitoring
- Expanded explainability analysis

---

# Tech Stack

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- XGBoost
- SHAP

---

# Project Structure

```text
credit-risk-prediction/
│
├── data/
├── notebooks/
│   └── Credit.ipynb
│
├── images/
│
├── README.md
│
├── requirements.txt
│
└── app.py
```

---

# Conclusion

This project demonstrates how machine learning can be used to support credit risk assessment through predictive modeling and explainable AI.

After comparing multiple models, XGBoost emerged as the best-performing solution. Feature importance and SHAP analysis confirmed that the model's behavior aligns with established lending principles, making it both accurate and interpretable.

The final solution provides a strong foundation for real-world deployment as a credit risk decision-support system.