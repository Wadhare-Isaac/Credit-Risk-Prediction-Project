# Credit Risk Prediction | Explainable Machine Learning

**An end-to-end classification case study: from exploratory analysis to credit-risk decision support.**

![Credit-risk feature importance](images/Feature%20importance-xgboost.png)

> **Project status:** Analysis, model evaluation and SHAP explainability completed. Deployment is planned, **not yet validated or live**. This is an educational proof of concept, not an automated lending decision system.

## Executive summary

Using **32,581 loan records** and **12 original columns**, I investigated borrower characteristics associated with default and compared Logistic Regression, Random Forest, and XGBoost. The data has a **21.8% default rate**, so identifying defaults matters more than reporting accuracy alone.

**XGBoost achieved the highest ROC-AUC (0.903)** on the 6,517-row held-out test split. At an illustrative **0.40 classification threshold**, it detected **82% of observed defaults**, with **53% precision**. The threshold is a *candidate operating point*, not a financially optimized policy.

| XGBoost at 0.40 | Result |
|---|---:|
| ROC-AUC (threshold-independent) | **0.903** |
| Accuracy | 0.80 |
| Default recall | **0.82** |
| Default precision | 0.53 |
| Default F1 | 0.64 |
| False negatives (missed defaults) | 255 |
| False positives (non-defaulters flagged) | 1,033 |

## Business question

**Can borrower and loan information help identify likely defaults early enough to support more consistent credit reviews?** False negatives expose lenders to credit losses; false positives may deny or delay credit for borrowers who would repay. A useful system must balance both, and its performance must be assessed beyond accuracy.

## Dataset

- **32,581 rows, 12 original columns**; binary outcome `loan_status` (`0` = non-default, `1` = default).
- Target split: **78.2% non-default**, **21.8% default**.
- Examples of original fields: `person_income`, `person_emp_length`, `person_home_ownership`, `loan_amnt`, `loan_intent`, `loan_grade`, `loan_int_rate`, `cb_person_default_on_file`, `cb_person_cred_hist_length`.
- Missing values observed: **3,116** in `loan_int_rate` and **895** in `person_emp_length`.
- Dataset provenance, observation period, and geographic coverage should be documented before any real-world generalization claims.

![Target distribution](images/Target%20distribution%20plot.png)

## Exploratory findings

**Observed default rates** (descriptive associations, not causal effects):

| Segment | Default rate |
|---|---:|
| Debt consolidation loans | 28.6% |
| Medical loans | 26.7% |
| Education loans | 17.2% |
| Venture loans | 14.8% |
| Renters | 31.6% |
| Mortgage holders | 12.6% |
| Homeowners (`OWN`) | 7.5% |

Loan-grade default rates ranged from **10.0% (A)** to **98.4% (G)** in this sample, although grade was excluded from the eventual model. Differences in group size and confounding may affect interpretation.

| Housing patterns | Loan-purpose patterns |
|---|---|
| ![Home ownership vs loan status](images/Home%20ownership%20vs%20loan%20status.png) | ![Loan intent vs loan status](images/loan%20intent%20vs%20loan%20status.png) |

![Correlation heatmap](images/Correlation%20heatmap.png)

## Methodology

1. **Audit the data:** shape, data types, missingness, class distribution and exploratory plots.
2. **Select features:** exclude `loan_grade`, `loan_int_rate`, `loan_percent_income` and `person_age` in this experiment. Grade and rate may encode lender decisions, but *leakage depends on the real prediction time*; exclusion alone does not prove leakage.
3. **Engineer features:** `debt_to_income = loan_amnt / person_income`; income quartiles; employment-duration bands; and `credit_exp_ratio = credit_history_length / (employment_length + 1)`.
4. **Encode categories:** one-hot encoding with `pd.get_dummies(drop_first=True)`.
5. **Split:** stratified 80/20 training/test split (`random_state=42`); training set 26,064 and test set 6,517.
6. **Preprocess:** fill missing model inputs with training medians; standardize numeric model matrix for Logistic Regression.
7. **Fit and evaluate:** baseline Logistic Regression, class-weighted Logistic Regression, balanced Random Forest, and class-weighted XGBoost.
8. **Explain:** model-native feature importance and SHAP beeswarm analysis.

**Important reproducibility caveat:** the notebook creates income quantile bins and encodes features before the train/test split. For production-quality validation, learn bin boundaries, imputation statistics and encoding mappings on training data only and reuse them in an inference pipeline. Also inspect the employment-bin edges: `pd.cut` excludes zero by default and values above 50 are not covered.

## Model comparison

All ROC-AUC scores below come from the notebook's same test partition. Thresholded classification metrics depend on the decision cutoff.

| Model | ROC-AUC | Default recall at 0.50 | Default precision at 0.50 |
|---|---:|---:|---:|
| Logistic Regression (baseline) | 0.814 | 0.41 | 0.74 |
| Logistic Regression (balanced) | 0.817 | 0.72 | 0.46 |
| Random Forest (balanced) | 0.867 | 0.51 | 0.92 |
| **XGBoost** | **0.903** | **0.75** | **0.67** |

XGBoost used 200 estimators, max depth 5, learning rate 0.1 and an imbalance-adjusted `scale_pos_weight`. These results establish a promising **experimental** ranking, not proof of superior performance on new populations.

### XGBoost threshold trade-off

| Threshold | Precision (default) | Recall (default) | Accuracy | Missed defaults (FN) | Non-defaulters flagged (FP) |
|---|---:|---:|---:|---:|---:|
| 0.50 | 0.67 | 0.75 | 0.86 | 356 | 537 |
| **0.40** | **0.53** | **0.82** | **0.80** | **255** | **1,033** |
| 0.30 | 0.43 | 0.90 | 0.72 | 144 | 1,669 |

Reducing the cutoff detects more defaulters but flags more borrowers who did not default. **0.40 is a useful demonstration threshold, not an empirically proven optimum.** A bank would need a separate validation set, probability calibration, expected loss assumptions and approval-policy review before selecting a cutoff.

## Model explainability

### XGBoost feature importance

![XGBoost feature importance](images/Feature%20importance-xgboost.png)

The highest model-native importance scores were: previous-default flag (**0.199**), renting (**0.163**), debt-to-income (**0.118**), owning (**0.114**), and venture intent (**0.061**). Feature importance measures relative model use; it **does not by itself reveal whether a variable raises or lowers risk**.

### SHAP explanations

![SHAP beeswarm](images/sharp%20analysis.png)

SHAP illustrates how feature values push model outputs higher or lower for individual records. In this analysis, affordability and prior credit behavior feature prominently. These are **model explanations, not causal claims**. Housing status and other potentially sensitive proxies require fairness review before operational use.

## Recommendations

- Use the score as a **review-prioritization signal**, not an automatic approve/deny decision.
- Request additional affordability verification for cases with high model-estimated risk, while providing a human review path.
- Investigate prior-default and loan-purpose patterns without treating a single feature as determinative.
- Assess group-level error rates and potential disparate impact, especially for housing-related features.
- Evaluate thresholds using realistic default-loss, false-positive and operational review costs.

## Limitations and validation risks

- **Single random holdout:** no cross-validation, out-of-time validation or external dataset validation is documented.
- **Preprocessing leakage risk:** income quartile boundaries were estimated before splitting; rebuild preprocessing inside a train-fitted pipeline.
- **Threshold selection on test outcomes:** choosing 0.40 after inspecting the test set can make final test performance optimistic; use validation data for selection and keep a fresh untouched test set.
- **No probability calibration:** `predict_proba` is not automatically a reliable real-world default probability.
- **Incomplete business context:** no verified cost matrix, lending policy, sample provenance, observation dates or economic regime.
- **Fairness and governance:** proxy discrimination, adverse-action explanations, drift, privacy and human oversight require evaluation.
- **No deployed application yet:** a saved model alone is not a reproducible inference system.

## Next steps

1. Rebuild feature engineering and encoding as a **train-fitted, serialized preprocessing pipeline**; confirm inference feature order and transformations.
2. Introduce train/validation/test or nested cross-validation; tune hyperparameters and the threshold on validation data only.
3. Test calibration (reliability plots, Brier score) and a documented business cost scenario.
4. Evaluate fairness, stability and out-of-time performance.
5. Build a **Streamlit demonstration app** with clear educational-use warnings, then publish a live demo when tested.
6. Document environment versions, reproducible training commands, and a model card.

## Repository structure

```text
credit-risk-prediction/
├── notebook
    └── Credit.ipynb              # Analysis notebook (rename uploaded notebook)
├── README.md
├── images/
│   ├── target.png
│   ├── home.png
│   ├── intent.png
│   ├── heatmap.png
│   ├── importance.png
│   └── shap.png
└── presentation/
    └── Credit_Risk_Portfolio_Presentation.pptx
```

### Reproduce the analysis

Create a Python environment and install the libraries used in the notebook:

```bash
python -m pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap jupyter
jupyter notebook Credit.ipynb
```

Place the dataset at the location expected by the notebook's data-loading cell, or update that path to your local file. A pinned `requirements.txt`, a stable data download reference and an automated training script remain future reproducibility improvements.

## Takeaway

**The strongest result is not merely an accuracy score.** This project demonstrates a full path from identifying class imbalance, through model comparison and interpretable predictions, to explicitly discussing the costs, limitations and governance needed for responsible credit-risk decision support.

*Portfolio case study • Python • scikit-learn • XGBoost • SHAP*
