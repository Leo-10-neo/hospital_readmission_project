# Hospital Readmission Risk Modeling — Final Report

## 1. Executive Summary
This project developed a machine learning pipeline to predict the risk of hospital readmission within 30 days for diabetic patients. Using a dataset of over 100,000 hospital encounters, we engineered clinical features and evaluated multiple models. The **Random Forest** classifier emerged as the best performer for identifying high-risk patients, with the **Comorbidity Index** and **Length of Stay** identified as the most critical predictors.

---

## 2. Dataset & Features
- **Total Samples:** 101,766 patients.
- **Target Variable:** `readmitted` (1 if readmitted <30 days, 0 otherwise).
- **Class Distribution:** Highly imbalanced (~11.16% readmission rate).
- **Feature Engineering:** 89 final features, including:
    - **Clinical Scores:** Comorbidity Index, Medication Complexity Score.
    - **Interaction Terms:** Age × Medication count, Age × Length of Stay.
    - **Diagnosis Flags:** Diabetes, Heart Disease, Respiratory Disease, etc.

---

## 3. Data Preparation Pipeline
1. **Cleaning:** Handled missing values (imputation) and capped outliers using the IQR method.
2. **Encoding:** One-Hot Encoding for categorical variables.
3. **Splitting:** Stratified 70% Train / 15% Validation / 15% Test splits to maintain class ratios.
4. **Scaling:** Applied `StandardScaler` (fit on training data only) to numerical features.
5. **Balancing:** Implemented **SMOTE** (Synthetic Minority Over-sampling Technique) to balance the training set (50/50 ratio).

---

## 4. Model Performance Comparison (Test Set)

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | 86.51% | 0.2327 | **0.0910** | **0.1309** | **0.6519** |
| Gradient Boosting | 88.11% | 0.2431 | 0.0311 | 0.0552 | 0.6456 |
| Logistic Regression| **0.8878** | **0.3448** | 0.0059 | 0.0115 | 0.6441 |

> [!NOTE]
> While Logistic Regression had higher accuracy, it was mostly predicting the majority class. **Random Forest** provided the best discrimination (ROC-AUC) and the highest recall for actual readmissions.

---

## 5. Key Predictors of Readmission
Analysis of the Random Forest model revealed the following top 5 predictors:

1. **Comorbidity Index:** Patients with multiple chronic conditions are significantly more likely to be readmitted.
2. **Length of Stay (`length_of_stay_log`):** Longer hospitalizations correlate with higher readmission risk, likely reflecting higher clinical complexity.
3. **Age (`age_risk_group` / `age_midpoint`):** Risk increases progressively with age, especially for patients over 70.
4. **Number of Medications:** Higher medication counts indicate complex therapeutic needs and higher risk.
5. **Number of Lab Procedures:** Frequent testing during stay often indicates unstable clinical status.

---

## 6. Visualizations Reference
All diagnostic plots are available in the `visualizations/` directory:
- **EDA:** `visualizations/eda/` (Age distribution, Readmission rates).
- **Training:** `visualizations/model_plots/` (ROC Curves, Confusion Matrices).
- **Insights:** `visualizations/feature_importance/` (Permutation Importance, PDP).

---

## 7. Recommendations & Future Work
- **Threshold Tuning:** The classification threshold (currently 0.5) should be lowered in a clinical setting to prioritize high Recall (catching more at-risk patients) even at the cost of lower Precision.
- **Ensemble Methods:** Further tuning of XGBoost or LightGBM could yield marginal gains in AUC.
- **Real-time Integration:** Deploy the model as a REST API to provide risk scores during the discharge planning process.
- **Clinical Intervention:** High-risk patients identified by the model should receive enhanced post-discharge follow-up or transitional care services.

---
**Report Generated:** 2026-05-08
**Project Status:** COMPLETED
