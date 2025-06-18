# Final Report: Insurance Risk AP Modeling

## Introduction
The Insurance Risk AP Modeling project aims to develop a data-driven, risk-based pricing system for insurance policies. This report summarizes the progression from initial data exploration (Task 1) and exploratory data analysis (Task 2) to hypothesis testing (Task 3) and predictive modeling with risk optimization (Task 4). The project leverages a dataset of approximately 1 million records, processed using DVC for versioning and Python for analysis. Refer to [README.md](README.md) for an overview and [requirements.txt](requirements.txt) for dependencies.

## Task 1: Data Exploration
### Objective
Understand the raw data structure and identify key variables for modeling.

### Methodology
- Loaded raw data from `data/raw/MachineLearningRating_v3.txt` using pandas with a `|` delimiter.
- Inspected column names, data types, and initial statistics.
- Identified potential features (e.g., `Gender`, `Province`, `TotalPremium`) and targets (e.g., `TotalClaims`).

### Findings
- Dataset contains ~1M rows with 38 columns, including mixed-type issues (e.g., column 32).
- Missing columns like `Age` and `PlanType` were noted, requiring feature engineering or exclusion.
- Initial visualization of `TotalClaims` distribution highlighted a sparse claim occurrence (~0.26%).

## Task 2: Exploratory Data Analysis (EDA)
### Objective
Clean and explore the dataset to prepare for hypothesis testing and modeling.

### Methodology
- Implemented `src/data_processing/data_cleaning.py` to handle missing values (median/mode imputation) and mixed types (comma-to-decimal conversion).
- Used `src/eda/` modules (`correlation.py`, `descriptive_stats.py`, `visualizations.py`) for analysis.
- Saved cleaned data to `data/processed/insurance_cleaned_data.csv`.

### Findings
- Correlation analysis showed `TotalPremium` and `CapitalOutstanding` were moderately correlated with `TotalClaims`.
- Descriptive statistics revealed skewed distributions for `TotalClaims`, necessitating log transformation in future models.
- Visualizations (in `outputs/plots`) confirmed regional variations in claim frequency.

## Task 3: Hypothesis Testing
### Objective
Test statistical hypotheses to identify significant factors affecting insurance margins.

### Methodology
- Configured tests in `config/hypothesis_config.yaml`.
- Used `scripts/run_hypothesis_tests.py` with `src/hypothesis_testing/` modules to segment data and apply t-tests/Mann-Whitney U tests.
- Results saved to `reports/task_3_hypothesis_summary.md`.

### Results
- **Province Margin (Gauteng vs. Western Cape)**: t-statistic = -0.9316, p = 0.3516, Fail to reject H₀ (no significant difference).
- **Gender Margin (Male vs. Female)**: U-statistic = 33994516.5, p = 0.0, Reject H₀ (significant difference).
- **Insight**: `Gender` is a key differentiator for margin, while `Province` effects are not significant.

## Task 4: Predictive Modeling & Risk-Based Pricing
### Objective
Build and evaluate models for claim probability, premium prediction, and claim severity to optimize risk-based pricing.

### Data Preparation
- Handled missing values with median/mode imputation and applied one-hot encoding (OHE) via `src/modeling/data_preparation.py`.
- Performed an 80/20 stratified split on `claim_frequency` for training and testing.

### Model Building
- **Classifier**: Predicts claim probability (XGBoost baseline).
- **Premium Regressor**: Predicts calculated premium amount (Random Forest).
- **Severity Regressor**: Predicts `TotalClaims` for claims > 0 (XGBoost).
- Models trained and saved as `models/classifier.joblib`, `models/regressor.joblib`, and `models/severity_regressor.joblib`.

### Evaluation
- **Classifier**: ROC-AUC = 0.9991, Accuracy = 0.95, Precision = 0.94, Recall = 0.96, F1 = 0.95 (from `reports/classifier_metrics.json`).
- **Premium Regressor**: R² = 0.991, RMSE = 150.23, MAE = 120.45 (from `reports/regressor_metrics.json`).
- **Severity Regressor**: R² = 0.313, RMSE = 4500.67, MAE = 3200.89 (from `reports/severity_metrics.json`).
- Metrics saved to `reports/*.json`.

### Comparison
See [model_comparison.md](reports/model_comparison.md) for a detailed table and SHAP insights.

### Interpretability
SHAP analysis (see `reports/shap_*.png`) identifies:
- **Severity Regressor**: `VehicleAge` increases claims by ~R 1,800/year; `Province_Eastern Cape` shows higher claim sizes.
- **Classifier**: `VehicleType_Taxi` raises claim probability; `NewVehicle` reduces risk.
- **Premium Regressor**: `VehicleMake_Toyota` dominates pricing; `TrackingDevice` lowers premiums.

## Business Recommendations
- **Premium Adjustments**: Increase rates for older vehicles (`VehicleAge`) and male policyholders (`Gender_Male`) based on Task 3 and 4 insights.
- **Regional Focus**: Target `Province_Eastern Cape` for claim mitigation and `Province_KwaZulu-Natal` for premium tuning.
- **Risk Mitigation**: Offer discounts for `TrackingDevice` and `AlarmImmobiliser` to reduce claim severity and probability.
- **Segmentation**: Use `VehicleType_Taxi` and `VehicleType_SUV` for targeted underwriting.

## Conclusion
The project successfully delivered a risk-based pricing framework, integrating hypothesis testing and predictive modeling. Limitations include the absence of `Age` data and moderate R² for severity (0.313), suggesting room for model improvement. Future work could incorporate external demographic data and explore deep learning for severity prediction.