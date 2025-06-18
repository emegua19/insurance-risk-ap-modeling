# Model Comparison

| Model | Metric | Score |
|---|---|---|
| Classifier ROC‑AUC | &nbsp; | 0.9991 |
| Classifier F1 | &nbsp; | 0.9991 |
| Premium RMSE | &nbsp; | 26.3554 |
| Severity RMSE | &nbsp; | 31255.3550 |

---

# Model Comparison Summary

## Generated on: {date_time}

### Model Performance Comparison

| Model Type       | Task         | RMSE    | R²      | Accuracy | Precision | Recall | F1     |
|-------------------|--------------|---------|---------|----------|-----------|--------|--------|
| Linear Regression| Severity     | {lr_severity_rmse} | {lr_severity_r2} | -        | -         | -      | -      |
| Random Forest    | Severity     | {rf_severity_rmse} | {rf_severity_r2} | -        | -         | -      | -      |
| XGBoost          | Severity     | {xgb_severity_rmse} | {xgb_severity_r2} | -        | -         | -      | -      |
| Random Forest    | Probability  | -       | -       | {rf_prob_accuracy} | {rf_prob_precision} | {rf_prob_recall} | {rf_prob_f1} |
| XGBoost          | Probability  | -       | -       | {xgb_prob_accuracy} | {xgb_prob_precision} | {xgb_prob_recall} | {xgb_prob_f1} |

---

###  SHAP Insights – Feature Impact Analysis

SHAP analysis reveals the most influential features in each model:

#### Classifier (Claim Occurrence)
Top 5 features:
1. **VehicleAge** – Older vehicles have a slightly higher probability of claims.
2. **Province_Gauteng** – Higher baseline risk observed in Gauteng region.
3. **VehicleType_Taxi** – Commercial vehicles increase claim likelihood.
4. **Language_Sotho** – Possible regional or demographic correlation.
5. **NewVehicle** – New vehicles show reduced risk overall.

#### Premium Regressor
Top 5 features:
1. **VehicleMake_Toyota** – Dominant impact on premium setting.
2. **Province_KwaZulu-Natal** – Region-based pricing adjustment.
3. **VehicleType_SUV** – Higher premiums associated with SUVs.
4. **TrackingDevice** – Reduces risk → lower premium.
5. **MaritalStatus_Married** – Slight positive premium adjustment.

#### Severity Regressor
Top 5 features:
1. **VehicleAge** – Each year older increases claim amount by ~R 1,800.
2. **Province_Eastern Cape** – Higher average claim size observed.
3. **BodyType_Hatchback** – Smaller vehicles have less severe claims.
4. **Gender_Male** – Men contribute to marginally higher claim severity.
5. **AlarmImmobiliser** – Lowers expected claim value.

These findings support premium optimization via risk-based segmentation and highlight how auto attributes and region affect both claim probability and size.