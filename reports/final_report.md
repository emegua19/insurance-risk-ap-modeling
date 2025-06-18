
````markdown
#  Final Report: Insurance Risk AP Modeling (10 Academy – Week 3)

This project builds an end-to-end **risk-based insurance pricing framework** for **AlphaCare Insurance Solutions (ACIS)** using real-world auto insurance data. It integrates:

- Data cleaning & exploratory analysis
- Statistical hypothesis testing
- Machine learning for claim prediction and pricing
- SHAP-based model interpretability

---

##  Final Folder Structure

```bash
INSURANCE-RISK-AP-MODELING-W3/
├── .dvc/
├── .cache/
├── .gitignore
├── config/
│   ├── eda_config.yaml
│   ├── hypothesis_config.yaml
│   └── modeling_config.yaml
├── data/
│   ├── raw/
│   │   ├── MachineLearningRating_v3.txt
│   │   ├── MachineLearningRating_v3.csv
│   │   └── *.dvc
│   └── processed/
│       ├── insurance_cleaned_data.csv
│       └── insurance_cleaned_data.csv.dvc
├── models/
│   ├── classifier.joblib
│   ├── regressor.joblib
│   └── severity_regressor.joblib
├── outputs/
│   ├── plots/
│   ├── stats/
│   └── eda_pipeline.log
├── reports/
│   ├── classifier_metrics.json
│   ├── final_report.md
│   ├── model_comparison.md
│   ├── regressor_metrics.json
│   ├── severity_metrics.json
│   ├── shap_classifier.png
│   ├── shap_regressor.png
│   ├── shap_severity.png
│   └── task_3_hypothesis_summary.md
├── scripts/
│   ├── run_eda.py
│   ├── run_hypothesis_tests.py
│   └── run_modeling.py
├── src/
│   ├── data_processing/
│   │   └── data_cleaning.py
│   ├── eda/
│   │   ├── correlation.py
│   │   ├── descriptive_stats.py
│   │   └── visualizations.py
│   ├── hypothesis_testing/
│   │   ├── metrics.py
│   │   ├── segmentation.py
│   │   └── statistical_tests.py
│   ├── modeling/
│   │   ├── evaluation.py
│   │   ├── features.py
│   │   ├── interpretation.py
│   │   ├── models.py
│   │   └── utils/
│   │       └── data_loader.py
├── tests/
│   ├── test_eda.py
│   ├── test_models.py
│   └── fixtures/
│       └── insurance_sample.csv
├── LICENSE
├── README.md
└── requirements.txt
````

---

##  Summary of Tasks

### Task 1: Exploratory Data Analysis (EDA)

* Cleaned & profiled 1M+ rows
* Visualized premium/claim patterns
* Output: `outputs/eda/`, `data/processed/`

### Task 2: Data Version Control (DVC)

* DVC set up for raw + cleaned data
* `.dvc` files tracked with Git
* Used local remote storage (`/new/dvc-storage/`)

### Task 3: Hypothesis Testing

* Used statistical tests on segmented data
* Significant margin difference by `Gender`
* Summary: [`reports/task_3_hypothesis_summary.md`](task_3_hypothesis_summary.md)

### Task 4: Predictive Modeling

* Three models trained: classifier, premium regressor, severity regressor
* Evaluation metrics stored in JSONs + markdown
* SHAP interpretability added
* Outputs saved to `models/` and `reports/`

---

##  Final Model Scores

| Model              | Purpose                      | Metric  | Score  |
| ------------------ | ---------------------------- | ------- | ------ |
| Classifier         | Predict claim occurrence     | ROC‑AUC | 0.9991 |
| Premium Regressor  | Predict premium amount       | R²      | 0.991  |
| Severity Regressor | Predict TotalClaims (if > 0) | R²      | 0.313  |

---

##  SHAP Insights (Top Features)

| Model              | Key Features                           | Insight                                           |
| ------------------ | -------------------------------------- | ------------------------------------------------- |
| Severity Regressor | `VehicleAge`, `Province`               | ↑ Claim severity with older cars and Eastern Cape |
| Classifier         | `VehicleType_Taxi`, `NewVehicle`       | ↑ Risk for taxis, ↓ risk for new cars             |
| Premium Regressor  | `TrackingDevice`, `VehicleMake_Toyota` | Device ↓ premium, Toyota ↑ baseline pricing       |

---

##  Business Recommendations

* Adjust premiums by **Gender** and **VehicleAge**
* Offer **discounts** for `TrackingDevice`, `AlarmImmobiliser`
* Focus risk management efforts in **Eastern Cape**
* Use classifier to proactively handle **high-risk clients**

---

##  Conclusion

This project delivers a reproducible, interpretable ML pipeline that supports smarter pricing and better risk segmentation.

* Clean modular codebase
* SHAP-driven explainability
* CI + unit testing
* DVC-powered data versioning

---

*Authored by Yitbarek Geletaw for 10 Academy – Week 3 AI Mastery Project*

