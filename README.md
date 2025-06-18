# Insurance Risk Analytics & Predictive Modeling <wbr/>(10 Academy – Week 3)

Predictive, data‑driven pricing for **AlphaCare Insurance Solutions (ACIS)** using historical auto‑insurance data (Feb 2014 – Aug 2015, ≈ 1 M records).

---

##  Project Milestones

| Task       | Deliverable                                  | Status     |
| ---------- | -------------------------------------------- | ---------- |
| **Task 1** | Data loading, cleaning & rich EDA plots      | **✅ Done** |
| **Task 2** | DVC tracking of raw/processed data           | **✅ Done** |
| **Task 3** | A/B hypothesis testing & business insight    | **✅ Done** |
| **Task 4** | Risk‑based ML models + SHAP interpretability | **✅ Done** |

---

##  Folder Layout (abridged)

```
insurance-risk-ap-modeling-w3/
├── configs/                 # YAML configs (EDA / tests / models)
├── data/                    # DVC‑tracked raw & processed CSVs
├── models/                  # Saved .joblib models (classifier / regressors)
├── outputs/eda/             # EDA plots + stats + log
├── reports/                 # JSON metrics, SHAP PNGs, Markdown summaries
├── scripts/                 # CLI entry points: run_eda / run_hypothesis_tests / run_modeling
├── src/                     # Modular Python package (eda, modeling, hypothesis_testing, utils)
├── tests/                   # Pytest unit tests (+ tiny CSV fixture)
└── dvc.yaml / .dvc/         # Data Version Control metadata
```

> **Full tree** shown in [`final_report.md`](reports/final_report.md).

---

## ⚙️ Quick‑start

```bash
# 1. clone & create env
git clone https://github.com/your‑org/insurance-risk-ap-modeling-w3
cd insurance-risk-ap-modeling-w3
python -m venv .venv && source .venv/bin/activate  # Win: .venv\Scripts\activate
pip install -r requirements.txt

# 2. pull data (if you want the full ~400 MB CSVs)
dvc pull                       # requires local remote or GDrive creds

# 3. end‑to‑end pipelines
python scripts/run_eda.py              --config configs/eda_config.yaml
python scripts/run_hypothesis_tests.py --config configs/hypothesis_config.yaml
python scripts/run_modeling.py         --config configs/modeling_config.yaml

# 4. run unit tests
pytest -q
```

---

##  Key Results

| Model                  | Metric  | Score                   |
| ---------------------- | ------- | ----------------------- |
| **Claim Classifier**   | ROC‑AUC | **0.9991**              |
| **Premium Regressor**  | R²      | **0.991**  (RMSE 26.36) |
| **Severity Regressor** | R²      | 0.313  (RMSE 31 255)    |

*Full tables & SHAP plots in* **`reports/`**.

---

##  Top Insights (SHAP)

| Risk Driver                | Effect                              |
| -------------------------- | ----------------------------------- |
| Older **VehicleAge**       | ↑ Severity by ≈ R 1 800 per year    |
| **VehicleType\_Taxi**      | ↑ Claim probability                 |
| **TrackingDevice**         | ↓ Premium & severity                |
| **Gender\_Male**           | ↑ Margin difference (Task 3 result) |
| **Province\_Eastern Cape** | ↑ Severity baseline                 |

---

##  Data Version Control (Task 2)

```bash
# track new artifact
dvc add data/processed/insurance_cleaned_data.csv
git add data/processed/insurance_cleaned_data.csv.dvc
git commit -m "data: track processed CSV with DVC"
dvc push    # sync to remote storage
```

CI runs on a **1 k‑row fixture** (`tests/fixtures/insurance_sample.csv`) to avoid heavy DVC pulls.

---

##  CI / Tests

GitHub Actions (`.github/workflows/ci.yml`) runs on every push / PR:

1. Install deps
2. Download fixture CSV
3. `pytest` (`tests/`): EDA + model sanity checks
4. Flake8 lint

All tests currently **pass**:

```
===== 8 passed in 46 s =====
```

---

##  Business Recommendations

1. **Age‑based pricing**: add loadings for vehicles > 10 yrs (see SHAP).
2. **Gender segmentation**: Male policyholders show higher margin variance — adjust base rate or leverage in marketing.
3. **Device discounts**: Incentivize `TrackingDevice` / `AlarmImmobiliser`; reduces expected severity.
4. **Province targeting**: Eastern Cape has highest claim severity; refine underwriting or set higher deductibles.

See detailed discussion in [`reports/final_report.md`](reports/final_report.md).

---

##  Roadmap

* [ ] Cross‑validation & hyper‑parameter search
* [ ] Streamlit dashboard for underwriters
* [ ] Incorporate external socio‑economic data
* [ ] Tweedie / GLM for severity tail modeling



