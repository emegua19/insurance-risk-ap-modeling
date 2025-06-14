
# Insurance Risk Analytics & Predictive Modeling

## Project Overview

This repository contains the codebase and resources for the **10 Academy: Artificial Intelligence Mastery** project focused on **End-to-End Insurance Risk Analytics & Predictive Modeling** for **AlphaCare Insurance Solutions (ACIS)**.

**Project Goal:**
Analyze historical car insurance claim data (Feb 2014–Aug 2015) to:
- Identify low-risk segments
- Optimize premiums
- Inform marketing strategies in South Africa

---

## Project Tasks

| Task   | Description                     | Status      |
| ------ | ------------------------------- | ----------- |
| Task 1 | Exploratory Data Analysis (EDA) | ✅ Completed |
| Task 2 | Data Version Control (DVC)      | ✅ Completed |
| Task 3 | A/B Hypothesis Testing          | 🔜 Upcoming |
| Task 4 | Predictive Modeling             | 🔜 Upcoming |

---

## Folder Structure

```txt
insurance-risk-ap-modeling/
├── .dvc/                     # DVC metadata
├── .github/workflows/       # GitHub Actions CI/CD config
│   └── ci.yml               # CI pipeline config (pytest + linting)
├── configs/
│   └── eda_config.yaml      # YAML config for EDA pipeline
├── data/
│   ├── raw/                 # Raw data (DVC-tracked)
│   └── processed/           # Cleaned data outputs
├── outputs/
│   └── eda/
│       ├── plots/           # Visualizations (histograms, boxplots, insights)
│       ├── stats/           # Summary CSVs (dtypes, missing, correlation)
│       └── eda_pipeline.log # Run log
├── scripts/
│   └── run_eda.py           # CLI script to run the EDA pipeline
├── src/
│   ├── data_processing/
│   │   ├── data_cleaning.py # Missing value & type handling
│   ├── eda/
│   │   ├── correlation.py   # Correlation matrix, scatter by geography
│   │   ├── descriptive_stats.py # Summary statistics & data types
│   │   ├── visualizations.py    # Histograms, bar plots, insights
│   ├── hypothesis_testing/  # Task 3 logic (planned)
│   ├── modeling/            # Task 4 model logic (planned)
│   ├── utils/
│   │   └── data_loader.py   # Loads and converts raw text data
├── tests/
│   ├── test_eda.py          # Unit tests for EDA functions
│   └── test_models.py       # Unit tests for modeling (placeholder)
├── .dvcignore               # DVC ignore rules
├── .gitignore               # Git ignore rules
├── LICENSE.md               # Project license (MIT)
├── dvc.yaml                 # (optional) DVC pipeline definition (WIP)
├── README.md                # Project overview and instructions
└── requirements.txt         # Python dependencies
```

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/emegua19/insurance-risk-ap-modeling
cd insurance-risk-ap-modeling
```

### 2. Create & Activate a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run the EDA Pipeline

```bash
python scripts/run_eda.py --config configs/eda_config.yaml
```

### 4. View Outputs

- Visualizations: `outputs/eda/plots/`
- CSV Summaries: `outputs/eda/stats/`
- Log File: `outputs/eda/eda_pipeline.log`

---

## DVC Setup (Data Version Control)

### Initial Setup
- Initialize DVC and commit the configuration:
  ```bash
  dvc init
  git add .dvc/ .dvcignore
  git commit -m "Initialize DVC for Task 2"
  ```

### Configure Local Remote
- Set up a local storage directory and configure it as the default remote:
  ```bash
  mkdir /new/dvc-storage
  dvc remote add -d localstorage /new/dvc-storage
  git add .dvc/config
  git commit -m "Configure local remote storage at /new/dvc-storage for DVC"
  ```

### Track and Push Data
- Track the datasets and commit the `.dvc` files:
  ```bash
  dvc add data/raw/MachineLearningRating_v3.csv
  dvc add data/raw/MachineLearningRating_v3.txt
  dvc add data/processed/insurance_cleaned_data.csv
  git add data/raw/MachineLearningRating_v3.csv.dvc data/raw/MachineLearningRating_v3.txt.dvc data/processed/insurance_cleaned_data.csv.dvc
  git commit -m "Track data/raw/MachineLearningRating_v3.csv, data/raw/MachineLearningRating_v3.txt, and data/processed/insurance_cleaned_data.csv with DVC"
  dvc push
  ```

### Reproduce from Remote
- Clone the repository and pull the data:
  ```bash
  git clone https://github.com/emegua19/insurance-risk-ap-modeling
  cd insurance-risk-ap-modeling
  dvc pull
  ```

---

## Git Workflow

### Branching
Create branches for each task:
```bash
git checkout -b task-1
git checkout -b task-2-v2  # Updated to reflect current branch
...
```

### Commit Example
```bash
git add .
git commit -m "Add histogram visualizations for TotalPremium"
```

### Pull Requests
Open PRs from `task-*` branches into `main`.

---

## ⚙️ CI/CD with GitHub Actions

GitHub Actions runs on push/PR:
- Runs `pytest`

### CI Config: `.github/workflows/ci.yml`
See earlier README version for full YAML example.

---

## Task 1: EDA Highlights
- `DataLoader`, `DataCleaner` classes built with OOP
- Visualizations: histograms, boxplots, bar charts, and 3 insight plots
- Output saved in structured folders
- Fully modular and YAML-driven

---

## Task 2: DVC Integration
- DVC initialized and `.dvcignore` configured
- Local remote storage set up at `/new/dvc-storage`
- Datasets (`data/raw/MachineLearningRating_v3.csv`, `data/raw/MachineLearningRating_v3.txt`, `data/processed/insurance_cleaned_data.csv`) tracked with `.dvc` files
- Data pushed to `/new/dvc-storage`
- `.gitignore` updated to exclude data files while allowing `.dvc` tracking

---

## Upcoming Tasks

### Task 3: Hypothesis Testing (Planned)
- Statistical validation (t-tests, chi-square)
- YAML-driven test config
- Results saved as reports or plots
- Code in `src/hypothesis_testing/`

### Task 4: Predictive Modeling (Planned)
- Model training on cleaned data
- Evaluation metrics
- DVC model tracking
- Code in `src/modeling/` and `scripts/train_models.py`

---

## Next Steps
- Merge `task-2-v2` to `main`
- Start `task-3` on June 15, 2025
- Submit GitHub repo by **June 15, 2025, 5:00 PM EAT**

---

## Testing
Run tests:
```bash
pytest tests/
```