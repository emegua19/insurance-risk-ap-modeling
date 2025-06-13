# Insurance Risk Analytics & Predictive Modeling

## Project Overview

This repository contains the codebase and resources for the **10 Academy: Artificial Intelligence Mastery** project focused on **End-to-End Insurance Risk Analytics & Predictive Modeling** for **AlphaCare Insurance Solutions (ACIS)**.

**Goal:**  
Analyze historical car insurance claim data (Feb 2014–Aug 2015) to:
- Identify low-risk segments  
- Optimize premiums  
- Inform marketing strategies in South Africa  

### Project Tasks
- **Exploratory Data Analysis (EDA)** to uncover patterns in risk and profitability  
- **Data Version Control (DVC)** for reproducible pipelines  
- **A/B Hypothesis Testing** to validate risk drivers  
- **Predictive Modeling** to estimate claim severity and optimize premiums  

---

## Folder Structure

```

├── data/                     # Raw and processed datasets (tracked by DVC)
│   ├── raw/                 # Original, unprocessed data
│   ├── processed/           # Cleaned and transformed data (includes EDA plots)
├── src/                      # Source code for reusable modules and functions
│   ├── eda/                 # EDA-related functions
│   ├── modeling/            # Machine learning model implementations
│   ├── preprocessing/       # Data cleaning and feature engineering
│   └── utils/               # Utility functions (e.g., logging, helpers)
├── scripts/                  # Standalone scripts for specific tasks
│   ├── data\_preparation.py  # Script for data cleaning and preprocessing
│   ├── run\_eda.py           # Script for running EDA
│   ├── hypothesis\_tests.py  # Script for A/B hypothesis testing
│   └── train\_models.py      # Script for training and evaluating models
├── notebooks/                # Jupyter notebooks for exploratory analysis
│   ├── eda.ipynb            # Notebook for exploratory data analysis
│   ├── hypothesis\_testing.ipynb # Notebook for A/B testing
│   └── modeling.ipynb       # Notebook for model development
├── tests/                    # Unit tests for source code
│   ├── test\_eda.py          # Tests for EDA functions
│   └── test\_modeling.py     # Tests for modeling functions
├── configs/                  # Configuration files (e.g., model parameters, DVC settings)
│   ├── dvc.yaml             # DVC pipeline configuration
│   └── model\_config.yaml    # Model hyperparameters
├── .github/                  # GitHub Actions workflows
│   └── workflows/
│       └── ci.yml           # CI/CD pipeline configuration
├── .dvc/                     # DVC configuration and cache
├── .gitignore                # Git ignore file
├── dvc.yaml                  # DVC pipeline configuration
├── LICENSE.md                # Project license
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies

````

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/emegua19/insurance-risk-ap-modeling
cd insurance-risk-analytics
````

### 2. Set Up Python Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Initialize DVC

```bash
dvc init
dvc remote add -d localstorage /path/to/your/local/storage  # Replace with your storage path (e.g., ~/dvc_storage)
dvc pull  # Retrieve data from remote storage
```

### 4. Create Branch for Tasks

```bash
# For Task 1 (EDA)
git checkout -b task-1

# For Task 2 (DVC)
git checkout -b task-2

# For Task 3 (A/B Hypothesis Testing)
git checkout -b task-3

# For Task 4 (Predictive Modeling)
git checkout -b task-4
```

---

## Git Workflow

### Branching

* Create a new branch for each task: `task-1`, `task-2`, `task-3`, `task-4`, etc.

### Commits

* Commit at least **three times daily** with clear messages:

```bash
git add .
git commit -m "Add initial EDA script for data summarization"
git commit -m "Implement histogram plotting for numerical variables"
git commit -m "Add outlier detection using box plots"
```

### Pull Requests

* Merge task branches into `main` via pull requests on GitHub.

### Version Control

* Use Git for code versioning.
* Use DVC for dataset and pipeline versioning.

---

## CI/CD with GitHub Actions

A CI/CD pipeline is set up to:

* Run tests (`pytest`)
* Run linting (`flake8`)
* Trigger on push or PR to `main` or `task-*` branches

### CI/CD Configuration

File: `.github/workflows/ci.yml`

```yaml
name: CI Pipeline

on:
  push:
    branches:
      - main
  pull_request:
    branches:
      - main

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt

      - name: Run tests
        run: |
          pytest tests/

      - name: Lint code
        run: |
          pip install flake8
          flake8 src/ scripts/ --max-line-length=120
```

---

## Python Dependencies

The following versions are tested for **Python 3.10+** and **Ubuntu latest**:

```txt
pandas==2.2.2
numpy==1.26.4
matplotlib==3.8.4
seaborn==0.13.2
scikit-learn==1.4.2
xgboost==2.0.3
shap==0.44.1
dvc==3.51.1
pytest==8.2.1
flake8==7.0.0
jupyter==1.0.0
python-dotenv==1.0.1
```

---

## Task Progress

###  Task 1: Exploratory Data Analysis (EDA)

* **Status**: Completed
* **Details**:

  * Implemented `DataLoader` in `src/preprocessing/`
  * Implemented EDA modules: `EDAProcessor`, `CorrelationAnalysis`, `DescriptiveStats`, `Visualizer`
  * Created `scripts/run_eda.py` to run full EDA
  * Documented visuals in `notebooks/eda.ipynb`
  * Added 3 creative plots:

    * Loss Ratio by Province
    * Claim Frequency Over Time
    * Total Claims by Vehicle Make
  * Unit tests added in `tests/test_eda.py`
* **Branch**: Merged `task-1` into `main` via PR

###  Task 2: Data Version Control (DVC)

* **Status**: Completed
* **Details**:

  * Installed and initialized DVC (`dvc==3.51.1`)
  * Configured remote storage at `/path/to/your/local/storage`
  * Added raw and processed data to DVC tracking
  * Pushed data to remote
  * Created simple `dvc.yaml` for data pipeline
* **Branch**: Merged `task-2` into `main` via PR

---

##  Upcoming Tasks

###  Task 3: A/B Hypothesis Testing

* **Status**: Not Started
* **Next Step**:

  * Create `task-3` branch
  * Validate risk drivers using statistical tests (e.g., t-test, chi-square)
  * Implement in `src/hypothesis_testing/`
  * Document in `notebooks/hypothesis_testing.ipynb`

###  Task 4: Predictive Modeling

* **Status**: Not Started
* **Next Step**:

  * Create `task-4` branch
  * Build models to predict claim severity and optimize premium pricing
  * Use `src/modeling/` and document in `notebooks/modeling.ipynb`

---

##  Next Steps

* Verify `main` reflects Task 1 and Task 2
* Run:

  ```bash
  dvc pull
  python scripts/run_eda.py
  ```
* Start `task-3` branch on **June 14, 2025**
* Submit GitHub link by **June 15, 2025, 8:00 PM UTC**

---

##  Notes

* Replace `/path/to/your/local/storage` with your actual DVC path (e.g., `~/dvc_storage`)
* Ensure raw and cleaned CSVs are DVC-tracked

---

###  Commands to Update README

```bash
git add README.md
git commit -m "Update README.md with completed Task 1 and Task 2, and plan for Task 3 and 4"
git push origin task-2  # If you're still on task-2 branch
```

