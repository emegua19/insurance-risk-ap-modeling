# Insurance Risk Analytics & Predictive Modeling

##  Project Overview

This repository contains the codebase and resources for the **10 Academy: Artificial Intelligence Mastery** project focused on **End-to-End Insurance Risk Analytics & Predictive Modeling** for **AlphaCare Insurance Solutions (ACIS)**.

**Goal:**  
Analyze historical car insurance claim data (Feb 2014–Aug 2015) to:
- Identify low-risk segments
- Optimize premiums
- Inform marketing strategies in South Africa

###  Project Tasks
- **Exploratory Data Analysis (EDA)** to uncover patterns in risk and profitability  
- **Data Version Control (DVC)** for reproducible pipelines  
- **A/B Hypothesis Testing** to validate risk drivers  
- **Predictive Modeling** to estimate claim severity and optimize premiums  

---

##  Folder Structure

```

├── data/                     # Raw and processed datasets (tracked by DVC)
│   ├── raw/                 # Original, unprocessed data
│   ├── processed/           # Cleaned and transformed data
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
├── .gitignore                # Git ignore file
├── dvc.yaml                  # DVC pipeline configuration
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation

````

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/insurance-risk-analytics.git
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
dvc remote add -d localstorage /path/to/your/local/storage
dvc add data/raw/<data.csv>
dvc push
```

### 4. Create Branch for Task 1

```bash
git checkout -b task-1
```

---

##  Git Workflow

###  Branching

* Create a new branch for each task: `task-1`, `task-2`, etc.

###  Commits

Commit at least **three times daily** with clear messages:

```bash
git add .
git commit -m "Add initial EDA script for data summarization"
git commit -m "Implement histogram plotting for numerical variables"
git commit -m "Add outlier detection using box plots"
```

###  Pull Requests

* Merge task branches into `main` via pull requests on GitHub.

###  Version Control

* Use Git for code versioning
* Use DVC for dataset and pipeline versioning

---

##  CI/CD with GitHub Actions

A CI/CD pipeline is set up to:

* Run tests (`pytest`)
* Run linting (`flake8`)
* Trigger on push or PR to `main` or `task-*` branches

###  CI/CD Configuration

The file: `.github/workflows/ci.yml`

```yml
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

##  Python Dependencies

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

