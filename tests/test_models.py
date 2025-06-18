# tests/test_models.py

import pytest
import os
import pandas as pd
from sklearn.pipeline import Pipeline
from src.hypothesis_testing.metrics import add_metrics
from src.modeling.features import FeatureBuilder
from src.modeling.models import ClassifierModel, RegressorModel

DATA_PATH = "data/processed/insurance_cleaned_data.csv"
FALLBACK_SAMPLE = "tests/fixtures/insurance_sample.csv"

@pytest.fixture(scope="module")
def cleaned_df():
    path = DATA_PATH if os.path.exists(DATA_PATH) else FALLBACK_SAMPLE
    if not os.path.exists(path):
        pytest.skip("❌ No dataset found for modeling tests")
    from src.hypothesis_testing.metrics import add_metrics
    return add_metrics(pd.read_csv(path, low_memory=False))

def test_feature_builder(cleaned_df):
    num_cols = cleaned_df.select_dtypes("number").columns.tolist()
    for col in ["ClaimOccurred", "TotalPremium", "TotalClaims", "margin", "loss_ratio", "claim_frequency"]:
        if col in num_cols:
            num_cols.remove(col)

    cat_cols = cleaned_df.select_dtypes("object").columns.tolist()

    fb = FeatureBuilder(
        numeric_cols=num_cols,
        cat_cols=cat_cols,
        test_size=0.2,
        random_state=42,
        stratify_col="ClaimOccurred"
    )
    preproc = fb.build_preprocessor()
    assert preproc is not None

def test_classifier_model(cleaned_df):
    fb = FeatureBuilder(
        numeric_cols=cleaned_df.select_dtypes("number").drop(columns=["ClaimOccurred", "TotalPremium", "TotalClaims", "margin", "loss_ratio", "claim_frequency"]).columns.tolist(),
        cat_cols=cleaned_df.select_dtypes("object").columns.tolist(),
        test_size=0.2,
        random_state=42,
        stratify_col="ClaimOccurred"
    )
    X_train, X_test, y_train, y_test = fb.train_test_split(cleaned_df, target="ClaimOccurred")
    preproc = fb.build_preprocessor()

    clf = ClassifierModel("xgboost", {"use_label_encoder": False, "eval_metric": "logloss"})
    clf_pipe = Pipeline([("prep", preproc), ("model", clf.model)])
    clf_pipe.fit(X_train, y_train)

    assert clf_pipe.predict(X_test).shape[0] == y_test.shape[0]

def test_regressor_model(cleaned_df):
    fb = FeatureBuilder(
        numeric_cols=cleaned_df.select_dtypes("number").drop(columns=["ClaimOccurred", "TotalPremium", "TotalClaims", "margin", "loss_ratio", "claim_frequency"]).columns.tolist(),
        cat_cols=cleaned_df.select_dtypes("object").columns.tolist(),
        test_size=0.2,
        random_state=42,
        stratify_col="ClaimOccurred"
    )
    X_train, X_test, y_train, y_test = fb.train_test_split(cleaned_df, target="TotalPremium")
    preproc = fb.build_preprocessor()

    reg = RegressorModel("xgboost", {"n_estimators": 50})
    reg_pipe = Pipeline([("prep", preproc), ("model", reg.model)])
    reg_pipe.fit(X_train, y_train)

    y_pred = reg_pipe.predict(X_test)
    assert len(y_pred) == len(y_test)
