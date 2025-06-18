"""
Wrapper classes for classifier and regressor models.
"""

import joblib
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor


class ClassifierModel:
    """Handles training, evaluation, and saving of classification models."""

    def __init__(self, model_type: str, params: dict):
        if model_type == "xgboost":
            self.model = XGBClassifier(**params)
        elif model_type == "logistic":
            self.model = LogisticRegression(**params)
        elif model_type == "random_forest_cls":
            self.model = RandomForestClassifier(**params)
        else:
            raise ValueError(f"Unsupported classifier: {model_type}")

    def evaluate(self, y_pred, y_true):
        from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score

        return {
            "roc_auc": roc_auc_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred > 0.5),
            "precision": precision_score(y_true, y_pred > 0.5),
            "recall": recall_score(y_true, y_pred > 0.5),
            "accuracy": accuracy_score(y_true, y_pred > 0.5)
        }

    def save(self, path):
        joblib.dump(self.model, path)


# ------------------------------------------------------------------
# Existing imports (you already have these)
import joblib
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score,
                             recall_score, f1_score,
                             mean_squared_error, mean_absolute_error, r2_score)
# ------------------------------------------------------------------

class RegressorModel:
    """Wrapper that instantiates and evaluates a regression model."""

    def __init__(self, model_name: str, params: dict | None = None):
        params = params or {}
        self.model = self._init_model(model_name, params)

    # ---------- ADD / UPDATE THIS METHOD --------------------------
    def _init_model(self, model_name: str, params: dict):
        if model_name == "linear":
            return LinearRegression(**params)
        elif model_name == "random_forest":
            return RandomForestRegressor(**params)
        elif model_name in ("xgboost", "xgboost_reg"):
            return XGBRegressor(**params)
        else:
            raise ValueError(f"Unsupported regressor: {model_name}")
    # --------------------------------------------------------------

    def fit(self, X, y):
        self.model.fit(X, y)

    def evaluate(self, y_pred, y_true) -> dict:
        return {
            "rmse": mean_squared_error(y_true, y_pred, squared=False),
            "mae": mean_absolute_error(y_true, y_pred),
            "r2": r2_score(y_true, y_pred),
        }

    def save(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
