"""
Feature engineering: numeric scaling + categorical one-hot encoding,
returned as a scikit-learn ColumnTransformer inside a Pipeline.
"""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import pandas as pd


class FeatureBuilder:
    """Constructs preprocessing pipeline and train/test splits."""

    def __init__(self,
                 numeric_cols: list[str],
                 cat_cols: list[str],
                 test_size: float = 0.2,
                 random_state: int = 42,
                 stratify_col: str | None = None):
        self.numeric_cols = numeric_cols
        self.cat_cols = cat_cols
        self.test_size = test_size
        self.random_state = random_state
        self.stratify_col = stratify_col
        self.preprocessor: ColumnTransformer | None = None

    def build_preprocessor(self) -> ColumnTransformer:
        """Create a ColumnTransformer for numeric scaling and one-hot encoding."""
        num_pipe = Pipeline([
            ("scaler", StandardScaler())
        ])
        cat_pipe = Pipeline([("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=True))])


        self.preprocessor = ColumnTransformer([
            ("num", num_pipe, self.numeric_cols),
            ("cat", cat_pipe, self.cat_cols),
        ])

        return self.preprocessor

    def train_test_split(self, df: pd.DataFrame, target: str):
        """Split df into train/test, stratifying if requested."""
        from sklearn.model_selection import train_test_split

        X = df.drop(columns=[target])
        y = df[target]
        stratify = df[self.stratify_col] if self.stratify_col else None

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify
        )
        return X_train, X_test, y_train, y_test
