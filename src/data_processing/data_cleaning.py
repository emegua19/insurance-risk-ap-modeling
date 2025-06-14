# src/preprocessing/data_cleaning.py
import os
import pandas as pd
from typing import Optional, List, Union


class DataCleaner:
    """
    Basic data‑cleaning utility.

    Responsibilities
    ----------------
    1. Standardise data types (numeric, categorical, datetime, boolean)
    2. Impute missing values (numeric → median, categorical → mode)
    3. Provide cleaned DataFrame for downstream tasks
    4. (Optional) Save cleaned data to CSV via a separate method
    """

    def __init__(self, processed_output_file: str) -> None:
        self.processed_output_file = processed_output_file
        self.df: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform basic data cleaning.

        Parameters
        ----------
        df : pd.DataFrame
            Raw DataFrame to clean.

        Returns
        -------
        pd.DataFrame
            Cleaned copy of the original DataFrame.
        """
        if df is None:
            raise ValueError("DataFrame must be provided.")

        self.df = df.copy(deep=True)  # work on a copy to preserve original
        self._standardise_types()
        self._impute_missing()
        return self.df

    def save_cleaned_data(self) -> None:
        """
        Save the cleaned DataFrame to `self.processed_output_file`.
        Raises an error if `clean_data()` has not been called.
        """
        if self.df is None:
            raise ValueError("Data must be cleaned before saving.")

        processed_dir = os.path.dirname(self.processed_output_file)
        os.makedirs(processed_dir, exist_ok=True)
        self.df.to_csv(self.processed_output_file, index=False)
        print(f"[INFO] Cleaned data saved → {self.processed_output_file}")

    def get_cleaned_data(self) -> pd.DataFrame:
        """
        Return the cleaned DataFrame.

        Returns
        -------
        pd.DataFrame
            The cleaned data.
        """
        if self.df is None:
            raise ValueError("Data must be cleaned first.")
        return self.df

    # ------------------------------------------------------------------ #
    # Internal helpers (prefixed with _)                                 #
    # ------------------------------------------------------------------ #
    def _standardise_types(self) -> None:
        """
        Standardise common column types:
        - Dates → pandas datetime
        - Boolean‑like strings → bool
        """
        # Convert TransactionMonth (if present) to datetime
        if "TransactionMonth" in self.df.columns:
            self.df["TransactionMonth"] = pd.to_datetime(
                self.df["TransactionMonth"], errors="coerce"
            )

        # Convert yes/no or true/false strings to bool
        bool_like_cols: List[str] = [
            col
            for col in self.df.columns
            if self.df[col].dtype == object
            and set(self.df[col].str.lower().dropna().unique()).issubset(
                {"yes", "no", "true", "false"}
            )
        ]
        for col in bool_like_cols:
            self.df[col] = (
                self.df[col]
                .str.lower()
                .map({"yes": True, "true": True, "no": False, "false": False})
            )

    def _impute_missing(self) -> None:
        """
        Fill missing values:
        - numeric columns → median
        - categorical columns → mode (most frequent)
        - boolean → False
        """
        numeric_cols = self.df.select_dtypes(include=["int64", "float64"]).columns
        cat_cols = self.df.select_dtypes(include=["object"]).columns
        bool_cols = self.df.select_dtypes(include=["bool"]).columns

        # Numeric: median
        self.df[numeric_cols] = self.df[numeric_cols].fillna(
            self.df[numeric_cols].median()
        )

        # Categorical: mode (first mode value)
        for col in cat_cols:
            if self.df[col].isna().any():
                self.df[col].fillna(self.df[col].mode().iloc[0], inplace=True)

        # Boolean: default to False
        for col in bool_cols:
            self.df[col].fillna(False, inplace=True)
