# src/preprocessing/data_cleaning.py
import os
import pandas as pd
from typing import Optional, List, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

class DataCleaner:
    """
    Basic data-cleaning utility.

    Responsibilities
    ----------------
    1. Standardise data types (numeric, categorical, datetime, boolean)
    2. Impute missing values (numeric → median, categorical → mode)
    3. Provide cleaned DataFrame for downstream tasks
    4. (Optional) Save cleaned data to CSV via a separate method
    5. Handle specific data quality issues (e.g., zero/negative premiums, mixed types)
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
        self._handle_specific_issues()  # New method for targeted fixes
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
        logging.info(f"Cleaned data saved → {self.processed_output_file}")

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
        - Boolean-like strings → bool
        - Numeric strings (e.g., CapitalOutstanding) → float
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

        # Convert numeric-like strings to float (e.g., CapitalOutstanding)
        numeric_like_cols = ['CapitalOutstanding']  # Add other columns as needed
        for col in numeric_like_cols:
            if col in self.df.columns and self.df[col].dtype == object:
                self.df[col] = pd.to_numeric(
                    self.df[col].str.replace(',', ''),  # Remove commas
                    errors='coerce'  # Convert non-numeric to NaN
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

    def _handle_specific_issues(self) -> None:
        """
        Handle specific data quality issues:
        - Filter out rows with zero or negative TotalPremium
        - Ensure covariate columns (e.g., Age, PlanType) are present or mapped
        """
        # Filter out zero or negative TotalPremium
        if 'TotalPremium' in self.df.columns:
            initial_rows = len(self.df)
            self.df = self.df[self.df['TotalPremium'] > 0]
            removed_rows = initial_rows - len(self.df)
            if removed_rows > 0:
                logging.info(f"Removed {removed_rows} rows with zero/negative TotalPremium")

        # Validate or map covariate columns (e.g., Age, PlanType)
        expected_covariates = {'Age', 'PlanType'}  # Adjust based on your data
        present_covariates = set(self.df.columns) & expected_covariates
        missing_covariates = expected_covariates - present_covariates
        if missing_covariates:
            logging.warning(f"Missing covariates: {missing_covariates}. Check column names or data.")
        for cov in present_covariates:
            if self.df[cov].dtype in ['int64', 'float64']:
                continue  # Numeric types are fine
            elif self.df[cov].dtype == 'object':
                self.df[cov] = pd.to_numeric(self.df[cov], errors='coerce')
                if self.df[cov].isna().all():
                    logging.warning(f"Covariate {cov} contains no valid numeric data after conversion.")
            else:
                logging.warning(f"Covariate {cov} has unexpected type: {self.df[cov].dtype}")