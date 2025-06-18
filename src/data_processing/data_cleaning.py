import os
import pandas as pd


class DataCleaner:
    """
    Clean raw insurance data.

    * Drops rows with non‑positive premium
    * Median‑imputes numeric NaN
    * Mode‑imputes categorical NaN (safe, no chained‑assignment warnings)
    * Parses TransactionMonth
    * Drops duplicates
    """

    def __init__(self, output_path: str):
        self.output_path = output_path
        self.df: pd.DataFrame | None = None

    # --------------------------------------------------
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # ▸ 1. Keep only positive premium rows
        df = df[df["TotalPremium"] > 0]

        # ▸ 2. Impute numeric columns (median)
        num_cols = df.select_dtypes(include=["int64", "float64"]).columns
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())

        # ▸ 3. Impute categoricals (mode)
        cat_cols = df.select_dtypes(include="object").columns
        for col in cat_cols:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].mode().iloc[0])

        # ▸ 4. Parse dates
        df["TransactionMonth"] = pd.to_datetime(
            df["TransactionMonth"], errors="coerce"
        )
        df.dropna(subset=["TransactionMonth"], inplace=True)

        # ▸ 5. Drop duplicates
        df.drop_duplicates(subset=["UnderwrittenCoverID", "TransactionMonth"],
                           inplace=True)

        # ▸ 6. Add claim‑frequency flag (all zeros right now)
        df["ClaimOccurred"] = (df["TotalClaims"] > 0).astype(int)

        self.df = df
        return df

    # --------------------------------------------------
    def save_cleaned_data(self) -> None:
        assert self.df is not None, "Run clean_data() first"
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        self.df.to_csv(self.output_path, index=False)
