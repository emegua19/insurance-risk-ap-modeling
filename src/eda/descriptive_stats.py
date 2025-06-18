import pandas as pd

class DescriptiveStats:
    """Generate summary statistics and data‑type information."""

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def basic_summary(self) -> pd.DataFrame:
        return self.df.describe()

    def review_data_structure(self) -> pd.Series:
        return self.df.dtypes

    def missing_summary(self) -> pd.DataFrame:
        miss = self.df.isnull().sum()
        pct = (miss / len(self.df) * 100).round(2)
        return pd.DataFrame({"missing_count": miss, "missing_pct": pct})
