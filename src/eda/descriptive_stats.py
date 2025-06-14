import pandas as pd

class DescriptiveStats:
    """
    A class to compute basic descriptive statistics, data structure overview,
    and missing value summaries for EDA.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize with a cleaned DataFrame.
        """
        self.df = df

    def basic_summary(self) -> pd.DataFrame:
        """
        Returns
        -------
        pd.DataFrame
            Descriptive statistics (count, mean, std, min, max, quartiles) 
            for all numerical columns.
        """
        return self.df.describe()

    def review_data_structure(self) -> pd.Series:
        """
        Returns
        -------
        pd.Series
            Data types for all columns in the dataset.
        """
        return self.df.dtypes

    def check_missing_values(self) -> pd.Series:
        """
        Returns
        -------
        pd.Series
            Count of missing values per column.
        """
        return self.df.isnull().sum()

    def missing_summary(self) -> pd.DataFrame:
        """
        Returns
        -------
        pd.DataFrame
            Summary of missing values including both count and percentage.
        """
        missing = self.df.isnull().sum()
        percent = self.df.isnull().mean() * 100
        return pd.DataFrame({
            "Missing": missing,
            "Percent": percent.round(2)
        })

    def generate_report(self) -> dict:
        """
        Returns
        -------
        dict
            A combined dictionary of all core summaries:
            - summary statistics
            - data types
            - missing value count and percent
        """
        return {
            "summary_stats": self.basic_summary(),
            "data_types": self.review_data_structure(),
            "missing_report": self.missing_summary()
        }
