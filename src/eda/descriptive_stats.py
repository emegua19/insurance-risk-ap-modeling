import pandas as pd

class DescriptiveStats:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def basic_summary(self):
        """Calculate variability for numerical features."""
        return self.df.describe()

    def review_data_structure(self):
        """Review data types of each column."""
        return self.df.dtypes

    def check_missing_values(self):
        """Check for missing values in the dataset."""
        return self.df.isnull().sum()