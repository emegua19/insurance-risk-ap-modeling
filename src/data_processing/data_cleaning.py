import os

class DataCleaner:
    def __init__(self, processed_output_file):
        self.processed_output_file = processed_output_file
        self.df = None

    def clean_data(self, df):
        """Perform basic data cleaning: fill missing values."""
        if df is None:
            raise ValueError("DataFrame must be provided.")
        self.df = df.copy()  # Work with a copy to avoid modifying the input
        
        # Basic cleaning: fill missing numerical values with median, categorical with mode
        numeric_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].median())
        for col in categorical_cols:
            self.df[col] = self.df[col].fillna(self.df[col].mode()[0])

        # Save cleaned data
        processed_dir = os.path.dirname(self.processed_output_file)
        os.makedirs(processed_dir, exist_ok=True)
        self.df.to_csv(self.processed_output_file, index=False)
        return self.df

    def get_cleaned_data(self):
        """Return the cleaned DataFrame."""
        if self.df is None:
            raise ValueError("Data must be cleaned first.")
        return self.df