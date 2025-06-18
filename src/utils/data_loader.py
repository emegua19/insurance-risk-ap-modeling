import os
import pandas as pd

class DataLoader:
    """Load .txt (|‑delimited) or CSV and convert to DataFrame."""

    def __init__(self, input_dir: str, delimiter: str = ","):
        self.input_dir = input_dir
        self.delimiter = delimiter

    def load_txt(self, filename: str, convert_to_csv: bool = True) -> pd.DataFrame:
        path = os.path.join(self.input_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Raw file not found: {path}")
        df = pd.read_csv(path, sep=self.delimiter, low_memory=False)
        if convert_to_csv:
            csv_name = os.path.splitext(filename)[0] + ".csv"
            csv_path = os.path.join(self.input_dir, csv_name)
            df.to_csv(csv_path, index=False)
        return df

    def load_csv(self, filename: str) -> pd.DataFrame:
        path = os.path.join(self.input_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"CSV file not found: {path}")
        return pd.read_csv(path)
