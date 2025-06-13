import pandas as pd
import os

class DataLoader:
    def __init__(self, input_dir="data/raw/"):
        self.input_dir = input_dir
        self.df = None

    def load_and_convert(self, filename):
        """Load raw data from .txt file and convert to .csv, storing in data/raw/."""
        if not os.path.exists(self.input_dir):
            raise FileNotFoundError(f"Directory {self.input_dir} not found.")
        
        input_path = os.path.join(self.input_dir, filename)
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"{input_path} not found.")
        
        # Check if file is .txt
        if not input_path.lower().endswith('.txt'):
            raise ValueError(f"{filename} is not a .txt file.")
        
        # Load .txt file with '|' separator and convert to .csv
        self.df = pd.read_csv(input_path, sep="|", low_memory=False)
        csv_filename = os.path.splitext(filename)[0] + ".csv"
        output_path = os.path.join(self.input_dir, csv_filename)
        self.df.to_csv(output_path, index=False)
        return self.df, output_path

    def get_data(self):
        """Return the loaded DataFrame."""
        if self.df is None:
            raise ValueError("Data must be loaded first using load_and_convert().")
        return self.df