# src/utils/data_loader.py
import os
from pathlib import Path
from typing import Optional, Tuple, Dict

import pandas as pd
import logging

logger = logging.getLogger(__name__)

class DataLoader:
    """
    Loads raw insurance data (pipe-delimited .txt) and optionally converts it to CSV.

    Parameters
    ----------
    input_dir : str
        Directory containing raw files.
    delimiter : str
        Column delimiter for the raw text file (default '|').
    """

    def __init__(self, input_dir: str = "data/raw", delimiter: str = "|") -> None:
        self.input_dir = Path(input_dir)
        self.delimiter = delimiter
        self.df: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #
    def load_txt(self, filename: str, convert_to_csv: bool = True, dtype: Dict[str, str] = None, encoding: str = 'utf-8') -> pd.DataFrame:
        """
        Load a pipe-delimited .txt file. Optionally write a converted .csv.

        Parameters
        ----------
        filename : str
            Name of the input file.
        convert_to_csv : bool, optional
            Whether to save a .csv version (default True).
        dtype : dict, optional
            Dictionary of column names to data types (e.g., {'CapitalOutstanding': 'float'}).
        encoding : str, optional
            Encoding to use for reading the file (default 'utf-8').

        Returns
        -------
        pd.DataFrame
            Loaded DataFrame.
        """
        self._validate_input_dir()
        input_path = self._validate_file(filename, expected_ext=".txt")

        logger.info(f"Loading raw data from {input_path} with encoding {encoding}")
        self.df = pd.read_csv(input_path, sep=self.delimiter, low_memory=False, dtype=dtype, encoding=encoding)

        if convert_to_csv:
            csv_path = input_path.with_suffix(".csv")
            logger.info(f"Saving converted CSV to {csv_path}")
            self.df.to_csv(csv_path, index=False, encoding=encoding)

        self._validate_required_columns()  # Optional validation
        return self.df

    def load_csv(self, filename: str, dtype: Dict[str, str] = None, encoding: str = 'utf-8') -> pd.DataFrame:
        """
        Load an existing CSV file directly.

        Useful when conversion has already been done in a previous pipeline run.

        Parameters
        ----------
        filename : str
            Name of the input file.
        dtype : dict, optional
            Dictionary of column names to data types (e.g., {'CapitalOutstanding': 'float'}).
        encoding : str, optional
            Encoding to use for reading the file (default 'utf-8').
        """
        self._validate_input_dir()
        csv_path = self._validate_file(filename, expected_ext=".csv")
        logger.info(f"Loading CSV data from {csv_path} with encoding {encoding}")
        self.df = pd.read_csv(csv_path, low_memory=False, dtype=dtype, encoding=encoding)
        self._validate_required_columns()  # Optional validation
        return self.df

    def get_data(self) -> pd.DataFrame:
        """Return the cached DataFrame (raise if not loaded)."""
        if self.df is None:
            raise ValueError("No data loaded. Call load_txt() or load_csv() first.")
        return self.df

    # ------------------------------------------------------------------ #
    # Internal helpers                                                   #
    # ------------------------------------------------------------------ #
    def _validate_input_dir(self) -> None:
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Directory {self.input_dir} not found.")

    def _validate_file(self, filename: str, expected_ext: str) -> Path:
        path = self.input_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"{path} not found.")
        if path.suffix.lower() != expected_ext:
            raise ValueError(f"{filename} must have extension {expected_ext}")
        return path

    def _validate_required_columns(self) -> None:
        """Validate the presence of required columns (optional, customizable)."""
        required_columns = {'TotalClaims', 'TotalPremium', 'CapitalOutstanding'}  # Adjust as needed
        missing_columns = required_columns - set(self.df.columns)
        if missing_columns:
            logger.warning(f"Missing required columns: {missing_columns}")