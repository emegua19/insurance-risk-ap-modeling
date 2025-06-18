"""
Utility to split DataFrame into Group A vs Group B for A/B testing.
"""

import pandas as pd
import logging
from typing import Tuple, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


def segment_groups(df: pd.DataFrame,
                   feature: str,
                   group_a_val: Any,
                   group_b_val: Any) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits the DataFrame into two groups based on the value of a categorical feature.

    Args:
        df (pd.DataFrame): Input dataset.
        feature (str): Column name to segment on.
        group_a_val (Any): Value of feature representing Group A.
        group_b_val (Any): Value of feature representing Group B.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Group A and Group B DataFrames.
    """
    if feature not in df.columns:
        logging.error(f"Feature '{feature}' not found in DataFrame.")
        return pd.DataFrame(), pd.DataFrame()

    a_df = df[df[feature] == group_a_val]
    b_df = df[df[feature] == group_b_val]

    logging.info(f"Segmented '{feature}': Group A = '{group_a_val}' ({len(a_df)} rows), "
                 f"Group B = '{group_b_val}' ({len(b_df)} rows)")

    if a_df.empty or b_df.empty:
        logging.warning(f"One of the groups is empty. A: {len(a_df)} rows, B: {len(b_df)} rows")

    return a_df, b_df
