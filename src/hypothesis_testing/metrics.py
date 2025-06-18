"""
Compute portfolio-level KPIs safely, even when TotalClaims == 0.
"""

import pandas as pd
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


def add_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds calculated insurance KPIs to the DataFrame:
    - margin: TotalPremium - TotalClaims
    - loss_ratio: TotalClaims / TotalPremium (nan if TotalPremium == 0)
    - claim_frequency: Binary flag indicating whether a claim occurred
    - loss_ratio_capped: (optional) clipped version of loss_ratio to handle outliers

    Args:
        df (pd.DataFrame): Insurance dataset with TotalPremium and TotalClaims

    Returns:
        pd.DataFrame: DataFrame with additional KPI columns
    """
    df = df.copy()

    if "TotalPremium" not in df.columns or "TotalClaims" not in df.columns:
        logging.warning("Missing 'TotalPremium' or 'TotalClaims' in input data.")
        return df

    # Margin = Premium – Claims
    df["margin"] = df["TotalPremium"] - df["TotalClaims"]

    # Loss ratio (with safe division)
    df["loss_ratio"] = np.where(
        df["TotalPremium"] > 0,
        df["TotalClaims"] / df["TotalPremium"],
        np.nan
    )

    # Binary flag for claim frequency (1 if any claim, else 0)
    df["claim_frequency"] = (df["TotalClaims"] > 0).astype(int)

    # Optional: Capped loss ratio to reduce outlier impact
    df["loss_ratio_capped"] = df["loss_ratio"].clip(upper=5.0)

    # Log averages
    logging.info(f"Average Margin: {df['margin'].mean():.2f}")
    logging.info(f"Average Loss Ratio: {df['loss_ratio'].mean():.4f}")
    logging.info(f"Claim Frequency: {df['claim_frequency'].mean():.4f}")

    return df
