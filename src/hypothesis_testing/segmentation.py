import pandas as pd
import logging
from scipy.stats import ttest_ind, chi2_contingency

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

def ab_segment(df: pd.DataFrame, feature: str, group_a: str, group_b: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split DataFrame into two segments A and B based on values of 'feature'.

    Args:
        df (pd.DataFrame): Original DataFrame.
        feature (str): Column name to segment on (e.g., 'PostalCode').
        group_a (str): Value of feature for control group (e.g., '8000').
        group_b (str): Value of feature for test group (e.g., '2000').

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (df_a, df_b) DataFrames for each group.

    Raises:
        KeyError: If 'feature' is not in DataFrame columns.
        ValueError: If either group is empty after segmentation.
    """
    if feature not in df.columns:
        raise KeyError(f"Feature '{feature}' not found in DataFrame.")
    
    df_a = df[df[feature] == group_a].copy()
    df_b = df[df[feature] == group_b].copy()
    
    if df_a.empty or df_b.empty:
        logging.warning(f"Segmentation resulted in empty group(s): Group A ({len(df_a)}), Group B ({len(df_b)})")
        raise ValueError("One or both segments are empty.")
    
    logging.info(f"Segmented {feature}: Group A ({group_a}) = {len(df_a)}, Group B ({group_b}) = {len(df_b)}")
    return df_a, df_b

def ensure_balance(df_a: pd.DataFrame, df_b: pd.DataFrame, covariates: list[str]) -> pd.DataFrame:
    """
    Statistically compare the distributions of covariates between df_a and df_b.

    Args:
        df_a (pd.DataFrame): Group A data.
        df_b (pd.DataFrame): Group B data.
        covariates (list[str]): List of covariate column names to check (e.g., ['Age', 'PlanType']).

    Returns:
        pd.DataFrame: DataFrame with covariates, p-values, and test types.

    Notes:
        Uses t-test for numerical covariates and chi-squared for categorical covariates.
        Logs warnings for unbalanced covariates (p < 0.05).
    """
    balance_records = []

    for cov in covariates:
        if cov not in df_a.columns or cov not in df_b.columns:
            logging.warning(f"Covariate '{cov}' missing in one or both groups, skipping.")
            continue

        if df_a[cov].dtype in ['int64', 'float64']:
            # Numerical: t-test
            _, p = ttest_ind(df_a[cov].dropna(), df_b[cov].dropna(), equal_var=False)
            test_type = "t-test"
            balance_records.append({
                "covariate": cov,
                "p_value": p,
                "test_type": test_type,
                "A_mean": df_a[cov].mean(),
                "B_mean": df_b[cov].mean()
            })
        else:
            # Categorical: chi-squared
            contingency = pd.crosstab(df_a[cov].fillna('Unknown'), df_b[cov].fillna('Unknown'))
            if contingency.shape[0] < 2 or contingency.shape[1] < 2:
                logging.warning(f"Insufficient categories for {cov}, skipping.")
                continue
            _, p, _, _ = chi2_contingency(contingency)
            test_type = "chi-squared"
            balance_records.append({
                "covariate": cov,
                "p_value": p,
                "test_type": test_type
            })

        if p < 0.05:
            logging.warning(f"Covariate {cov} unbalanced (p={p:.4f}, {test_type})")
        else:
            logging.info(f"Covariate {cov} balanced (p={p:.4f}, {test_type})")

    return pd.DataFrame(balance_records)

def flag_imbalances(balance_df: pd.DataFrame, threshold: float = 0.05) -> pd.DataFrame:
    """
    Identify imbalanced covariates based on p-value threshold.

    Args:
        balance_df (pd.DataFrame): Output of ensure_balance() with 'p_value' column.
        threshold (float): Maximum allowed p-value for balance (default 0.05).

    Returns:
        pd.DataFrame: Subset of balance_df where p-value indicates imbalance (p < threshold).
    """
    if 'p_value' not in balance_df.columns:
        logging.error("balance_df must contain 'p_value' column.")
        return pd.DataFrame()
    
    imbalanced = balance_df[balance_df['p_value'] < threshold]
    if not imbalanced.empty:
        logging.warning(f"Imbalanced covariates detected:\n{imbalanced}")
    return imbalanced