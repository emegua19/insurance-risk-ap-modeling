import pandas as pd


def ab_segment(df: pd.DataFrame, feature: str, group_a: str, group_b: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split df into two segments A and B based on values of 'feature'.
    
    Parameters:
    - df: original DataFrame
    - feature: column name to segment on
    - group_a: value of feature for control group
    - group_b: value of feature for test group
    
    Returns: (df_A, df_B)
    """
    if feature not in df.columns:
        raise KeyError(f"Feature '{feature}' not found in DataFrame.")
    df_a = df[df[feature] == group_a].copy()
    df_b = df[df[feature] == group_b].copy()
    return df_a, df_b


def ensure_balance(df_a: pd.DataFrame, df_b: pd.DataFrame, covariates: list[str]) -> pd.DataFrame:
    """
    Return a DataFrame comparing the distributions of covariates between df_a and df_b.
    Use this to check whether the two groups are balanced.
    """
    balance = {}
    for cov in covariates:
        if df_a[cov].dtype.name == "category" or df_a[cov].dtype == object:
            # Compare proportions
            prop_a = df_a[cov].value_counts(normalize=True)
            prop_b = df_b[cov].value_counts(normalize=True)
            balance[cov] = pd.DataFrame({"A": prop_a, "B": prop_b}).fillna(0)
        else:
            # Compare means and std
            balance[cov] = {
                "A_mean": df_a[cov].mean(),
                "A_std": df_a[cov].std(),
                "B_mean": df_b[cov].mean(),
                "B_std": df_b[cov].std(),
            }
    return balance
