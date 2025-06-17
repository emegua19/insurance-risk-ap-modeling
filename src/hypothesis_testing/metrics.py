import pandas as pd


def add_has_claim(df: pd.DataFrame, claim_col: str = "TotalClaims") -> pd.DataFrame:
    """
    Add a binary column 'has_claim' indicating whether TotalClaims > 0.
    """
    df = df.copy()
    df["has_claim"] = (df[claim_col] > 0).astype(int)
    return df


def claim_frequency(df: pd.DataFrame, group_col: str = None) -> pd.Series:
    """
    Compute claim frequency as (number of policies with ≥1 claim) / (total policies).
    If group_col is provided, returns frequency per group.
    """
    df = add_has_claim(df)
    if group_col:
        freq = df.groupby(group_col)["has_claim"].mean()
    else:
        freq = df["has_claim"].mean()
    return freq


def claim_severity(df: pd.DataFrame, claim_col: str = "TotalClaims", group_col: str = None) -> pd.Series:
    """
    Compute average claim severity on policies with at least one claim.
    If group_col is provided, returns severity per group.
    """
    df_pos = df[df[claim_col] > 0]
    if group_col:
        sev = df_pos.groupby(group_col)[claim_col].mean()
    else:
        sev = df_pos[claim_col].mean()
    return sev


def loss_ratio(df: pd.DataFrame, claim_col: str = "TotalClaims", premium_col: str = "TotalPremium",
               group_col: str = None) -> pd.Series:
    """
    Compute loss ratio = TotalClaims / TotalPremium.
    If group_col is provided, returns loss ratio per group.
    """
    df = df.copy()
    df["loss_ratio"] = df[claim_col] / df[premium_col].replace({0: pd.NA})
    if group_col:
        lr = df.groupby(group_col)["loss_ratio"].mean()
    else:
        lr = df["loss_ratio"].mean()
    return lr
