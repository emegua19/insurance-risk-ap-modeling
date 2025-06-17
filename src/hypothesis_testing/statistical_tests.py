import numpy as np
import pandas as pd
from scipy import stats


def chi2_test_counts(df_a: pd.DataFrame, df_b: pd.DataFrame, column: str) -> tuple[float, float]:
    """
    Perform a chi-squared test of independence on counts of a binary column (e.g., has_claim).
    Returns (chi2_stat, p_value).
    """
    counts = pd.DataFrame({
        "A": df_a[column].value_counts(),
        "B": df_b[column].value_counts()
    }).fillna(0).astype(int)
    chi2, p, _, _ = stats.chi2_contingency(counts)
    return chi2, p


def two_proportion_z_test(df_a: pd.DataFrame, df_b: pd.DataFrame, column: str) -> tuple[float, float]:
    """
    Perform a two-proportion z-test for a binary outcome column.
    Returns (z_stat, p_value).
    """
    # successes and trials
    n_a = len(df_a)
    n_b = len(df_b)
    x_a = df_a[column].sum()
    x_b = df_b[column].sum()
    p_pool = (x_a + x_b) / (n_a + n_b)
    z = (x_a/n_a - x_b/n_b) / np.sqrt(p_pool*(1-p_pool)*(1/n_a + 1/n_b))
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return z, p


def welchs_t_test(df_a: pd.DataFrame, df_b: pd.DataFrame, column: str) -> tuple[float, float]:
    """
    Perform Welch's t-test on a continuous column.
    Returns (t_stat, p_value).
    """
    t, p = stats.ttest_ind(df_a[column].dropna(),
                            df_b[column].dropna(),
                            equal_var=False)
    return t, p


def mann_whitney_u(df_a: pd.DataFrame, df_b: pd.DataFrame, column: str) -> tuple[float, float]:
    """
    Perform Mann-Whitney U test (non-parametric) on a continuous column.
    Returns (u_stat, p_value).
    """
    u, p = stats.mannwhitneyu(df_a[column].dropna(),
                              df_b[column].dropna(),
                              alternative="two-sided")
    return u, p
