"""
Run appropriate statistical tests given KPI type.

- Numeric KPI   -> Welch's t-test (or Mann‑Whitney U if specified)
- Binary KPI    -> Chi‑square test of proportions
"""

import warnings
import numpy as np
from scipy import stats
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


def _welch_t(a, b):
    return stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")


def _mw_u(a, b):
    return stats.mannwhitneyu(a, b, alternative="two-sided")


def _chi2(count_a, n_a, count_b, n_b):
    obs = np.array([[count_a, n_a - count_a],
                    [count_b, n_b - count_b]])
    chi2, p, _, _ = stats.chi2_contingency(obs, correction=False)
    return chi2, p


def run_test(a_df, b_df, kpi: str, test_type: str):
    """
    Run the correct statistical test between Group A and Group B.

    Args:
        a_df (pd.DataFrame): Group A data.
        b_df (pd.DataFrame): Group B data.
        kpi (str): Metric to test (e.g., 'margin', 'claim_frequency').
        test_type (str): Type of test to run ('ttest', 'mw_u', 'chi2').

    Returns:
        tuple: (statistic, p_value)
    """

    if kpi == "claim_frequency":
        # Binary outcome → Chi-square test of proportions
        cnt_a, cnt_b = a_df[kpi].sum(), b_df[kpi].sum()
        n_a, n_b = len(a_df), len(b_df)
        logging.info(f"Chi-square test on claim_frequency: Group A size = {n_a}, Group B size = {n_b}")
        chi2, p = _chi2(cnt_a, n_a, cnt_b, n_b)
        return chi2, p

    # Apply claim filtering only for severity KPI (if it appears in config later)
    if kpi == "severity" and "TotalClaims" in a_df.columns and "TotalClaims" in b_df.columns:
        a_df = a_df[a_df["TotalClaims"] > 0]
        b_df = b_df[b_df["TotalClaims"] > 0]
        logging.info(f"Filtered to TotalClaims > 0: Group A = {len(a_df)}, Group B = {len(b_df)}")

    # Numeric tests (margin, severity, etc.)
    if test_type == "mw_u":
        stat, p = _mw_u(a_df[kpi], b_df[kpi])
        logging.info(f"Running Mann–Whitney U test on {kpi}")
    else:
        stat, p = _welch_t(a_df[kpi], b_df[kpi])
        logging.info(f"Running Welch's t-test on {kpi}")

    if np.isnan(p):
        warnings.warn("p-value is NaN — check sample sizes or input data")
    return stat, p
