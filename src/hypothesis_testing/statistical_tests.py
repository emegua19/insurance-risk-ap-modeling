import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, ttest_ind, mannwhitneyu
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

def chi2_test(df_a, df_b, kpi):
    """
    Perform chi-squared test for categorical KPI (e.g., claim presence).

    Args:
        df_a (pd.DataFrame): DataFrame for Group A.
        df_b (pd.DataFrame): DataFrame for Group B.
        kpi (str): KPI to test (e.g., "frequency").

    Returns:
        tuple[float, float]: (statistic, p_value) or (nan, nan) if test fails.

    Notes:
        Tests claim frequency (TotalClaims > 0) across groups.
        Returns NaN if 'TotalClaims' is missing, sample size is too small (< 30), or contingency fails.
    """
    if 'TotalClaims' not in df_a.columns or 'TotalClaims' not in df_b.columns:
        logging.warning("Missing 'TotalClaims' column in one or both groups.")
        return float('nan'), float('nan')
    
    if len(df_a) < 30 or len(df_b) < 30:
        logging.warning(f"Small sample size: Group A ({len(df_a)}), Group B ({len(df_b)})")
        return float('nan'), float('nan')

    has_claim_a = (df_a['TotalClaims'] > 0).astype(int)
    has_claim_b = (df_b['TotalClaims'] > 0).astype(int)

    count_a = has_claim_a.value_counts().reindex([0, 1], fill_value=0)
    count_b = has_claim_b.value_counts().reindex([0, 1], fill_value=0)

    contingency = pd.DataFrame([count_a, count_b], index=["A", "B"])

    try:
        statistic, p_value, _, _ = chi2_contingency(contingency)
        logging.info(f"Chi-squared test: statistic={statistic:.4f}, p={p_value:.4f}")
        return statistic, p_value
    except Exception as e:
        logging.error(f"Chi-squared test failed: {e}")
        return float('nan'), float('nan')

def ttest(df_a, df_b, kpi):
    """
    Perform t-test for numerical KPIs (e.g., severity, margin, loss_ratio).

    Args:
        df_a (pd.DataFrame): DataFrame for Group A.
        df_b (pd.DataFrame): DataFrame for Group B.
        kpi (str): KPI to test (e.g., "severity", "margin", "loss_ratio").

    Returns:
        tuple[float, float]: (statistic, p_value) or (nan, nan) if test fails.

    Notes:
        Returns NaN if required columns are missing, sample size is too small (< 30),
        no claims exist for severity, or zero/negative premiums for loss_ratio.
    """
    if len(df_a) < 30 or len(df_b) < 30:
        logging.warning(f"Small sample size: Group A ({len(df_a)}), Group B ({len(df_b)})")
        return float('nan'), float('nan')

    try:
        if kpi == "severity":
            data_a = df_a[df_a['TotalClaims'] > 0]['TotalClaims']
            data_b = df_b[df_b['TotalClaims'] > 0]['TotalClaims']
            if len(data_a) == 0 or len(data_b) == 0:
                logging.warning("No claim data for severity in one or both groups.")
                return float('nan'), float('nan')
        elif kpi == "margin":
            data_a = df_a['TotalPremium'] - df_a['TotalClaims']
            data_b = df_b['TotalPremium'] - df_b['TotalClaims']
        elif kpi == "loss_ratio":
            if 'TotalClaims' not in df_a.columns or 'TotalPremium' not in df_a.columns or \
               'TotalClaims' not in df_b.columns or 'TotalPremium' not in df_b.columns:
                logging.warning("Missing 'TotalClaims' or 'TotalPremium' column in one or both groups.")
                return float('nan'), float('nan')
            if (df_a['TotalPremium'] <= 0).any() or (df_b['TotalPremium'] <= 0).any():
                logging.warning("Zero or negative 'TotalPremium' values detected.")
                return float('nan'), float('nan')
            data_a = df_a['TotalClaims'] / df_a['TotalPremium']
            data_b = df_b['TotalClaims'] / df_b['TotalPremium']
        else:
            logging.error(f"Unsupported KPI for t-test: {kpi}")
            return float('nan'), float('nan')

        statistic, p_value = ttest_ind(data_a.dropna(), data_b.dropna(), equal_var=False)
        logging.info(f"t-test ({kpi}): statistic={statistic:.4f}, p={p_value:.4f}")
        return statistic, p_value

    except Exception as e:
        logging.error(f"t-test failed for {kpi}: {e}")
        return float('nan'), float('nan')

def mann_whitney_u(df_a, df_b, kpi):
    """
    Perform Mann-Whitney U test for non-parametric numerical KPIs (e.g., severity, loss_ratio).

    Args:
        df_a (pd.DataFrame): DataFrame for Group A.
        df_b (pd.DataFrame): DataFrame for Group B.
        kpi (str): KPI to test (e.g., "severity", "loss_ratio").

    Returns:
        tuple[float, float]: (statistic, p_value) or (nan, nan) if test fails.

    Notes:
        Returns NaN if required columns are missing, sample size is too small (< 30),
        no claims exist for severity, or zero/negative premiums for loss_ratio.
    """
    if len(df_a) < 30 or len(df_b) < 30:
        logging.warning(f"Small sample size: Group A ({len(df_a)}), Group B ({len(df_b)})")
        return float('nan'), float('nan')

    try:
        if kpi == "severity":
            data_a = df_a[df_a['TotalClaims'] > 0]['TotalClaims']
            data_b = df_b[df_b['TotalClaims'] > 0]['TotalClaims']
            if len(data_a) == 0 or len(data_b) == 0:
                logging.warning("No claim data for severity in one or both groups.")
                return float('nan'), float('nan')
        elif kpi == "loss_ratio":
            if 'TotalClaims' not in df_a.columns or 'TotalPremium' not in df_a.columns or \
               'TotalClaims' not in df_b.columns or 'TotalPremium' not in df_b.columns:
                logging.warning("Missing 'TotalClaims' or 'TotalPremium' column in one or both groups.")
                return float('nan'), float('nan')
            if (df_a['TotalPremium'] <= 0).any() or (df_b['TotalPremium'] <= 0).any():
                logging.warning("Zero or negative 'TotalPremium' values detected.")
                return float('nan'), float('nan')
            data_a = df_a['TotalClaims'] / df_a['TotalPremium']
            data_b = df_b['TotalClaims'] / df_b['TotalPremium']
        else:
            logging.error(f"Unsupported KPI for Mann-Whitney U test: {kpi}")
            return float('nan'), float('nan')

        statistic, p_value = mannwhitneyu(data_a.dropna(), data_b.dropna(), alternative='two-sided')
        logging.info(f"Mann-Whitney U test ({kpi}): statistic={statistic:.4f}, p={p_value:.4f}")
        return statistic, p_value

    except Exception as e:
        logging.error(f"Mann-Whitney U test failed for {kpi}: {e}")
        return float('nan'), float('nan')

def run_test(df_a, df_b, test_type, kpi):
    """
    Dispatch the appropriate statistical test.

    Args:
        df_a (pd.DataFrame): DataFrame for Group A.
        df_b (pd.DataFrame): DataFrame for Group B.
        test_type (str): Type of test ("chi2", "ttest", or "mw_u").
        kpi (str): KPI to test (e.g., "frequency", "severity", "margin", "loss_ratio").

    Returns:
        tuple[float, float]: (statistic, p_value) or (nan, nan) if test fails.

    Notes:
        Maps test_type to the corresponding function and handles exceptions.
        Supported KPIs: frequency (chi2), severity/margin/loss_ratio (ttest/mw_u).
    """
    tests = {
        "chi2": chi2_test,
        "ttest": ttest,
        "mw_u": mann_whitney_u
    }

    if test_type not in tests:
        logging.error(f"Unsupported test type: {test_type}")
        return float('nan'), float('nan')

    try:
        return tests[test_type](df_a, df_b, kpi)
    except Exception as e:
        logging.error(f"Error running {test_type} on {kpi}: {e}")
        return float('nan'), float('nan')