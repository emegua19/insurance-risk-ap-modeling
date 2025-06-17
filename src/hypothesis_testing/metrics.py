import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# Add to src/hypothesis_testing/metrics.py after calculate_margin
# Replace the calculate_loss_ratio function in src/hypothesis_testing/metrics.py
def calculate_loss_ratio(df, group_by=None, as_series=False):
    """Calculate loss ratio: TotalClaims / TotalPremium.

    Args:
        df (pd.DataFrame): DataFrame with 'TotalClaims' and 'TotalPremium' columns.
        group_by (str or list of str, optional): Column(s) to group by (e.g., 'PostalCode').
        as_series (bool): If True, return Series with per-group values; if False, return mean.

    Returns:
        float or pd.Series: Overall or group-wise loss ratio (NaN if invalid data).

    Notes:
        Returns NaN if 'TotalClaims' or 'TotalPremium' is missing or zero/negative premiums exist.
    """
    try:
        if 'TotalClaims' not in df.columns or 'TotalPremium' not in df.columns:
            logging.warning("Missing 'TotalClaims' or 'TotalPremium' column.")
            return float('nan')

        # Filter out rows with zero or negative TotalPremium
        valid_df = df[df['TotalPremium'] > 0].copy()
        if valid_df.empty:
            logging.warning("No valid data after filtering zero/negative TotalPremium.")
            return float('nan')

        loss_ratio = valid_df['TotalClaims'] / valid_df['TotalPremium']
        
        if group_by:
            if isinstance(group_by, str):
                group_by = [group_by]
            loss_ratio_grouped = valid_df.groupby(group_by).apply(lambda x: (x['TotalClaims'] / x['TotalPremium']).mean())
            if loss_ratio_grouped.empty:
                logging.warning(f"No data after grouping by {group_by}")
                return float('nan')
            logging.info(f"Loss ratio by {group_by}: min={loss_ratio_grouped.min():.4f}, max={loss_ratio_grouped.max():.4f}, count={len(loss_ratio_grouped)}")
            return loss_ratio_grouped if as_series else loss_ratio_grouped.mean()

        loss_ratio_mean = loss_ratio.mean()
        logging.info(f"Overall Loss Ratio: {loss_ratio_mean:.4f} (based on {len(valid_df)} policies)")
        return loss_ratio_mean

    except Exception as e:
        logging.error(f"Error calculating loss ratio: {e}")
        return float('nan')
    
def calculate_claim_frequency(df, group_by=None, as_series=False):
    """Calculate claim frequency: proportion of policies with at least one claim.

    Args:
        df (pd.DataFrame): DataFrame containing 'TotalClaims' column.
        group_by (str or list of str, optional): Column(s) to group by (e.g., 'PostalCode').
        as_series (bool): If True, return Series with per-group values; if False, return mean.

    Returns:
        float or pd.Series: Overall or group-wise claim frequency (NaN if invalid data).
    
    Notes:
        Returns NaN if 'TotalClaims' is missing or all groups are empty.
    """
    try:
        if 'TotalClaims' not in df.columns:
            logging.warning("Missing 'TotalClaims' column.")
            return float('nan')

        # Calculate claim presence without modifying df
        has_claim = (df['TotalClaims'] > 0).astype(int)
        
        if group_by:
            if isinstance(group_by, str):
                group_by = [group_by]
            frequency = df.groupby(group_by)['TotalClaims'].apply(lambda x: (x > 0).mean())
            if frequency.empty:
                logging.warning(f"No data after grouping by {group_by}")
                return float('nan')
            logging.info(f"Claim frequency by {group_by}: min={frequency.min():.4f}, max={frequency.max():.4f}, count={len(frequency)}")
            return frequency if as_series else frequency.mean()
        
        frequency = has_claim.mean()
        logging.info(f"Overall Claim Frequency: {frequency:.4f} (based on {len(df)} policies)")
        return frequency

    except Exception as e:
        logging.error(f"Error calculating claim frequency: {e}")
        return float('nan')

def calculate_claim_severity(df, group_by=None, as_series=False):
    """Calculate claim severity: average claim amount for policies with claims.

    Args:
        df (pd.DataFrame): DataFrame containing 'TotalClaims' column.
        group_by (str or list of str, optional): Column(s) to group by (e.g., 'PostalCode').
        as_series (bool): If True, return Series with per-group values; if False, return mean.

    Returns:
        float or pd.Series: Overall or group-wise claim severity (NaN if no claims).
    
    Notes:
        Returns NaN if 'TotalClaims' is missing or no claims exist in any group.
    """
    try:
        if 'TotalClaims' not in df.columns:
            logging.warning("Missing 'TotalClaims' column.")
            return float('nan')

        df_claims = df[df['TotalClaims'] > 0]
        if df_claims.empty:
            logging.warning("No claim records found.")
            return float('nan')

        if group_by:
            if isinstance(group_by, str):
                group_by = [group_by]
            severity = df_claims.groupby(group_by)['TotalClaims'].mean()
            if severity.empty:
                logging.warning(f"No claim data after grouping by {group_by}")
                return float('nan')
            logging.info(f"Claim severity by {group_by}: min={severity.min():.2f}, max={severity.max():.2f}, count={len(severity)}")
            return severity if as_series else severity.mean()

        severity = df_claims['TotalClaims'].mean()
        logging.info(f"Overall Claim Severity: {severity:.2f} (based on {len(df_claims)} claims)")
        return severity

    except Exception as e:
        logging.error(f"Error calculating claim severity: {e}")
        return float('nan')

def calculate_margin(df, group_by=None, as_series=False):
    """Calculate average margin: TotalPremium - TotalClaims.

    Args:
        df (pd.DataFrame): DataFrame with 'TotalPremium' and 'TotalClaims' columns.
        group_by (str or list of str, optional): Column(s) to group by (e.g., 'PostalCode').
        as_series (bool): If True, return Series with per-group values; if False, return mean.

    Returns:
        float or pd.Series: Overall or group-wise average margin (NaN if invalid data).
    
    Notes:
        Returns NaN if 'TotalPremium' or 'TotalClaims' is missing or all groups are empty.
    """
    try:
        if 'TotalPremium' not in df.columns or 'TotalClaims' not in df.columns:
            logging.warning("Missing 'TotalPremium' or 'TotalClaims' column.")
            return float('nan')

        # Calculate margin without modifying df
        margin = df['TotalPremium'] - df['TotalClaims']
        
        if group_by:
            if isinstance(group_by, str):
                group_by = [group_by]
            margin_grouped = df.groupby(group_by).apply(lambda x: (x['TotalPremium'] - x['TotalClaims']).mean())
            if margin_grouped.empty:
                logging.warning(f"No data after grouping by {group_by}")
                return float('nan')
            logging.info(f"Margin by {group_by}: min={margin_grouped.min():.2f}, max={margin_grouped.max():.2f}, count={len(margin_grouped)}")
            return margin_grouped if as_series else margin_grouped.mean()

        margin = margin.mean()
        logging.info(f"Overall Margin: {margin:.2f} (based on {len(df)} policies)")
        return margin

    except Exception as e:
        logging.error(f"Error calculating margin: {e}")
        return float('nan')