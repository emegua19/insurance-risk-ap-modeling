import yaml
import pandas as pd
import logging
import os
import sys
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.hypothesis_testing.metrics import (
    calculate_claim_frequency,
    calculate_claim_severity,
    calculate_margin,
    calculate_loss_ratio
)
from src.hypothesis_testing.segmentation import ab_segment, ensure_balance, flag_imbalances
from src.hypothesis_testing.statistical_tests import run_test

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', encoding='utf-8')

def load_config(config_path):
    """Load configuration from YAML file.

    Args:
        config_path (str): Path to the config file.

    Returns:
        dict: Configuration dictionary.

    Raises:
        FileNotFoundError: If config file is not found.
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        logging.error(f"Config file not found: {config_path}")
        raise

def write_markdown_summary(results, output_path):
    """Write hypothesis test results to a Markdown file.

    Args:
        results (list): List of test result dictionaries.
        output_path (str): Path to save the Markdown file.
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Hypothesis Testing Summary\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("| Test Name | KPI | Statistic | p-value | Decision |\n")
        f.write("|-----------|-----|-----------|---------|----------|\n")
        for result in results:
            decision = "Reject H₀" if result['p_value'] < result['alpha'] else "Fail to Reject H₀"
            f.write(f"| {result['name']} | {result['kpi']} | {result['statistic']:.4f} | {result['p_value']:.4f} | {decision} |\n")
    logging.info(f"Hypothesis summary written to {output_path}")

def run_tests(cfg):
    """Run hypothesis tests based on configuration.

    Args:
        cfg (dict): Configuration dictionary.

    Returns:
        list: List of test result dictionaries.
    """
    # Validate required columns with dtype specification for CapitalOutstanding
    required_columns = {'TotalClaims', 'TotalPremium'}
    df = pd.read_csv(cfg["data"]["cleaned_path"], 
                     dtype={'CapitalOutstanding': 'float'},
                     low_memory=False, 
                     encoding='utf-8')
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        logging.error(f"Missing required columns: {missing_columns}")
        return []

    results = []

    for test in cfg["tests"]:
        name = test["name"]
        kpi = test["kpi"]
        logging.info(f"Running test: {name} on KPI: {kpi}")

        # Segment the data
        try:
            df_a, df_b = ab_segment(df, test["feature"], test["group_a"], test["group_b"])
        except (KeyError, ValueError) as e:
            logging.warning(f"Skipping {name}: Segmentation failed - {str(e)}")
            continue

        # Balance check
        covariates = test.get("covariates", [])
        if covariates:
            balance_df = ensure_balance(df_a, df_b, covariates)
            if balance_df.empty or 'p_value' not in balance_df.columns:
                logging.warning(f"Skipping {name}: Balance check failed or no valid covariates found")
                continue
            imbalance_df = flag_imbalances(balance_df, threshold=0.05)
            if not imbalance_df.empty:
                logging.warning(f"Skipping {name}: Imbalance detected in covariates:\n{imbalance_df}")
                continue

        # Calculate KPI for each group
        if kpi == "frequency":
            kpi_a = calculate_claim_frequency(df_a)
            kpi_b = calculate_claim_frequency(df_b)
        elif kpi == "severity":
            kpi_a = calculate_claim_severity(df_a)
            kpi_b = calculate_claim_severity(df_b)
        elif kpi == "margin":
            kpi_a = calculate_margin(df_a)
            kpi_b = calculate_margin(df_b)
        elif kpi == "loss_ratio":
            kpi_a = calculate_loss_ratio(df_a)
            kpi_b = calculate_loss_ratio(df_b)
        else:
            logging.error(f"Unsupported KPI: {kpi}")
            continue

        logging.info(f"{name} - KPI (A): {kpi_a:.4f}, KPI (B): {kpi_b:.4f}")

        # Run statistical test
        stat, p_value = run_test(df_a, df_b, test["test"], kpi)
        results.append({
            "name": name,
            "kpi": kpi,
            "statistic": stat,
            "p_value": p_value,
            "alpha": cfg["alpha"]
        })

        logging.info(f"{name}: p = {p_value:.4f}, {'Reject H₀' if p_value < cfg['alpha'] else 'Fail to Reject H₀'}")

    return results

def main():
    """Main function to execute hypothesis testing."""
    config_path = "configs/hypothesis_config.yaml"
    try:
        cfg = load_config(config_path)
        logging.info(f"Starting hypothesis tests using: {config_path}")
        results = run_tests(cfg)
        write_markdown_summary(results, cfg["reports"]["summary_md"])
        logging.info(f"Completed {len(results)} hypothesis tests")
    except Exception as e:
        logging.error(f"Execution failed: {e}")

if __name__ == "__main__":
    main()