#!/usr/bin/env python3
# scripts/run_hypothesis_tests.py

import argparse
import logging
from pathlib import Path

import pandas as pd
import yaml

from src.hypothesis_testing.metrics import (
    add_has_claim,
    loss_ratio,
    claim_frequency,
    claim_severity,
)
from src.hypothesis_testing.segmentation import ab_segment, ensure_balance
from src.hypothesis_testing.statistical_tests import (
    chi2_test_counts,
    two_proportion_z_test,
    welchs_t_test,
    mann_whitney_u,
)


TEST_MAP = {
    "chi2": chi2_test_counts,
    "ztest": two_proportion_z_test,
    "ttest": welchs_t_test,
    "mw_u": mann_whitney_u,
}


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def compute_kpi(df: pd.DataFrame, kpi: str, temp_col: str = None) -> pd.DataFrame:
    """
    Ensure the KPI column exists on df.
    For 'frequency', adds 'has_claim' then returns df with that column.
    For 'severity', filters to >0 and returns df.
    For 'loss_ratio', adds 'loss_ratio' column.
    """
    if kpi == "frequency":
        return add_has_claim(df)
    elif kpi == "severity":
        # no new column needed, severity is computed groupwise later
        return df
    elif kpi == "loss_ratio":
        df = df.copy()
        df["loss_ratio"] = df["TotalClaims"] / (df["TotalPremium"].replace({0: float("nan")}))
        return df
    else:
        raise ValueError(f"Unknown KPI: {kpi}")


def run_tests(cfg: dict, df: pd.DataFrame) -> pd.DataFrame:
    """
    Run each hypothesis test defined in cfg["tests"].
    Returns a summary DataFrame.
    """
    results = []

    for test_cfg in cfg["tests"]:
        name = test_cfg["name"]
        feature = test_cfg["feature"]
        a_val = test_cfg["group_a"]
        b_val = test_cfg["group_b"]
        kpi = test_cfg["kpi"]
        test_type = test_cfg["test"]

        # 1. Prepare data & KPI
        df_kpi = compute_kpi(df, kpi)
        df_a, df_b = ab_segment(df_kpi, feature, a_val, b_val)

        # 2. Optional balance check
        if "covariates" in test_cfg:
            balance = ensure_balance(df_a, df_b, test_cfg["covariates"])
            logging.info("Balance for %s:\n%s", name, balance)

        # 3. Select test function
        test_fn = TEST_MAP.get(test_type)
        if test_fn is None:
            raise ValueError(f"Test type {test_type} not supported")

        # 4. Run test
        if kpi == "frequency":
            stat, p = test_fn(df_a, df_b, "has_claim")
        else:
            stat, p = test_fn(df_a, df_b, kpi if kpi != "severity" else "TotalClaims")

        # 5. Decision
        decision = "Reject H0" if p < cfg.get("alpha", 0.05) else "Fail to reject H0"

        results.append({
            "name": name,
            "feature": feature,
            "group_A": a_val,
            "group_B": b_val,
            "kpi": kpi,
            "test": test_type,
            "statistic": round(float(stat), 4),
            "p_value": round(float(p), 4),
            "decision": decision,
        })

    return pd.DataFrame(results)


def write_markdown(df: pd.DataFrame, out_path: Path) -> None:
    """
    Write the summary DataFrame to a Markdown table.
    """
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Task 3 – Hypothesis Testing Summary\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n")
        f.write("**Decision rule:** Reject H₀ if p < 0.05.\n")


def main(args):
    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # Load config
    cfg = load_config(Path(args.config))

    # Load cleaned data
    df = pd.read_csv(cfg["data"]["cleaned_path"])

    # Run tests
    summary = run_tests(cfg, df)
    logging.info("Completed %d hypothesis tests", len(summary))

    # Write report
    out_md = Path(cfg["reports"]["summary_md"])
    out_md.parent.mkdir(parents=True, exist_ok=True)
    write_markdown(summary, out_md)
    logging.info("Hypothesis summary written to %s", out_md)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ACIS hypothesis tests")
    parser.add_argument(
        "--config",
        default="configs/hypothesis_config.yaml",
        help="Path to hypothesis testing YAML config",
    )
    main(parser.parse_args())
