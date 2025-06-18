#!/usr/bin/env python3
"""
Run Task‑3 hypothesis tests from YAML config.
"""

import argparse
import yaml
from pathlib import Path
import logging
import pandas as pd
import os
import sys

# add src to path)
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))

from src.hypothesis_testing.metrics import add_metrics
from src.hypothesis_testing.segmentation import segment_groups
from src.hypothesis_testing.statistical_tests import run_test


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_logger():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")
    return logging.getLogger()


def main(cfg_path):
    cfg = load_yaml(cfg_path)
    log = setup_logger()

    # 1. Load cleaned data + add KPI columns
    df = pd.read_csv(cfg["data"]["cleaned_path"])
    df = add_metrics(df)

    results = []
    for spec in cfg["tests"]:
        log.info("Running %s", spec["name"])
        a_df, b_df = segment_groups(df, spec["feature"],
                                    spec["group_a"], spec["group_b"])
        if a_df.empty or b_df.empty:
            log.warning("Skipping %s: empty segment", spec["name"])
            continue

        stat, p = run_test(a_df, b_df, spec["kpi"], spec["test"])
        decision = "Reject H₀" if p < cfg["alpha"] else "Fail to reject H₀"

        results.append({
            "name":       spec["name"],
            "feature":    spec["feature"],
            "group_A":    spec["group_a"],
            "group_B":    spec["group_b"],
            "kpi":        spec["kpi"],
            "test":       spec["test"],
            "statistic":  round(stat, 4),
            "p_value":    round(p, 4),
            "decision":   decision
        })

    # 2. Write Markdown summary
    out_md = Path(cfg["reports"]["summary_md"])
    out_md.parent.mkdir(parents=True, exist_ok=True)

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# Task 3 – Hypothesis Testing Summary\n\n")
        if results:
            # Markdown table header
            header = "| " + " | ".join(results[0].keys()) + " |"
            line   = "|" + "|".join([":"+"-"*(len(k)+1) for k in results[0]]) + "|"
            f.write(header + "\n" + line + "\n")

            for res in results:
                row = "| " + " | ".join(str(v) for v in res.values()) + " |"
                f.write(row + "\n")

            f.write(f"\n**Decision rule:** Reject H₀ if p < {cfg['alpha']}.")
        else:
            f.write("_No valid tests were run (empty segments)._")

    log.info("Hypothesis summary written to %s", out_md)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/hypothesis_config.yaml")
    args = parser.parse_args()
    main(args.config)
