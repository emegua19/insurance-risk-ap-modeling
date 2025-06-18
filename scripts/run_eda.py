#!scripts/run_eda.py
"""
scripts/run_eda.py
Run the ACIS EDA pipeline with rich console output.

Usage:
    python scripts/run_eda.py --config configs/eda_config.yaml
"""
from __future__ import annotations
import argparse
import logging
from pathlib import Path
import yaml
import os
import sys
import pandas as pd

# ------------------------------------------------------------------ #
# Import project modules (add src to path)
# ------------------------------------------------------------------ #
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
from src.utils.data_loader import DataLoader
from src.data_processing.data_cleaning import DataCleaner
from src.eda.descriptive_stats import DescriptiveStats
from src.eda.visualizations import Visualizations
from src.eda.correlation import Correlation

# ------------------------------------------------------------------ #


# ------------------------- Helper functions ------------------------ #
def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_logger(log_path: Path) -> logging.Logger:
    fmt = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path, mode="w"),
        ],
    )
    return logging.getLogger(__name__)
# ------------------------------------------------------------------ #


def main(cfg_path: str) -> None:
    # --------------------- Load configuration ---------------------- #
    cfg = load_yaml(cfg_path)
    base_dir = Path(cfg["general"]["base_output_dir"])
    plot_dir = Path(cfg["general"]["plots_output_dir"])
    base_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    log = setup_logger(base_dir / "eda_pipeline.log")
    log.info("=== Starting EDA pipeline ===")

    # ------------------------- 1. LOAD DATA ------------------------ #
    ld_cfg = cfg["data_loader"]
    loader = DataLoader(ld_cfg["input_dir"], ld_cfg["delimiter"])
    df_raw = (
        loader.load_txt(ld_cfg["filename"])
        if ld_cfg["file_type"] == "txt"
        else loader.load_csv(ld_cfg["filename"])
    )
    log.info("Raw data loaded: %s", df_raw.shape)

    print("\n=== RAW DATA SAMPLE ===")
    print(df_raw.head())

    # ------------------------- 2. CLEAN DATA ----------------------- #
    cl_cfg = cfg["data_cleaner"]
    cleaner = DataCleaner(cl_cfg["output_path"])
    df_clean = cleaner.clean_data(df_raw)
    cleaner.save_cleaned_data()
    log.info("Cleaned data saved -> %s", cl_cfg["output_path"])

    # ------------------- 3. DESCRIPTIVE STATS ---------------------- #
    stats = DescriptiveStats(df_clean)
    stats_dir = base_dir / "stats"
    stats_dir.mkdir(exist_ok=True)

    # Save CSV summaries
    stats.basic_summary().to_csv(stats_dir / "basic_summary.csv")
    stats.review_data_structure().to_csv(stats_dir / "dtypes.csv")
    stats.missing_summary().to_csv(stats_dir / "missing_summary.csv")
    log.info("Descriptive statistics exported")

    # --- Print key stats to console
    numeric_summary = stats.basic_summary().loc[["mean", "std", "min", "50%", "max"]]
    print("\n=== BASIC SUMMARY (numeric) ===")
    print(numeric_summary)

    miss_top10 = (
        stats.missing_summary()
        .sort_values("missing_count", ascending=False)
        .head(10)
    )
    print("\n=== TOP‑10 MISSING VALUES ===")
    print(miss_top10)

    # -------------------- 4. VISUALISATIONS ----------------------- #
    viz_cfg = cfg["visualisations"]
    viz = Visualizations(df_clean, str(plot_dir))

    # Univariate
    for col in viz_cfg.get("histograms", []):
        viz.plot_histogram(col)
    for col in viz_cfg.get("bar_charts", []):
        viz.plot_bar_chart(col)
    for col in viz_cfg.get("box_plots", []):
        viz.plot_boxplot(col)

    # Insight / creative plots
    viz.create_insight_plots(viz_cfg)
    log.info("Visualisations generated -> %s", plot_dir)

    # -------------------- 5. CORRELATION -------------------------- #
    corr = Correlation(df_clean, str(plot_dir))
    corr_matrix = corr.explore_correlations()
    corr_matrix.to_csv(stats_dir / "correlation_matrix.csv")

    # Print first 5×5 slice
    print("\n=== CORRELATION (first 5×5 block) ===")
    print(corr_matrix.iloc[:5, :5])

    corr.scatter_plots_by_geo(viz_cfg.get("geo_column", "Province"))
    log.info("Correlation analysis completed")

    log.info("=== EDA pipeline finished ===")
    print(f"\nPlots saved to: {plot_dir}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ACIS EDA pipeline with console output")
    parser.add_argument("--config", default="configs/eda_config.yaml", help="Path to YAML config")
    args = parser.parse_args()
    main(args.config)
