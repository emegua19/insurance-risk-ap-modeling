"""
scripts/run_eda.py
Run the ACIS EDA pipeline using YAML configuration + structured logging.

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
# ---------------- internal imports ---------------- #
from src.utils.data_loader import DataLoader          # updated loader
from src.data_processing.data_cleaning import DataCleaner
from src.eda.descriptive_stats import DescriptiveStats
from src.eda.visualizations import Visualizations
from src.eda.correlation import Correlation
# -------------------------------------------------- #

# make src importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# ------------------------------------------------------------------ #
# Helper functions                                                   #
# ------------------------------------------------------------------ #
def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_logger(log_path: Path, level: int = logging.INFO) -> None:
    """Configure console + file logging."""
    fmt = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(
        level=level,
        format=fmt,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path, mode="w"),
        ],
    )


# ------------------------------------------------------------------ #
# Main pipeline                                                      #
# ------------------------------------------------------------------ #
def main(cfg_path: str) -> None:
    cfg = load_yaml(cfg_path)
    base_output_dir = Path(cfg["general"]["base_output_dir"])
    plots_output_dir = Path(cfg["general"]["plots_output_dir"])
    base_output_dir.mkdir(parents=True, exist_ok=True)
    plots_output_dir.mkdir(parents=True, exist_ok=True)

    setup_logger(base_output_dir / "eda_pipeline.log")
    log = logging.getLogger(__name__)

    # ------------------------------------------------------------------ #
    # 1. Load raw data
    # ------------------------------------------------------------------ #
    ld_cfg = cfg["data_loader"]
    loader = DataLoader(ld_cfg["input_dir"], delimiter=ld_cfg["delimiter"])
    df = (
        loader.load_txt(ld_cfg["filename"], convert_to_csv=True)
        if ld_cfg["file_type"] == "txt"
        else loader.load_csv(ld_cfg["filename"])
    )
    log.info("Raw data loaded: %s rows × %s cols", *df.shape)
    print("\n=== RAW SAMPLE ===")
    print(df.head(3))

    # ------------------------------------------------------------------ #
    # 2. Clean data
    # ------------------------------------------------------------------ #
    cl_cfg = cfg["data_cleaner"]
    cleaner = DataCleaner(cl_cfg["output_path"])
    df_clean = cleaner.clean_data(df)
    cleaner.save_cleaned_data()
    log.info("Cleaned data saved -> %s", cl_cfg["output_path"])
    print("\n=== CLEANED SAMPLE ===")
    print(df_clean.head(3))

    # ------------------------------------------------------------------ #
    # 3. Descriptive statistics
    # ------------------------------------------------------------------ #
    stats = DescriptiveStats(df_clean)
    (base_output_dir / "stats").mkdir(exist_ok=True)
    stats.basic_summary().to_csv(
        base_output_dir / "stats" / "basic_summary.csv")
    
    stats.review_data_structure().to_csv(
        base_output_dir / "stats" / "dtypes.csv")
    
    stats.missing_summary().to_csv(
        base_output_dir / "stats" / "missing_summary.csv")
    
    log.info("Descriptive stats exported")

    # >>> PRINT key summaries to console
    print("\n=== SUMMARY STATS (numerical) ===")
    print(stats.basic_summary().loc[["mean", "std", "min", "max"]])

    print("\n=== MISSING VALUES (top 10) ===")
    print(
        stats.missing_summary()
        .sort_values("Missing", ascending=False)
        .head(10)
    )

    # ------------------------------------------------------------------ #
    # 4. Visualisations
    # ------------------------------------------------------------------ #
    viz_cfg = cfg["visualisations"]
    viz = Visualizations(df_clean, output_dir=plots_output_dir)

    # 4A. Histograms, bar charts, box plots
    for col in viz_cfg.get("histograms", []):
        viz.plot_histogram(col)
    for col in viz_cfg.get("bar_charts", []):
        viz.plot_bar_chart(col)
    for col in viz_cfg.get("box_plots", []):
        viz.plot_boxplot(col)

    # 4B. Insight / creative plots
    for spec in viz_cfg.get("insight_plots", []):
        viz.create_insight_plots()

    log.info("Visualisations generated -> %s", plots_output_dir)
    print("Plots saved to:", plots_output_dir)

    # ------------------------------------------------------------------ #
    # 5. Correlation & geographic trends
    # ------------------------------------------------------------------ #
    corr = Correlation(df_clean)
    corr_matrix = corr.explore_correlations()
    corr_matrix.to_csv(base_output_dir / "stats" / "correlation_matrix.csv")
    corr.scatter_plots_by_geo(viz_cfg.get("geo_column", "Province"))
    corr.trends_over_geography()

    # >>> PRINT correlation snippet
    print("\n=== CORRELATION (first 5 cols) ===")
    print(corr_matrix.iloc[:5, :5])

    log.info("=== EDA pipeline finished successfully ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ACIS EDA pipeline")
    parser.add_argument("--config", default="configs/eda_config.yaml",
                        help="Path to YAML configuration file")
    main(parser.parse_args().config)
