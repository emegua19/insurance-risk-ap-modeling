# tests/test_run.py
"""
End‑to‑end test for scripts/run_eda.py using a temporary dataset
and temporary output folders.
"""

from pathlib import Path
import sys
import yaml
import pandas as pd

import pytest

# ------------------------------------------------------------------ #
# make project src importable                                        #
# ------------------------------------------------------------------ #
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from scripts import run_eda  # noqa: E402


@pytest.fixture
def mini_project(tmp_path: Path):
    """
    Set up a minimal raw dataset + config YAML in a temporary directory,
    return the path to the YAML config.
    """
    # 1. Directory scaffold
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    processed_dir = tmp_path / "data" / "processed"
    processed_dir.mkdir(parents=True)

    # 2. Create tiny raw .txt (pipe‑delimited)
    raw_txt = raw_dir / "mini.txt"
    pd.DataFrame(
        {
            "TotalPremium": [1000, 1500, 2000, 1200],
            "TotalClaims": [200, 300, 400, 250],
            "VehicleType": ["Sedan", "SUV", "Sedan", "Truck"],
            "Province": ["A", "B", "A", "C"],
            "CoverType": ["Comp", "Third", "Comp", "Third"],
            "Gender": ["M", "F", "M", "F"],
        }
    ).to_csv(raw_txt, sep="|", index=False)

    # 3. Build YAML config
    cfg = {
        "general": {
            "base_output_dir": str(tmp_path / "outputs" / "eda"),
            "plots_output_dir": str(tmp_path / "outputs" / "eda" / "plots"),
        },
        "data_loader": {
            "input_dir": str(raw_dir),
            "filename": "mini.txt",
            "delimiter": "|",
            "file_type": "txt",
        },
        "data_cleaner": {
            "output_path": str(processed_dir / "clean.csv"),
        },
        "visualisations": {
            "histograms": ["TotalPremium"],
            "bar_charts": ["VehicleType"],
            "box_plots": [],
            "geo_column": "Province",
            "insight_plots": [],
        },
    }
    cfg_path = tmp_path / "test_cfg.yaml"
    with cfg_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)

    return cfg_path, tmp_path


def test_run_eda_end_to_end(mini_project):
    """
    Run the entire pipeline on the mini dataset and assert artefacts exist.
    """
    cfg_path, tmp_path = mini_project

    # Execute the pipeline
    run_eda.main(str(cfg_path))

    # Paths to check
    base_out = tmp_path / "outputs" / "eda"
    plots_out = base_out / "plots"
    stats_out = base_out / "stats"

    # Assertions
    assert (stats_out / "basic_summary.csv").exists()
    assert any(plots_out.glob("histogram_TotalPremium*.png"))
    assert (stats_out / "correlation_matrix.csv").exists()
    assert (base_out / "eda_pipeline.log").exists()
