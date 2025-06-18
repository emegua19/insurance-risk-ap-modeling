import os
import shutil
import pytest
import pandas as pd
from pathlib import Path

from src.utils.data_loader import DataLoader
from src.data_processing.data_cleaning import DataCleaner
from src.eda.descriptive_stats import DescriptiveStats
from src.eda.visualizations import Visualizations
from src.eda.correlation import Correlation

TEST_DIR   = "tests/test_data"
PLOTS_DIR  = "outputs/test_plots"
CLEANED_CSV = os.path.join(TEST_DIR, "cleaned_data.csv")


# ------------------------------------------------------------------ #
# Fixture: build a tiny dataset, load → clean → yield clean DataFrame
# ------------------------------------------------------------------ #
@pytest.fixture(scope="module")
def sample_df():
    os.makedirs(TEST_DIR, exist_ok=True)
    txt_path = os.path.join(TEST_DIR, "sample_data.txt")

    df = pd.DataFrame({
        "UnderwrittenCoverID": [1001, 1002, 1001, 1003, 1004],
        "PolicyID":            [10,   20,   10,   30,   40],
        "TransactionMonth":    ["2023-01", "2023-02", "2023-01", "2023-03", "2023-01"],
        "TotalPremium":        [1200, 1500, None, 800, 2000],
        "TotalClaims":         [200, None, 400, 100, 600],
        "Gender":              ["Male", "Female", "Male", "Female", None],
        "Province":            ["A", "B", "A", "C", "B"],
        "VehicleType":         ["Sedan", "SUV", "Truck", "SUV", "Sedan"]
    })
    df.to_csv(txt_path, sep="|", index=False)

    loader = DataLoader(TEST_DIR, delimiter="|")
    df_loaded = loader.load_txt("sample_data.txt")
    cleaner = DataCleaner(output_path=CLEANED_CSV)
    df_clean = cleaner.clean_data(df_loaded)
    cleaner.save_cleaned_data()

    yield df_clean

    # Tidy up
    shutil.rmtree(TEST_DIR,  ignore_errors=True)
    shutil.rmtree(PLOTS_DIR, ignore_errors=True)


# ------------------------------------------------------------------ #
# 1. Data‑cleaning assertions
# ------------------------------------------------------------------ #
def test_data_cleaning(sample_df):
    df = sample_df
    assert not df.isna().any().any()
    assert pd.api.types.is_datetime64_any_dtype(df["TransactionMonth"])


# ------------------------------------------------------------------ #
# 2. Descriptive‑stats assertions
# ------------------------------------------------------------------ #
def test_descriptive_stats(sample_df):
    stats = DescriptiveStats(sample_df)
    assert "TotalPremium" in stats.basic_summary().columns
    assert "UnderwrittenCoverID" in stats.review_data_structure().index
    assert "missing_count" in stats.missing_summary().columns


# ------------------------------------------------------------------ #
# 3. Visualization output files
# ------------------------------------------------------------------ #
def test_visualizations(sample_df):
    Path(PLOTS_DIR).mkdir(parents=True, exist_ok=True)

    viz_cfg = {
        "histograms": ["TotalPremium"],
        "bar_charts": ["Gender"],
        "box_plots": ["TotalClaims"],
        "insight_plots": [
            {
                "kind": "bar",
                "name": "avg_premium_by_province",
                "x": "Province",
                "y": "TotalPremium",
                "agg": "mean",
            }
        ],
    }

    viz = Visualizations(sample_df, output_dir=PLOTS_DIR)
    viz.plot_histogram("TotalPremium")
    viz.plot_bar_chart("Gender")
    viz.plot_boxplot("TotalClaims")
    viz.create_insight_plots(viz_cfg)

    expected = [
        "hist_TotalPremium.png",
        "bar_Gender.png",
        "box_TotalClaims.png",
        "avg_premium_by_province.png",
    ]
    for fname in expected:
        assert (Path(PLOTS_DIR) / fname).exists(), f"{fname} was not created"


# ------------------------------------------------------------------ #
# 4. Correlation matrix + plot
# ------------------------------------------------------------------ #
def test_correlation(sample_df):
    Path(PLOTS_DIR).mkdir(parents=True, exist_ok=True)

    corr = Correlation(sample_df, output_dir=PLOTS_DIR)
    matrix = corr.explore_correlations()
    assert isinstance(matrix, pd.DataFrame)
    assert (Path(PLOTS_DIR) / "correlation_matrix.png").exists()
