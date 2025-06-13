import os
import shutil
import sys
import pytest
import pandas as pd  # Moved to global scope
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from src.data_processing.data_loader import DataLoader
from src.data_processing.data_cleaning import DataCleaner
from src.eda.descriptive_stats import DescriptiveStats
from src.eda.visualizations import Visualizations
from src.eda.correlation import Correlation

@pytest.fixture
def setup_test_data():
    """Fixture to set up temporary test data."""
    test_dir = "tests/test_data/"
    os.makedirs(test_dir, exist_ok=True)
    input_file = os.path.join(test_dir, "test_data.txt")
    raw_output_file = os.path.join(test_dir, "insurance_data.csv")
    processed_output_file = os.path.join(test_dir, "insurance_cleaned_data.csv")

    # Create sample dataset
    sample_data = {
        'TotalPremium': [1000, 1500, 2000, 1200, None],
        'TotalClaims': [200, 300, 400, 250, 150],
        'VehicleType': ['Sedan', 'SUV', 'Sedan', 'Truck', 'SUV'],
        'Province': ['A', 'B', 'A', 'C', 'B'],
        'CoverType': ['Comprehensive', 'ThirdParty', 'Comprehensive', 'ThirdParty', 'Comprehensive'],
        'Gender': ['M', 'F', 'M', 'F', 'M']
    }
    pd.DataFrame(sample_data).to_csv(input_file, sep='|', index=False)

    loader = DataLoader(test_dir)
    cleaner = DataCleaner(processed_output_file)
    df, _ = loader.load_and_convert("test_data.txt")
    cleaned_df = cleaner.clean_data(df)

    yield cleaned_df, loader, cleaner

    # Teardown: remove temporary files
    shutil.rmtree(test_dir, ignore_errors=True)

def test_descriptive_stats(setup_test_data):
    """Test DescriptiveStats methods."""
    cleaned_df, _, _ = setup_test_data
    stats = DescriptiveStats(cleaned_df)
    summary = stats.basic_summary()
    assert 'TotalPremium' in cleaned_df.columns  # Check column existence
    assert len(stats.check_missing_values()) == len(cleaned_df.columns)
    assert isinstance(stats.review_data_structure(), pd.Series)

def test_visualizations(setup_test_data):
    """Test Visualization methods."""
    cleaned_df, _, _ = setup_test_data
    viz = Visualizations(cleaned_df)
    viz.plot_histogram('TotalPremium')
    viz.plot_bar_chart('VehicleType')
    viz.plot_boxplot('TotalPremium')
    viz.create_insight_plots()
    assert os.path.exists("output/eda/histogram_TotalPremium.png")
    assert os.path.exists("output/eda/bar_chart_VehicleType.png")
    assert os.path.exists("output/eda/boxplot_TotalPremium.png")
    assert os.path.exists("output/eda/avg_premium_by_province.png")

def test_correlation(setup_test_data):
    """Test Correlation methods."""
    cleaned_df, _, _ = setup_test_data
    corr = Correlation(cleaned_df)
    matrix = corr.explore_correlations()
    assert isinstance(matrix, pd.DataFrame)
    corr.scatter_plots_by_geo('Province')
    corr.trends_over_geography()
    assert os.path.exists("output/eda/correlation_matrix.png")
    assert os.path.exists("output/eda/scatter_Province_premium_claims.png")
    assert os.path.exists("output/eda/premium_by_covertype.png")

if __name__ == '__main__':
    pytest.main([__file__])