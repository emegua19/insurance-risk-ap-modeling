import pytest
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from src.eda.correlation import CorrelationAnalysis
from src.eda.descriptive_stats import DescriptiveStats
from src.eda.visualizations import Visualizer

# Fixture to create synthetic test data
@pytest.fixture
def sample_df():
    data = {
        'TotalPremium': [1000, 1200, 800, 1500],
        'TotalClaims': [500, 600, 400, 700],
        'Gender': ['Male', 'Female', 'Male', 'Female'],
        'VehicleType': ['Sedan', 'SUV', 'Truck', 'Sedan'],
        'Province': ['Gauteng', 'Western Cape', 'KwaZulu-Natal', 'Gauteng']
    }
    return pd.DataFrame(data)

# Tests for CorrelationAnalysis
def test_correlation_analysis_init(sample_df):
    corr_analyzer = CorrelationAnalysis(sample_df)
    assert corr_analyzer.df.equals(sample_df)

def test_compute_correlation_matrix(sample_df):
    corr_analyzer = CorrelationAnalysis(sample_df)
    corr_matrix = corr_analyzer.compute_correlation_matrix()
    assert corr_matrix.shape == (2, 2)  # Only numerical columns: TotalPremium, TotalClaims
    assert 'TotalPremium' in corr_matrix.index
    assert corr_matrix.loc['TotalPremium', 'TotalClaims'] >= 0  # Positive correlation expected

def test_plot_correlation_heatmap(sample_df):
    corr_analyzer = CorrelationAnalysis(sample_df)
    corr_analyzer.plot_correlation_heatmap()
    # Note: plt.show() makes direct testing difficult; use file save or mock for CI
    assert True  # Placeholder; consider saving to file and checking existence

# Tests for DescriptiveStats
def test_descriptive_stats_init(sample_df):
    stats = DescriptiveStats(sample_df)
    assert stats.df.equals(sample_df)

def test_basic_summary(sample_df):
    stats = DescriptiveStats(sample_df)
    summary = stats.basic_summary()
    assert not summary.empty
    assert 'TotalPremium' in summary.columns
    assert 'Gender' in summary.columns  # Check column, not index
    assert summary.loc['mean', 'TotalPremium'] == 1125.0  # (1000 + 1200 + 800 + 1500) / 4

def test_calculate_loss_ratio(sample_df):
    stats = DescriptiveStats(sample_df)
    result = stats.calculate_loss_ratio()
    assert 'LossRatio' in result.columns
    assert len(result) == len(sample_df)
    assert result['LossRatio'].iloc[0] == 500 / (1000 + 1e-6)  # Approx 0.5

def test_group_loss_ratio(sample_df):
    stats = DescriptiveStats(sample_df)
    result = stats.group_loss_ratio('Province')
    assert 'LossRatio' in result.columns
    assert len(result) == sample_df['Province'].nunique()
    assert result.loc['Gauteng', 'LossRatio'] == (500 + 700) / (1000 + 1500 + 1e-6)  # Approx 0.8

# Tests for Visualizer
def test_visualizer_init(sample_df):
    viz = Visualizer(sample_df)
    assert viz.df.equals(sample_df)
    assert os.path.exists(viz.output_dir)

def test_plot_histogram(sample_df):
    viz = Visualizer(sample_df)
    viz.plot_histogram('TotalPremium')
    assert os.path.exists(os.path.join(viz.output_dir, 'histogram_TotalPremium.png'))

def test_plot_boxplot(sample_df):
    viz = Visualizer(sample_df)
    viz.plot_boxplot('TotalPremium')
    assert os.path.exists(os.path.join(viz.output_dir, 'boxplot_TotalPremium.png'))

def test_plot_bar_chart(sample_df):
    viz = Visualizer(sample_df)
    viz.plot_bar_chart('Gender')
    assert os.path.exists(os.path.join(viz.output_dir, 'bar_chart_Gender.png'))

def test_generate_insight_plots(sample_df):
    viz = Visualizer(sample_df)
    viz.generate_insight_plots()
    expected_files = [
        'bar_chart_Gender.png', 'bar_chart_VehicleType.png', 'bar_chart_Province.png',
        'boxplot_Gender.png', 'boxplot_VehicleType.png', 'boxplot_Province.png'
    ]
    for file in expected_files:
        assert os.path.exists(os.path.join(viz.output_dir, file))