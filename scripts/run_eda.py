# scripts/run_eda.py
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.eda.descriptive_stats import DescriptiveStats
from src.eda.visualizations import Visualizer
from src.eda.correlation import CorrelationAnalysis

def main():
    # Load data

    # Load the .txt file with '|' as the separator
    df = pd.read_csv("data/raw/MachineLearningRating_v3.txt", sep="|")

    # Save it as a .csv file
    df.to_csv("data/raw/insurance_data.csv", index=False)

    df = pd.read_csv("data/raw/insurance_data.csv")

    # Descriptive Statistics
    stats = DescriptiveStats(df)
    print(stats.basic_summary())
    print(stats.calculate_loss_ratio().head())
    print(stats.group_loss_ratio(by='Province'))
    
    # Visualization
    viz = Visualizer(df)
    viz.plot_histogram('TotalClaims')
    viz.plot_boxplot('CustomValueEstimate')
    viz.plot_bar_chart('VehicleType')

    # Correlation
    corr = CorrelationAnalysis(df)
    print(corr.compute_correlation_matrix())
    corr.plot_correlation_heatmap()

if __name__ == "__main__":
    main()
