import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_processing.data_loader import DataLoader
from src.data_processing.data_cleaning import DataCleaner
from src.eda.descriptive_stats import DescriptiveStats
from src.eda.visualizations import Visualizations
from src.eda.correlation import Correlation

def main():
    try:
        # Initialize data loader and cleaner
        loader = DataLoader()
        cleaner = DataCleaner(processed_output_file="data/processed/insurance_cleaned_data.csv")

        # Load and convert data
        df, raw_output_path = loader.load_and_convert("MachineLearningRating_v3.txt")
        print(f"Raw data converted and saved to {raw_output_path}")

        # Print available columns to debug
        print("Available columns:", df.columns.tolist())

        # Clean data
        cleaned_df = cleaner.clean_data(df)

        # Descriptive Statistics
        stats = DescriptiveStats(cleaned_df)
        print("Basic Summary:", stats.basic_summary())
        print("Data Types:", stats.review_data_structure())
        print("Missing Values:", stats.check_missing_values())

        # Visualization
        viz = Visualizations(cleaned_df)
        viz.plot_histogram('TotalPremium')
        viz.plot_histogram('TotalClaims')
        viz.plot_bar_chart('VehicleType')
        viz.plot_boxplot('TotalPremium')
        viz.create_insight_plots()

        # Correlation
        corr = Correlation(cleaned_df)
        print("Correlation Matrix:", corr.explore_correlations())
        corr.scatter_plots_by_geo('Province')  # Changed to use Province as fallback
        corr.trends_over_geography()

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except KeyError as e:
        print(f"Error: Column not found - {e}")
    except PermissionError as e:
        print(f"Permission Error: {e} - Ensure write access to data/processed/")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()