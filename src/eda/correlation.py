import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class Correlation:
    """Compute and plot correlation matrix and geo‑based scatterplots."""

    def __init__(self, df: pd.DataFrame, output_dir: str):
        self.df = df
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def explore_correlations(self) -> pd.DataFrame:
        num_cols = self.df.select_dtypes(include=["int64", "float64"]).columns
        corr = self.df[num_cols].corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap="coolwarm")
        plt.title("Correlation Matrix")
        plt.savefig(os.path.join(self.output_dir, "correlation_matrix.png"))
        plt.close()
        return corr

    def scatter_plots_by_geo(self, geo_column: str):
        if geo_column not in self.df:
            return
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=self.df[geo_column], y=self.df["TotalPremium"], label="Premium")
        sns.scatterplot(x=self.df[geo_column], y=self.df["TotalClaims"], label="Claims")
        plt.xticks(rotation=45)
        plt.legend()
        plt.title(f"Premium & Claims by {geo_column}")
        plt.savefig(os.path.join(self.output_dir, f"scatter_{geo_column}.png"))
        plt.close()
