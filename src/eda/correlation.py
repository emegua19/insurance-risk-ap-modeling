import seaborn as sns
import matplotlib.pyplot as plt
import os

class Correlation:
    def __init__(self, df):
        self.df = df
        self.output_dir = "output/eda/"
        os.makedirs(self.output_dir, exist_ok=True)

    def explore_correlations(self):
        """Explore correlations between numerical columns."""
        numeric_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
        correlation_matrix = self.df[numeric_cols].corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
        plt.title('Correlation Matrix')
        plt.savefig(os.path.join(self.output_dir, "correlation_matrix.png"))
        plt.close()
        return correlation_matrix

    def scatter_plots_by_geo(self, geo_column='Province'):
        """Scatter plots of TotalPremium and TotalClaims as a function of a geographical column."""
        if geo_column not in self.df.columns:
            print(f"Warning: {geo_column} not found. Using index as fallback.")
            x_values = range(len(self.df))
        else:
            x_values = self.df[geo_column]
        
        plt.figure(figsize=(10, 6))
        plt.scatter(x_values, self.df['TotalPremium'], c='blue', label='TotalPremium')
        plt.scatter(x_values, self.df['TotalClaims'], c='red', label='TotalClaims')
        plt.xlabel(geo_column if geo_column in self.df.columns else 'Index')
        plt.ylabel('Values')
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, f"scatter_{geo_column}_premium_claims.png"))
        plt.close()

    def trends_over_geography(self):
        """Compare trends over geography (e.g., premium by cover type)."""
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='CoverType', y='TotalPremium', data=self.df)
        plt.xticks(rotation=45)
        plt.title('TotalPremium by CoverType')
        plt.savefig(os.path.join(self.output_dir, "premium_by_covertype.png"))
        plt.close()