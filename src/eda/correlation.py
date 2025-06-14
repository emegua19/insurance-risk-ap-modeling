import seaborn as sns
import matplotlib.pyplot as plt
import os

class Correlation:
    """
    A class to handle correlation and relationship visualizations 
    between numerical and geographical variables in a dataset.
    """

    def __init__(self, df):
        """
        Initialize the Correlation class with a DataFrame and 
        create output directory for saving plots.
        """
        self.df = df
        self.output_dir = "outputs/eda/plots"
        os.makedirs(self.output_dir, exist_ok=True)

    def explore_correlations(self):
        """
        Generate a correlation matrix heatmap for numerical columns.
        
        Returns
        -------
        pd.DataFrame:
            The correlation matrix for numeric columns.
        """
        # Select only numeric columns (int and float types)
        numeric_cols = self.df.select_dtypes(include=['int64', 'float64']).columns

        # Compute the correlation matrix
        correlation_matrix = self.df[numeric_cols].corr()

        # Plot the correlation heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
        plt.title('Correlation Matrix')

        # Save the plot to file
        plt.savefig(os.path.join(self.output_dir, "correlation_matrix.png"))
        plt.close()

        return correlation_matrix

    def scatter_plots_by_geo(self, geo_column='Province'):
        """
        Create scatter plots of TotalPremium and TotalClaims 
        across a geographical category (e.g., Province or Region).

        Parameters
        ----------
        geo_column : str
            The name of the geographical column to plot against.
        """
        # If the geo column doesn't exist, fall back to row index
        if geo_column not in self.df.columns:
            print(f"Warning: {geo_column} not found. Using index as fallback.")
            x_values = range(len(self.df))
        else:
            x_values = self.df[geo_column]

        # Plot both TotalPremium and TotalClaims as scatter plots
        plt.figure(figsize=(10, 6))
        plt.scatter(x_values, self.df['TotalPremium'], c='blue', label='TotalPremium')
        plt.scatter(x_values, self.df['TotalClaims'], c='red', label='TotalClaims')

        # Label axes and add legend
        plt.xlabel(geo_column if geo_column in self.df.columns else 'Index')
        plt.ylabel('Values')
        plt.legend()

        # Save the plot to file
        plt.savefig(os.path.join(self.output_dir, f"scatter_{geo_column}_premium_claims.png"))
        plt.close()

    def trends_over_geography(self):
        """
        Generate a boxplot showing TotalPremium distribution across CoverType.
        Useful for visualizing how premiums vary by product category.
        """
        plt.figure(figsize=(10, 6))

        # Create a boxplot comparing premium distribution by cover type
        sns.boxplot(x='CoverType', y='TotalPremium', data=self.df)

        # Rotate x-axis labels for readability
        plt.xticks(rotation=45)

        # Add title and save plot
        plt.title('TotalPremium by CoverType')
        plt.savefig(os.path.join(self.output_dir, "premium_by_covertype.png"))
        plt.close()
