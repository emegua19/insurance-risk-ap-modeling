import matplotlib.pyplot as plt
import seaborn as sns
import os

class Visualizations:
    def __init__(self, df):
        self.df = df
        self.output_dir = "output/eda/"
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_histogram(self, column):
        """Plot histogram for numerical columns."""
        plt.figure(figsize=(8, 4))
        sns.histplot(self.df[column].dropna(), kde=True)
        plt.title(f"Distribution of {column}")
        plt.savefig(os.path.join(self.output_dir, f"histogram_{column}.png"))
        plt.close()

    def plot_bar_chart(self, column):
        """Plot bar chart for categorical columns."""
        plt.figure(figsize=(10, 5))
        sns.countplot(data=self.df, x=column, order=self.df[column].value_counts().index)
        plt.xticks(rotation=45)
        plt.title(f"Bar Chart of {column}")
        plt.savefig(os.path.join(self.output_dir, f"bar_chart_{column}.png"))
        plt.close()

    def plot_boxplot(self, column):
        """Plot boxplot to detect outliers in numerical data."""
        plt.figure(figsize=(8, 4))
        sns.boxplot(x=self.df[column])
        plt.title(f"Boxplot of {column}")
        plt.savefig(os.path.join(self.output_dir, f"boxplot_{column}.png"))
        plt.close()

    def create_insight_plots(self):
        """Generate 3 creative plots capturing key EDA insights."""
        # Plot 1: Average TotalPremium by Province
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Province', y='TotalPremium', data=self.df, estimator='mean')
        plt.xticks(rotation=45)
        plt.title('Average TotalPremium by Province')
        plt.savefig(os.path.join(self.output_dir, "avg_premium_by_province.png"))
        plt.close()

        # Plot 2: TotalClaims vs TotalPremium by VehicleType
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='TotalPremium', y='TotalClaims', hue='VehicleType', data=self.df)
        plt.title('TotalClaims vs TotalPremium by VehicleType')
        plt.savefig(os.path.join(self.output_dir, "claims_vs_premium_by_vehicle.png"))
        plt.close()

        # Plot 3: LossRatio Distribution by Gender
        self.df['LossRatio'] = self.df['TotalClaims'] / (self.df['TotalPremium'] + 1e-6)
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Gender', y='LossRatio', data=self.df)
        plt.title('LossRatio Distribution by Gender')
        plt.savefig(os.path.join(self.output_dir, "loss_ratio_by_gender.png"))
        plt.close()