import seaborn as sns
import matplotlib.pyplot as plt
import os

class Visualizer:
    def __init__(self, df):
        self.df = df
        self.output_dir = "output/eda/plots"
        os.makedirs(self.output_dir, exist_ok=True)  # Create directory if it doesn't exist

    def plot_histogram(self, column):
        plt.figure(figsize=(8, 4))
        sns.histplot(self.df[column].dropna(), kde=True)
        plt.title(f"Distribution of {column}")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"histogram_{column}.png"))
        plt.close()

    def plot_boxplot(self, column):
        plt.figure(figsize=(8, 4))
        sns.boxplot(x=self.df[column])
        plt.title(f"Boxplot of {column}")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"boxplot_{column}.png"))
        plt.close()

    def plot_bar_chart(self, column):
        plt.figure(figsize=(10, 5))
        sns.countplot(data=self.df, x=column, order=self.df[column].value_counts().index)
        plt.xticks(rotation=45)
        plt.title(f"Bar Chart of {column}")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"bar_chart_{column}.png"))
        plt.close()

    def generate_insight_plots(self):
        """Generate specific plots for Gender, VehicleType, and Province."""
        # Bar charts
        self.plot_bar_chart("Gender")
        self.plot_bar_chart("VehicleType")
        self.plot_bar_chart("Province")

        # Boxplots
        self.plot_boxplot("Gender")
        self.plot_boxplot("VehicleType")
        self.plot_boxplot("Province")