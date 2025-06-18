import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class Visualizations:
    """Produce univariate and creative EDA plots."""

    def __init__(self, df: pd.DataFrame, output_dir: str):
        self.df = df.copy()
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_histogram(self, column: str):
        plt.figure(figsize=(8, 4))
        sns.histplot(self.df[column].dropna(), kde=True)
        plt.title(f"Distribution of {column}")
        plt.savefig(os.path.join(self.output_dir, f"hist_{column}.png"))
        plt.close()

    def plot_bar_chart(self, column: str):
        plt.figure(figsize=(10, 5))
        order = self.df[column].value_counts().index
        sns.countplot(data=self.df, x=column, order=order)
        plt.xticks(rotation=45)
        plt.title(f"{column} Counts")
        plt.savefig(os.path.join(self.output_dir, f"bar_{column}.png"))
        plt.close()

    def plot_boxplot(self, column: str):
        plt.figure(figsize=(8, 4))
        sns.boxplot(x=self.df[column].dropna())
        plt.title(f"Boxplot of {column}")
        plt.savefig(os.path.join(self.output_dir, f"box_{column}.png"))
        plt.close()

    def create_insight_plots(self, viz_cfg: dict):
        # Add LossRatio
        self.df["LossRatio"] = self.df["TotalClaims"] / self.df["TotalPremium"]
        for spec in viz_cfg.get("insight_plots", []):
            kind = spec["kind"]
            name = spec["name"]
            plt.figure(figsize=(10, 6))

            if kind == "bar":
                data = self.df.groupby(spec["x"])[spec["y"]].agg(spec["agg"]).reset_index()
                sns.barplot(x=spec["x"], y=spec["y"], data=data)
            elif kind == "scatter":
                sns.scatterplot(x=spec["x"], y=spec["y"], hue=spec.get("hue"), data=self.df)
            elif kind == "box":
                sns.boxplot(x=spec["x"], y=spec["y"], data=self.df)

            plt.title(name.replace("_", " ").title())
            plt.xticks(rotation=45)
            plt.savefig(os.path.join(self.output_dir, f"{name}.png"))
            plt.close()
