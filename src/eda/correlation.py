# src/eda/correlation.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class CorrelationAnalysis:
    def __init__(self, df):
        self.df = df

    def compute_correlation_matrix(self):
        numeric_df = self.df.select_dtypes(include='number')
        return numeric_df.corr()

    def plot_correlation_heatmap(self):
        corr = self.compute_correlation_matrix()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
        plt.title("Correlation Matrix")
        plt.tight_layout()
        plt.show()
