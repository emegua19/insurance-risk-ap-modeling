# src/eda/descriptive_stats.py

import pandas as pd

class DescriptiveStats:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def basic_summary(self):
        return self.df.describe(include='all')

    def calculate_loss_ratio(self):
        if 'TotalClaims' in self.df.columns and 'TotalPremium' in self.df.columns:
            self.df['LossRatio'] = self.df['TotalClaims'] / (self.df['TotalPremium'] + 1e-6)
        return self.df[['TotalPremium', 'TotalClaims', 'LossRatio']]

    def group_loss_ratio(self, by: str):
        return self.df.groupby(by)[['TotalPremium', 'TotalClaims']].sum().assign(
            LossRatio=lambda x: x['TotalClaims'] / (x['TotalPremium'] + 1e-6)
        )
