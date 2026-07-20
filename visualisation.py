import matplotlib.pyplot as plt
import seaborn as sns
import os

class DataVisualizer:
    def __init__(self, df):
        self.df = df
        os.makedirs('plots', exist_ok=True)

    def plot_feature_distribution(self):
        plt.figure(figsize=(10, 6))
        numeric_cols = self.df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            self.df[numeric_cols].hist(bins=15, figsize=(10, 6), color='#00d4ff')
            plt.suptitle("Feature Distribution - Islamic Finance Risk")
            plt.savefig('plots/feature_dist.png')
        plt.close()

    def plot_compliance_correlation(self):
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.df.corr(), annot=True, cmap='viridis', fmt=".2f")
        plt.title("Correlation Matrix - Compliance Factors")
        plt.savefig('plots/correlation.png')
        plt.close()
