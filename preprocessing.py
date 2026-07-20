import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class DataPreprocessor:
    def __init__(self, df):
        self.df = df
        self.df_cleaned = None

    def prepare_data(self):
        # Cleaning
        self.df_cleaned = self.df.dropna()
        
        # Target: 'compliance_status' (1 = Compliant/Reliable, 0 = Default/Risky)
        target_column = 'compliance_status'
        
        # Features: Income, Debt-to-Income ratio, Payment history, etc.
        features = [col for col in self.df_cleaned.columns if col != target_column]
        
        X = self.df_cleaned[features]
        y = self.df_cleaned[target_column]

        # Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        return X_train_scaled, X_test_scaled, y_train, y_test
