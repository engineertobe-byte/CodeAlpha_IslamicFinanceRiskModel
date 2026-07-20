import os
import pandas as pd
from src.preprocessing import DataPreprocessor
from src.training import ModelTrainer
from src.visualization import DataVisualizer

def main():
    print("🚀 Starting Islamic Finance Compliance Model (Task 1 - Custom)...")
    
    # 1. Load Data
    data_path = 'data/islamic_finance_data.csv'
    if not os.path.exists(data_path):
        print(f"❌ Error: File {data_path} not found.")
        return

    df = pd.read_csv(data_path)
    print("✅ Data loaded successfully.")

    # 2. Preprocessing
    preprocessor = DataPreprocessor(df)
    X_train, X_test, y_train, y_test = preprocessor.prepare_data()
    print("✅ Preprocessing completed.")

    # 3. Visualization
    visualizer = DataVisualizer(preprocessor.df_cleaned)
    visualizer.plot_feature_distribution()
    visualizer.plot_compliance_correlation()
    print("✅ Visualizations generated in 'plots/'.")

    # 4. Training and Evaluation
    trainer = ModelTrainer(X_train, X_test, y_train, y_test)
    best_model, metrics = trainer.train_and_evaluate()
    
    print("\n📊 Best Model Results:")
    for key, value in metrics.items():
        print(f"   {key}: {value:.4f}")
        
    print("\n🎉 Project completed! Model saved for ethical risk assessment.")

if __name__ == "__main__":
    main()
