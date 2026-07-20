# 🕌 Islamic Finance Compliance Model | CodeAlpha Internship Task 1

## 📌 Project Overview
This project predicts financial compliance for Islamic contracts using Machine Learning. It assesses client reliability for Sharia-compliant transactions like *Murabaha* and *Ijara* via Logistic Regression and Random Forest. Optimized for Precision, Recall, and F1-Score, it ensures ethical risk assessment and reduces *Gharar* (uncertainty) in financial agreements.

## 🛠️ Technologies Used
*   **Language:** Python
*   **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Joblib
*   **Algorithms:** Logistic Regression, Random Forest Classifier, Gradient Boosting Classifier

## 📂 Project Structure
```text
CodeAlpha_IslamicFinanceRiskModel/
│
├── data/                  # Dataset storage (islamic_finance_data.csv)
├── models/                # Saved trained models (.pkl)
├── plots/                 # Generated EDA visualizations
├── src/                   # Source code modules
│   ├── preprocessing.py   # Data cleaning and scaling
│   ├── training.py        # Model training and evaluation
│   └── visualization.py   # Data plotting functions
│
├── main.py                # Main execution script
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
```

## 🚀 How to Run
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/YourUsername/CodeAlpha_IslamicFinanceRiskModel.git
    cd CodeAlpha_IslamicFinanceRiskModel
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Download Dataset:**
    **Note:** The code requires a dataset to function. Please download a standard credit/financial dataset (e.g., German Credit Data) from Kaggle or UCI, rename it to `islamic_finance_data.csv`, and place it inside the `data/` folder. Ensure the target column is named `compliance_status`. Without this file, the application will not start.
4.  **Run the application:**
    ```bash
    python main.py
    ```

## 📊 Model Evaluation Metrics
The project evaluates performance using the following key metrics:
*   **Precision:** Accuracy of positive predictions.
*   **Recall:** Ability to find all compliant instances.
*   **F1-Score:** Harmonic mean of Precision and Recall.
*   **ROC-AUC:** Measure of separability between compliant and risky profiles.

## 📝 License
This project is part of the CodeAlpha Machine Learning Internship program.
