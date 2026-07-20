import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

class ModelTrainer:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42)
        }

    def train_and_evaluate(self):
        best_model = None
        best_f1 = 0
        best_metrics = {}

        for name, model in self.models.items():
            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)
            
            precision = precision_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            roc_auc = roc_auc_score(self.y_test, model.predict_proba(self.X_test)[:, 1])

            print(f"\n--- Model: {name} ---")
            print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}, ROC-AUC: {roc_auc:.4f}")

            if f1 > best_f1:
                best_f1 = f1
                best_model = model
                best_metrics = {
                    "Model": name,
                    "Precision": precision,
                    "Recall": recall,
                    "F1-Score": f1,
                    "ROC-AUC": roc_auc
                }

        joblib.dump(best_model, 'models/compliance_model.pkl')
        print(f"\n💾 Best model ({best_metrics['Model']}) saved.")
        
        return best_model, best_metrics
