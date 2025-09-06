import argparse
import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

# Import our validation
from src.data_validation import get_iris_data, validate_data

def train_model():
    # Start an MLflow run
    with mlflow.start_run():
        # 1. Get and Validate Data
        df = get_iris_data()
        validate_data(df)

        X = df.drop('target', axis=1)
        y = df['target']

        # 2. Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 3. Train Model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        # 4. Predict & Evaluate
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Model accuracy: {acc}")

        # 5. Log everything with MLflow
        mlflow.log_param("n_estimators", 10)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "model") # Logs the model as an artifact

        # 6. Save the model locally for now
        joblib.dump(model, "models/iris_rf_model.joblib")
        print("Model saved!")

if __name__ == "__main__":
    train_model()