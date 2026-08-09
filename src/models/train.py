import io
import os
import sys

import numpy as np
import pandas as pd

# Prevent UnicodeEncodeError on Windows stdout when printing emojis (e.g. from mlflow)
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

# Configure S3/MinIO environment variables so the local MLflow python client can upload artifacts
os.environ["AWS_ACCESS_KEY_ID"] = "minioadmin"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minioadmin"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"
# Ensure boto3/s3fs doesn't fail on local HTTP connection
os.environ["AWS_DEFAULT_REGION"] = "us-east-1"
os.environ["AWS_EC2_METADATA_DISABLED"] = "true"

import mlflow
import mlflow.sklearn
from feast import FeatureStore
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def main():
    # 1. Initialize Feast Feature Store
    print("Connecting to Feast Feature Store...")
    # Repo path is relative to the workspace root
    store = FeatureStore(repo_path="feature_store")

    # 2. Read entities (customerID, event_timestamp, and target Churn) from Parquet
    parquet_path = os.path.join("data", "churn_features.parquet")
    print(f"Reading training entities from {parquet_path}...")
    df_entities = pd.read_parquet(
        parquet_path, columns=["customerID", "event_timestamp", "Churn"]
    )

    # Convert target Churn to binary integer
    df_entities["target"] = (df_entities["Churn"] == "Yes").astype(int)

    # 3. Retrieve historical features from Feast
    print("Retrieving historical features from Feast offline store...")
    feature_refs = [
        "churn_features:gender",
        "churn_features:SeniorCitizen",
        "churn_features:Partner",
        "churn_features:Dependents",
        "churn_features:tenure",
        "churn_features:PhoneService",
        "churn_features:MultipleLines",
        "churn_features:InternetService",
        "churn_features:OnlineSecurity",
        "churn_features:OnlineBackup",
        "churn_features:DeviceProtection",
        "churn_features:TechSupport",
        "churn_features:StreamingTV",
        "churn_features:StreamingMovies",
        "churn_features:Contract",
        "churn_features:PaperlessBilling",
        "churn_features:PaymentMethod",
        "churn_features:MonthlyCharges",
        "churn_features:TotalCharges",
    ]

    training_data = store.get_historical_features(
        entity_df=df_entities[["customerID", "event_timestamp"]], features=feature_refs
    ).to_df()

    # Join target label back to retrieved features on customerID
    training_data = training_data.merge(
        df_entities[["customerID", "target"]], on="customerID"
    )

    # 4. Separate features and target
    X = training_data.drop(columns=["customerID", "event_timestamp", "target"])
    y = training_data["target"]

    # 5. Define column transformer for pre-processing
    categorical_cols = [
        "gender",
        "Partner",
        "Dependents",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
    ]
    numeric_cols = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", drop="first"),
                categorical_cols,
            ),
        ]
    )

    # Define model training pipeline
    n_estimators = 100
    max_depth = 6
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=n_estimators, max_depth=max_depth, random_state=42
                ),
            ),
        ]
    )

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 6. Configure MLflow Experiment Tracking
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("Telco_Customer_Churn_Training")

    print("Starting MLflow training run...")
    with mlflow.start_run() as run:
        # Fit pipeline
        print("Training Random Forest Classifier pipeline...")
        pipeline.fit(X_train, y_train)

        # Save baseline prediction probabilities (class 1: Churn = Yes)
        print("Generating and saving baseline prediction probabilities...")
        baseline_probs = pipeline.predict_proba(X)[:, 1]
        os.makedirs("data", exist_ok=True)
        np.save(os.path.join("data", "baseline_probabilities.npy"), baseline_probs)

        # Predictions & evaluation
        y_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print(
            f"Evaluation Metrics: Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}"
        )

        # Log Hyperparameters
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("test_size", 0.2)

        # Log Metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)

        # Log Model pipeline to S3/MinIO via MLflow registry
        print("Logging trained model pipeline to MLflow artifact store...")
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            registered_model_name="TelcoChurnModel",
        )

        print(f"Run completed successfully! Run ID: {run.info.run_id}")


if __name__ == "__main__":
    main()
