import io
import os
import sys
import time

import pandas as pd
from fastapi import FastAPI, HTTPException

from app.drift_monitor import (
    calculate_drift,
    calculate_evidently_drift,
    log_predictions,
)
from app.schema import ChurnPrediction, ChurnRequest, ChurnResponse

# Prevent Unicode errors on Windows terminal stdout
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

# Configure environment variables so MLflow can talk to Postgres/MinIO locally and inside Docker Compose
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin")
os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv(
    "MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000"
)
os.environ["AWS_DEFAULT_REGION"] = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
os.environ["AWS_EC2_METADATA_DISABLED"] = "true"

import mlflow
import mlflow.sklearn
from feast import FeatureStore

app = FastAPI(
    title="Telco Customer Churn Inference Service",
    description="FastAPI service integrated with Feast online store and MLflow model registry.",
    version="1.0.0",
)

# Global variables for model and Feast store
model = None
store = None


@app.on_event("startup")
def startup_event():
    global model, store

    # 1. Initialize Feast Feature Store
    print("Initializing Feast Feature Store connection...")
    try:
        # Repo path is relative to the app/ directory
        store = FeatureStore(repo_path="feature_store")
        print("Feast Feature Store initialized.")
    except Exception as e:
        print(f"Error initializing Feast Feature Store: {e!s}")
        # Don't fail the startup immediately, but log it

    # 2. Connect to MLflow and load the registered model
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)

    # We will try loading "models:/TelcoChurnModel/latest" and fall back to version 1 or 2 if needed
    model_uri = "models:/TelcoChurnModel/latest"

    max_retries = 12
    retry_delay = 5
    for i in range(max_retries):
        try:
            print(
                f"Connecting to MLflow tracking server at {tracking_uri} (Attempt {i + 1}/{max_retries})..."
            )
            # Load the scikit-learn model natively to preserve predict_proba function
            model = mlflow.sklearn.load_model(model_uri)
            print("Model loaded successfully from MLflow model registry!")
            break
        except Exception as e:
            print(f"Failed to load model: {e!s}")
            if i < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                # Attempt to fall back to a specific version (e.g. version 1)
                try:
                    fallback_uri = "models:/TelcoChurnModel/1"
                    print(
                        f"Attempting fallback to specific version URI: {fallback_uri}..."
                    )
                    model = mlflow.sklearn.load_model(fallback_uri)
                    print("Model loaded successfully using fallback version!")
                    break
                except Exception as e_fallback:
                    print(f"Fallback also failed: {e_fallback!s}")
                    raise RuntimeError(
                        "Fatal: Could not load the model from MLflow registry after all retries."
                    )


@app.post("/predict", response_model=ChurnResponse)
def predict(request: ChurnRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    if store is None:
        raise HTTPException(status_code=503, detail="Feast store is not initialized.")

    if not request.customer_ids:
        raise HTTPException(
            status_code=400, detail="customer_ids list cannot be empty."
        )

    # 1. Fetch features from Feast online store
    feature_names = [
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "tenure",
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
        "MonthlyCharges",
        "TotalCharges",
    ]
    feature_refs = [f"churn_features:{feat}" for feat in feature_names]

    try:
        entity_rows = [{"customerID": cid} for cid in request.customer_ids]
        print(f"Fetching online features for {len(entity_rows)} customer(s)...")
        feast_response = store.get_online_features(
            features=feature_refs, entity_rows=entity_rows
        )

        # 2. Convert response to Pandas DataFrame
        features_dict = feast_response.to_dict()
        df_features = pd.DataFrame(features_dict)

        # Ensure columns are ordered exactly as defined during training
        X = df_features[feature_names]
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve or parse online features from Feast: {e!s}",
        )

    # 3. Generate predictions
    try:
        probabilities = model.predict_proba(X)[:, 1]
        predictions_labels = model.predict(X)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Inference execution failed: {e!s}"
        )

    # 4. Format output and log results locally for drift monitoring
    prediction_list = []
    log_payload = []
    for cid, prob, label in zip(
        request.customer_ids, probabilities, predictions_labels
    ):
        pred_label = "Yes" if label == 1 else "No"
        prediction_list.append(
            ChurnPrediction(
                customer_id=cid, churn_probability=float(prob), prediction=pred_label
            )
        )
        log_payload.append(
            {
                "customer_id": cid,
                "churn_probability": float(prob),
                "prediction": pred_label,
            }
        )

    # Log predictions asynchronously or synchronously to CSV file
    try:
        log_predictions(log_payload)
    except Exception as e:
        print(f"Warning: Failed to write predictions log: {e!s}")

    return ChurnResponse(predictions=prediction_list)


@app.get("/drift")
def check_drift():
    """
    Calculates prediction drift using Kolmogorov-Smirnov test against baseline.
    """
    result = calculate_drift()
    if result["status"] == "error":
        raise HTTPException(status_code=500, detail=result["message"])

    # Update Prometheus metrics using Evidently calculation if data is sufficient
    if result["status"] == "success":
        evidently_result = calculate_evidently_drift()
        if evidently_result:
            EVIDENTLY_DRIFT_P_VALUE.set(evidently_result["p_value"])
            EVIDENTLY_DRIFT_SCORE.set(evidently_result["drift_score"])
            EVIDENTLY_DRIFT_DETECTED.set(
                1.0 if evidently_result["drift_detected"] else 0.0
            )
            result["evidently_drift"] = evidently_result
    return result


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "feast_initialized": store is not None,
    }


# Initialize Prometheus telemetry and custom drift metrics last to prevent registry clear issues
from prometheus_client import Gauge
from prometheus_fastapi_instrumentator import Instrumentator

Instrumentator().instrument(app).expose(app)

EVIDENTLY_DRIFT_P_VALUE = Gauge(
    "evidently_prediction_drift_p_value",
    "P-value of KS test for prediction probability drift calculated by Evidently AI",
)
EVIDENTLY_DRIFT_SCORE = Gauge(
    "evidently_prediction_drift_score", "Drift score calculated by Evidently AI"
)
EVIDENTLY_DRIFT_DETECTED = Gauge(
    "evidently_prediction_drift_detected",
    "Binary flag indicating if prediction drift is detected by Evidently AI (1 = yes, 0 = no)",
)
