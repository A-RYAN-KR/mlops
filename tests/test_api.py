from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# Apply fixtures to mock MLflow loading and Feast FeatureStore before import/lifespan
@pytest.fixture(autouse=True)
def mock_mlflow_and_feast():
    with (
        patch("mlflow.sklearn.load_model") as mock_load_model,
        patch("feast.FeatureStore") as mock_feature_store,
    ):
        # Mock MLflow model
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0, 1])
        mock_model.predict_proba.return_value = np.array([[0.9, 0.1], [0.2, 0.8]])
        mock_load_model.return_value = mock_model

        # Mock Feast response with matching features
        mock_feast_response = MagicMock()
        mock_feast_response.to_dict.return_value = {
            "customerID": ["7590-VHVEG", "5575-GNVDE"],
            "gender": ["Female", "Male"],
            "SeniorCitizen": [0, 0],
            "Partner": ["Yes", "No"],
            "Dependents": ["No", "No"],
            "tenure": [1, 34],
            "PhoneService": ["No", "Yes"],
            "MultipleLines": ["No phone service", "No"],
            "InternetService": ["DSL", "DSL"],
            "OnlineSecurity": ["No", "Yes"],
            "OnlineBackup": ["Yes", "No"],
            "DeviceProtection": ["No", "Yes"],
            "TechSupport": ["No", "No"],
            "StreamingTV": ["No", "No"],
            "StreamingMovies": ["No", "No"],
            "Contract": ["Month-to-month", "One year"],
            "PaperlessBilling": ["Yes", "No"],
            "PaymentMethod": ["Electronic check", "Mailed check"],
            "MonthlyCharges": [29.85, 56.95],
            "TotalCharges": [29.85, 1889.5],
        }
        mock_store = MagicMock()
        mock_store.get_online_features.return_value = mock_feast_response
        mock_feature_store.return_value = mock_store

        yield mock_model, mock_store


# Import client after patching starts
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "model_loaded": True,
        "feast_initialized": True,
    }


def test_predict_endpoint_success(client):
    payload = {"customer_ids": ["7590-VHVEG", "5575-GNVDE"]}
    # Mock log_predictions to avoid writing to local csv files during tests
    with patch("app.main.log_predictions") as mock_log:
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        predictions = response.json()["predictions"]

        assert len(predictions) == 2
        # Check mock mapping predictions
        assert predictions[0]["customer_id"] == "7590-VHVEG"
        assert predictions[0]["prediction"] == "No"  # class 0 mapped
        assert predictions[0]["churn_probability"] == 0.1

        assert predictions[1]["customer_id"] == "5575-GNVDE"
        assert predictions[1]["prediction"] == "Yes"  # class 1 mapped
        assert predictions[1]["churn_probability"] == 0.8

        mock_log.assert_called_once()


def test_predict_endpoint_empty_payload(client):
    payload = {"customer_ids": []}
    response = client.post("/predict", json=payload)
    assert response.status_code == 400
    assert "cannot be empty" in response.json()["detail"]


def test_drift_endpoint_insufficient_data(client):
    with patch("app.main.calculate_drift") as mock_calc:
        mock_calc.return_value = {
            "status": "insufficient_data",
            "count": 5,
            "message": "Not enough logged predictions",
        }
        response = client.get("/drift")
        assert response.status_code == 200
        assert response.json()["status"] == "insufficient_data"
        assert response.json()["count"] == 5
