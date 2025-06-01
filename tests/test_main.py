# tests/test_main.py
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock

# Adjust the import path
from src.credit_risk_app.main import app, get_prediction_service
from src.credit_risk_app.services import PredictionService


@pytest.fixture(scope="module")
def client():
    """Fixture to create a TestClient for the FastAPI app."""
    return TestClient(app)


# --- Test Cases for FastAPI Endpoints ---


def test_ping_endpoint(client):
    """
    Test 5: Check the /ping endpoint.
    Explanation: A simple health check endpoint. Verifies the API is responsive
                 and returns the expected 'pong' message.
    """
    response = client.get("/ping")
    assert response.status_code == 200
    assert response.json() == {"message": "pong"}
    print("\nTest 5 Passed: /ping endpoint returned 200 and 'pong'.")


def test_healthz_endpoint_service_ready(client):
    """
    Test 6: Check the /healthz endpoint when the service is available.
    Explanation: This tests the application's health check. When the prediction
                 service is (mocked as) ready, it should report 'ok'.
    """
    app.state.prediction_service = MagicMock(spec=PredictionService)

    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "message": "Service is ready."}

    app.state.prediction_service = None  # Cleanup
    print("\nTest 6 Passed: /healthz reported 'ok' when service is ready.")


def test_healthz_endpoint_service_unavailable(client):
    """
    Test 7: Check the /healthz endpoint when the service is NOT available.
    Explanation: Ensures the health check correctly reports 'unavailable' if the
                 prediction service failed to load (simulated here).
    """
    app.state.prediction_service = None

    response = client.get("/healthz")
    assert response.status_code == 503
    assert response.json()["status"] == "unavailable"
    print("\nTest 7 Passed: /healthz reported 'unavailable' correctly.")


def test_predict_random_client_endpoint_success(client):
    """
    Test 8: Check the /predict endpoint for a successful prediction.
    Explanation: This tests the main prediction endpoint. It replaces the actual
                 PredictionService with a 'mock' one that returns predefined data.
                 This allows testing the endpoint's behavior (like correct response
                 format and status code) without needing a real model.
    """
    mock_service_instance = MagicMock(spec=PredictionService)
    expected_prediction_data = {
        "threshold": 0.5,
        "probability_neg": 0.8,
        "probability_pos": 0.2,
        "loan_id": 12345,
        "credit_amount": 50000.0,
    }
    mock_service_instance.get_prediction_for_random_client.return_value = (
        expected_prediction_data
    )

    app.dependency_overrides[get_prediction_service] = lambda: mock_service_instance

    response = client.get("/predict")

    assert response.status_code == 200
    json_response = response.json()

    # Compare against the Pydantic model's schema field names (aliases)
    assert json_response["loan_id"] == expected_prediction_data["loan_id"]
    assert json_response["threshold"] == expected_prediction_data["threshold"]
    assert (
        json_response["probabilityClass0"]
        == expected_prediction_data["probability_neg"]
    )
    assert (
        json_response["probabilityClass1"]
        == expected_prediction_data["probability_pos"]
    )
    assert json_response["credit_amount"] == expected_prediction_data["credit_amount"]

    app.dependency_overrides = {}  # Cleanup
    print(
        f"\nTest 8 Passed: /predict endpoint returned successful mock prediction. Response: {json_response}"
    )


def test_predict_random_client_endpoint_service_error(client):
    """
    Test 9: Check the /predict endpoint when the service raises an error.
    Explanation: Tests how the endpoint handles errors from the underlying service.
                 If the (mocked) service raises an exception, the API should catch
                 it and return an appropriate HTTP error code (e.g., 500).
    """
    mock_service_instance = MagicMock(spec=PredictionService)
    mock_service_instance.get_prediction_for_random_client.side_effect = Exception(
        "Mock service internal error"
    )

    app.dependency_overrides[get_prediction_service] = lambda: mock_service_instance

    response = client.get("/predict")

    assert response.status_code == 500
    assert "detail" in response.json()
    assert response.json()["detail"] == "Internal server error during prediction."

    app.dependency_overrides = {}  # Cleanup
    print("\nTest 9 Passed: /predict endpoint handled service error correctly.")
