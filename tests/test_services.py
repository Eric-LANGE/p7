# tests/test_services.py
import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch

# Adjust the import path
from src.credit_risk_app.services import PredictionService


# Mock MLflow and its PyFuncModel to test the service logic
class MockPyFuncModel:
    def predict(self, data):
        # Simulate a prediction output: [probability_class_0, probability_class_1]
        # Ensure the output shape matches what PredictionService expects
        return pd.DataFrame([[0.6, 0.4]])


@pytest.fixture
def mock_pyfunc_model():
    return MockPyFuncModel()


@pytest.fixture
def sample_test_data():
    data = {
        "SK_ID_CURR": [100002, 100003],
        "AMT_CREDIT": [406597.5, 1293502.5],
        "FEATURE_A": [1, 0],
        "FEATURE_B": [0, 1],
    }
    df = pd.DataFrame(data)
    if "SK_ID_CURR" in df.columns:
        df = df.set_index("SK_ID_CURR")
    return df


@pytest.fixture
def expected_features_list():
    return ["FEATURE_A", "FEATURE_B"]


@pytest.fixture
def default_threshold():
    return 0.5


@pytest.fixture
def prediction_service(
    mock_pyfunc_model, sample_test_data, expected_features_list, default_threshold
):
    """Fixture to create a PredictionService instance with mocked dependencies."""
    return PredictionService(
        pyfunc_model=mock_pyfunc_model,
        test_data=sample_test_data,
        expected_features=expected_features_list,
        threshold=default_threshold,
    )


# --- Test Cases for PredictionService ---


def test_prediction_service_initialization(
    prediction_service,
    mock_pyfunc_model,
    sample_test_data,
    expected_features_list,
    default_threshold,
):
    """
    Test 1: Check if the PredictionService initializes correctly with valid inputs.
    Explanation: Verifies that the service can be created when all necessary components
                 (model, data, features, threshold) are provided.
    """
    assert prediction_service.pyfunc_model is mock_pyfunc_model
    assert prediction_service.test_data.equals(sample_test_data)
    assert prediction_service.expected_features == expected_features_list
    assert prediction_service.threshold == default_threshold
    print("\nTest 1 Passed: PredictionService initialized successfully.")


def test_prediction_service_init_invalid_model():
    """
    Test 2: Check if PredictionService raises ValueError for None model.
    Explanation: Ensures the service correctly handles a critical missing component (the model)
                 at startup by raising an error.
    """
    with pytest.raises(
        ValueError, match="requires a valid MLflow pyfunc model instance"
    ):
        PredictionService(
            pyfunc_model=None,
            test_data=MagicMock(),
            expected_features=["X"],
            threshold=0.5,
        )
    print("\nTest 2 Passed: PredictionService raised ValueError for None model.")


def test_prediction_service_init_invalid_test_data(mock_pyfunc_model):
    """
    Test 3: Check if PredictionService raises ValueError for empty test data.
    Explanation: Ensures the service correctly handles missing or invalid input data
                 by raising an error, preventing issues later during prediction.
    """
    with pytest.raises(ValueError, match="requires valid, non-empty test data"):
        PredictionService(
            pyfunc_model=mock_pyfunc_model,
            test_data=pd.DataFrame(),
            expected_features=["X"],
            threshold=0.5,
        )
    print("\nTest 3 Passed: PredictionService raised ValueError for empty test_data.")


@patch("src.credit_risk_app.services.apply_transformations")  # Mock the preprocessing
@patch(
    "src.credit_risk_app.services.random.choice"
)  # Mock random.choice for predictable client selection
def test_get_prediction_for_random_client(
    mock_random_choice,
    mock_apply_transformations,
    prediction_service,
    sample_test_data,
    expected_features_list,
    default_threshold,  # Added default_threshold fixture
):
    """
    Test 4: Check the structure and basic content of get_prediction_for_random_client.
    Explanation: This test checks if the main prediction method runs and returns a dictionary
                 with the expected keys and data types. It uses 'mocks' for components
                 like data selection and preprocessing to isolate the service's logic.
    """
    # Configure mock for random.choice to pick the first client
    first_client_id = sample_test_data.index[0]
    mock_random_choice.return_value = first_client_id

    # Configure the mock for apply_transformations
    mock_processed_df = pd.DataFrame([[0.1, 0.9]], columns=expected_features_list)
    mock_apply_transformations.return_value = mock_processed_df

    # Define expected values based on the first client in sample_test_data
    expected_loan_id = int(first_client_id)
    expected_credit_amount = float(sample_test_data.loc[first_client_id]["AMT_CREDIT"])

    result = prediction_service.get_prediction_for_random_client()

    assert isinstance(result, dict)
    assert "loan_id" in result
    assert "probability_neg" in result
    assert "probability_pos" in result
    assert "threshold" in result
    assert "credit_amount" in result

    assert result["loan_id"] == expected_loan_id
    assert isinstance(result["probability_neg"], float)
    assert isinstance(result["probability_pos"], float)
    assert (
        result["threshold"] == default_threshold
    )  # Check against the service's threshold
    assert result["credit_amount"] == round(
        expected_credit_amount, 2
    )  # Used expected_credit_amount

    assert 0 <= result["probability_neg"] <= 1
    assert 0 <= result["probability_pos"] <= 1

    mock_apply_transformations.assert_called_once()
    # Get the DataFrame that was passed to apply_transformations
    call_args_df = mock_apply_transformations.call_args[0][0]
    # Check if the DataFrame passed to apply_transformations corresponds to the selected client
    pd.testing.assert_frame_equal(call_args_df, sample_test_data.loc[[first_client_id]])

    print(
        f"\nTest 4 Passed: get_prediction_for_random_client returned expected structure. Result: {result}"
    )
