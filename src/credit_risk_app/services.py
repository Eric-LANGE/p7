# services.py
import logging
import random
import mlflow
from typing import Any, Dict, List

import pandas as pd
import numpy as np

from fastapi import HTTPException

from .preprocessing import apply_transformations

logger = logging.getLogger(__name__)

# Redirect mlflow file‐store
mlflow.set_tracking_uri("file:///tmp/mlruns-disabled")


class PredictionService:
    """
    Handles the business logic for generating credit risk predictions using
    an MLflow pyfunc model.
    """

    def __init__(
        self,
        pyfunc_model: mlflow.pyfunc.PyFuncModel,
        test_data: pd.DataFrame,
        expected_features: List[str],
        threshold: float,
    ):
        if pyfunc_model is None:
            raise ValueError(
                "PredictionService requires a valid MLflow pyfunc model instance."
            )
        if test_data is None or test_data.empty:
            raise ValueError("PredictionService requires valid, non-empty test data.")
        if (
            not isinstance(test_data.index, pd.Index)
            or test_data.index.name != "SK_ID_CURR"
        ):
            logger.error(
                "Test data index is not named 'SK_ID_CURR' or is not a proper index."
            )
            raise ValueError(
                "PredictionService requires test_data to have 'SK_ID_CURR' as its index name."
            )
        if not expected_features:
            raise ValueError(
                "PredictionService requires a list of expected features from the model signature."
            )
        if not (0 <= threshold <= 1):
            raise ValueError(
                f"PredictionService requires a valid threshold (0-1), received: {threshold}"
            )

        self.pyfunc_model = pyfunc_model
        self.test_data = test_data
        self.expected_features = expected_features
        self.threshold = threshold
        logger.info(
            f"PredictionService initialized with threshold: {self.threshold:.4f} "
            f"and {len(self.expected_features)} expected features. Test data index: {self.test_data.index.name}"
        )

    def _validate_data_columns(
        self, df: pd.DataFrame, required_cols: list[str]
    ) -> None:
        """Validates presence of required columns in the DataFrame."""
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            logger.error(
                f"Missing required informational columns in selected client data row: {missing}. Available: {list(df.columns)}"
            )
            raise HTTPException(
                status_code=500,
                detail=f"Internal error: Missing essential data columns {missing} for prediction response.",
            )

    def _select_random_client_data(self) -> tuple[pd.DataFrame, int]:
        """
        Selects a random client's original data and loan ID.
        """
        if self.test_data.empty:
            logger.error("Attempted prediction with empty test_data DataFrame.")
            raise HTTPException(
                status_code=500, detail="Internal server error: Test data is empty."
            )
        if self.test_data.index.empty:
            logger.error(
                "Test data index ('SK_ID_CURR') is empty, cannot select random client."
            )
            raise HTTPException(
                status_code=500,
                detail="Internal server error: Test data index is empty.",
            )

        random_loan_id_val = random.choice(self.test_data.index.tolist())

        try:
            loan_id = int(random_loan_id_val)
        except (ValueError, TypeError) as e:
            logger.error(
                f"Could not convert selected SK_ID_CURR ('{random_loan_id_val}') from index to int: {e}"
            )
            raise HTTPException(
                status_code=500,
                detail="Internal server error: Invalid Loan ID format in test data index.",
            )

        logger.info(f"Selected random Loan ID (from index 'SK_ID_CURR'): {loan_id}")

        row_df_orig = self.test_data.loc[[loan_id]].copy()

        return row_df_orig, loan_id

    def _extract_client_info(self, row_df_orig: pd.DataFrame, loan_id: int) -> float:
        """
        Extracts and validates basic client info (credit amount) from the original selected row.
        """
        required_info_cols = ["AMT_CREDIT"]
        self._validate_data_columns(row_df_orig, required_info_cols)

        try:
            credit_amount_raw = row_df_orig["AMT_CREDIT"].item()
        except Exception as e:
            logger.error(
                f"Error extracting AMT_CREDIT for Loan ID {loan_id} from row: {row_df_orig.head()}: {e}"
            )
            raise HTTPException(
                status_code=500,
                detail="Internal server error: Could not extract credit amount.",
            )

        return float(credit_amount_raw) if pd.notna(credit_amount_raw) else 0.0

    def _preprocess_client_data(
        self, row_df_orig: pd.DataFrame, loan_id: int
    ) -> pd.DataFrame:
        """
        Preprocesses the client's data using expected features.
        """
        logger.debug(
            f"Preprocessing data for Loan ID: {loan_id} (Index: {row_df_orig.index.item()}) "
            f"using {len(self.expected_features)} expected features."
        )
        processed_row_df = apply_transformations(row_df_orig, self.expected_features)

        logger.debug(
            f"Preprocessing complete for Loan ID: {loan_id}. "
            f"Processed DataFrame shape: {processed_row_df.shape}, Index: {processed_row_df.index.item()}"
        )

        # Final check on columns
        if list(processed_row_df.columns) != self.expected_features:
            logger.error(
                f"Column mismatch after preprocessing for Loan ID {loan_id}. "
                f"Expected: {self.expected_features}, Got: {list(processed_row_df.columns)}"
            )
            raise HTTPException(
                status_code=500,
                detail="Internal error: Feature mismatch after preprocessing.",
            )
        return processed_row_df

    def _get_model_prediction(
        self, processed_row_df: pd.DataFrame, loan_id: int
    ) -> tuple[float, float]:
        """
        Gets prediction probabilities from the pyfunc model.
        """
        logger.info(
            f"Predicting probabilities using pyfunc model for Loan ID: {loan_id} (Index: {processed_row_df.index.item()})..."
        )
        prediction_output = self.pyfunc_model.predict(processed_row_df)
        logger.info(f"Prediction successful for Loan ID: {loan_id}.")

        p_neg: float
        p_pos: float

        if (
            isinstance(prediction_output, np.ndarray)
            and prediction_output.ndim == 2
            and prediction_output.shape == (1, 2)
        ):
            p_neg, p_pos = prediction_output[0, 0], prediction_output[0, 1]
        elif isinstance(
            prediction_output, pd.DataFrame
        ) and prediction_output.shape == (1, 2):
            try:
                p_neg, p_pos = (
                    prediction_output.iloc[0, 0],
                    prediction_output.iloc[0, 1],
                )
            except Exception as e:
                log_msg = (
                    f"Could not extract probabilities from DataFrame output for "
                    f"Loan ID {loan_id}: {e}. Output columns: {prediction_output.columns}, Output: {prediction_output.values}"
                )
                logger.error(log_msg)
                raise HTTPException(
                    status_code=500,
                    detail="Internal error: Unexpected prediction output format (DataFrame access).",
                )
        else:
            log_msg = (
                f"Unexpected prediction output format or shape for Loan ID {loan_id}. "
                f"Expected (1,2) ndarray or DataFrame. "
                f"Got Type: {type(prediction_output)}, Shape/Content: {prediction_output}"
            )
            logger.error(log_msg)
            raise HTTPException(
                status_code=500,
                detail="Internal error: Unexpected prediction output format from model.",
            )
        return float(p_neg), float(p_pos)

    def _format_prediction_response(
        self, p_neg: float, p_pos: float, loan_id: int, credit_amount: float
    ) -> Dict[str, Any]:
        """Formats the final prediction response dictionary."""
        log_msg = (
            f"Loan ID {loan_id} probabilities -> Negative class - Repaid: {p_neg:.4f}, "
            f"Positive class - Not Repaid: {p_pos:.4f}, threshold={self.threshold:.4f}"
        )
        logger.info(log_msg)
        return {
            "threshold": round(float(self.threshold), 4),
            "probability_neg": round(p_neg, 4),
            "probability_pos": round(p_pos, 4),
            "loan_id": loan_id,  # Already int
            "credit_amount": round(credit_amount, 2),
        }

    def get_prediction_for_random_client(self) -> Dict[str, Any]:
        """
        Orchestrates the full process of selecting a client, preprocessing,
        predicting, and formatting the response.
        """
        loan_id_for_log: Any = "unknown_yet"
        try:
            row_df_orig, loan_id = self._select_random_client_data()
            loan_id_for_log = loan_id

            credit_amount = self._extract_client_info(row_df_orig, loan_id)
            logger.info(
                f"Processing prediction request for Loan ID: {loan_id}, Amount: {credit_amount}"
            )

            processed_row_df = self._preprocess_client_data(row_df_orig, loan_id)

            p_neg, p_pos = self._get_model_prediction(processed_row_df, loan_id)

            return self._format_prediction_response(
                p_neg, p_pos, loan_id, credit_amount
            )

        except IndexError as e:
            logger.error(
                f"IndexError during random client selection or data access (Loan ID context: {loan_id_for_log}). Error: {e}",
                exc_info=True,
            )
            raise HTTPException(
                status_code=500,
                detail="Internal server error: Could not select or access client data due to an indexing issue.",
            )
        except KeyError as e:
            logger.error(
                f"Missing key (column) during data access or processing for Loan ID {loan_id_for_log}: {e}",
                exc_info=True,
            )
            raise HTTPException(
                status_code=500,
                detail=f"Internal error: Missing expected data field '{e}'.",
            )
        except (
            AttributeError,
            TypeError,
        ) as e:
            logger.error(
                f"Type or attribute error during prediction logic for Loan ID {loan_id_for_log}: {e}",
                exc_info=True,
            )
            raise HTTPException(
                status_code=500,
                detail="Internal error during prediction processing due to type or attribute mismatch.",
            )
        except ValueError as ve:
            logger.error(
                f"Value error during prediction service execution for Loan ID {loan_id_for_log}: {ve}",
                exc_info=True,
            )
            raise HTTPException(
                status_code=400 if "Invalid data" in str(ve) else 500,
                detail=f"Prediction failed: {ve}",
            )
        except HTTPException as http_exc:
            raise http_exc
        except Exception as e:
            logger.critical(
                f"An unexpected critical error occurred in PredictionService for Loan ID {loan_id_for_log}: {e}",
                exc_info=True,
            )
            raise HTTPException(
                status_code=500,
                detail="An unexpected internal server error occurred during prediction.",
            )

    def get_prediction_for_specific_client(self, loan_id: int) -> Dict[str, Any]:
        """
        Orchestrates the full process of preprocessing, predicting,
        and formatting the response for a specific client ID.
        """
        logger.info(f"Processing prediction request for specific Loan ID: {loan_id}")
        try:
            if loan_id not in self.test_data.index:
                logger.warning(f"Loan ID {loan_id} not found in test data index.")
                raise HTTPException(
                    status_code=404, detail=f"Loan ID {loan_id} not found."
                )
            row_df_orig = self.test_data.loc[[loan_id]].copy()
            logger.debug(
                f"Data selected for Loan ID {loan_id}. Shape: {row_df_orig.shape}"
            )
            credit_amount = self._extract_client_info(row_df_orig, loan_id)
            logger.info(
                f"Extracted info for Loan ID: {loan_id}, Amount: {credit_amount}"
            )
            processed_row_df = self._preprocess_client_data(row_df_orig, loan_id)

            p_neg, p_pos = self._get_model_prediction(processed_row_df, loan_id)

            return self._format_prediction_response(
                p_neg, p_pos, loan_id, credit_amount
            )

        except HTTPException as http_exc:
            logger.warning(
                f"HTTP exception for Loan ID {loan_id}: {http_exc.status_code} - {http_exc.detail}"
            )
            raise http_exc
        except KeyError:
            logger.error(
                f"Loan ID {loan_id} not found in test data during .loc access (should have been caught earlier).",
                exc_info=True,
            )
            raise HTTPException(
                status_code=404,
                detail=f"Loan ID {loan_id} not found (processing error).",
            )
        except IndexError as e:
            logger.error(
                f"IndexError during data access for Loan ID {loan_id}: {e}",
                exc_info=True,
            )
            raise HTTPException(
                status_code=500,
                detail="Internal server error: Could not access client data due to an indexing issue.",
            )
        except (
            AttributeError,
            TypeError,
        ) as e:
            logger.error(
                f"Type or attribute error during prediction logic for Loan ID {loan_id}: {e}",
                exc_info=True,
            )
            raise HTTPException(
                status_code=500,
                detail="Internal error during prediction processing due to type or attribute mismatch.",
            )
        except ValueError as ve:
            logger.error(
                f"Value error during prediction service execution for Loan ID {loan_id}: {ve}",
                exc_info=True,
            )
            raise HTTPException(
                status_code=(400 if "Invalid data" in str(ve) else 500),
                detail=f"Prediction failed for Loan ID {loan_id}: {ve}",
            )
        except Exception as e:
            logger.critical(
                f"An unexpected critical error occurred in PredictionService for Loan ID {loan_id}: {e}",
                exc_info=True,
            )
            raise HTTPException(
                status_code=500,
                detail=f"An unexpected internal server error occurred during prediction for Loan ID {loan_id}.",
            )
