# main.py
import logging
import anyio
import mlflow
from typing import Optional, List
from pathlib import Path
from contextlib import asynccontextmanager

import pandas as pd
import uvicorn
from fastapi import FastAPI, Request, HTTPException, Depends, Response
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, ConfigDict, computed_field

from .config import COLUMNS_TO_IMPORT, DEFAULT_PREDICTION_THRESHOLD
from .services import PredictionService

# --- Configuration & Setup ---

# Redirect mlflow file‐store
mlflow.set_tracking_uri("file:///tmp/mlruns-disabled")

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT_DIR = APP_DIR.parent.parent

MODEL_PATH = PROJECT_ROOT_DIR / "models" / "gradient_boosting"
TEST_DATA_PATH = PROJECT_ROOT_DIR / "data" / "application_test.csv"
STATIC_DIR = PROJECT_ROOT_DIR / "static"
TEMPLATES_DIR = PROJECT_ROOT_DIR / "templates"

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

templates = Jinja2Templates(directory=TEMPLATES_DIR)


# --- Custom Exceptions ---
class ResourceError(Exception):
    """Base class for resource-related errors."""

    pass


class ModelError(ResourceError):
    """Base class for model loading or processing errors."""

    pass


class ModelNotFoundError(ModelError):
    """Raised when a model file/directory is not found."""

    pass


class ModelLoadError(ModelError):
    """Raised for general errors during model loading."""

    pass


class ModelSignatureError(ModelError):
    """Raised for errors related to model signature (e.g., missing inputs)."""

    pass


class ThresholdError(ModelError):
    """Raised for errors related to extracting the model threshold if critical."""

    pass


class DataError(ResourceError):
    """Base class for data loading or processing errors."""

    pass


class DataNotFoundError(DataError):
    """Raised when a data file is not found."""

    pass


class DataLoadError(DataError):
    """Raised for general errors during data loading or validation."""

    pass


# --- Pydantic Models ---
class PredictionResponse(BaseModel):
    threshold: float = Field(
        ..., ge=0, le=1, description="Decision threshold used for classification."
    )
    probability_neg: float = Field(
        ...,
        ge=0,
        le=1,
        alias="probabilityClass0",
        description="Predicted probability of the negative class (class 0 - likely accepted).",
    )
    probability_pos: float = Field(
        ...,
        ge=0,
        le=1,
        alias="probabilityClass1",
        description="Predicted probability of the positive class (class 1 - likely refused).",
    )
    loan_id: int = Field(
        ..., gt=0, description="Unique identifier for the loan application."
    )
    credit_amount: float = Field(
        ..., ge=0, description="The amount of credit requested."
    )

    @computed_field
    @property
    def decision(self) -> str:
        """
        Determines the loan decision based on the probability of the positive class
        (Not Repaid) and the threshold.
        """
        if self.probability_pos >= self.threshold:
            return "refused"
        else:
            return "accepted"

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "threshold": 0.5,
                "probabilityClass0": 0.8532,
                "probabilityClass1": 0.1468,
                "loan_id": 123456,
                "credit_amount": 50000.00,
            }
        },
    )


# --- Resource Loaders ---
class ModelLoader:
    def __init__(self, model_path: str, default_threshold: float):
        self.model_path = model_path
        self.default_threshold = default_threshold
        self.logger = logging.getLogger(self.__class__.__name__)
        self._pyfunc_model: Optional[mlflow.pyfunc.PyFuncModel] = None
        self._expected_features: Optional[List[str]] = None
        self._threshold: Optional[float] = None

    def load_all(self) -> None:
        """Loads all model-related resources and extracts metadata."""
        self._load_pyfunc_model()
        self._extract_features_from_signature()
        self._extract_threshold_from_model()

    @property
    def pyfunc_model(self) -> mlflow.pyfunc.PyFuncModel:
        if self._pyfunc_model is None:
            raise ModelError("Pyfunc model accessed before successful loading.")
        return self._pyfunc_model

    @property
    def expected_features(self) -> List[str]:
        if self._expected_features is None:
            raise ModelError("Expected features accessed before successful extraction.")
        return self._expected_features

    @property
    def threshold(self) -> float:
        if self._threshold is None:
            raise ModelError(
                "Threshold accessed before being set (should be defaulted)."
            )
        return self._threshold

    def _load_pyfunc_model(self) -> None:
        try:
            self._pyfunc_model = mlflow.pyfunc.load_model(self.model_path)
            self.logger.info("Pyfunc model loaded successfully.")
        except FileNotFoundError as e:
            msg = f"Pyfunc model directory not found at {self.model_path}"
            self.logger.error(msg)
            raise ModelNotFoundError(msg) from e
        except (mlflow.exceptions.MlflowException, ImportError, Exception) as e:
            msg = f"Error loading MLflow pyfunc model from {self.model_path}: {e}"
            self.logger.error(msg, exc_info=True)
            raise ModelLoadError(msg) from e

    def _extract_features_from_signature(self) -> None:
        if not self._pyfunc_model:
            raise ModelSignatureError(
                "Cannot extract features: Pyfunc model not loaded."
            )
        self.logger.info("Extracting features from pyfunc model signature...")
        try:
            if (
                hasattr(self._pyfunc_model, "metadata")
                and self._pyfunc_model.metadata
                and hasattr(self._pyfunc_model.metadata, "signature")
                and self._pyfunc_model.metadata.signature
                and hasattr(self._pyfunc_model.metadata.signature, "inputs")
                and self._pyfunc_model.metadata.signature.inputs
            ):
                signature_inputs = self._pyfunc_model.metadata.signature.inputs
                feature_names = None
                # Check for standard ways to get input names
                if hasattr(signature_inputs, "input_names") and callable(
                    signature_inputs.input_names
                ):
                    feature_names = signature_inputs.input_names()
                elif hasattr(
                    signature_inputs, "names"
                ):  # Older MLflow versions might use this
                    feature_names = signature_inputs.names

                if feature_names:
                    self._expected_features = feature_names
                    self.logger.info(
                        f"Extracted {len(self._expected_features)} expected features."
                    )
                    return
                else:
                    msg = "Could not extract input names from model signature's inputs schema."
                    self.logger.error(msg)
                    raise ModelSignatureError(msg)
            else:
                msg = "Pyfunc model metadata or signature structure is incomplete or missing for feature extraction."
                self.logger.error(msg)
                raise ModelSignatureError(msg)
        except Exception as e:
            msg = f"Error reading pyfunc model signature details: {e}"
            self.logger.error(msg, exc_info=True)
            raise ModelSignatureError(msg) from e

    def _extract_threshold_from_model(self) -> None:
        self._threshold = self.default_threshold

        if (
            not self._pyfunc_model
            or not hasattr(self._pyfunc_model, "metadata")
            or not self._pyfunc_model.metadata
        ):
            self.logger.warning(
                "Pyfunc model or its top-level metadata is not available for threshold extraction. "
                f"Using default threshold: {self.default_threshold}"
            )
            # self._threshold is already default_threshold
            self.logger.info(f"Final threshold set to: {self._threshold:.4f}")
            return

        # Access the custom metadata dictionary (passed as `metadata` param during mlflow.pyfunc.log_model)
        custom_model_metadata = self._pyfunc_model.metadata.metadata

        if custom_model_metadata and isinstance(custom_model_metadata, dict):
            optimal_threshold_value = custom_model_metadata.get("optimal_threshold")
            if optimal_threshold_value is not None:
                try:
                    extracted_threshold = float(optimal_threshold_value)
                    if 0 <= extracted_threshold <= 1:
                        self._threshold = extracted_threshold
                        self.logger.info(
                            f"Extracted threshold from pyfunc model metadata 'optimal_threshold': {self._threshold:.4f}"
                        )
                    else:
                        self.logger.warning(
                            f"Extracted 'optimal_threshold' ({extracted_threshold}) is outside the valid range [0, 1]. "
                            f"Using default threshold: {self.default_threshold}"
                        )
                except (TypeError, ValueError) as e:
                    self.logger.warning(
                        "Error converting 'optimal_threshold' from model metadata to float: "
                        f"{e}. Using default threshold: {self.default_threshold}",
                        exc_info=True,
                    )
            else:
                self.logger.info(
                    "'optimal_threshold' key not found in pyfunc model's custom metadata. "
                    f"Using default threshold: {self.default_threshold}"
                )
        else:
            self.logger.info(
                "No custom metadata dictionary found in pyfunc model (or it's not a dictionary). "
                f"Using default threshold: {self.default_threshold}"
            )

        self.logger.info(f"Final threshold set to: {self._threshold:.4f}")


class DataLoader:
    def __init__(
        self, test_data_path: str, columns_to_import: Optional[List[str]] = None
    ):
        self.test_data_path = test_data_path
        self.columns_to_import = columns_to_import
        self.logger = logging.getLogger(self.__class__.__name__)
        self._test_data: Optional[pd.DataFrame] = None

    def load(self) -> None:
        try:
            cols_kwarg = (
                {"usecols": self.columns_to_import} if self.columns_to_import else {}
            )
            num_cols_info = f"using {len(self.columns_to_import) if self.columns_to_import else 'all available'} columns"

            read_csv_kwargs = {}
            read_csv_kwargs["low_memory"] = False
            self.logger.info(f"Reading CSV ({num_cols_info}).")

            current_test_data = pd.read_csv(
                self.test_data_path, **cols_kwarg, **read_csv_kwargs
            )

            essential_cols = [
                "SK_ID_CURR",
                "AMT_CREDIT",
            ]
            missing_essential = [
                c for c in essential_cols if c not in current_test_data.columns
            ]
            if missing_essential:
                msg = f"Essential columns {missing_essential} missing from loaded test data."
                self.logger.error(msg)
                raise DataLoadError(msg)

            self.logger.info(
                f"Test data loaded successfully. Shape: {current_test_data.shape}"
            )

            try:
                current_test_data.set_index(
                    "SK_ID_CURR", inplace=True, verify_integrity=True
                )
                self.logger.info(
                    "Set 'SK_ID_CURR' as index for test data. Index is now verified unique."
                )
            except (
                Exception
            ) as e:  # Catches errors like non-unique values in SK_ID_CURR
                msg = (
                    f"Critical error: Could not set 'SK_ID_CURR' as a unique index. "
                    f"SK_ID_CURR must be unique for consistent client identification. Error: {e}"
                )
                self.logger.error(msg, exc_info=True)
                raise DataLoadError(msg) from e
            self._test_data = current_test_data

        except FileNotFoundError as e:
            msg = f"Test data file not found at {self.test_data_path}"
            self.logger.error(msg)
            raise DataNotFoundError(msg) from e
        except ValueError as e:
            msg = f"Value error loading or processing test data (e.g. column mismatch, bad data): {e}"
            self.logger.error(msg, exc_info=True)
            raise DataLoadError(msg) from e
        except Exception as e:
            msg = f"Unexpected error loading or processing test data: {e}"
            self.logger.error(msg, exc_info=True)
            raise DataLoadError(msg) from e

    @property
    def test_data(self) -> pd.DataFrame:
        if self._test_data is None:
            raise DataError("Test data accessed before successful loading.")
        return self._test_data


class ResourceManager:  # Orchestrator
    def __init__(self, model_loader: ModelLoader, data_loader: DataLoader):
        self.model_loader = model_loader
        self.data_loader = data_loader
        self.logger = logging.getLogger(self.__class__.__name__)

        self._pyfunc_model: Optional[mlflow.pyfunc.PyFuncModel] = None
        self._test_data: Optional[pd.DataFrame] = None
        self._expected_features: Optional[List[str]] = None
        self._threshold: Optional[float] = None

    def load_resources(self) -> None:
        self.logger.info("Orchestrating resource loading...")
        try:
            self.model_loader.load_all()
            self._pyfunc_model = self.model_loader.pyfunc_model
            self._expected_features = self.model_loader.expected_features
            self._threshold = self.model_loader.threshold
            self.logger.info("Model resources loaded and processed by orchestrator.")

            self.data_loader.load()
            self._test_data = self.data_loader.test_data
            self.logger.info("Data resources loaded and processed by orchestrator.")

            self.logger.info("All resources successfully loaded and orchestrated.")

        except ResourceError as e:
            self.logger.error(
                f"Resource loading failed during orchestration: {e}", exc_info=True
            )
            raise
        except Exception as e:
            self.logger.error(
                f"Unexpected error during resource orchestration: {e}", exc_info=True
            )
            raise ResourceError(f"Unexpected orchestration error: {e}") from e

    def release_resources(self) -> None:
        self.logger.info("Releasing orchestrated resources...")
        self._pyfunc_model = None
        self._test_data = None
        self._expected_features = None
        self._threshold = None
        self.logger.info("Orchestrated resources released.")

    @property
    def pyfunc_model(self) -> mlflow.pyfunc.PyFuncModel:
        if self._pyfunc_model is None:
            raise ResourceError("Pyfunc model not available via ResourceManager.")
        return self._pyfunc_model

    @property
    def test_data(self) -> pd.DataFrame:
        if self._test_data is None:
            raise ResourceError("Test data not available via ResourceManager.")
        return self._test_data

    @property
    def expected_features(self) -> List[str]:
        if self._expected_features is None:
            raise ResourceError("Expected features not available via ResourceManager.")
        return self._expected_features

    @property
    def threshold(self) -> float:
        if self._threshold is None:
            raise ResourceError("Threshold not available via ResourceManager.")
        return self._threshold


# --- Lifespan Context Manager ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application startup: Initializing resource loaders...")
    model_loader = ModelLoader(MODEL_PATH, DEFAULT_PREDICTION_THRESHOLD)
    data_loader = DataLoader(TEST_DATA_PATH, columns_to_import=COLUMNS_TO_IMPORT)
    resource_manager = ResourceManager(model_loader, data_loader)

    app.state.resource_manager = resource_manager
    app.state.prediction_service = None

    try:
        logger.info(
            "Application startup: Offloading resource loading to thread pool..."
        )
        await anyio.to_thread.run_sync(resource_manager.load_resources)
        logger.info("Application startup: Resources reported as loaded by manager.")

        logger.info("Application startup: Initializing PredictionService...")
        app.state.prediction_service = PredictionService(
            pyfunc_model=resource_manager.pyfunc_model,
            test_data=resource_manager.test_data,
            expected_features=resource_manager.expected_features,
            threshold=resource_manager.threshold,
        )
        logger.info("Application startup: PredictionService initialized successfully.")

    except ResourceError as e:
        err_msg = (
            f"Critical startup failure: Failed to load essential resources. Reason: {e}"
        )
        logger.critical(err_msg, exc_info=True)
        raise RuntimeError(err_msg) from e
    except (
        ValueError,  # Can be raised by PredictionService constructor
        TypeError,
    ) as service_init_error:  # Broad catch for other unexpected service init issues
        err_msg = f"Critical startup error: Failed to initialize PredictionService: {service_init_error}"
        logger.critical(err_msg, exc_info=True)
        raise RuntimeError(err_msg) from service_init_error
    except Exception as e:  # Catch-all for any other unexpected startup error
        logger.critical(
            f"Application startup failed critically and unexpectedly: {e}",
            exc_info=True,
        )
        error_message = (
            "Application startup failed unexpectedly during resource loading or "
            f"service initialization: {e}"
        )
        raise RuntimeError(error_message) from e

    logger.info("Application startup: Completed successfully, yielding control.")
    yield

    logger.info("Application shutdown: Releasing resources...")
    # Ensure resource_manager exists before trying to release
    if (
        hasattr(app.state, "resource_manager")
        and app.state.resource_manager is not None
    ):
        await anyio.to_thread.run_sync(app.state.resource_manager.release_resources)
    app.state.prediction_service = None  # Explicitly clear service
    logger.info("Application shutdown: Completed.")


# --- FastAPI App ---
app = FastAPI(
    lifespan=lifespan,
    title="Credit Risk Prediction API",
    version="1.0.0",
    description=(
        "API to predict credit risk for loan applicants. "
        "Uses MLflow pyfunc model, relies on its signature for feature handling, "
        "and extracts threshold from model metadata. "
    ),
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# --- Dependency Injection ---
def get_prediction_service(request: Request) -> PredictionService:
    if (
        not hasattr(request.app.state, "prediction_service")
        or request.app.state.prediction_service is None
    ):
        logger.error(
            "Prediction service accessed but is not available. This indicates a startup failure or incomplete initialization."
        )
        # 503 Service Unavailable is appropriate here.
        raise HTTPException(
            status_code=503,
            detail="Service unavailable. Critical resources may have failed to load on startup. Check server logs.",
        )
    return request.app.state.prediction_service


# --- API Endpoints ---
@app.get("/ping", tags=["Health"])
async def ping():
    logger.info("---- PING ENDPOINT HIT ----")
    return {"message": "pong"}


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def get_home(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/favicon.ico", include_in_schema=False)
async def favicon() -> FileResponse:
    favicon_path: Path = STATIC_DIR / "favicon.ico"
    if not favicon_path.exists():
        logger.debug(f"Favicon not found at {favicon_path}")
        raise HTTPException(status_code=404, detail="Favicon not found")
    return FileResponse(favicon_path)


@app.get("/healthz", tags=["Health"])
async def health_check(request: Request) -> Response:
    if (
        hasattr(request.app.state, "prediction_service")
        and request.app.state.prediction_service is not None
    ):
        logger.debug("Health check: Service is ready.")
        return JSONResponse({"status": "ok", "message": "Service is ready."})
    else:
        logger.warning(
            "Health check: Service is unavailable due to startup issues or incomplete initialization."
        )
        return JSONResponse(
            {
                "status": "unavailable",
                "message": "Prediction service is not initialized.",
            },
            status_code=503,  # Service Unavailable
        )


@app.get("/predict", response_model=PredictionResponse)
async def predict_random_client_endpoint(
    service: PredictionService = Depends(get_prediction_service),
) -> PredictionResponse:
    logger.debug("Received request for /predict endpoint.")
    try:
        prediction_data = service.get_prediction_for_random_client()
        return PredictionResponse(**prediction_data)
    except HTTPException as http_exc:
        logger.warning(
            f"HTTP exception during prediction: {http_exc.status_code} - {http_exc.detail}"
        )
        raise http_exc
    except ValueError as ve:
        logger.error(f"Value error during prediction: {ve}", exc_info=True)
        raise HTTPException(
            status_code=400,
            detail=f"Prediction failed due to invalid data or configuration: {ve}",
        )
    except Exception as e:
        logger.error(f"Unexpected error in /predict endpoint: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="Internal server error during prediction."
        )


@app.get("/predict/{loan_id}", response_model=PredictionResponse, tags=["Predictions"])
async def predict_specific_client_endpoint(
    loan_id: int,  # Path parameter from URL
    service: PredictionService = Depends(get_prediction_service),
) -> PredictionResponse:
    logger.debug(f"Received request for /predict/{loan_id} endpoint.")
    try:
        prediction_data = service.get_prediction_for_specific_client(loan_id)
        return PredictionResponse(**prediction_data)
    except HTTPException as http_exc:
        logger.warning(
            f"HTTP exception for specific client ID {loan_id}: {http_exc.status_code} - {http_exc.detail}"
        )
        raise http_exc
    except ValueError as ve:
        logger.error(
            f"Value error during prediction for specific client ID {loan_id}: {ve}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=400,
            detail=f"Prediction failed due to invalid data or ID format: {ve}",
        )
    except Exception as e:
        logger.error(
            f"Unexpected error in /predict/{loan_id} endpoint: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error during prediction for loan ID {loan_id}.",
        )


# --- Main Execution Guard ---
if __name__ == "__main__":
    logger.info(
        "Starting Uvicorn server for main execution (host=0.0.0.0, port=8080)..."
    )
    uvicorn.run("src.credit_risk_app.main:app", host="0.0.0.0", port=8080, reload=False)
