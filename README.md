---
title: Credit Risk Analysis
emoji: 📊
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: false
---

# Credit Risk Prediction API & Dashboard

## Overview

This project provides a Credit Risk Prediction tool accessible via a web dashboard and an API. The system uses a machine learning model, managed with MLflow, to predict the probability of a loan applicant defaulting. Users can input client data or request a random client's prediction and view the risk assessment through an interactive dashboard.

## Features

* **Credit Risk Prediction:** Predicts the likelihood of credit default for a given client.
* **Interactive Dashboard:** A web-based interface to visualize prediction scores, client information.
* **REST API:** Provides endpoints for predictions, client data retrieval, and model information.
* **MLflow Integration:** Uses MLflow for model tracking and management.

## Project Structure

```
project_p7/
├── .github/                 # GitHub Actions workflows
├── data/
│   └── application_test.csv # Sample data for testing predictions
├── src/
│   └── credit_risk_app/
│       ├── __init__.py
│       ├── config.py        # Configuration for features, paths, model
│       ├── main.py          # FastAPI application, API endpoints
│       ├── preprocessing.py # Data preprocessing and transformation logic
│       └── services.py      # Business logic for predictions, data loading
├── static/
│   ├── css/
│   │   └── style.css      # CSS for the dashboard
│   └── js/
│       ├── api.js           # JavaScript for API communication (frontend)
│       └── script.js        # JavaScript for dashboard interactivity
├── templates/
│   └── index.html           # HTML structure for the dashboard
├── tests/
│   ├── test_main.py       # Tests for API endpoints
│   └── test_services.py   # Tests for prediction and data services
├── .dockerignore            # Specifies intentionally untracked files for Docker
├── .gitignore               # Specifies intentionally untracked files for Git
├── credit_risk_env.yml      # Conda environment definition
├── Dockerfile               # Docker configuration
├── entrypoint.sh            # Script to run the application
├── pytest.ini               # Pytest configuration
├── README.md                # This file

## Running the Application

uvicorn main:app --reload --host 0.0.0.0 --port 8000

## API Endpoints

The application exposes the following API endpoints through FastAPI:

* **`GET /`**: Serves the main HTML dashboard.
* **`GET /api/schema/`**: Returns the schema of the input data expected by the model.
* **`GET /api/random-prediction/`**: Returns a prediction for a randomly selected client from `application_test.csv`.
* **`GET /api/client-info/{loan_id}`**: Returns a prediction for a specific client from `application_test.csv` by `loan_id`.

## Testing

The project uses `pytest` for testing.
* Tests for API endpoints are located in `tests/test_main.py`.
* Tests for service layer functionalities (prediction logic, data loading...) are in `tests/test_services.py`.
* PYTHONPATH=. pytest tests/ to run the tests


