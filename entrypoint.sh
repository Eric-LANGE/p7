#!/usr/bin/env bash
set -euo pipefail

# Redirect mlflow tracking to /tmp
export MLFLOW_TRACKING_URI="file:///tmp/mlruns-disabled"

# Name of Conda env
CONDA_ENV_NAME="credit_risk_env"

echo "--- Entrypoint: Initializing Micromamba shell ---"
eval "$(micromamba shell hook --shell bash)"

echo "--- Entrypoint: Activating Micromamba env: ${CONDA_ENV_NAME} ---"
micromamba activate "${CONDA_ENV_NAME}"

# Verify activation
if [[ "${CONDA_DEFAULT_ENV:-}" != "${CONDA_ENV_NAME}" ]]; then
  echo "❌ Failed to activate ${CONDA_ENV_NAME}" >&2
  exit 1
fi

echo "--- Entrypoint: Setting PYTHONPATH to include /app/src ---"
export PYTHONPATH="/app/src:${PYTHONPATH:-}"
echo "PYTHONPATH=${PYTHONPATH}"

echo "--- Entrypoint: Executing: $@ ---"
exec "$@"
