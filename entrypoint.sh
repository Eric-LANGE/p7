#!/usr/bin/env bash
set -euo pipefail

# Redirect mlflow tracking to /tmp
export MLFLOW_TRACKING_URI="file:///tmp/mlruns-disabled"

echo "--- Entrypoint: Setting PYTHONPATH to include /app/src ---"
export PYTHONPATH="/app/src:${PYTHONPATH:-}"
echo "PYTHONPATH=${PYTHONPATH}"

echo "--- Entrypoint: Executing: $@ ---"
exec "$@"
