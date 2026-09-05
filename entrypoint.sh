#!/usr/bin/env sh
# Guarantees `docker-compose up` alone produces a fully working app from a
# clean clone: trains the model (which generates synthetic data if no real
# CSV is present) and derives business rules on first start, then launches
# the UI. On subsequent restarts with the same mounted ./models volume this
# is a no-op skip, so restarts are fast.
set -e

if [ ! -f "/app/models/model.joblib" ]; then
    echo "[entrypoint] No trained model found - running the training pipeline once..."
    python -m src.ml.train
    python -m src.rules.rule_engine
else
    echo "[entrypoint] Found existing model artifacts, skipping training."
fi

exec streamlit run app/streamlit_app.py \
    --server.address 0.0.0.0 \
    --server.port "${APP_PORT:-8501}" \
    --server.headless true
