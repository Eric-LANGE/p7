# Dockerfile (Single-Stage Micromamba)

# 1. base image
FROM mambaorg/micromamba:latest

# 2. Set working dir
WORKDIR /app

# 3. Copy and install env
COPY credit_risk_env.yml /app/
RUN micromamba create -y --name credit_risk_env -f /app/credit_risk_env.yml && \
micromamba clean -afy

# 4. Entrypoint script
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# 5. Copy pytest config
COPY pytest.ini /app/pytest.ini

# 6. Copy application code & assets
COPY src/      /app/src/
COPY models/   /app/models/
COPY data/     /app/data/
COPY static/   /app/static/
COPY templates /app/templates/
RUN mkdir -p /app/tests
COPY tests/    /app/tests/

# 7. Expose & launch
EXPOSE 7860
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["uvicorn", "src.credit_risk_app.main:app", "--host", "0.0.0.0", "--port", "7860"]
