# Dockerfile (Single-Stage Micromamba)

# 1. base image
FROM mambaorg/micromamba:latest

# 2. Set working dir
WORKDIR /app

# 3. Copy and install env
COPY credit_risk_env.yml /app/

# 4. Ensure base env is on PATH (so pytest, uvicorn, etc. are available)
ENV MAMBA_ROOT_PREFIX=/opt/conda \
PATH=/opt/conda/bin:$PATH

# 5. Install packages directly into the base environment
RUN micromamba install -y -n base -f /app/credit_risk_env.yml && \
micromamba clean -afy

# 6. Copy application code & assets
COPY . /app/

# 7. Set entrypoint permissions (must be done AFTER copying all files)
COPY --chmod=755 entrypoint.sh /app/entrypoint.sh

# 8. Expose & launch
EXPOSE 7860
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["uvicorn", "src.credit_risk_app.main:app", "--host", "0.0.0.0", "--port", "7860"]
