# Multi-Service ML Stack for Hugging Face Spaces
# Base: Python 3.10 slim for efficient container size
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    APP_HOME=/app \
    MLFLOW_HOME=/mlflow \
    LOGS_HOME=/logs \
    DATA_HOME=/data \
    USER_ID=1000

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    supervisor \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Create application directories with proper permissions for UID 1000
# Create user with UID 1000 (required by Hugging Face Spaces)
RUN useradd -m -u ${USER_ID} -s /bin/bash appuser && \
    mkdir -p ${APP_HOME} ${MLFLOW_HOME} ${LOGS_HOME} ${DATA_HOME} \
    /var/log/supervisor \
    && chown -R ${USER_ID}:${USER_ID} ${APP_HOME} ${MLFLOW_HOME} ${LOGS_HOME} ${DATA_HOME} /var/log/supervisor

# Set working directory
WORKDIR ${APP_HOME}

# Copy and install Python dependencies first (for layer caching)
COPY --chown=${USER_ID}:${USER_ID} requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Install additional MLOps dependencies
RUN pip install --no-cache-dir \
    supervisor \
    prometheus-client \
    mlflow \
    fastapi \
    uvicorn[standard] \
    streamlit \
    aiofiles \
    httpx

# Copy application source code
COPY --chown=${USER_ID}:${USER_ID} . .

# Copy supervisor configuration
COPY --chown=${USER_ID}:${USER_ID} supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Copy startup script
COPY --chown=${USER_ID}:${USER_ID} start.sh ${APP_HOME}/start.sh
RUN chmod +x ${APP_HOME}/start.sh

# Create Streamlit config directory
RUN mkdir -p /home/user/.streamlit && chown -R ${USER_ID}:${USER_ID} /home/user

# Copy Streamlit configuration
COPY --chown=${USER_ID}:${USER_ID} .streamlit/config.toml /home/user/.streamlit/config.toml

# Switch to non-root user (required by Hugging Face Spaces)
USER ${USER_ID}

# Expose the primary port (Streamlit UI)
EXPOSE 7860

# Health check for the main Streamlit service
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/_stcore/health || exit 1

# Entrypoint: Start all services via supervisor
ENTRYPOINT ["./start.sh"]
