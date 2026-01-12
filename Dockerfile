FROM python:3.10-slim

WORKDIR / app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*


# Copy requirements
COPY requirements_api.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements_api.txt

# Copy application code 
COPY src/ ./src/
COPY models/ ./models/
COPY config/ ./config/

# Expose port 
EXPOSE 8000

# Run the application
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]