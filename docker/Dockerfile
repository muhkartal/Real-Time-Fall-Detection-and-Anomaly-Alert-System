# Base image for x86_64 architecture
FROM python:3.11-slim as base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on

# Install Python dependencies separately (for better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories for data and models
RUN mkdir -p data/up-fall data/ur-fall models logs

# Install ONNX optimizations
RUN python -m pip install --no-cache-dir onnxruntime-extensions

# Create a non-root user
RUN useradd -m -u 1000 appuser
RUN chown -R appuser:appuser /app
USER appuser

# Default command - will be overridden by docker-compose
CMD ["bash"]

# Inference Service image
FROM base as inference-service
CMD ["uvicorn", "src.inference_service.app:app", "--host", "0.0.0.0", "--port", "8000"]

# Dashboard image
FROM base as dashboard
CMD ["streamlit", "run", "src/dashboard/app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]

# Training image
FROM base as training
CMD ["python", "src/train.py"]