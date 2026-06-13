# Stage 1: Dependency builder stage
FROM python:3.10-slim as builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install dependencies, prioritizing CPU-only torch to minimize container footprints
RUN pip install --no-cache-dir --user -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Stage 2: Runtime runner stage
FROM python:3.10-slim as runner

WORKDIR /app

# Install system shared dependencies required by OpenCV image operations
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed libraries from builder stage
COPY --from=builder /root/.local /root/.local
COPY . .

ENV PATH=/root/.local/bin:$PATH
ENV PORT=5000
ENV MODEL_NAME=efficientnet_b0
ENV MODEL_WEIGHTS_PATH=/app/plant_disease_model_latest.pt

EXPOSE 5000

# Start Gunicorn server binding to configured PORT
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "src.app.main:create_app()"]
