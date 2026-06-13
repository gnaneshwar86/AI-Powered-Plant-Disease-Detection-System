import time
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from flask import Response

# Start time marker
START_TIME = time.time()

# 1. Total prediction counter with disease and response status label
PREDICTION_COUNT = Counter(
    "plant_disease_predictions_total",
    "Total number of leaf disease predictions made by the application.",
    ["disease_name", "status"]
)

# 2. Latency histogram for model prediction steps
PREDICTION_LATENCY = Histogram(
    "plant_disease_prediction_latency_seconds",
    "Time spent performing crop disease prediction inference.",
    buckets=[0.05, 0.1, 0.2, 0.4, 0.8, 1.5, 3.0, 5.0]
)

# 3. Uptime gauge (in seconds)
UPTIME_GAUGE = Gauge(
    "plant_disease_uptime_seconds",
    "Uptime of the plant disease prediction application in seconds."
)

# 4. Error counter tracking exception classifications
ERROR_COUNT = Counter(
    "plant_disease_prediction_errors_total",
    "Total number of errors encountered during prediction processing.",
    ["error_type"]
)

def metrics_endpoint():
    """Flask endpoint handler returning Prometheus formatted metrics."""
    # Update uptime before gathering latest metrics
    UPTIME_GAUGE.set(time.time() - START_TIME)
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)
