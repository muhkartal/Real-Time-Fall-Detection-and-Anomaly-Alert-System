#!/usr/bin/env python3
"""
Prometheus metrics for EdgeVision-Guard.

This module provides metrics for monitoring the EdgeVision-Guard inference service.
"""

import os
import time
from typing import Dict, Optional

from dotenv import load_dotenv
from fastapi import FastAPI
from prometheus_client import Counter, Gauge, Histogram, Summary
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route

# Load environment variables
load_dotenv()

# Enable metrics
METRICS_ENABLED = os.getenv("METRICS_ENABLED", "true").lower() in ("true", "1", "yes")
METRICS_PORT = int(os.getenv("METRICS_PORT", "9090"))

# Prometheus metrics
REQUEST_COUNT = Counter(
    "edgevision_inference_requests_total",
    "Total number of inference requests",
    ["method", "endpoint", "status_code"]
)

REQUEST_LATENCY = Histogram(
    "edgevision_inference_latency_seconds", 
    "Histogram of inference request latency in seconds",
    ["method", "endpoint"],
    buckets=[0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0]
)

ANOMALY_COUNT = Counter(
    "edgevision_anomaly_detections_total",
    "Total number of anomalies detected",
    ["class"]
)

FALSE_POSITIVE_RATE = Gauge(
    "edgevision_false_positive_rate",
    "Estimated false positive rate"
)

MODEL_VERSION = Gauge(
    "edgevision_model_version",
    "Current model version in use",
    ["version", "type"]
)

IN_PROGRESS = Gauge(
    "edgevision_inference_in_progress",
    "Number of inference requests in progress"
)

EXCEPTION_COUNT = Counter(
    "edgevision_exceptions_total",
    "Total number of exceptions",
    ["type"]
)

WEBSOCKET_CONNECTIONS = Gauge(
    "edgevision_websocket_connections",
    "Number of active WebSocket connections"
)

# Set initial values
FALSE_POSITIVE_RATE.set(0.05)  # Example initial value


class PrometheusMiddleware:
    """Middleware to capture request metrics for Prometheus."""
    
    def __init__(self, app: FastAPI):
        """
        Initialize the middleware.
        
        Args:
            app: FastAPI application
        """
        self.app = app
    
    async def __call__(self, request: Request, call_next):
        """
        Process the request and record metrics.
        
        Args:
            request: Request object
            call_next: Next middleware in chain
        
        Returns:
            Response from next middleware
        """
        method = request.method
        path = request.url.path
        
        if path == "/metrics":
            # Skip metrics endpoint to avoid recursion
            return await call_next(request)
        
        start_time = time.time()
        IN_PROGRESS.inc()
        
        try:
            response = await call_next(request)
            status_code = response.status_code
            
            # Record request count and latency
            REQUEST_COUNT.labels(method=method, endpoint=path, status_code=status_code).inc()
            REQUEST_LATENCY.labels(method=method, endpoint=path).observe(time.time() - start_time)
            
            return response
        except Exception as e:
            # Record exception
            EXCEPTION_COUNT.labels(type=type(e).__name__).inc()
            raise
        finally:
            IN_PROGRESS.dec()


def set_model_version(version: str, model_type: str = "onnx") -> None:
    """
    Set the current model version.
    
    Args:
        version: Model version string
        model_type: Type of model (onnx, pytorch, etc.)
    """
    MODEL_VERSION.labels(version=version, type=model_type).set(1)


def record_anomaly_detection(class_name: str) -> None:
    """
    Record an anomaly detection.
    
    Args:
        class_name: Name of the detected class
    """
    ANOMALY_COUNT.labels(class=class_name).inc()


def update_false_positive_rate(rate: float) -> None:
    """
    Update the estimated false positive rate.
    
    Args:
        rate: False positive rate (0.0-1.0)
    """
    FALSE_POSITIVE_RATE.set(rate)


def update_websocket_connections(count: int) -> None:
    """
    Update the number of active WebSocket connections.
    
    Args:
        count: Number of connections
    """
    WEBSOCKET_CONNECTIONS.set(count)


async def metrics_endpoint(request: Request) -> Response:
    """
    Endpoint to expose Prometheus metrics.
    
    Args:
        request: Request object
    
    Returns:
        Response with Prometheus metrics
    """
    return Response(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


def create_metrics_app() -> FastAPI:
    """
    Create a FastAPI app for metrics.
    
    Returns:
        FastAPI application
    """
    routes = [
        Route("/metrics", metrics_endpoint, methods=["GET"]),
    ]
    return FastAPI(routes=routes)


def setup_metrics(app: FastAPI) -> None:
    """
    Set up metrics for the application.
    
    Args:
        app: FastAPI application
    """
    if not METRICS_ENABLED:
        return
    
    # Add middleware
    app.add_middleware(PrometheusMiddleware)
    
    # Add metrics endpoint
    @app.get("/metrics")
    async def metrics():
        return Response(
            generate_latest(),
            media_type=CONTENT_TYPE_LATEST
        )
    
    # Log that metrics are enabled
    print(f"Prometheus metrics enabled at /metrics")


if __name__ == "__main__":
    import uvicorn
    
    # Create a standalone metrics app
    metrics_app = create_metrics_app()
    
    # Run server
    uvicorn.run(metrics_app, host="0.0.0.0", port=METRICS_PORT)