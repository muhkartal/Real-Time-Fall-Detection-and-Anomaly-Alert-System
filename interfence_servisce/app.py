#!/usr/bin/env python3
"""
FastAPI Inference Service for EdgeVision-Guard.

This module provides a FastAPI application with REST and WebSocket endpoints
for real-time fall detection inference.
"""

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import onnxruntime as ort
from dotenv import load_dotenv
from fastapi import (BackgroundTasks, Depends, FastAPI, File, Form, HTTPException,
                  Request, UploadFile, WebSocket, WebSocketDisconnect)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from .auth import setup_auth, require_roles, get_current_user, User
from .metrics import setup_metrics
from .utils import (Connection, ConnectionManager, ImageProcessor,
                 OnnxInferenceSession, VideoProcessor, load_keypoints_from_image)
from .integrations.kafka_connector import setup_kafka_integration, get_kafka_producer
from .integrations.mqtt_connector import setup_mqtt_integration, get_mqtt_client
from .integrations.webhook_connector import setup_webhook_integration, send_prediction_webhook

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Constants
MODEL_PATH = os.getenv("MODEL_PATH", "./models/fall_detector.onnx")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.7))
SEQUENCE_LENGTH = int(os.getenv("SEQUENCE_LENGTH", 30))
INPUT_SIZE = int(os.getenv("INPUT_SIZE", 51))
NUM_CLASSES = int(os.getenv("NUM_CLASSES", 3))
USE_GPU = os.getenv("USE_GPU", "false").lower() in ("true", "1", "yes")
CLASS_NAMES = ["No Fall", "Fall", "Other"]
AUTH_ENABLED = os.getenv("AUTH_ENABLED", "false").lower() in ("true", "1", "yes")

# Initialize FastAPI app
app = FastAPI(
    title="EdgeVision-Guard Inference API",
    description="API for real-time fall detection inference",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize connection manager for WebSockets
connection_manager = ConnectionManager()

# Set the app startup time for usage in integrations
app.state.startup_time = time.time()

# Set up authentication if enabled
if AUTH_ENABLED:
    setup_auth(app)

# Set up Prometheus metrics
setup_metrics(app)

# Set up integrations
setup_kafka_integration(app)
setup_mqtt_integration(app)
setup_webhook_integration(app)


# Response models
class PredictionResponse(BaseModel):
    """Response model for prediction API."""
    
    prediction: int = Field(..., description="Predicted class index")
    class_name: str = Field(..., description="Predicted class name")
    confidence: float = Field(..., description="Confidence score")
    timestamp: float = Field(..., description="Timestamp of prediction")
    anomaly_score: float = Field(..., description="Anomaly score")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")


class HealthResponse(BaseModel):
    """Response model for health check API."""
    
    status: str = Field(..., description="Service status")
    model: str = Field(..., description="Model name")
    version: str = Field(..., description="API version")


# Dependency to get the ONNX inference session
@app.on_event("startup")
async def startup_event():
    """Initialize the ONNX inference session on startup."""
    global inference_session, image_processor, video_processor
    
    # Create ONNX inference session
    logger.info(f"Loading ONNX model: {MODEL_PATH}")
    
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model file not found: {MODEL_PATH}")
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    
    # Initialize session with appropriate provider
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if USE_GPU else ["CPUExecutionProvider"]
    inference_session = OnnxInferenceSession(MODEL_PATH, providers=providers)
    
    # Initialize processors
    image_processor = ImageProcessor()
    video_processor = VideoProcessor(
        inference_session=inference_session,
        sequence_length=SEQUENCE_LENGTH,
        confidence_threshold=CONFIDENCE_THRESHOLD,
    )
    
    logger.info("Inference service initialized successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    if "inference_session" in globals():
        del globals()["inference_session"]
    
    if "video_processor" in globals():
        del globals()["video_processor"]
    
    if "image_processor" in globals():
        del globals()["image_processor"]
    
    logger.info("Inference service shut down")


def get_inference_session() -> OnnxInferenceSession:
    """Get the ONNX inference session."""
    return inference_session


def get_image_processor() -> ImageProcessor:
    """Get the image processor."""
    return image_processor


def get_video_processor() -> VideoProcessor:
    """Get the video processor."""
    return video_processor


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        model=os.path.basename(MODEL_PATH),
        version="1.0.0",
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    inference_session: OnnxInferenceSession = Depends(get_inference_session),
    image_processor: ImageProcessor = Depends(get_image_processor),
    current_user: User = Depends(require_roles(["admin", "viewer", "api"])) if AUTH_ENABLED else None,
):
    """
    Predict from a single image.
    
    Args:
        file: Image file
        inference_session: ONNX inference session
        image_processor: Image processor
        current_user: Authenticated user (if AUTH_ENABLED)
    
    Returns:
        Prediction response
    """
    start_time = time.time()
    
    try:
        # Read image
        contents = await file.read()
        image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Extract keypoints
        keypoints = image_processor.extract_keypoints(image)
        
        # Prepare input for the model
        # Since the model expects a sequence, we repeat the keypoints
        keypoints_seq = np.tile(keypoints, (SEQUENCE_LENGTH, 1))
        keypoints_seq = keypoints_seq.reshape(1, SEQUENCE_LENGTH, INPUT_SIZE).astype(np.float32)
        
        # Run inference
        outputs = inference_session.run(keypoints_seq)
        
        # Process outputs
        prediction_idx = int(np.argmax(outputs))
        confidence = float(outputs[prediction_idx])
        class_name = CLASS_NAMES[prediction_idx]
        
        # Calculate anomaly score (probability of fall)
        anomaly_score = float(outputs[1]) if NUM_CLASSES > 1 else 0.0
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000
        
        # Create response
        result = PredictionResponse(
            prediction=prediction_idx,
            class_name=class_name,
            confidence=confidence,
            timestamp=time.time(),
            anomaly_score=anomaly_score,
            processing_time_ms=processing_time,
        )
        
        # Send to integrations if anomaly detected
        if anomaly_score >= CONFIDENCE_THRESHOLD:
            # Send to Kafka if enabled
            try:
                kafka_producer = get_kafka_producer()
                if hasattr(kafka_producer, 'send_anomaly'):
                    kafka_producer.send_anomaly(result.dict())
            except Exception as e:
                logger.error(f"Error sending to Kafka: {e}")
            
            # Send to MQTT if enabled
            try:
                mqtt_client = get_mqtt_client()
                if hasattr(mqtt_client, 'publish_anomaly'):
                    mqtt_client.publish_anomaly(result.dict())
            except Exception as e:
                logger.error(f"Error sending to MQTT: {e}")
            
            # Send to webhook if enabled
            try:
                asyncio.create_task(send_prediction_webhook(result.dict()))
            except Exception as e:
                logger.error(f"Error sending to webhook: {e}")
            
            # Update metrics
            try:
                from .metrics import record_anomaly_detection
                record_anomaly_detection(class_name)
            except Exception as e:
                logger.error(f"Error updating metrics: {e}")
        
        return result
    
    except Exception as e:
        logger.error(f"Error processing prediction: {e}")
        
        # Record exception in metrics
        try:
            from .metrics import EXCEPTION_COUNT
            EXCEPTION_COUNT.labels(type=type(e).__name__).inc()
        except:
            pass
            
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/stream")
async def predict_stream(
    background_tasks: BackgroundTasks,
    request: Request,
    video_processor: VideoProcessor = Depends(get_video_processor),
    current_user: User = Depends(require_roles(["admin", "viewer", "api"])) if AUTH_ENABLED else None,
):
    """
    Predict from a video stream.
    
    Args:
        background_tasks: FastAPI background tasks
        request: FastAPI request
        video_processor: Video processor
        current_user: Authenticated user (if AUTH_ENABLED)
    
    Returns:
        Stream of predictions
    """
    # Start processing stream
    async def generate():
        buffer = b""
        async for chunk in request.stream():
            buffer += chunk
            frames = video_processor.extract_frames(buffer)
            buffer = b""
            
            for frame in frames:
                # Process frame
                result = video_processor.process_frame(frame)
                
                if result:
                    # Convert result to JSON
                    result_json = result.json()
                    yield f"data: {result_json}\n\n"
                    
                    # Broadcast result to WebSocket clients
                    await connection_manager.broadcast(result_json)
                    
                    # Send to integrations if anomaly detected
                    result_dict = result.dict()
                    if result_dict.get("anomaly_score", 0) >= CONFIDENCE_THRESHOLD:
                        # Send to Kafka if enabled
                        try:
                            kafka_producer = get_kafka_producer()
                            if hasattr(kafka_producer, 'send_anomaly'):
                                kafka_producer.send_anomaly(result_dict)
                        except Exception as e:
                            logger.error(f"Error sending to Kafka: {e}")
                        
                        # Send to MQTT if enabled
                        try:
                            mqtt_client = get_mqtt_client()
                            if hasattr(mqtt_client, 'publish_anomaly'):
                                mqtt_client.publish_anomaly(result_dict)
                        except Exception as e:
                            logger.error(f"Error sending to MQTT: {e}")
                        
                        # Send to webhook if enabled
                        try:
                            asyncio.create_task(send_prediction_webhook(result_dict))
                        except Exception as e:
                            logger.error(f"Error sending to webhook: {e}")
                        
                        # Update metrics
                        try:
                            from .metrics import record_anomaly_detection
                            record_anomaly_detection(result_dict.get("class_name", "Unknown"))
                        except Exception as e:
                            logger.error(f"Error updating metrics: {e}")
    
    # Return streaming response
    return StreamingResponse(generate(), media_type="text/event-stream")


@app.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    video_processor: VideoProcessor = Depends(get_video_processor),
):
    """
    WebSocket endpoint for real-time predictions.
    
    Args:
        websocket: WebSocket connection
        video_processor: Video processor
    """
    # Accept connection
    await connection_manager.connect(websocket)
    
    try:
        # Add to global list of connections
        connection = Connection(websocket)
        connection_manager.active_connections.append(connection)
        
        # Update metrics
        try:
            from .metrics import update_websocket_connections
            update_websocket_connections(len(connection_manager.active_connections))
        except Exception as e:
            logger.error(f"Error updating metrics: {e}")
        
        # Send welcome message
        await websocket.send_json({"type": "connected", "message": "Connected to EdgeVision-Guard"})
        
        # Process incoming messages
        async for data in websocket.iter_bytes():
            try:
                # Process frame
                frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                
                if frame is not None:
                    # Process the frame
                    result = video_processor.process_frame(frame)
                    
                    if result:
                        # Send result back to the client
                        await websocket.send_json(result.dict())
                        
                        # Send to integrations if anomaly detected
                        result_dict = result.dict()
                        if result_dict.get("anomaly_score", 0) >= CONFIDENCE_THRESHOLD:
                            # Send to Kafka if enabled
                            try:
                                kafka_producer = get_kafka_producer()
                                if hasattr(kafka_producer, 'send_anomaly'):
                                    kafka_producer.send_anomaly(result_dict)
                            except Exception as e:
                                logger.error(f"Error sending to Kafka: {e}")
                            
                            # Send to MQTT if enabled
                            try:
                                mqtt_client = get_mqtt_client()
                                if hasattr(mqtt_client, 'publish_anomaly'):
                                    mqtt_client.publish_anomaly(result_dict)
                            except Exception as e:
                                logger.error(f"Error sending to MQTT: {e}")
                            
                            # Send to webhook if enabled
                            try:
                                asyncio.create_task(send_prediction_webhook(result_dict))
                            except Exception as e:
                                logger.error(f"Error sending to webhook: {e}")
                            
                            # Broadcast to other clients if it's an anomaly
                            await connection_manager.broadcast_excluding(
                                result.json(), websocket
                            )
            except Exception as e:
                logger.error(f"Error processing WebSocket frame: {e}")
                
                # Record exception in metrics
                try:
                    from .metrics import EXCEPTION_COUNT
                    EXCEPTION_COUNT.labels(type=type(e).__name__).inc()
                except:
                    pass
                
                await websocket.send_json({"type": "error", "message": str(e)})
    
    except WebSocketDisconnect:
        # Remove from connections on disconnect
        connection_manager.disconnect(websocket)
        logger.info("WebSocket client disconnected")
        
        # Update metrics
        try:
            from .metrics import update_websocket_connections
            update_websocket_connections(len(connection_manager.active_connections))
        except Exception as e:
            logger.error(f"Error updating metrics: {e}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        connection_manager.disconnect(websocket)


# Add system endpoints
@app.get("/system/status", response_model=Dict)
async def system_status(
    current_user: User = Depends(require_roles(["admin"])) if AUTH_ENABLED else None,
):
    """
    Get system status information.
    
    Args:
        current_user: Authenticated user (if AUTH_ENABLED)
    
    Returns:
        System status information
    """
    import platform
    import psutil
    
    # Get system information
    system_info = {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "cpu_count": psutil.cpu_count(),
        "cpu_percent": psutil.cpu_percent(interval=0.5),
        "memory": {
            "total": psutil.virtual_memory().total,
            "available": psutil.virtual_memory().available,
            "percent": psutil.virtual_memory().percent,
        },
        "disk": {
            "total": psutil.disk_usage("/").total,
            "free": psutil.disk_usage("/").free,
            "percent": psutil.disk_usage("/").percent,
        },
        "network": {
            "connections": len(psutil.net_connections(kind="inet")),
        },
        "uptime": time.time() - app.state.startup_time,
    }
    
    # Get connections information
    connections_info = {
        "websocket_connections": len(connection_manager.active_connections),
        "websocket_stats": connection_manager.get_connection_stats(),
    }
    
    # Get model information
    model_info = {
        "model_path": MODEL_PATH,
        "model_name": os.path.basename(MODEL_PATH),
        "providers": inference_session.session.get_providers(),
    }
    
    return {
        "status": "ok",
        "timestamp": time.time(),
        "system": system_info,
        "connections": connections_info,
        "model": model_info,
    }


# Add configuration endpoint
@app.post("/system/configure")
async def configure_system(
    config: Dict,
    current_user: User = Depends(require_roles(["admin"])) if AUTH_ENABLED else None,
):
    """
    Configure system parameters.
    
    Args:
        config: Configuration dictionary
        current_user: Authenticated user (if AUTH_ENABLED)
    
    Returns:
        Updated configuration
    """
    # Update thresholds
    if "confidence_threshold" in config:
        global CONFIDENCE_THRESHOLD
        CONFIDENCE_THRESHOLD = float(config["confidence_threshold"])
        
        # Update video processor
        video_processor.confidence_threshold = CONFIDENCE_THRESHOLD
    
    # Return current configuration
    return {
        "status": "ok",
        "config": {
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "model_path": MODEL_PATH,
            "use_gpu": USE_GPU,
        }
    }