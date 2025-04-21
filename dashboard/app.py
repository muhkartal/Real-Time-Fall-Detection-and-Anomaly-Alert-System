#!/usr/bin/env python3
"""
Streamlit dashboard for EdgeVision-Guard.

This module provides a real-time dashboard for visualizing fall detection
results and anomaly scores.
"""

import json
import logging
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from dotenv import load_dotenv
from PIL import Image

from .explainer import GradCAMVisualizer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Constants
INFERENCE_SERVICE_URL = os.getenv("INFERENCE_SERVICE_URL", "http://localhost:8000")
WEBSOCKET_URL = os.getenv("WEBSOCKET_URL", "ws://localhost:8000/ws")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.7))
MODEL_PATH = os.getenv("MODEL_PATH", "./models/fall_detector.onnx")
UPDATE_INTERVAL = 0.1  # seconds
MAX_HISTORY = 100
CLASS_NAMES = ["No Fall", "Fall", "Other"]


# Initialize session state
def init_session_state():
    """Initialize Streamlit session state."""
    if "predictions" not in st.session_state:
        st.session_state.predictions = []
    
    if "latest_frame" not in st.session_state:
        st.session_state.latest_frame = None
    
    if "latest_keypoints" not in st.session_state:
        st.session_state.latest_keypoints = None
    
    if "anomaly_history" not in st.session_state:
        st.session_state.anomaly_history = []
    
    if "alert_log" not in st.session_state:
        st.session_state.alert_log = []
    
    if "connection_status" not in st.session_state:
        st.session_state.connection_status = "Disconnected"
    
    if "start_time" not in st.session_state:
        st.session_state.start_time = time.time()
    
    if "frame_count" not in st.session_state:
        st.session_state.frame_count = 0
    
    if "fps" not in st.session_state:
        st.session_state.fps = 0.0
    
    if "webcam_active" not in st.session_state:
        st.session_state.webcam_active = False


# WebSocket Connection
def connect_websocket():
    """Connect to the WebSocket server."""
    try:
        import websocket
        import threading
        
        def on_message(ws, message):
            try:
                data = json.loads(message)
                process_prediction(data)
            except Exception as e:
                logger.error(f"Error processing message: {e}")
        
        def on_error(ws, error):
            logger.error(f"WebSocket error: {error}")
            st.session_state.connection_status = f"Error: {error}"
        
        def on_close(ws, close_status_code, close_msg):
            logger.info("WebSocket connection closed")
            st.session_state.connection_status = "Disconnected"
        
        def on_open(ws):
            logger.info("WebSocket connection opened")
            st.session_state.connection_status = "Connected"
        
        # Create WebSocket connection
        ws = websocket.WebSocketApp(
            WEBSOCKET_URL,
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
        )
        
        # Start WebSocket connection in a thread
        def run():
            ws.run_forever()
        
        threading.Thread(target=run, daemon=True).start()
        
        return ws
    except Exception as e:
        logger.error(f"Error connecting to WebSocket: {e}")
        st.session_state.connection_status = f"Connection Error: {e}"
        return None


# Process prediction data
def process_prediction(data: Dict):
    """
    Process prediction data from WebSocket.
    
    Args:
        data: Prediction data
    """
    # Add to predictions list
    st.session_state.predictions.append(data)
    if len(st.session_state.predictions) > MAX_HISTORY:
        st.session_state.predictions.pop(0)
    
    # Update anomaly history
    timestamp = data.get("timestamp", time.time())
    anomaly_score = data.get("anomaly_score", 0.0)
    st.session_state.anomaly_history.append({"timestamp": timestamp, "score": anomaly_score})
    if len(st.session_state.anomaly_history) > MAX_HISTORY:
        st.session_state.anomaly_history.pop(0)
    
    # Check if it's an anomaly
    if anomaly_score > CONFIDENCE_THRESHOLD:
        # Add to alert log
        alert = {
            "timestamp": datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S"),
            "score": anomaly_score,
            "class": data.get("class_name", "Unknown"),
        }
        st.session_state.alert_log.append(alert)
        if len(st.session_state.alert_log) > 100:
            st.session_state.alert_log.pop(0)
    
    # Update keypoints if available
    if "keypoints" in data:
        st.session_state.latest_keypoints = data["keypoints"]
    
    # Update FPS calculation
    st.session_state.frame_count += 1
    elapsed = time.time() - st.session_state.start_time
    if elapsed > 0:
        st.session_state.fps = st.session_state.frame_count / elapsed


# Draw keypoints on image
def draw_keypoints(image: np.ndarray, keypoints: List[List[float]]) -> np.ndarray:
    """
    Draw keypoints on an image.
    
    Args:
        image: Input image
        keypoints: List of keypoints [x, y, confidence]
    
    Returns:
        Image with keypoints
    """
    if image is None or keypoints is None:
        return image
    
    # Create a copy of the image
    result = image.copy()
    
    # Get image dimensions
    h, w, _ = result.shape
    
    # Draw keypoints
    for i, (x, y, conf) in enumerate(keypoints):
        if conf < 0.5:
            continue
        
        # Convert normalized coordinates to image coordinates
        x_px = int(x * w)
        y_px = int(y * h)
        
        # Draw keypoint
        cv2.circle(result, (x_px, y_px), 5, (0, 255, 0), -1)
    
    # Draw connections (simplified skeleton)
    # Define connections as pairs of keypoint indices
    connections = [
        (0, 1),  # nose to left eye
        (0, 2),  # nose to right eye
        (1, 3),  # left eye to left ear
        (2, 4),  # right eye to right ear
        (5, 6),  # left shoulder to right shoulder
        (5, 7),  # left shoulder to left elbow
        (6, 8),  # right shoulder to right elbow
        (7, 9),  # left elbow to left wrist
        (8, 10),  # right elbow to right wrist
        (5, 11),  # left shoulder to left hip
        (6, 12),  # right shoulder to right hip
        (11, 12),  # left hip to right hip
        (11, 13),  # left hip to left knee
        (12, 14),  # right hip to right knee
        (13, 15),  # left knee to left ankle
        (14, 16),  # right knee to right ankle
    ]
    
    for connection in connections:
        start_idx, end_idx = connection
        
        # Skip if index is out of range
        if start_idx >= len(keypoints) or end_idx >= len(keypoints):
            continue
        
        # Skip if keypoint confidence is too low
        if keypoints[start_idx][2] < 0.5 or keypoints[end_idx][2] < 0.5:
            continue
        
        # Convert normalized coordinates to image coordinates
        start_x = int(keypoints[start_idx][0] * w)
        start_y = int(keypoints[start_idx][1] * h)
        end_x = int(keypoints[end_idx][0] * w)
        end_y = int(keypoints[end_idx][1] * h)
        
        # Draw connection
        cv2.line(result, (start_x, start_y), (end_x, end_y), (0, 255, 255), 2)
    
    return result


# Health check
def check_service_health() -> Dict:
    """
    Check the health of the inference service.
    
    Returns:
        Health status
    """
    try:
        response = requests.get(f"{INFERENCE_SERVICE_URL}/health", timeout=2)
        return response.json()
    except Exception as e:
        logger.error(f"Error checking service health: {e}")
        return {"status": "error", "message": str(e)}


# Upload and process image
def process_uploaded_image(image_file) -> Dict:
    """
    Process an uploaded image.
    
    Args:
        image_file: Uploaded image file
    
    Returns:
        Prediction result
    """
    try:
        # Read image
        contents = image_file.read()
        
        # Send to inference service
        response = requests.post(
            f"{INFERENCE_SERVICE_URL}/predict",
            files={"file": contents},
            timeout=10,
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Error from inference service: {response.text}")
            return {"error": response.text}
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        return {"error": str(e)}


# Create anomaly score chart
def create_anomaly_chart(history: List[Dict]) -> go.Figure:
    """
    Create an anomaly score chart.
    
    Args:
        history: List of anomaly scores with timestamps
    
    Returns:
        Plotly figure
    """
    if not history:
        # Create an empty chart
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=[],
                y=[],
                mode="lines",
                name="Anomaly Score",
            )
        )
        fig.add_shape(
            type="line",
            x0=0,
            y0=CONFIDENCE_THRESHOLD,
            x1=1,
            y1=CONFIDENCE_THRESHOLD,
            line=dict(color="red", width=2, dash="dash"),
        )
    else:
        # Create a DataFrame from history
        df = pd.DataFrame(history)
        
        # Create the chart
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["score"],
                mode="lines",
                name="Anomaly Score",
                line=dict(color="blue", width=2),
            )
        )
        
        # Add threshold line
        fig.add_shape(
            type="line",
            x0=min(df["timestamp"]),
            y0=CONFIDENCE_THRESHOLD,
            x1=max(df["timestamp"]),
            y1=CONFIDENCE_THRESHOLD,
            line=dict(color="red", width=2, dash="dash"),
        )
    
    # Configure layout
    fig.update_layout(
        title="Anomaly Score Over Time",
        xaxis_title="Time",
        yaxis_title="Anomaly Score",
        yaxis=dict(range=[0, 1.05]),
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    
    return fig


# Main dashboard
def main():
    """Main dashboard function."""
    # Set page title and icon
    st.set_page_config(
        page_title="EdgeVision-Guard Dashboard",
        page_icon="🛡️",
        layout="wide",
    )
    
    # Initialize session state
    init_session_state()
    
    # Header
    st.title("🛡️ EdgeVision-Guard Dashboard")
    st.markdown(
        """
        Real-time fall detection and anomaly alert system using computer vision.
        """
    )
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        
        # Service status
        st.subheader("Service Status")
        health = check_service_health()
        if health.get("status") == "ok":
            st.success(f"Service: Online | Model: {health.get('model', 'Unknown')}")
        else:
            st.error(f"Service: Offline | Error: {health.get('message', 'Unknown')}")
        
        # WebSocket status
        st.metric("WebSocket", st.session_state.connection_status)
        
        # Connect button
        if st.button("Connect WebSocket"):
            connect_websocket()
        
        # FPS counter
        st.metric("FPS", f"{st.session_state.fps:.1f}")
        
        # Threshold slider
        threshold = st.slider(
            "Anomaly Threshold",
            min_value=0.0,
            max_value=1.0,
            value=CONFIDENCE_THRESHOLD,
            step=0.05,
        )
        
        # Upload image
        st.subheader("Test with Image")
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            result = process_uploaded_image(uploaded_file)
            if "error" not in result:
                st.json(result)
            else:
                st.error(result["error"])
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Video feed with keypoints
        st.subheader("Video Feed")
        video_placeholder = st.empty()
        
        # Initialize webcam
        if st.button("Toggle Webcam"):
            st.session_state.webcam_active = not st.session_state.webcam_active
            
            if st.session_state.webcam_active:
                # Reset FPS counter
                st.session_state.start_time = time.time()
                st.session_state.frame_count = 0
        
        # Anomaly score chart
        st.subheader("Anomaly Score")
        chart_placeholder = st.empty()
    
    with col2:
        # Latest prediction
        st.subheader("Latest Prediction")
        prediction_placeholder = st.empty()
        
        # Alert log
        st.subheader("Alert Log")
        alert_placeholder = st.empty()
        
        # Grad-CAM visualization
        st.subheader("Explanation (Grad-CAM)")
        gradcam_placeholder = st.empty()
    
    # Initialize webcam (if activated)
    if st.session_state.webcam_active:
        cap = cv2.VideoCapture(0)
        
        # Check if webcam is opened successfully
        if not cap.isOpened():
            st.error("Error: Could not open webcam")
            st.session_state.webcam_active = False
    
    # Main loop
    try:
        while True:
            # Update chart
            chart = create_anomaly_chart(st.session_state.anomaly_history)
            chart_placeholder.plotly_chart(chart, use_container_width=True)
            
            # Update alert log
            if st.session_state.alert_log:
                alert_df = pd.DataFrame(st.session_state.alert_log)
                alert_placeholder.dataframe(
                    alert_df[["timestamp", "class", "score"]].sort_values(
                        by="timestamp", ascending=False
                    ),
                    hide_index=True,
                )
            else:
                alert_placeholder.info("No alerts detected yet")
            
            # Update latest prediction
            if st.session_state.predictions:
                latest = st.session_state.predictions[-1]
                prediction_placeholder.json(latest)
            else:
                prediction_placeholder.info("No predictions yet")
            
            # Process webcam frame
            if st.session_state.webcam_active:
                # Capture frame
                ret, frame = cap.read()
                if ret:
                    # Update frame count
                    st.session_state.frame_count += 1
                    
                    # Calculate FPS
                    elapsed = time.time() - st.session_state.start_time
                    if elapsed > 0:
                        st.session_state.fps = st.session_state.frame_count / elapsed
                    
                    # Draw keypoints if available
                    if st.session_state.latest_keypoints:
                        frame = draw_keypoints(frame, st.session_state.latest_keypoints)
                    
                    # Convert BGR to RGB for Streamlit
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    video_placeholder.image(frame_rgb, caption="Live Feed", use_column_width=True)
                    
                    # Store latest frame
                    st.session_state.latest_frame = frame
                    
                    # Process frame with inference service
                    # (This would typically be done via WebSocket,
                    # but we're showing how it could be done here)
                    try:
                        # Encode frame as JPEG
                        _, buffer = cv2.imencode(".jpg", frame)
                        
                        # Send to inference service
                        response = requests.post(
                            f"{INFERENCE_SERVICE_URL}/predict",
                            files={"file": buffer.tobytes()},
                            timeout=1,
                        )
                        
                        if response.status_code == 200:
                            # Process prediction
                            process_prediction(response.json())
                    except Exception as e:
                        logger.error(f"Error sending frame to inference service: {e}")
            
            # Sleep to control update rate
            time.sleep(UPDATE_INTERVAL)
    
    except Exception as e:
        st.error(f"Error: {e}")
    
    finally:
        # Clean up resources
        if st.session_state.webcam_active and "cap" in locals():
            cap.release()


if __name__ == "__main__":
    main()