#!/usr/bin/env python3
"""
Utility functions and classes for the EdgeVision-Guard inference service.

This module provides helper classes for processing images, videos,
and managing WebSocket connections.
"""

import io
import json
import logging
import queue
import threading
import time
from collections import deque
from typing import Dict, List, Optional, Tuple, Union

import cv2
import mediapipe as mp
import numpy as np
import onnxruntime as ort
from fastapi import WebSocket
from pydantic import BaseModel, Field

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class PredictionResult(BaseModel):
    """Model for prediction results."""
    
    prediction: int = Field(..., description="Predicted class index")
    class_name: str = Field(..., description="Predicted class name")
    confidence: float = Field(..., description="Confidence score")
    timestamp: float = Field(..., description="Timestamp of prediction")
    anomaly_score: float = Field(..., description="Anomaly score")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    keypoints: Optional[List[List[float]]] = Field(None, description="Detected keypoints")
    frame_id: Optional[int] = Field(None, description="Frame ID")


class Connection:
    """Class representing a WebSocket connection."""
    
    def __init__(self, websocket: WebSocket):
        """
        Initialize the connection.
        
        Args:
            websocket: WebSocket connection
        """
        self.websocket = websocket
        self.client_id = id(websocket)
        self.connected_at = time.time()
        self.last_activity = time.time()
        self.metadata = {}
    
    def update_activity(self):
        """Update the last activity timestamp."""
        self.last_activity = time.time()
    
    def idle_time(self) -> float:
        """
        Calculate idle time.
        
        Returns:
            Idle time in seconds
        """
        return time.time() - self.last_activity
    
    def connection_time(self) -> float:
        """
        Calculate total connection time.
        
        Returns:
            Connection time in seconds
        """
        return time.time() - self.connected_at


class ConnectionManager:
    """Manager for WebSocket connections."""
    
    def __init__(self):
        """Initialize the connection manager."""
        self.active_connections: List[Connection] = []
    
    async def connect(self, websocket: WebSocket):
        """
        Connect a new WebSocket.
        
        Args:
            websocket: WebSocket connection
        """
        await websocket.accept()
        logger.info(f"New WebSocket connection: {id(websocket)}")
    
    def disconnect(self, websocket: WebSocket):
        """
        Disconnect a WebSocket.
        
        Args:
            websocket: WebSocket connection
        """
        # Find and remove the connection
        for i, connection in enumerate(self.active_connections):
            if connection.websocket == websocket:
                self.active_connections.pop(i)
                logger.info(f"WebSocket disconnected: {connection.client_id}")
                break
    
    async def broadcast(self, message: str):
        """
        Broadcast a message to all connections.
        
        Args:
            message: Message to broadcast
        """
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.websocket.send_text(message)
                connection.update_activity()
            except Exception as e:
                logger.error(f"Error broadcasting to {connection.client_id}: {e}")
                disconnected.append(connection)
        
        # Clean up disconnected connections
        for connection in disconnected:
            self.active_connections.remove(connection)
    
    async def broadcast_excluding(self, message: str, exclude_websocket: WebSocket):
        """
        Broadcast a message to all connections except one.
        
        Args:
            message: Message to broadcast
            exclude_websocket: WebSocket to exclude
        """
        disconnected = []
        for connection in self.active_connections:
            if connection.websocket != exclude_websocket:
                try:
                    await connection.websocket.send_text(message)
                    connection.update_activity()
                except Exception as e:
                    logger.error(f"Error broadcasting to {connection.client_id}: {e}")
                    disconnected.append(connection)
        
        # Clean up disconnected connections
        for connection in disconnected:
            self.active_connections.remove(connection)
    
    def get_connection_stats(self) -> Dict:
        """
        Get statistics about current connections.
        
        Returns:
            Dictionary with connection statistics
        """
        return {
            "active_connections": len(self.active_connections),
            "longest_connection": max([c.connection_time() for c in self.active_connections], default=0),
            "idle_connections": sum(1 for c in self.active_connections if c.idle_time() > 60),
        }


class OnnxInferenceSession:
    """Wrapper for ONNX Runtime inference session."""
    
    def __init__(
        self,
        model_path: str,
        providers: Optional[List[str]] = None,
        class_names: List[str] = ["No Fall", "Fall", "Other"],
    ):
        """
        Initialize the ONNX inference session.
        
        Args:
            model_path: Path to the ONNX model
            providers: ONNX Runtime providers
            class_names: Names of the output classes
        """
        self.model_path = model_path
        self.providers = providers or ["CPUExecutionProvider"]
        self.class_names = class_names
        
        # Create session
        logger.info(f"Creating ONNX Runtime session with providers: {self.providers}")
        self.session = ort.InferenceSession(model_path, providers=self.providers)
        
        # Get input and output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        logger.info(f"Model input name: {self.input_name}")
        logger.info(f"Model output name: {self.output_name}")
    
    def run(self, input_data: np.ndarray) -> np.ndarray:
        """
        Run inference.
        
        Args:
            input_data: Input data
        
        Returns:
            Model output
        """
        # Run inference
        outputs = self.session.run([self.output_name], {self.input_name: input_data})
        
        # Process outputs
        output = outputs[0][0]  # First output, first batch item
        
        # Apply softmax
        exp_output = np.exp(output - np.max(output))
        probabilities = exp_output / exp_output.sum()
        
        return probabilities
    
    def get_class_names(self) -> List[str]:
        """
        Get the class names.
        
        Returns:
            List of class names
        """
        return self.class_names


class ImageProcessor:
    """Processor for images and keypoint extraction."""
    
    def __init__(self, min_detection_confidence: float = 0.5):
        """
        Initialize the image processor.
        
        Args:
            min_detection_confidence: Minimum detection confidence for MediaPipe
        """
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            min_detection_confidence=min_detection_confidence,
        )
    
    def extract_keypoints(self, image: np.ndarray) -> np.ndarray:
        """
        Extract keypoints from an image.
        
        Args:
            image: Input image
        
        Returns:
            Keypoints as a numpy array
        """
        # Convert to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image
        results = self.pose.process(image_rgb)
        
        # Extract keypoints
        keypoints = np.zeros((33, 3))  # 33 keypoints, (x, y, visibility)
        
        if results.pose_landmarks:
            for i, landmark in enumerate(results.pose_landmarks.landmark):
                keypoints[i] = [landmark.x, landmark.y, landmark.visibility]
        
        # Flatten keypoints for the model (using only 17 keypoints for upper body)
        flat_keypoints = keypoints[:17, :].flatten()
        
        return flat_keypoints
    
    def draw_keypoints(self, image: np.ndarray, keypoints: np.ndarray) -> np.ndarray:
        """
        Draw keypoints on an image.
        
        Args:
            image: Input image
            keypoints: Keypoints to draw
        
        Returns:
            Image with keypoints drawn
        """
        # Create a copy of the image
        image_copy = image.copy()
        
        # Reshape keypoints to original shape
        keypoints_reshaped = keypoints.reshape(-1, 3)
        
        # Get image dimensions
        h, w, _ = image_copy.shape
        
        # Draw keypoints
        for i, (x, y, v) in enumerate(keypoints_reshaped):
            # Skip if visibility is low
            if v < 0.5:
                continue
            
            # Convert normalized coordinates to image coordinates
            x_px = int(x * w)
            y_px = int(y * h)
            
            # Draw keypoint
            cv2.circle(image_copy, (x_px, y_px), 5, (0, 255, 0), -1)
            
            # Draw keypoint number
            cv2.putText(
                image_copy,
                str(i),
                (x_px + 5, y_px - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )
        
        # Draw connections
        connections = self.mp_pose.POSE_CONNECTIONS
        for connection in connections:
            start_idx = connection[0]
            end_idx = connection[1]
            
            # Skip if indices are out of range
            if start_idx >= len(keypoints_reshaped) or end_idx >= len(keypoints_reshaped):
                continue
            
            # Skip if visibility is low
            if keypoints_reshaped[start_idx][2] < 0.5 or keypoints_reshaped[end_idx][2] < 0.5:
                continue
            
            # Convert normalized coordinates to image coordinates
            start_x = int(keypoints_reshaped[start_idx][0] * w)
            start_y = int(keypoints_reshaped[start_idx][1] * h)
            end_x = int(keypoints_reshaped[end_idx][0] * w)
            end_y = int(keypoints_reshaped[end_idx][1] * h)
            
            # Draw line
            cv2.line(
                image_copy,
                (start_x, start_y),
                (end_x, end_y),
                (0, 255, 255),
                2,
            )
        
        return image_copy
    
    def __del__(self):
        """Clean up resources."""
        if hasattr(self, "pose"):
            self.pose.close()


class VideoProcessor:
    """Processor for video streams."""
    
    def __init__(
        self,
        inference_session: OnnxInferenceSession,
        sequence_length: int = 30,
        stride: int = 15,
        confidence_threshold: float = 0.7,
    ):
        """
        Initialize the video processor.
        
        Args:
            inference_session: ONNX inference session
            sequence_length: Length of frame sequences
            stride: Stride between sequences
            confidence_threshold: Confidence threshold for anomaly detection
        """
        self.inference_session = inference_session
        self.sequence_length = sequence_length
        self.stride = stride
        self.confidence_threshold = confidence_threshold
        
        # Initialize image processor
        self.image_processor = ImageProcessor()
        
        # Buffer for keypoints
        self.keypoints_buffer = deque(maxlen=sequence_length * 2)
        
        # Frame counter
        self.frame_counter = 0
    
    def extract_frames(self, data: bytes) -> List[np.ndarray]:
        """
        Extract frames from binary data.
        
        Args:
            data: Binary image data
        
        Returns:
            List of extracted frames
        """
        frames = []
        
        try:
            # Try to decode as a single image
            frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if frame is not None:
                frames.append(frame)
        except Exception as e:
            logger.error(f"Error extracting frames: {e}")
        
        return frames
    
    def process_frame(self, frame: np.ndarray) -> Optional[PredictionResult]:
        """
        Process a video frame.
        
        Args:
            frame: Input frame
        
        Returns:
            Prediction result if available
        """
        start_time = time.time()
        
        try:
            # Extract keypoints
            keypoints = self.image_processor.extract_keypoints(frame)
            
            # Add to buffer
            self.keypoints_buffer.append(keypoints)
            
            # Increment frame counter
            self.frame_counter += 1
            
            # Only process when we have enough frames
            if len(self.keypoints_buffer) >= self.sequence_length and self.frame_counter % self.stride == 0:
                # Take the last sequence_length keypoints
                keypoints_seq = list(self.keypoints_buffer)[-self.sequence_length:]
                keypoints_arr = np.array(keypoints_seq).reshape(1, self.sequence_length, -1).astype(np.float32)
                
                # Run inference
                outputs = self.inference_session.run(keypoints_arr)
                
                # Process outputs
                prediction_idx = int(np.argmax(outputs))
                confidence = float(outputs[prediction_idx])
                class_name = self.inference_session.get_class_names()[prediction_idx]
                
                # Calculate anomaly score (probability of fall)
                anomaly_score = float(outputs[1]) if len(outputs) > 1 else 0.0
                
                # Calculate processing time
                processing_time = (time.time() - start_time) * 1000
                
                # Return prediction result
                return PredictionResult(
                    prediction=prediction_idx,
                    class_name=class_name,
                    confidence=confidence,
                    timestamp=time.time(),
                    anomaly_score=anomaly_score,
                    processing_time_ms=processing_time,
                    keypoints=keypoints.reshape(-1, 3).tolist(),
                    frame_id=self.frame_counter,
                )
        
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
        
        return None


def load_keypoints_from_image(image_path: str) -> np.ndarray:
    """
    Load keypoints from an image.
    
    Args:
        image_path: Path to the image
    
    Returns:
        Keypoints as a numpy array
    """
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Extract keypoints
    processor = ImageProcessor()
    keypoints = processor.extract_keypoints(image)
    
    return keypoints


if __name__ == "__main__":
    # Example usage
    image_processor = ImageProcessor()
    
    # Load an example image
    image_path = "example.jpg"
    try:
        image = cv2.imread(image_path)
        if image is not None:
            keypoints = image_processor.extract_keypoints(image)
            print(f"Extracted keypoints shape: {keypoints.shape}")
            
            # Draw keypoints on image
            image_with_keypoints = image_processor.draw_keypoints(image, keypoints)
            
            # Save the image
            cv2.imwrite("example_keypoints.jpg", image_with_keypoints)
            print("Saved image with keypoints")
    except Exception as e:
        print(f"Error processing image: {e}")