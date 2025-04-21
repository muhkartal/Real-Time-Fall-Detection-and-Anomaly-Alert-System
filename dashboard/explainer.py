#!/usr/bin/env python3
"""
Explainability module for EdgeVision-Guard dashboard.

This module provides utilities for visualizing model explanations 
using Gradient-weighted Class Activation Mapping (Grad-CAM).
"""

import logging
import os
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import onnxruntime as ort
from dotenv import load_dotenv

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


class GradCAMVisualizer:
    """
    Grad-CAM visualization for ONNX models.
    
    This class uses Grad-CAM to visualize the regions of the input that 
    contribute most to the model's prediction.
    """
    
    def __init__(
        self,
        model_path: str,
        target_layer: Optional[str] = None,
        providers: Optional[List[str]] = None,
    ):
        """
        Initialize the Grad-CAM visualizer.
        
        Args:
            model_path: Path to the ONNX model
            target_layer: Name of the target layer for Grad-CAM
            providers: ONNX Runtime providers
        """
        self.model_path = model_path
        self.target_layer = target_layer or "mobilenet.features.8"  # Default target layer
        self.providers = providers or ["CPUExecutionProvider"]
        
        # Load ONNX model
        self._load_model()
    
    def _load_model(self):
        """Load the ONNX model."""
        try:
            logger.info(f"Loading ONNX model from {self.model_path}")
            self.session = ort.InferenceSession(self.model_path, providers=self.providers)
            logger.info("ONNX model loaded successfully")
            
            # Get input and output names
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            logger.info(f"Model input name: {self.input_name}")
            logger.info(f"Model output name: {self.output_name}")
        except Exception as e:
            logger.error(f"Error loading ONNX model: {e}")
            raise
    
    def _preprocess_keypoints(
        self, keypoints: np.ndarray, sequence_length: int = 30
    ) -> np.ndarray:
        """
        Preprocess keypoints for the model.
        
        Args:
            keypoints: Keypoints array
            sequence_length: Length of the input sequence
        
        Returns:
            Preprocessed keypoints
        """
        # Ensure keypoints are in the correct shape
        if keypoints.ndim == 2:  # Single frame of keypoints
            # Repeat keypoints to create a sequence
            keypoints_seq = np.tile(keypoints, (sequence_length, 1))
        elif keypoints.ndim == 3:  # Already a sequence
            keypoints_seq = keypoints
        else:
            raise ValueError(f"Unexpected keypoints shape: {keypoints.shape}")
        
        # Ensure we have exactly sequence_length frames
        if keypoints_seq.shape[0] < sequence_length:
            # Pad with zeros
            pad_length = sequence_length - keypoints_seq.shape[0]
            keypoints_seq = np.pad(
                keypoints_seq,
                ((0, pad_length), (0, 0)),
                mode="constant",
                constant_values=0,
            )
        elif keypoints_seq.shape[0] > sequence_length:
            # Truncate
            keypoints_seq = keypoints_seq[:sequence_length]
        
        # Add batch dimension
        keypoints_seq = keypoints_seq.reshape(1, sequence_length, -1).astype(np.float32)
        
        return keypoints_seq
    
    def _compute_gradcam(
        self,
        preprocessed_input: np.ndarray,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """
        Compute Grad-CAM for the input.
        
        Note: This is a pseudo-implementation for ONNX models, as direct gradient
        computation is not available in ONNX Runtime. A real implementation would
        require PyTorch.
        
        Args:
            preprocessed_input: Preprocessed input data
            target_class: Target class index (if None, uses the predicted class)
        
        Returns:
            Grad-CAM heatmap
        """
        logger.warning(
            "Using pseudo-Grad-CAM for ONNX model. "
            "For accurate Grad-CAM, use PyTorch model."
        )
        
        # Run inference
        outputs = self.session.run([self.output_name], {self.input_name: preprocessed_input})
        output = outputs[0][0]
        
        # Get class index if not provided
        if target_class is None:
            target_class = np.argmax(output)
        
        # For demonstration, we'll create a simple heatmap based on the model's attention
        # In a real implementation, this would use gradients
        
        # Simulate the Grad-CAM heatmap
        # For keypoint data, we'll simply highlight parts with higher activation
        seq_len = preprocessed_input.shape[1]
        feature_dim = preprocessed_input.shape[2]
        
        # Calculate feature importance
        importance = np.abs(preprocessed_input[0])
        
        # Normalize importance
        importance = importance / np.max(importance)
        
        # Create a heatmap
        heatmap = np.mean(importance, axis=1)
        heatmap = np.maximum(heatmap, 0)
        heatmap = heatmap / np.max(heatmap)
        
        return heatmap
    
    def generate_heatmap(
        self,
        keypoints: np.ndarray,
        target_class: Optional[int] = None,
        sequence_length: int = 30,
    ) -> np.ndarray:
        """
        Generate a Grad-CAM heatmap for the keypoints.
        
        Args:
            keypoints: Keypoints array
            target_class: Target class index (if None, uses the predicted class)
            sequence_length: Length of the input sequence
        
        Returns:
            Grad-CAM heatmap
        """
        try:
            # Preprocess keypoints
            preprocessed = self._preprocess_keypoints(keypoints, sequence_length)
            
            # Compute Grad-CAM
            heatmap = self._compute_gradcam(preprocessed, target_class)
            
            return heatmap
        except Exception as e:
            logger.error(f"Error generating heatmap: {e}")
            # Return empty heatmap
            return np.zeros((1, keypoints.shape[1]))
    
    def overlay_heatmap(
        self,
        image: np.ndarray,
        keypoints: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.5,
        colormap: int = cv2.COLORMAP_JET,
    ) -> np.ndarray:
        """
        Overlay the heatmap on the image.
        
        Args:
            image: Input image
            keypoints: Keypoints array
            heatmap: Grad-CAM heatmap
            alpha: Blend factor
            colormap: OpenCV colormap
        
        Returns:
            Image with heatmap overlay
        """
        # Ensure heatmap has the right shape
        if heatmap.ndim == 1:
            heatmap = heatmap.reshape(1, -1)
        
        # Convert heatmap to uint8
        heatmap_normalized = np.uint8(255 * heatmap)
        
        # Apply colormap
        colored_heatmap = cv2.applyColorMap(heatmap_normalized, colormap)
        
        # Create a copy of the image
        result = image.copy()
        
        # Get image dimensions
        h, w, _ = result.shape
        
        # Draw keypoints with heatmap coloring
        for i, (x, y, conf) in enumerate(keypoints.reshape(-1, 3)):
            if conf < 0.5:
                continue
            
            # Convert normalized coordinates to image coordinates
            x_px = int(x * w)
            y_px = int(y * h)
            
            # Get heatmap value for this keypoint
            if i < heatmap.shape[1]:
                heat_value = heatmap[0, i]
                
                # Calculate color based on heatmap
                color = colored_heatmap[0, i].tolist()
                
                # Draw keypoint with heatmap color
                cv2.circle(result, (x_px, y_px), 8, color, -1)
                
                # Draw larger circle with radius based on importance
                radius = int(5 + 15 * heat_value)
                cv2.circle(result, (x_px, y_px), radius, color, 2)
        
        return result
    
    def visualize(
        self,
        image: np.ndarray,
        keypoints: np.ndarray,
        target_class: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Visualize Grad-CAM on the image.
        
        Args:
            image: Input image
            keypoints: Keypoints array
            target_class: Target class index (if None, uses the predicted class)
        
        Returns:
            Tuple of (image with overlay, raw heatmap)
        """
        # Generate heatmap
        heatmap = self.generate_heatmap(keypoints, target_class)
        
        # Overlay heatmap on image
        result = self.overlay_heatmap(image, keypoints, heatmap)
        
        return result, heatmap


if __name__ == "__main__":
    # Example usage
    try:
        # Initialize visualizer
        visualizer = GradCAMVisualizer(MODEL_PATH)
        
        # Load an example image
        image_path = "example.jpg"
        image = cv2.imread(image_path)
        
        if image is not None:
            # Create dummy keypoints
            keypoints = np.random.rand(17, 3)
            
            # Generate visualization
            result, heatmap = visualizer.visualize(image, keypoints)
            
            # Save result
            cv2.imwrite("example_gradcam.jpg", result)
            print("Saved Grad-CAM visualization")
            
            # Print heatmap shape
            print(f"Heatmap shape: {heatmap.shape}")
        else:
            print(f"Could not read image: {image_path}")
    except Exception as e:
        print(f"Error: {e}")