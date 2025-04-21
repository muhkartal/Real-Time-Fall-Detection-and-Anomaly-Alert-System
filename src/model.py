#!/usr/bin/env python3
"""
Model architecture for EdgeVision-Guard.

This module defines the MobileNetV3-Small + Bi-LSTM architecture for fall detection.
"""

import os
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from dotenv import load_dotenv
from torchvision.models import mobilenet_v3_small

# Load environment variables
load_dotenv()

# Model configuration
NUM_CLASSES = int(os.getenv("NUM_CLASSES", 3))
HIDDEN_SIZE = int(os.getenv("HIDDEN_SIZE", 128))
NUM_LAYERS = int(os.getenv("NUM_LAYERS", 2))
SEQUENCE_LENGTH = int(os.getenv("SEQUENCE_LENGTH", 30))
INPUT_SIZE = int(os.getenv("INPUT_SIZE", 51))  # 17 keypoints * 3 (x, y, confidence)


class KeypointEmbedding(nn.Module):
    """
    Embeds keypoint data into a feature vector using a small MLP.
    
    This module processes raw keypoint data (x, y, confidence) into a feature vector
    that can be used by the MobileNetV3 backbone.
    """
    
    def __init__(
        self, 
        input_size: int = INPUT_SIZE, 
        hidden_size: int = 128, 
        output_size: int = 224*224*3
    ):
        """
        Initialize the keypoint embedding layer.
        
        Args:
            input_size: Size of input keypoint vector
            hidden_size: Size of hidden layer
            output_size: Size of output feature vector
        """
        super().__init__()
        
        self.embedding = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid(),  # Normalize to [0, 1]
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input keypoint tensor of shape [batch_size, sequence_length, input_size]
        
        Returns:
            Feature tensor of shape [batch_size, sequence_length, output_size]
        """
        batch_size, seq_len, _ = x.size()
        x = x.view(-1, INPUT_SIZE)  # Reshape to [batch_size * seq_len, input_size]
        x = self.embedding(x)
        return x.view(batch_size, seq_len, -1)  # Reshape back to [batch_size, seq_len, output_size]


class FallDetectionModel(nn.Module):
    """
    Fall detection model using MobileNetV3-Small + Bi-LSTM.
    
    This model processes a sequence of keypoint frames to detect falls and other anomalies.
    """
    
    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        hidden_size: int = HIDDEN_SIZE,
        num_layers: int = NUM_LAYERS,
        input_size: int = INPUT_SIZE,
        dropout: float = 0.5,
    ):
        """
        Initialize the fall detection model.
        
        Args:
            num_classes: Number of output classes
            hidden_size: LSTM hidden size
            num_layers: Number of LSTM layers
            input_size: Size of input keypoint vector
            dropout: Dropout probability
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Keypoint embedding
        self.keypoint_embedding = KeypointEmbedding(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=32,  # Reduced size to avoid memory issues
        )
        
        # Feature extraction with MobileNetV3-Small
        self.mobilenet = mobilenet_v3_small(pretrained=True)
        
        # Replace the first layer to accept 1-channel input instead of 3
        self.mobilenet.features[0][0] = nn.Conv2d(
            1, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
        )
        
        # Replace the classifier to output feature vectors
        self.mobilenet.classifier = nn.Sequential(
            nn.Linear(576, hidden_size),
            nn.Hardswish(),
            nn.Dropout(p=dropout, inplace=True),
        )
        
        # Bi-LSTM for sequence processing
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),  # * 2 for bidirectional
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize model weights."""
        for name, param in self.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)
    
    def forward(
        self, 
        x: torch.Tensor, 
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [batch_size, sequence_length, input_size]
            hidden: Initial hidden state for LSTM
        
        Returns:
            Output tensor of shape [batch_size, num_classes]
            If return_hidden is True, also returns the final hidden state
        """
        batch_size, seq_len, _ = x.size()
        
        # Embed keypoints
        x = self.keypoint_embedding(x)
        
        # Reshape for MobileNetV3
        x = x.view(-1, 1, 8, 4)  # Reshape to [batch_size * seq_len, 1, 8, 4]
        
        # Extract features with MobileNetV3
        x = self.mobilenet(x)  # Output: [batch_size * seq_len, hidden_size]
        
        # Reshape for LSTM
        x = x.view(batch_size, seq_len, -1)  # [batch_size, seq_len, hidden_size]
        
        # Process sequence with LSTM
        x, hidden = self.lstm(x, hidden)  # Output: [batch_size, seq_len, hidden_size*2]
        
        # Only use the last output from the sequence
        x = x[:, -1, :]  # [batch_size, hidden_size*2]
        
        # Apply classifier
        x = self.classifier(x)  # [batch_size, num_classes]
        
        return x, hidden


class GradCAM:
    """
    Grad-CAM implementation for model explainability.
    
    This class implements Gradient-weighted Class Activation Mapping for visualizing
    regions of interest in the input data.
    """
    
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        """
        Initialize Grad-CAM.
        
        Args:
            model: Model to explain
            target_layer: Target layer for Grad-CAM
        """
        self.model = model
        self.target_layer = target_layer
        self.hooks = []
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self) -> None:
        """Register forward and backward hooks."""
        
        # Forward hook
        def forward_hook(module, input, output):
            self.activations = output
        
        # Backward hook
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        # Register hooks
        forward_handle = self.target_layer.register_forward_hook(forward_hook)
        backward_handle = self.target_layer.register_backward_hook(backward_hook)
        
        self.hooks = [forward_handle, backward_handle]
    
    def remove_hooks(self) -> None:
        """Remove hooks."""
        for hook in self.hooks:
            hook.remove()
    
    def __call__(
        self, 
        inputs: torch.Tensor, 
        class_idx: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate Grad-CAM for the input.
        
        Args:
            inputs: Input tensor
            class_idx: Target class index (if None, uses the predicted class)
        
        Returns:
            Grad-CAM heatmap
        """
        # Forward pass
        outputs, _ = self.model(inputs)
        
        # Get class index if not provided
        if class_idx is None:
            class_idx = outputs.argmax(dim=1)
        
        # Backward pass
        outputs[:, class_idx].backward()
        
        # Compute weights
        gradients = self.gradients.mean(dim=(2, 3), keepdim=True)
        
        # Compute weighted activations
        heatmap = torch.sum(gradients * self.activations, dim=1)
        
        # Apply ReLU
        heatmap = F.relu(heatmap)
        
        # Normalize
        heatmap = F.interpolate(
            heatmap.unsqueeze(0).unsqueeze(0),
            size=inputs.size(2, 3),
            mode="bilinear",
            align_corners=False,
        ).squeeze()
        
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        return heatmap


# Function to create the model
def create_model(
    num_classes: int = NUM_CLASSES,
    hidden_size: int = HIDDEN_SIZE,
    num_layers: int = NUM_LAYERS,
    input_size: int = INPUT_SIZE,
    dropout: float = 0.5,
) -> FallDetectionModel:
    """
    Create a new model instance.
    
    Args:
        num_classes: Number of output classes
        hidden_size: LSTM hidden size
        num_layers: Number of LSTM layers
        input_size: Size of input keypoint vector
        dropout: Dropout probability
    
    Returns:
        Initialized model
    """
    return FallDetectionModel(
        num_classes=num_classes,
        hidden_size=hidden_size,
        num_layers=num_layers,
        input_size=input_size,
        dropout=dropout,
    )


# Load a trained model from disk
def load_model(model_path: str, device: str = "cpu") -> FallDetectionModel:
    """
    Load a trained model from disk.
    
    Args:
        model_path: Path to the model file
        device: Device to load the model on ('cpu' or 'cuda')
    
    Returns:
        Loaded model
    """
    # Create model
    model = create_model()
    
    # Load state dict
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    
    # Move to device
    model = model.to(device)
    
    # Set to evaluation mode
    model.eval()
    
    return model


if __name__ == "__main__":
    # Example usage
    model = create_model()
    print(model)
    
    # Print model size
    params = sum(p.numel() for p in model.parameters())
    print(f"Model size: {params:,} parameters")
    
    # Example input
    batch_size = 2
    seq_len = SEQUENCE_LENGTH
    x = torch.rand(batch_size, seq_len, INPUT_SIZE)
    
    # Forward pass
    with torch.no_grad():
        outputs, _ = model(x)
        print(f"Output shape: {outputs.shape}")