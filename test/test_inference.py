#!/usr/bin/env python3
"""
Unit tests for the inference module.

This module tests the inference functionality of the EdgeVision-Guard system.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.inference_service.utils import ImageProcessor, VideoProcessor
from src.model import create_model


class TestImageProcessor(unittest.TestCase):
    """Tests for the ImageProcessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.processor = ImageProcessor()
    
    def tearDown(self):
        """Clean up after tests."""
        del self.processor
    
    def test_initialization(self):
        """Test that the processor initializes correctly."""
        self.assertIsNotNone(self.processor.pose)
    
    @patch('mediapipe.solutions.pose.Pose')
    def test_extract_keypoints(self, mock_pose):
        """Test keypoint extraction from an image."""
        # Create a mock for pose.process
        mock_pose_instance = MagicMock()
        mock_pose.return_value = mock_pose_instance
        
        # Create a mock for the landmarks
        mock_results = MagicMock()
        mock_results.pose_landmarks = MagicMock()
        
        # Create mock landmarks
        landmarks = []
        for i in range(33):
            landmark = MagicMock()
            landmark.x = i / 100.0
            landmark.y = (i + 10) / 100.0
            landmark.visibility = 0.9
            landmarks.append(landmark)
        
        mock_results.pose_landmarks.landmark = landmarks
        mock_pose_instance.process.return_value = mock_results
        
        # Create test image
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Create processor with mock
        processor = ImageProcessor()
        
        # Extract keypoints
        keypoints = processor.extract_keypoints(test_image)
        
        # Check that the keypoints have the right shape (17 keypoints * 3 values)
        self.assertEqual(keypoints.shape, (51,))
        
        # Check that the landmark data was used
        for i in range(17):
            # Each keypoint is 3 values (x, y, visibility)
            self.assertAlmostEqual(keypoints[i*3], i / 100.0)
            self.assertAlmostEqual(keypoints[i*3 + 1], (i + 10) / 100.0)
            self.assertAlmostEqual(keypoints[i*3 + 2], 0.9)


class TestModelPrediction(unittest.TestCase):
    """Tests for model prediction."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a small model for testing
        self.model = create_model(
            num_classes=3,
            hidden_size=32,
            num_layers=1,
            input_size=51,
            dropout=0.0,
        )
        self.model.eval()
    
    def test_model_forward(self):
        """Test forward pass of the model."""
        # Create a random batch of input sequences
        batch_size = 2
        seq_length = 30
        input_size = 51
        
        x = torch.rand(batch_size, seq_length, input_size)
        
        # Run forward pass
        with torch.no_grad():
            output, _ = self.model(x)
        
        # Check output shape
        self.assertEqual(output.shape, (batch_size, 3))
        
        # Check output values are valid probabilities after softmax
        # (the model returns logits, so we need to apply softmax)
        probs = torch.softmax(output, dim=1)
        self.assertTrue(torch.all(probs >= 0.0))
        self.assertTrue(torch.all(probs <= 1.0))
        
        # Check that probabilities sum to approximately 1
        sums = torch.sum(probs, dim=1)
        self.assertTrue(torch.all(torch.isclose(sums, torch.ones_like(sums))))


class TestVideoProcessor(unittest.TestCase):
    """Tests for the VideoProcessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock inference session
        self.mock_inference_session = MagicMock()
        self.mock_inference_session.run.return_value = np.array([[0.1, 0.8, 0.1]])
        self.mock_inference_session.get_class_names.return_value = ["No Fall", "Fall", "Other"]
        
        # Create processor
        self.processor = VideoProcessor(
            inference_session=self.mock_inference_session,
            sequence_length=5,  # Small sequence length for testing
            stride=1,
        )
    
    def test_extract_frames(self):
        """Test extracting frames from binary data."""
        # Create a simple test image
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        test_image[25:75, 25:75] = 255  # White square in the middle
        
        # Encode to JPEG
        import cv2
        _, buffer = cv2.imencode('.jpg', test_image)
        
        # Extract frames
        frames = self.processor.extract_frames(buffer.tobytes())
        
        # Check that we got a frame
        self.assertEqual(len(frames), 1)
        
        # Check frame shape
        self.assertEqual(frames[0].shape, (100, 100, 3))
    
    @patch('src.inference_service.utils.ImageProcessor.extract_keypoints')
    def test_process_frame(self, mock_extract_keypoints):
        """Test processing a video frame."""
        # Create mock keypoints
        mock_keypoints = np.random.rand(51).astype(np.float32)
        mock_extract_keypoints.return_value = mock_keypoints
        
        # Create test frame
        test_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Process enough frames to get a prediction
        result = None
        for _ in range(10):  # More than sequence_length to ensure prediction
            result = self.processor.process_frame(test_frame)
        
        # Check that we got a result after enough frames
        self.assertIsNotNone(result)
        
        # Check prediction values
        self.assertEqual(result.prediction, 1)  # Fall class
        self.assertEqual(result.class_name, "Fall")
        self.assertAlmostEqual(result.anomaly_score, 0.8)
        
        # Check that the inference session was called
        self.mock_inference_session.run.assert_called()


if __name__ == '__main__':
    unittest.main()