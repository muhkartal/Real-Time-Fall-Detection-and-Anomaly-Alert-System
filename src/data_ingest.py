#!/usr/bin/env python3
"""
Data ingestion module for the EdgeVision-Guard project.

This script downloads and preprocesses the UP-Fall and URFall datasets,
and generates keypoints using MediaPipe Pose.
"""

import argparse
import logging
import os
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import gdown
import mediapipe as mp
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Dataset URLs and information
DATASET_INFO = {
    "up-fall": {
        "url": "https://sites.google.com/up.edu.mx/har-up/datasets/up-fall-detection-dataset",
        "gdrive_id": "1VcJ7IFdqL4xhG301wLoOD7zPWz01jGNI",  # Example Google Drive ID
        "description": "UP-Fall Detection Dataset from Universidad Panamericana",
        "license": "CC-BY 4.0",
    },
    "ur-fall": {
        "url": "http://fenix.univ.rzeszow.pl/~mkepski/ds/uf.html",
        "gdrive_id": "1d7DL3seUkvELqCr8Lq15x7h73aSK9BXl",  # Example Google Drive ID
        "description": "UR Fall Detection Dataset from University of Rzeszów",
        "license": "CC-BY-NC 3.0",
    },
}


class DatasetDownloader:
    """Downloads and extracts datasets."""

    def __init__(self, dataset_name: str, output_dir: str):
        """
        Initialize the dataset downloader.

        Args:
            dataset_name: Name of the dataset ('up-fall' or 'ur-fall')
            output_dir: Directory to save the dataset
        """
        self.dataset_name = dataset_name.lower()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if self.dataset_name not in DATASET_INFO:
            raise ValueError(
                f"Dataset {dataset_name} not supported. "
                f"Available datasets: {list(DATASET_INFO.keys())}"
            )

        self.dataset_info = DATASET_INFO[self.dataset_name]
        self.zip_path = self.output_dir / f"{self.dataset_name}.zip"

    def download(self) -> None:
        """Download the dataset using gdown."""
        if self.zip_path.exists():
            logger.info(f"Dataset {self.dataset_name} already downloaded.")
            return

        logger.info(f"Downloading {self.dataset_name} dataset...")
        gdrive_id = self.dataset_info["gdrive_id"]
        
        try:
            gdown.download(
                id=gdrive_id,
                output=str(self.zip_path),
                quiet=False,
            )
            logger.info(f"Downloaded {self.dataset_name} dataset to {self.zip_path}")
        except Exception as e:
            logger.error(f"Failed to download dataset: {e}")
            raise

    def extract(self) -> None:
        """Extract the downloaded zip file."""
        if not self.zip_path.exists():
            raise FileNotFoundError(f"Zip file {self.zip_path} not found.")

        extract_dir = self.output_dir / self.dataset_name
        if extract_dir.exists() and any(extract_dir.iterdir()):
            logger.info(f"Dataset {self.dataset_name} already extracted.")
            return

        logger.info(f"Extracting {self.dataset_name} dataset...")
        extract_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            with zipfile.ZipFile(self.zip_path, "r") as zip_ref:
                zip_ref.extractall(extract_dir)
            logger.info(f"Extracted {self.dataset_name} dataset to {extract_dir}")
        except Exception as e:
            logger.error(f"Failed to extract dataset: {e}")
            raise


class KeypointGenerator:
    """Generates skeleton keypoints from video frames using MediaPipe Pose."""

    def __init__(self, dataset_dir: str, output_dir: Optional[str] = None):
        """
        Initialize the keypoint generator.

        Args:
            dataset_dir: Directory containing dataset frames
            output_dir: Directory to save the keypoints (defaults to dataset_dir/keypoints)
        """
        self.dataset_dir = Path(dataset_dir)
        self.output_dir = Path(output_dir) if output_dir else self.dataset_dir / "keypoints"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def _process_image(self, image_path: Path) -> np.ndarray:
        """
        Process a single image and extract keypoints.

        Args:
            image_path: Path to image file

        Returns:
            Numpy array of keypoints [x, y, confidence] for each landmark
        """
        image = cv2.imread(str(image_path))
        if image is None:
            logger.warning(f"Failed to read image: {image_path}")
            return np.zeros((33, 3))
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image
        results = self.pose.process(image_rgb)
        
        # Extract keypoints
        keypoints = np.zeros((33, 3))
        if results.pose_landmarks:
            for i, landmark in enumerate(results.pose_landmarks.landmark):
                keypoints[i] = [landmark.x, landmark.y, landmark.visibility]
        
        return keypoints

    def generate_keypoints(self) -> None:
        """Generate keypoints for all frames in the dataset."""
        # Find all image files
        image_files = []
        for ext in ["*.jpg", "*.jpeg", "*.png"]:
            image_files.extend(list(self.dataset_dir.glob(f"**/{ext}")))
        
        logger.info(f"Found {len(image_files)} images in {self.dataset_dir}")
        
        # Process each image
        for image_path in tqdm(image_files, desc="Generating keypoints"):
            # Create relative path for saving
            rel_path = image_path.relative_to(self.dataset_dir)
            output_path = self.output_dir / rel_path.with_suffix(".npy")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Skip if already processed
            if output_path.exists():
                continue
            
            # Process image and save keypoints
            try:
                keypoints = self._process_image(image_path)
                np.save(output_path, keypoints)
            except Exception as e:
                logger.warning(f"Error processing {image_path}: {e}")

    def close(self) -> None:
        """Release resources."""
        if hasattr(self, 'pose'):
            self.pose.close()


class DataPreprocessor:
    """Preprocesses the dataset for training."""

    def __init__(
        self, 
        dataset_name: str, 
        dataset_dir: str,
        output_dir: Optional[str] = None,
        sequence_length: int = 30,
    ):
        """
        Initialize the data preprocessor.

        Args:
            dataset_name: Name of the dataset ('up-fall' or 'ur-fall')
            dataset_dir: Directory containing the dataset
            output_dir: Directory to save the preprocessed data
            sequence_length: Length of sequences for LSTM input
        """
        self.dataset_name = dataset_name.lower()
        self.dataset_dir = Path(dataset_dir)
        self.output_dir = Path(output_dir) if output_dir else self.dataset_dir / "processed"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sequence_length = sequence_length

    def _load_metadata(self) -> pd.DataFrame:
        """
        Load dataset metadata.

        Returns:
            DataFrame containing metadata
        """
        if self.dataset_name == "up-fall":
            metadata_path = self.dataset_dir / "metadata.csv"
            if not metadata_path.exists():
                raise FileNotFoundError(f"Metadata file {metadata_path} not found.")
            
            return pd.read_csv(metadata_path)
        
        elif self.dataset_name == "ur-fall":
            # For URFall, create metadata from directory structure
            metadata = []
            for activity_dir in self.dataset_dir.glob("*/"):
                activity = activity_dir.name
                label = 1 if "fall" in activity.lower() else 0
                
                for sequence_dir in activity_dir.glob("*/"):
                    sequence_id = sequence_dir.name
                    metadata.append({
                        "subject_id": sequence_id.split("_")[0] if "_" in sequence_id else "unknown",
                        "activity": activity,
                        "sequence_id": sequence_id,
                        "label": label,
                    })
            
            return pd.DataFrame(metadata)
        
        else:
            raise ValueError(f"Dataset {self.dataset_name} not supported.")

    def preprocess(self) -> None:
        """Preprocess the dataset and save sequences."""
        # Load metadata
        metadata = self._load_metadata()
        logger.info(f"Loaded metadata with {len(metadata)} entries.")
        
        # Process each sequence
        for _, row in tqdm(metadata.iterrows(), desc="Preprocessing sequences", total=len(metadata)):
            self._process_sequence(row)
    
    def _process_sequence(self, metadata_row: pd.Series) -> None:
        """
        Process a single sequence and save as a numpy array.

        Args:
            metadata_row: Row from metadata DataFrame
        """
        if self.dataset_name == "up-fall":
            sequence_id = metadata_row["sequence_id"]
            activity = metadata_row["activity"]
            label = metadata_row["label"]
            
            # Find keypoint files for this sequence
            keypoint_dir = self.dataset_dir / "keypoints" / activity / sequence_id
            keypoint_files = sorted(list(keypoint_dir.glob("*.npy")))
            
        elif self.dataset_name == "ur-fall":
            activity = metadata_row["activity"]
            sequence_id = metadata_row["sequence_id"]
            label = metadata_row["label"]
            
            # Find keypoint files for this sequence
            keypoint_dir = self.dataset_dir / "keypoints" / activity / sequence_id
            keypoint_files = sorted(list(keypoint_dir.glob("*.npy")))
        
        # Skip if not enough frames
        if len(keypoint_files) < self.sequence_length:
            logger.warning(
                f"Sequence {sequence_id} has only {len(keypoint_files)} frames, "
                f"which is less than required {self.sequence_length}. Skipping."
            )
            return
        
        # Create output path
        output_path = self.output_dir / f"{self.dataset_name}_{activity}_{sequence_id}.npz"
        if output_path.exists():
            return
        
        # Load keypoints
        sequences = []
        for i in range(0, len(keypoint_files) - self.sequence_length + 1, self.sequence_length // 2):
            seq_files = keypoint_files[i:i+self.sequence_length]
            seq_data = []
            
            for kp_file in seq_files:
                keypoints = np.load(kp_file)
                
                # Flatten the keypoints to [x1, y1, c1, x2, y2, c2, ...]
                flat_keypoints = keypoints.flatten()
                
                # For simplicity, we'll use only the first 17 keypoints (upper body)
                # Each keypoint has 3 values (x, y, confidence)
                flat_keypoints = flat_keypoints[:51]  # 17 keypoints * 3 values
                
                seq_data.append(flat_keypoints)
            
            sequences.append(np.array(seq_data))
        
        # Save the sequences
        if sequences:
            sequences_array = np.array(sequences)
            np.savez(
                output_path,
                sequences=sequences_array,
                labels=np.full(sequences_array.shape[0], label),
                metadata={
                    "dataset": self.dataset_name,
                    "activity": activity,
                    "sequence_id": sequence_id,
                    "sequence_length": self.sequence_length,
                },
            )
            logger.debug(f"Saved {sequences_array.shape[0]} sequences to {output_path}")


def main() -> None:
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Dataset downloader and preprocessor")
    parser.add_argument(
        "--dataset", 
        type=str, 
        choices=list(DATASET_INFO.keys()),
        required=True,
        help="Dataset to download and process",
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default=None,
        help="Output directory for the dataset",
    )
    parser.add_argument(
        "--download-only", 
        action="store_true",
        help="Only download the dataset without preprocessing",
    )
    parser.add_argument(
        "--generate-keypoints", 
        action="store_true",
        help="Generate keypoints from video frames",
    )
    parser.add_argument(
        "--preprocess", 
        action="store_true",
        help="Preprocess the dataset for training",
    )
    parser.add_argument(
        "--sequence-length", 
        type=int, 
        default=int(os.getenv("SEQUENCE_LENGTH", 30)),
        help="Length of sequences for LSTM input",
    )
    
    args = parser.parse_args()
    
    # Set output directory
    output_dir = args.output or os.getenv("DATA_DIR", "./data")
    
    # Download dataset
    if not args.generate_keypoints and not args.preprocess:
        downloader = DatasetDownloader(args.dataset, output_dir)
        downloader.download()
        
        if not args.download_only:
            downloader.extract()
    
    # Generate keypoints
    if args.generate_keypoints:
        dataset_dir = Path(output_dir) / args.dataset
        keypoint_generator = KeypointGenerator(dataset_dir)
        try:
            keypoint_generator.generate_keypoints()
        finally:
            keypoint_generator.close()
    
    # Preprocess dataset
    if args.preprocess:
        dataset_dir = Path(output_dir) / args.dataset
        preprocessor = DataPreprocessor(
            args.dataset, 
            dataset_dir,
            sequence_length=args.sequence_length,
        )
        preprocessor.preprocess()


if __name__ == "__main__":
    main()