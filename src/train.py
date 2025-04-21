#!/usr/bin/env python3
"""
Training script for the EdgeVision-Guard fall detection model.

This script trains the MobileNetV3-Small + Bi-LSTM model on the preprocessed datasets.
"""

import argparse
import json
import logging
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dotenv import load_dotenv
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                          f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset

from model import create_model

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Training constants
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 32))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", 0.001))
WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY", 0.0001))
EPOCHS = int(os.getenv("EPOCHS", 30))
EARLY_STOPPING_PATIENCE = int(os.getenv("EARLY_STOPPING_PATIENCE", 5))
VALIDATION_SPLIT = float(os.getenv("VALIDATION_SPLIT", 0.2))
TEST_SPLIT = float(os.getenv("TEST_SPLIT", 0.1))
SEQUENCE_LENGTH = int(os.getenv("SEQUENCE_LENGTH", 30))
INPUT_SIZE = int(os.getenv("INPUT_SIZE", 51))
MODELS_DIR = os.getenv("MODELS_DIR", "./models")
LOGS_DIR = os.getenv("LOGS_DIR", "./logs")


class FallDetectionDataset(Dataset):
    """Dataset for fall detection training."""

    def __init__(
        self,
        data_dir: str,
        transform: Optional[callable] = None,
        subset: float = 1.0,
    ):
        """
        Initialize the dataset.

        Args:
            data_dir: Directory containing preprocessed data files
            transform: Optional transform to apply to the data
            subset: Fraction of data to use (0.0-1.0)
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.subset = subset

        # Find all .npz files
        self.data_files = list(self.data_dir.glob("**/*.npz"))
        logger.info(f"Found {len(self.data_files)} data files in {data_dir}")

        # Take a subset if requested
        if subset < 1.0:
            num_files = max(1, int(len(self.data_files) * subset))
            self.data_files = random.sample(self.data_files, num_files)
            logger.info(f"Using {len(self.data_files)} files ({subset:.1%} subset)")

        # Load all data
        self._load_data()

    def _load_data(self):
        """Load all data files into memory."""
        self.sequences = []
        self.labels = []

        for file_path in self.data_files:
            try:
                data = np.load(file_path, allow_pickle=True)
                sequences = data["sequences"]
                labels = data["labels"]

                # Add all sequences and labels
                self.sequences.extend(list(sequences))
                self.labels.extend(list(labels))
            except Exception as e:
                logger.warning(f"Error loading {file_path}: {e}")

        # Convert to numpy arrays
        self.sequences = np.array(self.sequences)
        self.labels = np.array(self.labels)

        logger.info(f"Loaded {len(self.sequences)} sequences")
        logger.info(f"Label distribution: {np.bincount(self.labels)}")

    def __len__(self) -> int:
        """Return the number of sequences."""
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sequence and its label.

        Args:
            idx: Index of the item to retrieve

        Returns:
            Tuple of (sequence, label)
        """
        sequence = self.sequences[idx]
        label = self.labels[idx]

        # Apply transform if provided
        if self.transform:
            sequence = self.transform(sequence)

        # Convert to torch tensors
        sequence = torch.tensor(sequence, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)

        return sequence, label


class Trainer:
    """Trainer class for the fall detection model."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
        lr: float = LEARNING_RATE,
        weight_decay: float = WEIGHT_DECAY,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize the trainer.

        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader
            lr: Learning rate
            weight_decay: Weight decay
            device: Device to train on
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.lr = lr
        self.weight_decay = weight_decay

        # Define loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=3, verbose=True
        )

        # Training state
        self.best_val_loss = float("inf")
        self.best_model_state = None
        self.patience_counter = 0
        self.training_history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "lr": [],
        }

        # Create output directories
        os.makedirs(MODELS_DIR, exist_ok=True)
        os.makedirs(LOGS_DIR, exist_ok=True)

        # Generate a unique run ID
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.info(f"Starting training run {self.run_id} on device: {device}")

    def train_epoch(self) -> Tuple[float, float]:
        """
        Train the model for one epoch.

        Returns:
            Tuple of (average loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)

            # Zero the gradients
            self.optimizer.zero_grad()

            # Forward pass
            output, _ = self.model(data)
            loss = self.criterion(output, target)

            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()

            # Accumulate statistics
            total_loss += loss.item()
            
            # Get predictions
            _, preds = torch.max(output, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(target.cpu().numpy())

            # Print progress
            if (batch_idx + 1) % 10 == 0:
                logger.info(
                    f"Train Batch: {batch_idx+1}/{len(self.train_loader)} "
                    f"Loss: {loss.item():.4f}"
                )

        # Calculate metrics
        avg_loss = total_loss / len(self.train_loader)
        accuracy = accuracy_score(all_labels, all_preds)

        return avg_loss, accuracy

    def validate(self) -> Tuple[float, float]:
        """
        Validate the model.

        Returns:
            Tuple of (average loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)

                # Forward pass
                output, _ = self.model(data)
                loss = self.criterion(output, target)

                # Accumulate statistics
                total_loss += loss.item()
                
                # Get predictions
                _, preds = torch.max(output, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(target.cpu().numpy())

        # Calculate metrics
        avg_loss = total_loss / len(self.val_loader)
        accuracy = accuracy_score(all_labels, all_preds)

        return avg_loss, accuracy

    def train(self, epochs: int, patience: int = EARLY_STOPPING_PATIENCE) -> Dict:
        """
        Train the model for the specified number of epochs.

        Args:
            epochs: Number of epochs to train for
            patience: Early stopping patience

        Returns:
            Training history
        """
        logger.info(f"Starting training for {epochs} epochs...")
        start_time = time.time()

        for epoch in range(1, epochs + 1):
            epoch_start_time = time.time()

            # Train and validate
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()

            # Update scheduler
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Update history
            self.training_history["train_loss"].append(train_loss)
            self.training_history["val_loss"].append(val_loss)
            self.training_history["train_acc"].append(train_acc)
            self.training_history["val_acc"].append(val_acc)
            self.training_history["lr"].append(current_lr)

            # Print epoch summary
            epoch_time = time.time() - epoch_start_time
            logger.info(
                f"Epoch: {epoch}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Train Acc: {train_acc:.4f} | "
                f"Val Acc: {val_acc:.4f} | "
                f"LR: {current_lr:.6f} | "
                f"Time: {epoch_time:.2f}s"
            )

            # Save best model
            if val_loss < self.best_val_loss:
                logger.info(f"Validation loss improved from {self.best_val_loss:.4f} to {val_loss:.4f}")
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict().copy()
                self.patience_counter = 0
                
                # Save model
                self._save_model(f"model_{self.run_id}_best.pth")
            else:
                self.patience_counter += 1
                logger.info(f"Validation loss did not improve. Patience: {self.patience_counter}/{patience}")

            # Check early stopping
            if self.patience_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch} epochs")
                break

            # Save checkpoint
            if epoch % 5 == 0 or epoch == epochs:
                self._save_model(f"model_{self.run_id}_epoch_{epoch}.pth")
                self._save_history()

        # Restore best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            logger.info("Restored best model")

        # Calculate total training time
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.2f} seconds")

        # Save final model and history
        self._save_model(f"model_{self.run_id}_final.pth")
        self._save_history()

        # Evaluate on test set if available
        if self.test_loader is not None:
            self.evaluate()

        return self.training_history

    def evaluate(self) -> Dict:
        """
        Evaluate the model on the test set.

        Returns:
            Dictionary of evaluation metrics
        """
        if self.test_loader is None:
            logger.warning("No test loader provided for evaluation")
            return {}

        logger.info("Evaluating model on test set...")
        self.model.eval()
        all_preds = []
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)

                # Forward pass
                output, _ = self.model(data)
                
                # Get predictions and probabilities
                probs = torch.softmax(output, dim=1)
                _, preds = torch.max(output, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(target.cpu().numpy())

        # Convert to numpy arrays
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)

        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average="weighted")
        recall = recall_score(all_labels, all_preds, average="weighted")
        f1 = f1_score(all_labels, all_preds, average="weighted")
        
        # ROC AUC for multi-class
        try:
            roc_auc = roc_auc_score(
                np.eye(self.model.num_classes)[all_labels], all_probs, multi_class="ovr"
            )
        except ValueError:
            roc_auc = 0.0
            logger.warning("Could not calculate ROC AUC score")

        # Create classification report
        report = classification_report(all_labels, all_preds, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)

        # Compile metrics
        metrics = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "roc_auc": float(roc_auc),
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
        }

        # Log results
        logger.info(f"Test Accuracy: {accuracy:.4f}")
        logger.info(f"Test Precision: {precision:.4f}")
        logger.info(f"Test Recall: {recall:.4f}")
        logger.info(f"Test F1 Score: {f1:.4f}")
        logger.info(f"Test ROC AUC: {roc_auc:.4f}")

        # Save metrics
        self._save_metrics(metrics)

        return metrics

    def _save_model(self, filename: str) -> None:
        """
        Save the model.

        Args:
            filename: Name of the model file
        """
        model_path = os.path.join(MODELS_DIR, filename)
        torch.save(self.model.state_dict(), model_path)
        logger.info(f"Model saved to {model_path}")

    def _save_history(self) -> None:
        """Save the training history."""
        history_path = os.path.join(LOGS_DIR, f"history_{self.run_id}.json")
        with open(history_path, "w") as f:
            json.dump(self.training_history, f, indent=2)
        logger.info(f"Training history saved to {history_path}")

    def _save_metrics(self, metrics: Dict) -> None:
        """
        Save evaluation metrics.

        Args:
            metrics: Dictionary of metrics
        """
        metrics_path = os.path.join(LOGS_DIR, f"metrics_{self.run_id}.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Evaluation metrics saved to {metrics_path}")


def main() -> None:
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Train the fall detection model")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=os.getenv("DATA_DIR", "./data"),
        help="Directory containing the preprocessed data",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Batch size for training",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=LEARNING_RATE,
        help="Learning rate",
    )
    parser.add_argument(
        "--subset",
        type=float,
        default=1.0,
        help="Fraction of data to use (0.0-1.0)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
        help="Device to train on",
    )
    
    args = parser.parse_args()
    
    # Print arguments
    logger.info(f"Training with arguments: {args}")
    
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # Load dataset
    dataset = FallDetectionDataset(
        data_dir=args.data_dir,
        subset=args.subset,
    )
    
    # Split dataset
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    np.random.shuffle(indices)
    
    test_split = int(np.floor(TEST_SPLIT * dataset_size))
    val_split = int(np.floor(VALIDATION_SPLIT * dataset_size))
    
    test_indices = indices[:test_split]
    val_indices = indices[test_split:test_split+val_split]
    train_indices = indices[test_split+val_split:]
    
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)
    
    logger.info(f"Training set size: {len(train_dataset)}")
    logger.info(f"Validation set size: {len(val_dataset)}")
    logger.info(f"Test set size: {len(test_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True if args.device == "cuda" else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True if args.device == "cuda" else False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True if args.device == "cuda" else False,
    )
    
    # Create model
    model = create_model()
    logger.info(f"Created model with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        lr=args.lr,
        device=args.device,
    )
    
    # Train model
    trainer.train(epochs=args.epochs)
    
    # Evaluate model
    metrics = trainer.evaluate()
    
    logger.info("Training complete!")


if __name__ == "__main__":
    main()