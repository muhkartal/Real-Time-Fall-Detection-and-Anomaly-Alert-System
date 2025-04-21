#!/usr/bin/env python3
"""
Hyperparameter optimization for EdgeVision-Guard model.

This module provides functionality for optimizing model hyperparameters
using techniques like Bayesian optimization, random search, and grid search.
"""

import argparse
import json
import logging
import os
import sys
import time
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import optuna
import torch
import torch.nn as nn
from dotenv import load_dotenv
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.model_selection import train_test_split

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from model import create_model
from train import FallDetectionDataset, Trainer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load hyperparameter search configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
    
    Returns:
        Configuration dictionary
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Loaded configuration from {config_path}")
    return config


def get_param_value(trial: optuna.Trial, param_config: Dict[str, Any], param_name: str) -> Any:
    """
    Get parameter value from trial based on configuration.
    
    Args:
        trial: Optuna trial
        param_config: Parameter configuration
        param_name: Parameter name
    
    Returns:
        Parameter value
    """
    if "categorical" in param_config:
        return trial.suggest_categorical(param_name, param_config["categorical"])
    
    elif "integer" in param_config:
        low, high = param_config["integer"]
        return trial.suggest_int(param_name, low, high)
    
    elif "uniform" in param_config:
        low, high = param_config["uniform"]
        return trial.suggest_float(param_name, low, high)
    
    elif "log_uniform" in param_config:
        low, high = param_config["log_uniform"]
        return trial.suggest_float(param_name, low, high, log=True)
    
    elif "quantized_log_uniform" in param_config:
        low, high, step = param_config["quantized_log_uniform"]
        # Get continuous value and quantize
        continuous = trial.suggest_float(f"{param_name}_raw", np.log(low), np.log(high), log=False)
        quantized = round(np.exp(continuous) / step) * step
        return min(max(quantized, low), high)
    
    elif "conditional" in param_config:
        parent = param_config["conditional"]["parent"]
        values = param_config["conditional"]["values"]
        
        # Get the parent value from trial params
        parent_value = trial.params.get(parent)
        
        # If parent value is available and has configuration for this value
        if parent_value is not None and parent_value in values:
            child_config = values[parent_value]
            return get_param_value(trial, child_config, param_name)
        
        # Default return value if not handled by conditional
        return None
    
    # Unknown parameter type
    logger.warning(f"Unknown parameter type for {param_name}: {param_config}")
    return None


def check_constraints(params: Dict[str, Any], constraints: List[Dict[str, str]]) -> bool:
    """
    Check if parameters satisfy all constraints.
    
    Args:
        params: Parameter dictionary
        constraints: List of constraint dictionaries
    
    Returns:
        True if all constraints are satisfied, False otherwise
    """
    for constraint in constraints:
        # Create a local dictionary with flattened parameters
        local_dict = {}
        for param_name, param_value in params.items():
            parts = param_name.split(".")
            
            # Build nested dictionary
            current = local_dict
            for i, part in enumerate(parts[:-1]):
                if part not in current:
                    current[part] = {}
                current = current[part]
            
            # Set value in the innermost dictionary
            current[parts[-1]] = param_value
        
        try:
            # Evaluate constraint
            if not eval(constraint["check"], {"__builtins__": {}}, local_dict):
                logger.info(f"Constraint violated: {constraint['description']}")
                return False
        except Exception as e:
            logger.warning(f"Error evaluating constraint {constraint['name']}: {e}")
            return False
    
    return True


def objective(trial: optuna.Trial, config: Dict[str, Any], data_dir: str, device: str) -> float:
    """
    Objective function for hyperparameter optimization.
    
    Args:
        trial: Optuna trial
        config: Configuration dictionary
        data_dir: Data directory
        device: Device to run on
    
    Returns:
        Optimization metric value
    """
    # Extract search space and fixed parameters
    search_space = config["search_space"]
    fixed_params = config.get("fixed_parameters", {})
    constraints = config.get("constraints", [])
    
    # Build parameter dictionary
    params = {}
    
    # Process search space
    def process_search_space(space: Dict[str, Any], parent_key: str = ""):
        for key, value in space.items():
            param_name = f"{parent_key}.{key}" if parent_key else key
            
            if isinstance(value, dict) and not any(k in value for k in ["categorical", "integer", "uniform", "log_uniform", "quantized_log_uniform", "conditional"]):
                # This is a nested dictionary, recurse
                process_search_space(value, param_name)
            else:
                # This is a parameter specification
                param_value = get_param_value(trial, value, param_name)
                if param_value is not None:
                    params[param_name] = param_value
    
    # Process fixed parameters
    def process_fixed_params(fixed: Dict[str, Any], parent_key: str = ""):
        for key, value in fixed.items():
            param_name = f"{parent_key}.{key}" if parent_key else key
            
            if isinstance(value, dict):
                # This is a nested dictionary, recurse
                process_fixed_params(value, param_name)
            else:
                # This is a fixed parameter
                params[param_name] = value
    
    # Build parameter dictionary
    process_search_space(search_space)
    process_fixed_params(fixed_params)
    
    # Check constraints
    if not check_constraints(params, constraints):
        # If constraints are violated, set a very bad score
        return float("-inf")
    
    # Extract model parameters
    model_params = {
        "num_classes": params.get("model.classifier.num_classes", 3),
        "hidden_size": params.get("model.sequence_model.hidden_size", 128),
        "num_layers": params.get("model.sequence_model.num_layers", 2),
        "input_size": params.get("model.input.feature_size", 51),
        "dropout": params.get("model.feature_extractor.dropout", 0.5),
    }
    
    # Extract training parameters
    train_params = {
        "batch_size": params.get("training.batch_size", 32),
        "lr": params.get("training.optimizer.learning_rate", 0.001),
        "weight_decay": params.get("training.optimizer.weight_decay", 0.0001),
        "epochs": params.get("training.epochs", 30),
        "early_stopping_patience": params.get("training.early_stopping_patience", 5),
    }
    
    # Log parameters
    logger.info(f"Trial {trial.number}: Testing parameters: {params}")
    
    try:
        # Load dataset
        dataset = FallDetectionDataset(
            data_dir=data_dir,
            transform=None,  # TODO: Add data augmentation based on params
            subset=0.5,  # Use a subset for faster optimization
        )
        
        # Split dataset
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        np.random.shuffle(indices)
        
        val_split = params.get("training.validation_split", 0.2)
        test_split = params.get("training.test_split", 0.1)
        
        test_size = int(np.floor(test_split * dataset_size))
        val_size = int(np.floor(val_split * dataset_size))
        
        test_indices = indices[:test_size]
        val_indices = indices[test_size:test_size+val_size]
        train_indices = indices[test_size+val_size:]
        
        # Create subset datasets
        from torch.utils.data import Subset
        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)
        test_dataset = Subset(dataset, test_indices)
        
        # Create data loaders
        from torch.utils.data import DataLoader
        train_loader = DataLoader(
            train_dataset,
            batch_size=train_params["batch_size"],
            shuffle=True,
            num_workers=2,
            pin_memory=True if device == "cuda" else False,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=train_params["batch_size"],
            shuffle=False,
            num_workers=2,
            pin_memory=True if device == "cuda" else False,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=train_params["batch_size"],
            shuffle=False,
            num_workers=2,
            pin_memory=True if device == "cuda" else False,
        )
        
        # Create model
        model = create_model(**model_params)
        
        # Create trainer
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            lr=train_params["lr"],
            weight_decay=train_params["weight_decay"],
            device=device,
        )
        
        # Train model
        history = trainer.train(
            epochs=train_params["epochs"],
            patience=train_params["early_stopping_patience"],
        )
        
        # Evaluate model
        metrics = trainer.evaluate()
        
        # Get optimization metric
        metric_name = config.get("optimization_metric", "val_f1")
        if metric_name.startswith("val_"):
            # Get validation metric
            metric_key = metric_name[4:]  # Remove "val_" prefix
            if metric_key == "loss":
                metric_value = min(history["val_loss"])
            elif metric_key == "acc":
                metric_value = max(history["val_acc"])
            else:
                # For other metrics, use test metrics
                metric_value = metrics.get(metric_key, float("-inf"))
        else:
            # Get test metric
            metric_value = metrics.get(metric_name, float("-inf"))
        
        # Report intermediate values for pruning
        intermediate_values = {}
        for epoch in range(len(history["val_loss"])):
            intermediate_values[epoch] = -history["val_loss"][epoch] if metric_name == "val_loss" else history["val_acc"][epoch]
        
        for epoch, value in intermediate_values.items():
            trial.report(value, epoch)
            
            # Handle pruning
            if trial.should_prune():
                logger.info(f"Trial {trial.number} pruned at epoch {epoch}")
                raise optuna.exceptions.TrialPruned()
        
        # If optimization direction is minimize, negate the value
        if config.get("optimization_direction", "maximize") == "minimize":
            metric_value = -metric_value
        
        logger.info(f"Trial {trial.number} finished with {metric_name} = {metric_value}")
        
        return metric_value
    
    except Exception as e:
        logger.error(f"Error in trial {trial.number}: {e}")
        return float("-inf")


def run_hyperparameter_optimization(config_path: str, data_dir: str, output_dir: str, device: str, n_trials: Optional[int] = None, study_name: Optional[str] = None) -> None:
    """
    Run hyperparameter optimization.
    
    Args:
        config_path: Path to configuration file
        data_dir: Data directory
        output_dir: Output directory
        device: Device to run on
        n_trials: Number of trials to run (overrides config if provided)
        study_name: Study name (overrides config if provided)
    """
    # Load configuration
    config = load_config(config_path)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set study name
    if study_name is None:
        study_name = config.get("experiment_name", "fall_detection_hyperopt")
    
    # Set number of trials
    if n_trials is None:
        n_trials = config.get("max_trials", 50)
    
    # Set up sampler based on search algorithm
    search_algorithm = config.get("search_algorithm", "bayesian")
    if search_algorithm == "bayesian":
        sampler = TPESampler(seed=42)
    elif search_algorithm == "random":
        sampler = optuna.samplers.RandomSampler(seed=42)
    elif search_algorithm == "grid":
        # For grid search, we would need to define a grid
        logger.warning("Grid search not fully implemented. Using random sampling instead.")
        sampler = optuna.samplers.RandomSampler(seed=42)
    else:
        logger.warning(f"Unknown search algorithm: {search_algorithm}. Using Bayesian optimization.")
        sampler = TPESampler(seed=42)
    
    # Set up pruner
    early_stopping_config = config.get("early_stopping", {})
    if early_stopping_config.get("enabled", True):
        pruner = MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=10,
            interval_steps=1,
        )
    else:
        pruner = optuna.pruners.NopPruner()
    
    # Create study
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",  # We will handle minimization by negating values
        sampler=sampler,
        pruner=pruner,
        storage=f"sqlite:///{output_dir}/optuna.db",
        load_if_exists=True,
    )
    
    # Run optimization
    study.optimize(
        lambda trial: objective(trial, config, data_dir, device),
        n_trials=n_trials,
        timeout=None,  # No timeout
        n_jobs=1,  # Single job for now
        show_progress_bar=True,
    )
    
    # Get best trial
    best_trial = study.best_trial
    logger.info(f"Best trial: {best_trial.number}")
    logger.info(f"Best value: {best_trial.value}")
    logger.info(f"Best parameters: {best_trial.params}")
    
    # Save best parameters
    with open(os.path.join(output_dir, "best_params.json"), "w") as f:
        json.dump(best_trial.params, f, indent=2)
    
    # Save study results
    df = study.trials_dataframe()
    df.to_csv(os.path.join(output_dir, "study_results.csv"))
    
    # Create importance plot
    try:
        from optuna.visualization import plot_param_importances
        import matplotlib.pyplot as plt
        
        # Plot parameter importances
        fig = plot_param_importances(study)
        fig.write_image(os.path.join(output_dir, "param_importances.png"))
        
        # Plot optimization history
        from optuna.visualization import plot_optimization_history
        fig = plot_optimization_history(study)
        fig.write_image(os.path.join(output_dir, "optimization_history.png"))
    except Exception as e:
        logger.warning(f"Could not create visualization: {e}")
    
    # Log completion
    logger.info(f"Hyperparameter optimization completed. Results saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter optimization for EdgeVision-Guard")
    parser.add_argument(
        "--config",
        type=str,
        default="config/hyperparameter_space.yaml",
        help="Path to hyperparameter search configuration file",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=os.getenv("DATA_DIR", "./data"),
        help="Directory containing the preprocessed data",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./hyperopt_results",
        help="Directory to save optimization results",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
        help="Device to run on",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=None,
        help="Number of trials to run (overrides config if provided)",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default=None,
        help="Study name (overrides config if provided)",
    )
    
    args = parser.parse_args()
    
    run_hyperparameter_optimization(
        config_path=args.config,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        device=args.device,
        n_trials=args.n_trials,
        study_name=args.study_name,
    )