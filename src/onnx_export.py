#!/usr/bin/env python3
"""
ONNX Export Script for EdgeVision-Guard.

This script exports the PyTorch model to ONNX format and performs INT8 quantization.
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import onnx
import onnxruntime as ort
import torch
from dotenv import load_dotenv
from onnxruntime.quantization import (CalibrationDataReader, QuantFormat,
                                    QuantType, quantize_dynamic,
                                    quantize_static)

from model import create_model, load_model

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Constants
OPSET_VERSION = int(os.getenv("ONNX_OPSET_VERSION", 14))
SEQUENCE_LENGTH = int(os.getenv("SEQUENCE_LENGTH", 30))
INPUT_SIZE = int(os.getenv("INPUT_SIZE", 51))
MODELS_DIR = os.getenv("MODELS_DIR", "./models")
QUANTIZE_MODEL = os.getenv("QUANTIZE_MODEL", "true").lower() in ("true", "1", "yes")


class FallDetectionCalibrationDataReader(CalibrationDataReader):
    """
    Calibration data reader for quantization.
    
    This class provides sample data for ONNX Runtime quantization calibration.
    """
    
    def __init__(self, n_samples: int = 100):
        """
        Initialize the calibration data reader.
        
        Args:
            n_samples: Number of random samples to generate
        """
        self.n_samples = n_samples
        self.data = self._generate_random_data()
        self.current_index = 0
    
    def _generate_random_data(self) -> List[Dict[str, np.ndarray]]:
        """
        Generate random calibration data.
        
        Returns:
            List of dictionaries with input names and values
        """
        data = []
        for _ in range(self.n_samples):
            # Generate random sequence data
            x = np.random.randn(1, SEQUENCE_LENGTH, INPUT_SIZE).astype(np.float32)
            data.append({"input": x})
        
        return data
    
    def get_next(self) -> Optional[Dict[str, np.ndarray]]:
        """
        Get the next calibration data item.
        
        Returns:
            Dictionary with input names and values, or None if end of data
        """
        if self.current_index >= self.n_samples:
            return None
        
        sample = self.data[self.current_index]
        self.current_index += 1
        return sample
    
    def rewind(self) -> None:
        """Reset the data reader to the beginning."""
        self.current_index = 0


def export_to_onnx(
    model: torch.nn.Module,
    output_path: str,
    input_shape: Tuple[int, int, int] = (1, SEQUENCE_LENGTH, INPUT_SIZE),
    opset_version: int = OPSET_VERSION,
    dynamic_axes: Optional[Dict] = None,
    verbose: bool = True,
) -> str:
    """
    Export PyTorch model to ONNX format.
    
    Args:
        model: PyTorch model
        output_path: Path to save the ONNX model
        input_shape: Input tensor shape
        opset_version: ONNX opset version
        dynamic_axes: Dynamic axes configuration
        verbose: Whether to print verbose output
    
    Returns:
        Path to the exported ONNX model
    """
    # Ensure model is in eval mode
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(input_shape, requires_grad=True)
    
    # Default dynamic axes if not provided
    if dynamic_axes is None:
        dynamic_axes = {
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        }
    
    # Export the model
    logger.info(f"Exporting model to ONNX: {output_path}")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        verbose=verbose,
    )
    
    logger.info(f"Model exported to: {output_path}")
    
    # Verify the ONNX model
    logger.info("Verifying ONNX model...")
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    logger.info("ONNX model verified successfully.")
    
    return output_path


def quantize_onnx_model(
    model_path: str,
    output_path: Optional[str] = None,
    quantization_type: str = "dynamic",
    per_channel: bool = False,
    reduce_range: bool = False,
) -> str:
    """
    Quantize ONNX model to INT8.
    
    Args:
        model_path: Path to the ONNX model
        output_path: Path to save the quantized model
        quantization_type: Type of quantization ('dynamic' or 'static')
        per_channel: Whether to quantize per channel
        reduce_range: Whether to reduce range for compatibility with certain hardware
    
    Returns:
        Path to the quantized model
    """
    # Create output path if not provided
    if output_path is None:
        path = Path(model_path)
        output_path = str(path.with_stem(f"{path.stem}_quantized"))
    
    logger.info(f"Quantizing model ({quantization_type}): {model_path}")
    
    if quantization_type == "dynamic":
        # Dynamic quantization
        quantize_dynamic(
            model_input=model_path,
            model_output=output_path,
            per_channel=per_channel,
            reduce_range=reduce_range,
            weight_type=QuantType.QInt8,
        )
    else:
        # Static quantization with calibration
        calibration_data_reader = FallDetectionCalibrationDataReader(n_samples=100)
        
        quantize_static(
            model_input=model_path,
            model_output=output_path,
            calibration_data_reader=calibration_data_reader,
            quant_format=QuantFormat.QDQ,
            per_channel=per_channel,
            reduce_range=reduce_range,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
        )
    
    logger.info(f"Quantized model saved to: {output_path}")
    
    return output_path


def validate_onnx_model(
    model_path: str,
    input_shape: Tuple[int, int, int] = (1, SEQUENCE_LENGTH, INPUT_SIZE),
    device: str = "cpu",
) -> None:
    """
    Validate the ONNX model by running inference.
    
    Args:
        model_path: Path to the ONNX model
        input_shape: Input tensor shape
        device: Device to run inference on
    """
    logger.info(f"Validating ONNX model: {model_path}")
    
    # Create ONNX Runtime session
    if device.lower() == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers():
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        logger.info("Using CUDA for ONNX validation")
    else:
        providers = ["CPUExecutionProvider"]
        logger.info("Using CPU for ONNX validation")
    
    session = ort.InferenceSession(model_path, providers=providers)
    
    # Create random input
    input_data = np.random.randn(*input_shape).astype(np.float32)
    
    # Run inference
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    start_time = time.time()
    ort_outputs = session.run([output_name], {input_name: input_data})
    end_time = time.time()
    
    # Log results
    logger.info(f"ONNX model validation successful.")
    logger.info(f"Inference time: {(end_time - start_time) * 1000:.2f} ms")
    logger.info(f"Output shape: {ort_outputs[0].shape}")


def compare_pytorch_vs_onnx(
    pytorch_model: torch.nn.Module,
    onnx_model_path: str,
    input_shape: Tuple[int, int, int] = (1, SEQUENCE_LENGTH, INPUT_SIZE),
    device: str = "cpu",
) -> None:
    """
    Compare PyTorch and ONNX model outputs.
    
    Args:
        pytorch_model: PyTorch model
        onnx_model_path: Path to the ONNX model
        input_shape: Input tensor shape
        device: Device to run inference on
    """
    logger.info("Comparing PyTorch and ONNX model outputs...")
    
    # Set PyTorch model to evaluation mode
    pytorch_model.eval()
    pytorch_model.to(device)
    
    # Create ONNX Runtime session
    if device.lower() == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers():
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]
    
    session = ort.InferenceSession(onnx_model_path, providers=providers)
    
    # Create random input
    input_data = np.random.randn(*input_shape).astype(np.float32)
    input_tensor = torch.tensor(input_data, dtype=torch.float32).to(device)
    
    # Run PyTorch inference
    with torch.no_grad():
        pytorch_output, _ = pytorch_model(input_tensor)
        pytorch_output = pytorch_output.cpu().numpy()
    
    # Run ONNX inference
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    onnx_output = session.run([output_name], {input_name: input_data})[0]
    
    # Compare outputs
    max_diff = np.max(np.abs(pytorch_output - onnx_output))
    mean_diff = np.mean(np.abs(pytorch_output - onnx_output))
    
    logger.info(f"Maximum absolute difference: {max_diff:.6f}")
    logger.info(f"Mean absolute difference: {mean_diff:.6f}")
    
    if max_diff < 1e-4:
        logger.info("PyTorch and ONNX models produce equivalent outputs.")
    else:
        logger.warning("PyTorch and ONNX models produce different outputs!")


def benchmark_model(
    model_path: str,
    input_shape: Tuple[int, int, int] = (1, SEQUENCE_LENGTH, INPUT_SIZE),
    n_iterations: int = 100,
    warmup: int = 10,
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Benchmark the ONNX model.
    
    Args:
        model_path: Path to the ONNX model
        input_shape: Input tensor shape
        n_iterations: Number of iterations to run
        warmup: Number of warmup iterations
        device: Device to run inference on
    
    Returns:
        Dictionary with benchmark results
    """
    logger.info(f"Benchmarking ONNX model: {model_path}")
    
    # Create ONNX Runtime session
    if device.lower() == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers():
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        logger.info("Using CUDA for benchmarking")
    else:
        providers = ["CPUExecutionProvider"]
        logger.info("Using CPU for benchmarking")
    
    session = ort.InferenceSession(model_path, providers=providers)
    
    # Create random input
    input_data = np.random.randn(*input_shape).astype(np.float32)
    
    # Get input and output names
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    # Warmup
    for _ in range(warmup):
        session.run([output_name], {input_name: input_data})
    
    # Benchmark
    latencies = []
    start_time = time.time()
    
    for _ in range(n_iterations):
        iter_start = time.time()
        session.run([output_name], {input_name: input_data})
        latencies.append(time.time() - iter_start)
    
    total_time = time.time() - start_time
    
    # Calculate statistics
    latencies = np.array(latencies) * 1000  # Convert to ms
    avg_latency = np.mean(latencies)
    min_latency = np.min(latencies)
    max_latency = np.max(latencies)
    p50_latency = np.percentile(latencies, 50)
    p95_latency = np.percentile(latencies, 95)
    p99_latency = np.percentile(latencies, 99)
    throughput = n_iterations / total_time
    
    # Log results
    logger.info(f"Average latency: {avg_latency:.2f} ms")
    logger.info(f"Minimum latency: {min_latency:.2f} ms")
    logger.info(f"Maximum latency: {max_latency:.2f} ms")
    logger.info(f"P50 latency: {p50_latency:.2f} ms")
    logger.info(f"P95 latency: {p95_latency:.2f} ms")
    logger.info(f"P99 latency: {p99_latency:.2f} ms")
    logger.info(f"Throughput: {throughput:.2f} inferences/sec")
    
    # Return results
    return {
        "avg_latency_ms": float(avg_latency),
        "min_latency_ms": float(min_latency),
        "max_latency_ms": float(max_latency),
        "p50_latency_ms": float(p50_latency),
        "p95_latency_ms": float(p95_latency),
        "p99_latency_ms": float(p99_latency),
        "throughput": float(throughput),
    }


if __name__ == "__main__":
    import time
    
    parser = argparse.ArgumentParser(description="Export PyTorch model to ONNX")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the PyTorch model",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for the ONNX model",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        default=QUANTIZE_MODEL,
        help="Quantize the model to INT8",
    )
    parser.add_argument(
        "--quantization-type",
        type=str,
        choices=["dynamic", "static"],
        default="dynamic",
        help="Type of quantization to perform",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate the exported model",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Benchmark the exported model",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run validation and benchmarking on",
    )
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output) if args.output else MODELS_DIR, exist_ok=True)
    
    # Set default output path if not provided
    if args.output is None:
        model_filename = os.path.basename(args.model_path)
        model_name = os.path.splitext(model_filename)[0]
        args.output = os.path.join(MODELS_DIR, f"{model_name}.onnx")
    
    # Load PyTorch model
    logger.info(f"Loading PyTorch model: {args.model_path}")
    model = load_model(args.model_path, device="cpu")
    
    # Export to ONNX
    onnx_path = export_to_onnx(
        model=model,
        output_path=args.output,
    )
    
    # Compare PyTorch and ONNX outputs
    compare_pytorch_vs_onnx(
        pytorch_model=model,
        onnx_model_path=onnx_path,
        device=args.device,
    )
    
    # Quantize model if requested
    if args.quantize:
        quantized_onnx_path = quantize_onnx_model(
            model_path=onnx_path,
            quantization_type=args.quantization_type,
        )
        
        # Update path for validation and benchmarking
        onnx_path = quantized_onnx_path
    
    # Validate model if requested
    if args.validate:
        validate_onnx_model(
            model_path=onnx_path,
            device=args.device,
        )
    
    # Benchmark model if requested
    if args.benchmark:
        benchmark_model(
            model_path=onnx_path,
            device=args.device,
        )
    
    logger.info("ONNX export completed successfully!")