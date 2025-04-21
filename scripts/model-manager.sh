#!/bin/bash
# Model manager script for EdgeVision-Guard
# This script manages model versions and deployment

set -e

# Configuration
MODEL_REGISTRY_DIR="./models"
MODEL_CONFIG_FILE="./models/registry.json"
DEFAULT_MODEL_PATH="./models/fall_detector.onnx"
ACTIVE_MODEL_SYMLINK="./models/active_model.onnx"

# Create model registry directory if it doesn't exist
mkdir -p "$MODEL_REGISTRY_DIR"

# Create model registry file if it doesn't exist
if [ ! -f "$MODEL_CONFIG_FILE" ]; then
    echo '{"models": []}' > "$MODEL_CONFIG_FILE"
fi

# Function to display usage information
usage() {
    echo "EdgeVision-Guard Model Manager"
    echo ""
    echo "Usage: $0 <command> [options]"
    echo ""
    echo "Commands:"
    echo "  list                      List all registered models"
    echo "  register                  Register a new model"
    echo "  activate                  Activate a model by version"
    echo "  info                      Display information about a model"
    echo "  delete                    Delete a model"
    echo "  benchmark                 Benchmark a model"
    echo ""
    echo "Options for 'register':"
    echo "  --model-path <path>       Path to the model file (required)"
    echo "  --version <version>       Model version (required)"
    echo "  --name <name>             Model name (default: extracted from filename)"
    echo "  --description <desc>      Model description"
    echo "  --metrics <json>          Performance metrics as JSON string"
    echo ""
    echo "Options for 'activate':"
    echo "  --version <version>       Model version to activate (required)"
    echo ""
    echo "Options for 'info' and 'delete':"
    echo "  --version <version>       Model version (required)"
    echo ""
    echo "Options for 'benchmark':"
    echo "  --version <version>       Model version to benchmark"
    echo "  --iterations <n>          Number of benchmark iterations (default: 100)"
    echo "  --device <device>         Device to use (cpu, cuda) (default: cpu)"
    echo ""
    echo "Examples:"
    echo "  $0 list"
    echo "  $0 register --model-path ./models/my_model.onnx --version 1.2.0"
    echo "  $0 activate --version 1.2.0"
    echo "  $0 info --version 1.2.0"
    echo "  $0 delete --version 1.2.0"
    echo "  $0 benchmark --version 1.2.0 --iterations 200 --device cuda"
    exit 1
}

# Function to list all registered models
list_models() {
    echo "===== Registered Models ====="
    if [ "$(jq '.models | length' "$MODEL_CONFIG_FILE")" -eq 0 ]; then
        echo "No models registered"
        return
    fi
    
    echo "Version | Name | Date | Status | Size | Description"
    echo "--------|------|------|--------|------|------------"
    
    # Get active model version
    local active_model=""
    if [ -L "$ACTIVE_MODEL_SYMLINK" ]; then
        active_model=$(basename "$(readlink "$ACTIVE_MODEL_SYMLINK")" | sed 's/fall_detector_\(.*\)\.onnx/\1/')
    fi
    
    # List all models
    jq -c '.models[]' "$MODEL_CONFIG_FILE" | while read -r model; do
        version=$(echo "$model" | jq -r '.version')
        name=$(echo "$model" | jq -r '.name')
        date=$(echo "$model" | jq -r '.registered_date')
        size=$(echo "$model" | jq -r '.size')
        description=$(echo "$model" | jq -r '.description')
        
        # Check if this is the active model
        status=""
        if [ "$version" == "$active_model" ]; then
            status="ACTIVE"
        fi
        
        echo "$version | $name | $date | $status | $size | $description"
    done
}

# Function to register a new model
register_model() {
    local model_path=""
    local version=""
    local name=""
    local description=""
    local metrics="{}"
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --model-path)
                model_path="$2"
                shift 2
                ;;
            --version)
                version="$2"
                shift 2
                ;;
            --name)
                name="$2"
                shift 2
                ;;
            --description)
                description="$2"
                shift 2
                ;;
            --metrics)
                metrics="$2"
                shift 2
                ;;
            *)
                echo "Unknown option: $1"
                usage
                ;;
        esac
    done
    
    # Validate arguments
    if [ -z "$model_path" ]; then
        echo "Error: --model-path is required"
        usage
    fi
    
    if [ -z "$version" ]; then
        echo "Error: --version is required"
        usage
    fi
    
    if [ ! -f "$model_path" ]; then
        echo "Error: Model file not found: $model_path"
        exit 1
    fi
    
    # Get file details
    local file_size=$(du -h "$model_path" | cut -f1)
    local file_name=$(basename "$model_path")
    
    # If name is not provided, use the filename
    if [ -z "$name" ]; then
        name=$(basename "$model_path" | sed 's/\.[^.]*$//')
    fi
    
    # Check if the version is already registered
    if jq -e --arg version "$version" '.models[] | select(.version == $version)' "$MODEL_CONFIG_FILE" > /dev/null; then
        echo "Error: Model with version $version is already registered"
        exit 1
    fi
    
    # Copy the model file to registry with versioned name
    local ext="${model_path##*.}"
    local registry_path="$MODEL_REGISTRY_DIR/fall_detector_${version}.${ext}"
    cp "$model_path" "$registry_path"
    
    # Update registry
    local registry_content=$(jq --arg version "$version" \
                               --arg name "$name" \
                               --arg desc "$description" \
                               --arg path "$registry_path" \
                               --arg size "$file_size" \
                               --arg date "$(date +"%Y-%m-%d %H:%M:%S")" \
                               --argjson metrics "$metrics" \
                               '.models += [{
                                   "version": $version,
                                   "name": $name,
                                   "description": $desc,
                                   "path": $path,
                                   "size": $size,
                                   "registered_date": $date,
                                   "metrics": $metrics
                               }]' "$MODEL_CONFIG_FILE")
    
    echo "$registry_content" > "$MODEL_CONFIG_FILE"
    
    echo "Model successfully registered:"
    echo "  Version: $version"
    echo "  Name: $name"
    echo "  Path: $registry_path"
    echo "  Size: $file_size"
}

# Function to activate a model
activate_model() {
    local version=""
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --version)
                version="$2"
                shift 2
                ;;
            *)
                echo "Unknown option: $1"
                usage
                ;;
        esac
    done
    
    # Validate arguments
    if [ -z "$version" ]; then
        echo "Error: --version is required"
        usage
    fi
    
    # Check if the model exists
    local model_path=$(jq -r --arg version "$version" '.models[] | select(.version == $version) | .path' "$MODEL_CONFIG_FILE")
    if [ -z "$model_path" ] || [ "$model_path" == "null" ]; then
        echo "Error: Model with version $version not found"
        exit 1
    fi
    
    # Create or update symlink
    if [ -L "$ACTIVE_MODEL_SYMLINK" ]; then
        rm "$ACTIVE_MODEL_SYMLINK"
    fi
    
    ln -s "$model_path" "$ACTIVE_MODEL_SYMLINK"
    
    echo "Activated model version $version"
    echo "Active model symlink: $ACTIVE_MODEL_SYMLINK -> $model_path"
}

# Function to display model info
get_model_info() {
    local version=""
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --version)
                version="$2"
                shift 2
                ;;
            *)
                echo "Unknown option: $1"
                usage
                ;;
        esac
    done
    
    # Validate arguments
    if [ -z "$version" ]; then
        echo "Error: --version is required"
        usage
    fi
    
    # Get model info
    local model_info=$(jq -r --arg version "$version" '.models[] | select(.version == $version)' "$MODEL_CONFIG_FILE")
    if [ -z "$model_info" ] || [ "$model_info" == "null" ]; then
        echo "Error: Model with version $version not found"
        exit 1
    fi
    
    # Display info
    echo "===== Model Info ====="
    echo "Version: $(echo "$model_info" | jq -r '.version')"
    echo "Name: $(echo "$model_info" | jq -r '.name')"
    echo "Description: $(echo "$model_info" | jq -r '.description')"
    echo "Path: $(echo "$model_info" | jq -r '.path')"
    echo "Size: $(echo "$model_info" | jq -r '.size')"
    echo "Registered: $(echo "$model_info" | jq -r '.registered_date')"
    
    echo -e "\nPerformance Metrics:"
    echo "$model_info" | jq '.metrics'
}

# Function to delete a model
delete_model() {
    local version=""
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --version)
                version="$2"
                shift 2
                ;;
            *)
                echo "Unknown option: $1"
                usage
                ;;
        esac
    done
    
    # Validate arguments
    if [ -z "$version" ]; then
        echo "Error: --version is required"
        usage
    fi
    
    # Get model path
    local model_path=$(jq -r --arg version "$version" '.models[] | select(.version == $version) | .path' "$MODEL_CONFIG_FILE")
    if [ -z "$model_path" ] || [ "$model_path" == "null" ]; then
        echo "Error: Model with version $version not found"
        exit 1
    fi
    
    # Check if this is the active model
    if [ -L "$ACTIVE_MODEL_SYMLINK" ] && [ "$(readlink "$ACTIVE_MODEL_SYMLINK")" == "$model_path" ]; then
        echo "Error: Cannot delete the active model. Activate another model first."
        exit 1
    fi
    
    # Delete model file
    rm -f "$model_path"
    
    # Update registry
    local registry_content=$(jq --arg version "$version" '.models = [.models[] | select(.version != $version)]' "$MODEL_CONFIG_FILE")
    echo "$registry_content" > "$MODEL_CONFIG_FILE"
    
    echo "Model version $version deleted"
}

# Function to benchmark a model
benchmark_model() {
    local version=""
    local iterations=100
    local device="cpu"
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --version)
                version="$2"
                shift 2
                ;;
            --iterations)
                iterations="$2"
                shift 2
                ;;
            --device)
                device="$2"
                shift 2
                ;;
            *)
                echo "Unknown option: $1"
                usage
                ;;
        esac
    done
    
    # If no version specified, use active model
    if [ -z "$version" ]; then
        if [ ! -L "$ACTIVE_MODEL_SYMLINK" ]; then
            echo "Error: No active model found and no version specified"
            exit 1
        fi
        model_path="$ACTIVE_MODEL_SYMLINK"
        version="active"
    else
        # Get model path from registry
        model_path=$(jq -r --arg version "$version" '.models[] | select(.version == $version) | .path' "$MODEL_CONFIG_FILE")
        if [ -z "$model_path" ] || [ "$model_path" == "null" ]; then
            echo "Error: Model with version $version not found"
            exit 1
        fi
    fi
    
    echo "Benchmarking model $([ "$version" == "active" ] && echo "(active)" || echo "version $version")"
    echo "Model path: $model_path"
    echo "Iterations: $iterations"
    echo "Device: $device"
    
    # Run benchmark using Python
    python -c "
import sys
import time
import numpy as np
import onnxruntime as ort

model_path = '$model_path'
iterations = $iterations
device = '$device'

print(f'Loading model: {model_path}')
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']

try:
    session = ort.InferenceSession(model_path, providers=providers)
    
    # Get input details
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    
    # If batch size is dynamic (-1), set it to 1
    if input_shape[0] == -1:
        input_shape = (1,) + tuple(input_shape[1:])
    
    print(f'Input shape: {input_shape}')
    
    # Generate random input data
    input_data = np.random.randn(*input_shape).astype(np.float32)
    
    # Warmup
    print('Warming up...')
    for _ in range(10):
        session.run([session.get_outputs()[0].name], {input_name: input_data})
    
    # Benchmark
    print(f'Running benchmark with {iterations} iterations...')
    latencies = []
    start_time = time.time()
    
    for i in range(iterations):
        iter_start = time.time()
        session.run([session.get_outputs()[0].name], {input_name: input_data})
        latencies.append(time.time() - iter_start)
        
        if (i + 1) % 10 == 0:
            print(f'Progress: {i + 1}/{iterations}')
    
    total_time = time.time() - start_time
    
    # Calculate statistics
    latencies_ms = np.array(latencies) * 1000  # Convert to ms
    avg_latency = np.mean(latencies_ms)
    p50_latency = np.percentile(latencies_ms, 50)
    p95_latency = np.percentile(latencies_ms, 95)
    p99_latency = np.percentile(latencies_ms, 99)
    throughput = iterations / total_time
    
    print('\\nBenchmark Results:')
    print(f'Average latency: {avg_latency:.2f} ms')
    print(f'Median latency (P50): {p50_latency:.2f} ms')
    print(f'P95 latency: {p95_latency:.2f} ms')
    print(f'P99 latency: {p99_latency:.2f} ms')
    print(f'Throughput: {throughput:.2f} inferences/sec')
    
    # Output JSON for metrics
    metrics_json = '{{'
    metrics_json += f'\"avg_latency_ms\": {avg_latency:.2f}, '
    metrics_json += f'\"p50_latency_ms\": {p50_latency:.2f}, '
    metrics_json += f'\"p95_latency_ms\": {p95_latency:.2f}, '
    metrics_json += f'\"p99_latency_ms\": {p99_latency:.2f}, '
    metrics_json += f'\"throughput\": {throughput:.2f}, '
    metrics_json += f'\"device\": \"{device}\"'
    metrics_json += '}}'
    
    print('\\nMetrics JSON (for model registration):')
    print(metrics_json)
    
except Exception as e:
    print(f'Error: {e}', file=sys.stderr)
    sys.exit(1)
"
}

# Main command processing
if [ $# -eq 0 ]; then
    usage
fi

command="$1"
shift

case "$command" in
    list)
        list_models
        ;;
    register)
        register_model "$@"
        ;;
    activate)
        activate_model "$@"
        ;;
    info)
        get_model_info "$@"
        ;;
    delete)
        delete_model "$@"
        ;;
    benchmark)
        benchmark_model "$@"
        ;;
    *)
        echo "Unknown command: $command"
        usage
        ;;
esac

exit 0