#!/bin/bash
# Start CXR Inference Worker
cd "$(dirname "$0")/.."
source venv/bin/activate
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

echo "Starting CXR Inference Worker..."
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Config: worker/config.yaml"

python worker/inference_worker.py --config config.yaml "$@"
