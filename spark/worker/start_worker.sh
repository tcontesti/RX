#!/bin/bash
set -e

cd "$(dirname "$0")/.."
source venv/bin/activate
export PYTHONPATH="$(pwd)"

echo "=== CXR Inference Worker ==="
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Config: worker/config.yaml"
echo "PID: $$"
echo "==========================="

# Verificar GPU
python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'" || { echo "ERROR: No GPU available"; exit 1; }

# Verificar config
[ -f worker/config.yaml ] || { echo "ERROR: worker/config.yaml not found"; exit 1; }

exec python3 -u worker/inference_worker.py --config config.yaml "$@"
