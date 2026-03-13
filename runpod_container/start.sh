#!/bin/bash
# ARIA Voice Agent - Runpod Start Script

echo "========================================"
echo "ARIA Voice Agent - Starting..."
echo "========================================"

# Set environment
export MODELS_DIR=${MODELS_DIR:-/app/models}
export PYTHONUNBUFFERED=1

# Create directories
mkdir -p $MODELS_DIR
mkdir -p /app/voices

# Check GPU
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# Download models if not present
if [ ! -d "$MODELS_DIR/mistral-7b-instruct" ]; then
    echo "Downloading models (this may take 20-30 minutes)..."
    python /app/download_models.py
fi

# Start the API server
echo "Starting API server on port 8000..."
python /app/server.py
