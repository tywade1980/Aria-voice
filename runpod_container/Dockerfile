# ARIA Voice Agent - Runpod Container
# Single container with: Fish Speech 1.5, XTTS v2, StyleTTS2, Mistral 7B

FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

# System dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3-pip \
    git \
    git-lfs \
    wget \
    curl \
    ffmpeg \
    libsndfile1 \
    libportaudio2 \
    portaudio19-dev \
    espeak-ng \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Create app directory
WORKDIR /app

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support
RUN pip install torch==2.1.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121

# Install TTS engines
# 1. Fish Speech
RUN pip install fish-audio-sdk

# 2. Coqui TTS (XTTS v2)
RUN pip install coqui-tts

# 3. StyleTTS2
RUN pip install styletts2

# Install LLM dependencies
RUN pip install transformers accelerate bitsandbytes scipy

# Install vLLM for efficient Mistral inference
RUN pip install vllm

# Install API framework
RUN pip install fastapi uvicorn python-multipart aiofiles pydantic

# Install additional utilities
RUN pip install soundfile librosa numpy requests huggingface_hub

# Copy the API server
COPY server.py /app/server.py
COPY download_models.py /app/download_models.py

# Pre-download models (can be done at build time or runtime)
# Uncomment to bake models into image (larger image but faster startup)
# RUN python /app/download_models.py

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the server
CMD ["python", "-u", "server.py"]
