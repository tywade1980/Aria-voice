# ARIA Voice Agent - Runpod Deployment

## Overview
Single container deployment for Runpod containing:
- **Fish Speech 1.5** - High quality multilingual TTS
- **XTTS v2** - Voice cloning TTS (Coqui)
- **StyleTTS2** - Style-based TTS
- **Mistral 7B Instruct** - Open source LLM

## Requirements
- Runpod GPU Pod with **48GB+ VRAM** (A40, A6000, A100 recommended)
- ~50GB disk space for models

## Quick Deploy on Runpod

### Option 1: Build and Push to Docker Hub

```bash
# Build the image
docker build -t yourusername/aria-voice-agent:latest .

# Push to Docker Hub
docker push yourusername/aria-voice-agent:latest
```

Then in Runpod:
1. Go to **Pods** > **Deploy**
2. Select GPU (A40 48GB recommended)
3. Use custom Docker image: `yourusername/aria-voice-agent:latest`
4. Set volume mount: `/app/models` (for model persistence)
5. Expose port: `8000`

### Option 2: Direct Runpod Template

Create a Runpod template with:
```
Image: nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04
Volume: /app/models (50GB)
Port: 8000
Start Command: /start.sh
```

Then SSH into pod and run:
```bash
# Clone this setup
git clone <your-repo-with-these-files>
cd aria-container

# Install dependencies
pip install -r requirements.txt

# Download models (takes 20-30 min)
python download_models.py

# Start server
python server.py
```

## API Endpoints

Base URL: `http://<your-pod-ip>:8000`

### Health Check
```bash
curl http://localhost:8000/health
```

### Text-to-Speech
```bash
# XTTS v2
curl -X POST http://localhost:8000/api/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "engine": "xtts", "language": "en"}'

# Fish Speech
curl -X POST http://localhost:8000/api/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "engine": "fish"}'

# StyleTTS2
curl -X POST http://localhost:8000/api/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "engine": "styletts2"}'
```

### Voice Cloning
```bash
# Upload reference audio
curl -X POST http://localhost:8000/api/voice/clone \
  -F "name=my_voice" \
  -F "reference_audio=@voice_sample.wav"

# Use cloned voice
curl -X POST http://localhost:8000/api/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello", "engine": "xtts", "voice": "/app/voices/my_voice.wav"}'
```

### Chat with Mistral
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is the capital of France?"}
    ],
    "max_tokens": 256,
    "temperature": 0.7
  }'
```

### Speech-to-Text
```bash
curl -X POST http://localhost:8000/api/stt \
  -F "file=@audio.wav" \
  -F "language=en"
```

## Response Formats

### TTS Response
```json
{
  "audio_base64": "<base64-encoded-wav>",
  "format": "wav",
  "engine": "xtts",
  "duration": 2.5
}
```

### Chat Response
```json
{
  "response": "The capital of France is Paris.",
  "tokens_used": 42,
  "model": "mistral-7b-instruct"
}
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| MODELS_DIR | Model storage path | /app/models |
| FISH_API_KEY | Fish Audio API key (optional) | - |
| HF_TOKEN | Hugging Face token | - |

## Memory Requirements

| Model | VRAM | Disk |
|-------|------|------|
| XTTS v2 | ~4GB | ~2GB |
| Fish Speech 1.5 | ~4GB | ~1.5GB |
| StyleTTS2 | ~2GB | ~1GB |
| Mistral 7B (4-bit) | ~8GB | ~4GB |
| Mistral 7B (fp16) | ~16GB | ~14GB |
| **Total (4-bit)** | **~18GB** | **~10GB** |
| **Total (fp16)** | **~26GB** | **~20GB** |

Recommended: **48GB VRAM** GPU for comfortable headroom

## Troubleshooting

### Out of Memory
- Use 4-bit quantization for Mistral (default)
- Load models lazily (on-demand)
- Reduce max_model_len in vLLM config

### Slow First Request
- Models are loaded on first use
- Pre-load on startup by uncommenting in server.py

### Model Download Fails
- Ensure HF_TOKEN is set for gated models
- Check disk space (50GB+ recommended)
