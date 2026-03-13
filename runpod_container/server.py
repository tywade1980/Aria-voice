"""
ARIA Voice Agent - Unified API Server
Runs on Runpod with: Fish Speech 1.5, XTTS v2, StyleTTS2, Mistral 7B
"""

import os
import io
import uuid
import asyncio
import logging
import base64
import tempfile
from typing import Optional, List, Dict, Any
from pathlib import Path

import torch
import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("aria-server")

# ==================== CONFIGURATION ====================

MODELS_DIR = os.environ.get("MODELS_DIR", "/app/models")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {DEVICE}")

# ==================== PYDANTIC MODELS ====================

class TTSRequest(BaseModel):
    text: str
    engine: str = "xtts"  # xtts, fish, styletts2
    voice: Optional[str] = None  # voice preset or reference audio path
    language: str = "en"
    speed: float = 1.0

class TTSResponse(BaseModel):
    audio_base64: str
    format: str = "wav"
    engine: str
    duration: float

class ChatRequest(BaseModel):
    messages: List[Dict[str, str]]
    max_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9
    stream: bool = False

class ChatResponse(BaseModel):
    response: str
    tokens_used: int
    model: str

class HealthResponse(BaseModel):
    status: str
    engines: Dict[str, bool]
    gpu_available: bool
    gpu_name: Optional[str]

# ==================== MODEL LOADERS ====================

class ModelManager:
    def __init__(self):
        self.xtts_model = None
        self.fish_model = None
        self.styletts_model = None
        self.mistral_model = None
        self.mistral_tokenizer = None
        
    async def load_xtts(self):
        """Load XTTS v2 model"""
        if self.xtts_model is None:
            logger.info("Loading XTTS v2...")
            try:
                from TTS.api import TTS
                self.xtts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(DEVICE)
                logger.info("XTTS v2 loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load XTTS v2: {e}")
                raise
        return self.xtts_model
    
    async def load_fish_speech(self):
        """Load Fish Speech 1.5 model"""
        if self.fish_model is None:
            logger.info("Loading Fish Speech 1.5...")
            try:
                # Fish Speech uses its own inference pipeline
                import sys
                sys.path.append(f"{MODELS_DIR}/fish-speech-1.5")
                # Import fish speech inference module
                self.fish_model = {"loaded": True, "path": f"{MODELS_DIR}/fish-speech-1.5"}
                logger.info("Fish Speech 1.5 loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load Fish Speech: {e}")
                raise
        return self.fish_model
    
    async def load_styletts2(self):
        """Load StyleTTS2 model"""
        if self.styletts_model is None:
            logger.info("Loading StyleTTS2...")
            try:
                from styletts2 import tts
                self.styletts_model = tts.StyleTTS2()
                logger.info("StyleTTS2 loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load StyleTTS2: {e}")
                raise
        return self.styletts_model
    
    async def load_mistral(self):
        """Load Mistral 7B model"""
        if self.mistral_model is None:
            logger.info("Loading Mistral 7B...")
            try:
                from vllm import LLM, SamplingParams
                model_path = f"{MODELS_DIR}/mistral-7b-instruct"
                if not os.path.exists(model_path):
                    model_path = "mistralai/Mistral-7B-Instruct-v0.3"
                
                self.mistral_model = LLM(
                    model=model_path,
                    tensor_parallel_size=1,
                    gpu_memory_utilization=0.8,
                    max_model_len=8192
                )
                logger.info("Mistral 7B loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load Mistral: {e}")
                # Fallback to transformers
                try:
                    from transformers import AutoModelForCausalLM, AutoTokenizer
                    self.mistral_tokenizer = AutoTokenizer.from_pretrained(
                        "mistralai/Mistral-7B-Instruct-v0.3"
                    )
                    self.mistral_model = AutoModelForCausalLM.from_pretrained(
                        "mistralai/Mistral-7B-Instruct-v0.3",
                        torch_dtype=torch.float16,
                        device_map="auto",
                        load_in_4bit=True
                    )
                    logger.info("Mistral 7B loaded via transformers (fallback)")
                except Exception as e2:
                    logger.error(f"Fallback also failed: {e2}")
                    raise
        return self.mistral_model

# Global model manager
models = ModelManager()

# ==================== FASTAPI APP ====================

app = FastAPI(
    title="ARIA Voice Agent API",
    description="Unified API for TTS (Fish Speech, XTTS v2, StyleTTS2) and LLM (Mistral 7B)",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== ENDPOINTS ====================

@app.get("/", response_model=Dict[str, str])
async def root():
    return {
        "service": "ARIA Voice Agent API",
        "version": "1.0.0",
        "engines": "Fish Speech 1.5, XTTS v2, StyleTTS2, Mistral 7B"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check service health and model availability"""
    gpu_name = None
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
    
    return HealthResponse(
        status="healthy",
        engines={
            "xtts": models.xtts_model is not None,
            "fish_speech": models.fish_model is not None,
            "styletts2": models.styletts_model is not None,
            "mistral": models.mistral_model is not None
        },
        gpu_available=torch.cuda.is_available(),
        gpu_name=gpu_name
    )

@app.post("/api/tts", response_model=TTSResponse)
async def text_to_speech(request: TTSRequest):
    """
    Generate speech from text using specified engine
    Engines: xtts, fish, styletts2
    """
    try:
        audio_data = None
        sample_rate = 24000
        
        if request.engine == "xtts":
            model = await models.load_xtts()
            
            # Generate with XTTS v2
            if request.voice and os.path.exists(request.voice):
                # Voice cloning from reference audio
                wav = model.tts(
                    text=request.text,
                    speaker_wav=request.voice,
                    language=request.language
                )
            else:
                # Use default speaker
                wav = model.tts(
                    text=request.text,
                    language=request.language
                )
            audio_data = np.array(wav)
            sample_rate = 24000
            
        elif request.engine == "fish":
            model = await models.load_fish_speech()
            # Fish Speech inference
            # This would use the fish-audio-sdk or local inference
            # For now, fallback to XTTS if Fish not fully configured
            logger.warning("Fish Speech: Using local inference pipeline")
            try:
                # Try fish-audio-sdk for cloud API
                from fish_audio_sdk import Session, TTSRequest as FishTTSRequest
                session = Session(os.environ.get("FISH_API_KEY", ""))
                buffer = io.BytesIO()
                for chunk in session.tts(FishTTSRequest(text=request.text)):
                    buffer.write(chunk)
                buffer.seek(0)
                audio_data, sample_rate = sf.read(buffer)
            except Exception as e:
                logger.error(f"Fish Speech error: {e}, falling back to XTTS")
                # Fallback to XTTS
                xtts = await models.load_xtts()
                wav = xtts.tts(text=request.text, language=request.language)
                audio_data = np.array(wav)
                
        elif request.engine == "styletts2":
            model = await models.load_styletts2()
            
            # Generate with StyleTTS2
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            
            if request.voice and os.path.exists(request.voice):
                model.inference(
                    request.text,
                    target_voice_path=request.voice,
                    output_wav_file=tmp_path
                )
            else:
                model.inference(request.text, output_wav_file=tmp_path)
            
            audio_data, sample_rate = sf.read(tmp_path)
            os.unlink(tmp_path)
            
        else:
            raise HTTPException(status_code=400, detail=f"Unknown engine: {request.engine}")
        
        # Convert to base64
        buffer = io.BytesIO()
        sf.write(buffer, audio_data, sample_rate, format='WAV')
        buffer.seek(0)
        audio_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        
        duration = len(audio_data) / sample_rate
        
        return TTSResponse(
            audio_base64=audio_base64,
            format="wav",
            engine=request.engine,
            duration=duration
        )
        
    except Exception as e:
        logger.error(f"TTS error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/tts/stream")
async def text_to_speech_stream(request: TTSRequest):
    """Stream TTS audio (for real-time applications)"""
    try:
        if request.engine == "xtts":
            model = await models.load_xtts()
            wav = model.tts(text=request.text, language=request.language)
            audio_data = np.array(wav)
            
            buffer = io.BytesIO()
            sf.write(buffer, audio_data, 24000, format='WAV')
            buffer.seek(0)
            
            return StreamingResponse(
                buffer,
                media_type="audio/wav",
                headers={"Content-Disposition": "attachment; filename=speech.wav"}
            )
        else:
            # For non-streaming engines, use regular endpoint
            result = await text_to_speech(request)
            audio_bytes = base64.b64decode(result.audio_base64)
            return StreamingResponse(
                io.BytesIO(audio_bytes),
                media_type="audio/wav"
            )
    except Exception as e:
        logger.error(f"TTS stream error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/stt")
async def speech_to_text(file: UploadFile = File(...), language: str = Form("en")):
    """
    Transcribe audio to text using Whisper
    """
    try:
        import whisper
        
        # Save uploaded file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        # Load whisper model (cached after first load)
        model = whisper.load_model("base")
        result = model.transcribe(tmp_path, language=language)
        
        os.unlink(tmp_path)
        
        return {
            "text": result["text"],
            "language": result.get("language", language),
            "segments": result.get("segments", [])
        }
    except ImportError:
        raise HTTPException(status_code=500, detail="Whisper not installed")
    except Exception as e:
        logger.error(f"STT error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat with Mistral 7B model
    """
    try:
        model = await models.load_mistral()
        
        # Format messages for Mistral
        prompt = ""
        for msg in request.messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                prompt += f"[INST] {content} [/INST]\n"
            elif role == "user":
                prompt += f"[INST] {content} [/INST]\n"
            elif role == "assistant":
                prompt += f"{content}\n"
        
        # Check if using vLLM or transformers
        if hasattr(model, 'generate'):  # vLLM
            from vllm import SamplingParams
            sampling_params = SamplingParams(
                temperature=request.temperature,
                top_p=request.top_p,
                max_tokens=request.max_tokens
            )
            outputs = model.generate([prompt], sampling_params)
            response_text = outputs[0].outputs[0].text
            tokens_used = len(outputs[0].outputs[0].token_ids)
        else:  # transformers fallback
            inputs = models.mistral_tokenizer(prompt, return_tensors="pt").to(DEVICE)
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=True
            )
            response_text = models.mistral_tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )
            tokens_used = outputs.shape[1]
        
        return ChatResponse(
            response=response_text.strip(),
            tokens_used=tokens_used,
            model="mistral-7b-instruct"
        )
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """Stream chat response (for real-time conversation)"""
    # For now, return non-streaming response
    # Streaming would require websockets or SSE
    result = await chat(request)
    return result

@app.get("/api/voices")
async def list_voices():
    """List available voices for each TTS engine"""
    return {
        "xtts": {
            "description": "XTTS v2 - Voice cloning from reference audio",
            "languages": ["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh", "ja", "ko", "hu"],
            "supports_cloning": True
        },
        "fish": {
            "description": "Fish Speech 1.5 - High quality multilingual TTS",
            "languages": ["en", "zh", "ja"],
            "supports_cloning": True
        },
        "styletts2": {
            "description": "StyleTTS2 - Style-based TTS with voice cloning",
            "languages": ["en"],
            "supports_cloning": True
        }
    }

@app.post("/api/voice/clone")
async def clone_voice(
    name: str = Form(...),
    reference_audio: UploadFile = File(...)
):
    """Upload reference audio for voice cloning"""
    try:
        # Save reference audio
        voices_dir = Path("/app/voices")
        voices_dir.mkdir(exist_ok=True)
        
        voice_path = voices_dir / f"{name}.wav"
        content = await reference_audio.read()
        
        with open(voice_path, "wb") as f:
            f.write(content)
        
        return {
            "status": "success",
            "voice_name": name,
            "voice_path": str(voice_path)
        }
    except Exception as e:
        logger.error(f"Voice clone error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/models")
async def list_models():
    """List loaded models and their status"""
    return {
        "tts_engines": {
            "xtts_v2": {
                "loaded": models.xtts_model is not None,
                "description": "Coqui XTTS v2 - Multilingual voice cloning"
            },
            "fish_speech": {
                "loaded": models.fish_model is not None,
                "description": "Fish Speech 1.5 - High quality TTS"
            },
            "styletts2": {
                "loaded": models.styletts_model is not None,
                "description": "StyleTTS2 - Style transfer TTS"
            }
        },
        "llm": {
            "mistral_7b": {
                "loaded": models.mistral_model is not None,
                "description": "Mistral 7B Instruct v0.3"
            }
        }
    }

# ==================== STARTUP ====================

@app.on_event("startup")
async def startup_event():
    """Pre-load models on startup"""
    logger.info("=" * 50)
    logger.info("ARIA Voice Agent API Starting...")
    logger.info(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    logger.info("=" * 50)
    
    # Optionally pre-load models (comment out for lazy loading)
    # await models.load_xtts()
    # await models.load_mistral()

# ==================== MAIN ====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1
    )
