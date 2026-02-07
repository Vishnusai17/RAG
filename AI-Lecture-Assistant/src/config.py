"""Configuration management for LectureChat."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base directories
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
VIDEOS_DIR = DATA_DIR / "videos"
AUDIO_DIR = DATA_DIR / "audio"
FRAMES_DIR = DATA_DIR / "frames"
CHROMA_DIR = DATA_DIR / "chroma_db"

# Model settings
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
CLIP_MODEL = os.getenv("CLIP_MODEL", "ViT-L-14")

# Processing settings
SCENE_THRESHOLD = float(os.getenv("SCENE_THRESHOLD", "27.0"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.90"))
MAX_FRAMES_PER_HOUR = int(os.getenv("MAX_FRAMES_PER_HOUR", "50"))

# Retrieval settings
TOP_K_TEXT = int(os.getenv("TOP_K_TEXT", "3"))
TOP_K_IMAGE = int(os.getenv("TOP_K_IMAGE", "1"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))

# Ensure directories exist
for directory in [VIDEOS_DIR, AUDIO_DIR, FRAMES_DIR, CHROMA_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
