# LectureChat - Multimodal RAG Video Assistant

A Python-based application that allows you to chat with lecture videos using multimodal RAG (Retrieval-Augmented Generation). Upload a video or provide a YouTube URL, and ask questions about both the spoken content and visual slides.

## ✨ Features

- **🎬 Smart Slide Extraction**: Automatically detects unique slides using AdaptiveDetector with perceptual hash deduplication
- **🎙️ Audio Transcription**: Whisper-powered transcription with word-level timestamps
- **🖼️ Visual Understanding**: CLIP (ViT-L/14) embeddings for semantic image search
- **🤖 Local LLM**: Privacy-first generation using Ollama (Mistral/Llama3)
- **💬 Multimodal Chat**: Ask questions that combine audio and visual context
- **📊 Vector Storage**: ChromaDB for persistent multimodal indexing

## 🏗️ Architecture

```
Input (YouTube/MP4) 
    ↓
Video Processing (SceneDetector + Whisper)
    ↓
Multimodal Indexing (ChromaDB)
    ├─ Text Embeddings (sentence-transformers)
    └─ Image Embeddings (CLIP ViT-L/14)
    ↓
Retrieval (Hybrid text + image)
    ↓
Generation (Ollama LLM)
    ↓
Chat Response with Sources
```

## 📋 Prerequisites

### 1. Install Ollama

```bash
# macOS
brew install ollama

# Start Ollama service
ollama serve

# Pull a model
ollama pull mistral
# or
ollama pull llama3
```

### 2. Python 3.10+

Check your Python version:
```bash
python --version
```

## 🚀 Installation

1. **Clone or navigate to the project directory:**
```bash
cd "/Users/vishnu/Projects/Anti-P/Agentic-system/RAG/Multimodal Rag"
```

2. **Create a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables (optional):**
```bash
cp .env.example .env
# Edit .env if you want to customize settings
```

## 🎯 Usage

1. **Start the application:**
```bash
streamlit run app.py
```

2. **In the web interface:**
   - Choose **YouTube URL** or **Upload MP4** in the sidebar
   - Click **Process Video** (this will take a few minutes)
   - Once processing is complete, start chatting!

3. **Example questions:**
   - "What is the main topic of this lecture?"
   - "Can you explain the diagram shown at 5:30?"
   - "Summarize the key points from the introduction"
   - "What does the slide about neural networks show?"

## 📁 Project Structure

```
.
├── app.py                 # Streamlit UI
├── requirements.txt       # Python dependencies
├── .env.example          # Environment variables template
├── src/
│   ├── config.py         # Configuration management
│   ├── utils.py          # Helper functions
│   ├── video_processor.py # Video ingestion & smart scene detection
│   ├── transcription.py  # Whisper audio transcription
│   ├── embedding.py      # CLIP image embeddings
│   └── rag_engine.py     # Multimodal RAG pipeline
└── data/                 # Auto-created storage
    ├── videos/
    ├── audio/
    ├── frames/
    └── chroma_db/
```

## ⚙️ Configuration

Default settings in `.env`:

```bash
# Ollama
OLLAMA_MODEL=mistral
OLLAMA_BASE_URL=http://localhost:11434

# Models
WHISPER_MODEL=base
CLIP_MODEL=ViT-L-14

# Processing
SCENE_THRESHOLD=27.0
SIMILARITY_THRESHOLD=0.90
MAX_FRAMES_PER_HOUR=50

# Retrieval
TOP_K_TEXT=3
TOP_K_IMAGE=1
CHUNK_SIZE=500
```

## 🔧 How It Works

### 1. Video Processing
- **Download/Upload**: Accepts YouTube URLs (via `yt-dlp`) or direct MP4 uploads
- **Audio Extraction**: Extracts audio to WAV using `moviepy`
- **Smart Frame Extraction**:
  - Uses `AdaptiveDetector(threshold=27.0)` to detect scene changes
  - Extracts middle frame of each scene to avoid blur
  - Applies perceptual hash deduplication (>90% similarity filtered out)
  - Results in 20-50 unique slides per hour

### 2. Multimodal Indexing
- **Text**: Whisper transcription → chunked → embedded with sentence-transformers → ChromaDB
- **Images**: CLIP ViT-L/14 embeddings → ChromaDB

### 3. Retrieval
- Query triggers two parallel searches:
  - **Top 3 text chunks** (semantic search on transcription)
  - **Top 1 image** (CLIP text-to-image similarity)

### 4. Generation
- Combined context sent to Ollama
- Response includes timestamps and relevant slide images

## 🎨 UI Features

- **Split-screen layout**: Video player on left, chat on right
- **Source attribution**: Every answer shows relevant timestamps and slides
- **Visual preview**: Extracted slides displayed inline with responses
- **Session management**: Clear and restart functionality

## 🐛 Troubleshooting

### Ollama not running
```bash
ollama serve
```

### CUDA/GPU errors
The app works on CPU, but will be slower. To use GPU, ensure PyTorch CUDA is installed:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Memory issues
Reduce the Whisper model size in `.env`:
```bash
WHISPER_MODEL=tiny  # or base
```

### Scene detection finds too few slides
Lower the threshold in `.env`:
```bash
SCENE_THRESHOLD=20.0
```

## 📝 License

MIT License - feel free to use for your projects!

## 🙏 Acknowledgments

- **OpenAI Whisper** for transcription
- **OpenCLIP** for vision embeddings
- **scenedetect** for smart slide extraction
- **LlamaIndex** for RAG orchestration
- **ChromaDB** for vector storage
- **Ollama** for local LLM inference
