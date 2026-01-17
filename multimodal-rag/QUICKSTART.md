# LectureChat - Quick Start Guide

## Prerequisites
- Python 3.9+
- Ollama installed and running

## Installation (One Command)

```bash
./setup.sh
```

This will:
1. Check Python and Ollama
2. Create virtual environment
3. Install all dependencies
4. Set up .env file

## Manual Installation

```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Copy environment file
cp .env.example .env
```

## Start Ollama

```bash
# Terminal 1: Start Ollama service
ollama serve

# Terminal 2: Pull model (one time only)
ollama pull mistral
```

## Run LectureChat

```bash
# Activate virtual environment
source venv/bin/activate

# Start app
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Usage

1. **Upload video**: Choose YouTube URL or upload MP4
2. **Process**: Click "Process Video" and wait (~5-15 min)
3. **Chat**: Ask questions about the video content

## Example Questions

- "What is the main topic of this lecture?"
- "Can you explain what happens at 5:30?"
- "What does the diagram on slide 3 show?"
- "Summarize the first 10 minutes"

## Troubleshooting

### Ollama not running
```bash
ollama serve
```

### Virtual environment issues
```bash
deactivate
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Dependencies fail to install
```bash
pip install --upgrade pip
pip install -r requirements.txt --no-cache-dir
```

## Configuration

Edit `.env` to customize:
- `WHISPER_MODEL`: Change to `tiny` for faster processing
- `SCENE_THRESHOLD`: Lower (20.0) for more slides, higher (30.0) for fewer
- `OLLAMA_MODEL`: Switch between `mistral` and `llama3`

## Performance Tips

- **GPU**: Install CUDA-enabled PyTorch for faster processing
- **RAM**: Close other apps if processing large videos
- **Storage**: Clear `data/` folder periodically to save space

## File Locations

- **Videos**: `data/videos/`
- **Frames**: `data/frames/`
- **Database**: `data/chroma_db/`

To start fresh, delete the `data/` folder.
