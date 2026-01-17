#!/bin/bash

# Setup script for LectureChat

echo "🎓 LectureChat Setup Script"
echo "============================"
echo ""

# Check Python version
echo "1️⃣ Checking Python version..."
python3 --version

if [ $? -ne 0 ]; then
    echo "❌ Python 3 not found. Please install Python 3.10 or higher."
    exit 1
fi

# Check Ollama
echo ""
echo "2️⃣ Checking Ollama installation..."
if command -v ollama &> /dev/null; then
    echo "✅ Ollama found: $(which ollama)"
    
    echo ""
    echo "Checking if Ollama is running..."
    if curl -s http://localhost:11434/api/tags &> /dev/null; then
        echo "✅ Ollama service is running"
        
        echo ""
        echo "Installed models:"
        ollama list
        
        # Check for required models
        if ollama list | grep -q "mistral\|llama"; then
            echo "✅ Compatible model found"
        else
            echo "⚠️  No compatible model found. Installing mistral..."
            ollama pull mistral
        fi
    else
        echo "⚠️  Ollama is installed but not running."
        echo "Please start Ollama with: ollama serve"
    fi
else
    echo "❌ Ollama not found. Please install Ollama:"
    echo "   brew install ollama"
    echo "   Then run: ollama serve"
    echo "   And pull a model: ollama pull mistral"
    exit 1
fi

# Create virtual environment
echo ""
echo "3️⃣ Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo ""
echo "4️⃣ Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo ""
echo "5️⃣ Installing dependencies (this may take a few minutes)..."
pip install --upgrade pip
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✅ Dependencies installed successfully"
else
    echo "❌ Failed to install dependencies"
    exit 1
fi

# Copy .env file
echo ""
echo "6️⃣ Setting up environment variables..."
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "✅ Created .env file from template"
else
    echo "✅ .env file already exists"
fi

# Final instructions
echo ""
echo "✅ Setup complete!"
echo ""
echo "🚀 To start LectureChat:"
echo "   1. Make sure Ollama is running: ollama serve"
echo "   2. Activate virtual environment: source venv/bin/activate"
echo "   3. Run the app: streamlit run app.py"
echo ""
echo "📖 See README.md for detailed usage instructions."
