
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import ollama
from config import settings
from extraction.prompts import ENTITY_EXTRACTION_PROMPT

def debug_ollama():
    print("Using official ollama library...")
    print(f"Model: {settings.ollama_model}")
    
    text = "John Doe works at Acme Corp in New York."
    # Simplified prompt
    prompt = f"Extract entities from this text: {text}"
    
    print("Prompt length:", len(prompt))
    print("Sending request to Ollama...")
    
    try:
        stream = ollama.chat(
            model=settings.ollama_model,
            messages=[{'role': 'user', 'content': prompt}],
            stream=True
        )
        print("Response stream started:")
        for chunk in stream:
            print(chunk['message']['content'], end='', flush=True)
        print("\nDone!")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    debug_ollama()
