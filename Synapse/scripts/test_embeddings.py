
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from langchain_community.embeddings import OllamaEmbeddings
from config import settings

def test_embeddings():
    print(f"Testing embeddings with model: {settings.ollama_model}")
    embeddings = OllamaEmbeddings(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model
    )
    
    text = "John Doe works at Acme Corp."
    print("Generating embedding...")
    try:
        vec = embeddings.embed_query(text)
        print(f"✅ Success! Vector length: {len(vec)}")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_embeddings()
