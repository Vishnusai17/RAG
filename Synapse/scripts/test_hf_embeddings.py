
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from langchain_huggingface import HuggingFaceEmbeddings
from config import settings

def test_hf_embeddings():
    print(f"Testing embeddings with model: {settings.embedding_model}")
    embeddings = HuggingFaceEmbeddings(
        model_name=settings.embedding_model or "sentence-transformers/all-MiniLM-L6-v2"
    )
    
    text = "John Doe works at Acme Corp."
    print("Generating embedding...")
    try:
        vec = embeddings.embed_query(text)
        print(f"✅ Success! Vector length: {len(vec)}")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_hf_embeddings()
