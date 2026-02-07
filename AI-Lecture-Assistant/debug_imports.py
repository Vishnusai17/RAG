import sys
print("1. Testing imports...")

try:
    print("- Importing logging...")
    import logging
    
    print("- Importing sqlite3...")
    import sqlite3
    print(f"  sqlite3 version: {sqlite3.sqlite_version}")
    
    print("- Importing chromadb...")
    import chromadb
    print(f"  chromadb version: {chromadb.__version__}")
    
    print("- Importing llama_index...")
    import llama_index.core
    print("  llama_index imported")
    
    print("- Importing torch...")
    import torch
    print(f"  torch version: {torch.__version__}")
    
    print("- Importing CLIP...")
    import open_clip
    print("  open_clip imported")
    
    print("- Importing whisper...")
    import whisper
    print("  whisper imported")
    
    print("✅ All core imports successful!")
    
except Exception as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)
