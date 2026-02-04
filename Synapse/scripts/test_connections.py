
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings
from neo4j import GraphDatabase
from langchain_community.chat_models import ChatOllama
from langchain.schema import HumanMessage

def test_neo4j():
    print(f"Testing Neo4j connection...")
    print(f"URI: {settings.neo4j_uri}")
    print(f"User: {settings.neo4j_username}")
    print(f"Password: {'*' * len(settings.neo4j_password)}")
    
    try:
        driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_username, settings.neo4j_password)
        )
        driver.verify_connectivity()
        print("✅ Neo4j connection successful!")
        driver.close()
        return True
    except Exception as e:
        print(f"❌ Neo4j connection failed: {e}")
        return False

def test_ollama():
    print(f"\nTesting Ollama connection...")
    print(f"URL: {settings.ollama_base_url}")
    print(f"Model: {settings.ollama_model}")
    
    try:
        llm = ChatOllama(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model
        )
        response = llm.invoke([HumanMessage(content="Hi")])
        print(f"✅ Ollama connection successful! Response: {response.content}")
        return True
    except Exception as e:
        print(f"❌ Ollama connection failed: {e}")
        return False

if __name__ == "__main__":
    neo4j_ok = test_neo4j()
    ollama_ok = test_ollama()
    
    if neo4j_ok and ollama_ok:
        sys.exit(0)
    else:
        sys.exit(1)
