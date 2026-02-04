import os
import sys
from dotenv import load_dotenv
from langchain_community.graphs import Neo4jGraph

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

def test_graph_init():
    print("Initializing Neo4jGraph with enhanced_schema=False...")
    try:
        graph = Neo4jGraph(
            url=os.getenv("NEO4J_URI"),
            username=os.getenv("NEO4J_USERNAME"),
            password=os.getenv("NEO4J_PASSWORD"),
            enhanced_schema=False
        )
        print("Initialization success!")
        print("Schema:", graph.schema[:100] + "...")
    except Exception as e:
        print(f"Initialization failed: {e}")

    print("\nInitializing Neo4jGraph WITHOUT enhanced_schema flag (default)...")
    try:
        graph = Neo4jGraph(
            url=os.getenv("NEO4J_URI"),
            username=os.getenv("NEO4J_USERNAME"),
            password=os.getenv("NEO4J_PASSWORD")
        )
        print("Default initialization success!")
    except Exception as e:
        print(f"Default initialization failed: {e}")

if __name__ == "__main__":
    test_graph_init()
