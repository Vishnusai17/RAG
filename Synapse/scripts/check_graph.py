
from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

load_dotenv()

URI = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
USER = os.getenv("NEO4J_USERNAME", "neo4j")
PASSWORD = os.getenv("NEO4J_PASSWORD", "secret123")

def check_graph():
    driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
    with driver.session() as session:
        result = session.run("MATCH (n) RETURN count(n) as count")
        count = result.single()["count"]
        print(f"Total Nodes: {count}")
        
        if count > 0:
            print("\nRecent Nodes:")
            result = session.run("MATCH (n) RETURN labels(n) as labels, n.name as name LIMIT 5")
            for record in result:
                print(f"- {record['labels']}: {record['name']}")
                
    driver.close()

if __name__ == "__main__":
    check_graph()
