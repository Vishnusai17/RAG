#!/usr/bin/env python3
"""
CLI Script for Querying the Knowledge Graph
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.query_engine import QueryEngine
from graph.neo4j_client import get_client


def main():
    parser = argparse.ArgumentParser(
        description="Query the GraphRAG knowledge graph"
    )
    parser.add_argument(
        "--query", "-q",
        help="Natural language query"
    )
    parser.add_argument(
        "--path", "-p",
        nargs=2,
        metavar=("ENTITY1", "ENTITY2"),
        help="Find path between two entities"
    )
    parser.add_argument(
        "--entity", "-e",
        help="Get information about an entity"
    )
    parser.add_argument(
        "--list-entities",
        action="store_true",
        help="List all entities in the graph"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show graph statistics"
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Interactive query mode"
    )
    
    args = parser.parse_args()
    
    client = get_client()
    
    # Check connection
    if not client.verify_connection():
        print("❌ Failed to connect to Neo4j. Check your configuration.")
        sys.exit(1)
    
    if args.stats:
        show_stats(client)
    elif args.list_entities:
        list_entities(client)
    elif args.entity:
        show_entity(client, args.entity)
    elif args.path:
        find_path(client, args.path[0], args.path[1])
    elif args.query:
        execute_query(args.query)
    elif args.interactive:
        interactive_mode()
    else:
        parser.print_help()


def show_stats(client):
    """Display graph statistics"""
    print("\n📊 Graph Statistics")
    print("="*40)
    
    stats = client.get_graph_stats()
    print(f"Total nodes:         {stats.get('nodeCount', 0)}")
    print(f"Total relationships: {stats.get('relCount', 0)}")
    
    if 'labels' in stats:
        print("\nNode Types:")
        for label, count in stats['labels'].items():
            print(f"  {label}: {count}")
    
    if 'relTypes' in stats:
        print("\nRelationship Types:")
        for rel_type, count in stats['relTypes'].items():
            print(f"  {rel_type}: {count}")


def list_entities(client, limit=50):
    """List all entities"""
    print("\n📋 Entities in Graph")
    print("="*40)
    
    for entity_type in ["Person", "Organization", "Location"]:
        query = f"""
        MATCH (n:{entity_type})
        RETURN n.name as name
        ORDER BY n.name
        LIMIT {limit}
        """
        results = client.run_cypher(query)
        
        if results:
            print(f"\n{entity_type}s:")
            for r in results:
                print(f"  • {r['name']}")


def show_entity(client, name):
    """Show entity details and relationships"""
    print(f"\n🔍 Entity: {name}")
    print("="*40)
    
    relationships = client.get_relationships(name, limit=20)
    
    if not relationships:
        print("Entity not found or has no relationships.")
        return
    
    # Group by relationship type
    grouped = {}
    for rel in relationships:
        rel_type = rel['relationship_type']
        if rel_type not in grouped:
            grouped[rel_type] = []
        grouped[rel_type].append(rel['target']['name'])
    
    for rel_type, targets in grouped.items():
        print(f"\n{rel_type}:")
        for target in targets:
            print(f"  → {target}")


def find_path(client, entity1, entity2):
    """Find paths between two entities"""
    print(f"\n🔗 Finding paths: {entity1} ↔ {entity2}")
    print("="*40)
    
    paths = client.get_path_between(entity1, entity2, max_hops=4)
    
    if not paths:
        print("No paths found between these entities.")
        return
    
    for i, path in enumerate(paths, 1):
        print(f"\nPath {i}:")
        nodes = path['nodes']
        rels = path['relationships']
        
        for j, node in enumerate(nodes):
            print(f"  [{node['labels'][0]}] {node['name']}")
            if j < len(rels):
                print(f"    ↓ {rels[j]['type']}")


def execute_query(question):
    """Execute a natural language query"""
    print(f"\n❓ Question: {question}")
    print("="*40)
    
    engine = QueryEngine()
    result = engine.query(question)
    
    print(f"\n📝 Answer:")
    print(result.answer)
    
    if result.cypher_query:
        print(f"\n🔧 Cypher Query:")
        print(result.cypher_query)
    
    print(f"\n📊 Query Type: {result.query_type.value}")
    print(f"🎯 Confidence: {result.confidence}")


def interactive_mode():
    """Interactive query mode"""
    print("\n🎯 GraphRAG Interactive Query Mode")
    print("="*40)
    print("Type your questions, or 'quit' to exit.\n")
    
    engine = QueryEngine()
    
    while True:
        try:
            question = input("📝 > ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not question:
                continue
            
            result = engine.query(question)
            
            print(f"\n💡 {result.answer}\n")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    main()
