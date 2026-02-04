#!/usr/bin/env python3
"""
CLI Script for Document Ingestion
Processes documents and populates the knowledge graph
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ingestion.document_loader import DocumentLoader
from ingestion.chunker import DocumentChunker
from extraction.entity_extractor import EntityExtractor
from extraction.relationship_extractor import RelationshipExtractor
from graph.builder import GraphBuilder
from graph.neo4j_client import get_client
from rag.vector_store import get_vector_store


def main():
    parser = argparse.ArgumentParser(
        description="Ingest documents into the GraphRAG knowledge graph"
    )
    parser.add_argument(
        "--path", "-p",
        required=True,
        help="Path to document or directory to ingest"
    )
    parser.add_argument(
        "--recursive", "-r",
        action="store_true",
        default=True,
        help="Recursively process directories (default: True)"
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable LLM-based extraction (faster but less accurate)"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode - process but don't write to database"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--skip-vectors",
        action="store_true",
        help="Skip vector storage (useful if embeddings hang)"
    )
    
    args = parser.parse_args()
    
    # Validate path
    path = Path(args.path)
    if not path.exists():
        print(f"Error: Path not found: {args.path}")
        sys.exit(1)
    
    print(f"📂 Loading documents from: {args.path}")
    
    # Initialize components
    loader = DocumentLoader()
    chunker = DocumentChunker()
    entity_extractor = EntityExtractor(use_llm=not args.no_llm)
    relationship_extractor = RelationshipExtractor()
    graph_builder = GraphBuilder()
    
    # Only initialize vector store if needed
    vector_store = None
    if not args.skip_vectors:
        vector_store = get_vector_store()
    
    # Initialize schema
    if not args.test:
        print("🔧 Setting up graph schema...")
        graph_builder.initialize()
    
    # Load documents
    if path.is_file():
        documents = [loader.load_file(str(path))]
    else:
        documents = loader.load_directory(str(path), recursive=args.recursive)
    
    documents = [d for d in documents if d is not None]
    print(f"📄 Found {len(documents)} documents")
    
    if not documents:
        print("No documents to process.")
        sys.exit(0)
    
    # Process each document
    total_stats = {
        "documents": 0,
        "entities": 0,
        "relationships": 0,
        "chunks": 0
    }
    
    for i, doc in enumerate(documents, 1):
        print(f"\n[{i}/{len(documents)}] Processing: {doc.source}")
        
        try:
            # Extract entities
            if args.verbose:
                print("  Extracting entities...")
            entities = entity_extractor.extract_entities(
                doc.content,
                doc_id=doc.id,
                doc_type=doc.doc_type
            )
            if args.verbose:
                print(f"  Found {len(entities)} entities")
                for entity in entities[:5]:
                    print(f"    - {entity.name} ({entity.entity_type.value})")
            
            # Extract relationships
            if args.verbose:
                print("  Extracting relationships...")
            relationships = relationship_extractor.extract_relationships(
                doc.content,
                entities,
                doc_id=doc.id,
                doc_type=doc.doc_type
            )
            if args.verbose:
                print(f"  Found {len(relationships)} relationships")
            
            # Build graph
            if not args.test:
                if args.verbose:
                    print("  Building graph...")
                stats = graph_builder.build_from_extraction(doc, entities, relationships)
                total_stats["documents"] += stats["documents"]
                total_stats["entities"] += stats["entities"]
                total_stats["relationships"] += stats["relationships"]
            
            # Add to vector store
            if not args.test and not args.skip_vectors:
                if args.verbose:
                    print("  Adding to vector store...")
                chunks = chunker.chunk_document(doc)
                vector_store.add_chunks(chunks)
                total_stats["chunks"] += len(chunks)
            
            print(f"  ✓ Processed: {len(entities)} entities, {len(relationships)} relationships")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            continue
    
    # Summary
    print("\n" + "="*50)
    print("📊 Ingestion Summary")
    print("="*50)
    print(f"Documents processed: {total_stats['documents']}")
    print(f"Entities created:    {total_stats['entities']}")
    print(f"Relationships:       {total_stats['relationships']}")
    print(f"Vector chunks:       {total_stats['chunks']}")
    
    if args.test:
        print("\n⚠️  Test mode - no data was written to database")
    else:
        print("\n✅ Ingestion complete!")


if __name__ == "__main__":
    main()
