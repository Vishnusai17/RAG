"""
API Routes
Endpoints for document ingestion, querying, and graph operations
"""

import os
import shutil
from typing import List, Optional
from pathlib import Path

from fastapi import APIRouter, File, UploadFile, HTTPException, BackgroundTasks
from pydantic import BaseModel

from ingestion.document_loader import DocumentLoader, Document
from ingestion.chunker import DocumentChunker
from extraction.entity_extractor import EntityExtractor
from extraction.relationship_extractor import RelationshipExtractor
from graph.builder import GraphBuilder
from graph.neo4j_client import get_client
from rag.vector_store import get_vector_store
from rag.query_engine import QueryEngine


router = APIRouter()

# Upload directory
UPLOAD_DIR = "./data/uploads"


# Request/Response models
class QueryRequest(BaseModel):
    question: str
    query_type: Optional[str] = None  # relationship, entity, document, hybrid


class QueryResponse(BaseModel):
    answer: str
    query_type: str
    sources: List[dict]
    cypher_query: Optional[str] = None
    confidence: str


class IngestRequest(BaseModel):
    path: str
    recursive: bool = True


class EntityResponse(BaseModel):
    name: str
    type: str
    relationships: List[dict]


class GraphStatsResponse(BaseModel):
    node_count: int
    relationship_count: int
    labels: dict


class PathRequest(BaseModel):
    entity1: str
    entity2: str
    max_hops: int = 4


# Endpoints

@router.post("/query", response_model=QueryResponse)
async def query_graph(request: QueryRequest):
    """
    Query the knowledge graph with natural language.
    Automatically determines the best query strategy.
    """
    try:
        engine = QueryEngine()
        result = engine.query(request.question)
        
        return QueryResponse(
            answer=result.answer,
            query_type=result.query_type.value,
            sources=result.sources[:10] if result.sources else [],
            cypher_query=result.cypher_query,
            confidence=result.confidence
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ingest")
async def ingest_documents(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...)
):
    """
    Upload and ingest documents into the knowledge graph.
    Processing happens in the background.
    """
    uploaded_files = []
    
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    
    for file in files:
        # Save uploaded file
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        uploaded_files.append(file_path)
    
    # Process in background
    background_tasks.add_task(process_documents, uploaded_files)
    
    return {
        "message": f"Processing {len(files)} files",
        "files": [f.filename for f in files],
        "status": "processing"
    }


@router.post("/ingest/path")
async def ingest_from_path(
    request: IngestRequest,
    background_tasks: BackgroundTasks
):
    """
    Ingest documents from a local path.
    """
    if not os.path.exists(request.path):
        raise HTTPException(status_code=404, detail="Path not found")
    
    background_tasks.add_task(process_path, request.path, request.recursive)
    
    return {
        "message": f"Processing documents from {request.path}",
        "status": "processing"
    }


@router.get("/entities")
async def list_entities(
    entity_type: Optional[str] = None,
    limit: int = 50
):
    """
    List entities in the knowledge graph.
    """
    try:
        client = get_client()
        
        if entity_type:
            query = f"""
            MATCH (n:{entity_type})
            RETURN n.name as name, labels(n) as labels
            LIMIT $limit
            """
        else:
            query = """
            MATCH (n)
            WHERE n:Person OR n:Organization OR n:Location
            RETURN n.name as name, labels(n) as labels
            LIMIT $limit
            """
        
        results = client.run_cypher(query, {"limit": limit})
        
        return {
            "entities": [
                {"name": r["name"], "type": r["labels"][0] if r["labels"] else "Unknown"}
                for r in results
            ],
            "count": len(results)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/entities/{name}")
async def get_entity(name: str):
    """
    Get details about a specific entity including relationships.
    """
    try:
        client = get_client()
        relationships = client.get_relationships(name, limit=30)
        
        if not relationships:
            # Try to find the entity at least
            for label in ["Person", "Organization", "Location"]:
                node = client.get_node_by_name(label, name)
                if node:
                    return {
                        "name": name,
                        "type": label,
                        "properties": node,
                        "relationships": []
                    }
            raise HTTPException(status_code=404, detail="Entity not found")
        
        # Get entity type from first relationship
        entity_type = "Unknown"
        if relationships:
            source = relationships[0].get("source", {})
            entity_type = source.get("labels", ["Unknown"])[0] if "labels" in relationships[0] else "Unknown"
        
        return {
            "name": name,
            "type": entity_type,
            "relationships": relationships
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/entities/path")
async def find_path(request: PathRequest):
    """
    Find connection paths between two entities.
    """
    try:
        client = get_client()
        paths = client.get_path_between(
            request.entity1,
            request.entity2,
            max_hops=request.max_hops
        )
        
        return {
            "entity1": request.entity1,
            "entity2": request.entity2,
            "paths": paths,
            "path_count": len(paths)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/graph/stats")
async def get_graph_stats():
    """
    Get statistics about the knowledge graph.
    """
    try:
        client = get_client()
        stats = client.get_graph_stats()
        
        return {
            "node_count": stats.get("nodeCount", 0),
            "relationship_count": stats.get("relCount", 0),
            "labels": stats.get("labels", {}),
            "relationship_types": stats.get("relTypes", {})
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/graph/data")
async def get_graph_data(limit: int = 100):
    """
    Get graph data for visualization.
    Returns nodes and edges in a format suitable for vis.js or D3.
    """
    try:
        client = get_client()
        
        # Get nodes
        nodes_query = """
        MATCH (n)
        WHERE n:Person OR n:Organization OR n:Location
        RETURN id(n) as id, n.name as name, labels(n) as labels
        LIMIT $limit
        """
        nodes = client.run_cypher(nodes_query, {"limit": limit})
        
        # Get edges
        edges_query = """
        MATCH (a)-[r]->(b)
        WHERE (a:Person OR a:Organization OR a:Location)
          AND (b:Person OR b:Organization OR b:Location)
        RETURN id(a) as source, id(b) as target, type(r) as type
        LIMIT $limit
        """
        edges = client.run_cypher(edges_query, {"limit": limit * 2})
        
        return {
            "nodes": [
                {
                    "id": n["id"],
                    "label": n["name"],
                    "group": n["labels"][0] if n["labels"] else "Unknown"
                }
                for n in nodes
            ],
            "edges": [
                {
                    "from": e["source"],
                    "to": e["target"],
                    "label": e["type"]
                }
                for e in edges
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/graph/clear")
async def clear_graph():
    """
    Clear all data from the graph. USE WITH CAUTION.
    """
    try:
        client = get_client()
        success = client.clear_graph()
        
        if success:
            return {"message": "Graph cleared successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to clear graph")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Background processing functions

async def process_documents(file_paths: List[str]):
    """Process uploaded documents in the background"""
    loader = DocumentLoader()
    chunker = DocumentChunker()
    entity_extractor = EntityExtractor()
    relationship_extractor = RelationshipExtractor()
    graph_builder = GraphBuilder()
    vector_store = get_vector_store()
    
    # Initialize schema
    graph_builder.initialize()
    
    for file_path in file_paths:
        try:
            # Load document
            doc = loader.load_file(file_path)
            if not doc:
                continue
            
            # Extract entities
            entities = entity_extractor.extract_entities(
                doc.content,
                doc_id=doc.id,
                doc_type=doc.doc_type
            )
            
            # Extract relationships
            relationships = relationship_extractor.extract_relationships(
                doc.content,
                entities,
                doc_id=doc.id,
                doc_type=doc.doc_type
            )
            
            # Build graph
            graph_builder.build_from_extraction(doc, entities, relationships)
            
            # Add to vector store
            chunks = chunker.chunk_document(doc)
            vector_store.add_chunks(chunks)
            
            print(f"Processed: {file_path}")
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")


async def process_path(path: str, recursive: bool):
    """Process documents from a path in the background"""
    loader = DocumentLoader()
    
    if os.path.isfile(path):
        docs = [loader.load_file(path)]
    else:
        docs = loader.load_directory(path, recursive=recursive)
    
    # Process each document
    file_paths = [doc.source for doc in docs if doc]
    await process_documents(file_paths)
