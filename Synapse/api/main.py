"""
FastAPI Application
Main API endpoints for GraphRAG
"""

import os
import shutil
from typing import List, Optional
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from config import settings
from api.routes import router as api_router


# Create FastAPI app
app = FastAPI(
    title="GraphRAG API",
    description="Knowledge Graph powered RAG for investigative document analysis",
    version="1.0.0"
)

# Include API routes
app.include_router(api_router, prefix="/api")

# Serve frontend static files
frontend_path = Path(__file__).parent.parent / "frontend"
if frontend_path.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")


@app.get("/")
async def root():
    """Serve the main frontend page"""
    index_path = frontend_path / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "GraphRAG API is running", "docs": "/docs"}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "version": "1.0.0"}


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    print("GraphRAG API starting up...")
    
    # Create data directories
    os.makedirs("./data/uploads", exist_ok=True)
    os.makedirs("./data/vector_store", exist_ok=True)


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("GraphRAG API shutting down...")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True
    )
