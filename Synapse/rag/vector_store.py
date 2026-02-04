"""
Vector Store Module
FAISS-based vector store for semantic search
"""

import os
from typing import List, Dict, Any, Optional
from pathlib import Path

from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document as LCDocument

from config import settings
from ingestion.document_loader import Document
from ingestion.chunker import Chunk


class VectorStore:
    """FAISS-based vector store for document embeddings"""
    
    def __init__(
        self,
        store_path: str = None,
        embedding_model: str = None
    ):
        self.store_path = store_path or settings.vector_store_path
        self.embedding_model = embedding_model or settings.embedding_model
        
        
        # Initialize embeddings
        # Initialize embeddings
        if os.getenv("SKIP_VECTORS", "false").lower() == "true":
            self.embeddings = None
            print("⚠️ Vector store disabled (SKIP_VECTORS=true)")
            return

        if settings.llm_provider == "ollama":
            # Use local HuggingFace embeddings for Ollama setup (faster/reliable)
            try:
                self.embeddings = HuggingFaceEmbeddings(
                    model_name=self.embedding_model or "sentence-transformers/all-MiniLM-L6-v2",
                    model_kwargs={'device': 'cpu'}
                )
            except Exception as e:
                print(f"⚠️ Failed to initialize HuggingFace embeddings: {e}")
                self.embeddings = None
        else:
            self.embeddings = OpenAIEmbeddings(
                api_key=settings.openai_api_key,
                model=self.embedding_model
            )
        
        self._store: Optional[FAISS] = None
    
    @property
    def store(self) -> Optional[FAISS]:
        """Lazy load the vector store"""
        if self.embeddings is None:
            return None
            
        if self._store is None and os.path.exists(self.store_path):
            try:
                self._store = FAISS.load_local(
                    self.store_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
            except Exception as e:
                print(f"Warning: Could not load vector store: {e}")
                
        return self._store
    
    def add_documents(self, documents: List[Document]) -> int:
        """Add documents to the vector store"""
        if self.embeddings is None:
            return 0
            
        lc_docs = []
        for doc in documents:
            lc_docs.append(LCDocument(
                page_content=doc.content,
                metadata={
                    "doc_id": doc.id,
                    "source": doc.source,
                    "doc_type": doc.doc_type,
                    **doc.metadata
                }
            ))
        
        return self._add_lc_documents(lc_docs)
    
    def add_chunks(self, chunks: List[Chunk]) -> int:
        """Add document chunks to the vector store"""
        if self.embeddings is None:
            return 0
            
        lc_docs = []
        for chunk in chunks:
            lc_docs.append(LCDocument(
                page_content=chunk.content,
                metadata={
                    "chunk_id": chunk.id,
                    "source_doc_id": chunk.source_doc_id,
                    "chunk_index": chunk.chunk_index,
                    **chunk.metadata
                }
            ))
        
        return self._add_lc_documents(lc_docs)
    
    def _add_lc_documents(self, documents: List[LCDocument]) -> int:
        """Add LangChain documents to the store"""
        if not documents or self.embeddings is None:
            return 0
        
        if self._store is None:
            # Create new store
            self._store = FAISS.from_documents(documents, self.embeddings)
        else:
            # Add to existing store
            self._store.add_documents(documents)
        
        # Save the store
        self.save()
        
        return len(documents)
    
    def search(
        self,
        query: str,
        k: int = 5,
        filter_dict: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        if not self.store:
            return []
        
        # Perform similarity search
        if filter_dict:
            results = self.store.similarity_search_with_score(
                query,
                k=k,
                filter=filter_dict
            )
        else:
            results = self.store.similarity_search_with_score(query, k=k)
        
        return [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": score
            }
            for doc, score in results
        ]
    
    def search_by_embedding(
        self,
        embedding: List[float],
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """Search by embedding vector"""
        if not self.store:
            return []
        
        results = self.store.similarity_search_by_vector(embedding, k=k)
        return [
            {
                "content": doc.page_content,
                "metadata": doc.metadata
            }
            for doc in results
        ]
    
    def save(self):
        """Save the vector store to disk"""
        if self._store:
            Path(self.store_path).parent.mkdir(parents=True, exist_ok=True)
            self._store.save_local(self.store_path)
    
    def load(self) -> bool:
        """Load the vector store from disk"""
        if os.path.exists(self.store_path):
            self._store = FAISS.load_local(
                self.store_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            return True
        return False
    
    def get_retriever(self, k: int = 5):
        """Get a LangChain retriever"""
        if self.store:
            return self.store.as_retriever(search_kwargs={"k": k})
        return None


# Global instance
_vector_store: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    """Get the global vector store instance"""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store
