"""Multimodal RAG engine with ChromaDB and Ollama."""

import chromadb
from chromadb.config import Settings
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings as LlamaSettings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path

from src.config import (
    CHROMA_DIR, OLLAMA_MODEL, OLLAMA_BASE_URL,
    TOP_K_TEXT, TOP_K_IMAGE
)
from src.embedding import ImageEmbedder
from src.utils import format_timestamp


class MultimodalIndex:
    """
    Multimodal vector index for text and image embeddings.
    Uses ChromaDB for storage and retrieval.
    """
    
    def __init__(self, video_id: str):
        """
        Initialize multimodal index.
        
        Args:
            video_id: Unique video ID for this session
        """
        self.video_id = video_id
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        
        # Create or get collections
        self.text_collection = self.client.get_or_create_collection(
            name=f"text_{video_id}",
            metadata={"hnsw:space": "cosine"}
        )
        
        self.image_collection = self.client.get_or_create_collection(
            name=f"images_{video_id}",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Initialize embedders
        self.text_embedder = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.image_embedder = ImageEmbedder()
        
        print(f"Initialized multimodal index for video {video_id}")
    
    def build_index(self, transcription_chunks: List[Dict], 
                   frame_data: List[Tuple[str, float]]):
        """
        Build multimodal index from transcription and frames.
        
        Args:
            transcription_chunks: List of dicts with 'text', 'start_time', 'end_time'
            frame_data: List of (frame_path, timestamp) tuples
        """
        print("Building multimodal index...")
        
        # Index text chunks
        if transcription_chunks:
            texts = [chunk['text'] for chunk in transcription_chunks]
            text_ids = [f"text_{i}" for i in range(len(texts))]
            
            # Generate embeddings
            text_embeddings = [
                self.text_embedder.get_text_embedding(text)
                for text in texts
            ]
            
            # Create metadata
            metadatas = [
                {
                    'start_time': chunk['start_time'],
                    'end_time': chunk['end_time'],
                    'timestamp_str': format_timestamp(chunk['start_time'])
                }
                for chunk in transcription_chunks
            ]
            
            # Add to collection
            self.text_collection.add(
                ids=text_ids,
                embeddings=text_embeddings,
                documents=texts,
                metadatas=metadatas
            )
            
            print(f"Indexed {len(texts)} text chunks")
        
        # Index images
        if frame_data:
            frame_paths = [path for path, _ in frame_data]
            timestamps = [ts for _, ts in frame_data]
            
            # Generate CLIP embeddings
            image_embeddings = self.image_embedder.embed_images_batch(frame_paths)
            
            # Create IDs and metadata
            image_ids = [f"frame_{i}" for i in range(len(frame_paths))]
            metadatas = [
                {
                    'frame_path': path,
                    'timestamp': ts,
                    'timestamp_str': format_timestamp(ts)
                }
                for path, ts in frame_data
            ]
            
            # Add to collection (ChromaDB expects list of lists for embeddings)
            self.image_collection.add(
                ids=image_ids,
                embeddings=image_embeddings.tolist(),
                metadatas=metadatas
            )
            
            print(f"Indexed {len(frame_paths)} images")
        
        print("Multimodal index built successfully")
    
    def retrieve(self, query: str, top_k_text: int = TOP_K_TEXT, 
                top_k_image: int = TOP_K_IMAGE) -> Dict:
        """
        Retrieve relevant context for a query.
        
        Args:
            query: User question
            top_k_text: Number of text chunks to retrieve
            top_k_image: Number of images to retrieve
            
        Returns:
            Dict with 'text_contexts' and 'image_contexts'
        """
        # Retrieve text chunks
        text_embedding = self.text_embedder.get_text_embedding(query)
        
        text_results = self.text_collection.query(
            query_embeddings=[text_embedding],
            n_results=top_k_text
        )
        
        text_contexts = []
        if text_results['documents'] and text_results['documents'][0]:
            for i, doc in enumerate(text_results['documents'][0]):
                metadata = text_results['metadatas'][0][i]
                text_contexts.append({
                    'text': doc,
                    'start_time': metadata['start_time'],
                    'end_time': metadata['end_time'],
                    'timestamp_str': metadata['timestamp_str']
                })
        
        # Retrieve images using CLIP text-to-image similarity
        image_embedding = self.image_embedder.embed_text(query)
        
        image_results = self.image_collection.query(
            query_embeddings=[image_embedding.tolist()],
            n_results=top_k_image
        )
        
        image_contexts = []
        if image_results['metadatas'] and image_results['metadatas'][0]:
            for metadata in image_results['metadatas'][0]:
                image_contexts.append({
                    'frame_path': metadata['frame_path'],
                    'timestamp': metadata['timestamp'],
                    'timestamp_str': metadata['timestamp_str']
                })
        
        return {
            'text_contexts': text_contexts,
            'image_contexts': image_contexts
        }


class LLMEngine:
    """LLM generation engine using Ollama."""
    
    def __init__(self):
        """Initialize Ollama LLM."""
        self.setup_ollama()
    
    def setup_ollama(self, model_name: str = OLLAMA_MODEL):
        """
        Initialize Ollama LLM via LlamaIndex.
        
        Args:
            model_name: Ollama model name (e.g., 'mistral', 'llama3')
        """
        print(f"Setting up Ollama with model '{model_name}'...")
        
        self.llm = Ollama(
            model=model_name,
            base_url=OLLAMA_BASE_URL,
            temperature=0.7,
            request_timeout=120.0
        )
        
        # Configure LlamaIndex settings
        LlamaSettings.llm = self.llm
        
        print("Ollama LLM initialized")
    
    def generate_response(self, query: str, context: Dict) -> Dict:
        """
        Generate response using retrieved context.
        
        Args:
            query: User question
            context: Retrieved context from multimodal index
            
        Returns:
            Dict with 'response', 'sources' (timestamps and frames)
        """
        # Format text context
        text_context_str = ""
        for i, ctx in enumerate(context['text_contexts'], 1):
            text_context_str += f"\n[Transcript {i} at {ctx['timestamp_str']}]:\n{ctx['text']}\n"
        
        # Format image context
        image_context_str = ""
        for i, ctx in enumerate(context['image_contexts'], 1):
            image_context_str += f"\n[Slide {i} at {ctx['timestamp_str']}]"
        
        # Create prompt
        prompt = f"""You are an assistant helping users understand a lecture video. 
Answer the question based on the provided transcript excerpts and slide information.

Transcript excerpts from the lecture:
{text_context_str}

Visual slides shown:
{image_context_str}

Question: {query}

Instructions:
- Provide a clear, concise answer based on the evidence above
- Reference specific timestamps when relevant
- If the context doesn't contain enough information, say so
- Be helpful and educational

Answer:"""
        
        # Generate response
        response = self.llm.complete(prompt)
        
        # Collect sources
        sources = {
            'text_timestamps': [
                {'time': ctx['timestamp_str'], 'start_time': ctx['start_time']}
                for ctx in context['text_contexts']
            ],
            'image_frames': [
                {'frame_path': ctx['frame_path'], 
                 'timestamp': ctx['timestamp'],
                 'timestamp_str': ctx['timestamp_str']}
                for ctx in context['image_contexts']
            ]
        }
        
        return {
            'response': response.text,
            'sources': sources
        }


class RAGEngine:
    """Main RAG engine orchestrating multimodal retrieval and generation."""
    
    def __init__(self, video_id: str):
        """
        Initialize RAG engine.
        
        Args:
            video_id: Unique video ID for this session
        """
        self.video_id = video_id
        self.index = MultimodalIndex(video_id)
        self.llm_engine = LLMEngine()
    
    def build_index(self, transcription_chunks: List[Dict], 
                   frame_data: List[Tuple[str, float]]):
        """
        Build the multimodal index.
        
        Args:
            transcription_chunks: List of transcription dicts
            frame_data: List of (frame_path, timestamp) tuples
        """
        self.index.build_index(transcription_chunks, frame_data)
    
    def query(self, question: str) -> Dict:
        """
        Process a user query end-to-end.
        
        Args:
            question: User question
            
        Returns:
            Dict with 'response' and 'sources'
        """
        # Retrieve context
        context = self.index.retrieve(question)
        
        # Generate response
        result = self.llm_engine.generate_response(question, context)
        
        return result
