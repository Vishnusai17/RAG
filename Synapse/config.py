"""
GraphRAG Configuration Module
Centralized configuration management using pydantic-settings
"""

import os
from functools import lru_cache
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # Neo4j Configuration
    neo4j_uri: str = Field(default="bolt://localhost:7687", description="Neo4j connection URI")
    neo4j_username: str = Field(default="neo4j", description="Neo4j username")
    neo4j_password: str = Field(default="password", description="Neo4j password")
    
    # LLM Provider Configuration
    llm_provider: str = Field(default="openai", description="LLM provider: 'openai' or 'ollama'")

    # OpenAI Configuration
    openai_api_key: str = Field(default="", description="OpenAI API key")
    openai_model: str = Field(default="gpt-4-turbo-preview", description="OpenAI model to use")
    
    # Ollama Configuration
    ollama_base_url: str = Field(default="http://localhost:11434", description="Ollama base URL")
    ollama_model: str = Field(default="mistral", description="Ollama model to use")
    
    # Google Gemini Configuration
    gemini_api_key: str = Field(default="", description="Google Gemini API key")
    gemini_model: str = Field(default="gemini-1.5-flash", description="Gemini model to use")
    
    # Embedding Configuration
    embedding_model: str = Field(default="text-embedding-3-small", description="Embedding model (OpenAI or Ollama)")
    
    # Vector Store Configuration
    vector_store_path: str = Field(default="./data/vector_store", description="Path to vector store")
    
    # API Configuration
    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default=8000, description="API port")
    
    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    
    # Entity Extraction
    spacy_model: str = Field(default="en_core_web_lg", description="spaCy model for NER")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


# Convenience exports
settings = get_settings()
