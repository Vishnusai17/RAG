"""
Query Engine Module
Unified interface for hybrid graph + vector search
"""

from typing import List, Dict, Any, Optional
from enum import Enum
from dataclasses import dataclass

from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain.schema import HumanMessage

from config import settings
from rag.cypher_chain import CypherChain
from rag.vector_store import VectorStore, get_vector_store
from graph.neo4j_client import Neo4jClient, get_client


class QueryType(Enum):
    """Types of queries the engine can handle"""
    RELATIONSHIP = "relationship"  # "How is A connected to B?"
    ENTITY_INFO = "entity_info"    # "Who is X?" / "What is Y?"
    DOCUMENT_SEARCH = "document"   # "Find documents about..."
    HYBRID = "hybrid"              # Complex queries needing both


@dataclass
class QueryResult:
    """Result from a query"""
    answer: str
    sources: List[Dict[str, Any]]
    query_type: QueryType
    graph_results: Optional[List[Dict[str, Any]]] = None
    vector_results: Optional[List[Dict[str, Any]]] = None
    cypher_query: Optional[str] = None
    confidence: str = "Medium"


class QueryEngine:
    """
    Unified query engine combining:
    1. Graph queries (Neo4j + Cypher)
    2. Vector search (FAISS)
    3. LLM for query understanding and answer synthesis
    """
    
    RELATIONSHIP_KEYWORDS = [
        "connected", "related", "relationship", "link", "between",
        "knows", "works", "sent", "email", "contact", "path"
    ]
    
    ENTITY_KEYWORDS = [
        "who is", "what is", "tell me about", "information on",
        "details about", "describe"
    ]
    
    def __init__(
        self,
        neo4j_client: Neo4jClient = None,
        vector_store: VectorStore = None
    ):
        self.neo4j_client = neo4j_client or get_client()
        self.vector_store = vector_store or get_vector_store()
        self.cypher_chain = CypherChain()
        
        if settings.llm_provider == "ollama":
            self.llm = ChatOllama(
                base_url=settings.ollama_base_url,
                model=settings.ollama_model,
                temperature=0.2
            )
        elif settings.openai_api_key:
            self.llm = ChatOpenAI(
                model=settings.openai_model,
                api_key=settings.openai_api_key,
                temperature=0.2
            )
        else:
            self.llm = None
    
    def query(self, question: str) -> QueryResult:
        """
        Process a natural language query using appropriate strategy.
        """
        # Classify the query type
        query_type = self._classify_query(question)
        
        if query_type == QueryType.RELATIONSHIP:
            return self._handle_relationship_query(question)
        elif query_type == QueryType.ENTITY_INFO:
            return self._handle_entity_query(question)
        elif query_type == QueryType.DOCUMENT_SEARCH:
            return self._handle_document_query(question)
        else:
            return self._handle_hybrid_query(question)
    
    def _classify_query(self, question: str) -> QueryType:
        """Classify the query type based on keywords"""
        question_lower = question.lower()
        
        # Check for relationship queries
        if any(kw in question_lower for kw in self.RELATIONSHIP_KEYWORDS):
            return QueryType.RELATIONSHIP
        
        # Check for entity info queries
        if any(kw in question_lower for kw in self.ENTITY_KEYWORDS):
            return QueryType.ENTITY_INFO
        
        # Check for document-specific queries
        if "document" in question_lower or "email" in question_lower:
            return QueryType.DOCUMENT_SEARCH
        
        # Default to hybrid
        return QueryType.HYBRID
    
    def _handle_relationship_query(self, question: str) -> QueryResult:
        """Handle relationship/connection queries using graph"""
        # Use the Cypher chain for relationship queries
        result = self.cypher_chain.query(question)
        
        # Also try to find explicit paths if two entities are mentioned
        entities = self._extract_entity_names(question)
        path_results = []
        
        if len(entities) >= 2:
            paths = self.neo4j_client.get_path_between(
                entities[0], entities[1], max_hops=4
            )
            path_results = paths
        
        return QueryResult(
            answer=result["answer"],
            sources=result.get("intermediate_steps", []),
            query_type=QueryType.RELATIONSHIP,
            graph_results=path_results,
            cypher_query=result.get("cypher_query"),
            confidence="High" if result["success"] else "Low"
        )
    
    def _handle_entity_query(self, question: str) -> QueryResult:
        """Handle entity information queries"""
        entities = self._extract_entity_names(question)
        
        all_info = []
        for entity_name in entities:
            # Get entity info from graph
            relationships = self.neo4j_client.get_relationships(entity_name)
            all_info.extend(relationships)
        
        # Also get related documents from vector store
        vector_results = self.vector_store.search(question, k=3)
        
        # Synthesize answer
        answer = self._synthesize_entity_answer(question, all_info, vector_results)
        
        return QueryResult(
            answer=answer,
            sources=all_info[:10],
            query_type=QueryType.ENTITY_INFO,
            graph_results=all_info,
            vector_results=vector_results,
            confidence="Medium"
        )
    
    def _handle_document_query(self, question: str) -> QueryResult:
        """Handle document search queries"""
        # Search vector store
        vector_results = self.vector_store.search(question, k=5)
        
        # Synthesize answer from document content
        answer = self._synthesize_document_answer(question, vector_results)
        
        return QueryResult(
            answer=answer,
            sources=vector_results,
            query_type=QueryType.DOCUMENT_SEARCH,
            vector_results=vector_results,
            confidence="Medium"
        )
    
    def _handle_hybrid_query(self, question: str) -> QueryResult:
        """Handle complex queries needing both graph and vector search"""
        # Graph query
        graph_result = self.cypher_chain.query(question)
        
        # Vector search
        vector_results = self.vector_store.search(question, k=3)
        
        # Combine results
        answer = self._synthesize_hybrid_answer(
            question,
            graph_result,
            vector_results
        )
        
        return QueryResult(
            answer=answer,
            sources=vector_results,
            query_type=QueryType.HYBRID,
            graph_results=graph_result.get("intermediate_steps", []),
            vector_results=vector_results,
            cypher_query=graph_result.get("cypher_query"),
            confidence="Medium"
        )
    
    def _extract_entity_names(self, question: str) -> List[str]:
        """Extract potential entity names from the question"""
        if not self.llm:
            # Simple extraction without LLM
            # Look for capitalized words/phrases
            import re
            matches = re.findall(r'[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*', question)
            return matches[:3]  # Limit to 3 entities
        
        prompt = f"""Extract entity names (people, companies, places) from this question.
Return only the names, one per line.

Question: {question}

Entity names:"""
        
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            names = [n.strip() for n in response.content.strip().split('\n') if n.strip()]
            return names[:5]
        except:
            return []
    
    def _synthesize_entity_answer(
        self,
        question: str,
        graph_info: List[Dict],
        vector_results: List[Dict]
    ) -> str:
        """Synthesize an answer about an entity"""
        if not self.llm:
            if graph_info:
                return f"Found {len(graph_info)} relationships for this entity."
            return "No information found for this entity."
        
        context = f"""Graph relationships found:
{self._format_relationships(graph_info[:10])}

Related document excerpts:
{self._format_vector_results(vector_results[:3])}"""
        
        prompt = f"""Based on the following information, answer the question.

{context}

Question: {question}

Answer:"""
        
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return response.content
        except Exception as e:
            return f"Error generating answer: {e}"
    
    def _synthesize_document_answer(
        self,
        question: str,
        vector_results: List[Dict]
    ) -> str:
        """Synthesize an answer from document search results"""
        if not vector_results:
            return "No relevant documents found."
        
        if not self.llm:
            return f"Found {len(vector_results)} relevant documents."
        
        context = self._format_vector_results(vector_results)
        
        prompt = f"""Based on the following document excerpts, answer the question.

{context}

Question: {question}

Answer:"""
        
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return response.content
        except Exception as e:
            return f"Error generating answer: {e}"
    
    def _synthesize_hybrid_answer(
        self,
        question: str,
        graph_result: Dict,
        vector_results: List[Dict]
    ) -> str:
        """Synthesize an answer from both graph and vector results"""
        if not self.llm:
            return graph_result.get("answer", "No answer generated.")
        
        graph_answer = graph_result.get("answer", "No graph results.")
        vector_context = self._format_vector_results(vector_results[:3])
        
        prompt = f"""Combine the following information to answer the question comprehensively.

Graph analysis result:
{graph_answer}

Related document excerpts:
{vector_context}

Question: {question}

Comprehensive answer:"""
        
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return response.content
        except Exception as e:
            return graph_answer  # Fall back to graph answer
    
    def _format_relationships(self, relationships: List[Dict]) -> str:
        """Format relationships for the prompt"""
        if not relationships:
            return "No relationships found."
        
        lines = []
        for rel in relationships[:10]:
            source = rel.get("source", {}).get("name", "Unknown")
            rel_type = rel.get("relationship_type", "RELATED_TO")
            target = rel.get("target", {}).get("name", "Unknown")
            lines.append(f"- {source} --[{rel_type}]--> {target}")
        
        return "\n".join(lines)
    
    def _format_vector_results(self, results: List[Dict]) -> str:
        """Format vector search results for the prompt"""
        if not results:
            return "No documents found."
        
        lines = []
        for i, result in enumerate(results, 1):
            content = result.get("content", "")[:300]
            source = result.get("metadata", {}).get("source", "Unknown")
            lines.append(f"[{i}] Source: {source}\n{content}...")
        
        return "\n\n".join(lines)


# Convenience function
def query(question: str) -> QueryResult:
    """Query the knowledge graph with natural language"""
    engine = QueryEngine()
    return engine.query(question)
