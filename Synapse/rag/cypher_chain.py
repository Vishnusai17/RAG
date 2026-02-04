"""
Cypher Chain Module
Custom implementation for natural language to Cypher queries
Bypasses LangChain's GraphCypherQAChain to avoid APOC dependencies.
"""

from typing import Dict, Any, Optional, List
import json
import re

from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain.schema import HumanMessage, SystemMessage

from config import settings
from graph.neo4j_client import get_client, Neo4jClient


# Schema description for the LLM
# We hardcode this to avoid needing APOC to introspect the DB repeatedly
GRAPH_SCHEMA = """
The graph contains the following node types and properties:
- Person: {name, normalized_name, email, aliases}
- Organization: {name, normalized_name, type}
- Location: {name, type}
- Document: {doc_id, title, source, doc_type, date}
- Date: {value, normalized}

Relationships:
- (:Person)-[:WORKS_FOR]->(:Organization)
- (:Person)-[:EMPLOYED_BY]->(:Organization)
- (:Person)-[:SENT_TO]->(:Person)
- (:Person)-[:CC_TO]->(:Person)
- (:Person)-[:RECEIVED_FROM]->(:Person)
- (:Person)-[:REPORTED_TO]->(:Person)
- (:Person)-[:MET_WITH]->(:Person)
- (:Person)-[:KNOWS]->(:Person)
- (:Organization)-[:CONTRACTED_WITH]->(:Organization)
- (:Organization)-[:PAID_TO]->(:Organization)
- (:Organization)-[:AFFILIATED_WITH]->(:Organization)
- (:Person)-[:AFFILIATED_WITH]->(:Organization)
- (Any)-[:LOCATED_IN]->(:Location)
- (Any)-[:MENTIONED_IN]->(:Document)
- (Any)-[:OCCURRED_ON]->(:Date)
"""

CYPHER_SYSTEM_PROMPT = f"""You are a Neo4j Cypher expert analyzing an investigative knowledge graph.

{GRAPH_SCHEMA}

Task: Generate a valid Cypher query to answer the user's question.

CRITICAL RULES:
1. Return ONLY the Cypher query. No markdown, no explanations, no extra text.
2. Use case-insensitive matching: WHERE toLower(n.name) CONTAINS toLower('search term')
3. Limit results to 50 unless specified otherwise.
4. Do NOT use APOC procedures.
5. When matching relationships, use pattern: MATCH (a)-[r]->(b) or MATCH (a)-[r]-(b)
6. ALWAYS return relationship details as: type(r), properties(r)
7. ALWAYS return node names: a.name, b.name

EXAMPLES:

Question: "How is John Doe connected to Acme Corp?"
Good query:
MATCH (a)-[r]-(b)
WHERE toLower(a.name) CONTAINS toLower('John Doe') 
  AND toLower(b.name) CONTAINS toLower('Acme Corp')
RETURN a.name as source, type(r) as relationship_type, properties(r) as rel_props, b.name as target
LIMIT 10

Question: "Who works for Acme Corp?"
Good query:
MATCH (p:Person)-[r:WORKS_FOR|EMPLOYED_BY]->(o:Organization)
WHERE toLower(o.name) CONTAINS toLower('Acme Corp')
RETURN p.name as person, type(r) as relationship, o.name as organization
LIMIT 50

Question: "What organizations does John Doe know?"
Good query:
MATCH (p:Person)-[r]-(o:Organization)
WHERE toLower(p.name) CONTAINS toLower('John Doe')
RETURN p.name as person, type(r) as relationship, o.name as organization
LIMIT 50
"""

QA_SYSTEM_PROMPT = """You are an investigative analyst answering questions about a knowledge graph.
Based on the provided query results, write a clear, concise answer.
If results are empty, say "No information found in the graph."
Cite the specific nodes/relationships found if relevant.
"""

class CypherChain:
    """Custom wrapper for Graph RAG operations"""
    
    def __init__(self):
        self.client = get_client()
        self._llm = None
        
    @property
    def llm(self):
        """Lazy load LLM"""
        if self._llm is None:
            if settings.llm_provider == "ollama":
                self._llm = ChatOllama(
                    base_url=settings.ollama_base_url,
                    model=settings.ollama_model,
                    temperature=0
                )
            elif settings.llm_provider == "gemini":
                from langchain_google_genai import ChatGoogleGenerativeAI
                self._llm = ChatGoogleGenerativeAI(
                    model=settings.gemini_model,
                    google_api_key=settings.gemini_api_key,
                    temperature=0
                )
            else:
                self._llm = ChatOpenAI(
                    model=settings.openai_model,
                    api_key=settings.openai_api_key,
                    temperature=0
                )
        return self._llm
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        Execute a natural language query against the graph.
        """
        try:
            # 1. Generate Cypher
            cypher_query = self._generate_cypher(question)
            if not cypher_query:
                return {"answer": "Could not generate a valid query.", "success": False}
            
            # Clean up potential markdown formatting
            cypher_query = self._clean_cypher(cypher_query)
            
            # 2. Execute Cypher
            try:
                results = self.client.run_cypher(cypher_query)
            except Exception as e:
                return {
                    "answer": f"Error executing database query: {str(e)}",
                    "cypher_query": cypher_query,
                    "results": [],
                    "success": False
                }
            
            # 3. Generate Answer
            answer = self._generate_answer(question, results)
            
            return {
                "answer": answer,
                "cypher_query": cypher_query,
                "results": results,
                "success": True
            }
            
        except Exception as e:
            return {
                "answer": f"Error processing query: {str(e)}",
                "success": False,
                "error": str(e)
            }
    
    def _generate_cypher(self, question: str) -> str:
        """Call LLM to generate Cypher"""
        messages = [
            SystemMessage(content=CYPHER_SYSTEM_PROMPT),
            HumanMessage(content=f"Question: {question}")
        ]
        response = self.llm.invoke(messages)
        return response.content
    
    def _clean_cypher(self, query: str) -> str:
        """Remove markdown code blocks if present"""
        query = query.strip()
        # Remove ```cypher ... ``` or ``` ... ```
        match = re.search(r"```(?:cypher)?\n?(.*?)```", query, re.DOTALL)
        if match:
            return match.group(1).strip()
        return query
    
    def _generate_answer(self, question: str, results: List[Dict[str, Any]]) -> str:
        """Call LLM to synthesize answer from results"""
        if not results:
            return "No information found in the knowledge graph matching your request."
            
        # Format results as string
        context = json.dumps(results, default=str, indent=2)
        
        # Truncate context if too large (simple heuristic)
        if len(context) > 10000:
            context = context[:10000] + "...(truncated)"
            
        messages = [
            SystemMessage(content=QA_SYSTEM_PROMPT),
            HumanMessage(content=f"Question: {question}\n\nQuery Results:\n{context}")
        ]
        response = self.llm.invoke(messages)
        return response.content
    
    def execute_cypher(self, cypher_query: str) -> Dict[str, Any]:
        """Execute raw Cypher query"""
        try:
            results = self.client.run_cypher(cypher_query)
            return {"results": results, "success": True}
        except Exception as e:
            return {"results": [], "success": False, "error": str(e)}
    
    def refresh_schema(self):
        """No-op for custom chain"""
        pass

# Convenience function
def query_graph(question: str) -> Dict[str, Any]:
    chain = CypherChain()
    return chain.query(question)
