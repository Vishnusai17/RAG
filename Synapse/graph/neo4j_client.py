"""
Neo4j Client Module
Manages connection and operations with Neo4j database
"""

from typing import List, Dict, Any, Optional
from contextlib import contextmanager

from neo4j import GraphDatabase, Driver, Session
from neo4j.exceptions import ServiceUnavailable, AuthError

from config import settings
from graph.schema import SCHEMA_SETUP_QUERIES, NodeLabel, RelLabel


class Neo4jClient:
    """Neo4j database client with connection pooling"""
    
    def __init__(
        self,
        uri: str = None,
        username: str = None,
        password: str = None
    ):
        self.uri = uri or settings.neo4j_uri
        self.username = username or settings.neo4j_username
        self.password = password or settings.neo4j_password
        self._driver: Optional[Driver] = None
    
    @property
    def driver(self) -> Driver:
        """Lazy initialization of driver"""
        if self._driver is None:
            self._driver = GraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password)
            )
        return self._driver
    
    def verify_connection(self) -> bool:
        """Test the database connection"""
        try:
            with self.session() as session:
                result = session.run("RETURN 1 AS test")
                return result.single()["test"] == 1
        except (ServiceUnavailable, AuthError) as e:
            print(f"Connection failed: {e}")
            return False
    
    def close(self):
        """Close the driver connection"""
        if self._driver:
            self._driver.close()
            self._driver = None
    
    @contextmanager
    def session(self) -> Session:
        """Context manager for database sessions"""
        session = self.driver.session()
        try:
            yield session
        finally:
            session.close()
    
    def setup_schema(self) -> bool:
        """Create indexes and constraints"""
        try:
            with self.session() as session:
                for query in SCHEMA_SETUP_QUERIES:
                    try:
                        session.run(query)
                    except Exception as e:
                        # Some constraints might already exist
                        print(f"Schema setup note: {e}")
            return True
        except Exception as e:
            print(f"Schema setup failed: {e}")
            return False
    
    def run_cypher(
        self,
        query: str,
        parameters: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """Execute a Cypher query and return results"""
        with self.session() as session:
            result = session.run(query, parameters or {})
            return [dict(record) for record in result]
    
    def create_node(
        self,
        label: NodeLabel,
        properties: Dict[str, Any]
    ) -> Optional[int]:
        """
        Create a node with the given label and properties.
        Returns the node ID.
        """
        # Build property string
        prop_parts = []
        for key, value in properties.items():
            if value is not None:
                prop_parts.append(f"{key}: ${key}")
        
        prop_string = ", ".join(prop_parts)
        query = f"CREATE (n:{label.value} {{{prop_string}}}) RETURN id(n) as node_id"
        
        result = self.run_cypher(query, properties)
        return result[0]["node_id"] if result else None
    
    def merge_node(
        self,
        label: NodeLabel,
        match_properties: Dict[str, Any],
        set_properties: Dict[str, Any] = None
    ) -> Optional[int]:
        """
        Merge a node (create if not exists, update if exists).
        Returns the node ID.
        """
        # Build match property string
        match_parts = [f"{k}: ${k}" for k in match_properties.keys()]
        match_string = ", ".join(match_parts)
        
        query = f"MERGE (n:{label.value} {{{match_string}}})"
        
        params = {**match_properties}
        
        if set_properties:
            set_parts = [f"n.{k} = ${k}_set" for k in set_properties.keys()]
            query += f" ON CREATE SET {', '.join(set_parts)}"
            query += f" ON MATCH SET {', '.join(set_parts)}"
            params.update({f"{k}_set": v for k, v in set_properties.items()})
        
        query += " RETURN id(n) as node_id"
        
        result = self.run_cypher(query, params)
        return result[0]["node_id"] if result else None
    
    def create_relationship(
        self,
        source_label: NodeLabel,
        source_match: Dict[str, Any],
        target_label: NodeLabel,
        target_match: Dict[str, Any],
        rel_type: RelLabel,
        rel_properties: Dict[str, Any] = None
    ) -> bool:
        """Create a relationship between two nodes"""
        # Build match properties
        source_match_str = ", ".join([f"{k}: $source_{k}" for k in source_match.keys()])
        target_match_str = ", ".join([f"{k}: $target_{k}" for k in target_match.keys()])
        
        query = f"""
        MATCH (s:{source_label.value} {{{source_match_str}}})
        MATCH (t:{target_label.value} {{{target_match_str}}})
        MERGE (s)-[r:{rel_type.value}]->(t)
        """
        
        params = {}
        params.update({f"source_{k}": v for k, v in source_match.items()})
        params.update({f"target_{k}": v for k, v in target_match.items()})
        
        if rel_properties:
            set_parts = [f"r.{k} = $rel_{k}" for k in rel_properties.keys()]
            query += f" SET {', '.join(set_parts)}"
            params.update({f"rel_{k}": v for k, v in rel_properties.items()})
        
        query += " RETURN count(r) as created"
        
        result = self.run_cypher(query, params)
        return result[0]["created"] > 0 if result else False
    
    def get_node_by_name(
        self,
        label: NodeLabel,
        name: str
    ) -> Optional[Dict[str, Any]]:
        """Get a node by its name property"""
        query = f"""
        MATCH (n:{label.value})
        WHERE toLower(n.name) = toLower($name)
        RETURN n, id(n) as node_id
        LIMIT 1
        """
        result = self.run_cypher(query, {"name": name})
        if result:
            node = dict(result[0]["n"])
            node["_id"] = result[0]["node_id"]
            return node
        return None
    
    def search_nodes(
        self,
        query_text: str,
        labels: List[NodeLabel] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Search nodes by name using full-text index"""
        if labels:
            label_filter = ":".join([l.value for l in labels])
            query = f"""
            CALL db.index.fulltext.queryNodes('entity_search', $query)
            YIELD node, score
            WHERE node:{label_filter}
            RETURN node, labels(node) as labels, score
            ORDER BY score DESC
            LIMIT $limit
            """
        else:
            query = """
            CALL db.index.fulltext.queryNodes('entity_search', $query)
            YIELD node, score
            RETURN node, labels(node) as labels, score
            ORDER BY score DESC
            LIMIT $limit
            """
        
        result = self.run_cypher(query, {"query": query_text, "limit": limit})
        return [
            {"node": dict(r["node"]), "labels": r["labels"], "score": r["score"]}
            for r in result
        ]
    
    def get_relationships(
        self,
        entity_name: str,
        direction: str = "both",
        rel_types: List[RelLabel] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get relationships for an entity"""
        rel_filter = ""
        if rel_types:
            rel_filter = ":" + "|".join([r.value for r in rel_types])
        
        if direction == "outgoing":
            pattern = f"(n)-[r{rel_filter}]->(m)"
        elif direction == "incoming":
            pattern = f"(n)<-[r{rel_filter}]-(m)"
        else:
            pattern = f"(n)-[r{rel_filter}]-(m)"
        
        query = f"""
        MATCH {pattern}
        WHERE toLower(n.name) = toLower($name)
        RETURN n, type(r) as rel_type, r, m, labels(m) as target_labels
        LIMIT $limit
        """
        
        result = self.run_cypher(query, {"name": entity_name, "limit": limit})
        return [
            {
                "source": dict(r["n"]),
                "relationship_type": r["rel_type"],
                "relationship_props": dict(r["r"]) if r["r"] else {},
                "target": dict(r["m"]),
                "target_labels": r["target_labels"],
            }
            for r in result
        ]
    
    def get_path_between(
        self,
        entity1: str,
        entity2: str,
        max_hops: int = 4
    ) -> List[Dict[str, Any]]:
        """Find paths between two entities"""
        query = f"""
        MATCH path = shortestPath((a)-[*1..{max_hops}]-(b))
        WHERE toLower(a.name) = toLower($entity1) 
          AND toLower(b.name) = toLower($entity2)
        RETURN path,
               [n in nodes(path) | {{name: n.name, labels: labels(n)}}] as nodes,
               [r in relationships(path) | {{type: type(r), props: properties(r)}}] as rels
        LIMIT 5
        """
        
        result = self.run_cypher(query, {"entity1": entity1, "entity2": entity2})
        return [
            {"nodes": r["nodes"], "relationships": r["rels"]}
            for r in result
        ]
    
    def get_graph_stats(self) -> Dict[str, Any]:
        """Get statistics about the graph"""
        query = """
        CALL apoc.meta.stats() YIELD nodeCount, relCount, labels, relTypes
        RETURN nodeCount, relCount, labels, relTypes
        """
        try:
            result = self.run_cypher(query)
            return result[0] if result else {}
        except:
            # APOC might not be installed, use basic query
            query = """
            MATCH (n) WITH count(n) as nodeCount
            MATCH ()-[r]->() WITH nodeCount, count(r) as relCount
            RETURN nodeCount, relCount
            """
            result = self.run_cypher(query)
            return result[0] if result else {"nodeCount": 0, "relCount": 0}
    
    def clear_graph(self) -> bool:
        """Delete all nodes and relationships - USE WITH CAUTION"""
        query = "MATCH (n) DETACH DELETE n"
        try:
            self.run_cypher(query)
            return True
        except Exception as e:
            print(f"Error clearing graph: {e}")
            return False


# Global client instance
_client: Optional[Neo4jClient] = None


def get_client() -> Neo4jClient:
    """Get the global Neo4j client instance"""
    global _client
    if _client is None:
        _client = Neo4jClient()
    return _client
