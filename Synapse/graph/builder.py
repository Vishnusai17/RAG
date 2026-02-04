"""
Graph Builder Module
Populates Neo4j graph from extracted entities and relationships
"""

from typing import List, Dict, Any, Optional
from datetime import datetime

from graph.neo4j_client import Neo4jClient, get_client
from graph.schema import NodeLabel, RelLabel
from extraction.entity_extractor import Entity, EntityType
from extraction.relationship_extractor import Relationship, RelationshipType
from ingestion.document_loader import Document


# Mapping EntityType to NodeLabel
ENTITY_TO_NODE_LABEL = {
    EntityType.PERSON: NodeLabel.PERSON,
    EntityType.ORGANIZATION: NodeLabel.ORGANIZATION,
    EntityType.LOCATION: NodeLabel.LOCATION,
    EntityType.DATE: NodeLabel.DATE,
    EntityType.EMAIL: NodeLabel.EMAIL_ADDRESS,
    EntityType.EVENT: NodeLabel.EVENT,
}

# Mapping RelationshipType to RelLabel
REL_TYPE_TO_LABEL = {
    RelationshipType.WORKS_FOR: RelLabel.WORKS_FOR,
    RelationshipType.EMPLOYED_BY: RelLabel.EMPLOYED_BY,
    RelationshipType.SENT_TO: RelLabel.SENT_TO,
    RelationshipType.RECEIVED_FROM: RelLabel.RECEIVED_FROM,
    RelationshipType.AFFILIATED_WITH: RelLabel.AFFILIATED_WITH,
    RelationshipType.LOCATED_IN: RelLabel.LOCATED_IN,
    RelationshipType.MET_WITH: RelLabel.MET_WITH,
    RelationshipType.MENTIONED_WITH: RelLabel.MENTIONED_WITH,
    RelationshipType.REPORTED_TO: RelLabel.REPORTED_TO,
    RelationshipType.CONTRACTED_WITH: RelLabel.CONTRACTED_WITH,
    RelationshipType.PAID_TO: RelLabel.PAID_TO,
    RelationshipType.OCCURRED_ON: RelLabel.OCCURRED_ON,
    RelationshipType.CC_TO: RelLabel.CC_TO,
    RelationshipType.KNOWS: RelLabel.KNOWS,
    RelationshipType.OWNS: RelLabel.OWNS,
    RelationshipType.PART_OF: RelLabel.PART_OF,
    RelationshipType.MENTIONS: RelLabel.MENTIONED_IN,
}


class GraphBuilder:
    """Build knowledge graph from extracted data"""
    
    def __init__(self, client: Neo4jClient = None):
        self.client = client or get_client()
        self._entity_cache: Dict[str, Dict[str, Any]] = {}
    
    def initialize(self) -> bool:
        """Initialize the graph schema"""
        return self.client.setup_schema()
    
    def add_document(self, document: Document) -> Optional[int]:
        """Add a document node to the graph"""
        properties = {
            "doc_id": document.id,
            "source": document.source,
            "doc_type": document.doc_type,
            "content_preview": document.content[:500] if document.content else "",
            "created_at": datetime.now().isoformat(),
        }
        
        # Add metadata
        if document.metadata:
            if "title" in document.metadata:
                properties["title"] = document.metadata["title"]
            if "subject" in document.metadata:
                properties["subject"] = document.metadata["subject"]
            if "date" in document.metadata:
                properties["date"] = document.metadata["date"]
        
        return self.client.merge_node(
            NodeLabel.DOCUMENT,
            match_properties={"doc_id": document.id},
            set_properties=properties
        )
    
    def add_entity(self, entity: Entity) -> Optional[int]:
        """Add an entity node to the graph"""
        node_label = ENTITY_TO_NODE_LABEL.get(entity.entity_type)
        if not node_label:
            return None
        
        # Prepare properties
        properties = {
            "name": entity.name,
            "normalized_name": entity.name.lower(),
            "confidence": entity.confidence,
        }
        
        # Add entity-specific properties
        if entity.entity_type == EntityType.EMAIL:
            properties["address"] = entity.name
            domain = entity.name.split("@")[-1] if "@" in entity.name else ""
            properties["domain"] = domain
        
        if entity.mentions:
            properties["aliases"] = list(set(entity.mentions))
        
        # Add any extra properties
        properties.update(entity.properties)
        
        # Merge the node
        node_id = self.client.merge_node(
            node_label,
            match_properties={"normalized_name": entity.name.lower()},
            set_properties=properties
        )
        
        # Cache for relationship creation
        cache_key = f"{entity.entity_type.value}:{entity.name.lower()}"
        self._entity_cache[cache_key] = {
            "label": node_label,
            "name": entity.name,
            "node_id": node_id,
        }
        
        return node_id
    
    def add_relationship(self, relationship: Relationship) -> bool:
        """Add a relationship to the graph"""
        rel_label = REL_TYPE_TO_LABEL.get(relationship.relationship_type)
        if not rel_label:
            rel_label = RelLabel.AFFILIATED_WITH  # Default fallback
        
        # Find source and target entities
        source_info = self._find_entity_in_cache(relationship.source)
        target_info = self._find_entity_in_cache(relationship.target)
        
        if not source_info or not target_info:
            # Entities not in cache, try to find in graph
            source_node = self._find_entity_in_graph(relationship.source)
            target_node = self._find_entity_in_graph(relationship.target)
            
            if not source_node or not target_node:
                return False
            
            source_info = source_node
            target_info = target_node
        
        # Create relationship
        rel_properties = {
            "confidence": relationship.confidence,
            "source_doc_id": relationship.source_doc_id,
        }
        
        if relationship.evidence:
            rel_properties["evidence"] = relationship.evidence[:500]
        
        rel_properties.update(relationship.properties)
        
        return self.client.create_relationship(
            source_label=source_info["label"],
            source_match={"normalized_name": source_info["name"].lower()},
            target_label=target_info["label"],
            target_match={"normalized_name": target_info["name"].lower()},
            rel_type=rel_label,
            rel_properties=rel_properties
        )
    
    def link_entity_to_document(
        self,
        entity: Entity,
        document: Document
    ) -> bool:
        """Create MENTIONED_IN relationship between entity and document"""
        entity_info = self._find_entity_in_cache(entity.name)
        if not entity_info:
            entity_info = self._find_entity_in_graph(entity.name)
        
        if not entity_info:
            return False
        
        return self.client.create_relationship(
            source_label=entity_info["label"],
            source_match={"normalized_name": entity.name.lower()},
            target_label=NodeLabel.DOCUMENT,
            target_match={"doc_id": document.id},
            rel_type=RelLabel.MENTIONED_IN,
            rel_properties={"confidence": entity.confidence}
        )
    
    def build_from_extraction(
        self,
        document: Document,
        entities: List[Entity],
        relationships: List[Relationship]
    ) -> Dict[str, int]:
        """
        Build graph from extraction results.
        Returns counts of created nodes and relationships.
        """
        stats = {
            "documents": 0,
            "entities": 0,
            "relationships": 0,
            "entity_links": 0,
        }
        
        # Add document
        if self.add_document(document):
            stats["documents"] += 1
        
        # Add entities
        for entity in entities:
            if self.add_entity(entity):
                stats["entities"] += 1
                
                # Link to document
                if self.link_entity_to_document(entity, document):
                    stats["entity_links"] += 1
        
        # Add relationships
        for rel in relationships:
            if self.add_relationship(rel):
                stats["relationships"] += 1
        
        return stats
    
    def _find_entity_in_cache(self, name: str) -> Optional[Dict[str, Any]]:
        """Find an entity in the local cache"""
        name_lower = name.lower()
        
        # Try exact match with different types
        for entity_type in EntityType:
            cache_key = f"{entity_type.value}:{name_lower}"
            if cache_key in self._entity_cache:
                return self._entity_cache[cache_key]
        
        # Try fuzzy match
        for key, info in self._entity_cache.items():
            if name_lower in key or info["name"].lower() in name_lower:
                return info
        
        return None
    
    def _find_entity_in_graph(self, name: str) -> Optional[Dict[str, Any]]:
        """Find an entity in the graph database"""
        # Try each node type
        for node_label in [NodeLabel.PERSON, NodeLabel.ORGANIZATION, NodeLabel.LOCATION]:
            node = self.client.get_node_by_name(node_label, name)
            if node:
                return {
                    "label": node_label,
                    "name": node.get("name", name),
                    "node_id": node.get("_id"),
                }
        
        return None
    
    def clear_cache(self):
        """Clear the entity cache"""
        self._entity_cache.clear()


# Convenience function
def build_graph(
    document: Document,
    entities: List[Entity],
    relationships: List[Relationship]
) -> Dict[str, int]:
    """Build graph from a single document's extraction"""
    builder = GraphBuilder()
    return builder.build_from_extraction(document, entities, relationships)
