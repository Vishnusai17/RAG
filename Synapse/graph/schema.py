"""
Neo4j Graph Schema Definitions
"""

from enum import Enum
from typing import Dict, List, Any


class NodeLabel(Enum):
    """Node labels in the knowledge graph"""
    PERSON = "Person"
    ORGANIZATION = "Organization"
    LOCATION = "Location"
    DOCUMENT = "Document"
    DATE = "Date"
    EMAIL_ADDRESS = "EmailAddress"
    EVENT = "Event"
    TOPIC = "Topic"


class RelLabel(Enum):
    """Relationship labels in the knowledge graph"""
    WORKS_FOR = "WORKS_FOR"
    EMPLOYED_BY = "EMPLOYED_BY"
    SENT_TO = "SENT_TO"
    RECEIVED_FROM = "RECEIVED_FROM"
    AFFILIATED_WITH = "AFFILIATED_WITH"
    LOCATED_IN = "LOCATED_IN"
    MET_WITH = "MET_WITH"
    MENTIONED_IN = "MENTIONED_IN"
    MENTIONED_WITH = "MENTIONED_WITH"
    REPORTED_TO = "REPORTED_TO"
    CONTRACTED_WITH = "CONTRACTED_WITH"
    PAID_TO = "PAID_TO"
    OCCURRED_ON = "OCCURRED_ON"
    CC_TO = "CC_TO"
    KNOWS = "KNOWS"
    OWNS = "OWNS"
    PART_OF = "PART_OF"
    HAS_EMAIL = "HAS_EMAIL"
    ABOUT = "ABOUT"


# Schema definitions for each node type
NODE_SCHEMAS: Dict[NodeLabel, Dict[str, Any]] = {
    NodeLabel.PERSON: {
        "properties": {
            "name": {"type": "string", "required": True},
            "normalized_name": {"type": "string"},
            "aliases": {"type": "list"},
            "email": {"type": "string"},
            "title": {"type": "string"},
        },
        "indexes": ["name", "normalized_name"],
    },
    NodeLabel.ORGANIZATION: {
        "properties": {
            "name": {"type": "string", "required": True},
            "normalized_name": {"type": "string"},
            "aliases": {"type": "list"},
            "type": {"type": "string"},  # company, government, nonprofit, etc.
        },
        "indexes": ["name", "normalized_name"],
    },
    NodeLabel.LOCATION: {
        "properties": {
            "name": {"type": "string", "required": True},
            "type": {"type": "string"},  # city, state, country, address
            "coordinates": {"type": "point"},
        },
        "indexes": ["name"],
    },
    NodeLabel.DOCUMENT: {
        "properties": {
            "doc_id": {"type": "string", "required": True},
            "title": {"type": "string"},
            "source": {"type": "string"},
            "doc_type": {"type": "string"},
            "date": {"type": "datetime"},
            "content_preview": {"type": "string"},
        },
        "indexes": ["doc_id", "source"],
    },
    NodeLabel.DATE: {
        "properties": {
            "value": {"type": "string", "required": True},
            "normalized": {"type": "date"},
            "year": {"type": "integer"},
            "month": {"type": "integer"},
            "day": {"type": "integer"},
        },
        "indexes": ["value", "normalized"],
    },
    NodeLabel.EMAIL_ADDRESS: {
        "properties": {
            "address": {"type": "string", "required": True},
            "domain": {"type": "string"},
        },
        "indexes": ["address"],
    },
    NodeLabel.EVENT: {
        "properties": {
            "name": {"type": "string", "required": True},
            "description": {"type": "string"},
            "date": {"type": "datetime"},
        },
        "indexes": ["name"],
    },
    NodeLabel.TOPIC: {
        "properties": {
            "name": {"type": "string", "required": True},
            "description": {"type": "string"},
        },
        "indexes": ["name"],
    },
}


# Cypher queries for schema setup
SCHEMA_SETUP_QUERIES = [
    # Constraints
    "CREATE CONSTRAINT person_name IF NOT EXISTS FOR (p:Person) REQUIRE p.name IS NOT NULL",
    "CREATE CONSTRAINT org_name IF NOT EXISTS FOR (o:Organization) REQUIRE o.name IS NOT NULL",
    "CREATE CONSTRAINT doc_id IF NOT EXISTS FOR (d:Document) REQUIRE d.doc_id IS UNIQUE",
    "CREATE CONSTRAINT email_address IF NOT EXISTS FOR (e:EmailAddress) REQUIRE e.address IS UNIQUE",
    
    # Indexes for faster lookups
    "CREATE INDEX person_normalized IF NOT EXISTS FOR (p:Person) ON (p.normalized_name)",
    "CREATE INDEX org_normalized IF NOT EXISTS FOR (o:Organization) ON (o.normalized_name)",
    "CREATE INDEX location_name IF NOT EXISTS FOR (l:Location) ON (l.name)",
    "CREATE INDEX date_value IF NOT EXISTS FOR (d:Date) ON (d.value)",
    
    # Full-text search indexes
    """
    CREATE FULLTEXT INDEX entity_search IF NOT EXISTS 
    FOR (n:Person|Organization|Location) 
    ON EACH [n.name, n.normalized_name]
    """,
]


def get_node_properties(label: NodeLabel) -> List[str]:
    """Get list of property names for a node type"""
    schema = NODE_SCHEMAS.get(label, {})
    return list(schema.get("properties", {}).keys())


def get_required_properties(label: NodeLabel) -> List[str]:
    """Get required properties for a node type"""
    schema = NODE_SCHEMAS.get(label, {})
    props = schema.get("properties", {})
    return [k for k, v in props.items() if v.get("required", False)]
