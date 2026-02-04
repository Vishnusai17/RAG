"""
LLM Prompts for Entity and Relationship Extraction
"""

ENTITY_EXTRACTION_PROMPT = """Extract all entities (Person, Organization, Location, Date) from the text below.
Respond ONLY with a valid JSON array.

TEXT:
{text}

JSON RESPONSE:
[
  {{
    "name": "Entity Name",
    "type": "PERSON",
    "mentions": ["Entity Name"],
    "confidence": "High"
  }}
]"""


RELATIONSHIP_EXTRACTION_PROMPT = """You are an expert at identifying relationships between entities in documents.

Given the following text and a list of entities, identify all relationships between them.

ENTITIES:
{entities}

TEXT:
{text}

For each relationship, provide:
- source: The source entity name (must match an entity from the list)
- target: The target entity name (must match an entity from the list)  
- relationship_type: One of the following:
  * WORKS_FOR - person works for organization
  * EMPLOYED_BY - person employed by organization
  * SENT_TO - person/email sent to person
  * RECEIVED_FROM - received communication from
  * AFFILIATED_WITH - general affiliation
  * LOCATED_IN - entity located in place
  * MET_WITH - person met with person
  * MENTIONED_WITH - entities mentioned together
  * REPORTED_TO - person reports to person
  * CONTRACTED_WITH - business relationship
  * PAID_TO - money transferred to
  * RECEIVED_FROM - money received from
  * OCCURRED_ON - event occurred on date
- evidence: The text evidence supporting this relationship
- confidence: High, Medium, or Low

Only extract relationships that are clearly supported by the text.

Respond with a JSON array:
```json
[
  {{
    "source": "Source Entity",
    "target": "Target Entity", 
    "relationship_type": "RELATIONSHIP_TYPE",
    "evidence": "Text supporting this relationship",
    "confidence": "High"
  }}
]
```

JSON RESPONSE:"""


ENTITY_RESOLUTION_PROMPT = """You are an expert at entity resolution and coreference.

Given multiple entity mentions, determine which ones refer to the same real-world entity.

ENTITIES TO ANALYZE:
{entities}

CONTEXT FROM DOCUMENTS:
{context}

Group entities that refer to the same real-world entity. For each group, provide:
- canonical_name: The best/most complete name for this entity
- type: The entity type
- aliases: All the different names/mentions for this entity
- merge_confidence: How confident you are these are the same entity (High/Medium/Low)

Respond with a JSON array:
```json
[
  {{
    "canonical_name": "Full Official Name",
    "type": "ENTITY_TYPE",
    "aliases": ["alias1", "alias2", "nickname"],
    "merge_confidence": "High"
  }}
]
```

JSON RESPONSE:"""


EMAIL_ENTITY_EXTRACTION_PROMPT = """Analyze this email and extract key information for an investigative knowledge graph.

EMAIL:
{text}

Extract the following in JSON format:
1. sender: {{name, email, organization (if mentioned)}}
2. recipients: [{{name, email, organization}}]
3. entities_mentioned: [{{name, type, context}}]
4. topics: [main topics/subjects discussed]
5. action_items: [any tasks, requests, or action items]
6. sentiment: overall tone (neutral, urgent, friendly, hostile, etc.)
7. key_facts: [important factual statements]

JSON RESPONSE:"""


DOCUMENT_SUMMARY_PROMPT = """Summarize this document focusing on key entities and their relationships.

DOCUMENT:
{text}

Provide a structured summary:
1. Main entities involved
2. Key relationships between entities
3. Important dates/events
4. Central topics/themes
5. Notable facts or claims

Keep the summary concise but comprehensive for building a knowledge graph.

SUMMARY:"""
