"""
Relationship Extractor Module
Extract relationships between entities using LLM
"""

import json
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

import spacy
from langchain_openai import ChatOpenAI
import ollama
from langchain.schema import HumanMessage

from config import settings
from extraction.entity_extractor import Entity, EntityType
from extraction.prompts import RELATIONSHIP_EXTRACTION_PROMPT


class RelationshipType(Enum):
    """Types of relationships between entities"""
    WORKS_FOR = "WORKS_FOR"
    EMPLOYED_BY = "EMPLOYED_BY"
    SENT_TO = "SENT_TO"
    RECEIVED_FROM = "RECEIVED_FROM"
    AFFILIATED_WITH = "AFFILIATED_WITH"
    LOCATED_IN = "LOCATED_IN"
    MET_WITH = "MET_WITH"
    MENTIONED_WITH = "MENTIONED_WITH"
    REPORTED_TO = "REPORTED_TO"
    CONTRACTED_WITH = "CONTRACTED_WITH"
    PAID_TO = "PAID_TO"
    OCCURRED_ON = "OCCURRED_ON"
    CC_TO = "CC_TO"
    KNOWS = "KNOWS"
    OWNS = "OWNS"
    PART_OF = "PART_OF"
    MENTIONS = "MENTIONS"


@dataclass
class Relationship:
    """Represents a relationship between two entities"""
    source: str
    target: str
    relationship_type: RelationshipType
    evidence: str = ""
    confidence: str = "Medium"
    source_doc_id: str = ""
    properties: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "type": self.relationship_type.value,
            "evidence": self.evidence,
            "confidence": self.confidence,
            "source_doc_id": self.source_doc_id,
            "properties": self.properties,
        }


class RelationshipExtractor:
    """Extract relationships between entities"""
    
    def __init__(self, llm_model: str = None):
        if settings.llm_provider == "ollama":
            self.llm = "ollama"  # Marker to use ollama library
        elif settings.openai_api_key:
            self.llm = ChatOpenAI(
                model=llm_model or settings.openai_model,
                api_key=settings.openai_api_key,
                temperature=0.1
            )
        else:
            self.llm = None
    
    def extract_relationships(
        self,
        text: str,
        entities: List[Entity],
        doc_id: str = "",
        doc_type: str = "text"
    ) -> List[Relationship]:
        """
        Extract relationships between entities in text
        """
        relationships = []
        
        # Extract implicit relationships from email metadata
        if doc_type == "email":
            email_rels = self._extract_email_relationships(text, entities, doc_id)
            relationships.extend(email_rels)
        
        # Use LLM for deeper relationship extraction
        if self.llm and entities:
            llm_rels = self._extract_with_llm(text, entities, doc_id)
            relationships.extend(llm_rels)
        
        # Extract co-occurrence relationships
        cooccur_rels = self._extract_cooccurrence(entities, doc_id)
        relationships.extend(cooccur_rels)
        
        return self._deduplicate_relationships(relationships)
    
    def _extract_email_relationships(
        self,
        text: str,
        entities: List[Entity],
        doc_id: str
    ) -> List[Relationship]:
        """Extract relationships from email structure"""
        relationships = []
        
        # Parse email headers
        lines = text.split('\n')
        sender = None
        recipients = []
        cc_recipients = []
        
        for line in lines[:10]:  # Check first 10 lines for headers
            if line.startswith('From:'):
                sender = line[5:].strip()
            elif line.startswith('To:'):
                recipients = [r.strip() for r in line[3:].split(',')]
            elif line.startswith('CC:'):
                cc_recipients = [r.strip() for r in line[3:].split(',')]
        
        # Create SENT_TO relationships
        if sender:
            sender_entity = self._find_entity_by_mention(sender, entities)
            
            for recipient in recipients:
                recipient_entity = self._find_entity_by_mention(recipient, entities)
                if sender_entity and recipient_entity:
                    relationships.append(Relationship(
                        source=sender_entity.name,
                        target=recipient_entity.name,
                        relationship_type=RelationshipType.SENT_TO,
                        evidence=f"Email from {sender} to {recipient}",
                        confidence="High",
                        source_doc_id=doc_id
                    ))
            
            for cc in cc_recipients:
                cc_entity = self._find_entity_by_mention(cc, entities)
                if sender_entity and cc_entity:
                    relationships.append(Relationship(
                        source=sender_entity.name,
                        target=cc_entity.name,
                        relationship_type=RelationshipType.CC_TO,
                        evidence=f"Email CC'd to {cc}",
                        confidence="High",
                        source_doc_id=doc_id
                    ))
        
        return relationships
    
    def _extract_with_llm(
        self,
        text: str,
        entities: List[Entity],
        doc_id: str
    ) -> List[Relationship]:
        """Extract relationships using LLM"""
        try:
            # Prepare entity list for prompt
            entity_list = "\n".join([
                f"- {e.name} ({e.entity_type.value})"
                for e in entities
            ])
            
            # Truncate text if needed
            max_chars = 3000
            truncated_text = text[:max_chars] if len(text) > max_chars else text
            
            prompt = RELATIONSHIP_EXTRACTION_PROMPT.format(
                entities=entity_list,
                text=truncated_text
            )
            
            if self.llm == "ollama":
                response = ollama.chat(model=settings.ollama_model, messages=[
                    {'role': 'user', 'content': prompt},
                ])
                content = response['message']['content']
            else:
                response = self.llm.invoke([HumanMessage(content=prompt)])
                content = response.content
            
            return self._parse_llm_response(content, doc_id)
            
        except Exception as e:
            print(f"LLM relationship extraction error: {e}")
            return []
    
    def _parse_llm_response(self, response: str, doc_id: str) -> List[Relationship]:
        """Parse LLM response for relationships"""
        relationships = []
        
        try:
            # Extract JSON from response
            json_match = re.search(r'\[[\s\S]*\]', response)
            if json_match:
                data = json.loads(json_match.group())
                
                for item in data:
                    rel_type = self._map_string_to_relationship_type(
                        item.get("relationship_type", "")
                    )
                    if rel_type:
                        relationships.append(Relationship(
                            source=item.get("source", ""),
                            target=item.get("target", ""),
                            relationship_type=rel_type,
                            evidence=item.get("evidence", ""),
                            confidence=item.get("confidence", "Medium"),
                            source_doc_id=doc_id
                        ))
        except json.JSONDecodeError as e:
            print(f"JSON parse error in relationships: {e}")
        
        return relationships
    
    def _extract_cooccurrence(
        self,
        entities: List[Entity],
        doc_id: str
    ) -> List[Relationship]:
        """
        Extract weak relationships based on co-occurrence
        (entities mentioned in the same document)
        """
        relationships = []
        
        # Only create co-occurrence for entities from same doc
        doc_entities = [e for e in entities if e.source_doc_id == doc_id]
        
        # Create MENTIONED_WITH relationships between persons and orgs
        persons = [e for e in doc_entities if e.entity_type == EntityType.PERSON]
        orgs = [e for e in doc_entities if e.entity_type == EntityType.ORGANIZATION]
        
        for person in persons:
            for org in orgs:
                relationships.append(Relationship(
                    source=person.name,
                    target=org.name,
                    relationship_type=RelationshipType.MENTIONED_WITH,
                    evidence="Co-occurred in same document",
                    confidence="Low",
                    source_doc_id=doc_id
                ))
        
        return relationships
    
    def _find_entity_by_mention(
        self,
        mention: str,
        entities: List[Entity]
    ) -> Optional[Entity]:
        """Find an entity that matches a mention"""
        mention_lower = mention.lower()
        
        for entity in entities:
            if entity.name.lower() in mention_lower or mention_lower in entity.name.lower():
                return entity
            for m in entity.mentions:
                if m.lower() in mention_lower or mention_lower in m.lower():
                    return entity
        
        return None
    
    def _map_string_to_relationship_type(self, type_str: str) -> Optional[RelationshipType]:
        """Map string to RelationshipType enum"""
        type_str = type_str.upper().replace(" ", "_").replace("-", "_")
        try:
            return RelationshipType[type_str]
        except KeyError:
            return None
    
    def _deduplicate_relationships(
        self,
        relationships: List[Relationship]
    ) -> List[Relationship]:
        """Remove duplicate relationships"""
        seen = set()
        unique = []
        
        for rel in relationships:
            key = (rel.source.lower(), rel.target.lower(), rel.relationship_type)
            if key not in seen:
                seen.add(key)
                unique.append(rel)
        
        return unique


# Convenience function
def extract_relationships(
    text: str,
    entities: List[Entity],
    doc_id: str = ""
) -> List[Relationship]:
    """Extract relationships between entities in text"""
    extractor = RelationshipExtractor()
    return extractor.extract_relationships(text, entities, doc_id)
