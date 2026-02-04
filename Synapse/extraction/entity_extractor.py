"""
Entity Extractor Module
Hybrid entity extraction using spaCy NER + LLM enhancement
"""

import json
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

import spacy
import spacy
from langchain_openai import ChatOpenAI
import ollama
from langchain.schema import HumanMessage

from config import settings
from extraction.prompts import ENTITY_EXTRACTION_PROMPT, EMAIL_ENTITY_EXTRACTION_PROMPT


class EntityType(Enum):
    PERSON = "PERSON"
    ORGANIZATION = "ORGANIZATION"
    LOCATION = "LOCATION"
    DATE = "DATE"
    EMAIL = "EMAIL"
    PHONE = "PHONE"
    MONEY = "MONEY"
    EVENT = "EVENT"
    DOCUMENT = "DOCUMENT"


@dataclass
class Entity:
    """Represents an extracted entity"""
    name: str
    entity_type: EntityType
    mentions: List[str] = field(default_factory=list)
    confidence: str = "Medium"
    source_doc_id: str = ""
    properties: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.entity_type.value,
            "mentions": self.mentions,
            "confidence": self.confidence,
            "source_doc_id": self.source_doc_id,
            "properties": self.properties,
        }


class EntityExtractor:
    """
    Hybrid entity extraction combining:
    1. spaCy NER for fast initial extraction
    2. LLM for enhancement, disambiguation, and additional entities
    """
    
    # Mapping spaCy labels to our EntityType
    SPACY_TYPE_MAP = {
        "PERSON": EntityType.PERSON,
        "ORG": EntityType.ORGANIZATION,
        "GPE": EntityType.LOCATION,
        "LOC": EntityType.LOCATION,
        "DATE": EntityType.DATE,
        "TIME": EntityType.DATE,
        "MONEY": EntityType.MONEY,
        "EVENT": EntityType.EVENT,
        "FAC": EntityType.LOCATION,
        "NORP": EntityType.ORGANIZATION,
    }
    
    def __init__(
        self,
        use_llm: bool = True,
        spacy_model: str = None,
        llm_model: str = None
    ):
        self.use_llm = use_llm
        
        # Load spaCy model
        model_name = spacy_model or settings.spacy_model
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            print(f"Downloading spaCy model: {model_name}")
            spacy.cli.download(model_name)
            self.nlp = spacy.load(model_name)
        
        # Initialize LLM if needed
        if self.use_llm:
            if settings.llm_provider == "ollama":
                self.llm = "ollama"  # Marker to use ollama library
            else:
                self.llm = ChatOpenAI(
                    api_key=settings.openai_api_key,
                    model=settings.openai_model,
                    temperature=0
                )
        else:
            self.llm = None
    
    def extract_entities(
        self,
        text: str,
        doc_id: str = "",
        doc_type: str = "text"
    ) -> List[Entity]:
        """
        Extract entities from text using hybrid approach
        """
        # Step 1: spaCy NER extraction
        spacy_entities = self._extract_with_spacy(text, doc_id)
        
        # Step 2: Extract emails and phones with regex
        pattern_entities = self._extract_patterns(text, doc_id)
        
        # Step 3: LLM enhancement (if enabled)
        if self.llm and self.use_llm:
            if doc_type == "email":
                llm_entities = self._extract_with_llm_email(text, doc_id)
            else:
                llm_entities = self._extract_with_llm(text, doc_id)
            
            # Merge entities
            all_entities = self._merge_entities(spacy_entities + pattern_entities, llm_entities)
        else:
            all_entities = spacy_entities + pattern_entities
        
        return self._deduplicate_entities(all_entities)
    
    def _extract_with_spacy(self, text: str, doc_id: str) -> List[Entity]:
        """Extract entities using spaCy NER"""
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            if ent.label_ in self.SPACY_TYPE_MAP:
                entity_type = self.SPACY_TYPE_MAP[ent.label_]
                
                # Normalize the entity text
                name = self._normalize_name(ent.text, entity_type)
                
                entities.append(Entity(
                    name=name,
                    entity_type=entity_type,
                    mentions=[ent.text],
                    confidence="Medium",
                    source_doc_id=doc_id
                ))
        
        return entities
    
    def _extract_patterns(self, text: str, doc_id: str) -> List[Entity]:
        """Extract entities using regex patterns (emails, phones, etc.)"""
        entities = []
        
        # Email pattern
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        for match in re.finditer(email_pattern, text):
            entities.append(Entity(
                name=match.group().lower(),
                entity_type=EntityType.EMAIL,
                mentions=[match.group()],
                confidence="High",
                source_doc_id=doc_id
            ))
        
        # Phone pattern (various formats)
        phone_pattern = r'[\+]?[(]?[0-9]{1,3}[)]?[-\s\.]?[(]?[0-9]{1,4}[)]?[-\s\.]?[0-9]{1,4}[-\s\.]?[0-9]{1,9}'
        for match in re.finditer(phone_pattern, text):
            phone = match.group()
            # Filter out numbers that are too short or look like dates
            if len(re.sub(r'\D', '', phone)) >= 7:
                entities.append(Entity(
                    name=phone,
                    entity_type=EntityType.PHONE,
                    mentions=[phone],
                    confidence="Medium",
                    source_doc_id=doc_id
                ))
        
        return entities
    
    def _extract_with_llm(self, text: str, doc_id: str) -> List[Entity]:
        """Extract entities using LLM"""
        try:
            # Truncate text if too long
            max_chars = 4000
            truncated_text = text[:max_chars] if len(text) > max_chars else text
            
            prompt = ENTITY_EXTRACTION_PROMPT.format(text=truncated_text)
            if self.llm == "ollama":
                response = ollama.chat(model=settings.ollama_model, messages=[
                    {'role': 'user', 'content': prompt},
                ])
                content = response['message']['content']
            else:
                response = self.llm.invoke([HumanMessage(content=prompt)])
                content = response.content
            
            return self._parse_llm_entity_response(content, doc_id)
        except Exception as e:
            print(f"LLM extraction error: {e}")
            return []
    
    def _extract_with_llm_email(self, text: str, doc_id: str) -> List[Entity]:
        """Extract entities from email using specialized prompt"""
        try:
            prompt = EMAIL_ENTITY_EXTRACTION_PROMPT.format(text=text[:4000])
            if self.llm == "ollama":
                response = ollama.chat(model=settings.ollama_model, messages=[
                    {'role': 'user', 'content': prompt},
                ])
                content = response['message']['content']
            else:
                response = self.llm.invoke([HumanMessage(content=prompt)])
                content = response.content
            
            return self._parse_email_llm_response(content, doc_id)
        except Exception as e:
            print(f"Email LLM extraction error: {e}")
            return []
    
    def _parse_llm_entity_response(self, response: str, doc_id: str) -> List[Entity]:
        """Parse LLM response for entity extraction"""
        entities = []
        
        try:
            # Extract JSON from response
            json_match = re.search(r'\[[\s\S]*\]', response)
            if json_match:
                data = json.loads(json_match.group())
                
                for item in data:
                    entity_type = self._map_string_to_entity_type(item.get("type", ""))
                    if entity_type:
                        entities.append(Entity(
                            name=item.get("name", ""),
                            entity_type=entity_type,
                            mentions=item.get("mentions", []),
                            confidence=item.get("confidence", "Medium"),
                            source_doc_id=doc_id
                        ))
        except json.JSONDecodeError as e:
            print(f"JSON parse error: {e}")
        
        return entities
    
    def _parse_email_llm_response(self, response: str, doc_id: str) -> List[Entity]:
        """Parse LLM response for email entity extraction"""
        entities = []
        
        try:
            # Extract JSON from response
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
                
                # Extract sender
                if "sender" in data and data["sender"]:
                    sender = data["sender"]
                    if sender.get("name"):
                        entities.append(Entity(
                            name=sender["name"],
                            entity_type=EntityType.PERSON,
                            mentions=[sender["name"]],
                            confidence="High",
                            source_doc_id=doc_id,
                            properties={"email": sender.get("email", "")}
                        ))
                
                # Extract recipients
                for recipient in data.get("recipients", []):
                    if recipient.get("name"):
                        entities.append(Entity(
                            name=recipient["name"],
                            entity_type=EntityType.PERSON,
                            mentions=[recipient["name"]],
                            confidence="High",
                            source_doc_id=doc_id,
                            properties={"email": recipient.get("email", "")}
                        ))
                
                # Extract mentioned entities
                for entity_data in data.get("entities_mentioned", []):
                    entity_type = self._map_string_to_entity_type(entity_data.get("type", ""))
                    if entity_type:
                        entities.append(Entity(
                            name=entity_data.get("name", ""),
                            entity_type=entity_type,
                            mentions=[entity_data.get("name", "")],
                            confidence="Medium",
                            source_doc_id=doc_id
                        ))
        except json.JSONDecodeError as e:
            print(f"Email JSON parse error: {e}")
        
        return entities
    
    def _map_string_to_entity_type(self, type_str: str) -> Optional[EntityType]:
        """Map string to EntityType enum"""
        type_str = type_str.upper()
        try:
            return EntityType[type_str]
        except KeyError:
            # Try mapping common variations
            mappings = {
                "ORG": EntityType.ORGANIZATION,
                "COMPANY": EntityType.ORGANIZATION,
                "PLACE": EntityType.LOCATION,
                "GPE": EntityType.LOCATION,
            }
            return mappings.get(type_str)
    
    def _normalize_name(self, name: str, entity_type: EntityType) -> str:
        """Normalize entity name"""
        # Strip whitespace
        name = name.strip()
        
        # For persons, title case
        if entity_type == EntityType.PERSON:
            # Remove common prefixes like "Mr.", "Ms.", etc.
            prefixes = ["mr.", "ms.", "mrs.", "dr.", "prof."]
            lower_name = name.lower()
            for prefix in prefixes:
                if lower_name.startswith(prefix):
                    name = name[len(prefix):].strip()
                    break
            name = name.title()
        
        # For organizations, preserve case but clean up
        elif entity_type == EntityType.ORGANIZATION:
            name = ' '.join(name.split())  # Normalize whitespace
        
        return name
    
    def _merge_entities(
        self,
        base_entities: List[Entity],
        llm_entities: List[Entity]
    ) -> List[Entity]:
        """Merge base entities with LLM entities, preferring LLM for conflicts"""
        merged = {e.name.lower(): e for e in base_entities}
        
        for llm_entity in llm_entities:
            key = llm_entity.name.lower()
            if key in merged:
                # Merge mentions
                existing = merged[key]
                existing.mentions = list(set(existing.mentions + llm_entity.mentions))
                # Prefer LLM confidence if higher
                if llm_entity.confidence == "High":
                    existing.confidence = "High"
            else:
                merged[key] = llm_entity
        
        return list(merged.values())
    
    def _deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """Remove duplicate entities, merging mentions"""
        seen = {}
        
        for entity in entities:
            key = (entity.name.lower(), entity.entity_type)
            if key in seen:
                seen[key].mentions = list(set(seen[key].mentions + entity.mentions))
            else:
                seen[key] = entity
        
        return list(seen.values())


# Convenience function
def extract_entities(text: str, doc_id: str = "", use_llm: bool = True) -> List[Entity]:
    """Extract entities from text"""
    extractor = EntityExtractor(use_llm=use_llm)
    return extractor.extract_entities(text, doc_id)
