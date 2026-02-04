"""
Document Chunker Module
Smart chunking strategies for documents to optimize entity extraction
"""

from typing import List, Optional
from dataclasses import dataclass
from ingestion.document_loader import Document


@dataclass
class Chunk:
    """Represents a chunk of a document"""
    content: str
    source_doc_id: str
    chunk_index: int
    start_char: int
    end_char: int
    metadata: dict
    
    @property
    def id(self) -> str:
        return f"{self.source_doc_id}_chunk_{self.chunk_index}"


class DocumentChunker:
    """Chunk documents using various strategies"""
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        min_chunk_size: int = 100
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
    
    def chunk_document(self, document: Document) -> List[Chunk]:
        """Chunk a document based on its type"""
        if document.doc_type == 'email':
            return self._chunk_email(document)
        elif document.doc_type == 'pdf':
            return self._chunk_by_paragraphs(document)
        else:
            return self._chunk_by_size(document)
    
    def chunk_documents(self, documents: List[Document]) -> List[Chunk]:
        """Chunk multiple documents"""
        all_chunks = []
        for doc in documents:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)
        return all_chunks
    
    def _chunk_email(self, document: Document) -> List[Chunk]:
        """
        Special chunking for emails - keep header with body
        For short emails, keep as single chunk
        """
        content = document.content
        
        # Find the email body (after the headers)
        header_end = content.find('\n\n')
        if header_end == -1:
            header_end = 0
        
        header = content[:header_end + 2] if header_end > 0 else ""
        body = content[header_end + 2:] if header_end > 0 else content
        
        # For short emails, keep as single chunk
        if len(content) <= self.chunk_size * 1.5:
            return [Chunk(
                content=content,
                source_doc_id=document.id,
                chunk_index=0,
                start_char=0,
                end_char=len(content),
                metadata={
                    **document.metadata,
                    'doc_type': document.doc_type,
                    'source': document.source,
                }
            )]
        
        # For longer emails, chunk the body and prepend header context
        chunks = []
        chunk_index = 0
        pos = 0
        
        while pos < len(body):
            end_pos = min(pos + self.chunk_size, len(body))
            
            # Try to break at a paragraph or sentence
            if end_pos < len(body):
                # Look for paragraph break
                para_break = body.rfind('\n\n', pos, end_pos)
                if para_break > pos + self.min_chunk_size:
                    end_pos = para_break + 2
                else:
                    # Look for sentence break
                    sentence_break = body.rfind('. ', pos, end_pos)
                    if sentence_break > pos + self.min_chunk_size:
                        end_pos = sentence_break + 2
            
            chunk_content = header + body[pos:end_pos] if chunk_index == 0 else f"[continued...]\n{body[pos:end_pos]}"
            
            chunks.append(Chunk(
                content=chunk_content,
                source_doc_id=document.id,
                chunk_index=chunk_index,
                start_char=pos,
                end_char=end_pos,
                metadata={
                    **document.metadata,
                    'doc_type': document.doc_type,
                    'source': document.source,
                }
            ))
            
            pos = end_pos - self.chunk_overlap
            chunk_index += 1
        
        return chunks
    
    def _chunk_by_paragraphs(self, document: Document) -> List[Chunk]:
        """Chunk by paragraphs, combining small paragraphs"""
        content = document.content
        paragraphs = content.split('\n\n')
        
        chunks = []
        current_chunk = ""
        current_start = 0
        chunk_index = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # If adding this paragraph exceeds chunk size, save current and start new
            if len(current_chunk) + len(para) + 2 > self.chunk_size and len(current_chunk) >= self.min_chunk_size:
                chunks.append(Chunk(
                    content=current_chunk.strip(),
                    source_doc_id=document.id,
                    chunk_index=chunk_index,
                    start_char=current_start,
                    end_char=current_start + len(current_chunk),
                    metadata={
                        **document.metadata,
                        'doc_type': document.doc_type,
                        'source': document.source,
                    }
                ))
                chunk_index += 1
                current_start = current_start + len(current_chunk) - self.chunk_overlap
                # Keep some overlap
                current_chunk = current_chunk[-self.chunk_overlap:] if len(current_chunk) > self.chunk_overlap else ""
            
            current_chunk += para + "\n\n"
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append(Chunk(
                content=current_chunk.strip(),
                source_doc_id=document.id,
                chunk_index=chunk_index,
                start_char=current_start,
                end_char=current_start + len(current_chunk),
                metadata={
                    **document.metadata,
                    'doc_type': document.doc_type,
                    'source': document.source,
                }
            ))
        
        return chunks
    
    def _chunk_by_size(self, document: Document) -> List[Chunk]:
        """Simple size-based chunking with overlap"""
        content = document.content
        chunks = []
        pos = 0
        chunk_index = 0
        
        while pos < len(content):
            end_pos = min(pos + self.chunk_size, len(content))
            
            # Try to break at word boundary
            if end_pos < len(content):
                space_pos = content.rfind(' ', pos, end_pos)
                if space_pos > pos + self.min_chunk_size:
                    end_pos = space_pos + 1
            
            chunk_text = content[pos:end_pos].strip()
            
            if chunk_text:
                chunks.append(Chunk(
                    content=chunk_text,
                    source_doc_id=document.id,
                    chunk_index=chunk_index,
                    start_char=pos,
                    end_char=end_pos,
                    metadata={
                        **document.metadata,
                        'doc_type': document.doc_type,
                        'source': document.source,
                    }
                ))
                chunk_index += 1
            
            pos = end_pos - self.chunk_overlap
        
        return chunks


# Convenience function
def chunk_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[Chunk]:
    """Chunk a list of documents"""
    chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return chunker.chunk_documents(documents)
