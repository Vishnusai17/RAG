"""
Document Loader Module
Handles loading various document types: text files, PDFs, emails, etc.
"""

import os
import re
import email
from email import policy
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field

from pypdf import PdfReader
from bs4 import BeautifulSoup


@dataclass
class Document:
    """Represents a loaded document with content and metadata"""
    content: str
    source: str
    doc_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def id(self) -> str:
        """Generate a unique ID based on source path"""
        return f"doc_{hash(self.source) % 10**8}"


class DocumentLoader:
    """Load documents from various file formats"""
    
    SUPPORTED_EXTENSIONS = {
        '.txt': 'text',
        '.pdf': 'pdf',
        '.eml': 'email',
        '.msg': 'email',
        '.html': 'html',
        '.htm': 'html',
    }
    
    def __init__(self):
        self.loaded_docs: List[Document] = []
    
    def load_directory(self, directory: str, recursive: bool = True) -> List[Document]:
        """Load all supported documents from a directory"""
        documents = []
        path = Path(directory)
        
        if not path.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        pattern = '**/*' if recursive else '*'
        
        for file_path in path.glob(pattern):
            if file_path.is_file():
                ext = file_path.suffix.lower()
                if ext in self.SUPPORTED_EXTENSIONS:
                    try:
                        doc = self.load_file(str(file_path))
                        if doc:
                            documents.append(doc)
                    except Exception as e:
                        print(f"Error loading {file_path}: {e}")
        
        self.loaded_docs.extend(documents)
        return documents
    
    def load_file(self, file_path: str) -> Optional[Document]:
        """Load a single file based on its extension"""
        path = Path(file_path)
        ext = path.suffix.lower()
        
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {ext}")
        
        doc_type = self.SUPPORTED_EXTENSIONS[ext]
        
        loaders = {
            'text': self._load_text,
            'pdf': self._load_pdf,
            'email': self._load_email,
            'html': self._load_html,
        }
        
        return loaders[doc_type](file_path)
    
    def _load_text(self, file_path: str) -> Document:
        """Load a plain text file"""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        return Document(
            content=content,
            source=file_path,
            doc_type='text',
            metadata={
                'filename': Path(file_path).name,
                'loaded_at': datetime.now().isoformat(),
            }
        )
    
    def _load_pdf(self, file_path: str) -> Document:
        """Load a PDF file"""
        reader = PdfReader(file_path)
        
        pages = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                pages.append(f"[Page {i+1}]\n{text}")
        
        content = "\n\n".join(pages)
        
        # Extract metadata
        pdf_meta = reader.metadata or {}
        
        return Document(
            content=content,
            source=file_path,
            doc_type='pdf',
            metadata={
                'filename': Path(file_path).name,
                'page_count': len(reader.pages),
                'title': pdf_meta.get('/Title', ''),
                'author': pdf_meta.get('/Author', ''),
                'loaded_at': datetime.now().isoformat(),
            }
        )
    
    def _load_email(self, file_path: str) -> Document:
        """Load an email file (.eml format)"""
        with open(file_path, 'rb') as f:
            msg = email.message_from_binary_file(f, policy=policy.default)
        
        # Extract headers
        sender = msg.get('From', '')
        recipients = msg.get('To', '')
        cc = msg.get('Cc', '')
        subject = msg.get('Subject', '')
        date = msg.get('Date', '')
        
        # Extract body
        body = ""
        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                if content_type == 'text/plain':
                    payload = part.get_payload(decode=True)
                    if payload:
                        body = payload.decode('utf-8', errors='ignore')
                        break
                elif content_type == 'text/html' and not body:
                    payload = part.get_payload(decode=True)
                    if payload:
                        html_content = payload.decode('utf-8', errors='ignore')
                        soup = BeautifulSoup(html_content, 'html.parser')
                        body = soup.get_text(separator='\n', strip=True)
        else:
            payload = msg.get_payload(decode=True)
            if payload:
                body = payload.decode('utf-8', errors='ignore')
        
        # Structure the content for entity extraction
        content = f"""From: {sender}
To: {recipients}
CC: {cc}
Subject: {subject}
Date: {date}

{body}"""
        
        return Document(
            content=content,
            source=file_path,
            doc_type='email',
            metadata={
                'filename': Path(file_path).name,
                'sender': self._extract_email_address(sender),
                'sender_name': self._extract_name_from_email(sender),
                'recipients': self._parse_recipient_list(recipients),
                'cc': self._parse_recipient_list(cc),
                'subject': subject,
                'date': date,
                'loaded_at': datetime.now().isoformat(),
            }
        )
    
    def _load_html(self, file_path: str) -> Document:
        """Load an HTML file"""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            html_content = f.read()
        
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style elements
        for element in soup(['script', 'style', 'nav', 'footer']):
            element.decompose()
        
        # Get text content
        text = soup.get_text(separator='\n', strip=True)
        
        # Get title
        title = soup.title.string if soup.title else ''
        
        return Document(
            content=text,
            source=file_path,
            doc_type='html',
            metadata={
                'filename': Path(file_path).name,
                'title': title,
                'loaded_at': datetime.now().isoformat(),
            }
        )
    
    def _extract_email_address(self, email_string: str) -> str:
        """Extract email address from a string like 'Name <email@example.com>'"""
        match = re.search(r'<([^>]+)>', email_string)
        if match:
            return match.group(1)
        # If no angle brackets, assume the whole string is the email
        return email_string.strip()
    
    def _extract_name_from_email(self, email_string: str) -> str:
        """Extract name from email string"""
        match = re.search(r'^([^<]+)<', email_string)
        if match:
            return match.group(1).strip().strip('"')
        return ''
    
    def _parse_recipient_list(self, recipients: str) -> List[Dict[str, str]]:
        """Parse a comma-separated list of recipients"""
        if not recipients:
            return []
        
        result = []
        for recipient in recipients.split(','):
            recipient = recipient.strip()
            if recipient:
                result.append({
                    'email': self._extract_email_address(recipient),
                    'name': self._extract_name_from_email(recipient),
                })
        return result


# Convenience function
def load_documents(path: str, recursive: bool = True) -> List[Document]:
    """Load documents from a file or directory"""
    loader = DocumentLoader()
    
    if os.path.isfile(path):
        doc = loader.load_file(path)
        return [doc] if doc else []
    else:
        return loader.load_directory(path, recursive)
