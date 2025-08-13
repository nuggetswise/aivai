from app.models import SourceDoc
from app.config import settings
from typing import List
import os
import asyncio
import hashlib
import re
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from urllib.parse import urlparse, urljoin
import logging
import numpy as np
import yaml

from app.config import settings
from app.deps import get_search_client, get_embeddings_client
from app.retrieval.store import get_vector_store
from app.models import Bundle, Evidence, Citation, CitationType

logger = logging.getLogger(__name__)

class ContentChunker:
    """Handles text chunking for embedding"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Split text into overlapping chunks"""
        if not text or len(text) < 50:
            return []
        
        # Clean text
        text = self._clean_text(text)
        
        # Split into sentences for better chunk boundaries
        sentences = re.split(r'[.!?]+', text)
        
        chunks = []
        current_chunk = ""
        current_length = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            sentence_length = len(sentence)
            
            # If adding this sentence would exceed chunk size
            if current_length + sentence_length > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_id = self._generate_chunk_id(current_chunk, metadata)
                chunks.append({
                    'chunk_id': chunk_id,
                    'text': current_chunk.strip(),
                    'metadata': metadata or {},
                    'start_pos': len(chunks) * (self.chunk_size - self.chunk_overlap),
                    'length': len(current_chunk)
                })
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk)
                current_chunk = overlap_text + " " + sentence
                current_length = len(current_chunk)
            else:
                # Add sentence to current chunk
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
                current_length += sentence_length
        
        # Don't forget the last chunk
        if current_chunk.strip():
            chunk_id = self._generate_chunk_id(current_chunk, metadata)
            chunks.append({
                'chunk_id': chunk_id,
                'text': current_chunk.strip(),
                'metadata': metadata or {},
                'start_pos': len(chunks) * (self.chunk_size - self.chunk_overlap),
                'length': len(current_chunk)
            })
        
        return chunks
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s.,!?;:()\-"]', '', text)
        return text.strip()
    
    def _get_overlap_text(self, text: str) -> str:
        """Get overlap text from end of chunk"""
        words = text.split()
        overlap_words = int(len(words) * 0.1)  # 10% overlap
        return " ".join(words[-overlap_words:]) if overlap_words > 0 else ""
    
    def _generate_chunk_id(self, text: str, metadata: Dict[str, Any]) -> str:
        """Generate unique ID for chunk"""
        content = text + str(metadata.get('url', ''))
        return hashlib.md5(content.encode()).hexdigest()[:12]

class DocumentIndexer:
    """Handles document fetching, processing, and indexing"""
    
    def __init__(self):
        self.search_client = get_search_client()
        self.embeddings_client = get_embeddings_client()
        self.vector_store = get_vector_store()
        self.chunker = ContentChunker()
        
        # Load domain whitelist/blacklist
        self.trusted_domains = self._load_domain_list("whitelist")
        self.blocked_domains = self._load_domain_list("blacklist")
    
    def _load_domain_list(self, list_type: str) -> List[str]:
        """Load domain whitelist or blacklist"""
        # For testing, use a more permissive whitelist that includes legitimate sources
        if list_type == "whitelist":
            return [
                # Government and official sources
                "gov", "edu", "europa.eu", "who.int", "nist.gov", "nasa.gov", 
                "noaa.gov", "gov.uk", "canada.ca", "undp.org",
                
                # Academic and research
                "mit.edu", "stanford.edu", "ucar.edu", "nature.com", "acm.org",
                
                # Trusted news and media
                "reuters.com", "bbc.com", "oecd.org",
                
                # Other legitimate sources
                "wikipedia.org", "earth.org"
            ]
        elif list_type == "blacklist":
            return ["contentfarm.*", "presswire.*", "*-seo-*"]
        return []
    
    def index_from_search(self, query: str, max_results: int = None) -> Bundle:
        """Search web and index results"""
        max_results = max_results or settings.MAX_SEARCH_RESULTS
        
        logger.info(f"Searching and indexing: {query}")
        
        # Search for documents with raw content
        search_results = self.search_client.search(
            query, 
            max_results=max_results,
            include_raw_content=True,  # Use Tavily's raw content to avoid scraping issues
            include_answer=False,
            include_images=False
        )
        
        # Process and index documents
        documents = []
        citations = []
        
        for i, result in enumerate(search_results[:max_results]):
            try:
                doc = self._process_search_result(result, i + 1)
                if doc:
                    documents.append(doc)
                    
                    # Create citation
                    citation = Citation(
                        id=f"S{i + 1}",
                        type=CitationType.WEB,
                        url=result.get('url'),
                        title=result.get('title'),
                        excerpt=result.get('content', '')[:200] + "...",
                        confidence=result.get('score', 0.8),
                        timestamp=datetime.utcnow(),
                        trust_score=self._calculate_trust_score(result.get('url', ''))
                    )
                    citations.append(citation)
            
            except Exception as e:
                logger.warning(f"Failed to process search result {i + 1}: {e}")
                continue
        
        # Chunk documents and generate embeddings
        all_chunks = []
        for doc in documents:
            chunks = self.chunker.chunk_text(doc['content'], doc)
            all_chunks.extend(chunks)
        
        if all_chunks:
            # Generate embeddings for all chunks
            chunk_texts = [chunk['text'] for chunk in all_chunks]
            try:
                embeddings = self.embeddings_client.embed(chunk_texts)
                embeddings_array = np.array(embeddings)
                
                # Add to vector store
                chunk_docs = []
                for chunk, embedding in zip(all_chunks, embeddings):
                    chunk_doc = {
                        'doc_id': chunk['chunk_id'],
                        'url': chunk['metadata'].get('url'),
                        'title': chunk['metadata'].get('title'),
                        'content': chunk['metadata'].get('content', ''),
                        'chunk_text': chunk['text'],
                        'created_at': datetime.utcnow(),
                        'trust_score': self._calculate_trust_score(chunk['metadata'].get('url', '')),
                        'freshness_score': self._calculate_freshness_score(chunk['metadata'].get('published_date')),
                        'topic_tags': [query]  # Simple topic tagging
                    }
                    chunk_docs.append(chunk_doc)
                
                # Add to vector store
                self.vector_store.add_documents(chunk_docs, embeddings_array)
                
                logger.info(f"Indexed {len(chunk_docs)} chunks from {len(documents)} documents")
            except Exception as e:
                logger.warning(f"Failed to generate embeddings or add to vector store: {e}")
        
        # Create evidence bundle
        evidence_list = []
        for doc, citation in zip(documents, citations):
            evidence = Evidence(
                text=doc['content'][:500] + "..." if len(doc['content']) > 500 else doc['content'],
                citations=[citation],
                confidence=citation.confidence,
                topic_relevance=1.0  # Will be refined by ranking
            )
            evidence_list.append(evidence)
        
        bundle = Bundle(
            topic=query,
            query=query,
            claims=evidence_list,
            source_count=len(documents),
            freshness_score=np.mean([self._calculate_freshness_score(doc.get('published_date')) for doc in documents]) if documents else 0.0
        )
        
        # Save bundle to store
        try:
            self.vector_store.save_bundle(bundle)
        except Exception as e:
            logger.warning(f"Failed to save bundle: {e}")
        
        return bundle
    
    def index_local_corpus(self, corpus_path: str) -> int:
        """Index documents from local corpus directory"""
        from pathlib import Path
        
        corpus_dir = Path(corpus_path)
        if not corpus_dir.exists():
            logger.warning(f"Corpus directory not found: {corpus_path}")
            return 0
        
        document_count = 0
        all_chunks = []
        
        # Process all text files in corpus
        for file_path in corpus_dir.rglob("*.txt"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if len(content) < 100:  # Skip very short files
                    continue
                
                metadata = {
                    'url': f"file://{file_path}",
                    'title': file_path.stem,
                    'content': content,
                    'source_type': 'local_corpus'
                }
                
                chunks = self.chunker.chunk_text(content, metadata)
                all_chunks.extend(chunks)
                document_count += 1
                
            except Exception as e:
                logger.warning(f"Failed to process corpus file {file_path}: {e}")
                continue
        
        if all_chunks:
            # Generate embeddings
            chunk_texts = [chunk['text'] for chunk in all_chunks]
            embeddings = self.embeddings_client.embed(chunk_texts)
            embeddings_array = np.array(embeddings)
            
            # Prepare documents for vector store
            chunk_docs = []
            for chunk in all_chunks:
                chunk_doc = {
                    'doc_id': chunk['chunk_id'],
                    'url': chunk['metadata']['url'],
                    'title': chunk['metadata']['title'],
                    'content': chunk['metadata']['content'],
                    'chunk_text': chunk['text'],
                    'created_at': datetime.utcnow(),
                    'trust_score': 8,  # Local corpus is trusted
                    'freshness_score': 0.8,  # Assume reasonably fresh
                    'topic_tags': ['corpus']
                }
                chunk_docs.append(chunk_doc)
            
            # Add to vector store
            self.vector_store.add_documents(chunk_docs, embeddings_array)
            
            logger.info(f"Indexed {len(chunk_docs)} chunks from {document_count} local documents")
        
        return document_count
    
    def _process_search_result(self, result: Dict[str, Any], citation_num: int) -> Optional[Dict[str, Any]]:
        """Process a single search result - accept all results and let LLM prompts handle quality assessment"""
        url = result.get('url', '')
        
        # Prefer Tavily's raw_content if available, fallback to content/snippet
        raw_content = result.get('raw_content', '')
        content = result.get('content', '')
        snippet = result.get('snippet', '')
        
        # Use the best available content source
        if raw_content and len(raw_content) > len(content):
            used_content = raw_content
        elif content:
            used_content = content
        else:
            used_content = snippet
        
        # Clean and extract meaningful content
        used_content = self._clean_extracted_content(used_content)
        
        # Only filter out very short content after cleaning
        if not used_content or len(used_content) < 100:
            logger.debug(f"Skipping result with insufficient content: {url}")
            return None
        
        # Accept the result and let LLM prompts handle source quality assessment
        return {
            'url': url,
            'title': result.get('title', ''),
            'content': used_content,
            'published_date': result.get('published_date'),
            'citation_num': citation_num,
            # Still calculate trust score for ranking, but don't filter on it
            'trust_score': self._calculate_trust_score(url)
        }
    
    def _clean_extracted_content(self, content: str) -> str:
        """Clean extracted content to remove navigation, images, and boilerplate"""
        if not content:
            return ""
        
        # Remove common navigation patterns and page elements
        lines = content.split('\n')
        cleaned_lines = []
        
        # Flag to track if we're in main content
        in_main_content = False
        
        # Extra aggressive cleaning - remove all EU domain notices and standard web elements
        skip_patterns = [
            # EU/Gov notices
            "official european union website", "europa.eu", "**europa.eu**", "institutions and bodies",
            "**lock**", "https://", "safely connected", "share sensitive information",
            # Navigation
            "skip to", "jump to", "main page", "contents", "random article",
            "menu", "search", "login", "register", "home page", "main navigation",
            # Links patterns
            "[home]", "[data]", "[tools]", "[training]", "[about]", "[contact]",
            # Headers/footers
            "breadcrumb", "sidebar", "cookie policy", "privacy policy", "copyright",
            "all rights reserved", "terms of use", "disclaimer",
            # UI elements
            "close", "share", "link copied", "clipboard"
        ]
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
            
            # Skip lines with these patterns
            if any(pattern.lower() in line.lower() for pattern in skip_patterns):
                continue
            
            # Remove image tags and alt text patterns
            if line.startswith('![') or line.startswith('<img') or '](' in line and any(ext in line for ext in ['.png', '.jpg', '.gif', '.svg']):
                continue
                
            # Skip lines starting with common navigation markers
            if line.startswith('*') and len(line) < 100 and '[' in line and ']' in line:
                continue
            
            # Skip links that are likely navigation
            if line.startswith('[') and '](' in line and len(line) < 80:
                continue
                
            # Skip lines that are just navigation or UI elements
            if line.lower() in ['home', 'about', 'contact', 'news', 'services', 'faq', 'help']:
                continue
                
            # Skip lines with only special characters, numbers, or very short text
            if len(line) < 20 or all(c.isdigit() or c in '-|>/\\.' for c in line.replace(' ', '')):
                continue
            
            # If we see patterns that indicate the start of main content, mark as in main content
            if any(starter in line.lower() for starter in ['introduction', 'abstract', 'summary', 'overview']):
                in_main_content = True
            
            # Add line to cleaned content
            cleaned_lines.append(line)
        
        # Rejoin and extract meaningful paragraphs
        cleaned_content = '\n'.join(cleaned_lines)
        
        # Remove HTML tags if any remain
        cleaned_content = re.sub(r'<[^>]+>', '', cleaned_content)
        # Remove markdown links but keep the text
        cleaned_content = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', cleaned_content)
        # Remove remaining URLs
        cleaned_content = re.sub(r'https?://\S+', '', cleaned_content)
        
        # Find paragraphs (text blocks with substantial content)
        paragraphs = []
        current_paragraph = []
        
        for line in cleaned_lines:
            if len(line) > 50:  # Likely a substantial content line
                current_paragraph.append(line)
            else:
                if current_paragraph and len(' '.join(current_paragraph)) > 100:
                    paragraphs.append(' '.join(current_paragraph))
                current_paragraph = []
        
        # Don't forget the last paragraph
        if current_paragraph and len(' '.join(current_paragraph)) > 100:
            paragraphs.append(' '.join(current_paragraph))
        
        # Return the first few substantial paragraphs or cleaned content
        if paragraphs:
            return '\n\n'.join(paragraphs[:5])  # First 5 substantial paragraphs
        else:
            # Fallback: return cleaned content if no substantial paragraphs found
            return cleaned_content[:2000] if len(cleaned_content) > 100 else ""
    
    def _calculate_trust_score(self, url: str) -> float:
        """Calculate trust score for a URL - use softer scoring instead of hard filtering"""
        if not url:
            return 0.5
        
        domain = urlparse(url).netloc.lower()
        
        # High trust domains
        if any(trusted in domain for trusted in ['.gov', '.edu', 'europa.eu', 'who.int', 'nist.gov', 'nasa.gov', 'nature.com', 'acm.org']):
            return 1.0
        
        # Medium-high trust
        if any(medium in domain for medium in ['reuters.com', 'bbc.com', 'oecd.org', 'wikipedia.org']):
            return 0.9
        
        # Medium trust
        if any(ok in domain for ok in ['.org', '.com']):
            return 0.7
        
        # Lower trust but not blocked
        if any(lower in domain for lower in ['blog', 'press']):
            return 0.4
        
        # Default moderate trust
        return 0.6
    
    def _calculate_freshness_score(self, published_date: Optional[str]) -> float:
        """Calculate freshness score based on publication date"""
        if not published_date:
            return 0.5  # Unknown date gets neutral score
        
        try:
            # Parse date (assuming ISO format or common formats)
            from dateutil import parser
            pub_date = parser.parse(published_date)
            
            # Calculate age in days
            age_days = (datetime.utcnow() - pub_date.replace(tzinfo=None)).days
            
            # Fresher content gets higher scores
            if age_days <= 30:
                return 1.0
            elif age_days <= 90:
                return 0.8
            elif age_days <= 365:
                return 0.6
            elif age_days <= 730:
                return 0.4
            else:
                return 0.2
                
        except Exception:
            return 0.5  # Default for unparseable dates
    
    def extract_evidence_from_document(self, document: str) -> List[Evidence]:
        """Extract evidence from a local document"""
        evidence_list = []
        
        try:
            # For now, treat the whole document as one piece of evidence
            # Could be enhanced to extract specific claims
            citation = Citation(
                id=f"L{len(evidence_list) + 1}",
                type=CitationType.LOCAL,
                url=f"local://{document}",
                title=document,
                excerpt=document[:200] + "..." if len(document) > 200 else document,
                confidence=0.9,
                timestamp=datetime.utcnow(),
                trust_score=0.9  # Local documents are trusted
            )
            
            evidence = Evidence(
                text=document,
                citations=[citation],
                confidence=0.9,
                topic_relevance=1.0
            )
            
            evidence_list.append(evidence)
            
        except Exception as e:
            logger.warning(f"Failed to extract evidence from document: {e}")
        
        return evidence_list

# Singleton instance
_indexer = None

def get_indexer() -> DocumentIndexer:
    """Get singleton indexer instance"""
    global _indexer
    if _indexer is None:
        _indexer = DocumentIndexer()
    return _indexer
