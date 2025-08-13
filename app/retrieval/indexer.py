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
        # For now, return hardcoded lists based on pipeline.yaml
        if list_type == "whitelist":
            return [
                "europa.eu", "nist.gov", "oecd.org", "who.int",
                "reuters.com", "bbc.com", "nature.com", "acm.org",
                "mit.edu", "stanford.edu", "gov.uk", "canada.ca"
            ]
        elif list_type == "blacklist":
            return ["contentfarm.*", "presswire.*", "*-seo-*"]
        return []
    
    def index_from_search(self, query: str, max_results: int = None) -> Bundle:
        """Search web and index results"""
        max_results = max_results or settings.MAX_SEARCH_RESULTS
        
        logger.info(f"Searching and indexing: {query}")
        
        # Search for documents
        search_results = self.search_client.search(query, max_results=max_results)
        
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
        self.vector_store.save_bundle(bundle)
        
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
        """Process a single search result"""
        url = result.get('url', '')
        
        # Apply domain filtering
        if not self._is_domain_allowed(url):
            logger.debug(f"Skipping blocked domain: {url}")
            return None
        
        # Basic content extraction (placeholder - would use app/io/scraper.py)
        content = result.get('content', result.get('snippet', ''))
        
        if len(content) < 50:  # Skip very short content
            return None
        
        return {
            'url': url,
            'title': result.get('title', ''),
            'content': content,
            'published_date': result.get('published_date'),
            'citation_num': citation_num
        }
    
    def _is_domain_allowed(self, url: str) -> bool:
        """Check if domain is allowed based on whitelist/blacklist"""
        if not url:
            return False
        
        try:
            domain = urlparse(url).netloc.lower()
            
            # Check blacklist first
            for blocked in self.blocked_domains:
                if blocked.replace('*', '') in domain:
                    return False
            
            # If whitelist exists, domain must be in it
            if self.trusted_domains:
                return any(trusted in domain for trusted in self.trusted_domains)
            
            return True
        
        except Exception:
            return False
    
    def _calculate_trust_score(self, url: str) -> int:
        """Calculate trust score based on domain"""
        if not url:
            return 5
        
        domain = urlparse(url).netloc.lower()
        
        # High trust domains
        high_trust = ['gov', 'edu', 'europa.eu', 'who.int', 'nist.gov']
        if any(ht in domain for ht in high_trust):
            return 9
        
        # Medium trust domains
        medium_trust = ['reuters.com', 'bbc.com', 'nature.com']
        if any(mt in domain for mt in medium_trust):
            return 7
        
        # Default trust
        return 5
    
    def _calculate_freshness_score(self, published_date: Optional[str]) -> float:
        """Calculate freshness score based on publication date"""
        if not published_date:
            return 0.5  # Unknown date gets neutral score
        
        try:
            # Parse date (implementation depends on date format)
            # For now, return a placeholder calculation
            days_old = 30  # Placeholder
            max_days = settings.FRESHNESS_DAYS
            
            if days_old <= 7:
                return 1.0
            elif days_old <= 30:
                return 0.8
            elif days_old <= max_days:
                return 0.6
            else:
                return 0.3
        
        except Exception:
            return 0.5

# Singleton instance
_indexer = None

def get_indexer() -> DocumentIndexer:
    """Get singleton indexer instance"""
    global _indexer
    if _indexer is None:
        _indexer = DocumentIndexer()
    return _indexer
