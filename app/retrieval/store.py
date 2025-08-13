import faiss
import numpy as np
import sqlite3
import pickle
import json
import os
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import logging
from pathlib import Path

from app.config import settings
from app.models import Evidence, Citation, Bundle

logger = logging.getLogger(__name__)

class VectorStore:
    """FAISS-based vector store with SQLite metadata"""
    
    def __init__(self, index_path: str = None, metadata_db_path: str = None):
        self.index_path = index_path or os.path.join(settings.DATA_DIR, "indices", "faiss.index")
        self.metadata_db_path = metadata_db_path or os.path.join(settings.DATA_DIR, "indices", "metadata.db")
        
        # Ensure directories exist
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.metadata_db_path), exist_ok=True)
        
        # Get dimension from embeddings client
        from app.deps import get_embeddings_client
        embeddings_client = get_embeddings_client()
        # Test with a small sample to get dimension
        test_embedding = embeddings_client.embed(["test"])
        self.dimension = len(test_embedding[0])
        logger.info(f"Using embedding dimension: {self.dimension}")
        
        self.index = None
        self.metadata_conn = None
        
        self._initialize_index()
        self._initialize_metadata_db()
    
    def _initialize_index(self):
        """Initialize or load FAISS index"""
        if os.path.exists(self.index_path):
            try:
                logger.info(f"Loading existing FAISS index from {self.index_path}")
                existing_index = faiss.read_index(self.index_path)
                
                # Check if dimensions match
                if existing_index.d != self.dimension:
                    logger.warning(f"Dimension mismatch: existing index has {existing_index.d}, need {self.dimension}. Recreating index.")
                    os.remove(self.index_path)
                    self.index = faiss.IndexFlatIP(self.dimension)
                else:
                    self.index = existing_index
                    logger.info(f"Loaded existing index with {self.index.ntotal} vectors")
            except Exception as e:
                logger.warning(f"Failed to load existing index: {e}. Creating new index.")
                if os.path.exists(self.index_path):
                    os.remove(self.index_path)
                self.index = faiss.IndexFlatIP(self.dimension)
        else:
            logger.info(f"Creating new FAISS index with dimension {self.dimension}")
            self.index = faiss.IndexFlatIP(self.dimension)
    
    def _initialize_metadata_db(self):
        """Initialize SQLite metadata database"""
        self.metadata_conn = sqlite3.connect(self.metadata_db_path, check_same_thread=False)
        self.metadata_conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id TEXT UNIQUE,
                url TEXT,
                title TEXT,
                content TEXT,
                chunk_text TEXT,
                embedding_id INTEGER,
                created_at TIMESTAMP,
                trust_score INTEGER,
                freshness_score REAL,
                topic_tags TEXT  -- JSON array
            )
        """)
        
        self.metadata_conn.execute("""
            CREATE TABLE IF NOT EXISTS bundles (
                id TEXT PRIMARY KEY,
                topic TEXT,
                query TEXT,
                created_at TIMESTAMP,
                source_count INTEGER,
                bundle_data TEXT  -- JSON serialized Bundle
            )
        """)
        
        self.metadata_conn.commit()
    
    def add_documents(self, documents: List[Dict[str, Any]], embeddings: np.ndarray) -> List[int]:
        """Add documents with embeddings to the store"""
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")
        
        # Normalize embeddings for cosine similarity
        embeddings = embeddings.astype('float32')
        faiss.normalize_L2(embeddings)
        
        # Add to FAISS index
        start_id = self.index.ntotal
        self.index.add(embeddings)
        
        # Add metadata to SQLite
        embedding_ids = list(range(start_id, start_id + len(documents)))
        
        for doc, embedding_id in zip(documents, embedding_ids):
            self.metadata_conn.execute("""
                INSERT OR REPLACE INTO documents 
                (doc_id, url, title, content, chunk_text, embedding_id, created_at, trust_score, freshness_score, topic_tags)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                doc.get('doc_id'),
                doc.get('url'),
                doc.get('title'),
                doc.get('content'),
                doc.get('chunk_text'),
                embedding_id,
                doc.get('created_at', datetime.utcnow()),
                doc.get('trust_score', 5),
                doc.get('freshness_score', 1.0),
                json.dumps(doc.get('topic_tags', []))
            ))
        
        self.metadata_conn.commit()
        self.save_index()
        
        return embedding_ids
    
    def search(self, query_embedding: np.ndarray, k: int = 10, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        if self.index.ntotal == 0:
            logger.warning("Index is empty, returning no results")
            return []
        
        # Normalize query embedding
        query_embedding = query_embedding.astype('float32').reshape(1, -1)
        faiss.normalize_L2(query_embedding)
        
        # Search FAISS index
        scores, indices = self.index.search(query_embedding, min(k * 2, self.index.ntotal))  # Get extra for filtering
        
        # Retrieve metadata
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for invalid indices
                continue
                
            # Get document metadata
            cursor = self.metadata_conn.execute("""
                SELECT doc_id, url, title, content, chunk_text, created_at, trust_score, freshness_score, topic_tags
                FROM documents WHERE embedding_id = ?
            """, (int(idx),))
            
            row = cursor.fetchone()
            if row:
                doc_id, url, title, content, chunk_text, created_at, trust_score, freshness_score, topic_tags = row
                
                doc = {
                    'doc_id': doc_id,
                    'url': url,
                    'title': title,
                    'content': content,
                    'chunk_text': chunk_text,
                    'created_at': created_at,
                    'trust_score': trust_score,
                    'freshness_score': freshness_score,
                    'topic_tags': json.loads(topic_tags) if topic_tags else [],
                    'similarity_score': float(score),
                    'embedding_id': int(idx)
                }
                
                # Apply filters
                if self._passes_filters(doc, filters):
                    results.append(doc)
        
        # Sort by combined score and return top k
        results = sorted(results, key=lambda x: self._calculate_combined_score(x), reverse=True)
        return results[:k]
    
    def _passes_filters(self, doc: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if document passes filter criteria"""
        if not filters:
            return True
        
        # Trust score filter
        if 'min_trust_score' in filters:
            if doc['trust_score'] < filters['min_trust_score']:
                return False
        
        # Freshness filter
        if 'min_freshness' in filters:
            if doc['freshness_score'] < filters['min_freshness']:
                return False
        
        # URL domain filter
        if 'allowed_domains' in filters and doc['url']:
            from urllib.parse import urlparse
            domain = urlparse(doc['url']).netloc
            if domain not in filters['allowed_domains']:
                return False
        
        return True
    
    def _calculate_combined_score(self, doc: Dict[str, Any]) -> float:
        """Calculate combined ranking score"""
        similarity = doc['similarity_score']
        trust = doc['trust_score'] / 10.0  # normalize to 0-1
        freshness = doc['freshness_score']
        
        # Weighted combination
        return (0.6 * similarity) + (0.3 * trust) + (0.1 * freshness)
    
    def save_bundle(self, bundle: Bundle) -> str:
        """Save evidence bundle to metadata store"""
        self.metadata_conn.execute("""
            INSERT OR REPLACE INTO bundles (id, topic, query, created_at, source_count, bundle_data)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            bundle.id,
            bundle.topic,
            bundle.query,
            bundle.created_at,
            bundle.source_count,
            bundle.json()
        ))
        self.metadata_conn.commit()
        return bundle.id
    
    def load_bundle(self, bundle_id: str) -> Optional[Bundle]:
        """Load evidence bundle from metadata store"""
        cursor = self.metadata_conn.execute("""
            SELECT bundle_data FROM bundles WHERE id = ?
        """, (bundle_id,))
        
        row = cursor.fetchone()
        if row:
            return Bundle.parse_raw(row[0])
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get store statistics"""
        doc_count = self.index.ntotal if self.index else 0
        
        cursor = self.metadata_conn.execute("SELECT COUNT(*) FROM bundles")
        bundle_count = cursor.fetchone()[0]
        
        return {
            'document_count': doc_count,
            'bundle_count': bundle_count,
            'index_path': self.index_path,
            'metadata_db_path': self.metadata_db_path
        }
    
    def save_index(self):
        """Save FAISS index to disk"""
        if self.index:
            faiss.write_index(self.index, self.index_path)
            logger.debug(f"Saved FAISS index to {self.index_path}")
    
    def close(self):
        """Close connections and save index"""
        self.save_index()
        if self.metadata_conn:
            self.metadata_conn.close()

# Singleton instance
_vector_store = None

def get_vector_store() -> VectorStore:
    """Get singleton vector store instance"""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store
