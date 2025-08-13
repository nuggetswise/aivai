from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import logging
from datetime import datetime, timedelta
from urllib.parse import urlparse

from app.config import settings
from app.models import Bundle, Evidence, Citation
from app.deps import get_embeddings_client
from app.retrieval.store import get_vector_store

logger = logging.getLogger(__name__)

class EvidenceRanker:
    """Handles ranking and filtering of evidence for debate turns"""
    
    def __init__(self):
        self.embeddings_client = get_embeddings_client()
        self.vector_store = get_vector_store()
        
        # Ranking weights
        self.weights = {
            'semantic_similarity': 0.4,
            'trust_score': 0.3,
            'freshness': 0.2,
            'citation_quality': 0.1
        }
    
    def rank_evidence_for_query(self, query: str, max_results: int = 10, filters: Dict[str, Any] = None) -> Bundle:
        """Retrieve and rank evidence for a specific query"""
        logger.info(f"Ranking evidence for query: {query}")
        
        # Generate query embedding
        query_embeddings = self.embeddings_client.embed([query])
        query_embedding = np.array(query_embeddings[0])
        
        # Apply filters
        search_filters = self._build_search_filters(filters)
        
        # Search vector store
        search_results = self.vector_store.search(
            query_embedding, 
            k=max_results * 2,  # Get extra for ranking
            filters=search_filters
        )
        
        if not search_results:
            logger.warning(f"No evidence found for query: {query}")
            return Bundle(
                topic=query,
                query=query,
                claims=[],
                source_count=0,
                freshness_score=0.0
            )
        
        # Convert search results to evidence
        evidence_list = []
        citations_used = set()
        
        for i, result in enumerate(search_results[:max_results]):
            try:
                evidence = self._create_evidence_from_result(result, i + 1, citations_used)
                if evidence:
                    evidence_list.append(evidence)
            except Exception as e:
                logger.warning(f"Failed to create evidence from result {i}: {e}")
                continue
        
        # Detect contradictions
        contradictions = self._detect_contradictions(evidence_list)
        
        # Calculate bundle-level metrics
        freshness_scores = [e.topic_relevance for e in evidence_list if e.topic_relevance]
        avg_freshness = np.mean(freshness_scores) if freshness_scores else 0.0
        
        bundle = Bundle(
            topic=query,
            query=query,
            claims=evidence_list,
            contradictions=contradictions,
            source_count=len(set(e.citations[0].url for e in evidence_list if e.citations)),
            freshness_score=avg_freshness
        )
        
        # Save bundle
        self.vector_store.save_bundle(bundle)
        
        return bundle
    
    def rerank_evidence(self, evidence_list: List[Evidence], query: str, intent: str = "general") -> List[Evidence]:
        """Re-rank evidence based on query and intent"""
        if not evidence_list:
            return evidence_list
        
        # Calculate enhanced scores for each evidence
        scored_evidence = []
        for evidence in evidence_list:
            score = self._calculate_enhanced_score(evidence, query, intent)
            scored_evidence.append((evidence, score))
        
        # Sort by score and return
        scored_evidence.sort(key=lambda x: x[1], reverse=True)
        return [evidence for evidence, score in scored_evidence]
    
    def filter_contradictory_evidence(self, evidence_list: List[Evidence]) -> Tuple[List[Evidence], List[str]]:
        """Filter out contradictory evidence and return filtered list with notes"""
        if len(evidence_list) <= 1:
            return evidence_list, []
        
        filtered_evidence = []
        contradiction_notes = []
        
        # Simple contradiction detection based on keywords
        contradiction_patterns = [
            (['increase', 'rise', 'grow', 'up'], ['decrease', 'fall', 'decline', 'down']),
            (['safe', 'secure', 'protected'], ['dangerous', 'risky', 'threat']),
            (['effective', 'successful', 'works'], ['ineffective', 'failed', 'doesn\'t work']),
            (['support', 'approve', 'favor'], ['oppose', 'reject', 'against'])
        ]
        
        for evidence in evidence_list:
            text_lower = evidence.text.lower()
            is_contradictory = False
            
            # Check against already filtered evidence
            for existing in filtered_evidence:
                existing_text_lower = existing.text.lower()
                
                for positive_words, negative_words in contradiction_patterns:
                    has_positive = any(word in text_lower for word in positive_words)
                    has_negative = any(word in existing_text_lower for word in negative_words)
                    
                    existing_has_positive = any(word in existing_text_lower for word in positive_words)
                    has_negative_current = any(word in text_lower for word in negative_words)
                    
                    if (has_positive and has_negative) or (existing_has_positive and has_negative_current):
                        is_contradictory = True
                        contradiction_notes.append(
                            f"Evidence contradicts existing claim: '{evidence.text[:100]}...' vs '{existing.text[:100]}...'"
                        )
                        break
                
                if is_contradictory:
                    break
            
            if not is_contradictory:
                filtered_evidence.append(evidence)
        
        return filtered_evidence, contradiction_notes
    
    def _build_search_filters(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Build search filters from ranking parameters"""
        search_filters = {}
        
        if filters:
            # Trust score filter
            if 'min_trust_score' in filters:
                search_filters['min_trust_score'] = filters['min_trust_score']
            elif settings.TRUST_SCORE_THRESHOLD:
                search_filters['min_trust_score'] = settings.TRUST_SCORE_THRESHOLD
            
            # Freshness filter
            if 'min_freshness' in filters:
                search_filters['min_freshness'] = filters['min_freshness']
            
            # Domain filter
            if 'allowed_domains' in filters:
                search_filters['allowed_domains'] = filters['allowed_domains']
        
        return search_filters
    
    def _create_evidence_from_result(self, result: Dict[str, Any], citation_num: int, citations_used: set) -> Optional[Evidence]:
        """Create Evidence object from search result"""
        url = result.get('url', '')
        
        # Avoid duplicate citations from same URL
        if url in citations_used:
            return None
        
        citations_used.add(url)
        
        # Create citation
        citation = Citation(
            id=f"S{citation_num}",
            type="web",
            url=url,
            title=result.get('title', ''),
            excerpt=result.get('chunk_text', '')[:200] + "...",
            confidence=min(result.get('similarity_score', 0.7), 1.0),
            timestamp=datetime.utcnow(),
            trust_score=result.get('trust_score', 5)
        )
        
        # Create evidence
        evidence = Evidence(
            text=result.get('chunk_text', ''),
            citations=[citation],
            confidence=citation.confidence,
            topic_relevance=result.get('similarity_score', 0.7)
        )
        
        return evidence
    
    def _calculate_enhanced_score(self, evidence: Evidence, query: str, intent: str) -> float:
        """Calculate enhanced ranking score based on multiple factors"""
        base_score = evidence.confidence
        
        # Trust score component
        trust_component = 0.0
        if evidence.citations:
            avg_trust = np.mean([cite.trust_score or 5 for cite in evidence.citations])
            trust_component = avg_trust / 10.0  # Normalize to 0-1
        
        # Freshness component (from topic_relevance as proxy)
        freshness_component = evidence.topic_relevance
        
        # Citation quality component
        citation_quality = self._calculate_citation_quality(evidence.citations)
        
        # Intent-specific adjustments
        intent_multiplier = self._get_intent_multiplier(evidence.text, intent)
        
        # Weighted combination
        enhanced_score = (
            self.weights['semantic_similarity'] * base_score +
            self.weights['trust_score'] * trust_component +
            self.weights['freshness'] * freshness_component +
            self.weights['citation_quality'] * citation_quality
        ) * intent_multiplier
        
        return enhanced_score
    
    def _calculate_citation_quality(self, citations: List[Citation]) -> float:
        """Calculate quality score for citations"""
        if not citations:
            return 0.0
        
        quality_factors = []
        
        for citation in citations:
            quality = 0.5  # Base quality
            
            # URL quality
            if citation.url:
                domain = urlparse(citation.url).netloc.lower()
                if any(trusted in domain for trusted in ['gov', 'edu', '.org']):
                    quality += 0.3
                elif any(news in domain for news in ['reuters', 'bbc', 'ap']):
                    quality += 0.2
            
            # Title quality
            if citation.title and len(citation.title) > 10:
                quality += 0.1
            
            # Excerpt quality
            if citation.excerpt and len(citation.excerpt) > 50:
                quality += 0.1
            
            quality_factors.append(min(quality, 1.0))
        
        return np.mean(quality_factors)
    
    def _get_intent_multiplier(self, text: str, intent: str) -> float:
        """Get intent-specific score multiplier"""
        text_lower = text.lower()
        
        intent_keywords = {
            'opening': ['introduction', 'overview', 'generally', 'broadly'],
            'positioning': ['argue', 'position', 'stance', 'believe', 'maintain'],
            'rebuttal': ['however', 'but', 'contrary', 'disagree', 'refute'],
            'closing': ['conclusion', 'summary', 'therefore', 'in summary']
        }
        
        keywords = intent_keywords.get(intent, [])
        if any(keyword in text_lower for keyword in keywords):
            return 1.2  # Boost score for intent-relevant content
        
        return 1.0  # No change
    
    def _detect_contradictions(self, evidence_list: List[Evidence]) -> List[str]:
        """Detect contradictions within evidence list"""
        if len(evidence_list) <= 1:
            return []
        
        contradictions = []
        
        # Simple contradiction detection using opposing keywords
        opposing_pairs = [
            (['increase', 'rise', 'more', 'higher'], ['decrease', 'fall', 'less', 'lower']),
            (['safe', 'secure', 'protect'], ['dangerous', 'risk', 'threat', 'harm']),
            (['effective', 'successful', 'beneficial'], ['ineffective', 'failed', 'harmful']),
            (['support', 'favor', 'recommend'], ['oppose', 'against', 'discourage'])
        ]
        
        for i, evidence1 in enumerate(evidence_list):
            for j, evidence2 in enumerate(evidence_list[i+1:], i+1):
                text1_lower = evidence1.text.lower()
                text2_lower = evidence2.text.lower()
                
                for positive_words, negative_words in opposing_pairs:
                    has_positive_1 = any(word in text1_lower for word in positive_words)
                    has_negative_2 = any(word in text2_lower for word in negative_words)
                    
                    has_negative_1 = any(word in text1_lower for word in negative_words)
                    has_positive_2 = any(word in text2_lower for word in positive_words)
                    
                    if (has_positive_1 and has_negative_2) or (has_negative_1 and has_positive_2):
                        contradictions.append(
                            f"Conflicting evidence found between sources {i+1} and {j+1}"
                        )
                        break
        
        return contradictions

# Singleton instance
_ranker = None

def get_ranker() -> EvidenceRanker:
    """Get singleton evidence ranker instance"""
    global _ranker
    if _ranker is None:
        _ranker = EvidenceRanker()
    return _ranker
