from typing import List, Dict, Any, Optional
import logging
from datetime import datetime

from app.models import (
    ResearcherInput, ResearcherOutput, 
    Bundle, Evidence, Citation, TurnIntent
)
from app.config import settings
from app.deps import get_llm_router, get_search_client
from app.retrieval.indexer import get_indexer
from app.retrieval.rank import get_ranker
from app.retrieval.store import get_vector_store

logger = logging.getLogger(__name__)

class ResearcherAgent:
    """
    Fact-only researcher that gathers evidence from web and local corpus.
    Uses only provided excerpts, quotes tightly, and attaches citations.
    Flags contradictions and stale data.
    """
    
    def __init__(self):
        self.llm = get_llm_router().get_basic_llm()  # Use cost-effective tier
        self.search_client = get_search_client()
        self.indexer = get_indexer()
        self.ranker = get_ranker()
        self.vector_store = get_vector_store()
    
    def research(self, research_input: ResearcherInput) -> ResearcherOutput:
        """
        Main research method that coordinates web search, local retrieval, and evidence compilation.
        
        Args:
            research_input: ResearcherInput with topic, intent, opponent_point, local_corpus
            
        Returns:
            ResearcherOutput with claims, contradictions, omissions, and bundle_id
        """
        logger.info(f"Starting research for topic: {research_input.topic}")
        
        try:
            # Step 1: Build comprehensive search query
            search_queries = self._build_search_queries(research_input)
            
            # Step 2: Gather evidence from multiple sources
            web_evidence = self._gather_web_evidence(search_queries)
            local_evidence = self._gather_local_evidence(research_input)
            
            # Step 3: Combine and rank all evidence
            all_evidence = web_evidence + local_evidence
            ranked_evidence = self.ranker.rerank_evidence(
                all_evidence, 
                research_input.topic, 
                research_input.intent.value
            )
            
            # Step 4: Filter contradictions and limit results
            filtered_evidence, contradiction_notes = self.ranker.filter_contradictory_evidence(
                ranked_evidence[:20]  # Limit to top 20 before filtering
            )
            
            # Step 5: Identify omissions using LLM analysis
            omissions = self._identify_omissions(research_input, filtered_evidence)
            
            # Step 6: Create evidence bundle
            bundle = Bundle(
                topic=research_input.topic,
                query=search_queries[0] if search_queries else research_input.topic,
                claims=filtered_evidence[:12],  # Final limit
                contradictions=contradiction_notes,
                omissions=omissions,
                source_count=len(set(e.citations[0].url for e in filtered_evidence if e.citations))
            )
            
            # Step 7: Save bundle and return
            bundle_id = self.vector_store.save_bundle(bundle)
            
            logger.info(f"Research complete: {len(bundle.claims)} claims, {len(bundle.contradictions)} contradictions")
            
            return ResearcherOutput(
                claims=bundle.claims,
                contradictions=bundle.contradictions,
                omissions=bundle.omissions,
                bundle_id=bundle_id
            )
            
        except Exception as e:
            logger.error(f"Research failed for topic '{research_input.topic}': {e}")
            # Return empty result on failure
            return ResearcherOutput(
                claims=[],
                contradictions=[f"Research failed: {str(e)}"],
                omissions=[],
                bundle_id=""
            )
    
    def _build_search_queries(self, research_input: ResearcherInput) -> List[str]:
        """Build search queries based on topic, intent, and opponent context"""
        queries = [research_input.topic]  # Base query
        
        # Add intent-specific queries
        if research_input.intent == TurnIntent.REBUTTAL and research_input.opponent_point:
            # For rebuttals, search for counter-evidence
            queries.append(f"{research_input.topic} counter arguments")
            queries.append(f"criticism {research_input.topic}")
        
        elif research_input.intent == TurnIntent.OPENING:
            # For openings, search for overview and definitions
            queries.append(f"{research_input.topic} overview")
            queries.append(f"what is {research_input.topic}")
        
        elif research_input.intent == TurnIntent.POSITIONING:
            # For positioning, search for specific evidence and studies
            queries.append(f"{research_input.topic} evidence studies")
            queries.append(f"{research_input.topic} research findings")
        
        return queries[:3]  # Limit to 3 queries max
    
    def _gather_web_evidence(self, queries: List[str]) -> List[Evidence]:
        """Gather evidence from web search"""
        all_evidence = []
        
        for query in queries:
            try:
                # Use indexer to search and index new content
                bundle = self.indexer.index_from_search(query, max_results=6)
                all_evidence.extend(bundle.claims)
                
            except Exception as e:
                logger.warning(f"Web search failed for query '{query}': {e}")
                continue
        
        return all_evidence
    
    def _gather_local_evidence(self, research_input: ResearcherInput) -> List[Evidence]:
        """Gather evidence from local corpus"""
        if not research_input.local_corpus:
            return []
        
        evidence_list = []
        
        try:
            # Search local corpus using vector similarity
            bundle = self.ranker.rank_evidence_for_query(
                research_input.topic,
                max_results=8,
                filters={'min_trust_score': 6}  # Local corpus should be trusted
            )
            
            # Filter for local sources only
            for evidence in bundle.claims:
                if evidence.citations and any(
                    'file://' in cite.url or cite.type.value == 'local' 
                    for cite in evidence.citations
                ):
                    evidence_list.append(evidence)
                    
        except Exception as e:
            logger.warning(f"Local corpus search failed: {e}")
        
        return evidence_list
    
    def _identify_omissions(self, research_input: ResearcherInput, evidence_list: List[Evidence]) -> List[str]:
        """Use LLM to identify key omissions in the evidence"""
        if not evidence_list:
            return ["No evidence found for this topic"]
        
        # Prepare evidence summary for LLM
        evidence_texts = [e.text[:200] for e in evidence_list[:10]]  # Limit for context
        evidence_summary = "\n".join([f"- {text}" for text in evidence_texts])
        
        # LLM prompt to identify omissions
        messages = [
            {
                "role": "system",
                "content": """You are a fact-checking researcher. Given a topic and collected evidence, identify 2-3 key aspects or questions that are missing from the evidence.

Focus on:
- Important perspectives not covered
- Key data points that would be expected
- Critical context that's absent
- Methodological gaps

Be specific and factual. Each omission should be 1-2 sentences."""
            },
            {
                "role": "user", 
                "content": f"""Topic: {research_input.topic}
Intent: {research_input.intent.value}

Evidence collected:
{evidence_summary}

What key omissions or gaps do you identify in this evidence?"""
            }
        ]
        
        try:
            response = self.llm.generate(messages, temperature=0.3, max_tokens=300)
            
            # Parse omissions from response
            omissions = []
            for line in response.split('\n'):
                line = line.strip()
                if line and not line.startswith('#') and len(line) > 20:
                    # Clean up formatting
                    if line.startswith('-') or line.startswith('•'):
                        line = line[1:].strip()
                    if line.startswith(('1.', '2.', '3.')):
                        line = line[2:].strip()
                    omissions.append(line)
            
            return omissions[:3]  # Limit to 3 omissions
            
        except Exception as e:
            logger.warning(f"Failed to identify omissions: {e}")
            return []
    
    def quick_search(self, query: str, max_results: int = 5) -> List[Evidence]:
        """Quick search for immediate evidence needs"""
        try:
            bundle = self.ranker.rank_evidence_for_query(query, max_results=max_results)
            return bundle.claims
        except Exception as e:
            logger.error(f"Quick search failed for '{query}': {e}")
            return []

# Singleton instance
_researcher = None

def get_researcher() -> ResearcherAgent:
    """Get singleton researcher instance"""
    global _researcher
    if _researcher is None:
        _researcher = ResearcherAgent()
    return _researcher
