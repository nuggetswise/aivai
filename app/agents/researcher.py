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
        topic = research_input.topic
        
        # Use varied query templates without hardcoded domain restrictions
        queries = [
            f"{topic}",  # Simple base query
            f"{topic} recent developments",
            f"{topic} current research",
            f"{topic} key measures OR policy brief OR framework",
            f"{topic} case studies OR examples OR implementation"
        ]
        
        # Add intent-specific queries
        if research_input.intent == TurnIntent.REBUTTAL and research_input.opponent_point:
            # For rebuttals, search for counter-evidence without domain restrictions
            queries.extend([
                f"{topic} criticism OR limitations OR challenges",
                f"counter arguments {topic} OR debate OR controversy"
            ])
        
        elif research_input.intent == TurnIntent.OPENING:
            # For openings, search for overview and definitions without domain restrictions
            queries.extend([
                f"{topic} overview OR definition OR introduction",
                f"what is {topic} OR introduction OR basics"
            ])
        
        elif research_input.intent == TurnIntent.POSITIONING:
            # For positioning, search for specific evidence and studies without domain restrictions
            queries.extend([
                f"{topic} evidence OR studies OR research findings",
                f"{topic} data OR statistics OR report"
            ])
        
        return queries[:5]  # Limit to 5 queries max for MVP
    
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
        """Gather evidence from local corpus based on user-uploaded documents"""
        evidence_list = []
        
        try:
            # Iterate over each document in the local corpus
            for document in research_input.local_corpus:
                # Extract relevant evidence using the indexer
                extracted_evidence = self.indexer.extract_evidence_from_document(document)
                evidence_list.extend(extracted_evidence)
                
        except Exception as e:
            logger.warning(f"Local corpus search failed: {e}")
        
        return evidence_list
    
    def _identify_omissions(self, research_input: ResearcherInput, evidence_list: List[Evidence]) -> List[str]:
        """Use LLM to identify key omissions in the evidence"""
        if not evidence_list:
            return ["No evidence found for this topic"]
        
        # Load researcher prompt template
        try:
            with open("app/prompts/researcher.txt", "r") as f:
                researcher_prompt = f.read()
        except FileNotFoundError:
            researcher_prompt = "You are a fact-only researcher."
        
        # Prepare evidence summary for LLM
        evidence_texts = [e.text[:200] for e in evidence_list[:10]]  # Limit for context
        evidence_summary = "\n".join([f"- {text}" for text in evidence_texts])
        
        # Enhanced prompt with source quality awareness
        messages = [
            {
                "role": "system",
                "content": f"""{researcher_prompt}

Given a topic and collected evidence, identify 2-3 key aspects or questions that are missing from the evidence.

Focus on:
- Important perspectives not covered by credible sources
- Key data points from government/academic sources that would be expected
- Critical context from established institutions that's absent
- Areas where only low-quality sources were found (blogs, press releases)

Prioritize gaps that could be filled by:
- Government agencies (.gov, .edu)
- Peer-reviewed research
- Established news organizations
- International organizations (WHO, UN, OECD)

Be specific and factual. Each omission should be 1-2 sentences."""
            },
            {
                "role": "user", 
                "content": f"""Topic: {research_input.topic}
Intent: {research_input.intent.value}

Evidence collected from sources:
{evidence_summary}

What key omissions or gaps do you identify in this evidence? Focus on areas where we lack credible, authoritative sources."""
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
