from typing import List, Dict, Any, Optional
import logging
from datetime import datetime

from app.models import (
    CommentatorInput, CommentatorOutput,
    Bundle, Evidence, Citation, Persona, EpisodePhase, TurnIntent
)
from app.config import settings
from app.deps import get_llm_router

logger = logging.getLogger(__name__)

class CommentatorAgent:
    """
    Generates persona-locked debate turns from evidence bundles.
    Strictly uses only provided evidence, respects persona constraints,
    and ensures every factual sentence has proper citations.
    """
    
    def __init__(self):
        self.llm = get_llm_router().get_reasoning_llm()  # Use high-quality tier for persona generation
    
    def generate_turn(self, commentator_input: CommentatorInput) -> CommentatorOutput:
        """
        Generate a debate turn based on evidence bundle and persona constraints.
        
        Args:
            commentator_input: CommentatorInput with topic, phase, intent, persona, evidence_bundle
            
        Returns:
            CommentatorOutput with text and citations
        """
        logger.info(f"Generating turn for {commentator_input.persona.name} on {commentator_input.topic}")
        
        try:
            # Step 1: Prepare context for LLM
            context = self._prepare_context(commentator_input)
            
            # Step 2: Build persona-specific prompt
            messages = self._build_prompt(commentator_input, context)
            
            # Step 3: Generate response using reasoning LLM
            response_text = self.llm.generate(
                messages, 
                temperature=0.25,  # Low temperature for consistency
                max_tokens=800     # Reasonable turn length
            )
            
            # Step 4: Extract and validate citations
            citations = self._extract_citations(response_text, commentator_input.evidence_bundle)
            
            # Step 5: Clean and validate response
            final_text = self._clean_response(response_text, commentator_input.persona)
            
            logger.info(f"Generated turn: {len(final_text)} characters, {len(citations)} citations")
            
            return CommentatorOutput(
                text=final_text,
                citations=citations
            )
            
        except Exception as e:
            logger.error(f"Turn generation failed for {commentator_input.persona.name}: {e}")
            # Return fallback response
            return CommentatorOutput(
                text=commentator_input.persona.default_unknown,
                citations=[]
            )
    
    def _prepare_context(self, commentator_input: CommentatorInput) -> Dict[str, Any]:
        """Prepare context information for the LLM prompt"""
        # Extract key evidence points
        evidence_points = []
        available_citations = []
        
        for i, evidence in enumerate(commentator_input.evidence_bundle.claims[:10]):  # Limit to top 10
            if evidence.citations:
                citation_id = evidence.citations[0].id
                evidence_points.append(f"[{citation_id}] {evidence.text}")
                available_citations.extend(evidence.citations)
        
        # Build opponent context
        opponent_context = ""
        if commentator_input.opponent_summary:
            opponent_context = f"Your opponent recently argued: {commentator_input.opponent_summary}"
        
        # Phase-specific guidance
        phase_guidance = {
            EpisodePhase.OPENING: "Provide a clear introduction to your position.",
            EpisodePhase.POSITIONS: "Present your main arguments with strong evidence.",
            EpisodePhase.CROSSFIRE: "Address counterarguments and strengthen your position.",
            EpisodePhase.CLOSING: "Summarize your key points and provide a compelling conclusion."
        }
        
        return {
            'evidence_points': evidence_points,
            'available_citations': available_citations,
            'opponent_context': opponent_context,
            'phase_guidance': phase_guidance.get(commentator_input.phase, "Present your perspective clearly."),
            'contradictions': commentator_input.evidence_bundle.contradictions
        }
    
    def _build_prompt(self, commentator_input: CommentatorInput, context: Dict[str, Any]) -> List[Dict[str, str]]:
        """Build the persona-specific prompt for the LLM"""
        
        # Persona characteristics
        persona = commentator_input.persona
        quirks_text = ", ".join(persona.speech_quirks) if persona.speech_quirks else "direct communication"
        
        # Evidence section
        evidence_text = "\n".join(context['evidence_points']) if context['evidence_points'] else "No specific evidence available."
        
        # System prompt
        system_prompt = f"""You are {persona.name}, a {persona.role}.

PERSONALITY & TONE:
- Speak in {persona.tone} tone
- Communication style: {quirks_text}
- If information is unsupported by evidence, say: "{persona.default_unknown}"

EVIDENCE RULES:
- Use ONLY the evidence provided below
- Every factual sentence MUST have a bracketed citation like [S1] or [L2]
- Do not add new facts or claims beyond the evidence
- Quote or tightly paraphrase the evidence

FORBIDDEN TOPICS: {', '.join(persona.forbidden_topics) if persona.forbidden_topics else 'None'}

DEBATE CONTEXT:
- Topic: {commentator_input.topic}
- Phase: {commentator_input.phase.value}
- Intent: {commentator_input.intent.value}
- {context['phase_guidance']}

{context['opponent_context']}

AVAILABLE EVIDENCE:
{evidence_text}

Generate a debate turn that reflects your persona while strictly using only the provided evidence."""
        
        # User prompt
        user_prompt = f"""Topic: {commentator_input.topic}

As {persona.name}, provide your {commentator_input.intent.value} response. Remember:
1. Stay in character with your {persona.tone} tone
2. Use only the provided evidence with proper citations
3. End with one concise takeaway
4. Keep response to 2-3 paragraphs

Your response:"""
        
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    
    def _extract_citations(self, response_text: str, evidence_bundle: Bundle) -> List[Citation]:
        """Extract citation references from response and map to actual citations"""
        import re
        
        # Find all citation patterns like [S1], [L2], [R3]
        citation_pattern = r'\[([SLR]\d+)\]'
        found_citations = re.findall(citation_pattern, response_text)
        
        # Map found citations to actual Citation objects
        citations = []
        citation_map = {}
        
        # Build citation map from evidence bundle
        for evidence in evidence_bundle.claims:
            for citation in evidence.citations:
                citation_map[citation.id] = citation
        
        # Add found citations
        for cite_id in found_citations:
            if cite_id in citation_map:
                citations.append(citation_map[cite_id])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_citations = []
        for citation in citations:
            if citation.id not in seen:
                seen.add(citation.id)
                unique_citations.append(citation)
        
        return unique_citations
    
    def _clean_response(self, response_text: str, persona: Persona) -> str:
        """Clean and validate the response text"""
        # Remove any potential system artifacts
        response_text = response_text.strip()
        
        # Remove common artifacts
        artifacts = [
            "As an AI,", "I cannot", "I don't have access", 
            "I'm not able to", "I should mention", "It's worth noting"
        ]
        
        for artifact in artifacts:
            if artifact in response_text:
                # If persona constraint is violated, return default
                logger.warning(f"Response contained AI artifact: {artifact}")
                return persona.default_unknown
        
        # Check for forbidden topics
        if persona.forbidden_topics:
            response_lower = response_text.lower()
            for forbidden in persona.forbidden_topics:
                if forbidden.lower() in response_lower:
                    logger.warning(f"Response contained forbidden topic: {forbidden}")
                    return persona.default_unknown
        
        # Ensure response isn't too short or too long
        if len(response_text) < 50:
            return persona.default_unknown
        elif len(response_text) > 2000:
            # Truncate at last complete sentence
            sentences = response_text.split('.')
            truncated = '.'.join(sentences[:-1]) + '.'
            return truncated
        
        return response_text
    
    def validate_persona_consistency(self, response: str, persona: Persona) -> float:
        """Validate how well the response matches the persona (0-1 score)"""
        score = 1.0
        response_lower = response.lower()
        
        # Check for tone consistency (basic keyword matching)
        tone_keywords = {
            'formal': ['therefore', 'furthermore', 'consequently', 'however'],
            'casual': ['basically', 'pretty much', 'sort of', 'kind of'],
            'technical': ['according to', 'data shows', 'research indicates'],
            'passionate': ['absolutely', 'crucial', 'essential', 'vital']
        }
        
        expected_keywords = tone_keywords.get(persona.tone.lower(), [])
        if expected_keywords:
            found_keywords = sum(1 for kw in expected_keywords if kw in response_lower)
            keyword_score = min(found_keywords / len(expected_keywords), 1.0)
            score *= (0.7 + 0.3 * keyword_score)  # Weight keyword matching
        
        # Check for speech quirks
        if persona.speech_quirks:
            quirk_matches = sum(1 for quirk in persona.speech_quirks if quirk.lower() in response_lower)
            quirk_score = min(quirk_matches / len(persona.speech_quirks), 1.0)
            score *= (0.8 + 0.2 * quirk_score)  # Weight quirk matching
        
        return score

# Singleton instance
_commentator = None

def get_commentator() -> CommentatorAgent:
    """Get singleton commentator instance"""
    global _commentator
    if _commentator is None:
        _commentator = CommentatorAgent()
    return _commentator
