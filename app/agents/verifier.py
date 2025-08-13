from typing import List, Dict, Any, Optional
import re
import logging
from datetime import datetime

from app.models import (
    VerifierInput, VerifierOutput,
    Bundle, Evidence, Citation, Persona
)
from app.config import settings
from app.deps import get_llm_router

logger = logging.getLogger(__name__)

class VerifierAgent:
    """
    Checks factual support for each sentence in debate turns.
    Validates citations, removes unsupported claims, and ensures
    all statements are grounded in provided evidence.
    """
    
    def __init__(self):
        self.llm = get_llm_router().get_basic_llm()  # Use cost-effective tier
    
    def verify_turn(self, verifier_input: VerifierInput) -> VerifierOutput:
        """
        Verify factual support for a debate turn and fix unsupported claims.
        
        Args:
            verifier_input: VerifierInput with draft, evidence_bundle, persona
            
        Returns:
            VerifierOutput with verified text, citations, and notes
        """
        logger.info(f"Verifying turn for {verifier_input.persona.name}")
        
        try:
            # Step 1: Parse citations and sentences from draft
            sentences = self._split_into_sentences(verifier_input.draft)
            citation_map = self._build_citation_map(verifier_input.evidence_bundle)
            
            # Step 2: Verify each sentence
            verified_sentences = []
            verification_notes = []
            used_citations = set()
            
            for sentence in sentences:
                verified_sentence, notes, citations = self._verify_sentence(
                    sentence, citation_map, verifier_input.persona
                )
                verified_sentences.append(verified_sentence)
                verification_notes.extend(notes)
                used_citations.update(cite.id for cite in citations)
            
            # Step 3: Reconstruct verified text
            verified_text = " ".join(verified_sentences).strip()
            
            # Step 4: Extract final citations
            final_citations = [citation_map[cite_id] for cite_id in used_citations if cite_id in citation_map]
            
            # Step 5: Final validation pass
            if not self._passes_final_validation(verified_text, verifier_input.persona):
                verified_text = verifier_input.persona.default_unknown
                final_citations = []
                verification_notes.append("Turn failed final validation, replaced with default response")
            
            logger.info(f"Verification complete: {len(verified_sentences)} sentences, {len(final_citations)} citations")
            
            return VerifierOutput(
                text=verified_text,
                citations=final_citations,
                verification_notes=verification_notes
            )
            
        except Exception as e:
            logger.error(f"Verification failed: {e}")
            return VerifierOutput(
                text=verifier_input.persona.default_unknown,
                citations=[],
                verification_notes=[f"Verification error: {str(e)}"]
            )
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences for individual verification"""
        # Simple sentence splitting on periods, exclamation marks, question marks
        sentences = re.split(r'[.!?]+', text)
        
        # Clean and filter sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 10:  # Skip very short fragments
                # Ensure sentence ends with punctuation
                if not sentence.endswith(('.', '!', '?')):
                    sentence += '.'
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
    def _build_citation_map(self, evidence_bundle: Bundle) -> Dict[str, Citation]:
        """Build mapping from citation IDs to Citation objects"""
        citation_map = {}
        
        for evidence in evidence_bundle.claims:
            for citation in evidence.citations:
                citation_map[citation.id] = citation
        
        return citation_map
    
    def _verify_sentence(self, sentence: str, citation_map: Dict[str, Citation], persona: Persona) -> tuple:
        """Verify a single sentence against available evidence"""
        verification_notes = []
        sentence_citations = []
        
        # Extract citation references from sentence
        citation_pattern = r'\[([SLR]\d+)\]'
        found_citations = re.findall(citation_pattern, sentence)
        
        # Check if sentence contains factual claims that need verification
        if self._contains_factual_claims(sentence):
            if not found_citations:
                # Factual sentence without citations
                verification_notes.append(f"Removed unsupported factual claim: '{sentence[:50]}...'")
                return persona.default_unknown, verification_notes, []
            
            # Verify each citation exists
            valid_citations = []
            for cite_id in found_citations:
                if cite_id in citation_map:
                    valid_citations.append(cite_id)
                    sentence_citations.append(citation_map[cite_id])
                else:
                    verification_notes.append(f"Invalid citation {cite_id} in sentence")
            
            if not valid_citations:
                # No valid citations for factual claim
                verification_notes.append(f"Removed sentence with invalid citations: '{sentence[:50]}...'")
                return persona.default_unknown, verification_notes, []
        
        # Sentence passes verification
        return sentence, verification_notes, sentence_citations
    
    def _contains_factual_claims(self, sentence: str) -> bool:
        """Determine if sentence contains factual claims that need citation"""
        # Opinion/belief markers that don't require citations
        opinion_markers = [
            'i believe', 'i think', 'in my opinion', 'personally', 'i feel',
            'it seems to me', 'arguably', 'perhaps', 'possibly', 'might',
            'could be', 'appears to', 'suggests that'
        ]
        
        sentence_lower = sentence.lower()
        
        # If sentence contains opinion markers, it may not need citations
        if any(marker in sentence_lower for marker in opinion_markers):
            return False
        
        # Factual claim indicators
        factual_indicators = [
            'research shows', 'studies indicate', 'data reveals', 'according to',
            'statistics show', 'evidence suggests', 'findings indicate',
            'reports state', 'analysis confirms', '%', 'percent', 'million',
            'billion', 'year', 'study', 'survey', 'report'
        ]
        
        # If sentence contains factual indicators, it needs citations
        if any(indicator in sentence_lower for indicator in factual_indicators):
            return True
        
        # Check for definitive statements that need support
        definitive_patterns = [
            r'\b(is|are|was|were)\s+\w+',  # "X is Y" statements
            r'\b(has|have|had)\s+\w+',     # "X has Y" statements
            r'\b(will|would|can|cannot)\s+\w+',  # Prediction/capability statements
        ]
        
        for pattern in definitive_patterns:
            if re.search(pattern, sentence_lower):
                return True
        
        return False
    
    def _passes_final_validation(self, text: str, persona: Persona) -> bool:
        """Final validation of the entire verified text"""
        if not text or len(text.strip()) < 20:
            return False
        
        # Check for AI artifacts that shouldn't appear
        ai_artifacts = [
            "as an ai", "i'm an ai", "i cannot", "i don't have access",
            "i'm not able to", "as a language model", "i should note"
        ]
        
        text_lower = text.lower()
        if any(artifact in text_lower for artifact in ai_artifacts):
            return False
        
        # Check forbidden topics
        if persona.forbidden_topics:
            for forbidden in persona.forbidden_topics:
                if forbidden.lower() in text_lower:
                    return False
        
        # Check for coherence (basic)
        sentences = self._split_into_sentences(text)
        if len(sentences) < 1:
            return False
        
        return True
    
    def check_citation_accuracy(self, text: str, evidence_bundle: Bundle) -> List[str]:
        """Check accuracy of citations against evidence bundle"""
        issues = []
        citation_pattern = r'\[([SLR]\d+)\]'
        found_citations = re.findall(citation_pattern, text)
        
        # Build evidence map for quick lookup
        evidence_map = {}
        for evidence in evidence_bundle.claims:
            for citation in evidence.citations:
                evidence_map[citation.id] = evidence.text
        
        # Check each citation
        for cite_id in found_citations:
            if cite_id not in evidence_map:
                issues.append(f"Citation {cite_id} not found in evidence bundle")
                continue
            
            # Get the sentence containing this citation
            cite_pattern = rf'\[{re.escape(cite_id)}\]'
            sentences_with_cite = [s for s in text.split('.') if re.search(cite_pattern, s)]
            
            for sentence in sentences_with_cite:
                sentence_clean = re.sub(r'\[[^\]]+\]', '', sentence).strip()
                evidence_text = evidence_map[cite_id]
                
                # Basic relevance check (could be more sophisticated)
                if len(sentence_clean) > 20 and not self._check_relevance(sentence_clean, evidence_text):
                    issues.append(f"Citation {cite_id} may not support the claim in: '{sentence_clean[:50]}...'")
        
        return issues
    
    def _check_relevance(self, claim: str, evidence: str) -> bool:
        """Basic relevance check between claim and evidence"""
        # Simple keyword overlap check
        claim_words = set(claim.lower().split())
        evidence_words = set(evidence.lower().split())
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
        claim_words -= stop_words
        evidence_words -= stop_words
        
        if not claim_words:
            return False
        
        # Calculate overlap ratio
        overlap = len(claim_words & evidence_words)
        overlap_ratio = overlap / len(claim_words)
        
        return overlap_ratio >= 0.3  # At least 30% word overlap

# Singleton instance
_verifier = None

def get_verifier() -> VerifierAgent:
    """Get singleton verifier instance"""
    global _verifier
    if _verifier is None:
        _verifier = VerifierAgent()
    return _verifier
