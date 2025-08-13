from typing import List, Dict, Any, Optional
import re
import logging
from datetime import datetime

from app.models import (
    StyleInput, StyleOutput, Persona
)
from app.config import settings
from app.deps import get_llm_router

logger = logging.getLogger(__name__)

class StyleAgent:
    """
    Adjusts tone and style to match persona without altering facts or citations.
    Focuses purely on stylistic elements: sentence rhythm, persona quirks, 
    tone consistency while preserving all factual content and citations intact.
    """
    
    def __init__(self):
        self.llm = get_llm_router().get_basic_llm()  # Use cost-effective tier
    
    def apply_style(self, style_input: StyleInput) -> StyleOutput:
        """
        Apply persona-specific styling to verified text without changing facts.
        
        Args:
            style_input: StyleInput with verified_text and persona
            
        Returns:
            StyleOutput with styled_text
        """
        logger.info(f"Applying style for {style_input.persona.name}")
        
        try:
            # Step 1: Preserve citations and factual content
            citations_preserved = self._preserve_citations(style_input.verified_text)
            
            # Step 2: Apply persona-specific styling
            styled_text = self._apply_persona_styling(
                style_input.verified_text, 
                style_input.persona,
                citations_preserved
            )
            
            # Step 3: Validate that facts and citations remain intact
            if not self._validate_preservation(style_input.verified_text, styled_text):
                logger.warning("Style application may have altered facts, reverting to original")
                styled_text = style_input.verified_text
            
            logger.info(f"Style applied: {len(styled_text)} characters")
            
            return StyleOutput(styled_text=styled_text)
            
        except Exception as e:
            logger.error(f"Style application failed: {e}")
            # Return original text on failure
            return StyleOutput(styled_text=style_input.verified_text)
    
    def _preserve_citations(self, text: str) -> Dict[str, str]:
        """Extract and preserve all citations from text"""
        citation_pattern = r'\[([SLR]\d+)\]'
        citations = re.findall(citation_pattern, text)
        
        citation_map = {}
        for cite in citations:
            citation_map[f"[{cite}]"] = f"[{cite}]"  # Preserve exact format
        
        return citation_map
    
    def _apply_persona_styling(self, text: str, persona: Persona, citations: Dict[str, str]) -> str:
        """Apply persona-specific styling using LLM"""
        # Build style adjustment prompt
        quirks_description = self._format_quirks_for_prompt(persona.speech_quirks)
        
        messages = [
            {
                "role": "system",
                "content": f"""You are a style editor that adjusts text tone and rhythm to match a specific persona.

CRITICAL RULES:
1. DO NOT change any facts, numbers, or claims
2. DO NOT add or remove citations [S#], [L#], [R#] 
3. DO NOT alter the meaning of any sentence
4. Only adjust: sentence rhythm, word choice, transitions, persona quirks

PERSONA: {persona.name}, {persona.role}
TONE: {persona.tone}
SPEECH QUIRKS: {quirks_description}

Your job is to make the text sound like {persona.name} would say it, while keeping all facts and citations exactly the same."""
            },
            {
                "role": "user",
                "content": f"""Adjust the style of this text to match {persona.name}'s tone and quirks:

{text}

Remember: Keep all facts and citations [S#] exactly the same. Only change the style and tone."""
            }
        ]
        
        try:
            styled_response = self.llm.generate(
                messages,
                temperature=0.3,  # Low temperature for consistency
                max_tokens=1000
            )
            
            return styled_response.strip()
            
        except Exception as e:
            logger.error(f"LLM styling failed: {e}")
            return text  # Return original on failure
    
    def _format_quirks_for_prompt(self, quirks: List[str]) -> str:
        """Format speech quirks for LLM prompt"""
        if not quirks:
            return "Direct, clear communication"
        
        formatted_quirks = []
        for quirk in quirks:
            # Add specific guidance for common quirk types
            if "analogy" in quirk.lower() or "analogi" in quirk.lower():
                formatted_quirks.append(f"{quirk} (use comparisons and metaphors)")
            elif "question" in quirk.lower():
                formatted_quirks.append(f"{quirk} (include rhetorical questions)")
            elif "pause" in quirk.lower():
                formatted_quirks.append(f"{quirk} (use ellipses and dashes)")
            else:
                formatted_quirks.append(quirk)
        
        return "; ".join(formatted_quirks)
    
    def _validate_preservation(self, original: str, styled: str) -> bool:
        """Validate that facts and citations are preserved"""
        # Check citation preservation
        original_citations = set(re.findall(r'\[([SLR]\d+)\]', original))
        styled_citations = set(re.findall(r'\[([SLR]\d+)\]', styled))
        
        if original_citations != styled_citations:
            logger.warning(f"Citations changed: {original_citations} -> {styled_citations}")
            return False
        
        # Check for significant content changes (basic keyword preservation)
        original_keywords = self._extract_keywords(original)
        styled_keywords = self._extract_keywords(styled)
        
        # Calculate keyword preservation ratio
        if original_keywords:
            preserved_ratio = len(original_keywords & styled_keywords) / len(original_keywords)
            if preserved_ratio < 0.7:  # At least 70% keywords should be preserved
                logger.warning(f"Too many keywords changed: {preserved_ratio:.2f} preservation ratio")
                return False
        
        # Check for unwanted additions
        unwanted_additions = [
            "in my opinion", "i think", "personally", "it seems",
            "perhaps", "maybe", "possibly", "might be"
        ]
        
        original_lower = original.lower()
        styled_lower = styled.lower()
        
        for addition in unwanted_additions:
            if addition not in original_lower and addition in styled_lower:
                logger.warning(f"Unwanted opinion marker added: {addition}")
                return False
        
        return True
    
    def _extract_keywords(self, text: str) -> set:
        """Extract important keywords from text for validation"""
        # Remove citations for keyword extraction
        text_clean = re.sub(r'\[[^\]]+\]', '', text)
        
        # Split into words and filter
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text_clean.lower())
        
        # Remove common stop words
        stop_words = {
            'the', 'and', 'but', 'for', 'are', 'with', 'they', 'this', 'that',
            'have', 'from', 'not', 'more', 'can', 'had', 'her', 'was', 'one',
            'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'its',
            'may', 'new', 'now', 'old', 'see', 'two', 'who', 'boy', 'did',
            'does', 'she', 'use', 'way', 'will', 'you'
        }
        
        keywords = {word for word in words if word not in stop_words and len(word) > 3}
        return keywords
    
    def enhance_persona_consistency(self, text: str, persona: Persona) -> str:
        """Apply additional persona consistency enhancements"""
        # Apply tone-specific adjustments without LLM
        if persona.tone.lower() == 'formal':
            text = self._make_more_formal(text)
        elif persona.tone.lower() == 'casual':
            text = self._make_more_casual(text)
        elif persona.tone.lower() == 'passionate':
            text = self._make_more_passionate(text)
        
        return text
    
    def _make_more_formal(self, text: str) -> str:
        """Apply formal tone adjustments"""
        # Basic formal replacements (preserving meaning)
        replacements = {
            r'\bcan\'t\b': 'cannot',
            r'\bwon\'t\b': 'will not',
            r'\bdon\'t\b': 'do not',
            r'\bisn\'t\b': 'is not',
            r'\baren\'t\b': 'are not',
        }
        
        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text
    
    def _make_more_casual(self, text: str) -> str:
        """Apply casual tone adjustments"""
        # Keep as-is for casual tone (most text is already appropriately casual)
        return text
    
    def _make_more_passionate(self, text: str) -> str:
        """Apply passionate tone adjustments"""
        # Add emphasis without changing facts (very conservative)
        # This is mainly handled by the LLM styling
        return text

# Singleton instance
_style_agent = None

def get_style_agent() -> StyleAgent:
    """Get singleton style agent instance"""
    global _style_agent
    if _style_agent is None:
        _style_agent = StyleAgent()
    return _style_agent
