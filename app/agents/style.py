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
        Apply persona-specific style to verified text.
        
        Args:
            style_input: StyleInput with verified_text and persona
            
        Returns:
            StyleOutput with styled text
        """
        logger.info(f"Styling text for {style_input.persona.name}")
        
        try:
            # Step 1: Load style prompt
            with open("app/prompts/style.txt", "r") as f:
                style_prompt = f.read()
        except FileNotFoundError:
            style_prompt = "You are a style specialist."
        
        # Step 2: Replace persona variables in prompt template
        prompt = style_prompt
        for field in ["name", "role", "tone", "speech_quirks"]:
            value = getattr(style_input.persona, field)
            if isinstance(value, list):
                value = ", ".join(value)
            replacement = f"{{{{persona.{field}}}}}"
            prompt = prompt.replace(replacement, str(value))
        
        # Step 3: Generate styled text
        messages = [
            {
                "role": "system",
                "content": prompt
            },
            {
                "role": "user",
                "content": f"""Here is the verified debate text that needs styling and emotional expressions:

```
{style_input.verified_text}
```

IMPORTANT GUIDELINES:
1. Preserve ALL factual content and citations exactly as written
2. Add natural emotional expressions like (thoughtful), (pause), (chuckle), etc.
3. Aim to include 3-4 emotional expressions distributed through the text
4. Refine the style to match {style_input.persona.name}'s voice
5. The emotions should match the content and the persona's character

Return the styled text that incorporates the persona's speech patterns and appropriate emotional expressions.
"""
            }
        ]
        
        styled_text = self.llm.generate(
            messages,
            temperature=0.7,  # Higher temperature for creative styling
            max_tokens=2000
        )
        
        # Remove any unwanted formatting or explanations
        styled_text = self._clean_output(styled_text)
        
        # Ensure we have emotional annotations
        styled_text = self._ensure_emotional_annotations(styled_text, style_input.persona)
        
        logger.info("Styling complete")
        
        return StyleOutput(
            styled_text=styled_text
        )
    
    def _clean_output(self, text: str) -> str:
        """Clean up the output text by removing unwanted markers and formatting"""
        # Remove markdown code blocks
        text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
        
        # Remove explanatory headers/footers
        text = re.sub(r'^(Here is|I\'ve styled|The styled text|Styled text:).*?\n', '', text)
        text = re.sub(r'\n(Note:|As requested|I\'ve preserved|I made sure).*?$', '', text)
        
        return text.strip()
    
    def _ensure_emotional_annotations(self, text: str, persona: Persona) -> str:
        """Ensure the text has emotional annotations, add if missing"""
        # Check if text already has emotions
        emotion_pattern = r'\([a-z\s]+\)'
        emotions = re.findall(emotion_pattern, text)
        
        # If emotions are already present, return as is
        if len(emotions) >= 2:
            return text
        
        # If no emotions, add some based on persona and content
        logger.warning("No emotional annotations found in styled text, adding defaults")
        
        # Split into paragraphs
        paragraphs = text.split("\n\n")
        if len(paragraphs) < 2:
            paragraphs = text.split(".")
        
        # Customize emotions based on persona
        if persona.name.lower() == "nova rivers" or "passionate" in persona.tone.lower():
            # Nova-specific emotions based on her passionate, technical style
            if not re.search(emotion_pattern, paragraphs[0]):
                paragraphs[0] = f"(confident) {paragraphs[0]}"
            
            # Add technical enthusiasm in middle sections
            if len(paragraphs) > 1 and not re.search(emotion_pattern, paragraphs[1]):
                sentences = paragraphs[1].split(".")
                if len(sentences) > 2:
                    mid_idx = len(sentences) // 2
                    sentences[mid_idx] = f"{sentences[mid_idx]} (enthusiastic)"
                    paragraphs[1] = ".".join(sentences)
            
            # Add emphasis at the end
            if len(paragraphs) > 2 and not re.search(emotion_pattern, paragraphs[-1]):
                if "furthermore" in paragraphs[-1].lower():
                    paragraphs[-1] = paragraphs[-1].replace("Furthermore", "(passionate) Furthermore")
                else:
                    paragraphs[-1] = f"(serious tone) {paragraphs[-1]}"
            
            return "\n\n".join(paragraphs)
            
        else:
            # Generic emotion handling for other personas (e.g., Alex)
            # For first paragraph, add at beginning or end
            if not re.search(emotion_pattern, paragraphs[0]):
                if persona.tone.lower() in ["thoughtful", "analytical", "academic"]:
                    paragraphs[0] = f"(thoughtful) {paragraphs[0]}"
                else:
                    end_sent = paragraphs[0].split(".")[-2] + "."
                    paragraphs[0] = paragraphs[0].replace(end_sent, f"{end_sent} (pause)")
            
            # For middle paragraph, add in middle
            if len(paragraphs) > 1 and not re.search(emotion_pattern, paragraphs[1]):
                sentences = paragraphs[1].split(".")
                if len(sentences) > 2:
                    mid_idx = len(sentences) // 2
                    emotion = "serious tone" if "climate" in paragraphs[1].lower() else "thoughtful"
                    sentences[mid_idx] = f"{sentences[mid_idx]} ({emotion})"
                    paragraphs[1] = ".".join(sentences)
            
            # For last paragraph, add emphasis
            if len(paragraphs) > 2 and not re.search(emotion_pattern, paragraphs[-1]):
                if "?" in paragraphs[-1]:
                    paragraphs[-1] = f"(curious) {paragraphs[-1]}"
                else:
                    paragraphs[-1] = f"(confident) {paragraphs[-1]}"
            
            return "\n\n".join(paragraphs)
        
        # If text structure doesn't allow paragraph-based placement
        return f"({persona.tone.split(',')[0]}) {text}"
    
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
