import re
import logging
from typing import List, Optional, Dict, Any
import unicodedata
import html

logger = logging.getLogger(__name__)

class TextCleaner:
    """
    Text cleaning utility that normalizes and sanitizes text from various sources.
    Handles whitespace normalization, HTML entity decoding, and citation formatting.
    """
    
    def __init__(self):
        self.max_paragraph_length = 2000  # Maximum paragraph length before splitting
        self.min_paragraph_length = 40    # Minimum paragraph length to keep
    
    def clean_text(self, text: str, aggressive: bool = False) -> str:
        """
        Clean text with a series of transformations.
        
        Args:
            text: The text to clean
            aggressive: Whether to apply more aggressive cleaning
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Basic cleaning
        cleaned = self._decode_html_entities(text)
        cleaned = self._normalize_whitespace(cleaned)
        cleaned = self._normalize_unicode(cleaned)
        cleaned = self._remove_boilerplate(cleaned)
        
        # Additional cleaning if aggressive mode is enabled
        if aggressive:
            cleaned = self._remove_urls(cleaned)
            cleaned = self._remove_special_chars(cleaned)
            cleaned = self._collapse_newlines(cleaned)
        
        return cleaned.strip()
    
    def clean_paragraphs(self, text: str) -> List[str]:
        """
        Split text into cleaned paragraphs.
        
        Args:
            text: Text to split and clean
            
        Returns:
            List of cleaned paragraphs
        """
        if not text:
            return []
        
        # Basic text cleaning first
        cleaned = self.clean_text(text)
        
        # Split by paragraph breaks (multiple newlines)
        paragraphs = re.split(r'\n\s*\n', cleaned)
        
        # Clean each paragraph and filter out short or empty ones
        result = []
        for para in paragraphs:
            para = self._normalize_whitespace(para)
            if para and len(para) >= self.min_paragraph_length:
                # Split overly long paragraphs
                if len(para) > self.max_paragraph_length:
                    split_paras = self._split_long_paragraph(para)
                    result.extend(split_paras)
                else:
                    result.append(para)
        
        return result
    
    def format_citation_text(self, text: str, source_label: str) -> str:
        """
        Format text with citation marker.
        
        Args:
            text: Text to format
            source_label: Source label (e.g., S1, L2)
            
        Returns:
            Text with citation marker appended
        """
        if not text or not source_label:
            return text
        
        # Clean text first
        cleaned = self.clean_text(text)
        
        # Make sure source_label is properly formatted
        if not source_label.startswith('['):
            source_label = f'[{source_label}]'
        
        # Add citation at the end if not already present
        if not cleaned.endswith(source_label):
            cleaned = f"{cleaned} {source_label}"
        
        return cleaned
    
    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace in text"""
        if not text:
            return ""
        
        # Replace multiple spaces with single space
        text = re.sub(r' +', ' ', text)
        
        # Normalize line breaks
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # Trim whitespace from start/end of each line
        lines = [line.strip() for line in text.split('\n')]
        
        # Rejoin with newlines
        return '\n'.join(lines)
    
    def _normalize_unicode(self, text: str) -> str:
        """Normalize Unicode characters"""
        if not text:
            return ""
        
        # Convert to NFKC form (compatible composition)
        text = unicodedata.normalize('NFKC', text)
        
        # Replace common Unicode equivalents
        replacements = {
            '–': '-',  # en dash
            '—': '-',  # em dash
            ''': "'",  # curly quote
            ''': "'",  # curly quote
            '"': '"',  # curly quote
            '"': '"',  # curly quote
            '…': '...',  # ellipsis
            '•': '*',  # bullet
        }
        
        for orig, repl in replacements.items():
            text = text.replace(orig, repl)
        
        return text
    
    def _decode_html_entities(self, text: str) -> str:
        """Decode HTML entities in text"""
        if not text:
            return ""
        
        # Decode HTML entities
        return html.unescape(text)
    
    def _remove_boilerplate(self, text: str) -> str:
        """Remove common boilerplate text patterns"""
        if not text:
            return ""
        
        # Common patterns to remove
        patterns = [
            r'Share this:.*',  # Social media sharing
            r'Copyright ©.*',  # Copyright notices
            r'All rights reserved\..*',  # Rights statements
            r'Terms of (Use|Service).*',  # Terms
            r'Privacy Policy.*',  # Privacy
            r'Cookie Policy.*',  # Cookies
            r'Subscribe to our newsletter.*',  # Newsletter
            r'Follow us on.*',  # Social media
            r'Related Articles:.*',  # Related content
            r'Click here to.*',  # Generic click prompts
        ]
        
        # Apply each pattern
        for pattern in patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)
        
        return text
    
    def _remove_urls(self, text: str) -> str:
        """Remove URLs from text"""
        if not text:
            return ""
        
        # URL pattern
        url_pattern = r'https?://\S+|www\.\S+'
        return re.sub(url_pattern, '', text)
    
    def _remove_special_chars(self, text: str) -> str:
        """Remove special characters from text"""
        if not text:
            return ""
        
        # Keep alphanumeric, punctuation, whitespace
        return re.sub(r'[^\w\s.,;:!?"\'-]', '', text)
    
    def _collapse_newlines(self, text: str) -> str:
        """Collapse multiple newlines to a maximum of two"""
        if not text:
            return ""
        
        return re.sub(r'\n{3,}', '\n\n', text)
    
    def _split_long_paragraph(self, paragraph: str) -> List[str]:
        """Split a long paragraph into smaller chunks at sentence boundaries"""
        if len(paragraph) <= self.max_paragraph_length:
            return [paragraph]
        
        # Try to split at sentence boundaries (periods, question marks, exclamation marks)
        sentences = re.split(r'(?<=[.!?])\s+', paragraph)
        
        result = []
        current_chunk = ""
        
        for sentence in sentences:
            # If adding this sentence would exceed max length, start a new chunk
            if len(current_chunk) + len(sentence) > self.max_paragraph_length and current_chunk:
                result.append(current_chunk.strip())
                current_chunk = sentence
            else:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
        
        # Add the final chunk if it's not empty
        if current_chunk:
            result.append(current_chunk.strip())
        
        return result
    
    def clean_and_split_content(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean and split web content into structured parts.
        
        Args:
            content: Dictionary with title, text, and metadata
            
        Returns:
            Cleaned content with paragraphs list added
        """
        result = content.copy()
        
        # Clean title
        if 'title' in result:
            result['title'] = self.clean_text(result['title'])
        
        # Clean main text and split into paragraphs
        if 'text' in result:
            result['text'] = self.clean_text(result['text'])
            result['paragraphs'] = self.clean_paragraphs(result['text'])
        
        # Clean description
        if 'description' in result:
            result['description'] = self.clean_text(result['description'])
        
        return result

# Singleton instance
_cleaner = None

def get_cleaner() -> TextCleaner:
    """Get singleton text cleaner instance"""
    global _cleaner
    if _cleaner is None:
        _cleaner = TextCleaner()
    return _cleaner
