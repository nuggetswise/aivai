import httpx
import logging
from typing import Dict, Any, Optional, List
import time
from urllib.parse import urlparse
import trafilatura
from readability import Document
from bs4 import BeautifulSoup
import re

from app.config import settings

logger = logging.getLogger(__name__)

class WebScraper:
    """
    Web scraping utility for fetching and extracting content from URLs.
    Uses multiple extraction methods (trafilatura, readability, BS4) for best results.
    Handles rate limiting, timeouts, and content extraction.
    """
    
    def __init__(self):
        self.session = httpx.Client(
            timeout=15.0,  # 15 second timeout
            follow_redirects=True,
            headers={
                "User-Agent": settings.USER_AGENT,
                "Accept": "text/html,application/xhtml+xml,application/xml",
                "Accept-Language": "en-US,en;q=0.9",
            }
        )
        self.rate_limit_delay = 1.0  # Default 1 second between requests
        self.last_request_time = 0
        
    def fetch_url(self, url: str) -> Optional[str]:
        """
        Fetch HTML content from a URL with rate limiting and error handling.
        
        Args:
            url: The URL to fetch
            
        Returns:
            HTML content as string or None if failed
        """
        # Apply rate limiting
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        
        try:
            logger.info(f"Fetching content from {url}")
            self.last_request_time = time.time()
            
            # Check URL validity
            parsed_url = urlparse(url)
            if not parsed_url.scheme or not parsed_url.netloc:
                logger.warning(f"Invalid URL format: {url}")
                return None
                
            # Check domain whitelist/blacklist if configured
            if not self._is_allowed_domain(parsed_url.netloc):
                logger.warning(f"Domain not allowed: {parsed_url.netloc}")
                return None
                
            # Fetch content
            response = self.session.get(url)
            response.raise_for_status()
            
            # Check content type
            content_type = response.headers.get("content-type", "").lower()
            if not content_type.startswith("text/html"):
                logger.warning(f"Unsupported content type: {content_type}")
                return None
                
            return response.text
            
        except httpx.HTTPError as e:
            logger.error(f"HTTP error fetching {url}: {e}")
        except Exception as e:
            logger.error(f"Error fetching {url}: {e}")
            
        return None
    
    def extract_content(self, html: str, url: str) -> Dict[str, Any]:
        """
        Extract cleaned content from HTML using multiple extraction methods.
        
        Args:
            html: Raw HTML content
            url: Source URL for reference
            
        Returns:
            Dict with extracted title, text, metadata
        """
        if not html:
            return self._empty_result()
            
        try:
            # Try trafilatura first (usually best results)
            traf_result = self._extract_with_trafilatura(html, url)
            if self._is_good_extraction(traf_result):
                return traf_result
                
            # Fallback to readability
            read_result = self._extract_with_readability(html, url)
            if self._is_good_extraction(read_result):
                return read_result
                
            # Last resort: basic BeautifulSoup extraction
            bs_result = self._extract_with_bs4(html, url)
            return bs_result
            
        except Exception as e:
            logger.error(f"Error extracting content from {url}: {e}")
            return self._empty_result()
    
    def _extract_with_trafilatura(self, html: str, url: str) -> Dict[str, Any]:
        """Extract content using Trafilatura"""
        try:
            extracted = trafilatura.extract(
                html,
                output_format='json',
                with_metadata=True,
                date_extraction_params={'extensive_search': True},
                url=url
            )
            
            if extracted:
                import json
                parsed = json.loads(extracted)
                
                return {
                    'title': parsed.get('title', ''),
                    'text': parsed.get('text', ''),
                    'author': parsed.get('author', ''),
                    'date': parsed.get('date', ''),
                    'description': parsed.get('description', ''),
                    'categories': parsed.get('categories', []),
                    'tags': parsed.get('tags', []),
                    'extractor': 'trafilatura'
                }
        except Exception as e:
            logger.warning(f"Trafilatura extraction failed: {e}")
            
        return self._empty_result()
    
    def _extract_with_readability(self, html: str, url: str) -> Dict[str, Any]:
        """Extract content using Readability"""
        try:
            doc = Document(html)
            title = doc.title()
            content = doc.summary()
            
            # Clean up readability output with BeautifulSoup
            soup = BeautifulSoup(content, 'html.parser')
            text = soup.get_text(separator='\n\n', strip=True)
            
            # Try to extract date and author from meta tags
            soup = BeautifulSoup(html, 'html.parser')
            date = ''
            author = ''
            
            # Common meta tag patterns for dates and authors
            date_meta_names = ['date', 'article:published_time', 'publication_date', 'pubdate']
            author_meta_names = ['author', 'article:author', 'dcterms.creator']
            
            # Try to extract date
            for name in date_meta_names:
                meta = soup.find('meta', attrs={'name': name}) or soup.find('meta', attrs={'property': name})
                if meta and meta.get('content'):
                    date = meta.get('content')
                    break
                    
            # Try to extract author
            for name in author_meta_names:
                meta = soup.find('meta', attrs={'name': name}) or soup.find('meta', attrs={'property': name})
                if meta and meta.get('content'):
                    author = meta.get('content')
                    break
                    
            return {
                'title': title,
                'text': text,
                'author': author,
                'date': date,
                'description': '',
                'categories': [],
                'tags': [],
                'extractor': 'readability'
            }
        except Exception as e:
            logger.warning(f"Readability extraction failed: {e}")
            
        return self._empty_result()
    
    def _extract_with_bs4(self, html: str, url: str) -> Dict[str, Any]:
        """Basic extraction using BeautifulSoup"""
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Remove script, style elements
            for script in soup(["script", "style", "noscript", "iframe", "header", "footer", "nav"]):
                script.extract()
                
            # Get title
            title = ''
            if soup.title:
                title = soup.title.get_text(strip=True)
                
            # Get main content based on common container tags
            content_tags = ['article', 'main', 'div[role="main"]', '.content', '.post', '.article']
            main_content = None
            
            for tag in content_tags:
                if '[' in tag and ']' in tag:  # Handle attribute selectors
                    tag_name, attr = tag.split('[', 1)
                    attr_name, attr_value = attr.rstrip(']').split('=', 1)
                    attr_value = attr_value.strip('"\'')
                    elements = soup.find_all(tag_name, {attr_name: attr_value})
                elif '.' in tag:  # Handle class selectors
                    class_name = tag.lstrip('.')
                    elements = soup.find_all(class_=class_name)
                else:  # Handle tag selectors
                    elements = soup.find_all(tag)
                
                if elements:
                    # Use the longest content block
                    main_content = max(elements, key=lambda x: len(x.get_text()))
                    break
            
            # If we couldn't find main content, use body
            if not main_content:
                main_content = soup.body
                
            # Extract text from paragraphs
            paragraphs = []
            if main_content:
                for p in main_content.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                    text = p.get_text().strip()
                    if text and len(text) > 20:  # Skip short paragraphs
                        paragraphs.append(text)
                        
            text = '\n\n'.join(paragraphs)
            
            # Try to extract meta description
            description = ''
            meta_desc = soup.find('meta', attrs={'name': 'description'})
            if meta_desc:
                description = meta_desc.get('content', '')
                
            return {
                'title': title,
                'text': text,
                'author': '',
                'date': '',
                'description': description,
                'categories': [],
                'tags': [],
                'extractor': 'beautifulsoup'
            }
        except Exception as e:
            logger.warning(f"BeautifulSoup extraction failed: {e}")
            
        return self._empty_result()
    
    def _is_good_extraction(self, result: Dict[str, Any]) -> bool:
        """Check if extraction result is good quality"""
        # Minimum content length to consider extraction successful
        min_text_length = 100
        min_title_length = 3
        
        return (
            result.get('text', '') and
            len(result.get('text', '')) >= min_text_length and
            result.get('title', '') and
            len(result.get('title', '')) >= min_title_length
        )
    
    def _empty_result(self) -> Dict[str, Any]:
        """Return empty extraction result"""
        return {
            'title': '',
            'text': '',
            'author': '',
            'date': '',
            'description': '',
            'categories': [],
            'tags': [],
            'extractor': 'none'
        }
    
    def _is_allowed_domain(self, domain: str) -> bool:
        """Check if domain is allowed based on whitelist/blacklist"""
        # Check blacklist first
        if settings.blacklist_domains and any(
            blacklisted in domain for blacklisted in settings.blacklist_domains
        ):
            return False
            
        # Check whitelist if configured
        if settings.whitelist_domains:
            return any(whitelisted in domain for whitelisted in settings.whitelist_domains)
            
        # If no whitelist, all non-blacklisted domains are allowed
        return True
    
    def fetch_multiple_urls(self, urls: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Fetch and extract content from multiple URLs.
        
        Args:
            urls: List of URLs to fetch
            
        Returns:
            Dict mapping URLs to their extracted content
        """
        results = {}
        
        for url in urls:
            html = self.fetch_url(url)
            if html:
                content = self.extract_content(html, url)
                results[url] = content
                
        return results

# Singleton instance
_scraper = None

def get_scraper() -> WebScraper:
    """Get singleton scraper instance"""
    global _scraper
    if _scraper is None:
        _scraper = WebScraper()
    return _scraper
