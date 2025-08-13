from app.config import settings
from typing import Optional, Any, Dict
import httpx
import time
import random
from openai import OpenAI
import logging

logger = logging.getLogger(__name__)

class LLMClient:
    """Wrapper for OpenAI-compatible LLM clients with retry logic"""
    
    def __init__(self, model: str, api_key: str, base_url: str, tier: str):
        self.model = model
        self.tier = tier
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self._session = None
    
    def get_session(self) -> httpx.AsyncClient:
        """Get or create HTTP session with retry configuration"""
        if self._session is None:
            self._session = httpx.AsyncClient(
                timeout=30.0,
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
            )
        return self._session
    
    def generate(self, messages: list, temperature: float = 0.7, max_tokens: Optional[int] = None, **kwargs) -> str:
        """Generate text with retry logic and backoff"""
        max_retries = 3
        base_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs
                )
                return response.choices[0].message.content
            
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"LLM {self.tier} failed after {max_retries} attempts: {e}")
                    raise
                
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                logger.warning(f"LLM {self.tier} attempt {attempt + 1} failed, retrying in {delay:.2f}s: {e}")
                time.sleep(delay)
    
    async def generate_async(self, messages: list, temperature: float = 0.7, max_tokens: Optional[int] = None, **kwargs) -> str:
        """Async version of generate"""
        # Implementation would use async OpenAI client
        # For now, falling back to sync
        return self.generate(messages, temperature, max_tokens, **kwargs)
    
    def stream(self, messages: list, temperature: float = 0.7, **kwargs):
        """Stream generation for real-time responses"""
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                stream=True,
                **kwargs
            )
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error(f"LLM {self.tier} streaming failed: {e}")
            raise

class LLMRouter:
    """Routes requests to appropriate LLM tier based on task complexity"""
    
    def __init__(self):
        self._reasoning_client = None
        self._basic_client = None
        self._vl_client = None
        self._search_client = None
        self._embeddings_client = None
    
    def get_reasoning_llm(self) -> LLMClient:
        """Get reasoning-tier LLM for complex tasks (persona generation, complex reasoning)"""
        if self._reasoning_client is None:
            api_key = settings.REASONING_API_KEY or settings.OPENAI_API_KEY
            if not api_key:
                raise ValueError("No API key configured for reasoning LLM")
            
            self._reasoning_client = LLMClient(
                model=settings.REASONING_MODEL,
                api_key=api_key,
                base_url=settings.REASONING_BASE_URL,
                tier="reasoning"
            )
        return self._reasoning_client
    
    def get_basic_llm(self) -> LLMClient:
        """Get basic-tier LLM for simpler tasks (verification, style, research)"""
        if self._basic_client is None:
            api_key = settings.BASIC_API_KEY or settings.OPENAI_API_KEY
            if not api_key:
                raise ValueError("No API key configured for basic LLM")
            
            self._basic_client = LLMClient(
                model=settings.BASIC_MODEL,
                api_key=api_key,
                base_url=settings.BASIC_BASE_URL,
                tier="basic"
            )
        return self._basic_client
    
    def get_vl_llm(self) -> LLMClient:
        """Get vision-language LLM for tasks involving images"""
        if self._vl_client is None:
            api_key = settings.VL_API_KEY or settings.OPENAI_API_KEY
            if not api_key:
                raise ValueError("No API key configured for VL LLM")
            
            self._vl_client = LLMClient(
                model=settings.VL_MODEL,
                api_key=api_key,
                base_url=settings.VL_BASE_URL,
                tier="vision-language"
            )
        return self._vl_client

class SearchClient:
    """Unified interface for search providers (Tavily, SERP, etc.)"""
    
    def __init__(self):
        self.tavily_available = bool(settings.TAVILY_API_KEY)
        self.serp_available = bool(settings.SERP_API_KEY)
    
    def search(self, query: str, max_results: int = None, **kwargs) -> list:
        """Search with fallback providers"""
        max_results = max_results or settings.MAX_SEARCH_RESULTS
        
        if self.tavily_available:
            return self._search_tavily(query, max_results, **kwargs)
        elif self.serp_available:
            return self._search_serp(query, max_results, **kwargs)
        else:
            logger.warning("No search provider configured, returning empty results")
            return []
    
    def _search_tavily(self, query: str, max_results: int, **kwargs) -> list:
        """Search using Tavily API"""
        # Placeholder - implement Tavily integration
        logger.info(f"Searching with Tavily: {query}")
        return []
    
    def _search_serp(self, query: str, max_results: int, **kwargs) -> list:
        """Search using SERP API as fallback"""
        # Placeholder - implement SERP integration
        logger.info(f"Searching with SERP: {query}")
        return []

class EmbeddingsClient:
    """Client for text embeddings"""
    
    def __init__(self):
        self.client = OpenAI(
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.BASIC_BASE_URL
        )
    
    def embed(self, texts: list[str], model: str = "text-embedding-3-small") -> list[list[float]]:
        """Generate embeddings for texts"""
        try:
            response = self.client.embeddings.create(
                model=model,
                input=texts
            )
            return [data.embedding for data in response.data]
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise

# Singleton instances
_llm_router = None
_search_client = None
_embeddings_client = None

def get_llm_router() -> LLMRouter:
    """Get singleton LLM router instance"""
    global _llm_router
    if _llm_router is None:
        _llm_router = LLMRouter()
    return _llm_router

def get_search_client() -> SearchClient:
    """Get singleton search client instance"""
    global _search_client
    if _search_client is None:
        _search_client = SearchClient()
    return _search_client

def get_embeddings_client() -> EmbeddingsClient:
    """Get singleton embeddings client instance"""
    global _embeddings_client
    if _embeddings_client is None:
        _embeddings_client = EmbeddingsClient()
    return _embeddings_client

def get_vector_store():
    """Get vector store instance"""
    from app.retrieval.store import get_vector_store as _get_vector_store
    return _get_vector_store()

# Legacy compatibility
def get_llm_client():
    """Legacy compatibility - returns basic LLM"""
    return get_llm_router().get_basic_llm()
