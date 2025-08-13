from app.config import settings
from typing import Optional, Any, Dict
import httpx
import time
import random
import logging
import google.generativeai as genai
from google.generativeai.types import GenerationConfig

logger = logging.getLogger(__name__)

class LLMClient:
    """Wrapper for LLM clients with retry logic"""
    
    def __init__(self, model: str, api_key: str, tier: str):
        self.model = model
        self.tier = tier
        
        # Configure Gemini
        if "gemini" in model.lower():
            genai.configure(api_key=api_key)
            self.client_type = "gemini"
        else:
            # Fallback to default implementation
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
            self.client_type = "openai"
            
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
                if self.client_type == "gemini":
                    return self._generate_with_gemini(messages, temperature, max_tokens, **kwargs)
                else:
                    # OpenAI fallback
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
    
    def _generate_with_gemini(self, messages: list, temperature: float = 0.7, max_tokens: Optional[int] = None, **kwargs) -> str:
        """Generate text using Gemini API"""
        # Convert OpenAI-style messages to Gemini format
        gemini_messages = []
        
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            # Map OpenAI roles to Gemini roles
            if role == "system":
                # Add system message as a user message followed by model response acknowledging it
                gemini_messages.append({"role": "user", "parts": [content]})
                gemini_messages.append({"role": "model", "parts": ["I understand these instructions and will follow them."]})
            else:
                # Map user and assistant to user and model
                gemini_role = "user" if role == "user" else "model"
                gemini_messages.append({"role": gemini_role, "parts": [content]})
        
        # Configure generation parameters
        generation_config = GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens if max_tokens else 1024,
            top_p=kwargs.get("top_p", 0.95),
        )
        
        # Get model and generate
        model = genai.GenerativeModel(model_name=self.model, generation_config=generation_config)
        response = model.generate_content(gemini_messages)
        
        return response.text
    
    async def generate_async(self, messages: list, temperature: float = 0.7, max_tokens: Optional[int] = None, **kwargs) -> str:
        """Async version of generate"""
        # Implementation would use async client
        # For now, falling back to sync
        return self.generate(messages, temperature, max_tokens, **kwargs)
    
    def stream(self, messages: list, temperature: float = 0.7, **kwargs):
        """Stream generation for real-time responses"""
        try:
            if self.client_type == "gemini":
                # Configure Gemini streaming
                generation_config = GenerationConfig(
                    temperature=temperature,
                    top_p=kwargs.get("top_p", 0.95),
                )
                
                # Convert OpenAI-style messages to Gemini format
                gemini_messages = []
                for msg in messages:
                    role = msg["role"]
                    content = msg["content"]
                    
                    # Map OpenAI roles to Gemini roles
                    if role == "system":
                        gemini_messages.append({"role": "user", "parts": [content]})
                        gemini_messages.append({"role": "model", "parts": ["I understand these instructions and will follow them."]})
                    else:
                        gemini_role = "user" if role == "user" else "model"
                        gemini_messages.append({"role": gemini_role, "parts": [content]})
                
                # Get model and generate stream
                model = genai.GenerativeModel(model_name=self.model, generation_config=generation_config)
                response = model.generate_content(gemini_messages, stream=True)
                
                for chunk in response:
                    if chunk.text:
                        yield chunk.text
            else:
                # OpenAI streaming
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
            api_key = settings.REASONING_API_KEY or settings.GEMINI_API_KEY
            if not api_key:
                raise ValueError("No API key configured for reasoning LLM")
            
            self._reasoning_client = LLMClient(
                model=settings.REASONING_MODEL,
                api_key=api_key,
                tier="reasoning"
            )
        return self._reasoning_client
    
    def get_basic_llm(self) -> LLMClient:
        """Get basic-tier LLM for simpler tasks (verification, style, research)"""
        if self._basic_client is None:
            api_key = settings.BASIC_API_KEY or settings.GEMINI_API_KEY
            if not api_key:
                raise ValueError("No API key configured for basic LLM")
            
            self._basic_client = LLMClient(
                model=settings.BASIC_MODEL,
                api_key=api_key,
                tier="basic"
            )
        return self._basic_client
    
    def get_vl_llm(self) -> LLMClient:
        """Get vision-language LLM for tasks involving images"""
        if self._vl_client is None:
            api_key = settings.VL_API_KEY or settings.GEMINI_API_KEY
            if not api_key:
                raise ValueError("No API key configured for VL LLM")
            
            self._vl_client = LLMClient(
                model=settings.VL_MODEL,
                api_key=api_key,
                tier="vision-language"
            )
        return self._vl_client

class SearchClient:
    """Unified interface for search providers (Tavily, SERP, etc.)"""
    
    def __init__(self):
        self.tavily_available = bool(settings.TAVILY_API_KEY)
        self.serp_available = bool(settings.SERP_API_KEY)
    
    def search(self, query: str, max_results: int = None, include_raw_content: bool = False, include_answer: bool = False, include_images: bool = False) -> list:
        """Search with fallback providers"""
        max_results = max_results or settings.MAX_SEARCH_RESULTS
        
        if self.tavily_available:
            return self._search_tavily(query, max_results, include_raw_content, include_answer, include_images)
        elif self.serp_available:
            return self._search_serp(query, max_results, **kwargs)
        else:
            logger.warning("No search provider configured, returning empty results")
            return []
    
    def _search_tavily(self, query: str, max_results: int, include_raw_content: bool, include_answer: bool, include_images: bool) -> list:
        """Search using Tavily API"""
        try:
            # Using TavilyClient instead of the deprecated direct tavily.search
            from tavily import TavilyClient
            
            client = TavilyClient(api_key=settings.TAVILY_API_KEY)
            
            search_params = {
                "query": query,
                "max_results": max_results,
                "search_depth": "advanced",
                "include_raw_content": include_raw_content,  # Key addition for better content
                "include_answer": include_answer,
                "include_images": include_images,
                # Remove hardcoded domain restrictions - let the system be more permissive
                # include_domains can be added back selectively if needed
            }
            
            response = client.search(**search_params)
            results = response.get("results", [])
            
            # Process results to ensure consistent format
            processed_results = []
            for result in results:
                processed_result = {
                    'title': result.get('title', ''),
                    'url': result.get('url', ''),
                    'content': result.get('content', ''),
                    'raw_content': result.get('raw_content', ''),  # Use Tavily's raw content
                    'score': result.get('score', 0.8),
                    'published_date': result.get('published_date'),
                    'snippet': result.get('content', '')[:300] + "..." if result.get('content') else ""
                }
                processed_results.append(processed_result)
            
            logger.info(f"Tavily search returned {len(processed_results)} results for: {query}")
            return processed_results
            
        except Exception as e:
            logger.error(f"Tavily search failed: {e}")
            return []
    
    def _search_serp(self, query: str, max_results: int, **kwargs) -> list:
        """Search using SERP API as fallback"""
        # Placeholder - implement SERP integration
        logger.info(f"Searching with SERP: {query}")
        return []

class EmbeddingsClient:
    """Client for text embeddings"""
    
    def __init__(self):
        backend = settings.EMBEDDINGS_BACKEND.lower()
        model_id = settings.EMBEDDINGS_MODEL

        if backend != "hf":
            raise RuntimeError("For MVP, set EMBEDDINGS_BACKEND=hf")

        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_id)
            logger.info(f"Loaded HuggingFace embedding model: {model_id}")
        except ImportError:
            raise RuntimeError("sentence-transformers not available. pip install sentence-transformers")
        except Exception as e:
            logger.error(f"Failed to load HuggingFace model {model_id}: {e}")
            raise

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for texts"""
        if not texts:
            return []
        
        try:
            embeddings = self.model.encode(texts, normalize_embeddings=True)
            return embeddings.tolist()
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
