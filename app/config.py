import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # Three-tier LLM Configuration
    # Reasoning LLM (for complex reasoning tasks like persona generation)
    REASONING_MODEL: str = os.getenv("REASONING_MODEL", "gemini-2.0-flash-exp")
    REASONING_API_KEY: str = os.getenv("REASONING_API_KEY", "")
    REASONING_BASE_URL: str = os.getenv("REASONING_BASE_URL", "https://generativelanguage.googleapis.com/v1beta")
    
    # Basic LLM (for simpler tasks like verification, style adjustment)
    BASIC_MODEL: str = os.getenv("BASIC_MODEL", "gemini-2.0-flash-exp")
    BASIC_API_KEY: str = os.getenv("BASIC_API_KEY", "")
    BASIC_BASE_URL: str = os.getenv("BASIC_BASE_URL", "https://generativelanguage.googleapis.com/v1beta")
    
    # Vision-Language LLM (for tasks involving images)
    VL_MODEL: str = os.getenv("VL_MODEL", "gemini-2.0-flash-exp")
    VL_API_KEY: str = os.getenv("VL_API_KEY", "")
    VL_BASE_URL: str = os.getenv("VL_BASE_URL", "https://generativelanguage.googleapis.com/v1beta")
    
    # Gemini API Key (fallback for all Gemini models)
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    
    # Legacy/Fallback LLM Configuration
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    
    # Embeddings Configuration
    EMBEDDINGS_BACKEND: str = os.getenv("EMBEDDINGS_BACKEND", "hf")  # "hf" or "google"
    EMBEDDINGS_MODEL: str = os.getenv("EMBEDDINGS_MODEL", "intfloat/e5-small-v2")  # HF model or "models/text-embedding-004" for Google
    
    # Search and Retrieval APIs
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")
    JINA_API_KEY: str = os.getenv("JINA_API_KEY", "")
    SERP_API_KEY: str = os.getenv("SERP_API_KEY", "")
    
    # TTS Configuration
    DIA_TTS_MODEL: str = os.getenv("DIA_TTS_MODEL", "nari-labs/dia")
    TTS_VENDOR: str = os.getenv("TTS_VENDOR", "dia")
    TTS_ENABLED: bool = os.getenv("TTS_ENABLED", "true").lower() == "true"
    
    # Data and Storage Paths
    DATA_DIR: str = os.getenv("DATA_DIR", "data")
    CORPUS_DIR: str = os.getenv("CORPUS_DIR", "corpus")
    WHITELIST_PATH: str = os.getenv("WHITELIST_PATH", "whitelist.yaml")
    
    # Content Filtering and Quality
    FRESHNESS_DAYS: int = int(os.getenv("FRESHNESS_DAYS", "120"))
    MAX_SEARCH_RESULTS: int = int(os.getenv("MAX_SEARCH_RESULTS", "12"))
    TRUST_SCORE_THRESHOLD: int = int(os.getenv("TRUST_SCORE_THRESHOLD", "7"))
    
    # Audio Processing
    AUDIO_QUALITY: str = os.getenv("AUDIO_QUALITY", "high")
    SAMPLE_RATE: int = int(os.getenv("SAMPLE_RATE", "22050"))
    CHANNELS: int = int(os.getenv("CHANNELS", "1"))
    
    # Episode Configuration
    MAX_TURNS_PER_PHASE: int = int(os.getenv("MAX_TURNS_PER_PHASE", "4"))
    DEFAULT_EPISODE_LENGTH: int = int(os.getenv("DEFAULT_EPISODE_LENGTH", "20"))
    CITATION_REQUIRED: bool = os.getenv("CITATION_REQUIRED", "true").lower() == "true"
    
    # Browser Configuration
    CHROME_INSTANCE_PATH: str = os.getenv("CHROME_INSTANCE_PATH", "")
    USER_AGENT: str = os.getenv("USER_AGENT", "AIvAI-Bot/1.0")
    
    # Development and Debug
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    ENABLE_CACHING: bool = os.getenv("ENABLE_CACHING", "true").lower() == "true"
    CACHE_TTL: int = int(os.getenv("CACHE_TTL", "3600"))
    
    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///data/metadata.db")
    
    # Security
    API_KEY_HEADER: str = os.getenv("API_KEY_HEADER", "X-API-Key")
    CORS_ORIGINS: list = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:8000").split(",")
    RATE_LIMIT_PER_MINUTE: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
    
    # Episode ID generation
    EPISODE_ID_PREFIX: str = os.getenv("EPISODE_ID_PREFIX", "ep")
    EPISODE_ID_LENGTH: int = int(os.getenv("EPISODE_ID_LENGTH", "8"))

settings = Settings()
