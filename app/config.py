import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # API keys
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")
    DIA_TTS_MODEL: str = os.getenv("DIA_TTS_MODEL", "nari-labs/dia")
    # Data paths
    DATA_DIR: str = os.getenv("DATA_DIR", "data")
    CORPUS_DIR: str = os.getenv("CORPUS_DIR", "corpus")
    # Other settings
    FRESHNESS_DAYS: int = int(os.getenv("FRESHNESS_DAYS", "30"))
    WHITELIST_PATH: str = os.getenv("WHITELIST_PATH", "whitelist.yaml")
    # LLM model selection
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    # TTS vendor
    TTS_VENDOR: str = os.getenv("TTS_VENDOR", "dia")

settings = Settings()
