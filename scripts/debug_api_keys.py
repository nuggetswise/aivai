#!/usr/bin/env python
"""
Debug script for API key loading in the langmanus environment
This script checks how API keys are loaded and available to the application
"""

import os
import sys
from pathlib import Path

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

def print_key_status(key_name, env_value, settings_value=None):
    """Print status of a key with masking for security"""
    if env_value:
        masked = env_value[:4] + "*" * (len(env_value) - 8) + env_value[-4:] if len(env_value) > 8 else "***"
        print(f"✅ {key_name} from env: {masked}")
    else:
        print(f"❌ {key_name} not found in environment variables")
    
    if settings_value is not None:
        if settings_value:
            masked = settings_value[:4] + "*" * (len(settings_value) - 8) + settings_value[-4:] if len(settings_value) > 8 else "***"
            print(f"✅ {key_name} from settings: {masked}")
        else:
            print(f"❌ {key_name} not found in settings")

def main():
    print("Checking API keys availability in environment...")
    
    # Check keys directly from environment
    print("\nDirect environment check:")
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    tavily_key = os.environ.get("TAVILY_API_KEY", "")
    reasoning_key = os.environ.get("REASONING_API_KEY", "")
    
    print_key_status("GEMINI_API_KEY", gemini_key)
    print_key_status("TAVILY_API_KEY", tavily_key)
    print_key_status("REASONING_API_KEY", reasoning_key)
    
    # Try explicitly loading .env file
    print("\nTrying explicit .env loading:")
    try:
        from dotenv import load_dotenv
        load_dotenv(override=True)
        print("✅ .env file explicitly loaded")
        
        # Check keys after explicit loading
        gemini_key = os.environ.get("GEMINI_API_KEY", "")
        tavily_key = os.environ.get("TAVILY_API_KEY", "")
        print_key_status("GEMINI_API_KEY (after explicit loading)", gemini_key)
        print_key_status("TAVILY_API_KEY (after explicit loading)", tavily_key)
    except Exception as e:
        print(f"❌ Error loading .env file: {e}")
    
    # Check how the application config loads the keys
    print("\nChecking application config loading:")
    try:
        # Import configuration after explicit .env loading
        from app.config import settings
        
        print_key_status("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY", ""), settings.GEMINI_API_KEY)
        print_key_status("TAVILY_API_KEY", os.environ.get("TAVILY_API_KEY", ""), settings.TAVILY_API_KEY)
        print_key_status("REASONING_API_KEY", os.environ.get("REASONING_API_KEY", ""), settings.REASONING_API_KEY)
        
        # Check how Gemini key is actually accessed in the code
        print("\nChecking effective API key for LLM models:")
        print(f"REASONING_MODEL is set to: {settings.REASONING_MODEL}")
        if "gemini" in settings.REASONING_MODEL.lower():
            effective_key = settings.REASONING_API_KEY or settings.GEMINI_API_KEY
            print(f"Effective API key for reasoning model: {'✅ Present' if effective_key else '❌ Missing'}")
        else:
            print("Reasoning model is not Gemini, checking OpenAI")
            print(f"OpenAI API key: {'✅ Present' if settings.OPENAI_API_KEY else '❌ Missing'}")
    except Exception as e:
        print(f"❌ Error loading application config: {e}")
    
    print("\nDiagnostic complete. Review the output to identify API key loading issues.")

if __name__ == "__main__":
    main()