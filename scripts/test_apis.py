#!/usr/bin/env python
"""
Test script for directly testing Gemini and Tavily APIs
"""

import os
import sys
import requests
from pathlib import Path
import json

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

# Try to load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("Loaded .env file")
except ImportError:
    print("dotenv package not found, skipping .env loading")

def test_gemini_api():
    """Test the Gemini API directly"""
    print("\n=== Testing Gemini API ===")
    
    # Try to get API key from environment variables
    api_key = os.environ.get("GEMINI_API_KEY", "")
    
    if not api_key:
        print("❌ No Gemini API key found in environment variables")
        return False
    
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    headers = {
        'Content-Type': 'application/json',
        'x-goog-api-key': api_key
    }
    data = {
        "contents": [
            {
                "parts": [
                    {
                        "text": "Explain how AI works in a few words"
                    }
                ]
            }
        ]
    }
    
    try:
        print("Sending request to Gemini API...")
        response = requests.post(url, headers=headers, json=data)
        print(f"Response status code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            content = result.get('candidates', [{}])[0].get('content', {})
            text = content.get('parts', [{}])[0].get('text', '')
            print(f"Response text: {text[:100]}..." if len(text) > 100 else f"Response text: {text}")
            print("✅ Gemini API test successful")
            return True
        else:
            print(f"❌ Gemini API error: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Exception when calling Gemini API: {e}")
        return False

def test_tavily_api():
    """Test the Tavily API directly"""
    print("\n=== Testing Tavily API ===")
    
    # Try to import Tavily client
    try:
        from tavily import TavilyClient
    except ImportError:
        print("❌ Tavily Python client not installed. Install with: pip install tavily-python")
        return False
    
    # Use the provided API key
    api_key = "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"
    
    try:
        print("Creating Tavily client...")
        client = TavilyClient(api_key)
        
        print("Sending search request to Tavily API...")
        response = client.search(query="latest ai research")
        
        print(f"Got {len(response.get('results', []))} search results")
        # Print first result if available
        if response.get('results'):
            first_result = response['results'][0]
            print(f"First result title: {first_result.get('title', 'No title')}")
            print(f"First result URL: {first_result.get('url', 'No URL')}")
        
        print("✅ Tavily API test successful")
        return True
    except Exception as e:
        print(f"❌ Exception when calling Tavily API: {e}")
        return False

def main():
    """Run API tests"""
    print("Starting API tests...")
    
    gemini_success = test_gemini_api()
    tavily_success = test_tavily_api()
    
    print("\n=== Test Summary ===")
    print(f"Gemini API: {'✅ PASSED' if gemini_success else '❌ FAILED'}")
    print(f"Tavily API: {'✅ PASSED' if tavily_success else '❌ FAILED'}")
    
    if gemini_success and tavily_success:
        print("\n✅ All API tests passed! The langmanus test failure is likely due to another issue.")
    else:
        print("\n❌ Some API tests failed. This may explain why the langmanus test is failing.")
    
    return gemini_success and tavily_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)