#!/usr/bin/env python
"""
Simple test script for just the Tavily API
"""

import sys

# Try to import Tavily client
try:
    from tavily import TavilyClient
except ImportError:
    print("❌ Tavily Python client not installed. Install with: pip install tavily-python")
    sys.exit(1)

def test_tavily():
    """Test the Tavily API with the provided key"""
    print("=== Testing Tavily API ===")
    
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

if __name__ == "__main__":
    success = test_tavily()
    sys.exit(0 if success else 1)