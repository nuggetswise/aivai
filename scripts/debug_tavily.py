#!/usr/bin/env python
"""
Direct Tavily search test to debug the researcher agent issues
"""

import sys
import os
from pathlib import Path
import json
import logging

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

# Set up verbose logging
logging.basicConfig(level=logging.DEBUG)

# Import Tavily client
from app.deps import get_search_client
from app.config import settings

def main():
    """Test Tavily search directly and print raw results"""
    print("=== DIRECT TAVILY SEARCH DEBUG ===")
    
    # Ensure we have an API key
    tavily_key = settings.TAVILY_API_KEY
    print(f"Tavily API key available: {bool(tavily_key)}")
    
    # Get search client
    search_client = get_search_client()
    print(f"Search client tavily_available: {search_client.tavily_available}")
    
    # Test query
    topic = "Climate change adaptation strategies"
    print(f"Searching for: {topic}")
    
    try:
        # Direct search
        results = search_client.search(topic, max_results=5)
        print(f"Search returned {len(results)} results")
        
        # Save raw results for inspection
        debug_path = "debug_tavily_results.json"
        with open(debug_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Raw results saved to {debug_path}")
        
        # Check content availability
        raw_content_count = sum(1 for r in results if "raw_content" in r)
        content_count = sum(1 for r in results if "content" in r)
        
        print(f"Results with raw_content: {raw_content_count}/{len(results)}")
        print(f"Results with content: {content_count}/{len(results)}")
        
        # Process manually like the indexer would
        print("\nManually processing results like the indexer:")
        
        # Track why documents might be excluded
        passed = []
        failed = []
        
        for i, result in enumerate(results):
            url = result.get("url", "")
            content = result.get("raw_content") or result.get("content", result.get("snippet", ""))
            
            if not content or len(content) < 50:
                failed.append((i, url, "content too short"))
                continue
                
            # Document passes filters
            passed.append((i, url))
        
        print(f"\n✅ Passed documents: {len(passed)}/{len(results)}")
        for i, url in passed:
            print(f"  {i+1}. {url}")
            
        print(f"\n❌ Failed documents: {len(failed)}/{len(results)}")
        for i, url, reason in failed:
            print(f"  {i+1}. {url} - {reason}")
            
        # Look at first passing document
        if passed:
            idx = passed[0][0]
            doc = results[idx]
            print("\nSample passing document:")
            print(f"URL: {doc.get('url')}")
            print(f"Title: {doc.get('title')}")
            content_preview = (doc.get('raw_content') or doc.get('content', ''))[:150].replace('\n', ' ')
            print(f"Content: {content_preview}...")
            
        return bool(passed)
        
    except Exception as e:
        import traceback
        print(f"ERROR: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    print(f"\nDebug {'succeeded' if success else 'failed'}")