#!/usr/bin/env python
"""
Debug script to trace the exact flow and identify where claims are lost
"""

import sys
import logging
from pathlib import Path

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

# Set up detailed logging
logging.basicConfig(level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')

from app.deps import get_search_client
from app.retrieval.indexer import get_indexer

def debug_tavily_to_claims():
    """Trace the exact path from Tavily results to claims"""
    topic = "Climate change adaptation strategies"
    
    print("=== STEP 1: Direct Tavily Search ===")
    search_client = get_search_client()
    results = search_client.search(topic, max_results=3)
    print(f"Tavily returned {len(results)} results")
    
    for i, result in enumerate(results):
        print(f"\nResult {i+1}:")
        print(f"  URL: {result.get('url', 'N/A')}")
        print(f"  Title: {result.get('title', 'N/A')}")
        raw_content = result.get('raw_content', '')
        content = result.get('content', '')
        snippet = result.get('snippet', '')
        print(f"  Raw content length: {len(raw_content)}")
        print(f"  Content length: {len(content)}")
        print(f"  Snippet length: {len(snippet)}")
        
        # Show what _process_search_result would use
        used_content = raw_content or content or snippet
        print(f"  Used content length: {len(used_content)}")
        print(f"  First 100 chars: {used_content[:100]}...")
    
    print("\n=== STEP 2: Indexer Processing ===")
    indexer = get_indexer()
    bundle = indexer.index_from_search(topic, max_results=3)
    
    print(f"Bundle created with {len(bundle.claims)} claims")
    print(f"Bundle source_count: {bundle.source_count}")
    
    if bundle.claims:
        for i, claim in enumerate(bundle.claims):
            print(f"\nClaim {i+1}:")
            print(f"  Text: {claim.text[:100]}...")
            print(f"  Citations: {len(claim.citations)}")
            print(f"  Confidence: {claim.confidence}")
    else:
        print("❌ NO CLAIMS CREATED - This is the problem!")
        
    return len(bundle.claims) > 0

if __name__ == "__main__":
    success = debug_tavily_to_claims()
    print(f"\nDebug result: {'SUCCESS' if success else 'FAILED - NO CLAIMS'}")