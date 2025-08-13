#!/usr/bin/env python
"""
Simple diagnostic script to test just the researcher component of Langmanus
"""


import sys
import argparse
from pathlib import Path
import logging

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

# Set up logging - make more verbose
logging.basicConfig(level=logging.DEBUG, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Import components
from app.agents.researcher import get_researcher
from app.deps import get_search_client
from app.models import ResearcherInput, TurnIntent

def test_researcher(topic="Solar energy efficiency improvements", debug=True):
    """Test the researcher agent in isolation"""
    print("Starting researcher test...")
    
    # Get researcher instance
    researcher = get_researcher()
    print("Researcher instance created")
    
    # Simple research input
    research_input = ResearcherInput(
        topic=topic,
        intent=TurnIntent.OPENING,
        opponent_point="",
        local_corpus=[]
    )
    print(f"Created research input for topic: {topic}")
    
    # Debug: Check Tavily search directly
    if debug:
        print("\n=== DEBUGGING TAVILY SEARCH ===")
        try:
            search_client = get_search_client()
            print(f"Search client available: {search_client.tavily_available}")
            
            if search_client.tavily_available:
                print("Performing direct search...")
                results = search_client.search(topic, max_results=5)
                print(f"Direct search returned {len(results)} results")
                
                # Show first result preview
                if results:
                    print("\nFirst result preview:")
                    result = results[0]
                    print(f"URL: {result.get('url', 'N/A')}")
                    print(f"Title: {result.get('title', 'N/A')}")
                    print(f"Has raw_content: {'raw_content' in result}")
                    content_preview = result.get('raw_content', result.get('content', ''))[:100] + '...'
                    print(f"Content preview: {content_preview}")
        except Exception as e:
            print(f"Debug search error: {e}")
    
    try:
        # Run research
        print("\n=== RUNNING RESEARCHER AGENT ===")
        print("Calling researcher.research()...")
        research_output = researcher.research(research_input)
        
        # Print results
        print(f"\nResearch complete! Found {len(research_output.claims)} claims")
        
        if research_output.claims:
            print("\nSample claims:")
            for i, claim in enumerate(research_output.claims[:3]):
                print(f"\n--- Claim {i+1} ---")
                print(claim.text)
                print("Citations:", [c.id for c in claim.citations])
        else:
            print("\n⚠️ WARNING: No claims were found!")
        
        print("\nContradictions:", research_output.contradictions)
        print("\nOmissions:", research_output.omissions)
        print("\nBundle ID:", research_output.bundle_id)
        
        return True, research_output
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False, None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the researcher agent")
    parser.add_argument("--topic", default="Solar energy efficiency improvements", help="Research topic")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    success, output = test_researcher(topic=args.topic, debug=args.debug)
    print(f"\nTest {'succeeded' if success else 'failed'}")
    
    # Additional debug for no claims case
    if success and not output.claims:
        print("\n=== DEBUG FOR EMPTY CLAIMS ===")
        print("Check that:")
        print("1. Tavily search is returning results")
        print("2. Raw content is available in the search results")
        print("3. No domain filtering is blocking all results")
        print("4. Content chunking and processing is working correctly")
