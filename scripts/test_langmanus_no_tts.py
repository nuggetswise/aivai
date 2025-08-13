#!/usr/bin/env python
"""
Test script for Langmanus without Dia TTS
"""


import sys
import os
from pathlib import Path
import logging
import argparse

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

def main():
    """Run a simplified Langmanus test without TTS"""
    parser = argparse.ArgumentParser(description="Test Langmanus without TTS")
    parser.add_argument("--topic", type=str, default="Artificial Intelligence Ethics", 
                        help="Topic for the test")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    print(f"Testing Langmanus (No TTS) with topic: {args.topic}")
    
    # Import components - done here to apply config first
    from app.config import settings
    
    # Explicitly disable TTS
    settings.TTS_ENABLED = False
    print("TTS explicitly disabled")
    
    try:
        # Import minimal components for testing
        from app.models import ResearcherInput, TurnIntent, CommentatorInput, Persona
        from app.agents.researcher import get_researcher
        from app.agents.commentator import get_commentator
        
        # Check for required API keys
        if "gemini" in settings.REASONING_MODEL.lower():
            if not os.environ.get("GEMINI_API_KEY") and not settings.GEMINI_API_KEY and not os.environ.get("REASONING_API_KEY") and not settings.REASONING_API_KEY:
                print("WARNING: No Gemini API key found. Tests may fail.")
            else:
                print("Using Gemini model with available API key.")
        else:
            # Fallback to OpenAI check
            if not os.environ.get("OPENAI_API_KEY") and not settings.OPENAI_API_KEY:
                print("WARNING: No OpenAI API key found. Tests may fail.")
            else:
                print("Using OpenAI model with available API key.")
                
        # Check for Tavily API key if we're using it for search
        if not os.environ.get("TAVILY_API_KEY") and not settings.TAVILY_API_KEY:
            print("WARNING: No Tavily API key found. Research may fail.")
        
        # Create a simple persona for testing
        test_persona = Persona(
            name="Test Avatar",
            role="Technology Expert",
            tone="Professional and informative",
            speech_quirks=["Uses clear language"],
            voice={
                "vendor": "none",  # Disable TTS
                "ref_audio": "none",
                "speaker_tag": "none"
            }
        )
        
        # Step 1: Use researcher to get evidence
        print("\n--- Step 1: Research ---")
        researcher = get_researcher()
        research_input = ResearcherInput(
            topic=args.topic,
            intent=TurnIntent.OPENING,
            opponent_point="",
            local_corpus=[]
        )
        print("Calling researcher...")
        research_output = researcher.research(research_input)
        print(f"Research complete: {len(research_output.claims)} claims")
        
        # Step 2: Generate commentary
        print("\n--- Step 2: Commentary ---")
        commentator = get_commentator()
        commentator_input = CommentatorInput(
            topic=args.topic,
            phase="opening",
            intent=TurnIntent.OPENING,
            opponent_summary="",
            persona=test_persona,
            evidence_bundle={
                "topic": args.topic,
                "query": args.topic,  # Adding query field
                "claims": research_output.claims,
                "contradictions": research_output.contradictions,
                "omissions": research_output.omissions,
            }
        )
        print("Calling commentator...")
        commentary_output = commentator.generate_commentary(commentator_input)
        print("\nCommentary result:")
        print(commentary_output.text)
        
        print("\nTest completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
