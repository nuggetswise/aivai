#!/usr/bin/env python
"""
Test script for Langmanus orchestration framework.
This script tests the core functionality of the Langmanus pipeline
with a single debate turn using the Alex avatar.
"""

import os
import sys
import json
import yaml
import argparse
import logging
import traceback
from pathlib import Path

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

# Import app components - using getter functions instead of direct imports
from app.config import settings
from app.deps import get_llm_router, get_search_client, get_embeddings_client
from app.retrieval.store import get_vector_store
from app.retrieval.indexer import get_indexer
from app.retrieval.rank import get_ranker
from app.agents.researcher import get_researcher
from app.agents.commentator import get_commentator
from app.agents.verifier import get_verifier
from app.agents.style import get_style_agent
from app.models import ResearcherInput, CommentatorInput, VerifierInput, StyleInput, Persona, TurnIntent, EpisodePhase, Bundle

# Override TTS setting
settings.TTS_ENABLED = False

# Check for required API keys
if "gemini" in settings.REASONING_MODEL.lower():
    if not os.environ.get("GEMINI_API_KEY") and not settings.GEMINI_API_KEY and not os.environ.get("REASONING_API_KEY") and not settings.REASONING_API_KEY:
        print("WARNING: No Gemini API key found. Tests may fail.")
    else:
        print("Using Gemini model with available API key.")
        
def load_avatar(avatar_path):
    """Load avatar configuration from YAML file"""
    with open(avatar_path, 'r') as f:
        return yaml.safe_load(f)

def create_persona_from_avatar(avatar_data):
    """Convert avatar YAML data to Persona object"""
    return Persona(
        name=avatar_data.get('name', 'Unknown'),
        role=avatar_data.get('role', 'Debater'),
        tone=avatar_data.get('tone', 'neutral'),
        speech_quirks=avatar_data.get('speech_quirks', []),
        default_unknown=avatar_data.get('default_unknown', "I don't have enough information on that."),
        voice=avatar_data.get('voice', {}),
        background=avatar_data.get('background'),
        forbidden_topics=avatar_data.get('forbidden_topics', []),
        bias_indicators=avatar_data.get('bias_indicators', [])
    )

def initialize_components():
    """Initialize all required components and test connections"""
    try:
        print("Initializing components...")
        
        # Initialize LLM router and test connection
        llm_router = get_llm_router()
        print("✓ LLM Router initialized")
        
        # Initialize search client and test connection
        search_client = get_search_client()
        if search_client.tavily_available:
            print("✓ Tavily search client initialized")
        else:
            print("⚠ Tavily search client not available - will use fallback")
        
        # Initialize embedding client
        embeddings_client = get_embeddings_client()
        print("✓ Embeddings client initialized")
        
        # Initialize vector store
        vector_store = get_vector_store()
        stats = vector_store.get_stats()
        print(f"✓ Vector store initialized with {stats['document_count']} documents")
        
        # Initialize indexer
        indexer = get_indexer()
        print("✓ Document indexer initialized")
        
        # Initialize ranker
        ranker = get_ranker()
        print("✓ Evidence ranker initialized")
        
        return True
    except Exception as e:
        print(f"❌ Error initializing components: {e}")
        traceback.print_exc()
        return False

def test_langmanus_pipeline(topic, avatar_path, save_output=True):
    """Run a single debate turn through the Langmanus pipeline"""
    print(f"Testing Langmanus pipeline with topic: {topic}")
    print(f"Using avatar: {avatar_path}")
    
    # Load avatar configuration
    avatar_data = load_avatar(avatar_path)
    avatar_name = avatar_data.get('name', 'Unknown')
    print(f"Avatar loaded: {avatar_name}")
    
    # Convert to Persona object
    persona = create_persona_from_avatar(avatar_data)
    
    # Create output directories if they don't exist
    if save_output:
        os.makedirs("data/turns", exist_ok=True)
        os.makedirs("data/audio", exist_ok=True)
    
    try:
        # Step 1: Research phase
        print("\n🔍 Step 1: Researcher gathering evidence...")
        researcher = get_researcher()
        research_input = ResearcherInput(
            topic=topic,
            intent=TurnIntent.OPENING,
            opponent_point="",
            local_corpus=[]  # No local corpus for simple test
        )
        research_output = researcher.research(research_input)
        print(f"✅ Evidence gathered: {len(research_output.claims)} claims found")
        
        # Check for empty claims and provide fallback with retry
        if not research_output.claims:
            print("⚠️ No claims found; retrying with relaxed policy...")
            
            # Retry with more permissive search
            relaxed_input = ResearcherInput(
                topic=f"{topic} overview OR summary OR basics",  # More generic query
                intent=TurnIntent.OPENING,
                opponent_point="",
                local_corpus=[]
            )
            research_output = researcher.research(relaxed_input)
            print(f"🔄 Retry gathered: {len(research_output.claims)} claims found")
            
            # Final fallback if still no claims
            if not research_output.claims:
                print("⚠️ No claims found after retry; falling back to default_unknown turn.")
                return persona.default_unknown
        
        # Normalize ResearcherOutput to Bundle with required fields
        bundle = Bundle.model_validate({
            "topic": topic,
            "query": f"{topic} evidence and analysis",
            **research_output.model_dump()
        })
        
        # Step 2: Commentator phase
        print("\n💬 Step 2: Commentator generating response...")
        commentator = get_commentator()
        commentator_input = CommentatorInput(
            topic=topic,
            phase=EpisodePhase.OPENING,
            intent=TurnIntent.OPENING,
            opponent_summary="",
            persona=persona,
            evidence_bundle=bundle
        )
        commentary = commentator.generate_turn(commentator_input)
        print(f"✅ Commentary generated: {len(commentary.text.split())} words")
        
        # Step 3: Verifier phase
        print("\n🔎 Step 3: Verifier checking facts...")
        verifier = get_verifier()
        verifier_input = VerifierInput(
            draft=commentary.text,
            evidence_bundle=bundle,
            persona=persona
        )
        verified_text = verifier.verify_turn(verifier_input)
        print("✅ Verification complete")
        
        # Step 4: Style phase
        print("\n🎭 Step 4: Style adjusting tone and adding emotional expressions...")
        styler = get_style_agent()
        style_input = StyleInput(
            verified_text=verified_text.text,
            persona=persona
        )
        styled_text = styler.apply_style(style_input)
        print("✅ Style adjustment complete")
        
        # Final output - no TTS needed
        print("\n📝 Final debate turn ready")
        
        # Save outputs
        if save_output:
            turn_data = {
                "topic": topic,
                "avatar": avatar_name,
                "evidence_bundle": bundle.model_dump(),
                "commentary": commentary.dict() if hasattr(commentary, "dict") else {},
                "verified_text": verified_text.dict() if hasattr(verified_text, "dict") else {},
                "styled_text": styled_text.dict() if hasattr(styled_text, "dict") else {},
                "final_turn": styled_text.styled_text  # The complete turn with emotions
            }
            
            output_path = f"data/turns/test_{avatar_name.lower()}.json"
            with open(output_path, 'w') as f:
                json.dump(turn_data, f, indent=2, default=str)  # default=str handles datetime objects
            print(f"\n💾 Results saved to {output_path}")
        
        # Return the final text with emotional annotations
        return styled_text.styled_text
        
    except Exception as e:
        print(f"❌ Error during pipeline execution: {e}")
        traceback.print_exc()
        return f"Pipeline failed: {str(e)}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the Langmanus pipeline with a single debate turn")
    parser.add_argument("--topic", default="Climate change adaptation strategies", help="Debate topic")
    parser.add_argument("--avatar", default="avatars/alex.yaml", help="Path to avatar YAML file")
    parser.add_argument("--no-save", action="store_true", help="Don't save outputs to disk")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Set up logging if debug is enabled
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
        print("Debug logging enabled")
    else:
        logging.basicConfig(level=logging.INFO)
    
    # Initialize all components before running the pipeline
    if not initialize_components():
        print("❌ Failed to initialize components. Exiting.")
        sys.exit(1)
    
    print("\n=== Starting Pipeline Test ===\n")
    
    result = test_langmanus_pipeline(
        topic=args.topic,
        avatar_path=args.avatar,
        save_output=not args.no_save
    )
    
    print("\n📝 Final result:\n")
    print(result)