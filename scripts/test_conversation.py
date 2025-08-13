#!/usr/bin/env python
"""
Test script for LangManus conversation between two avatars.
This script tests a full conversation between Alex Chen and Nova Rivers.
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

# Import app components
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

def generate_debate_turn(topic, avatar_path, phase, intent, opponent_point="", save_output=True):
    """Run a single debate turn for an avatar"""
    # Load avatar configuration
    avatar_data = load_avatar(avatar_path)
    avatar_name = avatar_data.get('name', 'Unknown')
    print(f"\n=== {avatar_name}'s Turn ===")
    
    # Convert to Persona object
    persona = create_persona_from_avatar(avatar_data)
    
    # Create output directories if they don't exist
    if save_output:
        os.makedirs("data/turns", exist_ok=True)
        os.makedirs("data/transcripts", exist_ok=True)
    
    try:
        # Step 1: Research phase
        print("\n🔍 Researcher gathering evidence...")
        researcher = get_researcher()
        research_input = ResearcherInput(
            topic=topic,
            intent=intent,
            opponent_point=opponent_point,
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
                intent=intent,
                opponent_point=opponent_point,
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
        print("\n💬 Commentator generating response...")
        commentator = get_commentator()
        commentator_input = CommentatorInput(
            topic=topic,
            phase=phase,
            intent=intent,
            opponent_summary=opponent_point,
            persona=persona,
            evidence_bundle=bundle
        )
        commentary = commentator.generate_turn(commentator_input)
        print(f"✅ Commentary generated: {len(commentary.text.split())} words")
        
        # Step 3: Verifier phase
        print("\n🔎 Verifier checking facts...")
        verifier = get_verifier()
        verifier_input = VerifierInput(
            draft=commentary.text,
            evidence_bundle=bundle,
            persona=persona
        )
        verified_text = verifier.verify_turn(verifier_input)
        print("✅ Verification complete")
        
        # Step 4: Style phase
        print("\n🎭 Style adjusting tone and adding emotional expressions...")
        styler = get_style_agent()
        style_input = StyleInput(
            verified_text=verified_text.text,
            persona=persona
        )
        styled_text = styler.apply_style(style_input)
        print("✅ Style adjustment complete")
        
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
            
            output_path = f"data/turns/{avatar_name.lower().replace(' ', '_')}_{phase.value}.json"
            with open(output_path, 'w') as f:
                json.dump(turn_data, f, indent=2, default=str)
            print(f"\n💾 Results saved to {output_path}")
        
        # Return the final text with emotional annotations
        return styled_text.styled_text
        
    except Exception as e:
        print(f"❌ Error during pipeline execution: {e}")
        traceback.print_exc()
        return f"Pipeline failed: {str(e)}"

def run_conversation(topic, alex_path="avatars/alex.yaml", nova_path="avatars/nova.yaml", save_output=True):
    """Run a full conversation between Alex and Nova"""
    print(f"\n=== Starting Conversation on Topic: {topic} ===\n")
    
    # Create output directory for transcript
    if save_output:
        os.makedirs("data/transcripts", exist_ok=True)
    
    transcript = []
    
    try:
        # Opening phase
        print("\n== OPENING PHASE ==")
        
        # Alex's opening
        alex_opening = generate_debate_turn(
            topic=topic, 
            avatar_path=alex_path, 
            phase=EpisodePhase.OPENING,
            intent=TurnIntent.OPENING,
            save_output=save_output
        )
        transcript.append({
            "avatar": "Alex Chen",
            "phase": "opening",
            "text": alex_opening
        })
        print("\n📝 Alex Opening:\n")
        print(alex_opening)
        
        # Nova's opening
        nova_opening = generate_debate_turn(
            topic=topic, 
            avatar_path=nova_path, 
            phase=EpisodePhase.OPENING,
            intent=TurnIntent.OPENING,
            opponent_point=alex_opening,
            save_output=save_output
        )
        transcript.append({
            "avatar": "Nova Rivers",
            "phase": "opening",
            "text": nova_opening
        })
        print("\n📝 Nova Opening:\n")
        print(nova_opening)
        
        # Crossfire phase
        print("\n== CROSSFIRE PHASE ==")
        
        # Alex's rebuttal
        alex_rebuttal = generate_debate_turn(
            topic=topic, 
            avatar_path=alex_path, 
            phase=EpisodePhase.CROSSFIRE,
            intent=TurnIntent.REBUTTAL,
            opponent_point=nova_opening,
            save_output=save_output
        )
        transcript.append({
            "avatar": "Alex Chen",
            "phase": "crossfire",
            "text": alex_rebuttal
        })
        print("\n📝 Alex Rebuttal:\n")
        print(alex_rebuttal)
        
        # Nova's rebuttal
        nova_rebuttal = generate_debate_turn(
            topic=topic, 
            avatar_path=nova_path, 
            phase=EpisodePhase.CROSSFIRE,
            intent=TurnIntent.REBUTTAL,
            opponent_point=alex_rebuttal,
            save_output=save_output
        )
        transcript.append({
            "avatar": "Nova Rivers",
            "phase": "crossfire",
            "text": nova_rebuttal
        })
        print("\n📝 Nova Rebuttal:\n")
        print(nova_rebuttal)
        
        # Closing phase
        print("\n== CLOSING PHASE ==")
        
        # Alex's closing
        alex_closing = generate_debate_turn(
            topic=topic, 
            avatar_path=alex_path, 
            phase=EpisodePhase.CLOSING,
            intent=TurnIntent.CLOSING,
            opponent_point=nova_rebuttal,
            save_output=save_output
        )
        transcript.append({
            "avatar": "Alex Chen",
            "phase": "closing",
            "text": alex_closing
        })
        print("\n📝 Alex Closing:\n")
        print(alex_closing)
        
        # Nova's closing
        nova_closing = generate_debate_turn(
            topic=topic, 
            avatar_path=nova_path, 
            phase=EpisodePhase.CLOSING,
            intent=TurnIntent.CLOSING,
            opponent_point=alex_closing,
            save_output=save_output
        )
        transcript.append({
            "avatar": "Nova Rivers",
            "phase": "closing",
            "text": nova_closing
        })
        print("\n📝 Nova Closing:\n")
        print(nova_closing)
        
        # Save complete transcript
        if save_output:
            transcript_path = f"data/transcripts/debate_{topic.lower().replace(' ', '_')}.json"
            with open(transcript_path, 'w') as f:
                json.dump(transcript, f, indent=2, default=str)
            print(f"\n💾 Complete transcript saved to {transcript_path}")
        
        return transcript
        
    except Exception as e:
        print(f"❌ Error during conversation: {e}")
        traceback.print_exc()
        return []

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a conversation between Alex Chen and Nova Rivers")
    parser.add_argument("--topic", default="Climate change adaptation strategies", help="Debate topic")
    parser.add_argument("--alex", default="avatars/alex.yaml", help="Path to Alex avatar YAML file")
    parser.add_argument("--nova", default="avatars/nova.yaml", help="Path to Nova avatar YAML file")
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
    
    # Run full conversation
    transcript = run_conversation(
        topic=args.topic,
        alex_path=args.alex,
        nova_path=args.nova,
        save_output=not args.no_save
    )
    
    print("\n=== Conversation Complete ===")
    print(f"Generated {len(transcript)} turns in the conversation")