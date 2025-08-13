#!/usr/bin/env python3
"""
Simple CLI script to run a debate episode.
Usage: python scripts/run_episode.py --topic "AI Safety" --avatar-a avatars/alex.yaml --avatar-b avatars/nova.yaml
"""

import asyncio
import argparse
import sys
from pathlib import Path

# Add app to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.models import EpisodeConfig
from app.orchestrator.episode_runner import get_episode_runner
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def run_episode_cli(topic: str, avatar_a_path: str, avatar_b_path: str):
    """Run a debate episode via CLI"""
    
    # Create episode configuration
    config = EpisodeConfig(
        topic=topic,
        avatar_a_path=avatar_a_path,
        avatar_b_path=avatar_b_path,
        max_turns_per_phase=2,  # Keep it short for testing
        enable_verification=True,
        target_duration_minutes=15
    )
    
    # Get episode runner
    runner = get_episode_runner()
    
    print(f"\n🎬 Starting debate episode on: {topic}")
    print(f"Avatar A: {avatar_a_path}")
    print(f"Avatar B: {avatar_b_path}")
    print("-" * 60)
    
    try:
        # Run episode with streaming updates
        async for event in runner.run_episode(config):
            event_type = event["event"]
            data = event["data"]
            
            if event_type == "episode_started":
                print(f"📝 Episode {data['episode_id']} started")
                
            elif event_type == "phase_started":
                print(f"\n🔄 Phase: {data['phase'].upper()}")
                
            elif event_type == "research_complete":
                print(f"🔍 Research complete: {data['bundle_count']} bundles")
                
            elif event_type == "turn_started":
                print(f"  💬 {data['avatar_name']} speaking (Turn {data['turn_number']})")
                
            elif event_type == "turn_complete":
                print(f"  ✅ Turn complete: {data['text_length']} chars, {data['citation_count']} citations")
                if data['audio_path']:
                    print(f"     🔊 Audio: {data['audio_path']}")
                
            elif event_type == "episode_complete":
                print(f"\n🎉 Episode complete!")
                print(f"   Total turns: {data['turn_count']}")
                print(f"   Duration: {data['duration_minutes']:.1f} minutes")
                print(f"   Transcript: {data['transcript_path']}")
                
            elif event_type == "episode_error":
                print(f"❌ Error: {data['error']}")
                
    except KeyboardInterrupt:
        print("\n⏹️  Episode interrupted by user")
    except Exception as e:
        print(f"❌ Episode failed: {e}")

def main():
    parser = argparse.ArgumentParser(description="Run an AI debate episode")
    parser.add_argument("--topic", required=True, help="Debate topic")
    parser.add_argument("--avatar-a", required=True, help="Path to first avatar YAML")
    parser.add_argument("--avatar-b", required=True, help="Path to second avatar YAML")
    parser.add_argument("--turns", type=int, default=2, help="Max turns per phase")
    
    args = parser.parse_args()
    
    # Validate avatar files exist
    if not Path(args.avatar_a).exists():
        print(f"❌ Avatar A file not found: {args.avatar_a}")
        sys.exit(1)
        
    if not Path(args.avatar_b).exists():
        print(f"❌ Avatar B file not found: {args.avatar_b}")
        sys.exit(1)
    
    # Run the episode
    asyncio.run(run_episode_cli(args.topic, args.avatar_a, args.avatar_b))

if __name__ == "__main__":
    main()
