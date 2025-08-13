from app.models import Avatar, EvidenceBundle, DebateTurn
from typing import List

# The Commentator agent generates debate turns for an avatar using the evidence bundle.
def generate_debate_turns(topic: str, avatar: Avatar, bundle: EvidenceBundle, turn_index: int) -> DebateTurn:
    """
    Generate a debate turn for the avatar, referencing evidence from the bundle.
    """
    # TODO: Integrate with LLM and prompt templates for persona-locked generation
    # For now, create a stub turn
    text = f"[{avatar.name}] On the topic '{topic}', here is my argument. [CITATIONS]"
    citations = [src.id for src in bundle.sources[:1]]  # Example: cite first source
    turn = DebateTurn(
        id=f"turn-{avatar.id}-{turn_index}",
        episode_id="",  # To be filled by orchestrator
        avatar_id=avatar.id,
        text=text,
        citations=citations,
        audio_path=None,
        turn_index=turn_index
    )
    return turn
