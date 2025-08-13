from app.models import DebateTurn, Avatar

# The Style agent applies persona-specific quirks, tone, and style to debate turns.
def apply_style(turn: DebateTurn, avatar: Avatar) -> DebateTurn:
    """
    Modify the debate turn's text to reflect the avatar's persona, quirks, and tone.
    """
    # TODO: Integrate with prompt templates and persona data
    # For now, append a persona signature
    turn.text += f"\n-- {avatar.name} persona style applied"
    return turn
