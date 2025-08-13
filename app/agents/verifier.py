from app.models import DebateTurn, EvidenceBundle
from typing import List

# The Verifier agent checks that every factual claim in a debate turn is supported by evidence.
def verify_turn(turn: DebateTurn, bundle: EvidenceBundle) -> DebateTurn:
    """
    Ensure every factual claim in the turn is supported by evidence in the bundle.
    If not, rewrite or remove unsupported claims and enforce citation format.
    """
    # TODO: Implement LLM-based or rule-based verification and citation enforcement
    # For now, just ensure citations exist
    if not turn.citations:
        turn.text += " [NO CITATION]"
    return turn
