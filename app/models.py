from pydantic import BaseModel, Field
from typing import List, Optional, Dict

class Avatar(BaseModel):
    id: str
    name: str
    persona: str
    voice_ref: Optional[str] = None  # Path to reference audio
    corpus: Optional[str] = None     # Path to seed corpus

class SourceDoc(BaseModel):
    id: str
    title: str
    url: str
    content: str
    date: Optional[str] = None
    trust: Optional[float] = None
    local: bool = False

class EvidenceBundle(BaseModel):
    id: str
    avatar_id: str
    topic: str
    sources: List[SourceDoc]
    created_at: Optional[str] = None

class DebateTurn(BaseModel):
    id: str
    episode_id: str
    avatar_id: str
    text: str
    citations: List[str] = Field(default_factory=list)
    audio_path: Optional[str] = None
    turn_index: int

class Episode(BaseModel):
    id: str
    topic: str
    avatars: List[str]
    turns: List[DebateTurn] = Field(default_factory=list)
    bundles: Dict[str, EvidenceBundle] = Field(default_factory=dict)
    transcript_path: Optional[str] = None
    audio_path: Optional[str] = None
    created_at: Optional[str] = None
