from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum
import uuid

class SourceDoc(BaseModel):
    """Document source with content and metadata"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    url: Optional[str] = None
    title: str 
    content: str
    source_type: str = Field(default="web")  # web, local, reference
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    embedding_id: Optional[str] = None
    trust_score: int = Field(default=5, ge=0, le=10)

class CitationType(str, Enum):
    """Types of citations"""
    WEB = "web"  # [S#] - web source
    LOCAL = "local"  # [L#] - local corpus
    REFERENCE = "ref"  # [R#] - reference material

class TurnIntent(str, Enum):
    """Intent/purpose of a debate turn"""
    OPENING = "opening"
    POSITIONING = "positioning" 
    REBUTTAL = "rebuttal"
    CLOSING = "closing"
    EVIDENCE_HARVEST = "evidence_harvest"

class EpisodePhase(str, Enum):
    """Phases of a debate episode"""
    PRE_RESEARCH = "pre-research"
    OPENING = "opening"
    POSITIONS = "positions"
    CROSSFIRE = "crossfire"
    CLOSING = "closing"
    COMPLETE = "complete"

class VoiceSettings(BaseModel):
    """Voice configuration for TTS"""
    vendor: str = "dia"
    ref_audio: str = Field(..., description="Path to reference audio file")
    speaker_tag: str = Field(..., description="Speaker identification tag")
    pitch: Optional[float] = Field(1.0, ge=0.5, le=2.0)
    speed: Optional[float] = Field(1.0, ge=0.5, le=2.0)
    emotion: Optional[str] = None

class Persona(BaseModel):
    """Avatar personality and voice configuration"""
    name: str
    role: str = Field(..., description="Professional role or expertise area")
    tone: str = Field(..., description="Speaking tone and style")
    speech_quirks: List[str] = Field(default_factory=list)
    default_unknown: str = Field(default="I don't have enough information to answer that.")
    voice: VoiceSettings
    background: Optional[str] = None
    forbidden_topics: List[str] = Field(default_factory=list)
    bias_indicators: List[str] = Field(default_factory=list)

class Avatar(BaseModel):
    """Complete avatar definition with persona and metadata"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    persona: Persona
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    version: str = "1.0"
    tags: List[str] = Field(default_factory=list)

class Citation(BaseModel):
    """Single citation with source information"""
    id: str = Field(..., description="Citation identifier like S1, L2, R3")
    type: CitationType
    url: Optional[str] = None
    title: Optional[str] = None
    excerpt: Optional[str] = None
    confidence: float = Field(1.0, ge=0.0, le=1.0)
    timestamp: Optional[datetime] = None
    trust_score: Optional[float] = Field(None, ge=0.0, le=10.0)  # Accept float trust scores

class Evidence(BaseModel):
    """Single piece of evidence with citations and metadata"""
    text: str = Field(..., description="The evidence statement")
    citations: List[Citation] = Field(default_factory=list)
    confidence: float = Field(1.0, ge=0.0, le=1.0, description="Confidence in this evidence")
    contradictions: List[str] = Field(default_factory=list)
    supporting_evidence: List[str] = Field(default_factory=list)
    topic_relevance: float = Field(1.0, ge=0.0, le=1.0)

class Bundle(BaseModel):
    """Collection of evidence for a specific query/topic"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    topic: str
    query: str
    claims: List[Evidence] = Field(default_factory=list)
    contradictions: List[str] = Field(default_factory=list)
    omissions: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    source_count: int = Field(default=0)
    freshness_score: float = Field(1.0, ge=0.0, le=1.0)

class Turn(BaseModel):
    """Single debate turn by an avatar"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    avatar_id: str
    avatar_key: str = Field(..., description="A1 or A2 for tracking in episode")
    phase: EpisodePhase
    intent: TurnIntent
    text: str = Field(..., description="Final styled text of the turn")
    citations: List[Citation] = Field(default_factory=list)
    beats: List["Beat"] = Field(default_factory=list)
    evidence_bundle_id: Optional[str] = None
    opponent_summary: Optional[str] = None
    audio_path: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    generation_stats: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('avatar_key')
    def validate_avatar_key(cls, v):
        if v not in ['A1', 'A2']:
            raise ValueError('avatar_key must be A1 or A2')
        return v

class EmotionType(str, Enum):
    NEUTRAL = "neutral"
    CONFIDENT = "confident"
    QUESTIONING = "questioning"
    SURPRISED = "surprised"
    EMPHATIC = "emphatic"
    CAUTIOUS = "cautious"
    CONCERNED = "concerned"
    EXCITED = "excited"
    THOUGHTFUL = "thoughtful"
    AMUSED = "amused"
    SERIOUS = "serious"

class TextSpan(BaseModel):
    """A span of text with emotional annotation"""
    start: int
    end: int
    text: str
    emotion: EmotionType = EmotionType.NEUTRAL
    intensity: float = Field(1.0, ge=0.0, le=1.0)

class RichText(BaseModel):
    """Rich text with emotional annotations"""
    plain_text: str
    spans: List[TextSpan] = []

    def to_json(self) -> Dict[str, Any]:
        """Convert to JSON format for frontend rendering"""
        return {
            "text": self.plain_text,
            "spans": [span.dict() for span in self.spans]
        }

class EpisodeConfig(BaseModel):
    """Configuration for a debate episode"""
    topic: str
    avatar_a_path: str = Field(..., description="Path to avatar A YAML file")
    avatar_b_path: str = Field(..., description="Path to avatar B YAML file") 
    max_turns_per_phase: int = Field(default=4)
    enable_citations: bool = Field(default=True)
    enable_verification: bool = Field(default=True)
    freshness_days: int = Field(default=120)
    whitelist_domains: List[str] = Field(default_factory=list)
    blacklist_domains: List[str] = Field(default_factory=list)
    phases: List[EpisodePhase] = Field(default_factory=list)
    max_duration_seconds: int = Field(default=600)  # 10 minutes

class Episode(BaseModel):
    """Complete debate episode with all turns and metadata"""
    id: str = Field(default_factory=lambda: f"ep-{uuid.uuid4().hex[:8]}")
    config: EpisodeConfig
    avatar_a: Avatar
    avatar_b: Avatar
    turns: List[Turn] = Field(default_factory=list)
    current_phase: EpisodePhase = EpisodePhase.PRE_RESEARCH
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    audio_url: Optional[str] = None
    transcript_path: Optional[str] = None
    notes_path: Optional[str] = None
    bundles: Dict[str, Bundle] = Field(default_factory=dict)
    
    # Generated media assets (audio, captions, videos, clips)
    assets: Optional["EpisodeAssets"] = None
    
    @property
    def duration_minutes(self) -> Optional[float]:
        """Calculate episode duration in minutes"""
        if self.completed_at and self.started_at:
            delta = self.completed_at - self.started_at
            return delta.total_seconds() / 60
        return None
    
    @property
    def turn_count(self) -> int:
        """Total number of turns in episode"""
        return len(self.turns)
    
    def get_turns_by_avatar(self, avatar_key: str) -> List[Turn]:
        """Get all turns by specific avatar"""
        return [turn for turn in self.turns if turn.avatar_key == avatar_key]


class EpisodeAssets(BaseModel):
    """Produced media asset paths for an episode"""
    mp3: Optional[str] = None
    srt: Optional[str] = None
    vtt: Optional[str] = None
    videos: List[str] = Field(default_factory=list)
    clips: List[str] = Field(default_factory=list)

# Agent Input/Output Schemas (matching pipeline YAML)

class ResearcherInput(BaseModel):
    """Input schema for Researcher agent"""
    topic: str
    intent: TurnIntent
    opponent_point: str = ""
    local_corpus: List[str] = Field(default_factory=list)

class ResearcherOutput(BaseModel):
    """Output schema for Researcher agent"""
    claims: List[Evidence]
    contradictions: List[str] = Field(default_factory=list)
    omissions: List[str] = Field(default_factory=list)
    bundle_id: str

class CommentatorInput(BaseModel):
    """Input schema for Commentator agent"""
    topic: str
    phase: EpisodePhase
    intent: TurnIntent
    opponent_summary: str = ""
    persona: Persona
    evidence_bundle: Bundle

class CommentatorOutput(BaseModel):
    """Output schema for Commentator agent"""
    text: str
    citations: List[Citation]

class BeatProsody(BaseModel):
    """Prosodic guidance for a beat"""
    pace: Optional[str] = None  # slow|medium|fast
    pitch: Optional[str] = None  # low|neutral|high
    pauses: Optional[str] = None  # few|moderate|many

class Beat(BaseModel):
    """Structured speaking unit used for TTS pacing and naturalness"""
    label: str
    target_duration_s: Optional[float] = None
    text: str  # spoken text (clean; no [S#]/[L#]/[R#])
    emotion: Optional[str] = None
    prosody: Optional[BeatProsody] = None
    references: List[str] = Field(default_factory=list)

class VerifierInput(BaseModel):
    """Input schema for Verifier agent"""
    draft: str
    evidence_bundle: Bundle
    persona: Persona

class VerifierOutput(BaseModel):
    """Output schema for Verifier agent"""
    text: str
    citations: List[Citation]
    verification_notes: List[str] = Field(default_factory=list)

class StyleInput(BaseModel):
    """Input schema for Style agent"""
    verified_text: str
    persona: Persona

class StyleOutput(BaseModel):
    """Output schema for Style agent"""
    styled_text: str

class TTSInput(BaseModel):
    """Input schema for TTS agent"""
    styled_text: str
    persona: Persona

class TTSOutput(BaseModel):
    """Output schema for TTS agent"""
    audio_path: str
    duration_seconds: Optional[float] = None
    references: List[str] = Field(default_factory=list)

# API Schemas

class EpisodeStartRequest(BaseModel):
    """Request schema for starting an episode"""
    topic: str
    avatar_a_path: str
    avatar_b_path: str
    episode_id: Optional[str] = None
    config: Optional[Dict[str, Any]] = Field(default_factory=dict)

class EpisodeProgressEvent(BaseModel):
    """SSE event for episode progress"""
    event: str = Field(..., description="Event type: research_progress, turn_generated, etc.")
    data: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class ResearchRequest(BaseModel):
    """Request schema for research endpoint"""
    topic: str
    max_results: int = Field(default=12)
    freshness_days: int = Field(default=120)
    include_local: bool = Field(default=True)

class AvatarCreateRequest(BaseModel):
    """Request schema for creating/updating avatars"""
    persona: Persona
    tags: List[str] = Field(default_factory=list)

# Utility functions for model conversion

def bundle_to_dict(bundle: Bundle) -> Dict[str, Any]:
    """Convert Bundle to dict for agent consumption"""
    return bundle.dict()

def persona_from_yaml_dict(data: Dict[str, Any]) -> Persona:
    """Create Persona from YAML data"""
    return Persona(**data)

def episode_to_transcript_format(episode: Episode) -> List[Dict[str, Any]]:
    """Convert episode to transcript format for JSON export"""
    return [
        {
            "avatar": turn.avatar_key,
            "phase": turn.phase.value,
            "text": turn.text,
            "citations": [cite.dict() for cite in turn.citations],
            "audio_path": turn.audio_path,
            "timestamp": turn.created_at.isoformat()
        }
        for turn in episode.turns
    ]
