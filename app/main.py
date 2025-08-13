from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Set
import asyncio
import json
import uuid
import os
from pathlib import Path
import yaml
from datetime import datetime, timedelta
import random

from app.orchestrator.episode_runner import EpisodeRunner
from app.models import EpisodeConfig, Avatar as AvatarModel
from app.deps import get_config

app = FastAPI(title="AIvAI Debate Platform", version="1.0.0")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # Next.js dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (audio, transcripts)
app.mount("/static", StaticFiles(directory="data"), name="static")

# Models
class Avatar(BaseModel):
    path: str
    name: str
    role: str
    tone: str

class DebateRequest(BaseModel):
    topic: str
    avatar_a: str
    avatar_b: str
    max_turns_per_phase: int = 2
    enable_verification: bool = True

class EpisodeResponse(BaseModel):
    episode_id: str
    status: str
    message: str

class EpisodeStatus(BaseModel):
    episode_id: str
    status: str
    progress: float
    current_phase: str
    message: str
    created_at: datetime
    updated_at: datetime

class Citation(BaseModel):
    source: str
    title: str
    url: str
    snippet: str
    trust_score: Optional[float] = None

class Evidence(BaseModel):
    source_id: str
    url: str
    title: str
    content: str
    publish_date: Optional[str] = None
    trust_score: Optional[float] = None
    domain: str
    citation_count: int = 0

class EvidenceBundle(BaseModel):
    episode_id: str
    topic: str
    evidence: List[Evidence] = []
    query_terms: List[str] = []
    generated_at: datetime

class Turn(BaseModel):
    turn_id: str
    avatar_id: str
    avatar_name: str
    phase: str
    text: Optional[str] = None
    audio_path: Optional[str] = None
    duration: float = 0.0
    citations: List[Citation] = []
    timestamp: datetime

class Episode(BaseModel):
    episode_id: str
    topic: str
    avatar_a: str
    avatar_b: str
    status: str
    turns: List[Turn] = []
    transcript_path: Optional[str] = None
    total_duration: float = 0.0
    created_at: datetime
    completed_at: Optional[datetime] = None

# In-memory storage (in production, use a proper database)
episodes: Dict[str, Episode] = {}
episode_statuses: Dict[str, EpisodeStatus] = {}
evidence_bundles: Dict[str, EvidenceBundle] = {}
active_websockets: Dict[str, List[WebSocket]] = {}

@app.get("/")
async def root():
    return {"message": "AIvAI Debate Platform API", "version": "1.0.0"}

@app.get("/avatars", response_model=List[Avatar])
async def get_avatars():
    """Get list of available avatars"""
    avatars_dir = Path("avatars")
    avatars = []
    
    if avatars_dir.exists():
        for avatar_file in avatars_dir.glob("*.yaml"):
            try:
                with open(avatar_file, 'r') as f:
                    avatar_data = yaml.safe_load(f)
                    
                avatars.append(Avatar(
                    path=str(avatar_file),
                    name=avatar_data.get('name', avatar_file.stem.title()),
                    role=avatar_data.get('role', 'Debater'),
                    tone=avatar_data.get('tone', 'Professional')
                ))
            except Exception as e:
                print(f"Error loading avatar {avatar_file}: {e}")
    
    # Return default avatars if none found
    if not avatars:
        avatars = [
            Avatar(
                path="avatars/alex.yaml",
                name="Alex",
                role="Analytical Debater",
                tone="Professional"
            ),
            Avatar(
                path="avatars/nova.yaml", 
                name="Nova",
                role="Creative Advocate",
                tone="Passionate"
            )
        ]
    
    return avatars

@app.post("/episodes", response_model=EpisodeResponse)
async def create_episode(request: DebateRequest):
    """Create a new debate episode"""
    episode_id = str(uuid.uuid4())
    
    # Create episode
    episode = Episode(
        episode_id=episode_id,
        topic=request.topic,
        avatar_a=request.avatar_a,
        avatar_b=request.avatar_b,
        status="created",
        created_at=datetime.now()
    )
    
    # Create status
    status = EpisodeStatus(
        episode_id=episode_id,
        status="created",
        progress=0.0,
        current_phase="initialization",
        message="Episode created, ready to start",
        created_at=datetime.now(),
        updated_at=datetime.now()
    )
    
    episodes[episode_id] = episode
    episode_statuses[episode_id] = status
    active_websockets[episode_id] = []
    
    return EpisodeResponse(
        episode_id=episode_id,
        status="created",
        message="Episode created successfully"
    )

@app.post("/episodes/{episode_id}/start")
async def start_episode(episode_id: str, background_tasks: BackgroundTasks):
    """Start a debate episode"""
    if episode_id not in episodes:
        raise HTTPException(status_code=404, detail="Episode not found")
    
    episode = episodes[episode_id]
    if episode.status != "created":
        raise HTTPException(status_code=400, detail="Episode already started or completed")
    
    # Update status
    episode.status = "running"
    episode_statuses[episode_id].status = "running"
    episode_statuses[episode_id].current_phase = "research"
    episode_statuses[episode_id].message = "Starting debate episode..."
    episode_statuses[episode_id].updated_at = datetime.now()
    
    # Start episode in background
    background_tasks.add_task(run_episode, episode_id)
    
    return {"message": "Episode started", "episode_id": episode_id}

@app.get("/episodes/{episode_id}/status", response_model=EpisodeStatus)
async def get_episode_status(episode_id: str):
    """Get episode status"""
    if episode_id not in episode_statuses:
        raise HTTPException(status_code=404, detail="Episode not found")
    
    return episode_statuses[episode_id]

@app.get("/episodes/{episode_id}", response_model=Episode)
async def get_episode(episode_id: str):
    """Get episode details"""
    if episode_id not in episodes:
        raise HTTPException(status_code=404, detail="Episode not found")
    
    return episodes[episode_id]

# WebSocket connection handler
@app.websocket("/ws/{episode_id}")
async def websocket_endpoint(websocket: WebSocket, episode_id: str):
    await websocket.accept()
    
    if episode_id not in active_websockets:
        active_websockets[episode_id] = []
    active_websockets[episode_id].append(websocket)
    
    try:
        while True:
            # Keep connection alive and handle client messages
            data = await websocket.receive_text()
            msg = json.loads(data)
            
            if msg.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
            
            # Client can request current status
            elif msg.get("type") == "get_status":
                if episode_id in episode_statuses:
                    await websocket.send_json({
                        "type": "status_update",
                        "episode_status": episode_statuses[episode_id].dict()
                    })
            
            # Client can request transcript so far
            elif msg.get("type") == "get_transcript":
                if episode_id in episodes:
                    await websocket.send_json({
                        "type": "transcript_update",
                        "turns": [turn.dict() for turn in episodes[episode_id].turns]
                    })
    
    except WebSocketDisconnect:
        # Remove this websocket from active connections
        if episode_id in active_websockets:
            active_websockets[episode_id].remove(websocket)

@app.get("/files/audio/{filename}")
async def get_audio_file(filename: str):
    """Serve audio files"""
    file_path = Path("data/audio") / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
    
    return FileResponse(file_path, media_type="audio/wav")

@app.get("/files/transcripts/{filename}")
async def get_transcript_file(filename: str):
    """Serve transcript files"""
    file_path = Path("data/transcripts") / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Transcript file not found")
    
    return FileResponse(file_path, media_type="text/plain")

# Helper to broadcast message to all connected clients for an episode
async def broadcast_to_websockets(episode_id: str, message: dict):
    if episode_id in active_websockets:
        # Make a copy to avoid modification during iteration issues
        websockets_copy = active_websockets[episode_id].copy()
        for websocket in websockets_copy:
            try:
                await websocket.send_json(message)
            except Exception:
                # Remove dead connections
                if websocket in active_websockets[episode_id]:
                    active_websockets[episode_id].remove(websocket)

# New function to notify clients about debate progress with rich updates
async def notify_turn_generated(episode_id: str, turn: Turn):
    if episode_id in episodes:
        turn_data = {
            "turn_id": turn.turn_id,
            "avatar_id": turn.avatar_id,
            "avatar_name": turn.avatar_name,
            "phase": turn.phase,
            "text": turn.text,
            "rich_text": turn.rich_text.to_json() if turn.rich_text else None,
            "audio_path": turn.audio_path,
            "citations": [citation.dict() for citation in turn.citations],
            "timestamp": turn.timestamp.isoformat(),
            "duration": turn.duration
        }
        
        await broadcast_to_websockets(episode_id, {
            "type": "turn_generated",
            "turn": turn_data,
            "progress": episode_statuses[episode_id].progress,
            "current_phase": episode_statuses[episode_id].current_phase
        })

# New function to notify about research progress
async def notify_research_progress(episode_id: str, avatar_id: str, progress: float, message: str):
    await broadcast_to_websockets(episode_id, {
        "type": "research_progress",
        "avatar_id": avatar_id,
        "progress": progress,
        "message": message
    })

# New function to notify about TTS progress
async def notify_tts_progress(episode_id: str, turn_id: str, progress: float, message: str):
    await broadcast_to_websockets(episode_id, {
        "type": "tts_progress",
        "turn_id": turn_id,
        "progress": progress,
        "message": message
    })

# New function to notify about phase change
async def notify_phase_change(episode_id: str, phase: str, message: str):
    if episode_id in episode_statuses:
        episode_status = episode_statuses[episode_id]
        episode_status.current_phase = phase
        episode_status.message = message
        
        await broadcast_to_websockets(episode_id, {
            "type": "phase_change",
            "phase": phase,
            "message": message,
            "episode_status": episode_status.dict()
        })

async def update_episode_status(episode_id: str, status: str, progress: float, phase: str, message: str):
    """Update episode status and broadcast to WebSocket clients"""
    if episode_id in episode_statuses:
        episode_statuses[episode_id].status = status
        episode_statuses[episode_id].progress = progress
        episode_statuses[episode_id].current_phase = phase
        episode_statuses[episode_id].message = message
        episode_statuses[episode_id].updated_at = datetime.now()
        
        # Broadcast update
        await broadcast_to_websockets(episode_id, {
            "type": "status_update",
            "episode_status": episode_statuses[episode_id].dict()
        })

async def run_episode(episode_id: str):
    """Run the actual debate episode"""
    try:
        episode = episodes[episode_id]
        
        # Simulate episode phases
        phases = [
            ("research", "Researching topic and gathering evidence..."),
            ("opening", "Generating opening statements..."),
            ("debate", "Conducting debate rounds..."),
            ("closing", "Preparing closing arguments..."),
            ("synthesis", "Generating final synthesis...")
        ]
        
        for i, (phase, message) in enumerate(phases):
            progress = (i / len(phases)) * 100
            await update_episode_status(episode_id, "running", progress, phase, message)
            
            # Simulate work
            await asyncio.sleep(3)
            
            # Add sample turns for demonstration
            if phase == "debate":
                for turn_num in range(2):  # 2 turns per avatar
                    for avatar_id in ["A1", "A2"]:
                        avatar_name = "Alex" if avatar_id == "A1" else "Nova"
                        
                        turn = Turn(
                            turn_id=f"{episode_id}_{phase}_{avatar_id}_{turn_num}",
                            avatar_id=avatar_id,
                            avatar_name=avatar_name,
                            phase=phase,
                            text=f"This is a sample argument from {avatar_name} during the {phase} phase.",
                            citations=[],
                            timestamp=datetime.now()
                        )
                        
                        episodes[episode_id].turns.append(turn)
                        
                        # Broadcast turn completion
                        await broadcast_to_websockets(episode_id, {
                            "type": "turn_complete",
                            "turn": turn.dict()
                        })
                        
                        await asyncio.sleep(2)
        
        # Complete episode
        episodes[episode_id].status = "complete"
        episodes[episode_id].completed_at = datetime.now()
        episodes[episode_id].total_duration = 300  # 5 minutes sample
        
        await update_episode_status(episode_id, "complete", 100.0, "complete", "Debate completed successfully!")
        
        # Broadcast completion
        await broadcast_to_websockets(episode_id, {
            "type": "episode_complete",
            "episode": episodes[episode_id].dict()
        })
        
    except Exception as e:
        print(f"Error running episode {episode_id}: {e}")
        await update_episode_status(episode_id, "error", 0.0, "error", f"Error: {str(e)}")

# Evidence/Source inspection endpoints
@app.get("/episodes/{episode_id}/evidence", response_model=List[EvidenceBundle])
async def get_episode_evidence(episode_id: str):
    """Get all evidence bundles used in a debate episode"""
    if episode_id not in episodes:
        raise HTTPException(status_code=404, detail=f"Episode {episode_id} not found")
    
    # Collect all evidence bundles used across turns
    evidence_bundles = []
    for turn in episodes[episode_id].turns:
        if hasattr(turn, 'evidence_bundle') and turn.evidence_bundle:
            evidence_bundles.append(turn.evidence_bundle)
    
    return evidence_bundles

@app.get("/episodes/{episode_id}/turns/{turn_id}/evidence")
async def get_turn_evidence(episode_id: str, turn_id: str):
    """Get the evidence bundle for a specific turn"""
    if episode_id not in episodes:
        raise HTTPException(status_code=404, detail=f"Episode {episode_id} not found")
    
    for turn in episodes[episode_id].turns:
        if turn.turn_id == turn_id:
            if hasattr(turn, 'evidence_bundle') and turn.evidence_bundle:
                return turn.evidence_bundle
            else:
                raise HTTPException(status_code=404, detail="No evidence bundle for this turn")
    
    raise HTTPException(status_code=404, detail=f"Turn {turn_id} not found")

@app.get("/episodes/{episode_id}/sources", response_model=List[Source])
async def get_episode_sources(episode_id: str):
    """Get all sources used in a debate episode"""
    if episode_id not in episodes:
        raise HTTPException(status_code=404, detail=f"Episode {episode_id} not found")
    
    # Collect all sources used across turns
    sources = {}  # Using dict to deduplicate sources by URL
    for turn in episodes[episode_id].turns:
        if hasattr(turn, 'citations'):
            for citation in turn.citations:
                if citation.source_id:
                    source = get_source_by_id(citation.source_id)
                    if source:
                        sources[source.url] = source
    
    return list(sources.values())

# Helper function to get source by ID
def get_source_by_id(source_id: str) -> Optional[Source]:
    """Retrieve a source by its ID from the database"""
    # This should be implemented to fetch from your actual source storage
    # For now, return a stub example
    if source_id.startswith("S"):
        return Source(
            source_id=source_id,
            url=f"https://example.com/source/{source_id}",
            title="Example Source",
            author="Author Name",
            published_date=datetime.now() - timedelta(days=random.randint(1, 30)),
            domain="example.com",
            trust_score=0.8,
            content_snippet="This is a snippet of the source content...",
        )
    return None

# Citation resolution endpoint
@app.get("/citations/{citation_id}")
async def get_citation(citation_id: str):
    """Get details for a specific citation"""
    # This should be implemented to fetch the actual citation from your storage
    source_id = citation_id.split("-")[0] if "-" in citation_id else None
    
    if not source_id:
        raise HTTPException(status_code=404, detail=f"Invalid citation ID format")
    
    source = get_source_by_id(source_id)
    if not source:
        raise HTTPException(status_code=404, detail=f"Source not found")
    
    return {
        "citation_id": citation_id,
        "source": source.dict(),
        "snippet": "This is the specific text snippet being cited...",
        "context": "This is additional context around the citation..."
    }

@app.post("/episodes/{episode_id}/control")
async def control_episode(
    episode_id: str, 
    action: str = Query(..., enum=["pause", "resume", "next_turn", "cancel"])
):
    """Control episode execution (pause, resume, next turn, cancel)"""
    if episode_id not in episodes:
        raise HTTPException(status_code=404, detail="Episode not found")
    
    episode_status = episode_statuses[episode_id]
    
    if action == "pause" and episode_status.status == "running":
        episode_status.status = "paused"
        episode_status.message = "Episode paused"
        await broadcast_to_websockets(episode_id, {
            "type": "status_update",
            "episode_status": episode_status.dict()
        })
        return {"status": "paused", "message": "Episode paused successfully"}
    
    elif action == "resume" and episode_status.status == "paused":
        episode_status.status = "running"
        episode_status.message = f"Episode resumed at {episode_status.current_phase} phase"
        await broadcast_to_websockets(episode_id, {
            "type": "status_update",
            "episode_status": episode_status.dict()
        })
        return {"status": "running", "message": "Episode resumed successfully"}
    
    elif action == "next_turn" and episode_status.status in ["running", "paused"]:
        # Signal to advance to next turn (implementation depends on your runner)
        episode_status.message = "Advancing to next turn..."
        await broadcast_to_websockets(episode_id, {
            "type": "manual_advance",
            "episode_status": episode_status.dict()
        })
        return {"status": "advancing", "message": "Advancing to next turn"}
    
    elif action == "cancel":
        episode_status.status = "cancelled"
        episode_status.message = "Episode cancelled by user"
        await broadcast_to_websockets(episode_id, {
            "type": "status_update",
            "episode_status": episode_status.dict()
        })
        return {"status": "cancelled", "message": "Episode cancelled successfully"}
    
    else:
        raise HTTPException(status_code=400, detail=f"Invalid action '{action}' for current status '{episode_status.status}'")

# Interactive Control Endpoints
@app.post("/episodes/{episode_id}/pause")
async def pause_episode(episode_id: str):
    """Pause an ongoing episode"""
    if episode_id not in episodes:
        raise HTTPException(status_code=404, detail=f"Episode {episode_id} not found")
    
    episode = episodes[episode_id]
    if episode.status == "completed":
        raise HTTPException(status_code=400, detail="Episode already completed")
    
    # Store previous status to allow proper resuming
    episode.previous_status = episode.status
    episode.status = "paused"
    
    # Broadcast status update
    await broadcast_status_update(episode_id)
    
    return {"message": f"Episode {episode_id} paused", "status": "paused"}

@app.post("/episodes/{episode_id}/resume")
async def resume_episode(episode_id: str):
    """Resume a paused episode"""
    if episode_id not in episodes:
        raise HTTPException(status_code=404, detail=f"Episode {episode_id} not found")
    
    episode = episodes[episode_id]
    if episode.status != "paused":
        raise HTTPException(status_code=400, detail="Episode is not paused")
    
    # Restore previous status
    episode.status = episode.previous_status or "in_progress"
    episode.previous_status = None
    
    # Broadcast status update
    await broadcast_status_update(episode_id)
    
    return {"message": f"Episode {episode_id} resumed", "status": episode.status}

@app.post("/episodes/{episode_id}/next_turn")
async def trigger_next_turn(
    episode_id: str, 
    request: Optional[Dict] = Body(None)
):
    """Manually trigger the next turn in an episode"""
    if episode_id not in episodes:
        raise HTTPException(status_code=404, detail=f"Episode {episode_id} not found")
    
    episode = episodes[episode_id]
    if episode.status == "completed":
        raise HTTPException(status_code=400, detail="Episode already completed")
    
    # Optional override parameters from request body
    override_avatar = request.get("avatar_id") if request else None
    override_prompt = request.get("prompt") if request else None
    
    # Update status
    episode.status = "generating_turn"
    await broadcast_status_update(episode_id)
    
    # In a real implementation, this would trigger the orchestrator to generate the next turn
    # For demo purposes, we'll create a placeholder turn and broadcast after a delay
    turn_id = f"turn_{len(episode.turns) + 1}"
    avatar_id = override_avatar or (
        "nova" if len(episode.turns) % 2 == 0 else "alex"
    )
    
    # Create a background task to simulate turn generation
    background_tasks.add_task(
        generate_turn_background,
        episode_id=episode_id,
        turn_id=turn_id,
        avatar_id=avatar_id,
        override_prompt=override_prompt
    )
    
    return {"message": "Generating next turn", "turn_id": turn_id}

async def generate_turn_background(
    episode_id: str,
    turn_id: str,
    avatar_id: str,
    override_prompt: Optional[str] = None
):
    """Background task to simulate turn generation"""
    # Simulate processing time
    await asyncio.sleep(2)
    
    # Create the new turn
    episode = episodes[episode_id]
    
    # Get the avatar details
    avatar = next((a for a in avatars if a.avatar_id == avatar_id), None)
    if not avatar:
        logger.error(f"Avatar {avatar_id} not found")
        return
    
    # Generate a simulated turn
    current_phase = "opening" if len(episode.turns) < 2 else (
        "crossfire" if len(episode.turns) < 6 else "closing"
    )
    
    # Use override_prompt if provided, otherwise generate based on phase
    content = override_prompt or f"{avatar.name}'s {current_phase} statement on {episode.topic}"
    
    # Create rich text with emotion annotations
    rich_text = [
        RichTextSegment(
            text=content,
            emotion="neutral",
            emphasis=1.0
        )
    ]
    
    # Create evidence bundle and citations for this turn
    evidence_bundle = EvidenceBundle(
        bundle_id=f"bundle_{turn_id}",
        claims=[
            Claim(
                claim_id=f"claim_{turn_id}_1",
                text="This is a sample claim",
                confidence=0.85,
                citations=["S1-1", "S2-3"]
            )
        ],
        sources=[
            Source(
                source_id="S1",
                url="https://example.com/source1",
                title="Example Source 1",
                author="Author Name",
                published_date=datetime.now() - timedelta(days=5),
                domain="example.com",
                trust_score=0.8
            )
        ]
    )
    
    # Create the new turn
    new_turn = Turn(
        turn_id=turn_id,
        avatar_id=avatar_id,
        phase=current_phase,
        text=content,
        rich_text=rich_text,
        audio_path=f"/files/audio/placeholder_{avatar_id}.mp3",
        evidence_bundle=evidence_bundle,
        citations=[
            Citation(
                citation_id="S1-1",
                source_id="S1",
                text="Citation text",
                context="Citation context"
            )
        ],
        created_at=datetime.now(),
        duration_seconds=15
    )
    
    # Add turn to episode
    episode.turns.append(new_turn)
    
    # Update episode status
    if len(episode.turns) >= 8:  # Assuming 8 turns for a complete debate
        episode.status = "completed"
    else:
        episode.status = "waiting_for_input" if episode.manual_mode else "in_progress"
    
    # Broadcast the new turn and status update
    await broadcast_turn_update(episode_id, new_turn)
    await broadcast_status_update(episode_id)
