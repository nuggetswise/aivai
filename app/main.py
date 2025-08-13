from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import asyncio
import json
import uuid
import os
from pathlib import Path
import yaml
from datetime import datetime

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

@app.websocket("/ws/{episode_id}")
async def websocket_endpoint(websocket: WebSocket, episode_id: str):
    """WebSocket endpoint for real-time updates"""
    await websocket.accept()
    
    if episode_id not in active_websockets:
        active_websockets[episode_id] = []
    
    active_websockets[episode_id].append(websocket)
    
    try:
        # Send current status
        if episode_id in episode_statuses:
            await websocket.send_text(json.dumps({
                "type": "status_update",
                "episode_status": episode_statuses[episode_id].dict()
            }))
        
        # Keep connection alive
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
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

async def broadcast_to_websockets(episode_id: str, message: dict):
    """Broadcast message to all WebSocket connections for an episode"""
    if episode_id in active_websockets:
        disconnected = []
        for websocket in active_websockets[episode_id]:
            try:
                await websocket.send_text(json.dumps(message))
            except:
                disconnected.append(websocket)
        
        # Remove disconnected websockets
        for ws in disconnected:
            active_websockets[episode_id].remove(ws)

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
