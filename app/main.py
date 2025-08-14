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
import sys
import subprocess
import shutil
import yaml
from datetime import datetime, timedelta
import random

from app.orchestrator.episode_runner import EpisodeRunner
from app.models import EpisodeConfig, Avatar as AvatarModel
from app.tts.adapter import get_tts_adapter
from app.models import Persona, TTSInput
from app.deps import get_search_client, get_embeddings_client, get_llm_router

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
# Serve generated media assets (output directory)
app.mount("/media", StaticFiles(directory="output"), name="media")

# Convenience: serve any file under output via /files/output/{path}
@app.get("/files/output/{path:path}")
async def get_output_file(path: str):
    file_path = Path("output") / path
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="Output file not found")
    # Best-effort content type based on suffix
    suffix = file_path.suffix.lower()
    media_type = "application/octet-stream"
    if suffix == ".mp3":
        media_type = "audio/mpeg"
    elif suffix == ".wav":
        media_type = "audio/wav"
    elif suffix == ".mp4":
        media_type = "video/mp4"
    elif suffix == ".vtt":
        media_type = "text/vtt"
    elif suffix == ".srt":
        media_type = "text/plain"
    return FileResponse(file_path, media_type=media_type)


@app.get("/health")
async def health():
    ffmpeg_ok = shutil.which("ffmpeg") is not None
    env = {
        "ELEVEN_API_KEY": bool(os.environ.get("ELEVEN_API_KEY")),
        "GEMINI_API_KEY": bool(os.environ.get("GEMINI_API_KEY")),
        "TAVILY_API_KEY": bool(os.environ.get("TAVILY_API_KEY")),
    }
    return {"ok": True, "ffmpeg": ffmpeg_ok, "env": env}

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

class ResearchRequestBody(BaseModel):
    topic: str
    perspective: Optional[str] = None
    max_results: int = 6

class KnowledgeProcessBody(BaseModel):
    avatarId: str
    topic: str
    researchResults: List[Dict[str, Any]]

class KnowledgeSearchBody(BaseModel):
    avatarId: str
    query: str
    maxResults: int = 5

class DebateTurnRequest(BaseModel):
    topic: str
    avatars: List[Dict[str, Any]]
    avatarId: str
    phase: Optional[str] = "main"
    messages: List[Dict[str, Any]] = []
    opponentLastMessage: Optional[Dict[str, Any]] = None

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

class Source(BaseModel):
    source_id: str
    url: str
    title: str
    author: Optional[str] = None
    published_date: Optional[datetime] = None
    domain: Optional[str] = None
    trust_score: Optional[float] = None
    content_snippet: Optional[str] = None

class Claim(BaseModel):
    claim_id: str
    text: str
    confidence: float
    citations: List[str] = []

class RichTextSegment(BaseModel):
    text: str
    emotion: str = "neutral"
    emphasis: float = 1.0

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

# In-memory stores
knowledge_store: Dict[str, Dict[str, Any]] = {}
idempotency_cache: Dict[str, Any] = {}
task_store: Dict[str, Dict[str, Any]] = {}

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


@app.post("/research")
async def research_endpoint(body: ResearchRequestBody):
    try:
        sc = get_search_client()
        results = sc.search(body.topic, max_results=body.max_results, include_raw_content=True, include_answer=True)
        if body.perspective:
            key = "benefit" if body.perspective == "pro" else "risk"
            results = [r for r in results if key in (r.get("content") or r.get("raw_content") or "").lower()] or results
        def extract_points(text: str) -> List[str]:
            text = (text or "")[:800]
            sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 20]
            return sentences[:5]
        keyPoints: List[str] = []
        for r in results[:3]:
            keyPoints += extract_points(r.get("content") or r.get("raw_content") or "")
        payload = {
            "success": True,
            "data": {
                "keyPoints": keyPoints[:5],
                "sources": [
                    {
                        "title": r.get("title", ""),
                        "content": r.get("content", r.get("raw_content", "")),
                        "url": r.get("url", ""),
                        "score": r.get("score", 0.7),
                    } for r in results
                ],
                "perspective": body.perspective or "neutral",
            }
        }
        return payload
    except Exception as e:
        return {"success": False, "error": {"code": "RESEARCH_ERROR", "message": str(e), "details": {}}}


@app.post("/knowledge/process")
async def knowledge_process_endpoint(body: KnowledgeProcessBody):
    try:
        embedder = get_embeddings_client()
        chunks: List[Dict[str, Any]] = []
        def chunk_text(text: str, source: str, url: Optional[str]) -> List[Dict[str, Any]]:
            text = (text or "").strip()
            if not text:
                return []
            size = 500
            overlap = 50
            out: List[Dict[str, Any]] = []
            i = 0
            idx = 0
            while i < len(text) and idx < 20:
                j = min(len(text), i + size)
                k = text.rfind('.', i, j)
                if k > i + int(size * 0.6):
                    j = k + 1
                out.append({
                    "id": f"{source}-chunk-{idx}",
                    "content": text[i:j].strip(),
                    "metadata": {"source": source, "url": url, "timestamp": int(datetime.now().timestamp()*1000), "chunkIndex": idx}
                })
                i = max(j - overlap, i + 1)
                idx += 1
            return out
        for r in body.researchResults:
            src = r.get("title", "source")
            url = r.get("url")
            text = r.get("content") or r.get("raw_content") or ""
            chunks += chunk_text(text, src, url)
        embeddings = embedder.embed([c["content"] for c in chunks]) if chunks else []
        for c, emb in zip(chunks, embeddings):
            c["embedding"] = emb
        corpus = {
            "avatarId": body.avatarId,
            "topic": body.topic,
            "chunks": chunks,
            "summary": f"Knowledge corpus for {body.topic} with {len(chunks)} chunks.",
            "keyPoints": [c["content"][:80] for c in chunks[:5]],
            "createdAt": int(datetime.now().timestamp()*1000),
            "updatedAt": int(datetime.now().timestamp()*1000),
        }
        knowledge_store[body.avatarId] = corpus
        return {"success": True, "data": {"corpusId": body.avatarId, "chunksCount": len(chunks), "summary": corpus["summary"], "keyPoints": corpus["keyPoints"], "processedAt": corpus["createdAt"]}}
    except Exception as e:
        return {"success": False, "error": {"code": "KNOWLEDGE_PROCESS_ERROR", "message": str(e), "details": {}}}


@app.post("/knowledge/search")
async def knowledge_search_endpoint(body: KnowledgeSearchBody):
    try:
        corpus = knowledge_store.get(body.avatarId)
        if not corpus:
            return {"success": False, "error": {"code": "NOT_FOUND", "message": "Knowledge corpus not found", "details": {}}}
        embedder = get_embeddings_client()
        query_emb = embedder.embed([body.query])[0]
        def cos(a: List[float], b: List[float]) -> float:
            import math
            dot = sum(x*y for x, y in zip(a, b))
            na = math.sqrt(sum(x*x for x in a))
            nb = math.sqrt(sum(y*y for y in b))
            return (dot / (na*nb)) if na and nb else 0.0
        ranked: List[Dict[str, Any]] = []
        for c in corpus["chunks"]:
            sim = cos(query_emb, c.get("embedding", [])) if c.get("embedding") else 0.0
            ranked.append({
                "id": c["id"],
                "content": c["content"],
                "source": c["metadata"]["source"],
                "url": c["metadata"].get("url"),
                "similarity": sim,
            })
        ranked.sort(key=lambda x: x["similarity"], reverse=True)
        ranked = ranked[: body.maxResults]
        return {"success": True, "data": {"query": body.query, "results": ranked, "totalResults": len(ranked)}}
    except Exception as e:
        return {"success": False, "error": {"code": "KNOWLEDGE_SEARCH_ERROR", "message": str(e), "details": {}}}


@app.post("/debate/generate")
async def debate_generate_endpoint(body: DebateTurnRequest):
    try:
        avatar = next((a for a in body.avatars if a.get("id") == body.avatarId), None)
        if not avatar:
            raise HTTPException(status_code=404, detail="Avatar not found")
        stance = avatar.get("stance", "pro")
        knowledge = avatar.get("knowledge", [])
        stance_desc = "supporting" if stance == "pro" else "opposing"
        knowledge_context = ". ".join(knowledge[:5])
        system = f"You are {avatar.get('name','Debater')}, an AI avatar {stance_desc} the topic \"{body.topic}\". Keep responses concise (2-3 sentences)."
        context = f"Topic: {body.topic}\nYour stance: {stance.upper()}\nYour knowledge base includes: {knowledge_context}"
        if body.phase == "opening":
            user = f"Give your opening statement on \"{body.topic}\". Introduce your position and key arguments."
        elif body.phase == "closing":
            opp = " ".join(m.get("content","") for m in body.messages if m.get("avatarId") != body.avatarId)[-200:]
            user = f"Give your closing statement. Summarize your position and address the main points raised by your opponent. Opponent's main arguments: {opp}"
        else:
            if body.opponentLastMessage and body.opponentLastMessage.get("content"):
                user = f"Respond to your opponent's argument: \"{body.opponentLastMessage['content']}\". Present your counterargument using your knowledge."
            else:
                user = f"Present your main argument about \"{body.topic}\" using evidence from your knowledge base."
        llm = get_llm_router().get_basic_llm()
        content = llm.generate([
            {"role": "system", "content": system},
            {"role": "user", "content": f"{context}\n\n{user}"},
        ], temperature=0.7, max_tokens=200)
        return {"success": True, "data": {"content": content, "avatarId": body.avatarId, "timestamp": int(datetime.now().timestamp()*1000)}}
    except HTTPException as he:
        raise he
    except Exception as e:
        return {"success": False, "error": {"code": "DEBATE_ERROR", "message": str(e), "details": {}}}


class RenderRequest(BaseModel):
    captions: Optional[bool] = True
    videos: Optional[List[str]] = ["vertical", "square", "horizontal"]
    font: Optional[str] = "Arial"
    bg: Optional[str] = "black"
    waveform_ratio: Optional[float] = 0.35


@app.post("/episodes/{episode_id}/render")
async def render_episode_assets(episode_id: str, request: RenderRequest):
    """Render audio, captions (SRT/VTT), and videos for a completed episode using md_podcast_build.py"""
    repo_root = Path(__file__).resolve().parents[1]
    transcripts_dir = repo_root / "data" / "transcripts"
    md_path = transcripts_dir / f"{episode_id}_transcript.md"
    builder = repo_root / "md_podcast_build.py"

    if not md_path.exists():
        raise HTTPException(status_code=404, detail=f"Transcript Markdown not found for episode {episode_id}")
    if not builder.exists():
        raise HTTPException(status_code=500, detail="Media builder script not found")

    # Invoke builder synchronously for now
    try:
        args = [
            sys.executable, str(builder),
            "--md", str(md_path),
            "--slug", episode_id,
        ]
        voices_yaml = repo_root / "config" / "tts_voices.yaml"
        if voices_yaml.exists():
            args += ["--voices", str(voices_yaml)]
        subprocess.run(args, check=True)
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Render failed: {e}")

    # Return discovered assets
    return await get_episode_assets(episode_id)


class ClipsRequest(BaseModel):
    n: int = 3
    minSec: float = 15.0
    maxSec: float = 40.0
    filterSpeaker: str = ""
    font: str = "Arial"
    bg: str = "black"
    wf_ratio: float = 0.35


@app.post("/episodes/{episode_id}/clips")
async def generate_clips(episode_id: str, request: ClipsRequest):
    """Generate short clips for the episode using clip_maker.py"""
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = repo_root / "output"
    audio = output_dir / f"{episode_id}.mp3"
    srt = output_dir / f"{episode_id}.srt"
    if not audio.exists() or not srt.exists():
        raise HTTPException(status_code=404, detail="Episode audio/SRT not found. Render assets first.")

    clip_maker = repo_root / "clip_maker.py"
    if not clip_maker.exists():
        raise HTTPException(status_code=500, detail="clip_maker.py not found")

    outdir = output_dir / "clips" / episode_id
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        subprocess.run([
            sys.executable, str(clip_maker),
            "--audio", str(audio),
            "--srt", str(srt),
            "--outdir", str(outdir),
            "--n", str(request.n),
            "--min_s", str(request.minSec),
            "--max_s", str(request.maxSec),
            "--filter_speaker", request.filterSpeaker,
            "--font", request.font,
            "--bg", request.bg,
            "--wf_ratio", str(request.wf_ratio),
        ], check=True)
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Clip generation failed: {e}")

    clips = [str(p.resolve()) for p in outdir.glob("*.mp4")]
    return {"episode_id": episode_id, "clips": sorted(clips)}


@app.get("/episodes/{episode_id}/clips")
async def list_clips(episode_id: str):
    repo_root = Path(__file__).resolve().parents[1]
    outdir = repo_root / "output" / "clips" / episode_id
    if not outdir.exists():
        return {"episode_id": episode_id, "clips": []}
    clips = [str(p.resolve()) for p in outdir.glob("*.mp4")]
    return {"episode_id": episode_id, "clips": sorted(clips)}


@app.get("/config/tts-voices")
async def get_tts_voices():
    """Return the TTS voice mapping config if present"""
    repo_root = Path(__file__).resolve().parents[1]
    cfg = repo_root / "config" / "tts_voices.yaml"
    if not cfg.exists():
        return {"voices": {}}
    with open(cfg, "r") as f:
        data = yaml.safe_load(f) or {}
    return data


@app.put("/config/tts-voices")
async def put_tts_voices(payload: Dict[str, Any]):
    """Update the TTS voice mapping config"""
    repo_root = Path(__file__).resolve().parents[1]
    cfg = repo_root / "config" / "tts_voices.yaml"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    # Very light validation
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Invalid payload")
    with open(cfg, "w") as f:
        yaml.safe_dump(payload, f, sort_keys=False)
    return {"ok": True}


class TTSSynthesizeRequest(BaseModel):
    text: str
    persona: Dict[str, Any]


@app.post("/tts/synthesize")
async def tts_synthesize(req: TTSSynthesizeRequest):
    """Server-side TTS synthesis using mock when ElevenLabs is not configured."""
    try:
        persona = Persona(**req.persona)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid persona: {e}")

    adapter = get_tts_adapter()
    if not os.environ.get("ELEVEN_API_KEY"):
        adapter.default_engine = "mock"
    output = adapter.synthesize(TTSInput(styled_text=req.text, persona=persona))
    # Fallback to mock if primary engine failed
    if (not output.audio_path) and hasattr(adapter, "engines") and "mock" in adapter.engines:
        output = adapter.engines["mock"].synthesize(TTSInput(styled_text=req.text, persona=persona))
    if not output.audio_path:
        raise HTTPException(status_code=500, detail="TTS synthesis failed")
    return {"audio_path": output.audio_path, "duration": output.duration_seconds}

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


@app.get("/episodes/{episode_id}/assets")
async def get_episode_assets(episode_id: str):
    """Return produced asset paths for the episode if available"""
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = repo_root / "output"
    slug = episode_id

    assets = {
        "mp3": str((output_dir / f"{slug}.mp3").resolve()),
        "srt": str((output_dir / f"{slug}.srt").resolve()),
        "vtt": str((output_dir / f"{slug}.vtt").resolve()),
        "videos": [
            str((output_dir / f"{slug}_vertical.mp4").resolve()),
            str((output_dir / f"{slug}_square.mp4").resolve()),
            str((output_dir / f"{slug}_horizontal.mp4").resolve()),
        ],
        "clips": [str(p.resolve()) for p in (output_dir / "clips" / slug).glob("*.mp4")] if (output_dir / "clips" / slug).exists() else []
    }

    # Filter to existing files only
    def exists(p: str) -> bool:
        try:
            return Path(p).exists()
        except Exception:
            return False

    assets["videos"] = [p for p in assets["videos"] if exists(p)]
    assets["clips"] = [p for p in assets["clips"] if exists(p)]
    for k in ["mp3", "srt", "vtt"]:
        if not exists(assets[k]):
            assets[k] = None

    return {"episode_id": episode_id, "assets": assets}

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
    await broadcast_to_websockets(episode_id, {"type": "status_update", "episode_status": episode_statuses[episode_id].dict()})
    
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
    await broadcast_to_websockets(episode_id, {"type": "status_update", "episode_status": episode_statuses[episode_id].dict()})
    
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
    await broadcast_to_websockets(episode_id, {"type": "status_update", "episode_status": episode_statuses[episode_id].dict()})
    
    # In a real implementation, this would trigger the orchestrator to generate the next turn
    # For demo purposes, we'll create a placeholder turn and broadcast after a delay
    turn_id = f"turn_{len(episode.turns) + 1}"
    avatar_id = override_avatar or (
        "nova" if len(episode.turns) % 2 == 0 else "alex"
    )
    
    # Create a background task to simulate turn generation
    asyncio.create_task(generate_turn_background(
        episode_id=episode_id,
        turn_id=turn_id,
        avatar_id=avatar_id,
        override_prompt=override_prompt
    ))
    
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
    avatar = {"avatar_id": avatar_id, "name": avatar_id.upper()}
    
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
    # In this demo scaffolding, simply broadcast over websockets if connected
    await broadcast_to_websockets(episode_id, {"type": "turn_generated", "turn": new_turn.dict()})
    await broadcast_to_websockets(episode_id, {"type": "status_update", "episode_status": episode_statuses[episode_id].dict()})
