from typing import List, Dict, Any, Optional, AsyncGenerator
import asyncio
import logging
from datetime import datetime
from pathlib import Path
import sys

from app.models import (
    Episode, EpisodeConfig, Avatar, Turn, EpisodePhase, TurnIntent,
    ResearcherInput, CommentatorInput, VerifierInput, StyleInput, TTSInput,
    EpisodeAssets
)
from app.config import settings
from app.agents.researcher import get_researcher
from app.agents.commentator import get_commentator
from app.agents.verifier import get_verifier
from app.agents.style import get_style_agent
from app.tts.adapter import get_tts_adapter

logger = logging.getLogger(__name__)

class EpisodeRunner:
    """
    Orchestrates complete debate episodes by coordinating all agents
    through the defined phases: opening → positions → crossfire → closing
    """
    
    def __init__(self):
        self.researcher = get_researcher()
        self.commentator = get_commentator()
        self.verifier = get_verifier()
        self.style_agent = get_style_agent()
        self.tts_adapter = get_tts_adapter()
    
    async def run_episode(self, episode_config: EpisodeConfig) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Run a complete debate episode with streaming progress updates.
        
        Args:
            episode_config: EpisodeConfig with topic, avatars, and settings
            
        Yields:
            Progress events for real-time updates
        """
        logger.info(f"Starting episode: {episode_config.topic}")
        
        try:
            # Step 1: Load avatars and create episode
            avatar_a = self._load_avatar(episode_config.avatar_a_path)
            avatar_b = self._load_avatar(episode_config.avatar_b_path)
            
            episode = Episode(
                config=episode_config,
                avatar_a=avatar_a,
                avatar_b=avatar_b,
                current_phase=EpisodePhase.PRE_RESEARCH
            )
            
            yield {"event": "episode_started", "data": {"episode_id": episode.id, "topic": episode_config.topic}}
            
            # Step 2: Pre-research phase
            yield {"event": "phase_started", "data": {"phase": "pre_research"}}
            research_bundles = await self._conduct_pre_research(episode_config.topic)
            episode.bundles = research_bundles
            yield {"event": "research_complete", "data": {"bundle_count": len(research_bundles)}}
            
            # Step 3: Run debate phases
            phases = [
                (EpisodePhase.OPENING, TurnIntent.OPENING, 1),
                (EpisodePhase.POSITIONS, TurnIntent.POSITIONING, episode_config.max_turns_per_phase),
                (EpisodePhase.CROSSFIRE, TurnIntent.REBUTTAL, episode_config.max_turns_per_phase), 
                (EpisodePhase.CLOSING, TurnIntent.CLOSING, 1)
            ]
            
            for phase, intent, max_turns in phases:
                episode.current_phase = phase
                yield {"event": "phase_started", "data": {"phase": phase.value}}
                
                async for turn_event in self._run_phase(episode, phase, intent, max_turns):
                    yield turn_event
            
            # Step 4: Finalize episode
            episode.current_phase = EpisodePhase.COMPLETE
            episode.completed_at = datetime.utcnow()
            
            # Generate final artifacts
            transcript_artifacts = await self._generate_transcript(episode)
            transcript_path = transcript_artifacts.get("transcript_path") if isinstance(transcript_artifacts, dict) else str(transcript_artifacts)
            media_assets = transcript_artifacts.get("assets") if isinstance(transcript_artifacts, dict) else None
            episode.transcript_path = transcript_path
            
            # Persist assets on episode if available
            if media_assets:
                episode.assets = EpisodeAssets(
                    mp3=media_assets.get("audio"),
                    srt=media_assets.get("srt"),
                    vtt=media_assets.get("vtt"),
                    videos=media_assets.get("videos", []),
                    clips=[]
                )
            
            # Persist assets if present
            if media_assets:
                try:
                    from app.models import EpisodeAssets
                    episode.assets = EpisodeAssets(
                        mp3=media_assets.get("audio"),
                        srt=media_assets.get("srt"),
                        vtt=media_assets.get("vtt"),
                        videos=media_assets.get("videos", []),
                        clips=media_assets.get("clips", []),
                    )
                except Exception:
                    pass

            yield {"event": "episode_complete", "data": {
                "episode_id": episode.id,
                "turn_count": episode.turn_count,
                "duration_minutes": episode.duration_minutes,
                "transcript_path": transcript_path,
                "media_assets": media_assets
            }}
            
        except Exception as e:
            logger.error(f"Episode failed: {e}")
            yield {"event": "episode_error", "data": {"error": str(e)}}
    
    async def _conduct_pre_research(self, topic: str) -> Dict[str, Any]:
        """Conduct initial research for both sides of the debate"""
        logger.info("Conducting pre-research")
        
        # Research queries for different perspectives
        research_queries = [
            f"{topic} benefits advantages",
            f"{topic} risks concerns criticism", 
            f"{topic} evidence studies research",
            f"{topic} expert opinions analysis"
        ]
        
        bundles = {}
        
        for i, query in enumerate(research_queries):
            research_input = ResearcherInput(
                topic=query,
                intent=TurnIntent.EVIDENCE_HARVEST,
                opponent_point="",
                local_corpus=[]
            )
            
            bundle = self.researcher.research(research_input)
            bundles[f"research_{i+1}"] = bundle
        
        return bundles
    
    async def _run_phase(self, episode: Episode, phase: EpisodePhase, intent: TurnIntent, max_turns: int) -> AsyncGenerator[Dict[str, Any], None]:
        """Run a single debate phase with alternating turns"""
        turn_count = 0
        current_avatar = "A1"  # Start with Avatar A
        
        while turn_count < max_turns:
            # Determine which avatar is speaking
            avatar = episode.avatar_a if current_avatar == "A1" else episode.avatar_b
            opponent_avatar = episode.avatar_b if current_avatar == "A1" else episode.avatar_a
            
            yield {"event": "turn_started", "data": {
                "avatar": current_avatar,
                "avatar_name": avatar.persona.name,
                "phase": phase.value,
                "turn_number": turn_count + 1
            }}
            
            # Generate turn
            turn = await self._generate_turn(
                episode=episode,
                avatar=avatar,
                avatar_key=current_avatar,
                phase=phase,
                intent=intent,
                opponent_summary=self._get_opponent_summary(episode, opponent_avatar.id)
            )
            
            episode.turns.append(turn)
            
            yield {"event": "turn_complete", "data": {
                "turn_id": turn.id,
                "avatar": current_avatar,
                "text_length": len(turn.text),
                "citation_count": len(turn.citations),
                "audio_path": turn.audio_path
            }}
            
            # Switch to other avatar
            current_avatar = "A2" if current_avatar == "A1" else "A1"
            turn_count += 1
    
    async def _generate_turn(self, episode: Episode, avatar: Avatar, avatar_key: str, 
                           phase: EpisodePhase, intent: TurnIntent, opponent_summary: str) -> Turn:
        """Generate a complete turn through the agent pipeline"""
        
        # Step 1: Research (if needed)
        if intent in [TurnIntent.POSITIONING, TurnIntent.REBUTTAL]:
            research_input = ResearcherInput(
                topic=episode.config.topic,
                intent=intent,
                opponent_point=opponent_summary,
                local_corpus=[]
            )
            research_output = self.researcher.research(research_input)
            evidence_bundle = episode.bundles.get("research_1", research_output)  # Use pre-research or new
        else:
            # Use pre-research for opening/closing
            evidence_bundle = list(episode.bundles.values())[0] if episode.bundles else None
        
        # Step 2: Generate initial response
        commentator_input = CommentatorInput(
            topic=episode.config.topic,
            phase=phase,
            intent=intent,
            opponent_summary=opponent_summary,
            persona=avatar.persona,
            evidence_bundle=evidence_bundle
        )
        commentator_output = self.commentator.generate_turn(commentator_input)
        
        # Step 3: Verify factual support
        if episode.config.enable_verification:
            verifier_input = VerifierInput(
                draft=commentator_output.text,
                evidence_bundle=evidence_bundle,
                persona=avatar.persona
            )
            verifier_output = self.verifier.verify_turn(verifier_input)
            verified_text = verifier_output.text
            final_citations = verifier_output.citations
        else:
            verified_text = commentator_output.text
            final_citations = commentator_output.citations
        
        # Step 4: Apply styling
        style_input = StyleInput(
            verified_text=verified_text,
            persona=avatar.persona
        )
        style_output = self.style_agent.apply_style(style_input)
        final_text = style_output.styled_text
        
        # Step 5: Generate audio (clean speech only; citations are stripped downstream too)
        tts_input = TTSInput(
            styled_text=final_text,
            persona=avatar.persona
        )
        tts_output = self.tts_adapter.synthesize(tts_input)
        
        # Step 6: Create turn object (populate a minimal beats[] for unified schema)
        beat_label = {
            TurnIntent.OPENING: "opening",
            TurnIntent.POSITIONING: "position",
            TurnIntent.REBUTTAL: "rebuttal",
            TurnIntent.CLOSING: "closing",
            TurnIntent.EVIDENCE_HARVEST: "evidence",
        }.get(intent, "segment")

        from app.models import Beat, BeatProsody
        beat_references = [c.id for c in final_citations] if final_citations else []
        beats = [
            Beat(
                label=beat_label,
                target_duration_s=None,
                text=final_text,
                emotion=None,
                prosody=BeatProsody(),
                references=beat_references,
            )
        ]

        turn = Turn(
            avatar_id=avatar.id,
            avatar_key=avatar_key,
            phase=phase,
            intent=intent,
            text=final_text,
            citations=final_citations,
            beats=beats,
            evidence_bundle_id=evidence_bundle.id if evidence_bundle else None,
            opponent_summary=opponent_summary,
            audio_path=tts_output.audio_path,
            generation_stats={
                "audio_duration": tts_output.duration_seconds,
                "citation_count": len(final_citations),
                "verification_enabled": episode.config.enable_verification
            }
        )
        
        return turn
    
    def _load_avatar(self, avatar_path: str) -> Avatar:
        """Load avatar from YAML file"""
        import yaml
        from app.models import persona_from_yaml_dict
        
        try:
            with open(avatar_path, 'r') as f:
                avatar_data = yaml.safe_load(f)
            
            persona = persona_from_yaml_dict(avatar_data)
            
            return Avatar(
                persona=persona,
                tags=[avatar_path.split('/')[-1].replace('.yaml', '')]
            )
            
        except Exception as e:
            logger.error(f"Failed to load avatar from {avatar_path}: {e}")
            raise
    
    def _get_opponent_summary(self, episode: Episode, opponent_avatar_id: str) -> str:
        """Get summary of opponent's recent turns"""
        opponent_turns = [
            turn for turn in episode.turns[-3:]  # Last 3 turns
            if turn.avatar_id == opponent_avatar_id
        ]
        
        if not opponent_turns:
            return ""
        
        # Simple summary: combine recent turn texts
        recent_points = [turn.text[:100] + "..." for turn in opponent_turns]
        return " ".join(recent_points)
    
    async def _generate_transcript(self, episode: Episode) -> Dict[str, Any]:
        """Generate and save episode transcript JSON and a scriptfix-formatted Markdown file.

        Returns a dict with keys: transcript_path, markdown_path, assets (optional)
        """
        from app.models import episode_to_transcript_format
        import json
        import subprocess
        
        transcript_data = {
            "episode_id": episode.id,
            "topic": episode.config.topic,
            "avatars": {
                "A1": episode.avatar_a.persona.name,
                "A2": episode.avatar_b.persona.name
            },
            "started_at": episode.started_at.isoformat(),
            "completed_at": episode.completed_at.isoformat() if episode.completed_at else None,
            "turns": episode_to_transcript_format(episode)
        }
        
        # Save transcript
        transcript_dir = Path(settings.DATA_DIR) / "transcripts"
        transcript_dir.mkdir(parents=True, exist_ok=True)
        
        transcript_path = transcript_dir / f"{episode.id}_transcript.json"
        
        with open(transcript_path, 'w') as f:
            json.dump(transcript_data, f, indent=2)
        
        logger.info(f"Transcript saved: {transcript_path}")
        
        # Also generate scriptfix-compliant Markdown alongside JSON
        assets = None
        try:
            md_output = transcript_dir / f"{episode.id}_transcript.md"
            repo_root = Path(__file__).resolve().parents[2]
            script_path = repo_root / "scripts" / "generate_transcript.py"
            subprocess.run([
                sys.executable, str(script_path),
                "--input", str(transcript_path),
                "--output", str(md_output),
                "--tighten"
            ], check=True)
            logger.info(f"Markdown transcript saved: {md_output}")

            # Build media assets (audio, captions, videos)
            media_builder = repo_root / "md_podcast_build.py"
            if media_builder.exists():
                slug = f"{episode.id}"
                subprocess.run([
                    sys.executable, str(media_builder),
                    "--md", str(md_output),
                    "--slug", slug,
                    "--voices", str(repo_root / "config" / "tts_voices.yaml")
                ], check=True)
                logger.info("Media assets built successfully")
                # Optionally, collect and log asset paths
                output_dir = repo_root / "output"
                assets = {
                    "audio": str((output_dir / f"{slug}.mp3").resolve()),
                    "srt": str((output_dir / f"{slug}.srt").resolve()),
                    "vtt": str((output_dir / f"{slug}.vtt").resolve()),
                    "videos": [
                        str((output_dir / f"{slug}_vertical.mp4").resolve()),
                        str((output_dir / f"{slug}_square.mp4").resolve()),
                        str((output_dir / f"{slug}_horizontal.mp4").resolve()),
                    ],
                }
                logger.info(f"Assets: {assets}")

                # Auto-generate 3 clips if clip_maker is present
                clip_maker = repo_root / "clip_maker.py"
                if clip_maker.exists():
                    try:
                        subprocess.run([
                            sys.executable, str(clip_maker),
                            "--audio", str(output_dir / f"{slug}.mp3"),
                            "--srt", str(output_dir / f"{slug}.srt"),
                            "--outdir", str(output_dir / "clips" / slug),
                            "--n", "3"
                        ], check=True)
                        logger.info("Clips generated successfully")
                    except Exception as ce:
                        logger.warning(f"Clip generation failed: {ce}")
            else:
                logger.warning("Media builder not found; skipping A/V rendering")
        except Exception as e:
            logger.error(f"Failed to generate Markdown and/or media assets: {e}")

        # Return artifact info
        result: Dict[str, Any] = {
            "transcript_path": str(transcript_path),
            "markdown_path": str(md_output),
        }
        if assets:
            logger.info(f"Episode assets ready for {episode.id}: {assets}")
            result["assets"] = assets
        return result

# Singleton instance
_episode_runner = None

def get_episode_runner() -> EpisodeRunner:
    """Get singleton episode runner instance"""
    global _episode_runner
    if _episode_runner is None:
        _episode_runner = EpisodeRunner()
    return _episode_runner
