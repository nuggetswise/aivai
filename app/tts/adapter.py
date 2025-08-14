from typing import Optional, Dict, Any
import os
import logging
from pathlib import Path
import requests

from app.config import settings
from app.models import Persona, TTSInput, TTSOutput

logger = logging.getLogger(__name__)

class ElevenLabsTTSEngine:
    """ElevenLabs TTS engine using REST API."""

    def __init__(self):
        self.output_dir = Path(settings.DATA_DIR) / "audio" / "turns"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.api_key = os.environ.get("ELEVEN_API_KEY", "")
        if not self.api_key:
            raise RuntimeError("ELEVEN_API_KEY is not set")
        # Optional default voice id
        self.default_voice_id = os.environ.get("ELEVEN_VOICE_ID", "")

    def synthesize(self, tts_input: TTSInput) -> TTSOutput:
        try:
            voice_id = self._resolve_voice_id(tts_input.persona)
            if not voice_id:
                raise RuntimeError("No ElevenLabs voice id configured for persona and ELEVEN_VOICE_ID not set")

            filename = self._generate_audio_filename(tts_input.persona)
            out_path = self.output_dir / filename

            url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
            headers = {
                "xi-api-key": self.api_key,
                "accept": "audio/mpeg",
                "content-type": "application/json",
            }
            clean_text, references = self._split_text_and_references(tts_input.styled_text)
            payload = {
                "text": clean_text,
                "model_id": "eleven_monolingual_v1",
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.5,
                    "style": 0.5,
                    "use_speaker_boost": True,
                },
            }

            resp = requests.post(url, headers=headers, json=payload, timeout=60)
            if resp.status_code != 200:
                raise RuntimeError(f"ElevenLabs API error: {resp.status_code} {resp.text[:200]}")

            with open(out_path, "wb") as f:
                f.write(resp.content)

            duration = self._estimate_duration(clean_text)
            return TTSOutput(audio_path=str(out_path), duration_seconds=duration, references=references)
        except Exception as e:
            logger.error(f"ElevenLabs TTS failed: {e}")
            return TTSOutput(audio_path="", duration_seconds=0.0)

    def _resolve_voice_id(self, persona: Persona) -> Optional[str]:
        # Try config/tts_voices.yaml mapping by persona name
        try:
            import yaml
            repo_root = Path(__file__).resolve().parents[2]
            cfg = repo_root / "config" / "tts_voices.yaml"
            if cfg.exists():
                with open(cfg, "r") as f:
                    data = yaml.safe_load(f) or {}
                voices = data.get("voices", {})
                entry = voices.get(persona.name) or voices.get(persona.name.strip())
                if isinstance(entry, dict):
                    # Support nested provider structure
                    return entry.get("voiceId") or entry.get("voice_id")
                if isinstance(entry, str):
                    return entry
        except Exception:
            pass
        # Fallback to env voice id
        return self.default_voice_id or None

    def _split_text_and_references(self, text: str) -> tuple[str, list]:
        import re
        refs = re.findall(r"\[(S\d+|L\d+|R\d+)\]", text)
        clean_text = re.sub(r"\[(?:S\d+|L\d+|R\d+)\]", "", text)
        clean_text = re.sub(r"\s+", " ", clean_text).strip()
        return clean_text, refs

    def _generate_audio_filename(self, persona: Persona) -> str:
        import uuid
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        safe_name = persona.name.lower().replace(" ", "_").replace("-", "_")
        return f"{safe_name}_{timestamp}_{unique_id}.mp3"

class TTSAdapter:
    """
    TTS adapter that supports multiple TTS vendors and provides a unified interface.
    Currently supports Dia TTS with extensibility for other providers.
    """
    
    def __init__(self):
        self.engines: Dict[str, Any] = {}
        # Prefer ElevenLabs when available
        if os.environ.get("ELEVEN_API_KEY"):
            try:
                self.engines["elevenlabs"] = ElevenLabsTTSEngine()
            except Exception as e:
                logger.warning(f"ElevenLabs init failed: {e}")
        # Always provide mock as fallback
        self.engines["mock"] = MockTTSEngine()

        # Default engine: 'elevenlabs' if present else 'mock'
        self.default_engine = "elevenlabs" if "elevenlabs" in self.engines else "mock"
    
    def synthesize(self, tts_input: TTSInput) -> TTSOutput:
        """
        Synthesize speech using the appropriate TTS engine.
        
        Args:
            tts_input: TTSInput with styled_text and persona
            
        Returns:
            TTSOutput with audio_path and duration
        """
        # Determine which engine to use
        # Ignore persona vendor; pick available engine based on environment
        engine = self.engines.get(self.default_engine) or self.engines.get("mock")
        return engine.synthesize(tts_input)
    
    def get_available_voices(self, vendor: str = None) -> Dict[str, Any]:
        """Get available voices for a TTS vendor"""
        vendor = vendor or self.default_engine
        
        # For Dia TTS, voices are generated dynamically
        return {
            'dia': {
                'speakers': ['S1', 'S2'],  # Dia uses these speaker tags
                'nonverbals': [
                    '(laughs)', '(clears throat)', '(sighs)', '(gasps)', 
                    '(coughs)', '(chuckle)', '(whistles)', '(applause)'
                ],
                'voice_cloning': True,
                'max_duration_seconds': 20,  # Per README guidelines
                'min_duration_seconds': 5
            }
        }
    
    def validate_voice_settings(self, voice_settings: Any) -> bool:
        """Validate voice settings for TTS synthesis"""
        # Dia doesn't need traditional voice settings validation
        # It works with speaker tags and reference audio
        return True


class MockTTSEngine:
    """Simple mock TTS engine that generates a short silent WAV file."""
    def __init__(self):
        self.output_dir = Path(settings.DATA_DIR) / "audio" / "turns"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def synthesize(self, tts_input: TTSInput) -> TTSOutput:
        try:
            filename = self._generate_audio_filename(tts_input.persona)
            out_path = self.output_dir / filename
            self._write_silence_wav(out_path, duration_seconds=self._estimate_duration(tts_input.styled_text))
            return TTSOutput(audio_path=str(out_path), duration_seconds=self._estimate_duration(tts_input.styled_text))
        except Exception as e:
            logger.error(f"Mock TTS failed: {e}")
            return TTSOutput(audio_path="", duration_seconds=0.0)

    def _write_silence_wav(self, path: Path, duration_seconds: float, sample_rate: int = 16000):
        import wave
        import struct
        num_samples = int(duration_seconds * sample_rate)
        with wave.open(str(path), 'w') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample_rate)
            silence = struct.pack('<h', 0)
            for _ in range(num_samples):
                wf.writeframes(silence)

    def _generate_audio_filename(self, persona: Persona) -> str:
        import uuid
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        safe_name = persona.name.lower().replace(" ", "_").replace("-", "_")
        return f"{safe_name}_{timestamp}_{unique_id}.wav"

    def _estimate_duration(self, text: str) -> float:
        words = max(1, len(text.split()))
        return max(1.0, (words / 150.0) * 60.0)

# Singleton instance
_tts_adapter = None

def get_tts_adapter() -> TTSAdapter:
    """Get singleton TTS adapter instance"""
    global _tts_adapter
    if _tts_adapter is None:
        _tts_adapter = TTSAdapter()
    return _tts_adapter
