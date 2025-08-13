from typing import Optional, Dict, Any
import os
import logging
from pathlib import Path
import torch
from transformers import AutoProcessor, DiaForConditionalGeneration

from app.config import settings
from app.models import Persona, TTSInput, TTSOutput

logger = logging.getLogger(__name__)

class DiaTTSEngine:
    """
    Dia TTS engine integration using the nari-labs/Dia model.
    Supports voice cloning with reference audio and speaker tags.
    """
    
    def __init__(self):
        self.model_checkpoint = "nari-labs/Dia-1.6B-0626"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.output_dir = Path(settings.DATA_DIR) / "audio" / "turns"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model and processor
        self.processor = None
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load Dia model and processor"""
        try:
            logger.info("Loading Dia TTS model...")
            self.processor = AutoProcessor.from_pretrained(self.model_checkpoint)
            self.model = DiaForConditionalGeneration.from_pretrained(self.model_checkpoint).to(self.device)
            logger.info(f"Dia TTS model loaded on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load Dia TTS model: {e}")
            raise
    
    def synthesize(self, tts_input: TTSInput) -> TTSOutput:
        """
        Synthesize speech from text using Dia TTS.
        
        Args:
            tts_input: TTSInput with styled_text and persona
            
        Returns:
            TTSOutput with audio_path and duration
        """
        logger.info(f"Synthesizing speech for {tts_input.persona.name}")
        
        try:
            # Step 1: Prepare text for Dia format
            dia_text = self._prepare_text_for_dia(tts_input.styled_text, tts_input.persona)
            
            # Step 2: Generate unique filename
            audio_filename = self._generate_audio_filename(tts_input.persona)
            audio_path = self.output_dir / audio_filename
            
            # Step 3: Generate audio with Dia
            success = self._generate_with_dia(dia_text, str(audio_path), tts_input.persona)
            
            if not success:
                raise Exception("Dia TTS synthesis failed")
            
            # Step 4: Get audio duration (estimate)
            duration = self._estimate_duration(dia_text)
            
            logger.info(f"TTS synthesis complete: {audio_path} ({duration:.2f}s)")
            
            return TTSOutput(
                audio_path=str(audio_path),
                duration_seconds=duration
            )
            
        except Exception as e:
            logger.error(f"TTS synthesis failed: {e}")
            return TTSOutput(
                audio_path="",
                duration_seconds=0.0
            )
    
    def _prepare_text_for_dia(self, text: str, persona: Persona) -> str:
        """
        Prepare text for Dia format with speaker tags and voice cloning.
        
        Dia expects:
        - Text to start with [S1] 
        - Alternating [S1] and [S2] tags
        - Voice cloning: transcript of reference audio + new text
        """
        import re
        
        # Remove citations for speech
        clean_text = re.sub(r'\[[^\]]+\]', '', text)
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        
        # Add speaker tag for single speaker
        speaker_tag = "[S1]"
        
        # Handle voice cloning with reference audio
        if hasattr(persona.voice, 'ref_audio') and persona.voice.ref_audio and os.path.exists(persona.voice.ref_audio):
            # For voice cloning, we need the transcript of the reference audio
            # This is a placeholder - in practice, you'd need the actual transcript
            ref_transcript = f"[S1] {persona.name} speaking in their natural voice."
            dia_text = f"{ref_transcript} {speaker_tag} {clean_text}"
        else:
            # No voice cloning, just use speaker tag
            dia_text = f"{speaker_tag} {clean_text}"
        
        # Add non-verbal cues based on persona
        dia_text = self._add_nonverbal_cues(dia_text, persona)
        
        # Ensure proper ending for audio quality (end with speaker tag)
        if not dia_text.strip().endswith(speaker_tag):
            dia_text += f" {speaker_tag}"
        
        return dia_text
    
    def _add_nonverbal_cues(self, text: str, persona: Persona) -> str:
        """Add appropriate non-verbal cues based on persona"""
        # Supported non-verbals from README
        # (laughs), (clears throat), (sighs), (gasps), (coughs), (singing), etc.
        
        # Add subtle non-verbal cues based on persona tone
        if persona.tone.lower() == 'passionate':
            # Add occasional emphasis for passionate tone
            if 'important' in text.lower() or 'crucial' in text.lower():
                text = text.replace('important', 'important (sighs)')
                text = text.replace('crucial', 'crucial (gasps)')
        
        elif persona.tone.lower() == 'thoughtful':
            # Add occasional pauses for thoughtful tone
            sentences = text.split('.')
            if len(sentences) > 2:
                # Add a throat clear in the middle
                mid_point = len(sentences) // 2
                sentences[mid_point] = sentences[mid_point] + ' (clears throat)'
                text = '.'.join(sentences)
        
        # Add personality-specific cues sparingly
        if hasattr(persona, 'speech_quirks') and persona.speech_quirks:
            for quirk in persona.speech_quirks:
                if 'humor' in quirk.lower() or 'laugh' in quirk.lower():
                    # Add occasional laughter
                    if '.' in text:
                        text = text.replace('.', ' (chuckle).', 1)  # Add one chuckle
                        break
        
        return text
    
    def _generate_with_dia(self, dia_text: str, output_path: str, persona: Persona) -> bool:
        """Generate audio using Dia model"""
        try:
            if self.processor is None or self.model is None:
                logger.error("Dia model not loaded")
                return False
            
            # Prepare inputs
            inputs = self.processor(text=[dia_text], padding=True, return_tensors="pt").to(self.device)
            
            # Generate audio with Dia parameters
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=3072,
                    guidance_scale=3.0,
                    temperature=1.8,
                    top_p=0.90,
                    top_k=45
                )
            
            # Decode and save audio
            decoded_outputs = self.processor.batch_decode(outputs)
            self.processor.save_audio(decoded_outputs, output_path)
            
            return os.path.exists(output_path)
            
        except Exception as e:
            logger.error(f"Dia generation failed: {e}")
            return False
    
    def _estimate_duration(self, dia_text: str) -> float:
        """
        Estimate audio duration based on text length.
        Dia README suggests: 1 second ≈ 86 tokens
        """
        # Rough estimation: count words and convert to tokens
        words = len(dia_text.split())
        estimated_tokens = words * 1.3  # Rough word-to-token ratio
        estimated_duration = estimated_tokens / 86  # Tokens per second from README
        
        return max(estimated_duration, 1.0)  # Minimum 1 second
    
    def _generate_audio_filename(self, persona: Persona) -> str:
        """Generate unique filename for audio output"""
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
        self.engines = {
            'dia': DiaTTSEngine()
        }
        self.default_engine = settings.TTS_VENDOR
    
    def synthesize(self, tts_input: TTSInput) -> TTSOutput:
        """
        Synthesize speech using the appropriate TTS engine.
        
        Args:
            tts_input: TTSInput with styled_text and persona
            
        Returns:
            TTSOutput with audio_path and duration
        """
        # Determine which engine to use
        vendor = tts_input.persona.voice.vendor
        if vendor not in self.engines:
            logger.warning(f"TTS vendor '{vendor}' not available, using default '{self.default_engine}'")
            vendor = self.default_engine
        
        if vendor not in self.engines:
            logger.error(f"No TTS engine available for vendor '{vendor}'")
            return TTSOutput(audio_path="", duration_seconds=0.0)
        
        # Use the appropriate engine
        engine = self.engines[vendor]
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

# Singleton instance
_tts_adapter = None

def get_tts_adapter() -> TTSAdapter:
    """Get singleton TTS adapter instance"""
    global _tts_adapter
    if _tts_adapter is None:
        _tts_adapter = TTSAdapter()
    return _tts_adapter
