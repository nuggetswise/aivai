import logging
import os
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
import uuid
import tempfile
from datetime import datetime

import ffmpeg
from pydub import AudioSegment

from app.config import settings

logger = logging.getLogger(__name__)

class AudioProcessor:
    """
    Audio processing utility for working with speech audio files.
    Handles format conversion, normalization, and mixing of debate turns.
    """
    
    def __init__(self):
        self.output_dir = Path(settings.DATA_DIR) / "audio"
        self.temp_dir = Path(tempfile.gettempdir()) / "aivai_audio"
        self.target_loudness = -16.0  # Target LUFS (Loudness Units Full Scale)
        self.target_sample_rate = settings.SAMPLE_RATE
        self.target_channels = settings.CHANNELS
        
        # Ensure directories exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
    
    def normalize_audio(self, input_path: Union[str, Path], output_path: Optional[Union[str, Path]] = None) -> str:
        """
        Normalize audio loudness to target LUFS.
        
        Args:
            input_path: Path to input audio file
            output_path: Path for output audio file (generated if None)
            
        Returns:
            Path to normalized audio file
        """
        input_path = Path(input_path)
        
        if not input_path.exists():
            logger.error(f"Input audio file not found: {input_path}")
            raise FileNotFoundError(f"Audio file not found: {input_path}")
        
        if not output_path:
            output_path = self.output_dir / f"norm_{uuid.uuid4().hex[:8]}{input_path.suffix}"
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            logger.info(f"Normalizing audio: {input_path} -> {output_path}")
            
            # Get audio loudness
            probe = ffmpeg.probe(str(input_path), v='error')
            input_loudness = None
            
            for stream in probe['streams']:
                if stream['codec_type'] == 'audio':
                    # Use FFmpeg's loudnorm filter to analyze
                    loudness_data = (
                        ffmpeg
                        .input(str(input_path))
                        .filter('loudnorm', i=self.target_loudness, print_format='json')
                        .output('-', format='null')
                        .run(capture_stdout=True, capture_stderr=True)
                    )
                    
                    # Parse loudness info from stderr
                    loudness_info = loudness_data[1].decode('utf-8')
                    if "input_i" in loudness_info:
                        import re
                        match = re.search(r'input_i\s*:\s*([-\d.]+)', loudness_info)
                        if match:
                            input_loudness = float(match.group(1))
                    break
            
            # Apply normalization based on measured loudness
            if input_loudness is not None:
                logger.info(f"Measured input loudness: {input_loudness} LUFS, target: {self.target_loudness} LUFS")
                
                # Apply loudnorm filter
                (
                    ffmpeg
                    .input(str(input_path))
                    .filter('loudnorm', i=self.target_loudness, lra=11, tp=-1.5)
                    .output(str(output_path), ar=self.target_sample_rate, ac=self.target_channels)
                    .overwrite_output()
                    .run(quiet=True)
                )
            else:
                # Fallback to simple normalization
                logger.warning("Could not measure input loudness, using simple normalization")
                
                audio = AudioSegment.from_file(input_path)
                target_dbfs = -20  # Target dBFS as fallback
                change_in_dbfs = target_dbfs - audio.dBFS
                normalized = audio.apply_gain(change_in_dbfs)
                normalized.export(output_path, format=output_path.suffix.lstrip('.'))
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Audio normalization failed: {e}")
            # Return original file if normalization fails
            return str(input_path)
    
    def convert_audio_format(self, input_path: Union[str, Path], output_format: str = 'mp3') -> str:
        """
        Convert audio to another format.
        
        Args:
            input_path: Path to input audio file
            output_format: Target format (mp3, wav, etc.)
            
        Returns:
            Path to converted audio file
        """
        input_path = Path(input_path)
        
        if not input_path.exists():
            logger.error(f"Input audio file not found: {input_path}")
            raise FileNotFoundError(f"Audio file not found: {input_path}")
        
        output_path = self.output_dir / f"{input_path.stem}.{output_format}"
        
        try:
            logger.info(f"Converting audio: {input_path} -> {output_path}")
            
            (
                ffmpeg
                .input(str(input_path))
                .output(str(output_path), acodec='libmp3lame' if output_format == 'mp3' else None)
                .overwrite_output()
                .run(quiet=True)
            )
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Audio conversion failed: {e}")
            return str(input_path)
    
    def mix_audio_sequence(self, audio_paths: List[Union[str, Path]], output_path: Optional[Union[str, Path]] = None) -> str:
        """
        Mix a sequence of audio files with crossfades.
        
        Args:
            audio_paths: List of audio file paths in sequence
            output_path: Path for output mixed audio file
            
        Returns:
            Path to mixed audio file
        """
        if not audio_paths:
            logger.error("No audio paths provided for mixing")
            raise ValueError("No audio paths provided for mixing")
        
        if not output_path:
            output_path = self.output_dir / f"mix_{uuid.uuid4().hex[:8]}.mp3"
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            logger.info(f"Mixing {len(audio_paths)} audio files -> {output_path}")
            
            # Convert all to the same format and normalize first
            normalized_paths = []
            for i, path in enumerate(audio_paths):
                path = Path(path)
                if not path.exists():
                    logger.warning(f"Audio file not found, skipping: {path}")
                    continue
                
                # Normalize each file
                norm_path = self.temp_dir / f"norm_{i}_{path.name}"
                normalized_paths.append(self.normalize_audio(path, norm_path))
            
            if not normalized_paths:
                logger.error("No valid audio files for mixing")
                raise ValueError("No valid audio files for mixing")
            
            if len(normalized_paths) == 1:
                # Just one file, convert to desired format
                return self.convert_audio_format(normalized_paths[0], output_path.suffix.lstrip('.'))
            
            # Mix files using pydub with crossfade
            mixed = AudioSegment.from_file(normalized_paths[0])
            crossfade_duration = 50  # 50ms crossfade
            
            for i in range(1, len(normalized_paths)):
                next_segment = AudioSegment.from_file(normalized_paths[i])
                mixed = mixed.append(next_segment, crossfade=crossfade_duration)
            
            # Add 1 second of silence at the beginning and end
            silence = AudioSegment.silent(duration=1000)
            mixed = silence + mixed + silence
            
            # Export final mix
            mixed.export(output_path, format=output_path.suffix.lstrip('.'))
            
            # Clean up temp files
            for path in normalized_paths:
                try:
                    if path.startswith(str(self.temp_dir)):
                        os.unlink(path)
                except:
                    pass
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Audio mixing failed: {e}")
            # Return the first file if mixing fails
            if audio_paths:
                return str(audio_paths[0])
            raise
    
    def get_audio_duration(self, audio_path: Union[str, Path]) -> float:
        """
        Get duration of audio file in seconds.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Duration in seconds
        """
        audio_path = Path(audio_path)
        
        if not audio_path.exists():
            logger.error(f"Audio file not found: {audio_path}")
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        try:
            probe = ffmpeg.probe(str(audio_path))
            duration = float(probe['format']['duration'])
            return duration
        except Exception as e:
            logger.error(f"Failed to get audio duration: {e}")
            return 0.0
    
    def create_episode_mixdown(self, turn_audio_paths: Dict[str, str], output_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """
        Create a complete episode mixdown from turn audio files.
        
        Args:
            turn_audio_paths: Dict mapping turn IDs to audio file paths
            output_path: Path for output episode audio file
            
        Returns:
            Dict with mixdown path and metadata
        """
        if not turn_audio_paths:
            logger.error("No turn audio paths provided for episode mixdown")
            raise ValueError("No turn audio paths provided for episode mixdown")
        
        episode_id = f"episode_{uuid.uuid4().hex[:8]}"
        if not output_path:
            output_path = self.output_dir / f"{episode_id}.mp3"
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            logger.info(f"Creating episode mixdown with {len(turn_audio_paths)} turns -> {output_path}")
            
            # Sort turns by their key (assuming sequential turn IDs)
            sorted_turns = sorted(turn_audio_paths.items())
            audio_paths = [path for _, path in sorted_turns]
            
            # Mix the episode
            mixdown_path = self.mix_audio_sequence(audio_paths, output_path)
            
            # Get total duration
            total_duration = self.get_audio_duration(mixdown_path)
            
            # Create turn timestamps (approximate)
            timestamps = []
            current_time = 1.0  # Start after 1 second silence
            
            for turn_id, path in sorted_turns:
                duration = self.get_audio_duration(path)
                timestamps.append({
                    "turn_id": turn_id,
                    "start_time": current_time,
                    "end_time": current_time + duration,
                    "duration": duration
                })
                current_time += duration - (0.05 if current_time > 1.0 else 0)  # Account for crossfade
            
            result = {
                "episode_id": episode_id,
                "mixdown_path": mixdown_path,
                "total_duration": total_duration,
                "turn_count": len(turn_audio_paths),
                "timestamps": timestamps,
                "created_at": datetime.now().isoformat()
            }
            
            # Save metadata
            metadata_path = Path(mixdown_path).with_suffix('.json')
            with open(metadata_path, 'w') as f:
                import json
                json.dump(result, f, indent=2, default=str)
            
            return result
            
        except Exception as e:
            logger.error(f"Episode mixdown failed: {e}")
            raise

# Singleton instance
_audio_processor = None

def get_audio_processor() -> AudioProcessor:
    """Get singleton audio processor instance"""
    global _audio_processor
    if _audio_processor is None:
        _audio_processor = AudioProcessor()
    return _audio_processor
