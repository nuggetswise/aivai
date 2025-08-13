import os
import json
import logging
import yaml
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import uuid
import datetime

from app.config import settings

logger = logging.getLogger(__name__)

class FileManager:
    """
    File management utility for saving and loading data files,
    including transcripts, evidence bundles, and audio metadata.
    """
    
    def __init__(self):
        # Set up directory paths
        self.data_dir = Path(settings.DATA_DIR)
        self.corpus_dir = Path(settings.CORPUS_DIR)
        
        # Ensure data subdirectories exist
        self.dirs = {
            'audio': self.data_dir / 'audio',
            'bundles': self.data_dir / 'bundles',
            'indices': self.data_dir / 'indices',
            'sources': self.data_dir / 'sources',
            'transcripts': self.data_dir / 'transcripts',
            'turns': self.data_dir / 'turns',
            'voices': self.data_dir / 'voices',
        }
        
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Create all required directories if they don't exist"""
        for dir_path in self.dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
            
        # Also ensure corpus directory exists
        self.corpus_dir.mkdir(parents=True, exist_ok=True)
    
    def save_json(self, data: Dict[str, Any], filepath: Union[str, Path]) -> str:
        """
        Save data as JSON file.
        
        Args:
            data: Data to save
            filepath: Path to save to (absolute or relative to data dir)
            
        Returns:
            Absolute path to saved file
        """
        filepath = self._resolve_path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=self._json_serializer)
            
            logger.debug(f"Saved JSON to {filepath}")
            return str(filepath.absolute())
        except Exception as e:
            logger.error(f"Error saving JSON to {filepath}: {e}")
            raise
    
    def load_json(self, filepath: Union[str, Path]) -> Dict[str, Any]:
        """
        Load data from JSON file.
        
        Args:
            filepath: Path to load from (absolute or relative to data dir)
            
        Returns:
            Loaded data
        """
        filepath = self._resolve_path(filepath)
        
        if not filepath.exists():
            logger.warning(f"JSON file not found: {filepath}")
            return {}
            
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading JSON from {filepath}: {e}")
            return {}
    
    def save_yaml(self, data: Dict[str, Any], filepath: Union[str, Path]) -> str:
        """
        Save data as YAML file.
        
        Args:
            data: Data to save
            filepath: Path to save to (absolute or relative to data dir)
            
        Returns:
            Absolute path to saved file
        """
        filepath = self._resolve_path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
            
            logger.debug(f"Saved YAML to {filepath}")
            return str(filepath.absolute())
        except Exception as e:
            logger.error(f"Error saving YAML to {filepath}: {e}")
            raise
    
    def load_yaml(self, filepath: Union[str, Path]) -> Dict[str, Any]:
        """
        Load data from YAML file.
        
        Args:
            filepath: Path to load from (absolute or relative to data dir)
            
        Returns:
            Loaded data
        """
        filepath = self._resolve_path(filepath)
        
        if not filepath.exists():
            logger.warning(f"YAML file not found: {filepath}")
            return {}
            
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(f"Error loading YAML from {filepath}: {e}")
            return {}
    
    def save_text(self, text: str, filepath: Union[str, Path]) -> str:
        """
        Save text to file.
        
        Args:
            text: Text to save
            filepath: Path to save to (absolute or relative to data dir)
            
        Returns:
            Absolute path to saved file
        """
        filepath = self._resolve_path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(text)
            
            logger.debug(f"Saved text to {filepath}")
            return str(filepath.absolute())
        except Exception as e:
            logger.error(f"Error saving text to {filepath}: {e}")
            raise
    
    def load_text(self, filepath: Union[str, Path]) -> str:
        """
        Load text from file.
        
        Args:
            filepath: Path to load from (absolute or relative to data dir)
            
        Returns:
            Loaded text
        """
        filepath = self._resolve_path(filepath)
        
        if not filepath.exists():
            logger.warning(f"Text file not found: {filepath}")
            return ""
            
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error loading text from {filepath}: {e}")
            return ""
    
    def save_avatar(self, avatar_data: Dict[str, Any], name: str) -> str:
        """
        Save avatar data to corpus directory.
        
        Args:
            avatar_data: Avatar data to save
            name: Avatar name (used for filename)
            
        Returns:
            Path to saved file
        """
        # Sanitize name for filename
        safe_name = name.lower().replace(' ', '_').replace('-', '_')
        if not safe_name.endswith('.yaml'):
            safe_name += '.yaml'
            
        filepath = self.corpus_dir / safe_name
        return self.save_yaml(avatar_data, filepath)
    
    def load_avatar(self, name: str) -> Dict[str, Any]:
        """
        Load avatar data from corpus directory.
        
        Args:
            name: Avatar name or path
            
        Returns:
            Avatar data
        """
        # Handle both name and path inputs
        if not name.endswith('.yaml'):
            name += '.yaml'
        
        # If it's a full path, use it directly
        if os.path.isabs(name):
            return self.load_yaml(name)
            
        # Check in corpus directory
        filepath = self.corpus_dir / name
        return self.load_yaml(filepath)
    
    def save_transcript(self, transcript_data: Dict[str, Any], episode_id: str) -> str:
        """
        Save episode transcript.
        
        Args:
            transcript_data: Transcript data to save
            episode_id: Episode identifier
            
        Returns:
            Path to saved file
        """
        filepath = self.dirs['transcripts'] / f"{episode_id}_transcript.json"
        return self.save_json(transcript_data, filepath)
    
    def append_transcript(self, episode_id: str, turn_data: Dict[str, Any]) -> bool:
        """
        Append a turn to an episode transcript.
        
        Args:
            episode_id: Episode identifier
            turn_data: Turn data to append
            
        Returns:
            Success status
        """
        filepath = self.dirs['transcripts'] / f"{episode_id}_transcript.json"
        
        try:
            # Load existing transcript
            transcript = self.load_json(filepath)
            
            # Initialize turns list if needed
            if 'turns' not in transcript:
                transcript['turns'] = []
                
            # Append turn data
            transcript['turns'].append(turn_data)
            
            # Update last_updated timestamp
            transcript['last_updated'] = datetime.datetime.now().isoformat()
            
            # Save back to file
            self.save_json(transcript, filepath)
            return True
        except Exception as e:
            logger.error(f"Error appending to transcript {episode_id}: {e}")
            return False
    
    def save_bundle(self, bundle_data: Dict[str, Any], bundle_id: Optional[str] = None) -> str:
        """
        Save evidence bundle.
        
        Args:
            bundle_data: Bundle data to save
            bundle_id: Optional bundle identifier (generated if not provided)
            
        Returns:
            Bundle ID
        """
        if not bundle_id:
            bundle_id = f"bundle-{uuid.uuid4().hex[:8]}"
            
        # Ensure bundle has ID
        bundle_data['id'] = bundle_id
        
        # Add timestamp if not present
        if 'created_at' not in bundle_data:
            bundle_data['created_at'] = datetime.datetime.now().isoformat()
            
        filepath = self.dirs['bundles'] / f"{bundle_id}.json"
        self.save_json(bundle_data, filepath)
        
        return bundle_id
    
    def load_bundle(self, bundle_id: str) -> Dict[str, Any]:
        """
        Load evidence bundle.
        
        Args:
            bundle_id: Bundle identifier
            
        Returns:
            Bundle data
        """
        filepath = self.dirs['bundles'] / f"{bundle_id}.json"
        return self.load_json(filepath)
    
    def list_bundles(self) -> List[str]:
        """
        List available evidence bundles.
        
        Returns:
            List of bundle IDs
        """
        bundle_dir = self.dirs['bundles']
        
        try:
            return [
                f.stem for f in bundle_dir.glob("*.json") 
                if f.is_file() and not f.name.startswith('.')
            ]
        except Exception as e:
            logger.error(f"Error listing bundles: {e}")
            return []
    
    def list_avatars(self) -> List[Dict[str, Any]]:
        """
        List available avatars with basic info.
        
        Returns:
            List of avatar data (name, path, etc.)
        """
        try:
            avatars = []
            for f in self.corpus_dir.glob("*.yaml"):
                if f.is_file():
                    try:
                        data = self.load_yaml(f)
                        if 'name' in data:
                            avatars.append({
                                'name': data['name'],
                                'path': str(f),
                                'role': data.get('role', ''),
                                'tone': data.get('tone', '')
                            })
                    except Exception as e:
                        logger.warning(f"Failed to load avatar from {f}: {e}")
                        
            return avatars
        except Exception as e:
            logger.error(f"Error listing avatars: {e}")
            return []
    
    def save_source(self, source_data: Dict[str, Any], source_id: Optional[str] = None) -> str:
        """
        Save source document.
        
        Args:
            source_data: Source data to save
            source_id: Optional source identifier (generated if not provided)
            
        Returns:
            Source ID
        """
        if not source_id:
            source_id = f"src-{uuid.uuid4().hex[:8]}"
            
        # Ensure source has ID
        source_data['id'] = source_id
        
        # Add timestamp if not present
        if 'fetched_at' not in source_data:
            source_data['fetched_at'] = datetime.datetime.now().isoformat()
            
        filepath = self.dirs['sources'] / f"{source_id}.json"
        self.save_json(source_data, filepath)
        
        return source_id
    
    def copy_file(self, source_path: Union[str, Path], dest_subdir: str) -> str:
        """
        Copy file to a subdirectory in the data directory.
        
        Args:
            source_path: Path to source file
            dest_subdir: Destination subdirectory (e.g., 'audio', 'voices')
            
        Returns:
            Path to copied file
        """
        source_path = Path(source_path)
        
        if not source_path.exists():
            logger.error(f"Source file not found: {source_path}")
            raise FileNotFoundError(f"Source file not found: {source_path}")
        
        if dest_subdir not in self.dirs:
            logger.error(f"Invalid destination subdirectory: {dest_subdir}")
            raise ValueError(f"Invalid destination subdirectory: {dest_subdir}")
        
        dest_dir = self.dirs[dest_subdir]
        dest_path = dest_dir / source_path.name
        
        try:
            shutil.copy2(source_path, dest_path)
            logger.debug(f"Copied {source_path} to {dest_path}")
            return str(dest_path)
        except Exception as e:
            logger.error(f"Error copying file: {e}")
            raise
    
    def _resolve_path(self, path: Union[str, Path]) -> Path:
        """Resolve path to absolute path"""
        path = Path(path)
        
        # If path is already absolute, return it
        if path.is_absolute():
            return path
            
        # Otherwise, resolve relative to data dir
        return self.data_dir / path
    
    def _json_serializer(self, obj):
        """JSON serializer for objects not serializable by default"""
        if isinstance(obj, (datetime.datetime, datetime.date)):
            return obj.isoformat()
        if isinstance(obj, uuid.UUID):
            return str(obj)
        
        # Try to convert to dict if object has a dict method
        if hasattr(obj, 'dict') and callable(obj.dict):
            return obj.dict()
            
        # Try to convert to dict if object has __dict__
        if hasattr(obj, '__dict__'):
            return obj.__dict__
            
        raise TypeError(f"Type not serializable: {type(obj)}")

# Singleton instance
_file_manager = None

def get_file_manager() -> FileManager:
    """Get singleton file manager instance"""
    global _file_manager
    if _file_manager is None:
        _file_manager = FileManager()
    return _file_manager
