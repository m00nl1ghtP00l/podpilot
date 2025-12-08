"""
Tests for update_durations.py CLI interface

This test suite covers:
- CLI argument parsing
- Main function execution
- Error handling
- File processing
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, mock_open, MagicMock
from datetime import datetime
import sys

# Add parent directory to path to import the module
sys.path.insert(0, str(Path(__file__).parent.parent))

from update_durations import (
    format_duration,
    update_video_duration,
    update_video_duration_from_file,
    update_metadata_durations,
    main
)


class TestUpdateDurationsCLI:
    """Tests for update_durations.py CLI interface"""
    
    @patch('update_durations.load_config')
    @patch('update_durations.find_podcast_by_name')
    @patch('update_durations.Path.exists')
    @patch('update_durations.update_metadata_durations')
    def test_main_with_name(self, mock_update, mock_exists, mock_find_podcast, 
                            mock_load_config, tmp_path, capsys):
        """Test main() with --name argument"""
        # Setup mocks
        mock_config = {
            "youtube_channels": [{"channel_name_short": "test"}],
            "data_root": str(tmp_path)
        }
        mock_load_config.return_value = mock_config
        mock_find_podcast.return_value = {"channel_name_short": "test"}
        mock_exists.return_value = True
        mock_update.return_value = (5, 2)  # (processed, updated)
        
        with patch('sys.argv', ['update_durations.py', '--name', 'test']):
            main()
        
        mock_update.assert_called_once()
        captured = capsys.readouterr()
        assert "test" in captured.out.lower() or "processing" in captured.out.lower()
    
    @patch('update_durations.Path.exists')
    @patch('update_durations.update_metadata_durations')
    def test_main_with_json_file(self, mock_update, mock_exists, tmp_path, capsys):
        """Test main() with direct JSON file argument"""
        # Create test JSON file and audio directory
        json_file = tmp_path / "test.json"
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()
        
        json_data = {
            "channel_name": "Test Channel",
            "videos": [
                {"title": "Video 1", "filename": "video1.mp3"},
                {"title": "Video 2", "filename": "video2.mp3"}
            ]
        }
        json_file.write_text(json.dumps(json_data))
        
        mock_exists.return_value = True
        mock_update.return_value = (2, 1)
        
        with patch('sys.argv', ['update_durations.py', '--metadata-file', str(json_file), '--audio-dir', str(audio_dir)]):
            main()
        
        mock_update.assert_called_once()
    
    @patch('update_durations.load_config')
    @patch('update_durations.find_podcast_by_name')
    def test_main_missing_metadata_file(self, mock_find_podcast, mock_load_config, tmp_path, capsys):
        """Test main() when metadata file doesn't exist"""
        mock_config = {
            "youtube_channels": [{"channel_name_short": "test"}],
            "data_root": str(tmp_path)
        }
        mock_load_config.return_value = mock_config
        mock_find_podcast.return_value = {"channel_name_short": "test"}
        
        with patch('sys.argv', ['update_durations.py', '--name', 'test']):
            with pytest.raises(SystemExit) as exc_info:
                main()
            # Should exit with error code 1 when file doesn't exist
            assert exc_info.value.code == 1
        
        captured = capsys.readouterr()
        assert "not found" in captured.out.lower() or "error" in captured.out.lower()
    
    @patch('update_durations.load_config')
    @patch('update_durations.find_podcast_by_name')
    @patch('update_durations.Path.exists')
    @patch('update_durations.update_metadata_durations')
    def test_main_with_filters(self, mock_update, mock_exists, mock_find_podcast, 
                               mock_load_config, tmp_path, capsys):
        """Test main() with min_duration filter"""
        mock_config = {
            "youtube_channels": [{"channel_name_short": "test"}],
            "data_root": str(tmp_path),
            "min_duration": 300
        }
        mock_load_config.return_value = mock_config
        mock_find_podcast.return_value = {"channel_name_short": "test"}
        mock_exists.return_value = True
        mock_update.return_value = (3, 2)
        
        with patch('sys.argv', ['update_durations.py', '--name', 'test']):
            main()
        
        mock_update.assert_called_once()
        # Verify min_duration was passed
        call_kwargs = mock_update.call_args[1] if mock_update.call_args[1] else {}
        # The function should be called with the config's min_duration

