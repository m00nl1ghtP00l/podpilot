"""
Tests for update_durations.py

This test suite covers:
- Duration formatting
- Audio file duration extraction
- Video duration updates
- Filename sanitization
- Metadata duration updates
- File finding logic
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys

# Add parent directory to path to import the module
sys.path.insert(0, str(Path(__file__).parent.parent))

from update_durations import (
    format_duration,
    update_video_duration,
    update_video_duration_from_file,
    get_audio_duration,
    find_audio_file,
    update_metadata_durations
)
from channel_fetcher import sanitize_filename


class TestFormatDuration:
    """Tests for format_duration function"""
    
    def test_format_seconds(self):
        """Test formatting seconds (< 1 minute)"""
        assert format_duration(45) == "00:45"
        assert format_duration(0) == "00:00"
        assert format_duration(59) == "00:59"
    
    def test_format_minutes(self):
        """Test formatting minutes (< 1 hour)"""
        assert format_duration(60) == "01:00"
        assert format_duration(125) == "02:05"
        assert format_duration(3599) == "59:59"
    
    def test_format_hours(self):
        """Test formatting hours"""
        assert format_duration(3600) == "01:00:00"
        assert format_duration(3661) == "01:01:01"
        assert format_duration(7325) == "02:02:05"  # 2 hours, 2 minutes, 5 seconds
    
    def test_format_negative(self):
        """Test formatting negative duration"""
        assert format_duration(-10) == "00:00"
    
    def test_format_float(self):
        """Test formatting float duration"""
        assert format_duration(90.5) == "01:30"
        assert format_duration(125.7) == "02:05"


class TestUpdateVideoDuration:
    """Tests for update_video_duration function"""
    
    def test_update_video_duration(self):
        """Test updating a video with duration"""
        video = {'title': 'Test Video', 'id': 'test123'}
        result = update_video_duration(video, 300)
        
        assert result == video  # Returns same dict
        assert video['duration_seconds'] == 300
        assert video['duration'] == "05:00"
    
    def test_update_video_duration_overwrites(self):
        """Test that updating overwrites existing duration"""
        video = {
            'title': 'Test Video',
            'duration_seconds': 100,
            'duration': '01:40'
        }
        update_video_duration(video, 500)
        
        assert video['duration_seconds'] == 500
        assert video['duration'] == "08:20"
    
    def test_update_video_duration_float(self):
        """Test updating with float duration"""
        video = {'title': 'Test Video'}
        update_video_duration(video, 125.7)
        
        assert video['duration_seconds'] == 125
        assert video['duration'] == "02:05"


class TestSanitizeFilename:
    """Tests for sanitize_filename function"""
    
    def test_sanitize_no_invalid_chars(self):
        """Test sanitizing filename with no invalid characters"""
        assert sanitize_filename("test_file.mp3") == "test_file.mp3"
        assert sanitize_filename("2025-01-01_Title") == "2025-01-01_Title"
    
    def test_sanitize_slash(self):
        """Test sanitizing forward slash"""
        assert sanitize_filename("test/file.mp3") == "test_file.mp3"
        assert sanitize_filename("f/w@test") == "f_w@test"
    
    def test_sanitize_backslash(self):
        """Test sanitizing backslash"""
        assert sanitize_filename("test\\file.mp3") == "test_file.mp3"
    
    def test_sanitize_colon(self):
        """Test sanitizing colon"""
        assert sanitize_filename("test:file.mp3") == "test_file.mp3"
    
    def test_sanitize_multiple_invalid(self):
        """Test sanitizing multiple invalid characters"""
        assert sanitize_filename("test/file:name*?.mp3") == "test_file_name_.mp3"
    
    def test_sanitize_multiple_underscores(self):
        """Test that multiple underscores are collapsed"""
        assert sanitize_filename("test___file.mp3") == "test_file.mp3"
        assert sanitize_filename("test//file.mp3") == "test_file.mp3"
    
    def test_sanitize_empty(self):
        """Test sanitizing empty string"""
        assert sanitize_filename("") == ""
    
    def test_sanitize_trailing_underscores(self):
        """Test that trailing underscores are removed"""
        assert sanitize_filename("test_file__") == "test_file"


class TestGetAudioDuration:
    """Tests for get_audio_duration function"""
    
    @patch('update_durations.MP3')
    def test_get_audio_duration_success(self, mock_mp3):
        """Test successfully getting duration from audio file"""
        mock_audio = MagicMock()
        mock_audio.info.length = 300.5
        mock_mp3.return_value = mock_audio
        
        result = get_audio_duration(Path("test.mp3"))
        
        assert result is not None
        duration_seconds, duration_formatted = result
        assert duration_seconds == 300.5
        assert duration_formatted == "05:00"
    
    @patch('update_durations.MP3')
    def test_get_audio_duration_failure(self, mock_mp3):
        """Test handling failure when getting duration"""
        from mutagen import MutagenError
        mock_mp3.side_effect = MutagenError("File not found")
        
        result = get_audio_duration(Path("test.mp3"))
        assert result is None
    
    @patch('update_durations.MP3')
    def test_get_audio_duration_exception(self, mock_mp3):
        """Test handling general exception"""
        mock_mp3.side_effect = Exception("Unexpected error")
        
        result = get_audio_duration(Path("test.mp3"))
        assert result is None


class TestFindAudioFile:
    """Tests for find_audio_file function"""
    
    def test_find_by_clean_filename(self, tmp_path):
        """Test finding file by clean_filename"""
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()
        test_file = audio_dir / "2025-01-01_Test_Title_video123.mp3"
        test_file.write_bytes(b"fake audio")
        
        result = find_audio_file(audio_dir, "video123", "2025-01-01_Test_Title_video123")
        
        assert result == test_file
    
    def test_find_by_video_id(self, tmp_path):
        """Test finding file by video ID in filename"""
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()
        test_file = audio_dir / "2025-01-01_Test_video123.mp3"
        test_file.write_bytes(b"fake audio")
        
        result = find_audio_file(audio_dir, "video123", "2025-01-01_Test_Title")
        
        assert result == test_file
    
    def test_find_by_date_prefix(self, tmp_path):
        """Test finding file by date prefix and title keywords"""
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()
        test_file = audio_dir / "2025-01-01_Test_Title_Other_Stuff.mp3"
        test_file.write_bytes(b"fake audio")
        
        result = find_audio_file(audio_dir, "", "2025-01-01_Test_Title")
        
        assert result == test_file
    
    def test_find_not_found(self, tmp_path):
        """Test when file is not found"""
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()
        
        result = find_audio_file(audio_dir, "nonexistent", "2025-01-01_Test")
        
        assert result is None


class TestUpdateVideoDurationFromFile:
    """Tests for update_video_duration_from_file function"""
    
    @patch('update_durations.get_audio_duration')
    @patch('update_durations.find_audio_file')
    def test_update_from_file_success(self, mock_find, mock_get_duration, tmp_path):
        """Test successfully updating duration from file"""
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()
        test_file = audio_dir / "test.mp3"
        test_file.write_bytes(b"fake audio")
        
        mock_find.return_value = test_file
        mock_get_duration.return_value = (300.0, "05:00")
        
        video = {'id': 'test123', 'clean_filename': 'test'}
        updated, should_keep = update_video_duration_from_file(video, audio_dir, min_duration=0)
        
        assert updated is True
        assert should_keep is True
        assert video['duration_seconds'] == 300
        assert video['duration'] == "05:00"
    
    @patch('update_durations.get_audio_duration')
    @patch('update_durations.find_audio_file')
    def test_update_from_file_too_short(self, mock_find, mock_get_duration, tmp_path):
        """Test updating file that's too short"""
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()
        test_file = audio_dir / "test.mp3"
        test_file.write_bytes(b"fake audio")
        
        mock_find.return_value = test_file
        mock_get_duration.return_value = (100.0, "01:40")
        
        video = {'id': 'test123', 'clean_filename': 'test'}
        updated, should_keep = update_video_duration_from_file(video, audio_dir, min_duration=300)
        
        assert updated is True
        assert should_keep is False  # Too short, should be removed
        assert video['duration_seconds'] == 100
    
    @patch('update_durations.find_audio_file')
    def test_update_from_file_already_has_duration(self, mock_find, tmp_path):
        """Test skipping file that already has duration"""
        audio_dir = tmp_path / "audio"
        
        video = {
            'id': 'test123',
            'duration_seconds': 500,
            'duration': '08:20'
        }
        updated, should_keep = update_video_duration_from_file(video, audio_dir, min_duration=0)
        
        assert updated is False  # Didn't update (already had duration)
        assert should_keep is True
        mock_find.assert_not_called()  # Shouldn't try to find file
    
    @patch('update_durations.find_audio_file')
    def test_update_from_file_already_has_duration_too_short(self, mock_find, tmp_path):
        """Test file with duration that's too short"""
        audio_dir = tmp_path / "audio"
        
        video = {
            'id': 'test123',
            'duration_seconds': 100,
            'duration': '01:40'
        }
        updated, should_keep = update_video_duration_from_file(video, audio_dir, min_duration=300)
        
        assert updated is False  # Didn't update (already had duration)
        assert should_keep is False  # Too short, should be removed
    
    @patch('update_durations.find_audio_file')
    def test_update_from_file_not_found(self, mock_find, tmp_path):
        """Test when audio file is not found"""
        audio_dir = tmp_path / "audio"
        mock_find.return_value = None
        
        video = {'id': 'test123', 'clean_filename': 'test'}
        updated, should_keep = update_video_duration_from_file(video, audio_dir, min_duration=0)
        
        assert updated is False
        assert should_keep is True  # Keep in metadata even if file not found


class TestUpdateMetadataDurations:
    """Tests for update_metadata_durations function"""
    
    def test_update_metadata_durations(self, tmp_path):
        """Test updating metadata with durations"""
        # Create test metadata
        metadata_file = tmp_path / "metadata.json"
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()
        
        metadata = {
            'videos': [
                {
                    'id': 'test1',
                    'title': 'Test 1',
                    'clean_filename': '2025-01-01_Test1_test1'
                },
                {
                    'id': 'test2',
                    'title': 'Test 2',
                    'clean_filename': '2025-01-02_Test2_test2'
                }
            ]
        }
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f)
        
        # Create test audio files
        test_file1 = audio_dir / "2025-01-01_Test1_test1.mp3"
        test_file1.write_bytes(b"fake audio")
        test_file2 = audio_dir / "2025-01-02_Test2_test2.mp3"
        test_file2.write_bytes(b"fake audio")
        
        with patch('update_durations.get_audio_duration') as mock_get:
            mock_get.side_effect = [
                (300.0, "05:00"),
                (600.0, "10:00")
            ]
            
            result = update_metadata_durations(metadata_file, audio_dir, verbose=False, min_duration=0)
            
            assert len(result['videos']) == 2
            assert result['videos'][0]['duration_seconds'] == 300
            assert result['videos'][1]['duration_seconds'] == 600
    
    def test_update_metadata_durations_filters_short(self, tmp_path):
        """Test that short videos are filtered out"""
        metadata_file = tmp_path / "metadata.json"
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()
        
        metadata = {
            'videos': [
                {
                    'id': 'test1',
                    'title': 'Short Video',
                    'clean_filename': '2025-01-01_Short_test1'
                },
                {
                    'id': 'test2',
                    'title': 'Long Video',
                    'clean_filename': '2025-01-02_Long_test2'
                }
            ]
        }
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f)
        
        test_file1 = audio_dir / "2025-01-01_Short_test1.mp3"
        test_file1.write_bytes(b"fake audio")
        test_file2 = audio_dir / "2025-01-02_Long_test2.mp3"
        test_file2.write_bytes(b"fake audio")
        
        with patch('update_durations.get_audio_duration') as mock_get:
            mock_get.side_effect = [
                (100.0, "01:40"),  # Too short (< 300s)
                (600.0, "10:00")   # Long enough
            ]
            
            result = update_metadata_durations(metadata_file, audio_dir, verbose=False, min_duration=300)
            
            # Only long video should remain
            assert len(result['videos']) == 1
            assert result['videos'][0]['title'] == 'Long Video'
    
    def test_update_metadata_durations_skips_existing(self, tmp_path):
        """Test that videos with existing duration are skipped"""
        metadata_file = tmp_path / "metadata.json"
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()
        
        metadata = {
            'videos': [
                {
                    'id': 'test1',
                    'title': 'Has Duration',
                    'clean_filename': '2025-01-01_Has_test1',
                    'duration_seconds': 500,
                    'duration': '08:20'
                },
                {
                    'id': 'test2',
                    'title': 'No Duration',
                    'clean_filename': '2025-01-02_No_test2'
                }
            ]
        }
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f)
        
        test_file2 = audio_dir / "2025-01-02_No_test2.mp3"
        test_file2.write_bytes(b"fake audio")
        
        with patch('update_durations.get_audio_duration') as mock_get:
            mock_get.return_value = (600.0, "10:00")
            
            result = update_metadata_durations(metadata_file, audio_dir, verbose=False, min_duration=0)
            
            # Both should remain
            assert len(result['videos']) == 2
            # First should keep original duration
            assert result['videos'][0]['duration_seconds'] == 500
            # Second should get new duration
            assert result['videos'][1]['duration_seconds'] == 600

