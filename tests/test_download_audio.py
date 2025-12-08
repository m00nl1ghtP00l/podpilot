"""
Tests for download_audio.py

This test suite covers:
- JSON loading and validation
- Date parsing and range checking
- File processing and transcoding
- File downloading (with mocks)
- YouTube downloading (with mocks)
- Audio file validation
- Main download orchestration
"""

import pytest
import json
import os
import tempfile
import argparse
import requests
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import Mock, patch, mock_open, MagicMock
import sys

# Add parent directory to path to import the module
sys.path.insert(0, str(Path(__file__).parent.parent))

from download_audio import (
    load_json,
    parse_date_arg,
    is_date_in_range,
    process_existing_file,
    transcode_file,
    download_file,
    download_audio_from_youtube,
    is_valid_audio_file,
    download_mode,
    get_youtube_duration,
    main
)
from channel_fetcher import load_config, find_podcast_by_name
from extract_duration import format_duration, update_video_duration


class TestLoadJson:
    """Tests for load_json function"""
    
    def test_load_json_valid(self, tmp_path):
        """Test loading valid JSON file"""
        json_file = tmp_path / "test.json"
        data = {"videos": [{"title": "Test"}]}
        json_file.write_text(json.dumps(data), encoding='utf-8')
        
        result = load_json(str(json_file))
        assert result == data
    
    def test_load_json_invalid_json(self, tmp_path):
        """Test loading invalid JSON file"""
        json_file = tmp_path / "test.json"
        json_file.write_text("invalid json", encoding='utf-8')
        
        with pytest.raises(ValueError, match="Invalid JSON"):
            load_json(str(json_file))
    
    def test_load_json_missing_file(self):
        """Test loading non-existent file"""
        with pytest.raises(ValueError, match="File not found"):
            load_json("/nonexistent/file.json")
    
    def test_load_json_missing_videos_key(self, tmp_path):
        """Test JSON file missing 'videos' key"""
        json_file = tmp_path / "test.json"
        json_file.write_text(json.dumps({"other": "data"}), encoding='utf-8')
        
        with pytest.raises(ValueError, match="Missing 'videos' key"):
            load_json(str(json_file))


class TestParseDateArg:
    """Tests for parse_date_arg function"""
    
    def test_parse_simple_date_format(self):
        """Test parsing YYYY-MM-DD format"""
        result = parse_date_arg("2024-01-15")
        assert isinstance(result, datetime)
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15
        assert result.tzinfo == timezone.utc
    
    def test_parse_iso_format(self):
        """Test parsing ISO format"""
        result = parse_date_arg("2024-01-15T10:30:00+00:00")
        assert isinstance(result, datetime)
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15
        assert result.hour == 10
        assert result.minute == 30
    
    def test_parse_invalid_format(self):
        """Test parsing invalid date format"""
        with pytest.raises(argparse.ArgumentTypeError):
            parse_date_arg("invalid-date")
    
    def test_parse_relative_date_7d(self):
        """Test parsing relative date '7d' (7 days ago)"""
        result = parse_date_arg("7d")
        assert isinstance(result, datetime)
        # Should be 7 days ago at midnight
        from datetime import timedelta
        today = datetime.now(timezone.utc).date()
        expected_date = today - timedelta(days=7)
        expected = datetime.combine(expected_date, datetime.min.time()).replace(tzinfo=timezone.utc)
        assert result.date() == expected.date()
        assert result.hour == 0 and result.minute == 0 and result.second == 0
    
    def test_parse_relative_date_1w(self):
        """Test parsing relative date '1w' (1 week ago)"""
        result = parse_date_arg("1w")
        assert isinstance(result, datetime)
        # Should be 1 week ago at midnight
        from datetime import timedelta
        today = datetime.now(timezone.utc).date()
        expected_date = today - timedelta(weeks=1)
        expected = datetime.combine(expected_date, datetime.min.time()).replace(tzinfo=timezone.utc)
        assert result.date() == expected.date()
        assert result.hour == 0 and result.minute == 0 and result.second == 0
    
    def test_parse_relative_date_1m(self):
        """Test parsing relative date '1m' (1 month ago)"""
        result = parse_date_arg("1m")
        assert isinstance(result, datetime)
        # Should be 30 days ago at midnight
        from datetime import timedelta
        today = datetime.now(timezone.utc).date()
        expected_date = today - timedelta(days=30)
        expected = datetime.combine(expected_date, datetime.min.time()).replace(tzinfo=timezone.utc)
        assert result.date() == expected.date()
        assert result.hour == 0 and result.minute == 0 and result.second == 0
    
    def test_parse_alias_today(self):
        """Test parsing alias 'today'"""
        result = parse_date_arg("today")
        assert isinstance(result, datetime)
        today = datetime.now(timezone.utc).date()
        assert result.date() == today
    
    def test_parse_alias_yesterday(self):
        """Test parsing alias 'yesterday'"""
        result = parse_date_arg("yesterday")
        assert isinstance(result, datetime)
        from datetime import timedelta
        yesterday = datetime.now(timezone.utc).date() - timedelta(days=1)
        assert result.date() == yesterday
    
    def test_parse_alias_last_week(self):
        """Test parsing alias 'last-week'"""
        result = parse_date_arg("last-week")
        assert isinstance(result, datetime)
        from datetime import timedelta
        last_week = datetime.now(timezone.utc).date() - timedelta(weeks=1)
        assert result.date() == last_week
    
    def test_parse_alias_last_month(self):
        """Test parsing alias 'last-month'"""
        result = parse_date_arg("last-month")
        assert isinstance(result, datetime)
        from datetime import timedelta
        last_month = datetime.now(timezone.utc).date() - timedelta(days=30)
        assert result.date() == last_month


class TestIsDateInRange:
    """Tests for is_date_in_range function"""
    
    def test_date_in_range(self):
        """Test date within range"""
        date_str = "2024-01-15T12:00:00Z"
        from_date = datetime(2024, 1, 10, tzinfo=timezone.utc)
        to_date = datetime(2024, 1, 20, tzinfo=timezone.utc)
        
        assert is_date_in_range(date_str, from_date, to_date) is True
    
    def test_date_before_range(self):
        """Test date before range"""
        date_str = "2024-01-05T12:00:00Z"
        from_date = datetime(2024, 1, 10, tzinfo=timezone.utc)
        to_date = datetime(2024, 1, 20, tzinfo=timezone.utc)
        
        assert is_date_in_range(date_str, from_date, to_date) is False
    
    def test_date_after_range(self):
        """Test date after range"""
        date_str = "2024-01-25T12:00:00Z"
        from_date = datetime(2024, 1, 10, tzinfo=timezone.utc)
        to_date = datetime(2024, 1, 20, tzinfo=timezone.utc)
        
        assert is_date_in_range(date_str, from_date, to_date) is False
    
    def test_date_on_from_boundary(self):
        """Test date on from_date boundary"""
        date_str = "2024-01-10T12:00:00Z"
        from_date = datetime(2024, 1, 10, tzinfo=timezone.utc)
        to_date = datetime(2024, 1, 20, tzinfo=timezone.utc)
        
        assert is_date_in_range(date_str, from_date, to_date) is True
    
    def test_date_on_to_boundary(self):
        """Test date on to_date boundary"""
        date_str = "2024-01-20T12:00:00Z"
        from_date = datetime(2024, 1, 10, tzinfo=timezone.utc)
        to_date = datetime(2024, 1, 20, tzinfo=timezone.utc)
        
        assert is_date_in_range(date_str, from_date, to_date) is True
    
    def test_date_with_only_from_date(self):
        """Test date filtering with only from_date"""
        date_str = "2024-01-15T12:00:00Z"
        from_date = datetime(2024, 1, 10, tzinfo=timezone.utc)
        
        assert is_date_in_range(date_str, from_date, None) is True
    
    def test_date_with_only_to_date(self):
        """Test date filtering with only to_date"""
        date_str = "2024-01-15T12:00:00Z"
        to_date = datetime(2024, 1, 20, tzinfo=timezone.utc)
        
        assert is_date_in_range(date_str, None, to_date) is True
    
    def test_date_with_no_dates(self):
        """Test date filtering with no date constraints"""
        date_str = "2024-01-15T12:00:00Z"
        
        assert is_date_in_range(date_str, None, None) is True


class TestProcessExistingFile:
    """Tests for process_existing_file function"""
    
    @patch('download_audio.transcode')
    @patch('download_audio.os.path.getsize')
    @patch('builtins.print')
    def test_process_existing_file_valid(self, mock_print, mock_getsize, mock_transcode):
        """Test processing existing valid file (under 25MB)"""
        mock_getsize.return_value = 10 * 1024 * 1024  # 10 MB
        
        result = process_existing_file("/path/to/file.mp3", simulate=False)
        assert result is True
        mock_transcode.assert_not_called()
    
    @patch('download_audio.transcode')
    @patch('download_audio.os.path.getsize')
    @patch('builtins.print')
    def test_process_existing_file_too_large(self, mock_print, mock_getsize, mock_transcode):
        """Test processing existing file that's too large"""
        mock_getsize.return_value = 30 * 1024 * 1024  # 30 MB
        mock_transcode.return_value = {'success': True, 'original_size_mb': 30, 'new_size_mb': 20, 'bitrate': 128}
        
        result = process_existing_file("/path/to/file.mp3", simulate=False)
        assert result is True
        mock_transcode.assert_called_once()
    
    @patch('download_audio.os.path.getsize')
    @patch('builtins.print')
    def test_process_existing_file_exception(self, mock_print, mock_getsize):
        """Test processing existing file with exception"""
        mock_getsize.side_effect = OSError("File not found")
        
        result = process_existing_file("/path/to/file.mp3", simulate=False)
        assert result is False


class TestTranscodeFile:
    """Tests for transcode_file function"""
    
    @patch('download_audio.transcode')
    @patch('builtins.print')
    def test_transcode_file_success(self, mock_print, mock_transcode):
        """Test successful transcoding"""
        mock_transcode.return_value = {'success': True, 'original_size_mb': 30, 'new_size_mb': 20, 'bitrate': 128}
        
        result = transcode_file("/path/to/file.mp3", simulate=False)
        assert result == "/path/to/file.mp3"
        mock_transcode.assert_called_once_with("/path/to/file.mp3", target_size_mb=25, show_progress=True)
    
    @patch('download_audio.transcode')
    @patch('builtins.print')
    def test_transcode_file_failure(self, mock_print, mock_transcode):
        """Test failed transcoding"""
        mock_transcode.return_value = {'success': False, 'error': 'Transcoding failed'}
        
        result = transcode_file("/path/to/file.mp3", simulate=False)
        assert result is None
    
    @patch('builtins.print')
    def test_transcode_file_simulate(self, mock_print):
        """Test transcoding in simulate mode"""
        result = transcode_file("/path/to/file.mp3", simulate=True)
        assert result is None


class TestDownloadFile:
    """Tests for download_file function"""
    
    @patch('download_audio.is_valid_audio_file')
    @patch('download_audio.os.path.getsize')
    @patch('download_audio.requests.get')
    @patch('download_audio.os.makedirs')
    @patch('download_audio.tqdm')
    @patch('builtins.print')
    @patch('builtins.open', new_callable=mock_open)
    def test_download_file_success(self, mock_file, mock_print, mock_tqdm, mock_makedirs, mock_get, mock_getsize, mock_is_valid):
        """Test successful file download"""
        mock_response = Mock()
        mock_response.content = b"audio content"
        mock_response.raise_for_status = Mock()
        mock_response.headers = {'content-length': '1000', 'content-type': 'audio/mpeg'}
        mock_response.iter_content = Mock(return_value=iter([b"audio content"]))
        mock_get.return_value = mock_response
        mock_getsize.return_value = 1000
        mock_is_valid.return_value = True
        
        result = download_file("https://example.com/audio.mp3", "/path/to/output.mp3", simulate=False)
        
        assert result is True
        mock_get.assert_called_once_with("https://example.com/audio.mp3", stream=True)
    
    @patch('download_audio.requests.get')
    @patch('builtins.print')
    def test_download_file_failure(self, mock_print, mock_get):
        """Test failed file download"""
        mock_get.side_effect = requests.RequestException("Network error")
        
        result = download_file("https://example.com/audio.mp3", "/path/to/output.mp3", simulate=False)
        assert result is False
    
    @patch('builtins.print')
    def test_download_file_simulate(self, mock_print):
        """Test download in simulate mode"""
        result = download_file("https://example.com/audio.mp3", "/path/to/output.mp3", simulate=True)
        assert result is True


class TestDownloadAudioFromYouTube:
    """Tests for download_audio_from_youtube function"""
    
    @patch('download_audio.transcode')
    @patch('shutil.move')
    @patch('os.remove')
    @patch('os.listdir')
    @patch('os.path.getctime')
    @patch('download_audio.yt_dlp.YoutubeDL')
    @patch('os.path.getsize')
    @patch('os.path.exists')
    @patch('os.makedirs')
    @patch('tempfile.gettempdir')
    @patch('builtins.print')
    def test_download_audio_from_youtube_success(self, mock_print, mock_gettempdir, mock_makedirs, mock_exists, mock_getsize, mock_ytdlp_class, mock_getctime, mock_listdir, mock_remove, mock_move, mock_transcode):
        """Test successful YouTube download"""
        mock_ytdlp = Mock()
        mock_ytdlp_class.return_value.__enter__.return_value = mock_ytdlp
        mock_ytdlp.download.return_value = None  # Success
        mock_gettempdir.return_value = "/tmp"
        
        # Mock exists to return True for temp file, False for final file (so it doesn't try to remove)
        def exists_side_effect(path):
            return "/tmp/ytdlp_" in path
        mock_exists.side_effect = exists_side_effect
        
        # Mock listdir to return temp file
        mock_listdir.return_value = ["ytdlp_abc12345.mp3"]
        mock_getctime.return_value = 1000.0
        mock_getsize.return_value = 10 * 1024 * 1024  # 10 MB
        
        result = download_audio_from_youtube("https://youtube.com/watch?v=test", "/path/to/output.mp3")
        assert result is True
        mock_ytdlp.download.assert_called_once()
    
    @patch('download_audio.yt_dlp.YoutubeDL')
    @patch('builtins.print')
    def test_download_audio_from_youtube_failure(self, mock_print, mock_ytdlp_class):
        """Test failed YouTube download"""
        mock_ytdlp = Mock()
        mock_ytdlp_class.return_value.__enter__.return_value = mock_ytdlp
        mock_ytdlp.download.side_effect = Exception("Download error")
        
        result = download_audio_from_youtube("https://youtube.com/watch?v=test", "/path/to/output.mp3")
        assert result is False


class TestIsValidAudioFile:
    """Tests for is_valid_audio_file function"""
    
    @patch('download_audio.subprocess.run')
    def test_is_valid_audio_file_valid(self, mock_run):
        """Test valid audio file"""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_run.return_value = mock_result
        
        result = is_valid_audio_file("/path/to/file.mp3")
        assert result is True
        mock_run.assert_called_once()
    
    @patch('download_audio.subprocess.run')
    def test_is_valid_audio_file_invalid(self, mock_run):
        """Test invalid audio file"""
        # Create a mock result with empty stdout (invalid file)
        mock_result = Mock()
        mock_result.stdout = ""  # Empty stdout means no audio codec found
        mock_run.return_value = mock_result
        
        result = is_valid_audio_file("/path/to/file.mp3")
        # Should return False when stdout is empty (bool("".strip()) == False)
        assert result is False
        # Verify subprocess.run was called
        mock_run.assert_called_once()


class TestDownloadMode:
    """Tests for download_mode function"""
    
    @patch('download_audio.os.makedirs')
    @patch('download_audio.os.listdir')
    @patch('download_audio.download_file')
    def test_download_mode_basic(self, mock_download, mock_listdir, mock_makedirs):
        """Test basic download mode"""
        mock_listdir.return_value = []
        
        json_data = {
            'videos': [
                {
                    'title': 'Test Video',
                    'published': '2024-01-15T12:00:00Z',
                    'url': 'https://example.com/video.mp3',
                    'clean_filename': '2024-01-15_test_video'
                }
            ]
        }
        
        args = Mock()
        args.from_date = None
        args.to_date = None
        args.simulate = False
        
        download_mode(json_data, "/tmp/test", args)
        
        assert mock_download.called
    
    @patch('download_audio.os.makedirs')
    @patch('download_audio.os.listdir')
    @patch('download_audio.process_existing_file')
    @patch('download_audio.download_file')
    def test_download_mode_existing_file(self, mock_download, mock_process, mock_listdir, mock_makedirs):
        """Test download mode with existing file"""
        mock_listdir.return_value = ['2024-01-15_test_video.mp3']
        mock_process.return_value = True
        
        json_data = {
            'videos': [
                {
                    'title': 'Test Video',
                    'published': '2024-01-15T12:00:00Z',
                    'url': 'https://example.com/video.mp3',
                    'clean_filename': '2024-01-15_test_video'
                }
            ]
        }
        
        args = Mock()
        args.from_date = None
        args.to_date = None
        args.simulate = False
        
        download_mode(json_data, "/tmp/test", args)
        
        mock_process.assert_called()
        mock_download.assert_not_called()
    
    @patch('download_audio.os.makedirs')
    @patch('download_audio.os.listdir')
    def test_download_mode_date_filtering(self, mock_listdir, mock_makedirs):
        """Test date filtering in download mode"""
        mock_listdir.return_value = []
        
        json_data = {
            'videos': [
                {
                    'title': 'Old Video',
                    'published': '2024-01-05T12:00:00Z',
                    'url': 'https://example.com/video1.mp3',
                    'clean_filename': '2024-01-05_old_video'
                },
                {
                    'title': 'New Video',
                    'published': '2024-01-20T12:00:00Z',
                    'url': 'https://example.com/video2.mp3',
                    'clean_filename': '2024-01-20_new_video'
                }
            ]
        }
        
        args = Mock()
        args.from_date = datetime(2024, 1, 10, tzinfo=timezone.utc)
        args.to_date = datetime(2024, 1, 31, tzinfo=timezone.utc)
        args.simulate = False
        
        with patch('download_audio.download_file', return_value=True) as mock_download:
            download_mode(json_data, "/tmp/test", args)
            # Should only download the new video (Jan 20)
            assert mock_download.call_count == 1
    
    @patch('download_audio.os.makedirs')
    @patch('download_audio.os.listdir')
    def test_download_mode_simulate(self, mock_listdir, mock_makedirs):
        """Test simulate mode"""
        mock_listdir.return_value = []
        
        json_data = {
            'videos': [
                {
                    'title': 'Test Video',
                    'published': '2024-01-15T12:00:00Z',
                    'url': 'https://example.com/video.mp3',
                    'clean_filename': '2024-01-15_test_video'
                }
            ]
        }
        
        args = Mock()
        args.from_date = None
        args.to_date = None
        args.simulate = True
        
        with patch('download_audio.download_file') as mock_download:
            download_mode(json_data, "/tmp/test", args)
            mock_download.assert_not_called()


class TestMain:
    """Tests for main CLI function"""
    
    @patch('download_audio.download_mode')
    @patch('download_audio.load_json')
    def test_main_requires_json_or_config(self, mock_load_json, mock_download_mode):
        """Test that main requires either json_file or --config with --name"""
        with patch('sys.argv', ['download_audio.py']):
            with pytest.raises(SystemExit):
                main()
    
    @patch('download_audio.download_mode')
    @patch('download_audio.load_json')
    def test_main_default_audio_dir(self, mock_load_json, mock_download_mode):
        """Test default audio directory"""
        mock_load_json.return_value = {'videos': []}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json_file = f.name
            json.dump({'videos': []}, f)
        
        try:
            with patch('sys.argv', ['download_audio.py', json_file]):
                with patch('builtins.print'):
                    main()
            
            mock_download_mode.assert_called_once()
            call_args = mock_download_mode.call_args
            assert call_args[0][1] == './downloads'  # default audio_dir
        finally:
            if os.path.exists(json_file):
                os.remove(json_file)
    
    @patch('download_audio.download_mode')
    @patch('download_audio.load_json')
    def test_main_custom_audio_dir(self, mock_load_json, mock_download_mode):
        """Test custom audio directory"""
        mock_load_json.return_value = {'videos': []}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json_file = f.name
            json.dump({'videos': []}, f)
        
        try:
            with patch('sys.argv', ['download_audio.py', json_file, '-a', '/custom/path']):
                with patch('builtins.print'):
                    main()
            
            mock_download_mode.assert_called_once()
            call_args = mock_download_mode.call_args
            assert call_args[0][1] == '/custom/path'
        finally:
            if os.path.exists(json_file):
                os.remove(json_file)
    
    @patch('download_audio.download_mode')
    @patch('download_audio.load_json')
    def test_main_from_date(self, mock_load_json, mock_download_mode):
        """Test --from-date option"""
        mock_load_json.return_value = {'videos': []}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json_file = f.name
            json.dump({'videos': []}, f)
        
        try:
            with patch('sys.argv', ['download_audio.py', json_file, '--from-date', '2024-01-15']):
                with patch('builtins.print'):
                    main()
            
            mock_download_mode.assert_called_once()
            call_args = mock_download_mode.call_args
            args = call_args[0][2]
            assert args.from_date is not None
        finally:
            if os.path.exists(json_file):
                os.remove(json_file)
    
    @patch('download_audio.download_mode')
    @patch('download_audio.load_json')
    def test_main_to_date(self, mock_load_json, mock_download_mode):
        """Test --to-date option"""
        mock_load_json.return_value = {'videos': []}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json_file = f.name
            json.dump({'videos': []}, f)
        
        try:
            with patch('sys.argv', ['download_audio.py', json_file, '--to-date', '2024-01-31']):
                with patch('builtins.print'):
                    main()
            
            mock_download_mode.assert_called_once()
            call_args = mock_download_mode.call_args
            args = call_args[0][2]
            assert args.to_date is not None
        finally:
            if os.path.exists(json_file):
                os.remove(json_file)
    
    @patch('download_audio.download_mode')
    @patch('download_audio.load_json')
    def test_main_simulate(self, mock_load_json, mock_download_mode):
        """Test --simulate option"""
        mock_load_json.return_value = {'videos': []}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json_file = f.name
            json.dump({'videos': []}, f)
        
        try:
            with patch('sys.argv', ['download_audio.py', json_file, '--simulate']):
                with patch('builtins.print'):
                    main()
            
            mock_download_mode.assert_called_once()
            call_args = mock_download_mode.call_args
            args = call_args[0][2]
            assert args.simulate is True
        finally:
            if os.path.exists(json_file):
                os.remove(json_file)
    
    @patch('download_audio.download_mode')
    @patch('download_audio.load_json')
    def test_main_id_option(self, mock_load_json, mock_download_mode):
        """Test --id option"""
        mock_load_json.return_value = {
            'videos': [
                {'id': 'test123', 'title': 'Test', 'published': '2024-01-15T12:00:00Z', 'url': 'https://example.com/video.mp3', 'clean_filename': 'test'}
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json_file = f.name
            json.dump({'videos': [{'id': 'test123'}]}, f)
        
        try:
            with patch('sys.argv', ['download_audio.py', json_file, '--id', 'test123']):
                with patch('builtins.print'):
                    main()
            
            mock_download_mode.assert_called_once()
        finally:
            if os.path.exists(json_file):
                os.remove(json_file)
    
    @patch('download_audio.load_json')
    def test_main_id_not_found(self, mock_load_json):
        """Test --id option with non-existent ID"""
        mock_load_json.return_value = {'videos': []}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json_file = f.name
            json.dump({'videos': []}, f)
        
        try:
            with patch('sys.argv', ['download_audio.py', json_file, '--id', 'nonexistent']):
                with patch('builtins.print') as mock_print:
                    main()
                    # Should print error message
                    assert any('Error' in str(call) or 'No video found' in str(call) for call in mock_print.call_args_list)
        finally:
            if os.path.exists(json_file):
                os.remove(json_file)
    
    @patch('download_audio.download_mode')
    @patch('download_audio.load_json')
    def test_main_id_with_date_warning(self, mock_load_json, mock_download_mode):
        """Test warning when --id is used with date filters"""
        mock_load_json.return_value = {'videos': [{'id': 'test123'}]}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json_file = f.name
            json.dump({'videos': [{'id': 'test123'}]}, f)
        
        try:
            with patch('sys.argv', ['download_audio.py', json_file, '--id', 'test123', '--from-date', '2024-01-15']):
                with patch('builtins.print') as mock_print:
                    main()
                    # Should print warning
                    assert any('Warning' in str(call) for call in mock_print.call_args_list)
        finally:
            if os.path.exists(json_file):
                os.remove(json_file)
    
    @patch('download_audio.download_mode')
    @patch('download_audio.load_json')
    def test_main_date_range(self, mock_load_json, mock_download_mode):
        """Test date range filtering"""
        mock_load_json.return_value = {'videos': []}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json_file = f.name
            json.dump({'videos': []}, f)
        
        try:
            with patch('sys.argv', ['download_audio.py', json_file, '--from-date', '2024-01-15', '--to-date', '2024-01-31']):
                with patch('builtins.print'):
                    main()
            
            mock_download_mode.assert_called_once()
            call_args = mock_download_mode.call_args
            args = call_args[0][2]
            assert args.from_date is not None
            assert args.to_date is not None
        finally:
            if os.path.exists(json_file):
                os.remove(json_file)
    
    @patch('download_audio.download_mode')
    @patch('download_audio.load_json')
    def test_main_short_audio_dir(self, mock_load_json, mock_download_mode):
        """Test short form -a for audio-dir"""
        mock_load_json.return_value = {'videos': []}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json_file = f.name
            json.dump({'videos': []}, f)
        
        try:
            with patch('sys.argv', ['download_audio.py', json_file, '-a', '/custom/path']):
                with patch('builtins.print'):
                    main()
            
            mock_download_mode.assert_called_once()
            call_args = mock_download_mode.call_args
            assert call_args[0][1] == '/custom/path'
        finally:
            if os.path.exists(json_file):
                os.remove(json_file)
    
    @patch('download_audio.download_mode')
    @patch('download_audio.load_json')
    def test_main_short_simulate(self, mock_load_json, mock_download_mode):
        """Test short form -s for simulate"""
        mock_load_json.return_value = {'videos': []}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json_file = f.name
            json.dump({'videos': []}, f)
        
        try:
            with patch('sys.argv', ['download_audio.py', json_file, '-s']):
                with patch('builtins.print'):
                    main()
            
            mock_download_mode.assert_called_once()
            call_args = mock_download_mode.call_args
            args = call_args[0][2]
            assert args.simulate is True
        finally:
            if os.path.exists(json_file):
                os.remove(json_file)
    
    @patch('download_audio.download_mode')
    @patch('download_audio.load_config')
    @patch('download_audio.find_podcast_by_name')
    @patch('download_audio.load_json')
    def test_main_with_config_and_name(self, mock_load_json, mock_find, mock_load_config, mock_download_mode, tmp_path):
        """Test using --config and --name to find metadata file"""
        # Setup config
        config_data = {
            "data_root": str(tmp_path),
            "youtube_channels": [
                {
                    "channel_name_short": "test",
                    "channel_name_long": "Test Podcast",
                    "channel_id": "UC1234567890123456789012"
                }
            ]
        }
        mock_load_config.return_value = config_data
        mock_find.return_value = config_data["youtube_channels"][0]
        
        # Create metadata file
        metadata_file = tmp_path / "test.json"
        metadata_file.write_text(json.dumps({"videos": []}), encoding='utf-8')
        
        mock_load_json.return_value = {"videos": []}
        
        with patch('sys.argv', ['download_audio.py', '--config', 'config/podcasts.json', '--name', 'test']):
            with patch('builtins.print'):
                main()
        
        # Verify config was loaded and podcast found
        mock_load_config.assert_called_once_with('config/podcasts.json')
        mock_find.assert_called_once()
        # Verify it tried to load the metadata file from data_root
        assert mock_load_json.called
    
    @patch('download_audio.load_config')
    def test_main_config_without_name(self, mock_load_config):
        """Test that --config without --name raises error"""
        with patch('sys.argv', ['download_audio.py', '--config', 'config/podcasts.json']):
            with pytest.raises(SystemExit):
                main()
    
    @patch('download_audio.load_config')
    def test_main_config_invalid_podcast_name(self, mock_load_config):
        """Test that invalid podcast name in config raises error"""
        config_data = {
            "youtube_channels": [
                {"channel_name_short": "test", "channel_id": "UC123"}
            ]
        }
        mock_load_config.return_value = config_data
        
        with patch('download_audio.find_podcast_by_name') as mock_find:
            mock_find.side_effect = ValueError("Podcast not found")
            with patch('sys.argv', ['download_audio.py', '--config', 'config/podcasts.json', '--name', 'nonexistent']):
                with pytest.raises(SystemExit):
                    main()


class TestDurationFunctions:
    """Tests for duration-related functions"""
    
    def test_format_duration_imported(self):
        """Test that format_duration is imported from extract_duration"""
        assert format_duration is not None
        assert callable(format_duration)
    
    def test_format_duration_usage(self):
        """Test using format_duration function"""
        assert format_duration(300) == "05:00"
        assert format_duration(3661) == "01:01:01"
        assert format_duration(45) == "00:45"
    
    def test_update_video_duration_imported(self):
        """Test that update_video_duration is imported"""
        assert update_video_duration is not None
        assert callable(update_video_duration)
    
    def test_update_video_duration_usage(self):
        """Test using update_video_duration function"""
        video = {'title': 'Test'}
        result = update_video_duration(video, 500)
        
        assert result == video
        assert video['duration_seconds'] == 500
        assert video['duration'] == "08:20"
    
    @patch('download_audio.yt_dlp.YoutubeDL')
    def test_get_youtube_duration_success(self, mock_ydl_class):
        """Test getting duration from YouTube successfully"""
        mock_ydl = MagicMock()
        mock_ydl_class.return_value.__enter__.return_value = mock_ydl
        mock_ydl.extract_info.return_value = {'duration': 300}
        
        result = get_youtube_duration("https://www.youtube.com/watch?v=test123")
        
        assert result == 300
        mock_ydl.extract_info.assert_called_once_with("https://www.youtube.com/watch?v=test123", download=False)
    
    @patch('download_audio.yt_dlp.YoutubeDL')
    def test_get_youtube_duration_no_duration(self, mock_ydl_class):
        """Test getting duration when video info has no duration"""
        mock_ydl = MagicMock()
        mock_ydl_class.return_value.__enter__.return_value = mock_ydl
        mock_ydl.extract_info.return_value = {}  # No duration key
        
        result = get_youtube_duration("https://www.youtube.com/watch?v=test123")
        
        assert result is None
    
    @patch('download_audio.yt_dlp.YoutubeDL')
    def test_get_youtube_duration_exception(self, mock_ydl_class):
        """Test handling exception when getting duration"""
        mock_ydl = MagicMock()
        mock_ydl_class.return_value.__enter__.return_value = mock_ydl
        mock_ydl.extract_info.side_effect = Exception("Network error")
        
        result = get_youtube_duration("https://www.youtube.com/watch?v=test123")
        
        assert result is None


class TestDownloadModeWithDuration:
    """Tests for download_mode with duration handling"""
    
    @patch('download_audio.download_file')
    @patch('extract_duration.update_video_duration_from_file')
    @patch('download_audio.get_youtube_duration')
    def test_download_mode_saves_youtube_duration(self, mock_get_duration, mock_update_from_file, mock_download, tmp_path):
        """Test that YouTube duration is saved during download"""
        json_file = tmp_path / "metadata.json"
        output_dir = tmp_path / "downloads"
        output_dir.mkdir()
        
        metadata = {
            'videos': [
                {
                    'id': 'test123',
                    'title': 'Test Video',
                    'link': 'https://www.youtube.com/watch?v=test123',
                    'published': '2025-01-01',
                    'clean_filename': '2025-01-01_Test_test123'
                }
            ]
        }
        
        with open(json_file, 'w') as f:
            json.dump(metadata, f)
        
        mock_get_duration.return_value = 500
        mock_download.return_value = True
        
        class Args:
            simulate = False
            min_duration = None
            from_date = None
            to_date = None
        
        args = Args()
        
        download_mode(metadata, str(output_dir), args, str(json_file))
        
        # Verify duration was checked
        mock_get_duration.assert_called_once()
        # Verify video was updated with duration
        assert metadata['videos'][0]['duration_seconds'] == 500
        assert metadata['videos'][0]['duration'] == "08:20"
    
    @patch('download_audio.download_file')
    @patch('extract_duration.update_video_duration_from_file')
    @patch('download_audio.get_youtube_duration')
    def test_download_mode_processes_non_youtube_duration(self, mock_get_duration, mock_update_from_file, mock_download, tmp_path):
        """Test that non-YouTube files get duration from audio file"""
        json_file = tmp_path / "metadata.json"
        output_dir = tmp_path / "downloads"
        output_dir.mkdir()
        
        test_file = output_dir / "2025-01-01_Test_test123.mp3"
        test_file.write_bytes(b"fake audio")
        
        metadata = {
            'videos': [
                {
                    'id': 'test123',
                    'title': 'Test Video',
                    'link': 'https://example.com/audio.mp3',
                    'published': '2025-01-01',
                    'clean_filename': '2025-01-01_Test_test123'
                }
            ]
        }
        
        with open(json_file, 'w') as f:
            json.dump(metadata, f)
        
        mock_download.return_value = True
        mock_update_from_file.return_value = (True, True)  # Updated, should keep
        
        class Args:
            simulate = False
            min_duration = None
            from_date = None
            to_date = None
        
        args = Args()
        
        download_mode(metadata, str(output_dir), args, str(json_file))
        
        # Verify update_video_duration_from_file was called for non-YouTube file
        mock_update_from_file.assert_called()
        # YouTube duration should not be checked
        mock_get_duration.assert_not_called()
    
    @patch('download_audio.download_file')
    @patch('download_audio.get_youtube_duration')
    def test_download_mode_filters_short_videos(self, mock_get_duration, mock_download, tmp_path):
        """Test that short videos are filtered out"""
        json_file = tmp_path / "metadata.json"
        output_dir = tmp_path / "downloads"
        output_dir.mkdir()
        
        metadata = {
            'videos': [
                {
                    'id': 'test123',
                    'title': 'Short Video',
                    'link': 'https://www.youtube.com/watch?v=test123',
                    'published': '2025-01-01',
                    'clean_filename': '2025-01-01_Short_test123'
                }
            ]
        }
        
        with open(json_file, 'w') as f:
            json.dump(metadata, f)
        
        mock_get_duration.return_value = 100  # Short video
        mock_download.return_value = True
        
        class Args:
            simulate = False
            min_duration = 300  # 5 minutes minimum
            from_date = None
            to_date = None
        
        args = Args()
        
        download_mode(metadata, str(output_dir), args, str(json_file))
        
        # Video should be removed from metadata
        assert len(metadata['videos']) == 0
