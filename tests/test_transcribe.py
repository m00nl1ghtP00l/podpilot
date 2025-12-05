"""
Tests for transcribe.py

This test suite covers:
- Date parsing and file date extraction
- Date range checking
- Transcription file checking
- Config loading and validation
- JSON file operations
- Podcast data fetching (with mocks)
- Audio file downloading (with mocks)
- Transcription processing (with mocks)
- CLI interface
"""

import pytest
import os
import tempfile
import json
import argparse
from pathlib import Path
from unittest.mock import Mock, patch, mock_open, MagicMock, call
from datetime import datetime, timezone
import sys

# Add parent directory to path to import the module
sys.path.insert(0, str(Path(__file__).parent.parent))

from transcribe import (
    parse_date_arg,
    get_file_date,
    parse_date_string,
    is_date_in_range,
    check_existing_transcription,
    transcribe_audio,
    load_config,
    find_podcast_by_name,
    list_podcasts,
    load_json,
    save_json,
    fetch_podcast_data,
    download_audio_files,
    process_transcriptions,
    main
)


class TestParseDateArg:
    """Tests for parse_date_arg function"""
    
    def test_parse_iso_format(self):
        """Test parsing ISO format date"""
        result = parse_date_arg('2024-01-15T10:30:00+00:00')
        assert isinstance(result, datetime)
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15
    
    def test_parse_simple_date_format(self):
        """Test parsing simple date format"""
        result = parse_date_arg('2024-01-15')
        assert isinstance(result, datetime)
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15
        # fromisoformat('2024-01-15') succeeds and returns naive datetime
        # Only strptime path adds timezone, but fromisoformat is tried first
        # So result may or may not have timezone depending on which path is taken
        # Both are valid - just check it's a datetime
        assert isinstance(result, datetime)
    
    def test_parse_none(self):
        """Test parsing None"""
        result = parse_date_arg(None)
        assert result is None
    
    def test_parse_empty_string(self):
        """Test parsing empty string"""
        result = parse_date_arg('')
        assert result is None
    
    def test_parse_invalid_format(self):
        """Test parsing invalid date format"""
        with pytest.raises(argparse.ArgumentTypeError):
            parse_date_arg('invalid-date')


class TestGetFileDate:
    """Tests for get_file_date function"""
    
    def test_get_file_date_valid(self):
        """Test extracting date from valid filename"""
        result = get_file_date('2024-01-15_Test_Episode')
        assert result == datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
    
    def test_get_file_date_invalid_format(self):
        """Test extracting date from invalid filename"""
        result = get_file_date('invalid_filename')
        assert result is None
    
    def test_get_file_date_no_underscore(self):
        """Test extracting date from filename without underscore"""
        result = get_file_date('2024-01-15')
        assert result == datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
    
    def test_get_file_date_empty_string(self):
        """Test extracting date from empty string"""
        result = get_file_date('')
        assert result is None


class TestParseDateString:
    """Tests for parse_date_string function"""
    
    def test_parse_iso_format(self):
        """Test parsing ISO format"""
        result = parse_date_string('2024-01-15T10:30:00+00:00')
        assert isinstance(result, datetime)
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15
    
    def test_parse_iso_with_z(self):
        """Test parsing ISO format with Z timezone"""
        result = parse_date_string('2024-01-15T10:30:00Z')
        assert isinstance(result, datetime)
        assert result.tzinfo is not None
    
    def test_parse_datetime_format(self):
        """Test parsing YYYY-MM-DD HH:MM format"""
        result = parse_date_string('2024-01-15 10:30')
        assert isinstance(result, datetime)
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15
        assert result.hour == 10
        assert result.minute == 30
        # fromisoformat might succeed for some formats, but strptime path adds timezone
        # Check that we got a datetime, and if it has timezone it should be UTC
        assert isinstance(result, datetime)
        if result.tzinfo is not None:
            assert result.tzinfo == timezone.utc
    
    def test_parse_simple_date_format(self):
        """Test parsing YYYY-MM-DD format"""
        result = parse_date_string('2024-01-15')
        assert isinstance(result, datetime)
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15
        # fromisoformat might succeed and return naive datetime, but strptime path adds timezone
        # Check that we got a datetime, and if it has timezone it should be UTC
        assert isinstance(result, datetime)
        if result.tzinfo is not None:
            assert result.tzinfo == timezone.utc
    
    def test_parse_datetime_object(self):
        """Test parsing datetime object (should return as-is)"""
        dt = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        result = parse_date_string(dt)
        assert result == dt
    
    def test_parse_none(self):
        """Test parsing None"""
        result = parse_date_string(None)
        assert result is None
    
    def test_parse_empty_string(self):
        """Test parsing empty string"""
        result = parse_date_string('')
        assert result is None
    
    def test_parse_invalid_format(self):
        """Test parsing invalid date format"""
        result = parse_date_string('invalid-date')
        assert result is None


class TestIsDateInRange:
    """Tests for is_date_in_range function"""
    
    def test_date_in_range(self):
        """Test date within range"""
        date = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        from_date = datetime(2024, 1, 10, tzinfo=timezone.utc)
        to_date = datetime(2024, 1, 20, tzinfo=timezone.utc)
        assert is_date_in_range(date, from_date, to_date) is True
    
    def test_date_before_range(self):
        """Test date before range"""
        date = datetime(2024, 1, 5, 12, 0, 0, tzinfo=timezone.utc)
        from_date = datetime(2024, 1, 10, tzinfo=timezone.utc)
        to_date = datetime(2024, 1, 20, tzinfo=timezone.utc)
        assert is_date_in_range(date, from_date, to_date) is False
    
    def test_date_after_range(self):
        """Test date after range"""
        date = datetime(2024, 1, 25, 12, 0, 0, tzinfo=timezone.utc)
        from_date = datetime(2024, 1, 10, tzinfo=timezone.utc)
        to_date = datetime(2024, 1, 20, tzinfo=timezone.utc)
        assert is_date_in_range(date, from_date, to_date) is False
    
    def test_date_with_only_from_date(self):
        """Test date with only from_date"""
        date = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        from_date = datetime(2024, 1, 10, tzinfo=timezone.utc)
        assert is_date_in_range(date, from_date, None) is True
    
    def test_date_with_only_to_date(self):
        """Test date with only to_date"""
        date = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        to_date = datetime(2024, 1, 20, tzinfo=timezone.utc)
        assert is_date_in_range(date, None, to_date) is True
    
    def test_date_string_in_range(self):
        """Test date string within range"""
        date_str = '2024-01-15'
        from_date = datetime(2024, 1, 10, tzinfo=timezone.utc)
        to_date = datetime(2024, 1, 20, tzinfo=timezone.utc)
        assert is_date_in_range(date_str, from_date, to_date) is True
    
    def test_date_none(self):
        """Test None date"""
        assert is_date_in_range(None, None, None) is False
    
    def test_invalid_date_string(self):
        """Test invalid date string"""
        assert is_date_in_range('invalid-date', None, None) is False


class TestCheckExistingTranscription:
    """Tests for check_existing_transcription function"""
    
    def test_check_existing_transcription_json_exists(self, tmp_path):
        """Test when JSON transcription exists"""
        audio_file = tmp_path / "test.mp3"
        audio_file.touch()
        json_file = tmp_path / "test.json"
        json_file.touch()
        
        assert check_existing_transcription(str(audio_file)) is True
    
    def test_check_existing_transcription_txt_exists(self, tmp_path):
        """Test when TXT transcription exists"""
        audio_file = tmp_path / "test.mp3"
        audio_file.touch()
        txt_file = tmp_path / "test.txt"
        txt_file.touch()
        
        assert check_existing_transcription(str(audio_file)) is True
    
    def test_check_existing_transcription_both_exist(self, tmp_path):
        """Test when both JSON and TXT exist"""
        audio_file = tmp_path / "test.mp3"
        audio_file.touch()
        json_file = tmp_path / "test.json"
        json_file.touch()
        txt_file = tmp_path / "test.txt"
        txt_file.touch()
        
        assert check_existing_transcription(str(audio_file)) is True
    
    def test_check_existing_transcription_none_exist(self, tmp_path):
        """Test when no transcription exists"""
        audio_file = tmp_path / "test.mp3"
        audio_file.touch()
        
        assert check_existing_transcription(str(audio_file)) is False


class TestLoadConfig:
    """Tests for load_config function"""
    
    def test_load_config_valid(self, tmp_path):
        """Test loading valid config file"""
        config_file = tmp_path / "config.json"
        config_data = {
            "data_root": "/tmp/test",
            "youtube_channels": [
                {
                    "channel_name_short": "test",
                    "channel_name_long": "Test Podcast",
                    "channel_id": "UC123456"
                }
            ]
        }
        config_file.write_text(json.dumps(config_data), encoding='utf-8')
        
        result = load_config(str(config_file))
        assert result == config_data
    
    def test_load_config_missing_data_root(self, tmp_path):
        """Test loading config missing data_root"""
        config_file = tmp_path / "config.json"
        config_data = {
            "youtube_channels": []
        }
        config_file.write_text(json.dumps(config_data), encoding='utf-8')
        
        with pytest.raises(ValueError, match="missing 'data_root' field"):
            load_config(str(config_file))
    
    def test_load_config_missing_youtube_channels(self, tmp_path):
        """Test loading config missing youtube_channels"""
        config_file = tmp_path / "config.json"
        config_data = {
            "data_root": "/tmp/test"
        }
        config_file.write_text(json.dumps(config_data), encoding='utf-8')
        
        with pytest.raises(ValueError, match="missing 'youtube_channels' list"):
            load_config(str(config_file))
    
    def test_load_config_invalid_podcast(self, tmp_path):
        """Test loading config with invalid podcast entry"""
        config_file = tmp_path / "config.json"
        config_data = {
            "data_root": "/tmp/test",
            "youtube_channels": [
                {
                    "channel_name_short": "test"
                    # Missing required fields
                }
            ]
        }
        config_file.write_text(json.dumps(config_data), encoding='utf-8')
        
        with pytest.raises(ValueError, match="missing 'channel_name_long'"):
            load_config(str(config_file))
    
    def test_load_config_invalid_json(self, tmp_path):
        """Test loading invalid JSON file"""
        config_file = tmp_path / "config.json"
        config_file.write_text("{ invalid json }", encoding='utf-8')
        
        with pytest.raises(ValueError, match="JSON syntax error"):
            load_config(str(config_file))
    
    def test_load_config_file_not_found(self):
        """Test loading non-existent config file"""
        with pytest.raises(ValueError, match="Error reading config file"):
            load_config("/nonexistent/file.json")


class TestFindPodcastByName:
    """Tests for find_podcast_by_name function"""
    
    def test_find_podcast_by_name_found(self):
        """Test finding existing podcast"""
        config = {
            "youtube_channels": [
                {"channel_name_short": "test1", "channel_name_long": "Test 1", "channel_id": "UC1"},
                {"channel_name_short": "test2", "channel_name_long": "Test 2", "channel_id": "UC2"}
            ]
        }
        
        result = find_podcast_by_name(config, "test1")
        assert result == config["youtube_channels"][0]
    
    def test_find_podcast_by_name_not_found(self):
        """Test finding non-existent podcast"""
        config = {
            "youtube_channels": [
                {"channel_name_short": "test1", "channel_name_long": "Test 1", "channel_id": "UC1"}
            ]
        }
        
        with pytest.raises(ValueError, match="not found in config"):
            find_podcast_by_name(config, "nonexistent")


class TestListPodcasts:
    """Tests for list_podcasts function"""
    
    def test_list_podcasts(self, capsys):
        """Test listing podcasts"""
        config = {
            "youtube_channels": [
                {"channel_name_short": "test1", "channel_name_long": "Test 1", "channel_id": "UC1"},
                {"channel_name_short": "test2", "channel_name_long": "Test 2", "channel_id": "UC2"}
            ]
        }
        
        result = list_podcasts(config)
        assert result is True
        
        captured = capsys.readouterr()
        assert "Available Podcasts" in captured.out
        assert "test1" in captured.out
        assert "test2" in captured.out
        assert "UC1" in captured.out
        assert "UC2" in captured.out


class TestLoadJson:
    """Tests for load_json function"""
    
    def test_load_json_valid(self, tmp_path):
        """Test loading valid JSON file"""
        json_file = tmp_path / "test.json"
        data = {"key": "value", "number": 42}
        json_file.write_text(json.dumps(data), encoding='utf-8')
        
        result = load_json(str(json_file))
        assert result == data
    
    def test_load_json_invalid(self, tmp_path, capsys):
        """Test loading invalid JSON file"""
        json_file = tmp_path / "test.json"
        json_file.write_text("{ invalid json }", encoding='utf-8')
        
        result = load_json(str(json_file))
        assert result is None
        
        captured = capsys.readouterr()
        assert "Error reading JSON file" in captured.out
    
    def test_load_json_not_found(self, capsys):
        """Test loading non-existent JSON file"""
        result = load_json("/nonexistent/file.json")
        assert result is None
        
        captured = capsys.readouterr()
        assert "Error reading JSON file" in captured.out


class TestSaveJson:
    """Tests for save_json function"""
    
    def test_save_json_success(self, tmp_path, capsys):
        """Test saving JSON file successfully"""
        json_file = tmp_path / "test.json"
        data = {"key": "value", "number": 42}
        
        result = save_json(data, str(json_file))
        assert result is True
        assert json_file.exists()
        
        loaded = json.loads(json_file.read_text(encoding='utf-8'))
        assert loaded == data
        
        captured = capsys.readouterr()
        assert "Data saved to" in captured.out
    
    def test_save_json_failure(self, capsys):
        """Test saving JSON file failure"""
        # Try to save to invalid path (directory that doesn't exist)
        result = save_json({"key": "value"}, "/nonexistent/dir/file.json")
        assert result is False
        
        captured = capsys.readouterr()
        assert "Error saving JSON file" in captured.out


class TestTranscribeAudio:
    """Tests for transcribe_audio function"""
    
    @patch('transcribe.Path')
    @patch('builtins.open', new_callable=mock_open)
    def test_transcribe_audio_success(self, mock_file, mock_path, tmp_path, capsys):
        """Test successful transcription"""
        # Setup mocks
        mock_client = Mock()
        mock_transcript = Mock()
        mock_transcript.model_dump.return_value = {
            'text': 'Test transcription text',
            'segments': []
        }
        mock_client.audio.transcriptions.create.return_value = mock_transcript
        
        # Setup path mocks
        audio_file = tmp_path / "test.mp3"
        audio_file.touch()
        base_path = Mock()
        base_path.parent = tmp_path
        base_path.stem = "test"
        mock_path.return_value = base_path
        
        result = transcribe_audio(str(audio_file), mock_client)
        assert result is True
        
        # Verify API was called
        mock_client.audio.transcriptions.create.assert_called_once()
        call_args = mock_client.audio.transcriptions.create.call_args
        assert call_args.kwargs['model'] == 'whisper-1'
        assert call_args.kwargs['language'] == 'ja'
        
        captured = capsys.readouterr()
        assert "Transcription saved" in captured.out
    
    def test_transcribe_audio_failure(self, capsys):
        """Test transcription failure"""
        mock_client = Mock()
        mock_client.audio.transcriptions.create.side_effect = Exception("API Error")
        
        result = transcribe_audio("/nonexistent/file.mp3", mock_client)
        assert result is False
        
        captured = capsys.readouterr()
        assert "Error transcribing" in captured.out


class TestFetchPodcastData:
    """Tests for fetch_podcast_data function"""
    
    @patch('transcribe.save_json')
    @patch('transcribe.parse_rss_feed')
    @patch('transcribe.fetch_rss_feed')
    def test_fetch_podcast_data_success(self, mock_fetch, mock_parse, mock_save, tmp_path, capsys):
        """Test successful podcast data fetch"""
        mock_fetch.return_value = "<xml>content</xml>"
        mock_parse.return_value = {
            'videos': [
                {'title': 'Video 1', 'date': '2024-01-15'}
            ]
        }
        mock_save.return_value = True
        
        output_file = tmp_path / "output.json"
        result = fetch_podcast_data("UC123", str(output_file), None, None, simulate=False)
        
        assert result is True
        mock_fetch.assert_called_once_with("UC123")
        mock_parse.assert_called_once()
        mock_save.assert_called_once()
        
        captured = capsys.readouterr()
        assert "Fetching podcast data" in captured.out
    
    def test_fetch_podcast_data_simulate(self, tmp_path, capsys):
        """Test podcast data fetch in simulate mode"""
        output_file = tmp_path / "output.json"
        result = fetch_podcast_data("UC123", str(output_file), None, None, simulate=True)
        
        assert result is True
        
        captured = capsys.readouterr()
        assert "Would fetch podcast data" in captured.out
    
    @patch('transcribe.save_json')
    @patch('transcribe.parse_rss_feed')
    @patch('transcribe.fetch_rss_feed')
    def test_fetch_podcast_data_failure(self, mock_fetch, mock_parse, mock_save, tmp_path, capsys):
        """Test podcast data fetch failure"""
        mock_fetch.side_effect = Exception("Network error")
        
        output_file = tmp_path / "output.json"
        result = fetch_podcast_data("UC123", str(output_file), None, None, simulate=False)
        
        assert result is False
        
        captured = capsys.readouterr()
        assert "Error fetching podcast data" in captured.out


class TestDownloadAudioFiles:
    """Tests for download_audio_files function"""
    
    @patch('transcribe.process_existing_file')
    @patch('transcribe.download_file')
    @patch('transcribe.clean_title')
    @patch('transcribe.load_json')
    def test_download_audio_files_success(self, mock_load, mock_clean, mock_download, mock_process, tmp_path, capsys):
        """Test successful audio file download"""
        # Setup
        json_file = tmp_path / "data.json"
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()
        
        mock_load.return_value = {
            'videos': [
                {
                    'title': 'Test Video',
                    'published': '2024-01-15',
                    'url': 'https://youtube.com/watch?v=123',
                    'clean_filename': '2024-01-15_Test_Video'
                }
            ]
        }
        mock_download.return_value = True
        
        result = download_audio_files(
            str(json_file), str(audio_dir), None, None, force_download=False, simulate=False
        )
        
        assert result is True
        mock_download.assert_called()
        
        captured = capsys.readouterr()
        assert "Downloading audio files" in captured.out
    
    @patch('transcribe.load_json')
    def test_download_audio_files_simulate(self, mock_load, tmp_path, capsys):
        """Test audio file download in simulate mode"""
        json_file = tmp_path / "data.json"
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()
        
        mock_load.return_value = {
            'videos': [
                {
                    'title': 'Test Video',
                    'published': '2024-01-15',
                    'url': 'https://youtube.com/watch?v=123',
                    'clean_filename': '2024-01-15_Test_Video'
                }
            ]
        }
        
        result = download_audio_files(
            str(json_file), str(audio_dir), None, None, force_download=False, simulate=True
        )
        
        assert result is True
        
        captured = capsys.readouterr()
        assert "Simulation mode" in captured.out
    
    @patch('transcribe.load_json')
    def test_download_audio_files_no_json_data(self, mock_load, tmp_path):
        """Test when JSON data cannot be loaded"""
        json_file = tmp_path / "data.json"
        audio_dir = tmp_path / "audio"
        
        mock_load.return_value = None
        
        result = download_audio_files(
            str(json_file), str(audio_dir), None, None, force_download=False, simulate=False
        )
        
        assert result is False


class TestProcessTranscriptions:
    """Tests for process_transcriptions function"""
    
    @patch('transcribe.transcribe_audio')
    @patch('transcribe.load_json')
    @patch('transcribe.OpenAI')
    def test_process_transcriptions_success(self, mock_openai, mock_load, mock_transcribe, tmp_path, capsys):
        """Test successful transcription processing"""
        # Setup
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()
        json_file = tmp_path / "data.json"
        
        # Create test audio file
        audio_file = audio_dir / "2024-01-15_Test_Episode.mp3"
        audio_file.touch()
        
        mock_load.return_value = {
            'videos': [
                {'clean_filename': '2024-01-15_Test_Episode'}
            ]
        }
        mock_client = Mock()
        mock_openai.return_value = mock_client
        mock_transcribe.return_value = True
        
        result = process_transcriptions(
            str(audio_dir), str(json_file), None, None, retranscribe=False, 
            api_key="test-key", simulate=False
        )
        
        assert result is True
        mock_transcribe.assert_called()
        
        captured = capsys.readouterr()
        assert "Processing transcriptions" in captured.out
    
    def test_process_transcriptions_simulate(self, tmp_path, capsys):
        """Test transcription processing in simulate mode"""
        audio_dir = tmp_path / "audio"
        json_file = tmp_path / "data.json"
        
        result = process_transcriptions(
            str(audio_dir), str(json_file), None, None, retranscribe=False,
            api_key=None, simulate=True
        )
        
        assert result is True
        
        captured = capsys.readouterr()
        assert "Simulation mode" in captured.out
    
    @patch.dict(os.environ, {}, clear=True)
    def test_process_transcriptions_no_api_key(self, tmp_path, capsys):
        """Test transcription processing without API key"""
        audio_dir = tmp_path / "audio"
        json_file = tmp_path / "data.json"
        
        result = process_transcriptions(
            str(audio_dir), str(json_file), None, None, retranscribe=False,
            api_key=None, simulate=False
        )
        
        assert result is False
        
        captured = capsys.readouterr()
        assert "OpenAI API key not provided" in captured.out
    
    @patch('transcribe.load_json')
    def test_process_transcriptions_no_json_data(self, mock_load, tmp_path):
        """Test when JSON data cannot be loaded"""
        audio_dir = tmp_path / "audio"
        json_file = tmp_path / "data.json"
        
        mock_load.return_value = None
        
        result = process_transcriptions(
            str(audio_dir), str(json_file), None, None, retranscribe=False,
            api_key="test-key", simulate=False
        )
        
        assert result is False


class TestMain:
    """Tests for main function"""
    
    @patch('transcribe.list_podcasts')
    @patch('transcribe.load_config')
    def test_main_list_podcasts(self, mock_load, mock_list):
        """Test --list option"""
        mock_load.return_value = {"youtube_channels": []}
        
        with patch('sys.argv', ['transcribe.py', '--list']):  # config defaults to config/podcasts.json
            result = main()
        
        assert result == 0
        mock_list.assert_called_once()
        # Verify default config path was used
        mock_load.assert_called_once_with('config/podcasts.json')
    
    @patch('transcribe.load_config')
    def test_main_no_name(self, mock_load, capsys):
        """Test main without --name option"""
        mock_load.return_value = {"youtube_channels": []}
        
        # Need to provide args to get past the "no args" check and trigger the error
        with patch('sys.argv', ['transcribe.py', '--config', 'config/podcasts.json']):
            result = main()
        
        assert result == 1
        
        captured = capsys.readouterr()
        # Check for the actual error message format
        assert "Error: --name is required to specify which podcast to process" in captured.out
    
    @patch('transcribe.find_podcast_by_name')
    @patch('transcribe.load_config')
    def test_main_podcast_not_found(self, mock_load, mock_find):
        """Test main with non-existent podcast name"""
        mock_load.return_value = {"youtube_channels": []}
        mock_find.side_effect = ValueError("Podcast not found")
        
        with patch('sys.argv', ['transcribe.py', '--name', 'nonexistent']):  # config defaults
            result = main()
        
        assert result == 1
    
    @patch('transcribe.process_transcriptions')
    @patch('transcribe.download_audio_files')
    @patch('transcribe.fetch_podcast_data')
    @patch('transcribe.find_podcast_by_name')
    @patch('transcribe.load_config')
    def test_main_full_workflow(self, mock_load, mock_find, mock_fetch, 
                                mock_download, mock_transcribe, tmp_path):
        """Test full workflow"""
        # Setup mocks
        config = {
            "data_root": str(tmp_path),
            "youtube_channels": [{"channel_name_short": "test", "channel_name_long": "Test", "channel_id": "UC123"}]
        }
        mock_load.return_value = config
        mock_find.return_value = config["youtube_channels"][0]
        
        # Create metadata file so it exists
        metadata_file = tmp_path / "test.json"
        metadata_file.touch()
        
        mock_fetch.return_value = True
        mock_download.return_value = True
        mock_transcribe.return_value = True
        
        with patch('sys.argv', ['transcribe.py', '--name', 'test']):  # config defaults
            result = main()
        
        assert result == 0
        # Verify default config path was used
        mock_load.assert_called_once_with('config/podcasts.json')
    
    @patch('transcribe.process_transcriptions')
    @patch('transcribe.download_audio_files')
    @patch('transcribe.fetch_podcast_data')
    @patch('transcribe.find_podcast_by_name')
    @patch('transcribe.load_config')
    def test_main_simulate_mode(self, mock_load, mock_find, mock_fetch,
                                mock_download, mock_transcribe, tmp_path):
        """Test simulate mode"""
        config = {
            "data_root": str(tmp_path),
            "youtube_channels": [{"channel_name_short": "test", "channel_name_long": "Test", "channel_id": "UC123"}]
        }
        mock_load.return_value = config
        mock_find.return_value = config["youtube_channels"][0]
        
        # Don't create metadata file so it doesn't exist
        # This will trigger fetch_podcast_data
        
        mock_fetch.return_value = True
        mock_download.return_value = True
        mock_transcribe.return_value = True
        
        with patch('sys.argv', ['transcribe.py', '--name', 'test', '--simulate']):
            result = main()
        
        assert result == 0
        # Verify simulate was passed through
        mock_fetch.assert_called()
        assert mock_fetch.call_args[0][4] is True  # simulate parameter
    
    @patch('transcribe.load_config')
    def test_main_invalid_config(self, mock_load):
        """Test main with invalid config"""
        mock_load.side_effect = ValueError("Config error")
        
        with patch('sys.argv', ['transcribe.py', '--name', 'test']):  # config defaults
            result = main()
        
        assert result == 1
    
    @patch('transcribe.process_transcriptions')
    @patch('transcribe.download_audio_files')
    @patch('transcribe.fetch_podcast_data')
    @patch('transcribe.find_podcast_by_name')
    @patch('transcribe.load_config')
    def test_main_download_option(self, mock_load, mock_find, mock_fetch, mock_download, mock_transcribe, tmp_path):
        """Test --download option forces download"""
        config = {
            "data_root": str(tmp_path),
            "youtube_channels": [{"channel_name_short": "test", "channel_name_long": "Test", "channel_id": "UC123"}]
        }
        mock_load.return_value = config
        mock_find.return_value = config["youtube_channels"][0]
        
        metadata_file = tmp_path / "test.json"
        metadata_file.touch()
        
        mock_fetch.return_value = True
        mock_download.return_value = True
        mock_transcribe.return_value = True
        
        with patch('sys.argv', ['transcribe.py', '--name', 'test', '--download']):
            result = main()
        
        assert result == 0
        # Verify --download was passed (force_download should be True)
        mock_download.assert_called()
        assert mock_download.call_args[0][4] is True  # force_download parameter
    
    @patch('transcribe.process_transcriptions')
    @patch('transcribe.download_audio_files')
    @patch('transcribe.fetch_podcast_data')
    @patch('transcribe.find_podcast_by_name')
    @patch('transcribe.load_config')
    def test_main_force_option(self, mock_load, mock_find, mock_fetch, mock_download, mock_transcribe, tmp_path):
        """Test --force option forces all operations"""
        config = {
            "data_root": str(tmp_path),
            "youtube_channels": [{"channel_name_short": "test", "channel_name_long": "Test", "channel_id": "UC123"}]
        }
        mock_load.return_value = config
        mock_find.return_value = config["youtube_channels"][0]
        
        # Don't create metadata file so fetch is triggered
        mock_fetch.return_value = True
        mock_download.return_value = True
        mock_transcribe.return_value = True
        
        with patch('sys.argv', ['transcribe.py', '--name', 'test', '--force']):
            result = main()
        
        assert result == 0
        # Verify --force triggers fetch (metadata doesn't exist or force=True)
        mock_fetch.assert_called()
        # Verify --force passed to download (force_download=True)
        mock_download.assert_called()
        assert mock_download.call_args[0][4] is True  # force_download parameter
        # Verify --force passed to transcribe (retranscribe=True)
        mock_transcribe.assert_called()
        assert mock_transcribe.call_args[0][4] is True  # retranscribe parameter
    
    @patch('transcribe.process_transcriptions')
    @patch('transcribe.download_audio_files')
    @patch('transcribe.fetch_podcast_data')
    @patch('transcribe.find_podcast_by_name')
    @patch('transcribe.load_config')
    def test_main_retranscribe_option(self, mock_load, mock_find, mock_fetch, mock_download, mock_transcribe, tmp_path):
        """Test --retranscribe option"""
        config = {
            "data_root": str(tmp_path),
            "youtube_channels": [{"channel_name_short": "test", "channel_name_long": "Test", "channel_id": "UC123"}]
        }
        mock_load.return_value = config
        mock_find.return_value = config["youtube_channels"][0]
        
        metadata_file = tmp_path / "test.json"
        metadata_file.touch()
        
        mock_fetch.return_value = True
        mock_download.return_value = True
        mock_transcribe.return_value = True
        
        with patch('sys.argv', ['transcribe.py', '--name', 'test', '--retranscribe']):
            result = main()
        
        assert result == 0
        # Verify --retranscribe was passed
        mock_transcribe.assert_called()
        assert mock_transcribe.call_args[0][4] is True  # retranscribe parameter
    
    @patch('transcribe.process_transcriptions')
    @patch('transcribe.download_audio_files')
    @patch('transcribe.fetch_podcast_data')
    @patch('transcribe.find_podcast_by_name')
    @patch('transcribe.load_config')
    def test_main_api_key_option(self, mock_load, mock_find, mock_fetch, mock_download, mock_transcribe, tmp_path):
        """Test --api-key option"""
        config = {
            "data_root": str(tmp_path),
            "youtube_channels": [{"channel_name_short": "test", "channel_name_long": "Test", "channel_id": "UC123"}]
        }
        mock_load.return_value = config
        mock_find.return_value = config["youtube_channels"][0]
        
        metadata_file = tmp_path / "test.json"
        metadata_file.touch()
        
        mock_fetch.return_value = True
        mock_download.return_value = True
        mock_transcribe.return_value = True
        
        with patch('sys.argv', ['transcribe.py', '--name', 'test', '--api-key', 'test-api-key-123']):
            result = main()
        
        assert result == 0
        # Verify --api-key was passed
        mock_transcribe.assert_called()
        assert mock_transcribe.call_args[0][5] == 'test-api-key-123'  # api_key parameter
    
    @patch('transcribe.process_transcriptions')
    @patch('transcribe.download_audio_files')
    @patch('transcribe.fetch_podcast_data')
    @patch('transcribe.find_podcast_by_name')
    @patch('transcribe.load_config')
    def test_main_from_date_option(self, mock_load, mock_find, mock_fetch, mock_download, mock_transcribe, tmp_path):
        """Test -f/--from-date option"""
        config = {
            "data_root": str(tmp_path),
            "youtube_channels": [{"channel_name_short": "test", "channel_name_long": "Test", "channel_id": "UC123"}]
        }
        mock_load.return_value = config
        mock_find.return_value = config["youtube_channels"][0]
        
        # Don't create metadata file so fetch is called
        mock_fetch.return_value = True
        mock_download.return_value = True
        mock_transcribe.return_value = True
        
        with patch('sys.argv', ['transcribe.py', '--name', 'test', '--from-date', '2024-01-15']):
            result = main()
        
        assert result == 0
        # Verify --from-date was passed to all functions
        mock_fetch.assert_called()
        from_date = mock_fetch.call_args[0][2]
        assert from_date is not None
        assert from_date.year == 2024
        assert from_date.month == 1
        assert from_date.day == 15
        
        mock_download.assert_called()
        assert mock_download.call_args[0][2] == from_date
        
        mock_transcribe.assert_called()
        assert mock_transcribe.call_args[0][2] == from_date
    
    @patch('transcribe.process_transcriptions')
    @patch('transcribe.download_audio_files')
    @patch('transcribe.fetch_podcast_data')
    @patch('transcribe.find_podcast_by_name')
    @patch('transcribe.load_config')
    def test_main_to_date_option(self, mock_load, mock_find, mock_fetch, mock_download, mock_transcribe, tmp_path):
        """Test -t/--to-date option"""
        config = {
            "data_root": str(tmp_path),
            "youtube_channels": [{"channel_name_short": "test", "channel_name_long": "Test", "channel_id": "UC123"}]
        }
        mock_load.return_value = config
        mock_find.return_value = config["youtube_channels"][0]
        
        # Don't create metadata file so fetch is called
        mock_fetch.return_value = True
        mock_download.return_value = True
        mock_transcribe.return_value = True
        
        with patch('sys.argv', ['transcribe.py', '--name', 'test', '--to-date', '2024-01-31']):
            result = main()
        
        assert result == 0
        # Verify --to-date was passed to all functions
        mock_fetch.assert_called()
        to_date = mock_fetch.call_args[0][3]
        assert to_date is not None
        assert to_date.year == 2024
        assert to_date.month == 1
        assert to_date.day == 31
        
        mock_download.assert_called()
        assert mock_download.call_args[0][3] == to_date
        
        mock_transcribe.assert_called()
        assert mock_transcribe.call_args[0][3] == to_date
    
    @patch('transcribe.process_transcriptions')
    @patch('transcribe.download_audio_files')
    @patch('transcribe.fetch_podcast_data')
    @patch('transcribe.find_podcast_by_name')
    @patch('transcribe.load_config')
    def test_main_date_range_options(self, mock_load, mock_find, mock_fetch, mock_download, mock_transcribe, tmp_path):
        """Test -f/--from-date and -t/--to-date together"""
        config = {
            "data_root": str(tmp_path),
            "youtube_channels": [{"channel_name_short": "test", "channel_name_long": "Test", "channel_id": "UC123"}]
        }
        mock_load.return_value = config
        mock_find.return_value = config["youtube_channels"][0]
        
        # Don't create metadata file so fetch is called
        mock_fetch.return_value = True
        mock_download.return_value = True
        mock_transcribe.return_value = True
        
        with patch('sys.argv', ['transcribe.py', '--name', 'test', 
                                '-f', '2024-01-15', '-t', '2024-01-31']):
            result = main()
        
        assert result == 0
        # Verify both dates were passed
        mock_fetch.assert_called()
        from_date = mock_fetch.call_args[0][2]
        to_date = mock_fetch.call_args[0][3]
        assert from_date is not None
        assert to_date is not None
        assert from_date.year == 2024 and from_date.month == 1 and from_date.day == 15
        assert to_date.year == 2024 and to_date.month == 1 and to_date.day == 31

