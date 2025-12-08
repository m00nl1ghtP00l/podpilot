"""
Tests for local_whisper_transcribe.py

This test suite covers:
- Date parsing and file date extraction
- Date range checking
- Transcription file checking
- SRT parsing
- Transcript writing
- Japanese text segmentation
- Metadata handling
- Configuration
- Transcription function (with mocks)
- CLI interface
"""

import pytest
import os
import tempfile
import json
import argparse
from pathlib import Path
from unittest.mock import Mock, patch, mock_open, MagicMock
from datetime import datetime, timezone
import sys

# Add parent directory to path to import the module
sys.path.insert(0, str(Path(__file__).parent.parent))

from local_whisper_transcribe import (
    parse_date_arg,
    get_file_date,
    is_date_in_range,
    check_existing_transcription,
    find_whisper_executable,
    parse_srt_to_segments,
    write_clean_transcript,
    segment_japanese_text,
    get_video_id_from_metadata,
    load_metadata,
    get_config_from_environment,
    find_audio_path_by_id,
    transcribe_audio,
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
    
    def test_get_file_date_empty(self):
        """Test extracting date from empty filename"""
        result = get_file_date('')
        assert result is None


class TestIsDateInRange:
    """Tests for is_date_in_range function"""
    
    def test_date_in_range_both_limits(self):
        """Test date within both limits"""
        date = datetime(2024, 1, 15, tzinfo=timezone.utc)
        from_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
        to_date = datetime(2024, 1, 31, tzinfo=timezone.utc)
        assert is_date_in_range(date, from_date, to_date) is True
    
    def test_date_before_from_date(self):
        """Test date before from_date"""
        date = datetime(2024, 1, 5, tzinfo=timezone.utc)
        from_date = datetime(2024, 1, 10, tzinfo=timezone.utc)
        to_date = datetime(2024, 1, 31, tzinfo=timezone.utc)
        assert is_date_in_range(date, from_date, to_date) is False
    
    def test_date_after_to_date(self):
        """Test date after to_date"""
        date = datetime(2024, 2, 5, tzinfo=timezone.utc)
        from_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
        to_date = datetime(2024, 1, 31, tzinfo=timezone.utc)
        assert is_date_in_range(date, from_date, to_date) is False
    
    def test_date_no_limits(self):
        """Test date with no limits"""
        date = datetime(2024, 1, 15, tzinfo=timezone.utc)
        assert is_date_in_range(date, None, None) is True
    
    def test_date_none(self):
        """Test with None date"""
        assert is_date_in_range(None, None, None) is False


class TestCheckExistingTranscription:
    """Tests for check_existing_transcription function"""
    
    def test_check_existing_transcription_json_exists(self, tmp_path):
        """Test when JSON transcription exists"""
        audio_file = tmp_path / "test.mp3"
        audio_file.touch()
        json_file = tmp_path / "test.json"
        json_file.touch()
        
        result = check_existing_transcription(str(audio_file))
        assert result is True
    
    def test_check_existing_transcription_txt_exists(self, tmp_path):
        """Test when TXT transcription exists"""
        audio_file = tmp_path / "test.mp3"
        audio_file.touch()
        txt_file = tmp_path / "test.txt"
        txt_file.touch()
        
        result = check_existing_transcription(str(audio_file))
        assert result is True
    
    def test_check_existing_transcription_none_exists(self, tmp_path):
        """Test when no transcription exists"""
        audio_file = tmp_path / "test.mp3"
        audio_file.touch()
        
        result = check_existing_transcription(str(audio_file))
        assert result is False


class TestFindWhisperExecutable:
    """Tests for find_whisper_executable function"""
    
    @patch('local_whisper_transcribe.shutil.which')
    def test_find_whisper_executable_found(self, mock_which):
        """Test finding whisper executable"""
        mock_which.return_value = '/usr/local/bin/whisper-cli'
        result = find_whisper_executable()
        assert result == '/usr/local/bin/whisper-cli'
    
    @patch('local_whisper_transcribe.shutil.which')
    def test_find_whisper_executable_not_found(self, mock_which):
        """Test when whisper executable not found"""
        mock_which.return_value = None
        result = find_whisper_executable()
        assert result is None


class TestParseSrtToSegments:
    """Tests for parse_srt_to_segments function"""
    
    def test_parse_srt_to_segments_valid(self, tmp_path):
        """Test parsing valid SRT file"""
        srt_file = tmp_path / "test.srt"
        srt_content = """1
00:00:00,000 --> 00:00:05,000
First segment text

2
00:00:05,000 --> 00:00:10,000
Second segment text
"""
        srt_file.write_text(srt_content, encoding='utf-8')
        
        result = parse_srt_to_segments(str(srt_file))
        assert len(result) == 2
        assert result[0]['start'] == '00:00:00.000'
        assert result[0]['end'] == '00:00:05.000'
        assert result[0]['text'] == 'First segment text'
    
    def test_parse_srt_to_segments_empty(self, tmp_path):
        """Test parsing empty SRT file"""
        srt_file = tmp_path / "test.srt"
        srt_file.write_text('', encoding='utf-8')
        
        result = parse_srt_to_segments(str(srt_file))
        assert result == []
    
    def test_parse_srt_to_segments_invalid_format(self, tmp_path):
        """Test parsing invalid SRT format"""
        srt_file = tmp_path / "test.srt"
        srt_file.write_text('Invalid content', encoding='utf-8')
        
        result = parse_srt_to_segments(str(srt_file))
        assert result == []


class TestWriteCleanTranscript:
    """Tests for write_clean_transcript function"""
    
    def test_write_clean_transcript_removes_timestamps(self, tmp_path):
        """Test that clean transcript removes timestamp lines"""
        text_file = tmp_path / "test.txt"
        text_content = """[00:00:00.000] https://www.youtube.com/watch?v=abc123&t=0
This is the transcript text.

[00:00:05.000] https://www.youtube.com/watch?v=abc123&t=5
More text here.
"""
        text_file.write_text(text_content, encoding='utf-8')
        
        write_clean_transcript(text_file)
        
        clean_file = tmp_path / "test_transcript.txt"
        assert clean_file.exists()
        content = clean_file.read_text(encoding='utf-8')
        assert 'youtube.com' not in content
        assert 'This is the transcript text.' in content


class TestSegmentJapaneseText:
    """Tests for segment_japanese_text function"""
    
    def test_segment_japanese_text_simple(self):
        """Test segmenting simple Japanese text"""
        text = "これはテストです。これは別の文です。"
        result = segment_japanese_text(text)
        assert len(result) == 2
        assert result[0] == "これはテストです。"
        assert result[1] == "これは別の文です。"
    
    def test_segment_japanese_text_with_question(self):
        """Test segmenting text with question mark"""
        text = "これは何ですか？これは答えです。"
        result = segment_japanese_text(text)
        assert len(result) == 2
        assert "？" in result[0]
    
    def test_segment_japanese_text_multiline(self):
        """Test segmenting multiline text"""
        text = "これは最初の文です。\nこれは二番目の文です。"
        result = segment_japanese_text(text)
        assert len(result) == 2
    
    def test_segment_japanese_text_empty(self):
        """Test segmenting empty text"""
        result = segment_japanese_text('')
        assert result == []
    
    def test_segment_japanese_text_no_punctuation(self):
        """Test segmenting text without punctuation"""
        text = "これは句読点のない文です"
        result = segment_japanese_text(text)
        assert len(result) == 1
        assert result[0] == "これは句読点のない文です"


class TestGetVideoIdFromMetadata:
    """Tests for get_video_id_from_metadata function"""
    
    def test_get_video_id_from_metadata_found(self, tmp_path):
        """Test extracting video ID from metadata"""
        # The function looks for metadata in parent.parent / "{parent.name}.json"
        # So if audio_file is /tmp/audio/test_file.mp3, it looks for /tmp/audio.json
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()
        audio_file = audio_dir / "test_file.mp3"
        audio_file.touch()
        
        metadata_file = tmp_path / "audio.json"
        metadata = {
            'videos': [
                {
                    'clean_filename': 'test_file',
                    'id': 'abc123'
                }
            ]
        }
        metadata_file.write_text(json.dumps(metadata), encoding='utf-8')
        
        result = get_video_id_from_metadata(str(audio_file))
        assert result == 'abc123'
    
    def test_get_video_id_from_metadata_youtube_url(self, tmp_path):
        """Test extracting ID from YouTube URL in metadata"""
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()
        audio_file = audio_dir / "test_file.mp3"
        audio_file.touch()
        
        metadata_file = tmp_path / "audio.json"
        metadata = {
            'videos': [
                {
                    'clean_filename': 'test_file',
                    'id': 'https://www.youtube.com/watch?v=xyz789'
                }
            ]
        }
        metadata_file.write_text(json.dumps(metadata), encoding='utf-8')
        
        result = get_video_id_from_metadata(str(audio_file))
        assert result == 'xyz789'
    
    def test_get_video_id_from_metadata_not_found(self, tmp_path):
        """Test when video ID not found"""
        audio_file = tmp_path / "audio" / "test_file.mp3"
        audio_file.parent.mkdir()
        audio_file.touch()
        
        metadata_file = tmp_path / "test.json"
        metadata = {'videos': []}
        metadata_file.write_text(json.dumps(metadata), encoding='utf-8')
        
        result = get_video_id_from_metadata(str(audio_file))
        assert result is None


class TestLoadMetadata:
    """Tests for load_metadata function"""
    
    def test_load_metadata_valid(self, tmp_path):
        """Test loading valid metadata file"""
        metadata_file = tmp_path / "metadata.json"
        metadata = {
            'videos': [
                {'id': 'abc123', 'title': 'Test Video'}
            ]
        }
        metadata_file.write_text(json.dumps(metadata), encoding='utf-8')
        
        result = load_metadata(str(metadata_file))
        assert result == metadata
    
    def test_load_metadata_missing_videos_key(self, tmp_path):
        """Test loading metadata without videos key"""
        metadata_file = tmp_path / "metadata.json"
        metadata = {'other_key': 'value'}
        metadata_file.write_text(json.dumps(metadata), encoding='utf-8')
        
        with patch('builtins.print'):
            result = load_metadata(str(metadata_file))
        assert result is None
    
    def test_load_metadata_nonexistent_file(self):
        """Test loading nonexistent metadata file"""
        with patch('builtins.print'):
            result = load_metadata('/nonexistent/file.json')
        assert result is None


class TestGetConfigFromEnvironment:
    """Tests for get_config_from_environment function"""
    
    @patch('local_whisper_transcribe.os.environ.get')
    @patch('local_whisper_transcribe.find_whisper_executable')
    def test_get_config_from_environment_full(self, mock_find, mock_get):
        """Test getting full config from environment"""
        mock_get.return_value = '/path/to/model.bin'
        mock_find.return_value = '/usr/bin/whisper-cli'
        
        with patch('local_whisper_transcribe.os.path.exists', return_value=True):
            result = get_config_from_environment()
        
        assert result['model_path'] == '/path/to/model.bin'
        assert result['executable'] == '/usr/bin/whisper-cli'
    
    @patch('local_whisper_transcribe.os.environ.get')
    @patch('local_whisper_transcribe.find_whisper_executable')
    def test_get_config_from_environment_partial(self, mock_find, mock_get):
        """Test getting partial config"""
        mock_get.return_value = None
        mock_find.return_value = '/usr/bin/whisper-cli'
        
        result = get_config_from_environment()
        
        assert 'model_path' not in result
        assert result['executable'] == '/usr/bin/whisper-cli'


class TestFindAudioPathById:
    """Tests for find_audio_path_by_id function"""
    
    def test_find_audio_path_by_id_found(self):
        """Test finding audio path by ID"""
        metadata = {
            'videos': [
                {'id': 'abc123', 'clean_filename': 'test_file'},
                {'id': 'def456', 'clean_filename': 'other_file'}
            ]
        }
        
        result = find_audio_path_by_id(metadata, 'abc123', '/audio/dir')
        assert result == '/audio/dir/test_file.mp3'
    
    def test_find_audio_path_by_id_not_found(self):
        """Test when ID not found"""
        metadata = {
            'videos': [
                {'id': 'abc123', 'clean_filename': 'test_file'}
            ]
        }
        
        result = find_audio_path_by_id(metadata, 'nonexistent', '/audio/dir')
        assert result is None
    
    def test_find_audio_path_by_id_no_clean_filename(self):
        """Test when clean_filename is missing"""
        metadata = {
            'videos': [
                {'id': 'abc123'}
            ]
        }
        
        result = find_audio_path_by_id(metadata, 'abc123', '/audio/dir')
        assert result is None


class TestTranscribeAudio:
    """Tests for transcribe_audio function"""
    
    @patch('local_whisper_transcribe.subprocess.run')
    @patch('local_whisper_transcribe.parse_srt_to_segments')
    @patch('local_whisper_transcribe.write_clean_transcript')
    @patch('local_whisper_transcribe.get_video_id_from_metadata')
    def test_transcribe_audio_success(self, mock_get_id, mock_write_clean, mock_parse, mock_subprocess, tmp_path):
        """Test successful transcription"""
        audio_file = tmp_path / "test.mp3"
        audio_file.touch()
        srt_file = tmp_path / "test.srt"
        srt_file.touch()
        
        mock_subprocess.return_value = Mock(returncode=0, stderr='')
        mock_parse.return_value = [
            {'start': '00:00:00.000', 'end': '00:00:05.000', 'text': 'Test transcript'}
        ]
        mock_get_id.return_value = None
        
        config = {
            'executable': '/usr/bin/whisper-cli',
            'model_path': '/path/to/model.bin'
        }
        
        with patch('builtins.print'):
            result = transcribe_audio(str(audio_file), config, 'ja', None, None)
        
        assert result is True
        mock_subprocess.assert_called_once()
    
    @patch('local_whisper_transcribe.subprocess.run')
    def test_transcribe_audio_subprocess_failure(self, mock_subprocess, tmp_path):
        """Test transcription failure due to subprocess error"""
        audio_file = tmp_path / "test.mp3"
        audio_file.touch()
        
        mock_subprocess.return_value = Mock(returncode=1, stderr='Error message')
        
        config = {
            'executable': '/usr/bin/whisper-cli',
            'model_path': '/path/to/model.bin'
        }
        
        with patch('builtins.print'):
            result = transcribe_audio(str(audio_file), config, 'ja', None, None)
        
        assert result is False
    
    @patch('local_whisper_transcribe.subprocess.run')
    def test_transcribe_audio_no_executable(self, mock_subprocess, tmp_path):
        """Test transcription failure when executable missing"""
        audio_file = tmp_path / "test.mp3"
        audio_file.touch()
        
        config = {
            'executable': None,
            'model_path': '/path/to/model.bin'
        }
        
        with patch('builtins.print'):
            result = transcribe_audio(str(audio_file), config, 'ja', None, None)
        
        assert result is False
    
    @patch('local_whisper_transcribe.subprocess.run')
    @patch('local_whisper_transcribe.Path.exists')
    def test_transcribe_audio_no_srt_file(self, mock_exists, mock_subprocess, tmp_path):
        """Test transcription failure when SRT file not created"""
        audio_file = tmp_path / "test.mp3"
        audio_file.touch()
        
        mock_subprocess.return_value = Mock(returncode=0, stderr='')
        mock_exists.return_value = False  # SRT file doesn't exist
        
        config = {
            'executable': '/usr/bin/whisper-cli',
            'model_path': '/path/to/model.bin'
        }
        
        with patch('builtins.print'):
            result = transcribe_audio(str(audio_file), config, 'ja', None, None)
        
        assert result is False


class TestMain:
    """Tests for main CLI function"""
    
    @patch('local_whisper_transcribe.get_config_from_environment')
    def test_main_no_executable(self, mock_config):
        """Test error when executable not found"""
        mock_config.return_value = {}
        
        with patch('sys.argv', ['local_whisper_transcribe.py', '-a', '/audio/dir']):
            with patch('builtins.print'):
                with pytest.raises(SystemExit):
                    main()
    
    @patch('local_whisper_transcribe.get_config_from_environment')
    def test_main_no_model_path(self, mock_config):
        """Test error when model path not specified"""
        mock_config.return_value = {'executable': '/usr/bin/whisper-cli'}
        
        with patch('sys.argv', ['local_whisper_transcribe.py', '-a', '/audio/dir']):
            with patch('builtins.print'):
                with pytest.raises(SystemExit):
                    main()
    
    @patch('local_whisper_transcribe.transcribe_audio')
    @patch('local_whisper_transcribe.get_config_from_environment')
    def test_main_single_file(self, mock_config, mock_transcribe):
        """Test transcribing single file"""
        mock_config.return_value = {
            'executable': '/usr/bin/whisper-cli',
            'model_path': '/path/to/model.bin'
        }
        mock_transcribe.return_value = True
        
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as f:
            file_path = f.name
        
        try:
            with patch('sys.argv', ['local_whisper_transcribe.py', '--single-file', file_path]):
                with patch('builtins.print'):
                    with pytest.raises(SystemExit) as exc_info:
                        main()
                    assert exc_info.value.code == 0
            
            mock_transcribe.assert_called_once()
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)
    
    @patch('local_whisper_transcribe.transcribe_audio')
    @patch('local_whisper_transcribe.get_config_from_environment')
    def test_main_single_file_not_found(self, mock_config, mock_transcribe):
        """Test error when single file not found"""
        mock_config.return_value = {
            'executable': '/usr/bin/whisper-cli',
            'model_path': '/path/to/model.bin'
        }
        
        with patch('sys.argv', ['local_whisper_transcribe.py', '--single-file', '/nonexistent/file.mp3']):
            with patch('builtins.print'):
                with pytest.raises(SystemExit):
                    main()
    
    @patch('local_whisper_transcribe.transcribe_audio')
    @patch('local_whisper_transcribe.find_audio_path_by_id')
    @patch('local_whisper_transcribe.load_metadata')
    @patch('local_whisper_transcribe.get_config_from_environment')
    def test_main_id_option(self, mock_config, mock_load, mock_find, mock_transcribe):
        """Test transcribing by video ID"""
        mock_config.return_value = {
            'executable': '/usr/bin/whisper-cli',
            'model_path': '/path/to/model.bin'
        }
        mock_load.return_value = {'videos': []}
        mock_find.return_value = '/audio/test.mp3'
        mock_transcribe.return_value = True
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a minimal config pointing data_root to tmpdir and podcast name "metadata"
            config_file = Path(tmpdir) / "config.json"
            config_data = {
                "data_root": str(tmpdir),
                "youtube_channels": [
                    {
                        "channel_name_short": "metadata",
                        "channel_name_long": "Test Podcast",
                        "channel_id": "UC1234567890123456789012"
                    }
                ]
            }
            config_file.write_text(json.dumps(config_data), encoding='utf-8')
            
            metadata_file = Path(tmpdir) / "metadata.json"
            metadata_file.write_text('{"videos": []}', encoding='utf-8')
            
            audio_file = Path(tmpdir) / "test.mp3"
            audio_file.touch()
            
            with patch('sys.argv', ['local_whisper_transcribe.py', '--id', 'abc123',
                                    '--name', 'metadata', '--config', str(config_file),
                                    '-a', tmpdir]):
                with patch('builtins.print'):
                    with patch('local_whisper_transcribe.Path.exists', return_value=True):
                        with pytest.raises(SystemExit) as exc_info:
                            main()
                        assert exc_info.value.code == 0
            
            mock_transcribe.assert_called_once()
    
    @patch('local_whisper_transcribe.get_config_from_environment')
    def test_main_id_missing_required_args(self, mock_config):
        """Test error when --id used without required args"""
        mock_config.return_value = {
            'executable': '/usr/bin/whisper-cli',
            'model_path': '/path/to/model.bin'
        }
        
        with patch('sys.argv', ['local_whisper_transcribe.py', '--id', 'abc123']):
            with patch('builtins.print'):
                with pytest.raises(SystemExit):
                    main()
    
    @patch('local_whisper_transcribe.get_config_from_environment')
    def test_main_missing_audio_dir(self, mock_config):
        """Test error when -a not provided"""
        mock_config.return_value = {
            'executable': '/usr/bin/whisper-cli',
            'model_path': '/path/to/model.bin'
        }
        
        with patch('sys.argv', ['local_whisper_transcribe.py']):
            with patch('builtins.print'):
                with pytest.raises(SystemExit):
                    main()
    
    @patch('local_whisper_transcribe.transcribe_audio')
    @patch('local_whisper_transcribe.get_file_date')
    @patch('local_whisper_transcribe.is_date_in_range')
    @patch('local_whisper_transcribe.check_existing_transcription')
    @patch('local_whisper_transcribe.get_config_from_environment')
    def test_main_date_filtering(self, mock_config, mock_check, mock_in_range, mock_get_date, mock_transcribe):
        """Test date filtering"""
        mock_config.return_value = {
            'executable': '/usr/bin/whisper-cli',
            'model_path': '/path/to/model.bin'
        }
        mock_get_date.return_value = datetime(2024, 1, 15, tzinfo=timezone.utc)
        mock_in_range.return_value = True  # File is in range
        mock_check.return_value = False  # Not already transcribed
        mock_transcribe.return_value = True
        
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_file = Path(tmpdir) / "2024-01-15_test.mp3"
            audio_file.touch()
            
            with patch('sys.argv', ['local_whisper_transcribe.py', '-a', tmpdir, 
                                    '--from-date', '2024-01-01', '--to-date', '2024-01-31']):
                with patch('builtins.print'):
                    with patch('local_whisper_transcribe.tqdm') as mock_tqdm:
                        # Mock tqdm to iterate over files
                        def tqdm_side_effect(iterable, **kwargs):
                            return iterable
                        mock_tqdm.side_effect = tqdm_side_effect
                        try:
                            main()
                        except SystemExit as e:
                            # main() exits with 0 on success
                            assert e.code == 0
            
            # Verify transcribe was called since file is in range and not transcribed
            mock_transcribe.assert_called()
    
    @patch('local_whisper_transcribe.transcribe_audio')
    @patch('local_whisper_transcribe.get_config_from_environment')
    def test_main_custom_model_path(self, mock_config, mock_transcribe):
        """Test custom model path via CLI"""
        config_dict = {
            'executable': '/usr/bin/whisper-cli'
        }
        mock_config.return_value = config_dict
        mock_transcribe.return_value = True
        
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_file = Path(tmpdir) / "test.mp3"
            audio_file.touch()
            
            with patch('sys.argv', ['local_whisper_transcribe.py', '--single-file', str(audio_file),
                                    '--model-path', '/custom/path/model.bin']):
                with patch('builtins.print'):
                    with pytest.raises(SystemExit) as exc_info:
                        main()
                    assert exc_info.value.code == 0
            
            # Verify config was updated with model path
            assert config_dict.get('model_path') == '/custom/path/model.bin'
            mock_transcribe.assert_called_once()
    
    @patch('local_whisper_transcribe.transcribe_audio')
    @patch('local_whisper_transcribe.get_config_from_environment')
    def test_main_custom_language(self, mock_config, mock_transcribe):
        """Test custom language option"""
        mock_config.return_value = {
            'executable': '/usr/bin/whisper-cli',
            'model_path': '/path/to/model.bin'
        }
        mock_transcribe.return_value = True
        
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as f:
            file_path = f.name
        
        try:
            with patch('sys.argv', ['local_whisper_transcribe.py', '--single-file', file_path,
                                    '--language', 'en']):
                with patch('builtins.print'):
                    with pytest.raises(SystemExit) as exc_info:
                        main()
                    assert exc_info.value.code == 0
            
            # Verify transcribe was called with custom language
            mock_transcribe.assert_called_once()
            call_args = mock_transcribe.call_args
            assert call_args[0][2] == 'en'  # language is third argument
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)
    
    @patch('local_whisper_transcribe.transcribe_audio')
    @patch('local_whisper_transcribe.get_file_date')
    @patch('local_whisper_transcribe.is_date_in_range')
    @patch('local_whisper_transcribe.check_existing_transcription')
    @patch('local_whisper_transcribe.get_config_from_environment')
    def test_main_retranscribe(self, mock_config, mock_check, mock_in_range, mock_get_date, mock_transcribe):
        """Test --retranscribe option"""
        mock_config.return_value = {
            'executable': '/usr/bin/whisper-cli',
            'model_path': '/path/to/model.bin'
        }
        mock_get_date.return_value = datetime(2024, 1, 15, tzinfo=timezone.utc)
        mock_in_range.return_value = True
        mock_check.return_value = True  # Already transcribed, but retranscribe=True
        mock_transcribe.return_value = True
        
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_file = Path(tmpdir) / "2024-01-15_test.mp3"
            audio_file.touch()
            
            with patch('sys.argv', ['local_whisper_transcribe.py', '-a', tmpdir, '--retranscribe']):
                with patch('builtins.print'):
                    with patch('local_whisper_transcribe.tqdm') as mock_tqdm:
                        def tqdm_side_effect(iterable, **kwargs):
                            return iterable
                        mock_tqdm.side_effect = tqdm_side_effect
                        try:
                            main()
                        except SystemExit as e:
                            assert e.code == 0
            
            # Verify transcribe was called even though file was already transcribed
            mock_transcribe.assert_called()
    
    @patch('local_whisper_transcribe.transcribe_audio')
    @patch('local_whisper_transcribe.get_config_from_environment')
    def test_main_test_duration(self, mock_config, mock_transcribe):
        """Test --test-duration option"""
        mock_config.return_value = {
            'executable': '/usr/bin/whisper-cli',
            'model_path': '/path/to/model.bin'
        }
        mock_transcribe.return_value = True
        
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as f:
            file_path = f.name
        
        try:
            with patch('sys.argv', ['local_whisper_transcribe.py', '--single-file', file_path,
                                    '--test-duration', '10.5']):
                with patch('builtins.print'):
                    with pytest.raises(SystemExit) as exc_info:
                        main()
                    assert exc_info.value.code == 0
            
            # Verify transcribe was called with test_duration
            mock_transcribe.assert_called_once()
            call_args = mock_transcribe.call_args
            assert call_args[0][3] == 10.5  # test_duration is fourth argument
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)
    
    @patch('local_whisper_transcribe.transcribe_audio')
    @patch('local_whisper_transcribe.load_metadata')
    @patch('local_whisper_transcribe.get_file_date')
    @patch('local_whisper_transcribe.is_date_in_range')
    @patch('local_whisper_transcribe.check_existing_transcription')
    @patch('local_whisper_transcribe.get_config_from_environment')
    def test_main_metadata_file(self, mock_config, mock_check, mock_in_range, mock_get_date, mock_load, mock_transcribe):
        """Test metadata-based filtering using config + name"""
        mock_config.return_value = {
            'executable': '/usr/bin/whisper-cli',
            'model_path': '/path/to/model.bin'
        }
        mock_load.return_value = {
            'videos': [
                {'clean_filename': '2024-01-15_test'}
            ]
        }
        mock_get_date.return_value = datetime(2024, 1, 15, tzinfo=timezone.utc)
        mock_in_range.return_value = True
        mock_check.return_value = False
        mock_transcribe.return_value = True
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a minimal config pointing data_root to tmpdir and podcast name "test"
            config_file = Path(tmpdir) / "config.json"
            config_data = {
                "data_root": str(tmpdir),
                "youtube_channels": [
                    {
                        "channel_name_short": "test",
                        "channel_name_long": "Test Podcast",
                        "channel_id": "UC1234567890123456789012"
                    }
                ]
            }
            config_file.write_text(json.dumps(config_data), encoding='utf-8')
            
            metadata_file = Path(tmpdir) / "test.json"
            metadata_file.write_text('{"videos": [{"clean_filename": "2024-01-15_test"}]}', encoding='utf-8')
            
            audio_file = Path(tmpdir) / "2024-01-15_test.mp3"
            audio_file.touch()
            
            with patch('sys.argv', ['local_whisper_transcribe.py', '-a', tmpdir,
                                    '--name', 'test', '--config', str(config_file)]):
                with patch('builtins.print'):
                    with patch('local_whisper_transcribe.tqdm') as mock_tqdm:
                        def tqdm_side_effect(iterable, **kwargs):
                            return iterable
                        mock_tqdm.side_effect = tqdm_side_effect
                        try:
                            main()
                        except SystemExit as e:
                            assert e.code == 0
            
            mock_load.assert_called_once_with(str(metadata_file))
            mock_transcribe.assert_called()

