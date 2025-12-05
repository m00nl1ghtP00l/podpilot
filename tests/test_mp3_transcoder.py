"""
Tests for mp3_transcoder.py

This test suite covers:
- Utility functions (bytes conversion, time formatting/parsing)
- Bitrate calculation
- Main transcode function (with mocks)
- CLI interface
"""

import pytest
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys

# Add parent directory to path to import the module
sys.path.insert(0, str(Path(__file__).parent.parent))

from mp3_transcoder import (
    _bytes_to_mb,
    _format_time,
    _parse_time,
    _calculate_target_bitrate,
    transcode,
    main
)


class TestBytesToMb:
    """Tests for _bytes_to_mb function"""
    
    def test_bytes_to_mb_zero(self):
        """Test converting zero bytes"""
        result = _bytes_to_mb(0)
        assert result == 0.0
    
    def test_bytes_to_mb_one_mb(self):
        """Test converting 1 MB"""
        result = _bytes_to_mb(1024 * 1024)
        assert result == 1.0
    
    def test_bytes_to_mb_fractional(self):
        """Test converting fractional MB"""
        result = _bytes_to_mb(512 * 1024)  # 0.5 MB
        assert result == 0.5
    
    def test_bytes_to_mb_large_value(self):
        """Test converting large value"""
        result = _bytes_to_mb(25 * 1024 * 1024)  # 25 MB
        assert result == 25.0
    
    def test_bytes_to_mb_precision(self):
        """Test precision of conversion"""
        result = _bytes_to_mb(1536 * 1024)  # 1.5 MB
        assert result == 1.5


class TestFormatTime:
    """Tests for _format_time function"""
    
    def test_format_time_zero(self):
        """Test formatting zero seconds"""
        result = _format_time(0)
        assert result == "00:00:00"
    
    def test_format_time_seconds_only(self):
        """Test formatting seconds only"""
        result = _format_time(45)
        assert result == "00:00:45"
    
    def test_format_time_minutes(self):
        """Test formatting minutes"""
        result = _format_time(125)  # 2 minutes 5 seconds
        assert result == "00:02:05"
    
    def test_format_time_hours(self):
        """Test formatting hours"""
        result = _format_time(3665)  # 1 hour 1 minute 5 seconds
        assert result == "01:01:05"
    
    def test_format_time_large(self):
        """Test formatting large time"""
        result = _format_time(7325)  # 2 hours 2 minutes 5 seconds
        assert result == "02:02:05"
    
    def test_format_time_fractional_seconds(self):
        """Test that fractional seconds are truncated"""
        result = _format_time(45.7)
        assert result == "00:00:45"


class TestParseTime:
    """Tests for _parse_time function"""
    
    def test_parse_time_seconds_only(self):
        """Test parsing seconds only"""
        result = _parse_time("00:00:45")
        assert result == 45.0
    
    def test_parse_time_minutes(self):
        """Test parsing minutes"""
        result = _parse_time("00:02:05")
        assert abs(result - 125.0) < 0.01
    
    def test_parse_time_hours(self):
        """Test parsing hours"""
        result = _parse_time("01:01:05")
        assert abs(result - 3665.0) < 0.01
    
    def test_parse_time_with_milliseconds(self):
        """Test parsing time with milliseconds"""
        result = _parse_time("00:01:30.500")
        assert abs(result - 90.5) < 0.01
    
    def test_parse_time_zero(self):
        """Test parsing zero time"""
        result = _parse_time("00:00:00")
        assert result == 0.0


class TestCalculateTargetBitrate:
    """Tests for _calculate_target_bitrate function"""
    
    def test_calculate_bitrate_basic(self):
        """Test basic bitrate calculation"""
        # 25MB file, 60 minutes = 3600 seconds
        # Should calculate reasonable bitrate
        result = _calculate_target_bitrate(3600, 25 * 1000 * 1000)
        assert 32 <= result <= 320
    
    def test_calculate_bitrate_minimum(self):
        """Test that bitrate doesn't go below 32 kbps"""
        # Very long duration should still return at least 32
        result = _calculate_target_bitrate(100000, 1 * 1000 * 1000)
        assert result == 32
    
    def test_calculate_bitrate_maximum(self):
        """Test that bitrate doesn't exceed 320 kbps"""
        # Very short duration should cap at 320
        result = _calculate_target_bitrate(1, 100 * 1000 * 1000)
        assert result == 320
    
    def test_calculate_bitrate_typical(self):
        """Test typical use case"""
        # 25MB target, 30 minute audio (1800 seconds)
        result = _calculate_target_bitrate(1800, 25 * 1000 * 1000)
        # Should be around 100 kbps (25MB * 8 * 0.9 / 1800 / 1000)
        assert 80 <= result <= 120
    
    def test_calculate_bitrate_short_audio(self):
        """Test with short audio"""
        # 25MB target, 5 minute audio (300 seconds)
        result = _calculate_target_bitrate(300, 25 * 1000 * 1000)
        # Should be higher bitrate
        assert result > 100


class TestTranscode:
    """Tests for transcode function"""
    
    @patch('mp3_transcoder.os.path.getsize')
    def test_transcode_file_already_small_enough(self, mock_getsize):
        """Test that file under target size is not transcoded"""
        mock_getsize.return_value = 20 * 1024 * 1024  # 20MB (under 25MB)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as f:
            file_path = f.name
        
        try:
            result = transcode(file_path, target_size_mb=25, show_progress=False)
            assert result['success'] is True
            assert result['original_size_mb'] == 20.0
            assert result['new_size_mb'] == 20.0
            assert result['bitrate'] == 0  # No transcoding needed
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)
    
    @patch('mp3_transcoder.shutil.move')
    @patch('mp3_transcoder.shutil.copy2')
    @patch('mp3_transcoder.subprocess.run')
    @patch('mp3_transcoder.MP3')
    @patch('mp3_transcoder.os.path.getsize')
    @patch('mp3_transcoder.tempfile.NamedTemporaryFile')
    def test_transcode_success(self, mock_tempfile, mock_getsize, mock_mp3, mock_subprocess, mock_copy, mock_move):
        """Test successful transcoding"""
        # Setup mocks
        mock_getsize.side_effect = [30 * 1024 * 1024, 24 * 1024 * 1024]  # Original 30MB, new 24MB
        
        # Mock MP3 audio object
        mock_audio = Mock()
        mock_audio.info.length = 1800  # 30 minutes
        mock_mp3.return_value = mock_audio
        
        # Mock temp file
        mock_temp = Mock()
        mock_temp.name = '/tmp/test_output.mp3'
        mock_temp.__enter__ = Mock(return_value=mock_temp)
        mock_temp.__exit__ = Mock(return_value=None)
        mock_tempfile.return_value = mock_temp
        
        # Mock subprocess (ffmpeg success)
        mock_process = Mock()
        mock_process.returncode = 0
        mock_subprocess.return_value = mock_process
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as f:
            file_path = f.name
        
        try:
            result = transcode(file_path, target_size_mb=25, show_progress=False)
            assert result['success'] is True
            assert result['original_size_mb'] == 30.0
            assert result['new_size_mb'] == 24.0
            assert result['bitrate'] > 0
            mock_subprocess.assert_called_once()
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)
    
    @patch('mp3_transcoder.shutil.copy2')
    @patch('mp3_transcoder.subprocess.run')
    @patch('mp3_transcoder.MP3')
    @patch('mp3_transcoder.os.path.getsize')
    def test_transcode_ffmpeg_failure(self, mock_getsize, mock_mp3, mock_subprocess, mock_copy):
        """Test handling FFmpeg failure"""
        mock_getsize.return_value = 30 * 1024 * 1024  # 30MB
        
        mock_audio = Mock()
        mock_audio.info.length = 1800
        mock_mp3.return_value = mock_audio
        
        # Mock subprocess failure
        mock_process = Mock()
        mock_process.returncode = 1
        mock_process.stderr = b'FFmpeg error message'
        mock_subprocess.return_value = mock_process
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as f:
            file_path = f.name
            f.write(b'fake mp3 data')
        
        try:
            result = transcode(file_path, target_size_mb=25, show_progress=False)
            assert result['success'] is False
            assert 'error' in result
            assert 'FFmpeg error' in result['error']
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)
            # Clean up any .orig backup files
            orig_path = Path(file_path).parent / f"{Path(file_path).stem}.orig{Path(file_path).suffix}"
            if os.path.exists(orig_path):
                os.remove(orig_path)
    
    @patch('mp3_transcoder.shutil.copy2')
    @patch('mp3_transcoder.subprocess.run')
    @patch('mp3_transcoder.MP3')
    @patch('mp3_transcoder.os.path.getsize')
    def test_transcode_validation_failure(self, mock_getsize, mock_mp3, mock_subprocess, mock_copy):
        """Test handling validation failure (wrong duration)"""
        mock_getsize.side_effect = [30 * 1024 * 1024, 24 * 1024 * 1024]  # Original and transcoded sizes
        
        # Original audio
        mock_audio_orig = Mock()
        mock_audio_orig.info.length = 1800  # 30 minutes
        
        # Transcoded audio (wrong duration)
        mock_audio_transcoded = Mock()
        mock_audio_transcoded.info.length = 100  # Very different duration
        mock_mp3.side_effect = [mock_audio_orig, mock_audio_transcoded]
        
        mock_process = Mock()
        mock_process.returncode = 0
        mock_subprocess.return_value = mock_process
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as f:
            file_path = f.name
            f.write(b'fake mp3 data')
        
        try:
            result = transcode(file_path, target_size_mb=25, show_progress=False)
            assert result['success'] is False
            assert 'error' in result
            assert 'duration' in result['error'].lower()
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)
            # Clean up any .orig backup files
            orig_path = Path(file_path).parent / f"{Path(file_path).stem}.orig{Path(file_path).suffix}"
            if os.path.exists(orig_path):
                os.remove(orig_path)
    
    @patch('mp3_transcoder.os.path.getsize')
    def test_transcode_file_not_found(self, mock_getsize):
        """Test handling file not found"""
        mock_getsize.side_effect = FileNotFoundError("File not found")
        
        result = transcode("/nonexistent/file.mp3", target_size_mb=25, show_progress=False)
        assert result['success'] is False
        assert 'error' in result
    
    @patch('mp3_transcoder.shutil.move')
    @patch('mp3_transcoder.shutil.copy2')
    @patch('mp3_transcoder.subprocess.run')
    @patch('mp3_transcoder.MP3')
    @patch('mp3_transcoder.os.path.getsize')
    def test_transcode_custom_target_size(self, mock_getsize, mock_mp3, mock_subprocess, mock_copy, mock_move):
        """Test transcoding with custom target size"""
        mock_getsize.side_effect = [50 * 1024 * 1024, 10 * 1024 * 1024]  # 50MB -> 10MB
        
        mock_audio = Mock()
        mock_audio.info.length = 3600  # 1 hour
        mock_mp3.return_value = mock_audio
        
        mock_process = Mock()
        mock_process.returncode = 0
        mock_subprocess.return_value = mock_process
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as f:
            file_path = f.name
            f.write(b'fake mp3 data')
        
        try:
            result = transcode(file_path, target_size_mb=10, show_progress=False)
            assert result['success'] is True
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)
            # Clean up any .orig backup files
            orig_path = Path(file_path).parent / f"{Path(file_path).stem}.orig{Path(file_path).suffix}"
            if os.path.exists(orig_path):
                os.remove(orig_path)
    
    @patch('mp3_transcoder.shutil.move')
    @patch('mp3_transcoder.shutil.copy2')
    @patch('mp3_transcoder.subprocess.run')
    @patch('mp3_transcoder.MP3')
    @patch('mp3_transcoder.os.path.getsize')
    @patch('builtins.print')
    def test_transcode_show_progress(self, mock_print, mock_getsize, mock_mp3, mock_subprocess, mock_copy, mock_move):
        """Test that progress is shown when show_progress=True"""
        mock_getsize.side_effect = [30 * 1024 * 1024, 24 * 1024 * 1024]
        
        mock_audio = Mock()
        mock_audio.info.length = 1800
        mock_mp3.return_value = mock_audio
        
        mock_process = Mock()
        mock_process.returncode = 0
        mock_subprocess.return_value = mock_process
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as f:
            file_path = f.name
            f.write(b'fake mp3 data')
        
        try:
            result = transcode(file_path, target_size_mb=25, show_progress=True)
            assert result['success'] is True
            # Verify print was called (progress messages)
            assert mock_print.called
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)
            # Clean up any .orig backup files
            orig_path = Path(file_path).parent / f"{Path(file_path).stem}.orig{Path(file_path).suffix}"
            if os.path.exists(orig_path):
                os.remove(orig_path)


class TestMain:
    """Tests for main CLI function"""
    
    def test_main_no_input_file(self):
        """Test that main exits when no input file provided"""
        with patch('sys.argv', ['mp3_transcoder.py']):
            with pytest.raises(SystemExit):
                main()
    
    @patch('mp3_transcoder.transcode')
    @patch('mp3_transcoder.Path.exists')
    def test_main_file_not_found(self, mock_exists, mock_transcode):
        """Test handling file not found"""
        mock_exists.return_value = False
        
        with patch('sys.argv', ['mp3_transcoder.py', '/nonexistent/file.mp3']):
            with pytest.raises(SystemExit):
                main()
    
    @patch('mp3_transcoder.transcode')
    @patch('mp3_transcoder.Path.exists')
    def test_main_wrong_file_type(self, mock_exists, mock_transcode):
        """Test handling non-MP3 file"""
        mock_exists.return_value = True
        
        with patch('sys.argv', ['mp3_transcoder.py', 'file.txt']):
            with pytest.raises(SystemExit):
                main()
    
    @patch('mp3_transcoder.transcode')
    @patch('mp3_transcoder.Path.exists')
    def test_main_success(self, mock_exists, mock_transcode):
        """Test successful transcoding via CLI"""
        mock_exists.return_value = True
        mock_transcode.return_value = {
            'success': True,
            'original_size_mb': 30.0,
            'new_size_mb': 24.0,
            'bitrate': 128
        }
        
        with patch('sys.argv', ['mp3_transcoder.py', 'test.mp3', '-s', '25']):
            with patch('sys.exit'):
                main()
        
        mock_transcode.assert_called_once_with(
            Path('test.mp3'),
            target_size_mb=25,
            show_progress=True
        )
    
    @patch('mp3_transcoder.transcode')
    @patch('mp3_transcoder.Path.exists')
    def test_main_quiet_mode(self, mock_exists, mock_transcode):
        """Test quiet mode flag"""
        mock_exists.return_value = True
        mock_transcode.return_value = {'success': True}
        
        with patch('sys.argv', ['mp3_transcoder.py', 'test.mp3', '-q']):
            with patch('sys.exit'):
                main()
        
        mock_transcode.assert_called_once()
        call_args = mock_transcode.call_args
        assert call_args[1]['show_progress'] is False
    
    @patch('mp3_transcoder.transcode')
    @patch('mp3_transcoder.Path.exists')
    def test_main_transcode_failure(self, mock_exists, mock_transcode):
        """Test handling transcoding failure"""
        mock_exists.return_value = True
        mock_transcode.return_value = {
            'success': False,
            'error': 'Transcoding failed'
        }
        
        with patch('sys.argv', ['mp3_transcoder.py', 'test.mp3']):
            with pytest.raises(SystemExit):
                main()
    
    @patch('mp3_transcoder.transcode')
    @patch('mp3_transcoder.Path.exists')
    def test_main_default_target_size(self, mock_exists, mock_transcode):
        """Test that default target size (25MB) is used when -s is not provided"""
        mock_exists.return_value = True
        mock_transcode.return_value = {
            'success': True,
            'original_size_mb': 30.0,
            'new_size_mb': 24.0,
            'bitrate': 128
        }
        
        with patch('sys.argv', ['mp3_transcoder.py', 'test.mp3']):
            with patch('sys.exit'):
                main()
        
        mock_transcode.assert_called_once_with(
            Path('test.mp3'),
            target_size_mb=25,  # Default value
            show_progress=True
        )
    
    @patch('mp3_transcoder.transcode')
    @patch('mp3_transcoder.Path.exists')
    def test_main_custom_target_size(self, mock_exists, mock_transcode):
        """Test custom target size via -s option"""
        mock_exists.return_value = True
        mock_transcode.return_value = {
            'success': True,
            'original_size_mb': 50.0,
            'new_size_mb': 10.0,
            'bitrate': 64
        }
        
        with patch('sys.argv', ['mp3_transcoder.py', 'test.mp3', '-s', '10']):
            with patch('sys.exit'):
                main()
        
        mock_transcode.assert_called_once_with(
            Path('test.mp3'),
            target_size_mb=10,  # Custom value
            show_progress=True
        )
    
    @patch('mp3_transcoder.transcode')
    @patch('mp3_transcoder.Path.exists')
    def test_main_quiet_mode_with_custom_size(self, mock_exists, mock_transcode):
        """Test combining -q and -s options"""
        mock_exists.return_value = True
        mock_transcode.return_value = {'success': True}
        
        with patch('sys.argv', ['mp3_transcoder.py', 'test.mp3', '-q', '-s', '50']):
            with patch('sys.exit'):
                main()
        
        mock_transcode.assert_called_once_with(
            Path('test.mp3'),
            target_size_mb=50,
            show_progress=False  # Quiet mode
        )
    
    @patch('mp3_transcoder.transcode')
    @patch('mp3_transcoder.Path.exists')
    def test_main_long_options(self, mock_exists, mock_transcode):
        """Test that long option names work (--size, --quiet)"""
        mock_exists.return_value = True
        mock_transcode.return_value = {'success': True}
        
        with patch('sys.argv', ['mp3_transcoder.py', 'test.mp3', '--size', '30', '--quiet']):
            with patch('sys.exit'):
                main()
        
        mock_transcode.assert_called_once_with(
            Path('test.mp3'),
            target_size_mb=30,
            show_progress=False
        )

