"""
Tests for generate_lesson.py CLI interface and batch processing

This test suite covers:
- CLI argument parsing
- Single file processing
- Batch processing with date filters
- Parallel processing
- Error handling
- Utility functions
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, mock_open
from datetime import datetime
import sys
import json

# Add parent directory to path to import the module
sys.path.insert(0, str(Path(__file__).parent.parent))

from generate_lesson import (
    expand_env_vars,
    get_file_date_from_name,
    find_transcription_files,
    calculate_worker_count,
    process_single_file_for_parallel,
    main
)
from llm_providers import LLMProvider


class TestExpandEnvVars:
    """Tests for expand_env_vars function"""
    
    def test_expand_env_var_braces(self, monkeypatch):
        """Test expanding ${VAR} syntax"""
        monkeypatch.setenv("TEST_VAR", "test_value")
        result = expand_env_vars("${TEST_VAR}")
        assert result == "test_value"
    
    def test_expand_env_var_dollar(self, monkeypatch):
        """Test expanding $VAR syntax"""
        monkeypatch.setenv("TEST_VAR", "test_value")
        result = expand_env_vars("$TEST_VAR")
        assert result == "test_value"
    
    def test_expand_env_var_mixed(self, monkeypatch):
        """Test expanding multiple env vars"""
        monkeypatch.setenv("VAR1", "value1")
        monkeypatch.setenv("VAR2", "value2")
        result = expand_env_vars("${VAR1}/path/$VAR2")
        assert result == "value1/path/value2"
    
    def test_expand_env_var_not_found(self):
        """Test when env var doesn't exist"""
        result = expand_env_vars("${NONEXISTENT_VAR}")
        assert result == "${NONEXISTENT_VAR}"  # Returns original if not found
    
    def test_expand_env_var_non_string(self):
        """Test with non-string input"""
        result = expand_env_vars(123)
        assert result == 123


class TestGetFileDateFromName:
    """Tests for get_file_date_from_name function"""
    
    def test_get_file_date_valid(self):
        """Test extracting date from valid filename"""
        result = get_file_date_from_name("2024-01-15_transcript")
        assert result == datetime(2024, 1, 15)
    
    def test_get_file_date_with_underscores(self):
        """Test extracting date from filename with multiple underscores"""
        result = get_file_date_from_name("2024-12-25_some_long_filename")
        assert result == datetime(2024, 12, 25)
    
    def test_get_file_date_invalid_format(self):
        """Test with invalid date format"""
        result = get_file_date_from_name("invalid-date_format")
        assert result is None
    
    def test_get_file_date_no_underscore(self):
        """Test with filename without underscore"""
        result = get_file_date_from_name("2024-01-15")
        assert result == datetime(2024, 1, 15)
    
    def test_get_file_date_invalid_date(self):
        """Test with invalid date values"""
        result = get_file_date_from_name("2024-13-45_invalid")
        assert result is None


class TestFindTranscriptionFiles:
    """Tests for find_transcription_files function"""
    
    def test_find_transcription_files_basic(self, tmp_path):
        """Test finding basic transcription files"""
        (tmp_path / "2024-01-15_transcript.txt").write_text("Test")
        (tmp_path / "2024-01-16_transcript.txt").write_text("Test")
        
        result = find_transcription_files(tmp_path)
        
        assert len(result) == 2
        assert any("2024-01-15" in str(p) for p in result)
        assert any("2024-01-16" in str(p) for p in result)
    
    def test_find_transcription_files_skip_lesson_files(self, tmp_path):
        """Test that lesson files are skipped"""
        (tmp_path / "2024-01-15_transcript.txt").write_text("Test")
        (tmp_path / "2024-01-15_transcript_lesson.md").write_text("Lesson")
        
        result = find_transcription_files(tmp_path)
        
        assert len(result) == 1
        assert "_lesson" not in str(result[0])
    
    def test_find_transcription_files_prefer_clean(self, tmp_path):
        """Test that _transcript.txt files are preferred over .txt"""
        # Create both formatted and clean versions
        (tmp_path / "2024-01-15_some_title.txt").write_text("Formatted")
        (tmp_path / "2024-01-15_some_title_transcript.txt").write_text("Clean")
        
        result = find_transcription_files(tmp_path)
        
        # Should only return the clean version (the _transcript.txt file)
        assert len(result) == 1
        assert result[0].stem.endswith("_transcript")
    
    def test_find_transcription_files_with_from_date(self, tmp_path):
        """Test filtering with from_date"""
        (tmp_path / "2024-01-15_transcript.txt").write_text("Test")
        (tmp_path / "2024-01-20_transcript.txt").write_text("Test")
        (tmp_path / "2024-01-25_transcript.txt").write_text("Test")
        
        from_date = datetime(2024, 1, 18)
        result = find_transcription_files(tmp_path, from_date=from_date)
        
        assert len(result) == 2
        assert all("2024-01-20" in str(p) or "2024-01-25" in str(p) for p in result)
    
    def test_find_transcription_files_with_to_date(self, tmp_path):
        """Test filtering with to_date"""
        (tmp_path / "2024-01-15_transcript.txt").write_text("Test")
        (tmp_path / "2024-01-20_transcript.txt").write_text("Test")
        (tmp_path / "2024-01-25_transcript.txt").write_text("Test")
        
        to_date = datetime(2024, 1, 22)
        result = find_transcription_files(tmp_path, to_date=to_date)
        
        assert len(result) == 2
        assert all("2024-01-15" in str(p) or "2024-01-20" in str(p) for p in result)
    
    def test_find_transcription_files_with_date_range(self, tmp_path):
        """Test filtering with both from_date and to_date"""
        (tmp_path / "2024-01-15_transcript.txt").write_text("Test")
        (tmp_path / "2024-01-20_transcript.txt").write_text("Test")
        (tmp_path / "2024-01-25_transcript.txt").write_text("Test")
        
        from_date = datetime(2024, 1, 18)
        to_date = datetime(2024, 1, 22)
        result = find_transcription_files(tmp_path, from_date=from_date, to_date=to_date)
        
        assert len(result) == 1
        assert "2024-01-20" in str(result[0])
    
    def test_find_transcription_files_empty_directory(self, tmp_path):
        """Test finding files in empty directory"""
        result = find_transcription_files(tmp_path)
        assert result == []


class TestCalculateWorkerCount:
    """Tests for calculate_worker_count function"""
    
    @patch('generate_lesson.multiprocessing.cpu_count')
    def test_calculate_worker_count_auto_detect(self, mock_cpu_count):
        """Test auto-detection with 0 jobs"""
        mock_cpu_count.return_value = 8
        result = calculate_worker_count(0, 0.8)
        # Should use 80% of 8 cores = 6.4, rounded down = 6
        assert result == 6
    
    @patch('generate_lesson.multiprocessing.cpu_count')
    def test_calculate_worker_count_specified(self, mock_cpu_count):
        """Test with specified job count"""
        mock_cpu_count.return_value = 8
        result = calculate_worker_count(4, 0.8)
        assert result == 4
    
    @patch('generate_lesson.multiprocessing.cpu_count')
    def test_calculate_worker_count_capped_at_cpu(self, mock_cpu_count):
        """Test that worker count is capped at CPU count"""
        mock_cpu_count.return_value = 4
        result = calculate_worker_count(10, 0.8)  # Request more than available
        assert result == 4  # Capped at CPU count
    
    @patch('generate_lesson.multiprocessing.cpu_count')
    def test_calculate_worker_count_minimum_one(self, mock_cpu_count):
        """Test that minimum worker count is 1"""
        mock_cpu_count.return_value = 1
        result = calculate_worker_count(0, 0.1)  # Would calculate to < 1
        assert result == 1


class TestProcessSingleFileForParallel:
    """Tests for process_single_file_for_parallel function"""
    
    @patch('generate_lesson.get_provider')
    @patch('generate_lesson.load_transcription')
    @patch('generate_lesson.generate_lesson')
    @patch('generate_lesson.save_lesson')
    def test_process_single_file_success(self, mock_save, mock_generate, mock_load, mock_get_provider, tmp_path):
        """Test successful file processing"""
        mock_provider = Mock(spec=LLMProvider)
        mock_get_provider.return_value = mock_provider
        mock_load.return_value = "Test transcription"
        mock_generate.return_value = {"markdown": "Test lesson"}
        
        txt_file = tmp_path / "test_transcript.txt"
        txt_file.write_text("Test")
        
        result = process_single_file_for_parallel(
            txt_file, "ollama", {"model": "test"}, "markdown"
        )
        
        assert result == (txt_file, True, None)
        mock_generate.assert_called_once()
        mock_save.assert_called_once()
    
    @patch('generate_lesson.get_provider')
    @patch('generate_lesson.load_transcription')
    def test_process_single_file_empty_transcription(self, mock_load, mock_get_provider, tmp_path):
        """Test handling empty transcription"""
        mock_provider = Mock(spec=LLMProvider)
        mock_get_provider.return_value = mock_provider
        mock_load.return_value = ""
        
        txt_file = tmp_path / "test_transcript.txt"
        txt_file.write_text("")
        
        result = process_single_file_for_parallel(
            txt_file, "ollama", {"model": "test"}, "markdown"
        )
        
        assert result == (txt_file, False, "Empty transcription")
    
    @patch('generate_lesson.get_provider')
    @patch('generate_lesson.load_transcription')
    @patch('generate_lesson.generate_lesson')
    def test_process_single_file_error(self, mock_generate, mock_load, mock_get_provider, tmp_path):
        """Test error handling during processing"""
        mock_provider = Mock(spec=LLMProvider)
        mock_get_provider.return_value = mock_provider
        mock_load.return_value = "Test transcription"
        mock_generate.side_effect = RuntimeError("Generation failed")
        
        txt_file = tmp_path / "test_transcript.txt"
        txt_file.write_text("Test")
        
        result = process_single_file_for_parallel(
            txt_file, "ollama", {"model": "test"}, "markdown"
        )
        
        assert result == (txt_file, False, "Generation failed")


class TestMainCLI:
    """Tests for main() CLI function"""
    
    @patch('generate_lesson.load_config')
    @patch('generate_lesson.find_podcast_by_name')
    @patch('generate_lesson.load_transcription')
    @patch('generate_lesson.get_provider')
    @patch('generate_lesson.generate_lesson')
    @patch('generate_lesson.save_lesson')
    @patch('generate_lesson.get_language_adapter')
    def test_main_single_file_mode(self, mock_get_adapter, mock_save, mock_generate, 
                                   mock_get_provider, mock_load_trans, mock_find_podcast, 
                                   mock_load_config, tmp_path, capsys):
        """Test main() with single file mode"""
        # Setup mocks
        mock_config = {
            "youtube_channels": [{"channel_name_short": "test"}],
            "analysis": {"provider": "ollama", "model": "test", "base_url": "http://localhost:11434"},
            "language": "ja"
        }
        mock_load_config.return_value = mock_config
        mock_find_podcast.return_value = {"channel_name_short": "test"}
        mock_load_trans.return_value = "Test transcription"
        mock_provider = Mock(spec=LLMProvider)
        mock_provider.is_available.return_value = True
        mock_get_provider.return_value = mock_provider
        mock_generate.return_value = {"markdown": "Test lesson"}
        
        # Mock language adapter
        mock_adapter = Mock()
        mock_adapter.language_name = "Japanese"
        mock_adapter.language_code = "ja"
        mock_get_adapter.return_value = mock_adapter
        
        # Create test file
        test_file = tmp_path / "test_transcript.txt"
        test_file.write_text("Test")
        # Ensure lesson file doesn't exist (skip-existing logic)
        lesson_file = tmp_path / "test_transcript_lesson.md"
        assert not lesson_file.exists()
        
        with patch('sys.argv', ['generate_lesson.py', '--name', 'test', str(test_file)]):
            main()
        
        # Verify calls were made
        mock_generate.assert_called_once()
        mock_save.assert_called_once()
    
    @patch('generate_lesson.load_config')
    @patch('generate_lesson.find_podcast_by_name')
    @patch('generate_lesson.find_transcription_files')
    @patch('generate_lesson.load_transcription')
    @patch('generate_lesson.get_provider')
    @patch('generate_lesson.generate_lesson')
    @patch('generate_lesson.save_lesson')
    @patch('generate_lesson.get_language_adapter')
    def test_main_batch_mode(self, mock_get_adapter, mock_save, mock_generate, 
                             mock_get_provider, mock_load_trans, mock_find_files, 
                             mock_find_podcast, mock_load_config, tmp_path, capsys):
        """Test main() with batch processing mode"""
        # Setup mocks
        mock_config = {
            "youtube_channels": [{"channel_name_short": "test"}],
            "analysis": {"provider": "ollama", "model": "test", "base_url": "http://localhost:11434"},
            "language": "ja",
            "data_root": str(tmp_path)
        }
        mock_load_config.return_value = mock_config
        mock_find_podcast.return_value = {"channel_name_short": "test"}
        
        # Create audio directory structure
        audio_dir = tmp_path / "test"
        audio_dir.mkdir()
        test_file = audio_dir / "2024-01-15_transcript.txt"
        test_file.write_text("Test")
        mock_find_files.return_value = [test_file]
        
        mock_load_trans.return_value = "Test transcription"
        mock_provider = Mock(spec=LLMProvider)
        mock_provider.is_available.return_value = True
        mock_get_provider.return_value = mock_provider
        mock_generate.return_value = {"markdown": "Test lesson"}
        
        # Mock language adapter
        mock_adapter = Mock()
        mock_adapter.language_name = "Japanese"
        mock_adapter.language_code = "ja"
        mock_get_adapter.return_value = mock_adapter
        
        with patch('sys.argv', ['generate_lesson.py', '--name', 'test', '--from-date', '2024-01-01']):
            main()
        
        mock_find_files.assert_called_once()
        # With default jobs=1, it processes sequentially, not parallel
        mock_load_trans.assert_called_once()
        mock_generate.assert_called_once()
    
    @patch('generate_lesson.load_config')
    @patch('generate_lesson.find_podcast_by_name')
    @patch('generate_lesson.find_transcription_files')
    @patch('generate_lesson.get_language_adapter')
    def test_main_simulate_mode(self, mock_get_adapter, mock_find_files, mock_find_podcast, 
                                mock_load_config, tmp_path, capsys):
        """Test main() with --simulate flag"""
        mock_config = {
            "youtube_channels": [{"channel_name_short": "test"}],
            "analysis": {"provider": "ollama", "model": "test", "base_url": "http://localhost:11434"},
            "language": "ja",
            "data_root": str(tmp_path)
        }
        mock_load_config.return_value = mock_config
        mock_find_podcast.return_value = {"channel_name_short": "test"}
        
        # Create audio directory structure
        audio_dir = tmp_path / "test"
        audio_dir.mkdir()
        test_file = audio_dir / "2024-01-15_transcript.txt"
        test_file.write_text("Test")
        mock_find_files.return_value = [test_file]
        
        # Mock language adapter
        mock_adapter = Mock()
        mock_adapter.language_name = "Japanese"
        mock_adapter.language_code = "ja"
        mock_get_adapter.return_value = mock_adapter
        
        with patch('sys.argv', ['generate_lesson.py', '--name', 'test', '--simulate']):
            main()
        
        captured = capsys.readouterr()
        assert "simulate" in captured.out.lower() or "would process" in captured.out.lower() or "simulation" in captured.out.lower()
    
    @patch('generate_lesson.load_config')
    @patch('generate_lesson.find_podcast_by_name')
    @patch('generate_lesson.get_provider')
    @patch('generate_lesson.load_transcription')
    @patch('generate_lesson.get_language_adapter')
    def test_main_timeout_exits_immediately(self, mock_get_adapter, mock_load_trans, 
                                            mock_get_provider, mock_find_podcast, 
                                            mock_load_config, tmp_path):
        """Test that timeout errors cause immediate exit"""
        mock_config = {
            "youtube_channels": [{"channel_name_short": "test"}],
            "analysis": {"provider": "ollama", "model": "test", "base_url": "http://localhost:11434"},
            "language": "ja"
        }
        mock_load_config.return_value = mock_config
        mock_find_podcast.return_value = {"channel_name_short": "test"}
        
        mock_provider = Mock(spec=LLMProvider)
        mock_provider.is_available.return_value = True
        mock_provider.generate.side_effect = RuntimeError("Request timed out")
        mock_get_provider.return_value = mock_provider
        
        mock_load_trans.return_value = "Test transcription"
        
        # Mock language adapter
        mock_adapter = Mock()
        mock_adapter.language_name = "Japanese"
        mock_adapter.language_code = "ja"
        mock_get_adapter.return_value = mock_adapter
        
        test_file = tmp_path / "test_transcript.txt"
        test_file.write_text("Test")
        # Ensure lesson file doesn't exist
        lesson_file = tmp_path / "test_transcript_lesson.md"
        assert not lesson_file.exists()
        
        with patch('sys.argv', ['generate_lesson.py', '--name', 'test', str(test_file)]):
            with patch('sys.exit') as mock_exit:
                main()
                # Should exit on timeout
                mock_exit.assert_called_once_with(1)

