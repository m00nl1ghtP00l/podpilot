"""
Tests for batch processing features in generate_lesson.py

This test suite covers:
- Finding transcription files
- Extracting dates from filenames
- Calculating worker count
- Processing single files (parallel mode)
- Batch processing
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import sys

# Add parent directory to path to import the module
sys.path.insert(0, str(Path(__file__).parent.parent))

from generate_lesson import (
    find_transcription_files,
    get_file_date_from_name,
    calculate_worker_count,
    process_single_file_for_parallel,
    main
)


class TestFindTranscriptionFiles:
    """Tests for find_transcription_files function"""
    
    def test_find_transcription_files_basic(self, tmp_path):
        """Test finding transcription files"""
        # Create test files
        (tmp_path / "2024-01-15_transcript.txt").write_text("Test")
        (tmp_path / "2024-01-16_transcript.txt").write_text("Test")
        (tmp_path / "2024-01-15_transcript_lesson.json").write_text("{}")  # Should be skipped
        
        result = find_transcription_files(tmp_path)
        
        assert len(result) == 2
        assert any("2024-01-15" in str(p) for p in result)
        assert any("2024-01-16" in str(p) for p in result)
    
    def test_find_transcription_files_skip_lesson_files(self, tmp_path):
        """Test that lesson files are skipped"""
        (tmp_path / "2024-01-15_transcript.txt").write_text("Test")
        (tmp_path / "2024-01-15_transcript_lesson.txt").write_text("Test")
        
        result = find_transcription_files(tmp_path)
        
        assert len(result) == 1
        assert "_lesson" not in str(result[0])
    
    def test_find_transcription_files_with_date_filter(self, tmp_path):
        """Test finding files with date filtering"""
        (tmp_path / "2024-01-15_transcript.txt").write_text("Test")
        (tmp_path / "2024-01-20_transcript.txt").write_text("Test")
        (tmp_path / "2024-01-25_transcript.txt").write_text("Test")
        
        from_date = datetime(2024, 1, 18)
        to_date = datetime(2024, 1, 22)
        
        result = find_transcription_files(tmp_path, from_date=from_date, to_date=to_date)
        
        assert len(result) == 1
        assert "2024-01-20" in str(result[0])


class TestGetFileDateFromName:
    """Tests for get_file_date_from_name function"""
    
    def test_get_file_date_from_name_valid(self):
        """Test extracting date from valid filename"""
        result = get_file_date_from_name("2024-01-15_transcript")
        
        assert result == datetime(2024, 1, 15)
    
    def test_get_file_date_from_name_invalid(self):
        """Test extracting date from invalid filename"""
        result = get_file_date_from_name("invalid_filename")
        
        assert result is None
    
    def test_get_file_date_from_name_no_date(self):
        """Test extracting date from filename without date"""
        result = get_file_date_from_name("transcript")
        
        assert result is None


class TestCalculateWorkerCount:
    """Tests for calculate_worker_count function"""
    
    @patch('generate_lesson.multiprocessing.cpu_count')
    def test_calculate_worker_count_auto(self, mock_cpu_count):
        """Test auto worker count calculation"""
        mock_cpu_count.return_value = 8
        
        result = calculate_worker_count(0, 0.7)
        
        # Should be 70% of 8 = 5.6, rounded down to 5, but min is 1
        assert result >= 1
        assert result <= 8
    
    @patch('generate_lesson.multiprocessing.cpu_count')
    def test_calculate_worker_count_specified(self, mock_cpu_count):
        """Test specified worker count"""
        mock_cpu_count.return_value = 8
        
        result = calculate_worker_count(4, 0.7)
        
        assert result == 4
    
    @patch('generate_lesson.multiprocessing.cpu_count')
    def test_calculate_worker_count_capped(self, mock_cpu_count):
        """Test worker count is capped at CPU count"""
        mock_cpu_count.return_value = 4
        
        result = calculate_worker_count(10, 0.7)
        
        assert result == 4
    
    @patch('generate_lesson.multiprocessing.cpu_count')
    def test_calculate_worker_count_minimum(self, mock_cpu_count):
        """Test worker count has minimum of 1"""
        mock_cpu_count.return_value = 1
        
        result = calculate_worker_count(0, 0.1)
        
        assert result == 1


class TestProcessSingleFile:
    """Tests for process_single_file function"""
    
    @patch('generate_lesson.get_provider')
    @patch('generate_lesson.load_transcription')
    @patch('generate_lesson.generate_lesson')
    @patch('generate_lesson.save_lesson')
    def test_process_single_file_success(self, mock_save, mock_generate, mock_load, mock_get_provider, tmp_path):
        """Test successful processing of single file"""
        txt_file = tmp_path / "2024-01-15_transcript.txt"
        txt_file.write_text("Test transcription")
        
        mock_provider = Mock()
        mock_get_provider.return_value = mock_provider
        mock_load.return_value = "Test transcription"
        mock_generate.return_value = {"vocabulary": [], "summary": "Test"}
        
        result = process_single_file_for_parallel(txt_file, "ollama", {"model": "test"}, "json")
        
        assert result[0] == txt_file
        assert result[1] is True
        assert result[2] is None
        mock_generate.assert_called_once()
        mock_save.assert_called_once()
    
    @patch('process_lessons.get_provider')
    @patch('process_lessons.load_transcription')
    def test_process_single_file_empty_transcription(self, mock_load, mock_get_provider, tmp_path):
        """Test processing file with empty transcription"""
        txt_file = tmp_path / "2024-01-15_transcript.txt"
        
        mock_provider = Mock()
        mock_get_provider.return_value = mock_provider
        mock_load.return_value = ""
        
        result = process_single_file_for_parallel(txt_file, "ollama", {"model": "test"}, "json")
        
        assert result[0] == txt_file
        assert result[1] is False
        assert "Empty transcription" in result[2]
    
    @patch('process_lessons.get_provider')
    @patch('process_lessons.load_transcription')
    def test_process_single_file_error(self, mock_load, mock_get_provider, tmp_path):
        """Test processing file with error"""
        txt_file = tmp_path / "2024-01-15_transcript.txt"
        
        mock_provider = Mock()
        mock_get_provider.return_value = mock_provider
        mock_load.side_effect = Exception("Test error")
        
        result = process_single_file_for_parallel(txt_file, "ollama", {"model": "test"}, "json")
        
        assert result[0] == txt_file
        assert result[1] is False
        assert "Test error" in result[2]

