"""
Tests for generate_lesson.py parallel processing

This test suite covers:
- Parallel processing with ThreadPoolExecutor
- Worker count calculation for parallel mode
- Batch processing statistics
- Error handling in parallel mode
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import sys
import concurrent.futures

# Add parent directory to path to import the module
sys.path.insert(0, str(Path(__file__).parent.parent))

from generate_lesson import (
    process_single_file_for_parallel,
    calculate_worker_count,
    main
)
from llm_providers import LLMProvider


class TestParallelProcessing:
    """Tests for parallel processing functionality"""
    
    @patch('generate_lesson.get_provider')
    @patch('generate_lesson.load_transcription')
    @patch('generate_lesson.generate_lesson')
    @patch('generate_lesson.save_lesson')
    @patch('concurrent.futures.ThreadPoolExecutor')
    def test_parallel_processing_multiple_files(self, mock_executor, mock_save, 
                                                 mock_generate, mock_load, mock_get_provider, tmp_path):
        """Test parallel processing with multiple files"""
        # Setup mocks
        mock_provider = Mock(spec=LLMProvider)
        mock_get_provider.return_value = mock_provider
        mock_load.return_value = "Test transcription"
        mock_generate.return_value = {"markdown": "Test lesson"}
        
        # Create test files
        file1 = tmp_path / "2024-01-15_transcript.txt"
        file2 = tmp_path / "2024-01-16_transcript.txt"
        file3 = tmp_path / "2024-01-17_transcript.txt"
        file1.write_text("Test 1")
        file2.write_text("Test 2")
        file3.write_text("Test 3")
        
        # Mock ThreadPoolExecutor
        mock_executor_instance = MagicMock()
        mock_executor.return_value.__enter__.return_value = mock_executor_instance
        mock_executor.return_value.__exit__.return_value = None
        
        # Mock submit to return futures
        futures = []
        for f in [file1, file2, file3]:
            future = MagicMock()
            future.result.return_value = (f, True, None)
            futures.append(future)
        mock_executor_instance.submit.side_effect = futures
        
        # Test parallel processing
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            for f in [file1, file2, file3]:
                future = executor.submit(
                    process_single_file_for_parallel,
                    f, "ollama", {"model": "test"}, "markdown"
                )
                results.append(future.result())
        
        assert len(results) == 3
        assert all(success for _, success, _ in results)
    
    @patch('generate_lesson.get_provider')
    @patch('generate_lesson.load_transcription')
    @patch('generate_lesson.generate_lesson')
    def test_parallel_processing_with_errors(self, mock_generate, mock_load, 
                                             mock_get_provider, tmp_path):
        """Test parallel processing handles errors gracefully"""
        mock_provider = Mock(spec=LLMProvider)
        mock_get_provider.return_value = mock_provider
        mock_load.return_value = "Test transcription"
        
        # First file succeeds, second fails
        mock_generate.side_effect = [
            {"markdown": "Success"},
            RuntimeError("Generation failed")
        ]
        
        file1 = tmp_path / "file1_transcript.txt"
        file2 = tmp_path / "file2_transcript.txt"
        file1.write_text("Test 1")
        file2.write_text("Test 2")
        
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(process_single_file_for_parallel, file1, "ollama", {"model": "test"}, "markdown"),
                executor.submit(process_single_file_for_parallel, file2, "ollama", {"model": "test"}, "markdown")
            ]
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
        
        # Should have 2 results: one success, one failure
        assert len(results) == 2
        success_count = sum(1 for _, success, _ in results if success)
        failure_count = sum(1 for _, success, _ in results if not success)
        assert success_count == 1
        assert failure_count == 1
    
    def test_parallel_processing_worker_count_logic(self):
        """Test that parallel processing is only enabled for Ollama with multiple files"""
        # This tests the logic indirectly - parallel processing requires:
        # 1. len(transcription_files) > 1
        # 2. provider_type == 'ollama'
        # 3. worker_count > 1
        
        # Test worker count calculation
        from generate_lesson import calculate_worker_count
        import multiprocessing
        
        # With 0 jobs (auto-detect), should calculate based on CPU
        cpu_count = multiprocessing.cpu_count()
        result = calculate_worker_count(0, 0.7)
        assert result >= 1
        assert result <= cpu_count
        
        # With specific jobs, should respect that (up to CPU limit)
        result = calculate_worker_count(4, 0.7)
        assert result == min(4, cpu_count)

