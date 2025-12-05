"""
Tests for generate_lesson.py

This test suite covers:
- Loading transcriptions from files
- Generating lessons from transcriptions
- Saving lessons in different formats
- Formatting lessons as markdown
- CLI interface
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, mock_open, MagicMock
import sys

# Add parent directory to path to import the module
sys.path.insert(0, str(Path(__file__).parent.parent))

from generate_lesson import (
    load_transcription,
    generate_lesson,
    save_lesson,
    format_lesson_markdown,
    main
)
from llm_providers import LLMProvider


class TestLoadTranscription:
    """Tests for load_transcription function"""
    
    def test_load_transcription_from_txt(self, tmp_path):
        """Test loading transcription from .txt file"""
        txt_file = tmp_path / "transcript.txt"
        txt_file.write_text("This is a test transcription.\nWith multiple lines.")
        
        result = load_transcription(txt_file)
        
        assert "This is a test transcription" in result
        assert "With multiple lines" in result
    
    def test_load_transcription_from_txt_with_timestamps(self, tmp_path):
        """Test loading transcription with timestamps (should be removed)"""
        txt_file = tmp_path / "transcript.txt"
        txt_file.write_text("[00:00:00.000] This is a test.\n[00:00:05.000] More text.\nClean line.")
        
        result = load_transcription(txt_file)
        
        assert "[00:00:00.000]" not in result
        assert "Clean line" in result
    
    def test_load_transcription_from_txt_with_urls(self, tmp_path):
        """Test loading transcription with URLs (should be removed)"""
        txt_file = tmp_path / "transcript.txt"
        txt_file.write_text("Text here.\nhttps://youtube.com/watch?v=test\nMore text.")
        
        result = load_transcription(txt_file)
        
        assert "https://youtube.com" not in result
        assert "Text here" in result
        assert "More text" in result
    
    def test_load_transcription_from_json_with_text(self, tmp_path):
        """Test loading transcription from JSON with 'text' field"""
        json_file = tmp_path / "transcript.json"
        json_data = {"text": "This is the transcription text"}
        json_file.write_text(json.dumps(json_data))
        
        result = load_transcription(json_file)
        
        assert result == "This is the transcription text"
    
    def test_load_transcription_from_json_with_segments(self, tmp_path):
        """Test loading transcription from JSON with 'segments' field"""
        json_file = tmp_path / "transcript.json"
        json_data = {
            "segments": [
                {"text": "First segment"},
                {"text": "Second segment"}
            ]
        }
        json_file.write_text(json.dumps(json_data))
        
        result = load_transcription(json_file)
        
        assert "First segment" in result
        assert "Second segment" in result
    
    def test_load_transcription_file_not_found(self, tmp_path):
        """Test loading transcription from non-existent file"""
        missing_file = tmp_path / "missing.txt"
        
        with pytest.raises(RuntimeError, match="Error loading transcription"):
            load_transcription(missing_file)


class TestGenerateLesson:
    """Tests for generate_lesson function"""
    
    def test_generate_lesson_success(self):
        """Test successful lesson generation"""
        mock_provider = Mock(spec=LLMProvider)
        mock_provider.generate.return_value = json.dumps({
            "vocabulary": [{"word": "テスト", "meaning": "test"}],
            "grammar_points": [],
            "key_phrases": [],
            "summary": "Test summary"
        })
        
        result = generate_lesson(mock_provider, "Test transcription text")
        
        assert "vocabulary" in result
        assert "summary" in result
        mock_provider.generate.assert_called_once()
    
    def test_generate_lesson_with_episode_title(self):
        """Test lesson generation with episode title"""
        mock_provider = Mock(spec=LLMProvider)
        mock_provider.generate.return_value = json.dumps({
            "vocabulary": [],
            "grammar_points": [],
            "key_phrases": [],
            "summary": "Test"
        })
        
        result = generate_lesson(mock_provider, "Test text", episode_title="Test Episode")
        
        assert result is not None
        # Check that title was included in prompt
        call_args = mock_provider.generate.call_args
        assert "Test Episode" in call_args[1]["prompt"] or "Test Episode" in call_args[0][0]
    
    def test_generate_lesson_with_markdown_wrapped_json(self):
        """Test lesson generation with JSON wrapped in markdown code blocks"""
        mock_provider = Mock(spec=LLMProvider)
        mock_provider.generate.return_value = """```json
{
  "vocabulary": [],
  "summary": "Test"
}
```"""
        
        result = generate_lesson(mock_provider, "Test text")
        
        assert "vocabulary" in result
        assert "summary" in result
    
    def test_generate_lesson_invalid_json(self):
        """Test lesson generation with invalid JSON response"""
        mock_provider = Mock(spec=LLMProvider)
        mock_provider.generate.return_value = "This is not JSON"
        
        with pytest.raises(RuntimeError, match="Failed to parse lesson JSON"):
            generate_lesson(mock_provider, "Test text")


class TestSaveLesson:
    """Tests for save_lesson function"""
    
    def test_save_lesson_json(self, tmp_path):
        """Test saving lesson as JSON"""
        lesson_data = {
            "vocabulary": [{"word": "テスト", "meaning": "test"}],
            "summary": "Test summary"
        }
        output_path = tmp_path / "lesson.json"
        
        save_lesson(lesson_data, output_path, format="json")
        
        assert output_path.exists()
        loaded = json.loads(output_path.read_text())
        assert loaded["vocabulary"] == lesson_data["vocabulary"]
    
    def test_save_lesson_markdown(self, tmp_path):
        """Test saving lesson as markdown"""
        lesson_data = {
            "vocabulary": [{"word": "テスト", "meaning": "test"}],
            "summary": "Test summary"
        }
        output_path = tmp_path / "lesson.md"
        
        save_lesson(lesson_data, output_path, format="markdown")
        
        assert output_path.exists()
        content = output_path.read_text()
        assert "テスト" in content
        assert "Test summary" in content


class TestFormatLessonMarkdown:
    """Tests for format_lesson_markdown function"""
    
    def test_format_lesson_markdown_basic(self):
        """Test formatting basic lesson as markdown"""
        lesson_data = {
            "vocabulary": [
                {"word": "テスト", "reading": "てすと", "meaning": "test", "jlpt_level": "N5"}
            ],
            "grammar_points": [],
            "key_phrases": [],
            "summary": "Test summary"
        }
        
        result = format_lesson_markdown(lesson_data)
        
        assert "テスト" in result
        assert "てすと" in result
        assert "test" in result
        assert "N5" in result
        assert "Test summary" in result
    
    def test_format_lesson_markdown_with_grammar(self):
        """Test formatting lesson with grammar points"""
        lesson_data = {
            "vocabulary": [],
            "grammar_points": [
                {
                    "pattern": "〜です",
                    "explanation": "Polite copula",
                    "jlpt_level": "N5",
                    "example_sentence": "これは本です",
                    "example_translation": "This is a book"
                }
            ],
            "key_phrases": [],
            "summary": "Test"
        }
        
        result = format_lesson_markdown(lesson_data)
        
        assert "〜です" in result
        assert "Polite copula" in result
        assert "これは本です" in result

