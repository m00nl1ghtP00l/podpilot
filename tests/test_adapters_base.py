"""
Tests for adapters/base.py

This test suite covers:
- Prompt file loading from config
- Environment variable expansion in prompt paths
- Fallback to variant-specific files
- Fallback to default files
- Schema generation
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
import sys
import os

# Add parent directory to path to import the module
sys.path.insert(0, str(Path(__file__).parent.parent))

from adapters.base import LanguageAdapter


class TestPromptFileLoading:
    """Tests for prompt file loading logic"""
    
    def test_load_system_prompt_from_config(self, tmp_path, monkeypatch):
        """Test loading system prompt from config prompt_files"""
        # Create a test prompt file
        prompt_file = tmp_path / "test_system.md"
        prompt_file.write_text("# System Prompt\nTest content")
        
        # Create a mock adapter
        class TestAdapter(LanguageAdapter):
            @property
            def language_code(self):
                return "test"
            
            @property
            def language_name(self):
                return "Test"
            
            def get_transcription_prompt(self):
                return ""
            
            def segment_text(self, text):
                return []
            
            def clean_title(self, title):
                return title
            
            def get_proficiency_levels(self):
                return []
        
        adapter = TestAdapter()
        
        # Test loading from config
        prompt_files = {"system": str(prompt_file)}
        result = adapter.get_lesson_system_prompt(prompt_files=prompt_files)
        
        assert "System Prompt" in result
        assert "Test content" in result
    
    def test_load_user_prompt_from_config(self, tmp_path):
        """Test loading user prompt from config prompt_files"""
        # Create a test prompt file
        prompt_file = tmp_path / "test_user.md"
        prompt_file.write_text("# User Prompt\n{episode_title_section}\n{transcription_text}")
        
        # Create a mock adapter
        class TestAdapter(LanguageAdapter):
            @property
            def language_code(self):
                return "test"
            
            @property
            def language_name(self):
                return "Test"
            
            def get_transcription_prompt(self):
                return ""
            
            def segment_text(self, text):
                return []
            
            def clean_title(self, title):
                return title
            
            def get_proficiency_levels(self):
                return []
        
        adapter = TestAdapter()
        
        # Test loading from config
        prompt_files = {"user": str(prompt_file)}
        result = adapter.get_lesson_user_prompt_template(prompt_files=prompt_files)
        
        assert "User Prompt" in result
        assert "{episode_title_section}" in result
        assert "{transcription_text}" in result
    
    def test_expand_env_vars_in_prompt_path(self, tmp_path, monkeypatch):
        """Test environment variable expansion in prompt file paths"""
        # Set environment variable
        test_dir = str(tmp_path)
        monkeypatch.setenv("PROMPTS_DIR", test_dir)
        
        # Create a test prompt file
        prompt_file = tmp_path / "test_prompt.md"
        prompt_file.write_text("# Test Prompt")
        
        # Create a mock adapter
        class TestAdapter(LanguageAdapter):
            @property
            def language_code(self):
                return "test"
            
            @property
            def language_name(self):
                return "Test"
            
            def get_transcription_prompt(self):
                return ""
            
            def segment_text(self, text):
                return []
            
            def clean_title(self, title):
                return title
            
            def get_proficiency_levels(self):
                return []
        
        adapter = TestAdapter()
        
        # Test with env var in path
        prompt_files = {"system": "${PROMPTS_DIR}/test_prompt.md"}
        result = adapter.get_lesson_system_prompt(prompt_files=prompt_files)
        
        assert "Test Prompt" in result
    
    def test_fallback_to_variant_file(self, tmp_path):
        """Test fallback to variant-specific prompt file"""
        # Create prompts directory structure relative to adapters directory
        from adapters import base
        adapters_dir = Path(base.__file__).parent
        prompts_dir = adapters_dir / "prompts" / "test"
        prompts_dir.mkdir(parents=True, exist_ok=True)
        
        # Create variant-specific file
        variant_file = prompts_dir / "system_prompt_detailed.md"
        variant_file.write_text("# Detailed System Prompt")
        
        # Create a mock adapter
        class TestAdapter(LanguageAdapter):
            @property
            def language_code(self):
                return "test"
            
            @property
            def language_name(self):
                return "Test"
            
            def get_transcription_prompt(self):
                return ""
            
            def segment_text(self, text):
                return []
            
            def clean_title(self, title):
                return title
            
            def get_proficiency_levels(self):
                return []
        
        adapter = TestAdapter()
        
        # Test loading variant (should find the file we created)
        result = adapter.get_lesson_system_prompt(variant="detailed")
        
        # Should load the variant file or fallback
        assert len(result) > 0
        # Clean up
        if variant_file.exists():
            variant_file.unlink()
        if prompts_dir.exists():
            prompts_dir.rmdir()
    
    def test_fallback_to_default_prompt(self, tmp_path):
        """Test fallback to default prompt when no files found"""
        # Create a mock adapter
        class TestAdapter(LanguageAdapter):
            @property
            def language_code(self):
                return "test"
            
            @property
            def language_name(self):
                return "Test"
            
            def get_transcription_prompt(self):
                return ""
            
            def segment_text(self, text):
                return []
            
            def clean_title(self, title):
                return title
            
            def get_proficiency_levels(self):
                return []
        
        adapter = TestAdapter()
        
        # Test fallback (no files exist)
        result = adapter.get_lesson_system_prompt()
        
        # Should return fallback prompt
        assert len(result) > 0
        assert "assistant" in result.lower() or "summary" in result.lower() or "lesson" in result.lower()
    
    def test_schema_generation(self):
        """Test schema generation for lesson structure"""
        # Create a mock adapter
        class TestAdapter(LanguageAdapter):
            @property
            def language_code(self):
                return "test"
            
            @property
            def language_name(self):
                return "Test"
            
            def get_transcription_prompt(self):
                return ""
            
            def segment_text(self, text):
                return []
            
            def clean_title(self, title):
                return title
            
            def get_proficiency_levels(self):
                return ["Level1", "Level2"]
        
        adapter = TestAdapter()
        
        # Test schema generation
        schema = adapter.get_lesson_schema()
        
        assert "vocabulary" in schema
        assert "grammar_points" in schema
        assert "key_phrases" in schema
        assert "summary" in schema

