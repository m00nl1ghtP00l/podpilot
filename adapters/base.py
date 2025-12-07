#!/usr/bin/env python3
"""
Base classes for language and LLM adapters
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, List
from pathlib import Path


class LanguageAdapter(ABC):
    """Base class for language-specific adapters
    
    Language adapters provide:
    - Language-specific prompts for transcription and lesson generation
    - Text segmentation rules
    - Character handling (preservation, cleaning)
    - Proficiency level systems (e.g., JLPT for Japanese, CEFR for European languages)
    """
    
    @property
    @abstractmethod
    def language_code(self) -> str:
        """ISO 639-1 language code (e.g., 'ja', 'en', 'es')"""
        pass
    
    @property
    @abstractmethod
    def language_name(self) -> str:
        """Human-readable language name"""
        pass
    
    @abstractmethod
    def get_transcription_prompt(self) -> str:
        """Get the prompt for transcription (language-specific)"""
        pass
    
    def get_lesson_system_prompt(self, variant: Optional[str] = None, prompt_files: Optional[Dict] = None) -> str:
        """Get the system prompt for lesson generation
        
        Args:
            variant: Optional prompt variant name (e.g., "detailed", "simple")
                    If provided, looks for system_prompt_{variant}.md
                    Otherwise uses system_prompt.md
            prompt_files: Optional dict from config with "system" and "user" file paths
                        If provided, loads from these paths (with env var expansion)
                        Takes precedence over variant and default files
        
        Priority:
        1. prompt_files["system"] (if provided in config)
        2. adapters/prompts/{language_code}/system_prompt_{variant}.md (if variant specified)
        3. adapters/prompts/{language_code}/system_prompt.md (default)
        4. Simple fallback prompt
        """
        # Check config prompt_files first
        if prompt_files and "system" in prompt_files:
            prompt_path = self._resolve_prompt_path(prompt_files["system"])
            if prompt_path and prompt_path.exists():
                return prompt_path.read_text(encoding='utf-8').strip()
        
        # Fall back to file-based lookup
        prompts_dir = Path(__file__).parent / "prompts" / self.language_code
        if variant:
            prompt_path = prompts_dir / f"system_prompt_{variant}.md"
        else:
            prompt_path = prompts_dir / "system_prompt.md"
        
        if prompt_path.exists():
            return prompt_path.read_text(encoding='utf-8').strip()
        # Simple fallback prompt
        return "You are a helpful assistant that will provide a summary of the input text, list the key themes, key phrases, and key vocabulary. At the end, provide a list of the phrases used with 5 multiple choice answers; 4 could be the answer but only 1 is the answer. Provide 10 questions. At the end provide the answers to each question"
    
    def get_lesson_user_prompt_template(self, variant: Optional[str] = None, prompt_files: Optional[Dict] = None) -> str:
        """Get the user prompt template for lesson generation
        
        Args:
            variant: Optional prompt variant name (e.g., "detailed", "simple")
                    If provided, looks for user_prompt_template_{variant}.md
                    Otherwise uses user_prompt_template.md
            prompt_files: Optional dict from config with "system" and "user" file paths
                        If provided, loads from these paths (with env var expansion)
                        Takes precedence over variant and default files
        
        Priority:
        1. prompt_files["user"] (if provided in config)
        2. adapters/prompts/{language_code}/user_prompt_template_{variant}.md (if variant specified)
        3. adapters/prompts/{language_code}/user_prompt_template.md (default)
        4. Simple fallback template
        
        Should include placeholders like {episode_title_section} and {transcription_text}
        """
        # Check config prompt_files first
        if prompt_files and "user" in prompt_files:
            prompt_path = self._resolve_prompt_path(prompt_files["user"])
            if prompt_path and prompt_path.exists():
                return prompt_path.read_text(encoding='utf-8').strip()
        
        # Fall back to file-based lookup
        prompts_dir = Path(__file__).parent / "prompts" / self.language_code
        if variant:
            prompt_path = prompts_dir / f"user_prompt_template_{variant}.md"
        else:
            prompt_path = prompts_dir / "user_prompt_template.md"
        
        if prompt_path.exists():
            return prompt_path.read_text(encoding='utf-8').strip()
        # Simple fallback template
        return """Analyze the following text:

{episode_title_section}

{transcription_text}

Provide a summary, key themes, key phrases, and key vocabulary."""
    
    @abstractmethod
    def segment_text(self, text: str) -> List[str]:
        """Segment text into sentences based on language-specific rules"""
        pass
    
    @abstractmethod
    def clean_title(self, title: str) -> str:
        """Clean title while preserving language-specific characters"""
        pass
    
    @abstractmethod
    def get_proficiency_levels(self) -> List[str]:
        """Get list of proficiency levels for this language
        
        Examples:
        - Japanese: ["N5", "N4", "N3", "N2", "N1"]
        - European languages: ["A1", "A2", "B1", "B2", "C1", "C2"]
        - Custom: ["Beginner", "Intermediate", "Advanced"]
        """
        pass
    
    def get_lesson_schema(self) -> Dict:
        """Get the JSON schema for lesson structure
        
        Can be overridden to customize lesson structure per language.
        Default returns a generic structure.
        """
        return {
            "vocabulary": [
                {
                    "word": "word in target language",
                    "reading": "pronunciation/reading (if applicable)",
                    "meaning": "English meaning",
                    "proficiency_level": "level from get_proficiency_levels()",
                    "example_sentence": "Example sentence using the word",
                    "example_translation": "English translation of example"
                }
            ],
            "grammar_points": [
                {
                    "pattern": "Grammar pattern name",
                    "explanation": "Explanation of how to use this grammar",
                    "proficiency_level": "level from get_proficiency_levels()",
                    "example_sentence": "Example sentence",
                    "example_translation": "English translation"
                }
            ],
            "key_phrases": [
                {
                    "phrase": "phrase in target language",
                    "translation": "English translation",
                    "context": "When/where this phrase is used"
                }
            ],
            "summary": "Brief summary of the lesson content"
        }
    
    def format_schema_for_prompt(self) -> str:
        """Format the lesson schema as a markdown code block for use in prompts
        
        This generates a concise JSON schema example from get_lesson_schema()
        to avoid duplication and keep prompts maintainable.
        """
        import json
        schema = self.get_lesson_schema()
        # Format with indentation for readability
        schema_json = json.dumps(schema, indent=2, ensure_ascii=False)
        return f"```json\n{schema_json}\n```"
    
    def _resolve_prompt_path(self, path_str: str) -> Optional[Path]:
        """Resolve prompt file path with environment variable expansion
        
        Args:
            path_str: Path string that may contain environment variables (${VAR} or $VAR)
        
        Returns:
            Resolved Path object, or None if path is invalid
        """
        import os
        import re
        
        # Expand environment variables
        def replace_var(match):
            var_name = match.group(1) or match.group(2)
            return os.environ.get(var_name, match.group(0))  # Return original if not found
        
        expanded_path = re.sub(r'\$\{(\w+)\}|\$(\w+)', replace_var, path_str)
        
        # Resolve path (relative to project root or absolute)
        path = Path(expanded_path)
        if path.is_absolute():
            return path
        
        # Relative paths: try relative to project root (where adapters/ is)
        # Go up from adapters/base.py to project root
        project_root = Path(__file__).parent.parent
        return project_root / path


# Registry for language adapters
_language_adapters: Dict[str, LanguageAdapter] = {}


def register_language_adapter(language_code: str, adapter: LanguageAdapter):
    """Register a language adapter
    
    Args:
        language_code: Language code (e.g., 'ja', 'en', 'es')
        adapter: LanguageAdapter instance
    """
    _language_adapters[language_code.lower()] = adapter


def get_language_adapter(language_code: str) -> Optional[LanguageAdapter]:
    """Get a language adapter by language code
    
    Args:
        language_code: Language code (e.g., 'ja', 'en', 'es')
    
    Returns:
        LanguageAdapter instance or None if not found
    """
    return _language_adapters.get(language_code.lower())


def list_available_adapters() -> List[str]:
    """List all registered language adapter codes"""
    return list(_language_adapters.keys())

