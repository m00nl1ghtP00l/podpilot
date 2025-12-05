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
    
    @abstractmethod
    def get_lesson_system_prompt(self) -> str:
        """Get the system prompt for lesson generation"""
        pass
    
    @abstractmethod
    def get_lesson_user_prompt_template(self) -> str:
        """Get the user prompt template for lesson generation
        
        Should include placeholders like {episode_title} and {transcription_text}
        """
        pass
    
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

