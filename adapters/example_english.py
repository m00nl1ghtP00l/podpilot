#!/usr/bin/env python3
"""
Example English language adapter

This serves as a template for creating custom language adapters.
Users can copy this file and modify it for their target language.
"""

from typing import List, Optional, Dict
from .base import LanguageAdapter


class EnglishAdapter(LanguageAdapter):
    """Adapter for English language processing"""
    
    @property
    def language_code(self) -> str:
        return "en"
    
    @property
    def language_name(self) -> str:
        return "English"
    
    def get_transcription_prompt(self) -> str:
        return "This audio is in English. Please transcribe it as accurately as possible. Add line breaks at the end of sentences."
    
    def get_lesson_system_prompt(self, variant: Optional[str] = None, prompt_files: Optional[Dict] = None) -> str:
        """Get system prompt from file, with schema injection for English"""
        # Get prompt (from config files, variant, or default)
        prompt = super().get_lesson_system_prompt(variant=variant, prompt_files=prompt_files)
        
        # Inject schema if {schema_block} placeholder exists
        if "{schema_block}" in prompt:
            schema_block = self.format_schema_for_prompt()
            prompt = prompt.replace("{schema_block}", schema_block)
        
        return prompt
    
    def get_lesson_user_prompt_template(self, variant: Optional[str] = None, prompt_files: Optional[Dict] = None) -> str:
        """Get user prompt template from file"""
        # Just use base class implementation (handles config files and variants)
        return super().get_lesson_user_prompt_template(variant=variant, prompt_files=prompt_files)
    
    def segment_text(self, text: str) -> List[str]:
        """Split English text into sentences.
        
        English sentences typically end with punctuation marks like ., ?, or !.
        """
        import re
        # Split on sentence-ending punctuation followed by space or newline
        sentences = re.split(r'([.!?]+)\s+', text)
        result = []
        for i in range(0, len(sentences) - 1, 2):
            if i + 1 < len(sentences):
                sentence = (sentences[i] + sentences[i + 1]).strip()
                if sentence:
                    result.append(sentence)
        if len(sentences) % 2 == 1 and sentences[-1].strip():
            result.append(sentences[-1].strip())
        return result if result else [text]
    
    def clean_title(self, title: str) -> str:
        """Clean title for English (standard filename sanitization)"""
        # Replace invalid filename characters
        invalid_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
        cleaned = title
        for char in invalid_chars:
            cleaned = cleaned.replace(char, '_')
        # Replace spaces with underscores
        cleaned = cleaned.replace(' ', '_')
        return cleaned.strip('_')
    
    def get_proficiency_levels(self) -> List[str]:
        return ["A1", "A2", "B1", "B2", "C1", "C2"]
    
    def get_lesson_schema(self):
        """Get the JSON schema for English lesson structure"""
        from typing import Dict, Any
        return {
            "vocabulary": [
                {
                    "word": "word in English",
                    "pronunciation": "phonetic pronunciation (IPA or similar)",
                    "meaning": "definition or translation",
                    "cefr_level": "A1|A2|B1|B2|C1|C2",
                    "example_sentence": "Example sentence using the word",
                    "example_translation": "Translation if applicable"
                }
            ],
            "grammar_points": [
                {
                    "pattern": "Grammar pattern name",
                    "explanation": "Explanation of how to use this grammar",
                    "cefr_level": "A1|A2|B1|B2|C1|C2",
                    "example_sentence": "Example sentence",
                    "example_translation": "Translation if applicable"
                }
            ],
            "key_phrases": [
                {
                    "phrase": "English phrase",
                    "translation": "Translation if applicable",
                    "context": "When/where this phrase is used"
                }
            ],
            "summary": "Brief summary of the lesson content"
        }

