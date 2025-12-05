#!/usr/bin/env python3
"""
Example English language adapter

This serves as a template for creating custom language adapters.
Users can copy this file and modify it for their target language.
"""

from typing import List
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
    
    def get_lesson_system_prompt(self) -> str:
        schema_block = self.format_schema_for_prompt()
        return f"""# Role
You are an expert English language teacher specializing in ESL (English as a Second Language) preparation.

# Task
Analyze English text and create structured lessons with vocabulary and grammar explanations.

# Output Format
Always respond in **valid JSON format only** (no markdown code blocks, no explanatory text). Use the following structure:

{schema_block}

# Instructions
- Extract important vocabulary words with pronunciations, meanings, and CEFR levels
- Identify grammar patterns and structures with clear explanations
- Include key phrases that are useful for learners
- Provide a brief summary of the content
- Focus on words and grammar useful for ESL learners (A1-C2 levels)
- Ensure all JSON is valid and properly formatted"""
    
    def get_lesson_user_prompt_template(self) -> str:
        return """# Analysis Request

{episode_title_section}

## English Text to Analyze

{transcription_text}

## Task
Create a comprehensive ESL-style lesson by extracting:

1. **Vocabulary**: Important words with pronunciations, meanings, and CEFR levels
2. **Grammar**: Patterns and structures with explanations
3. **Key Phrases**: Useful phrases with context
4. **Summary**: Brief overview of the content

Focus on content useful for ESL learners (A1-C2 levels)."""
    
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

