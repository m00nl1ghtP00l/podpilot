#!/usr/bin/env python3
"""
Japanese language adapter

Provides Japanese-specific:
- Transcription prompts
- Lesson generation prompts
- Text segmentation (Japanese punctuation)
- Character handling (preserve Kanji, Hiragana, Katakana)
- JLPT proficiency levels
"""

import re
from typing import List, Optional, Dict
from .base import LanguageAdapter


class JapaneseAdapter(LanguageAdapter):
    """Adapter for Japanese language processing"""
    
    # Japanese character pattern (compiled once for efficiency)
    _JAPANESE_CHARS_PATTERN = re.compile(
        r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF\u3400-\u4DBF]'  # Hiragana, Katakana, Kanji
    )
    
    @property
    def language_code(self) -> str:
        return "ja"
    
    @property
    def language_name(self) -> str:
        return "Japanese"
    
    def get_transcription_prompt(self) -> str:
        return "この音声は日本語です。できるだけ正確に文字起こししてください。文末に改行を入れてください."
    
    def get_lesson_system_prompt(self, variant: Optional[str] = None, prompt_files: Optional[Dict] = None) -> str:
        """Get system prompt from file, with schema injection for Japanese"""
        from pathlib import Path
        from typing import Dict
        
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
        """Split Japanese text into sentences.
        
        Japanese sentences typically end with punctuation marks like 。, ？, or ！.
        """
        # Define Japanese sentence-ending punctuation
        end_marks = ['。', '？', '！', '…']
        
        # Split the text into initial chunks based on line breaks
        chunks = text.split('\n')
        sentences = []
        
        for chunk in chunks:
            if not chunk.strip():
                continue
                
            # Current position in the chunk
            current_pos = 0
            chunk_len = len(chunk)
            
            # Process the chunk character by character
            for i in range(chunk_len):
                # Check if this character is a sentence-ending punctuation
                if i < chunk_len and chunk[i] in end_marks:
                    # Extract the sentence (including the ending punctuation)
                    sentence = chunk[current_pos:i+1].strip()
                    if sentence:
                        sentences.append(sentence)
                    current_pos = i + 1
            
            # Add any remaining text as a sentence
            if current_pos < chunk_len:
                remaining = chunk[current_pos:].strip()
                if remaining:
                    sentences.append(remaining)
        
        return sentences
    
    def clean_title(self, title: str) -> str:
        """Clean title while preserving Japanese characters"""
        def _extract_japanese_segments(text: str):
            """Extract Japanese character segments with their positions."""
            segments = []
            for match in re.finditer(f'({self._JAPANESE_CHARS_PATTERN.pattern}+)', text):
                segments.append((match.start(), match.end(), match.group()))
            return segments
        
        def _remove_emojis_preserving_japanese(text: str, jp_segments):
            """Remove emojis from text while preserving Japanese character segments."""
            result = []
            last_end = 0
            
            for start, end, jp_text in jp_segments:
                # Process the part before this Japanese segment
                before = text[last_end:start]
                before_clean = ''.join(c for c in before if ord(c) < 0x1F000 or (0x1F300 <= ord(c) <= 0x1F9FF))
                result.append(before_clean)
                
                # Add the Japanese segment as is
                result.append(jp_text)
                last_end = end
            
            # Process the remaining part after the last Japanese segment
            after = text[last_end:]
            after_clean = ''.join(c for c in after if ord(c) < 0x1F000 or (0x1F300 <= ord(c) <= 0x1F9FF))
            result.append(after_clean)
            
            return ''.join(result)
        
        # Extract Japanese segments
        jp_segments = _extract_japanese_segments(title)
        
        # Remove emojis while preserving Japanese
        result = _remove_emojis_preserving_japanese(title, jp_segments)
        
        # Extract Japanese segments again from emoji-removed result
        jp_segments_final = _extract_japanese_segments(result)
        
        # Convert spaces to underscores while preserving Japanese
        final_result = []
        last_end = 0
        
        for start, end, jp_text in jp_segments_final:
            # Process the part before this Japanese segment
            before = result[last_end:start]
            before_clean = before.replace(' ', '_').replace('|', '_')
            final_result.append(before_clean)
            
            # Add the Japanese segment as is
            final_result.append(jp_text)
            last_end = end
        
        # Process the remaining part after the last Japanese segment
        after = result[last_end:]
        after_clean = after.replace(' ', '_').replace('|', '_')
        final_result.append(after_clean)
        
        # Final cleanup: remove invalid filename characters except those in Japanese
        cleaned = ''.join(final_result)
        # Replace invalid filename characters (but preserve Japanese)
        invalid_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
        for char in invalid_chars:
            cleaned = cleaned.replace(char, '_')
        
        return cleaned.strip('_')
    
    def get_proficiency_levels(self) -> List[str]:
        return ["N5", "N4", "N3", "N2", "N1"]
    
    def get_lesson_schema(self):
        """Get the JSON schema for Japanese lesson structure"""
        from typing import Dict, Any
        return {
            "vocabulary": [
                {
                    "word": "word in Kanji, Hiragana, or Katakana",
                    "reading": "hiragana/katakana reading",
                    "meaning": "English meaning",
                    "jlpt_level": "N5|N4|N3|N2|N1",
                    "example_sentence": "Example sentence using the word",
                    "example_translation": "English translation of example"
                }
            ],
            "grammar_points": [
                {
                    "pattern": "Grammar pattern name",
                    "explanation": "Explanation of how to use this grammar",
                    "jlpt_level": "N5|N4|N3|N2|N1",
                    "example_sentence": "Example sentence",
                    "example_translation": "English translation"
                }
            ],
            "key_phrases": [
                {
                    "phrase": "Japanese phrase",
                    "translation": "English translation",
                    "context": "When/where this phrase is used"
                }
            ],
            "summary": "Brief summary of the lesson content"
        }

