#!/usr/bin/env python3
"""
Lesson Search and Retrieval Module
Searches through lesson files and extracts relevant content
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass


@dataclass
class LessonContent:
    """Structured lesson content"""
    file_path: Path
    episode_title: str
    date: Optional[datetime]
    summary: str
    vocabulary: List[Dict]
    grammar_points: List[Dict]
    key_phrases: List[Dict]
    raw_content: str
    jlpt_levels: List[str]  # List of JLPT levels found in this lesson


def parse_lesson_markdown(file_path: Path) -> Optional[LessonContent]:
    """Parse a lesson markdown file and extract structured content"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        return None
    
    # Extract episode title from filename
    filename = file_path.stem.replace('_lesson', '')
    episode_title = filename
    
    # Extract date from filename (format: YYYY-MM-DD_...)
    date = None
    date_match = re.match(r'(\d{4}-\d{2}-\d{2})', filename)
    if date_match:
        try:
            date = datetime.strptime(date_match.group(1), '%Y-%m-%d')
        except ValueError:
            pass
    
    # Extract summary
    summary = ""
    summary_match = re.search(r'##\s*Summary\s*\n\n(.*?)(?=\n##|\Z)', content, re.DOTALL | re.IGNORECASE)
    if summary_match:
        summary = summary_match.group(1).strip()
    
    # Extract vocabulary
    vocabulary = []
    vocab_section = re.search(r'##\s*Vocabulary\s*\n(.*?)(?=\n##|\Z)', content, re.DOTALL | re.IGNORECASE)
    if vocab_section:
        vocab_text = vocab_section.group(1)
        # Try to extract vocabulary items (various formats)
        vocab_items = re.findall(r'(?:^|\n)(?:[-*]|\d+\.)\s*\*\*([^*]+)\*\*(?:[^(]*\(([^)]+)\))?[^-]*?JLPT\s*Level:\s*(N[1-5])', vocab_text, re.MULTILINE | re.IGNORECASE)
        for match in vocab_items:
            word = match[0].strip()
            reading = match[1].strip() if match[1] else ""
            level = match[2].strip()
            vocabulary.append({
                'word': word,
                'reading': reading,
                'jlpt_level': level
            })
    
    # Extract grammar points
    grammar_points = []
    grammar_section = re.search(r'##\s*Grammar\s*(?:Points)?\s*\n(.*?)(?=\n##|\Z)', content, re.DOTALL | re.IGNORECASE)
    if grammar_section:
        grammar_text = grammar_section.group(1)
        # Try to extract grammar patterns
        grammar_items = re.findall(r'(?:^|\n)(?:[-*]|\d+\.)\s*\*\*([^*]+)\*\*[^-]*?JLPT\s*Level:\s*(N[1-5])', grammar_text, re.MULTILINE | re.IGNORECASE)
        for match in grammar_items:
            pattern = match[0].strip()
            level = match[1].strip()
            grammar_points.append({
                'pattern': pattern,
                'jlpt_level': level
            })
    
    # Extract key phrases
    key_phrases = []
    phrases_section = re.search(r'##\s*Key\s*Phrases\s*\n(.*?)(?=\n##|\Z)', content, re.DOTALL | re.IGNORECASE)
    if phrases_section:
        phrases_text = phrases_section.group(1)
        phrases_items = re.findall(r'(?:^|\n)(?:[-*]|\d+\.)\s*\*\*([^*]+)\*\*', phrases_text, re.MULTILINE)
        for match in phrases_items:
            phrase = match.strip()
            key_phrases.append({'phrase': phrase})
    
    # Extract all JLPT levels mentioned
    jlpt_levels = list(set(re.findall(r'N[1-5]', content, re.IGNORECASE)))
    jlpt_levels.sort()
    
    return LessonContent(
        file_path=file_path,
        episode_title=episode_title,
        date=date,
        summary=summary,
        vocabulary=vocabulary,
        grammar_points=grammar_points,
        key_phrases=key_phrases,
        raw_content=content,
        jlpt_levels=jlpt_levels
    )


def find_lesson_files(data_root: Path, channel_name: Optional[str] = None, 
                     from_date: Optional[datetime] = None, 
                     to_date: Optional[datetime] = None) -> List[Path]:
    """Find all lesson files in the data directory"""
    lesson_files = []
    
    if channel_name:
        # Search in specific channel directory
        channel_dir = data_root / channel_name
        if channel_dir.exists():
            lesson_files.extend(channel_dir.glob("*_lesson.md"))
    else:
        # Search in all channel directories
        for item in data_root.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                lesson_files.extend(item.glob("*_lesson.md"))
    
    # Filter by date if specified
    if from_date or to_date:
        filtered = []
        for lesson_file in lesson_files:
            date_match = re.match(r'(\d{4}-\d{2}-\d{2})', lesson_file.stem)
            if date_match:
                try:
                    file_date = datetime.strptime(date_match.group(1), '%Y-%m-%d')
                    if from_date and file_date < from_date:
                        continue
                    if to_date and file_date > to_date:
                        continue
                    filtered.append(lesson_file)
                except ValueError:
                    pass
            elif not (from_date or to_date):
                # If no date in filename but no date filter, include it
                filtered.append(lesson_file)
        lesson_files = filtered
    
    return sorted(lesson_files)


def search_lessons_by_keywords(lessons: List[LessonContent], keywords: List[str]) -> List[LessonContent]:
    """Search lessons by keywords (case-insensitive)"""
    keywords_lower = [kw.lower() for kw in keywords]
    matched = []
    
    for lesson in lessons:
        # Search in summary, vocabulary, grammar, and raw content
        search_text = (
            lesson.summary + " " + 
            " ".join([v.get('word', '') for v in lesson.vocabulary]) +
            " ".join([g.get('pattern', '') for g in lesson.grammar_points]) +
            " ".join([p.get('phrase', '') for p in lesson.key_phrases]) +
            lesson.raw_content
        ).lower()
        
        # Check if any keyword matches
        if any(kw in search_text for kw in keywords_lower):
            matched.append(lesson)
    
    return matched


def search_lessons_by_jlpt_level(lessons: List[LessonContent], levels: List[str]) -> List[LessonContent]:
    """Search lessons by JLPT level"""
    levels_upper = [level.upper() for level in levels]
    matched = []
    
    for lesson in lessons:
        # Check if lesson contains any of the requested levels
        if any(level in lesson.jlpt_levels for level in levels_upper):
            matched.append(lesson)
    
    return matched


def get_lessons_for_channel(data_root: Path, channel_name: str,
                            from_date: Optional[datetime] = None,
                            to_date: Optional[datetime] = None) -> List[LessonContent]:
    """Get all lessons for a specific channel"""
    lesson_files = find_lesson_files(data_root, channel_name, from_date, to_date)
    lessons = []
    
    for lesson_file in lesson_files:
        lesson = parse_lesson_markdown(lesson_file)
        if lesson:
            lessons.append(lesson)
    
    return lessons


def get_all_lessons(data_root: Path,
                    from_date: Optional[datetime] = None,
                    to_date: Optional[datetime] = None) -> List[LessonContent]:
    """Get all lessons across all channels"""
    lesson_files = find_lesson_files(data_root, None, from_date, to_date)
    lessons = []
    
    for lesson_file in lesson_files:
        lesson = parse_lesson_markdown(lesson_file)
        if lesson:
            lessons.append(lesson)
    
    return lessons

