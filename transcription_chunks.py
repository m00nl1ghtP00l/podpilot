#!/usr/bin/env python3
"""
Transcription Chunking Module - Split transcriptions into semantic chunks
"""

import re
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

from generate_lesson import load_transcription, find_transcription_files as _find_transcription_files


def chunk_transcription(transcription_text: str, episode_info: Dict) -> List[Dict]:
    """
    Split transcription into semantic chunks by Japanese sentences.
    
    Args:
        transcription_text: Full transcription text
        episode_info: Dictionary with episode metadata:
            - source: Episode identifier (e.g., "2024-01-15_Business_Success_#48")
            - episode_date: Date string (e.g., "2024-01-15")
            - channel: Channel name (e.g., "hnh")
            - transcription_file: Path to transcription file
            - lesson_file: Path to lesson file (optional)
    
    Returns:
        List of chunk dictionaries, each containing:
        - chunk_id: Unique identifier
        - content: The transcription text for this chunk
        - source: Episode identifier
        - episode_date: Episode date
        - channel: Channel name
        - transcription_file: Path to transcription file
        - lesson_file: Path to lesson file (if exists)
        - sentence_count: Number of sentences in chunk
        - position: Position in episode (0.0-1.0)
    """
    if not transcription_text or not transcription_text.strip():
        return []
    
    # Split by Japanese sentence boundaries
    # Japanese sentences end with: 。(period), ？(question), ！(exclamation)
    # Also handle English punctuation that might appear
    sentence_pattern = r'([^。？！\n]+[。？！\n]*)'
    sentences = [s.strip() for s in re.findall(sentence_pattern, transcription_text) if s.strip()]
    
    if not sentences:
        # Fallback: if no sentence markers, treat entire text as one chunk
        sentences = [transcription_text.strip()]
    
    # Group sentences into chunks (2-5 sentences per chunk, ~200-400 tokens)
    # Rough estimate: 1 Japanese character ≈ 1-2 tokens
    chunks = []
    current_chunk = []
    current_length = 0
    max_chunk_length = 400  # Leave headroom under 512 token limit
    
    total_length = sum(len(s) for s in sentences)
    
    for i, sentence in enumerate(sentences):
        sentence_length = len(sentence)
        
        # If single sentence exceeds limit, truncate it
        if sentence_length > max_chunk_length:
            # Add current chunk if it has content
            if current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_length = 0
            
            # Truncate long sentence
            truncated = sentence[:max_chunk_length]
            chunks.append(truncated)
            continue
        
        # Add sentence to current chunk
        if current_length + sentence_length <= max_chunk_length and len(current_chunk) < 5:
            current_chunk.append(sentence)
            current_length += sentence_length
        else:
            # Save current chunk and start new one
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_length
    
    # Add final chunk if it has content
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    # Create chunk dictionaries
    result = []
    for idx, chunk_content in enumerate(chunks, 1):
        # Calculate position in episode (0.0-1.0)
        chunk_start = sum(len(chunks[i]) for i in range(idx - 1))
        position = chunk_start / total_length if total_length > 0 else 0.0
        
        # Count sentences in chunk (approximate)
        sentence_count = len(re.findall(r'[。？！]', chunk_content)) + 1
        
        chunk_id = f"{episode_info['source']}_chunk_{idx:03d}"
        
        chunk_dict = {
            "chunk_id": chunk_id,
            "content": chunk_content,
            "source": episode_info['source'],
            "episode_date": episode_info.get('episode_date', ''),
            "channel": episode_info.get('channel', ''),
            "transcription_file": str(episode_info.get('transcription_file', '')),
            "lesson_file": str(episode_info.get('lesson_file', '')),
            "sentence_count": sentence_count,
            "position": round(position, 3)
        }
        result.append(chunk_dict)
    
    return result


def find_transcription_files(data_root: Path, channel_name: Optional[str] = None,
                            from_date: Optional[datetime] = None,
                            to_date: Optional[datetime] = None) -> List[Path]:
    """
    Find transcription files in the data directory.
    
    Args:
        data_root: Root data directory
        channel_name: Optional channel name to filter by
        from_date: Optional start date filter
        to_date: Optional end date filter
    
    Returns:
        List of transcription file paths (prefers _transcript.txt over .txt)
    """
    if channel_name:
        audio_dir = data_root / channel_name
    else:
        audio_dir = data_root
    
    if not audio_dir.exists():
        return []
    
    # Reuse the existing function from generate_lesson.py
    return _find_transcription_files(audio_dir, from_date, to_date)


def get_episode_info_from_file(transcription_file: Path, data_root: Path) -> Dict:
    """
    Extract episode information from transcription file path.
    
    Args:
        transcription_file: Path to transcription file
        data_root: Root data directory
    
    Returns:
        Dictionary with episode metadata
    """
    # Extract channel from path
    channel = transcription_file.parent.name if transcription_file.parent != data_root else ''
    
    # Extract date and title from filename
    filename = transcription_file.stem.replace('_transcript', '')
    
    # Try to extract date
    date_match = re.match(r'(\d{4}-\d{2}-\d{2})', filename)
    episode_date = date_match.group(1) if date_match else ''
    
    # Use filename as source (episode identifier)
    source = filename
    
    # Check if lesson file exists
    lesson_file = transcription_file.parent / f"{filename}_lesson.md"
    lesson_path = str(lesson_file) if lesson_file.exists() else ''
    
    return {
        "source": source,
        "episode_date": episode_date,
        "channel": channel,
        "transcription_file": str(transcription_file.relative_to(data_root)),
        "lesson_file": lesson_path
    }


def parse_transcript_sentences(transcript_file: Path) -> List[str]:
    """
    Parse _transcript.txt file efficiently.
    The file already has sentences separated by blank lines (\n\n).
    Just split by blank lines - no regex needed!
    
    Format:
        みなさんこんにちは、スピークジャープにスナッチオリーパーキャーストの踏みです。
        
        このチャンネルでは日本語や日本文化について紹介します。
        
        チャンネルメンバーシップもやっています。
    
    Returns:
        List of sentence strings (already separated)
    """
    try:
        with open(transcript_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split by blank lines - each block is already a sentence!
        sentences = [block.strip() for block in content.split('\n\n') if block.strip()]
        
        return sentences
    
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"Could not parse transcript file: {e}. Falling back to regex parsing.")
        return []


def chunk_transcription_file(transcription_file: Path, data_root: Path) -> List[Dict]:
    """
    Load transcription file and chunk it efficiently.
    Uses the sentence-level structure already in _transcript.txt (blank lines).
    Much more efficient than regex parsing!
    
    Args:
        transcription_file: Path to transcription file (prefer _transcript.txt)
        data_root: Root data directory
    
    Returns:
        List of chunk dictionaries
    """
    try:
        # Get episode info
        episode_info = get_episode_info_from_file(transcription_file, data_root)
        
        # Parse sentences directly from _transcript.txt (already separated by blank lines)
        sentences = parse_transcript_sentences(transcription_file)
        
        if sentences:
            # Use parsed sentences directly - already at sentence level!
            # Group 2-5 sentences per chunk
            chunks = []
            current_chunk = []
            current_length = 0
            max_chunk_length = 400
            
            total_length = sum(len(s) for s in sentences)
            
            for sentence_text in sentences:
                sentence_length = len(sentence_text)
                
                # If single sentence exceeds limit, truncate it
                if sentence_length > max_chunk_length:
                    if current_chunk:
                        chunks.append(' '.join(current_chunk))
                        current_chunk = []
                        current_length = 0
                    
                    chunks.append(sentence_text[:max_chunk_length])
                    continue
                
                # Add sentence to current chunk
                if current_length + sentence_length <= max_chunk_length and len(current_chunk) < 5:
                    current_chunk.append(sentence_text)
                    current_length += sentence_length
                else:
                    if current_chunk:
                        chunks.append(' '.join(current_chunk))
                    current_chunk = [sentence_text]
                    current_length = sentence_length
            
            # Add final chunk
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            
            # Convert to chunk dictionaries
            result = []
            for idx, chunk_content in enumerate(chunks, 1):
                chunk_start = sum(len(chunks[i]) for i in range(idx - 1))
                position = chunk_start / total_length if total_length > 0 else 0.0
                
                # Count sentences in chunk (approximate - count by splitting)
                sentence_count = len([s for s in sentences if s in chunk_content or chunk_content.startswith(s[:20])])
                if sentence_count == 0:
                    sentence_count = 1  # At least 1 sentence per chunk
                
                chunk_id = f"{episode_info['source']}_chunk_{idx:03d}"
                
                chunk_dict = {
                    "chunk_id": chunk_id,
                    "content": chunk_content,
                    "source": episode_info['source'],
                    "episode_date": episode_info.get('episode_date', ''),
                    "channel": episode_info.get('channel', ''),
                    "transcription_file": str(episode_info.get('transcription_file', '')),
                    "lesson_file": str(episode_info.get('lesson_file', '')),
                    "sentence_count": sentence_count,
                    "position": round(position, 3)
                }
                result.append(chunk_dict)
            
            return result
        
        # Fallback: use the old method (regex parsing) if blank-line parsing fails
        transcription_text = load_transcription(transcription_file)
        chunks = chunk_transcription(transcription_text, episode_info)
        return chunks
    
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Error chunking transcription file {transcription_file}: {e}")
        return []

