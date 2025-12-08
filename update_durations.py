#!/usr/bin/env python3
"""
Update metadata JSON with duration information from downloaded audio files
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Tuple
from mutagen.mp3 import MP3
from mutagen import MutagenError

# Import config loading functions
try:
    from channel_fetcher import load_config, find_podcast_by_name, format_date
except ImportError:
    def load_config(config_file):
        with open(config_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def find_podcast_by_name(config, name):
        for podcast in config["youtube_channels"]:
            if podcast["channel_name_short"] == name:
                return podcast
        raise ValueError(f"Podcast '{name}' not found")


def format_duration(seconds: float) -> str:
    """Format duration in seconds to HH:MM:SS or MM:SS format."""
    if seconds < 0:
        return "00:00"
    
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes:02d}:{secs:02d}"


def update_video_duration(video: dict, duration_seconds: float) -> dict:
    """Update a single video entry with duration information.
    
    Args:
        video: Video dictionary from metadata
        duration_seconds: Duration in seconds
        
    Returns:
        Updated video dictionary
    """
    video['duration_seconds'] = int(duration_seconds)
    video['duration'] = format_duration(duration_seconds)
    return video


def get_audio_duration(audio_path: Path) -> Optional[Tuple[float, str]]:
    """Get duration from audio file. Returns (seconds, formatted) or None."""
    try:
        audio = MP3(audio_path)
        duration_seconds = audio.info.length
        duration_formatted = format_duration(duration_seconds)
        return (duration_seconds, duration_formatted)
    except MutagenError:
        return None
    except Exception:
        return None


def update_video_duration_from_file(video: dict, audio_dir: Path, min_duration: int = 0) -> Tuple[bool, bool]:
    """Update a single video's duration from its audio file.
    
    Args:
        video: Video dictionary from metadata
        audio_dir: Directory containing audio files
        min_duration: Minimum duration in seconds (0 = don't filter)
        
    Returns:
        Tuple of (updated: bool, should_keep: bool)
        - updated: True if duration was successfully updated
        - should_keep: False if video should be removed (too short)
    """
    # Skip if already has duration
    if 'duration_seconds' in video and 'duration' in video:
        duration_seconds = video.get('duration_seconds', 0)
        if min_duration > 0 and duration_seconds < min_duration:
            return (False, False)  # Already has duration, but too short
        return (False, True)  # Already has duration, keep it
    
    # Find audio file
    video_id = video.get('id', '')
    clean_filename = video.get('clean_filename', '')
    audio_path = find_audio_file(audio_dir, video_id, clean_filename)
    
    if not audio_path or not audio_path.exists():
        return (False, True)  # File not found, but keep in metadata
    
    # Get duration from audio file
    duration_info = get_audio_duration(audio_path)
    if duration_info:
        duration_seconds, duration_formatted = duration_info
        update_video_duration(video, duration_seconds)
        
        # Check if too short
        if min_duration > 0 and duration_seconds < min_duration:
            return (True, False)  # Updated but too short, remove
        
        return (True, True)  # Updated and OK, keep
    
    return (False, True)  # Couldn't get duration, but keep in metadata


def find_audio_file(audio_dir: Path, video_id: str, clean_filename: str) -> Optional[Path]:
    """Find audio file by video ID or filename matching."""
    if not video_id and not clean_filename:
        return None
    
    # Normalize video ID (remove underscores, they're not in actual YouTube IDs)
    normalized_video_id = video_id.replace('_', '') if video_id else ''
    
    # Strategy 1: Try exact match with clean_filename
    if clean_filename:
        audio_path = audio_dir / f"{clean_filename}.mp3"
        if audio_path.exists():
            return audio_path
    
    # Strategy 2: Try to find by video ID in filename (with and without underscore)
    if video_id:
        for audio_file in audio_dir.glob("*.mp3"):
            stem = audio_file.stem
            # Try both original and normalized video ID
            if video_id in stem or normalized_video_id in stem:
                return audio_file
    
    # Strategy 3: Try to match by date prefix (YYYY-MM-DD) and partial title
    if clean_filename:
        parts = clean_filename.split('_')
        if len(parts) >= 2:
            date_prefix = parts[0]
            # Get title part (everything except date and video ID)
            title_parts = parts[1:-1] if video_id and parts[-1] == video_id else parts[1:]
            title_keywords = ' '.join(title_parts[:3])  # First few words of title
            
            if date_prefix and len(date_prefix) == 10:  # YYYY-MM-DD format
                for audio_file in audio_dir.glob(f"{date_prefix}*.mp3"):
                    # Check if title keywords match
                    if any(keyword.lower() in audio_file.stem.lower() for keyword in title_parts[:2] if keyword):
                        return audio_file
    
    return None


def update_metadata_durations(metadata_file: Path, audio_dir: Path, verbose: bool = False, 
                              min_duration: int = 300, delete_audio: bool = False) -> dict:
    """Update metadata JSON with durations from audio files."""
    # Load metadata
    with open(metadata_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    if 'videos' not in metadata:
        print(f"Error: Metadata file missing 'videos' key")
        return metadata
    
    updated_count = 0
    not_found_count = 0
    already_has_duration = 0
    removed_count = 0
    
    # Process each video (iterate backwards so we can safely remove items)
    videos_to_keep = []
    
    for video in metadata['videos']:
        # Check if already has duration
        has_duration = 'duration_seconds' in video and 'duration' in video
        
        # If already has duration, check if it's too short
        if has_duration:
            duration_seconds = video.get('duration_seconds', 0)
            if duration_seconds < min_duration:
                if verbose:
                    print(f"  Removing: {video.get('title', 'Unknown')[:50]} (duration: {video.get('duration', 'N/A')} < {format_duration(min_duration)})")
                
                # Try to find and delete audio file if requested
                if delete_audio:
                    video_id = video.get('id', '')
                    clean_filename = video.get('clean_filename', '')
                    audio_path = find_audio_file(audio_dir, video_id, clean_filename)
                    if audio_path and audio_path.exists():
                        try:
                            audio_path.unlink()
                            if verbose:
                                print(f"    Deleted audio file: {audio_path.name}")
                        except Exception as e:
                            print(f"    Warning: Could not delete {audio_path.name}: {e}")
                
                removed_count += 1
                continue  # Don't add to videos_to_keep
            
            already_has_duration += 1
            videos_to_keep.append(video)  # Keep videos that already have duration and are long enough
            continue
        
        # Find corresponding audio file
        video_id = video.get('id', '')
        clean_filename = video.get('clean_filename', '')
        
        audio_path = find_audio_file(audio_dir, video_id, clean_filename)
        
        if not audio_path:
            if verbose:
                print(f"  Not found: {video.get('title', 'Unknown')[:50]} (ID: {video_id}, filename: {clean_filename})")
            not_found_count += 1
            continue
        
        if verbose:
            print(f"  Found: {video.get('title', 'Unknown')[:50]} -> {audio_path.name}")
        
        # Get duration from audio file
        duration_info = get_audio_duration(audio_path)
        if duration_info:
            duration_seconds, duration_formatted = duration_info
            video['duration_seconds'] = int(duration_seconds)
            video['duration'] = duration_formatted
            updated_count += 1
            
            # Check if duration is too short
            if duration_seconds < min_duration:
                if verbose:
                    print(f"  Removing: {video.get('title', 'Unknown')[:50]} (duration: {duration_formatted} < {format_duration(min_duration)})")
                
                # Delete audio file if requested
                if delete_audio and audio_path.exists():
                    try:
                        audio_path.unlink()
                        if verbose:
                            print(f"    Deleted audio file: {audio_path.name}")
                    except Exception as e:
                        print(f"    Warning: Could not delete {audio_path.name}: {e}")
                
                removed_count += 1
                continue  # Don't add to videos_to_keep
        
        # Add to keep list if duration is OK or not found
        videos_to_keep.append(video)
    
    # Update metadata with filtered videos
    metadata['videos'] = videos_to_keep
    
    print(f"Updated {updated_count} videos with duration information")
    if already_has_duration > 0:
        print(f"Skipped {already_has_duration} videos that already have duration")
    if removed_count > 0:
        print(f"Removed {removed_count} videos shorter than {format_duration(min_duration)} from metadata")
    if not_found_count > 0:
        print(f"Warning: {not_found_count} audio files not found")
    
    return metadata


def main():
    parser = argparse.ArgumentParser(
        description='Update metadata JSON with duration from downloaded audio files',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--name',
                       help='Short name of the podcast from config file (loads paths from config)')
    parser.add_argument('--config', default='config/podcasts.json',
                       help='Path to podcasts.json config file (use with --name)')
    parser.add_argument('--metadata-file', type=Path,
                       help='Path to metadata JSON file (required if not using --name)')
    parser.add_argument('-a', '--audio-dir', type=Path,
                       help='Directory containing audio files (required if not using --name)')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Show detailed matching information')
    parser.add_argument('--min-duration', type=int,
                       help='Minimum duration in seconds to keep (overrides config). Videos shorter than this will be removed from metadata.')
    parser.add_argument('--delete-audio', action='store_true',
                       help='Also delete audio files that are shorter than minimum duration')
    
    # If run with no arguments, show usage and help
    if len(sys.argv) == 1:
        parser.print_usage()
        print()
        parser.print_help()
        sys.exit(1)
    
    args = parser.parse_args()
    
    # Load config if using --name
    config_data = None
    if args.name:
        try:
            config_data = load_config(args.config)
        except Exception as e:
            print(f"Error loading config: {e}")
            sys.exit(1)
    
    # Determine paths
    if args.name:
        try:
            podcast = find_podcast_by_name(config_data, args.name)
            data_root = Path(config_data.get("data_root", "."))
            channel_name = podcast["channel_name_short"]
            metadata_file = data_root / f"{channel_name}.json"
            audio_dir = data_root / channel_name
            print(f"Using config:")
            print(f"  Metadata file: {metadata_file}")
            print(f"  Audio directory: {audio_dir}")
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
    else:
        if not args.metadata_file or not args.audio_dir:
            parser.error("Either --name or both --metadata-file and --audio-dir must be provided")
        metadata_file = args.metadata_file
        audio_dir = args.audio_dir
    
    # Get min_duration from config if not specified on command line
    if args.min_duration is None and config_data:
        args.min_duration = config_data.get('min_duration', 300)  # Default to 5 minutes
    elif args.min_duration is None:
        args.min_duration = 300  # Default if no config
    
    # Validate paths
    if not metadata_file.exists():
        print(f"Error: Metadata file not found: {metadata_file}")
        sys.exit(1)
    
    if not audio_dir.exists():
        print(f"Error: Audio directory not found: {audio_dir}")
        sys.exit(1)
    
    # Get min_duration from config if not specified on command line
    if args.min_duration is None and config_data:
        args.min_duration = config_data.get('min_duration', 300)  # Default to 5 minutes
    elif args.min_duration is None:
        args.min_duration = 300  # Default if no config
    
    # Update durations
    print(f"\nUpdating durations from audio files...")
    if args.min_duration and args.min_duration > 0:
        print(f"Minimum duration: {format_duration(args.min_duration)} (videos shorter will be removed)")
    updated_metadata = update_metadata_durations(
        metadata_file, audio_dir, args.verbose, args.min_duration, args.delete_audio
    )
    
    # Save updated metadata
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(updated_metadata, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Metadata updated: {metadata_file}")


if __name__ == '__main__':
    main()

