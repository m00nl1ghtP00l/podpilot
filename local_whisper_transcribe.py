#!/usr/bin/env python3
import os
import sys
import argparse
from pathlib import Path
import json
import re
from datetime import datetime
import subprocess
from tqdm import tqdm
from datetime import timezone
import shutil
from find_podcasts import load_config, find_podcast_by_name, sanitize_filename

def parse_date_arg(date_str):
    """Parse date argument from command line"""
    if not date_str:
        return None
    try:
        # First try ISO format
        return datetime.fromisoformat(date_str)
    except ValueError:
        try:
            # Then try just date
            date = datetime.strptime(date_str, '%Y-%m-%d')
            return date.replace(tzinfo=timezone.utc)
        except ValueError:
            raise argparse.ArgumentTypeError(
                'Invalid date format. Use YYYY-MM-DD or ISO format'
            )

def get_file_date(filename):
    """Extract date from filename in YYYY-MM-DD format"""
    try:
        date_str = filename.split('_')[0]
        date = datetime.strptime(date_str, '%Y-%m-%d')
        return date.replace(hour=12, tzinfo=timezone.utc)
    except (ValueError, IndexError):
        return None

def is_date_in_range(date, from_date, to_date):
    """Check if date falls within specified range"""
    if not date:
        return False
    
    date_only = date.date()
    
    if from_date and date_only < from_date.date():
        return False
        
    if to_date and date_only > to_date.date():
        return False
        
    return True

def check_existing_transcription(file_path):
    """Check if transcription files already exist for this audio file"""
    base_path = Path(file_path).with_suffix('')
    json_exists = base_path.with_suffix('.json').exists()
    txt_exists = base_path.with_suffix('.txt').exists()
    return json_exists or txt_exists

def find_whisper_executable():
    """Find the whisper-cli executable in PATH"""
    return shutil.which("whisper-cli")

def parse_srt_to_segments(srt_path):
    segments = []
    with open(srt_path, 'r', encoding='utf-8') as f:
        content = f.read()

    blocks = content.strip().split('\n\n')
    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) >= 3:
            timestamp_line = lines[1]
            text = ' '.join(lines[2:]).strip()
            match = re.match(r'(\d{2}:\d{2}:\d{2}),\d{3} --> (\d{2}:\d{2}:\d{2}),\d{3}', timestamp_line)
            if match:
                start_time = match.group(1) + '.000'
                end_time = match.group(2) + '.000'
                segments.append({'start': start_time, 'end': end_time, 'text': text})
    return segments

def write_clean_transcript(text_path):
    clean_path = text_path.with_name(text_path.stem + "_transcript.txt")
    with open(text_path, "r", encoding="utf-8") as fin, open(clean_path, "w", encoding="utf-8") as fout:
        block = []
        for line in fin:
            # Skip lines with timestamps (with or without YouTube URL)
            # Pattern 1: Timestamp with YouTube URL
            if re.match(r'^\[\d{2}:\d{2}:\d{2}\.\d{3}\] https://www\.youtube\.com/watch\?v=[^&\s]+&t=\d+', line):
                continue
            # Pattern 2: Timestamp only (no URL)
            if re.match(r'^\[\d{2}:\d{2}:\d{2}\.\d{3}\]\s*$', line):
                continue
            # Write block on blank line
            if line.strip() == "":
                if block:
                    fout.write("".join(block).strip() + "\n\n")
                    block = []
            else:
                block.append(line)
        if block:
            fout.write("".join(block).strip() + "\n")

def transcribe_audio(file_path, model_config, language="ja", test_duration=None, video_id=None):
    try:
        print(f"Transcribing: {file_path}")

        whisper_executable = model_config.get("executable")
        model_path = model_config.get("model_path")

        if not whisper_executable:
            print("Error: whisper-cli executable not found in PATH")
            return False

        input_file_path = Path(file_path)
        base_filename = input_file_path.stem
        output_dir = input_file_path.parent
        srt_path = input_file_path.with_suffix(input_file_path.suffix + ".srt")
        text_path = output_dir / f"{base_filename}.txt"

        cmd = [
            whisper_executable,
            "-m", model_path,
            "-l", language,
            "-osrt",
            "-f", str(input_file_path)
        ]

        if test_duration is not None and test_duration >= 1:
            ms_duration = max(int(test_duration * 1000), 1000)
            cmd.extend(["-d", str(ms_duration)])

        print(f"Running command: {' '.join(cmd)}")
        process = subprocess.run(cmd, capture_output=True, text=True)

        if process.returncode != 0:
            print(f"Error running whisper-cli: {process.stderr}")
            return False

        print(f"Checking for SRT file at: {srt_path}")
        if not srt_path.exists():
            print(f"Error: SRT file not found at {srt_path}")
            return False

        segments = parse_srt_to_segments(srt_path)
        print(f"Parsed {len(segments)} segments from SRT file")
        if not segments:
            print("No segments found in SRT file")
            return False

        if not video_id:
            video_id = get_video_id_from_metadata(file_path)

        with open(text_path, 'w', encoding='utf-8') as f:
            for segment in segments:
                start_time = segment['start']
                text = segment['text']
                h, m, s = start_time.split(':')
                total_seconds = int(h) * 3600 + int(m) * 60 + int(float(s))
                timestamp = f"[{start_time}]"
                if video_id:
                    youtube_link = f"https://www.youtube.com/watch?v={video_id}&t={total_seconds}"
                    f.write(f"{timestamp} {youtube_link}\n")
                else:
                    f.write(f"{timestamp}\n")
                f.write(f"{text}\n\n")
        print(f"Formatted transcript saved to {text_path}")
        # Write clean transcript
        write_clean_transcript(text_path)
        print(f"Clean transcript saved to {text_path.with_name(text_path.stem + '_transcript.txt')}")
        return True

    except Exception as e:
        print(f"Error transcribing {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return False

def get_video_id_from_metadata(file_path):
    """Try to extract YouTube video ID from metadata file or filename"""
    try:
        # Get the base name of the audio file
        base_name = Path(file_path).stem
        
        # Check if metadata file exists in the same directory
        metadata_path = Path(file_path).parent.parent / f"{Path(file_path).parent.name}.json"
        
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # Use sanitize_filename for consistent matching (handles underscore differences)
            from find_podcasts import sanitize_filename
            sanitized_base_name = sanitize_filename(base_name)
            
            # Look for the video with matching filename
            for video in metadata.get('videos', []):
                clean_fn = video.get('clean_filename', '')
                if clean_fn:
                    sanitized_clean_fn = sanitize_filename(clean_fn)
                    if sanitized_clean_fn == sanitized_base_name:
                        # Check for video ID or URL
                        video_id = video.get('id')
                        
                        # If it's a URL, extract the ID
                        if video_id and ('youtube.com' in video_id or 'youtu.be' in video_id):
                            if 'youtube.com/watch?v=' in video_id:
                                return video_id.split('youtube.com/watch?v=')[1].split('&')[0]
                            elif 'youtu.be/' in video_id:
                                return video_id.split('youtu.be/')[1].split('?')[0]
                        elif video_id:
                            # It might be just the ID (clean it - remove leading/trailing underscores)
                            cleaned_id = video_id.strip('_')
                            # If it starts with underscore, it's likely part of the filename, extract just the ID part
                            if cleaned_id and len(cleaned_id) == 11:  # YouTube IDs are 11 characters
                                return cleaned_id
                            # Otherwise try to extract from the ID if it contains the actual ID
                            # Sometimes the ID might be like "_iwGxHbltp4" - extract the last 11 chars
                            if len(video_id) >= 11:
                                # Try to find a valid YouTube ID pattern (11 alphanumeric chars)
                                import re
                                id_match = re.search(r'([A-Za-z0-9_-]{11})', video_id)
                                if id_match:
                                    return id_match.group(1).strip('_')
                            return cleaned_id if cleaned_id else video_id
        
        return None
    except Exception as e:
        print(f"Error extracting video ID: {e}")
        return None

def segment_japanese_text(text):
    """Split Japanese text into sentences.
    
    Japanese sentences typically end with punctuation marks like 。, ？, or ！.
    This function splits the text at these marks and ensures each sentence is on its own line.
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


def load_metadata(metadata_file):
    """Load metadata from JSON file and return the full metadata dictionary"""
    try:
        with open(metadata_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        if 'videos' not in data:
            print(f"Error: Invalid metadata format. 'videos' key not found in {metadata_file}")
            return None
            
        return data  # Return the full metadata dictionary instead of just filenames
    except Exception as e:
        print(f"Error loading metadata file {metadata_file}: {e}")
        return None

def expand_env_vars(value: str) -> str:
    """Expand environment variables in a string (supports $VAR and ${VAR} syntax)"""
    if not isinstance(value, str):
        return value
    import re
    # Replace ${VAR} or $VAR with environment variable value
    def replace_var(match):
        var_name = match.group(1) or match.group(2)
        return os.environ.get(var_name, match.group(0))  # Return original if not found
    return re.sub(r'\$\{(\w+)\}|\$(\w+)', replace_var, value)

def get_config_from_environment():
    """Get configuration from environment variables"""
    config = {}
    
    # Look for WHISPER_MODEL_PATH environment variable
    model_path = os.environ.get("WHISPER_MODEL_PATH")
    if model_path and os.path.exists(model_path):
        config["model_path"] = model_path
        
    # Look for whisper-cli executable
    executable = find_whisper_executable()
    if executable:
        config["executable"] = executable
        
    return config

def find_audio_path_by_id(metadata, video_id, audio_dir):
    for video in metadata.get('videos', []):
        if video.get('id') == video_id:
            clean_filename = video.get('clean_filename')
            if clean_filename:
                return str(Path(audio_dir) / (clean_filename + '.mp3'))
    return None

def main():
    parser = argparse.ArgumentParser(
        description='Transcribe audio files using local whisper-cli. '
                    'Requires WHISPER_MODEL_PATH environment variable or --model-path option.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        usage='%(prog)s [-h] [--config CONFIG] [--name NAME] [options]',
        epilog='''Metadata file selection:
  Config-based: use --name (requires --config, which defaults to config/podcasts.json)

Examples:
  # Transcribe single file:
  local_whisper_transcribe.py --single-file audio.mp3

  # Transcribe directory with date filter (using config):
  local_whisper_transcribe.py --name hnh -a ./audio --from-date 2024-01-01 --to-date 2024-01-31

  # Transcribe by video ID (using config):
  local_whisper_transcribe.py --name hnh --id abc123 -a ./audio'''
    )
    parser.add_argument('--config', default='config/podcasts.json',
                       help='Path to podcasts.json config file (default: config/podcasts.json, use with --name)')
    parser.add_argument('--name',
                       help='Short name of the podcast from config file. Use with --config to find metadata JSON file.')
    parser.add_argument('-f', '--from-date', type=parse_date_arg, 
                       help='Start date for filtering files (YYYY-MM-DD or ISO format). Use with --to-date for date range.')
    parser.add_argument('-t', '--to-date', type=parse_date_arg, 
                       help='End date for filtering files (YYYY-MM-DD or ISO format). Use with --from-date for date range.')
    parser.add_argument('--language', default='ja', 
                       help='Language code for transcription (default: ja for Japanese). Examples: en, ja, es, fr')
    parser.add_argument('--retranscribe', action='store_true', 
                       help='Retranscribe files even if transcription files (.json or .txt) already exist')
    parser.add_argument('--single-file', 
                       help='Transcribe a single file instead of processing a directory. Overrides -a option.')
    parser.add_argument('--model-path', 
                       help='Path to whisper.cpp model file. Overrides WHISPER_MODEL_PATH environment variable.')
    parser.add_argument('--test-duration', type=float, 
                       help='Process only this many seconds of audio (minimum 1 second). Useful for testing.')
    parser.add_argument('--id', 
                       help='Transcribe file by video ID from metadata. Requires --name and -a/--audio-dir options.')
    parser.add_argument('-a', '--audio-dir',
                       help='Directory containing audio files. Required when processing directory (not with --single-file). Also required with --id.',
                       dest='audio_dir')
    
    # If run with no arguments, show usage and help to guide the user.
    if len(sys.argv) == 1:
        parser.print_usage()
        print()
        parser.print_help()
        sys.exit(1)
    
    args = parser.parse_args()
    
    # Get configuration
    config = get_config_from_environment()
    
    # Override with command-line arguments if provided
    if args.model_path:
        config["model_path"] = args.model_path
    
    # Determine metadata file from args (config + name mode)
    metadata_file = None
    config_data = None
    if args.name:
        try:
            config_data = load_config(args.config)
            podcast = find_podcast_by_name(config_data, args.name)
            
            # Apply transcription settings from config if present
            transcription_cfg = config_data.get("transcription", {})
            provider = transcription_cfg.get("provider")
            if provider and provider not in ("whisper.cpp", "whispercpp"):
                parser.error(f"Transcription provider '{provider}' is not supported in local_whisper_transcribe.py (only whisper.cpp).")
            # Resolve model_path: prefer CLI, then env from get_config_from_environment, then config
            if "model_path" not in config:
                mp_cfg = transcription_cfg.get("model_path")
                if mp_cfg:
                    # Expand environment variables in model_path
                    config["model_path"] = expand_env_vars(mp_cfg)
            if "executable" not in config and "executable" in transcription_cfg:
                config["executable"] = transcription_cfg["executable"]
            
            # Use data_root from config to find metadata file (same pattern as other scripts)
            data_root = Path(config_data.get("data_root", "."))
            channel_name = podcast["channel_name_short"]
            metadata_file = str(data_root / f"{channel_name}.json")
            
            # If audio_dir not specified, use data_root/channel_name (like other scripts)
            if not args.audio_dir:
                args.audio_dir = str(data_root / channel_name)
        except ValueError as e:
            parser.error(f"Error: {e}")
    
    # Validate required arguments before processing IDs
    if args.id:
        if not args.name or not args.audio_dir:
            parser.error('--name and -a/--audio-dir are required with --id')
    
    if args.single_file:
        file_path = Path(args.single_file)
        if not file_path.exists():
            parser.error(f'File not found: {args.single_file}')
    
    if not args.single_file and not args.audio_dir:
        parser.error("-a/--audio-dir is required when not using --single-file")
    
    if "executable" not in config:
        print("Error: whisper-cli executable not found in PATH and not set in config")
        exit(1)
        
    if "model_path" not in config:
        print("Error: Model path not specified. Set transcription.model_path in config or use --model-path/WHISPER_MODEL_PATH")
        exit(1)
    
    print(f"Using whisper executable: {config['executable']}")
    print(f"Using model: {config['model_path']}")
    
    if args.id:
        metadata = load_metadata(metadata_file)
        if not metadata:
            parser.error(f"Error loading metadata file: {metadata_file}")
        file_path = find_audio_path_by_id(metadata, args.id, args.audio_dir)
        if file_path is None:
            parser.error(f'Video ID {args.id} not found in metadata file')
        if not Path(file_path).exists():
            parser.error(f'Audio file for ID {args.id} not found in directory: {file_path}')
        transcribe_audio(file_path, config, args.language, args.test_duration)
        exit(0)

    if args.single_file:
        transcribe_audio(str(file_path), config, args.language, args.test_duration)
        exit(0)
        
    # Check audio directory
    audio_path = Path(args.audio_dir)
    if not audio_path.exists():
        print(f"Error: Directory not found: {args.audio_dir}")
        exit(1)
    
    # Get list of MP3 files
    if metadata_file:
        # Use metadata file to filter valid files
        metadata = load_metadata(metadata_file)
        if not metadata:
            print(f"Error loading metadata file: {metadata_file}")
            exit(1)
        
        # Sanitize metadata filenames to match what might be on disk
        valid_filenames = {sanitize_filename(f"{video['clean_filename']}.mp3") for video in metadata.get('videos', [])}
        print(f"Found {len(valid_filenames)} valid files in metadata")
        
        # Get all MP3 files in directory and match them to metadata
        all_mp3_files = list(audio_path.glob('*.mp3'))
        mp3_files = []
        
        # Create a mapping of sanitized filenames to actual file paths
        sanitized_to_file = {sanitize_filename(f.name): f for f in all_mp3_files}
        
        # Match metadata filenames to actual files
        for metadata_filename in valid_filenames:
            if metadata_filename in sanitized_to_file:
                mp3_files.append(sanitized_to_file[metadata_filename])
        
        # If no matches found, fall back to using all MP3 files in directory
        if not mp3_files and all_mp3_files:
            print(f"Warning: No metadata matches found, using all {len(all_mp3_files)} MP3 files in directory")
            mp3_files = all_mp3_files
                
        # Report files in metadata but not found in directory
        missing_files = []
        for metadata_filename in valid_filenames:
            if metadata_filename not in sanitized_to_file:
                missing_files.append(metadata_filename)
        
        if missing_files:
            print(f"\nWARNING: {len(missing_files)} files listed in metadata were not found in the audio directory:")
            for missing in missing_files[:10]:  # Show first 10 to avoid clutter
                print(f"  {missing}")
            if len(missing_files) > 10:
                print(f"  ... and {len(missing_files) - 10} more")
    else:
        # Get all MP3 files in directory
        mp3_files = list(audio_path.glob('*.mp3'))
    
    if not mp3_files:
        print(f"No MP3 files found in {args.audio_dir}")
        exit(1)
    
    # Filter files by date and transcription status
    files_to_process = []
    date_skipped = []
    already_transcribed = []
    parsing_failed = []
    
    for file in mp3_files:
        file_date = get_file_date(file.stem)
        if not file_date:
            parsing_failed.append(file)
            continue
            
        if not is_date_in_range(file_date, args.from_date, args.to_date):
            date_skipped.append((file, file_date))
            continue
            
        if check_existing_transcription(file) and not args.retranscribe:
            already_transcribed.append((file, file_date))
            continue
            
        files_to_process.append((file, file_date))
    
    # Sort files by date
    files_to_process.sort(key=lambda x: x[1])
    
    # Report status
    print(f"\nFound {len(files_to_process)} files to process:")
    for file, date in files_to_process:
        print(f"  {file.name} ({date.date()})")
    
    if date_skipped:
        print(f"\nSkipping {len(date_skipped)} files outside date range:")
        for file, date in date_skipped[:5]:  # Show only first 5
            print(f"  {file.name} ({date.date()})")
        if len(date_skipped) > 5:
            print(f"  ... and {len(date_skipped) - 5} more")
            
    if already_transcribed:
        print(f"\nSkipping {len(already_transcribed)} already transcribed files (use --retranscribe to override)")
            
    if parsing_failed:
        print(f"\nSkipping {len(parsing_failed)} files with invalid date format")
    
    if not files_to_process:
        print("\nNo files to process")
        exit(0)
    
    # Process files
    success = 0
    failed = 0
    
    for audio_file, file_date in tqdm(files_to_process, desc="Transcribing files"):
        if transcribe_audio(str(audio_file), config, args.language, args.test_duration):
            success += 1
        else:
            failed += 1
            
    # Final report
    print(f"\nTranscription complete!")
    print(f"Successfully transcribed: {success}")
    print(f"Failed transcriptions: {failed}")
    print(f"Skipped {len(date_skipped)} files outside date range")
    print(f"Skipped {len(already_transcribed)} already transcribed files")
    print(f"Skipped {len(parsing_failed)} files with invalid date format")

if __name__ == '__main__':
    main()
