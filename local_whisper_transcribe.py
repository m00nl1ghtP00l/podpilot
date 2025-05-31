#!/usr/bin/env python3
import os
import argparse
from pathlib import Path
import json
import re
from datetime import datetime
import subprocess
from tqdm import tqdm
from datetime import timezone
import shutil

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
            # Skip lines with timestamp and YouTube URL
            if re.match(r'^\[\d{2}:\d{2}:\d{2}\.\d{3}\] https://www\.youtube\.com/watch\?v=[^&\s]+&t=\d+', line):
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
            
            # Look for the video with matching filename
            for video in metadata.get('videos', []):
                if video.get('clean_filename') == base_name:
                    # Check for video ID or URL
                    video_id = video.get('id')
                    
                    # If it's a URL, extract the ID
                    if video_id and ('youtube.com' in video_id or 'youtu.be' in video_id):
                        if 'youtube.com/watch?v=' in video_id:
                            return video_id.split('youtube.com/watch?v=')[1].split('&')[0]
                        elif 'youtu.be/' in video_id:
                            return video_id.split('youtu.be/')[1].split('?')[0]
                    else:
                        # It might be just the ID
                        return video_id
        
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
    """Load metadata from JSON file and extract valid filenames"""
    try:
        with open(metadata_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        if 'videos' not in data:
            print(f"Error: Invalid metadata format. 'videos' key not found in {metadata_file}")
            return set()
            
        valid_filenames = set()
        for video in data['videos']:
            # Check for clean_filename field, which is used in download_audio.py
            if 'clean_filename' in video:
                valid_filenames.add(f"{video['clean_filename']}.mp3")
                
        return valid_filenames
    except Exception as e:
        print(f"Error loading metadata file {metadata_file}: {e}")
        return set()

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

def main():
    parser = argparse.ArgumentParser(description='Transcribe audio files using whisper-cli')
    parser.add_argument('audio_dir', nargs='?', help='Directory containing audio files (not required with --single-file)')
    parser.add_argument('--metadata-file', help='JSON file containing video metadata (optional)')
    parser.add_argument('--from-date', type=parse_date_arg, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--to-date', type=parse_date_arg, help='End date (YYYY-MM-DD)')
    parser.add_argument('--language', default='ja', help='Language code (default: ja for Japanese)')
    parser.add_argument('--retranscribe', action='store_true', 
                       help='Retranscribe files even if they already have transcriptions')
    parser.add_argument('--single-file', help='Transcribe a single file instead of a directory')
    parser.add_argument('--model-path', help='Path to whisper.cpp model file')
    parser.add_argument('--test-duration', type=float, 
                   help='Process only this many seconds of audio (minimum 1 second)')
    args = parser.parse_args()
    
    # Get configuration
    config = get_config_from_environment()
    
    # Override with command-line arguments if provided
    if args.model_path:
        config["model_path"] = args.model_path
    
    # Check if we have the required configuration
    if "executable" not in config:
        print("Error: whisper-cli executable not found in PATH")
        exit(1)
        
    if "model_path" not in config:
        print("Error: Model path not specified. Please set WHISPER_MODEL_PATH environment variable or use --model-path")
        print("Example: export WHISPER_MODEL_PATH=/path/to/ggml-large-v2.bin")
        exit(1)
        
    print(f"Using whisper executable: {config['executable']}")
    print(f"Using model: {config['model_path']}")
    
    # Handle single file mode
    if args.single_file:
        file_path = Path(args.single_file)
        if not file_path.exists():
            print(f"Error: File not found: {args.single_file}")
            exit(1)
            
        success = transcribe_audio(file_path, config, args.language, args.test_duration)
        exit(0 if success else 1)
    
    # If we're here, we need audio_dir
    if not args.audio_dir:
        print("Error: audio_dir is required when not using --single-file")
        parser.print_help()
        exit(1)
        
    # Check audio directory
    audio_path = Path(args.audio_dir)
    if not audio_path.exists():
        print(f"Error: Directory not found: {args.audio_dir}")
        exit(1)
    
    # Get list of MP3 files
    if args.metadata_file:
        # Use metadata file to filter valid files
        valid_filenames = load_metadata(args.metadata_file)
        if not valid_filenames:
            print(f"No valid filenames found in metadata file: {args.metadata_file}")
            exit(1)
            
        print(f"Found {len(valid_filenames)} valid files in metadata")
        
        # Get list of MP3 files that match the metadata
        mp3_files = []
        for filename in valid_filenames:
            file_path = audio_path / filename
            if file_path.exists():
                mp3_files.append(file_path)
                
        # Report files in metadata but not found in directory
        missing_files = []
        for filename in valid_filenames:
            file_path = audio_path / filename
            if not file_path.exists():
                missing_files.append(filename)
        
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
