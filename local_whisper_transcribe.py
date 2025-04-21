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

def clean_transcript(text):
    """Remove timestamps from transcription"""
    # Pattern to match timestamp lines like [00:00:00.000 --> 00:00:02.000]
    pattern = r'\[\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}\]\s*'
    
    # Remove timestamps
    clean_text = re.sub(pattern, '', text)
    
    # Remove any extra whitespace
    clean_text = re.sub(r'\s+', ' ', clean_text)
    
    return clean_text.strip()

def find_whisper_executable():
    """Find the whisper-cli executable in PATH"""
    return shutil.which("whisper-cli")

def transcribe_audio(file_path, model_config, language="ja"):
    """Transcribe audio using whisper-cli"""
    try:
        print(f"Transcribing: {file_path}")
        
        # Get configuration for transcription
        whisper_executable = model_config.get("executable")
        model_path = model_config.get("model_path")
        
        if not whisper_executable:
            print("Error: whisper-cli executable not found in PATH")
            return False
            
        if not model_path:
            print("Error: Model path not specified")
            return False
            
        if not os.path.exists(model_path):
            print(f"Error: Model file not found at {model_path}")
            return False
        
        # Create command
        cmd = [
            whisper_executable,
            "-m", model_path,
            "-l", language,
            "-otxt",  # Output as text
            "-f", str(file_path)
        ]
        
        # Run the command
        print(f"Running command: {' '.join(cmd)}")
        process = subprocess.run(cmd, capture_output=True, text=True)
        
        if process.returncode != 0:
            print(f"Error running whisper-cli: {process.stderr}")
            return False
        
        # Get output text
        output_text = process.stdout
        
        # Clean the text (remove timestamps)
        clean_text = clean_transcript(output_text)
        
        # Create the output path with the correct extension
        # Use the same filename but with .txt extension instead of adding .txt
        input_file_path = Path(file_path)
        text_path = input_file_path.with_suffix('.txt')
        
        # Save plain text without timestamps
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(clean_text)
            
        print(f"Plain text saved to {text_path}")
        return True
            
    except Exception as e:
        print(f"Error transcribing {file_path}: {e}")
        return False


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
            
        success = transcribe_audio(file_path, config, args.language)
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
        if transcribe_audio(str(audio_file), config, args.language):
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
