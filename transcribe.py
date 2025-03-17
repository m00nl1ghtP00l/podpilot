import os
import argparse
from pathlib import Path
from openai import OpenAI
import json
from datetime import datetime
from zoneinfo import ZoneInfo
from datetime import timezone

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

def transcribe_audio(file_path, client):
    """Transcribe audio using OpenAI Whisper API"""
    try:
        with open(file_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="verbose_json",
                language="ja",
                prompt="この音声は日本語です。できるだけ正確に文字起こししてください。文末に改行を入れてください."
            )
            
            transcript_dict = transcript.model_dump()
            base_filename = Path(file_path).stem
            
            # Save JSON output
            output_path = Path(file_path).parent / f"{base_filename}.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(transcript_dict, f, ensure_ascii=False, indent=2)
            
            # Save plain text
            text_path = Path(file_path).parent / f"{base_filename}.txt"
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write(transcript_dict['text'])
                
            print(f"Transcription saved to {output_path}")
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

def main():
    parser = argparse.ArgumentParser(description='Transcribe Japanese audio files using OpenAI Whisper API')
    parser.add_argument('audio_dir', help='Directory containing audio files')
    parser.add_argument('metadata_file', help='JSON file containing video metadata')
    parser.add_argument('--api-key', help='OpenAI API key (or set OPENAI_API_KEY env variable)')
    parser.add_argument('--from-date', type=parse_date_arg, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--to-date', type=parse_date_arg, help='End date (YYYY-MM-DD)')
    parser.add_argument('--retranscribe', action='store_true', 
                       help='Retranscribe files even if they already have transcriptions')
    args = parser.parse_args()
    
    # Check for API key
    api_key = args.api_key or os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("Error: OpenAI API key not provided. Use --api-key or set OPENAI_API_KEY environment variable")
        exit(1)
        
    client = OpenAI(api_key=api_key)
    
    # Check audio directory
    audio_path = Path(args.audio_dir)
    if not audio_path.exists():
        print(f"Error: Directory not found: {args.audio_dir}")
        exit(1)
    
    # Load metadata to get valid filenames
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
    
    if not mp3_files:
        print(f"No matching MP3 files found in {args.audio_dir}")
        exit(1)
    
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
        for file, date in date_skipped:
            print(f"  {file.name} ({date.date()})")
            
    if already_transcribed:
        print(f"\nSkipping {len(already_transcribed)} already transcribed files (use --retranscribe to override):")
        for file, date in already_transcribed:
            print(f"  {file.name} ({date.date()})")
            
    if parsing_failed:
        print(f"\nSkipping {len(parsing_failed)} files with invalid date format:")
        for file in parsing_failed:
            print(f"  {file.name}")
    
    if not files_to_process:
        print("\nNo files to process")
        exit(0)
    
    # Process files
    success = 0
    failed = 0
    
    for audio_file, file_date in files_to_process:
        print(f"\nProcessing: {audio_file.name} (date: {file_date.date()})")
        if transcribe_audio(str(audio_file), client):
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