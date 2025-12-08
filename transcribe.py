#!/usr/bin/env python3
import os
import argparse
from pathlib import Path
from openai import OpenAI
import json
from datetime import datetime
from datetime import timezone
import sys

# Import functions from other modules
from channel_fetcher import clean_title, fetch_rss_feed, parse_rss_feed, load_config
from download_audio import download_file, process_existing_file

# Import language adapter system
try:
    from adapters import get_language_adapter
    ADAPTERS_AVAILABLE = True
except ImportError:
    ADAPTERS_AVAILABLE = False

# Constants
WHISPER_MODEL = "whisper-1"
DEFAULT_LANGUAGE = "ja"  # Default to Japanese for backward compatibility
DEFAULT_FILE_HOUR = 12  # Default hour for file dates (noon UTC)
YOUTUBE_VIDEO_ID_LENGTH = 11  # Standard YouTube video ID length

def print_step(message):
    """Print a step header to make progress clear"""
    print("\n" + "=" * 80)
    print(f"{message}")
    print("=" * 80 + "\n")

def print_substep(message):
    """Print a substep header"""
    print("\n" + "-" * 60)
    print(f"  {message}")
    print("-" * 60)

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
        return date.replace(hour=DEFAULT_FILE_HOUR, tzinfo=timezone.utc)
    except (ValueError, IndexError):
        return None

def parse_date_string(date_str):
    """Parse a date string in various formats to a datetime object.
    
    Supports:
    - ISO format (with or without 'Z' timezone)
    - 'YYYY-MM-DD HH:MM' format
    - 'YYYY-MM-DD' format
    
    Returns datetime with UTC timezone, or None if parsing fails.
    """
    if not date_str:
        return None
    
    if isinstance(date_str, datetime):
        return date_str
    
    try:
        # Try ISO format first
        return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
    except ValueError:
        try:
            # Try 'YYYY-MM-DD HH:MM' format
            date = datetime.strptime(date_str, '%Y-%m-%d %H:%M')
            return date.replace(tzinfo=timezone.utc)
        except ValueError:
            try:
                # Try 'YYYY-MM-DD' format
                date = datetime.strptime(date_str, '%Y-%m-%d')
                return date.replace(tzinfo=timezone.utc)
            except ValueError:
                return None

def is_date_in_range(date, from_date, to_date):
    """Check if date falls within specified range"""
    if not date:
        return False
    
    # Parse date string if needed
    parsed_date = parse_date_string(date) if isinstance(date, str) else date
    if not parsed_date:
        return False
    
    date_only = parsed_date.date()
    
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

def transcribe_audio(file_path, client, language=None, prompt=None):
    """Transcribe audio using OpenAI Whisper API
    
    Args:
        file_path: Path to audio file
        client: OpenAI client instance
        language: Language code (defaults to 'ja' for backward compatibility)
        prompt: Transcription prompt (defaults to language-specific prompt if adapter available)
    """
    # Use defaults if not provided
    if language is None:
        language = DEFAULT_LANGUAGE
    if prompt is None:
        # Try to get prompt from language adapter
        if ADAPTERS_AVAILABLE:
            adapter = get_language_adapter(language)
            if adapter:
                prompt = adapter.get_transcription_prompt()
        # Fallback to Japanese prompt if no adapter
        if prompt is None:
            prompt = "この音声は日本語です。できるだけ正確に文字起こししてください。文末に改行を入れてください."
    
    try:
        print(f"Starting transcription of {os.path.basename(file_path)}...")
        with open(file_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model=WHISPER_MODEL,
                file=audio_file,
                response_format="verbose_json",
                language=language,
                prompt=prompt
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
                
            print(f"✓ Transcription saved to {output_path}")
            print(f"✓ Plain text saved to {text_path}")
            return True
            
    except Exception as e:
        print(f"✗ Error transcribing {file_path}: {e}")
        return False

def load_config(config_file):
    """Load and validate config file"""
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # Validate required fields
        if "data_root" not in config:
            raise ValueError("Config file missing 'data_root' field")
        if "youtube_channels" not in config or not isinstance(config["youtube_channels"], list):
            raise ValueError("Config file missing 'youtube_channels' list or it's not a list")
        
        # Validate each podcast entry
        for i, podcast in enumerate(config["youtube_channels"]):
            if "channel_name_short" not in podcast:
                raise ValueError(f"Podcast at index {i} missing 'channel_name_short'")
            if "channel_name_long" not in podcast:
                raise ValueError(f"Podcast at index {i} missing 'channel_name_long'")
            if "channel_id" not in podcast:
                raise ValueError(f"Podcast at index {i} missing 'channel_id'")
        
        return config
    except json.JSONDecodeError as e:
        line = int(str(e).split("line", 1)[1].split()[0])
        raise ValueError(f"JSON syntax error in config file (line {line}): {e}")
    except Exception as e:
        raise ValueError(f"Error reading config file: {e}")

def find_podcast_by_name(config, name):
    """Find a podcast in the config by its channel_name_short"""
    for podcast in config["youtube_channels"]:
        if podcast["channel_name_short"] == name:
            return podcast
    
    # Podcast not found, list available options
    available = [p["channel_name_short"] for p in config["youtube_channels"]]
    raise ValueError(f"Podcast '{name}' not found in config. Available options: {', '.join(available)}")

def list_podcasts(config):
    """List all podcasts in the config"""
    print_step("Available Podcasts")
    
    for i, podcast in enumerate(config["youtube_channels"], 1):
        print(f"{i}. {podcast['channel_name_short']} - {podcast['channel_name_long']}")
        print(f"   Channel ID: {podcast['channel_id']}")
    
    return True

def load_json(json_file):
    """Load JSON data from file"""
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading JSON file: {e}")
        return None

def save_json(data, json_file):
    """Save JSON data to file"""
    try:
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"✓ Data saved to {json_file}")
        return True
    except Exception as e:
        print(f"✗ Error saving JSON file: {e}")
        return False

def fetch_podcast_data(channel_id, output_file, from_date, to_date, simulate=False):
    """Fetch podcast data from YouTube channel"""
    print_substep(f"Fetching podcast data for channel: {channel_id}")
    
    if simulate:
        print("Would fetch podcast data from YouTube RSS feed")
        return True
    
    try:
        # Fetch and parse RSS feed
        print("Fetching RSS feed...")
        xml_content = fetch_rss_feed(channel_id)
        
        print("Parsing RSS feed...")
        channel_info = parse_rss_feed(
            xml_content, 
            from_date=from_date, 
            to_date=to_date,
            include_author=True,
            include_description=True
        )
        
        print(f"Found {len(channel_info['videos'])} videos in the specified date range")
        
        # Save JSON data
        save_json(channel_info, output_file)
        return True
    except Exception as e:
        print(f"✗ Error fetching podcast data: {e}")
        return False

def download_audio_files(json_file, audio_dir, from_date, to_date, force_download=False, simulate=False):
    """Download audio files from JSON data"""
    print_substep("Downloading audio files")
    
    # Create output directory if it doesn't exist
    audio_path = Path(audio_dir)
    audio_path.mkdir(parents=True, exist_ok=True)
    
    # Load JSON data
    json_data = load_json(json_file)
    if not json_data:
        return False
    
    # Get list of existing files
    existing_files = {f.name for f in audio_path.iterdir() if f.is_file()}
    
    downloads = []
    process_existing = []
    
    print("Analyzing podcast data...")
    for video in json_data['videos']:
        # Use published date if available, otherwise fall back to date
        date_str = video.get('published', video.get('date'))
        if not date_str:
            print(f"Warning: No date found for video {video.get('title', 'Unknown')}")
            continue
            
        if not is_date_in_range(date_str, from_date, to_date):
            continue
        
        # Extract necessary information
        title = video.get('title', '')
        
        # Check for URL in various potential fields
        url = None
        for url_field in ['url', 'link', 'id']:
            if url_field in video and video[url_field]:
                url_value = video[url_field]
                
                # Handle YouTube video IDs
                if url_field == 'id' and len(url_value) == YOUTUBE_VIDEO_ID_LENGTH and not url_value.startswith('http'):
                    url = f"https://www.youtube.com/watch?v={url_value}"
                else:
                    url = url_value
                
                break
        
        if not url:
            print(f"Warning: No URL found for video {title}")
            continue
        
        # Parse the date using shared utility function
        date = parse_date_string(date_str)
        if not date:
            print(f"Warning: Could not parse date {date_str} for video {title}")
            continue
        
        # Use clean_filename from channel_fetcher if available, otherwise generate one
        if 'clean_filename' in video:
            filename = f"{video['clean_filename']}.mp3"
        else:
            # Generate cleaned title
            clean_title_str = clean_title(title)
            date_prefix = date.strftime('%Y-%m-%d')
            filename = f"{date_prefix}_{clean_title_str}.mp3"
        
        output_path = Path(audio_dir) / filename

        if filename in existing_files and not force_download:
            process_existing.append((output_path, date))
            continue
            
        downloads.append((url, output_path, filename, date))
    
    # Sort by date
    downloads.sort(key=lambda x: x[3])
    process_existing.sort(key=lambda x: x[1])
    
    print(f"Found {len(downloads)} files to download:")
    for _, _, filename, _ in downloads:
        print(f"  {filename}")
    
    if process_existing:
        print(f"\nFound {len(process_existing)} existing files:")
        for file_path, _ in process_existing:
            print(f"  {os.path.basename(file_path)}")
    
    if not downloads and not process_existing:
        print("No files to process")
        return True
    
    # Process existing files
    for file_path, _ in process_existing:
        process_existing_file(str(file_path), simulate)
    
    if simulate:
        print("\nSimulation mode: would download files")
        return True
        
    # Download new files
    success = 0
    failed = 0
    
    for url, output_path, filename, _ in downloads:
        print(f"\nDownloading: {filename}")
        print(f"URL: {url}")
        if download_file(url, output_path, simulate):
            success += 1
        else:
            failed += 1
    
    if downloads:
        print(f"\nDownloads complete: {success} successful, {failed} failed")
    
    return True

def process_transcriptions(audio_dir, json_file, from_date, to_date, retranscribe=False, api_key=None, simulate=False, language=None, prompt=None):
    """Process audio transcriptions"""
    print_substep("Processing transcriptions")
    
    if simulate:
        print("Simulation mode: would process transcriptions")
        return True
    
    # Check for API key
    if not api_key:
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            print("Error: OpenAI API key not provided. Use --api-key or set OPENAI_API_KEY environment variable")
            return False
    
    # Create OpenAI client
    client = OpenAI(api_key=api_key)
    
    # Load metadata to get valid filenames
    print("Loading metadata...")
    json_data = load_json(json_file)
    if not json_data:
        return False
    
    valid_filenames = set()
    for video in json_data['videos']:
        if 'clean_filename' in video:
            valid_filenames.add(f"{video['clean_filename']}.mp3")
    
    print(f"Found {len(valid_filenames)} valid files in metadata")
    
    # Get list of MP3 files that match the metadata
    audio_path = Path(audio_dir)
    mp3_files = []
    for filename in valid_filenames:
        file_path = audio_path / filename
        if file_path.exists():
            mp3_files.append(file_path)
    
    # If we don't have metadata references, just look for all .mp3 files
    if not mp3_files and audio_path.exists():
        mp3_files = list(audio_path.glob("*.mp3"))
        print(f"No metadata matches found, using all {len(mp3_files)} MP3 files in directory")
    
    if not mp3_files:
        print(f"No MP3 files found in {audio_dir}")
        return False
    
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
            
        if not is_date_in_range(file_date, from_date, to_date):
            date_skipped.append((file, file_date))
            continue
            
        if check_existing_transcription(file) and not retranscribe:
            already_transcribed.append((file, file_date))
            continue
            
        files_to_process.append((file, file_date))
    
    # Sort files by date
    files_to_process.sort(key=lambda x: x[1])
    
    # Report status
    print(f"\nFound {len(files_to_process)} files to transcribe:")
    for file, date in files_to_process:
        print(f"  {file.name} ({date.date()})")
    
    if date_skipped:
        print(f"\nSkipping {len(date_skipped)} files outside date range")
        if len(date_skipped) <= 5:
            for file, date in date_skipped:
                print(f"  {file.name} ({date.date()})")
        else:
            for file, date in date_skipped[:5]:
                print(f"  {file.name} ({date.date()})")
            print(f"  ... and {len(date_skipped) - 5} more")
            
    if already_transcribed:
        print(f"\nSkipping {len(already_transcribed)} already transcribed files")
        if len(already_transcribed) <= 5:
            for file, date in already_transcribed:
                print(f"  {file.name} ({date.date()})")
        else:
            for file, date in already_transcribed[:5]:
                print(f"  {file.name} ({date.date()})")
            print(f"  ... and {len(already_transcribed) - 5} more")
            
    if parsing_failed:
        print(f"\nSkipping {len(parsing_failed)} files with invalid date format")
        if len(parsing_failed) <= 5:
            for file in parsing_failed:
                print(f"  {file.name}")
        else:
            for file in parsing_failed[:5]:
                print(f"  {file.name}")
            print(f"  ... and {len(parsing_failed) - 5} more")
    
    if not files_to_process:
        print("\nNo files to transcribe")
        return True
    
    # Process files
    success = 0
    failed = 0
    
    for audio_file, file_date in files_to_process:
        print(f"\nProcessing: {audio_file.name} (date: {file_date.date()})")
        if transcribe_audio(str(audio_file), client, language=language, prompt=prompt):
            success += 1
        else:
            failed += 1
            
    # Final report
    print(f"\nTranscription complete!")
    print(f"Successfully transcribed: {success}")
    print(f"Failed transcriptions: {failed}")
    
    return True

def main():
    parser = argparse.ArgumentParser(
        description='Podcast downloading and transcription tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        usage='%(prog)s [-h] [--config CONFIG] (--name NAME | --list) [options]',
        epilog='''REQUIRED: Either --name or --list must be provided.
  --name: Process a specific podcast (REQUIRED unless using --list)
  --list: List all podcasts in config file (alternative to --name)'''
    )
    
    parser.add_argument('--config', default='config/podcasts.json', 
                       help='JSON config file with podcast information (default: config/podcasts.json)')
    parser.add_argument('--name', help='Short name of the podcast to process from config file (REQUIRED unless using --list)')
    parser.add_argument('--list', action='store_true', help='List all podcasts in the config (alternative to --name)')
    
    # Date range options
    parser.add_argument('-f', '--from-date', type=parse_date_arg, help='Start date (YYYY-MM-DD)')
    parser.add_argument('-t', '--to-date', type=parse_date_arg, help='End date (YYYY-MM-DD)')
    
    # Processing options
    parser.add_argument('--download', action='store_true', help='Force download of audio files even if they exist')
    parser.add_argument('--retranscribe', action='store_true', help='Retranscribe files even if they already have transcriptions')
    parser.add_argument('--force', action='store_true', help='Force all operations regardless of existing files')
    parser.add_argument('-s', '--simulate', action='store_true', help='Simulation mode (no actual downloads/transcriptions)')
    parser.add_argument('--api-key', help='OpenAI API key (or set OPENAI_API_KEY env variable)')
    
    # If run with no arguments, show usage and help to explain how to drive the tool.
    if len(sys.argv) == 1:
        parser.print_usage()
        print()
        parser.print_help()
        return 1
    
    args = parser.parse_args()
    
    # Load config file
    try:
        config = load_config(args.config)
    except ValueError as e:
        print(f"Error: {e}")
        return 1
    
    # Load language adapter from config
    language = None
    prompt = None
    if ADAPTERS_AVAILABLE:
        language_code = config.get('language', DEFAULT_LANGUAGE)
        language = language_code
        adapter = get_language_adapter(language_code)
        if adapter:
            prompt = adapter.get_transcription_prompt()
            print(f"Using language adapter: {adapter.language_name} ({adapter.language_code})")
    
    # List podcasts if requested
    if args.list:
        list_podcasts(config)
        return 0
    
    # Check if a podcast name is provided
    if not args.name:
        print("Error: --name is required to specify which podcast to process")
        print("Use --list to see available podcasts")
        return 1
    
    # Find the requested podcast
    try:
        podcast = find_podcast_by_name(config, args.name)
    except ValueError as e:
        print(f"Error: {e}")
        return 1
    
    # Setup paths
    data_root = Path(config["data_root"])
    channel_name = podcast["channel_name_short"]
    audio_dir = data_root / channel_name
    metadata_file = data_root / f"{channel_name}.json"
    
    print("=" * 80)
    print(f"PODCAST: {podcast['channel_name_long']} ({channel_name})".center(80))
    print("=" * 80)
    
    print(f"Channel ID: {podcast['channel_id']}")
    print(f"Audio directory: {audio_dir}")
    print(f"Metadata file: {metadata_file}")
    
    if args.from_date or args.to_date:
        print(f"Date range: {args.from_date.date() if args.from_date else 'Any'} to {args.to_date.date() if args.to_date else 'Any'}")
    
    if args.simulate:
        print("SIMULATION MODE: No actual changes will be made")
    
    # Determine if metadata file exists
    metadata_exists = metadata_file.exists()
    
    # Step 1: Fetch podcast data if metadata doesn't exist or force is enabled
    if not metadata_exists or args.force:
        print_step("Fetching podcast data")
        success = fetch_podcast_data(
            podcast["channel_id"], 
            metadata_file, 
            args.from_date, 
            args.to_date,
            args.simulate
        )
        
        if not success:
            print("Failed to fetch podcast data")
            return 1
    else:
        print_step("Using existing metadata file")
        print(f"Metadata file exists at {metadata_file}")
        print("Use --force to refresh metadata")
    
    # Step 2: Download missing audio files or all if force download
    print_step("Processing audio files")
    success = download_audio_files(
        metadata_file, 
        audio_dir, 
        args.from_date, 
        args.to_date,
        args.download or args.force,
        args.simulate
    )
    
    if not success:
        print("Failed to process audio files")
        return 1
    
    # Step 3: Transcribe audio files
    print_step("Transcribing audio files")
    success = process_transcriptions(
        audio_dir,
        metadata_file,
        args.from_date,
        args.to_date,
        args.retranscribe or args.force,
        args.api_key,
        args.simulate,
        language=language,
        prompt=prompt
    )
    
    if not success:
        print("Failed to process transcriptions")
        return 1
    
    print("\n" + "=" * 80)
    print("PROCESSING COMPLETE".center(80))
    print("=" * 80)
    
    return 0

if __name__ == '__main__':
    sys.exit(main())