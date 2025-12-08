import os
import sys
import argparse
from pathlib import Path
import json
from datetime import datetime, timedelta
from datetime import timezone
import requests
from tqdm import tqdm
import re
import subprocess
from mp3_transcoder import transcode
import unicodedata
import yt_dlp  # Added yt-dlp for reliable audio extraction

# Import functions from channel_fetcher.py
from channel_fetcher import clean_title, load_config, find_podcast_by_name, sanitize_filename

# Import duration utilities from extract_duration.py
from extract_duration import format_duration, update_video_duration, extract_metadata_duration

def load_json(json_file):
    """Load and validate JSON data"""
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, dict) or 'videos' not in data:
            raise ValueError("Missing 'videos' key")
        return data
    except FileNotFoundError:
        # Provide helpful error message suggesting to run channel_fetcher.py first
        raise ValueError(f"File not found: {json_file}\n"
                        f"Hint: Run 'channel_fetcher.py --name <podcast_name>' first to create the metadata file.")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}")
    except ValueError:
        # Re-raise ValueError as-is (for missing 'videos' key)
        raise
    except Exception as e:
        raise ValueError(f"Error reading JSON file: {e}")

def parse_date_arg(date_str):
    """
    Parse date argument from command line. Supports multiple formats:
    - YYYY-MM-DD (e.g., "2024-01-15")
    - ISO format (e.g., "2024-01-15T10:30:00+00:00")
    - Relative dates: "7d" (7 days ago), "1w" (1 week ago), "1m" (1 month ago)
    - Aliases: "today", "yesterday", "last-week", "last-month"
    """
    if not date_str:
        return None
    
    today = datetime.now(timezone.utc).date()
    
    # Handle aliases
    aliases = {
        'today': today,
        'yesterday': today - timedelta(days=1),
        'last-week': today - timedelta(weeks=1),
        'last-month': today - timedelta(days=30),
    }
    if date_str.lower() in aliases:
        result_date = aliases[date_str.lower()]
        return datetime.combine(result_date, datetime.min.time()).replace(tzinfo=timezone.utc)
    
    # Handle relative dates (e.g., "7d", "2w", "3m")
    if len(date_str) > 1 and date_str[-1].lower() in ['d', 'w', 'm']:
        try:
            value = int(date_str[:-1])
            unit = date_str[-1].lower()
            
            if unit == 'd':
                result_date = today - timedelta(days=value)
            elif unit == 'w':
                result_date = today - timedelta(weeks=value)
            elif unit == 'm':
                result_date = today - timedelta(days=value * 30)  # Approximate month
            else:
                raise ValueError()
            
            return datetime.combine(result_date, datetime.min.time()).replace(tzinfo=timezone.utc)
        except (ValueError, IndexError):
            pass  # Fall through to try other formats
    
    # Try ISO format first
    try:
        return datetime.fromisoformat(date_str).replace(tzinfo=timezone.utc)
    except ValueError:
        try:
            # Try YYYY-MM-DD format
            date = datetime.strptime(date_str, '%Y-%m-%d')
            return date.replace(tzinfo=timezone.utc)
        except ValueError:
            raise argparse.ArgumentTypeError('Invalid date format. Use YYYY-MM-DD, ISO format, relative dates (7d/1w/1m), or aliases (today/yesterday/last-week/last-month)')

def is_date_in_range(date_str, from_date, to_date):
    """Check if date falls within specified range (inclusive)"""
    if not date_str:
        return False
    
    date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
    date = date.astimezone(timezone.utc)
    date_only = date.date()
    
    if from_date and date_only < from_date.date():
        return False
    if to_date and date_only > to_date.date():
        return False
    return True

def process_existing_file(file_path, simulate=False):
    """Check and transcode existing file if needed"""
    try:
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # Convert to MB
        print(f"Processing existing file: {os.path.basename(file_path)}, {file_size:.2f}MB")
        
        if file_size > 25:
            if not simulate:
                print("File is over 25MB limit, transcoding...")
                result = transcode(file_path, target_size_mb=25, show_progress=True)
                if result['success']:
                    print(f"Successfully transcoded: {result['original_size_mb']:.2f}MB -> {result['new_size_mb']:.2f}MB")
                    print(f"Final bitrate: {result['bitrate']}kbps")
                    return True
                else:
                    print(f"Transcoding failed: {result['error']}")
                    print("Keeping original file")
            else:
                print(f"Would transcode: {file_path}")
        else:
            if not simulate:
                print("File is under 25MB limit, no transcoding needed")
        return True
    except Exception as e:
        print(f"Error processing existing file: {e}")
        return False

def transcode_file(input_path, simulate=False):
    """
    This function is maintained for backward compatibility.
    It now uses the mp3_transcoder module internally.
    """
    if simulate:
        print(f"Would transcode: {input_path}")
        return None
    
    try:
        result = transcode(input_path, target_size_mb=25, show_progress=True)
        if result['success']:
            print(f"Successfully transcoded: {result['original_size_mb']:.2f}MB -> {result['new_size_mb']:.2f}MB")
            return input_path
        else:
            print(f"Transcoding failed: {result['error']}")
            return None
    except Exception as e:
        print(f"Error during transcoding: {e}")
        return None

def download_file(url, output_path, simulate=False, min_duration=None, duration=None):
    """Download file with progress bar and transcode if needed
    
    Args:
        url: URL to download
        output_path: Where to save the file
        simulate: If True, don't actually download
        min_duration: Minimum duration in seconds to download
        duration: Pre-calculated duration in seconds (for YouTube URLs)
    """
    if simulate:
        print(f"Would download: {url}")
        print(f"          to: {output_path}")
        return True
    
    try:
        # Check if the URL is a YouTube video URL
        if 'youtube.com' in url or 'youtu.be' in url:
            print(f"Detected YouTube URL: {url}")
            return download_audio_from_youtube(url, output_path, min_duration, duration)
        
        # For direct file downloads
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        content_type = response.headers.get('content-type', '')
        if not content_type.startswith('audio/'):
            print(f"Warning: Content-Type is {content_type}, not audio. This may not be an audio file.")
        
        total_size = int(response.headers.get('content-length', 0))
        
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=os.path.basename(output_path)) as pbar:
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        # Check file size after download
        file_size = os.path.getsize(output_path) / (1024 * 1024)  # Convert to MB
        print(f"\nFile size: {file_size:.2f}MB")
        
        # Verify it's a valid audio file
        if not is_valid_audio_file(output_path):
            print(f"Warning: The downloaded file does not appear to be a valid audio file.")
            
            # If the file is too small, it might be an error page
            if file_size < 0.1:  # Less than 100KB
                print(f"File is very small ({file_size:.2f}MB), likely not a valid audio file.")
                os.remove(output_path)
                return False
        
        # Use mp3_transcoder if file is too large
        if file_size > 25:
            print("File is over 25MB limit, transcoding...")
            result = transcode(output_path, target_size_mb=25, show_progress=True)
            if result['success']:
                print(f"Successfully transcoded: {result['original_size_mb']:.2f}MB -> {result['new_size_mb']:.2f}MB")
                print(f"Final bitrate: {result['bitrate']}kbps")
            else:
                print(f"Transcoding failed: {result['error']}")
                print("Keeping original file")
        else:
            print("File is under 25MB limit, no transcoding needed")
            
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        if os.path.exists(output_path):
            os.remove(output_path)
        return False

def get_youtube_duration(url):
    """Get video duration from YouTube without downloading. Returns duration in seconds or None."""
    try:
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'skip_download': True,  # Don't download, just get info
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            if info and 'duration' in info:
                return info['duration']
    except Exception:
        pass
    return None


def download_audio_from_youtube(url, output_path, min_duration=None, duration=None):
    """Download audio from YouTube using yt-dlp
    
    Args:
        url: YouTube URL
        output_path: Where to save the file
        min_duration: Minimum duration in seconds to download
        duration: Pre-calculated duration in seconds (optional, will be fetched if not provided)
    """
    # Check duration before downloading if min_duration is specified
    if min_duration:
        if duration is None:
            print(f"Checking video duration...")
            duration = get_youtube_duration(url)
        
        if duration is not None:
            duration_min = duration // 60
            duration_sec = duration % 60
            print(f"Video duration: {duration_min}:{duration_sec:02d}")
            if duration < min_duration:
                print(f"Skipping: Video is shorter than {min_duration // 60}:{min_duration % 60:02d}")
                return False
        else:
            print("Warning: Could not get video duration, proceeding with download...")
    
    print(f"Extracting audio from YouTube video: {url}")
    
    # Use a temporary directory and simple filename to prevent yt-dlp from creating subdirectories
    # yt-dlp can misinterpret paths and create unwanted directory structures
    import tempfile
    import uuid
    import shutil
    
    output_dir = os.path.dirname(output_path)
    
    # Create a temporary file in the system temp directory to avoid any path interpretation issues
    temp_dir = tempfile.gettempdir()
    temp_filename = f"ytdlp_{uuid.uuid4().hex[:8]}.%(ext)s"
    temp_outtmpl = os.path.join(temp_dir, temp_filename)
    
    # Define options for yt-dlp
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': temp_outtmpl,  # Use temp directory with simple filename
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'progress_hooks': [lambda d: print(f"Download progress: {d['status']}", end='\r') 
                          if d['status'] != 'finished' else print("\nDownload complete, post-processing...")],
        'quiet': True,  # Suppress most output
        'no_warnings': True,  # Suppress warnings
        'extract_flat': False,  # Don't extract playlist info
        'ignoreerrors': True,  # Continue on download errors
    }
    
    temp_file_path = None
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                ydl.download([url])
            except Exception as e:
                print(f"\nError during YouTube download: {str(e)}")
                return False
        
        # Find the downloaded file (yt-dlp creates it with .mp3 extension)
        # Extract the UUID from the temp filename to search for the actual file
        uuid_part = temp_filename.split('_')[1].split('.')[0]  # Get UUID part
        temp_pattern = f"ytdlp_{uuid_part}.mp3"
        temp_file_path = os.path.join(temp_dir, temp_pattern)
        
        # If the expected temp file doesn't exist, search for it
        if not os.path.exists(temp_file_path):
            # Look for any temp file that matches our pattern
            try:
                temp_files = [f for f in os.listdir(temp_dir) 
                             if f.startswith(f"ytdlp_{uuid_part}") and f.endswith('.mp3')]
                if temp_files:
                    temp_file_path = os.path.join(temp_dir, temp_files[0])
                else:
                    # Last resort: find most recently created .mp3 in temp dir
                    print(f"Warning: Expected temp file not found, searching temp directory...")
                    mp3_files = [f for f in os.listdir(temp_dir) if f.endswith('.mp3')]
                    if mp3_files:
                        latest_file = max(mp3_files, key=lambda f: os.path.getctime(os.path.join(temp_dir, f)))
                        temp_file_path = os.path.join(temp_dir, latest_file)
                        print(f"Found: {temp_file_path}")
                    else:
                        print(f"Error: Could not find downloaded file in temp directory")
                        return False
            except OSError as e:
                print(f"Error accessing temp directory: {e}")
                return False
        
        # Move the temp file to the final location
        if os.path.exists(temp_file_path):
            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)
            
            # Remove output file if it exists
            if os.path.exists(output_path):
                os.remove(output_path)
            
            # Move temp file to final location
            shutil.move(temp_file_path, output_path)
            print(f"File saved to: {output_path}")
        else:
            print(f"Error: Temp file {temp_file_path} does not exist")
            return False
            
        # Process file size and potentially transcode
        file_size = os.path.getsize(output_path) / (1024 * 1024)  # Convert to MB
        print(f"File size: {file_size:.2f}MB")
        
        if file_size > 25:
            print("File is over 25MB limit, transcoding...")
            result = transcode(output_path, target_size_mb=25, show_progress=True)
            if result['success']:
                print(f"Successfully transcoded: {result['original_size_mb']:.2f}MB -> {result['new_size_mb']:.2f}MB")
                print(f"Final bitrate: {result['bitrate']}kbps")
            else:
                print(f"Transcoding failed: {result['error']}")
                print("Keeping original file")
        
        return True
    except Exception as e:
        # Clean up temp file if it exists
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except:
                pass
        print(f"Error downloading audio from YouTube: {e}")
        if os.path.exists(output_path):
            os.remove(output_path)
        return False
    except Exception as e:
        print(f"Error downloading audio from YouTube: {e}")
        if os.path.exists(output_path):
            os.remove(output_path)
        return False

def is_valid_audio_file(file_path):
    """Check if the file is a valid audio file using ffprobe"""
    try:
        # Use ffprobe to get file information
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'a', '-show_entries', 
             'stream=codec_name', '-of', 'default=noprint_wrappers=1:nokey=1', file_path],
            capture_output=True, text=True
        )
        
        # If ffprobe returns an audio codec, the file should be valid
        return bool(result.stdout.strip())
    except Exception as e:
        print(f"Error checking audio file validity: {e}")
        return True  # Assume it's valid if we can't check

def download_mode(json_data, output_dir, args, json_file=None):
    """Download new files from JSON data"""
    os.makedirs(output_dir, exist_ok=True)
    existing_files = set(os.listdir(output_dir))
    
    downloads = []
    process_existing = []
    
    for video in json_data['videos']:
        # Use published date if available, otherwise fall back to date
        date_str = video.get('published', video.get('date'))
        if not date_str:
            print(f"Warning: No date found for video {video.get('title', 'Unknown')}")
            continue
            
        if not is_date_in_range(date_str, args.from_date, args.to_date):
            continue
        
        # Extract necessary information
        title = video.get('title', '')
        
        # Check for URL in various potential fields
        url = None
        for url_field in ['url', 'link', 'id']:
            if url_field in video and video[url_field]:
                url_value = video[url_field]
                
                # Handle YouTube video IDs
                if url_field == 'id' and len(url_value) == 11 and not url_value.startswith('http'):
                    url = f"https://www.youtube.com/watch?v={url_value}"
                else:
                    url = url_value
                
                break
        
        if not url:
            print(f"Warning: No URL found for video {title}")
            continue
        
        # Parse the date
        try:
            if 'T' in date_str:  # ISO format
                date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            else:  # Simple date format
                date = datetime.strptime(date_str, '%Y-%m-%d %H:%M')
        except ValueError:
            try:
                # Try alternate format
                date = datetime.strptime(date_str, '%Y-%m-%d')
            except ValueError:
                print(f"Warning: Could not parse date {date_str} for video {title}")
                continue
        
        # Use clean_filename from channel_fetcher if available, otherwise generate one
        if 'clean_filename' in video:
            # Sanitize the filename to remove any invalid characters (e.g., /, \, :, etc.)
            # This handles cases where old metadata files might have invalid characters
            base_filename = sanitize_filename(video['clean_filename'])
            filename = f"{base_filename}.mp3"
        else:
            # Generate cleaned title using the imported clean_title function
            clean_title_str = clean_title(title)
            date_prefix = date.strftime('%Y-%m-%d')
            filename = f"{date_prefix}_{clean_title_str}.mp3"
        
        output_path = os.path.join(output_dir, filename)

        print(f"\nProcessing video: {title}")
        print(f"Date: {date.strftime('%Y-%m-%d')}")
        print(f"URL: {url}")
        print(f"Filename: {filename}")

        if filename in existing_files:
            process_existing.append((output_path, date))
            continue
            
        downloads.append((url, output_path, filename, date, video))  # Include video dict for duration updates
    
    # Sort by date
    downloads.sort(key=lambda x: x[3])
    process_existing.sort(key=lambda x: x[1])
    
    print(f"\nFound {len(downloads)} files to download:")
    for _, _, filename, _, _ in downloads:
        print(f"  {filename}")
    
    if process_existing:
        print(f"\nFound {len(process_existing)} existing files to check:")
        for file_path, _ in process_existing:
            print(f"  {os.path.basename(file_path)}")
    
    if not downloads and not process_existing:
        print("No files to process")
        return
    
    # Process existing files
    existing_videos_to_update = []
    for file_path, date in process_existing:
        process_existing_file(file_path, args.simulate)
        # Find the corresponding video entry for duration update (only if missing duration)
        filename = os.path.basename(file_path)
        filename_stem = os.path.splitext(filename)[0]  # Remove .mp3 extension
        for video in json_data['videos']:
            # Only check duration if it's missing from JSON
            if 'duration_seconds' in video and 'duration' in video:
                continue  # Skip files that already have duration
            
            # Match by clean_filename (use sanitize_filename for consistent matching)
            clean_fn = video.get('clean_filename', '')
            if clean_fn:
                # Sanitize both to handle any filename differences
                if sanitize_filename(filename_stem) == sanitize_filename(clean_fn):
                    existing_videos_to_update.append((file_path, video))
                    break
    
    if args.simulate:
        return
        
    # Download new files
    success = 0
    failed = 0
    skipped = 0
    duration_updated = False
    
    for url, output_path, filename, _, video in downloads:
        # Check duration before downloading (for YouTube URLs) and save to JSON
        duration = None
        if 'youtube.com' in url or 'youtu.be' in url:
            duration = get_youtube_duration(url)
            if duration is not None:
                # Save duration to video entry in JSON using centralized function
                update_video_duration(video, duration)
                duration_updated = True
        
        result = download_file(url, output_path, args.simulate, args.min_duration, duration)
        if result is False:
            # Check if it was skipped due to duration (YouTube URLs only)
            if args.min_duration and ('youtube.com' in url or 'youtu.be' in url):
                skipped += 1  # Skipped due to duration - no download occurred
                print(f"Skipped {filename} (duration < {args.min_duration // 60}:{args.min_duration % 60:02d})")
            else:
                failed += 1
        elif result is True:
            success += 1
        else:
            failed += 1
    
    if downloads:
        print(f"\nDownloads complete: {success} successful, {failed} failed", end="")
        if skipped > 0:
            print(f", {skipped} skipped (too short)")
        else:
            print()
    
    # Update durations for successfully downloaded files and existing files that don't have duration yet
    # (e.g., non-YouTube files need duration extracted from audio files)
    if json_file and (success > 0 or len(existing_videos_to_update) > 0):
        try:
            from pathlib import Path
            audio_dir_path = Path(output_dir)
            min_dur = args.min_duration if args.min_duration else 0
            
            # Track which videos to remove (too short)
            videos_to_remove = []
            duration_updates = 0
            
            # Process videos we just downloaded
            for url, output_path, filename, _, video in downloads:
                # Skip if already has duration (YouTube videos we set earlier)
                if 'duration_seconds' in video and 'duration' in video:
                    # Check if too short and should be removed
                    if min_dur > 0 and video.get('duration_seconds', 0) < min_dur:
                        videos_to_remove.append(video)
                    continue
                
                # For non-YouTube files or YouTube files where duration check failed,
                # get duration from the downloaded audio file
                if os.path.exists(output_path):
                    from extract_duration import update_video_duration_from_file
                    updated, should_keep = update_video_duration_from_file(video, audio_dir_path, min_dur)
                    if updated:
                        duration_updates += 1
                    if not should_keep:
                        videos_to_remove.append(video)
            
            # Process existing files that don't have duration yet
            for file_path, video in existing_videos_to_update:
                # Skip if already has duration
                if 'duration_seconds' in video and 'duration' in video:
                    # Check if too short and should be removed
                    if min_dur > 0 and video.get('duration_seconds', 0) < min_dur:
                        videos_to_remove.append(video)
                    continue
                
                # Get duration from the existing audio file
                if os.path.exists(file_path):
                    from extract_duration import update_video_duration_from_file
                    updated, should_keep = update_video_duration_from_file(video, audio_dir_path, min_dur)
                    if updated:
                        duration_updates += 1
                    if not should_keep:
                        videos_to_remove.append(video)
            
            # Remove videos that are too short
            if videos_to_remove:
                json_data['videos'] = [v for v in json_data['videos'] if v not in videos_to_remove]
                print(f"Removed {len(videos_to_remove)} videos shorter than {min_dur // 60}:{min_dur % 60:02d} from metadata")
            
            # Save updated metadata
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2)
            
            if duration_updates > 0 or duration_updated:
                print(f"✓ Metadata updated with duration information: {json_file}")
        except Exception as e:
            print(f"Warning: Could not update duration information: {e}")
            # Fallback: just save what we have (YouTube durations already set)
            if duration_updated:
                try:
                    with open(json_file, 'w', encoding='utf-8') as f:
                        json.dump(json_data, f, ensure_ascii=False, indent=2)
                    print(f"✓ Saved metadata (some duration updates may be missing): {json_file}")
                except Exception as e2:
                    print(f"Error: Could not save metadata: {e2}")
    elif json_file and duration_updated:
        # Only YouTube durations were set, no files downloaded, just save
        try:
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2)
            print(f"✓ Metadata updated with duration information: {json_file}")
        except Exception as e:
            print(f"Error: Could not save metadata: {e}")

def main():
    parser = argparse.ArgumentParser(
        description='Download audio from YouTube videos listed in JSON feed',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        usage='%(prog)s [-h] (json_file | --name NAME) [--config CONFIG] [options]',
        epilog='''Input selection (REQUIRED - choose one):
  1. Direct: provide json_file as positional argument
  2. Config-based: use --name (requires --config, which defaults to config/podcasts.json)'''
    )
    
    # Input selection: either JSON file or config + name
    parser.add_argument('json_file', nargs='?', 
                       help='JSON file containing video URLs (REQUIRED if not using --name)')
    parser.add_argument('--config', default='config/podcasts.json',
                       help='Path to podcasts.json config file (default: config/podcasts.json, REQUIRED with --name)')
    parser.add_argument('--name', 
                       help='Short name of the podcast from config file (REQUIRED if not using json_file, use with --config)')
    
    parser.add_argument('-a', '--audio-dir', default='./downloads',
                        help='Directory for audio files (default: ./downloads)')
    parser.add_argument('-f', '--from-date', type=parse_date_arg,
                        help='Start date filter (YYYY-MM-DD, ISO format, or relative like 7d/1w/1m)')
    parser.add_argument('-t', '--to-date', type=parse_date_arg,
                        help='End date filter (YYYY-MM-DD, ISO format, or relative like 7d/1w/1m)')
    parser.add_argument('-s', '--simulate', action='store_true',
                        help='Dry run mode: show what would be done without doing it')
    parser.add_argument('--id',
                        help='Download audio from specific video by ID (overrides date filters)')
    parser.add_argument('--min-duration', type=int,
                        help='Minimum duration in seconds to download (overrides config). Videos shorter than this will be skipped.')
    
    # If run with no arguments, show usage and full help instead of just an error.
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
    
    # Determine JSON file from args
    json_file = None
    if args.json_file:
        # Use provided JSON file
        json_file = args.json_file
    elif args.name:
        # Using config + name
        try:
            podcast = find_podcast_by_name(config_data, args.name)
            # Use data_root from config to find metadata file (same pattern as transcribe.py)
            data_root = Path(config_data.get("data_root", "."))
            channel_name = podcast["channel_name_short"]
            json_file = str(data_root / f"{channel_name}.json")
            
            # If audio_dir not specified, use data_root/channel_name (like transcribe.py)
            if args.audio_dir == './downloads':  # Only if using default
                args.audio_dir = str(data_root / channel_name)
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)
    else:
        # Neither json_file nor --name provided
        parser.error("Either json_file or --config with --name must be provided")
    
    # Get min_duration from config if not specified on command line
    if args.min_duration is None and config_data:
        args.min_duration = config_data.get('min_duration', 300)  # Default to 5 minutes
    elif args.min_duration is None:
        args.min_duration = 300  # Default if no config
    
    # Validate conflicting options
    if args.id and (args.from_date or args.to_date):
        print("Warning: --id specified, date filters (--from-date, --to-date) will be ignored")
    
    try:
        json_data = load_json(json_file)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # If ID is specified, only download that specific video
    if args.id:
        for video in json_data['videos']:
            if video.get('id') == args.id:
                # Create a single-item list with just this video
                json_data['videos'] = [video]
                break
        else:
            print(f"Error: No video found with ID {args.id}")
            return
    
    download_mode(json_data, args.audio_dir, args, json_file)

if __name__ == '__main__':
    main()
