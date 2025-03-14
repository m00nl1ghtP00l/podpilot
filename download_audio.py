import os
import argparse
from pathlib import Path
import json
from datetime import datetime
from datetime import timezone
import requests
from tqdm import tqdm
import re
import subprocess
from mp3_transcoder import transcode
import unicodedata

# Import functions from find_podcasts.py
from find_podcasts import clean_title

def load_json(json_file):
    """Load and validate JSON data"""
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, dict) or 'videos' not in data:
            raise ValueError("Invalid JSON structure: must contain 'videos' key")
        return data
    except Exception as e:
        print(f"Error reading JSON file: {e}")
        exit(1)

def parse_date_arg(date_str):
    """Parse date argument from command line"""
    if not date_str:
        return None
    try:
        return datetime.fromisoformat(date_str).replace(tzinfo=timezone.utc)
    except ValueError:
        try:
            date = datetime.strptime(date_str, '%Y-%m-%d')
            return date.replace(tzinfo=timezone.utc)
        except ValueError:
            raise argparse.ArgumentTypeError('Invalid date format. Use YYYY-MM-DD or ISO format')

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

def download_file(url, output_path, simulate=False):
    """Download file with progress bar and transcode if needed"""
    if simulate:
        print(f"Would download: {url}")
        print(f"          to: {output_path}")
        return True
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
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

def download_mode(json_data, output_dir, args):
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
        url = video.get('url', '')
        
        # Check if URL is missing
        if not url:
            # Try using 'link' instead (from find_podcasts output)
            url = video.get('link', '')
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
        
        # Use clean_filename from find_podcasts if available, otherwise generate one
        if 'clean_filename' in video:
            filename = f"{video['clean_filename']}.mp3"
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
            
        downloads.append((url, output_path, filename, date))
    
    # Sort by date
    downloads.sort(key=lambda x: x[3])
    process_existing.sort(key=lambda x: x[1])
    
    print(f"\nFound {len(downloads)} files to download:")
    for _, _, filename, _ in downloads:
        print(f"  {filename}")
    
    if process_existing:
        print(f"\nFound {len(process_existing)} existing files to check:")
        for file_path, _ in process_existing:
            print(f"  {os.path.basename(file_path)}")
    
    if not downloads and not process_existing:
        print("No files to process")
        return
    
    # Process existing files
    for file_path, _ in process_existing:
        process_existing_file(file_path, args.simulate)
    
    if args.simulate:
        return
        
    # Download new files
    success = 0
    failed = 0
    
    for url, output_path, filename, _ in downloads:
        if download_file(url, output_path, args.simulate):
            success += 1
        else:
            failed += 1
    
    if downloads:
        print(f"\nDownloads complete: {success} successful, {failed} failed")

def main():
    parser = argparse.ArgumentParser(description='Download audio files from JSON feed')
    parser.add_argument('json_file', help='JSON file containing audio URLs')
    parser.add_argument('-a','--audio_dir', help='Directory for audio files')
    parser.add_argument('--from-date', type=parse_date_arg, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--to-date', type=parse_date_arg, help='End date (YYYY-MM-DD)')
    parser.add_argument('-s', '--simulate', action='store_true',
                       help='Simulation mode: show what would be done without doing it')
    args = parser.parse_args()
    
    json_data = load_json(args.json_file)
    download_mode(json_data, args.audio_dir, args)

if __name__ == '__main__':
    main()
