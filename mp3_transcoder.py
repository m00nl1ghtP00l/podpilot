#!/usr/bin/env python3
import os
import sys
from pathlib import Path
import shutil
from mutagen.mp3 import MP3
import subprocess
import tempfile
from datetime import datetime
import argparse

__all__ = ['transcode']

def _bytes_to_mb(bytes_value):
    """Convert bytes to megabytes"""
    return bytes_value / (1024 * 1024)

def _format_time(seconds):
    """Format seconds as HH:MM:SS"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def _parse_time(time_str):
    """Parse ffmpeg time string (HH:MM:SS.ms) to seconds"""
    time_parts = time_str.split(':')
    hours = float(time_parts[0])
    minutes = float(time_parts[1])
    seconds = float(time_parts[2])
    return hours * 3600 + minutes * 60 + seconds

def _calculate_target_bitrate(duration, target_size):
    """Calculate target bitrate in kbps based on duration and target size"""
    # Formula: bitrate (kbps) = (target_size_bytes * 8) / (duration_seconds * 1000)
    # Subtract 10% for metadata and container overhead
    target_size_bits = target_size * 8 * 0.9
    bitrate = int(target_size_bits / (duration * 1000))
    
    # Ensure bitrate is within reasonable bounds (32-320 kbps)
    return max(32, min(320, bitrate))

def transcode(input_path, target_size_mb=25, show_progress=True):
    """
    Transcode an MP3 file to a target size by adjusting the bitrate.
    
    Args:
        input_path (str or Path): Path to the input MP3 file
        target_size_mb (int): Target size in megabytes
        show_progress (bool): Whether to show progress information
    
    Returns:
        dict: Result information including success status, original and new sizes
    """
    result = {
        'success': False,
        'original_size_mb': 0,
        'new_size_mb': 0,
        'bitrate': 0,
        'error': None
    }
    
    try:
        input_path = Path(input_path)
        target_size = target_size_mb * 1000 * 1000  # Convert MB to bytes
        file_size = os.path.getsize(input_path)
        result['original_size_mb'] = _bytes_to_mb(file_size)
        
        if file_size <= target_size:
            result['success'] = True
            result['new_size_mb'] = result['original_size_mb']
            return result
        
        # Get audio duration
        audio = MP3(input_path)
        duration = audio.info.length
        
        target_bitrate = _calculate_target_bitrate(duration, target_size)
        result['bitrate'] = target_bitrate
        
        if show_progress:
            print(f"Original size: {_bytes_to_mb(file_size):.1f}MB")
            print(f"Target bitrate: {target_bitrate}kbps")
            print(f"Total duration: {_format_time(duration)}")
        
        # Create backup of original file
        orig_path = input_path.parent / f"{input_path.stem}.orig{input_path.suffix}"
        shutil.copy2(input_path, orig_path)
        
        # Create temporary file for output
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
            temp_path = temp_file.name
        
        # Improved FFmpeg command with better options for compatibility
        cmd = [
            'ffmpeg',
            '-i', str(input_path),
            '-c:a', 'libmp3lame',
            '-b:a', f'{target_bitrate}k',
            '-map_metadata', '0',  # Copy all metadata
            '-id3v2_version', '3',  # Use ID3v2.3 for better compatibility
            '-write_xing', '1',     # Write Xing header for better seeking
            '-y',                   # Overwrite output file if it exists
            temp_path
        ]
        
        if show_progress:
            print("Transcoding...")
        else:
            cmd.insert(1, '-loglevel')
            cmd.insert(2, 'error')
        
        process = subprocess.run(cmd, capture_output=not show_progress)
        
        if process.returncode != 0:
            error_msg = process.stderr.decode() if not show_progress else "See console output for details"
            result['error'] = f"FFmpeg error: {error_msg}"
            os.unlink(temp_path)
            return result
        
        # Verify the transcoded file
        try:
            # Check if file is valid MP3
            test_audio = MP3(temp_path)
            # Check if duration is similar (within 1 second)
            if abs(test_audio.info.length - duration) > 1:
                result['error'] = "Transcoded file has incorrect duration"
                os.unlink(temp_path)
                return result
        except Exception as e:
            result['error'] = f"Validation error: {str(e)}"
            os.unlink(temp_path)
            return result
        
        # Check the new file size
        new_size = os.path.getsize(temp_path)
        result['new_size_mb'] = _bytes_to_mb(new_size)
        
        # Replace the original file with the transcoded one
        shutil.move(temp_path, input_path)
        
        if show_progress:
            print(f"New size: {result['new_size_mb']:.1f}MB")
        
        result['success'] = True
        return result
        
    except Exception as e:
        result['error'] = str(e)
        # Try to clean up temp file if it exists
        if 'temp_path' in locals() and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except:
                pass
        return result

def main():
    """Command line interface for the transcoder"""
    parser = argparse.ArgumentParser(
        description='Transcode MP3 files to target size by adjusting bitrate. '
                    'Creates a backup (.orig) of the original file. '
                    'Only transcodes if file exceeds target size. '
                    'Bitrate range: 32-320 kbps.',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('input_file', help='Input MP3 file to transcode (REQUIRED)')
    parser.add_argument('-s', '--size', type=int, default=25, 
                        help='Target size in MB (default: 25). File is only transcoded if it exceeds this size.')
    parser.add_argument('-q', '--quiet', action='store_true', 
                        help='Quiet mode (no progress output)')
    
    # If run with no arguments, show usage and help.
    if len(sys.argv) == 1:
        parser.print_usage()
        print()
        parser.print_help()
        sys.exit(1)
    
    args = parser.parse_args()
    
    input_path = Path(args.input_file)
    
    if not input_path.exists():
        print(f"Error: File {input_path} does not exist")
        sys.exit(1)
    
    if input_path.suffix.lower() != '.mp3':
        print("Error: File must be an MP3")
        sys.exit(1)
    
    result = transcode(input_path, target_size_mb=args.size, show_progress=not args.quiet)
    if not result['success']:
        print(f"Error during transcoding: {result['error']}")
        sys.exit(1)

if __name__ == "__main__":
    main()
