"""
YouTube RSS Feed Parser
Extracts video information from a YouTube channel's RSS feed with improved readability
"""

import argparse
import datetime
from typing import Optional, Dict, List, Tuple
from pathlib import Path
import os
import sys
import urllib.request
import urllib.error
import xml.etree.ElementTree as ET
import json
import re
from html import unescape
import unicodedata

# Constants
CHANNEL_ID_LENGTH = 24
CHANNEL_ID_PREFIX = 'UC'
MAX_DESCRIPTION_LENGTH = 300
YOUTUBE_RSS_BASE_URL = "https://www.youtube.com/feeds/videos.xml"

# Japanese character pattern (compiled once for efficiency)
_JAPANESE_CHARS_PATTERN = re.compile(
    r'[\u3000-\u303f\u3040-\u309f\u30a0-\u30ff\u3400-\u4dbf\u4e00-\u9fff'
    r'\uf900-\ufaff\uff00-\uffef]|'
    r'【|】|「|」|『|』'
)

# Emoji pattern (compiled once for efficiency)
_EMOJI_PATTERN = re.compile("["
    u"\U0001F600-\U0001F64F"  # emoticons
    u"\U0001F300-\U0001F5FF"  # symbols & pictographs
    u"\U0001F680-\U0001F6FF"  # transport & map symbols
    u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
    u"\U00002702-\U000027B0"  # Dingbats
    u"\U000024C2-\U0001F251"  # Enclosed characters
    u"\U00002728"            # Sparkles ✨
    "]+", flags=re.UNICODE)


def _extract_japanese_segments(text: str) -> List[Tuple[int, int, str]]:
    """Extract Japanese character segments with their positions."""
    segments = []
    for match in re.finditer(f'({_JAPANESE_CHARS_PATTERN.pattern}+)', text):
        segments.append((match.start(), match.end(), match.group(0)))
    return segments


def _remove_emojis_preserving_japanese(text: str, jp_segments: List[Tuple[int, int, str]]) -> str:
    """Remove emojis from text while preserving Japanese character segments."""
    result = ""
    last_pos = 0
    
    for start, end, segment in jp_segments:
        # Process the part before this Japanese segment
        before_part = text[last_pos:start]
        before_part = _EMOJI_PATTERN.sub('', before_part)
        result += before_part
        
        # Add the Japanese segment as is
        result += segment
        last_pos = end
    
    # Process the remaining part after the last Japanese segment
    if last_pos < len(text):
        last_part = text[last_pos:]
        last_part = _EMOJI_PATTERN.sub('', last_part)
        result += last_part
    
    return result


def _convert_spaces_to_underscores(text: str, jp_segments: List[Tuple[int, int, str]]) -> str:
    """Convert spaces and pipes to underscores, preserving Japanese segments."""
    result = ""
    last_pos = 0
    
    for start, end, segment in jp_segments:
        # Process the part before this Japanese segment
        before_part = text[last_pos:start]
        before_part = re.sub(r'[\s|]+', '_', before_part.strip())
        result += before_part
        
        # Add the Japanese segment as is
        result += segment
        last_pos = end
    
    # Process the remaining part after the last Japanese segment
    if last_pos < len(text):
        last_part = text[last_pos:]
        last_part = re.sub(r'[\s|]+', '_', last_part.strip())
        result += last_part
    
    return result


def sanitize_filename(filename: str) -> str:
    """Sanitize a filename by replacing invalid characters with underscores.
    
    This function handles filenames that may contain invalid filesystem characters,
    such as those from old metadata files created before sanitization was implemented.
    """
    if not filename:
        return ""
    
    # Replace invalid filename characters (/, \, :, *, ?, ", <, >, |) with underscores
    invalid_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    
    # Clean up multiple underscores
    filename = re.sub(r'_+', '_', filename)
    
    return filename.rstrip('_')


def clean_title(title: str) -> str:
    """Convert spaces and pipes to underscores while preserving Japanese characters."""
    if not title:
        return ""
    
    # Extract Japanese segments
    jp_segments = _extract_japanese_segments(title)
    
    # Remove emojis while preserving Japanese
    result = _remove_emojis_preserving_japanese(title, jp_segments)
    
    # Convert full-width spaces to regular spaces
    result = result.replace('\u3000', ' ')
    
    # Replace invalid filename characters using the shared sanitization function
    result = sanitize_filename(result)
    
    # Extract Japanese segments again from emoji-removed result
    jp_segments_final = _extract_japanese_segments(result)
    
    # Convert spaces to underscores while preserving Japanese
    final_result = _convert_spaces_to_underscores(result, jp_segments_final)
    
    # Clean up multiple underscores and remaining spaces
    final_result = re.sub(r'_+', '_', final_result)
    final_result = re.sub(r'[\s\u3000]+', '_', final_result)
    
    return final_result.rstrip('_')

def validate_channel_id(channel_id: str) -> str:
    """Validate that the channel ID is in the correct format."""
    if not channel_id.startswith(CHANNEL_ID_PREFIX) or len(channel_id) != CHANNEL_ID_LENGTH:
        raise ValueError(f"Channel ID must start with '{CHANNEL_ID_PREFIX}' and be {CHANNEL_ID_LENGTH} characters long")
    return channel_id


def parse_date(date_str: Optional[str]) -> Optional[datetime.date]:
    """
    Parse a date string. Supports multiple formats:
    - YYYY-MM-DD (e.g., "2024-01-15")
    - Relative dates: "7d" (7 days ago), "1w" (1 week ago), "1m" (1 month ago)
    - Aliases: "today", "yesterday", "last-week", "last-month"
    """
    if not date_str:
        return None
    
    today = datetime.date.today()
    
    # Handle aliases
    aliases = {
        'today': today,
        'yesterday': today - datetime.timedelta(days=1),
        'last-week': today - datetime.timedelta(weeks=1),
        'last-month': today - datetime.timedelta(days=30),
    }
    if date_str.lower() in aliases:
        return aliases[date_str.lower()]
    
    # Handle relative dates (e.g., "7d", "2w", "3m")
    if len(date_str) > 1 and date_str[-1].lower() in ['d', 'w', 'm']:
        try:
            number = int(date_str[:-1])
            unit = date_str[-1].lower()
            
            if unit == 'd':
                return today - datetime.timedelta(days=number)
            elif unit == 'w':
                return today - datetime.timedelta(weeks=number)
            elif unit == 'm':
                # Approximate month as 30 days
                return today - datetime.timedelta(days=number * 30)
        except ValueError:
            pass  # Fall through to try YYYY-MM-DD format
    
    # Try standard YYYY-MM-DD format
    try:
        return datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        raise ValueError(
            f"Invalid date format: {date_str}. "
            f"Use YYYY-MM-DD, relative dates (e.g., '7d', '1w', '1m'), "
            f"or aliases (e.g., 'today', 'yesterday', 'last-week', 'last-month')."
        )


def fetch_rss_feed(channel_id: str) -> str:
    """Fetch the RSS feed for the given channel ID."""
    url = f"{YOUTUBE_RSS_BASE_URL}?channel_id={channel_id}"
    try:
        with urllib.request.urlopen(url) as response:
            return response.read().decode('utf-8')
    except urllib.error.URLError as e:
        sys.stderr.write(f"Error fetching RSS feed: {e}\n")
        sys.exit(1)


def format_date(date_str: Optional[str]) -> Optional[str]:
    """Format ISO date strings to be more readable."""
    if not date_str:
        return None
    try:
        dt = datetime.datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S%z")
        return dt.strftime("%Y-%m-%d %H:%M")
    except (ValueError, TypeError):
        return date_str


def clean_description(description: Optional[str]) -> Optional[Dict[str, any]]:
    """Clean up the description by removing excessive Unicode spacing and formatting the text."""
    if not description:
        return None
    
    # Remove invisible Unicode spacing characters
    cleaned = re.sub(r'\u2060+', ' ', description)
    
    # Convert multiple spaces/newlines to single spaces
    cleaned = re.sub(r'\s+', ' ', cleaned)
    
    # Extract social media links for better readability
    links = re.findall(r'https?://[^\s]+', cleaned)
    
    # Truncate long descriptions
    if len(cleaned) > MAX_DESCRIPTION_LENGTH:
        cleaned = cleaned[:MAX_DESCRIPTION_LENGTH - 3] + "..."
    
    return {
        "text": cleaned,
        "links": links if links else []
    }


def parse_rss_feed(
    xml_content: str,
    from_date: Optional[datetime.date] = None,
    to_date: Optional[datetime.date] = None,
    include_author: bool = False,
    include_description: bool = False
) -> Dict[str, any]:
    """Parse the RSS feed XML content and extract video information."""
    try:
        # Define namespaces used in YouTube's RSS feeds
        namespaces = {
            'atom': 'http://www.w3.org/2005/Atom',
            'media': 'http://search.yahoo.com/mrss/',
            'yt': 'http://www.youtube.com/xml/schemas/2015'
        }
        
        root = ET.fromstring(xml_content)
        
        # Get channel information
        channel_title = root.find('./atom:title', namespaces).text
        channel_link = root.find('./atom:link[@rel="alternate"]', namespaces).get('href')
        
        channel_info = {
            'title': channel_title,
            'link': channel_link,
            'videos': []
        }
        
        # Extract video entries
        for entry in root.findall('./atom:entry', namespaces):
            published_text = entry.find('./atom:published', namespaces).text
            published_date = datetime.datetime.strptime(published_text, "%Y-%m-%dT%H:%M:%S%z").date()
            
            # Apply date filters if specified
            if from_date and published_date < from_date:
                continue
            if to_date and published_date > to_date:
                continue
                
            # Get title and create clean filename
            raw_title = entry.find('./atom:title', namespaces).text
            clean_title_str = clean_title(raw_title)
            
            # Create filename with date prefix
            date_str = published_date.strftime('%Y-%m-%d')
            video_id = entry.find('./yt:videoId', namespaces).text if entry.find('./yt:videoId', namespaces) is not None else None
            if video_id:
                # Include video ID in filename for easier lookup (e.g. grep/ls by ID)
                clean_filename = f"{date_str}_{clean_title_str}_{video_id}"
            else:
                clean_filename = f"{date_str}_{clean_title_str}"
            
            # Create video entry with basic information
            video = {
                'title': raw_title,  # Original title for display
                'clean_filename': clean_filename,  # Clean filename for file operations
                'link': entry.find('./atom:link', namespaces).get('href'),
                'id': video_id,
                'published': format_date(published_text),
                'updated': format_date(entry.find('./atom:updated', namespaces).text)
            }
            
            # Add optional fields if requested
            if include_author:
                video['author'] = entry.find('./atom:author/atom:name', namespaces).text
                
            if include_description:
                video['description'] = clean_description(entry.find('./media:group/media:description', namespaces).text if entry.find('./media:group/media:description', namespaces) is not None else None)
            
            channel_info['videos'].append(video)
        
        return channel_info
        
    except ET.ParseError as e:
        sys.stderr.write(f"Error parsing XML: {e}\n")
        sys.exit(1)


def display_readable_format(channel_info: Dict[str, any]) -> str:
    """Create a human-readable text output instead of raw JSON."""
    output = []
    
    # Channel header
    output.append("=" * 80)
    output.append(f"CHANNEL: {channel_info['title']}")
    output.append(f"URL: {channel_info['link']}")
    output.append(f"Total Videos: {len(channel_info['videos'])}")
    output.append("=" * 80)
    output.append("")
    
    # Video information
    for i, video in enumerate(channel_info['videos'], 1):
        output.append(f"VIDEO #{i}:")
        output.append(f"Title: {video['title']}")
        output.append(f"URL: {video['link']}")
        output.append(f"Published: {video['published']}")
        
        if 'duration' in video:
            output.append(f"Duration: {video['duration']}")
        
        if 'author' in video and video['author']:
            output.append(f"Author: {video['author']}")
        
        if 'description' in video and video['description']:
            output.append("\nDescription:")
            output.append(video['description']['text'])
            
            if video['description']['links']:
                output.append("\nLinks found in description:")
                for link in video['description']['links']:
                    output.append(f"- {link}")
        
        output.append("-" * 80)
        output.append("")
    
    return "\n".join(output)


def load_config(config_file):
    """Load and validate config file"""
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # Validate required fields
        if "youtube_channels" not in config or not isinstance(config["youtube_channels"], list):
            raise ValueError("Config file missing 'youtube_channels' list or it's not a list")
        
        # Validate each podcast entry
        for i, podcast in enumerate(config["youtube_channels"]):
            if "channel_name_short" not in podcast:
                raise ValueError(f"Podcast at index {i} missing 'channel_name_short'")
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

def main():
    """Main function to parse arguments and process the RSS feed."""
    parser = argparse.ArgumentParser(
        description='Parse YouTube RSS feed for a channel',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        usage='%(prog)s [-h] (channel_id | --name NAME) [--config CONFIG] [options]',
        epilog='''Channel selection (REQUIRED - choose one):
  1. Direct: provide channel_id as positional argument
  2. Config-based: use --name (requires --config, which defaults to config/podcasts.json)'''
    )
    
    # Channel selection: either direct channel_id or config file + name
    parser.add_argument('channel_id', nargs='?', type=str,
                       help='YouTube channel ID (REQUIRED if not using --name)')
    parser.add_argument('--config', default='config/podcasts.json',
                       help='Path to podcasts.json config file (default: config/podcasts.json, REQUIRED with --name)')
    parser.add_argument('--name', help='Short name of the podcast from config file (REQUIRED if not using channel_id, use with --config)')
    
    parser.add_argument('-o', '--output', dest='output', help='Output file path (defaults to stdout)')
    parser.add_argument('-f', '--from-date', type=parse_date,
                        help='Start date (YYYY-MM-DD, relative like "7d"/"1w"/"1m", or aliases like "today"/"yesterday"/"last-week")')
    parser.add_argument('-t', '--to-date', type=parse_date,
                        help='End date (YYYY-MM-DD, relative like "7d"/"1w"/"1m", or aliases like "today"/"yesterday"/"last-week")')
    parser.add_argument('-F', '--format', choices=['json', 'readable'], default='json',
                       help='Output format: json (default) or readable text')
    parser.add_argument('-c', '--clean-desc', action='store_true',
                       help='Clean and truncate descriptions in JSON output')
    parser.add_argument('-s', '--sort', choices=['date-asc', 'date-desc'],
                       help='Sort videos by publication date')
    parser.add_argument('-a', '--include-author', action='store_true',
                       help='Include author information')
    parser.add_argument('-d', '--include-description', action='store_true',
                       help='Include video descriptions')
    parser.add_argument('-v', '--view', nargs='?', const=True,
                       help='View existing JSON file in readable format. Provide file path, or use with --name to view from data_root')
    
    # If script is run with no arguments at all, show both usage and full help.
    # This makes `python find_podcasts.py` self-explanatory instead of just erroring.
    if len(sys.argv) == 1:
        parser.print_usage()
        print()
        parser.print_help()
        sys.exit(1)
    
    args = parser.parse_args()
    
    # Handle --view option (view existing JSON file)
    if args.view:
        # If --name is provided, use it to find the file in data_root
        if args.name:
            try:
                config_data = load_config(args.config)
                podcast = find_podcast_by_name(config_data, args.name)
                data_root = Path(config_data.get("data_root", "."))
                channel_name = podcast["channel_name_short"]
                view_file = str(data_root / f"{channel_name}.json")
            except ValueError as e:
                sys.stderr.write(f"Error: {e}\n")
                sys.exit(1)
        elif isinstance(args.view, str):
            # Use provided file path
            view_file = args.view
        else:
            parser.error("--view requires either a file path or --name to find the file in data_root")
        
        # Load and display the JSON file
        try:
            abs_path = os.path.abspath(view_file)
            print(f"Viewing: {abs_path}\n")
            with open(view_file, 'r', encoding='utf-8') as f:
                channel_info = json.load(f)
            print(display_readable_format(channel_info))
        except FileNotFoundError:
            sys.stderr.write(f"Error: File not found: {view_file}\n")
            sys.stderr.write(f"Hint: Run 'find_podcasts.py --name <podcast_name>' first to create the metadata file.\n")
            sys.exit(1)
        except json.JSONDecodeError as e:
            sys.stderr.write(f"Error: Invalid JSON in file: {e}\n")
            sys.exit(1)
        except Exception as e:
            sys.stderr.write(f"Error reading file: {e}\n")
            sys.exit(1)
        return
    
    # Determine channel_id from args
    # Check if --name is provided first (indicates config mode)
    channel_id = None
    config_data = None  # Cache config for later use
    if args.name:
        # Using config + name mode
        if not args.config:
            parser.error("--config is required when using --name")
        try:
            config_data = load_config(args.config)
            podcast = find_podcast_by_name(config_data, args.name)
            channel_id = podcast["channel_id"]
        except ValueError as e:
            sys.stderr.write(f"Error: {e}\n")
            sys.exit(1)
    elif args.channel_id:
        # Using direct channel_id mode
        try:
            channel_id = validate_channel_id(args.channel_id)
        except argparse.ArgumentTypeError as e:
            parser.error(str(e))
    else:
        parser.error("Either channel_id or --config with --name must be provided")
    
    # Fetch and parse the RSS feed
    xml_content = fetch_rss_feed(channel_id)
    channel_info = parse_rss_feed(
        xml_content, 
        from_date=args.from_date, 
        to_date=args.to_date,
        include_author=args.include_author,
        include_description=args.include_description
    )
    
    # Sort videos if requested
    if args.sort:
        if args.sort == 'date-asc':
            channel_info['videos'].sort(key=lambda v: v['published'])
        elif args.sort == 'date-desc':
            channel_info['videos'].sort(key=lambda v: v['published'], reverse=True)
    
    # Determine output format
    if args.format == 'readable':
        output = display_readable_format(channel_info)
    else:  # json format
        if args.include_description and not args.clean_desc:
            # Revert description to original format for raw JSON output
            for video in channel_info['videos']:
                if 'description' in video and video['description'] and isinstance(video['description'], dict):
                    video['description'] = video['description']['text']
        
        output = json.dumps(channel_info, indent=2, ensure_ascii=False)
    
    # Output results
    output_file = args.output
    if not output_file and args.name and config_data:
        # Auto-determine output file when using --name (same pattern as other scripts)
        # Only auto-write if data_root exists in config (user has set it up)
        try:
            if "data_root" in config_data:
                podcast = find_podcast_by_name(config_data, args.name)
                data_root = Path(config_data.get("data_root", "."))
                channel_name = podcast["channel_name_short"]
                output_file = str(data_root / f"{channel_name}.json")
        except (ValueError, KeyError):
            pass  # Fall back to stdout if config lookup fails
    
    if output_file:
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(output)
            # Show absolute path for clarity
            abs_path = os.path.abspath(output_file)
            print(f"Results written to {abs_path}")
            print(f"\nTo view this file: find_podcasts.py --view --name {args.name if args.name else 'PODCAST_NAME'}")
        except IOError as e:
            sys.stderr.write(f"Error writing to output file: {e}\n")
            sys.exit(1)
    else:
        print(output)


if __name__ == "__main__":
    main()
