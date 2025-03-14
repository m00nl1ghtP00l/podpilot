def clean_title(title):
    """Remove only emojis from title while preserving Japanese and other text characters."""
    if not title:
        return ""
    
    # Pattern to match emoji characters only (not CJK characters)
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F700-\U0001F77F"  # alchemical symbols
        "\U0001F780-\U0001F7FF"  # Geometric Shapes
        "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
        "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        "\U0001FA00-\U0001FA6F"  # Chess Symbols
        "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        "\U00002702-\U000027B0"  # Dingbats
        "\U000024C2-\U0001F251" 
        "]+", flags=re.UNICODE
    )
    
    # Remove emojis while preserving Japanese characters
    title = emoji_pattern.sub('', title)
    
    # Clean up any double spaces created by removing characters
    title = re.sub(r'\s+', ' ', title).strip()
    
    return title#!/usr/bin/env python3
"""
YouTube RSS Feed Parser
Extracts video information from a YouTube channel's RSS feed with improved readability
"""

import argparse
import datetime
import sys
import urllib.request
import xml.etree.ElementTree as ET
import json
import re
from html import unescape
import unicodedata


def validate_channel_id(channel_id):
    """Validate that the channel ID is in the correct format."""
    if not channel_id.startswith('UC') or len(channel_id) != 24:
        raise ValueError("Channel ID must start with 'UC' and be 24 characters long")
    return channel_id


def parse_date(date_str):
    """Parse a date string in ISO format (YYYY-MM-DD)."""
    if date_str:
        try:
            return datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            raise ValueError(f"Invalid date format: {date_str}. Use YYYY-MM-DD format.")
    return None


def fetch_rss_feed(channel_id):
    """Fetch the RSS feed for the given channel ID."""
    url = f"https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"
    try:
        with urllib.request.urlopen(url) as response:
            return response.read().decode('utf-8')
    except urllib.error.URLError as e:
        sys.stderr.write(f"Error fetching RSS feed: {e}\n")
        sys.exit(1)


def format_date(date_str):
    """Format ISO date strings to be more readable."""
    try:
        dt = datetime.datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S%z")
        return dt.strftime("%Y-%m-%d %H:%M")
    except (ValueError, TypeError):
        return date_str


def clean_description(description):
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
    if len(cleaned) > 300:
        cleaned = cleaned[:297] + "..."
    
    return {
        "text": cleaned,
        "links": links if links else []
    }


def parse_rss_feed(xml_content, from_date=None, to_date=None, include_author=False, include_description=False):
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
                
            video_id = entry.find('./yt:videoId', namespaces).text if entry.find('./yt:videoId', namespaces) is not None else None
            
            # Get title, decode HTML entities, and clean emojis
            title = entry.find('./atom:title', namespaces).text
            title = unescape(title) if title else ""
            title = clean_title(title)  # Remove emojis and special characters
            
            # Create video entry with basic information
            video = {
                'title': title,
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


def display_readable_format(channel_info):
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


def main():
    """Main function to parse arguments and process the RSS feed."""
    parser = argparse.ArgumentParser(description='Parse YouTube RSS feed for a channel')
    
    parser.add_argument('channel_id', type=validate_channel_id, help='YouTube channel ID (must start with UC, 24 characters total)')
    parser.add_argument('-o', '--output', dest='output', help='Output file path (optional, defaults to stdout)')
    parser.add_argument('--from-date', dest='from_date', type=parse_date, help='Start date in ISO format (YYYY-MM-DD)')
    parser.add_argument('--to-date', dest='to_date', type=parse_date, help='End date in ISO format (YYYY-MM-DD)')
    parser.add_argument('--format', dest='format', choices=['json', 'readable'], default='json',
                       help='Output format: json (default) or readable text')
    parser.add_argument('--clean-desc', dest='clean_descriptions', action='store_true',
                       help='Clean and truncate descriptions in JSON output')
    parser.add_argument('--sort', dest='sort', choices=['date-asc', 'date-desc'], 
                       help='Sort videos by publication date (ascending or descending)')
    parser.add_argument('--include-author', dest='include_author', action='store_true',
                       help='Include author information in the output')
    parser.add_argument('--include-description', dest='include_description', action='store_true',
                       help='Include video descriptions in the output')
    
    args = parser.parse_args()
    
    # Fetch and parse the RSS feed
    xml_content = fetch_rss_feed(args.channel_id)
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
        if args.include_description and not args.clean_descriptions:
            # Revert description to original format for raw JSON output
            for video in channel_info['videos']:
                if 'description' in video and video['description'] and isinstance(video['description'], dict):
                    video['description'] = video['description']['text']
        
        output = json.dumps(channel_info, indent=2, ensure_ascii=False)
    
    # Output results
    if args.output:
        try:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(output)
            print(f"Results written to {args.output}")
        except IOError as e:
            sys.stderr.write(f"Error writing to output file: {e}\n")
            sys.exit(1)
    else:
        print(output)


if __name__ == "__main__":
    main()
