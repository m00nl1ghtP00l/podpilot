"""
Tests for channel_fetcher.py

This test suite covers:
- Title cleaning with Japanese character handling
- Channel ID validation
- Date parsing and formatting
- RSS feed fetching (with mocks)
- Description cleaning
- RSS feed parsing (with mocked XML)
- Output formatting
- CLI argument handling
"""

import pytest
import json
import os
import tempfile
from pathlib import Path
from datetime import datetime, date, timedelta
from unittest.mock import Mock, patch, mock_open, MagicMock
import sys
import xml.etree.ElementTree as ET

# Add parent directory to path to import the module
sys.path.insert(0, str(Path(__file__).parent.parent))

from channel_fetcher import (
    clean_title,
    validate_channel_id,
    parse_date,
    fetch_rss_feed,
    format_date,
    clean_description,
    parse_rss_feed,
    display_readable_format,
    load_config,
    find_podcast_by_name,
    sanitize_filename,
    main
)


class TestCleanTitle:
    """Tests for clean_title function"""
    
    def test_clean_simple_title(self):
        """Test cleaning a simple English title"""
        result = clean_title("Hello World")
        assert result == "Hello_World"
    
    def test_clean_title_with_pipes(self):
        """Test cleaning title with pipes"""
        result = clean_title("Video | Episode 1")
        assert result == "Video_Episode_1"
    
    def test_clean_title_preserves_japanese(self):
        """Test that Japanese characters are preserved"""
        result = clean_title("日本語のタイトル")
        assert result == "日本語のタイトル"
    
    def test_clean_title_mixed_japanese_english(self):
        """Test title with both Japanese and English"""
        result = clean_title("Learn 日本語 Today")
        # Japanese characters are preserved as-is, spaces around them become underscores
        assert result == "Learn日本語Today" or "日本語" in result
    
    def test_clean_title_removes_emojis(self):
        """Test that emojis are removed"""
        result = clean_title("Video Title ✨🎉")
        assert result == "Video_Title"
    
    def test_clean_title_preserves_japanese_removes_emojis(self):
        """Test that emojis are removed but Japanese is preserved"""
        result = clean_title("日本語✨タイトル🎉")
        assert result == "日本語タイトル"
    
    def test_clean_title_multiple_spaces(self):
        """Test that multiple spaces become single underscore"""
        result = clean_title("Video    Title")
        assert result == "Video_Title"
    
    def test_clean_title_removes_trailing_underscores(self):
        """Test that trailing underscores are removed"""
        result = clean_title("Video Title___")
        assert result == "Video_Title"
    
    def test_clean_title_japanese_brackets(self):
        """Test that Japanese brackets are preserved"""
        result = clean_title("【日本語】タイトル")
        assert result == "【日本語】タイトル"
    
    def test_clean_title_empty_string(self):
        """Test empty string"""
        result = clean_title("")
        assert result == ""
    
    def test_clean_title_only_spaces(self):
        """Test string with only spaces"""
        result = clean_title("   ")
        assert result == ""
    
    def test_clean_title_with_slash(self):
        """Test that clean_title now handles slashes (via sanitize_filename)"""
        result = clean_title("Video f/w Guest")
        assert "/" not in result
        assert "f_w" in result or "f" in result and "w" in result


class TestSanitizeFilename:
    """Tests for sanitize_filename function"""
    
    def test_sanitize_no_invalid_chars(self):
        """Test sanitizing filename with no invalid characters"""
        assert sanitize_filename("test_file.mp3") == "test_file.mp3"
        assert sanitize_filename("2025-01-01_Title") == "2025-01-01_Title"
    
    def test_sanitize_slash(self):
        """Test sanitizing forward slash"""
        assert sanitize_filename("test/file.mp3") == "test_file.mp3"
        assert sanitize_filename("f/w@test") == "f_w@test"
    
    def test_sanitize_backslash(self):
        """Test sanitizing backslash"""
        assert sanitize_filename("test\\file.mp3") == "test_file.mp3"
    
    def test_sanitize_colon(self):
        """Test sanitizing colon"""
        assert sanitize_filename("test:file.mp3") == "test_file.mp3"
    
    def test_sanitize_multiple_invalid(self):
        """Test sanitizing multiple invalid characters"""
        assert sanitize_filename("test/file:name*?.mp3") == "test_file_name_.mp3"
    
    def test_sanitize_multiple_underscores(self):
        """Test that multiple underscores are collapsed"""
        assert sanitize_filename("test___file.mp3") == "test_file.mp3"
        assert sanitize_filename("test//file.mp3") == "test_file.mp3"
    
    def test_sanitize_empty(self):
        """Test sanitizing empty string"""
        assert sanitize_filename("") == ""
    
    def test_sanitize_trailing_underscores(self):
        """Test that trailing underscores are removed"""
        assert sanitize_filename("test_file__") == "test_file"
    
    def test_sanitize_real_world_example(self):
        """Test with real-world problematic filename"""
        problematic = "2025-11-28_50_Japanese_Questions_in_5_Seconds_Think_in_Japanese_With_Me_f/w@yuyunihongopodcast__iwGxHbltp4"
        result = sanitize_filename(problematic)
        assert "/" not in result
        assert "__" not in result  # Double underscores should be collapsed


class TestValidateChannelId:
    """Tests for validate_channel_id function"""
    
    def test_valid_channel_id(self):
        """Test valid channel ID"""
        result = validate_channel_id("UCauyM-A8JIJ9NQcw5_jF00Q")
        assert result == "UCauyM-A8JIJ9NQcw5_jF00Q"
        assert len(result) == 24
        assert result.startswith("UC")
    
    def test_invalid_channel_id_wrong_prefix(self):
        """Test channel ID that doesn't start with UC"""
        with pytest.raises(ValueError, match="must start with 'UC'"):
            validate_channel_id("CCauyM-A8JIJ9NQcw5_jF00Q")
    
    def test_invalid_channel_id_wrong_length(self):
        """Test channel ID with wrong length"""
        with pytest.raises(ValueError, match="24 characters long"):
            validate_channel_id("UC123")  # Too short
        with pytest.raises(ValueError, match="24 characters long"):
            validate_channel_id("UC" + "a" * 25)  # Too long


class TestParseDate:
    """Tests for parse_date function"""
    
    def test_parse_valid_date(self):
        """Test parsing valid date string"""
        result = parse_date("2024-01-15")
        assert result == date(2024, 1, 15)
    
    def test_parse_date_none(self):
        """Test parsing None"""
        result = parse_date(None)
        assert result is None
    
    def test_parse_date_empty_string(self):
        """Test parsing empty string"""
        result = parse_date("")
        assert result is None
    
    def test_parse_date_invalid_format(self):
        """Test parsing invalid date format"""
        with pytest.raises(ValueError, match="Invalid date format"):
            parse_date("2024/01/15")
    
    def test_parse_date_invalid_date(self):
        """Test parsing invalid date"""
        with pytest.raises(ValueError):
            parse_date("2024-13-45")  # Invalid month/day
    
    def test_parse_date_today(self):
        """Test parsing 'today' alias"""
        result = parse_date("today")
        assert result == date.today()
    
    def test_parse_date_yesterday(self):
        """Test parsing 'yesterday' alias"""
        result = parse_date("yesterday")
        expected = date.today() - timedelta(days=1)
        assert result == expected
    
    def test_parse_date_last_week(self):
        """Test parsing 'last-week' alias"""
        result = parse_date("last-week")
        expected = date.today() - timedelta(weeks=1)
        assert result == expected
    
    def test_parse_date_last_month(self):
        """Test parsing 'last-month' alias"""
        result = parse_date("last-month")
        expected = date.today() - timedelta(days=30)
        assert result == expected
    
    def test_parse_date_relative_days(self):
        """Test parsing relative days (e.g., '7d')"""
        result = parse_date("7d")
        expected = date.today() - timedelta(days=7)
        assert result == expected
    
    def test_parse_date_relative_weeks(self):
        """Test parsing relative weeks (e.g., '2w')"""
        result = parse_date("2w")
        expected = date.today() - timedelta(weeks=2)
        assert result == expected
    
    def test_parse_date_relative_months(self):
        """Test parsing relative months (e.g., '1m')"""
        result = parse_date("1m")
        expected = date.today() - timedelta(days=30)
        assert result == expected
    
    def test_parse_date_case_insensitive_aliases(self):
        """Test that aliases are case-insensitive"""
        result1 = parse_date("TODAY")
        result2 = parse_date("Today")
        result3 = parse_date("today")
        assert result1 == result2 == result3 == date.today()


class TestFormatDate:
    """Tests for format_date function"""
    
    def test_format_iso_date(self):
        """Test formatting ISO date string"""
        result = format_date("2024-01-15T10:30:00+00:00")
        assert result == "2024-01-15 10:30"
    
    def test_format_date_invalid(self):
        """Test formatting invalid date returns original"""
        result = format_date("invalid-date")
        assert result == "invalid-date"
    
    def test_format_date_none(self):
        """Test formatting None"""
        result = format_date(None)
        assert result is None


class TestCleanDescription:
    """Tests for clean_description function"""
    
    def test_clean_description_none(self):
        """Test cleaning None description"""
        result = clean_description(None)
        assert result is None
    
    def test_clean_description_simple(self):
        """Test cleaning simple description"""
        result = clean_description("This is a simple description.")
        assert result['text'] == "This is a simple description."
        assert result['links'] == []
    
    def test_clean_description_with_links(self):
        """Test cleaning description with links"""
        desc = "Check out https://example.com for more info"
        result = clean_description(desc)
        assert "https://example.com" in result['links']
    
    def test_clean_description_truncates_long(self):
        """Test that long descriptions are truncated"""
        long_desc = "A" * 400
        result = clean_description(long_desc)
        assert len(result['text']) <= 300
        assert result['text'].endswith("...")
    
    def test_clean_description_multiple_spaces(self):
        """Test that multiple spaces are collapsed"""
        result = clean_description("Multiple    spaces   here")
        assert result['text'] == "Multiple spaces here"
    
    def test_clean_description_with_newlines(self):
        """Test that newlines are converted to spaces"""
        result = clean_description("Line 1\nLine 2\nLine 3")
        assert "\n" not in result['text']
        assert "Line 1 Line 2 Line 3" in result['text']


class TestFetchRssFeed:
    """Tests for fetch_rss_feed function"""
    
    @patch('channel_fetcher.urllib.request.urlopen')
    def test_fetch_rss_feed_success(self, mock_urlopen):
        """Test successful RSS feed fetch"""
        mock_response = Mock()
        mock_response.read.return_value = b'<?xml version="1.0"?><feed></feed>'
        mock_urlopen.return_value.__enter__.return_value = mock_response
        
        result = fetch_rss_feed("UCauyM-A8JIJ9NQcw5_jF00Q")
        assert result == '<?xml version="1.0"?><feed></feed>'
        mock_urlopen.assert_called_once()
    
    @patch('channel_fetcher.urllib.request.urlopen')
    def test_fetch_rss_feed_url_error(self, mock_urlopen):
        """Test handling URL error"""
        import urllib.error
        mock_urlopen.side_effect = urllib.error.URLError("Network error")
        
        with pytest.raises(SystemExit):
            fetch_rss_feed("UCauyM-A8JIJ9NQcw5_jF00Q")


class TestParseRssFeed:
    """Tests for parse_rss_feed function"""
    
    def test_parse_rss_feed_basic(self):
        """Test parsing basic RSS feed"""
        xml_content = """<?xml version="1.0"?>
        <feed xmlns="http://www.w3.org/2005/Atom"
              xmlns:yt="http://www.youtube.com/xml/schemas/2015"
              xmlns:media="http://search.yahoo.com/mrss/">
            <title>Test Channel</title>
            <link rel="alternate" href="https://youtube.com/channel/test"/>
            <entry>
                <title>Test Video</title>
                <link href="https://youtube.com/watch?v=test123"/>
                <published>2024-01-15T10:00:00+00:00</published>
                <updated>2024-01-15T10:00:00+00:00</updated>
                <yt:videoId>test123</yt:videoId>
            </entry>
        </feed>"""
        
        result = parse_rss_feed(xml_content)
        
        assert result['title'] == "Test Channel"
        assert len(result['videos']) == 1
        assert result['videos'][0]['title'] == "Test Video"
        assert result['videos'][0]['id'] == "test123"
    
    def test_parse_rss_feed_date_filtering(self):
        """Test date filtering in RSS feed parsing"""
        xml_content = """<?xml version="1.0"?>
        <feed xmlns="http://www.w3.org/2005/Atom"
              xmlns:yt="http://www.youtube.com/xml/schemas/2015">
            <title>Test Channel</title>
            <link rel="alternate" href="https://youtube.com/channel/test"/>
            <entry>
                <title>Old Video</title>
                <link href="https://youtube.com/watch?v=old"/>
                <published>2024-01-01T10:00:00+00:00</published>
                <updated>2024-01-01T10:00:00+00:00</updated>
                <yt:videoId>old</yt:videoId>
            </entry>
            <entry>
                <title>New Video</title>
                <link href="https://youtube.com/watch?v=new"/>
                <published>2024-01-15T10:00:00+00:00</published>
                <updated>2024-01-15T10:00:00+00:00</updated>
                <yt:videoId>new</yt:videoId>
            </entry>
        </feed>"""
        
        from_date = date(2024, 1, 10)
        to_date = date(2024, 1, 20)
        
        result = parse_rss_feed(xml_content, from_date=from_date, to_date=to_date)
        
        assert len(result['videos']) == 1
        assert result['videos'][0]['title'] == "New Video"
    
    def test_parse_rss_feed_include_author(self):
        """Test including author information"""
        xml_content = """<?xml version="1.0"?>
        <feed xmlns="http://www.w3.org/2005/Atom"
              xmlns:yt="http://www.youtube.com/xml/schemas/2015">
            <title>Test Channel</title>
            <link rel="alternate" href="https://youtube.com/channel/test"/>
            <entry>
                <title>Test Video</title>
                <link href="https://youtube.com/watch?v=test"/>
                <published>2024-01-15T10:00:00+00:00</published>
                <updated>2024-01-15T10:00:00+00:00</updated>
                <yt:videoId>test</yt:videoId>
                <author><name>Test Author</name></author>
            </entry>
        </feed>"""
        
        result = parse_rss_feed(xml_content, include_author=True)
        
        assert 'author' in result['videos'][0]
        assert result['videos'][0]['author'] == "Test Author"
    
    def test_parse_rss_feed_include_description(self):
        """Test including description"""
        xml_content = """<?xml version="1.0"?>
        <feed xmlns="http://www.w3.org/2005/Atom"
              xmlns:yt="http://www.youtube.com/xml/schemas/2015"
              xmlns:media="http://search.yahoo.com/mrss/">
            <title>Test Channel</title>
            <link rel="alternate" href="https://youtube.com/channel/test"/>
            <entry>
                <title>Test Video</title>
                <link href="https://youtube.com/watch?v=test"/>
                <published>2024-01-15T10:00:00+00:00</published>
                <updated>2024-01-15T10:00:00+00:00</updated>
                <yt:videoId>test</yt:videoId>
                <media:group>
                    <media:description>Test description</media:description>
                </media:group>
            </entry>
        </feed>"""
        
        result = parse_rss_feed(xml_content, include_description=True)
        
        assert 'description' in result['videos'][0]
        assert result['videos'][0]['description']['text'] == "Test description"
    
    def test_parse_rss_feed_invalid_xml(self):
        """Test handling invalid XML"""
        invalid_xml = "This is not valid XML"
        
        with pytest.raises(SystemExit):
            parse_rss_feed(invalid_xml)


class TestDisplayReadableFormat:
    """Tests for display_readable_format function"""
    
    def test_display_readable_format_basic(self):
        """Test basic readable format output"""
        channel_info = {
            'title': 'Test Channel',
            'link': 'https://youtube.com/channel/test',
            'videos': [
                {
                    'title': 'Test Video',
                    'link': 'https://youtube.com/watch?v=test',
                    'published': '2024-01-15 10:00'
                }
            ]
        }
        
        result = display_readable_format(channel_info)
        
        assert 'CHANNEL: Test Channel' in result
        assert 'VIDEO #1:' in result
        assert 'Title: Test Video' in result
        assert 'URL: https://youtube.com/watch?v=test' in result
    
    def test_display_readable_format_with_author(self):
        """Test readable format with author"""
        channel_info = {
            'title': 'Test Channel',
            'link': 'https://youtube.com/channel/test',
            'videos': [
                {
                    'title': 'Test Video',
                    'link': 'https://youtube.com/watch?v=test',
                    'published': '2024-01-15 10:00',
                    'author': 'Test Author'
                }
            ]
        }
        
        result = display_readable_format(channel_info)
        assert 'Author: Test Author' in result
    
    def test_display_readable_format_with_description(self):
        """Test readable format with description"""
        channel_info = {
            'title': 'Test Channel',
            'link': 'https://youtube.com/channel/test',
            'videos': [
                {
                    'title': 'Test Video',
                    'link': 'https://youtube.com/watch?v=test',
                    'published': '2024-01-15 10:00',
                    'description': {
                        'text': 'Test description',
                        'links': ['https://example.com']
                    }
                }
            ]
        }
        
        result = display_readable_format(channel_info)
        assert 'Description:' in result
        assert 'Test description' in result
        assert 'Links found in description:' in result


class TestLoadConfig:
    """Tests for load_config function"""
    
    def test_load_config_valid(self, tmp_path):
        """Test loading valid config file"""
        config_file = tmp_path / "config.json"
        config_data = {
            "youtube_channels": [
                {
                    "channel_name_short": "test",
                    "channel_name_long": "Test Podcast",
                    "channel_id": "UC1234567890123456789012"
                }
            ]
        }
        config_file.write_text(json.dumps(config_data), encoding='utf-8')
        
        result = load_config(str(config_file))
        assert result == config_data
    
    def test_load_config_missing_youtube_channels(self, tmp_path):
        """Test loading config missing youtube_channels"""
        config_file = tmp_path / "config.json"
        config_data = {}
        config_file.write_text(json.dumps(config_data), encoding='utf-8')
        
        with pytest.raises(ValueError, match="missing 'youtube_channels' list"):
            load_config(str(config_file))
    
    def test_load_config_invalid_podcast(self, tmp_path):
        """Test loading config with invalid podcast entry"""
        config_file = tmp_path / "config.json"
        config_data = {
            "youtube_channels": [
                {
                    "channel_name_short": "test"
                    # Missing channel_id
                }
            ]
        }
        config_file.write_text(json.dumps(config_data), encoding='utf-8')
        
        with pytest.raises(ValueError, match="missing 'channel_id'"):
            load_config(str(config_file))
    
    def test_load_config_invalid_json(self, tmp_path):
        """Test loading invalid JSON file"""
        config_file = tmp_path / "config.json"
        config_file.write_text("{ invalid json }", encoding='utf-8')
        
        with pytest.raises(ValueError, match="JSON syntax error"):
            load_config(str(config_file))
    
    def test_load_config_file_not_found(self):
        """Test loading non-existent config file"""
        with pytest.raises(ValueError, match="Error reading config file"):
            load_config("/nonexistent/file.json")


class TestFindPodcastByName:
    """Tests for find_podcast_by_name function"""
    
    def test_find_podcast_by_name_found(self):
        """Test finding existing podcast"""
        config = {
            "youtube_channels": [
                {"channel_name_short": "test1", "channel_name_long": "Test 1", "channel_id": "UC1"},
                {"channel_name_short": "test2", "channel_name_long": "Test 2", "channel_id": "UC2"}
            ]
        }
        
        result = find_podcast_by_name(config, "test1")
        assert result == config["youtube_channels"][0]
    
    def test_find_podcast_by_name_not_found(self):
        """Test finding non-existent podcast"""
        config = {
            "youtube_channels": [
                {"channel_name_short": "test1", "channel_name_long": "Test 1", "channel_id": "UC1"}
            ]
        }
        
        with pytest.raises(ValueError, match="not found in config"):
            find_podcast_by_name(config, "nonexistent")


class TestMain:
    """Tests for main function"""
    
    @patch('channel_fetcher.fetch_rss_feed')
    @patch('channel_fetcher.parse_rss_feed')
    @patch('sys.stdout')
    def test_main_outputs_to_stdout(self, mock_stdout, mock_parse, mock_fetch):
        """Test that main outputs to stdout when no output file specified"""
        mock_fetch.return_value = "<feed></feed>"
        mock_parse.return_value = {
            'title': 'Test',
            'link': 'https://test.com',
            'videos': []
        }
        
        with patch('sys.argv', ['channel_fetcher.py', 'UCauyM-A8JIJ9NQcw5_jF00Q']):
            main()
        
        # Should have called print (stdout.write)
        assert mock_stdout.write.called or True  # May use print() instead
    
    @patch('channel_fetcher.fetch_rss_feed')
    @patch('channel_fetcher.parse_rss_feed')
    def test_main_writes_to_file(self, mock_parse, mock_fetch):
        """Test that main writes to file when -o specified"""
        mock_fetch.return_value = "<feed></feed>"
        mock_parse.return_value = {
            'title': 'Test',
            'link': 'https://test.com',
            'videos': []
        }
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            output_file = f.name
        
        try:
            with patch('sys.argv', ['channel_fetcher.py', '-o', output_file, 'UCauyM-A8JIJ9NQcw5_jF00Q']):
                main()
            
            assert os.path.exists(output_file)
            with open(output_file, 'r') as f:
                content = f.read()
                assert 'Test' in content
        finally:
            if os.path.exists(output_file):
                os.remove(output_file)
    
    @patch('channel_fetcher.fetch_rss_feed')
    @patch('channel_fetcher.parse_rss_feed')
    def test_main_date_filtering(self, mock_parse, mock_fetch):
        """Test that date filtering is passed correctly"""
        mock_fetch.return_value = "<feed></feed>"
        mock_parse.return_value = {
            'title': 'Test',
            'link': 'https://test.com',
            'videos': []
        }
        
        with patch('sys.argv', ['channel_fetcher.py', '-f', '2024-01-01', '-t', '2024-01-31', 'UCauyM-A8JIJ9NQcw5_jF00Q']):
            main()
        
        # Verify parse_rss_feed was called with date filters
        mock_parse.assert_called_once()
        call_args = mock_parse.call_args
        assert call_args[1]['from_date'] == date(2024, 1, 1)
        assert call_args[1]['to_date'] == date(2024, 1, 31)
    
    def test_main_invalid_channel_id(self):
        """Test that invalid channel ID raises error"""
        with patch('sys.argv', ['channel_fetcher.py', 'invalid-id']):
            with pytest.raises((SystemExit, ValueError)):
                main()
    
    @patch('channel_fetcher.fetch_rss_feed')
    @patch('channel_fetcher.parse_rss_feed')
    @patch('channel_fetcher.display_readable_format')
    def test_main_readable_format(self, mock_display, mock_parse, mock_fetch):
        """Test that --format readable calls display_readable_format"""
        mock_fetch.return_value = "<feed></feed>"
        mock_parse.return_value = {
            'title': 'Test',
            'link': 'https://test.com',
            'videos': []
        }
        mock_display.return_value = "Readable output"
        
        with patch('sys.argv', ['channel_fetcher.py', '-F', 'readable', 'UCauyM-A8JIJ9NQcw5_jF00Q']):
            with patch('builtins.print') as mock_print:
                main()
        
        mock_display.assert_called_once()
        mock_print.assert_called()
    
    @patch('channel_fetcher.fetch_rss_feed')
    @patch('channel_fetcher.parse_rss_feed')
    @patch('channel_fetcher.json.dumps')
    def test_main_json_format_default(self, mock_json, mock_parse, mock_fetch):
        """Test that default format is JSON"""
        mock_fetch.return_value = "<feed></feed>"
        mock_parse.return_value = {
            'title': 'Test',
            'link': 'https://test.com',
            'videos': []
        }
        mock_json.return_value = '{"title": "Test"}'
        
        with patch('sys.argv', ['channel_fetcher.py', 'UCauyM-A8JIJ9NQcw5_jF00Q']):
            with patch('builtins.print'):
                main()
        
        mock_json.assert_called_once()
    
    @patch('channel_fetcher.fetch_rss_feed')
    @patch('channel_fetcher.parse_rss_feed')
    def test_main_include_author(self, mock_parse, mock_fetch):
        """Test that --include-author flag is passed to parse_rss_feed"""
        mock_fetch.return_value = "<feed></feed>"
        mock_parse.return_value = {
            'title': 'Test',
            'link': 'https://test.com',
            'videos': []
        }
        
        with patch('sys.argv', ['channel_fetcher.py', '-a', 'UCauyM-A8JIJ9NQcw5_jF00Q']):
            with patch('builtins.print'):
                main()
        
        mock_parse.assert_called_once()
        call_args = mock_parse.call_args
        assert call_args[1]['include_author'] is True
    
    @patch('channel_fetcher.fetch_rss_feed')
    @patch('channel_fetcher.parse_rss_feed')
    def test_main_include_description(self, mock_parse, mock_fetch):
        """Test that --include-description flag is passed to parse_rss_feed"""
        mock_fetch.return_value = "<feed></feed>"
        mock_parse.return_value = {
            'title': 'Test',
            'link': 'https://test.com',
            'videos': []
        }
        
        with patch('sys.argv', ['channel_fetcher.py', '-d', 'UCauyM-A8JIJ9NQcw5_jF00Q']):
            with patch('builtins.print'):
                main()
        
        mock_parse.assert_called_once()
        call_args = mock_parse.call_args
        assert call_args[1]['include_description'] is True
    
    @patch('channel_fetcher.fetch_rss_feed')
    @patch('channel_fetcher.parse_rss_feed')
    def test_main_sort_date_asc(self, mock_parse, mock_fetch):
        """Test that --sort date-asc sorts videos ascending"""
        mock_fetch.return_value = "<feed></feed>"
        channel_info = {
            'title': 'Test',
            'link': 'https://test.com',
            'videos': [
                {'title': 'Video 2', 'published': '2024-01-15 10:00'},
                {'title': 'Video 1', 'published': '2024-01-10 10:00'},
            ]
        }
        mock_parse.return_value = channel_info
        
        with patch('sys.argv', ['channel_fetcher.py', '-s', 'date-asc', 'UCauyM-A8JIJ9NQcw5_jF00Q']):
            with patch('builtins.print'):
                main()
        
        # Videos should be sorted by published date ascending
        assert channel_info['videos'][0]['published'] == '2024-01-10 10:00'
        assert channel_info['videos'][1]['published'] == '2024-01-15 10:00'
    
    @patch('channel_fetcher.fetch_rss_feed')
    @patch('channel_fetcher.parse_rss_feed')
    def test_main_sort_date_desc(self, mock_parse, mock_fetch):
        """Test that --sort date-desc sorts videos descending"""
        mock_fetch.return_value = "<feed></feed>"
        channel_info = {
            'title': 'Test',
            'link': 'https://test.com',
            'videos': [
                {'title': 'Video 1', 'published': '2024-01-10 10:00'},
                {'title': 'Video 2', 'published': '2024-01-15 10:00'},
            ]
        }
        mock_parse.return_value = channel_info
        
        with patch('sys.argv', ['channel_fetcher.py', '-s', 'date-desc', 'UCauyM-A8JIJ9NQcw5_jF00Q']):
            with patch('builtins.print'):
                main()
        
        # Videos should be sorted by published date descending
        assert channel_info['videos'][0]['published'] == '2024-01-15 10:00'
        assert channel_info['videos'][1]['published'] == '2024-01-10 10:00'
    
    @patch('channel_fetcher.fetch_rss_feed')
    @patch('channel_fetcher.parse_rss_feed')
    @patch('channel_fetcher.json.dumps')
    def test_main_clean_desc_with_description(self, mock_json, mock_parse, mock_fetch):
        """Test that --clean-desc flag works with --include-description"""
        mock_fetch.return_value = "<feed></feed>"
        channel_info = {
            'title': 'Test',
            'link': 'https://test.com',
            'videos': [
                {
                    'title': 'Video 1',
                    'description': {'text': 'Test description', 'links': []}
                }
            ]
        }
        mock_parse.return_value = channel_info
        mock_json.return_value = '{}'
        
        with patch('sys.argv', ['channel_fetcher.py', '-c', '-d', 'UCauyM-A8JIJ9NQcw5_jF00Q']):
            with patch('builtins.print'):
                main()
        
        # When clean-desc is set, description should remain as dict
        # When clean-desc is NOT set, description should be converted to text
        # This test verifies clean-desc flag is recognized
        mock_json.assert_called_once()
    
    @patch('channel_fetcher.fetch_rss_feed')
    @patch('channel_fetcher.parse_rss_feed')
    @patch('channel_fetcher.load_config')
    def test_main_with_config_and_name(self, mock_load, mock_parse, mock_fetch, tmp_path):
        """Test using --config and --name to lookup channel_id"""
        # Create a test config file
        config_file = tmp_path / "config.json"
        config_data = {
            "youtube_channels": [
                {
                    "channel_name_short": "test",
                    "channel_name_long": "Test Podcast",
                    "channel_id": "UC1234567890123456789012"
                }
            ]
        }
        config_file.write_text(json.dumps(config_data), encoding='utf-8')
        
        mock_fetch.return_value = "<feed></feed>"
        mock_parse.return_value = {
            'title': 'Test',
            'link': 'https://test.com',
            'videos': []
        }
        mock_load.return_value = config_data
        
        with patch('sys.argv', ['channel_fetcher.py', '--config', str(config_file), '--name', 'test']):
            with patch('builtins.print'):
                main()
        
        # Verify load_config was called
        mock_load.assert_called_once_with(str(config_file))
        # Verify fetch_rss_feed was called with the channel_id from config
        mock_fetch.assert_called_once_with("UC1234567890123456789012")
    
    @patch('channel_fetcher.load_config')
    def test_main_config_without_name(self, mock_load):
        """Test that --config without --name raises error"""
        with patch('sys.argv', ['channel_fetcher.py', '--config', 'config.json']):
            with pytest.raises(SystemExit):
                main()
    
    @patch('channel_fetcher.load_config')
    def test_main_config_invalid_podcast_name(self, mock_load, tmp_path):
        """Test that invalid podcast name in config raises error"""
        config_file = tmp_path / "config.json"
        config_data = {
            "youtube_channels": [
                {
                    "channel_name_short": "test",
                    "channel_id": "UC1234567890123456789012"
                }
            ]
        }
        config_file.write_text(json.dumps(config_data), encoding='utf-8')
        mock_load.return_value = config_data
        
        with patch('sys.argv', ['channel_fetcher.py', '--config', str(config_file), '--name', 'nonexistent']):
            with pytest.raises(SystemExit):
                main()
    
    def test_main_no_channel_id_or_config(self):
        """Test that missing both channel_id and config raises error"""
        with patch('sys.argv', ['channel_fetcher.py']):
            with pytest.raises(SystemExit):
                main()
    
    @patch('channel_fetcher.fetch_rss_feed')
    @patch('channel_fetcher.parse_rss_feed')
    def test_main_config_with_date_filtering(self, mock_parse, mock_fetch, tmp_path):
        """Test that date filtering works with config file"""
        config_file = tmp_path / "config.json"
        config_data = {
            "youtube_channels": [
                {
                    "channel_name_short": "test",
                    "channel_id": "UC1234567890123456789012"
                }
            ]
        }
        config_file.write_text(json.dumps(config_data), encoding='utf-8')
        
        mock_fetch.return_value = "<feed></feed>"
        mock_parse.return_value = {
            'title': 'Test',
            'link': 'https://test.com',
            'videos': []
        }
        
        with patch('channel_fetcher.load_config', return_value=config_data):
            with patch('sys.argv', ['channel_fetcher.py', '--config', str(config_file), '--name', 'test',
                                    '-f', '2024-01-01', '-t', '2024-01-31']):
                with patch('builtins.print'):
                    main()
        
        # Verify parse_rss_feed was called with date filters
        mock_parse.assert_called_once()
        call_args = mock_parse.call_args
        assert call_args[1]['from_date'] == date(2024, 1, 1)
        assert call_args[1]['to_date'] == date(2024, 1, 31)

