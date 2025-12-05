"""
Tests for podcast_downloader.py

This test suite covers:
- Date parsing
- File downloading (with mocks)
- Episode date extraction
- Audio URL extraction
- Metadata saving
- Episode downloading
- CLI interface
"""

import pytest
import os
import tempfile
import json
import argparse
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, mock_open
from datetime import datetime
import sys

# Add parent directory to path to import the module
sys.path.insert(0, str(Path(__file__).parent.parent))

from podcast_downloader import (
    parse_date,
    download_file,
    get_episode_date,
    get_episode_audio_url,
    save_metadata,
    download_episode,
    main
)


class TestParseDate:
    """Tests for parse_date function"""
    
    def test_parse_date_valid(self):
        """Test parsing valid date string"""
        result = parse_date('2024-01-15')
        assert result == datetime(2024, 1, 15)
    
    def test_parse_date_invalid_format(self):
        """Test parsing invalid date format"""
        with pytest.raises(argparse.ArgumentTypeError):
            parse_date('01/15/2024')
    
    def test_parse_date_invalid_date(self):
        """Test parsing invalid date"""
        with pytest.raises(argparse.ArgumentTypeError):
            parse_date('2024-13-45')
    
    def test_parse_date_empty_string(self):
        """Test parsing empty string"""
        with pytest.raises(argparse.ArgumentTypeError):
            parse_date('')
    
    def test_parse_date_different_years(self):
        """Test parsing dates in different years"""
        result = parse_date('2020-12-31')
        assert result == datetime(2020, 12, 31)
        
        result = parse_date('2025-01-01')
        assert result == datetime(2025, 1, 1)


class TestGetEpisodeDate:
    """Tests for get_episode_date function"""
    
    def test_get_episode_date_published_parsed(self):
        """Test extracting date from published_parsed"""
        episode = Mock()
        episode.published_parsed = (2024, 1, 15, 10, 30, 0, 0, 0, 0)
        episode.updated_parsed = None
        episode.created_parsed = None
        
        result = get_episode_date(episode)
        assert result == datetime(2024, 1, 15, 10, 30, 0)
    
    def test_get_episode_date_updated_parsed(self):
        """Test extracting date from updated_parsed when published_parsed missing"""
        episode = Mock()
        del episode.published_parsed  # Remove attribute so hasattr returns False
        episode.updated_parsed = (2024, 2, 20, 14, 0, 0, 0, 0, 0)
        del episode.created_parsed
        
        result = get_episode_date(episode)
        assert result == datetime(2024, 2, 20, 14, 0, 0)
    
    def test_get_episode_date_created_parsed(self):
        """Test extracting date from created_parsed when others missing"""
        episode = Mock()
        del episode.published_parsed
        del episode.updated_parsed
        episode.created_parsed = (2024, 3, 10, 8, 15, 0, 0, 0, 0)
        
        result = get_episode_date(episode)
        assert result == datetime(2024, 3, 10, 8, 15, 0)
    
    def test_get_episode_date_none(self):
        """Test returning None when no date fields available"""
        episode = Mock()
        del episode.published_parsed
        del episode.updated_parsed
        del episode.created_parsed
        
        result = get_episode_date(episode)
        assert result is None
    
    def test_get_episode_date_priority(self):
        """Test that published_parsed takes priority over others"""
        episode = Mock()
        episode.published_parsed = (2024, 1, 1, 0, 0, 0, 0, 0, 0)
        episode.updated_parsed = (2024, 2, 1, 0, 0, 0, 0, 0, 0)
        episode.created_parsed = (2024, 3, 1, 0, 0, 0, 0, 0, 0)
        
        result = get_episode_date(episode)
        assert result == datetime(2024, 1, 1)


class TestGetEpisodeAudioUrl:
    """Tests for get_episode_audio_url function"""
    
    def test_get_episode_audio_url_mp3(self):
        """Test getting MP3 URL"""
        class Link:
            def __init__(self, type_val, href_val):
                self.type = type_val
                self.href = href_val
            def __contains__(self, key):
                return key == 'type'
        link1 = Link('audio/mpeg', 'http://example.com/episode.mp3')
        link2 = Link('audio/m4a', 'http://example.com/episode.m4a')
        
        class Episode:
            def __contains__(self, key):
                return key == 'links'
        episode = Episode()
        episode.links = [link1, link2]
        
        result = get_episode_audio_url(episode, 'mp3')
        assert result == 'http://example.com/episode.mp3'
    
    def test_get_episode_audio_url_m4a(self):
        """Test getting M4A URL"""
        class Link:
            def __init__(self, type_val, href_val):
                self.type = type_val
                self.href = href_val
            def __contains__(self, key):
                return key == 'type'
        link1 = Link('audio/mpeg', 'http://example.com/episode.mp3')
        link2 = Link('audio/m4a', 'http://example.com/episode.m4a')
        
        class Episode:
            def __contains__(self, key):
                return key == 'links'
        episode = Episode()
        episode.links = [link1, link2]
        
        result = get_episode_audio_url(episode, 'm4a')
        assert result == 'http://example.com/episode.m4a'
    
    def test_get_episode_audio_url_all(self):
        """Test getting first audio URL when format is 'all'"""
        class Link:
            def __init__(self, type_val, href_val):
                self.type = type_val
                self.href = href_val
            def __contains__(self, key):
                return key == 'type'
        link = Link('audio/mpeg', 'http://example.com/episode.mp3')
        
        class Episode:
            def __contains__(self, key):
                return key == 'links'
        episode = Episode()
        episode.links = [link]
        
        result = get_episode_audio_url(episode, 'all')
        assert result == 'http://example.com/episode.mp3'
    
    def test_get_episode_audio_url_no_links(self):
        """Test returning None when no links"""
        class Episode:
            def __contains__(self, key):
                return False
        episode = Episode()
        
        result = get_episode_audio_url(episode, 'mp3')
        assert result is None
    
    def test_get_episode_audio_url_no_audio_links(self):
        """Test returning None when no audio links"""
        class Link:
            def __init__(self, type_val, href_val):
                self.type = type_val
                self.href = href_val
            def __contains__(self, key):
                return key == 'type'
        link = Link('text/html', 'http://example.com/page.html')
        
        class Episode:
            def __contains__(self, key):
                return key == 'links'
        episode = Episode()
        episode.links = [link]
        
        result = get_episode_audio_url(episode, 'mp3')
        assert result is None
    
    def test_get_episode_audio_url_format_in_href(self):
        """Test matching format in href when type doesn't match exactly"""
        class Link:
            def __init__(self, type_val, href_val):
                self.type = type_val
                self.href = href_val
            def __contains__(self, key):
                return key == 'type'
        link = Link('audio/mpeg', 'http://example.com/episode.m4a')
        
        class Episode:
            def __contains__(self, key):
                return key == 'links'
        episode = Episode()
        episode.links = [link]
        
        result = get_episode_audio_url(episode, 'm4a')
        assert result == 'http://example.com/episode.m4a'


class TestSaveMetadata:
    """Tests for save_metadata function"""
    
    @patch('podcast_downloader.get_episode_date')
    @patch('podcast_downloader.get_episode_audio_url')
    def test_save_metadata_single_episode(self, mock_get_url, mock_get_date):
        """Test saving metadata for single episode"""
        mock_get_date.return_value = datetime(2024, 1, 15)
        mock_get_url.return_value = 'http://example.com/episode.mp3'
        
        episode = Mock()
        episode.title = 'Test Episode'
        episode.get = Mock(side_effect=lambda key, default='': {
            'description': 'Test description',
            'itunes_duration': '30:00',
            'author': 'Test Author',
            'id': 'episode-123'
        }.get(key, default))
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_metadata([episode], tmpdir)
            
            metadata_path = Path(tmpdir) / 'episode_metadata.json'
            assert metadata_path.exists()
            
            with open(metadata_path, 'r') as f:
                data = json.load(f)
            
            assert len(data) == 1
            assert data[0]['title'] == 'Test Episode'
            assert data[0]['audio_url'] == 'http://example.com/episode.mp3'
            assert data[0]['description'] == 'Test description'
    
    @patch('podcast_downloader.get_episode_date')
    @patch('podcast_downloader.get_episode_audio_url')
    def test_save_metadata_multiple_episodes(self, mock_get_url, mock_get_date):
        """Test saving metadata for multiple episodes"""
        mock_get_date.return_value = datetime(2024, 1, 15)
        mock_get_url.return_value = 'http://example.com/episode.mp3'
        
        episodes = []
        for i in range(3):
            episode = Mock()
            episode.title = f'Episode {i+1}'
            episode.get = Mock(return_value='')
            episodes.append(episode)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_metadata(episodes, tmpdir)
            
            metadata_path = Path(tmpdir) / 'episode_metadata.json'
            with open(metadata_path, 'r') as f:
                data = json.load(f)
            
            assert len(data) == 3


class TestDownloadFile:
    """Tests for download_file function"""
    
    @patch('podcast_downloader.tqdm')
    @patch('podcast_downloader.requests.get')
    @patch('builtins.open', new_callable=mock_open)
    def test_download_file_success(self, mock_file, mock_get, mock_tqdm):
        """Test successful file download"""
        # Mock response
        mock_response = Mock()
        mock_response.headers = {'content-length': '1000'}
        mock_response.iter_content = Mock(return_value=[b'chunk1', b'chunk2'])
        mock_get.return_value = mock_response
        
        # Mock tqdm
        mock_pbar = Mock()
        mock_tqdm.return_value.__enter__ = Mock(return_value=mock_pbar)
        mock_tqdm.return_value.__exit__ = Mock(return_value=None)
        
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            output_path = Path(tmp.name)
        
        try:
            download_file('http://example.com/file.mp3', output_path)
            
            mock_get.assert_called_once()
            mock_file.assert_called()
        finally:
            if output_path.exists():
                os.remove(output_path)
    
    @patch('podcast_downloader.tqdm')
    @patch('podcast_downloader.requests.get')
    @patch('builtins.open', new_callable=mock_open)
    def test_download_file_resume(self, mock_file, mock_get, mock_tqdm):
        """Test resuming interrupted download"""
        # Mock existing file
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(b'existing data')
            output_path = Path(tmp.name)
            existing_size = len(b'existing data')
        
        # Mock response with 206 (partial content)
        mock_response = Mock()
        mock_response.status_code = 206
        mock_response.headers = {'content-length': '500'}
        mock_response.iter_content = Mock(return_value=[b'new chunk'])
        mock_get.return_value = mock_response
        
        # Mock tqdm
        mock_pbar = Mock()
        mock_tqdm.return_value.__enter__ = Mock(return_value=mock_pbar)
        mock_tqdm.return_value.__exit__ = Mock(return_value=None)
        
        try:
            download_file('http://example.com/file.mp3', output_path, resume=True)
            
            # Check that Range header was set
            call_args = mock_get.call_args
            assert 'Range' in call_args[1]['headers']
            assert call_args[1]['headers']['Range'] == f'bytes={existing_size}-'
        finally:
            if output_path.exists():
                os.remove(output_path)


class TestDownloadEpisode:
    """Tests for download_episode function"""
    
    @patch('podcast_downloader.download_file')
    @patch('podcast_downloader.get_episode_audio_url')
    @patch('podcast_downloader.get_episode_date')
    def test_download_episode_success(self, mock_get_date, mock_get_url, mock_download):
        """Test successful episode download"""
        mock_get_date.return_value = datetime(2024, 1, 15)
        mock_get_url.return_value = 'http://example.com/episode.mp3'
        mock_download.return_value = None
        
        episode = Mock()
        episode.title = 'Test Episode'
        
        with tempfile.TemporaryDirectory() as tmpdir:
            result = download_episode(episode, tmpdir, 'mp3', None, 30, False)
            
            assert result is True
            mock_download.assert_called_once()
    
    @patch('podcast_downloader.get_episode_audio_url')
    def test_download_episode_no_audio_url(self, mock_get_url):
        """Test episode download when no audio URL found"""
        mock_get_url.return_value = None
        
        episode = Mock()
        episode.title = 'Test Episode'
        
        with tempfile.TemporaryDirectory() as tmpdir:
            result = download_episode(episode, tmpdir, 'mp3', None, 30, False)
            
            assert result is False
    
    @patch('podcast_downloader.requests.head')
    @patch('podcast_downloader.get_episode_audio_url')
    def test_download_episode_exceeds_max_size(self, mock_get_url, mock_head):
        """Test skipping episode that exceeds max size"""
        mock_get_url.return_value = 'http://example.com/episode.mp3'
        
        mock_response = Mock()
        mock_response.headers = {'content-length': str(30 * 1024 * 1024)}  # 30MB
        mock_head.return_value = mock_response
        
        episode = Mock()
        episode.title = 'Test Episode'
        
        with tempfile.TemporaryDirectory() as tmpdir:
            result = download_episode(episode, tmpdir, 'mp3', 25, 30, False)
            
            assert result is False
    
    @patch('podcast_downloader.download_file')
    @patch('podcast_downloader.requests.head')
    @patch('podcast_downloader.get_episode_audio_url')
    @patch('podcast_downloader.get_episode_date')
    def test_download_episode_within_max_size(self, mock_get_date, mock_get_url, mock_head, mock_download):
        """Test downloading episode within max size"""
        mock_get_date.return_value = datetime(2024, 1, 15)
        mock_get_url.return_value = 'http://example.com/episode.mp3'
        
        mock_response = Mock()
        mock_response.headers = {'content-length': str(20 * 1024 * 1024)}  # 20MB
        mock_head.return_value = mock_response
        
        episode = Mock()
        episode.title = 'Test Episode'
        
        with tempfile.TemporaryDirectory() as tmpdir:
            result = download_episode(episode, tmpdir, 'mp3', 25, 30, False)
            
            assert result is True
            mock_download.assert_called_once()
    
    @patch('podcast_downloader.download_file')
    @patch('podcast_downloader.get_episode_audio_url')
    @patch('podcast_downloader.get_episode_date')
    def test_download_episode_download_error(self, mock_get_date, mock_get_url, mock_download):
        """Test handling download error"""
        mock_get_date.return_value = datetime(2024, 1, 15)
        mock_get_url.return_value = 'http://example.com/episode.mp3'
        mock_download.side_effect = Exception('Download failed')
        
        episode = Mock()
        episode.title = 'Test Episode'
        
        with tempfile.TemporaryDirectory() as tmpdir:
            result = download_episode(episode, tmpdir, 'mp3', None, 30, False)
            
            assert result is False


class TestMain:
    """Tests for main CLI function"""
    
    @patch('podcast_downloader.feedparser.parse')
    def test_main_no_episodes(self, mock_parse):
        """Test handling feed with no episodes"""
        mock_feed = Mock()
        mock_feed.entries = []
        mock_parse.return_value = mock_feed
        
        with patch('sys.argv', ['podcast_downloader.py', 'http://example.com/feed.xml']):
            with patch('sys.exit'):
                main()
    
    @patch('podcast_downloader.feedparser.parse')
    @patch('podcast_downloader.get_episode_date')
    def test_main_list_episodes(self, mock_get_date, mock_parse):
        """Test listing episodes"""
        mock_get_date.return_value = datetime(2024, 1, 15)
        
        episode = Mock()
        episode.title = 'Test Episode'
        
        mock_feed = Mock()
        mock_feed.entries = [episode]
        mock_parse.return_value = mock_feed
        
        with patch('sys.argv', ['podcast_downloader.py', 'http://example.com/feed.xml', '--list']):
            with patch('builtins.print'):
                main()
    
    @patch('podcast_downloader.feedparser.parse')
    @patch('podcast_downloader.get_episode_date')
    def test_main_latest_episode(self, mock_get_date, mock_parse):
        """Test downloading latest episode"""
        mock_get_date.return_value = datetime(2024, 1, 15)
        
        episodes = [Mock(title=f'Episode {i}') for i in range(3)]
        
        mock_feed = Mock()
        mock_feed.entries = episodes
        mock_parse.return_value = mock_feed
        
        with patch('sys.argv', ['podcast_downloader.py', 'http://example.com/feed.xml', '--latest']):
            with patch('podcast_downloader.download_episode', return_value=True):
                with patch('builtins.print'):
                    main()
    
    @patch('podcast_downloader.feedparser.parse')
    @patch('podcast_downloader.get_episode_date')
    def test_main_simulate(self, mock_get_date, mock_parse):
        """Test simulate mode"""
        mock_get_date.return_value = datetime(2024, 1, 15)
        
        episode = Mock()
        episode.title = 'Test Episode'
        
        mock_feed = Mock()
        mock_feed.entries = [episode]
        mock_parse.return_value = mock_feed
        
        with patch('sys.argv', ['podcast_downloader.py', 'http://example.com/feed.xml', '--simulate']):
            with patch('builtins.print'):
                main()
    
    @patch('podcast_downloader.feedparser.parse')
    @patch('podcast_downloader.get_episode_date')
    def test_main_save_metadata(self, mock_get_date, mock_parse):
        """Test saving metadata"""
        mock_get_date.return_value = datetime(2024, 1, 15)
        
        episode = Mock()
        episode.title = 'Test Episode'
        episode.get = Mock(return_value='')
        episode.__contains__ = Mock(return_value=False)  # For date filtering
        
        mock_feed = Mock()
        mock_feed.entries = [episode]
        mock_parse.return_value = mock_feed
        
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch('sys.argv', ['podcast_downloader.py', 'http://example.com/feed.xml', '--metadata', '--output-dir', tmpdir]):
                with patch('podcast_downloader.save_metadata') as mock_save:
                    with patch('builtins.print'):
                        main()
                    
                    mock_save.assert_called_once()
    
    @patch('podcast_downloader.feedparser.parse')
    @patch('podcast_downloader.get_episode_date')
    def test_main_episode_number(self, mock_get_date, mock_parse):
        """Test downloading specific episode number"""
        mock_get_date.return_value = datetime(2024, 1, 15)
        
        episodes = [Mock(title=f'Episode {i}') for i in range(3)]
        
        mock_feed = Mock()
        mock_feed.entries = episodes
        mock_parse.return_value = mock_feed
        
        with patch('sys.argv', ['podcast_downloader.py', 'http://example.com/feed.xml', '--episode', '1']):
            with patch('podcast_downloader.download_episode', return_value=True):
                with patch('builtins.print'):
                    main()
    
    @patch('podcast_downloader.feedparser.parse')
    @patch('podcast_downloader.get_episode_date')
    def test_main_invalid_episode_number(self, mock_get_date, mock_parse):
        """Test handling invalid episode number"""
        mock_get_date.return_value = datetime(2024, 1, 15)
        
        episode = Mock()
        episode.title = 'Episode 0'
        
        mock_feed = Mock()
        mock_feed.entries = [episode]
        mock_parse.return_value = mock_feed
        
        with patch('sys.argv', ['podcast_downloader.py', 'http://example.com/feed.xml', '--episode', '5']):
            with patch('builtins.print'):
                with patch('sys.exit'):
                    main()
    
    @patch('podcast_downloader.feedparser.parse')
    @patch('podcast_downloader.get_episode_date')
    def test_main_date_filtering(self, mock_get_date, mock_parse):
        """Test date filtering"""
        # get_episode_date is called multiple times: once for filtering, once for sorting
        dates = [
            datetime(2024, 1, 10),  # First call for filtering
            datetime(2024, 1, 15),  # Second call for filtering
            datetime(2024, 1, 20),  # Third call for filtering
            datetime(2024, 1, 10),  # First call for sorting
            datetime(2024, 1, 15),  # Second call for sorting
            datetime(2024, 1, 20),  # Third call for sorting
        ]
        mock_get_date.side_effect = dates
        
        episodes = [Mock(title=f'Episode {i}') for i in range(3)]
        
        mock_feed = Mock()
        mock_feed.entries = episodes
        mock_parse.return_value = mock_feed
        
        with patch('sys.argv', ['podcast_downloader.py', 'http://example.com/feed.xml', 
                                '--from-date', '2024-01-12', '--to-date', '2024-01-18']):
            with patch('podcast_downloader.download_episode', return_value=True):
                with patch('builtins.print'):
                    with patch('sys.exit'):
                        main()
    
    @patch('podcast_downloader.feedparser.parse')
    @patch('podcast_downloader.get_episode_date')
    def test_main_since_date(self, mock_get_date, mock_parse):
        """Test --since date option"""
        dates = [
            datetime(2024, 1, 10),
            datetime(2024, 1, 15),
            datetime(2024, 1, 20),
            datetime(2024, 1, 15),  # For sorting
            datetime(2024, 1, 20),   # For sorting
        ]
        mock_get_date.side_effect = dates
        
        episodes = [Mock(title=f'Episode {i}') for i in range(3)]
        
        mock_feed = Mock()
        mock_feed.entries = episodes
        mock_parse.return_value = mock_feed
        
        with patch('sys.argv', ['podcast_downloader.py', 'http://example.com/feed.xml', 
                                '--since', '2024-01-12']):
            with patch('podcast_downloader.download_episode', return_value=True):
                with patch('builtins.print'):
                    main()
    
    @patch('podcast_downloader.feedparser.parse')
    @patch('podcast_downloader.get_episode_date')
    def test_main_custom_output_dir(self, mock_get_date, mock_parse):
        """Test custom output directory"""
        mock_get_date.return_value = datetime(2024, 1, 15)
        
        episode = Mock()
        episode.title = 'Test Episode'
        
        mock_feed = Mock()
        mock_feed.entries = [episode]
        mock_parse.return_value = mock_feed
        
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch('sys.argv', ['podcast_downloader.py', 'http://example.com/feed.xml', 
                                    '--output-dir', tmpdir]):
                with patch('podcast_downloader.download_episode', return_value=True) as mock_download:
                    with patch('builtins.print'):
                        main()
                    
                    # Verify download_episode was called with custom output dir
                    mock_download.assert_called_once()
                    call_args = mock_download.call_args
                    # output_dir is converted to Path in main(), so compare Path objects
                    from pathlib import Path
                    assert call_args[0][1] == Path(tmpdir)  # output_dir is second argument
    
    @patch('podcast_downloader.feedparser.parse')
    @patch('podcast_downloader.get_episode_date')
    def test_main_format_m4a(self, mock_get_date, mock_parse):
        """Test --format m4a option"""
        mock_get_date.return_value = datetime(2024, 1, 15)
        
        episode = Mock()
        episode.title = 'Test Episode'
        
        mock_feed = Mock()
        mock_feed.entries = [episode]
        mock_parse.return_value = mock_feed
        
        with patch('sys.argv', ['podcast_downloader.py', 'http://example.com/feed.xml', 
                                '--format', 'm4a']):
            with patch('podcast_downloader.download_episode', return_value=True) as mock_download:
                with patch('builtins.print'):
                    main()
                
                # Verify download_episode was called with m4a format
                mock_download.assert_called_once()
                call_args = mock_download.call_args
                assert call_args[0][2] == 'm4a'  # format is third argument
    
    @patch('podcast_downloader.feedparser.parse')
    @patch('podcast_downloader.get_episode_date')
    def test_main_format_all(self, mock_get_date, mock_parse):
        """Test --format all option"""
        mock_get_date.return_value = datetime(2024, 1, 15)
        
        episode = Mock()
        episode.title = 'Test Episode'
        
        mock_feed = Mock()
        mock_feed.entries = [episode]
        mock_parse.return_value = mock_feed
        
        with patch('sys.argv', ['podcast_downloader.py', 'http://example.com/feed.xml', 
                                '--format', 'all']):
            with patch('podcast_downloader.download_episode', return_value=True) as mock_download:
                with patch('builtins.print'):
                    main()
                
                mock_download.assert_called_once()
                call_args = mock_download.call_args
                assert call_args[0][2] == 'all'
    
    @patch('podcast_downloader.feedparser.parse')
    @patch('podcast_downloader.get_episode_date')
    def test_main_max_size(self, mock_get_date, mock_parse):
        """Test --max-size option"""
        mock_get_date.return_value = datetime(2024, 1, 15)
        
        episode = Mock()
        episode.title = 'Test Episode'
        
        mock_feed = Mock()
        mock_feed.entries = [episode]
        mock_parse.return_value = mock_feed
        
        with patch('sys.argv', ['podcast_downloader.py', 'http://example.com/feed.xml', 
                                '--max-size', '50']):
            with patch('podcast_downloader.download_episode', return_value=True) as mock_download:
                with patch('builtins.print'):
                    main()
                
                mock_download.assert_called_once()
                call_args = mock_download.call_args
                assert call_args[0][3] == 50  # max_size is fourth argument
    
    @patch('podcast_downloader.feedparser.parse')
    @patch('podcast_downloader.get_episode_date')
    def test_main_resume(self, mock_get_date, mock_parse):
        """Test --resume option"""
        mock_get_date.return_value = datetime(2024, 1, 15)
        
        episode = Mock()
        episode.title = 'Test Episode'
        
        mock_feed = Mock()
        mock_feed.entries = [episode]
        mock_parse.return_value = mock_feed
        
        with patch('sys.argv', ['podcast_downloader.py', 'http://example.com/feed.xml', 
                                '--resume']):
            with patch('podcast_downloader.download_episode', return_value=True) as mock_download:
                with patch('builtins.print'):
                    main()
                
                mock_download.assert_called_once()
                call_args = mock_download.call_args
                assert call_args[0][5] is True  # resume is sixth argument
    
    @patch('podcast_downloader.feedparser.parse')
    @patch('podcast_downloader.get_episode_date')
    def test_main_timeout(self, mock_get_date, mock_parse):
        """Test --timeout option"""
        mock_get_date.return_value = datetime(2024, 1, 15)
        
        episode = Mock()
        episode.title = 'Test Episode'
        
        mock_feed = Mock()
        mock_feed.entries = [episode]
        mock_parse.return_value = mock_feed
        
        with patch('sys.argv', ['podcast_downloader.py', 'http://example.com/feed.xml', 
                                '--timeout', '60']):
            with patch('podcast_downloader.download_episode', return_value=True) as mock_download:
                with patch('builtins.print'):
                    main()
                
                mock_download.assert_called_once()
                call_args = mock_download.call_args
                assert call_args[0][4] == 60  # timeout is fifth argument
    
    @patch('podcast_downloader.feedparser.parse')
    @patch('podcast_downloader.get_episode_date')
    @patch('podcast_downloader.download_episode')
    @patch('podcast_downloader.concurrent.futures.ThreadPoolExecutor')
    def test_main_parallel_downloads(self, mock_executor, mock_download, mock_get_date, mock_parse):
        """Test --parallel option for parallel downloads"""
        mock_get_date.return_value = datetime(2024, 1, 15)
        mock_download.return_value = True
        
        episodes = [Mock(title=f'Episode {i}') for i in range(3)]
        
        mock_feed = Mock()
        mock_feed.entries = episodes
        mock_parse.return_value = mock_feed
        
        # Set up ThreadPoolExecutor as a proper context manager
        mock_executor_instance = Mock()
        mock_executor_instance.submit.return_value = Mock()  # Return a future-like object
        
        mock_executor.return_value.__enter__ = Mock(return_value=mock_executor_instance)
        mock_executor.return_value.__exit__ = Mock(return_value=None)
        
        with patch('sys.argv', ['podcast_downloader.py', 'http://example.com/feed.xml', 
                                '--parallel', '3']):
            with patch('builtins.print'):
                with patch('podcast_downloader.concurrent.futures.wait'):  # Mock wait to avoid hanging
                    main()
            
            # Verify ThreadPoolExecutor was created with correct max_workers
            mock_executor.assert_called_once_with(max_workers=3)
            # Verify submit was called for each episode
            assert mock_executor_instance.submit.call_count == 3

