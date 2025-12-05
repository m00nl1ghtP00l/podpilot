# Podpilot

A Python pipeline for downloading, processing, and transcribing Japanese language learning podcasts from YouTube.

## Overview

Podpilot automates the process of:
1. **Finding** new podcast episodes from YouTube channels via RSS feeds
2. **Downloading** audio files from YouTube videos (with duration filtering)
3. **Transcoding** audio files to optimize size (target: 25MB for API limits)
4. **Transcribing** audio using either OpenAI Whisper API or local Whisper CLI
5. **Generating lessons** from transcriptions using LLMs (Ollama, OpenAI, Anthropic)
6. **Updating metadata** with duration information

## Features

- 🎯 **Japanese Language Focus** - Optimized for Japanese podcast transcription
- 📅 **Date Range Filtering** - Download/transcribe episodes from specific date ranges
- 🔄 **Dual Transcription** - Support for both OpenAI Whisper API and local Whisper CLI
- 📦 **Smart Transcoding** - Automatically reduces file size to meet API limits
- 🎨 **Japanese Character Handling** - Properly handles Japanese characters in filenames
- ✅ **Resume Support** - Can resume interrupted downloads
- 📊 **Progress Tracking** - Visual progress bars for downloads
- 🤖 **LLM Integration** - Generate JLPT-style lessons from transcriptions (Ollama, OpenAI, Anthropic)
- ⏱️ **Duration Management** - Automatic duration extraction and metadata updates
- ⚙️ **Flexible Configuration** - Centralized config with environment variable support

## Tracked Podcasts

Currently tracking 4 Japanese learning channels:
- **hnh** - Haru no nihongo
- **sjn** - Speak Japanese Naturally
- **ss** - Sayuri Saying
- **yuyu** - Yuyu no nihongo

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Podcasts

Copy the example configuration and edit it:

```bash
cp config/podcasts.json.example config/podcasts.json
```

Then edit `config/podcasts.json` to add or modify the channels you want to track:

```json
{
  "youtube_url": "https://www.youtube.com/feeds/videos.xml",
  "data_root": "/path/to/your/data/directory",
  "analysis": {
    "provider": "ollama",
    "model": "qwen2.5:14b"
  },
  "transcription": {
    "provider": "whisper.cpp",
    "model_path": "${WHISPER_MODEL_PATH}"
  },
  "min_duration": 300,
  "youtube_channels": [
    {
      "channel_name_short": "hnh",
      "channel_name_long": "Haru no nihongo",
      "channel_id": "UCauyM-A8JIJ9NQcw5_jF00Q"
    }
  ]
}
```

See `config/README.md` for detailed configuration options.

**Important:** The `data_root` directory is where all downloaded audio files, metadata JSON files, and transcriptions will be stored. Each podcast channel will have its own subdirectory under `data_root` named after the `channel_name_short`.

### Data Directory Structure

For example, if `data_root` is `/Users/eric/Documents/GitHub/podpilot-data` and you're tracking "hnh", files will be stored in:
- **Audio files**: `/Users/eric/Documents/GitHub/podpilot-data/hnh/`
  - Example: `2024-01-15_episode_title.mp3`
- **Metadata JSON**: `/Users/eric/Documents/GitHub/podpilot-data/hnh.json`
  - Contains video metadata (title, URL, publish date, etc.)
- **Transcriptions**: `/Users/eric/Documents/GitHub/podpilot-data/hnh/`
  - `.json` files (structured transcription data)
  - `.txt` files (plain text transcriptions)
  - `.srt` files (subtitle format, if using local Whisper)

### Where Files Are Downloaded

The location depends on which script you use:

1. **`transcribe.py`**: Always uses `{data_root}/{channel_name}` from config
   - Example: `/Users/eric/Documents/GitHub/podpilot-data/hnh/`

2. **`download_audio.py`**:
   - **With `--name`**: Uses `{data_root}/{channel_name}` from config
   - **With direct JSON file**: Defaults to `./downloads` (current directory)
     - Can override with `-a/--audio-dir` option

3. **`local_whisper_transcribe.py`**: Requires `-a/--audio-dir` to be specified
   - Typically points to `{data_root}/{channel_name}`

**Recommendation:** Use a directory outside your project folder (e.g., `../podpilot-data` or `/Users/yourname/Documents/podpilot-data`) to keep data separate from code. This prevents accidentally committing large audio files to git and allows for different backup strategies for code vs. data.

### 3. Set Up Transcription (Optional)

**For OpenAI Whisper API:**
- Set your OpenAI API key: `export OPENAI_API_KEY=your_key_here`

**For Local Whisper CLI:**
- Install whisper-cli: `pip install openai-whisper` or use your preferred installation method
- Ensure `whisper-cli` is in your PATH

## Usage

### Find New Podcast Episodes

```bash
python find_podcasts.py <channel_short_name> [--from-date YYYY-MM-DD] [--to-date YYYY-MM-DD]
```

Example:
```bash
python find_podcasts.py hnh --from-date 2024-01-01 --to-date 2024-01-31
```

### Download Audio Files

**Using direct JSON file:**
```bash
python download_audio.py <json_file> [-a AUDIO_DIR] [--from-date YYYY-MM-DD] [--to-date YYYY-MM-DD]
```

**Using config file (recommended):**
```bash
python download_audio.py --name <podcast_name> [--config CONFIG] [-a AUDIO_DIR] [--from-date YYYY-MM-DD] [--to-date YYYY-MM-DD]
```

Examples:
```bash
# Direct JSON file (defaults to ./downloads)
python download_audio.py hnh.json --from-date 2024-01-01

# Using config file (uses data_root/podcast_name from config)
python download_audio.py --name hnh --from-date 2024-01-01

# Custom audio directory
python download_audio.py --name hnh -a /custom/path --from-date 2024-01-01
```

**Note:** When using `--name`, audio files are stored in `{data_root}/{channel_name}` from your config. When using a direct JSON file, files default to `./downloads` unless you specify `-a/--audio-dir`.

### Transcribe Audio

**Using OpenAI Whisper API:**
```bash
python transcribe.py --name <podcast_name> [--config CONFIG] [--from-date YYYY-MM-DD] [--to-date YYYY-MM-DD]
```

**Using Local Whisper CLI:**
```bash
python local_whisper_transcribe.py -a <audio_dir> [--from-date YYYY-MM-DD] [--to-date YYYY-MM-DD]
```

Examples:
```bash
# Transcribe using OpenAI API (uses data_root/podcast_name from config)
python transcribe.py --name hnh --from-date 2024-01-01

# Transcribe using local Whisper CLI
python local_whisper_transcribe.py -a /Users/eric/Documents/GitHub/podpilot-data/hnh --from-date 2024-01-01
```

**Note:** `transcribe.py` automatically uses `{data_root}/{channel_name}` from your config file. `local_whisper_transcribe.py` requires you to specify the audio directory with `-a/--audio-dir`.

### Generate Lessons from Transcriptions

**Single file:**
```bash
python generate_lesson.py <transcription_file> [--name PODCAST_NAME] [--format json|markdown]
```

**Batch processing:**
```bash
python process_lessons.py -a <audio_dir> [--name PODCAST_NAME] [--from-date YYYY-MM-DD] [--to-date YYYY-MM-DD]
```

Examples:
```bash
# Generate lesson from single transcription
python generate_lesson.py transcript.txt --name hnh --format markdown

# Batch process all transcriptions
python process_lessons.py -a /path/to/audio --name hnh --from-date 2024-01-01
```

### Update Duration Metadata

```bash
python update_durations.py <json_file> [--audio-dir AUDIO_DIR]
```

This updates the metadata JSON file with duration information extracted from audio files.

## Project Structure

```
podpilot/
├── find_podcasts.py              # Find episodes from YouTube RSS feeds
├── download_audio.py             # Download audio from YouTube
├── transcribe.py                 # Transcribe using OpenAI Whisper API
├── local_whisper_transcribe.py   # Transcribe using local Whisper CLI
├── mp3_transcoder.py             # Transcode audio files to target size
├── update_durations.py           # Update metadata with duration info
├── generate_lesson.py            # Generate JLPT lessons from transcriptions
├── process_lessons.py            # Batch process transcriptions into lessons
├── llm_providers.py              # LLM provider implementations (Ollama, OpenAI, Anthropic)
├── llm_config.py                 # LLM configuration defaults
├── podcast_downloader.py        # Alternative downloader implementation
├── config/                       # Configuration files
│   ├── podcasts.json.example     # Example configuration
│   ├── podcasts.json             # Your configuration (gitignored)
│   └── README.md                 # Configuration documentation
├── docs/                         # Documentation
│   └── README.md                 # Testing guide
├── scripts/                      # Utility scripts
│   └── run_tests.sh
├── tests/                        # Test suite
│   ├── test_download_audio.py
│   ├── test_find_podcasts.py
│   ├── test_transcribe.py
│   ├── test_local_whisper_transcribe.py
│   ├── test_mp3_transcoder.py
│   ├── test_podcast_downloader.py
│   └── test_update_durations.py
├── requirements.txt              # Python dependencies
├── pyproject.toml                # Modern Python packaging config
└── downloads/                    # Downloaded audio files (gitignored)
```

## Testing

Run the test suite:

```bash
pytest
```

Or with coverage:

```bash
pytest --cov=. --cov-report=html
```

See `TESTING.md` and `tests/README.md` for more information.

## Requirements

- Python 3.10+
- ffmpeg (for audio transcoding)
- yt-dlp (for YouTube downloads)
- OpenAI API key (optional, for cloud transcription or lesson generation)
- whisper-cli (optional, for local transcription)
- Ollama (optional, for local lesson generation)
- Anthropic API key (optional, for Claude-based lesson generation)

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]
