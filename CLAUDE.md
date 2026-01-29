# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick Commands

```bash
# Core pipeline scripts
python channel_fetcher.py <channel> --from-date 7d               # Fetch new episodes
python download_audio.py --name <channel> --from-date 7d         # Download audio
python local_whisper_transcribe.py -a $HOME/podpilot-data/<channel> --from-date 7d  # Transcribe
python generate_lesson.py --name <channel> --from-date 7d        # Generate lessons

# Testing
pytest                            # Run all tests
pytest tests/test_channel_fetcher.py  # Single test file
pytest --cov=. --cov-report=html  # With coverage
```

## Channels

Configure your podcast channels in `config/podcasts.json` (copy from `podcasts.json.example`).

Each channel needs:
- Short name (e.g., "hnh", "sjn") - used in commands
- RSS feed URL
- Language settings

Example channel configuration:
```json
"channels": {
  "example": {
    "name": "Example Podcast",
    "rss_url": "https://youtube.com/feeds/videos.xml?channel_id=YOUR_CHANNEL_ID",
    "language": "ja"
  }
}
```

## Architecture

### Pipeline Flow

Discovery → Download → Transcription → Lesson Generation

The system uses standalone scripts for batch processing of podcast episodes.

### Core Scripts

- `channel_fetcher.py` - Fetches episodes from YouTube RSS feeds
- `download_audio.py` - Downloads audio via yt-dlp
- `local_whisper_transcribe.py` - Transcribes audio via whisper.cpp
- `transcribe.py` - Transcribes audio via OpenAI Whisper API
- `generate_lesson.py` - Generates learning lessons from transcripts using LLMs

### LLM Providers (`llm_providers.py`)

Abstract `LLMProvider` base with `OllamaProvider`, `OpenAIProvider`, `AnthropicProvider`. Factory function `get_provider()` creates the right one from config. 10-minute timeout for generation.

### Language Adapters (`adapters/`)

`LanguageAdapter` base class with `JapaneseAdapter` implementation. Provides language-specific prompts for transcription and lesson generation. Prompts can be overridden via config file paths (`analysis.prompt_files` in podcasts.json).

## Configuration

`config/podcasts.json` (gitignored, copy from `podcasts.json.example`):
- `data_root` - Where audio/transcripts are stored
- `analysis.provider` - LLM provider (`ollama`, `openai`, `anthropic`)
- `analysis.prompt_files` - Optional system/user prompt file overrides
- `transcription.provider` - `whisper.cpp` or `openai`
- Supports `$VAR` and `${VAR}` environment variable expansion

## Key Conventions

- Episode IDs are YouTube video IDs (alphanumeric)
- Data files live under `data_root/<channel>/` as `YYYY-MM-DD_title.{mp3,json,txt,srt}`
- Transcript format: timestamped lines with YouTube URLs (`[HH:MM:SS.mmm] url\ntext`)
- LLM-generated lessons are JSON with `summary`, `vocabulary`, `grammar_points`, `quiz_questions`

## Dependencies

- Python 3.10+, ffmpeg, yt-dlp, whisper.cpp, Ollama/OpenAI/Anthropic
