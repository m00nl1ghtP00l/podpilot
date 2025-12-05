# Testing Guide

## Setup

1. Install test dependencies:
```bash
pip install -r requirements.txt
```

Or if using a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Running Tests

### Run all tests:
```bash
pytest
```

### Run tests for a specific file:
```bash
pytest tests/test_download_audio.py
```

### Run a specific test class:
```bash
pytest tests/test_download_audio.py::TestLoadJson
```

### Run a specific test:
```bash
pytest tests/test_download_audio.py::TestLoadJson::test_load_valid_json
```

### Run with coverage report:
```bash
pytest --cov=. --cov-report=html
```

Then open `htmlcov/index.html` in your browser to see the coverage report.

### Run with verbose output:
```bash
pytest -v
```

## Test Structure

Tests are organized by the function/class they test. Test files include:
- `test_download_audio.py` - Tests for download functionality
- `test_find_podcasts.py` - Tests for podcast discovery
- `test_transcribe.py` - Tests for OpenAI transcription
- `test_local_whisper_transcribe.py` - Tests for local Whisper transcription
- `test_mp3_transcoder.py` - Tests for audio transcoding
- `test_podcast_downloader.py` - Tests for RSS podcast downloader
- `test_update_durations.py` - Tests for duration metadata updates
- `test_generate_lesson.py` - Tests for lesson generation
- `test_process_lessons.py` - Tests for batch lesson processing (now part of generate_lesson.py)
- `test_llm_providers.py` - Tests for LLM provider implementations

## Writing New Tests

When adding new functionality, add corresponding tests:
1. Create a new test class following the naming pattern `TestFunctionName`
2. Add test methods following the pattern `test_description_of_what_is_tested`
3. Use mocks for external dependencies (file I/O, network calls, subprocess)
4. Test both success and failure cases
5. Test edge cases and boundary conditions

