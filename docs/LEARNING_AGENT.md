# Learning Agent Documentation

## Overview

The Learning Agent is an AI-powered orchestrator that automates the entire podpilot pipeline to create personalized learning packs. It uses LLM to understand user queries and automatically executes the necessary tools.

## Architecture

```
User Query
    ↓
LLM Planning (determines which tools to use)
    ↓
ToolExecutor (executes Python tools programmatically)
    ├── find_episodes() → channel_fetcher.py
    ├── download_audio() → download_audio.py
    ├── transcribe_audio() → local_whisper_transcribe.py
    └── generate_lessons() → generate_lesson.py
    ↓
Lesson Selection (LLM selects relevant lessons)
    ↓
Learning Pack Generation (LLM creates comprehensive pack)
    ↓
Markdown Output File
```

## How It Works

### 1. Planning Phase

The agent uses LLM to analyze the user query and create an execution plan:

```python
# Example query: "I want to learn business Japanese from recent episodes"

Plan:
{
  "steps": [
    {"action": "find_episodes", "channel": "hnh", "keywords": ["business"]},
    {"action": "download_audio", "channel": "hnh"},
    {"action": "transcribe_audio"},
    {"action": "generate_lessons"},
    {"action": "create_learning_pack"}
  ]
}
```

### 2. Execution Phase

The `ToolExecutor` class executes each step:

```python
executor = ToolExecutor(config_data, provider)

# Find episodes
episodes = executor.find_episodes("hnh", keywords=["business"])

# Download audio
audio_files = executor.download_audio("hnh", episodes)

# Transcribe
transcriptions = executor.transcribe_audio(audio_files)

# Generate lessons
lessons = executor.generate_lessons(transcriptions)
```

### 3. Learning Pack Creation

The agent:
1. Selects relevant lessons based on the query
2. Extracts vocabulary, grammar, and key phrases
3. Uses LLM to generate a comprehensive learning pack

## Usage Examples

### Full Pipeline (Automatic)

```bash
# The agent will find, download, transcribe, and generate everything
python learning_agent.py "I want to learn business Japanese from recent episodes" --name hnh
```

### Use Existing Lessons Only

```bash
# Skip tool execution, only create pack from existing lessons
python learning_agent.py "I want to practice N3 grammar" --skip-tools
```

### With Date Filters

```bash
# Only process episodes from specific date range
python learning_agent.py "Recent travel vocabulary" --from-date 2024-01-01 --name sjn
```

### Force Re-execution

```bash
# Force re-download, re-transcribe, and re-generate
python learning_agent.py "Business Japanese basics" --force
```

## ToolExecutor Methods

### `find_episodes(channel_name, from_date, to_date, keywords)`

Uses `channel_fetcher.py` functions to find podcast episodes:
- Fetches RSS feed
- Parses episodes
- Filters by date and keywords

### `download_audio(channel_name, episodes, force)`

Uses `download_audio.py` functions to download audio files:
- Checks if files already exist
- Downloads missing files
- Handles transcoding automatically

### `transcribe_audio(audio_files, force)`

Uses `local_whisper_transcribe.py` functions to transcribe:
- Checks for existing transcriptions
- Runs whisper.cpp
- Creates clean transcript files

### `generate_lessons(transcription_files, force)`

Uses `generate_lesson.py` functions to generate lessons:
- Loads transcriptions
- Calls LLM to generate lessons
- Saves markdown lesson files

## Learning Pack Structure

The generated learning pack includes:

1. **Introduction** - Overview of what will be learned
2. **Learning Objectives** - Clear goals
3. **Vocabulary Focus** - Curated vocabulary by JLPT level/topic
4. **Grammar Focus** - Key grammar points with explanations
5. **Practice Exercises** - Fill-in-the-blank, multiple choice, etc.
6. **Review Section** - Summary of key points
7. **Next Steps** - Suggestions for further study

## Configuration

The agent uses your existing `config/podcasts.json`:
- LLM provider settings (Ollama, OpenAI, Anthropic)
- Data root directory
- Channel configurations
- Transcription settings

## Error Handling

- If planning fails, falls back to searching existing lessons
- If tools fail, continues with available data
- Skips steps if files already exist (unless `--force`)

## Integration

The agent integrates seamlessly with all existing podpilot tools:
- Uses same LLM providers
- Respects same configuration
- Follows same file structure
- Compatible with all existing workflows

