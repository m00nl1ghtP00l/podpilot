Config overview
==============

This project uses two top-level blocks to keep roles clear:

- analysis: LLM used for lesson generation/summaries (Ollama/OpenAI/Anthropic).
- transcription: audio → text (whisper.cpp locally or OpenAI Whisper).

Example
-------
```json
{
  "analysis": {
    "provider": "ollama",
    "model": "qwen2.5:14b"
  },
  "transcription": {
    "provider": "whisper.cpp",
    "model_path": "${WHISPER_MODEL_PATH}"
  },
  "youtube_channels": [ ... ]
}
```

Switching providers
-------------------
- analysis.provider: ollama | openai | anthropic
  - model: e.g., qwen2.5:14b (ollama), gpt-4o (openai), claude-3.5-sonnet (anthropic)
- transcription.provider: whisper.cpp | openai
  - whisper.cpp: set model_path (supports env vars: ${WHISPER_MODEL_PATH} or $WHISPER_MODEL_PATH)
  - openai: set openai_model (e.g., whisper-1); requires OPENAI_API_KEY

Precedence
----------
Config is the source of truth. CLI/env can override if you choose to keep them, but the recommended path is to set providers/models here.***

