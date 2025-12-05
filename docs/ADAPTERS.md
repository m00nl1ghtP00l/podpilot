# Creating Custom Language and LLM Adapters

Podpilot is designed to be extensible. You can create custom adapters for:
- **Language adapters**: Support for new languages (prompts, segmentation, character handling)
- **LLM providers**: Support for new LLM services or APIs

## Language Adapters

Language adapters provide language-specific functionality:
- Transcription prompts
- Lesson generation prompts and schemas
- Text segmentation rules
- Character handling (preserving special characters)
- Proficiency level systems (e.g., JLPT for Japanese, CEFR for European languages)

### Creating a Language Adapter

1. Create a new file in the `adapters/` directory (e.g., `adapters/spanish.py`)

2. Inherit from `LanguageAdapter` and implement required methods:

```python
from adapters.base import LanguageAdapter
from typing import List

class SpanishAdapter(LanguageAdapter):
    @property
    def language_code(self) -> str:
        return "es"
    
    @property
    def language_name(self) -> str:
        return "Spanish"
    
    def get_transcription_prompt(self) -> str:
        return "Este audio está en español. Por favor transcríbelo con la mayor precisión posible."
    
    def get_lesson_system_prompt(self) -> str:
        return """You are an expert Spanish language teacher..."""
    
    def get_lesson_user_prompt_template(self) -> str:
        return """Analyze the following Spanish text..."""
    
    def segment_text(self, text: str) -> List[str]:
        # Split Spanish text into sentences
        # Spanish uses ., ?, ! as sentence endings
        import re
        sentences = re.split(r'([.!?]+)\s+', text)
        # ... process and return list of sentences
        return sentences
    
    def clean_title(self, title: str) -> str:
        # Clean title while preserving Spanish characters (ñ, á, é, etc.)
        # Remove invalid filename characters
        invalid_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
        cleaned = title
        for char in invalid_chars:
            cleaned = cleaned.replace(char, '_')
        return cleaned.strip('_')
    
    def get_proficiency_levels(self) -> List[str]:
        return ["A1", "A2", "B1", "B2", "C1", "C2"]  # CEFR levels
```

3. Register your adapter in `adapters/__init__.py`:

```python
from .spanish import SpanishAdapter

register_language_adapter("es", SpanishAdapter())
register_language_adapter("spanish", SpanishAdapter())
```

4. Use it in your config:

```json
{
  "language": "es",
  "analysis": { ... },
  "transcription": { ... }
}
```

### Example: English Adapter

See `adapters/example_english.py` for a complete example that you can copy and modify.

## LLM Provider Adapters

To add support for a new LLM provider:

1. Create a class inheriting from `LLMProvider` in `llm_providers.py` or a separate file:

```python
from llm_providers import LLMProvider

class CustomLLMProvider(LLMProvider):
    def __init__(self, api_key: str, model: str = "custom-model"):
        self.api_key = api_key
        self.model = model
    
    def is_available(self) -> bool:
        # Check if provider is configured and available
        return self.api_key is not None
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None,
                 temperature: float = 0.7, max_tokens: Optional[int] = None,
                 format: Optional[str] = None) -> str:
        # Implement API call to your LLM service
        # ...
        return response_text
```

2. Register it in `llm_providers.py`'s `get_provider()` function or use a plugin system.

3. Use it in your config:

```json
{
  "analysis": {
    "provider": "custom",
    "model": "your-model-name",
    "api_key": "${CUSTOM_API_KEY}"
  }
}
```

## Configuration

Add language selection to your `config/podcasts.json`:

```json
{
  "language": "ja",
  "analysis": { ... },
  "transcription": {
    "provider": "whisper.cpp",
    "language": "ja"
  },
  "youtube_channels": [ ... ]
}
```

The `language` field determines which adapter to use for:
- Lesson generation prompts
- Text segmentation
- Title cleaning
- Proficiency levels

## Best Practices

1. **Preserve special characters**: When cleaning titles, preserve language-specific characters (e.g., Japanese Kanji, Spanish accents)

2. **Use standard language codes**: Use ISO 639-1 codes (e.g., 'ja', 'en', 'es', 'fr')

3. **Follow existing patterns**: Look at `adapters/japanese.py` for a complete reference implementation

4. **Test your adapter**: Create tests in `tests/test_adapters.py` following the pattern of existing tests

5. **Document proficiency levels**: Clearly document what each proficiency level means for your language

## Contributing Adapters

If you create a useful adapter, consider contributing it back to the project:
1. Follow the existing code style
2. Add tests
3. Update documentation
4. Submit a pull request

