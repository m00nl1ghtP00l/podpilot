"""
Language and LLM Adapter System

This module provides a plugin system for:
- Language-specific adapters (prompts, segmentation, character handling)
- Custom LLM provider implementations

Users can create their own adapters by:
1. Creating a language adapter class inheriting from LanguageAdapter
2. Creating an LLM provider class inheriting from LLMProvider (in llm_providers.py)
3. Registering them via config or entry points
"""

from .base import LanguageAdapter, get_language_adapter, register_language_adapter
from .japanese import JapaneseAdapter

# Register built-in adapters
register_language_adapter("ja", JapaneseAdapter())
register_language_adapter("japanese", JapaneseAdapter())

__all__ = [
    "LanguageAdapter",
    "get_language_adapter",
    "register_language_adapter",
    "JapaneseAdapter",
]

