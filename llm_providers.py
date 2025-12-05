#!/usr/bin/env python3
"""
LLM Provider Abstraction Layer
Supports local (Ollama) and cloud (OpenAI, Anthropic) providers
"""

import os
import json
from abc import ABC, abstractmethod
from typing import Optional, Dict, List, Any
import requests
from openai import OpenAI
try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None


class LLMProvider(ABC):
    """Base class for LLM providers"""
    
    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None, 
                 temperature: float = 0.7, max_tokens: Optional[int] = None,
                 format: Optional[str] = None) -> str:
        """Generate a response from the LLM"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is available/configured"""
        pass


class OllamaProvider(LLMProvider):
    """Ollama provider for local LLM inference"""
    
    def __init__(self, model: str = "llama3.1", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        self._check_availability()
    
    def _find_model_name(self, available_models):
        """Find the actual model name, handling :latest tag"""
        # Exact match
        if self.model in available_models:
            return self.model
        
        # Try with :latest tag
        model_with_tag = f"{self.model}:latest"
        if model_with_tag in available_models:
            return model_with_tag
        
        # Try matching by prefix (e.g., "qwen2.5" matches "qwen2.5:14b")
        for model_name in available_models:
            if model_name.startswith(self.model + ":"):
                return model_name
            # Also check if model_name without tag matches
            if model_name.split(':')[0] == self.model:
                return model_name
        
        return None
    
    def _check_availability(self):
        """Check if Ollama is running"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=2)
            if response.status_code == 200:
                available_models = [m['name'] for m in response.json().get('models', [])]
                actual_model = self._find_model_name(available_models)
                if not actual_model:
                    print(f"Warning: Model '{self.model}' not found in Ollama. Available models: {available_models}")
                elif actual_model != self.model:
                    # Update to use the actual model name with tag
                    self.model = actual_model
        except requests.exceptions.RequestException:
            pass  # Will be caught in is_available()
    
    def is_available(self) -> bool:
        """Check if Ollama is running and model is available"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=2)
            if response.status_code == 200:
                available_models = [m['name'] for m in response.json().get('models', [])]
                actual_model = self._find_model_name(available_models)
                if actual_model and actual_model != self.model:
                    # Update to use the actual model name
                    self.model = actual_model
                return actual_model is not None
            return False
        except requests.exceptions.RequestException:
            return False
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None,
                 temperature: float = 0.7, max_tokens: Optional[int] = None,
                 format: Optional[str] = None) -> str:
        """Generate response using Ollama API"""
        if not self.is_available():
            raise RuntimeError(f"Ollama is not available or model '{self.model}' is not installed")
        
        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Prepare request
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
            }
        }
        
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens
        
        if format == "json_object":
            # Ollama uses "json" format
            payload["format"] = "json"
        
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=300  # 5 minute timeout for long generations
            )
            response.raise_for_status()
            result = response.json()
            return result.get("message", {}).get("content", "")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Ollama API error: {e}")


class OpenAIProvider(LLMProvider):
    """OpenAI provider (ChatGPT)"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.client = None
        if self.api_key:
            self.client = OpenAI(api_key=self.api_key)
    
    def is_available(self) -> bool:
        """Check if OpenAI API key is configured"""
        return self.client is not None
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None,
                 temperature: float = 0.7, max_tokens: Optional[int] = None,
                 format: Optional[str] = None) -> str:
        """Generate response using OpenAI API"""
        if not self.is_available():
            raise RuntimeError("OpenAI API key not configured. Set OPENAI_API_KEY environment variable.")
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
        
        if format == "json_object":
            kwargs["response_format"] = {"type": "json_object"}
        
        try:
            response = self.client.chat.completions.create(**kwargs)
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"OpenAI API error: {e}")


class AnthropicProvider(LLMProvider):
    """Anthropic provider (Claude)"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-5-sonnet-20241022"):
        if Anthropic is None:
            raise ImportError("anthropic package not installed. Install with: pip install anthropic")
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model = model
        self.client = None
        if self.api_key:
            self.client = Anthropic(api_key=self.api_key)
    
    def is_available(self) -> bool:
        """Check if Anthropic API key is configured"""
        return self.client is not None
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None,
                 temperature: float = 0.7, max_tokens: Optional[int] = None,
                 format: Optional[str] = None) -> str:
        """Generate response using Anthropic API"""
        if not self.is_available():
            raise RuntimeError("Anthropic API key not configured. Set ANTHROPIC_API_KEY environment variable.")
        
        kwargs = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
        
        if system_prompt:
            kwargs["system"] = system_prompt
        
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
        
        try:
            response = self.client.messages.create(**kwargs)
            return response.content[0].text
        except Exception as e:
            raise RuntimeError(f"Anthropic API error: {e}")


# Registry for custom LLM providers
_provider_registry: Dict[str, type] = {
    "ollama": OllamaProvider,
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
}


def register_provider(provider_name: str, provider_class: type):
    """Register a custom LLM provider
    
    Args:
        provider_name: Name of the provider (e.g., "custom_llm")
        provider_class: Class that inherits from LLMProvider
    """
    if not issubclass(provider_class, LLMProvider):
        raise TypeError(f"Provider class must inherit from LLMProvider")
    _provider_registry[provider_name] = provider_class


def get_provider(provider_type: str = "auto", **kwargs) -> LLMProvider:
    """
    Factory function to get an LLM provider
    
    Args:
        provider_type: "ollama", "openai", "anthropic", "auto", or a custom registered provider name
        **kwargs: Provider-specific arguments
    
    Returns:
        LLMProvider instance
    """
    if provider_type == "auto":
        # Try providers in order of preference (local first)
        providers = [
            ("ollama", lambda: OllamaProvider(
                model=kwargs.get("model", "llama3.1"),
                base_url=kwargs.get("base_url", "http://localhost:11434")
            )),
            ("openai", lambda: OpenAIProvider(
                api_key=kwargs.get("api_key"),
                model=kwargs.get("model", "gpt-4o-mini")
            )),
            ("anthropic", lambda: AnthropicProvider(
                api_key=kwargs.get("api_key"),
                model=kwargs.get("model", "claude-3-5-sonnet-20241022")
            )),
        ]
        
        # Add custom registered providers
        for name, provider_class in _provider_registry.items():
            if name not in ["ollama", "openai", "anthropic"]:
                providers.append((name, lambda nc=provider_class: nc(**kwargs)))
        
        for name, provider_factory in providers:
            try:
                provider = provider_factory()
                if provider.is_available():
                    print(f"Using {name} provider")
                    return provider
            except Exception as e:
                print(f"Provider {name} not available: {e}")
                continue
        
        raise RuntimeError("No LLM provider available. Please configure Ollama, OpenAI, Anthropic, or a custom provider.")
    
    elif provider_type in _provider_registry:
        provider_class = _provider_registry[provider_type]
        return provider_class(**kwargs)
    else:
        raise ValueError(f"Unknown provider type: {provider_type}. Available: {list(_provider_registry.keys())}")

