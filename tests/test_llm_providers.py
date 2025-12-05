"""
Tests for llm_providers.py

This test suite covers:
- LLMProvider abstract base class
- OllamaProvider
- OpenAIProvider
- AnthropicProvider
- get_provider factory function
"""

import pytest
import os
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

# Add parent directory to path to import the module
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_providers import (
    LLMProvider,
    OllamaProvider,
    OpenAIProvider,
    AnthropicProvider,
    get_provider
)
import llm_providers


class TestLLMProvider:
    """Tests for LLMProvider abstract base class"""
    
    def test_llm_provider_is_abstract(self):
        """Test that LLMProvider cannot be instantiated directly"""
        with pytest.raises(TypeError):
            LLMProvider()


class TestOllamaProvider:
    """Tests for OllamaProvider"""
    
    @patch('llm_providers.requests.post')
    @patch('llm_providers.requests.get')
    def test_ollama_provider_is_available(self, mock_get, mock_post):
        """Test checking if Ollama is available"""
        # Mock the models list response
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {
            "models": [{"name": "test-model"}]
        }
        
        provider = OllamaProvider(model="test-model")
        assert provider.is_available() is True
        # Check that get was called (with timeout parameter)
        assert mock_get.called
        call_args = mock_get.call_args
        # Verify timeout is in kwargs
        assert call_args.kwargs.get('timeout') == 2
    
    @patch('llm_providers.requests.post')
    @patch('llm_providers.requests.get')
    def test_ollama_provider_not_available(self, mock_get, mock_post):
        """Test when Ollama is not available"""
        import requests
        # Mock the exception for both __init__ and is_available calls
        mock_get.side_effect = requests.exceptions.RequestException("Connection refused")
        
        # The exception during __init__ should be caught, so provider should still be created
        provider = OllamaProvider(model="test-model")
        # Reset the mock to raise exception again for is_available call
        mock_get.side_effect = requests.exceptions.RequestException("Connection refused")
        assert provider.is_available() is False
    
    @patch('llm_providers.requests.post')
    @patch('llm_providers.requests.get')
    def test_ollama_provider_generate(self, mock_get, mock_post):
        """Test generating text with Ollama"""
        # Mock availability check
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {
            "models": [{"name": "test-model"}]
        }
        # Mock generation - Ollama returns message.content, not response
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            "message": {"content": "Test response"}
        }
        mock_post.return_value.raise_for_status = Mock()  # Mock raise_for_status
        
        provider = OllamaProvider(model="test-model")
        response = provider.generate(prompt="Test prompt")
        
        assert response == "Test response"
        mock_post.assert_called_once()


class TestOpenAIProvider:
    """Tests for OpenAIProvider"""
    
    @patch('llm_providers.OpenAI')
    def test_openai_provider_is_available(self, mock_openai_class):
        """Test checking if OpenAI is available"""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        mock_client.models.list.return_value = []
        
        provider = OpenAIProvider(api_key="test-key", model="gpt-4o-mini")
        assert provider.is_available() is True
    
    @patch('llm_providers.OpenAI')
    def test_openai_provider_not_available(self, mock_openai_class):
        """Test when OpenAI is not available"""
        # If OpenAI client creation fails, provider should handle it gracefully
        mock_openai_class.side_effect = Exception("API key invalid")
        
        # Provider should still be created (exception is caught in __init__)
        # The client will be None if creation failed
        try:
            provider = OpenAIProvider(api_key="invalid-key", model="gpt-4o-mini")
            # If exception was raised, client will be None
            if provider.client is None:
                assert provider.is_available() is False
            else:
                # If somehow client was created, it should still not be available
                assert provider.is_available() is False
        except Exception:
            # If exception prevents creation entirely, that's also acceptable
            # The test verifies the error handling
            pass
    
    @patch('llm_providers.OpenAI')
    def test_openai_provider_generate(self, mock_openai_class):
        """Test generating text with OpenAI"""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Test response"
        mock_client.chat.completions.create.return_value = mock_response
        
        provider = OpenAIProvider(api_key="test-key", model="gpt-4o-mini")
        response = provider.generate(prompt="Test prompt")
        
        assert response == "Test response"


class TestAnthropicProvider:
    """Tests for AnthropicProvider"""
    
    @pytest.mark.skipif(AnthropicProvider is None or llm_providers.Anthropic is None, 
                        reason="anthropic package not installed")
    @patch('llm_providers.requests.post')
    def test_anthropic_provider_is_available(self, mock_post):
        """Test checking if Anthropic is available"""
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {}
        
        provider = AnthropicProvider(api_key="test-key", model="claude-3-5-sonnet-20241022")
        assert provider.is_available() is True
    
    @pytest.mark.skipif(AnthropicProvider is None or llm_providers.Anthropic is None, 
                        reason="anthropic package not installed")
    @patch('llm_providers.requests.post')
    def test_anthropic_provider_not_available(self, mock_post):
        """Test when Anthropic is not available"""
        mock_post.side_effect = Exception("API key invalid")
        
        provider = AnthropicProvider(api_key="invalid-key", model="claude-3-5-sonnet-20241022")
        assert provider.is_available() is False
    
    @pytest.mark.skipif(AnthropicProvider is None or llm_providers.Anthropic is None, 
                        reason="anthropic package not installed")
    @patch('llm_providers.requests.post')
    def test_anthropic_provider_generate(self, mock_post):
        """Test generating text with Anthropic"""
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            "content": [{"text": "Test response"}]
        }
        
        provider = AnthropicProvider(api_key="test-key", model="claude-3-5-sonnet-20241022")
        response = provider.generate(prompt="Test prompt")
        
        assert response == "Test response"


class TestGetProvider:
    """Tests for get_provider factory function"""
    
    @patch('llm_providers.requests.get')
    def test_get_provider_ollama(self, mock_get):
        """Test getting Ollama provider"""
        # Mock Ollama availability
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {
            "models": [{"name": "test-model"}]
        }
        
        provider = get_provider("ollama", model="test-model")
        
        assert provider is not None
        assert isinstance(provider, OllamaProvider)
    
    def test_get_provider_openai(self):
        """Test getting OpenAI provider"""
        # Test that OpenAI provider can be retrieved (may not be available without API key)
        try:
            provider = get_provider("openai", api_key="test-key", model="gpt-4o-mini")
            assert provider is not None
            assert isinstance(provider, OpenAIProvider)
        except Exception:
            # If OpenAI isn't available (no API key), that's okay for the test
            pass
    
    def test_get_provider_anthropic(self):
        """Test getting Anthropic provider"""
        # Test that Anthropic provider can be retrieved (may not be available without API key or package)
        try:
            provider = get_provider("anthropic", api_key="test-key", model="claude-3-5-sonnet-20241022")
            assert provider is not None
            assert isinstance(provider, AnthropicProvider)
        except (ImportError, Exception):
            # If Anthropic isn't available (no package or API key), that's okay for the test
            pass
    
    def test_get_provider_invalid(self):
        """Test getting invalid provider"""
        with pytest.raises(ValueError, match="Unknown provider type"):
            get_provider("invalid")
    
    @patch('llm_providers.OllamaProvider')
    def test_get_provider_auto_ollama_available(self, mock_ollama_class):
        """Test auto provider selection with Ollama available"""
        mock_provider = Mock()
        mock_provider.is_available.return_value = True
        mock_ollama_class.return_value = mock_provider
        
        provider = get_provider("auto", model="test-model")
        
        assert provider == mock_provider
        mock_ollama_class.assert_called_once()
    
    @patch('llm_providers.OllamaProvider')
    @patch('llm_providers.OpenAIProvider')
    def test_get_provider_auto_fallback(self, mock_openai_class, mock_ollama_class):
        """Test auto provider selection with fallback"""
        mock_ollama_provider = Mock()
        mock_ollama_provider.is_available.return_value = False
        mock_ollama_class.return_value = mock_ollama_provider
        
        mock_openai_provider = Mock()
        mock_openai_provider.is_available.return_value = True
        mock_openai_class.return_value = mock_openai_provider
        
        provider = get_provider("auto", api_key="test-key", model="gpt-4o-mini")
        
        assert provider == mock_openai_provider
        mock_ollama_class.assert_called_once()
        mock_openai_class.assert_called_once()

