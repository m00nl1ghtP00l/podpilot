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
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {}
        
        provider = OllamaProvider(model="test-model")
        assert provider.is_available() is True
        mock_get.assert_called_once()
    
    @patch('llm_providers.requests.post')
    @patch('llm_providers.requests.get')
    def test_ollama_provider_not_available(self, mock_get, mock_post):
        """Test when Ollama is not available"""
        mock_get.side_effect = Exception("Connection refused")
        
        provider = OllamaProvider(model="test-model")
        assert provider.is_available() is False
    
    @patch('llm_providers.requests.post')
    @patch('llm_providers.requests.get')
    def test_ollama_provider_generate(self, mock_get, mock_post):
        """Test generating text with Ollama"""
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {}
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            "response": "Test response"
        }
        
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
        mock_openai_class.side_effect = Exception("API key invalid")
        
        provider = OpenAIProvider(api_key="invalid-key", model="gpt-4o-mini")
        assert provider.is_available() is False
    
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
    
    @patch('llm_providers.requests.post')
    def test_anthropic_provider_is_available(self, mock_post):
        """Test checking if Anthropic is available"""
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {}
        
        provider = AnthropicProvider(api_key="test-key", model="claude-3-5-sonnet-20241022")
        assert provider.is_available() is True
    
    @patch('llm_providers.requests.post')
    def test_anthropic_provider_not_available(self, mock_post):
        """Test when Anthropic is not available"""
        mock_post.side_effect = Exception("API key invalid")
        
        provider = AnthropicProvider(api_key="invalid-key", model="claude-3-5-sonnet-20241022")
        assert provider.is_available() is False
    
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
    
    @patch('llm_providers.OllamaProvider')
    def test_get_provider_ollama(self, mock_ollama_class):
        """Test getting Ollama provider"""
        mock_provider = Mock()
        mock_provider.is_available.return_value = True
        mock_ollama_class.return_value = mock_provider
        
        provider = get_provider("ollama", model="test-model")
        
        assert provider == mock_provider
        mock_ollama_class.assert_called_once()
    
    @patch('llm_providers.OpenAIProvider')
    def test_get_provider_openai(self, mock_openai_class):
        """Test getting OpenAI provider"""
        mock_provider = Mock()
        mock_provider.is_available.return_value = True
        mock_openai_class.return_value = mock_provider
        
        provider = get_provider("openai", api_key="test-key", model="gpt-4o-mini")
        
        assert provider == mock_provider
        mock_openai_class.assert_called_once()
    
    @patch('llm_providers.AnthropicProvider')
    def test_get_provider_anthropic(self, mock_anthropic_class):
        """Test getting Anthropic provider"""
        mock_provider = Mock()
        mock_provider.is_available.return_value = True
        mock_anthropic_class.return_value = mock_provider
        
        provider = get_provider("anthropic", api_key="test-key", model="claude-3-5-sonnet-20241022")
        
        assert provider == mock_provider
        mock_anthropic_class.assert_called_once()
    
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

