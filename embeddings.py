#!/usr/bin/env python3
"""
Embeddings Module - Handle Ollama embeddings for semantic search
"""

import requests
from typing import List, Callable, Optional
import logging

logger = logging.getLogger(__name__)


def get_embedding(text: str, model: str = "mxbai-embed-large", 
                  base_url: str = "http://localhost:11434") -> List[float]:
    """
    Get embedding vector for text using Ollama.
    
    Args:
        text: Text to embed
        model: Embedding model name (default: mxbai-embed-large)
        base_url: Ollama base URL (default: http://localhost:11434)
    
    Returns:
        Embedding vector (list of floats, 1024 dimensions for mxbai-embed-large)
    
    Raises:
        RuntimeError: If Ollama is not available or model not found
        requests.RequestException: If API request fails
    """
    try:
        response = requests.post(
            f"{base_url}/api/embeddings",
            json={"model": model, "prompt": text},
            timeout=30
        )
        response.raise_for_status()
        result = response.json()
        
        if "embedding" not in result:
            raise RuntimeError(f"Invalid response from Ollama: {result}")
        
        return result["embedding"]
    
    except requests.exceptions.ConnectionError:
        raise RuntimeError(f"Cannot connect to Ollama at {base_url}. Make sure Ollama is running.")
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 400:
            error_text = e.response.text
            if "does not support" in error_text.lower():
                raise RuntimeError(f"Model '{model}' does not support embeddings. Try 'nomic-embed-text' or 'mxbai-embed-large'.")
        raise RuntimeError(f"Ollama API error: {e}")
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Error getting embedding from Ollama: {e}")


def create_ollama_embedding_function(model: str = "mxbai-embed-large", 
                                     base_url: str = "http://localhost:11434") -> Callable:
    """
    Create custom embedding function for Chroma that uses Ollama.
    
    Args:
        model: Embedding model name
        base_url: Ollama base URL
    
    Returns:
        Callable function that takes a list of texts and returns a list of embeddings
        Format: List[str] -> List[List[float]]
    
    Example:
        embedding_fn = create_ollama_embedding_function()
        embeddings = embedding_fn(["ビジネス", "商売"])
        # Returns: [[-0.478, ...], [-0.426, ...]]
    """
    def embed_texts(texts: List[str]) -> List[List[float]]:
        """
        Embed a list of texts using Ollama.
        
        Args:
            texts: List of text strings to embed
        
        Returns:
            List of embedding vectors (each is a list of floats)
        """
        embeddings = []
        for text in texts:
            try:
                embedding = get_embedding(text, model=model, base_url=base_url)
                embeddings.append(embedding)
            except Exception as e:
                logger.error(f"Error embedding text '{text[:50]}...': {e}")
                # Return zero vector as fallback (same dimension as model)
                # mxbai-embed-large has 1024 dimensions
                embeddings.append([0.0] * 1024)
        
        return embeddings
    
    return embed_texts

