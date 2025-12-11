#!/usr/bin/env python3
"""
Test script for embeddings - verify embedding models work correctly
"""

import argparse
import requests
import json
import sys
from typing import List

def test_ollama_embeddings(model: str, base_url: str = "http://localhost:11434"):
    """Test Ollama embeddings API"""
    print(f"Testing Ollama embeddings with model: {model}")
    print(f"Base URL: {base_url}")
    print("-" * 60)
    
    # Test text (can be word, phrase, or sentence)
    test_text = "ビジネス"
    
    try:
        # Check if model exists
        print(f"\n1. Checking if model '{model}' is available...")
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        if response.status_code != 200:
            print(f"❌ Error: Cannot connect to Ollama at {base_url}")
            print(f"   Make sure Ollama is running: ollama serve")
            return False
        
        available_models = [m['name'] for m in response.json().get('models', [])]
        print(f"   Available models: {', '.join(available_models)}")
        
        if model not in available_models:
            print(f"❌ Model '{model}' not found!")
            print(f"   Install it with: ollama pull {model}")
            return False
        
        print(f"   ✅ Model '{model}' found")
        
        # Test embeddings endpoint
        print(f"\n2. Testing embeddings API...")
        print(f"   Text: '{test_text}'")
        
        response = requests.post(
            f"{base_url}/api/embeddings",
            json={
                "model": model,
                "prompt": test_text
            },
            timeout=30
        )
        
        if response.status_code != 200:
            error_text = response.text
            print(f"❌ Error: {response.status_code} {error_text}")
            
            if "does not support" in error_text.lower():
                print(f"\n   ⚠️  This model might not support embeddings.")
                print(f"   Try a different model like:")
                print(f"   - nomic-embed-text")
                print(f"   - all-minilm")
                print(f"   - mxbai-embed-large (if supported)")
            
            return False
        
        result = response.json()
        embedding = result.get("embedding", [])
        
        print(f"   ✅ Success!")
        print(f"   Embedding dimension: {len(embedding)}")
        print(f"   First 5 values: {embedding[:5]}")
        
        # Test similarity
        print(f"\n3. Testing similarity search...")
        test_texts = [
            "ビジネス",  # Single word
            "商売",  # Single word
            "仕事",  # Single word
            "オフィス",  # Single word
            "ビジネスで成功する方法",  # Sentence
            "会社で働くことについて",  # Sentence
            "完全に違う話題"  # Unrelated sentence
        ]
        
        embeddings = []
        for text in test_texts:
            response = requests.post(
                f"{base_url}/api/embeddings",
                json={"model": model, "prompt": text},
                timeout=30
            )
            if response.status_code == 200:
                embeddings.append((text, response.json()["embedding"]))
        
        if len(embeddings) >= 2:
            # Calculate cosine similarity (without numpy for simplicity)
            def cosine_similarity(a, b):
                dot_product = sum(x * y for x, y in zip(a, b))
                norm_a = sum(x * x for x in a) ** 0.5
                norm_b = sum(x * x for x in b) ** 0.5
                return dot_product / (norm_a * norm_b) if (norm_a * norm_b) > 0 else 0.0
            
            query_emb = embeddings[0][1]
            print(f"\n   Similarity scores (higher = more similar):")
            for text, emb in embeddings[1:]:
                similarity = cosine_similarity(query_emb, emb)
                print(f"   '{test_texts[0]}' vs '{text}': {similarity:.3f}")
        
        print(f"\n✅ All tests passed!")
        return True
        
    except requests.exceptions.ConnectionError:
        print(f"❌ Error: Cannot connect to Ollama at {base_url}")
        print(f"   Make sure Ollama is running: ollama serve")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def list_embedding_models(base_url: str = "http://localhost:11434"):
    """List available embedding models"""
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        if response.status_code != 200:
            print(f"❌ Cannot connect to Ollama")
            return
        
        models = response.json().get('models', [])
        print("Available models:")
        for model in models:
            name = model.get('name', 'unknown')
            print(f"  - {name}")
        
        print("\nRecommended embedding models:")
        print("  - nomic-embed-text (best for multilingual)")
        print("  - all-minilm (fast, smaller)")
        print("  - mxbai-embed-large (if supported)")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test embeddings")
    parser.add_argument("--model", default="nomic-embed-text", help="Embedding model name")
    parser.add_argument("--base-url", default="http://localhost:11434", help="Ollama base URL")
    parser.add_argument("--list", action="store_true", help="List available models")
    
    args = parser.parse_args()
    
    if args.list:
        list_embedding_models(args.base_url)
    else:
        success = test_ollama_embeddings(args.model, args.base_url)
        sys.exit(0 if success else 1)

