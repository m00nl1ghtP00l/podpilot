#!/usr/bin/env python3
"""
Example: Semantic Search Implementation
Shows how to calculate cosine similarity and find similar lessons
"""

import requests
import json
from typing import List, Tuple, Dict
from pathlib import Path


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Formula: similarity = (A · B) / (||A|| × ||B||)
    
    Args:
        vec_a: First vector (embedding)
        vec_b: Second vector (embedding)
    
    Returns:
        Similarity score between -1.0 and 1.0 (higher = more similar)
    """
    # Step 1: Calculate dot product (A · B)
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    
    # Step 2: Calculate magnitudes (||A|| and ||B||)
    magnitude_a = sum(a * a for a in vec_a) ** 0.5
    magnitude_b = sum(b * b for b in vec_b) ** 0.5
    
    # Step 3: Calculate similarity
    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0
    
    similarity = dot_product / (magnitude_a * magnitude_b)
    return similarity


def get_embedding(text: str, model: str = "mxbai-embed-large", 
                  base_url: str = "http://localhost:11434") -> List[float]:
    """
    Get embedding vector for text using Ollama.
    
    Args:
        text: Text to embed
        model: Embedding model name
        base_url: Ollama base URL
    
    Returns:
        Embedding vector (list of floats)
    """
    response = requests.post(
        f"{base_url}/api/embeddings",
        json={"model": model, "prompt": text},
        timeout=30
    )
    response.raise_for_status()
    return response.json()["embedding"]


def search_semantic(query: str, documents: List[Dict], 
                   model: str = "mxbai-embed-large",
                   top_k: int = 5,
                   similarity_threshold: float = 0.5) -> List[Tuple[Dict, float]]:
    """
    Perform semantic search on documents.
    
    Args:
        query: Search query text
        documents: List of documents, each with 'text' and optionally 'id', 'title', etc.
        model: Embedding model name
        top_k: Number of top results to return
        similarity_threshold: Minimum similarity score (0.0 to 1.0)
    
    Returns:
        List of (document, similarity_score) tuples, sorted by similarity (highest first)
    
    Example:
        documents = [
            {"id": 1, "text": "ビジネスで成功する方法", "title": "Business Success"},
            {"id": 2, "text": "猫の飼い方", "title": "Cat Care"},
        ]
        results = search_semantic("ビジネス", documents)
        # Returns: [({"id": 1, ...}, 0.85), ({"id": 2, ...}, 0.12)]
    """
    # Step 1: Get query embedding
    print(f"Getting embedding for query: '{query}'")
    query_embedding = get_embedding(query, model)
    
    # Step 2: Get embeddings for all documents
    print(f"Getting embeddings for {len(documents)} documents...")
    results = []
    
    for doc in documents:
        # Get embedding for document text
        doc_embedding = get_embedding(doc["text"], model)
        
        # Calculate similarity
        similarity = cosine_similarity(query_embedding, doc_embedding)
        
        # Filter by threshold
        if similarity >= similarity_threshold:
            results.append((doc, similarity))
    
    # Step 3: Sort by similarity (highest first)
    results.sort(key=lambda x: x[1], reverse=True)
    
    # Step 4: Return top_k results
    return results[:top_k]


def search_lessons_semantic(query: str, lessons: List[Dict],
                           model: str = "mxbai-embed-large",
                           top_k: int = 10) -> List[Tuple[Dict, float]]:
    """
    Semantic search specifically for lesson files.
    
    Args:
        query: Search query (e.g., "ビジネス", "business vocabulary")
        lessons: List of lesson dictionaries with 'summary', 'title', etc.
        model: Embedding model name
        top_k: Number of top results
    
    Returns:
        List of (lesson, similarity_score) tuples
    """
    # Create searchable text from lesson (summary + vocabulary)
    documents = []
    for lesson in lessons:
        # Combine summary and key vocabulary for better search
        searchable_text = lesson.get("summary", "")
        if "vocabulary" in lesson:
            vocab_text = " ".join([v.get("word", "") for v in lesson["vocabulary"][:10]])
            searchable_text += " " + vocab_text
        
        documents.append({
            "id": lesson.get("id"),
            "title": lesson.get("title", ""),
            "text": searchable_text,
            "lesson": lesson  # Keep full lesson data
        })
    
    return search_semantic(query, documents, model, top_k)


# Example usage
if __name__ == "__main__":
    # Example 1: Simple word similarity
    print("=" * 70)
    print("Example 1: Word Similarity")
    print("=" * 70)
    
    word1 = "ビジネス"
    word2 = "商売"
    
    emb1 = get_embedding(word1)
    emb2 = get_embedding(word2)
    
    similarity = cosine_similarity(emb1, emb2)
    print(f"Similarity between '{word1}' and '{word2}': {similarity:.3f}")
    print()
    
    # Example 2: Document search
    print("=" * 70)
    print("Example 2: Document Search")
    print("=" * 70)
    
    documents = [
        {"id": 1, "text": "ビジネスで成功する方法について話しています。", "title": "Business Success"},
        {"id": 2, "text": "会社で働くことについて説明しています。", "title": "Working at Company"},
        {"id": 3, "text": "猫の飼い方と世話の仕方について", "title": "Cat Care"},
    ]
    
    query = "ビジネス"
    results = search_semantic(query, documents, top_k=3)
    
    print(f"\nSearch results for '{query}':")
    for doc, score in results:
        print(f"  [{score:.3f}] {doc['title']}: {doc['text'][:50]}...")

