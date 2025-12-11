# Semantic Search Implementation Guide

## Overview

This guide explains how to add semantic search to podpilot for finding lessons and vocabulary by meaning, not just exact keywords.

## Why Semantic Search?

**Current limitation:** Keyword search only finds exact matches. Searching for "business" won't find lessons about "commerce" or "workplace".

**With semantic search:** You can find related content even with different wording.

## Recommended Embeddings Models for Japanese

### Option 1: Ollama Embeddings (Easiest - No Extra Dependencies)
**Models:**
- `nomic-embed-text` - Good multilingual support, works with Ollama (8192 token limit)
- `mxbai-embed-large` - Multilingual, good for Japanese (**512 token limit**)

**Token Limits:**
- **mxbai-embed-large**: Maximum **512 tokens** per prompt
  - ~200-400 Japanese characters (depending on kanji/hiragana mix)
  - Text exceeding this limit will be truncated automatically
- **nomic-embed-text**: Maximum **8192 tokens** per prompt
  - Much longer text support (~3000-6000 Japanese characters)

**Pros:**
- Already using Ollama
- No additional dependencies
- Local, private

**Cons:**
- Requires Ollama to support embeddings API (check version)
- May be slower than specialized models
- **mxbai-embed-large has short context (512 tokens)** - use for sentences/phrases, not long paragraphs

**Usage:**
```python
# Check if Ollama supports embeddings
curl http://localhost:11434/api/embeddings -d '{
  "model": "nomic-embed-text",
  "prompt": "ビジネス"
}'
```

### Option 2: Sentence-Transformers (Best for Japanese)
**Models:**
- `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` - Excellent for Japanese
- `intfloat/multilingual-e5-large` - State-of-the-art multilingual
- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` - Faster, smaller

**Pros:**
- Excellent Japanese support
- Fast inference
- Well-maintained

**Cons:**
- Requires `sentence-transformers` library
- Model download (~500MB-2GB)

**Installation:**
```bash
pip install sentence-transformers
```

### Option 3: OpenAI Embeddings (Cloud)
**Model:** `text-embedding-3-small` or `text-embedding-3-large`

**Pros:**
- High quality
- No local model needed

**Cons:**
- Requires API key
- Costs money
- Data sent to OpenAI

## Implementation Architecture

### Step 1: Create Embeddings Module

Create `embeddings.py`:

```python
#!/usr/bin/env python3
"""
Embeddings module for semantic search
Supports multiple backends: Ollama, sentence-transformers, OpenAI
"""

from typing import List, Optional
import numpy as np
from pathlib import Path
import json

class EmbeddingsProvider:
    """Base class for embeddings providers"""
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding vector for text"""
        raise NotImplementedError
    
    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts (more efficient)"""
        return [self.get_embedding(text) for text in texts]


class OllamaEmbeddings(EmbeddingsProvider):
    """Ollama embeddings provider"""
    
    def __init__(self, model: str = "nomic-embed-text", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
    
    def get_embedding(self, text: str) -> List[float]:
        import requests
        response = requests.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self.model, "prompt": text},
            timeout=30
        )
        response.raise_for_status()
        return response.json()["embedding"]


class SentenceTransformersEmbeddings(EmbeddingsProvider):
    """Sentence-transformers embeddings provider"""
    
    def __init__(self, model_name: str = "paraphrase-multilingual-mpnet-base-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
    
    def get_embedding(self, text: str) -> List[float]:
        return self.model.encode(text).tolist()
    
    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts).tolist()


class OpenAIEmbeddings(EmbeddingsProvider):
    """OpenAI embeddings provider"""
    
    def __init__(self, model: str = "text-embedding-3-small", api_key: Optional[str] = None):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        self.model = model
    
    def get_embedding(self, text: str) -> List[float]:
        response = self.client.embeddings.create(
            model=self.model,
            input=text
        )
        return response.data[0].embedding
```

### Step 2: Add Semantic Search to lesson_search.py

```python
# Add to lesson_search.py

def search_lessons_semantic(
    lessons: List[LessonContent], 
    query: str,
    embeddings_provider: EmbeddingsProvider,
    top_k: int = 10,
    similarity_threshold: float = 0.5
) -> List[Tuple[LessonContent, float]]:
    """Search lessons using semantic similarity
    
    Returns:
        List of (lesson, similarity_score) tuples, sorted by similarity
    """
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Get query embedding
    query_embedding = np.array(embeddings_provider.get_embedding(query)).reshape(1, -1)
    
    # Get embeddings for all lessons (cache these!)
    lesson_embeddings = []
    for lesson in lessons:
        # Create searchable text from lesson
        searchable_text = f"{lesson.summary} {' '.join([v.get('word', '') for v in lesson.vocabulary])}"
        embedding = np.array(embeddings_provider.get_embedding(searchable_text))
        lesson_embeddings.append(embedding)
    
    # Calculate similarities
    similarities = cosine_similarity(query_embedding, np.array(lesson_embeddings))[0]
    
    # Filter and sort
    results = [
        (lesson, float(score))
        for lesson, score in zip(lessons, similarities)
        if score >= similarity_threshold
    ]
    results.sort(key=lambda x: x[1], reverse=True)
    
    return results[:top_k]
```

### Step 3: Cache Embeddings (Important!)

Don't recompute embeddings every time. Cache them:

```python
def get_or_compute_embedding(
    text: str,
    embeddings_provider: EmbeddingsProvider,
    cache_dir: Path
) -> List[float]:
    """Get embedding from cache or compute and cache it"""
    import hashlib
    import pickle
    
    # Create hash of text for cache key
    text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
    cache_file = cache_dir / f"{text_hash}.pkl"
    
    if cache_file.exists():
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    # Compute and cache
    embedding = embeddings_provider.get_embedding(text)
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_file, 'wb') as f:
        pickle.dump(embedding, f)
    
    return embedding
```

### Step 4: Integration Points

**In `conversational_agent.py`:**
```python
# When user asks for lessons, use semantic search instead of keyword search
if use_semantic_search:
    results = search_lessons_semantic(lessons, user_query, embeddings_provider)
else:
    results = search_lessons_by_keywords(lessons, keywords)
```

**In `learning_agent.py`:**
```python
# When selecting relevant lessons for learning pack
selected_lessons = search_lessons_semantic(lessons, args.query, embeddings_provider, top_k=10)
```

## Quick Start Example

```python
# 1. Choose provider
from embeddings import SentenceTransformersEmbeddings
embeddings = SentenceTransformersEmbeddings("paraphrase-multilingual-mpnet-base-v2")

# 2. Search semantically
results = search_lessons_semantic(lessons, "ビジネス", embeddings, top_k=5)

# 3. Results are sorted by similarity
for lesson, score in results:
    print(f"Similarity: {score:.2f} - {lesson.episode_title}")
```

## Configuration

Add to `config/podcasts.json`:

```json
{
  "embeddings": {
    "provider": "sentence-transformers",
    "model": "paraphrase-multilingual-mpnet-base-v2",
    "enabled": false,
    "cache_dir": ".embeddings_cache"
  }
}
```

## Text Length Limits

### Token Limits by Model

| Model | Max Tokens | Approx. Japanese Chars | Best For |
|-------|------------|------------------------|----------|
| **mxbai-embed-large** | 512 | ~200-400 | Sentences, short paragraphs |
| **nomic-embed-text** | 8192 | ~3000-6000 | Long paragraphs, full summaries |
| **paraphrase-multilingual-mpnet-base-v2** | 128 | ~50-100 | Single sentences |
| **multilingual-e5-large** | 512 | ~200-400 | Sentences, short paragraphs |

**Important:** 
- Tokens ≠ characters. Japanese text uses ~1-2 tokens per character (kanji = more tokens)
- If text exceeds limit, it will be truncated (first N tokens used)
- For long lesson summaries, consider:
  1. Split into chunks (e.g., by paragraph)
  2. Use a model with higher limit (nomic-embed-text)
  3. Embed key sections separately (summary, vocabulary, grammar)

### Handling Long Text

```python
def get_embedding_for_long_text(text: str, max_tokens: int = 512):
    """Handle long text by truncating or chunking"""
    # Option 1: Simple truncation (first N tokens)
    # Most embedding APIs handle this automatically
    
    # Option 2: Smart chunking (for very long text)
    if len(text) > max_tokens * 2:  # Rough estimate
        # Split by sentences or paragraphs
        chunks = text.split('。')[:5]  # First 5 sentences
        text = '。'.join(chunks)
    
    return get_embedding(text)
```

## Performance Considerations

1. **Cache embeddings** - Don't recompute for same text
2. **Batch processing** - Process multiple texts at once when possible
3. **Index pre-computation** - Pre-compute embeddings for all lessons on first run
4. **Similarity threshold** - Filter out low-similarity results (default 0.5)
5. **Text length** - Keep text within model limits (truncate long summaries if needed)

## Testing

### Step 1: Test Embeddings API (Quick Test)

**Using curl (works immediately):**
```bash
curl http://localhost:11434/api/embeddings -d '{
  "model": "mxbai-embed-large",
  "prompt": "ビジネス"
}'
```

You should get back a JSON response with an `embedding` array (vector of numbers).

**Using Python test script:**
```bash
# Install dependencies if needed
pip install requests

# Run the test
python test_embeddings.py --model mxbai-embed-large

# Or test with a different model
python test_embeddings.py --model nomic-embed-text

# List available models
python test_embeddings.py --list
```

The test script will:
1. ✅ Verify Ollama is running
2. ✅ Check if the model is available
3. ✅ Test embeddings generation
4. ✅ Test similarity between related words (ビジネス vs 商売 vs 仕事)

### Step 2: Test Semantic Search (After Implementation)

```python
# Test semantic search
lessons = get_all_lessons(data_root)
results = search_lessons_semantic(lessons, "ビジネス", embeddings_provider)

# Should find lessons about:
# - ビジネス (business)
# - 商売 (commerce)  
# - 仕事 (work)
# - オフィス (office)
# Even if they don't contain the exact word "ビジネス"
```

### Step 3: Test in Conversational Agent

Once integrated, test with:
```bash
python conversational_agent.py
# Then ask: "Find lessons about business vocabulary"
# Should use semantic search to find related content
```

## Recommended Model for Japanese

**Best choice:** `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`
- Excellent Japanese support
- Good balance of speed and quality
- Well-maintained
- ~420MB download

**Alternative:** `intfloat/multilingual-e5-large` (if you need highest quality)
- State-of-the-art performance
- Larger model (~2GB)
- Slower inference

## Vector Database Options

### Option 1: Chroma (Simplest - Recommended for Start)
**Best for:** Getting started quickly, Python-native

**Pros:**
- Easiest to use - just `pip install chromadb`
- Python-native, no separate server
- Automatic embedding management
- **Multilingual support**: Can use any embedding model (multilingual models work great!)
- Built-in persistence
- Good for small-medium scale

**Cons:**
- Less performant than specialized DBs
- Limited advanced features
- Default model (`all-MiniLM-L6-v2`) is English-focused (but you can change it!)

**Multilingual Support:**
Chroma supports multilingual embeddings! You can use:
- `paraphrase-multilingual-MiniLM-L12-v2` (good for Japanese)
- `paraphrase-multilingual-mpnet-base-v2` (excellent for Japanese)
- `LaBSE` (multilingual)
- Custom models (Ollama, OpenAI, etc.)

**Usage with Multilingual Model:**
```python
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

# Option 1: Use Sentence Transformers multilingual model
multilingual_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="paraphrase-multilingual-mpnet-base-v2"
)

client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory="./chroma_db"
))

# Create collection with multilingual embedding function
collection = client.create_collection(
    name="lessons",
    embedding_function=multilingual_ef  # Use multilingual model!
)

# Add lessons (automatic multilingual embedding)
collection.add(
    documents=[
        "ビジネスで成功する方法について話しています。",  # Japanese
        "会社で働くことについて説明しています。",  # Japanese
    ],
    metadatas=[
        {"title": "Business Success", "channel": "hnh"},
        {"title": "Working at Company", "channel": "sjn"},
    ],
    ids=["1", "2"]
)

# Semantic search works with Japanese!
results = collection.query(
    query_texts=["ビジネス"],  # Japanese query
    n_results=5
)
# Finds: "ビジネスで成功する方法" (high similarity)
# Also finds: "会社で働くこと" (related, lower similarity)
```

**Usage with Custom Embeddings (Ollama):**
```python
# Option 2: Use your own embedding function (e.g., Ollama)
def ollama_embedding_function(texts):
    import requests
    embeddings = []
    for text in texts:
        response = requests.post(
            "http://localhost:11434/api/embeddings",
            json={"model": "mxbai-embed-large", "prompt": text}
        )
        embeddings.append(response.json()["embedding"])
    return embeddings

collection = client.create_collection(
    name="lessons",
    embedding_function=ollama_embedding_function  # Your custom function
)
```

### Option 2: Qdrant (Best Performance)
**Best for:** Production use, high performance

**Pros:**
- Very fast similarity search
- Production-ready
- Good Python client
- Can run locally or cloud

**Cons:**
- Requires separate server (or use Docker)
- More setup complexity

**Usage:**
```bash
# Run with Docker
docker run -p 6333:6333 qdrant/qdrant
```

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

client = QdrantClient("localhost", port=6333)

client.create_collection(
    collection_name="lessons",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE)
)

# Add vectors
client.upsert(
    collection_name="lessons",
    points=[...]  # vectors with metadata
)

# Search
results = client.search(
    collection_name="lessons",
    query_vector=query_embedding,
    limit=5
)
```

### Option 3: SQLite + sqlite-vss (Hybrid Approach)
**Best for:** Want both structured queries AND vector search

**Pros:**
- Single file (easy backup)
- Structured queries (SQL) + vector search
- No separate server
- Familiar SQL interface

**Cons:**
- Need to compile extension
- Less optimized than dedicated vector DBs

**Usage:**
```sql
-- Install extension (one-time setup)
.load ./sqlite-vss

-- Create vector table
CREATE VIRTUAL TABLE lesson_vectors USING vss0(
    embedding(384),
    lesson_id INTEGER
);

-- Insert embeddings
INSERT INTO lesson_vectors(embedding, lesson_id) 
VALUES (?, ?);

-- Vector similarity search
SELECT lesson_id, distance 
FROM lesson_vectors 
WHERE vss_search(embedding, ?) 
LIMIT 5;
```

### Option 4: Valkey with Vector Module (Redis Fork)
**Best for:** Real-time applications, existing Redis infrastructure, low latency needs

**Pros:**
- **Ultra-low latency**: Single-digit millisecond search times
- **High performance**: Handles billions of vectors efficiently
- **Hybrid queries**: Combine vector search with Redis data structures (strings, hashes, sets)
- **Scalable**: Linear scaling with CPU cores, supports clustering
- **High availability**: Primary/replica architecture with automatic failover
- **Open source**: BSD 3-clause license (fully free)
- **Mature ecosystem**: If you already use Redis, familiar interface
- **In-memory**: Extremely fast (all data in RAM)

**Cons:**
- **Memory intensive**: All vectors stored in RAM (can be expensive)
- **Setup complexity**: Requires separate server setup (like Redis)
- **Evolving features**: Newer than specialized vector DBs, some features may be limited
- **Smaller community**: Less mature ecosystem than Redis/Qdrant
- **No automatic embedding**: You must generate embeddings yourself (unlike Chroma)
- **Persistence trade-offs**: RAM-based means need careful backup strategy

**Usage:**
```bash
# Install Valkey with vector module
# See: https://valkey.io/docs/getting-started/

# Run Valkey server
valkey-server --loadmodule /path/to/valkey-vector.so
```

```python
import redis
import numpy as np

# Connect to Valkey
r = redis.Redis(host='localhost', port=6379, decode_responses=True)

# Create index with vector field
r.execute_command(
    'FT.CREATE', 'lessons_idx',
    'ON', 'HASH',
    'PREFIX', '1', 'lesson:',
    'SCHEMA', 
    'title', 'TEXT',
    'summary', 'TEXT',
    'embedding', 'VECTOR', 'HNSW', '6', 'TYPE', 'FLOAT32', 'DIM', '1024', 'DISTANCE_METRIC', 'COSINE'
)

# Add document with vector
embedding = get_embedding("ビジネスで成功する方法")  # Your embedding function
r.hset('lesson:1', mapping={
    'title': 'Business Success',
    'summary': 'ビジネスで成功する方法について',
    'embedding': np.array(embedding, dtype=np.float32).tobytes()
})

# Vector similarity search
results = r.execute_command(
    'FT.SEARCH', 'lessons_idx',
    '*=>[KNN 5 @embedding $query_vec]',
    'PARAMS', '2', 'query_vec', np.array(query_embedding, dtype=np.float32).tobytes(),
    'RETURN', '2', 'title', 'summary'
)
```

### Option 5: Pinecone (Cloud - Managed)
**Best for:** Don't want to manage infrastructure

**Pros:**
- Fully managed
- Scalable
- No setup

**Cons:**
- Costs money
- Data in cloud
- Requires internet

## Comparison Table

| Database | Setup | Performance | Memory | Best For |
|----------|-------|-------------|--------|----------|
| **Chroma** | ⭐⭐⭐⭐⭐ Easiest | ⭐⭐⭐ Good | Low | Getting started |
| **Qdrant** | ⭐⭐⭐ Medium | ⭐⭐⭐⭐⭐ Best | Medium | Production |
| **SQLite-vss** | ⭐⭐⭐⭐ Easy | ⭐⭐⭐⭐ Very Good | Low | Hybrid needs |
| **Valkey** | ⭐⭐⭐ Medium | ⭐⭐⭐⭐⭐ Best | High (RAM) | Real-time, low latency |
| **Pinecone** | ⭐⭐⭐⭐⭐ Easiest | ⭐⭐⭐⭐⭐ Best | N/A (cloud) | Cloud-first |

## Recommendation for Your Use Case

**Start with Chroma** because:
1. Simplest setup - just `pip install chromadb`
2. Good enough performance for your scale
3. Automatic embedding management
4. Easy to migrate later if needed

**Upgrade to Qdrant** if:
- You have 1000+ lessons
- Need maximum performance
- Want production-grade features

**Use SQLite-vss** if:
- You want structured queries (SQL) AND vector search
- Prefer single-file database
- Want to avoid separate server

