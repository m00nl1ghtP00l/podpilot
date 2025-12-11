# RAG Architecture for Podpilot

## Current State: Partial RAG

Your `conversational_agent.py` already implements **basic RAG**:

```python
# In answer_question() method:
1. RETRIEVE: Gets lessons (currently keyword-based)
2. AUGMENT: Adds lesson content to prompt
3. GENERATE: LLM answers using that content
```

**What you have:**
- ✅ Retrieval (keyword search)
- ✅ Augmentation (adds to prompt)
- ✅ Generation (LLM answers)

**What's missing for full RAG:**
- ❌ Semantic search for retrieval (you're exploring this!)
- ❌ Chunking (lessons are used whole, not in chunks)
- ❌ Better ranking (top-K most relevant chunks)
- ❌ Source citation (doesn't cite which lesson/chunk)

## Should You Build Full RAG?

### ✅ YES, if you want:
1. **Better answers**: Answers based on YOUR lesson content, not general knowledge
2. **Source citations**: "According to lesson #48..."
3. **Precise retrieval**: Find exact vocabulary/grammar explanations
4. **Scalability**: Handle hundreds of lessons efficiently

### ❌ NO, if:
1. **Learning packs are enough**: You just want to generate study materials
2. **Simple search works**: Keyword search finds what you need
3. **Small scale**: < 100 lessons, current approach is fine

## Recommended RAG Architecture

### Option 1: Enhance Current Approach (Minimal Changes)

**What to add:**
1. **Semantic search** for retrieval (replace keyword search)
2. **Chunk lessons** into smaller pieces (vocabulary, grammar, examples)
3. **Rank chunks** by relevance to query

```python
# Enhanced answer_question() with semantic search
def answer_question(self, question: str) -> str:
    # 1. RETRIEVE: Use semantic search
    query_embedding = get_embedding(question)
    relevant_chunks = semantic_search(query_embedding, all_lesson_chunks, top_k=5)
    
    # 2. AUGMENT: Add retrieved chunks to prompt
    context = "\n\n".join([
        f"From {chunk['source']}:\n{chunk['content']}"
        for chunk in relevant_chunks
    ])
    
    # 3. GENERATE: LLM answers with citations
    prompt = f"""Answer based on these lesson excerpts:
    
{context}

Question: {question}

Answer using the content above. Cite sources (e.g., "from lesson #48")."""
    
    return self.provider.generate(prompt)
```

### Option 2: Full RAG with Vector Database

**Architecture:**
```
User Query
    ↓
Embed Query → [1024 numbers]
    ↓
Vector DB Search → Top 5 chunks
    ↓
Augment Prompt → Add chunks
    ↓
LLM Generation → Answer with citations
```

**Benefits:**
- Fast semantic search (indexed)
- Handles large scale (1000+ lessons)
- Precise chunk retrieval
- Source citations

**Implementation:**
1. **Chunk lessons** into:
   - Vocabulary entries (word + definition + example)
   - Grammar points (pattern + explanation + examples)
   - Summary sections
   
2. **Store chunks** in vector database (Chroma/Qdrant)

3. **Retrieve** top-K chunks for each query

4. **Generate** answers with citations

## Chunking Strategy

### Current: Whole Lessons
```python
# Uses entire lesson
lesson_context = f"{lesson.summary} {lesson.vocabulary} {lesson.grammar}"
```

### Better: Semantic Chunks
```python
# Chunk by semantic unit
chunks = [
    {
        "content": f"Vocabulary: {word} - {definition}\nExample: {example}",
        "source": f"lesson_{lesson_id}",
        "type": "vocabulary"
    },
    {
        "content": f"Grammar: {pattern}\nMeaning: {meaning}\nExample: {example}",
        "source": f"lesson_{lesson_id}",
        "type": "grammar"
    }
]
```

## Implementation Steps

### Step 1: Add Semantic Search (You're doing this!)
- ✅ Embedding model: `mxbai-embed-large`
- ✅ Cosine similarity calculation
- ⏳ Integrate into `answer_question()`

### Step 2: Chunk Lessons
```python
def chunk_lesson(lesson: LessonContent) -> List[Dict]:
    """Split lesson into searchable chunks"""
    chunks = []
    
    # Vocabulary chunks
    for vocab in lesson.vocabulary:
        chunks.append({
            "content": f"{vocab['word']} ({vocab.get('reading', '')}): {vocab.get('meaning', '')}",
            "source": f"{lesson.episode_title}",
            "type": "vocabulary",
            "metadata": {"jlpt_level": vocab.get("jlpt_level")}
        })
    
    # Grammar chunks
    for grammar in lesson.grammar_points:
        chunks.append({
            "content": f"{grammar['pattern']}: {grammar.get('meaning', '')}\nExample: {grammar.get('example', '')}",
            "source": f"{lesson.episode_title}",
            "type": "grammar"
        })
    
    return chunks
```

### Step 3: Store in Vector DB (Optional)
```python
# Store chunks in Chroma
collection.add(
    documents=[chunk["content"] for chunk in chunks],
    metadatas=[{"source": chunk["source"], "type": chunk["type"]} for chunk in chunks],
    ids=[f"{lesson_id}_{i}" for i in range(len(chunks))]
)
```

### Step 4: Enhanced Retrieval
```python
def retrieve_relevant_chunks(query: str, top_k: int = 5):
    """Retrieve most relevant chunks using semantic search"""
    query_embedding = get_embedding(query)
    
    # Option A: Manual (current approach)
    similarities = []
    for chunk in all_chunks:
        chunk_embedding = get_embedding(chunk["content"])
        similarity = cosine_similarity(query_embedding, chunk_embedding)
        similarities.append((chunk, similarity))
    
    # Sort and return top-K
    similarities.sort(key=lambda x: x[1], reverse=True)
    return [chunk for chunk, score in similarities[:top_k]]
    
    # Option B: Vector DB (better for scale)
    # results = collection.query(query_texts=[query], n_results=top_k)
```

## Recommendation

**For your current scale (< 1000 lessons):**

1. ✅ **Enhance current approach** (Option 1)
   - Add semantic search to `answer_question()`
   - Chunk lessons into vocabulary/grammar units
   - Add source citations
   - **No vector DB needed yet**

2. ⏳ **Add vector DB later** (Option 2)
   - When you have 1000+ lessons
   - When search becomes slow
   - When you need production performance

**You're already 80% there!** Just need:
- Semantic search integration
- Better chunking
- Source citations

## Example: Enhanced RAG Flow

```python
User: "What does ておく mean?"

1. RETRIEVE (semantic search):
   - Query embedding: [0.123, -0.456, ...]
   - Find chunks about "ておく"
   - Top result: "Grammar: ておく - do something in advance. Example: 買っておく (buy in advance)"

2. AUGMENT (add to prompt):
   """
   From lesson #48 "Business Japanese Basics":
   Grammar: ておく - do something in advance
   Example: 買っておく means "buy in advance"
   
   Question: What does ておく mean?
   """

3. GENERATE (LLM answers):
   "According to lesson #48, ておく means to do something in advance. 
   For example, 買っておく means 'buy in advance' - preparing something 
   before it's needed."
```

## Next Steps

1. **Now**: Add semantic search to `answer_question()` method
2. **Soon**: Implement chunking for better retrieval
3. **Later**: Add vector DB if scale requires it

You're building RAG - just enhance what you have!

