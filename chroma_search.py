#!/usr/bin/env python3
"""
Chroma Integration Module - Manage Chroma database and semantic search
"""

import chromadb
from chromadb.config import Settings
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import logging

from embeddings import create_ollama_embedding_function

logger = logging.getLogger(__name__)


def initialize_chroma(data_root: Path, embedding_model: str = "mxbai-embed-large",
                      base_url: str = "http://localhost:11434",
                      collection_name: str = "transcriptions") -> chromadb.Collection:
    """
    Initialize Chroma collection with Ollama embedding function.
    
    Args:
        data_root: Root data directory (Chroma will store in {data_root}/.chroma_db/)
        embedding_model: Ollama embedding model name
        base_url: Ollama base URL
        collection_name: Name of the Chroma collection
    
    Returns:
        Chroma collection object
    
    Raises:
        RuntimeError: If Chroma initialization fails
    """
    try:
        # Create Chroma client with persistent storage
        chroma_path = data_root / ".chroma_db"
        chroma_path.mkdir(parents=True, exist_ok=True)
        
        client = chromadb.PersistentClient(
            path=str(chroma_path),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Create custom embedding function using Ollama
        embedding_function = create_ollama_embedding_function(
            model=embedding_model,
            base_url=base_url
        )
        
        # Get or create collection
        try:
            collection = client.get_collection(name=collection_name)
            # Check if collection uses the correct embedding model
            collection_metadata = collection.metadata or {}
            existing_model = collection_metadata.get("embedding_model", "")
            
            if existing_model != embedding_model:
                logger.warning(f"Collection '{collection_name}' uses embedding model '{existing_model}', but config specifies '{embedding_model}'. Recreating collection...")
                # Delete old collection and recreate with correct embedding function
                client.delete_collection(name=collection_name)
                collection = client.create_collection(
                    name=collection_name,
                    embedding_function=embedding_function,
                    metadata={"embedding_model": embedding_model}
                )
                logger.info(f"Recreated Chroma collection '{collection_name}' with embedding model '{embedding_model}'")
            else:
                logger.info(f"Loaded existing Chroma collection: {collection_name} (using {embedding_model})")
        except Exception:
            # Collection doesn't exist, create it
            collection = client.create_collection(
                name=collection_name,
                embedding_function=embedding_function,
                metadata={"embedding_model": embedding_model}
            )
            logger.info(f"Created new Chroma collection: {collection_name} with embedding model '{embedding_model}'")
        
        return collection
    
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Chroma: {e}")


def store_transcription_chunks(chunks: List[Dict], collection: chromadb.Collection,
                               force: bool = False) -> int:
    """
    Store transcription chunks in Chroma.
    
    Args:
        chunks: List of chunk dictionaries (from chunk_transcription)
        collection: Chroma collection object
        force: If True, overwrite existing chunks. If False, skip duplicates.
    
    Returns:
        Number of chunks actually stored (may be less than input if duplicates skipped)
    """
    if not chunks:
        return 0
    
    # Check for existing chunks if not forcing
    existing_ids = set()
    if not force:
        try:
            # Get all existing IDs (limit to reasonable number)
            existing = collection.get(limit=10000)  # Adjust if needed
            existing_ids = set(existing.get('ids', []))
        except Exception as e:
            logger.warning(f"Could not check existing chunks: {e}")
    
    # Filter out duplicates if not forcing
    chunks_to_store = []
    if force:
        chunks_to_store = chunks
    else:
        chunks_to_store = [c for c in chunks if c['chunk_id'] not in existing_ids]
    
    if not chunks_to_store:
        logger.info(f"All {len(chunks)} chunks already exist in Chroma, skipping")
        return 0
    
    # Prepare data for Chroma
    documents = [chunk['content'] for chunk in chunks_to_store]
    ids = [chunk['chunk_id'] for chunk in chunks_to_store]
    metadatas = [
        {
            "source": chunk['source'],
            "episode_date": chunk.get('episode_date', ''),
            "channel": chunk.get('channel', ''),
            "transcription_file": chunk.get('transcription_file', ''),
            "lesson_file": chunk.get('lesson_file', ''),
            "sentence_count": chunk.get('sentence_count', 0),
            "position": chunk.get('position', 0.0)
        }
        for chunk in chunks_to_store
    ]
    
    try:
        # Add to Chroma (batch operation)
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        logger.info(f"Stored {len(chunks_to_store)} chunks in Chroma")
        return len(chunks_to_store)
    
    except Exception as e:
        logger.error(f"Error storing chunks in Chroma: {e}")
        raise


def search_transcriptions(query: str, collection: chromadb.Collection,
                         top_k: int = 5, similarity_threshold: float = 0.5) -> List[Dict]:
    """
    Perform semantic search on transcriptions using Chroma.
    
    Args:
        query: Search query text
        collection: Chroma collection object
        top_k: Number of top results to return
        similarity_threshold: Minimum similarity score (0.0-1.0)
    
    Returns:
        List of chunk dictionaries with similarity scores, sorted by similarity (highest first)
        Each dict includes all original chunk fields plus 'similarity' score
    """
    if not query or not query.strip():
        return []
    
    try:
        # Perform semantic search
        results = collection.query(
            query_texts=[query],
            n_results=min(top_k * 2, 20)  # Get more results to filter by threshold
        )
        
        # Extract results
        if not results['ids'] or not results['ids'][0]:
            return []
        
        chunks_with_scores = []
        ids = results['ids'][0]
        distances = results['distances'][0] if results.get('distances') else []
        metadatas = results['metadatas'][0] if results.get('metadatas') else []
        documents = results['documents'][0] if results.get('documents') else []
        
        # Convert distances to similarities (Chroma uses distance, we want similarity)
        # Distance 0 = similarity 1.0, higher distance = lower similarity
        # For cosine distance: similarity = 1 - distance
        for i, chunk_id in enumerate(ids):
            distance = distances[i] if i < len(distances) else 1.0
            similarity = 1.0 - distance if distance <= 1.0 else 0.0
            
            # Filter by threshold
            if similarity < similarity_threshold:
                continue
            
            # Reconstruct chunk dict
            metadata = metadatas[i] if i < len(metadatas) else {}
            content = documents[i] if i < len(documents) else ''
            
            chunk_dict = {
                "chunk_id": chunk_id,
                "content": content,
                "source": metadata.get('source', ''),
                "episode_date": metadata.get('episode_date', ''),
                "channel": metadata.get('channel', ''),
                "transcription_file": metadata.get('transcription_file', ''),
                "lesson_file": metadata.get('lesson_file', ''),
                "sentence_count": metadata.get('sentence_count', 0),
                "position": metadata.get('position', 0.0),
                "similarity": similarity
            }
            chunks_with_scores.append(chunk_dict)
        
        # Sort by similarity (highest first) and return top_k
        chunks_with_scores.sort(key=lambda x: x['similarity'], reverse=True)
        return chunks_with_scores[:top_k]
    
    except Exception as e:
        logger.error(f"Error searching transcriptions: {e}")
        return []


def get_chroma_collection(data_root: Path, config: Dict) -> Optional[chromadb.Collection]:
    """
    Get Chroma collection with configuration from config file.
    
    Args:
        data_root: Root data directory
        config: Configuration dictionary (from config/podcasts.json)
    
    Returns:
        Chroma collection object, or None if Chroma is disabled or unavailable
    """
    # Check if embeddings are enabled
    embeddings_cfg = config.get('embeddings', {})
    if embeddings_cfg.get('enabled', True) is False:
        logger.info("Embeddings disabled in config")
        return None
    
    # Get embedding model and base URL
    embedding_model = embeddings_cfg.get('model', 'mxbai-embed-large')
    
    # Try to get base_url from embeddings config, fallback to analysis config
    base_url = embeddings_cfg.get('base_url')
    if not base_url:
        analysis_cfg = config.get('analysis', {})
        base_url = analysis_cfg.get('base_url', 'http://localhost:11434')
    
    # Expand environment variables if present
    if isinstance(base_url, str) and ('$' in base_url or '{' in base_url):
        from generate_lesson import expand_env_vars
        base_url = expand_env_vars(base_url)
    
    try:
        collection = initialize_chroma(
            data_root=data_root,
            embedding_model=embedding_model,
            base_url=base_url
        )
        return collection
    except Exception as e:
        logger.warning(f"Could not initialize Chroma: {e}")
        return None


def check_chunks_exist(collection: chromadb.Collection, chunk_ids: List[str]) -> set:
    """
    Check which chunk IDs already exist in Chroma.
    
    Args:
        collection: Chroma collection object
        chunk_ids: List of chunk IDs to check
    
    Returns:
        Set of chunk IDs that exist in Chroma
    """
    if not chunk_ids:
        return set()
    
    try:
        # Get existing chunks by IDs
        existing = collection.get(ids=chunk_ids)
        return set(existing.get('ids', []))
    except Exception:
        # If any ID doesn't exist, Chroma raises error
        # Check one by one (slower but more reliable)
        existing_ids = set()
        for chunk_id in chunk_ids:
            try:
                collection.get(ids=[chunk_id])
                existing_ids.add(chunk_id)
            except Exception:
                pass
        return existing_ids

