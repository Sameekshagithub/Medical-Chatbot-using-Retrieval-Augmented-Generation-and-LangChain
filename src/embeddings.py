"""
============================================================
src/embeddings.py - Embedding Generation Module
============================================================

PURPOSE:
    Converts text chunks into dense numerical vectors (embeddings)
    that capture semantic meaning, enabling similarity search.

WHAT ARE EMBEDDINGS?
    Embeddings are numerical representations of text in high-dimensional space.
    
    Example:
        "chest pain"    → [0.23, -0.15, 0.87, ..., 0.42]  (384 numbers)
        "cardiac ache"  → [0.21, -0.14, 0.89, ..., 0.44]  (similar!)
        "pizza recipe"  → [-0.9,  0.88, -0.3, ..., 0.12]  (very different)
    
    Semantically similar texts produce similar vectors!
    This is the magic that powers semantic search.

MODEL: sentence-transformers/all-MiniLM-L6-v2
    - Lightweight: only 80MB, runs on CPU
    - Fast: ~14,000 sentences/second on GPU
    - Dimensions: 384 (vs. OpenAI's 1536)
    - Trained on: 1 billion sentence pairs
    - Perfect for: medical Q&A, document similarity

COSINE SIMILARITY:
    Measures angle between two vectors (range: -1 to 1)
    - 1.0  = identical meaning
    - 0.8+ = very similar
    - 0.5  = somewhat related
    - 0.0  = unrelated
    - -1.0 = opposite meaning

DATA FLOW:
    Text chunks → HuggingFaceEmbeddings model → 384-dim vectors → FAISS index
============================================================
"""

import os
from typing import List, Optional
from loguru import logger

from langchain_community.embeddings import HuggingFaceEmbeddings


class EmbeddingGenerator:
    """
    Generates semantic embeddings using HuggingFace sentence-transformers.
    
    These embeddings are stored in the vector database and used to find
    relevant document chunks when a user asks a question.
    """

    # Available embedding models (sorted by size/speed tradeoff)
    AVAILABLE_MODELS = {
        "all-MiniLM-L6-v2": {
            "full_name": "sentence-transformers/all-MiniLM-L6-v2",
            "dimensions": 384,
            "size_mb": 80,
            "speed": "very fast",
            "quality": "good",
            "description": "Best balance of speed and quality. Recommended."
        },
        "all-mpnet-base-v2": {
            "full_name": "sentence-transformers/all-mpnet-base-v2",
            "dimensions": 768,
            "size_mb": 420,
            "speed": "medium",
            "quality": "excellent",
            "description": "Higher quality embeddings, slower."
        },
        "all-MiniLM-L12-v2": {
            "full_name": "sentence-transformers/all-MiniLM-L12-v2",
            "dimensions": 384,
            "size_mb": 120,
            "speed": "fast",
            "quality": "good",
            "description": "Larger MiniLM variant, slightly better quality."
        }
    }

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
        normalize_embeddings: bool = True,
        cache_folder: Optional[str] = "./model_cache"
    ):
        """
        Initialize the embedding generator.
        
        Args:
            model_name:           HuggingFace model identifier
            device:               'cpu' or 'cuda' (GPU if available)
            normalize_embeddings: Normalize vectors to unit length (better cosine sim)
            cache_folder:         Where to cache downloaded model weights
        """
        self.model_name = model_name
        self.device = device
        
        logger.info(f"Loading embedding model: {model_name}")
        logger.info(f"Device: {device} | Normalize: {normalize_embeddings}")
        
        # Initialize HuggingFace embeddings
        # Model downloads automatically on first run (~80MB for MiniLM)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={
                "device": device        # cpu or cuda
            },
            encode_kwargs={
                "normalize_embeddings": normalize_embeddings,  # L2 normalization
                "batch_size": 32        # Process 32 chunks at a time
            },
            cache_folder=cache_folder   # Avoid re-downloading on every restart
        )
        
        logger.success(f"Embedding model loaded: {model_name.split('/')[-1]}")

    def get_embeddings(self) -> HuggingFaceEmbeddings:
        """
        Return the LangChain-compatible embeddings object.
        
        This is passed directly to FAISS/ChromaDB for indexing.
        
        Returns:
            HuggingFaceEmbeddings instance compatible with LangChain vector stores
        """
        return self.embeddings

    def embed_query(self, text: str) -> List[float]:
        """
        Generate embedding for a single query text.
        
        Used during retrieval: converts user question to vector,
        then finds nearest document chunk vectors in the database.
        
        Args:
            text: The query string (user's medical question)
            
        Returns:
            List of floats representing the embedding vector
            
        Example:
            vec = embed_query("What are symptoms of diabetes?")
            # Returns [0.23, -0.15, ..., 0.42] with 384 dimensions
        """
        return self.embeddings.embed_query(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a batch of text chunks.
        
        Used during indexing: converts all document chunks to vectors
        for storage in the vector database.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors (list of lists)
        """
        logger.info(f"Generating embeddings for {len(texts)} texts...")
        embeddings = self.embeddings.embed_documents(texts)
        logger.success(f"Generated {len(embeddings)} embeddings | Dimension: {len(embeddings[0])}")
        return embeddings

    @property
    def embedding_dimension(self) -> int:
        """Return the dimension of the embedding vectors."""
        # all-MiniLM-L6-v2 produces 384-dimensional vectors
        test_vec = self.embed_query("test")
        return len(test_vec)

    def get_model_info(self) -> dict:
        """Return information about the loaded embedding model."""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "short_name": self.model_name.split("/")[-1]
        }


# ─── Singleton pattern for efficient resource use ──────────────────────────
_embedding_instance: Optional[EmbeddingGenerator] = None

def get_embedding_generator(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: str = "cpu"
) -> EmbeddingGenerator:
    """
    Get or create a singleton EmbeddingGenerator instance.
    
    Loading an embedding model takes ~2-5 seconds and uses memory.
    Reusing the same instance avoids repeated initialization.
    
    Args:
        model_name: HuggingFace model identifier
        device:     'cpu' or 'cuda'
        
    Returns:
        EmbeddingGenerator singleton instance
    """
    global _embedding_instance
    
    if _embedding_instance is None:
        _embedding_instance = EmbeddingGenerator(
            model_name=model_name,
            device=device
        )
    
    return _embedding_instance


# ─── Standalone usage example ────────────────────────────────────────────────
if __name__ == "__main__":
    gen = EmbeddingGenerator()
    
    # Test similarity between medical terms
    vec1 = gen.embed_query("hypertension treatment guidelines")
    vec2 = gen.embed_query("high blood pressure management protocol")
    vec3 = gen.embed_query("chocolate cake recipe")
    
    # Compute cosine similarity manually
    import numpy as np
    def cosine_sim(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    sim12 = cosine_sim(vec1, vec2)
    sim13 = cosine_sim(vec1, vec3)
    
    print(f"'hypertension treatment' vs 'high blood pressure': {sim12:.3f} (should be HIGH)")
    print(f"'hypertension treatment' vs 'chocolate cake':      {sim13:.3f} (should be LOW)")
    print("EmbeddingGenerator module ready ✓")
