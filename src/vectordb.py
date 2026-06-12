"""
============================================================
src/vectordb.py - Vector Database Module
============================================================

PURPOSE:
    Stores document embeddings in a vector database and performs
    fast similarity search to find relevant chunks for user queries.

WHAT IS A VECTOR DATABASE?
    A specialized database optimized for storing and searching
    high-dimensional vectors (embeddings).
    
    Traditional DB: "SELECT * WHERE content LIKE '%hypertension%'"
    Vector DB:      "Find 5 vectors closest to query_vector"
    
    The vector DB finds semantically similar content, not just
    keyword matches. This is why RAG is so powerful!

FAISS (Facebook AI Similarity Search):
    - Open source by Meta AI Research
    - Extremely fast: searches millions of vectors in milliseconds
    - Runs entirely in memory (saves to disk as .pkl/.index files)
    - Algorithm: IndexFlatIP (Inner Product / cosine similarity)
    - Best for: smaller datasets, fast retrieval, no server needed

ChromaDB:
    - Open source, built for AI applications
    - Persists to SQLite automatically
    - Supports metadata filtering (filter by document source, page, etc.)
    - Best for: larger datasets, metadata filtering, production use

DATA FLOW:
    Chunks → Embeddings → FAISS/Chroma index → Retriever → LangChain RAG Chain
============================================================
"""

import os
import pickle
from pathlib import Path
from typing import List, Optional, Tuple
from loguru import logger
from langchain_core.documents import Document

from langchain_community.vectorstores import FAISS, Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


class VectorDatabase:
    """
    Manages vector storage and retrieval using FAISS or ChromaDB.
    
    Supports:
        - Creating a new index from document chunks
        - Loading a previously saved index
        - Similarity search (top-k retrieval)
        - Similarity search with relevance scores
        - Metadata filtering
    """

    def __init__(
        self,
        embeddings: HuggingFaceEmbeddings,
        db_type: str = "faiss",
        persist_directory: str = "./vectorstore"
    ):
        """
        Initialize the VectorDatabase.
        
        Args:
            embeddings:         HuggingFace embeddings model
            db_type:            "faiss" or "chroma"
            persist_directory:  Directory to save/load the index
        """
        self.embeddings = embeddings
        self.db_type = db_type.lower()
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        self.vectorstore = None  # Populated after create() or load()
        
        logger.info(f"VectorDatabase initialized | Type: {db_type} | Dir: {persist_directory}")

    # ─── Index Creation ────────────────────────────────────────────────────

    def create_from_documents(self, chunks: List[Document]) -> None:
        """
        Create a vector index from document chunks.
        
        Process:
            1. Extract text from each chunk
            2. Generate embeddings using the embedding model
            3. Store (text, embedding, metadata) in the vector index
            4. Save index to disk for future sessions
        
        Args:
            chunks: List of Document chunks from the text splitter
        """
        if not chunks:
            logger.error("Cannot create vector store from empty chunk list")
            raise ValueError("No chunks provided for indexing")
        
        logger.info(f"Creating {self.db_type.upper()} index from {len(chunks)} chunks...")
        
        if self.db_type == "faiss":
            self._create_faiss(chunks)
        elif self.db_type == "chroma":
            self._create_chroma(chunks)
        else:
            raise ValueError(f"Unsupported db_type: {self.db_type}. Use 'faiss' or 'chroma'")
        
        logger.success(f"Vector index created and saved to '{self.persist_directory}'")

    def _create_faiss(self, chunks: List[Document]) -> None:
        """Create and save a FAISS index."""
        # from_documents: embeds all chunks and builds FAISS index
        self.vectorstore = FAISS.from_documents(
            documents=chunks,
            embedding=self.embeddings
        )
        
        # Save FAISS index to disk (creates .pkl and .index files)
        save_path = str(self.persist_directory / "faiss_index")
        self.vectorstore.save_local(save_path)
        logger.info(f"FAISS index saved to: {save_path}")

    def _create_chroma(self, chunks: List[Document]) -> None:
        """Create and save a ChromaDB index."""
        chroma_path = str(self.persist_directory / "chroma_db")
        
        # ChromaDB persists automatically to the directory
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=chroma_path
        )
        
        # Explicit persist for older ChromaDB versions
        if hasattr(self.vectorstore, 'persist'):
            self.vectorstore.persist()
        
        logger.info(f"ChromaDB index saved to: {chroma_path}")

    # ─── Loading Saved Index ───────────────────────────────────────────────

    def load(self) -> bool:
        """
        Load a previously saved vector index from disk.
        
        Returns:
            True if loaded successfully, False if no saved index found
        """
        try:
            if self.db_type == "faiss":
                return self._load_faiss()
            elif self.db_type == "chroma":
                return self._load_chroma()
        except Exception as e:
            logger.error(f"Failed to load vector store: {e}")
            return False

    def _load_faiss(self) -> bool:
        """Load FAISS index from disk."""
        save_path = str(self.persist_directory / "faiss_index")
        
        if not Path(save_path).exists():
            logger.warning(f"No FAISS index found at: {save_path}")
            return False
        
        # allow_dangerous_deserialization needed for newer FAISS versions
        self.vectorstore = FAISS.load_local(
            folder_path=save_path,
            embeddings=self.embeddings,
            allow_dangerous_deserialization=True
        )
        logger.success(f"FAISS index loaded from: {save_path}")
        return True

    def _load_chroma(self) -> bool:
        """Load ChromaDB from disk."""
        chroma_path = str(self.persist_directory / "chroma_db")
        
        if not Path(chroma_path).exists():
            logger.warning(f"No ChromaDB found at: {chroma_path}")
            return False
        
        self.vectorstore = Chroma(
            persist_directory=chroma_path,
            embedding_function=self.embeddings
        )
        logger.success(f"ChromaDB loaded from: {chroma_path}")
        return True

    def index_exists(self) -> bool:
        """Check if a saved index already exists on disk."""
        if self.db_type == "faiss":
            return (self.persist_directory / "faiss_index").exists()
        elif self.db_type == "chroma":
            return (self.persist_directory / "chroma_db").exists()
        return False

    # ─── Retrieval ─────────────────────────────────────────────────────────

    def get_retriever(self, k: int = 5):
        """
        Create a LangChain-compatible retriever for use in RAG chains.
        
        The retriever is the bridge between the vector database and the LLM.
        When called with a query, it:
            1. Embeds the query
            2. Finds top-k most similar chunks
            3. Returns them as context for the LLM
        
        Args:
            k: Number of chunks to retrieve (top-k)
            
        Returns:
            LangChain VectorStoreRetriever object
        """
        if self.vectorstore is None:
            raise RuntimeError("Vector store not initialized. Call create_from_documents() or load() first.")
        
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",      # Options: "similarity", "mmr", "similarity_score_threshold"
            search_kwargs={
                "k": k                     # Return top-k most similar chunks
            }
        )
        
        logger.info(f"Retriever created with k={k}")
        return retriever

    def similarity_search(
        self,
        query: str,
        k: int = 5
    ) -> List[Document]:
        """
        Find the most similar document chunks for a given query.
        
        Args:
            query: User's question or search string
            k:     Number of results to return
            
        Returns:
            List of most similar Document chunks
        """
        if self.vectorstore is None:
            raise RuntimeError("Vector store not initialized")
        
        results = self.vectorstore.similarity_search(query, k=k)
        logger.info(f"Similarity search: found {len(results)} results for query")
        return results

    def similarity_search_with_scores(
        self,
        query: str,
        k: int = 5
    ) -> List[Tuple[Document, float]]:
        """
        Find similar chunks AND their similarity scores.
        
        Scores are used in the source citation system to show users
        how confident the retrieval was (higher = more relevant).
        
        Args:
            query: User's question
            k:     Number of results
            
        Returns:
            List of (Document, similarity_score) tuples
            Score range: 0.0 (unrelated) to 1.0 (identical)
        """
        if self.vectorstore is None:
            raise RuntimeError("Vector store not initialized")
        
        results = self.vectorstore.similarity_search_with_relevance_scores(
            query, k=k
        )
        
        logger.info(f"Retrieved {len(results)} chunks with scores")
        return results

    def add_documents(self, new_chunks: List[Document]) -> None:
        """
        Add new document chunks to an existing index (incremental indexing).
        
        Useful for adding new PDFs without re-indexing everything.
        
        Args:
            new_chunks: New document chunks to add
        """
        if self.vectorstore is None:
            raise RuntimeError("Vector store not initialized. Create index first.")
        
        self.vectorstore.add_documents(new_chunks)
        
        # Re-save for FAISS (ChromaDB auto-persists)
        if self.db_type == "faiss":
            save_path = str(self.persist_directory / "faiss_index")
            self.vectorstore.save_local(save_path)
        
        logger.success(f"Added {len(new_chunks)} new chunks to existing index")

    def get_collection_stats(self) -> dict:
        """Return statistics about the vector store."""
        if self.vectorstore is None:
            return {"status": "not initialized"}
        
        stats = {
            "db_type": self.db_type,
            "status": "initialized",
            "persist_directory": str(self.persist_directory)
        }
        
        # FAISS-specific stats
        if self.db_type == "faiss" and hasattr(self.vectorstore, 'index'):
            stats["total_vectors"] = self.vectorstore.index.ntotal
        
        return stats


# ─── Standalone usage example ────────────────────────────────────────────────
if __name__ == "__main__":
    print("VectorDatabase module ready ✓")
    print("Supports: FAISS and ChromaDB")
    print("Features: create, load, search, add documents, get retriever")
