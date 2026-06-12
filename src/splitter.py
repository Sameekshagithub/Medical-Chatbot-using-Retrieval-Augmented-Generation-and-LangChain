"""
============================================================
src/splitter.py - Text Chunking Module
============================================================

PURPOSE:
    Splits large medical documents into smaller, overlapping
    chunks that fit within LLM context windows and enable
    precise semantic search.

WHY CHUNKING MATTERS IN RAG:
    - LLMs have token limits (e.g., GPT-4 handles ~128k tokens)
    - A 200-page medical textbook far exceeds any context window
    - Searching at chunk level is more precise than full-doc search
    - Smaller chunks = more targeted retrieval = better answers
    - Overlap prevents losing context at chunk boundaries

CHUNK SIZE GUIDE:
    - chunk_size=500:  Fine-grained, precise but may miss context
    - chunk_size=1000: Balanced for most medical documents ✓
    - chunk_size=2000: Broad context, may dilute relevance

OVERLAP GUIDE:
    - chunk_overlap=0:   Fast but loses boundary context
    - chunk_overlap=200: Standard - ensures continuity ✓
    - chunk_overlap=500: Redundant but maximally safe

DATA FLOW:
    List[Document] → RecursiveCharacterTextSplitter → List[Document chunks]
============================================================
"""

import os
from typing import List, Optional
from loguru import logger

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

class TextChunker:
    """
    Splits large documents into smaller overlapping chunks for RAG.
    
    Uses RecursiveCharacterTextSplitter which tries to split on:
        1. Paragraph breaks (\n\n) → best split point
        2. Line breaks (\n)        → second choice
        3. Sentences (. ! ?)      → third choice
        4. Words (spaces)         → last resort
        5. Characters             → absolute fallback
        
    This hierarchy preserves semantic meaning as much as possible.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None
    ):
        """
        Initialize the TextChunker.
        
        Args:
            chunk_size:    Target size of each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
                           (prevents losing context at boundaries)
            separators:    Custom split points (defaults to standard hierarchy)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Default separators: try to split on these in order
        # Medical docs often use \n\n for paragraphs and \n for line breaks
        self.separators = separators or ["\n\n", "\n", ". ", "! ", "? ", " ", ""]
        
        # Initialize the LangChain text splitter
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=self.separators,
            length_function=len,       # Use character count (not tokens)
            is_separator_regex=False   # Treat separators as plain strings
        )
        
        logger.info(
            f"TextChunker initialized | "
            f"chunk_size={chunk_size} | "
            f"chunk_overlap={chunk_overlap}"
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split a list of Documents into smaller chunks.
        
        Each chunk inherits the metadata from its parent document,
        so we always know which PDF and page a chunk came from.
        
        Args:
            documents: List of Document objects from the loader
            
        Returns:
            List of smaller Document chunks with inherited metadata
            
        Example:
            # Before: 50 pages, ~3000 chars each = 150,000 chars total
            # After:  ~150 chunks of ~1000 chars each (with 200 overlap)
        """
        if not documents:
            logger.warning("No documents provided for splitting")
            return []
        
        logger.info(f"Splitting {len(documents)} documents into chunks...")
        
        # LangChain's split_documents handles metadata inheritance automatically
        chunks = self.splitter.split_documents(documents)
        
        # Add chunk index to metadata for debugging and citation
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = i
            chunk.metadata["chunk_size"] = len(chunk.page_content)
        
        logger.success(
            f"Created {len(chunks)} chunks | "
            f"avg size: {sum(len(c.page_content) for c in chunks) // max(len(chunks), 1)} chars"
        )
        
        return chunks

    def split_text(self, text: str, metadata: Optional[dict] = None) -> List[Document]:
        """
        Split a raw text string (not a Document object) into chunks.
        
        Useful when you have text from non-PDF sources.
        
        Args:
            text:     Raw text string to split
            metadata: Optional metadata dict to attach to all chunks
            
        Returns:
            List of Document chunks
        """
        texts = self.splitter.split_text(text)
        
        documents = []
        for i, text_chunk in enumerate(texts):
            doc = Document(
                page_content=text_chunk,
                metadata={
                    **(metadata or {}),
                    "chunk_id": i,
                    "chunk_size": len(text_chunk)
                }
            )
            documents.append(doc)
        
        return documents

    def get_chunk_stats(self, chunks: List[Document]) -> dict:
        """
        Return statistics about the generated chunks.
        
        Args:
            chunks: List of Document chunks
            
        Returns:
            Dict with statistics for UI display
        """
        if not chunks:
            return {}
        
        sizes = [len(c.page_content) for c in chunks]
        sources = list(set(c.metadata.get("source", "unknown") for c in chunks))
        
        return {
            "total_chunks": len(chunks),
            "avg_chunk_size": sum(sizes) // len(sizes),
            "min_chunk_size": min(sizes),
            "max_chunk_size": max(sizes),
            "total_characters": sum(sizes),
            "unique_sources": sources,
            "chunk_size_setting": self.chunk_size,
            "chunk_overlap_setting": self.chunk_overlap
        }

    def preview_chunks(self, chunks: List[Document], n: int = 3) -> None:
        """
        Print a preview of the first n chunks (for debugging).
        
        Args:
            chunks: List of Document chunks
            n:      Number of chunks to preview
        """
        for i, chunk in enumerate(chunks[:n]):
            logger.debug(
                f"\n{'='*50}\n"
                f"Chunk {i} | Source: {chunk.metadata.get('source')} | "
                f"Page: {chunk.metadata.get('page', 'N/A')} | "
                f"Size: {len(chunk.page_content)} chars\n"
                f"Preview: {chunk.page_content[:150]}..."
            )


# ─── Standalone usage example ────────────────────────────────────────────────
if __name__ == "__main__":
    chunker = TextChunker(chunk_size=1000, chunk_overlap=200)
    
    # Create a sample document to test
    sample_doc = Document(
        page_content="This is a medical document about hypertension. " * 50,
        metadata={"source": "test.pdf", "page": 0}
    )
    
    chunks = chunker.split_documents([sample_doc])
    stats = chunker.get_chunk_stats(chunks)
    
    print(f"Created {stats['total_chunks']} chunks")
    print(f"Average size: {stats['avg_chunk_size']} chars")
    print("TextChunker module ready ✓")
