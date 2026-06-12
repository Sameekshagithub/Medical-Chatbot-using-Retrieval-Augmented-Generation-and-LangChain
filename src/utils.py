"""
============================================================
src/utils.py - Utility Functions
============================================================

PURPOSE:
    Shared helper functions used across all modules.
    Includes: config loading, formatting, caching, validation.
============================================================
"""

import os
import json
import hashlib
import time
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
from loguru import logger
from dotenv import load_dotenv


# ─── Environment Configuration ────────────────────────────────────────────

def load_config() -> dict:
    """
    Load all configuration from .env file.
    
    Returns:
        Dict containing all app configuration values
    """
    # Load .env file (silently ignore if not found)
    load_dotenv(override=True)
    
    config = {
        # LLM Settings
        "llm_provider": os.getenv("LLM_PROVIDER", "groq"),
        "openai_model": os.getenv("OPENAI_MODEL", "gpt-4o"),
        "groq_model": os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile"),
        "llm_temperature": float(os.getenv("LLM_TEMPERATURE", "0.1")),
        "max_tokens": int(os.getenv("MAX_TOKENS", "1024")),
        
        # Embedding Settings
        "embedding_model": os.getenv(
            "EMBEDDING_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2"
        ),
        
        # Vector DB Settings
        "vector_db": os.getenv("VECTOR_DB", "faiss"),
        "vector_db_path": os.getenv("VECTOR_DB_PATH", "./vectorstore"),
        
        # RAG Settings
        "retrieval_k": int(os.getenv("RETRIEVAL_K", "5")),
        "chunk_size": int(os.getenv("CHUNK_SIZE", "1000")),
        "chunk_overlap": int(os.getenv("CHUNK_OVERLAP", "200")),
        
        # App Settings
        "app_title": os.getenv("APP_TITLE", "MediQuery AI"),
        "app_version": os.getenv("APP_VERSION", "1.0.0"),
        
        # Database
        "sqlite_db_path": os.getenv("SQLITE_DB_PATH", "./chat_history/conversations.db"),
        
        # Logging
        "log_level": os.getenv("LOG_LEVEL", "INFO"),
    }
    
    return config


def validate_api_keys(config: dict) -> tuple[bool, str]:
    """
    Validate that required API keys are present and not placeholder values.
    
    Args:
        config: Configuration dict from load_config()
        
    Returns:
        (is_valid, error_message) tuple
    """
    provider = config.get("llm_provider", "groq")
    
    if provider == "openai":
        key = os.getenv("OPENAI_API_KEY", "")
        if not key or key == "your_openai_api_key_here" or not key.startswith("sk-"):
            return False, (
                "❌ OpenAI API key is missing or invalid.\n"
                "Add OPENAI_API_KEY to your .env file.\n"
                "Get your key at: https://platform.openai.com/api-keys"
            )
    
    elif provider == "groq":
        key = os.getenv("GROQ_API_KEY", "")
        if not key or key == "your_groq_api_key_here":
            return False, (
                "❌ Groq API key is missing or invalid.\n"
                "Add GROQ_API_KEY to your .env file.\n"
                "Get your free key at: https://console.groq.com"
            )
    
    return True, "✅ API keys validated"


# ─── Formatting Utilities ─────────────────────────────────────────────────

def format_file_size(size_bytes: int) -> str:
    """Convert bytes to human-readable size string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 ** 2:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 ** 3:
        return f"{size_bytes / (1024**2):.1f} MB"
    else:
        return f"{size_bytes / (1024**3):.1f} GB"


def format_timestamp(dt: datetime = None) -> str:
    """
    Format a datetime as a readable timestamp.
    
    Args:
        dt: datetime object (uses current time if None)
        
    Returns:
        Formatted string like "2024-01-15 14:30:22"
    """
    if dt is None:
        dt = datetime.now()
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def truncate_text(text: str, max_length: int = 200, suffix: str = "...") -> str:
    """
    Truncate text to a maximum length with ellipsis.
    
    Args:
        text:       Input text
        max_length: Maximum character length
        suffix:     String to append when truncated
        
    Returns:
        Truncated text string
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def clean_text(text: str) -> str:
    """
    Clean extracted PDF text by removing common artifacts.
    
    PDF text extraction often includes:
    - Multiple consecutive whitespace
    - Hyphenated line breaks (re-join words)
    - Form feed characters
    
    Args:
        text: Raw extracted text
        
    Returns:
        Cleaned text string
    """
    import re
    
    # Remove form feed characters
    text = text.replace("\f", "\n")
    
    # Join hyphenated line breaks (e.g., "hyper-\ntension" → "hypertension")
    text = re.sub(r"-\n(\w)", r"\1", text)
    
    # Normalize multiple spaces to single space
    text = re.sub(r" {2,}", " ", text)
    
    # Normalize multiple newlines to double newline (paragraph break)
    text = re.sub(r"\n{3,}", "\n\n", text)
    
    return text.strip()


# ─── Caching Utilities ────────────────────────────────────────────────────

def compute_file_hash(file_path: str) -> str:
    """
    Compute MD5 hash of a file for cache invalidation.
    
    When a new PDF is uploaded, its hash changes, indicating
    the vector index needs to be rebuilt.
    
    Args:
        file_path: Path to the file
        
    Returns:
        MD5 hex digest string
    """
    hasher = hashlib.md5()
    
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    
    return hasher.hexdigest()


def compute_content_hash(content: str) -> str:
    """Compute MD5 hash of a string content."""
    return hashlib.md5(content.encode()).hexdigest()


# ─── Logging Setup ────────────────────────────────────────────────────────

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> None:
    """
    Configure loguru logging for the application.
    
    Args:
        log_level: Logging level ("DEBUG", "INFO", "WARNING", "ERROR")
        log_file:  Optional path to log file
    """
    from loguru import logger
    import sys
    
    # Remove default handler
    logger.remove()
    
    # Console handler with color
    logger.add(
        sys.stdout,
        level=log_level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        colorize=True
    )
    
    # File handler if specified
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_file,
            level=log_level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} - {message}",
            rotation="10 MB",    # Rotate when file exceeds 10MB
            retention="7 days",  # Keep logs for 7 days
            compression="zip"    # Compress old logs
        )
    
    logger.info(f"Logging configured | Level: {log_level}")


# ─── Source Citation Formatting ───────────────────────────────────────────

def format_sources_for_display(source_docs: list, scores: Optional[list] = None) -> List[Dict]:
    """
    Format retrieved source documents for UI display.
    
    Args:
        source_docs: List of LangChain Document objects
        scores:      Optional list of similarity scores
        
    Returns:
        List of formatted source dicts for Streamlit display
    """
    formatted = []
    
    for i, doc in enumerate(source_docs):
        score = scores[i] if scores and i < len(scores) else None
        
        source_info = {
            "index": i + 1,
            "source": doc.metadata.get("source", "Unknown Document"),
            "page": doc.metadata.get("page", "N/A"),
            "chunk_id": doc.metadata.get("chunk_id", i),
            "content_preview": truncate_text(doc.page_content, 300),
            "full_content": doc.page_content,
            "relevance_score": f"{score:.0%}" if score is not None else "N/A",
            "relevance_float": score
        }
        
        # Add page number offset (PyPDFLoader uses 0-indexed pages)
        if isinstance(source_info["page"], int):
            source_info["page_display"] = source_info["page"] + 1
        else:
            source_info["page_display"] = source_info["page"]
        
        formatted.append(source_info)
    
    return formatted


def get_sample_questions() -> List[str]:
    """
    Return sample medical questions for UI display and testing.
    
    Returns:
        List of example questions users might ask
    """
    return [
        "What are the symptoms of Type 2 diabetes?",
        "What medications are prescribed in this document?",
        "What are the recommended dosage instructions?",
        "What are the contraindications for this treatment?",
        "Summarize the patient's diagnosis and treatment plan",
        "What follow-up appointments are recommended?",
        "Are there any drug interactions mentioned?",
        "What lifestyle changes are recommended for hypertension?",
        "What are the side effects of the prescribed medication?",
        "What does the lab report indicate about cholesterol levels?"
    ]


# ─── Standalone usage example ────────────────────────────────────────────────
if __name__ == "__main__":
    setup_logging()
    config = load_config()
    print(f"Config loaded | Provider: {config['llm_provider']} | VectorDB: {config['vector_db']}")
    
    # Test text cleaning
    sample = "This is   a test\n\n\n\nWith multiple    spaces and hyper-\ntension"
    cleaned = clean_text(sample)
    print(f"Cleaned text: {cleaned}")
    print("Utils module ready ✓")
