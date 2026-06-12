"""
============================================================
src/loader.py - PDF Document Loader Module
============================================================

PURPOSE:
    Loads PDF documents from file paths or uploaded files and
    extracts raw text with metadata (filename, page number).

WORKFLOW:
    1. Accept PDF file path(s) or Streamlit UploadedFile objects
    2. Use LangChain's PyPDFLoader to extract text page by page
    3. Return list of LangChain Document objects with metadata
    4. Handle errors gracefully (corrupted PDFs, empty files)

WHY PyPDFLoader?
    - Preserves page-level metadata (page_number)
    - Integrates natively with LangChain Document schema
    - Handles multi-page PDFs automatically
    - Lightweight and reliable for medical documents

DATA FLOW:
    PDF File → PyPDFLoader → List[Document(page_content, metadata)]
============================================================
"""

import os
import tempfile
from pathlib import Path
from typing import List, Optional, Union
from loguru import logger

from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document


class DocumentLoader:
    """
    Handles loading of PDF medical documents into LangChain Document objects.
    
    Each Document object contains:
        - page_content: The extracted text from one PDF page
        - metadata: Dict with 'source' (filename) and 'page' (page number)
    """

    def __init__(self, data_dir: str = "./data"):
        """
        Initialize the DocumentLoader.
        
        Args:
            data_dir: Directory where uploaded PDFs will be saved temporarily
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"DocumentLoader initialized. Data directory: {self.data_dir}")

    def load_pdf(self, file_path: str) -> List[Document]:
        """
        Load a single PDF file and extract text as LangChain Documents.
        
        Args:
            file_path: Absolute or relative path to the PDF file
            
        Returns:
            List of Document objects, one per PDF page
            
        Example:
            docs = loader.load_pdf("./data/medical_report.pdf")
            # docs[0].page_content → "Patient Name: John Doe..."
            # docs[0].metadata → {"source": "medical_report.pdf", "page": 0}
        """
        try:
            logger.info(f"Loading PDF: {file_path}")
            
            # PyPDFLoader splits PDF into individual pages automatically
            pdf_loader = PyPDFLoader(file_path)
            documents = pdf_loader.load()
            
            # Add filename to metadata for source citation later
            file_name = Path(file_path).name
            for doc in documents:
                doc.metadata["source"] = file_name
                doc.metadata["file_path"] = str(file_path)
            
            logger.success(f"Loaded {len(documents)} pages from '{file_name}'")
            return documents
            
        except Exception as e:
            logger.error(f"Failed to load PDF '{file_path}': {e}")
            return []

    def load_uploaded_file(self, uploaded_file) -> List[Document]:
        """
        Load a Streamlit UploadedFile object (in-memory bytes).
        
        Streamlit uploaded files are BytesIO objects, not saved to disk.
        We temporarily save them to process with PyPDFLoader.
        
        Args:
            uploaded_file: Streamlit UploadedFile object
            
        Returns:
            List of Document objects extracted from the PDF
        """
        try:
            logger.info(f"Processing uploaded file: {uploaded_file.name}")
            
            # Save uploaded bytes to a temporary file on disk
            # PyPDFLoader needs a file path, not BytesIO
            temp_path = self.data_dir / uploaded_file.name
            
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.read())
            
            # Now load from disk
            documents = self.load_pdf(str(temp_path))
            return documents
            
        except Exception as e:
            logger.error(f"Failed to process uploaded file '{uploaded_file.name}': {e}")
            return []

    def load_multiple_pdfs(self, file_paths: List[str]) -> List[Document]:
        """
        Load multiple PDF files and combine into one document list.
        
        This enables multi-PDF RAG: user can upload their prescription,
        a research paper, and a hospital report simultaneously.
        
        Args:
            file_paths: List of PDF file paths
            
        Returns:
            Combined list of Documents from all PDFs
        """
        all_documents = []
        
        for path in file_paths:
            docs = self.load_pdf(path)
            all_documents.extend(docs)
        
        logger.info(f"Total documents loaded: {len(all_documents)} pages from {len(file_paths)} files")
        return all_documents

    def load_multiple_uploads(self, uploaded_files: list) -> List[Document]:
        """
        Load multiple Streamlit uploaded files at once.
        
        Args:
            uploaded_files: List of Streamlit UploadedFile objects
            
        Returns:
            Combined list of Documents from all uploaded PDFs
        """
        all_documents = []
        
        for uploaded_file in uploaded_files:
            docs = self.load_uploaded_file(uploaded_file)
            all_documents.extend(docs)
        
        logger.info(f"Loaded {len(all_documents)} total pages from {len(uploaded_files)} uploaded files")
        return all_documents

    def get_document_stats(self, documents: List[Document]) -> dict:
        """
        Get statistics about loaded documents for UI display.
        
        Args:
            documents: List of loaded Document objects
            
        Returns:
            Dict with stats: num_pages, num_files, avg_page_length, file_names
        """
        if not documents:
            return {"num_pages": 0, "num_files": 0, "avg_page_length": 0, "file_names": []}
        
        file_names = list(set(doc.metadata.get("source", "unknown") for doc in documents))
        total_chars = sum(len(doc.page_content) for doc in documents)
        
        return {
            "num_pages": len(documents),
            "num_files": len(file_names),
            "avg_page_length": total_chars // len(documents),
            "total_characters": total_chars,
            "file_names": file_names
        }


# ─── Standalone usage example ────────────────────────────────────────────────
if __name__ == "__main__":
    loader = DocumentLoader()
    # Example: load a test PDF
    # docs = loader.load_pdf("../data/sample_medical.pdf")
    # print(f"Pages loaded: {len(docs)}")
    # print(f"First page preview: {docs[0].page_content[:200]}")
    print("DocumentLoader module ready ✓")
