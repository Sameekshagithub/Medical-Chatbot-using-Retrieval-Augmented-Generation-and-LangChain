"""
============================================================
app.py - MediRAG AI - Main Streamlit Application
============================================================

PURPOSE:
    The main entry point for the MediRAG AI medical chatbot.
    Orchestrates all modules and renders the Streamlit UI.

ARCHITECTURE:
    app.py (UI/orchestration)
        ├── src/loader.py     → PDF loading
        ├── src/splitter.py   → Text chunking
        ├── src/embeddings.py → HuggingFace embeddings
        ├── src/vectordb.py   → FAISS/ChromaDB storage
        ├── src/rag_chain.py  → LangChain RAG pipeline
        ├── src/memory.py     → Conversational memory
        ├── src/database.py   → SQLite chat persistence
        ├── src/prompts.py    → Prompt templates
        └── src/utils.py      → Helper utilities

RUN:
    streamlit run app.py
============================================================
"""

import os
import sys
import time
import uuid
from pathlib import Path
from datetime import datetime
from typing import Optional

import streamlit as st
from dotenv import load_dotenv

# ─── Load Environment Variables ────────────────────────────────────────────
load_dotenv(override=True)

# ─── Add src to path ────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

# ─── Import App Modules ─────────────────────────────────────────────────────
from src.loader import DocumentLoader
from src.splitter import TextChunker
from src.embeddings import EmbeddingGenerator
from src.vectordb import VectorDatabase
from src.rag_chain import RAGChain
from src.memory import ChatMemoryManager
from src.database import ChatDatabase, generate_session_id
from src.prompts import format_source_citation, DISCLAIMER
from src.utils import (
    load_config,
    validate_api_keys,
    format_file_size,
    format_sources_for_display,
    get_sample_questions,
    setup_logging,
    truncate_text
)
from dotenv import load_dotenv
load_dotenv()

# ─── Logging Setup ────────────────────────────────────────────────────────
setup_logging(log_level="INFO")

# ─── Page Configuration ────────────────────────────────────────────────────
st.set_page_config(
    page_title="MediRAG AI",
    page_icon="֎🇦🇮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Load Custom CSS ───────────────────────────────────────────────────────
def load_css():
    css_path = Path(__file__).parent / "assets" / "styles.css"
    if css_path.exists():
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        # Inline fallback CSS
        st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=DM+Sans:wght@400;500;600&display=swap');
        :root {
            --pink-50: #fff0f5; --pink-100: #ffe0ec; --pink-200: #ffc1d9;
            --pink-300: #ff9dbf; --pink-400: #ff6b9d; --pink-500: #e84393;
            --pink-600: #c4306e; --text-dark: #2d1b2e; --text-mid: #6b4670;
        }
        * { font-family: 'DM Sans', sans-serif !important; }
        .stApp { background: linear-gradient(135deg, #fff0f5, #fde8f0) !important; }
        </style>
        """, unsafe_allow_html=True)

load_css()

# ─── Load Config ──────────────────────────────────────────────────────────
config = load_config()

# ═══════════════════════════════════════════════════════════════════════════
# SESSION STATE INITIALIZATION
# ═══════════════════════════════════════════════════════════════════════════

def init_session_state():
    """Initialize all Streamlit session state variables."""
    
    defaults = {
        # App state
        "session_id": generate_session_id(),
        "documents_processed": False,
        "num_chunks": 0,
        "num_pages": 0,
        "num_files": 0,
        "processing_error": None,
        
        # Chat state
        "messages": [],           # List of {role, content, timestamp, sources}
        "rag_chain": None,        # Initialized RAGChain
        "memory_manager": None,   # ChatMemoryManager
        
        # UI state
        "show_sources": True,
        "dark_mode": False,
        "selected_sample": None,
        
        # Component instances (lazy-loaded)
        "_loader": None,
        "_embeddings": None,
        "_vectordb": None,
        "_chat_db": None,
    }
    
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default

init_session_state()

# COMPONENT GETTERS (Lazy-loaded singletons)

@st.cache_resource(show_spinner=False)
def get_embedding_generator(model_name: str):
    """Cache the embedding model across Streamlit reruns."""
    return EmbeddingGenerator(model_name=model_name, device="cpu")

@st.cache_resource(show_spinner=False)
def get_chat_database(db_path: str):
    """Cache the database connection."""
    return ChatDatabase(db_path=db_path)

def get_memory_manager():
    """Get or create the memory manager for this session."""
    if st.session_state.memory_manager is None:
        st.session_state.memory_manager = ChatMemoryManager(
            memory_key="chat_history",
            max_messages=20,
            return_messages=True
        )
    return st.session_state.memory_manager


# DOCUMENT PROCESSING PIPELINE
def process_documents(uploaded_files, chunk_size: int, chunk_overlap: int, k: int):
    """
    Full RAG pipeline initialization:
        Upload → Load → Chunk → Embed → Index → Create RAG Chain
    """
    progress_bar = st.progress(0, text="Starting document processing...")
    status_area = st.empty()
    
    try:
        # ── Step 1: Load PDFs ────────────────────────────────────────────
        status_area.info("📂 Loading PDF documents...")
        progress_bar.progress(10, text="Loading PDFs...")
        
        loader = DocumentLoader(data_dir="./data")
        documents = loader.load_multiple_uploads(uploaded_files)
        
        if not documents:
            st.error("❌ Failed to extract text from uploaded PDFs. Please check the files.")
            return False
        
        doc_stats = loader.get_document_stats(documents)
        st.session_state.num_pages = doc_stats["num_pages"]
        st.session_state.num_files = doc_stats["num_files"]
        
        progress_bar.progress(25, text="PDFs loaded successfully...")
        
        # ── Step 2: Chunk Text ───────────────────────────────────────────
        status_area.info("✂️ Splitting documents into chunks...")
        progress_bar.progress(35, text="Chunking text...")
        
        chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = chunker.split_documents(documents)
        chunk_stats = chunker.get_chunk_stats(chunks)
        st.session_state.num_chunks = chunk_stats["total_chunks"]
        
        progress_bar.progress(50, text=f"Created {len(chunks)} chunks...")
        
        # ── Step 3: Load Embeddings ──────────────────────────────────────
        status_area.info("🧠 Loading embedding model (first time: ~30s download)...")
        progress_bar.progress(60, text="Loading embeddings model...")
        
        embedding_gen = get_embedding_generator(config["embedding_model"])
        embeddings = embedding_gen.get_embeddings()
        
        progress_bar.progress(70, text="Embedding model ready...")
        
        # ── Step 4: Create Vector Index ──────────────────────────────────
        status_area.info("🗄️ Building vector database...")
        progress_bar.progress(75, text="Indexing chunks in FAISS...")
        
        vector_db = VectorDatabase(
            embeddings=embeddings,
            db_type=config["vector_db"],
            persist_directory=config["vector_db_path"]
        )
        vector_db.create_from_documents(chunks)
        
        progress_bar.progress(87, text="Vector index created...")
        
        # ── Step 5: Create Retriever ─────────────────────────────────────
        retriever = vector_db.get_retriever(k=k)
        
        # ── Step 6: Initialize RAG Chain ─────────────────────────────────
        status_area.info("🤖 Connecting LLM and building RAG chain...")
        progress_bar.progress(93, text="Building RAG chain...")
        
        memory_manager = get_memory_manager()
        
        rag_chain = RAGChain(
            retriever=retriever,
            llm_provider=config["llm_provider"],
            temperature=config["llm_temperature"],
            max_tokens=config["max_tokens"],
            memory=memory_manager.get_langchain_memory(),
            return_source_documents=True
        )
        
        st.session_state.rag_chain = rag_chain
        st.session_state.documents_processed = True
        st.session_state.processing_error = None
        
        progress_bar.progress(100, text="✅ Processing complete!")
        status_area.success(
            f"✅ Successfully processed {doc_stats['num_files']} file(s) → "
            f"{doc_stats['num_pages']} pages → {chunk_stats['total_chunks']} chunks → ready!"
        )
        
        time.sleep(1.5)
        progress_bar.empty()
        status_area.empty()
        return True
        
    except ValueError as e:
        # API key errors
        progress_bar.empty()
        status_area.empty()
        st.session_state.processing_error = str(e)
        st.error(f"🔑 Configuration Error:\n{str(e)}")
        return False
        
    except Exception as e:
        progress_bar.empty()
        status_area.empty()
        st.session_state.processing_error = str(e)
        st.error(f"❌ Processing failed: {str(e)}\n\nCheck the logs for details.")
        return False


# ═══════════════════════════════════════════════════════════════════════════
# CHAT FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def handle_user_message(user_input: str):
    """
    Process a user message through the RAG pipeline and update UI.
    
    Steps:
        1. Add user message to chat history
        2. Show typing indicator
        3. Query RAG chain
        4. Add AI response to chat history
        5. Save to database
        6. Trigger UI rerun
    """
    if not user_input.strip():
        return
    
    memory_manager = get_memory_manager()
    chat_db = get_chat_database(config["sqlite_db_path"])
    
    # Add user message
    timestamp = datetime.now().strftime("%H:%M")
    st.session_state.messages.append({
        "role": "user",
        "content": user_input,
        "timestamp": timestamp,
        "sources": []
    })
    memory_manager.add_user_message(user_input)
    
    # Save to database
    chat_db.save_message(st.session_state.session_id, "user", user_input)
    
    # Query RAG chain
    with st.spinner("🔍 Searching medical documents..."):
        result = st.session_state.rag_chain.query(user_input)
    
    answer = result.get("answer", "I couldn't generate a response.")
    source_docs = result.get("source_documents", [])
    
    # Format sources for display
    formatted_sources = format_sources_for_display(source_docs)
    
    # Add AI response
    timestamp = datetime.now().strftime("%H:%M")
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "timestamp": timestamp,
        "sources": formatted_sources,
        "is_emergency": result.get("is_emergency", False)
    })
    
    memory_manager.add_ai_message(answer, formatted_sources)
    
    # Save AI response to database
    source_strings = [f"{s['source']} (p.{s['page_display']})" for s in formatted_sources]
    chat_db.save_message(
        st.session_state.session_id,
        "assistant",
        answer,
        sources=source_strings
    )
    
    st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
# UI RENDERING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def render_header():
    """Render the app header with logo and badges."""
    st.markdown("""
    <div class="mediquery-header">
        <div class="mediquery-logo"> ֎🇦🇮 MediRAG <span>Query</span> AI</div>
        <div class="mediquery-tagline">Medical Document Intelligence — Powered by RAG + LangChain</div>
        <div style="display:flex; justify-content:center; gap:8px; flex-wrap:wrap; margin-top:0.6rem;">
            <span class="mediquery-badge">🔒 Privacy-First</span>
            <span class="mediquery-badge">📄 RAG Architecture</span>
            <span class="mediquery-badge">⚕️ Medical AI</span>
            <span class="mediquery-badge">🧠 HuggingFace</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar():
    """Render the sidebar with upload controls and statistics."""
    
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; padding:0.5rem 0 1rem;">
            <div style="font-family:'Playfair Display',serif; font-size:1.3rem; color:#e84393; font-weight:700;">
                ֎🇦🇮 MediRAG AI
            </div>
            <div style="font-size:0.72rem; color:#9b7ba0; margin-top:2px;">v1.0.0 — Medical RAG Chatbot</div>
        </div>
        """, unsafe_allow_html=True)
        
        # ── Document Upload Section ──────────────────────────────────────
        st.markdown("### 📄 Upload Documents")
        
        uploaded_files = st.file_uploader(
            "Upload Medical PDFs",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload prescriptions, reports, research papers, treatment guides, or any medical PDF",
            label_visibility="collapsed"
        )
        
        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} file(s) selected")
            for f in uploaded_files:
                size_str = format_file_size(f.size)
                st.markdown(f"""
                <div style="background:#fff0f5; border:1px solid #ffc1d9; border-radius:8px; 
                            padding:0.4rem 0.7rem; margin:0.2rem 0; font-size:0.78rem; 
                            color:#6b4670; display:flex; justify-content:space-between;">
                    <span>📄 {f.name[:28]}{'...' if len(f.name)>28 else ''}</span>
                    <span style="color:#e84393; font-weight:600;">{size_str}</span>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # ── RAG Settings ─────────────────────────────────────────────────
        with st.expander("⚙️ RAG Settings", expanded=False):
            chunk_size = st.slider(
                "Chunk Size (chars)",
                min_value=200, max_value=2000,
                value=config["chunk_size"],
                step=100,
                help="Size of each text chunk. Smaller = more precise retrieval"
            )
            
            chunk_overlap = st.slider(
                "Chunk Overlap (chars)",
                min_value=0, max_value=500,
                value=config["chunk_overlap"],
                step=50,
                help="Overlap between chunks to preserve context at boundaries"
            )
            
            retrieval_k = st.slider(
                "Retrieved Chunks (k)",
                min_value=1, max_value=10,
                value=config["retrieval_k"],
                step=1,
                help="Number of document chunks to retrieve per query"
            )
            
            llm_provider = st.selectbox(
                "LLM Provider",
                ["groq", "openai"],
                index=0 if config["llm_provider"] == "groq" else 1,
                help="Groq is free and fast. OpenAI requires paid API key."
            )
            
            vector_db_type = st.selectbox(
                "Vector Database",
                ["faiss", "chroma"],
                index=0 if config["vector_db"] == "faiss" else 1
            )
        
        # ── Process Button ───────────────────────────────────────────────
        if uploaded_files:
            if st.button(
                " Process Documents" if not st.session_state.documents_processed else "🔄 Re-Process Documents",
                use_container_width=True,
                type="primary"
            ):
                # Update config with slider values
                config["chunk_size"] = chunk_size
                config["chunk_overlap"] = chunk_overlap
                config["retrieval_k"] = retrieval_k
                config["llm_provider"] = llm_provider
                config["vector_db"] = vector_db_type
                
                # Validate API keys before processing
                is_valid, msg = validate_api_keys(config)
                if not is_valid:
                    st.error(msg)
                else:
                    process_documents(
                        uploaded_files,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        k=retrieval_k
                    )
        else:
            st.info("👆 Upload PDFs to get started")
        
        st.markdown("---")
        
        # ── Statistics ───────────────────────────────────────────────────
        st.markdown("### 📊 Session Stats")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Files", st.session_state.num_files)
            st.metric("Chunks", st.session_state.num_chunks)
        with col2:
            st.metric("Pages", st.session_state.num_pages)
            st.metric("Messages", len(st.session_state.messages))
        
        st.markdown("---")
        
        # ── Controls ─────────────────────────────────────────────────────
        st.markdown("### 🎛️ Controls")
        
        show_sources = st.checkbox(
            "Show Source Citations",
            value=st.session_state.show_sources
        )
        st.session_state.show_sources = show_sources
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.messages = []
                memory_manager = get_memory_manager()
                memory_manager.clear()
                st.rerun()
        
        with col2:
            if st.session_state.messages:
                memory_manager = get_memory_manager()
                chat_text = memory_manager.export_to_text()
                st.download_button(
                    "💾 Export",
                    data=chat_text,
                    file_name=f"mediquery_chat_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
        
        st.markdown("---")
        
        # ── LLM Status ───────────────────────────────────────────────────
        provider_icon = "🟢" if st.session_state.documents_processed else "🔴"
        st.markdown(f"""
        <div style="background:white; border:1.5px solid #ffe0ec; border-radius:10px; 
                    padding:0.8rem; font-size:0.8rem; color:#6b4670;">
            <div style="font-weight:600; color:#e84393; margin-bottom:6px;">System Status</div>
            <div>{provider_icon} LLM: {config['llm_provider'].upper()}</div>
            <div>{'🟢' if st.session_state.documents_processed else '🔴'} Vector DB: {config['vector_db'].upper()}</div>
            <div>🔵 Embeddings: MiniLM-L6</div>
        </div>
        """, unsafe_allow_html=True)
        
        return uploaded_files


def render_message(msg: dict, index: int):
    """Render a single chat message bubble with sources."""
    is_user = msg["role"] == "user"
    
    if is_user:
        # User message (right-aligned, pink bubble)
        st.markdown(f"""
        <div class="message-row user">
            <div class="message-avatar avatar-user">👤</div>
            <div>
                <div class="message-bubble bubble-user">
                    {msg['content'].replace(chr(10), '<br>')}
                    <span class="message-time">{msg.get('timestamp', '')}</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # AI message (left-aligned, white bubble)
        is_emergency = msg.get("is_emergency", False)
        bubble_style = "bubble-bot"
        
        st.markdown(f"""
        <div class="message-row">
            <div class="message-avatar avatar-bot">🌸</div>
            <div style="flex:1; max-width:76%;">
                <div class="message-bubble {bubble_style}">
                    {msg['content'].replace(chr(10), '<br>')}
                    <span class="message-time" style="color:#9b7ba0;">{msg.get('timestamp', '')}</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Source citations
        if st.session_state.show_sources and msg.get("sources"):
            with st.expander(f"📚 View Sources ({len(msg['sources'])} documents retrieved)", expanded=False):
                for source in msg["sources"]:
                    relevance_color = "#e84393" if source.get("relevance_float", 0) and source.get("relevance_float") > 0.7 else "#9b7ba0"
                    st.markdown(f"""
                    <div class="source-card">
                        <div class="source-card-header">
                            <span class="source-filename">📄 {source['source']}</span>
                            <span class="source-badge">Page {source['page_display']} | {source['relevance_score']}</span>
                        </div>
                        <div class="source-preview">{source['content_preview']}</div>
                    </div>
                    """, unsafe_allow_html=True)


def render_welcome_screen():
    """Render the welcome screen shown before documents are uploaded."""
    st.markdown("""
    <div class="welcome-card">
        <h2>👋 Welcome to MediQuery AI</h2>
        <p style="color:#9b7ba0; font-size:0.9rem; margin-bottom:1.5rem;">
            Upload your medical documents and ask questions in natural language.<br>
            Powered by RAG architecture for accurate, grounded answers.
        </p>
        <div class="welcome-steps">
            <div class="welcome-step">
                <span class="step-icon">📄</span>
                <div class="step-label"><strong>1. Upload PDFs</strong><br>Prescriptions, reports,<br>research papers</div>
            </div>
            <div class="welcome-step">
                <span class="step-icon">⚡</span>
                <div class="step-label"><strong>2. Process</strong><br>Click "Process Documents"<br>to index your files</div>
            </div>
            <div class="welcome-step">
                <span class="step-icon">💬</span>
                <div class="step-label"><strong>3. Ask Questions</strong><br>Chat naturally about<br>your documents</div>
            </div>
            <div class="welcome-step">
                <span class="step-icon">📚</span>
                <div class="step-label"><strong>4. Cite Sources</strong><br>Every answer shows<br>which page it's from</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Sample questions
    st.markdown("""
    <div style="margin-top:1.5rem;">
        <div style="font-family:'Playfair Display',serif; color:#e84393; font-size:1rem; 
                    font-weight:600; margin-bottom:0.8rem;">
            💡 Sample Questions You Can Ask
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    questions = get_sample_questions()
    cols = st.columns(2)
    for i, question in enumerate(questions[:8]):
        with cols[i % 2]:
            if st.button(f"❓ {question}", key=f"sample_{i}", use_container_width=True):
                st.session_state.selected_sample = question
    
    # Disclaimer
    st.markdown("""
    <div class="disclaimer-box">
        <strong>⚠️ Medical Disclaimer</strong><br>
        MediQuery AI is an AI-powered document assistant for educational purposes only.
        It is <strong>NOT</strong> a licensed medical professional and cannot replace professional medical advice,
        diagnosis, or treatment. Always consult a qualified healthcare provider for medical decisions.
        In case of emergencies, call <strong>911</strong> or your local emergency number immediately.
    </div>
    """, unsafe_allow_html=True)


def render_chat_interface():
    """Render the main chat interface."""
    
    # ── Chat Header ──────────────────────────────────────────────────────
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        st.markdown("""
        <div style="font-family:'Playfair Display',serif; font-size:1.3rem; 
                    color:#e84393; font-weight:700; padding:0.3rem 0;">
            💬 Medical Chat
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div style="background:#fff0f5; border:1px solid #ffc1d9; border-radius:20px; 
                    padding:0.3rem 0.8rem; text-align:center; font-size:0.75rem; 
                    color:#e84393; font-weight:600; margin-top:0.3rem;">
            {st.session_state.num_chunks} chunks indexed
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div style="background:#fff0f5; border:1px solid #ffc1d9; border-radius:20px; 
                    padding:0.3rem 0.8rem; text-align:center; font-size:0.75rem; 
                    color:#e84393; font-weight:600; margin-top:0.3rem;">
            {st.session_state.num_files} file(s)
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ── Display Messages ─────────────────────────────────────────────────
    if not st.session_state.messages:
        # Show first-chat prompt
        st.markdown("""
        <div style="text-align:center; padding:2.5rem 1rem; color:#9b7ba0;">
            <div style="font-size:2.5rem; margin-bottom:1rem;">🌸</div>
            <div style="font-family:'Playfair Display',serif; font-size:1.2rem; 
                        color:#e84393; font-weight:600; margin-bottom:0.5rem;">
                Documents indexed successfully!
            </div>
            <div style="font-size:0.88rem;">
                Ask me anything about your medical documents below.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Quick sample questions
        st.markdown("**💡 Quick Questions:**")
        quick_qs = get_sample_questions()[:4]
        cols = st.columns(2)
        for i, q in enumerate(quick_qs):
            with cols[i % 2]:
                if st.button(f"❓ {q}", key=f"chat_sample_{i}", use_container_width=True):
                    handle_user_message(q)
    else:
        # Render all messages
        for i, msg in enumerate(st.session_state.messages):
            render_message(msg, i)
    
    st.markdown("---")
    
    # ── Chat Input ───────────────────────────────────────────────────────
    
    # Handle selected sample question
    prefill = st.session_state.get("selected_sample", "")
    if prefill:
        st.session_state.selected_sample = None  # Clear after use
    
    user_input = st.chat_input(
        placeholder="Ask a medical question about your documents... (e.g., 'What medications are prescribed?')",
    )
    
    if user_input:
        handle_user_message(user_input)
    elif prefill:
        handle_user_message(prefill)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN APP LAYOUT
# ═══════════════════════════════════════════════════════════════════════════

def main():
    """Main application entry point."""
    
    # ── Header ──────────────────────────────────────────────────────────
    render_header()
    
    # ── Sidebar ─────────────────────────────────────────────────────────
    uploaded_files = render_sidebar()
    
    # ── Main Content Area ────────────────────────────────────────────────
    
    # API Key warning
    is_valid, msg = validate_api_keys(config)
    if not is_valid:
        st.warning(f"""
        **🔑 API Key Required**
        
        {msg}
        
        1. Copy `.env.example` to `.env`
        2. Add your API key
        3. Restart the app: `streamlit run app.py`
        """)
    
    # Render appropriate view
    if st.session_state.documents_processed and st.session_state.rag_chain:
        render_chat_interface()
    else:
        if not uploaded_files:
            render_welcome_screen()
        else:
            # Files uploaded but not yet processed
            st.info("👆 Click **'🚀 Process Documents'** in the sidebar to index your PDFs and start chatting!")
            
            st.markdown("""
            <div style="background:white; border:1.5px solid #ffe0ec; border-radius:16px; 
                        padding:1.5rem; margin-top:1rem; text-align:center;">
                <div style="font-size:2rem; margin-bottom:0.8rem;">⚡</div>
                <div style="font-family:'Playfair Display',serif; color:#e84393; font-weight:600; margin-bottom:0.5rem;">
                    Ready to Process
                </div>
                <div style="font-size:0.85rem; color:#9b7ba0;">
                    {n} PDF(s) selected. Configure settings in the sidebar and click "Process Documents".
                </div>
            </div>
            """.replace("{n}", str(len(uploaded_files))), unsafe_allow_html=True)
            render_welcome_screen()
    
    # ── Footer ───────────────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center; padding:2rem 0 1rem; color:#c4b0c8; font-size:0.75rem;">
        🌸 MediQuery AI v1.0.0 &nbsp;|&nbsp; Built with LangChain + FAISS + HuggingFace &nbsp;|&nbsp; 
        <span style="color:#e84393;">Not for clinical use</span>
    </div>
    """, unsafe_allow_html=True)


# ─── Entry Point ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
