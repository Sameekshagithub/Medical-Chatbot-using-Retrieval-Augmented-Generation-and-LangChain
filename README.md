# 🌸 MediQuery AI — Medical RAG Chatbot

> **Retrieval-Augmented Generation for Medical Document Q&A**  
> Built with LangChain · FAISS · HuggingFace · Streamlit · Groq/OpenAI

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-0.2-green)](https://python.langchain.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.38-red)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-pink)](LICENSE)

---

## 📋 Table of Contents

- [What is MediQuery AI?](#what-is-mediquery-ai)
- [What is RAG?](#what-is-rag)
- [Architecture](#architecture)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the App](#running-the-app)
- [Sample Questions](#sample-questions)
- [Deployment](#deployment)
- [Future Improvements](#future-improvements)
- [Medical Disclaimer](#medical-disclaimer)

---

## What is MediQuery AI?

MediQuery AI is a **production-grade medical document chatbot** that uses **Retrieval-Augmented Generation (RAG)** to answer healthcare-related questions strictly from uploaded PDF documents.

**Key benefits over generic ChatGPT:**
- ✅ Answers only from YOUR medical documents (no hallucination)
- ✅ Cites exact sources (page number, document name)
- ✅ Works with your private prescriptions, lab reports, clinical guidelines
- ✅ Conversational memory (follow-up questions work naturally)
- ✅ Emergency safety detection

**Supported document types:**
- 💊 Prescriptions
- 🧪 Lab Reports
- 📖 Research Papers
- 🏥 Hospital Discharge Summaries
- 📋 Treatment Protocols
- 📚 Clinical Guidelines & Medical Textbooks

---

## What is RAG?

**Retrieval-Augmented Generation** is an AI architecture that:

```
User Question
     ↓
[Embed Question] → 384-dim vector
     ↓
[FAISS Search]  → Top-5 relevant document chunks
     ↓
[Build Prompt]  → "Context: {chunks} | Question: {query}"
     ↓
[LLM (Groq)]   → Grounded answer
     ↓
User sees: Answer + Source Citations
```

**Why RAG for Healthcare?**
| Problem | RAG Solution |
|---------|-------------|
| LLMs hallucinate medical facts | Only answers from retrieved documents |
| Generic training data | Your specific patient documents |
| No source attribution | Cites exact page and document |
| Static knowledge | Add new PDFs anytime |
| Context window limits | Chunks fit any LLM |

---

## Architecture

```
medical-rag-chatbot/
│
├── app.py                  ← Streamlit UI + Orchestration
│
├── src/
│   ├── loader.py           ← PDF → LangChain Documents
│   ├── splitter.py         ← Documents → Overlapping Chunks
│   ├── embeddings.py       ← Chunks → 384-dim Vectors
│   ├── vectordb.py         ← FAISS/Chroma Index + Retriever
│   ├── rag_chain.py        ← ConversationalRetrievalChain
│   ├── memory.py           ← ConversationBufferMemory
│   ├── prompts.py          ← Medical Prompt Templates
│   ├── database.py         ← SQLite Chat History
│   └── utils.py            ← Config, Logging, Helpers
│
├── assets/
│   └── styles.css          ← Light Pink Medical Theme
│
├── data/                   ← Uploaded PDFs (auto-created)
├── vectorstore/            ← FAISS/Chroma Index (auto-created)
├── chat_history/           ← SQLite DB (auto-created)
│
├── notebooks/
│   └── experimentation.ipynb
│
├── requirements.txt
├── .env.example
└── README.md
```

**Data Flow:**
```
PDF Upload → PyPDFLoader → RecursiveCharacterTextSplitter
          → HuggingFaceEmbeddings (all-MiniLM-L6-v2)
          → FAISS Index (saved to disk)
          → VectorStoreRetriever (top-k=5)
          → ConversationalRetrievalChain (Groq/OpenAI LLM)
          → Answer + Source Documents → Streamlit UI
```

---

## Features

| Feature | Implementation |
|---------|---------------|
| PDF Ingestion | PyPDFLoader (multi-file, multi-page) |
| Text Chunking | RecursiveCharacterTextSplitter (configurable size/overlap) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (384-dim) |
| Vector DB | FAISS (default) or ChromaDB |
| LLM | Groq (free, fast) or OpenAI GPT-4 |
| RAG Chain | ConversationalRetrievalChain |
| Memory | ConversationBufferMemory |
| Source Citation | Page number + similarity score + preview |
| Chat Storage | SQLite via SQLAlchemy |
| Safety | Emergency keyword detection |
| UI | Streamlit + custom light pink CSS theme |
| Export | Download chat history as .txt |

---

## Project Structure

```
medical-rag-chatbot/
├── app.py                  Main Streamlit application
├── requirements.txt        All Python dependencies (explained)
├── .env.example            Template for environment variables
├── README.md               This file
│
├── src/
│   ├── __init__.py
│   ├── loader.py           DocumentLoader class
│   ├── splitter.py         TextChunker class
│   ├── embeddings.py       EmbeddingGenerator class
│   ├── vectordb.py         VectorDatabase class
│   ├── rag_chain.py        RAGChain class
│   ├── memory.py           ChatMemoryManager class
│   ├── prompts.py          Prompt templates + safety checks
│   ├── database.py         ChatDatabase (SQLite)
│   └── utils.py            Config, logging, formatters
│
├── assets/
│   └── styles.css          Custom light pink CSS theme
│
├── data/                   (auto-created) Uploaded PDF storage
├── vectorstore/            (auto-created) FAISS/Chroma indices
├── chat_history/           (auto-created) SQLite database
│
└── notebooks/
    └── experimentation.ipynb
```

---

## Installation

### Prerequisites
- Python 3.10 or higher
- pip package manager
- Git

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/medical-rag-chatbot.git
cd medical-rag-chatbot
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Mac/Linux)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate

# Verify activation
which python  # Should show path with 'venv'
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

> ⏱️ First install takes 3-5 minutes. The HuggingFace model (~80MB) downloads on first run.

### Step 4: Configure Environment

```bash
# Copy the template
cp .env.example .env

# Edit with your API keys
nano .env  # or code .env / notepad .env
```

---

## Configuration

Edit your `.env` file:

```bash
# Choose your LLM provider:

# Option A: Groq (FREE, ultra-fast, recommended for beginners)
GROQ_API_KEY=gsk_your_groq_key_here
LLM_PROVIDER=groq
GROQ_MODEL=llama-3.1-70b-versatile

# Option B: OpenAI (paid, most powerful)
OPENAI_API_KEY=sk-your_openai_key_here
LLM_PROVIDER=openai
OPENAI_MODEL=gpt-4o
```

**Getting API Keys:**
- **Groq (Free):** https://console.groq.com → Sign up → API Keys → Create
- **OpenAI (Paid):** https://platform.openai.com/api-keys → Create new key

---

## Running the App

```bash
# Make sure virtual environment is active
source venv/bin/activate

# Run Streamlit
streamlit run app.py
```

The app opens at: **http://localhost:8501**

---

## Sample Questions for Testing

After uploading a medical PDF, try these questions:

```
Clinical Questions:
- "What are the symptoms of the condition described?"
- "What medications are prescribed and at what dosage?"
- "What are the contraindications mentioned?"
- "What follow-up tests are recommended?"
- "Summarize the patient's diagnosis"

Drug & Treatment:
- "What are the side effects of [medication name]?"
- "Are there any drug interactions mentioned?"
- "What is the recommended treatment protocol?"
- "What dietary restrictions are advised?"

Lab Reports:
- "What do the lab results indicate?"
- "Are any values outside normal range?"
- "What does the cholesterol level indicate?"
- "What follow-up is recommended based on these results?"
```

---

## Deployment

### Streamlit Cloud (Free)

1. Push to GitHub:
```bash
git init
git add .
git commit -m "Initial commit: Medical RAG Chatbot"
git remote add origin https://github.com/YOUR_USERNAME/medical-rag-chatbot.git
git push -u origin main
```

2. Go to https://share.streamlit.io
3. Connect your GitHub repo
4. Set **Main file path:** `app.py`
5. Add secrets in **Advanced Settings → Secrets:**
```toml
GROQ_API_KEY = "your_key_here"
LLM_PROVIDER = "groq"
GROQ_MODEL = "llama-3.1-70b-versatile"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_DB = "faiss"
```

### Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501"]
```

```bash
docker build -t mediquery-ai .
docker run -p 8501:8501 --env-file .env mediquery-ai
```

---

## Future Improvements

- [ ] **OCR Support** — Process scanned PDFs with Tesseract
- [ ] **Multi-language** — Support non-English medical documents
- [ ] **Audio Input** — Voice questions via Whisper API
- [ ] **Graph RAG** — Build knowledge graphs from medical entities
- [ ] **Fine-tuned Embeddings** — Medical-specific embedding model (BioMedBERT)
- [ ] **Hybrid Search** — Combine BM25 keyword + semantic vector search
- [ ] **User Authentication** — Multi-user support with separate document stores
- [ ] **DICOM Support** — Process medical imaging metadata
- [ ] **API Mode** — FastAPI backend for mobile app integration
- [ ] **Evaluation Pipeline** — RAGAS metrics for retrieval quality

---

## Technical Concepts Explained

### Why Chunk Size = 1000 chars?
- Too small (< 200): Loses context, retrieval returns fragments
- Too large (> 2000): Dilutes relevance, may exceed LLM context
- 1000 chars ≈ 150-200 words ≈ one medical paragraph ✓

### Why MiniLM-L6-v2 for Embeddings?
- 80MB vs 1.5GB for larger models
- Runs on CPU (no GPU needed)
- 384 dimensions: fast FAISS search
- Trained on 1B sentence pairs: understands medical terminology
- Free: no API calls, runs locally

### Why FAISS over ChromaDB?
- FAISS: in-memory, blazing fast, great for < 100k chunks
- ChromaDB: SQLite-backed, metadata filtering, better for large datasets
- For a medical document chatbot: FAISS is sufficient and simpler

---

## Medical Disclaimer

> ⚠️ **IMPORTANT:** MediQuery AI is an AI-powered tool designed for educational and informational purposes only. It is **NOT** a licensed medical professional and should **NOT** be used for:
> - Clinical diagnosis
> - Treatment decisions
> - Emergency medical situations
> - Prescribing medications
> 
> Always consult a qualified, licensed healthcare provider for medical advice, diagnosis, or treatment. In case of a medical emergency, call **911** (or your local emergency number) immediately.

---

## License

MIT License — Free for personal and commercial use.

---

*Built with ❤️ using LangChain, HuggingFace, FAISS, Groq, and Streamlit*
