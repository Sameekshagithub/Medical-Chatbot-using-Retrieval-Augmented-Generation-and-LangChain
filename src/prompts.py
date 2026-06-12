

from langchain.prompts import PromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate


# ─── System Message ────────────────────────────────────────────────────────

MEDICAL_SYSTEM_MESSAGE = """You are MediQuery AI, a specialized medical information assistant.

ROLE:
You help users understand information from their uploaded medical documents.
You are NOT a licensed medical professional and cannot replace actual medical advice.

STRICT RULES:
1. Answer ONLY using the provided context from retrieved documents
2. If the answer is not in the context, say: "I don't have enough information in the uploaded documents to answer this. Please consult a healthcare professional."
3. NEVER diagnose medical conditions
4. NEVER recommend specific medications or dosages without clear document support
5. ALWAYS recommend consulting a licensed doctor for personal medical decisions
6. For emergencies, immediately direct to emergency services (911, ER)
7. Cite the source document and page number when possible

RESPONSE FORMAT:
- Be clear, concise, and empathetic
- Use simple language (avoid unnecessary jargon)
- Structure long answers with bullet points or numbered lists
- End responses with the medical disclaimer when relevant

MEDICAL DISCLAIMER:
⚠️ This information is for educational purposes only. Always consult a qualified healthcare professional for personal medical advice, diagnosis, or treatment.
"""

# ─── Main QA Prompt Template ──────────────────────────────────────────────

QA_PROMPT_TEMPLATE = """You are MediQuery AI, a specialized medical information assistant.

IMPORTANT INSTRUCTIONS:
- Answer the question ONLY using the provided context below
- If the context does not contain enough information, say "I don't have sufficient information in the uploaded documents to answer this question accurately."
- Do not make up medical information
- Always recommend consulting a healthcare professional for personal medical decisions
- For emergencies, direct to emergency services immediately

CONTEXT FROM MEDICAL DOCUMENTS:
{context}

CONVERSATION HISTORY:
{chat_history}

CURRENT QUESTION:
{question}

ANSWER (based only on the context above):"""

# LangChain PromptTemplate object
QA_PROMPT = PromptTemplate(
    input_variables=["context", "chat_history", "question"],
    template=QA_PROMPT_TEMPLATE
)


# ─── Condense Question Prompt ─────────────────────────────────────────────
# Used in ConversationalRetrievalChain to reformulate follow-up questions
# 
# Example:
#   Chat history: "What is diabetes?"  →  "It's a metabolic disorder..."
#   Follow-up: "What are its symptoms?"
#   Condensed: "What are the symptoms of diabetes?" (standalone question)
#   Now the retriever can find relevant chunks without needing history

CONDENSE_QUESTION_TEMPLATE = """Given the following conversation history and a follow-up question, 
rephrase the follow-up question to be a standalone question that captures the full context.

IMPORTANT: 
- Keep medical terminology accurate
- Don't add medical information not present in the history
- Create a clear, specific question for document retrieval

Conversation History:
{chat_history}

Follow-up Question: {question}

Standalone Medical Question:"""

CONDENSE_QUESTION_PROMPT = PromptTemplate(
    input_variables=["chat_history", "question"],
    template=CONDENSE_QUESTION_TEMPLATE
)


# ─── Simple QA Prompt (without chat history) ──────────────────────────────

SIMPLE_QA_TEMPLATE = """You are a medical document assistant. Answer the question based ONLY on the provided context.

Context from medical documents:
{context}

Question: {question}

If the answer is not in the context, respond with:
"I don't have enough information in the uploaded documents to answer this question. Please consult a healthcare professional."

Answer:"""

SIMPLE_QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=SIMPLE_QA_TEMPLATE
)


# ─── Document Processing Prompts ──────────────────────────────────────────

SUMMARY_PROMPT_TEMPLATE = """Summarize the following medical document content in 3-5 bullet points.
Focus on: key medical information, diagnoses mentioned, treatments described, and important warnings.

Document Content:
{text}

Summary (bullet points):"""

SUMMARY_PROMPT = PromptTemplate(
    input_variables=["text"],
    template=SUMMARY_PROMPT_TEMPLATE
)


# ─── Safety Check Prompts ─────────────────────────────────────────────────

EMERGENCY_KEYWORDS = [
    "emergency", "911", "heart attack", "stroke", "overdose",
    "suicide", "poisoning", "unconscious", "not breathing",
    "severe bleeding", "anaphylaxis", "can't breathe"
]

EMERGENCY_RESPONSE = """🚨 **MEDICAL EMERGENCY DETECTED**

If this is a medical emergency, please:

1. **Call 911 (or your local emergency number) immediately**
2. **Go to the nearest Emergency Room**
3. **Call Poison Control: 1-800-222-1222** (for poisoning)
4. **Call National Suicide Prevention Lifeline: 988** (for mental health emergencies)

**I am an AI assistant and cannot provide emergency medical assistance.**

Please seek immediate professional medical help."""

DISCLAIMER = """
---
⚠️ **Medical Disclaimer**: This AI assistant provides information from uploaded medical documents for educational purposes only. It is NOT a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider for medical decisions.
"""


def get_emergency_check(query: str) -> bool:
    """
    Check if a user query contains emergency-related keywords.
    
    Args:
        query: User's question
        
    Returns:
        True if emergency keywords detected, False otherwise
    """
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in EMERGENCY_KEYWORDS)


def format_source_citation(doc, score: float = None) -> str:
    """
    Format a retrieved document chunk as a source citation.
    
    Args:
        doc:   LangChain Document object
        score: Similarity score (0-1)
        
    Returns:
        Formatted citation string for display
    """
    source = doc.metadata.get("source", "Unknown Document")
    page = doc.metadata.get("page", "N/A")
    chunk_id = doc.metadata.get("chunk_id", "")
    
    citation = f"📄 **{source}** | Page: {page}"
    if score is not None:
        citation += f" | Relevance: {score:.0%}"
    
    return citation


# ─── Standalone usage example ────────────────────────────────────────────────
if __name__ == "__main__":
    print("Prompt Templates Available:")
    print(f"  QA_PROMPT variables: {QA_PROMPT.input_variables}")
    print(f"  CONDENSE_QUESTION_PROMPT variables: {CONDENSE_QUESTION_PROMPT.input_variables}")
    print(f"  SIMPLE_QA_PROMPT variables: {SIMPLE_QA_PROMPT.input_variables}")
    print(f"  Emergency keywords: {len(EMERGENCY_KEYWORDS)}")
    print("Prompts module ready ✓")
