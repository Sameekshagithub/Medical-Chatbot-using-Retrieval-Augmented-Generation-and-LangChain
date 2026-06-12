"""
RAG PIPELINE FLOW:
    User Question
         ↓
    [Embedding Model] → Query Vector
         ↓
    [Vector DB] → Top-5 Relevant Chunks
         ↓
    [Prompt Template] → "Context: ... Question: ..."
         ↓
    [LLM (GPT-4/Llama)] → Generated Answer
         ↓
    [Memory] → Store conversation
         ↓
    User sees: Answer + Source Citations
"""

import os
from typing import Optional, Dict, Any, List, Tuple
from loguru import logger

from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.schema import BaseRetriever
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory

from src.prompts import (
    QA_PROMPT, 
    CONDENSE_QUESTION_PROMPT, 
    SIMPLE_QA_PROMPT,
    get_emergency_check,
    EMERGENCY_RESPONSE
)


class RAGChain:
    """
    Complete RAG pipeline for medical document Q&A.
    
    Supports:
        - OpenAI GPT-4/GPT-3.5
        - Groq (Llama3, Mixtral) - free and ultra-fast
        - Conversational memory (follow-up questions)
        - Source document retrieval and citation
        - Emergency safety checks
    """

    # Supported LLM configurations
    SUPPORTED_PROVIDERS = {
        "openai": {
            "models": ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
            "class": ChatOpenAI
        },
        "groq": {
            "models": ["llama-3.1-8b-instant", "mixtral-8x7b-32768", "gemma-7b-it"],
            "class": ChatGroq
        }
    }

    def __init__(
        self,
        retriever: BaseRetriever,
        llm_provider: str = "groq",
        model_name: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        memory: Optional[ConversationBufferMemory] = None,
        return_source_documents: bool = True
    ):
        """
        Initialize the RAG pipeline.
        
        Args:
            retriever:               Vector DB retriever (returns relevant chunks)
            llm_provider:            "openai" or "groq"
            model_name:              Specific model name (uses default if None)
            temperature:             LLM creativity (0.0=deterministic, 1.0=creative)
                                     Use 0.1 for medical accuracy
            max_tokens:              Maximum tokens in LLM response
            memory:                  LangChain ConversationBufferMemory object
            return_source_documents: Whether to return source chunks with answer
        """
        self.retriever = retriever
        self.llm_provider = llm_provider.lower()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.return_source_documents = return_source_documents
        
        # Initialize LLM
        self.llm = self._initialize_llm(
            provider=self.llm_provider,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Build the conversational RAG chain
        self.memory = memory or ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        self.chain = self._build_chain()
        
        logger.success(
            f"RAGChain initialized | "
            f"Provider: {llm_provider} | "
            f"Model: {self.llm.model_name if hasattr(self.llm, 'model_name') else 'unknown'} | "
            f"Temperature: {temperature}"
        )

    def _initialize_llm(
        self,
        provider: str,
        model_name: Optional[str],
        temperature: float,
        max_tokens: int
    ):
        """
        Initialize the Language Model based on provider.
        
        Args:
            provider:   "openai" or "groq"
            model_name: Specific model to use
            temperature: Sampling temperature
            max_tokens:  Max response length
            
        Returns:
            LangChain-compatible LLM instance
        """
        if provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key or api_key ==  "your_openai_api_key_here":
                raise ValueError(
                    "OPENAI_API_KEY not set. Add it to your .env file.\n"
                    "Get your key at: https://platform.openai.com/api-keys"
                )
            
            model = model_name or os.getenv("OPENAI_MODEL", "gpt-4o")
            
            llm = ChatOpenAI(
                model_name=model,
                temperature=temperature,
                max_tokens=max_tokens,
                openai_api_key=api_key
            )
            logger.info(f"OpenAI LLM initialized: {model}")
            return llm
            
        elif provider == "groq":
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key or api_key == "your_groq_api_key_here":
                raise ValueError(
                    "GROQ_API_KEY not set. Add it to your .env file.\n"
                    "Get your free key at: https://console.groq.com"
                )
            
            model = model_name or os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
            
            llm = ChatGroq(
                model_name=model,
                temperature=temperature,
                max_tokens=max_tokens,
                groq_api_key=api_key
            )
            logger.info(f"Groq LLM initialized: {model}")
            return llm
        
        else:
            raise ValueError(f"Unsupported provider: {provider}. Use 'openai' or 'groq'")

    def _build_chain(self) -> ConversationalRetrievalChain:
        """
        Build the ConversationalRetrievalChain.
        
        This chain:
            1. Takes user question + chat history
            2. Condenses to a standalone question (for retrieval)
            3. Retrieves relevant chunks from vector DB
            4. Generates answer using LLM + retrieved context
            5. Stores Q&A in memory for future turns
        
        Returns:
            LangChain ConversationalRetrievalChain
        """
        chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            memory=self.memory,
            
            # Prompt for condensing follow-up questions
            condense_question_prompt=CONDENSE_QUESTION_PROMPT,
            
            # Prompt for final Q&A with context
            combine_docs_chain_kwargs={"prompt": SIMPLE_QA_PROMPT},
            
            # Return source documents for citation
            return_source_documents=self.return_source_documents,
            
            # Verbose logging during development
            verbose=False,
            
            # Output key for memory storage
            output_key="answer"
        )
        
        logger.info("ConversationalRetrievalChain built successfully")
        return chain

    def query(self, question: str) -> Dict[str, Any]:
        """
        Process a user's medical question through the RAG pipeline.
        
        This is the main method called for every user interaction.
        
        Pipeline:
            1. Safety check (emergency keywords)
            2. Embed question → vector
            3. Retrieve top-k similar chunks
            4. Build prompt: context + history + question
            5. LLM generates answer
            6. Return answer + sources
        
        Args:
            question: User's medical question
            
        Returns:
            Dict with:
                - "answer": Generated text response
                - "source_documents": List of retrieved Document chunks
                - "question": Original question
        """
        # Step 1: Emergency safety check
        if get_emergency_check(question):
            logger.warning(f"Emergency keywords detected in query: {question[:50]}...")
            return {
                "answer": EMERGENCY_RESPONSE,
                "source_documents": [],
                "question": question,
                "is_emergency": True
            }
        
        try:
            logger.info(f"Processing query: '{question[:80]}...' " if len(question) > 80 else f"Processing query: '{question}'")
            
            # Step 2-5: Run through RAG chain
            # The chain handles: embedding → retrieval → prompting → LLM → response
            result = self.chain.invoke({
                "question": question
            })
            
            answer = result.get("answer", "I couldn't generate a response. Please try again.")
            source_docs = result.get("source_documents", [])
            
            logger.success(
                f"Query answered | "
                f"Answer length: {len(answer)} chars | "
                f"Sources retrieved: {len(source_docs)}"
            )
            
            return {
                "answer": answer,
                "source_documents": source_docs,
                "question": question,
                "is_emergency": False
            }
            
        except Exception as e:
            logger.error(f"RAG query failed: {e}")
            return {
                "answer": f"An error occurred while processing your question: {str(e)}\n\nPlease check your API key and try again.",
                "source_documents": [],
                "question": question,
                "is_emergency": False,
                "error": str(e)
            }

    def get_chain_info(self) -> dict:
        """Return information about the current RAG chain configuration."""
        return {
            "llm_provider": self.llm_provider,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "return_sources": self.return_source_documents,
            "memory_type": "ConversationBufferMemory"
        }


def create_rag_chain(
    retriever: BaseRetriever,
    llm_provider: str = "groq",
    memory: Optional[ConversationBufferMemory] = None,
    **kwargs
) -> RAGChain:
    """
    Factory function to create a RAGChain with sensible defaults.
    
    Args:
        retriever:    Vector DB retriever
        llm_provider: "openai" or "groq"
        memory:       Optional pre-initialized memory
        **kwargs:     Additional arguments passed to RAGChain
        
    Returns:
        Configured RAGChain instance
    """
    return RAGChain(
        retriever=retriever,
        llm_provider=llm_provider,
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.1")),
        max_tokens=int(os.getenv("MAX_TOKENS", "1024")),
        memory=memory,
        **kwargs
    )


# ─── Standalone usage example ────────────────────────────────────────────────
if __name__ == "__main__":
    print("RAGChain module ready ✓")
    print("Supported providers: openai, groq")
    print("Features: conversational memory, source citation, emergency detection")
