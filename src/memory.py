

from typing import List, Dict, Optional, Tuple
from datetime import datetime
from loguru import logger

from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain.schema import HumanMessage, AIMessage, BaseMessage


class ChatMemoryManager:
    """
    Manages conversation memory for the medical chatbot.
    
    Maintains a sliding window of recent messages to prevent
    exceeding LLM context limits while preserving conversation flow.
    """

    def __init__(
        self,
        memory_key: str = "chat_history",
        max_messages: int = 20,
        return_messages: bool = True
    ):
        """
        Initialize the memory manager.
        
        Args:
            memory_key:       Key used to pass history to LangChain chains
            max_messages:     Maximum conversation turns to remember (prevents context overflow)
            return_messages:  Return as Message objects (True) or plain string (False)
        """
        self.memory_key = memory_key
        self.max_messages = max_messages
        self.return_messages = return_messages
        
        # Primary LangChain memory object (used with chains)
        self.langchain_memory = ConversationBufferMemory(
            memory_key=memory_key,
            return_messages=return_messages,
            output_key="answer",       # Key in chain output to store as AI message
            input_key="question"       # Key in chain input that's the user message
        )
        
        # Our own message store for UI display and history export
        self.messages: List[Dict] = []
        self.session_start = datetime.now()
        
        logger.info(f"ChatMemoryManager initialized | max_messages={max_messages}")

    def add_user_message(self, message: str) -> None:
        """
        Record a user message in memory.
        
        Args:
            message: The user's question or input
        """
        self.messages.append({
            "role": "user",
            "content": message,
            "timestamp": datetime.now().isoformat()
        })
        
        # Enforce sliding window - remove oldest messages if limit exceeded
        self._enforce_window()

    def add_ai_message(self, message: str, sources: Optional[List] = None) -> None:
        """
        Record an AI response in memory.
        
        Args:
            message: The AI's response
            sources: Optional list of source documents used
        """
        self.messages.append({
            "role": "assistant",
            "content": message,
            "timestamp": datetime.now().isoformat(),
            "sources": sources or []
        })

    def get_chat_history_string(self) -> str:
        """
        Format conversation history as a readable string for LLM context.
        
        Returns:
            Formatted string like:
            "Human: What is diabetes?
             AI: Diabetes is a metabolic disorder...
             Human: What are its treatments?"
        """
        history_lines = []
        
        for msg in self.messages[-self.max_messages:]:
            role = "Human" if msg["role"] == "user" else "AI"
            history_lines.append(f"{role}: {msg['content']}")
        
        return "\n".join(history_lines)

    def get_langchain_memory(self) -> ConversationBufferMemory:
        """
        Return the LangChain memory object for use in chains.
        
        This is passed to ConversationalRetrievalChain which handles
        memory injection automatically.
        
        Returns:
            LangChain ConversationBufferMemory object
        """
        return self.langchain_memory

    def get_messages_for_display(self) -> List[Dict]:
        """
        Return messages formatted for Streamlit UI display.
        
        Returns:
            List of message dicts with role, content, timestamp
        """
        return self.messages.copy()

    def clear(self) -> None:
        """
        Clear all conversation history.
        
        Called when user clicks "Clear Chat" in the UI.
        Also resets the LangChain memory buffer.
        """
        self.messages = []
        self.langchain_memory.clear()
        self.session_start = datetime.now()
        logger.info("Conversation memory cleared")

    def get_session_stats(self) -> dict:
        """
        Return statistics about the current session.
        
        Returns:
            Dict with session duration, message count, etc.
        """
        now = datetime.now()
        duration = now - self.session_start
        
        user_messages = [m for m in self.messages if m["role"] == "user"]
        ai_messages = [m for m in self.messages if m["role"] == "assistant"]
        
        return {
            "session_start": self.session_start.strftime("%Y-%m-%d %H:%M"),
            "session_duration_minutes": int(duration.total_seconds() / 60),
            "total_messages": len(self.messages),
            "user_questions": len(user_messages),
            "ai_responses": len(ai_messages),
            "memory_window": self.max_messages
        }

    def export_to_text(self) -> str:
        """
        Export the full conversation as formatted text for download.
        
        Returns:
            Formatted conversation string for file download
        """
        lines = [
            f"MediQuery AI - Conversation Export",
            f"Session: {self.session_start.strftime('%Y-%m-%d %H:%M')}",
            f"{'='*60}",
            ""
        ]
        
        for msg in self.messages:
            timestamp = msg.get("timestamp", "")[:16].replace("T", " ")
            role = "You" if msg["role"] == "user" else "MediQuery AI"
            
            lines.append(f"[{timestamp}] {role}:")
            lines.append(msg["content"])
            lines.append("")
        
        lines.append(f"{'='*60}")
        lines.append("⚠️ Medical Disclaimer: This conversation is for informational purposes only.")
        lines.append("Always consult a qualified healthcare professional for medical advice.")
        
        return "\n".join(lines)

    def _enforce_window(self) -> None:
        """
        Keep only the most recent messages within the window limit.
        Also syncs the LangChain memory buffer.
        """
        if len(self.messages) > self.max_messages * 2:  # *2 because user+AI pairs
            # Keep the most recent messages
            self.messages = self.messages[-(self.max_messages * 2):]
            
            # Rebuild LangChain memory from our messages
            self.langchain_memory.clear()
            for i in range(0, len(self.messages) - 1, 2):
                if i + 1 < len(self.messages):
                    user_msg = self.messages[i]
                    ai_msg = self.messages[i + 1]
                    if user_msg["role"] == "user" and ai_msg["role"] == "assistant":
                        self.langchain_memory.save_context(
                            {"question": user_msg["content"]},
                            {"answer": ai_msg["content"]}
                        )


# ─── Standalone usage example ────────────────────────────────────────────────
if __name__ == "__main__":
    memory = ChatMemoryManager(max_messages=10)
    
    # Simulate a conversation
    memory.add_user_message("What is hypertension?")
    memory.add_ai_message("Hypertension is persistently elevated blood pressure above 140/90 mmHg.")
    memory.add_user_message("What are its risk factors?")
    
    print("Chat History:")
    print(memory.get_chat_history_string())
    print("\nSession Stats:")
    print(memory.get_session_stats())
    print("\nMemory module ready ✓")
