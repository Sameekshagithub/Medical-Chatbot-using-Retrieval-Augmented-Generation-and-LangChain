"""
============================================================
src/database.py - Chat History Storage (SQLite)
============================================================

PURPOSE:
    Persists conversation history to SQLite database so:
    - Chat history survives app restarts
    - Users can review past conversations
    - Data can be exported for analysis

SCHEMA:
    Table: conversations
        id          INTEGER PRIMARY KEY AUTOINCREMENT
        session_id  TEXT NOT NULL
        role        TEXT (user / assistant)
        content     TEXT NOT NULL
        sources     TEXT (JSON-serialized source citations)
        timestamp   DATETIME DEFAULT CURRENT_TIMESTAMP

WHY SQLITE:
    - Zero setup (file-based, no server needed)
    - Perfect for single-user Streamlit apps
    - SQLAlchemy ORM for clean, safe queries
    - Easy to export as CSV or JSON
============================================================
"""

import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict
from loguru import logger

from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

# ─── SQLAlchemy Setup ─────────────────────────────────────────────────────

Base = declarative_base()


class ConversationRecord(Base):
    """
    SQLAlchemy ORM model for conversation records.
    Represents one message (user or assistant) in the database.
    """
    __tablename__ = "conversations"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(36), nullable=False, index=True)
    role = Column(String(20), nullable=False)       # "user" or "assistant"
    content = Column(Text, nullable=False)           # Message content
    sources = Column(Text, nullable=True)            # JSON: list of source citations
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    def to_dict(self) -> dict:
        """Convert record to dictionary for export."""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "role": self.role,
            "content": self.content,
            "sources": json.loads(self.sources) if self.sources else [],
            "timestamp": self.timestamp.isoformat() if self.timestamp else None
        }


# ─── Database Manager ─────────────────────────────────────────────────────

class ChatDatabase:
    """
    Manages SQLite database for persistent chat history storage.
    
    Usage:
        db = ChatDatabase()
        db.save_message(session_id, "user", "What is diabetes?")
        history = db.get_session_history(session_id)
    """

    def __init__(self, db_path: str = "./chat_history/conversations.db"):
        """
        Initialize the database connection.
        
        Args:
            db_path: Path to the SQLite database file
        """
        # Ensure directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        self.db_path = db_path
        self.engine = create_engine(
            f"sqlite:///{db_path}",
            connect_args={"check_same_thread": False}  # Required for Streamlit
        )
        
        # Create tables if they don't exist
        Base.metadata.create_all(self.engine)
        
        # Create session factory
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )
        
        logger.info(f"ChatDatabase initialized: {db_path}")

    def get_session(self) -> Session:
        """Create a new database session."""
        return self.SessionLocal()

    def save_message(
        self,
        session_id: str,
        role: str,
        content: str,
        sources: Optional[List[Dict]] = None
    ) -> ConversationRecord:
        """
        Save a single chat message to the database.
        
        Args:
            session_id: Unique identifier for this chat session
            role:       "user" or "assistant"
            content:    Message text content
            sources:    Optional list of source citation dicts
            
        Returns:
            The saved ConversationRecord
        """
        with self.get_session() as session:
            record = ConversationRecord(
                session_id=session_id,
                role=role,
                content=content,
                sources=json.dumps(sources) if sources else None,
                timestamp=datetime.utcnow()
            )
            session.add(record)
            session.commit()
            session.refresh(record)
            
            logger.debug(f"Saved {role} message | Session: {session_id[:8]}...")
            return record

    def get_session_history(self, session_id: str) -> List[Dict]:
        """
        Retrieve all messages for a specific chat session.
        
        Args:
            session_id: The session identifier
            
        Returns:
            List of message dicts, ordered by timestamp
        """
        with self.get_session() as session:
            records = (
                session.query(ConversationRecord)
                .filter(ConversationRecord.session_id == session_id)
                .order_by(ConversationRecord.timestamp.asc())
                .all()
            )
            return [r.to_dict() for r in records]

    def get_all_sessions(self) -> List[Dict]:
        """
        Get summary information for all chat sessions.
        
        Returns:
            List of session summaries with ID, message count, last activity
        """
        with self.get_session() as session:
            # Group by session_id and get counts + last timestamp
            results = (
                session.query(
                    ConversationRecord.session_id,
                    func.count(ConversationRecord.id).label("message_count"),
                    func.max(ConversationRecord.timestamp).label("last_activity"),
                    func.min(ConversationRecord.timestamp).label("created_at")
                )
                .group_by(ConversationRecord.session_id)
                .order_by(func.max(ConversationRecord.timestamp).desc())
                .all()
            )
            
            return [
                {
                    "session_id": r.session_id,
                    "message_count": r.message_count,
                    "last_activity": r.last_activity.isoformat() if r.last_activity else None,
                    "created_at": r.created_at.isoformat() if r.created_at else None
                }
                for r in results
            ]

    def delete_session(self, session_id: str) -> int:
        """
        Delete all messages for a specific session.
        
        Args:
            session_id: Session to delete
            
        Returns:
            Number of records deleted
        """
        with self.get_session() as session:
            deleted = (
                session.query(ConversationRecord)
                .filter(ConversationRecord.session_id == session_id)
                .delete()
            )
            session.commit()
            logger.info(f"Deleted {deleted} records for session {session_id[:8]}...")
            return deleted

    def export_session_to_text(self, session_id: str) -> str:
        """
        Export a session's conversation as formatted text.
        
        Args:
            session_id: Session to export
            
        Returns:
            Formatted conversation text for download
        """
        messages = self.get_session_history(session_id)
        
        if not messages:
            return "No messages found for this session."
        
        lines = [
            "MediQuery AI - Conversation Export",
            f"Session ID: {session_id}",
            f"Messages: {len(messages)}",
            "=" * 60,
            ""
        ]
        
        for msg in messages:
            role = "You" if msg["role"] == "user" else "MediQuery AI"
            timestamp = msg["timestamp"][:16].replace("T", " ") if msg["timestamp"] else ""
            
            lines.append(f"[{timestamp}] {role}:")
            lines.append(msg["content"])
            
            if msg.get("sources"):
                lines.append("Sources:")
                for source in msg["sources"]:
                    lines.append(f"  - {source}")
            lines.append("")
        
        lines.extend([
            "=" * 60,
            "⚠️ Medical Disclaimer: This conversation is for educational purposes only.",
            "Always consult a qualified healthcare professional for medical advice."
        ])
        
        return "\n".join(lines)

    def get_database_stats(self) -> dict:
        """Return overall database statistics."""
        with self.get_session() as session:
            total_messages = session.query(ConversationRecord).count()
            total_sessions = (
                session.query(ConversationRecord.session_id)
                .distinct()
                .count()
            )
            
            return {
                "total_messages": total_messages,
                "total_sessions": total_sessions,
                "db_path": self.db_path,
                "db_size_kb": round(Path(self.db_path).stat().st_size / 1024, 2) if Path(self.db_path).exists() else 0
            }


def generate_session_id() -> str:
    """Generate a unique session identifier."""
    return str(uuid.uuid4())


# ─── Standalone usage example ────────────────────────────────────────────────
if __name__ == "__main__":
    # Test the database
    db = ChatDatabase(db_path="./test_db.db")
    
    session_id = generate_session_id()
    db.save_message(session_id, "user", "What is hypertension?")
    db.save_message(session_id, "assistant", "Hypertension is elevated blood pressure...")
    
    history = db.get_session_history(session_id)
    print(f"Saved and retrieved {len(history)} messages")
    print(f"Stats: {db.get_database_stats()}")
    
    # Cleanup test
    import os
    os.remove("./test_db.db")
    print("ChatDatabase module ready ✓")
