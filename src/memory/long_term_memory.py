# ==============================================================================
# src/memory/long_term_memory.py
# ==============================================================================
# PURPOSE:
#   Manages long-term persistent memory by storing and retrieving
#   conversation history in the PostgreSQL database.
#
# WHAT IS LONG-TERM MEMORY?
#   Unlike short-term memory (which lives in RAM and is lost on restart),
#   long-term memory persists indefinitely in the database.
#
#   It serves three purposes in this system:
#
#   1. AUDIT TRAIL
#      Every Q&A exchange is permanently logged so admins can review
#      what questions users are asking and what answers they received.
#
#   2. FREQUENTLY ASKED QUESTIONS (FAQ)
#      By querying which questions appear most often, we can identify
#      common medical topics users are asking about. This can be used
#      to pre-warm the cache or improve the knowledge base.
#
#   3. USER HISTORY
#      Users can view their past conversations (within their account)
#      to review information they received in previous sessions.
#      This is especially useful if they want to show a doctor
#      what the system told them.
#
# HOW IT DIFFERS FROM SHORT-TERM MEMORY:
#
#   Short-Term Memory          Long-Term Memory
#   ─────────────────          ────────────────
#   Lives in RAM               Lives in PostgreSQL
#   Lost on server restart     Persists forever
#   Per active session only    All sessions, all users
#   Used for follow-up Qs      Used for audit + analytics
#   Read/write every request   Write every request, read on demand
#   Max N turns per session    Unlimited rows
#
# DATABASE TABLE USED:
#   ConversationHistory (defined in src/database/models.py)
#   Columns: session_id, user_id, user_message, ai_response,
#            retrieved_chunks_count, was_flagged, created_at
#
# USED BY:
#   src/rag/pipeline.py (save_conversation_to_db)
#   src/api/routes.py   (get_user_history, get_frequent_questions)
# ==============================================================================

from loguru import logger
from sqlalchemy.orm import Session
from sqlalchemy import func, desc
from typing import Optional

from src.database.models import ConversationHistory


def save_conversation_to_db(
    session_id: str,
    user_id: str,
    user_message: str,
    ai_response: str,
    retrieved_chunks_count: int,
    was_flagged: bool,
    db: Session,
) -> Optional[ConversationHistory]:
    """
    Saves one complete Q&A exchange to the ConversationHistory table.

    Purpose:
        Called by pipeline.py at the end of every successful RAG pipeline run
        to permanently record the conversation exchange in PostgreSQL.
        This is the write path for long-term memory.

    Parameters:
        session_id (str):
            The UUID identifying the user's current browser session.
            Groups all messages from the same session together.
        user_id (str):
            The ID of the authenticated user who made this query.
        user_message (str):
            The exact question the user submitted.
        ai_response (str):
            The final cleaned response that was returned to the user
            (after output guardrails were applied).
        retrieved_chunks_count (int):
            How many FAISS chunks were retrieved for this query.
            Stored for retrieval quality analysis.
        was_flagged (bool):
            Whether this response was flagged for human review.
        db (Session):
            The active SQLAlchemy database session for this request.

    Returns:
        ConversationHistory: The newly created database record.
        None: If the save operation fails (error is logged, not raised).

    Example:
        save_conversation_to_db(
            session_id="abc-123",
            user_id="user-456",
            user_message="What are symptoms of diabetes?",
            ai_response="Diabetes symptoms include...",
            retrieved_chunks_count=5,
            was_flagged=False,
            db=db,
        )
    """
    try:
        # Create the new ConversationHistory record
        conversation_record = ConversationHistory(
            session_id=session_id,
            user_id=user_id,
            user_message=user_message,
            ai_response=ai_response,
            retrieved_chunks_count=retrieved_chunks_count,
            was_flagged=was_flagged,
        )

        # Add to the database session and commit
        db.add(conversation_record)
        db.commit()
        db.refresh(conversation_record)

        logger.debug(
            f"LongTermMemory: saved conversation record id={conversation_record.id} "
            f"for user {user_id[:8]}... session {session_id[:8]}..."
        )

        return conversation_record

    except Exception as error:
        # Roll back the failed transaction to keep the DB consistent
        db.rollback()

        # Log the error but do NOT re-raise it.
        # A failure to save long-term memory should never break the
        # main pipeline — the user already received their response.
        logger.error(f"LongTermMemory: failed to save conversation to DB: {error}")

        return None


def get_user_conversation_history(
    user_id: str,
    db: Session,
    limit: int = 50,
    session_id: Optional[str] = None,
) -> list[dict]:
    """
    Retrieves the conversation history for a specific user from the database.

    Purpose:
        Used by the API and Streamlit frontend to display a user's past
        conversations. Users can review what medical information they
        received in previous sessions.

    Parameters:
        user_id (str):
            The ID of the user whose history to retrieve.
        db (Session):
            The active SQLAlchemy database session.
        limit (int):
            Maximum number of records to return. Default: 50.
            Prevents returning thousands of rows at once.
        session_id (str, optional):
            If provided, only return history from this specific session.
            If None, return history from all sessions for this user.

    Returns:
        list[dict]: A list of conversation records, most recent first.
            Each dict contains:
                - id (int): The database record ID
                - session_id (str): Which session this belongs to
                - user_message (str): What the user asked
                - ai_response (str): What the AI replied
                - retrieved_chunks_count (int): How many chunks were used
                - was_flagged (bool): Whether this was flagged for review
                - created_at (str): ISO format timestamp

    Example:
        history = get_user_conversation_history(
            user_id="user-456",
            db=db,
            limit=10,
        )
        for record in history:
            print(record["user_message"])
            print(record["ai_response"])
    """
    try:
        # Build the base query for this user's history
        query = db.query(ConversationHistory).filter(
            ConversationHistory.user_id == user_id
        )

        # Optionally filter to a specific session only
        if session_id is not None:
            query = query.filter(ConversationHistory.session_id == session_id)

        # Order by most recent first, apply the limit
        records = (
            query.order_by(desc(ConversationHistory.created_at)).limit(limit).all()
        )

        # Convert ORM objects to plain dicts for the API response
        history = [_conversation_record_to_dict(record) for record in records]

        logger.debug(
            f"LongTermMemory: retrieved {len(history)} records "
            f"for user {user_id[:8]}..."
        )

        return history

    except Exception as error:
        logger.error(
            f"LongTermMemory: failed to retrieve history for user {user_id}: {error}"
        )
        return []


def get_frequently_asked_questions(
    db: Session,
    limit: int = 20,
    min_occurrences: int = 2,
) -> list[dict]:
    """
    Identifies the most frequently asked questions across all users.

    Purpose:
        Analyzes the ConversationHistory table to find which questions
        appear most often. This is useful for:
          - Pre-warming the query cache with common questions
          - Understanding what medical topics users care about most
          - Improving the knowledge base to better cover common queries
          - Displaying an FAQ section in the admin dashboard

    How it works:
        Groups all stored user_messages by their exact text and counts
        how many times each appears. Returns the most frequent ones.

        Note: This is a simple exact-match frequency count. A more
        sophisticated version would use semantic clustering to group
        similar questions (e.g., "diabetes symptoms" and "signs of
        diabetes" would be counted separately here).

    Parameters:
        db (Session): The active SQLAlchemy database session.
        limit (int): Maximum number of FAQs to return. Default: 20.
        min_occurrences (int):
            Minimum number of times a question must appear to be included.
            Default: 2 (filters out one-off questions).

    Returns:
        list[dict]: Most frequent questions, sorted by frequency descending.
            Each dict contains:
                - question (str): The user's message text
                - count (int): How many times this exact question was asked
                - first_asked (datetime): When it was first asked
                - last_asked (datetime): When it was most recently asked

    Example:
        faqs = get_frequently_asked_questions(db=db, limit=10)
        for faq in faqs:
            print(f"{faq['count']}x - {faq['question']}")
        # 47x - What are symptoms of diabetes?
        # 31x - What causes high blood pressure?
    """
    try:
        # Group by user_message and count occurrences
        # Also get the first and last time each question was asked
        faq_query = (
            db.query(
                ConversationHistory.user_message,
                func.count(ConversationHistory.id).label("count"),
                func.min(ConversationHistory.created_at).label("first_asked"),
                func.max(ConversationHistory.created_at).label("last_asked"),
            )
            .group_by(ConversationHistory.user_message)
            .having(func.count(ConversationHistory.id) >= min_occurrences)
            .order_by(desc("count"))
            .limit(limit)
        )

        results = faq_query.all()

        faqs = []
        for row in results:
            faqs.append(
                {
                    "question": row.user_message,
                    "count": row.count,
                    "first_asked": (
                        row.first_asked.isoformat() if row.first_asked else None
                    ),
                    "last_asked": (
                        row.last_asked.isoformat() if row.last_asked else None
                    ),
                }
            )

        logger.debug(
            f"LongTermMemory: found {len(faqs)} frequently asked questions "
            f"(min_occurrences={min_occurrences})"
        )

        return faqs

    except Exception as error:
        logger.error(
            f"LongTermMemory: failed to get frequently asked questions: {error}"
        )
        return []


def get_flagged_conversation_count(
    user_id: str,
    db: Session,
) -> int:
    """
    Returns the number of conversations flagged for review for a specific user.

    Purpose:
        Used by the admin panel to quickly check if a specific user has
        a pattern of generating risky responses. Could indicate the user
        is attempting to misuse the system.

    Parameters:
        user_id (str): The user to check.
        db (Session): The active database session.

    Returns:
        int: Count of flagged conversation records for this user.
    """
    try:
        count = (
            db.query(func.count(ConversationHistory.id))
            .filter(
                ConversationHistory.user_id == user_id,
                ConversationHistory.was_flagged == True,  # noqa: E712
            )
            .scalar()
        )
        return count or 0

    except Exception as error:
        logger.error(f"LongTermMemory: failed to count flagged conversations: {error}")
        return 0


def get_conversation_stats(db: Session) -> dict:
    """
    Returns high-level statistics about all stored conversations.

    Purpose:
        Used by the admin dashboard to display system-wide usage metrics
        such as total conversations, flagged rate, and daily activity.

    Parameters:
        db (Session): The active database session.

    Returns:
        dict: A dictionary of statistics:
            - total_conversations (int): Total Q&A pairs stored
            - flagged_conversations (int): How many were flagged
            - flagged_rate_percent (float): Percentage that were flagged
            - unique_users (int): Number of distinct users who have chatted
            - unique_sessions (int): Number of distinct sessions
    """
    try:
        total = db.query(func.count(ConversationHistory.id)).scalar() or 0

        flagged = (
            db.query(func.count(ConversationHistory.id))
            .filter(ConversationHistory.was_flagged == True)  # noqa: E712
            .scalar()
            or 0
        )

        unique_users = (
            db.query(func.count(func.distinct(ConversationHistory.user_id))).scalar()
            or 0
        )

        unique_sessions = (
            db.query(func.count(func.distinct(ConversationHistory.session_id))).scalar()
            or 0
        )

        flagged_rate = round((flagged / total * 100), 2) if total > 0 else 0.0

        return {
            "total_conversations": total,
            "flagged_conversations": flagged,
            "flagged_rate_percent": flagged_rate,
            "unique_users": unique_users,
            "unique_sessions": unique_sessions,
        }

    except Exception as error:
        logger.error(f"LongTermMemory: failed to get conversation stats: {error}")
        return {
            "total_conversations": 0,
            "flagged_conversations": 0,
            "flagged_rate_percent": 0.0,
            "unique_users": 0,
            "unique_sessions": 0,
        }


# ------------------------------------------------------------------------------
# PRIVATE HELPER
# ------------------------------------------------------------------------------


def _conversation_record_to_dict(record: ConversationHistory) -> dict:
    """
    Converts a ConversationHistory ORM object to a plain Python dictionary.

    Purpose:
        FastAPI cannot directly serialize SQLAlchemy ORM objects to JSON.
        This function converts each field to a JSON-safe Python type.

    Parameters:
        record (ConversationHistory): An ORM record from the database.

    Returns:
        dict: A plain dictionary with all fields as JSON-serializable types.
    """
    return {
        "id": record.id,
        "session_id": record.session_id,
        "user_id": record.user_id,
        "user_message": record.user_message,
        "ai_response": record.ai_response,
        "retrieved_chunks_count": record.retrieved_chunks_count,
        "was_flagged": record.was_flagged,
        "created_at": (record.created_at.isoformat() if record.created_at else None),
    }
