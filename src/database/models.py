# ==============================================================================
# src/database/models.py
# ==============================================================================
# PURPOSE:
#   This file defines every database table in the Healthcare RAG system
#   using SQLAlchemy ORM (Object Relational Mapper).
#
#   Each Python class here = one table in PostgreSQL.
#   Each class attribute  = one column in that table.
#
# TABLES DEFINED HERE:
#   1. User               - registered users, their roles, token budgets
#   2. TokenUsageLog      - one row per LLM API call, tracks cost
#   3. ConversationHistory- long-term storage of user Q&A pairs
#   4. QueryCache         - cached responses for repeated queries
#   5. HumanReviewQueue   - AI responses flagged for expert review
#
# HOW OTHER FILES USE THIS:
#   from src.database.models import User, TokenUsageLog
# ==============================================================================

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    Enum as SQLEnum,
)
from sqlalchemy.orm import relationship
import enum

from src.database.db import Base


# ------------------------------------------------------------------------------
# HELPER: get current UTC time
# ------------------------------------------------------------------------------
# We use this as the default value for all "created_at" columns.
# It ensures timestamps are always stored in UTC regardless of the
# server's local timezone.
# ------------------------------------------------------------------------------


def utc_now() -> datetime:
    """
    Returns the current date and time in UTC timezone.

    Purpose:
        Used as a default factory for DateTime columns so every
        record is timestamped in UTC at the moment of creation.

    Returns:
        datetime: Current UTC datetime.
    """
    return datetime.now(timezone.utc)


# ------------------------------------------------------------------------------
# ENUM: UserRole
# ------------------------------------------------------------------------------
# Defines the two possible roles a user can have in the system.
# "patient" = regular user who asks medical questions
# "admin"   = healthcare professional or system administrator
# ------------------------------------------------------------------------------


class UserRole(str, enum.Enum):
    """
    Enumeration of valid user roles in the system.

    Values:
        patient: A regular user who submits medical queries.
        admin:   A healthcare professional or system administrator
                 who can review flagged responses and manage users.
    """

    patient = "patient"
    admin = "admin"


# ------------------------------------------------------------------------------
# ENUM: ReviewStatus
# ------------------------------------------------------------------------------


class ReviewStatus(str, enum.Enum):
    """
    Enumeration of possible statuses for a human review queue entry.

    Values:
        pending:  The response has been flagged and is waiting for review.
        approved: A healthcare professional reviewed and approved the response.
        rejected: A healthcare professional reviewed and rejected the response.
    """

    pending = "pending"
    approved = "approved"
    rejected = "rejected"


# ==============================================================================
# TABLE 1: User
# ==============================================================================


class User(Base):
    """
    Represents a registered user of the Healthcare RAG system.

    Purpose:
        Stores user credentials, role, and token usage tracking data.
        Every query submitted to the system must be linked to a user
        so we can enforce per-user token limits.

    Relationships:
        - One User can have many TokenUsageLog entries (one per API call)
        - One User can have many ConversationHistory entries
        - One User can have many HumanReviewQueue entries (as the reviewer)

    Table name: users
    """

    __tablename__ = "users"

    # Primary key - we use a UUID string instead of an integer.
    # UUIDs are better for distributed systems because they are globally
    # unique and do not expose the total number of users.
    id = Column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
        comment="Unique identifier for the user (UUID format)",
    )

    # The user's email address - must be unique across all users.
    # Used as the login identifier.
    email = Column(
        String(255),
        unique=True,
        nullable=False,
        index=True,
        comment="User's email address, used for login",
    )

    # Hashed password - we never store plain text passwords.
    # The passlib library handles hashing in the auth module.
    hashed_password = Column(
        String(255),
        nullable=False,
        comment="Bcrypt-hashed password, never store plain text",
    )

    # Display name shown in the UI
    full_name = Column(
        String(255),
        nullable=True,
        comment="User's full name for display purposes",
    )

    # Role controls what the user can do in the system
    role = Column(
        SQLEnum(UserRole),
        nullable=False,
        default=UserRole.patient,
        comment="User role: patient (regular user) or admin (healthcare professional)",
    )

    # Whether this account is active. Admins can deactivate accounts
    # instead of deleting them to preserve audit history.
    is_active = Column(
        Boolean,
        nullable=False,
        default=True,
        comment="False means the account is deactivated and cannot log in",
    )

    # Running total of tokens consumed by this user across all requests.
    # Updated after every API call by the usage_logger module.
    tokens_used = Column(
        Integer,
        nullable=False,
        default=0,
        comment="Total number of tokens consumed by this user so far",
    )

    # The maximum number of tokens this user is allowed to consume.
    # Defaults to the value set in settings.default_token_limit.
    # Admins can override this per user.
    token_limit = Column(
        Integer,
        nullable=False,
        default=100_000,
        comment="Maximum tokens this user is allowed to use in total",
    )

    # Timestamps
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        comment="When this user account was created",
    )

    last_active_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="When this user last made a request to the system",
    )

    # --------------------------------------------------------------------------
    # Relationships - SQLAlchemy links these tables together automatically
    # --------------------------------------------------------------------------

    # All token usage records belonging to this user
    token_logs = relationship(
        "TokenUsageLog",
        back_populates="user",
        cascade="all, delete-orphan",
        # cascade: if we delete a user, delete all their logs too
    )

    # All conversation history records belonging to this user
    conversations = relationship(
        "ConversationHistory",
        back_populates="user",
        cascade="all, delete-orphan",
    )

    # --------------------------------------------------------------------------
    # Computed properties (not stored in DB, calculated on the fly)
    # --------------------------------------------------------------------------

    @property
    def tokens_remaining(self) -> int:
        """
        Calculates how many tokens this user can still consume.

        Returns:
            int: tokens_limit minus tokens_used, minimum 0.
        """
        remaining = self.token_limit - self.tokens_used
        return max(0, remaining)

    @property
    def is_over_limit(self) -> bool:
        """
        Returns True if this user has exceeded their token budget.

        Returns:
            bool: True if tokens_used >= token_limit.
        """
        return self.tokens_used >= self.token_limit

    def __repr__(self) -> str:
        return f"<User id={self.id} email={self.email} role={self.role}>"


# ==============================================================================
# TABLE 2: TokenUsageLog
# ==============================================================================


class TokenUsageLog(Base):
    """
    Records the token usage of every single LLM API call.

    Purpose:
        One row is inserted for every request that reaches the LLM.
        This gives us a complete audit trail of:
          - Who made the request
          - What query they asked
          - How many tokens it consumed
          - How much it cost in USD
          - When it happened

        The monitoring module reads this table to enforce budgets
        and the admin panel displays it for usage analytics.

    Table name: token_usage_logs
    """

    __tablename__ = "token_usage_logs"

    id = Column(
        Integer,
        primary_key=True,
        autoincrement=True,
        comment="Auto-incrementing primary key",
    )

    # Foreign key linking this log entry to the user who made the request
    user_id = Column(
        String(36),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="The user who made this request",
    )

    # The query the user submitted (truncated to 1000 chars for storage)
    query_text = Column(
        String(1000),
        nullable=True,
        comment="The user's query, truncated to 1000 characters",
    )

    # Token counts from the OpenAI API response
    input_tokens = Column(
        Integer,
        nullable=False,
        default=0,
        comment="Number of prompt/input tokens used in this request",
    )

    output_tokens = Column(
        Integer,
        nullable=False,
        default=0,
        comment="Number of completion/output tokens used in this request",
    )

    total_tokens = Column(
        Integer,
        nullable=False,
        default=0,
        comment="Total tokens used (input + output)",
    )

    # Cost in USD for this single request
    cost_usd = Column(
        Float,
        nullable=False,
        default=0.0,
        comment="Estimated cost of this request in US dollars",
    )

    # The OpenAI model that was used (e.g., gpt-4o-mini)
    model_used = Column(
        String(100),
        nullable=True,
        comment="The OpenAI model name used for this request",
    )

    # Whether this request was served from cache (if so, no tokens were used)
    served_from_cache = Column(
        Boolean,
        nullable=False,
        default=False,
        comment="True if this response came from cache (no LLM call was made)",
    )

    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        comment="When this request was made",
    )

    # Relationship back to the User
    user = relationship("User", back_populates="token_logs")

    def __repr__(self) -> str:
        return (
            f"<TokenUsageLog id={self.id} user_id={self.user_id} "
            f"total_tokens={self.total_tokens} cost_usd={self.cost_usd}>"
        )


# ==============================================================================
# TABLE 3: ConversationHistory
# ==============================================================================


class ConversationHistory(Base):
    """
    Stores the long-term history of user conversations.

    Purpose:
        This is the LONG-TERM memory store. Unlike short-term memory
        (which lives in RAM and disappears when the session ends),
        this table persists conversations to PostgreSQL permanently.

        It enables:
          - Users to review their past queries and answers
          - Admins to audit what questions users are asking
          - The long_term_memory module to surface frequently asked questions
          - Future personalization (e.g., "last time you asked about diabetes...")

        Each row represents one complete Q&A exchange:
          one user message + one AI response.

    Table name: conversation_history
    """

    __tablename__ = "conversation_history"

    id = Column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    # The session ID groups messages from the same browser/app session together.
    # This is a UUID generated by the frontend when a new session starts.
    session_id = Column(
        String(36),
        nullable=False,
        index=True,
        comment="Groups messages from the same user session together",
    )

    # Foreign key to the user
    user_id = Column(
        String(36),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="The user who had this conversation",
    )

    # The user's message
    user_message = Column(
        Text,
        nullable=False,
        comment="The exact message the user submitted",
    )

    # The AI's response
    ai_response = Column(
        Text,
        nullable=False,
        comment="The AI-generated response returned to the user",
    )

    # The number of source chunks that were retrieved for this response.
    # Useful for debugging retrieval quality.
    retrieved_chunks_count = Column(
        Integer,
        nullable=True,
        comment="How many FAISS chunks were retrieved for this query",
    )

    # Whether this response was flagged for human review
    was_flagged = Column(
        Boolean,
        nullable=False,
        default=False,
        comment="True if this response was sent to the human review queue",
    )

    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        comment="When this conversation exchange happened",
    )

    # Relationship back to the User
    user = relationship("User", back_populates="conversations")

    def __repr__(self) -> str:
        return (
            f"<ConversationHistory id={self.id} session_id={self.session_id} "
            f"user_id={self.user_id}>"
        )


# ==============================================================================
# TABLE 4: QueryCache
# ==============================================================================


class QueryCache(Base):
    """
    Caches AI-generated responses for repeated identical queries.

    Purpose:
        If two users ask the exact same question (e.g., "what is diabetes?"),
        we do not need to call the OpenAI API a second time. Instead, we
        return the cached response instantly.

        This saves money (no token cost) and reduces latency (no LLM wait time).

        How it works:
          1. When a query comes in, the caching module computes a SHA-256 hash
             of the query text (lowercased and stripped of whitespace).
          2. It looks up that hash in this table.
          3. If found and not expired: return the cached response.
          4. If not found: run the full RAG pipeline, then store the result here.

    Table name: query_cache
    """

    __tablename__ = "query_cache"

    id = Column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    # SHA-256 hash of the normalized query text.
    # We store the hash (not the full query) as the lookup key because:
    #   - Hashes are fixed-length (64 chars) making indexing fast
    #   - Two identical queries always produce the same hash
    query_hash = Column(
        String(64),
        unique=True,
        nullable=False,
        index=True,
        comment="SHA-256 hash of the normalized query text, used as lookup key",
    )

    # The original query text (stored for debugging/admin inspection)
    query_text = Column(
        Text,
        nullable=False,
        comment="The original query text before hashing",
    )

    # The cached AI response
    response_text = Column(
        Text,
        nullable=False,
        comment="The AI-generated response that was cached",
    )

    # How many times this cache entry has been served
    hit_count = Column(
        Integer,
        nullable=False,
        default=1,
        comment="Number of times this cached response has been returned",
    )

    # When this cache entry was first created
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        comment="When this cache entry was first created",
    )

    # When this cache entry was last used
    last_accessed_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        comment="When this cache entry was last returned to a user",
    )

    # Optional: when this cache entry expires.
    # None means it never expires.
    expires_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="When this cache entry expires. None means no expiry.",
    )

    def __repr__(self) -> str:
        return (
            f"<QueryCache id={self.id} hit_count={self.hit_count} "
            f"query_hash={self.query_hash[:8]}...>"
        )


# ==============================================================================
# TABLE 5: HumanReviewQueue
# ==============================================================================


class HumanReviewQueue(Base):
    """
    Holds AI-generated responses that have been flagged for human review.

    Purpose:
        When the output guardrails or risk scoring detects that an AI response
        might be unsafe, sensitive, or outside the system's confidence range,
        the response is placed in this queue instead of being returned
        directly to the user.

        A healthcare professional (admin user) then reviews the queue via
        the admin panel and either:
          - Approves the response (it is sent to the user)
          - Rejects the response (a safe fallback message is sent instead)

        This implements the "human-in-the-loop" requirement from the
        problem statement.

    Table name: human_review_queue
    """

    __tablename__ = "human_review_queue"

    id = Column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    # The session this flagged response belongs to
    session_id = Column(
        String(36),
        nullable=False,
        index=True,
        comment="The session ID of the user who triggered this review",
    )

    # The user who asked the question
    user_id = Column(
        String(36),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        comment="The user who submitted the query that triggered this review",
    )

    # The original query from the user
    user_query = Column(
        Text,
        nullable=False,
        comment="The original query submitted by the user",
    )

    # The AI response that was flagged
    ai_response = Column(
        Text,
        nullable=False,
        comment="The AI-generated response that was flagged for review",
    )

    # The numerical risk score that caused this to be flagged.
    # Higher = riskier. Flagged when above settings.human_review_risk_threshold.
    risk_score = Column(
        Float,
        nullable=False,
        default=0.0,
        comment="Risk score between 0.0 and 1.0 that caused this to be flagged",
    )

    # The reason why this response was flagged (e.g., "possible diagnosis detected")
    flag_reason = Column(
        String(500),
        nullable=True,
        comment="Human-readable reason why this response was flagged",
    )

    # Current status of this review item
    status = Column(
        SQLEnum(ReviewStatus),
        nullable=False,
        default=ReviewStatus.pending,
        index=True,
        comment="Current review status: pending, approved, or rejected",
    )

    # The admin user who reviewed this item (null until reviewed)
    reviewed_by_user_id = Column(
        String(36),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        comment="The admin user who reviewed this flagged response",
    )

    # Any notes the reviewer added during their review
    reviewer_notes = Column(
        Text,
        nullable=True,
        comment="Optional notes added by the healthcare professional reviewer",
    )

    # Timestamps
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        comment="When this item was added to the review queue",
    )

    reviewed_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="When this item was reviewed by a healthcare professional",
    )

    def __repr__(self) -> str:
        return (
            f"<HumanReviewQueue id={self.id} status={self.status} "
            f"risk_score={self.risk_score}>"
        )
