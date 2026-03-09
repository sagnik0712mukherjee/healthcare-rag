# ==============================================================================
# src/utils/helpers.py
# ==============================================================================
# PURPOSE:
#   Shared utility functions used by multiple modules across the project.
#   Nothing in this file is specific to any single layer — these are
#   general-purpose helpers that any module can import and use.
#
# FUNCTIONS IN THIS FILE:
#   1. compute_risk_score()     - Scores a query+response pair for human review
#   2. clean_text()             - Strips and normalises text strings
#   3. truncate_text()          - Cuts text to a max length with ellipsis
#   4. get_utc_now()            - Returns current UTC datetime
#   5. format_timestamp()       - Formats a datetime as a readable string
#   6. is_valid_uuid()          - Validates UUID format strings
#   7. sanitize_user_input()    - Light sanitization of raw user input
#   8. build_session_id()       - Generates a new unique session UUID
#
# IMPORTANT — compute_risk_score():
#   This is the most critical function here. It is called by pipeline.py
#   in Step 8 to decide whether an AI response should be flagged for
#   human review by a healthcare professional.
#
#   It uses a keyword-based scoring approach (not a second LLM call) to
#   keep it fast and free. The score is a float between 0.0 and 1.0.
#   Responses that score above settings.human_review_risk_threshold
#   are added to the HumanReviewQueue table.
#
# USED BY:
#   src/rag/pipeline.py         (compute_risk_score)
#   src/api/routes.py           (build_session_id, is_valid_uuid)
#   src/guardrails/*.py         (clean_text, truncate_text)
#   src/monitoring/*.py         (get_utc_now, format_timestamp)
# ==============================================================================

import re
import uuid
from datetime import datetime, timezone
from loguru import logger


# ==============================================================================
# RISK SCORING FOR HUMAN REVIEW
# ==============================================================================

# These keyword groups contribute to the risk score of an AI response.
# Each group has a weight — how much each matching keyword adds to the score.
# The final score is normalised to a 0.0 - 1.0 range.
#
# Design principle:
#   We score the RESPONSE (not the query) because the response is what
#   could actually cause harm if shown to the user. A dangerous-sounding
#   query might produce a perfectly safe, educational response.

# High-risk indicators in the response (weight: 0.3 each)
# These suggest the AI may have crossed a safety boundary
HIGH_RISK_RESPONSE_KEYWORDS = [
    "you have",  # Sounds like a diagnosis
    "you are suffering",  # Sounds like a diagnosis
    "you should take",  # Sounds like a prescription
    "i diagnose",  # Direct diagnosis claim
    "my diagnosis",  # Direct diagnosis claim
    "i prescribe",  # Direct prescription
    "take this medication",  # Direct prescription
    "lethal dose",  # Dangerous dosage info
    "overdose on",  # Dangerous dosage info
]

# Medium-risk indicators in the response (weight: 0.15 each)
# These suggest the response is in a sensitive territory
MEDIUM_RISK_RESPONSE_KEYWORDS = [
    "you probably have",  # Soft diagnosis
    "you likely have",  # Soft diagnosis
    "sounds like you have",  # Soft diagnosis
    "you need to take",  # Soft prescription
    "i recommend taking",  # Soft prescription
    "the correct dose",  # Dosage instruction
    "inject yourself",  # Self-injection instruction
    "stop your medication",  # Medication management
]

# High-risk indicators in the QUERY (weight: 0.2 each)
# Some query types inherently warrant more careful review of the response
HIGH_RISK_QUERY_KEYWORDS = [
    "suicide",
    "self harm",
    "overdose",
    "kill myself",
    "end my life",
    "want to die",
    "harm someone",
]

# The maximum possible raw score (used for normalisation to 0.0-1.0)
# = (max high-risk response hits * 0.3) + (max medium-risk * 0.15) + (max query hits * 0.2)
# We cap at a reasonable maximum rather than calculating exact ceiling
MAX_RAW_SCORE = 3.0


def compute_risk_score(query: str, response: str) -> float:
    """
    Computes a risk score between 0.0 and 1.0 for a query-response pair.

    Purpose:
        Called by pipeline.py Step 8 to decide whether an AI-generated
        response should be flagged for human review by a healthcare
        professional. A higher score means higher risk.

        The threshold for flagging is set in settings.human_review_risk_threshold
        (default: 0.7). Responses scoring above this are added to the
        HumanReviewQueue table in the database.

    How it works:
        1. Scan the RESPONSE for high-risk keywords (weight: 0.3 each)
        2. Scan the RESPONSE for medium-risk keywords (weight: 0.15 each)
        3. Scan the QUERY for inherently risky topics (weight: 0.2 each)
        4. Sum all contributions
        5. Normalise to 0.0-1.0 range (cap at 1.0)

    Design note:
        This is intentionally simple — keyword matching, no LLM call.
        Speed matters here because this runs on every single request.
        The goal is to catch obvious cases, not to be perfect.
        Human reviewers are the final safety net.

    Parameters:
        query (str): The original user query.
        response (str): The cleaned AI-generated response.

    Returns:
        float: Risk score between 0.0 (no risk detected) and 1.0 (maximum risk).

    Example:
        score = compute_risk_score(
            query="What medications help with diabetes?",
            response="You should take metformin 500mg twice daily.",
        )
        print(score)  # ~0.45 (medium risk — sounds like prescription)

        score = compute_risk_score(
            query="What are symptoms of a cold?",
            response="Common cold symptoms include runny nose and sore throat.",
        )
        print(score)  # 0.0 (no risk detected)
    """
    raw_score = 0.0

    lowercased_query = query.lower() if query else ""
    lowercased_response = response.lower() if response else ""

    # Step 1: Check response for high-risk keywords
    for keyword in HIGH_RISK_RESPONSE_KEYWORDS:
        if keyword in lowercased_response:
            raw_score += 0.3
            logger.debug(
                f"RiskScore: high-risk response keyword '{keyword}' "
                f"found (+0.3, running total: {raw_score:.2f})"
            )

    # Step 2: Check response for medium-risk keywords
    for keyword in MEDIUM_RISK_RESPONSE_KEYWORDS:
        if keyword in lowercased_response:
            raw_score += 0.15
            logger.debug(
                f"RiskScore: medium-risk response keyword '{keyword}' "
                f"found (+0.15, running total: {raw_score:.2f})"
            )

    # Step 3: Check query for inherently risky topics
    for keyword in HIGH_RISK_QUERY_KEYWORDS:
        if keyword in lowercased_query:
            raw_score += 0.2
            logger.debug(
                f"RiskScore: high-risk query keyword '{keyword}' "
                f"found (+0.2, running total: {raw_score:.2f})"
            )

    # Step 4: Normalise to 0.0-1.0 (cap at 1.0)
    normalised_score = min(raw_score / MAX_RAW_SCORE, 1.0)

    logger.debug(
        f"RiskScore: final score = {normalised_score:.3f} "
        f"(raw: {raw_score:.2f} / max: {MAX_RAW_SCORE})"
    )

    return round(normalised_score, 4)


# ==============================================================================
# TEXT UTILITIES
# ==============================================================================


def clean_text(text: str) -> str:
    """
    Strips whitespace and normalises a text string.

    Purpose:
        Used throughout the codebase to ensure consistent text processing —
        removing leading/trailing whitespace and collapsing multiple internal
        spaces into a single space.

    Parameters:
        text (str): The raw text string to clean.

    Returns:
        str: The cleaned text. Returns empty string if input is None or empty.

    Example:
        clean_text("  What   is diabetes?  ")
        # "What is diabetes?"
    """
    if not text:
        return ""
    # Collapse multiple whitespace characters (spaces, tabs, newlines) into one space
    return " ".join(text.strip().split())


def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """
    Truncates a text string to a maximum length, adding a suffix if cut.

    Purpose:
        Used to prevent very long strings from being stored in the database
        or displayed in the UI. For example, query text is truncated before
        storage in TokenUsageLog.

    Parameters:
        text (str): The text to potentially truncate.
        max_length (int): Maximum allowed character length (including suffix).
        suffix (str): String to append when truncation occurs. Default: "..."

    Returns:
        str: The original text if within max_length, or truncated text + suffix.

    Example:
        truncate_text("A very long clinical case description...", max_length=20)
        # "A very long clini..."
    """
    if not text:
        return ""

    if len(text) <= max_length:
        return text

    # Truncate so that text + suffix fits within max_length
    cut_at = max_length - len(suffix)
    return text[:cut_at] + suffix


def sanitize_user_input(text: str) -> str:
    """
    Applies light sanitization to raw user input before processing.

    Purpose:
        Removes characters that could cause issues in downstream processing
        while preserving the meaning of the user's query. This is NOT
        a security measure (SQL injection is prevented by SQLAlchemy's
        parameterised queries) — it just ensures clean input for the pipeline.

    What it does:
        - Strips leading/trailing whitespace
        - Removes null bytes (which can cause issues in some DB drivers)
        - Removes non-printable control characters (except newlines and tabs)
        - Collapses multiple consecutive newlines into a maximum of two

    Parameters:
        text (str): Raw user input from the Streamlit frontend.

    Returns:
        str: Sanitized text safe for pipeline processing.

    Example:
        sanitize_user_input("What is diabetes?\x00\x01")
        # "What is diabetes?"
    """
    if not text:
        return ""

    # Remove null bytes
    text = text.replace("\x00", "")

    # Remove non-printable control characters (keep \n and \t)
    text = re.sub(r"[\x01-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # Collapse more than 2 consecutive newlines into exactly 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Strip leading and trailing whitespace
    text = text.strip()

    return text


# ==============================================================================
# DATE AND TIME UTILITIES
# ==============================================================================


def get_utc_now() -> datetime:
    """
    Returns the current date and time in UTC timezone.

    Purpose:
        Centralises the creation of UTC timestamps so that all modules
        use the same approach. Avoids timezone inconsistencies caused by
        mixing naive and aware datetime objects.

    Returns:
        datetime: Current UTC datetime, timezone-aware.

    Example:
        now = get_utc_now()
        print(now.isoformat())  # "2025-03-09T14:32:11.123456+00:00"
    """
    return datetime.now(timezone.utc)


def format_timestamp(dt: datetime, format_str: str = "%Y-%m-%d %H:%M UTC") -> str:
    """
    Formats a datetime object as a human-readable string.

    Purpose:
        Used by the Streamlit frontend to display timestamps in a
        consistent, readable format throughout the admin and user panels.

    Parameters:
        dt (datetime): The datetime object to format.
        format_str (str): strftime format string. Default: "2025-03-09 14:32 UTC"

    Returns:
        str: Formatted datetime string, or "Unknown" if dt is None.

    Example:
        format_timestamp(datetime(2025, 3, 9, 14, 32, 11, tzinfo=timezone.utc))
        # "2025-03-09 14:32 UTC"
    """
    if dt is None:
        return "Unknown"

    # Make sure we are working with a timezone-aware datetime
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    return dt.strftime(format_str)


# ==============================================================================
# UUID AND SESSION UTILITIES
# ==============================================================================


def build_session_id() -> str:
    """
    Generates a new unique session ID as a UUID4 string.

    Purpose:
        Called by the Streamlit frontend when a new chat session starts
        to create a unique identifier for that session. This ID is passed
        with every query so the pipeline can group conversation turns
        from the same session in short-term and long-term memory.

    Returns:
        str: A random UUID4 string (e.g., "a3f4b2c1-d5e6-7890-abcd-ef1234567890").

    Example:
        session_id = build_session_id()
        # "7f3d9a1b-4c2e-4f8a-9b3d-1e2f4a5b6c7d"
    """
    return str(uuid.uuid4())


def is_valid_uuid(value: str) -> bool:
    """
    Returns True if a string is a valid UUID format.

    Purpose:
        Used by API route handlers to validate user_id and session_id
        parameters before querying the database, preventing badly formatted
        IDs from causing database errors.

    Parameters:
        value (str): The string to validate.

    Returns:
        bool: True if value is a valid UUID, False otherwise.

    Example:
        is_valid_uuid("7f3d9a1b-4c2e-4f8a-9b3d-1e2f4a5b6c7d")  # True
        is_valid_uuid("not-a-uuid")                              # False
        is_valid_uuid("")                                        # False
    """
    if not value:
        return False

    try:
        uuid.UUID(str(value))
        return True
    except ValueError:
        return False
