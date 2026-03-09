# ==============================================================================
# src/monitoring/usage_logger.py
# ==============================================================================
# PURPOSE:
#   Writes a detailed record of every LLM API call to the TokenUsageLog
#   table in PostgreSQL and updates the user's running token total in the
#   User table — all in a single database transaction.
#
# WHY TWO WRITES IN ONE TRANSACTION?
#   We need to update two things after every request:
#     1. TokenUsageLog: insert a new row with full request details
#        (tokens used, cost, model, query text, timestamp)
#     2. User.tokens_used: increment the user's running total
#        (this is what token_tracker.py checks on the NEXT request)
#
#   Both updates happen together in a single db.commit() so they are
#   always in sync. If the commit fails, both are rolled back — we never
#   have a log entry without the user total being updated, or vice versa.
#
# WHAT GETS LOGGED:
#   - Which user made the request (user_id)
#   - What they asked (query text, truncated to 1000 chars)
#   - How many tokens were used (input, output, total)
#   - What it cost in USD
#   - Which model was used
#   - Whether the response came from cache (in which case tokens = 0)
#   - When it happened (timestamp auto-set by the model)
#
# USED BY:
#   src/rag/pipeline.py (Step 10)
# ==============================================================================

from loguru import logger
from sqlalchemy.orm import Session

from src.database.models import TokenUsageLog, User


# Maximum characters of query text to store in the log
# We truncate to avoid storing very long queries that waste DB space
QUERY_TEXT_MAX_LENGTH = 1000


def log_token_usage(
    user_id: str,
    query: str,
    input_tokens: int,
    output_tokens: int,
    total_tokens: int,
    cost_usd: float,
    model_used: str,
    served_from_cache: bool,
    db: Session,
) -> bool:
    """
    Logs token usage for one request and updates the user's running token total.

    Purpose:
        Called by pipeline.py at Step 10 after every successful LLM generation
        (or cache hit, where tokens are 0). Creates a permanent audit record
        and keeps the user's token balance up to date for budget enforcement.

    Parameters:
        user_id (str):
            The ID of the user who made this request.
        query (str):
            The user's query text. Stored truncated to QUERY_TEXT_MAX_LENGTH
            characters to limit database storage.
        input_tokens (int):
            Number of prompt/input tokens consumed. 0 if served from cache.
        output_tokens (int):
            Number of completion/output tokens consumed. 0 if served from cache.
        total_tokens (int):
            Total tokens (input + output). 0 if served from cache.
        cost_usd (float):
            Estimated cost of this request in USD. 0.0 if served from cache.
        model_used (str):
            The OpenAI model name used (e.g., "gpt-4o-mini").
        served_from_cache (bool):
            True if the response came from the query cache (no LLM call made).
            In this case, token counts and cost should all be 0.
        db (Session):
            The active SQLAlchemy database session for this request.

    Returns:
        bool: True if logging succeeded, False if an error occurred.
              Errors are logged but not re-raised — a logging failure should
              never crash the pipeline or prevent the user from getting a response.

    Example:
        success = log_token_usage(
            user_id="user-abc",
            query="What are symptoms of diabetes?",
            input_tokens=450,
            output_tokens=312,
            total_tokens=762,
            cost_usd=0.0002541,
            model_used="gpt-4o-mini",
            served_from_cache=False,
            db=db,
        )
    """
    try:
        # ------------------------------------------------------------------
        # WRITE 1: Insert a new TokenUsageLog record
        # ------------------------------------------------------------------
        # Truncate the query text to avoid storing huge strings in the DB.
        # The full query is already stored in ConversationHistory if needed.
        truncated_query = query[:QUERY_TEXT_MAX_LENGTH] if query else ""

        log_entry = TokenUsageLog(
            user_id=user_id,
            query_text=truncated_query,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cost_usd=cost_usd,
            model_used=model_used,
            served_from_cache=served_from_cache,
        )

        db.add(log_entry)

        # ------------------------------------------------------------------
        # WRITE 2: Update the user's running token total
        # ------------------------------------------------------------------
        # We increment tokens_used by the total_tokens from this request.
        # This is what token_tracker.py reads on the next request to decide
        # whether the user is still within their budget.
        #
        # We skip incrementing for cache hits (total_tokens = 0 anyway)
        # but we still log the cache hit for analytics purposes.

        if total_tokens > 0:
            user = db.query(User).filter(User.id == user_id).first()

            if user is not None:
                user.tokens_used += total_tokens

                logger.debug(
                    f"UsageLogger: updated user {user_id[:8]}... tokens_used "
                    f"by +{total_tokens:,} "
                    f"(new total: {user.tokens_used:,})"
                )
            else:
                logger.warning(
                    f"UsageLogger: could not find user {user_id} to update "
                    f"tokens_used. Log entry will still be saved."
                )

        # ------------------------------------------------------------------
        # Commit both writes together in a single transaction
        # ------------------------------------------------------------------
        db.commit()

        logger.info(
            f"UsageLogger: logged request for user {user_id[:8]}... | "
            f"tokens={total_tokens:,} | "
            f"cost=${cost_usd:.6f} | "
            f"cached={served_from_cache} | "
            f"model={model_used}"
        )

        return True

    except Exception as error:
        # Roll back both writes if anything goes wrong
        db.rollback()

        # Log the error but return False rather than raising
        # A logging failure must never break the user's experience
        logger.error(
            f"UsageLogger: failed to log token usage for user {user_id}: {error}"
        )

        return False


def get_usage_logs_for_user(
    user_id: str,
    db: Session,
    limit: int = 100,
) -> list[dict]:
    """
    Retrieves recent token usage log entries for a specific user.

    Purpose:
        Used by the admin panel to show a detailed per-request breakdown
        of a user's token consumption. Helps admins investigate unusual
        usage patterns or verify billing calculations.

    Parameters:
        user_id (str): The user whose logs to retrieve.
        db (Session): The active database session.
        limit (int): Maximum number of log entries to return. Default: 100.

    Returns:
        list[dict]: List of usage log dicts, most recent first. Each dict:
            - id (int): Log entry ID
            - query_text (str): The query (truncated)
            - input_tokens (int)
            - output_tokens (int)
            - total_tokens (int)
            - cost_usd (float)
            - model_used (str)
            - served_from_cache (bool)
            - created_at (str): ISO format timestamp

    Example:
        logs = get_usage_logs_for_user(user_id="abc-123", db=db, limit=20)
        for log in logs:
            print(f"{log['created_at']}: {log['total_tokens']} tokens")
    """
    try:
        records = (
            db.query(TokenUsageLog)
            .filter(TokenUsageLog.user_id == user_id)
            .order_by(TokenUsageLog.created_at.desc())
            .limit(limit)
            .all()
        )

        return [_log_record_to_dict(record) for record in records]

    except Exception as error:
        logger.error(
            f"UsageLogger: failed to retrieve logs for user {user_id}: {error}"
        )
        return []


def get_system_wide_usage_stats(db: Session) -> dict:
    """
    Returns aggregated token usage statistics across all users.

    Purpose:
        Powers the admin dashboard's system-wide usage overview.
        Shows total tokens consumed, total estimated cost, and
        how many requests were served from cache vs. the LLM.

    Parameters:
        db (Session): The active database session.

    Returns:
        dict: System-wide statistics:
            - total_requests (int): Total log entries
            - total_tokens_used (int): Sum of all tokens consumed
            - total_cost_usd (float): Sum of all estimated costs
            - cache_hits (int): Requests served from cache
            - llm_calls (int): Requests that actually called the LLM
            - cache_hit_rate_percent (float): Percentage served from cache
    """
    try:
        from sqlalchemy import func

        stats = db.query(
            func.count(TokenUsageLog.id).label("total_requests"),
            func.sum(TokenUsageLog.total_tokens).label("total_tokens"),
            func.sum(TokenUsageLog.cost_usd).label("total_cost"),
            func.sum(
                # Count rows where served_from_cache is True
                # SQLAlchemy casts bool to int for summing
                TokenUsageLog.served_from_cache.cast(
                    db.bind.dialect.type_descriptor(__import__("sqlalchemy").Integer)
                )
            ).label("cache_hits"),
        ).first()

        total_requests = stats.total_requests or 0
        total_tokens = int(stats.total_tokens or 0)
        total_cost = float(stats.total_cost or 0.0)
        cache_hits = int(stats.cache_hits or 0)
        llm_calls = total_requests - cache_hits
        cache_hit_rate = (
            round((cache_hits / total_requests) * 100, 2) if total_requests > 0 else 0.0
        )

        return {
            "total_requests": total_requests,
            "total_tokens_used": total_tokens,
            "total_cost_usd": round(total_cost, 6),
            "cache_hits": cache_hits,
            "llm_calls": llm_calls,
            "cache_hit_rate_percent": cache_hit_rate,
        }

    except Exception as error:
        logger.error(f"UsageLogger: failed to get system stats: {error}")
        return {
            "total_requests": 0,
            "total_tokens_used": 0,
            "total_cost_usd": 0.0,
            "cache_hits": 0,
            "llm_calls": 0,
            "cache_hit_rate_percent": 0.0,
        }


# ------------------------------------------------------------------------------
# PRIVATE HELPER
# ------------------------------------------------------------------------------


def _log_record_to_dict(record: TokenUsageLog) -> dict:
    """
    Converts a TokenUsageLog ORM object to a plain Python dictionary.

    Parameters:
        record (TokenUsageLog): A database record from the token_usage_logs table.

    Returns:
        dict: All fields as JSON-serializable Python types.
    """
    return {
        "id": record.id,
        "user_id": record.user_id,
        "query_text": record.query_text,
        "input_tokens": record.input_tokens,
        "output_tokens": record.output_tokens,
        "total_tokens": record.total_tokens,
        "cost_usd": record.cost_usd,
        "model_used": record.model_used,
        "served_from_cache": record.served_from_cache,
        "created_at": (record.created_at.isoformat() if record.created_at else None),
    }
