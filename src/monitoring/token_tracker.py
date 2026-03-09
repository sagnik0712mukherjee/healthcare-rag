# ==============================================================================
# src/monitoring/token_tracker.py
# ==============================================================================
# PURPOSE:
#   Checks whether a user still has enough token budget remaining before
#   the RAG pipeline is allowed to make an OpenAI API call.
#
# WHY THIS EXISTS:
#   OpenAI charges per token. Without limits, a single user could exhaust
#   the entire application budget by sending thousands of queries.
#   Each user has a token_limit stored in the database (set at registration).
#   This module enforces that limit BEFORE the expensive LLM call happens.
#
# HOW IT FITS IN THE PIPELINE:
#   pipeline.py calls check_user_token_budget() as STEP 1 — the very first
#   thing that happens on every request. If the user is over their limit,
#   we return immediately without touching the retriever or the LLM.
#
#   Flow:
#     Request comes in
#       -> check_user_token_budget()
#           -> user is over limit? -> return error message immediately
#           -> user is under limit? -> continue to cache check, retrieval, generation
#
# WHAT "TOKEN LIMIT" MEANS IN PRACTICE:
#   The default limit is 100,000 tokens (set in config/settings.py).
#   With gpt-4o-mini pricing:
#     Input:  $0.00015 per 1K tokens
#     Output: $0.00060 per 1K tokens
#   100,000 tokens costs roughly $0.02-0.07 depending on input/output ratio.
#   This is intentionally low for an educational/demo system.
#   Production systems would set higher limits or use subscription tiers.
#
# USED BY:
#   src/rag/pipeline.py (Step 1)
# ==============================================================================

from loguru import logger
from sqlalchemy.orm import Session

from src.database.models import User
from config.settings import settings


def check_user_token_budget(
    user_id: str,
    db: Session,
) -> tuple[bool, str]:
    """
    Checks whether a user has remaining token budget to make a new request.

    Purpose:
        Called at the very start of the RAG pipeline before any expensive
        operations. Looks up the user's current token usage and limit in
        the database and decides whether to allow or block the request.

    Parameters:
        user_id (str):
            The ID of the user making the request.
        db (Session):
            The active SQLAlchemy database session for this request.

    Returns:
        tuple[bool, str]:
            - (True, "")
              The user is within their budget. The pipeline should continue.
              The empty string means no message needs to be shown.

            - (False, message)
              The user has exceeded their budget. The message string is a
              user-friendly explanation to display in the UI instead of
              an AI response.

    Example:
        budget_ok, message = check_user_token_budget(user_id="abc", db=db)

        if not budget_ok:
            return message  # Show this to the user

        # Otherwise continue the pipeline...
    """
    # Look up the user record in the database
    user = db.query(User).filter(User.id == user_id).first()

    # If user not found, something is wrong with authentication
    # Allow the request but log a warning — auth layer should have caught this
    if user is None:
        logger.warning(
            f"TokenTracker: user_id={user_id} not found in database. "
            f"Allowing request but this should be investigated."
        )
        return True, ""

    # Check if the user's account is active
    if not user.is_active:
        logger.warning(f"TokenTracker: user_id={user_id} account is deactivated.")
        return False, (
            "Your account has been deactivated. "
            "Please contact the administrator for assistance."
        )

    # Check if the user has exceeded their token limit
    if user.is_over_limit:
        tokens_used = user.tokens_used
        token_limit = user.token_limit

        # Calculate approximate cost for display
        cost_used_usd = settings.get_total_cost_usd(
            input_tokens=int(tokens_used * 0.7),  # rough 70/30 input/output split
            output_tokens=int(tokens_used * 0.3),
        )

        logger.warning(
            f"TokenTracker: user_id={user_id} has exceeded token limit. "
            f"Used: {tokens_used:,} / Limit: {token_limit:,} tokens."
        )

        return False, (
            f"You have reached your usage limit of {token_limit:,} tokens "
            f"(approximately ${cost_used_usd:.4f} USD used).\n\n"
            f"Your account has consumed {tokens_used:,} tokens in total. "
            f"Please contact the administrator to increase your limit or "
            f"reset your usage.\n\n"
            "This information is for educational purposes only. Please "
            "consult a qualified healthcare professional for medical advice, "
            "diagnosis, or treatment."
        )

    # User is within budget — log remaining allowance for monitoring
    tokens_remaining = user.tokens_remaining
    tokens_used = user.tokens_used
    token_limit = user.token_limit

    logger.debug(
        f"TokenTracker: user_id={user_id} budget OK. "
        f"Used: {tokens_used:,} / Limit: {token_limit:,} "
        f"({tokens_remaining:,} remaining)"
    )

    return True, ""


def get_user_token_summary(user_id: str, db: Session) -> dict:
    """
    Returns a complete token usage summary for a user.

    Purpose:
        Used by the admin panel and the user's profile page to display
        how much of their token budget has been consumed, what it cost,
        and how much they have remaining.

    Parameters:
        user_id (str): The user to look up.
        db (Session): The active database session.

    Returns:
        dict: A summary dictionary with the following keys:
            - user_id (str)
            - email (str)
            - tokens_used (int): Total tokens consumed so far
            - token_limit (int): Maximum tokens allowed
            - tokens_remaining (int): How many tokens are left
            - usage_percent (float): Percentage of limit consumed (0-100)
            - is_over_limit (bool): True if limit has been exceeded
            - estimated_cost_usd (float): Approximate dollar cost so far
            - found (bool): False if the user was not found in the database

    Example:
        summary = get_user_token_summary(user_id="abc-123", db=db)
        print(f"{summary['usage_percent']:.1f}% of budget used")
        print(f"${summary['estimated_cost_usd']:.4f} estimated cost")
    """
    user = db.query(User).filter(User.id == user_id).first()

    if user is None:
        logger.warning(
            f"TokenTracker: get_user_token_summary — user {user_id} not found."
        )
        return {"found": False, "user_id": user_id}

    # Calculate usage percentage safely (avoid division by zero)
    if user.token_limit > 0:
        usage_percent = round((user.tokens_used / user.token_limit) * 100, 2)
    else:
        usage_percent = 0.0

    # Estimate cost with a rough 70/30 input/output token split
    estimated_cost_usd = settings.get_total_cost_usd(
        input_tokens=int(user.tokens_used * 0.7),
        output_tokens=int(user.tokens_used * 0.3),
    )

    return {
        "found": True,
        "user_id": user.id,
        "email": user.email,
        "full_name": user.full_name,
        "tokens_used": user.tokens_used,
        "token_limit": user.token_limit,
        "tokens_remaining": user.tokens_remaining,
        "usage_percent": usage_percent,
        "is_over_limit": user.is_over_limit,
        "estimated_cost_usd": estimated_cost_usd,
        "is_active": user.is_active,
    }


def reset_user_token_usage(user_id: str, db: Session) -> bool:
    """
    Resets a user's token usage counter back to zero.

    Purpose:
        Used by administrators via the admin panel to give a user a fresh
        start — for example, at the beginning of a new billing period or
        after increasing their limit. This does NOT delete the usage logs
        (those stay in TokenUsageLog for audit purposes).

    Parameters:
        user_id (str): The user whose usage to reset.
        db (Session): The active database session.

    Returns:
        bool: True if reset was successful, False if user not found or error.

    Example:
        success = reset_user_token_usage(user_id="abc-123", db=db)
        if success:
            print("Token usage reset to 0.")
    """
    try:
        user = db.query(User).filter(User.id == user_id).first()

        if user is None:
            logger.warning(f"TokenTracker: reset failed — user {user_id} not found.")
            return False

        old_usage = user.tokens_used
        user.tokens_used = 0
        db.commit()

        logger.info(
            f"TokenTracker: reset token usage for user {user_id}. "
            f"Was: {old_usage:,} tokens. Now: 0."
        )

        return True

    except Exception as error:
        db.rollback()
        logger.error(f"TokenTracker: failed to reset usage for user {user_id}: {error}")
        return False


def update_user_token_limit(
    user_id: str,
    new_limit: int,
    db: Session,
) -> bool:
    """
    Updates a user's token limit to a new value.

    Purpose:
        Used by administrators to increase or decrease a specific user's
        token allowance. For example, giving a power user a higher limit
        or restricting a user who is misusing the system.

    Parameters:
        user_id (str): The user whose limit to change.
        new_limit (int): The new token limit. Must be a positive integer.
        db (Session): The active database session.

    Returns:
        bool: True if the update was successful, False otherwise.

    Example:
        success = update_user_token_limit(
            user_id="abc-123",
            new_limit=500_000,
            db=db,
        )
    """
    if new_limit < 0:
        logger.warning(
            f"TokenTracker: invalid new_limit={new_limit} for user {user_id}. "
            f"Limit must be a positive integer."
        )
        return False

    try:
        user = db.query(User).filter(User.id == user_id).first()

        if user is None:
            logger.warning(
                f"TokenTracker: update limit failed — user {user_id} not found."
            )
            return False

        old_limit = user.token_limit
        user.token_limit = new_limit
        db.commit()

        logger.info(
            f"TokenTracker: updated token limit for user {user_id}. "
            f"Old limit: {old_limit:,}. New limit: {new_limit:,}."
        )

        return True

    except Exception as error:
        db.rollback()
        logger.error(
            f"TokenTracker: failed to update limit for user {user_id}: {error}"
        )
        return False
