# ==============================================================================
# src/caching/query_cache.py
# ==============================================================================
# PURPOSE:
#   Caches AI-generated responses for repeated identical queries in PostgreSQL.
#   If the same question is asked again, we return the stored answer instantly
#   without calling the OpenAI API — saving both time and money.
#
# HOW IT WORKS (SHA-256 hashing):
#   1. Normalize the query: lowercase + strip whitespace
#      "What is Diabetes?" -> "what is diabetes?"
#
#   2. Hash the normalized query with SHA-256
#      "what is diabetes?" -> "a3f4b2c1..." (64-char hex string)
#
#   3. Look up the hash in the QueryCache table
#      FOUND:     return the cached response_text (no LLM call needed)
#      NOT FOUND: run the full RAG pipeline, then store the result
#
# WHY SHA-256 INSTEAD OF STORING THE FULL QUERY?
#   - Hashes are always exactly 64 characters — fast to index in PostgreSQL
#   - Two identical queries (after normalization) ALWAYS produce the same hash
#   - The hash protects against SQL injection in the lookup key
#   - The original query text is also stored separately for admin inspection
#
# WHY NORMALIZE BEFORE HASHING?
#   Without normalization, these three queries would produce different hashes
#   even though they are semantically identical:
#     "What is diabetes?"
#     "what is diabetes?"
#     "  What is diabetes?  "
#   After normalization (lowercase + strip), all three become "what is diabetes?"
#   and produce the same hash — so the cache hit works correctly.
#
# CACHE INVALIDATION:
#   Cache entries have an optional expires_at field. For this project we
#   set no expiry (entries live forever) since medical knowledge changes
#   slowly. The admin can manually clear the cache if needed.
#
# USED BY:
#   src/rag/pipeline.py (Step 2 — check, Step 9 — save)
# ==============================================================================

import hashlib
from datetime import datetime, timezone
from loguru import logger
from sqlalchemy.orm import Session
from typing import Optional

from src.database.models import QueryCache


def get_cached_response(query: str, db: Session) -> Optional[str]:
    """
    Looks up a query in the cache and returns the stored response if found.

    Purpose:
        Called by pipeline.py as Step 2 — immediately after the token budget
        check and before any expensive retrieval or generation operations.
        A cache hit means we skip Steps 3-10 entirely.

    How it works:
        1. Normalize and hash the query
        2. Query the QueryCache table for a matching hash
        3. If found and not expired: increment hit_count, update last_accessed_at,
           return the cached response text
        4. If not found or expired: return None (pipeline continues normally)

    Parameters:
        query (str):
            The raw user query, exactly as submitted.
        db (Session):
            The active SQLAlchemy database session.

    Returns:
        str: The cached response text if a valid cache entry exists.
        None: If no cache entry exists or the entry has expired.

    Example:
        cached = get_cached_response("What is diabetes?", db)

        if cached is not None:
            return cached   # Return instantly, no LLM call needed

        # Cache miss — continue with retrieval and generation
    """
    if not query or not query.strip():
        return None

    # Step 1: Compute the lookup hash for this query
    query_hash = _hash_query(query)

    # Step 2: Look up in the database
    cache_entry = (
        db.query(QueryCache).filter(QueryCache.query_hash == query_hash).first()
    )

    # Cache miss — no entry found
    if cache_entry is None:
        logger.debug(
            f"QueryCache: MISS for query hash {query_hash[:8]}... "
            f"(query: '{query[:60]}')"
        )
        return None

    # Check if this entry has expired
    if _is_expired(cache_entry):
        logger.info(
            f"QueryCache: EXPIRED entry found for hash {query_hash[:8]}... "
            f"Deleting and treating as a cache miss."
        )
        # Delete the expired entry so it gets rebuilt fresh
        db.delete(cache_entry)
        db.commit()
        return None

    # Cache hit — update statistics and return the stored response
    try:
        cache_entry.hit_count += 1
        cache_entry.last_accessed_at = datetime.now(timezone.utc)
        db.commit()
    except Exception as error:
        # Stats update failure should not prevent returning the cached response
        logger.warning(f"QueryCache: failed to update hit stats: {error}")
        db.rollback()

    logger.info(
        f"QueryCache: HIT for query hash {query_hash[:8]}... "
        f"(hit_count={cache_entry.hit_count}, "
        f"query: '{query[:60]}')"
    )

    return cache_entry.response_text


def save_response_to_cache(
    query: str,
    response_text: str,
    db: Session,
    expires_at: Optional[datetime] = None,
) -> bool:
    """
    Stores a query-response pair in the cache for future lookups.

    Purpose:
        Called by pipeline.py as Step 9, after the response has been
        generated and cleaned by output guardrails. Stores the result
        so the next identical query can be served instantly.

        Only called for non-flagged responses (flagged responses may be
        rejected by a human reviewer and should not be cached).

    Parameters:
        query (str):
            The original user query. Will be normalized before hashing.
        response_text (str):
            The final cleaned AI response to cache.
        db (Session):
            The active SQLAlchemy database session.
        expires_at (datetime, optional):
            When this cache entry should expire. None means never expires.
            Use this for time-sensitive medical information if needed.

    Returns:
        bool: True if the entry was saved successfully, False on error.
              A cache save failure never crashes the pipeline.

    Example:
        save_response_to_cache(
            query="What is diabetes?",
            response_text="Diabetes is a condition where...",
            db=db,
        )
    """
    if not query or not query.strip():
        logger.warning("QueryCache: save called with empty query. Skipping.")
        return False

    if not response_text or not response_text.strip():
        logger.warning("QueryCache: save called with empty response. Skipping.")
        return False

    query_hash = _hash_query(query)

    try:
        # Check if an entry already exists for this hash
        # (e.g., two simultaneous requests for the same query)
        existing_entry = (
            db.query(QueryCache).filter(QueryCache.query_hash == query_hash).first()
        )

        if existing_entry is not None:
            # Entry already exists — update it with the fresh response
            # This handles cases where the cache was cleared mid-flight
            existing_entry.response_text = response_text
            existing_entry.last_accessed_at = datetime.now(timezone.utc)
            existing_entry.expires_at = expires_at
            db.commit()

            logger.debug(
                f"QueryCache: UPDATED existing entry for hash {query_hash[:8]}..."
            )
            return True

        # Create a new cache entry
        new_entry = QueryCache(
            query_hash=query_hash,
            query_text=query,
            response_text=response_text,
            hit_count=1,
            expires_at=expires_at,
        )

        db.add(new_entry)
        db.commit()

        logger.info(
            f"QueryCache: SAVED new entry for hash {query_hash[:8]}... "
            f"(query: '{query[:60]}')"
        )

        return True

    except Exception as error:
        db.rollback()
        logger.error(
            f"QueryCache: failed to save entry for query '{query[:60]}': {error}"
        )
        return False


def delete_cache_entry(query: str, db: Session) -> bool:
    """
    Deletes a specific query from the cache by its hash.

    Purpose:
        Used by the admin panel to manually invalidate a specific cached
        response — for example, if the cached answer is outdated or incorrect
        and needs to be regenerated from the current knowledge base.

    Parameters:
        query (str): The original query whose cache entry should be deleted.
        db (Session): The active database session.

    Returns:
        bool: True if an entry was found and deleted, False if not found.
    """
    query_hash = _hash_query(query)

    try:
        entry = db.query(QueryCache).filter(QueryCache.query_hash == query_hash).first()

        if entry is None:
            logger.debug(
                f"QueryCache: delete called but no entry found for "
                f"hash {query_hash[:8]}..."
            )
            return False

        db.delete(entry)
        db.commit()

        logger.info(
            f"QueryCache: deleted entry for hash {query_hash[:8]}... "
            f"(query: '{query[:60]}')"
        )
        return True

    except Exception as error:
        db.rollback()
        logger.error(f"QueryCache: failed to delete entry: {error}")
        return False


def clear_all_cache(db: Session) -> int:
    """
    Deletes ALL entries from the query cache.

    Purpose:
        Admin-only operation. Used when the knowledge base (FAISS index)
        has been rebuilt with new data — in that case, all cached responses
        may be based on outdated retrieval results and should be cleared.

    Parameters:
        db (Session): The active database session.

    Returns:
        int: The number of cache entries that were deleted.

    Example:
        count = clear_all_cache(db)
        print(f"Cleared {count} cache entries.")
    """
    try:
        count = db.query(QueryCache).count()
        db.query(QueryCache).delete()
        db.commit()

        logger.info(f"QueryCache: cleared all {count} cache entries.")
        return count

    except Exception as error:
        db.rollback()
        logger.error(f"QueryCache: failed to clear all entries: {error}")
        return 0


def get_cache_stats(db: Session) -> dict:
    """
    Returns statistics about the current state of the query cache.

    Purpose:
        Used by the admin dashboard to show cache effectiveness —
        how many entries exist, total hits saved, and the most
        frequently cached queries.

    Parameters:
        db (Session): The active database session.

    Returns:
        dict:
            - total_entries (int): Number of cached queries
            - total_hits (int): Sum of all hit_counts (requests saved)
            - most_cached (list[dict]): Top 5 most frequently served entries
    """
    try:
        from sqlalchemy import func

        total_entries = db.query(func.count(QueryCache.id)).scalar() or 0
        total_hits = db.query(func.sum(QueryCache.hit_count)).scalar() or 0

        # Get the top 5 most frequently served cache entries
        top_entries = (
            db.query(QueryCache).order_by(QueryCache.hit_count.desc()).limit(5).all()
        )

        most_cached = [
            {
                "query_preview": entry.query_text[:80],
                "hit_count": entry.hit_count,
                "last_accessed_at": (
                    entry.last_accessed_at.isoformat()
                    if entry.last_accessed_at
                    else None
                ),
            }
            for entry in top_entries
        ]

        return {
            "total_entries": total_entries,
            "total_hits": int(total_hits),
            "most_cached": most_cached,
        }

    except Exception as error:
        logger.error(f"QueryCache: failed to get stats: {error}")
        return {
            "total_entries": 0,
            "total_hits": 0,
            "most_cached": [],
        }


# ------------------------------------------------------------------------------
# PRIVATE HELPERS
# ------------------------------------------------------------------------------


def _hash_query(query: str) -> str:
    """
    Produces a consistent SHA-256 hash for a query string.

    Purpose:
        Used as the cache lookup key. Normalizes the query first so that
        minor variations (capitalization, leading/trailing spaces) all
        map to the same hash and therefore the same cache entry.

    Normalization steps:
        1. Strip leading/trailing whitespace
        2. Convert to lowercase
        3. Collapse multiple internal spaces into one
           (so "what  is  diabetes" == "what is diabetes")

    Parameters:
        query (str): The raw query string to hash.

    Returns:
        str: A 64-character lowercase hex string (SHA-256 digest).

    Example:
        _hash_query("What is Diabetes?")
        # same result as _hash_query("what is diabetes?")
        # -> "a3f4b2c1d5e6..." (64 chars)
    """
    # Normalize: strip, lowercase, collapse multiple spaces
    normalized = " ".join(query.strip().lower().split())

    # Hash with SHA-256 and return as hex string
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _is_expired(cache_entry: QueryCache) -> bool:
    """
    Returns True if a cache entry has passed its expiry time.

    Parameters:
        cache_entry (QueryCache): The cache record to check.

    Returns:
        bool: True if expired, False if still valid or no expiry set.
    """
    if cache_entry.expires_at is None:
        # No expiry set — entry never expires
        return False

    now = datetime.now(timezone.utc)

    # Make expires_at timezone-aware if it is not already
    expires_at = cache_entry.expires_at
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)

    return now > expires_at
