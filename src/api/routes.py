# ==============================================================================
# src/api/routes.py
# ==============================================================================
# PURPOSE:
#   Defines all FastAPI route handlers. This is the HTTP interface that
#   the Streamlit frontend calls to interact with the RAG pipeline,
#   authentication system, and admin tools.
#
# ROUTE MAP:
#
#   PUBLIC ROUTES (no auth required):
#     POST /auth/register         Register a new user account
#     POST /auth/login            Login and receive a JWT token
#     GET  /health                Health check (DB, FAISS, memory status)
#
#   PATIENT ROUTES (requires valid JWT):
#     POST /query                 Submit a medical query → get AI response
#     GET  /history               Get the current user's conversation history
#
#   ADMIN ROUTES (requires valid JWT + admin role):
#     GET  /admin/review          List all pending flagged responses
#     POST /admin/review/{id}     Submit approve/reject for a flagged response
#     GET  /admin/users           List all registered users with token stats
#     POST /admin/users/{id}/reset-tokens    Reset a user's token usage to 0
#     POST /admin/users/{id}/token-limit     Update a user's token limit
#     GET  /admin/cache/stats     View query cache statistics
#     DELETE /admin/cache         Clear the entire query cache
#
# AUTHENTICATION:
#   We use JWT (JSON Web Tokens) via python-jose + passlib[bcrypt].
#   Flow:
#     1. User registers → password is bcrypt-hashed and stored
#     2. User logs in → password verified → JWT issued with user_id + role
#     3. User sends JWT in Authorization: Bearer <token> header
#     4. get_current_user() dependency decodes the JWT and loads the User
#     5. require_admin() dependency adds an additional role check on top
#
# USED BY:
#   src/api/main.py (registers this router with the FastAPI app)
# ==============================================================================

from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from loguru import logger
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from config.settings import settings
from src.api.schemas import (
    CacheClearResponse,
    CacheStatsResponse,
    HealthResponse,
    LoginRequest,
    MessageResponse,
    QueryRequest,
    QueryResponse,
    ReviewAction,
    ReviewQueueItemResponse,
    RetrievedChunkResponse,
    TokenLimitUpdateAction,
    TokenResetAction,
    TokenResponse,
    UserCreateRequest,
    UserResponse,
)
from src.caching.query_cache import clear_all_cache, get_cache_stats
from src.database.db import check_db_connection, get_db
from src.database.models import (
    ConversationHistory,
    HumanReviewQueue,
    ReviewStatus,
    User,
    UserRole,
)
from src.memory.short_term_memory import ShortTermMemory
from src.monitoring.token_tracker import (
    reset_user_token_usage,
    update_user_token_limit,
)
from src.rag.pipeline import RAGRequest, run_rag_pipeline
from src.rag import retriever as _retriever_module
from src.utils.helpers import is_valid_uuid, sanitize_user_input

# ------------------------------------------------------------------------------
# ROUTER INSTANCE
# Main router that main.py will include with app.include_router()
# ------------------------------------------------------------------------------
router = APIRouter()

# ------------------------------------------------------------------------------
# AUTH UTILITIES
# ------------------------------------------------------------------------------

# JWT configuration
JWT_ALGORITHM = "HS256"
JWT_EXPIRY_MINUTES = 60 * 24  # 24 hours

# bcrypt context for password hashing and verification
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# HTTP Bearer scheme for extracting JWT from Authorization header
bearer_scheme = HTTPBearer()


def _hash_password(plain_password: str) -> str:
    """Hashes a plain-text password using bcrypt."""
    return pwd_context.hash(plain_password)


def _verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifies a plain-text password against its bcrypt hash."""
    return pwd_context.verify(plain_password, hashed_password)


def _create_access_token(user_id: str, role: str) -> str:
    """
    Creates a signed JWT token containing the user's ID and role.

    The token expires after JWT_EXPIRY_MINUTES (default: 24 hours).
    The token is signed with settings.secret_key so it cannot be forged.
    """
    expire = datetime.now(timezone.utc) + timedelta(minutes=JWT_EXPIRY_MINUTES)
    payload = {
        "sub": user_id,  # Subject: the user's UUID
        "role": role,  # "patient" or "admin"
        "exp": expire,  # Expiry timestamp
    }
    return jwt.encode(payload, settings.secret_key, algorithm=JWT_ALGORITHM)


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    db: Session = Depends(get_db),
) -> User:
    """
    FastAPI dependency: decodes the JWT and returns the authenticated User.

    Purpose:
        Used as a Depends() argument on all protected routes. FastAPI
        automatically calls this before the route handler. If authentication
        fails for any reason, a 401 Unauthorized error is raised before
        the route handler runs.

    Raises:
        HTTPException 401: If the token is missing, expired, or invalid.
        HTTPException 401: If the user_id from the token doesn't exist in the DB.
        HTTPException 403: If the user's account has been deactivated.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(
            credentials.credentials,
            settings.secret_key,
            algorithms=[JWT_ALGORITHM],
        )
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = db.query(User).filter(User.id == user_id).first()

    if user is None:
        raise credentials_exception

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Your account has been deactivated. Contact an administrator.",
        )

    # Update last_active_at timestamp on each authenticated request
    try:
        user.last_active_at = datetime.now(timezone.utc)
        db.commit()
    except Exception:
        db.rollback()

    return user


def require_admin(current_user: User = Depends(get_current_user)) -> User:
    """
    FastAPI dependency: extends get_current_user() with an admin role check.

    Purpose:
        Used as a Depends() argument on all /admin/* routes. Ensures
        only users with role="admin" can access admin endpoints.

    Raises:
        HTTPException 403: If the authenticated user is not an admin.
    """
    if current_user.role != UserRole.admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required.",
        )
    return current_user


# ==============================================================================
# PUBLIC ROUTES
# ==============================================================================


@router.get("/health", response_model=HealthResponse, tags=["System"])
def health_check(db: Session = Depends(get_db)):
    """
    Returns the current health status of the application.

    Checks:
    - Database connectivity (SELECT 1)
    - FAISS index load status and vector count
    - Number of active short-term memory sessions

    Used by Railway deployment health checks and the Streamlit
    frontend to verify the API is reachable before rendering the UI.
    """
    # Check database connection
    db_status = "connected" if check_db_connection(db) else "error: unreachable"

    # Check FAISS index — read through module to get the live value,
    # not the stale None that was captured at import time
    _live_index = _retriever_module._faiss_index
    if _live_index is not None:
        faiss_status = f"loaded ({_live_index.ntotal:,} vectors)"
    else:
        faiss_status = "not loaded (run ingestion pipeline first)"

    # Count active short-term memory sessions
    memory = ShortTermMemory()
    active_sessions = memory.get_active_session_count()

    overall_status = "healthy" if "error" not in db_status else "degraded"

    return HealthResponse(
        status=overall_status,
        app_env=settings.app_env,
        database=db_status,
        faiss_index=faiss_status,
        active_sessions=active_sessions,
    )


@router.post("/auth/register", response_model=UserResponse, tags=["Auth"])
def register_user(
    request: UserCreateRequest,
    db: Session = Depends(get_db),
):
    """
    Registers a new user account.

    - Validates the email is not already taken
    - Hashes the password with bcrypt before storage
    - Assigns the default token limit from settings
    - Returns the new user's public profile (no password)

    Raises:
        400: If the email is already registered.
    """
    # Check for duplicate email
    existing = db.query(User).filter(User.email == request.email).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Email '{request.email}' is already registered.",
        )

    new_user = User(
        email=request.email,
        hashed_password=_hash_password(request.password),
        full_name=request.full_name,
        role=UserRole(request.role),
        token_limit=settings.default_token_limit,
    )

    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    logger.info(f"New user registered: {new_user.email} (role={new_user.role})")

    return new_user


@router.post("/auth/login", response_model=TokenResponse, tags=["Auth"])
def login(
    request: LoginRequest,
    db: Session = Depends(get_db),
):
    """
    Authenticates a user and returns a JWT access token.

    The token must be sent in the Authorization: Bearer <token> header
    on all subsequent protected requests.

    Raises:
        401: If the email doesn't exist or the password is incorrect.
    """
    user = db.query(User).filter(User.email == request.email).first()

    if user is None or not _verify_password(request.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password.",
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Your account has been deactivated.",
        )

    token = _create_access_token(user_id=user.id, role=user.role.value)

    logger.info(f"User logged in: {user.email}")

    return TokenResponse(
        access_token=token,
        user_id=user.id,
        email=user.email,
        role=user.role.value,
        full_name=user.full_name,
    )


# ==============================================================================
# PATIENT ROUTES (requires auth)
# ==============================================================================


@router.post("/query", response_model=QueryResponse, tags=["Query"])
def submit_query(
    request: QueryRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Submits a medical query to the RAG pipeline and returns the AI response.

    This is the core endpoint of the entire application. It:
    1. Sanitizes the user's query
    2. Passes it to run_rag_pipeline() which handles all 11 steps
    3. Returns the response with source citations and token usage

    The session_id groups this query with others in the same conversation
    so the AI has memory of previous turns.

    Raises:
        400: If the query is empty after sanitization.
        500: If the pipeline encounters an unrecoverable error.
    """
    # Sanitize the raw query before processing
    clean_query = sanitize_user_input(request.query)

    if not clean_query:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query cannot be empty.",
        )

    logger.info(
        f"Query received | user={current_user.email} | "
        f"session={request.session_id[:8]}... | "
        f"query='{clean_query[:60]}...'"
    )

    # Build the pipeline request and run it
    pipeline_request = RAGRequest(
        query=clean_query,
        user_id=current_user.id,
        session_id=request.session_id,
        db=db,
        top_k=request.top_k,
        source_filter=request.source_filter,
    )

    result = run_rag_pipeline(pipeline_request)

    # Map the retrieved chunks to the response schema
    chunks_response = [
        RetrievedChunkResponse(
            chunk_text=chunk["chunk_text"],
            similarity_score=chunk["similarity_score"],
            source=chunk["source"],
            chunk_id=chunk["chunk_id"],
            case_id=chunk.get("case_id"),
            patient_age=chunk.get("patient_age"),
            patient_gender=chunk.get("patient_gender"),
            image_id=chunk.get("image_id"),
            image_type=chunk.get("image_type"),
            labels=chunk.get("labels", []),
        )
        for chunk in result.retrieved_chunks
    ]

    return QueryResponse(
        response_text=result.response_text,
        retrieved_chunks=chunks_response,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        total_tokens=result.total_tokens,
        served_from_cache=result.served_from_cache,
        was_flagged_for_review=result.was_flagged_for_review,
        risk_score=result.risk_score,
        success=result.success,
        error=result.error,
    )


@router.get("/history", tags=["Query"])
def get_my_history(
    limit: int = 50,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Returns the authenticated user's conversation history.

    Returns the most recent `limit` Q&A pairs, sorted newest-first.
    Users can review past medical information they received.
    """
    limit = max(1, min(limit, 200))  # Clamp between 1 and 200

    records = (
        db.query(ConversationHistory)
        .filter(ConversationHistory.user_id == current_user.id)
        .order_by(ConversationHistory.created_at.desc())
        .limit(limit)
        .all()
    )

    return [
        {
            "id": r.id,
            "session_id": r.session_id,
            "user_message": r.user_message,
            "ai_response": r.ai_response,
            "retrieved_chunks_count": r.retrieved_chunks_count,
            "was_flagged": r.was_flagged,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in records
    ]


@router.get("/me", response_model=UserResponse, tags=["Auth"])
def get_my_profile(current_user: User = Depends(get_current_user)):
    """
    Returns the authenticated user's profile and token usage summary.
    """
    return current_user


# ==============================================================================
# ADMIN ROUTES — HUMAN REVIEW
# ==============================================================================


@router.get(
    "/admin/review",
    response_model=list[ReviewQueueItemResponse],
    tags=["Admin - Review"],
)
def list_review_queue(
    status_filter: Optional[str] = "pending",
    limit: int = 50,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """
    Lists all items in the human review queue.

    Query params:
    - status_filter: "pending", "approved", "rejected", or None for all
    - limit: max records to return (default 50)

    Returns flagged AI responses awaiting review by a healthcare professional.
    """
    query = db.query(HumanReviewQueue)

    if status_filter:
        try:
            status_enum = ReviewStatus(status_filter)
            query = query.filter(HumanReviewQueue.status == status_enum)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid status_filter '{status_filter}'. "
                f"Use 'pending', 'approved', or 'rejected'.",
            )

    items = (
        query.order_by(HumanReviewQueue.created_at.desc())
        .limit(max(1, min(limit, 200)))
        .all()
    )

    return items


@router.post(
    "/admin/review/{item_id}",
    response_model=MessageResponse,
    tags=["Admin - Review"],
)
def submit_review_decision(
    item_id: int,
    action: ReviewAction,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """
    Submits an approve or reject decision for a flagged AI response.

    - "approve": The response is deemed safe. Marks it as approved.
    - "reject":  The response was unsafe. Marks it as rejected.

    The reviewer's notes and timestamp are recorded for audit purposes.

    Raises:
        404: If the review item does not exist.
        400: If the item has already been reviewed.
    """
    item = db.query(HumanReviewQueue).filter(HumanReviewQueue.id == item_id).first()

    if item is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Review item {item_id} not found.",
        )

    if item.status != ReviewStatus.pending:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Item {item_id} has already been reviewed (status: {item.status.value}).",
        )

    item.status = ReviewStatus(action.action + "d")  # "approved" or "rejected"
    item.reviewed_by_user_id = admin.id
    item.reviewer_notes = action.reviewer_notes
    item.reviewed_at = datetime.now(timezone.utc)

    db.commit()

    logger.info(
        f"Review decision submitted: item_id={item_id}, "
        f"action={action.action}, reviewer={admin.email}"
    )

    return MessageResponse(
        success=True,
        message=f"Response {action.action}d successfully.",
    )


# ==============================================================================
# ADMIN ROUTES — USER MANAGEMENT
# ==============================================================================


@router.get(
    "/admin/users",
    response_model=list[UserResponse],
    tags=["Admin - Users"],
)
def list_all_users(
    limit: int = 100,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """
    Returns a list of all registered users with their token usage stats.

    Used by the admin panel to monitor user activity and manage budgets.
    """
    users = (
        db.query(User)
        .order_by(User.created_at.desc())
        .limit(max(1, min(limit, 500)))
        .all()
    )
    return users


@router.post(
    "/admin/users/{user_id}/reset-tokens",
    response_model=MessageResponse,
    tags=["Admin - Users"],
)
def reset_user_tokens(
    user_id: str,
    action: TokenResetAction,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """
    Resets a specific user's token usage counter back to zero.

    The user's token_limit is not changed — only their tokens_used.
    The full usage history remains in TokenUsageLog for audit purposes.

    Requires confirm=true in the request body to prevent accidental resets.

    Raises:
        400: If confirm is not True, or if user_id is not a valid UUID.
        404: If the user does not exist.
    """
    if not action.confirm:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Must set confirm=true to reset token usage.",
        )

    if not is_valid_uuid(user_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid user_id format.",
        )

    success = reset_user_token_usage(user_id=user_id, db=db)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User {user_id} not found.",
        )

    logger.info(f"Admin {admin.email} reset token usage for user {user_id}.")

    return MessageResponse(
        success=True,
        message=f"Token usage for user {user_id} has been reset to 0.",
    )


@router.post(
    "/admin/users/{user_id}/token-limit",
    response_model=MessageResponse,
    tags=["Admin - Users"],
)
def update_token_limit(
    user_id: str,
    action: TokenLimitUpdateAction,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """
    Updates the token limit for a specific user.

    Useful for giving power users a higher allowance or restricting
    users who are misusing the system.

    Raises:
        400: If user_id is not a valid UUID.
        404: If the user does not exist.
    """
    if not is_valid_uuid(user_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid user_id format.",
        )

    success = update_user_token_limit(
        user_id=user_id,
        new_limit=action.new_limit,
        db=db,
    )

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User {user_id} not found.",
        )

    logger.info(
        f"Admin {admin.email} updated token limit for user {user_id} "
        f"to {action.new_limit:,}."
    )

    return MessageResponse(
        success=True,
        message=f"Token limit for user {user_id} updated to {action.new_limit:,}.",
    )


# ==============================================================================
# ADMIN ROUTES — CACHE MANAGEMENT
# ==============================================================================


@router.get(
    "/admin/cache/stats",
    response_model=CacheStatsResponse,
    tags=["Admin - Cache"],
)
def get_query_cache_stats(
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """
    Returns statistics about the query cache.

    Shows total cached entries, total cache hits (requests saved),
    and the top 5 most frequently served queries.
    """
    stats = get_cache_stats(db=db)
    return CacheStatsResponse(**stats)


@router.delete(
    "/admin/cache",
    response_model=CacheClearResponse,
    tags=["Admin - Cache"],
)
def clear_query_cache(
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """
    Clears ALL entries from the query cache.

    Use this after rebuilding the FAISS index with new data — cached
    responses may be based on outdated retrieval results.

    This action is irreversible.
    """
    count = clear_all_cache(db=db)

    logger.info(
        f"Admin {admin.email} cleared the query cache. {count} entries removed."
    )

    return CacheClearResponse(
        entries_cleared=count,
        message=f"Successfully cleared {count} cache entries.",
    )
