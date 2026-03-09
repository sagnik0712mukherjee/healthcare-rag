# ==============================================================================
# src/rag/pipeline.py
# ==============================================================================
# PURPOSE:
#   Orchestrates the complete RAG flow from user query to final response.
#   This is the single function that api/routes.py calls for every query.
#
# THE COMPLETE FLOW (in order):
#
#   1.  Check token budget       <- Is this user allowed to make more queries?
#   2.  Check query cache        <- Has this exact question been asked before?
#   3.  Input guardrails         <- Is this query safe to process?
#   4.  Load short-term memory   <- What has this user said in this session?
#   5.  Retrieve context         <- Find relevant chunks from FAISS
#   6.  Generate response        <- Call OpenAI with query + context + history
#   7.  Output guardrails        <- Is the response safe to return?
#   8.  Human review check       <- Should this be flagged for expert review?
#   9.  Save to cache            <- Store result for future identical queries
#   10. Log token usage          <- Record cost and update user's token count
#   11. Save to memory           <- Update short-term and long-term memory
#   12. Return response          <- Send back to the user
#
# INPUT:
#   A RAGRequest dataclass (query, user_id, session_id, db session)
#
# OUTPUT:
#   A RAGResponse dataclass (response text, retrieved chunks, token usage, flags)
#
# USED BY:
#   src/api/routes.py
# ==============================================================================

from dataclasses import dataclass, field
from typing import Optional
from loguru import logger
from sqlalchemy.orm import Session

from src.rag.retriever import retrieve
from src.rag.generator import generate_response
from src.guardrails.input_guardrails import check_input_safety
from src.guardrails.output_guardrails import apply_output_guardrails
from src.memory.short_term_memory import ShortTermMemory
from src.memory.long_term_memory import save_conversation_to_db
from src.monitoring.token_tracker import check_user_token_budget
from src.monitoring.usage_logger import log_token_usage
from src.caching.query_cache import get_cached_response, save_response_to_cache
from src.database.models import HumanReviewQueue
from config.settings import settings
from src.utils.helpers import compute_risk_score


# ------------------------------------------------------------------------------
# REQUEST AND RESPONSE DATACLASSES
# ------------------------------------------------------------------------------


@dataclass
class RAGRequest:
    """
    All the inputs needed to run the RAG pipeline for one user query.

    Attributes:
        query (str): The user's question or medical query.
        user_id (str): The ID of the authenticated user making the request.
        session_id (str): The session identifier for short-term memory grouping.
        db (Session): The active SQLAlchemy database session for this request.
        top_k (int): How many chunks to retrieve. Defaults to settings value.
        source_filter (str, optional): Restrict retrieval to "clinical_case"
            or "image_caption". None returns both.
    """

    query: str
    user_id: str
    session_id: str
    db: Session
    top_k: int = None
    source_filter: Optional[str] = None

    def __post_init__(self):
        # Apply default top_k from settings if not provided
        if self.top_k is None:
            self.top_k = settings.retrieval_top_k


@dataclass
class RAGResponse:
    """
    All the outputs produced by running the RAG pipeline.

    Attributes:
        response_text (str): The final AI-generated response to show the user.
        retrieved_chunks (list[dict]): The context chunks used to generate it.
        input_tokens (int): Prompt tokens consumed by this request.
        output_tokens (int): Completion tokens consumed by this request.
        total_tokens (int): Total tokens consumed.
        served_from_cache (bool): True if the response came from cache (no LLM call).
        was_flagged_for_review (bool): True if this response needs human review.
        flag_reason (str): Why this response was flagged, if applicable.
        risk_score (float): Computed risk score for this response (0.0 to 1.0).
        error (str, optional): Error message if the pipeline failed.
        success (bool): False if the pipeline encountered a blocking error.
    """

    response_text: str = ""
    retrieved_chunks: list[dict] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    served_from_cache: bool = False
    was_flagged_for_review: bool = False
    flag_reason: str = ""
    risk_score: float = 0.0
    error: Optional[str] = None
    success: bool = True


# ------------------------------------------------------------------------------
# THE MAIN PIPELINE FUNCTION
# ------------------------------------------------------------------------------


def run_rag_pipeline(request: RAGRequest) -> RAGResponse:
    """
    Runs the complete RAG pipeline for a single user query.

    Purpose:
        This is the top-level orchestrator that ties together every module
        in the system. It is called once per user query from the API route
        handler and returns a complete RAGResponse.

        Each step is wrapped in clear logging so you can follow the pipeline
        in the application logs and debug any issues.

    Parameters:
        request (RAGRequest): Contains the query, user info, session, and db session.

    Returns:
        RAGResponse: Contains the response text, token usage, flags, and metadata.
            If a non-recoverable error occurs, returns a RAGResponse with
            success=False and an appropriate error message.

    Flow:
        Token budget check
            -> Cache check
                -> Input guardrails
                    -> Retrieve chunks
                        -> Generate response
                            -> Output guardrails
                                -> Human review check
                                    -> Cache save
                                        -> Token logging
                                            -> Memory save
                                                -> Return
    """
    logger.info(
        f"RAG pipeline started | "
        f"user_id={request.user_id} | "
        f"session_id={request.session_id} | "
        f"query='{request.query[:60]}...'"
    )

    # ==========================================================================
    # STEP 1: CHECK TOKEN BUDGET
    # ==========================================================================
    # Before doing any work, verify this user still has token budget remaining.
    # If they are over their limit, return immediately without calling the LLM.

    logger.info("Step 1/11: Checking user token budget...")

    budget_ok, budget_message = check_user_token_budget(
        user_id=request.user_id,
        db=request.db,
    )

    if not budget_ok:
        logger.warning(f"Token budget exceeded for user {request.user_id}.")
        return RAGResponse(
            response_text=budget_message,
            success=False,
            error="token_limit_exceeded",
        )

    # ==========================================================================
    # STEP 2: CHECK QUERY CACHE
    # ==========================================================================
    # If this exact query has been asked before, return the cached response
    # immediately — no retrieval, no LLM call, no token cost.

    logger.info("Step 2/11: Checking query cache...")

    cached_response = get_cached_response(
        query=request.query,
        db=request.db,
    )

    if cached_response is not None:
        logger.info("Cache HIT — returning cached response.")

        # Still log the cache hit (with 0 tokens) for audit purposes
        log_token_usage(
            user_id=request.user_id,
            query=request.query,
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            cost_usd=0.0,
            model_used=settings.openai_chat_model,
            served_from_cache=True,
            db=request.db,
        )

        return RAGResponse(
            response_text=cached_response,
            served_from_cache=True,
            success=True,
        )

    # ==========================================================================
    # STEP 3: INPUT GUARDRAILS
    # ==========================================================================
    # Check if the user's query is safe to process.
    # Unsafe queries (self-harm, drug abuse, dangerous medical advice requests)
    # are blocked here before any retrieval or generation happens.

    logger.info("Step 3/11: Running input guardrails...")

    is_safe, safety_message = check_input_safety(query=request.query)

    if not is_safe:
        logger.warning(
            f"Input guardrail triggered for user {request.user_id}: {safety_message}"
        )
        return RAGResponse(
            response_text=safety_message,
            success=False,
            error="input_blocked_by_guardrail",
        )

    # ==========================================================================
    # STEP 4: LOAD SHORT-TERM MEMORY
    # ==========================================================================
    # Get the conversation history for this session so the LLM has context
    # about previous turns (enables follow-up questions).

    logger.info("Step 4/11: Loading short-term conversation memory...")

    memory = ShortTermMemory()
    conversation_history = memory.get_history(session_id=request.session_id)

    logger.info(
        f"Loaded {len(conversation_history)} previous turns "
        f"for session {request.session_id}"
    )

    # ==========================================================================
    # STEP 5: RETRIEVE RELEVANT CHUNKS
    # ==========================================================================
    # Search the FAISS index for the most relevant clinical case chunks
    # and image captions for this query.

    logger.info(f"Step 5/11: Retrieving top-{request.top_k} chunks from FAISS...")

    try:
        retrieved_chunks = retrieve(
            query=request.query,
            top_k=request.top_k,
            source_filter=request.source_filter,
        )
    except Exception as error:
        logger.error(f"Retrieval failed: {error}")
        return RAGResponse(
            response_text=(
                "I'm sorry, I was unable to search the medical knowledge base "
                "at this time. Please try again in a moment."
            ),
            success=False,
            error=f"retrieval_error: {str(error)}",
        )

    logger.info(f"Retrieved {len(retrieved_chunks)} chunks.")

    # ==========================================================================
    # STEP 6: GENERATE RESPONSE
    # ==========================================================================
    # Call OpenAI with the query, retrieved context, and conversation history.

    logger.info("Step 6/11: Generating response with OpenAI...")

    try:
        generator_result = generate_response(
            query=request.query,
            retrieved_chunks=retrieved_chunks,
            conversation_history=conversation_history,
        )
    except Exception as error:
        logger.error(f"Generation failed: {error}")
        return RAGResponse(
            response_text=(
                "I'm sorry, I was unable to generate a response at this time. "
                "Please try again in a moment."
            ),
            success=False,
            error=f"generation_error: {str(error)}",
        )

    logger.info(
        f"Response generated. "
        f"Tokens: {generator_result.total_tokens} total "
        f"({generator_result.input_tokens} in, {generator_result.output_tokens} out)"
    )

    # ==========================================================================
    # STEP 7: OUTPUT GUARDRAILS
    # ==========================================================================
    # Clean the generated response:
    # - Remove any language that sounds like a diagnosis or prescription
    # - Ensure the medical disclaimer is present
    # The output guardrails return the cleaned response text.

    logger.info("Step 7/11: Applying output guardrails...")

    cleaned_response = apply_output_guardrails(
        response_text=generator_result.response_text,
    )

    # ==========================================================================
    # STEP 8: HUMAN REVIEW CHECK
    # ==========================================================================
    # Compute a risk score for the response.
    # If the score exceeds the threshold and human review is enabled,
    # flag this response for a healthcare professional to review.

    logger.info("Step 8/11: Computing risk score for human review check...")

    risk_score = compute_risk_score(
        query=request.query,
        response=cleaned_response,
    )

    was_flagged = False
    flag_reason = ""

    if (
        settings.human_review_enabled
        and risk_score >= settings.human_review_risk_threshold
    ):
        was_flagged = True
        flag_reason = (
            f"Risk score {risk_score:.2f} exceeded threshold "
            f"{settings.human_review_risk_threshold:.2f}"
        )

        logger.warning(
            f"Response flagged for human review. "
            f"risk_score={risk_score:.2f}, reason={flag_reason}"
        )

        # Save to the human review queue in the database
        _save_to_review_queue(
            request=request,
            response_text=cleaned_response,
            risk_score=risk_score,
            flag_reason=flag_reason,
        )

    # ==========================================================================
    # STEP 9: SAVE TO CACHE
    # ==========================================================================
    # Store the response so future identical queries can be served instantly.
    # We only cache non-flagged responses (flagged ones may be rejected later).

    logger.info("Step 9/11: Saving response to cache...")

    if not was_flagged:
        save_response_to_cache(
            query=request.query,
            response_text=cleaned_response,
            db=request.db,
        )

    # ==========================================================================
    # STEP 10: LOG TOKEN USAGE
    # ==========================================================================
    # Record this request's token usage in the database and update the
    # user's running total so we can enforce their budget on future queries.

    logger.info("Step 10/11: Logging token usage...")

    cost_usd = settings.get_total_cost_usd(
        input_tokens=generator_result.input_tokens,
        output_tokens=generator_result.output_tokens,
    )

    log_token_usage(
        user_id=request.user_id,
        query=request.query,
        input_tokens=generator_result.input_tokens,
        output_tokens=generator_result.output_tokens,
        total_tokens=generator_result.total_tokens,
        cost_usd=cost_usd,
        model_used=generator_result.model_used,
        served_from_cache=False,
        db=request.db,
    )

    # ==========================================================================
    # STEP 11: SAVE TO MEMORY
    # ==========================================================================
    # Update short-term memory with this turn so the next query in this
    # session has context about what was just discussed.
    # Also save to long-term memory (database) for audit and analytics.

    logger.info("Step 11/11: Saving to short-term and long-term memory...")

    # Update short-term (in-RAM) memory with this turn
    memory.add_turn(
        session_id=request.session_id,
        user_message=request.query,
        assistant_message=cleaned_response,
    )

    # Save to long-term (database) memory
    save_conversation_to_db(
        session_id=request.session_id,
        user_id=request.user_id,
        user_message=request.query,
        ai_response=cleaned_response,
        retrieved_chunks_count=len(retrieved_chunks),
        was_flagged=was_flagged,
        db=request.db,
    )

    # ==========================================================================
    # RETURN THE FINAL RESPONSE
    # ==========================================================================

    logger.info(
        f"RAG pipeline complete | "
        f"user_id={request.user_id} | "
        f"tokens={generator_result.total_tokens} | "
        f"cached=False | "
        f"flagged={was_flagged}"
    )

    return RAGResponse(
        response_text=cleaned_response,
        retrieved_chunks=retrieved_chunks,
        input_tokens=generator_result.input_tokens,
        output_tokens=generator_result.output_tokens,
        total_tokens=generator_result.total_tokens,
        served_from_cache=False,
        was_flagged_for_review=was_flagged,
        flag_reason=flag_reason,
        risk_score=risk_score,
        success=True,
    )


# ------------------------------------------------------------------------------
# PRIVATE HELPER: Save flagged response to the human review queue
# ------------------------------------------------------------------------------


def _save_to_review_queue(
    request: RAGRequest,
    response_text: str,
    risk_score: float,
    flag_reason: str,
) -> None:
    """
    Saves a flagged response to the HumanReviewQueue table in PostgreSQL.

    Purpose:
        When a response is deemed risky (above the threshold), this function
        creates a new row in the human_review_queue table so an admin can
        review and approve or reject it via the admin panel.

    Parameters:
        request (RAGRequest): The original pipeline request (for user/session info).
        response_text (str): The cleaned AI response that was flagged.
        risk_score (float): The computed risk score that triggered the flag.
        flag_reason (str): Human-readable explanation of why it was flagged.

    Returns:
        None
    """
    try:
        review_entry = HumanReviewQueue(
            session_id=request.session_id,
            user_id=request.user_id,
            user_query=request.query,
            ai_response=response_text,
            risk_score=risk_score,
            flag_reason=flag_reason,
        )

        request.db.add(review_entry)
        request.db.commit()

        logger.info("Flagged response saved to human review queue.")

    except Exception as error:
        # Do not let a review queue failure break the main pipeline
        # Log the error but continue returning the response
        logger.error(f"Failed to save to human review queue: {error}")
        request.db.rollback()
