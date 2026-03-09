# ==============================================================================
# src/rag/retriever.py
# ==============================================================================
# PURPOSE:
#   Loads the FAISS index from disk and uses it to find the most relevant
#   chunks for a given user query.
#
# WHAT RETRIEVAL MEANS IN RAG:
#   When a user asks "What are symptoms of lung cancer?", we do NOT search
#   by keywords. Instead, we:
#     1. Convert the query into an embedding vector (a list of 1536 numbers)
#     2. Ask FAISS: "which stored vectors are closest to this query vector?"
#     3. FAISS returns the top-k most similar chunk positions
#     4. We look up those positions in metadata.json to get the actual text
#
#   "Close in vector space" = "similar in meaning"
#   So a query about "lung cancer symptoms" will retrieve chunks about
#   "pulmonary carcinoma presentation" even without exact word matches.
#
# INPUT:
#   - A user query string (e.g., "What causes chest pain?")
#   - Optional: filter by source type ("clinical_case" or "image_caption")
#
# OUTPUT:
#   A list of the top-k most relevant chunk dicts, each containing:
#   - chunk_text: the actual text to give to the LLM as context
#   - similarity_score: how similar this chunk is to the query (0.0 to 1.0)
#   - source: "clinical_case" or "image_caption"
#   - metadata: patient info, image labels, etc.
#
# USED BY:
#   src/rag/pipeline.py
# ==============================================================================

import json
import os
import numpy as np
import faiss
from loguru import logger
from typing import Optional

from config.settings import settings
from src.ingestion.embeddings import generate_single_embedding


# ------------------------------------------------------------------------------
# MODULE-LEVEL CACHE
# ------------------------------------------------------------------------------
# We load the FAISS index and metadata once when this module is first imported,
# then keep them in memory for the lifetime of the application.
#
# Why? Loading from disk takes ~1-2 seconds. If we loaded on every query,
# the app would be very slow. By caching at module level, we load once
# at startup and every subsequent query is instant.
#
# These are set to None initially and populated by _load_index_and_metadata().
# ------------------------------------------------------------------------------

_faiss_index: Optional[faiss.Index] = None
_chunk_metadata: Optional[list[dict]] = None


def _load_index_and_metadata() -> None:
    """
    Loads the FAISS index and metadata JSON from disk into module-level cache.

    Purpose:
        Called automatically the first time retrieve() is called.
        After that, the index and metadata stay in memory and this
        function is never called again (thanks to the global cache check).

    Returns:
        None

    Raises:
        FileNotFoundError: If index.bin or metadata.json do not exist.
            This means build_faiss_index.py has not been run yet.
    """
    global _faiss_index, _chunk_metadata

    # Check that the FAISS index file exists
    if not os.path.exists(settings.faiss_index_file):
        raise FileNotFoundError(
            f"FAISS index not found at: {settings.faiss_index_file}\n"
            f"You must run the ingestion pipeline first:\n"
            f"  python src/ingestion/build_faiss_index.py"
        )

    # Check that the metadata file exists
    if not os.path.exists(settings.faiss_metadata_file):
        raise FileNotFoundError(
            f"FAISS metadata not found at: {settings.faiss_metadata_file}\n"
            f"You must run the ingestion pipeline first:\n"
            f"  python src/ingestion/build_faiss_index.py"
        )

    logger.info(f"Loading FAISS index from: {settings.faiss_index_file}")
    _faiss_index = faiss.read_index(settings.faiss_index_file)
    logger.info(f"FAISS index loaded. Total vectors: {_faiss_index.ntotal}")

    logger.info(f"Loading chunk metadata from: {settings.faiss_metadata_file}")
    with open(settings.faiss_metadata_file, "r", encoding="utf-8") as f:
        _chunk_metadata = json.load(f)
    logger.info(f"Metadata loaded. Total chunks: {len(_chunk_metadata)}")


def retrieve(
    query: str,
    top_k: Optional[int] = None,
    source_filter: Optional[str] = None,
) -> list[dict]:
    """
    Finds the most relevant chunks for a user query using FAISS vector search.

    Purpose:
        This is the core retrieval function. It converts the user's query
        into a vector, searches the FAISS index for the nearest neighbors,
        and returns the matching chunks with their metadata.

    Parameters:
        query (str):
            The user's question or medical query.
            Example: "What are the symptoms of type 2 diabetes?"
        top_k (int, optional):
            How many chunks to retrieve. Defaults to settings.retrieval_top_k.
            Increase for more context, decrease for faster/cheaper responses.
        source_filter (str, optional):
            If provided, only return chunks from this source type.
            Options: "clinical_case", "image_caption", or None (return both).
            Example: source_filter="image_caption" only returns image context.

    Returns:
        list[dict]: A list of up to top_k chunk dicts, sorted by relevance
        (most relevant first). Each dict contains:
            - chunk_text (str): The text content of this chunk
            - similarity_score (float): Cosine similarity score (0.0 to 1.0)
            - source (str): "clinical_case" or "image_caption"
            - chunk_id (str): Unique identifier for this chunk
            - case_id (str, optional): For clinical case chunks
            - patient_age (int, optional): For clinical case chunks
            - patient_gender (str, optional): For clinical case chunks
            - image_id (str, optional): For image caption chunks
            - image_type (str, optional): For image caption chunks
            - labels (list, optional): For image caption chunks
            - file_name (str, optional): For image caption chunks

    Example:
        results = retrieve("symptoms of diabetes", top_k=5)
        for result in results:
            print(result["similarity_score"], result["chunk_text"][:100])
    """
    global _faiss_index, _chunk_metadata

    # Load index and metadata on first call (lazy loading)
    if _faiss_index is None or _chunk_metadata is None:
        _load_index_and_metadata()

    if not query or not query.strip():
        logger.warning("retrieve() called with an empty query. Returning empty list.")
        return []

    if top_k is None:
        top_k = settings.retrieval_top_k

    logger.info(f"Retrieving top-{top_k} chunks for query: '{query[:80]}...'")

    # --------------------------------------------------------------------------
    # STEP 1: Embed the query
    # --------------------------------------------------------------------------
    # Convert the user's text query into a 1536-dimensional vector.
    # This uses the same model that was used to embed the chunks during ingestion.
    # Using the same model is critical — different models produce incompatible vectors.

    query_vector = generate_single_embedding(text=query)

    # FAISS expects a 2D array even for a single query: shape (1, 1536)
    # Our generate_single_embedding returns shape (1536,) so we add a dimension
    query_vector_2d = np.expand_dims(query_vector, axis=0)

    # Normalize the query vector to unit length (same as what we did during indexing)
    # Without this, the inner product scores would not represent cosine similarity
    faiss.normalize_L2(query_vector_2d)

    # --------------------------------------------------------------------------
    # STEP 2: Search the FAISS index
    # --------------------------------------------------------------------------
    # We retrieve more than top_k results initially, in case the source_filter
    # removes some results. For example, if we want 5 results but filter to
    # "image_caption" only, we might need to fetch 20 candidates to get 5 matches.

    # How many candidates to fetch from FAISS before applying source filter
    fetch_k = top_k * 4 if source_filter else top_k

    # Ensure we don't ask for more vectors than exist in the index
    fetch_k = min(fetch_k, _faiss_index.ntotal)

    # faiss.search returns two arrays:
    #   similarity_scores: shape (1, fetch_k) — cosine similarity for each result
    #   faiss_indices:     shape (1, fetch_k) — position of each result in the index
    similarity_scores, faiss_indices = _faiss_index.search(query_vector_2d, fetch_k)

    # Remove the outer batch dimension (we only had 1 query)
    # After squeeze: similarity_scores.shape == (fetch_k,)
    #                faiss_indices.shape     == (fetch_k,)
    similarity_scores = similarity_scores[0]
    faiss_indices = faiss_indices[0]

    # --------------------------------------------------------------------------
    # STEP 3: Look up metadata for each result
    # --------------------------------------------------------------------------
    # FAISS returns integer positions. We map each position to its chunk dict
    # using the metadata list where metadata[position] = chunk dict.

    results = []

    for position, score in zip(faiss_indices, similarity_scores):
        # FAISS returns -1 for invalid/empty positions (can happen when
        # fetch_k > number of vectors in the index)
        if position == -1:
            continue

        # Get the chunk metadata for this position
        chunk = _chunk_metadata[position]

        # Apply source filter if specified
        if source_filter and chunk.get("source") != source_filter:
            continue

        # Build the result dict with all useful fields
        result = {
            # The text that will be injected into the LLM prompt as context
            "chunk_text": chunk.get("chunk_text", ""),
            # How similar this chunk is to the query (higher = more relevant)
            # Range: 0.0 (completely unrelated) to 1.0 (identical meaning)
            "similarity_score": float(score),
            # Source type — lets the frontend label results appropriately
            "source": chunk.get("source", "unknown"),
            # Universal fields present in all chunk types
            "chunk_id": chunk.get("chunk_id", ""),
            # Clinical case specific fields (None for image captions)
            "case_id": chunk.get("case_id", None),
            "patient_age": chunk.get("patient_age", None),
            "patient_gender": chunk.get("patient_gender", None),
            "chunk_index": chunk.get("chunk_index", 0),
            "total_chunks": chunk.get("total_chunks", 1),
            # Image caption specific fields (None for clinical cases)
            "image_id": chunk.get("image_id", None),
            "image_type": chunk.get("image_type", None),
            "image_subtype": chunk.get("image_subtype", None),
            "labels": chunk.get("labels", []),
            "file_name": chunk.get("file_name", None),
        }

        results.append(result)

        # Stop once we have enough results after filtering
        if len(results) >= top_k:
            break

    logger.info(
        f"Retrieved {len(results)} chunks "
        f"(source_filter={source_filter}, top_k={top_k})"
    )

    # Log a brief summary of the top result for debugging
    if results:
        top = results[0]
        logger.debug(
            f"Top result: score={top['similarity_score']:.4f} "
            f"source={top['source']} "
            f"text='{top['chunk_text'][:80]}...'"
        )

    return results
