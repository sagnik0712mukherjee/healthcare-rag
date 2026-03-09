# ==============================================================================
# src/ingestion/embeddings.py
# ==============================================================================
# PURPOSE:
#   Generates vector embeddings for a list of text strings using the
#   OpenAI Embeddings API (text-embedding-3-small by default).
#
# WHAT IS AN EMBEDDING?
#   An embedding is a list of floating-point numbers (a vector) that
#   represents the semantic meaning of a piece of text.
#   Two texts with similar meanings will have vectors that are close
#   together in mathematical space.
#
#   Example:
#     "chest pain"       -> [0.023, -0.145, 0.891, ...]  (1536 numbers)
#     "heart discomfort" -> [0.019, -0.138, 0.887, ...]  (very similar!)
#     "pizza recipe"     -> [-0.421, 0.334, -0.102, ...]  (very different)
#
#   FAISS uses these vectors to find the most similar chunks to a query.
#
# HOW WE CALL THE OPENAI API:
#   OpenAI's API accepts up to 2048 texts per call (with text-embedding-3-small).
#   We send texts in batches to avoid hitting rate limits and to handle
#   large datasets efficiently.
#
# INPUT:
#   List of text strings (chunk_text values from chunking.py output)
#
# OUTPUT:
#   numpy array of shape (num_texts, embedding_dimension)
#   e.g., (5000, 1536) for 5000 chunks with text-embedding-3-small
#
# USED BY:
#   src/ingestion/build_faiss_index.py
#   src/rag/retriever.py  (to embed the user's query at search time)
# ==============================================================================

import time
import numpy as np
from openai import OpenAI
from loguru import logger
from typing import Optional

from config.settings import settings


# Create a single OpenAI client instance for this module.
# This is reused across all embedding calls to avoid creating
# a new HTTP client on every function call.
openai_client = OpenAI(api_key=settings.openai_api_key)

# Maximum number of texts to send to the OpenAI API in one batch.
# The API supports up to 2048, but we use a smaller batch size
# to be safe with rate limits and memory usage.
BATCH_SIZE = 500


def generate_embeddings(
    texts: list[str],
    model: Optional[str] = None,
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    """
    Generates vector embeddings for a list of text strings.

    Purpose:
        Calls the OpenAI Embeddings API in batches and returns all
        embeddings as a single numpy array. This array is then stored
        in the FAISS vector index by build_faiss_index.py.

    Parameters:
        texts (list[str]):
            List of text strings to embed. Each string should be one
            chunk_text value from the chunking step. Empty strings
            are automatically filtered out.
        model (str, optional):
            The OpenAI embedding model to use.
            Defaults to settings.openai_embedding_model
            (text-embedding-3-small, which produces 1536-dim vectors).
        batch_size (int):
            Number of texts per API call. Default: 500.
            Reduce this if you hit rate limit errors.

    Returns:
        np.ndarray: A 2D numpy array of shape (num_texts, embedding_dim).
            Each row is the embedding vector for the corresponding input text.
            dtype is float32 (required by FAISS).

    Raises:
        ValueError: If the texts list is empty after filtering.
        openai.APIError: If the OpenAI API call fails after retries.

    Example:
        texts = ["chest pain", "shortness of breath", "fever and chills"]
        embeddings = generate_embeddings(texts)
        print(embeddings.shape)  # (3, 1536)
        print(type(embeddings))  # <class 'numpy.ndarray'>
    """
    if model is None:
        model = settings.openai_embedding_model

    # Filter out any empty strings — the API will reject them
    clean_texts = [text.strip() for text in texts if text.strip()]

    if not clean_texts:
        raise ValueError(
            "generate_embeddings received an empty list of texts after filtering. "
            "Make sure your chunks have non-empty text content."
        )

    total_texts = len(clean_texts)
    logger.info(
        f"Generating embeddings for {total_texts} texts "
        f"using model: {model} "
        f"(batch size: {batch_size})"
    )

    all_embeddings = []
    total_batches = (total_texts + batch_size - 1) // batch_size

    # Process texts in batches
    for batch_number, batch_start in enumerate(range(0, total_texts, batch_size)):
        batch_end = min(batch_start + batch_size, total_texts)
        batch_texts = clean_texts[batch_start:batch_end]

        logger.info(
            f"Embedding batch {batch_number + 1}/{total_batches} "
            f"({len(batch_texts)} texts)..."
        )

        # Call the OpenAI API with retry logic
        batch_embeddings = _embed_batch_with_retry(
            texts=batch_texts,
            model=model,
        )

        all_embeddings.extend(batch_embeddings)

        # Small pause between batches to respect OpenAI rate limits
        # (60 requests per minute on the free tier)
        if batch_number < total_batches - 1:
            time.sleep(0.5)

    # Stack all embedding vectors into a single 2D numpy array
    # FAISS requires float32 — float64 (numpy default) will cause errors
    embeddings_array = np.array(all_embeddings, dtype=np.float32)

    logger.info(
        f"Embeddings complete. "
        f"Shape: {embeddings_array.shape} "
        f"(expected: ({total_texts}, {settings.openai_embedding_dimension}))"
    )

    return embeddings_array


def generate_single_embedding(text: str, model: Optional[str] = None) -> np.ndarray:
    """
    Generates an embedding vector for a single text string.

    Purpose:
        Used at query time by the retriever to embed the user's question
        so it can be compared against stored chunk embeddings in FAISS.

        This is separate from generate_embeddings() because:
        - We only need to embed one text (the user's query)
        - We want a 1D array (not a 2D batch), shape: (1536,)
        - Speed matters at query time (no batching overhead needed)

    Parameters:
        text (str): The text to embed (e.g., the user's query).
        model (str, optional): The embedding model. Defaults to settings value.

    Returns:
        np.ndarray: A 1D float32 array of shape (embedding_dim,).
            e.g., shape (1536,) for text-embedding-3-small.

    Raises:
        ValueError: If the text is empty.
        openai.APIError: If the API call fails.

    Example:
        query_vector = generate_single_embedding("What are symptoms of diabetes?")
        print(query_vector.shape)  # (1536,)
    """
    if not text or not text.strip():
        raise ValueError(
            "generate_single_embedding received an empty text. "
            "The user query must not be empty."
        )

    if model is None:
        model = settings.openai_embedding_model

    # Call the API for a single text
    embeddings = _embed_batch_with_retry(
        texts=[text.strip()],
        model=model,
    )

    # Return as 1D array (the first and only embedding in the result)
    return np.array(embeddings[0], dtype=np.float32)


# ------------------------------------------------------------------------------
# PRIVATE HELPER: API call with retry logic
# ------------------------------------------------------------------------------


def _embed_batch_with_retry(
    texts: list[str],
    model: str,
    max_retries: int = 3,
    retry_delay_seconds: float = 5.0,
) -> list[list[float]]:
    """
    Calls the OpenAI Embeddings API for a batch of texts, with retry logic.

    Purpose:
        Network errors and rate limit responses (HTTP 429) are common when
        calling external APIs. This function retries the request up to
        max_retries times with a delay between attempts. If all retries
        fail, it raises the final error so the caller can handle it.

    Parameters:
        texts (list[str]): Batch of text strings to embed.
        model (str): The embedding model name.
        max_retries (int): Maximum number of retry attempts. Default: 3.
        retry_delay_seconds (float): Seconds to wait between retries. Default: 5.

    Returns:
        list[list[float]]: Raw embedding vectors as Python lists.
            Each inner list contains embedding_dim float values.

    Raises:
        Exception: The last exception encountered after all retries are exhausted.
    """
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            # Call the OpenAI Embeddings API
            response = openai_client.embeddings.create(
                input=texts,
                model=model,
            )

            # Extract the embedding vectors from the API response.
            # The response.data list has one object per input text.
            # Each object has an .embedding attribute (list of floats).
            # We sort by index to ensure the output order matches input order.
            embeddings = [
                item.embedding for item in sorted(response.data, key=lambda x: x.index)
            ]

            return embeddings

        except Exception as error:
            last_error = error
            logger.warning(
                f"Embedding API call failed (attempt {attempt}/{max_retries}): {error}"
            )

            # Wait before retrying (longer wait on later attempts)
            if attempt < max_retries:
                wait_time = retry_delay_seconds * attempt
                logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)

    # All retries exhausted — raise the last error
    logger.error(
        f"Embedding API call failed after {max_retries} attempts. "
        f"Last error: {last_error}"
    )
    raise last_error
