# ==============================================================================
# src/ingestion/chunking.py
# ==============================================================================
# PURPOSE:
#   Splits long clinical case texts into smaller overlapping chunks.
#
# WHY DO WE CHUNK?
#   OpenAI's embedding model has a token limit (~8191 tokens per call).
#   Some clinical cases in MultiCaRe are much longer than this limit.
#   Also, retrieving a smaller, focused chunk is more useful than
#   retrieving a 5-page clinical report — the LLM gets more precise context.
#
# HOW IT WORKS (Sliding Window):
#   Given a text of 1000 tokens, with chunk_size=300 and overlap=50:
#
#   Chunk 1: tokens   0 -> 300
#   Chunk 2: tokens 250 -> 550   (starts 50 tokens back = overlap)
#   Chunk 3: tokens 500 -> 800
#   Chunk 4: tokens 750 -> 1000
#
#   The overlap ensures that sentences split across chunk boundaries
#   are still captured in at least one chunk. This prevents context loss
#   at the edges of each chunk.
#
# IMAGE CAPTIONS:
#   Image captions from load_images.py are usually very short (1-3 sentences).
#   We do NOT chunk captions — we treat each caption as a single chunk.
#   This function only chunks clinical case text.
#
# INPUT:
#   List of case dicts from load_cases.py
#
# OUTPUT:
#   List of chunk dicts. Each chunk dict is one unit that will be embedded.
#   Example:
#   {
#     "chunk_id":      "PMC7992397_chunk_0",
#     "chunk_text":    "A 50-year-old male presented with dyspnea...",
#     "case_id":       "PMC7992397",
#     "patient_age":   50,
#     "patient_gender":"Male",
#     "chunk_index":   0,
#     "total_chunks":  3,
#     "source":        "clinical_case"
#   }
#
# USED BY:
#   src/ingestion/build_faiss_index.py
# ==============================================================================

import tiktoken
from loguru import logger


# We use tiktoken to count tokens accurately.
# "cl100k_base" is the tokenizer used by all current OpenAI embedding models
# and GPT-4 family models. Using the correct tokenizer ensures our chunk
# sizes are exactly what we expect — not just an approximation.
TOKENIZER = tiktoken.get_encoding("cl100k_base")


def chunk_clinical_cases(
    cases: list[dict],
    chunk_size: int = 400,
    overlap: int = 50,
) -> list[dict]:
    """
    Splits a list of clinical cases into smaller overlapping text chunks.

    Purpose:
        Takes the output of load_cases.py and produces a new list where
        each long case text is broken into chunks that fit within the
        embedding model's token limit. Short cases that fit in one chunk
        are returned as a single chunk without splitting.

    Parameters:
        cases (list[dict]):
            List of clinical case dictionaries from load_cases.py.
            Each dict must have at least "case_text" and "case_id" keys.
        chunk_size (int):
            Maximum number of tokens per chunk.
            Default: 400 tokens. This leaves room for the image captions
            and query in the final LLM prompt (which has a ~4000 token context).
        overlap (int):
            Number of tokens that each chunk shares with the previous chunk.
            Default: 50 tokens. This prevents losing context at chunk boundaries.

    Returns:
        list[dict]: A flat list of chunk dictionaries. Each dict has:
            - chunk_id (str): Unique ID combining case_id and chunk index
            - chunk_text (str): The actual text of this chunk
            - case_id (str): The parent case this chunk came from
            - patient_age (int or None): Inherited from parent case
            - patient_gender (str): Inherited from parent case
            - chunk_index (int): Position of this chunk within its case (0-based)
            - total_chunks (int): Total number of chunks this case was split into
            - source (str): Always "clinical_case"

    Example:
        cases = load_clinical_cases(max_cases=10)
        chunks = chunk_clinical_cases(cases, chunk_size=400, overlap=50)
        print(len(chunks))  # Will be >= 10 (some cases split into multiple chunks)
        print(chunks[0]["chunk_id"])  # "PMC7992397_chunk_0"
    """
    if not cases:
        logger.warning("chunk_clinical_cases received an empty list of cases.")
        return []

    logger.info(
        f"Chunking {len(cases)} clinical cases "
        f"(chunk_size={chunk_size} tokens, overlap={overlap} tokens)..."
    )

    all_chunks = []

    for case in cases:
        case_text = case.get("case_text", "")
        case_id = case.get("case_id", "unknown")

        # Skip cases with no text (should have been filtered in load_cases.py
        # but we check again to be safe)
        if not case_text.strip():
            logger.warning(f"Skipping case {case_id}: empty text after loading.")
            continue

        # Split this single case text into chunks
        case_chunks = _split_text_into_chunks(
            text=case_text,
            chunk_size=chunk_size,
            overlap=overlap,
        )

        total_chunks = len(case_chunks)

        # Wrap each raw text chunk in a full dictionary with metadata
        for chunk_index, chunk_text in enumerate(case_chunks):
            chunk_dict = {
                # Unique ID: combine case ID with position so we can trace
                # any chunk back to its original clinical case
                "chunk_id": f"{case_id}_chunk_{chunk_index}",
                # The actual text content of this chunk
                "chunk_text": chunk_text,
                # Metadata inherited from the parent case
                # Stored alongside the embedding in FAISS metadata
                # so the retriever can return context about the source
                "case_id": case_id,
                "patient_age": case.get("patient_age", None),
                "patient_gender": case.get("patient_gender", "Unknown"),
                # Position tracking — useful for debugging retrieval
                "chunk_index": chunk_index,
                "total_chunks": total_chunks,
                # Source tag — distinguishes clinical case chunks from
                # image caption records in FAISS search results
                "source": "clinical_case",
            }

            all_chunks.append(chunk_dict)

    logger.info(
        f"Chunking complete: {len(cases)} cases -> {len(all_chunks)} chunks "
        f"(average {len(all_chunks) / max(len(cases), 1):.1f} chunks per case)"
    )

    return all_chunks


def chunk_image_captions(image_records: list[dict]) -> list[dict]:
    """
    Prepares image caption records as single-chunk units for embedding.

    Purpose:
        Image captions from load_images.py are typically short (1-3 sentences,
        well under 400 tokens). We do not split them — instead, we just
        reformat them into the same chunk dict structure that clinical case
        chunks use, so the embedding and FAISS indexing steps can treat
        all records uniformly.

    Parameters:
        image_records (list[dict]):
            List of image dictionaries from load_images.py.
            Each dict must have at least "caption" and "image_id" keys.

    Returns:
        list[dict]: A list of chunk-format dictionaries. Each dict has:
            - chunk_id (str): Same as image_id
            - chunk_text (str): The image caption text
            - image_id (str): The original image identifier
            - image_type (str): e.g., "radiology", "pathology"
            - image_subtype (str): e.g., "mri", "x_ray"
            - labels (list[str]): Classification labels for this image
            - file_name (str): The .webp filename for UI display
            - chunk_index (int): Always 0 (captions are not split)
            - total_chunks (int): Always 1
            - source (str): Always "image_caption"

    Example:
        images = load_image_captions(max_images=100)
        caption_chunks = chunk_image_captions(images)
        print(caption_chunks[0]["chunk_text"])
        # "Chest X-ray AP view showing bilateral infiltrates"
    """
    if not image_records:
        logger.warning("chunk_image_captions received an empty list.")
        return []

    logger.info(f"Preparing {len(image_records)} image captions as single chunks...")

    caption_chunks = []

    for image_record in image_records:
        caption = image_record.get("caption", "")
        image_id = image_record.get("image_id", "unknown")

        if not caption.strip():
            continue

        chunk_dict = {
            # Use image_id directly as the chunk_id (no splitting needed)
            "chunk_id": image_id,
            # The caption text is the chunk text for embedding
            "chunk_text": caption,
            # Image-specific metadata (not present in case chunks)
            "image_id": image_id,
            "image_type": image_record.get("image_type", "unknown"),
            "image_subtype": image_record.get("image_subtype", "unknown"),
            "labels": image_record.get("labels", []),
            "file_name": image_record.get("file_name", ""),
            # Position tracking (always 0/1 since captions are not split)
            "chunk_index": 0,
            "total_chunks": 1,
            # Source tag to distinguish from clinical case chunks
            "source": "image_caption",
        }

        caption_chunks.append(chunk_dict)

    logger.info(f"Prepared {len(caption_chunks)} image caption chunks")

    return caption_chunks


# ------------------------------------------------------------------------------
# PRIVATE HELPER: sliding window text splitter
# ------------------------------------------------------------------------------


def _split_text_into_chunks(
    text: str,
    chunk_size: int,
    overlap: int,
) -> list[str]:
    """
    Splits a single text string into overlapping token-based chunks.

    Purpose:
        This is the core splitting algorithm. It tokenizes the input text,
        then uses a sliding window to extract chunks of `chunk_size` tokens,
        each starting `overlap` tokens before the previous chunk ended.

    How it works step by step:
        1. Encode the full text into a list of integer token IDs using tiktoken
        2. Walk through the token list with a sliding window
        3. Decode each window back into a string
        4. Return the list of strings

    Parameters:
        text (str): The full text to split.
        chunk_size (int): Maximum tokens per chunk.
        overlap (int): Number of tokens to repeat at the start of each chunk.

    Returns:
        list[str]: List of text chunk strings.
                   Always returns at least one chunk (even if text is short).

    Example:
        chunks = _split_text_into_chunks("Long medical text...", 400, 50)
        len(chunks)  # 3 (for a ~1000 token text)
    """
    # Step 1: Convert the text string into a list of integer token IDs
    # tiktoken is much more accurate than splitting by words or characters
    token_ids = TOKENIZER.encode(text)

    total_tokens = len(token_ids)

    # If the text fits in one chunk, return it as-is without splitting
    if total_tokens <= chunk_size:
        return [text]

    # Step 2: Sliding window over token IDs
    chunks = []
    start = 0

    while start < total_tokens:
        # The end of this chunk (don't go past the end of the token list)
        end = min(start + chunk_size, total_tokens)

        # Extract the token IDs for this chunk
        chunk_token_ids = token_ids[start:end]

        # Step 3: Decode token IDs back to a readable string
        chunk_text = TOKENIZER.decode(chunk_token_ids)

        chunks.append(chunk_text)

        # If we just processed the last tokens, stop
        if end == total_tokens:
            break

        # Step 4: Move the window forward by (chunk_size - overlap)
        # The overlap means the next chunk starts `overlap` tokens before
        # where the current chunk ended
        start += chunk_size - overlap

    return chunks


def get_token_count(text: str) -> int:
    """
    Returns the number of tokens in a text string.

    Purpose:
        Utility function used by other modules to check how many tokens
        a piece of text will consume before sending it to the OpenAI API.
        Helps avoid exceeding context window limits.

    Parameters:
        text (str): Any text string.

    Returns:
        int: The number of tokens in the text.

    Example:
        count = get_token_count("What are symptoms of diabetes?")
        print(count)  # 6
    """
    return len(TOKENIZER.encode(text))
