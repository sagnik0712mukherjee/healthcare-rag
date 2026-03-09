# ==============================================================================
# src/ingestion/build_faiss_index.py
# ==============================================================================
# PURPOSE:
#   This is the master ingestion script that orchestrates the entire
#   data pipeline from raw MultiCaRe dataset files to a ready-to-query
#   FAISS vector index.
#
# RUN THIS SCRIPT ONCE before starting the application.
# It reads raw data, chunks it, embeds it, and saves the FAISS index to disk.
#
# HOW TO RUN:
#   python src/ingestion/build_faiss_index.py
#
#   For a quick test with limited data:
#   python src/ingestion/build_faiss_index.py --max-cases 500 --max-images 1000
#
# WHAT IT DOES (step by step):
#   1. Load clinical cases from cases.parquet
#   2. Load image captions from captions_and_labels.csv
#   3. Chunk the clinical cases into smaller text pieces
#   4. Prepare image captions as single-chunk records
#   5. Generate OpenAI embeddings for ALL chunks (cases + captions)
#   6. Build a FAISS index from the embeddings
#   7. Save the FAISS index binary to disk
#   8. Save the chunk metadata as a JSON file alongside the index
#
# OUTPUT FILES (saved to vectorstore/faiss_index/):
#   - index.bin      : The FAISS binary index (fast vector search)
#   - metadata.json  : List of chunk dicts (one per vector in the index)
#
# The metadata.json maps each FAISS vector position (0, 1, 2, ...)
# to the original chunk text and its source metadata. The retriever
# uses this to return human-readable results after a vector search.
# ==============================================================================

import os
import json
import argparse
import numpy as np
import faiss
from loguru import logger
from tqdm import tqdm

from config.settings import settings
from src.ingestion.load_cases import load_clinical_cases
from src.ingestion.load_images import load_image_captions
from src.ingestion.chunking import chunk_clinical_cases, chunk_image_captions
from src.ingestion.embeddings import generate_embeddings


def build_faiss_index(
    max_cases: int = None,
    max_images: int = None,
    chunk_size: int = 400,
    overlap: int = 50,
) -> None:
    """
    Runs the full ingestion pipeline and saves the FAISS index to disk.

    Purpose:
        This is the single entry point for ingesting the MultiCaRe dataset.
        It coordinates all ingestion sub-modules in the correct order and
        saves the results so the RAG retriever can load them at query time.

    Parameters:
        max_cases (int, optional):
            Maximum number of clinical cases to load.
            None = load all 93,816 cases (recommended for production).
            Use a small number like 500 for development/testing.
        max_images (int, optional):
            Maximum number of image caption records to load.
            None = load all 130,791 images.
            Use a small number like 1000 for development/testing.
        chunk_size (int):
            Maximum tokens per clinical case chunk. Default: 400.
        overlap (int):
            Token overlap between consecutive chunks. Default: 50.

    Returns:
        None

    Side effects:
        Creates the following files:
          - {settings.faiss_index_path}/index.bin
          - {settings.faiss_index_path}/metadata.json
    """
    logger.info("=" * 60)
    logger.info("HEALTHCARE RAG - FAISS INDEX BUILD PIPELINE")
    logger.info("=" * 60)

    # --------------------------------------------------------------------------
    # STEP 1: Load Clinical Cases
    # --------------------------------------------------------------------------
    logger.info("STEP 1/6: Loading clinical cases...")

    cases = load_clinical_cases(max_cases=max_cases)

    logger.info(f"Loaded {len(cases)} clinical cases.")

    # --------------------------------------------------------------------------
    # STEP 2: Load Image Captions
    # --------------------------------------------------------------------------
    logger.info("STEP 2/6: Loading image captions...")

    image_records = load_image_captions(max_images=max_images)

    logger.info(f"Loaded {len(image_records)} image caption records.")

    # --------------------------------------------------------------------------
    # STEP 3: Chunk Clinical Cases
    # --------------------------------------------------------------------------
    logger.info("STEP 3/6: Chunking clinical cases...")

    case_chunks = chunk_clinical_cases(
        cases=cases,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    logger.info(f"Created {len(case_chunks)} clinical case chunks.")

    # --------------------------------------------------------------------------
    # STEP 4: Prepare Image Captions as Chunks
    # --------------------------------------------------------------------------
    logger.info("STEP 4/6: Preparing image captions as chunks...")

    caption_chunks = chunk_image_captions(image_records=image_records)

    logger.info(f"Prepared {len(caption_chunks)} image caption chunks.")

    # --------------------------------------------------------------------------
    # Combine all chunks into one list
    # --------------------------------------------------------------------------
    # Both clinical case chunks and image caption chunks go into the same
    # FAISS index. The "source" field in each chunk's metadata tells the
    # retriever whether a result came from a case or an image.

    all_chunks = case_chunks + caption_chunks

    logger.info(
        f"Total chunks to embed and index: {len(all_chunks)} "
        f"({len(case_chunks)} case chunks + {len(caption_chunks)} caption chunks)"
    )

    if not all_chunks:
        raise ValueError(
            "No chunks were generated. Check that your data files exist "
            "at the paths specified in your .env file."
        )

    # --------------------------------------------------------------------------
    # STEP 5: Generate Embeddings for All Chunks
    # --------------------------------------------------------------------------
    logger.info("STEP 5/6: Generating embeddings (this may take a while)...")

    # Extract just the text strings for the embedding API call
    all_texts = [chunk["chunk_text"] for chunk in all_chunks]

    # This calls the OpenAI API in batches and returns a numpy array
    # Shape: (num_chunks, embedding_dimension) e.g., (50000, 1536)
    embeddings = generate_embeddings(texts=all_texts)

    logger.info(f"Generated embeddings array of shape: {embeddings.shape}")

    # Sanity check: number of embeddings must match number of chunks
    assert len(embeddings) == len(all_chunks), (
        f"Mismatch: {len(embeddings)} embeddings for {len(all_chunks)} chunks. "
        f"Something went wrong in the embedding step."
    )

    # --------------------------------------------------------------------------
    # STEP 6: Build the FAISS Index
    # --------------------------------------------------------------------------
    logger.info("STEP 6/6: Building FAISS index and saving to disk...")

    _build_and_save_index(
        embeddings=embeddings,
        chunks=all_chunks,
    )

    logger.info("=" * 60)
    logger.info("INGESTION PIPELINE COMPLETE!")
    logger.info(f"FAISS index saved to: {settings.faiss_index_path}/")
    logger.info(f"Total vectors indexed: {len(all_chunks)}")
    logger.info("You can now start the FastAPI backend.")
    logger.info("=" * 60)


# ------------------------------------------------------------------------------
# PRIVATE HELPER: Build and save the FAISS index
# ------------------------------------------------------------------------------


def _build_and_save_index(
    embeddings: np.ndarray,
    chunks: list[dict],
) -> None:
    """
    Builds a FAISS index from embedding vectors and saves it to disk.

    Purpose:
        Takes the numpy array of embeddings produced in Step 5 and
        creates a FAISS index that can be used for fast nearest-neighbor
        search at query time.

    FAISS index type used: IndexFlatIP
        - "Flat" means it compares the query against EVERY stored vector
          (no approximation). This is 100% accurate but slower for
          very large datasets (>1M vectors).
        - "IP" means Inner Product similarity. When vectors are normalized
          to unit length, inner product equals cosine similarity.
        - For this project's size (up to ~200K vectors), Flat is fast enough.
          For larger scale, switch to IndexIVFFlat or IndexHNSW.

    Why normalize vectors?
        OpenAI embeddings are NOT unit-normalized by default.
        FAISS IndexFlatIP gives cosine similarity only when vectors are
        normalized to length 1.0. We do this with faiss.normalize_L2().

    Parameters:
        embeddings (np.ndarray): Shape (num_chunks, embedding_dim), float32.
        chunks (list[dict]): The chunk metadata to save alongside the index.

    Returns:
        None

    Side effects:
        Creates the output directory and saves index.bin and metadata.json.
    """
    # Create the output directory if it does not already exist
    os.makedirs(settings.faiss_index_path, exist_ok=True)

    embedding_dim = embeddings.shape[1]

    logger.info(
        f"Building FAISS IndexFlatIP with dimension={embedding_dim} "
        f"and {len(embeddings)} vectors..."
    )

    # Normalize all vectors to unit length so that inner product = cosine similarity
    # This modifies the array in-place (faiss.normalize_L2 operates in-place)
    faiss.normalize_L2(embeddings)

    # Create the FAISS index
    index = faiss.IndexFlatIP(embedding_dim)

    # Add all embedding vectors to the index
    # After this, index.ntotal == len(embeddings)
    index.add(embeddings)

    logger.info(f"FAISS index built. Total vectors: {index.ntotal}")

    # --------------------------------------------------------------------------
    # Save the FAISS index binary file
    # --------------------------------------------------------------------------
    # faiss.write_index saves the index in FAISS's native binary format.
    # faiss.read_index loads it back at query time (in retriever.py).

    faiss.write_index(index, settings.faiss_index_file)
    logger.info(f"FAISS index saved to: {settings.faiss_index_file}")

    # --------------------------------------------------------------------------
    # Save the metadata JSON file
    # --------------------------------------------------------------------------
    # FAISS stores vectors by integer position (0, 1, 2, ...).
    # When the retriever finds the top-k closest vectors, it returns their
    # positions (indices). We need to look up those positions to get the
    # original chunk text and metadata.
    #
    # The metadata.json file is a list where:
    #   metadata[0] = chunk dict for the vector at FAISS position 0
    #   metadata[1] = chunk dict for the vector at FAISS position 1
    #   ...etc.
    #
    # This allows us to do:
    #   faiss_positions = [12, 437, 8821]
    #   results = [metadata[pos] for pos in faiss_positions]

    # Clean up chunks for JSON serialization
    # Some values (like numpy integers) need to be converted to Python types
    serializable_chunks = []
    for chunk in tqdm(chunks, desc="Preparing metadata for JSON"):
        clean_chunk = _make_json_serializable(chunk)
        serializable_chunks.append(clean_chunk)

    with open(settings.faiss_metadata_file, "w", encoding="utf-8") as f:
        json.dump(serializable_chunks, f, ensure_ascii=False, indent=2)

    logger.info(f"Metadata saved to: {settings.faiss_metadata_file}")

    # --------------------------------------------------------------------------
    # Save a summary file for reference
    # --------------------------------------------------------------------------
    _save_index_summary(
        total_vectors=index.ntotal,
        embedding_dim=embedding_dim,
        chunks=chunks,
    )


def _make_json_serializable(chunk: dict) -> dict:
    """
    Converts a chunk dictionary into a JSON-serializable format.

    Purpose:
        Some values in chunk dicts may be numpy integers, numpy floats,
        or other non-serializable types. This function converts them all
        to standard Python types so json.dump() works without errors.

    Parameters:
        chunk (dict): A chunk dictionary possibly containing non-serializable values.

    Returns:
        dict: A new dictionary with all values converted to JSON-safe types.
    """
    clean = {}
    for key, value in chunk.items():
        if isinstance(value, (np.integer,)):
            clean[key] = int(value)
        elif isinstance(value, (np.floating,)):
            clean[key] = float(value)
        elif isinstance(value, np.ndarray):
            clean[key] = value.tolist()
        elif value is None:
            clean[key] = None
        else:
            clean[key] = value
    return clean


def _save_index_summary(
    total_vectors: int,
    embedding_dim: int,
    chunks: list[dict],
) -> None:
    """
    Saves a human-readable summary of the built index.

    Purpose:
        Creates a summary.json file in the FAISS index directory that
        records key statistics about the index for debugging and auditing.

    Parameters:
        total_vectors (int): Number of vectors in the index.
        embedding_dim (int): Dimension of each vector.
        chunks (list[dict]): All chunk records (used to count by source type).

    Returns:
        None
    """
    case_count = sum(1 for c in chunks if c.get("source") == "clinical_case")
    caption_count = sum(1 for c in chunks if c.get("source") == "image_caption")

    summary = {
        "total_vectors": total_vectors,
        "embedding_dimension": embedding_dim,
        "embedding_model": settings.openai_embedding_model,
        "clinical_case_chunks": case_count,
        "image_caption_chunks": caption_count,
        "index_type": "IndexFlatIP (cosine similarity via L2 normalization)",
        "index_file": settings.faiss_index_file,
        "metadata_file": settings.faiss_metadata_file,
    }

    summary_path = os.path.join(settings.faiss_index_path, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Index summary saved to: {summary_path}")


# ------------------------------------------------------------------------------
# COMMAND LINE INTERFACE
# Run this file directly: python src/ingestion/build_faiss_index.py
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build the FAISS vector index for the Healthcare RAG system."
    )

    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help=(
            "Maximum number of clinical cases to load. "
            "Use a small number like 500 for a quick test. "
            "Leave empty to load all cases."
        ),
    )

    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help=(
            "Maximum number of image caption records to load. "
            "Leave empty to load all images."
        ),
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=400,
        help="Maximum number of tokens per clinical case chunk. Default: 400.",
    )

    parser.add_argument(
        "--overlap",
        type=int,
        default=50,
        help="Token overlap between consecutive chunks. Default: 50.",
    )

    args = parser.parse_args()

    build_faiss_index(
        max_cases=args.max_cases,
        max_images=args.max_images,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )
