# ==============================================================================
# src/ingestion/load_images.py
# ==============================================================================
# PURPOSE:
#   Loads image metadata and captions from the MultiCaRe dataset's
#   captions_and_labels.csv file and returns them as a clean list of dicts.
#
#   We do NOT load actual image pixel data here.
#   We only load the TEXT associated with each image:
#     - the image caption (e.g., "Chest X-ray showing bilateral infiltrates")
#     - the image labels (e.g., "radiology", "x_ray", "thorax")
#     - the image file name (so the UI can display it later)
#
#   The caption text is what gets embedded into FAISS, allowing users to
#   find relevant images by asking text questions like:
#   "Show me cases with chest X-ray findings"
#
# INPUT:
#   data/raw/captions_and_labels.csv
#
# OUTPUT:
#   A list of dicts, one per image. Example:
#   [
#     {
#       "image_id":   "PMC7992397_fig1_A_1_1",
#       "caption":    "Transthoracic echocardiogram showing a thrombus...",
#       "image_type": "radiology",
#       "image_subtype": "ultrasound",
#       "labels":     ["radiology", "ultrasound", "echocardiogram"],
#       "file_name":  "PMC7992397_fmed09-985235-g001_A_1_1.webp",
#       "source":     "image_caption"
#     },
#     ...
#   ]
#
# USED BY:
#   src/ingestion/build_faiss_index.py
# ==============================================================================

import os
import pandas as pd
from loguru import logger
from typing import Optional

from config.settings import settings


def load_image_captions(
    file_path: Optional[str] = None,
    max_images: Optional[int] = None,
) -> list[dict]:
    """
    Loads image captions and metadata from captions_and_labels.csv.

    Purpose:
        Reads the CSV file and returns each image's caption and labels
        as a dictionary. These captions are later embedded into FAISS
        alongside clinical case text chunks, enabling multimodal retrieval.

    What is captions_and_labels.csv?
        This file from the MultiCaRe dataset has one row per image.
        Key columns include:
          - file_name: the .webp image filename
          - caption: the text description of the image
          - image_type: top-level category (radiology, pathology, etc.)
          - image_subtype: more specific type (x_ray, mri, ct, etc.)
          - gt_labels_for_semisupervised_classification: high-quality labels
          - ml_labels_for_supervised_classification: ML-generated labels

    Parameters:
        file_path (str, optional):
            Path to captions_and_labels.csv.
            Defaults to settings.captions_csv_path.
        max_images (int, optional):
            If provided, only load this many image records.
            Useful for testing without loading all 130,791 images.

    Returns:
        list[dict]: A list of image caption dictionaries. Each dict has:
            - image_id (str): Unique identifier derived from the filename
            - caption (str): The text description of the image
            - image_type (str): Top-level image category (e.g., "radiology")
            - image_subtype (str): Specific image type (e.g., "mri", "x_ray")
            - labels (list[str]): All labels assigned to this image
            - file_name (str): The .webp filename for displaying the image
            - source (str): Always "image_caption" for filtering in retrieval

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If no rows remain after cleaning.

    Example:
        images = load_image_captions(max_images=2000)
        print(images[0]["caption"])
        # "Chest X-ray showing bilateral pulmonary infiltrates"
    """
    # Use the path from settings if none was provided
    if file_path is None:
        file_path = settings.captions_csv_path

    # Verify the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"captions_and_labels.csv not found at: {file_path}\n"
            f"Download the MultiCaRe dataset from: "
            f"https://zenodo.org/records/14994046\n"
            f"Then place captions_and_labels.csv in: data/raw/"
        )

    logger.info(f"Loading image captions from: {file_path}")

    # Read the CSV file
    # low_memory=False prevents pandas from guessing column types incorrectly
    # on large files with mixed data
    dataframe = pd.read_csv(file_path, low_memory=False)

    logger.info(
        f"Raw image data loaded: {len(dataframe)} rows, "
        f"columns: {list(dataframe.columns)}"
    )

    # --------------------------------------------------------------------------
    # DATA CLEANING
    # --------------------------------------------------------------------------

    # Drop rows where caption is missing — we cannot embed an image without text
    dataframe = dataframe.dropna(subset=["caption"])
    dataframe = dataframe[dataframe["caption"].str.strip() != ""]

    logger.info(f"After removing empty captions: {len(dataframe)} rows remain")

    # Optionally limit the number of images
    if max_images is not None:
        dataframe = dataframe.head(max_images)
        logger.info(f"Limited to first {max_images} image records for this run")

    # --------------------------------------------------------------------------
    # CONVERT TO LIST OF DICTS
    # --------------------------------------------------------------------------

    images = []

    for _, row in dataframe.iterrows():
        # Extract the file name (used as a unique ID and for display)
        file_name = _safe_string(row.get("file_name", ""), default=f"image_{_}")

        # Derive a clean image_id from the file name by stripping the extension
        image_id = file_name.replace(".webp", "").replace(".jpg", "")

        # Extract image classification labels
        # We try the high-quality caption-based labels first,
        # fall back to ML-generated labels if those are missing
        labels = _extract_labels(row)

        # Build the image record dictionary
        image_record = {
            # Unique identifier for this image
            "image_id": image_id,
            # The caption text — this is what gets embedded into FAISS
            "caption": _safe_string(row.get("caption", ""), default=""),
            # Top-level image type classification
            "image_type": _safe_string(row.get("image_type", ""), default="unknown"),
            # More specific image subtype (e.g., "mri", "x_ray", "ct")
            "image_subtype": _safe_string(
                row.get("image_subtype", ""), default="unknown"
            ),
            # All labels for this image as a Python list
            "labels": labels,
            # The actual filename — used by the frontend to display the image
            "file_name": file_name,
            # Tag to distinguish image records from clinical case records
            # when filtering FAISS results in the retriever
            "source": "image_caption",
        }

        # Only add if the caption has content
        if image_record["caption"]:
            images.append(image_record)

    logger.info(f"Successfully loaded {len(images)} image caption records")

    return images


# ------------------------------------------------------------------------------
# PRIVATE HELPER FUNCTIONS
# ------------------------------------------------------------------------------


def _extract_labels(row: pd.Series) -> list[str]:
    """
    Extracts a clean list of labels from a DataFrame row.

    Purpose:
        The MultiCaRe dataset stores labels in different columns depending
        on how they were generated. This function tries each column in
        order of quality preference and returns the best available labels.

    Label column preference order:
        1. gt_labels_for_semisupervised_classification (highest quality, caption-based)
        2. ml_labels_for_supervised_classification (ML-generated, more complete)
        3. image_type + image_subtype (always available, basic fallback)

    Parameters:
        row (pd.Series): A single row from the captions DataFrame.

    Returns:
        list[str]: A list of label strings for this image.
                   Empty list if no labels are found.

    Example:
        labels = _extract_labels(row)
        # ["radiology", "mri", "head", "axial"]
    """
    labels = []

    # Try high-quality caption-based labels first
    gt_labels = row.get("gt_labels_for_semisupervised_classification", None)
    if _is_valid_label_value(gt_labels):
        labels = _parse_label_string(str(gt_labels))
        if labels:
            return labels

    # Fall back to ML-generated labels
    ml_labels = row.get("ml_labels_for_supervised_classification", None)
    if _is_valid_label_value(ml_labels):
        labels = _parse_label_string(str(ml_labels))
        if labels:
            return labels

    # Final fallback: use image_type and image_subtype columns
    fallback_labels = []
    for col in ["image_type", "image_subtype", "radiology_region"]:
        val = row.get(col, None)
        if _is_valid_label_value(val):
            clean_val = str(val).strip()
            if clean_val and clean_val != "nan":
                fallback_labels.append(clean_val)

    return fallback_labels


def _parse_label_string(label_string: str) -> list[str]:
    """
    Parses a label string into a list of individual label strings.

    Purpose:
        In the dataset CSV, multiple labels are stored as a
        pipe-separated or comma-separated string like:
        "radiology|mri|head" or "radiology,mri,head"
        This function splits them into a clean Python list.

    Parameters:
        label_string (str): Raw label string from the CSV.

    Returns:
        list[str]: Individual label strings, cleaned and deduplicated.

    Example:
        _parse_label_string("radiology|mri|head|axial")
        # ["radiology", "mri", "head", "axial"]
    """
    if not label_string or label_string.strip() in ("nan", "None", ""):
        return []

    # Try pipe-separated first, then comma-separated
    if "|" in label_string:
        raw_labels = label_string.split("|")
    elif "," in label_string:
        raw_labels = label_string.split(",")
    else:
        # Single label
        raw_labels = [label_string]

    # Clean each label and remove empty strings
    clean_labels = []
    for label in raw_labels:
        label = label.strip()
        if label and label not in ("nan", "None", ""):
            clean_labels.append(label)

    # Remove duplicates while preserving order
    seen = set()
    unique_labels = []
    for label in clean_labels:
        if label not in seen:
            seen.add(label)
            unique_labels.append(label)

    return unique_labels


def _is_valid_label_value(value) -> bool:
    """
    Returns True if a label value is not null or empty.

    Parameters:
        value: Any value from a DataFrame cell.

    Returns:
        bool: True if the value is usable, False if null/empty/NaN.
    """
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    return str(value).strip() not in ("", "nan", "None")


def _safe_string(value, default: str = "") -> str:
    """
    Converts a value to a clean string safely.

    Parameters:
        value: Any value to convert.
        default (str): Fallback if value is null or empty.

    Returns:
        str: A clean string, or the default.
    """
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    result = str(value).strip()
    return result if result else default
