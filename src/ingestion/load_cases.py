# ==============================================================================
# src/ingestion/load_cases.py
# ==============================================================================
# PURPOSE:
#   Loads clinical case text data from the MultiCaRe dataset's cases.parquet
#   file and returns it as a clean list of Python dictionaries.
#
# INPUT:
#   data/raw/cases.parquet
#   (Downloaded from https://zenodo.org/records/14994046)
#
# OUTPUT:
#   A list of dicts, one per clinical case. Example:
#   [
#     {
#       "case_id":    "PMC7992397",
#       "case_text":  "A 50-year-old male presented with...",
#       "patient_age": 50,
#       "patient_gender": "Male",
#       "source": "clinical_case"
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


def load_clinical_cases(
    file_path: Optional[str] = None,
    max_cases: Optional[int] = None,
) -> list[dict]:
    """
    Loads clinical case records from the MultiCaRe cases.parquet file.

    Purpose:
        Reads the parquet file, cleans the data, and returns each clinical
        case as a dictionary that downstream modules (chunking, embedding)
        can easily work with.

    What is cases.parquet?
        This file comes from the MultiCaRe dataset. Each row represents
        one clinical case extracted from a PubMed Central case report.
        Columns include:
          - case_id or pmcid: the PubMed Central article identifier
          - case_text: the full text of the clinical case
          - age: patient age in years (may be NaN if not mentioned)
          - gender: "Male", "Female", "Transgender", or "Unknown"

    Parameters:
        file_path (str, optional):
            Path to the cases.parquet file.
            Defaults to settings.cases_parquet_path from config.
        max_cases (int, optional):
            If provided, only load this many cases. Useful for testing
            the pipeline without loading all 93,816 cases.
            Example: max_cases=1000 loads the first 1000 rows only.

    Returns:
        list[dict]: A list of clinical case dictionaries. Each dict has:
            - case_id (str): Unique identifier for this case
            - case_text (str): The full clinical case text
            - patient_age (int or None): Patient age, or None if unknown
            - patient_gender (str): "Male", "Female", "Transgender", or "Unknown"
            - source (str): Always "clinical_case" for text search filtering

    Raises:
        FileNotFoundError: If the parquet file does not exist at the given path.
        ValueError: If the file exists but has no valid rows after cleaning.

    Example:
        cases = load_clinical_cases(max_cases=500)
        print(len(cases))             # 500
        print(cases[0]["case_text"])  # "A 45-year-old female presented..."
    """
    # Use the path from settings if none was provided
    if file_path is None:
        file_path = settings.cases_parquet_path

    # Check the file exists before trying to read it
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"cases.parquet not found at: {file_path}\n"
            f"Download the MultiCaRe dataset from: "
            f"https://zenodo.org/records/14994046\n"
            f"Then place cases.parquet in: data/raw/"
        )

    logger.info(f"Loading clinical cases from: {file_path}")

    # Read the parquet file into a pandas DataFrame
    # Parquet is a binary columnar format — much faster to read than CSV
    dataframe = pd.read_parquet(file_path)

    logger.info(
        f"Raw data loaded: {len(dataframe)} rows, columns: {list(dataframe.columns)}"
    )

    # --------------------------------------------------------------------------
    # DATA CLEANING
    # --------------------------------------------------------------------------

    # Drop any rows where case_text is missing or empty
    # We cannot embed or retrieve a case with no text
    dataframe = dataframe.dropna(subset=_find_text_column(dataframe))
    dataframe = dataframe[dataframe[_find_text_column(dataframe)].str.strip() != ""]

    logger.info(f"After removing empty case texts: {len(dataframe)} rows remain")

    # Optionally limit the number of cases (useful for development/testing)
    if max_cases is not None:
        dataframe = dataframe.head(max_cases)
        logger.info(f"Limited to first {max_cases} cases for this run")

    # --------------------------------------------------------------------------
    # CONVERT TO LIST OF DICTS
    # --------------------------------------------------------------------------
    # We convert each DataFrame row into a simple Python dictionary.
    # This makes the data easy to pass between modules without pandas dependencies.

    cases = []

    text_column = _find_text_column(dataframe)
    id_column = _find_id_column(dataframe)

    for _, row in dataframe.iterrows():
        # Extract patient age safely (may be NaN in the dataset)
        patient_age = _safe_int(row.get("age", None))

        # Extract patient gender safely (may be NaN or missing)
        patient_gender = _safe_string(row.get("gender", "Unknown"), default="Unknown")

        # Build the case dictionary
        case = {
            # Unique identifier: use the PMCID if available
            "case_id": _safe_string(row.get(id_column, ""), default=f"case_{_}"),
            # The full clinical case text - this is what gets chunked and embedded
            "case_text": _safe_string(row.get(text_column, ""), default=""),
            # Patient demographics - stored in metadata alongside embeddings
            "patient_age": patient_age,
            "patient_gender": patient_gender,
            # Tag this record so the retriever knows it came from a clinical case
            # (as opposed to an image caption, which also goes into FAISS)
            "source": "clinical_case",
        }

        # Only add the case if it has actual text content
        if case["case_text"]:
            cases.append(case)

    logger.info(f"Successfully loaded {len(cases)} clinical cases")

    return cases


# ------------------------------------------------------------------------------
# PRIVATE HELPER FUNCTIONS
# These are used only within this file, hence the underscore prefix.
# They handle the fact that the MultiCaRe dataset column names
# may vary slightly between dataset versions.
# ------------------------------------------------------------------------------


def _find_text_column(dataframe: pd.DataFrame) -> str:
    """
    Finds the name of the column containing clinical case text.

    Purpose:
        The MultiCaRe dataset uses "case_text" as the column name,
        but we check for alternatives in case future versions differ.

    Parameters:
        dataframe (pd.DataFrame): The loaded cases DataFrame.

    Returns:
        str: The name of the text column.

    Raises:
        ValueError: If no recognizable text column is found.
    """
    # Try these column names in order of preference
    candidates = ["case_text", "text", "content", "clinical_case", "cases"]

    for candidate in candidates:
        if candidate in dataframe.columns:
            return candidate

    raise ValueError(
        f"Could not find a text column in cases.parquet. "
        f"Available columns: {list(dataframe.columns)}\n"
        f"Expected one of: {candidates}"
    )


def _find_id_column(dataframe: pd.DataFrame) -> str:
    """
    Finds the name of the column containing the case/article identifier.

    Parameters:
        dataframe (pd.DataFrame): The loaded cases DataFrame.

    Returns:
        str: The name of the ID column, or a fallback.
    """
    candidates = ["pmcid", "case_id", "id", "article_id", "pmc_id"]

    for candidate in candidates:
        if candidate in dataframe.columns:
            return candidate

    # If no ID column found, we will generate IDs by row index
    return "index"


def _safe_int(value) -> Optional[int]:
    """
    Converts a value to int safely, returning None if conversion fails.

    Purpose:
        Patient ages in the dataset may be NaN (float), a string like "45",
        or already an integer. This function handles all cases cleanly.

    Parameters:
        value: Any value that might represent an integer.

    Returns:
        int or None: The integer value, or None if conversion is not possible.
    """
    if value is None:
        return None

    try:
        # pd.isna() checks for NaN, None, pd.NaT etc.
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _safe_string(value, default: str = "") -> str:
    """
    Converts a value to a clean string safely.

    Purpose:
        Values in the dataset may be NaN, None, or already a string.
        This ensures we always get a clean string back.

    Parameters:
        value: Any value to convert.
        default (str): What to return if the value is null or empty.

    Returns:
        str: A clean string, or the default if value is null/empty.
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
