# ==============================================================================
# config/settings.py
# ==============================================================================
# PURPOSE:
#   This is the single source of truth for all application configuration.
#   Every other module imports settings from here instead of reading
#   environment variables directly.
#
# HOW IT WORKS:
#   Pydantic's BaseSettings class automatically reads values from:
#     1. The system environment variables
#     2. A .env file in the project root (loaded via python-dotenv)
#
# HOW TO USE IN OTHER MODULES:
#   from config.settings import settings
#   print(settings.openai_api_key)
# ==============================================================================

import os
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Central configuration class for the Healthcare RAG system.

    All attributes in this class map directly to environment variables
    defined in the .env file. Pydantic automatically:
      - Reads the value from the environment
      - Casts it to the correct Python type (str, int, float, bool)
      - Raises a clear error at startup if a required variable is missing

    The field names here are lowercase versions of the .env variable names.
    For example, OPENAI_API_KEY in .env becomes openai_api_key here.
    """

    # --------------------------------------------------------------------------
    # Pydantic configuration
    # This tells Pydantic where to find the .env file and to ignore
    # any extra fields that are not defined in this class.
    # --------------------------------------------------------------------------
    model_config = SettingsConfigDict(
        env_file=".env",  # Read from .env file in the project root
        env_file_encoding="utf-8",  # Use UTF-8 encoding when reading .env
        case_sensitive=False,  # OPENAI_API_KEY and openai_api_key both work
        extra="ignore",  # Silently ignore unknown env variables
    )

    # --------------------------------------------------------------------------
    # OPENAI CONFIGURATION
    # --------------------------------------------------------------------------

    openai_api_key: str
    # The secret API key from https://platform.openai.com/api-keys
    # Example value: sk-proj-abc123...
    # Required: Yes - the system cannot function without this

    openai_chat_model: str = "gpt-4o-mini"
    # Which OpenAI model to use for generating responses
    # Default: gpt-4o-mini (low cost, good quality for healthcare Q&A)
    # Other options: gpt-4o, gpt-3.5-turbo

    openai_embedding_model: str = "text-embedding-3-small"
    # Which OpenAI model to use for generating vector embeddings
    # Default: text-embedding-3-small (fast, cheap, 1536 dimensions)

    openai_embedding_dimension: int = 1536
    # The number of dimensions in each embedding vector.
    # This must match the model above:
    #   text-embedding-3-small  -> 1536
    #   text-embedding-ada-002  -> 1536
    #   text-embedding-3-large  -> 3072

    # --------------------------------------------------------------------------
    # DATABASE CONFIGURATION
    # --------------------------------------------------------------------------

    database_url: str
    # Full PostgreSQL connection string
    # Local example:  postgresql://postgres:password@localhost:5432/healthcare_rag
    # Railway example: postgresql://postgres:abc123@containers-us-west-1.railway.app:5432/railway
    # Required: Yes - the system needs a database to run

    # --------------------------------------------------------------------------
    # APPLICATION SETTINGS
    # --------------------------------------------------------------------------

    app_env: str = "development"
    # The environment the app is running in.
    # Values: "development", "staging", "production"
    # Used to enable/disable debug features and detailed error messages.

    secret_key: str = "healthcare-rag-internal-eval-key-2026"
    # Secret key used for signing JWT authentication tokens.
    # In production, this must be a long random string.
    # Generate one with: python -c "import secrets; print(secrets.token_hex(32))"

    app_port: int = 8000
    # The port FastAPI runs on locally.
    # In Railway deployment, $PORT overrides this automatically.

    # --------------------------------------------------------------------------
    # TOKEN USAGE LIMITS
    # --------------------------------------------------------------------------

    default_token_limit: int = 100_000
    # The maximum number of tokens a new user can consume in total.
    # This is applied when a new user registers.
    # Admins can override per-user limits in the database directly.

    cost_per_1k_input_tokens: float = 0.00015
    # Cost in USD per 1000 input (prompt) tokens for the chat model.
    # Used to calculate and display dollar-equivalent usage to admins.
    # Update this value if OpenAI changes their pricing.

    cost_per_1k_output_tokens: float = 0.00060
    # Cost in USD per 1000 output (completion) tokens for the chat model.

    # --------------------------------------------------------------------------
    # FAISS VECTOR STORE CONFIGURATION
    # --------------------------------------------------------------------------

    faiss_index_path: str = "vectorstore/faiss_index"
    # Directory path where the FAISS index file and metadata JSON are saved.
    # This directory is created by the ingestion pipeline (build_faiss_index.py).
    # The retriever loads the index from this path at startup.

    retrieval_top_k: int = 5
    # Number of most-similar chunks to retrieve for each user query.
    # Higher values give more context to the LLM but increase token usage.
    # Recommended range: 3 to 10

    # --------------------------------------------------------------------------
    # STORAGE CONFIGURATION
    # --------------------------------------------------------------------------

    use_s3: bool = False
    # Set to True to store and serve images from AWS S3.
    # Set to False to use local disk storage (recommended for development).

    local_image_path: str = "data/images"
    # Local directory path where .webp images from MultiCaRe are stored.
    # Only used when use_s3 is False.

    aws_access_key_id: str = ""
    # AWS access key ID for S3 access.
    # Only required when use_s3 is True.

    aws_secret_access_key: str = ""
    # AWS secret access key for S3 access.
    # Only required when use_s3 is True.

    aws_region: str = "us-east-1"
    # AWS region where the S3 bucket is located.

    s3_bucket_name: str = ""
    # Name of the S3 bucket where images are stored.
    # Only required when use_s3 is True.

    # --------------------------------------------------------------------------
    # DATA PATHS - MultiCaRe Dataset Files
    # --------------------------------------------------------------------------

    cases_parquet_path: str = "data/raw/cases.parquet"
    # Path to the cases.parquet file from the MultiCaRe dataset.
    # This file contains clinical case text, patient age, and gender.
    # Download from: https://zenodo.org/records/14994046

    captions_csv_path: str = "data/raw/captions_and_labels.csv"
    # Path to the captions_and_labels.csv file from MultiCaRe.
    # This file contains image captions, file names, and classification labels.

    metadata_parquet_path: str = "data/raw/metadata.parquet"
    # Path to the metadata.parquet file from MultiCaRe.
    # Contains article metadata: authors, year, journal, DOI, PMCID, etc.

    processed_data_path: str = "data/processed"
    # Directory where the ingestion pipeline saves processed chunk JSON files.
    # Created automatically by build_faiss_index.py if it does not exist.

    # --------------------------------------------------------------------------
    # MEMORY CONFIGURATION
    # --------------------------------------------------------------------------

    short_term_memory_max_turns: int = 10
    # Maximum number of conversation turns kept in short-term memory per session.
    # A single "turn" is: one user message + one AI response.
    # Older turns are dropped when this limit is reached (sliding window).
    # Higher values give more context but use more tokens per request.

    # --------------------------------------------------------------------------
    # HUMAN-IN-THE-LOOP REVIEW CONFIGURATION
    # --------------------------------------------------------------------------

    human_review_risk_threshold: float = 0.01
    # A risk score between 0.0 and 1.0.
    # Responses with a computed risk score above this value are flagged
    # for review by a healthcare professional before being returned to the user.
    # Lower threshold = more responses get flagged (more conservative).
    # Higher threshold = fewer responses get flagged (more permissive).

    human_review_enabled: bool = True
    # Master switch for the human review workflow.
    # Set to False to disable flagging entirely (e.g., in development).

    # --------------------------------------------------------------------------
    # CORS CONFIGURATION
    # --------------------------------------------------------------------------

    allowed_origins: str = "http://localhost:8501,http://localhost:3000"
    # Comma-separated list of allowed origins for CORS (Cross-Origin Resource Sharing).
    # This controls which frontend URLs are allowed to call the FastAPI backend.
    # In production: set this to your Streamlit Cloud URL.
    # Example: "https://your-app.streamlit.app"

    # --------------------------------------------------------------------------
    # LOGGING CONFIGURATION
    # --------------------------------------------------------------------------

    log_level: str = "INFO"
    # Log verbosity level.
    # Options (from most to least verbose): DEBUG, INFO, WARNING, ERROR, CRITICAL
    # Use DEBUG during development to see all details.
    # Use INFO or WARNING in production.

    admin_email: str = "admin@healthcare-rag.com"
    # Email address of the admin user.

    admin_password: str = "ChangeMe123!"
    # Password of the admin user.

    # --------------------------------------------------------------------------
    # COMPUTED PROPERTIES
    # These are not read from .env - they are calculated from other settings.
    # --------------------------------------------------------------------------

    @property
    def is_production(self) -> bool:
        """
        Returns True if the application is running in production mode.

        Purpose:
            Used to enable/disable features that should only run in production,
            such as strict error handling and detailed audit logging.

        Returns:
            bool: True if app_env is "production", False otherwise.
        """
        return self.app_env == "production"

    @property
    def is_development(self) -> bool:
        """
        Returns True if the application is running in development mode.

        Purpose:
            Used to enable developer-friendly features like detailed
            error tracebacks in API responses.

        Returns:
            bool: True if app_env is "development", False otherwise.
        """
        return self.app_env == "development"

    @property
    def faiss_index_file(self) -> str:
        """
        Returns the full file path to the FAISS index binary file.

        Purpose:
            The FAISS index is saved as a .bin file inside the faiss_index_path
            directory. This property constructs the full path.

        Returns:
            str: Full path like "vectorstore/faiss_index/index.bin"
        """
        return os.path.join(self.faiss_index_path, "index.bin")

    @property
    def faiss_metadata_file(self) -> str:
        """
        Returns the full file path to the FAISS metadata JSON file.

        Purpose:
            Alongside the binary index, we store a JSON file that maps
            each FAISS vector index position to its original text chunk
            and source metadata (case ID, patient info, image labels, etc.).

        Returns:
            str: Full path like "vectorstore/faiss_index/metadata.json"
        """
        return os.path.join(self.faiss_index_path, "metadata.json")

    @property
    def allowed_origins_list(self) -> list[str]:
        """
        Returns the allowed CORS origins as a Python list.

        Purpose:
            The .env file stores origins as a comma-separated string for
            simplicity. FastAPI's CORS middleware needs a list of strings.
            This property converts the string to a list.

        Example:
            Input:  "http://localhost:8501,https://my-app.streamlit.app"
            Output: ["http://localhost:8501", "https://my-app.streamlit.app"]

        Returns:
            list[str]: List of allowed origin URLs.
        """
        return [origin.strip() for origin in self.allowed_origins.split(",")]

    def get_total_cost_usd(self, input_tokens: int, output_tokens: int) -> float:
        """
        Calculates the total cost in USD for a single LLM API call.

        Purpose:
            Used by the token monitoring module to calculate how much money
            a single request cost, so it can be logged and compared against
            the user's dollar-equivalent budget.

        Parameters:
            input_tokens (int): Number of input (prompt) tokens used.
            output_tokens (int): Number of output (completion) tokens used.

        Returns:
            float: Total cost in USD, rounded to 8 decimal places.

        Example:
            settings.get_total_cost_usd(500, 200)
            # input cost:  500 / 1000 * 0.00015 = 0.000075
            # output cost: 200 / 1000 * 0.00060 = 0.00012
            # total:       0.000195 USD
        """
        input_cost = (input_tokens / 1000) * self.cost_per_1k_input_tokens
        output_cost = (output_tokens / 1000) * self.cost_per_1k_output_tokens
        total_cost = input_cost + output_cost
        return round(total_cost, 8)


# ------------------------------------------------------------------------------
# SINGLETON INSTANCE
# ------------------------------------------------------------------------------
# @lru_cache means this function is only called ONCE for the entire application
# lifetime. After the first call, the same Settings object is returned from
# cache. This prevents reading the .env file on every import.
# ------------------------------------------------------------------------------


@lru_cache()
def get_settings() -> Settings:
    """
    Returns the application settings as a cached singleton.

    Purpose:
        Creates a single Settings instance that is shared across the entire
        application. Using lru_cache ensures the .env file is only read once,
        making this safe and efficient to call from anywhere.

    Returns:
        Settings: The application settings object.

    Usage:
        from config.settings import get_settings
        settings = get_settings()
        print(settings.openai_api_key)
    """
    return Settings()


# ------------------------------------------------------------------------------
# CONVENIENCE ALIAS
# ------------------------------------------------------------------------------
# Most modules will import settings directly like:
#   from config.settings import settings
#
# This is equivalent to calling get_settings() every time, but reads
# more naturally in code.
# ------------------------------------------------------------------------------

settings = get_settings()
