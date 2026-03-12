# ==============================================================================
# src/api/main.py
# ==============================================================================
# PURPOSE:
#   The FastAPI application factory. Creates the app instance, configures
#   middleware (CORS), registers routes, and runs startup/shutdown hooks.
#
# THIS IS THE ENTRYPOINT FOR THE BACKEND SERVER.
#
# How it is launched:
#   Local development:
#     uvicorn src.api.main:app --reload --port 8000
#
#   Railway production (from Procfile):
#     uvicorn src.api.main:app --host 0.0.0.0 --port $PORT
#
# STARTUP SEQUENCE (what happens when the server boots):
#   1. Database tables are created if they don't exist (idempotent)
#   2. Database connection is verified with SELECT 1
#   3. FAISS index is pre-loaded into memory (so first query is fast)
#   4. A default admin user is created if no admin exists yet
#   5. Loguru logger is configured with the level from settings
#
# SHUTDOWN SEQUENCE (what happens when the server stops):
#   1. Any in-progress requests complete (FastAPI handles this)
#   2. Shutdown is logged
#
# USED BY:
#   Procfile (production deployment)
#   Local uvicorn command (development)
# ==============================================================================

import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from passlib.context import CryptContext

from config.settings import settings
from src.api.routes import router
from src.database.db import check_db_connection, create_tables, get_db
from src.database.models import User, UserRole


# ------------------------------------------------------------------------------
# LOGGER CONFIGURATION
# ------------------------------------------------------------------------------
# Configure Loguru to write to stdout (captured by Railway logs)
# with the log level from settings (DEBUG in dev, INFO in prod).
# We remove the default handler and add our own to control the format.

logger.remove()  # Remove the default stderr handler
logger.add(
    sys.stdout,
    level=settings.log_level.upper(),
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    ),
    colorize=True,
)


# ------------------------------------------------------------------------------
# PASSWORD HASHER (used for default admin creation)
# ------------------------------------------------------------------------------

_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# ==============================================================================
# LIFESPAN — STARTUP AND SHUTDOWN
# ==============================================================================
# FastAPI's lifespan context manager replaces the deprecated @app.on_event
# decorators. Everything before `yield` runs on startup; everything after
# runs on shutdown.
# ==============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs application startup and shutdown logic.

    Startup (before yield):
        - Creates all database tables
        - Verifies database connection
        - Pre-warms the FAISS index into memory
        - Creates a default admin user if none exists

    Shutdown (after yield):
        - Logs graceful shutdown message
    """
    # ==========================================================================
    # STARTUP
    # ==========================================================================

    logger.info("=" * 60)
    logger.info("Healthcare RAG API starting up...")
    logger.info(f"Environment: {settings.app_env}")
    logger.info(f"Log level:   {settings.log_level}")
    logger.info("=" * 60)

    # --------------------------------------------------------------------------
    # STEP 1: Create database tables
    # --------------------------------------------------------------------------
    # create_tables() calls Base.metadata.create_all() which is idempotent:
    # it creates tables that don't exist and ignores ones that do.
    # Safe to run on every startup.

    logger.info("Creating database tables (if not exist)...")
    try:
        create_tables()
        logger.info("Database tables ready.")
    except Exception as error:
        logger.critical(f"Failed to create database tables: {error}")
        logger.critical("Cannot start without a database. Exiting.")
        raise

    # --------------------------------------------------------------------------
    # STEP 2: Verify database connection
    # --------------------------------------------------------------------------

    logger.info("Verifying database connection...")
    db_gen = get_db()
    db = next(db_gen)

    try:
        if check_db_connection(db):
            logger.info("Database connection verified.")
        else:
            logger.error("Database connection check failed. API may be degraded.")
    except Exception as error:
        logger.error(f"Database connection check error: {error}")
    finally:
        try:
            next(db_gen)
        except StopIteration:
            pass

    # --------------------------------------------------------------------------
    # STEP 3: Pre-warm the FAISS index
    # --------------------------------------------------------------------------
    # We eagerly load the FAISS index at startup so the first user query
    # doesn't incur the 1-2 second index loading delay.
    # If the index doesn't exist yet, we log a warning but don't crash —
    # the ingestion pipeline may not have been run yet.

    logger.info("Pre-warming FAISS index...")
    try:
        from src.rag.retriever import _load_index_and_metadata

        _load_index_and_metadata()
        logger.info("FAISS index pre-warmed successfully.")
    except FileNotFoundError:
        logger.warning(
            "FAISS index not found. "
            "Run the ingestion pipeline before making queries: "
            "python src/ingestion/build_faiss_index.py"
        )
    except Exception as error:
        logger.error(f"Failed to pre-warm FAISS index: {error}")

    # --------------------------------------------------------------------------
    # STEP 4: Create default admin user if no admin exists
    # --------------------------------------------------------------------------
    # On first deploy, there are no users in the database.
    # We create a default admin so the system is immediately usable.
    # The password should be changed immediately after first login.

    logger.info("Checking for existing admin user...")
    db_gen2 = get_db()
    db2 = next(db_gen2)

    try:
        admin_exists = db2.query(User).filter(User.role == UserRole.admin).first()

        if admin_exists is None:
            logger.warning("No admin user found. Creating default admin account...")
            default_admin = User(
                email=settings.admin_email,
                hashed_password=_pwd_context.hash(settings.admin_password),
                full_name="System Administrator",
                role=UserRole.admin,
                token_limit=10_000_000,  # Admins get a large limit
            )
            db2.add(default_admin)
            db2.commit()

            logger.warning(
                "Default admin created: admin@healthcare-rag.com / ChangeMe123! "
                "— CHANGE THIS PASSWORD IMMEDIATELY!"
            )
        else:
            logger.info(f"Admin user exists: {admin_exists.email}")

    except Exception as error:
        logger.error(f"Failed to check/create default admin: {error}")
        db2.rollback()
    finally:
        try:
            next(db_gen2)
        except StopIteration:
            pass

    logger.info("Startup complete. API is ready to accept requests.")
    logger.info("=" * 60)

    # FastAPI is now running — yield control to the application
    yield

    # ==========================================================================
    # SHUTDOWN
    # ==========================================================================

    logger.info("=" * 60)
    logger.info("Healthcare RAG API shutting down gracefully...")
    logger.info("=" * 60)


# ==============================================================================
# APPLICATION FACTORY
# ==============================================================================


def create_app() -> FastAPI:
    """
    Creates and configures the FastAPI application instance.

    Purpose:
        Factory function that builds the complete app. Using a factory
        function (instead of a bare module-level app) makes testing easier —
        tests can call create_app() to get a fresh isolated instance.

    Returns:
        FastAPI: The configured application instance.
    """
    app = FastAPI(
        title="Healthcare RAG API",
        description=(
            "A Retrieval-Augmented Generation system for medical Q&A, "
            "backed by the MultiCaRe clinical case dataset (93,816 cases). "
            "Includes guardrails, human review, token budgeting, and caching."
        ),
        version="1.0.0",
        lifespan=lifespan,
        # OpenAPI docs are available at /docs (Swagger UI) and /redoc
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # --------------------------------------------------------------------------
    # CORS MIDDLEWARE
    # --------------------------------------------------------------------------
    # Allows the Streamlit frontend (different port) to call the FastAPI backend.
    # In production, only the Streamlit Cloud URL should be in allowed_origins.

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins_list,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )

    # --------------------------------------------------------------------------
    # REGISTER ROUTES
    # --------------------------------------------------------------------------
    # All routes defined in routes.py are registered under the /api/v1 prefix.
    # Example: POST /query becomes POST /api/v1/query

    app.include_router(router, prefix="/api/v1")

    logger.info(f"FastAPI app created with {len(router.routes)} routes under /api/v1")

    return app


# ==============================================================================
# APPLICATION INSTANCE
# ==============================================================================
# This is the object that uvicorn imports and runs:
#   uvicorn src.api.main:app
#
# It is also exported for testing:
#   from src.api.main import app

app = create_app()


# ==============================================================================
# DEVELOPMENT SERVER
# ==============================================================================
# Running this file directly starts a development server:
#   python src/api/main.py

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=settings.app_port,
        reload=settings.is_development,  # Auto-reload on file changes in dev
        log_level=settings.log_level.lower(),
    )
