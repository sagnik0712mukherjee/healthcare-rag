# ==============================================================================
# src/database/db.py
# ==============================================================================
# PURPOSE:
#   This file is responsible for one thing only:
#   creating and managing the connection to the PostgreSQL database.
#
# IT PROVIDES THREE THINGS to the rest of the application:
#   1. engine       - the raw SQLAlchemy connection to PostgreSQL
#   2. SessionLocal - a factory that creates new database sessions
#   3. get_db()     - a FastAPI dependency that hands a session to a route
#                     and closes it automatically when the request is done
#   4. create_tables() - creates all tables in the DB on first run
#
# HOW OTHER FILES USE THIS:
#   from src.database.db import get_db, engine
# ==============================================================================

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session, DeclarativeBase
from typing import Generator
from loguru import logger

from config.settings import settings


# ------------------------------------------------------------------------------
# STEP 1: Create the SQLAlchemy Engine
# ------------------------------------------------------------------------------
# The engine is the core connection to PostgreSQL.
# It reads the DATABASE_URL from settings (which reads from .env).
#
# pool_pre_ping=True means: before using a connection from the pool,
# send a small "ping" to check if it is still alive.
# This prevents errors when the DB drops idle connections (common on Railway).
#
# pool_size=5: keep up to 5 connections open at the same time.
# max_overflow=10: allow up to 10 additional connections in bursts.
# ------------------------------------------------------------------------------

engine = create_engine(
    url=settings.database_url,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
    echo=settings.is_development,
    # echo=True prints every SQL statement to the console.
    # We only do this in development to help with debugging.
    # In production this is False to avoid log spam.
)


# ------------------------------------------------------------------------------
# STEP 2: Create the Session Factory
# ------------------------------------------------------------------------------
# SessionLocal is a class (a factory) that creates new Session objects.
# Each Session represents one database "conversation" - a series of
# reads/writes that happen together.
#
# autocommit=False: we manually control when to commit (save) changes.
# autoflush=False:  we manually control when to flush (send SQL to DB).
# bind=engine:      all sessions created from this factory use our engine.
# ------------------------------------------------------------------------------

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)


# ------------------------------------------------------------------------------
# STEP 3: Create the Declarative Base
# ------------------------------------------------------------------------------
# All ORM model classes (in models.py) will inherit from this Base class.
# SQLAlchemy uses Base to keep track of all table definitions so it can
# create them all at once when we call Base.metadata.create_all(engine).
# ------------------------------------------------------------------------------


class Base(DeclarativeBase):
    """
    Base class for all SQLAlchemy ORM models.

    Purpose:
        Every table definition in models.py inherits from this class.
        SQLAlchemy uses this to register and track all table schemas.

    Usage in models.py:
        from src.database.db import Base

        class User(Base):
            __tablename__ = "users"
            ...
    """

    pass


# ------------------------------------------------------------------------------
# STEP 4: FastAPI Database Dependency - get_db()
# ------------------------------------------------------------------------------


def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency that provides a database session to route handlers.

    Purpose:
        This function is used as a dependency in FastAPI route functions.
        It opens a new database session at the start of each HTTP request
        and guarantees the session is closed when the request finishes,
        even if an exception occurs (thanks to the try/finally block).

    How FastAPI uses this:
        FastAPI sees the "yield" and understands this is a generator.
        It calls the function, runs code up to "yield", injects the session
        into the route handler, waits for the route to finish, then runs
        the code after "yield" (the finally block) to close the session.

    Usage in routes.py:
        from fastapi import Depends
        from src.database.db import get_db
        from sqlalchemy.orm import Session

        @router.post("/query")
        def handle_query(db: Session = Depends(get_db)):
            # db is a live database session here
            result = db.query(User).filter(User.id == 1).first()
            return result

    Yields:
        Session: An active SQLAlchemy database session.
    """
    # Create a new session from the factory
    db = SessionLocal()

    try:
        # Hand the session to the route handler
        yield db

    except Exception as error:
        # If anything goes wrong inside the route handler,
        # roll back any uncommitted database changes to keep
        # the database in a consistent state.
        logger.error(f"Database session error, rolling back: {error}")
        db.rollback()
        raise

    finally:
        # Always close the session when the request is done.
        # This returns the connection back to the pool for reuse.
        db.close()


# ------------------------------------------------------------------------------
# STEP 5: create_tables() - One-time setup function
# ------------------------------------------------------------------------------


def create_tables() -> None:
    """
    Creates all database tables defined in models.py.

    Purpose:
        This function is called once at application startup (in api/main.py)
        and also manually when setting up the project for the first time.
        It reads all classes that inherit from Base and creates their
        corresponding tables in PostgreSQL if they do not already exist.

        If a table already exists, SQLAlchemy skips it silently.
        This makes it safe to call on every startup.

    How to run manually (first-time setup):
        python -c "from src.database.db import create_tables; create_tables()"

    Returns:
        None
    """
    # Import models here so that SQLAlchemy's Base knows about all tables
    # before trying to create them. Without this import, Base.metadata
    # would be empty and no tables would be created.
    from src.database import models  # noqa: F401

    logger.info("Creating database tables if they do not exist...")

    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully.")

    except Exception as error:
        logger.error(f"Failed to create database tables: {error}")
        raise


# ------------------------------------------------------------------------------
# STEP 6: check_db_connection() - Health check utility
# ------------------------------------------------------------------------------


def check_db_connection() -> bool:
    """
    Checks whether the application can connect to the PostgreSQL database.

    Purpose:
        Used by the /health API endpoint to report database connectivity.
        Runs a trivial SQL query (SELECT 1) to verify the connection is alive.

    Returns:
        bool: True if the database is reachable, False otherwise.

    Usage:
        from src.database.db import check_db_connection
        is_healthy = check_db_connection()
    """
    try:
        # Open a raw connection and run a simple query
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return True

    except Exception as error:
        logger.error(f"Database health check failed: {error}")
        return False
