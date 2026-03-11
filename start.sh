#!/bin/bash
# ==============================================================================
# start.sh — Railway startup script
# ==============================================================================
# The FAISS index is pre-built locally and committed via Git LFS.
# Railway pulls the LFS files during the build step (see railway.json).
# This script NEVER rebuilds the index — it just starts the API server.
# ==============================================================================

set -e

echo "========================================"
echo "Healthcare RAG — Startup"
echo "========================================"

INDEX_PATH="vectorstore/faiss_index/index.bin"

# ------------------------------------------------------------------------------
# CHECK: FAISS Index must be present (pulled from Git LFS during build).
# We do NOT build it here — that would require 4GB+ RAM and kill the process.
# If the index is missing, fail loudly so the issue is immediately obvious.
# ------------------------------------------------------------------------------

if [ -f "$INDEX_PATH" ]; then
    echo "✅ FAISS index found — skipping build, starting API."
else
    echo "❌ ERROR: FAISS index not found at $INDEX_PATH"
    echo ""
    echo "This means Git LFS did not pull the index file during build."
    echo "Fix: ensure 'git lfs pull' runs in your Railway buildCommand."
    echo "See railway.json — buildCommand should be:"
    echo "  git lfs pull && pip install -r requirements.txt"
    echo ""
    echo "Exiting. Fix the LFS setup and redeploy."
    exit 1
fi

echo "========================================"
echo "Starting FastAPI server on port $PORT..."
echo "========================================"

exec uvicorn src.api.main:app --host 0.0.0.0 --port "$PORT"