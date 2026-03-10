#!/bin/bash
# ==============================================================================
# start.sh — Railway startup script
# Builds FAISS index if not present, then starts the API
# ==============================================================================

set -e

echo "========================================"
echo "Healthcare RAG — Startup Script"
echo "========================================"

# Check if FAISS index already exists on the volume
INDEX_PATH="vectorstore/faiss_index/index.bin"

if [ -f "$INDEX_PATH" ]; then
    echo "✅ FAISS index found at $INDEX_PATH — skipping build"
else
    echo "⚠️  FAISS index not found — building now..."
    echo "This will take ~15-20 minutes on first startup"
    
    python -m src.ingestion.build_faiss_index \
        --max-cases 5000 \
        --max-images 5000
    
    echo "✅ FAISS index built successfully!"
fi

echo "========================================"
echo "Starting FastAPI server..."
echo "========================================"

exec uvicorn src.api.main:app --host 0.0.0.0 --port $PORT
