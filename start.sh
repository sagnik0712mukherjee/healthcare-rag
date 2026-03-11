#!/bin/bash
# ==============================================================================
# start.sh — Railway startup script
# Downloads data from Zenodo if needed, builds FAISS index, starts API
# ==============================================================================

set -e

echo "========================================"
echo "Healthcare RAG — Startup Script"
echo "========================================"

INDEX_PATH="vectorstore/faiss_index/index.bin"
DATA_DIR="data/raw"
CASES_FILE="$DATA_DIR/cases.parquet"
CAPTIONS_FILE="$DATA_DIR/captions_and_labels.csv"

# LFS pointer files are tiny (~130 bytes), real files are much larger
CASES_SIZE=$(stat -c%s "$CASES_FILE" 2>/dev/null || echo "0")

if [ "$CASES_SIZE" -lt 1000 ]; then
    echo "⚠️  Data files are LFS pointers — downloading from Zenodo..."
    mkdir -p "$DATA_DIR"

    echo "Downloading cases.parquet (~50MB)..."
    curl -L "https://zenodo.org/records/14994046/files/cases.parquet?download=1" \
         -o "$CASES_FILE" --progress-bar

    echo "Downloading captions_and_labels.csv (~30MB)..."
    curl -L "https://zenodo.org/records/14994046/files/captions_and_labels.csv?download=1" \
         -o "$CAPTIONS_FILE" --progress-bar

    echo "✅ Data files downloaded!"
else
    echo "✅ Data files already present (${CASES_SIZE} bytes)"
fi

# Build FAISS index if not present on volume
if [ -f "$INDEX_PATH" ]; then
    echo "✅ FAISS index found — skipping build"
else
    echo "⚠️  Building FAISS index (~15-20 mins on first startup)..."
    python -m src.ingestion.build_faiss_index \
        --max-cases 5000 \
        --max-images 5000
    echo "✅ FAISS index built!"
fi

echo "========================================"
echo "Starting FastAPI server..."
echo "========================================"

exec uvicorn src.api.main:app --host 0.0.0.0 --port $PORT