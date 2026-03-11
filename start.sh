#!/bin/bash
# ==============================================================================
# start.sh — Railway startup script
# ==============================================================================
# How FAISS index delivery works:
#   The index.bin and metadata.json are stored in GitHub LFS.
#   When Railway clones the repo, LFS files come through as tiny pointer files
#   (~130 bytes) instead of the real content. We detect this and download
#   the real files directly from GitHub's LFS media CDN using curl.
#   No git-lfs binary needed anywhere.
# ==============================================================================

set -e

REPO="sagnik0712mukherjee/healthcare-rag"
BRANCH="main"
INDEX_DIR="vectorstore/faiss_index"
INDEX_BIN="$INDEX_DIR/index.bin"
METADATA_JSON="$INDEX_DIR/metadata.json"

echo "========================================"
echo "Healthcare RAG — Startup"
echo "========================================"

mkdir -p "$INDEX_DIR"

# ------------------------------------------------------------------------------
# Helper: check if a file is still a git-lfs pointer (< 1 KB means pointer)
# Real index.bin is ~114 MB. Real metadata.json is ~25 MB.
# ------------------------------------------------------------------------------
is_lfs_pointer() {
    local size
    size=$(stat -c%s "$1" 2>/dev/null || echo "0")
    [ "$size" -lt 1000 ]
}

# ------------------------------------------------------------------------------
# Download index.bin if missing or still a pointer file
# ------------------------------------------------------------------------------
if [ ! -f "$INDEX_BIN" ] || is_lfs_pointer "$INDEX_BIN"; then
    echo "Downloading index.bin from GitHub LFS CDN (~114 MB)..."
    curl -L --progress-bar \
        "https://media.githubusercontent.com/media/${REPO}/${BRANCH}/${INDEX_BIN}" \
        -o "$INDEX_BIN"
    echo "✅ index.bin downloaded ($(stat -c%s "$INDEX_BIN") bytes)"
else
    echo "✅ index.bin already present ($(stat -c%s "$INDEX_BIN") bytes)"
fi

# ------------------------------------------------------------------------------
# Download metadata.json if missing or still a pointer file
# ------------------------------------------------------------------------------
if [ ! -f "$METADATA_JSON" ] || is_lfs_pointer "$METADATA_JSON"; then
    echo "Downloading metadata.json from GitHub LFS CDN (~25 MB)..."
    curl -L --progress-bar \
        "https://media.githubusercontent.com/media/${REPO}/${BRANCH}/${METADATA_JSON}" \
        -o "$METADATA_JSON"
    echo "✅ metadata.json downloaded ($(stat -c%s "$METADATA_JSON") bytes)"
else
    echo "✅ metadata.json already present ($(stat -c%s "$METADATA_JSON") bytes)"
fi

# ------------------------------------------------------------------------------
# Final check — bail out clearly if downloads failed
# ------------------------------------------------------------------------------
if is_lfs_pointer "$INDEX_BIN"; then
    echo "❌ ERROR: index.bin is still a pointer file after download attempt."
    echo "Check that the GitHub repo is public and LFS files are pushed."
    exit 1
fi

echo "========================================"
echo "Starting FastAPI server on port $PORT..."
echo "========================================"

exec uvicorn src.api.main:app --host 0.0.0.0 --port "$PORT"