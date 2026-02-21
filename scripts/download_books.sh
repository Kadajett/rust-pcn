#!/usr/bin/env bash
# Download public domain books from Project Gutenberg for PCN training.
#
# Phase 1 books go to data/books/ (used from the start of training).
# Phase 2 books go to data/books-phase2/ (copied in mid-training for continual learning).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

BOOKS_DIR="$PROJECT_DIR/data/books"
PHASE2_DIR="$PROJECT_DIR/data/books-phase2"

mkdir -p "$BOOKS_DIR" "$PHASE2_DIR"

download_book() {
    local id="$1"
    local name="$2"
    local dest="$3"
    local url="https://www.gutenberg.org/cache/epub/${id}/pg${id}.txt"
    local output="$dest/${name}.txt"

    if [ -f "$output" ]; then
        echo "  Already exists: $output"
        return 0
    fi

    echo "  Downloading: $name (PG#$id)..."
    if curl -fsSL "$url" -o "$output"; then
        echo "  Saved: $output ($(wc -c < "$output") bytes)"
    else
        echo "  ERROR: Failed to download $name from $url" >&2
        return 1
    fi
}

echo "=== Phase 1 Books (data/books/) ==="
download_book 11339 "aesops-fables" "$BOOKS_DIR"
download_book 11 "alice-in-wonderland" "$BOOKS_DIR"

echo ""
echo "=== Phase 2 Books (data/books-phase2/) ==="
download_book 1952 "yellow-wallpaper" "$PHASE2_DIR"
download_book 1080 "modest-proposal" "$PHASE2_DIR"
download_book 2591 "grimms-fairy-tales" "$PHASE2_DIR"

echo ""
echo "Done. Phase 1 books are in $BOOKS_DIR"
echo "To start continual learning, copy phase 2 books:"
echo "  cp $PHASE2_DIR/*.txt $BOOKS_DIR/"
