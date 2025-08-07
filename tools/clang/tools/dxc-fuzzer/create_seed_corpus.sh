#!/bin/bash

# Script to create a seed corpus from HLSL examples for fuzzing

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLES_DIR="$SCRIPT_DIR/examples"
CORPUS_DIR="$SCRIPT_DIR/corpus"

echo "Creating seed corpus for MiniHLSL fuzzer..."
echo "Source: $EXAMPLES_DIR"
echo "Target: $CORPUS_DIR"

# Create corpus directory if it doesn't exist
mkdir -p "$CORPUS_DIR"

# Copy all HLSL files from examples to corpus
if [ -d "$EXAMPLES_DIR" ]; then
    echo "Copying HLSL files..."
    find "$EXAMPLES_DIR" -name "*.hlsl" -type f | while read -r file; do
        # Get the base filename
        basename=$(basename "$file")
        # Copy to corpus with a prefix for organization
        cp "$file" "$CORPUS_DIR/seed_$basename"
        echo "  Added: seed_$basename"
    done
else
    echo "Error: Examples directory not found at $EXAMPLES_DIR"
    exit 1
fi

# Count the files
NUM_FILES=$(find "$CORPUS_DIR" -name "*.hlsl" | wc -l)
echo ""
echo "Seed corpus created with $NUM_FILES HLSL files"

# Create a manifest file listing all seeds
echo "Creating manifest..."
find "$CORPUS_DIR" -name "*.hlsl" -type f | sort > "$CORPUS_DIR/manifest.txt"

echo "Done! Corpus is ready at: $CORPUS_DIR"
echo ""
echo "To use with fuzzer:"
echo "  ./fuzz-hlsl-file $CORPUS_DIR/seed_*.hlsl"