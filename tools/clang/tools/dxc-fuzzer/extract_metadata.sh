#!/bin/bash

# Metadata extraction script for GPU test suite
# Processes seed files to generate pattern metadata for all tests

set -e

# Show usage if requested
if [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]] || [[ $# -lt 1 ]]; then
    echo "Usage: $0 <test_suite_dir> [output_dir]"
    echo ""
    echo "Arguments:"
    echo "  test_suite_dir - Directory containing seeds/ and tests/ subdirectories"
    echo "  output_dir     - Directory for metadata output (default: test_suite_dir/metadata)"
    echo ""
    echo "Example:"
    echo "  $0 gpu_test_suite_wave64_threads64"
    echo "  $0 gpu_test_suite_wave32_threads64 /tmp/metadata"
    exit 0
fi

# Parse arguments
TEST_SUITE_DIR="$1"
OUTPUT_DIR="${2:-$TEST_SUITE_DIR/metadata}"

# Validate input directory
if [ ! -d "$TEST_SUITE_DIR" ]; then
    echo "Error: Test suite directory '$TEST_SUITE_DIR' not found"
    exit 1
fi

if [ ! -d "$TEST_SUITE_DIR/seeds" ]; then
    echo "Error: Seeds directory '$TEST_SUITE_DIR/seeds' not found"
    exit 1
fi

if [ ! -d "$TEST_SUITE_DIR/tests" ]; then
    echo "Error: Tests directory '$TEST_SUITE_DIR/tests' not found"
    exit 1
fi

# Extract wave size and thread count from directory name if possible
WAVE_SIZE=32
NUM_THREADS=64

if [[ "$TEST_SUITE_DIR" =~ wave([0-9]+) ]]; then
    WAVE_SIZE="${BASH_REMATCH[1]}"
fi

if [[ "$TEST_SUITE_DIR" =~ threads([0-9]+) ]]; then
    NUM_THREADS="${BASH_REMATCH[1]}"
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Build metadata extractor if not already built
echo "=== Building metadata extractor ==="
if [ ! -f "./bin/metadata-extractor" ]; then
    echo "Building metadata-extractor..."
    ninja metadata-extractor
fi

echo ""
echo "=== Metadata Extraction ==="
echo "Test suite: $TEST_SUITE_DIR"
echo "Seed directory: $TEST_SUITE_DIR/seeds"
echo "Test directory: $TEST_SUITE_DIR/tests"
echo "Output directory: $OUTPUT_DIR"
echo "Wave size: $WAVE_SIZE"
echo "Num threads: $NUM_THREADS"
echo ""

# Count files
SEED_COUNT=$(ls -1 "$TEST_SUITE_DIR/seeds"/*.bin 2>/dev/null | wc -l || echo "0")
TEST_COUNT=$(ls -1 "$TEST_SUITE_DIR/tests"/*.test 2>/dev/null | wc -l || echo "0")

echo "Found $SEED_COUNT seed files"
echo "Found $TEST_COUNT test files"
echo ""

# Run metadata extractor
echo "Extracting metadata..."
./bin/metadata-extractor \
    "$TEST_SUITE_DIR/seeds" \
    "$TEST_SUITE_DIR/tests" \
    "$OUTPUT_DIR" \
    "$WAVE_SIZE" \
    "$NUM_THREADS"

# Count generated metadata files
METADATA_COUNT=$(ls -1 "$OUTPUT_DIR"/*.meta.json 2>/dev/null | wc -l || echo "0")

echo ""
echo "=== Extraction Complete ==="
echo "Generated $METADATA_COUNT metadata files"
echo ""

# Generate pattern summary
echo "=== Pattern Summary ==="
python3 - <<EOF
import json
import glob
from collections import defaultdict

pattern_counts = defaultdict(int)
pattern_info = {}

for meta_file in glob.glob("$OUTPUT_DIR/*.meta.json"):
    try:
        with open(meta_file, 'r') as f:
            data = json.load(f)
            pattern_id = data['pattern']['id']
            pattern_counts[pattern_id] += 1
            
            if pattern_id not in pattern_info:
                pattern_info[pattern_id] = {
                    'category': data['pattern']['category'],
                    'description': data['pattern']['description'],
                    'complexity': data['pattern']['complexity']
                }
    except Exception as e:
        print(f"Error reading {meta_file}: {e}")

print(f"Total unique patterns: {len(pattern_counts)}")
print()
print("Pattern distribution:")
print("-" * 60)
print(f"{'ID':<6} {'Count':<8} {'Category':<20} {'Description':<30}")
print("-" * 60)

for pattern_id in sorted(pattern_counts.keys()):
    info = pattern_info[pattern_id]
    print(f"{pattern_id:<6} {pattern_counts[pattern_id]:<8} "
          f"{info['category']:<20} {info['description'][:30]:<30}")

print()
print("Complexity distribution:")
for complexity in range(1, 6):
    count = sum(1 for p in pattern_info.values() if p['complexity'] == complexity)
    print(f"  Level {complexity}: {count} patterns")
EOF

echo ""
echo "Metadata files are in: $OUTPUT_DIR"
echo ""
echo "Next steps:"
echo "1. Run tests on GPU: ninja check-offload-d3d12"
echo "2. Process results: ./scorecard-tool parse test_result.txt --metadata $OUTPUT_DIR"
echo "3. Generate reports: ./scorecard-tool report --format html,csv"