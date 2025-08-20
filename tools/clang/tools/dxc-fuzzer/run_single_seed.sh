#!/bin/bash

# Run fuzzer with a single seed file
# Useful for debugging or reproducing specific test cases

# Show usage if requested
if [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]] || [[ $# -lt 1 ]]; then
    echo "Usage: $0 <seed_file> [wave_size] [num_threads] [output_dir]"
    echo ""
    echo "Arguments:"
    echo "  seed_file   - Path to seed file (.bin)"
    echo "  wave_size   - Wave/subgroup size: 4, 8, 16, 32, or 64 (default: 32)"
    echo "  num_threads - Total number of threads (default: 64)"
    echo "  output_dir  - Output directory (default: single_run_output)"
    echo ""
    echo "Options:"
    echo "  --verbose   - Show full fuzzer output (add as last argument)"
    echo ""
    echo "Example:"
    echo "  $0 seeds/seed_123.bin 16 128"
    echo "  $0 seeds/seed_123.bin 32 64 my_output --verbose"
    exit 0
fi

# Parse command line arguments
SEED_FILE="$1"
WAVE_SIZE=${2:-32}
NUM_THREADS=${3:-64}
OUTPUT_DIR=${4:-"single_run_output"}

# Check for verbose flag
VERBOSE=0
if [[ "${@: -1}" == "--verbose" ]]; then
    VERBOSE=1
    # Adjust output dir if verbose was in 4th position
    if [[ "$4" == "--verbose" ]]; then
        OUTPUT_DIR="single_run_output"
    fi
fi

# Validate seed file
if [[ ! -f "$SEED_FILE" ]]; then
    echo "Error: Seed file '$SEED_FILE' does not exist"
    exit 1
fi

# Validate wave size
if [[ ! "$WAVE_SIZE" =~ ^(4|8|16|32|64)$ ]]; then
    echo "Error: Invalid wave size $WAVE_SIZE. Valid values are: 4, 8, 16, 32, 64"
    exit 1
fi

# Validate num threads
if [[ ! "$NUM_THREADS" =~ ^[0-9]+$ ]] || [ "$NUM_THREADS" -lt 1 ] || [ "$NUM_THREADS" -gt 1024 ]; then
    echo "Error: Invalid numthreads $NUM_THREADS. Must be between 1 and 1024"
    exit 1
fi

# Extract seed info
SEED_NAME=$(basename "$SEED_FILE" .bin)
SEED_ID="${SEED_NAME#seed_}"

# Create output directory
mkdir -p "$OUTPUT_DIR/programs" "$OUTPUT_DIR/tests"

# Configuration
export FUZZ_GENERATE_RANDOM=1
export FUZZ_INCREMENTAL_PIPELINE=1
export FUZZ_NUM_THREADS=$NUM_THREADS
export FUZZ_WAVE_SIZE=$WAVE_SIZE
export FUZZ_MAX_INCREMENTS=1
export FUZZ_SEED_ID=$SEED_ID
export FUZZ_OUTPUT_DIR="$OUTPUT_DIR/programs"

echo "=== Running Fuzzer with Single Seed ==="
echo "Seed file: $SEED_FILE"
echo "Seed ID: $SEED_ID"
echo "Wave size: $WAVE_SIZE"
echo "Num threads: $NUM_THREADS"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Run the fuzzer
if [[ $VERBOSE -eq 1 ]]; then
    echo "=== Fuzzer Output ==="
    ./bin/minihlsl-fuzzer "$SEED_FILE" $WAVE_SIZE
    RESULT=$?
else
    echo "Running fuzzer..."
    OUTPUT=$(./bin/minihlsl-fuzzer "$SEED_FILE" $WAVE_SIZE 2>&1)
    RESULT=$?
    
    # Show key information
    echo "$OUTPUT" | grep -E "(Using wave size:|Total Threads:|Generated program:|Saved|=== Pipeline Results ===)" || true
fi

echo ""

if [[ $RESULT -eq 0 ]]; then
    # Move test files
    if ls "$OUTPUT_DIR/programs/program_${SEED_ID}_"*.test 1> /dev/null 2>&1; then
        mv "$OUTPUT_DIR/programs/program_${SEED_ID}_"*.test "$OUTPUT_DIR/tests/" 2>/dev/null || true
    fi
    
    echo "=== Results ==="
    echo "Programs generated: $(ls -1 "$OUTPUT_DIR/programs"/*.hlsl 2>/dev/null | wc -l)"
    echo "Test files generated: $(ls -1 "$OUTPUT_DIR/tests"/*.test 2>/dev/null | wc -l)"
    
    # List generated files
    if [[ $(ls -1 "$OUTPUT_DIR/programs"/*.hlsl 2>/dev/null | wc -l) -gt 0 ]]; then
        echo ""
        echo "Generated programs:"
        ls -la "$OUTPUT_DIR/programs"/*.hlsl 2>/dev/null | awk '{print "  " $9}'
    fi
    
    if [[ $(ls -1 "$OUTPUT_DIR/tests"/*.test 2>/dev/null | wc -l) -gt 0 ]]; then
        echo ""
        echo "Generated test files:"
        ls -la "$OUTPUT_DIR/tests"/*.test 2>/dev/null | awk '{print "  " $9}'
    fi
else
    echo "Error: Fuzzer failed with exit code $RESULT"
    exit $RESULT
fi