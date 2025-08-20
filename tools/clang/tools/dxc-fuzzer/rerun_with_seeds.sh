#!/bin/bash

# Rerun fuzzer with existing seed files
# This script runs the fuzzer using previously generated seed files

# Show usage if requested
if [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]] || [[ $# -lt 1 ]]; then
    echo "Usage: $0 <seed_directory> [wave_size] [num_threads] [output_dir]"
    echo ""
    echo "Arguments:"
    echo "  seed_directory - Directory containing seed files (*.bin)"
    echo "  wave_size      - Wave/subgroup size: 4, 8, 16, 32, or 64 (default: 32)"
    echo "  num_threads    - Total number of threads (default: 64)"
    echo "  output_dir     - Output directory for results (default: rerun_output_waveX_threadsY)"
    echo ""
    echo "Example:"
    echo "  $0 gpu_test_suite_wave32_threads64/seeds 16 128"
    echo "  # Rerun all seeds with wave size 16 and 128 threads"
    echo ""
    echo "  $0 gpu_test_suite_wave32_threads64/seeds 32 64 custom_output"
    echo "  # Rerun with specific output directory"
    exit 0
fi

# Parse command line arguments
SEED_DIR="$1"
WAVE_SIZE=${2:-32}
NUM_THREADS=${3:-64}
OUTPUT_DIR=${4:-"rerun_output_wave${WAVE_SIZE}_threads${NUM_THREADS}"}

# Validate seed directory
if [[ ! -d "$SEED_DIR" ]]; then
    echo "Error: Seed directory '$SEED_DIR' does not exist"
    exit 1
fi

# Count seed files
SEED_COUNT=$(ls -1 "$SEED_DIR"/*.bin 2>/dev/null | wc -l)
if [[ $SEED_COUNT -eq 0 ]]; then
    echo "Error: No seed files (*.bin) found in '$SEED_DIR'"
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

# Create output directory structure
mkdir -p "$OUTPUT_DIR/programs" "$OUTPUT_DIR/tests" "$OUTPUT_DIR/logs"

# Configuration
export FUZZ_GENERATE_RANDOM=1
export FUZZ_INCREMENTAL_PIPELINE=1
export FUZZ_NUM_THREADS=$NUM_THREADS
export FUZZ_WAVE_SIZE=$WAVE_SIZE
export FUZZ_MAX_INCREMENTS=1  # Only increment 0

echo "=== Rerunning Fuzzer with Existing Seeds ==="
echo "Seed directory: $SEED_DIR"
echo "Seed files found: $SEED_COUNT"
echo "Wave size: $WAVE_SIZE"
echo "Num threads: $NUM_THREADS"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Process each seed file
PROCESSED=0
FAILED=0

for SEED_FILE in "$SEED_DIR"/*.bin; do
    if [[ -f "$SEED_FILE" ]]; then
        PROCESSED=$((PROCESSED + 1))
        SEED_NAME=$(basename "$SEED_FILE" .bin)
        
        # Extract seed ID from filename (assuming format: seed_<ID>.bin)
        SEED_ID="${SEED_NAME#seed_}"
        
        echo "[$PROCESSED/$SEED_COUNT] Processing $SEED_NAME (ID: $SEED_ID)"
        
        # Set seed ID for this run
        export FUZZ_SEED_ID=$SEED_ID
        export FUZZ_OUTPUT_DIR="$OUTPUT_DIR/programs"
        
        # Run the fuzzer with wave size parameter
        if ./bin/minihlsl-fuzzer "$SEED_FILE" $WAVE_SIZE > "$OUTPUT_DIR/logs/${SEED_NAME}.log" 2>&1; then
            # Check if test files were generated
            if ls "$OUTPUT_DIR/programs/program_${SEED_ID}_"*.test 1> /dev/null 2>&1; then
                mv "$OUTPUT_DIR/programs/program_${SEED_ID}_"*.test "$OUTPUT_DIR/tests/" 2>/dev/null || true
                echo "  ✓ Generated test files for $SEED_NAME"
            else
                echo "  ✓ Processed $SEED_NAME (no test files generated)"
            fi
            
            # Extract key information from log
            grep -E "(Using wave size:|Total Threads:|Generated program:|Saved)" "$OUTPUT_DIR/logs/${SEED_NAME}.log" | head -5 || true
        else
            FAILED=$((FAILED + 1))
            echo "  ✗ Failed to process $SEED_NAME (check log: $OUTPUT_DIR/logs/${SEED_NAME}.log)"
        fi
        
        echo ""
    fi
done

echo "=== Summary ==="
echo "Seeds processed: $PROCESSED"
echo "Failed: $FAILED"
echo "Programs generated: $(ls -1 "$OUTPUT_DIR/programs"/*.hlsl 2>/dev/null | wc -l)"
echo "Test files generated: $(ls -1 "$OUTPUT_DIR/tests"/*.test 2>/dev/null | wc -l)"
echo ""
echo "Output locations:"
echo "  Programs: $OUTPUT_DIR/programs/"
echo "  Tests: $OUTPUT_DIR/tests/"
echo "  Logs: $OUTPUT_DIR/logs/"

# Optional: Show differences if rerunning with different parameters
if [[ -n "$ORIGINAL_DIR" ]] && [[ -d "$ORIGINAL_DIR" ]]; then
    echo ""
    echo "=== Comparison with Original ==="
    echo "Original programs: $(ls -1 "$ORIGINAL_DIR/programs"/*.hlsl 2>/dev/null | wc -l)"
    echo "New programs: $(ls -1 "$OUTPUT_DIR/programs"/*.hlsl 2>/dev/null | wc -l)"
fi