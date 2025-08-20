#!/bin/bash

# GPU Test Suite Generator
# Generates multiple test cases with random seeds for GPU validation

# Show usage if requested
if [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]; then
    echo "Usage: $0 [num_tests] [wave_size] [num_threads]"
    echo ""
    echo "Arguments:"
    echo "  num_tests   - Number of test cases to generate (default: 100)"
    echo "  wave_size   - Wave/subgroup size: 4, 8, 16, 32, or 64 (default: 32)"
    echo "  num_threads - Total number of threads (default: 64)"
    echo ""
    echo "Example:"
    echo "  $0 50 16 128  # Generate 50 tests with wave size 16 and 128 threads"
    exit 0
fi

# Parse command line arguments
NUM_TESTS=${1:-100}  # Default to 100 tests
WAVE_SIZE=${2:-32}   # Default wave size to 32
NUM_THREADS=${3:-64} # Default numthreads to 64

# Validate wave size
if [[ ! "$WAVE_SIZE" =~ ^(4|8|16|32|64)$ ]]; then
    echo "Error: Invalid wave size $WAVE_SIZE. Valid values are: 4, 8, 16, 32, 64"
    exit 1
fi

# Calculate numthreads configuration based on total threads
# For simplicity, we'll use (NUM_THREADS, 1, 1) configuration
if [[ ! "$NUM_THREADS" =~ ^[0-9]+$ ]] || [ "$NUM_THREADS" -lt 1 ] || [ "$NUM_THREADS" -gt 1024 ]; then
    echo "Error: Invalid numthreads $NUM_THREADS. Must be between 1 and 1024"
    exit 1
fi

# Create output directory with wave size suffix
OUTPUT_DIR="gpu_test_suite_wave${WAVE_SIZE}_threads${NUM_THREADS}"

# Create directory structure
mkdir -p $OUTPUT_DIR/seeds $OUTPUT_DIR/programs $OUTPUT_DIR/tests

# Configuration
export FUZZ_GENERATE_RANDOM=1
export FUZZ_INCREMENTAL_PIPELINE=1
export FUZZ_NUM_THREADS=$NUM_THREADS
export FUZZ_WAVE_SIZE=$WAVE_SIZE

echo "=== GPU Test Suite Generator ==="
echo "Generating $NUM_TESTS test cases..."
echo "Wave size: $WAVE_SIZE"
echo "Num threads: $NUM_THREADS"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Generate tests
for i in $(seq 1 $NUM_TESTS); do
    # Generate unique seed ID (timestamp + counter)
    SEED_ID=$(date +%s%N)_$i
    echo "[$i/$NUM_TESTS] Generating test with seed ID: $SEED_ID"
    
    # Generate random seed data (1KB of random data)
    dd if=/dev/urandom of=$OUTPUT_DIR/seeds/seed_$SEED_ID.bin bs=1024 count=1 2>/dev/null
    
    # Run fuzzer with this seed
    # Set environment variables for the fuzzer
    export FUZZ_SEED_ID=$SEED_ID
    export FUZZ_OUTPUT_DIR=$OUTPUT_DIR/programs
    export FUZZ_MAX_INCREMENTS=1  # Only increment 0
    
    # Run the fuzzer with wave size parameter
    ./bin/minihlsl-fuzzer $OUTPUT_DIR/seeds/seed_$SEED_ID.bin $WAVE_SIZE 2>&1 | \
        grep -E "(Generated program:|Wave Size:|Saved|mutant|test|Using wave size)" || true
    
    # Move test files to tests directory
    if ls $OUTPUT_DIR/programs/program_${SEED_ID}_*.test 1> /dev/null 2>&1; then
        mv $OUTPUT_DIR/programs/program_${SEED_ID}_*.test $OUTPUT_DIR/tests/ 2>/dev/null || true
        echo "Moved test files for seed $SEED_ID to tests directory"
    fi
done

echo ""
echo "=== Generation Complete ==="
echo "Seeds generated: $(ls -1 $OUTPUT_DIR/seeds/*.bin 2>/dev/null | wc -l)"
echo "Programs generated: $(ls -1 $OUTPUT_DIR/programs/*.hlsl 2>/dev/null | wc -l)"
echo "Test files generated: $(ls -1 $OUTPUT_DIR/tests/*.test 2>/dev/null | wc -l)"
echo ""
echo "Directory structure:"
echo "  $OUTPUT_DIR/seeds/    - Random seed files"
echo "  $OUTPUT_DIR/programs/ - Generated HLSL programs"
echo "  $OUTPUT_DIR/tests/    - Test files for GPU validation"