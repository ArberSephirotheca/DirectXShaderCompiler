#!/bin/bash

# Script to build and run the MiniHLSL fuzzer

# Get absolute paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DXC_ROOT="$( cd "$SCRIPT_DIR/../../../.." && pwd )"
BUILD_DIR="$DXC_ROOT/build-fuzzer"
TEST_DIR="$SCRIPT_DIR/test_cases"

echo "MiniHLSL Fuzzer Runner"
echo "======================"
echo "Build Dir: $BUILD_DIR"
echo "Test Dir: $TEST_DIR"
echo

# Check if we're in the right directory
if [ ! -f "MiniHLSLFuzzer.cpp" ]; then
    echo "Error: Run this from the dxc-fuzzer directory"
    exit 1
fi

# Create clean build directory with clang++
echo "Creating clean build directory with clang++..."
if [ -d "$BUILD_DIR" ]; then
    echo "Removing existing build directory..."
    rm -rf "$BUILD_DIR"
fi
echo "Creating new build directory at: $BUILD_DIR"
mkdir -p "$BUILD_DIR"

# Configure with clang++
echo "Configuring with CMake using clang++..."
cd "$BUILD_DIR"
cmake "$DXC_ROOT" \
  -C "$DXC_ROOT/cmake/caches/PredefinedParams.cmake" \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_USE_SANITIZER='Address' \
  -DCMAKE_C_FLAGS="-fsanitize=address" \
  -DCMAKE_CXX_FLAGS="-fsanitize=address" \
  -G Ninja

if [ $? -ne 0 ]; then
    echo "CMake configuration failed!"
    exit 1
fi

# Build the MiniHLSL fuzzer
echo "Building minihlsl-fuzzer..."
cmake --build . --target minihlsl-fuzzer

if [ $? -ne 0 ]; then
    echo "Build failed!"
    exit 1
fi

echo "Build successful!"
echo

# Create a corpus directory for the fuzzer
CORPUS_DIR="$SCRIPT_DIR/minihlsl_corpus"
mkdir -p "$CORPUS_DIR"

# Copy valid test cases to corpus
if [ -f "$TEST_DIR/valid_minihlsl.hlsl" ]; then
    cp "$TEST_DIR/valid_minihlsl.hlsl" "$CORPUS_DIR/seed1.hlsl"
    echo "Added valid test case to corpus"
fi

# Create some additional seed inputs
cat > "$CORPUS_DIR/seed2.hlsl" << 'EOF'
[numthreads(32, 1, 1)]
void main() {
    uint lane = WaveGetLaneIndex();
    float sum = WaveActiveSum(float(lane));
}
EOF

cat > "$CORPUS_DIR/seed3.hlsl" << 'EOF'
[numthreads(64, 1, 1)]
void main() {
    uint idx = WaveGetLaneIndex();
    uint product = WaveActiveProduct(idx + 1);
    uint maxVal = WaveActiveMax(idx);
}
EOF

echo "Created corpus with seed inputs"
echo

# Run the fuzzer for a short time to test it works
echo "Running MiniHLSL fuzzer for 30 seconds..."
echo "This will generate and test MiniHLSL-compliant mutations..."

cd "$SCRIPT_DIR"
timeout 30s "$BUILD_DIR/bin/minihlsl-fuzzer" "$CORPUS_DIR" -max_total_time=30 -print_final_stats=1

echo
echo "Fuzzer completed. Check $CORPUS_DIR for generated test cases."
echo "Any crashes would indicate bugs in MiniHLSL order-independence properties."