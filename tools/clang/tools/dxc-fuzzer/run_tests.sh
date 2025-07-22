#!/bin/bash

# Simple script to build and run MiniHLSL validator tests

# Get absolute paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DXC_ROOT="$( cd "$SCRIPT_DIR/../../../.." && pwd )"
BUILD_DIR="$DXC_ROOT/build-fuzzer"
TEST_DIR="$SCRIPT_DIR/test_cases"

echo "Simple MiniHLSL Test Runner"
echo "============================"
echo "Debug: SCRIPT_DIR = $SCRIPT_DIR"
echo "Debug: DXC_ROOT = $DXC_ROOT" 
echo "Debug: BUILD_DIR = $BUILD_DIR"
echo "Debug: TEST_DIR = $TEST_DIR"
echo

# Check if we're in the right directory
if [ ! -f "MiniHLSLValidator.h" ]; then
    echo "Error: Run this from the dxc-fuzzer directory"
    exit 1
fi

# Create clean build directory in project root
echo "Creating clean build directory..."
if [ -d "$BUILD_DIR" ]; then
    echo "Removing existing build directory..."
    rm -rf "$BUILD_DIR"
fi
echo "Creating new build directory at: $BUILD_DIR"
mkdir -p "$BUILD_DIR"

# Configure with CMake from DXC root using clang++
echo "Configuring with CMake using clang++..."
echo "DXC Root: $DXC_ROOT"
echo "Build Dir: $BUILD_DIR"
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

# Build the test runner
echo "Building test_runner..."
cmake --build . --target test_runner

if [ $? -ne 0 ]; then
    echo "Build failed!"
    exit 1
fi

echo "Build successful!"
echo

# Run the test runner on test_cases directory
cd -
if [ -d "$TEST_DIR" ]; then
    echo "Running tests on $TEST_DIR directory..."
    "$BUILD_DIR/bin/test_runner" "$TEST_DIR"
else
    echo "Warning: $TEST_DIR directory not found"
    echo "Creating sample test files..."
    
    mkdir -p "$TEST_DIR"
    
    # Create a valid test
    cat > "$TEST_DIR/valid_simple.hlsl" << 'EOF'
[numthreads(32, 1, 1)]
void main() {
    uint lane = WaveGetLaneIndex();
    float sum = WaveActiveSum(float(lane));
}
EOF
    
    # Create an invalid test
    cat > "$TEST_DIR/invalid_prefix.hlsl" << 'EOF'
[numthreads(32, 1, 1)]
void main() {
    uint lane = WaveGetLaneIndex();
    float prefix = WavePrefixSum(float(lane));
}
EOF
    
    echo "Created sample test files. Running tests..."
    "$BUILD_DIR/bin/test_runner" "$TEST_DIR"
fi