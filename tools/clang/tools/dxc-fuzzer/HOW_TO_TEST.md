# How to Test MiniHLSLValidator

## Quick Test (Without Full DXC Build)

For a quick test without building all of DXC:

```bash
cd /home/zheyuan/dxc_workspace/DirectXShaderCompiler/tools/clang/tools/dxc-fuzzer
clang++ -std=c++17 -o test_simple test_simple.cpp
./test_simple
```

## Full Testing with CTest

### 1. First Build DXC

```bash
# Create build directory if it doesn't exist
mkdir -p /home/zheyuan/dxc_workspace/DirectXShaderCompiler/build
cd /home/zheyuan/dxc_workspace/DirectXShaderCompiler/build

# Configure with CMake
cmake .. -C ../cmake/caches/PredefinedParams.cmake

# Build DXC (this takes a while)
cmake --build .
```

### 2. Build the Test Targets

```bash
# From the build directory
cd /home/zheyuan/dxc_workspace/DirectXShaderCompiler/build

# Build just the validator tests
cmake --build . --target test_minihlsl_validator test_minihlsl_advanced
```

### 3. Run Tests with CTest

```bash
# From the build directory (this is important!)
cd /home/zheyuan/dxc_workspace/DirectXShaderCompiler/build

# Run all MiniHLSL tests
ctest -R "MiniHLSL" -V

# Or run all tests
ctest

# List available tests
ctest -N

# Run with parallel execution
ctest -j4
```

## Why CTest Must Run from Build Directory

CTest looks for test definitions in the build directory's test configuration files:
- `CTestTestfile.cmake` files throughout the build tree
- Test executables in their build locations
- CMake's test registry

Running from source directory won't find these files.

## Alternative: Direct Execution

If CTest isn't working, run the test executables directly:

```bash
# From build directory
./tools/clang/tools/dxc-fuzzer/test_minihlsl_validator
./tools/clang/tools/dxc-fuzzer/test_minihlsl_advanced

# Or from source directory with full path
/home/zheyuan/dxc_workspace/DirectXShaderCompiler/build/tools/clang/tools/dxc-fuzzer/test_minihlsl_validator
```

## Verifying CTest Integration

To verify tests are registered with CTest:

```bash
cd /home/zheyuan/dxc_workspace/DirectXShaderCompiler/build
grep -r "add_test.*MiniHLSL" .
```

## Common Issues

1. **"No tests found"** - You're not in the build directory
2. **Tests don't build** - DXC needs to be built first
3. **Missing dependencies** - Ensure all Clang/LLVM libraries are built

## Using CTest in LLVM/Clang Projects

Yes, CTest is standard in LLVM/Clang projects:
- LLVM uses CTest for CMake-based testing
- Integrates with `lit` (LLVM Integrated Tester)
- Works with continuous integration systems
- Provides test timing and parallel execution

The typical LLVM testing stack:
1. **CTest** - Test orchestration and discovery
2. **lit** - FileCheck-based integration tests
3. **gtest** - C++ unit tests (like DXC uses)
4. **FileCheck** - Output verification