# MiniHLSL Standalone Interpreter

A standalone command-line tool that can execute HLSL compute shader programs and verify their order independence.

## Building

```bash
# From the DirectXShaderCompiler build directory
ninja minihlsl-standalone
```

## Usage

```bash
./minihlsl-standalone [options] <hlsl_file>
```

### Options

- `-h, --help` - Show help message
- `-v, --verbose` - Enable verbose output
- `-g, --debug-graph` - Print dynamic execution graph (planned feature)
- `-n, --no-verify` - Skip order independence verification
- `-o, --orderings N` - Number of thread orderings to test (default: 10)
- `-w, --wave-size N` - Wave size (default: 32, must be power of 2)

### Examples

```bash
# Basic execution
./minihlsl-standalone examples/simple_wave_sum.hlsl

# Verbose output with custom wave size
./minihlsl-standalone -v --wave-size 16 examples/control_flow_test.hlsl

# Skip verification for quick testing
./minihlsl-standalone --no-verify examples/shared_memory_test.hlsl

# Test with many orderings and smaller wave size
./minihlsl-standalone --orderings 50 --wave-size 8 examples/simple_wave_sum.hlsl
```

## Features

### ✅ Implemented
- **HLSL Parsing**: Uses Clang to parse HLSL compute shaders
- **AST Conversion**: Converts Clang AST to interpreter program
- **Execution**: Runs programs with configurable thread/wave settings
- **Order Independence Verification**: Tests multiple thread execution orderings
- **Command Line Interface**: Full CLI with options and help

### 🚧 Planned
- **Debug Graph Output**: Visualize dynamic execution blocks
- **Performance Profiling**: Measure execution time and memory usage
- **Interactive Mode**: Step-by-step execution debugging
- **Output Formats**: JSON, XML output for tooling integration

## Supported HLSL Features

- ✅ Wave intrinsics (WaveActiveSum, WaveGetLaneIndex, etc.)
- ✅ Control flow (if/else, loops, switch)
- ✅ Shared memory with barriers
- ✅ Basic arithmetic and logical operations
- ✅ Thread/dispatch ID intrinsics
- ⚠️ Limited data types (int, float, bool)

## Example Output

```
MiniHLSL Interpreter
====================
Input file: examples/simple_wave_sum.hlsl
Wave size: 32
Thread configuration: [32, 1, 1]
Order verification: 10 orderings

✅ Found main function in HLSL source
✅ Successfully converted HLSL to interpreter program

=== Sequential Execution ===
✅ Execution successful
Thread Return Values:
  Thread 0: 10416
  Thread 1: 10416
  ...

=== Order Independence Verification ===
✅ PASS: Program is order-independent!
Verified across 10 different thread orderings.

✅ Execution completed successfully!
```

## Error Handling

The interpreter provides detailed error messages for:
- HLSL parsing errors
- AST conversion failures
- Runtime execution errors
- Order dependence violations
- File I/O issues

## Integration

The standalone interpreter can be integrated into:
- CI/CD pipelines for HLSL validation
- Automated testing frameworks
- Shader debugging workflowss
- Research and educational tools