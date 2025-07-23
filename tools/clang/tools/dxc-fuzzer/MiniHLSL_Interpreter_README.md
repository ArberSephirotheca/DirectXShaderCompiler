# MiniHLSL Interpreter

A complete interpreter implementation for the MiniHLSL language subset, designed to execute GPU shader programs and verify their order independence properties.

## Overview

The MiniHLSL Interpreter provides:
- **Parallel execution simulation** with configurable thread orderings
- **Order independence verification** by testing multiple execution orderings
- **Wave operation support** (reductions, lane queries)
- **Shared memory model** with barrier synchronization
- **Deterministic control flow** handling
- **Memory safety analysis** with data race detection

## Key Features

### 1. Execution Model
- Simulates threadgroup execution with waves and lanes
- Supports multiple thread execution orderings (sequential, random, interleaved)
- Models GPU memory hierarchy (private variables, shared memory)

### 2. Order Independence Testing
Instead of testing all possible orderings (which would explode exponentially), the interpreter uses a practical approach:
- **Default: 10 different thread orderings** including sequential, reverse, random, and interleaved patterns
- **Configurable**: You can specify the number of orderings to test
- **Smart selection**: Includes deterministic patterns plus random variations

### 3. Supported MiniHLSL Constructs

#### Expressions
- Literals: `42`, `3.14f`, `true`
- Lane intrinsics: `WaveGetLaneIndex()`, `WaveGetLaneCount()`
- Thread intrinsics: `GetThreadIndex()`, `WaveGetWaveIndex()`
- Arithmetic: `+`, `-`, `*`, `/`, `%`
- Comparisons: `==`, `!=`, `<`, `<=`, `>`, `>=`
- Logical: `&&`, `||`, `!`

#### Wave Operations (Order-Independent)
- `WaveActiveSum(expr)`
- `WaveActiveProduct(expr)`
- `WaveActiveMin(expr)`, `WaveActiveMax(expr)`
- `WaveActiveAnd(expr)`, `WaveActiveOr(expr)`, `WaveActiveXor(expr)`
- `WaveActiveCountBits(expr)`

#### Statements
- Variable declarations: `var name = expr;`
- Assignments: `name = expr;`
- Deterministic if statements: `if (deterministicCondition) { ... }`
- For loops: `for (i = 0; i < count; i++) { ... }`
- Return statements: `return expr;`
- Barriers: `GroupMemoryBarrierWithGroupSync();`
- Shared memory: `g_shared[addr] = value;`, `value = g_shared[addr];`

## Building

### Using Make
```bash
make -f Makefile_interpreter
```

### Using CMake
```bash
mkdir build
cd build
cmake .. -f ../CMakeLists_interpreter.txt
make
```

### Manual Compilation
```bash
g++ -std=c++17 -O2 -pthread MiniHLSLInterpreter.cpp test_interpreter.cpp -o test_interpreter
```

## Running Tests

```bash
./test_interpreter
```

The test suite includes:
1. **Wave Reduction Test**: Demonstrates order-independent wave operations
2. **Deterministic Branching Test**: Shows deterministic control flow based on lane indices
3. **Shared Memory Test**: Tests proper barrier synchronization
4. **Order-Dependent Test**: Demonstrates a program that fails verification

## Usage Examples

### Basic Usage

```cpp
#include "MiniHLSLInterpreter.h"
using namespace minihlsl::interpreter;

// Create a simple program
Program program;
program.numThreadsX = 32;

// Add statements
program.statements.push_back(makeVarDecl("laneId", makeLaneIndex()));
program.statements.push_back(makeVarDecl("sum", makeWaveSum(makeVariable("laneId"))));
program.statements.push_back(std::make_unique<ReturnStmt>(makeVariable("sum")));

// Create interpreter
MiniHLSLInterpreter interpreter;

// Verify order independence
auto result = interpreter.verifyOrderIndependence(program);
if (result.isOrderIndependent) {
    std::cout << "✅ Program is order-independent!" << std::endl;
} else {
    std::cout << "❌ Program has order dependence!" << std::endl;
    std::cout << result.divergenceReport << std::endl;
}
```

### Creating Custom Thread Orderings

```cpp
// Create custom orderings
std::vector<ThreadOrdering> orderings = {
    ThreadOrdering::sequential(32),
    ThreadOrdering::reverseSequential(32),
    ThreadOrdering::random(32, 12345),
    ThreadOrdering::evenOddInterleaved(32),
    ThreadOrdering::waveInterleaved(32, 16)
};

// Test each ordering
for (const auto& ordering : orderings) {
    auto result = interpreter.execute(program, ordering);
    std::cout << "Ordering: " << ordering.description 
              << ", Result: " << result.threadReturnValues[0].toString() << std::endl;
}
```

### Building Complex Programs

```cpp
// Create a program with deterministic control flow
Program program;
program.numThreadsX = 32;

// var laneId = WaveGetLaneIndex();
program.statements.push_back(makeVarDecl("laneId", makeLaneIndex()));

// if (laneId < 16) { ... } else { ... }
std::vector<std::unique_ptr<Statement>> thenBlock;
thenBlock.push_back(makeAssign("result", makeLiteral(Value(1))));

std::vector<std::unique_ptr<Statement>> elseBlock;
elseBlock.push_back(makeAssign("result", makeLiteral(Value(2))));

program.statements.push_back(makeIf(
    makeBinaryOp(makeVariable("laneId"), makeLiteral(Value(16)), BinaryOpExpr::Lt),
    std::move(thenBlock),
    std::move(elseBlock)
));
```

## Thread Ordering Strategies

The interpreter provides several built-in thread ordering strategies:

1. **Sequential**: `0 → 1 → 2 → 3 → ...`
2. **Reverse Sequential**: `31 → 30 → 29 → ...`  
3. **Even-Odd Interleaved**: `0 → 2 → 4 → ... → 1 → 3 → 5 → ...`
4. **Wave Interleaved**: Execute waves in reverse order
5. **Random**: Randomly shuffled execution order

This provides good coverage of potential ordering issues without the exponential explosion of testing all possible orderings.

## Architecture

### Core Components

- **Value**: Variant type supporting int, float, bool with automatic conversions
- **LaneContext**: Per-thread execution state and variables
- **WaveContext**: SIMD execution unit containing multiple lanes
- **ThreadgroupContext**: GPU threadgroup with multiple waves and shared memory
- **SharedMemory**: Thread-safe shared memory with access tracking
- **Expression/Statement**: AST nodes for program representation

### Execution Model

1. **Program Parsing**: Build AST from MiniHLSL constructs
2. **Thread Ordering**: Generate test orderings
3. **Execution**: Simulate parallel execution with chosen ordering
4. **Verification**: Compare results across different orderings
5. **Reporting**: Generate detailed reports on order dependence

## Limitations and Future Extensions

### Current Limitations
- Fixed wave size (32 threads)
- Simplified barrier model
- No texture/buffer operations
- Limited to compute shader model

### Potential Extensions
- Dynamic wave sizes
- More sophisticated barrier synchronization
- Texture and buffer operations
- Graphics shader support (vertex, pixel)
- Performance profiling
- Debugging and visualization tools

## Integration with Validation

The interpreter complements the formal MiniHLSL validator:

1. **Static Analysis** (Validator): Checks syntax and semantic rules
2. **Dynamic Analysis** (Interpreter): Verifies actual execution behavior
3. **Formal Proof** (Lean): Mathematical guarantees of order independence

This three-layer approach provides comprehensive verification of GPU shader correctness.

## Error Handling

The interpreter provides detailed error reporting for:
- **Data races**: Conflicting memory accesses
- **Divergence**: Order-dependent execution
- **Runtime errors**: Division by zero, invalid operations
- **Synchronization issues**: Barrier deadlocks

## Performance

The interpreter is designed for testing and verification rather than high performance:
- **Thread count**: Practical for up to 1024 threads
- **Ordering count**: Default 10 orderings provides good coverage
- **Memory usage**: Minimal overhead for typical shader sizes

For larger-scale testing, consider reducing the number of orderings or implementing more sophisticated sampling strategies.

## Contributing

When extending the interpreter:
1. Maintain order independence focus
2. Add comprehensive tests for new features
3. Update documentation
4. Consider formal verification implications
5. Profile performance impact

## References

- [MiniHLSL Specification](MiniHLSL.md)
- [Formal Proof Documentation](proof/README.md)
- [MiniHLSL Validator](MiniHLSLValidator.h)

The interpreter provides a practical tool for understanding and verifying GPU shader behavior while maintaining strong theoretical foundations through its integration with formal verification.