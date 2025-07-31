# Semantics-Preserving Fuzzing for MiniHLSL Interpreter

## Overview

This document outlines strategies for fuzzing the MiniHLSL interpreter while preserving program semantics. The goal is to generate diverse test cases that exercise different execution paths, thread scheduling, and SIMT behavior without changing the expected computational results.

## Core Principles

1. **Preserve Computational Logic**: Don't change the mathematical operations or control flow logic
2. **Vary Execution Context**: Change thread scheduling, wave operations, and synchronization patterns
3. **Maintain Determinism**: Ensure results remain predictable for validation
4. **Exercise Edge Cases**: Target boundary conditions and rare scheduling scenarios

## Fuzzing Strategies

### 1. Thread Scheduling Variations

#### 1.1 Lane Execution Order Permutation
```hlsl
// Original: Sequential lane execution 0,1,2,3
// Fuzzed: Random order like 2,0,3,1
[numthreads(4, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint laneId = WaveGetLaneIndex();
    // Same logic, different execution order
}
```

**Implementation**: Modify thread scheduler to randomize lane processing order while maintaining wave semantics.

#### 1.2 Interleaved Control Flow Execution
```hlsl
// Vary which lanes execute if/while bodies first
while (condition) {
    if (laneId == someValue) {
        // Vary which lanes enter this branch first
        continue;
    }
    result += WaveActiveSum(1);
}
```

**Implementation**: Insert randomized yield points to change control flow interleaving.

### 2. Wave Operation Scheduling

#### 2.1 Wave Synchronization Point Delays
```hlsl
// Original: Immediate wave operation
result += WaveActiveSum(value);

// Fuzzed: Delayed synchronization
result += WaveActiveSum(value); // Some lanes may reach this first
```

**Implementation**: Introduce controlled delays before wave operations to test different arrival patterns.

#### 2.2 Partial Wave Participation
```hlsl
// Test scenarios where not all lanes participate simultaneously
if (condition) {
    uint partialSum = WaveActiveSum(1); // Only some lanes participate initially
}
```

**Implementation**: Vary which lanes are "ready" when wave operations begin.

### 3. Control Flow Variations

#### 3.1 Loop Iteration Interleaving
```hlsl
// Original: All lanes progress through loop iterations together
for (int i = 0; i < 4; ++i) {
    // body
}

// Fuzzed: Lanes at different iteration stages
// Lane 0 at i=2, Lane 1 at i=1, Lane 2 at i=3, Lane 3 at i=0
```

**Implementation**: Allow different lanes to be at different loop iterations simultaneously.

#### 3.2 Nested Control Flow Scheduling
```hlsl
while (outerCondition) {
    if (innerCondition) {
        for (int j = 0; j < count; ++j) {
            // Vary execution order of nested structures
        }
    }
}
```

**Implementation**: Randomize which nested control structures execute first across lanes.

### 4. Memory Access Patterns

#### 4.1 Variable Read/Write Timing
```hlsl
// Vary when lanes read/write shared variables
uint sharedVar = computeValue();
// Some lanes may read old value, others new value
uint result = WaveActiveSum(sharedVar);
```

**Implementation**: Introduce timing variations in variable access while maintaining data dependencies.

#### 4.2 Register Allocation Variations
```hlsl
// Use different temporary variable arrangements
uint temp1 = a + b;
uint temp2 = c * d;
uint result = temp1 + temp2;

// vs

uint result = (a + b) + (c * d); // Same result, different register usage
```

### 5. Exception and State Management

#### 5.1 WaveOperationWaitException Timing
```hlsl
// Vary when WaveOperationWaitException is thrown and handled
try {
    result += WaveActiveSum(value); // May throw at different times
} catch (WaveOperationWaitException) {
    // Different lanes may be suspended at different points
}
```

#### 5.2 Thread State Transitions
```hlsl
// Test different sequences of thread state changes
// Ready -> WaitingForWave -> Ready
// Ready -> WaitingForResume -> Ready
// Ready -> WaitingAtBarrier -> Ready
```

## Specific Test Case Generators

### 1. Loop Continuation Fuzzer
```hlsl
// Template for continue statement fuzzing
while (i < maxIter) {
    if (fuzzCondition(laneId, i)) {  // Vary this condition
        i += fuzzIncrement();        // Vary increment timing
        continue;
    }
    result += WaveActiveSum(fuzzValue()); // Vary wave op timing
    i += fuzzIncrement();
}
```

**Parameters to Fuzz**:
- `fuzzCondition()`: Different lane-based conditions
- `fuzzIncrement()`: When increment happens (before/after operations)
- `fuzzValue()`: Values for wave operations
- Thread scheduling between iterations

### 2. Reconvergence Point Fuzzer
```hlsl
// Test different reconvergence scenarios
if (divergentCondition) {
    // Branch A
    blockA_operations();
} else {
    // Branch B  
    blockB_operations();
}
// Reconvergence point - vary arrival timing
post_divergence_operations();
```

**Fuzzing Targets**:
- Which lanes arrive at reconvergence point first
- Order of merge block creation
- Timing of block cleanup operations

### 3. Wave Size Variations
```hlsl
// Test with different effective wave sizes
[numthreads(X, 1, 1)]  // X = 1, 2, 4, 8, 16, 32
void main(uint3 id : SV_DispatchThreadID) {
    // Same logic, different wave participation patterns
}
```

## Implementation Framework

### 1. Fuzzer Configuration
```cpp
struct FuzzConfig {
    bool randomizeLaneOrder = true;
    bool delayWaveOps = true;
    bool interleaveControlFlow = true;
    uint32_t maxDelaySteps = 10;
    uint32_t randomSeed = 0;
};
```

### 2. Deterministic Randomization
```cpp
class DeterministicFuzzer {
    std::mt19937 rng;
    
public:
    DeterministicFuzzer(uint32_t seed) : rng(seed) {}
    
    bool shouldDelay() { return rng() % 4 == 0; }
    uint32_t getDelaySteps() { return rng() % maxDelaySteps + 1; }
    std::vector<uint32_t> shuffleLanes(std::vector<uint32_t> lanes);
};
```

### 3. Validation Framework
```cpp
struct TestResult {
    std::vector<uint32_t> laneResults;
    uint32_t waveSum;
    bool isCorrect;
    std::string executionTrace;
};

bool validateSemantics(const TestResult& original, const TestResult& fuzzed) {
    return original.laneResults == fuzzed.laneResults && 
           original.waveSum == fuzzed.waveSum;
}
```

## Test Categories

### 1. Stress Tests
- Maximum thread divergence scenarios
- Deep nested control flow with different scheduling
- Large wave sizes with complex synchronization

### 2. Edge Cases
- Single-lane waves
- Empty loop bodies with wave operations
- Immediate continue/break statements
- Nested wave operations

### 3. Regression Tests
- Known bug scenarios with different scheduling
- Previously fixed reconvergence issues
- Thread state management edge cases

## Validation Strategies

### 1. Golden Reference Comparison
```cpp
// Run same test with different scheduling
auto result1 = runWithScheduling(test, ScheduleType::Sequential);
auto result2 = runWithScheduling(test, ScheduleType::Random);
assert(validateSemantics(result1, result2));
```

### 2. Invariant Checking
```cpp
// Check that certain properties hold regardless of scheduling
assert(totalWaveSum == expectedSum);
assert(allLanesReachFinalState());
assert(noDataRaces());
```

### 3. Differential Testing
```cpp
// Compare against reference implementation
auto interpreterResult = runInterpreter(test);
auto referenceResult = runReference(test);
assert(resultsMatch(interpreterResult, referenceResult));
```

## Integration with Build System

### 1. Automated Fuzzing
```bash
# Generate and run fuzzed tests
./dxc-fuzzer --mode=fuzz --seed=12345 --iterations=1000
./dxc-fuzzer --mode=validate --reference-impl=hlsl
```

### 2. Continuous Integration
```yaml
# CI pipeline step
- name: Semantics-Preserving Fuzz Testing
  run: |
    ./build-fuzzer/dxc-fuzzer --fuzz-suite=basic --validate
    ./build-fuzzer/dxc-fuzzer --fuzz-suite=stress --validate
```

## Future Extensions

1. **Dynamic Test Generation**: Generate new test cases based on code coverage
2. **Mutation-Based Fuzzing**: Modify existing tests while preserving semantics  
3. **Property-Based Testing**: Use formal specifications to guide fuzzing
4. **Performance Fuzzing**: Test different scheduling impacts on performance
5. **Multi-Wave Scenarios**: Extend to multiple waves and cross-wave interactions

## Conclusion

Semantics-preserving fuzzing for the MiniHLSL interpreter focuses on varying execution context and timing while maintaining computational correctness. This approach helps uncover subtle bugs in thread scheduling, synchronization, and control flow handling that might not appear in deterministic tests.

The key is to maintain the mathematical and logical correctness of programs while exploring the vast space of possible execution interleavings and thread scheduling decisions that can occur in real SIMT hardware.