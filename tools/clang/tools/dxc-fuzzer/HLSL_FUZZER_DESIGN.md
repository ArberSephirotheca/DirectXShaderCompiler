# HLSL Wave Operation Fuzzer Design Document

## Overview
This document describes the design of an incremental HLSL program generator integrated with libFuzzer to find bugs in GPU wave operation implementations.

## Goals
1. Generate syntactically valid HLSL programs with complex control flow
2. Ensure wave operations occur under divergent control flow conditions
3. Cover edge cases in participant sets
4. Integrate seamlessly with libFuzzer for coverage-guided fuzzing
5. Apply semantic-preserving mutations only to newly generated code

## Architecture

### Core Components

```
libFuzzer Input → FuzzedDataProvider → Program Generator → Mutation Engine → Interpreter → Bug Detection
```

### Key Classes

1. **ConstrainedHLSLGenerator**: Main generator that creates programs with control flow
2. **ParticipantSetGenerator**: Generates conditions that create specific lane participation patterns
3. **LoopGenerator**: Creates loops with divergent exit patterns
4. **IncrementalMutator**: Applies mutations only to newly added code
5. **ProgramStateTracker**: Tracks which parts of program are new vs. existing

## Participant Set Patterns

### 1. Single Lane Participation
```hlsl
if (WaveGetLaneIndex() == k) {
    result = WaveActiveSum(value);
}
```
- Edge case: Only one lane participates
- Tests: Degenerate reductions, single-lane broadcasts

### 2. Sparse Non-Contiguous
```hlsl
if (WaveGetLaneIndex() == 0 || WaveGetLaneIndex() == 3 || WaveGetLaneIndex() == 5) {
    result = WaveActiveSum(value);
}
```
- Edge case: Non-consecutive lanes with gaps
- Tests: Hardware lane masking, sparse reductions

### 3. Contiguous Range
```hlsl
if (WaveGetLaneIndex() < N) {  // or >= N
    result = WaveActiveSum(value);
}
```
- Edge case: First/last N lanes
- Tests: Boundary conditions, partial wave operations

### 4. Disjoint Sets
```hlsl
if (WaveGetLaneIndex() < 2 || WaveGetLaneIndex() >= 30) {
    result = WaveActiveSum(value);
}
```
- Edge case: Two separate groups
- Tests: Multiple active regions in hardware

### 5. Parity Patterns
```hlsl
if ((WaveGetLaneIndex() & 1) == 0) {  // Even lanes
    result = WaveActiveSum(value);
}
```
- Edge case: Alternating lanes
- Tests: Stride patterns, SIMD lane pairing

### 6. Ensure Non-Empty (Wrapper Pattern)
- Always includes at least one guaranteed participant
- Prevents degenerate cases where no lanes execute wave op

## Control Flow Patterns

### 1. Simple Divergent If
```hlsl
if (lane_condition) {
    x = WaveActiveSum(y);
}
```

### 2. Nested Divergence
```hlsl
if (outer_condition) {
    x = WaveActiveSum(y);
    if (inner_condition) {
        z = WaveActiveProduct(x);
    }
}
```

### 3. Loop with Early Exit
```hlsl
for (int i = 0; i < N; ++i) {
    if (lane_specific_condition) continue;
    x = WaveActiveSum(i);
    if (iteration_and_lane_condition) break;
    y = WaveActiveProduct(x);
}
```

### 4. Cascading Conditions
```hlsl
if (condition1) {
    x = WaveOp1();
} else if (condition2) {
    x = WaveOp2();
} else {
    x = WaveOp3();
}
```

## WaveSize Support

### Interpreter Enhancement
```cpp
struct Program {
    uint32_t preferredWaveSize = 0; // 0 = no preference, 32/64 = specific size
};
```

### Attribute Parsing
- Support `[WaveSize(32)]` and `[WaveSize(64)]` attributes
- Test both wave sizes for same program
- Detect wave-size-dependent bugs

## Incremental Generation Strategy

### State Tracking
```cpp
struct ProgramState {
    interpreter::Program program;
    std::vector<StatementInfo> statements;
    GenerationHistory history;
};

struct StatementInfo {
    size_t index;
    bool isNewlyAdded;
    bool hasMutation;
    std::vector<WaveOpLocation> waveOps;
};
```

### Generation Pipeline
1. Start with minimal base program
2. Add control flow block with wave operations
3. Apply mutations to new wave operations only
4. Mark mutated code
5. Add more control flow (without mutating existing)
6. Apply mutations to newest additions
7. Repeat...

### Example Evolution
```hlsl
// Round 0: Base
void main(uint3 tid : SV_DispatchThreadID) {
    uint x = tid.x;
}

// Round 1: Add divergent if + mutate
void main(uint3 tid : SV_DispatchThreadID) {
    uint x = tid.x;
    if (WaveGetLaneIndex() < 2) {
        // LanePermutation applied here
        uint _perm_val = WaveReadLaneAt(x, (WaveGetLaneIndex() + 1) % 2);
        x = WaveActiveSum(_perm_val);
    }
}

// Round 2: Add loop + mutate new only
void main(uint3 tid : SV_DispatchThreadID) {
    uint x = tid.x;
    if (WaveGetLaneIndex() < 2) {
        uint _perm_val = WaveReadLaneAt(x, (WaveGetLaneIndex() + 1) % 2);
        x = WaveActiveSum(_perm_val);
    }
    // New loop - only this gets mutations
    for (int i = 0; i < 4; ++i) {
        if ((WaveGetLaneIndex() & 1) == 0) continue;
        // WaveParticipantTracking applied here
        y = WaveActiveProduct(i);
        _participant_check_sum[tid.x] += verify_participants();
    }
}
```

## libFuzzer Integration

### Entry Point
```cpp
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
    // Use fuzzer bytes to drive all generation decisions
    FuzzedDataProvider provider(Data, Size);
    
    // Generate program incrementally
    auto program = generateIncrementalProgram(provider);
    
    // Execute and verify
    return verifyProgram(program);
}
```

### Coverage Guidance
- libFuzzer tracks code coverage in the interpreter
- New control flow paths = new coverage = interesting input
- Mutations that trigger new GPU behaviors = preserved by fuzzer

### Deterministic Generation
- Same input bytes = same generated program
- Enables reproduction of bugs
- Allows minimization of crash inputs

## Implementation Plan

### Phase 1: Core Infrastructure
1. Add WaveSize attribute support to interpreter
2. Create ParticipantSetGenerator with all patterns
3. Implement basic ConstrainedHLSLGenerator

### Phase 2: Control Flow
1. Implement simple if generation with wave ops
2. Add nested control flow support
3. Implement loop generation with break/continue

### Phase 3: Incremental Generation
1. Create ProgramStateTracker
2. Implement incremental addition of control flow
3. Add mutation tracking to prevent re-mutation

### Phase 4: libFuzzer Integration
1. Create LLVMFuzzerTestOneInput entry point
2. Implement FuzzedDataProvider-based generation
3. Add crash reporting and minimization

### Phase 5: Testing & Refinement
1. Test all participant patterns
2. Verify incremental generation
3. Tune generation probabilities
4. Add more complex patterns as needed

## Success Metrics
1. Coverage of all participant set edge cases
2. Discovery of order-dependent behaviors
3. Generation of diverse, valid HLSL programs
4. Efficient fuzzing (programs/second)
5. Reproducible bug reports

## Future Extensions
1. Switch statements with wave ops
2. Do-while loops
3. Goto statements (if supported)
4. Multiple wave operations in single expression
5. Texture/buffer operations combined with wave ops