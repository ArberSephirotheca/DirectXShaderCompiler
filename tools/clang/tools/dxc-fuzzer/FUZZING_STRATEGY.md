# Trace-Guided Fuzzing Strategy for MiniHLSLInterpreter

## Overview

This document captures the design discussion and strategy for implementing a semantics-preserving fuzzer for the MiniHLSLInterpreter. The fuzzer is designed to find bugs in the interpreter's handling of control flow and wave operations while avoiding false positives.

## Key Insight: Semantics-Preserving Fuzzing

The core innovation is using a **trace-guided, semantics-preserving** approach:

1. **Trace Phase**: Execute the program once to capture complete execution behavior
2. **Mutation Phase**: Generate semantically-equivalent variants based on the trace
3. **Validation Phase**: Execute mutations and verify they produce identical results

Since every mutation is guaranteed to preserve semantics, any behavioral difference indicates a real interpreter bug.

## Initial Strategy Discussion

### Original Proposal

The initial strategy proposed:
- **Block-signature coverage**: Hash of (statement-id, phase, loop-iteration)
- **Wave-op witness trace**: Record operation, lane mask, and arguments
- **Final observable state digest**: Hash all outputs

### Evolution to Trace-Guided Approach

We evolved this to a more sophisticated trace-guided approach after realizing we could:
- Run the program once to capture **complete execution behavior**
- Use this trace to generate **guaranteed-safe mutations**
- Focus on interpreter-specific concerns (reconvergence, wave ops)

## Architecture Components

### 1. Execution Trace Capture

The `ExecutionTrace` structure captures:
- **Thread hierarchy**: Wave/lane organization matching interpreter structure
- **Dynamic block graph**: How blocks are created and lanes flow through them
- **Control flow decisions**: Exact conditions and branches taken per lane
- **Wave operations**: Participants, sync behavior, and results
- **Memory accesses**: Order and values for race detection

### 2. Trace-Guided Mutations

#### Control Flow Mutations
- **CF-01: Explicit Lane Divergence** - Convert implicit divergence to explicit lane tests
- **CF-02: Loop Unrolling** - Unroll with exact iteration counts from trace
- **CF-03: Split by Wave** - Handle each wave's divergence separately
- **CF-04: Redundant Branches** - Add redundant but equivalent control flow

#### Wave Operation Mutations
- **WV-01: Precompute Results** - Replace wave ops with traced results
- **WV-02: Redundant Sync** - Add extra synchronization that shouldn't change behavior
- **WV-03: Duplicate Operations** - Execute wave ops multiple times

#### Block Structure Mutations
- **BL-01: Force Block Boundaries** - Make implicit blocks explicit
- **BL-02: Split at Wave Ops** - Force new blocks around wave operations

#### Memory Access Mutations
- **MEM-01: Serialize Accesses** - Order concurrent accesses explicitly

### 3. Semantic Validation

The validator ensures mutations preserve:
- Final variable values
- Wave operation results
- Memory state
- Return values

## Integration with MiniHLSLInterpreter

### Alignment with Interpreter Architecture

The fuzzer leverages existing interpreter structures:
- `ThreadgroupContext` with multiple `WaveContext` objects
- `LaneContext` for per-lane state
- `DynamicExecutionBlock` for tracking participants
- `BlockMembershipRegistry` for lane-block relationships
- `WaveOperationSyncPoint` for wave op synchronization

### Key Corrections from Implementation Review

1. **LaneId is wave-local** (0 to waveSize-1), not global
2. **ThreadId is global**: `waveId * waveSize + laneId`
3. **Multiple waves** execute independently until barriers
4. **Dynamic blocks** track participants per wave

## Example Bugs This Approach Would Find

### 1. Reconvergence Bug
```hlsl
// Original
if (threadIdx.x < 16) {
  x = WaveActiveSum(1);
} else {
  x = WaveActiveSum(2);
}

// Mutated (adds always-true branch)
if (threadIdx.x < 16) {
  if (true) {  // New block created
    x = WaveActiveSum(1);
  }
} else {
  x = WaveActiveSum(2);
}
```
If the interpreter incorrectly handles nested blocks, participant sets might be wrong.

### 2. Wave Operation Ordering
```hlsl
// Original
a = WaveActiveSum(x);
b = WaveActiveProduct(y);

// Mutated (reordered)
b = WaveActiveProduct(y);
a = WaveActiveSum(x);
```
Should produce identical results if interpreter correctly handles independent wave ops.

### 3. Loop Reconvergence
```hlsl
// Original
for (int i = 0; i < laneId; i++) {
  sum += WaveActiveSum(1);
}

// Mutated (unrolled with guards)
if (laneId > 0) sum += WaveActiveSum(1);
if (laneId > 1) sum += WaveActiveSum(1);
// ...
```
Tests if interpreter correctly handles varying iteration counts across lanes.

## Benefits of This Approach

1. **No False Positives**: Every mutation is semantically equivalent
2. **Targeted Testing**: Focuses on interpreter-specific concerns
3. **Comprehensive Coverage**: Systematically explores execution patterns
4. **Debugging Support**: Complete traces available for analysis
5. **Scalable**: Can add new mutation strategies incrementally

## Implementation Status

### Completed
- [x] ExecutionTrace data structure design
- [x] TraceCaptureInterpreter hooks
- [x] Mutation strategy framework
- [x] Semantic validator design
- [x] Bug reporter structure
- [x] LibFuzzer integration points

### TODO
- [ ] Implement trace capture hooks in interpreter
- [ ] Implement mutation strategies
- [ ] Create seed corpus
- [ ] Integrate with build system
- [ ] Add test cases

## Future Extensions

1. **Cross-wave testing**: Test consistency across different wave configurations
2. **Barrier mutations**: Test different barrier placements
3. **Memory model testing**: More sophisticated concurrent access patterns
4. **Performance fuzzing**: Find pathological cases for interpreter performance

## References

- MiniHLSLInterpreter source code
- LibFuzzer documentation
- HLSL wave intrinsics specification