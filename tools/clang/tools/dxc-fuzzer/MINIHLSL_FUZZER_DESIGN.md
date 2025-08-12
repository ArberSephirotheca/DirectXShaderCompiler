# MiniHLSL Fuzzer Comprehensive Design Document

## Overview

This document combines and updates the design for the MiniHLSL fuzzer, a sophisticated trace-guided fuzzer for finding bugs in GPU wave operation implementations. The fuzzer uses dynamic execution blocks to enable context-aware mutations of wave operations, even within complex control flow like loops.

## Architecture

### Core Approach: Trace-Guided Semantics-Preserving Fuzzing

```
Input → Program Generation → Execution & Trace Capture → Mutation Based on Trace → Validation
```

The key innovation is using **dynamic execution blocks** to distinguish different instances of the same static statement, enabling precise context-aware mutations.

### Key Components

1. **IncrementalGenerator**: Creates HLSL programs with complex control flow
2. **TraceCaptureInterpreter**: Captures complete execution traces with dynamic blocks
3. **TraceGuidedFuzzer**: Applies mutations based on captured traces
4. **SemanticValidator**: Ensures mutations preserve program semantics
5. **IncrementalFuzzingPipeline**: Orchestrates the fuzzing process

## Dynamic Execution Blocks - The Key Innovation

### What Are Dynamic Blocks?

Dynamic blocks are runtime instances of static program blocks that capture:
- The specific execution context (loop iteration, branch taken)
- The participating lanes for that specific instance
- Parent-child relationships forming an execution graph

### Why Dynamic Blocks Matter

Consider this example:
```hlsl
for (int i = 0; i < 3; i++) {
    if (WaveGetLaneIndex() == i) {
        result = WaveActiveSum(value);  // Static: one wave op
    }
}
```

This creates **three different dynamic blocks**:
- Block 34 (iteration 0): Only lane 0 participates
- Block 134 (iteration 1): Only lane 1 participates  
- Block 234 (iteration 2): Only lane 2 participates

### How This Enables Context-Aware Mutations

With dynamic blocks, we can mutate specific instances:
```hlsl
for (int i = 0; i < 3; i++) {
    if (WaveGetLaneIndex() == i) {
        result = WaveActiveSum(value);
    }
    // Mutation targeting block 34 (iteration 0)
    else if (i == 0 && WaveGetLaneIndex() == 2) {
        result = WaveActiveSum(value);  // Add lane 2 to iteration 0
    }
    // Mutation targeting block 134 (iteration 1)
    else if (i == 1 && WaveGetLaneIndex() == 3) {
        result = WaveActiveSum(value);  // Add lane 3 to iteration 1
    }
}
```

This solves the fundamental challenge of mutating wave operations in loops where participant sets depend on dynamic execution context.

## Execution Trace Structure

```cpp
struct ExecutionTrace {
    // Thread hierarchy
    ThreadHierarchy threadHierarchy;
    
    // Dynamic execution blocks - THE KEY DATA STRUCTURE
    std::map<uint32_t, BlockExecutionRecord> blocks;
    
    // Wave operations with their block context
    std::vector<WaveOpRecord> waveOperations;
    
    // Control flow decisions
    std::vector<ControlFlowDecision> controlFlowHistory;
    
    // Final program state
    FinalState finalState;
};

struct BlockExecutionRecord {
    uint32_t blockId;                    // Unique identifier
    BlockType blockType;                 // If/Then/Else/Loop/etc
    const Statement* sourceStatement;    // Static statement
    uint32_t parentBlockId;             // Execution context
    
    // Per-wave participation
    std::map<WaveId, WaveParticipationInfo> waveParticipation;
};
```

## Mutation Strategies

### 1. Lane Permutation Mutations
Permute input values between lanes before wave operations:
```hlsl
// Original
result = WaveActiveSum(value);

// Mutated
uint _perm_val = WaveReadLaneAt(value, (laneId + 1) % participantCount);
result = WaveActiveSum(_perm_val);
```

### 2. Participant Tracking Mutations
Add lanes to specific dynamic blocks:
```hlsl
// Target: Add lane 3 to block 134 (iteration 1)
if (i == 1 && WaveGetLaneIndex() == 3) {
    result = WaveActiveSum(value);
}
```

### 3. Wave Operation Duplication
Execute wave operations multiple times to test consistency:
```hlsl
// Original
result = WaveActiveSum(value);

// Mutated
result = WaveActiveSum(value);
uint _verify = WaveActiveSum(value);
assert(result == _verify);
```

### 4. Explicit Reconvergence
Force reconvergence points to test interpreter behavior:
```hlsl
// Original
if (condition) {
    x = WaveActiveSum(y);
}

// Mutated
if (condition) {
    x = WaveActiveSum(y);
}
WaveBarrier();  // Force reconvergence
```

## Participant Pattern Generation

### Core Patterns

1. **Single Lane**: `WaveGetLaneIndex() == k`
2. **Sparse Non-Contiguous**: `lane == 0 || lane == 3 || lane == 5`
3. **Contiguous Range**: `lane < N` or `lane >= N`
4. **Disjoint Sets**: `lane < 2 || lane >= 30`
5. **Parity Patterns**: `(lane & 1) == 0` (even lanes)
6. **Complex Patterns**: `(lane % 3 == 0) && (lane < 16)`

### Pattern Composition in Loops

```cpp
class LoopAwarePattern : public ParticipantPattern {
    std::unique_ptr<Expression> generateCondition(
        uint32_t waveSize, 
        uint32_t loopIteration,  // NEW: iteration context
        FuzzedDataProvider& provider) {
        // Generate different patterns for different iterations
        if (loopIteration == 0) {
            return makeSingleLane(0);
        } else if (loopIteration == 1) {
            return makeParityPattern(1);  // Odd lanes
        } else {
            return makeContiguousRange(loopIteration, waveSize);
        }
    }
};
```

## Incremental Generation Pipeline

### Generation Phases

1. **Base Program**: Initialize with minimal structure
2. **Add Control Flow**: Insert if/loop with wave operations  
3. **Capture Trace**: Execute to get dynamic blocks
4. **Apply Mutations**: Target specific dynamic blocks
5. **Validate**: Ensure semantic equivalence
6. **Iterate**: Add more control flow, repeat

### State Tracking

```cpp
struct ProgramState {
    interpreter::Program program;
    std::vector<GenerationRound> history;
    std::set<std::string> declaredVariables;
    
    // Track mutations per dynamic block
    std::map<uint32_t, std::vector<AppliedMutation>> blockMutations;
};

struct GenerationRound {
    size_t roundNumber;
    std::vector<size_t> addedStatementIndices;
    std::vector<uint32_t> newDynamicBlocks;  // From execution trace
    std::map<uint32_t, MutationType> blockMutations;  // Per-block mutations
};
```

## Implementation Architecture

### File Structure
```
MiniHLSLInterpreterFuzzer.cpp         # Main fuzzer entry point
HLSLProgramGenerator.cpp              # Program generation
IncrementalFuzzingPipeline.cpp        # Orchestration
TraceGuidedFuzzer.cpp                 # Mutation engine
MiniHLSLInterpreterTraceCapture.cpp   # Trace capture
ExecutionTrace.h                      # Trace data structures
HLSLParticipantPatterns.cpp           # Pattern generation
```

### Key Algorithms

#### Dynamic Block Creation
```cpp
uint32_t findOrCreateBlockForPath(
    const ExecutionPath& path,
    const Statement* stmt,
    BlockType type,
    ThreadgroupContext& tg) {
    
    // Check if block exists for this path
    auto blockId = computeBlockId(path, stmt);
    
    if (tg.executionBlocks.find(blockId) == tg.executionBlocks.end()) {
        // Create new dynamic block
        DynamicExecutionBlock block(blockId, type, stmt);
        block.setParentBlock(path.getCurrentBlock());
        tg.executionBlocks[blockId] = block;
    }
    
    return blockId;
}
```

#### Context-Aware Mutation Generation
```cpp
std::unique_ptr<Statement> generateMutationForBlock(
    uint32_t targetBlockId,
    const BlockExecutionRecord& blockRecord,
    const ExecutionTrace& trace) {
    
    // Extract iteration context from block's execution path
    auto iterationInfo = extractIterationContext(blockRecord);
    
    // Generate condition that matches this specific iteration
    auto condition = generateIterationSpecificCondition(iterationInfo);
    
    // Add participants based on mutation strategy
    auto newParticipants = selectNewParticipants(blockRecord);
    
    return buildMutationStatement(condition, newParticipants);
}
```

## Example: Complete Fuzzing Cycle

### 1. Initial Program
```hlsl
[numthreads(4, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint result = 0;
    for (uint i = 0; i < 3; i++) {
        if (WaveGetLaneIndex() == i) {
            result = WaveActiveSum(result + i);
        }
    }
}
```

### 2. Execution Trace Captures
```
Block 10 (Loop Header i=0): All lanes
  └─ Block 20 (If condition): Lane 0 passes
      └─ Block 30 (Then body): Lane 0 executes WaveActiveSum

Block 110 (Loop Header i=1): All lanes  
  └─ Block 120 (If condition): Lane 1 passes
      └─ Block 130 (Then body): Lane 1 executes WaveActiveSum

Block 210 (Loop Header i=2): All lanes
  └─ Block 220 (If condition): Lane 2 passes  
      └─ Block 230 (Then body): Lane 2 executes WaveActiveSum
```

### 3. Mutation Applied
Target: Add lane 3 to iteration 1 (block 130)
```hlsl
for (uint i = 0; i < 3; i++) {
    if (WaveGetLaneIndex() == i) {
        result = WaveActiveSum(result + i);
    }
    // MUTATION: Specific to iteration 1, block 130
    else if (i == 1 && WaveGetLaneIndex() == 3) {
        result = WaveActiveSum(result + i);
    }
}
```

### 4. Validation
- Execute mutated program
- Verify wave operation results match expected values
- Any divergence indicates interpreter bug

## Bug Classes Detected

### 1. Reconvergence Bugs
- Incorrect participant tracking across control flow
- Missing reconvergence after divergent branches
- Wrong merge points in nested control flow

### 2. Loop-Related Bugs  
- Incorrect participant sets across iterations
- State leakage between iterations
- Wrong handling of break/continue with wave ops

### 3. Dynamic Block Management Bugs
- Incorrect block creation/reuse
- Wrong parent-child relationships
- Missing participant updates

### 4. Wave Operation Bugs
- Race conditions in collective operations
- Incorrect reduction algorithms
- Wrong broadcast behavior

## Performance Optimizations

1. **Trace Caching**: Reuse traces for similar programs
2. **Incremental Trace Updates**: Only trace new code
3. **Parallel Mutation Testing**: Test multiple mutations concurrently
4. **Smart Seed Selection**: Prioritize seeds that expose new blocks

## Future Extensions

1. **Multi-Wave Patterns**: Test wave group interactions
2. **Barrier Mutations**: Test synchronization primitives
3. **Memory Model Testing**: Add shared memory operations
4. **Performance Fuzzing**: Find performance anomalies
5. **Compiler Integration**: Test DXC's HLSL compilation

## Success Metrics

1. **Coverage**: All participant pattern combinations tested
2. **Bug Discovery**: Real interpreter bugs found
3. **Performance**: >100 programs tested per second
4. **Reliability**: Zero false positives
5. **Reproducibility**: All bugs have minimal repros

## Conclusion

The key innovation of using dynamic execution blocks enables precise, context-aware mutations of wave operations even in complex control flow. This approach systematically explores the interpreter's behavior space while maintaining semantic equivalence, ensuring all reported bugs are real implementation issues rather than test artifacts.