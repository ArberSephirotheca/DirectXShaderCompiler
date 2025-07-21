# Formal Specification of MiniHLSL

## 1. Introduction

MiniHLSL is a minimal subset of HLSL (High-Level Shading Language) designed for formal verification of GPU shader programs. This specification defines the syntax, semantics, and order independence properties of MiniHLSL programs.

### 1.1 Purpose
MiniHLSL serves as a formal model for reasoning about GPU program correctness, specifically focusing on:
- Order independence of thread execution
- Wave-level cooperative operations
- Threadgroup-level synchronization
- Memory safety in parallel contexts

### 1.2 Key Design Principles
1. **Order Independence**: Programs must produce identical results regardless of thread execution order
2. **Deterministic Control Flow**: Control flow decisions must be statically analyzable
3. **Memory Safety**: Concurrent memory accesses must follow strict safety rules
4. **Wave Coherence**: Wave operations require all participating threads to be active

## 2. Lexical Structure

### 2.1 Basic Types
```
LaneId        ::= Natural number (0 to WaveSize-1)
WaveId        ::= Natural number (0 to WaveCount-1)  
ThreadId      ::= Natural number (0 to ThreadgroupSize-1)
Value         ::= Integer
MemoryAddress ::= Natural number
```

### 2.2 Keywords
```
if, else, for, while, switch, case, default, break, continue, barrier
```

### 2.3 Built-in Functions
```
WaveGetLaneIndex, WaveGetLaneCount, WaveActiveSum, WaveActiveProduct,
WaveActiveMax, WaveActiveMin, WaveActiveCountBits, GroupMemoryBarrierWithGroupSync
```

## 3. Type System

### 3.1 Value Types
- `int`: 32-bit signed integer
- `uint`: 32-bit unsigned integer  
- `float`: 32-bit floating point (treated as integer in formal model)

### 3.2 Memory Types
- **Private Memory**: Per-thread storage (variables)
- **Shared Memory**: Per-threadgroup storage (groupshared)

## 4. Syntax

### 4.1 Expressions

#### 4.1.1 Pure Expressions
Pure expressions have no side effects and always evaluate to the same value in a given context.

```
PureExpr ::= literal(Value)
           | laneIndex                    // WaveGetLaneIndex()
           | waveIndex                    // Current wave ID
           | threadIndex                  // Global thread ID
           | add(PureExpr, PureExpr)
           | mul(PureExpr, PureExpr)
           | comparison(PureExpr, PureExpr)
```

#### 4.1.2 Compile-Time Deterministic Expressions
Expressions that can be fully evaluated at compile time given thread/lane/wave indices.

```
CompileTimeDeterministicExpr ::= literal(Value)
                               | laneIndex
                               | waveIndex
                               | threadIndex
                               | add(CompileTimeDeterministicExpr, CompileTimeDeterministicExpr)
                               | mul(CompileTimeDeterministicExpr, CompileTimeDeterministicExpr)
                               | comparison(CompileTimeDeterministicExpr, CompileTimeDeterministicExpr)
```

### 4.2 Operations

#### 4.2.1 Wave Operations
Cooperative operations that execute across all active lanes in a wave.

```
WaveOp ::= activeSum(PureExpr)         // Sum across active lanes
         | activeProduct(PureExpr)     // Product across active lanes
         | activeMax(PureExpr)         // Maximum across active lanes
         | activeMin(PureExpr)         // Minimum across active lanes
         | activeCountBits(PureExpr)   // Count of non-zero values
         | getLaneCount                // Number of active lanes
```

#### 4.2.2 Threadgroup Operations
Operations that involve shared memory or synchronization across waves.

```
ThreadgroupOp ::= barrier                              // GroupMemoryBarrierWithGroupSync
                | sharedRead(MemoryAddress)            // Read from shared memory
                | sharedWrite(MemoryAddress, PureExpr) // Write to shared memory
                | InterlockedAdd(MemoryAddress, PureExpr) // Atomic add
```

### 4.3 Statements

```
Stmt ::= assign(Variable, PureExpr)           // Variable assignment
       | waveAssign(Variable, WaveOp)         // Wave operation result assignment
       | threadgroupAssign(Variable, ThreadgroupOp) // Threadgroup op assignment
       | barrier                              // Synchronization barrier
       | deterministicIf(CompileTimeDeterministicExpr, List[Stmt], List[Stmt])
       | deterministicFor(CompileTimeDeterministicExpr, CompileTimeDeterministicExpr,
                         CompileTimeDeterministicExpr, List[Stmt])
       | deterministicWhile(CompileTimeDeterministicExpr, List[Stmt])
       | deterministicSwitch(CompileTimeDeterministicExpr,
                            List[(CompileTimeDeterministicExpr, List[Stmt])], List[Stmt])
       | breakStmt
       | continueStmt
```

### 4.4 Programs
```
Program ::= List[Stmt]
```

## 5. Semantic Rules

### 5.1 Execution Model

#### 5.1.1 Thread Organization
- **Threadgroup**: Collection of threads executing together
- **Wave**: Hardware unit of SIMD execution (typically 32 or 64 threads)
- **Lane**: Individual thread within a wave

#### 5.1.2 Execution Contexts

**Wave Context**:
```
WaveContext = {
  waveId: WaveId,
  laneCount: Natural,
  activeLanes: Set[LaneId],
  laneValues: LaneId → Value
}
```

**Threadgroup Context**:
```
ThreadgroupContext = {
  threadgroupSize: Natural,
  waveSize: Natural,
  waveCount: Natural,
  activeWaves: Set[WaveId],
  waveContexts: WaveId → WaveContext,
  sharedMemory: MemoryAddress → Value,
  constraint: threadgroupSize = waveCount × waveSize
}
```

### 5.2 Evaluation Semantics

#### 5.2.1 Pure Expression Evaluation
```
evalPureExpr : PureExpr → WaveContext → ThreadgroupContext → LaneId → Value

evalPureExpr (literal v) _ _ _ = v
evalPureExpr laneIndex _ _ lane = lane
evalPureExpr waveIndex ctx _ _ = ctx.waveId
evalPureExpr threadIndex ctx tgCtx lane = ctx.waveId * tgCtx.waveSize + lane
evalPureExpr (add e1 e2) ctx tgCtx lane = 
  evalPureExpr e1 ctx tgCtx lane + evalPureExpr e2 ctx tgCtx lane
evalPureExpr (mul e1 e2) ctx tgCtx lane =
  evalPureExpr e1 ctx tgCtx lane * evalPureExpr e2 ctx tgCtx lane
evalPureExpr (comparison e1 e2) ctx tgCtx lane =
  if evalPureExpr e1 ctx tgCtx lane = evalPureExpr e2 ctx tgCtx lane then 1 else 0
```

#### 5.2.2 Wave Operation Evaluation
```
evalWaveOp : WaveOp → WaveContext → ThreadgroupContext → Value

evalWaveOp (activeSum expr) ctx tgCtx =
  Σ_{lane ∈ ctx.activeLanes} evalPureExpr expr ctx tgCtx lane

evalWaveOp (activeProduct expr) ctx tgCtx =
  Π_{lane ∈ ctx.activeLanes} evalPureExpr expr ctx tgCtx lane

evalWaveOp (activeMax expr) ctx tgCtx =
  max_{lane ∈ ctx.activeLanes} evalPureExpr expr ctx tgCtx lane

evalWaveOp (activeMin expr) ctx tgCtx =
  min_{lane ∈ ctx.activeLanes} evalPureExpr expr ctx tgCtx lane

evalWaveOp (activeCountBits expr) ctx tgCtx =
  |{lane ∈ ctx.activeLanes : evalPureExpr expr ctx tgCtx lane ≠ 0}|

evalWaveOp getLaneCount ctx _ = |ctx.activeLanes|
```

### 5.3 Order Independence Properties

#### 5.3.1 Wave-Level Order Independence
A wave operation is order-independent if it produces the same result regardless of the order in which lanes are processed:

```
∀ π : Permutation(activeLanes), 
  evalWaveOp op ctx tgCtx = evalWaveOp op (permute ctx π) tgCtx
```

#### 5.3.2 Threadgroup-Level Order Independence
A threadgroup program is order-independent if it produces the same final state regardless of wave execution order:

```
∀ π : Permutation(activeWaves),
  execProgram program tgCtx = execProgram program (permute tgCtx π)
```

### 5.4 Safety Constraints

#### 5.4.1 Data-Race-Free Memory Model

MiniHLSL follows a data-race-free memory model similar to C++11:

**Memory Access Definitions**:
- A memory access is either a read or write operation to a memory location
- Two memory accesses **conflict** if:
  - They access the same memory location
  - At least one of them is a write operation
  - They are performed by different threads

**Data Race Definition**:
A program has a **data race** if it contains two conflicting memory accesses that are not ordered by synchronization. Specifically, a data race occurs when:
1. Two threads access the same memory location concurrently
2. At least one access is a write
3. The accesses are not synchronized by:
   - A barrier operation (`GroupMemoryBarrierWithGroupSync`)
   - Atomic operations (`InterlockedAdd`)
   - Happens-before relationships established by synchronization

**Undefined Behavior**:
If a data race occurs, the behavior of the program is **undefined**.

#### 5.4.2 Memory Access Patterns
For order independence and data-race freedom, shared memory accesses must follow one of these patterns:

1. **Read-Only**: All threads only read from an address (no data race possible)
2. **Disjoint Writes**: Each thread writes to different addresses (no conflicting accesses)
3. **Synchronized Access**: Accesses are ordered by barriers or atomic operations
4. **Commutative Atomic Operations**: All writes use commutative atomic operations (e.g., `InterlockedAdd`)

#### 5.4.3 Hybrid Approach for Compound Read-Modify-Write Operations

MiniHLSL uses a **hybrid approach** to handle compound operations that both read and write memory:

**Simple RMW Constraint** (`simpleRMWRequiresAtomic`):
- **Scope**: Same-thread operations to the same address
- **Rule**: If a thread performs both a read and write to the same address, then either:
  1. The operations are synchronized by barriers, OR
  2. They must be replaced by atomic operations (`InterlockedAdd`, etc.)
- **Rationale**: Prevents intra-thread races in compound expressions like `x = x + 1`

**Complex Operations Constraint** (`complexOperationsAreSynchronized`):
- **Scope**: Cross-thread operations to the same address  
- **Rule**: If different threads access the same address and at least one isn't atomic, then:
  1. The operations must be synchronized by barriers, OR
  2. They must be ordered by happens-before relationships
- **Rationale**: Prevents inter-thread races for complex memory patterns

**Examples**:

```hlsl
// ✅ VALID: Simple atomic RMW
InterlockedAdd(g_counter, 1);  // Atomic handles read-modify-write

// ✅ VALID: Barrier-synchronized RMW  
int temp = g_data[tid];               // Read
GroupMemoryBarrierWithGroupSync();    // Barrier
g_data[tid] = temp + 1;              // Write (synchronized)

// ❌ INVALID: Unsynchronized compound RMW
g_data[0] += threadId;  // Multiple threads, same address, no synchronization
```

**Implementation Benefits**:
- **Flexibility**: Allows both atomic and barrier-based approaches
- **Completeness**: Covers both intra-thread and inter-thread scenarios
- **Verification**: Formally proven to prevent data races

#### 5.4.4 Synchronization Primitives

**Barriers**:
- `GroupMemoryBarrierWithGroupSync` ensures all memory operations before the barrier complete before any operations after the barrier begin
- All threads in the threadgroup must reach the barrier (a happens-before edge is established)

**Atomic Operations**:
- `InterlockedAdd` and similar operations provide atomic read-modify-write semantics
- Multiple concurrent atomic operations to the same location are serialized

#### 5.4.5 Control Flow Constraints
1. **Deterministic Conditions**: All control flow conditions must be compile-time deterministic
2. **Barrier Synchronization**: All waves must reach barriers together

## 6. Valid Program Examples

### 6.1 Wave Reduction
```hlsl
[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint laneId = WaveGetLaneIndex();
    float value = float(laneId);
    float sum = WaveActiveSum(value);  // Order-independent reduction
}
```

### 6.2 Threadgroup Reduction with Barriers
```hlsl
groupshared float g_data[256];

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint threadId = tid.x;
    
    // Phase 1: Each thread writes to its own location
    g_data[threadId] = float(threadId);
    
    GroupMemoryBarrierWithGroupSync();  // Synchronization
    
    // Phase 2: Wave-level reductions
    if (threadId < 64) {
        float sum = WaveActiveSum(g_data[threadId]);
    }
}
```

### 6.3 Deterministic Control Flow
```hlsl
[numthreads(64, 1, 1)]
void main() {
    uint laneId = WaveGetLaneIndex();
    float result = 0.0f;
    
    // Deterministic condition based on lane index
    if (laneId < 32) {
        result = 1.0f;
    } else {
        result = 2.0f;
    }
    
    // All lanes participate in wave operation
    float sum = WaveActiveSum(result);  // Safe: deterministic divergence
}
```

### 6.4 Wave Operations in Deterministic Divergent Control Flow
```hlsl
[numthreads(64, 1, 1)]
void main() {
    uint laneId = WaveGetLaneIndex();
    float value = float(laneId);
    
    // Deterministic divergent control flow with wave operations
    if (laneId < 16) {
        // Only lanes 0-15 are active here
        float partialSum = WaveActiveSum(value);  // Valid: deterministic divergence
    } else if (laneId < 32) {
        // Only lanes 16-31 are active here  
        float partialMax = WaveActiveMax(value);  // Valid: deterministic divergence
    }
    // Testing hardware behavior with deterministic but non-uniform execution
}
```

### 6.5 Data-Race-Free Shared Memory Access
```hlsl
groupshared float g_sum;
groupshared float g_data[256];

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint threadId = tid.x;
    
    // Phase 1: Disjoint writes (no data race)
    g_data[threadId] = float(threadId);
    
    GroupMemoryBarrierWithGroupSync();  // Synchronization point
    
    // Phase 2: Read shared data (safe after barrier)
    float neighborValue = g_data[(threadId + 1) % 256];
    
    // Phase 3: Atomic operations (no data race)
    InterlockedAdd(g_sum, neighborValue);  // Atomic add
    
    GroupMemoryBarrierWithGroupSync();  // Synchronization point
    
    // Phase 4: All threads can safely read the sum
    float finalSum = g_sum;  // Safe read after barrier
}
```

## 7. Invalid Program Examples

### 7.1 Non-Deterministic Control Flow
```hlsl
// INVALID: Runtime-dependent condition (not compile-time deterministic)
if (someRuntimeValue > threshold) {
    float sum = WaveActiveSum(1.0f);  // ERROR: Non-deterministic divergence
}
```

### 7.2 Data Races
```hlsl
// INVALID: Conflicting memory accesses without synchronization
groupshared float g_data[64];

[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint id = tid.x;
    
    // ERROR: Data race - multiple threads write to same location
    g_data[0] += float(id);  // Conflicting writes without synchronization
    
    // ERROR: Read-write data race without synchronization
    float value = g_data[id % 32];  // Thread 0 reads while thread 32 writes
    g_data[id % 32] = value * 2.0f;
}
```

### 7.3 Order-Dependent Operations
```hlsl
// INVALID: Result depends on execution order
groupshared float g_data[64];

void main(uint3 tid : SV_DispatchThreadID) {
    uint id = tid.x;
    g_data[0] = g_data[0] * 2 + id;  // ERROR: Non-commutative operation
}
```

## 8. Formal Properties

### 8.1 Theorems

**Theorem 1 (Wave Operation Commutativity)**:
All MiniHLSL wave operations are commutative and associative over their active lane sets.

**Theorem 2 (Deterministic Control Flow Order Independence)**:
Programs with only compile-time deterministic control flow are order-independent.

**Theorem 3 (Data-Race Freedom)**:
Programs with properly synchronized shared memory accesses are data-race-free and have well-defined behavior.

**Theorem 4 (Memory Safety and Order Independence)**:
Programs following the memory access patterns (read-only, disjoint writes, synchronized access, or atomic operations) are both data-race-free and order-independent.

**Theorem 5 (Program Composition)**:
Sequential composition of order-independent statements preserves order independence.

**Theorem 6 (Hybrid RMW Safety)**:
Programs following the hybrid approach for compound read-modify-write operations (simple RMW atomics + complex operation synchronization) are guaranteed to be data-race-free and maintain order independence.

### 8.2 Counterexamples

**Counterexample 1**: Prefix operations (scan) are not order-independent
**Counterexample 2**: Overlapping non-commutative writes break order independence
**Counterexample 3**: Runtime-dependent control flow may cause deadlock
**Counterexample 4**: Unsynchronized concurrent accesses create data races and undefined behavior

## 9. Implementation Notes

### 9.1 Static Analysis Requirements
A MiniHLSL validator must check:
1. All control flow conditions are compile-time deterministic
2. Memory access patterns follow safety constraints
<!-- 3. Wave operations occur only in converged control flow -->
3. No data races in shared memory accesses

### 9.2 Runtime Guarantees
Valid MiniHLSL programs guarantee:
1. Deterministic results regardless of thread scheduling
2. No deadlocks or race conditions
3. Portable execution across different GPU architectures
<!-- 4. Optimization opportunities for parallel execution -->

<!-- ## 10. Future Extensions

Planned extensions to MiniHLSL include:
1. Additional atomic operations
2. Texture and buffer operations
3. Extended type system (vectors, matrices)
4. Function definitions and calls -->

---

This specification provides the formal foundation for reasoning about GPU program correctness in the context of order independence. All constructs are designed to be verifiable through static analysis and formal proof systems.