# OrderIndependenceProof.lean Documentation

This document explains each definition and theorem in the OrderIndependenceProof.lean file, which provides a formal framework for proving order independence in miniHLSL shader programs at both wave-level and threadgroup-level.

## Core Type Definitions

### Basic Types
- **`WaveId`**: Identifies a specific wave within a threadgroup
- **`LaneId`**: Identifies a specific lane within a wave  
- **`MemoryAddress`**: Represents shared memory addresses
- **`Value`**: Represents computed values (aliased to `Int`)

### Memory Access Patterns
```lean
inductive MemoryAccessPattern where
  | readOnly : MemoryAddress → MemoryAccessPattern
  | writeDisjoint : MemoryAddress → WaveId → MemoryAccessPattern
  | writeReduction : MemoryAddress → MemoryAccessPattern
```
**Purpose**: Models how waves access shared memory
- `readOnly`: Memory address is only read from
- `writeDisjoint`: Each wave writes to different addresses (no conflicts)
- `writeReduction`: All waves write the same reduction result

### Execution Context Structures

#### `WaveContext`
```lean
structure WaveContext where
  activeLanes : Finset LaneId
  laneCount : Nat
  laneValues : LaneId → Value
```
**Purpose**: Represents the execution state of a single GPU wave
- `activeLanes`: Which lanes are currently active (not diverged)
- `laneCount`: Total number of lanes in the wave
- `laneValues`: Values stored in each lane

#### `SharedMemory`
```lean
structure SharedMemory where
  data : MemoryAddress → Value
  accessPattern : MemoryAddress → Finset WaveId
```
**Purpose**: Models shared memory accessible by all waves in a threadgroup
- `data`: Values stored at each memory address
- `accessPattern`: Which waves access each memory address

#### `ThreadgroupContext`
```lean
structure ThreadgroupContext where
  threadgroupSize : Nat
  waveSize : Nat
  waveCount : Nat
  activeWaves : Finset WaveId
  waveContexts : WaveId → WaveContext
  sharedMemory : SharedMemory
  h_size_constraint : threadgroupSize = waveCount * waveSize
```
**Purpose**: Represents the execution state of an entire GPU threadgroup
- Contains multiple waves and shared memory
- Enforces the constraint that threadgroup size equals wave count × wave size

## Expression and Operation Types

### `PureExpr`
```lean
inductive PureExpr where
  | literal : Value → PureExpr
  | laneIndex : PureExpr
  | waveIndex : PureExpr
  | threadIndex : PureExpr
  | add : PureExpr → PureExpr → PureExpr
  | mul : PureExpr → PureExpr → PureExpr
  | comparison : PureExpr → PureExpr → PureExpr
```
**Purpose**: Represents pure expressions that don't have side effects
- Always produces the same output for the same input
- Forms the foundation for order-independent operations

### `WaveOp`
```lean
inductive WaveOp where
  | activeSum : PureExpr → WaveOp
  | activeProduct : PureExpr → WaveOp
  | activeMax : PureExpr → WaveOp
  | activeMin : PureExpr → WaveOp
  | activeCountBits : PureExpr → WaveOp
  | getLaneCount : WaveOp
```
**Purpose**: Wave-level operations that work within a single wave
- All operations are commutative and associative
- Results don't depend on lane execution order

### `ThreadgroupOp`
```lean
inductive ThreadgroupOp where
  | barrier : ThreadgroupOp
  | sharedRead : MemoryAddress → ThreadgroupOp
  | sharedWrite : MemoryAddress → PureExpr → ThreadgroupOp
  | sharedAtomicAdd : MemoryAddress → PureExpr → ThreadgroupOp
```
**Purpose**: Threadgroup-level operations that work across waves
- Requires careful synchronization and memory access patterns
- `barrier`: Synchronizes all waves
- `sharedAtomicAdd`: Atomic addition (commutative, order-independent)

### `Stmt`
```lean
inductive Stmt where
  | assign : String → PureExpr → Stmt
  | uniformIf : PureExpr → List Stmt → List Stmt → Stmt
  | waveAssign : String → WaveOp → Stmt
  | threadgroupAssign : String → ThreadgroupOp → Stmt
  | barrier : Stmt
```
**Purpose**: MiniHLSL statements for building programs
- Restricted to constructs that can maintain order independence

## Evaluation Functions

### `evalPureExpr`
```lean
def evalPureExpr (expr : PureExpr) (tgCtx : ThreadgroupContext) (waveId : WaveId) (laneId : LaneId) : Int
```
**Purpose**: Evaluates pure expressions in a given context
- Deterministic: same inputs always produce same outputs
- Foundation for proving order independence

### `evalWaveOp`
```lean
def evalWaveOp (op : WaveOp) (tgCtx : ThreadgroupContext) (waveId : WaveId) : Int
```
**Purpose**: Evaluates wave operations within a specific wave
- Uses commutative/associative operations (sum, product, max, min)
- Results are independent of lane execution order

### `evalThreadgroupOp`
```lean
def evalThreadgroupOp (op : ThreadgroupOp) (tgCtx : ThreadgroupContext) : Int
```
**Purpose**: Evaluates threadgroup operations across all waves
- Handles shared memory access and synchronization
- Requires additional constraints for order independence

## Safety Constraints

### `hasDisjointWrites`
```lean
def hasDisjointWrites (tgCtx : ThreadgroupContext) : Prop
```
**Purpose**: Ensures no two waves write to the same memory address
- Prevents race conditions
- Necessary condition for threadgroup order independence

### `hasOnlyCommutativeOps`
```lean
def hasOnlyCommutativeOps (tgCtx : ThreadgroupContext) : Prop
```
**Purpose**: Ensures only commutative operations are used
- Operations like addition, multiplication are allowed
- Subtraction, division are forbidden (non-commutative)

## Order Independence Properties

### `isWaveOrderIndependent`
```lean
def isWaveOrderIndependent (op : WaveOp) : Prop
```
**Purpose**: Defines when a wave operation is order-independent
- Result doesn't depend on lane execution order within the wave
- Requires same lane values and wave structure

### `isThreadgroupOrderIndependent`
```lean
def isThreadgroupOrderIndependent (op : ThreadgroupOp) : Prop
```
**Purpose**: Defines when a threadgroup operation is order-independent
- Result doesn't depend on wave execution order within the threadgroup
- Requires safety constraints (disjoint writes, commutative ops)

### `isThreadgroupProgramOrderIndependent`
```lean
def isThreadgroupProgramOrderIndependent (program : List Stmt) : Prop
```
**Purpose**: Defines when an entire program is order-independent
- **Key insight**: The program parameter is now actually used! The definition analyzes each statement in the program
- Checks that every statement uses only order-independent operations:
  - `Stmt.threadgroupAssign`: Must use order-independent threadgroup operations
  - `Stmt.waveAssign`: Must use order-independent wave operations  
  - `Stmt.assign`: Pure assignments are always order-independent
  - `Stmt.barrier`: Synchronization points are order-independent
  - `Stmt.uniformIf`: Uniform control flow is order-independent
- Combines wave-level and threadgroup-level requirements
- **Fixed**: Previous version had unused program parameter - now it properly analyzes the program structure

## Key Theorems

### `evalPureExpr_deterministic`
```lean
lemma evalPureExpr_deterministic (expr : PureExpr) (tgCtx1 tgCtx2 : ThreadgroupContext) (waveId : WaveId) (laneId : LaneId)
```
**Purpose**: Proves that pure expressions are deterministic
- Same lane values + same wave size → same result
- Foundation for all order independence proofs

### `minihlsl_wave_operations_order_independent`
```lean
theorem minihlsl_wave_operations_order_independent : ∀ (op : WaveOp), isWaveOrderIndependent op
```
**Purpose**: Proves all MiniHLSL wave operations are order-independent
- Covers activeSum, activeProduct, activeMax, activeMin, activeCountBits, getLaneCount
- Uses induction on operation types and relies on commutativity/associativity

### `minihlsl_threadgroup_operations_order_independent`
```lean
theorem minihlsl_threadgroup_operations_order_independent : ∀ (op : ThreadgroupOp), isThreadgroupOrderIndependent op
```
**Purpose**: Proves MiniHLSL threadgroup operations are order-independent under constraints
- Requires safety constraints (disjoint writes, commutative ops)
- Handles barrier synchronization and atomic operations

## Safety Validation Theorems

### `disjoint_writes_preserve_order_independence`
```lean
theorem disjoint_writes_preserve_order_independence
```
**Purpose**: Proves that disjoint writes are sufficient for order independence
- If no two waves write to the same address, no race conditions occur
- Validates the `hasDisjointWrites` constraint

### `commutative_ops_preserve_order_independence`
```lean
theorem commutative_ops_preserve_order_independence
```
**Purpose**: Proves that commutative operations preserve order independence
- Operations like atomic add don't depend on execution order
- Validates the `hasOnlyCommutativeOps` constraint

## Counterexample Theorems

### `prefix_operations_not_order_independent`
```lean
theorem prefix_operations_not_order_independent
```
**Purpose**: Proves that prefix operations violate order independence
- Prefix sums depend on lane execution order
- Justifies excluding prefix operations from MiniHLSL

### `overlapping_writes_not_order_independent`
```lean
theorem overlapping_writes_not_order_independent
```
**Purpose**: Proves that overlapping writes break order independence
- Multiple waves writing to same address creates race conditions
- Justifies the `hasDisjointWrites` requirement

### `non_commutative_ops_not_order_independent`
```lean
theorem non_commutative_ops_not_order_independent
```
**Purpose**: Proves that non-commutative operations break order independence
- Operations like subtraction (A - B ≠ B - A) depend on execution order
- Justifies the `hasOnlyCommutativeOps` requirement

## High-Level Integration Theorems

### `threadgroup_example_program_order_independent`
```lean
theorem threadgroup_example_program_order_independent
```
**Purpose**: Proves that a concrete MiniHLSL program is order-independent
- Demonstrates the framework applied to a real program
- Shows how constraints ensure order independence

### `main_threadgroup_order_independence`
```lean
theorem main_threadgroup_order_independence
```
**Purpose**: Main theorem proving MiniHLSL programs are order-independent
- If safety constraints are satisfied, the program is order-independent
- Provides the theoretical foundation for a practical checker

## Implementation Implications

This formal framework provides:

1. **Specification**: Precise definitions of what "order independence" means
2. **Validation**: Proof that MiniHLSL operations satisfy the requirements
3. **Constraints**: Identification of necessary safety conditions
4. **Counterexamples**: Examples of patterns that break order independence
5. **Completeness**: Coverage of both positive and negative results

The framework can be directly implemented as a static analysis tool that:
- Checks if a shader program uses only allowed MiniHLSL constructs
- Verifies that memory access patterns are disjoint
- Ensures only commutative operations are used
- Flags potentially problematic patterns (prefix ops, overlapping writes, non-commutative ops)

This provides a solid theoretical foundation for detecting reconvergence bugs in GPU shader compilation.