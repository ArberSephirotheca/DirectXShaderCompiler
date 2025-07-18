# Adding Loop Support to MiniHLSL Order Independence Framework

## Current Status

**❌ Problem**: Our C++ validator supports loops but our Lean4 formal proof doesn't, creating a gap between theory and implementation.

**✅ Solution**: Extend our formal model first, then update the C++ validator to match.

## What We Have vs What We Need

### Current Lean4 Model (`OrderIndependenceProof.lean`)
```lean
inductive Stmt where
  | assign : String → PureExpr → Stmt
  | uniformIf : PureExpr → List Stmt → List Stmt → Stmt  -- Only uniform if
  | waveAssign : String → WaveOp → Stmt
  | threadgroupAssign : String → ThreadgroupOp → Stmt
  | barrier : Stmt
  -- NO LOOPS!
```

### Extended Model (`OrderIndependenceProof_Extended.lean`)
```lean
inductive Stmt where
  | assign : String → PureExpr → Stmt
  | uniformIf : PureExpr → List Stmt → List Stmt → Stmt
  | waveAssign : String → WaveOp → Stmt
  | threadgroupAssign : String → ThreadgroupOp → Stmt
  | barrier : Stmt
  -- NEW: Loop constructs with uniformity requirements
  | uniformFor : String → PureExpr → PureExpr → PureExpr → List Stmt → Stmt
  | uniformWhile : PureExpr → List Stmt → Stmt
  | uniformSwitch : PureExpr → List (PureExpr × List Stmt) → List Stmt → Stmt
  | break : Stmt
  | continue : Stmt
```

## Key Theoretical Requirements

### 1. **Uniform Loop Conditions**
- **Requirement**: Loop conditions must be uniform across all threads
- **Rationale**: Divergent loops break wave operation safety
- **Example**: 
  ```hlsl
  // GOOD: Uniform condition
  for (uint i = 0; i < 4; i++) { ... }
  
  // BAD: Divergent condition  
  for (uint i = 0; i < WaveGetLaneIndex(); i++) { ... }
  ```

### 2. **Wave Operation Placement**
- **Requirement**: Wave operations only allowed after loop convergence
- **Rationale**: Ensures all threads participate in wave operations
- **Example**:
  ```hlsl
  // GOOD: Wave op after uniform loop
  for (uint i = 0; i < 4; i++) { total += i; }
  float waveSum = WaveActiveSum(total);
  
  // BAD: Wave op in potentially divergent loop
  for (uint i = 0; i < condition; i++) {
    float sum = WaveActiveSum(1.0f);  // ERROR!
  }
  ```

### 3. **Break/Continue Semantics**
- **Requirement**: Break/continue must preserve uniformity
- **Rationale**: Non-uniform control flow breaks wave operations
- **Implementation**: Track loop state uniformly across all threads

## Formal Theorems to Prove

### 1. **Uniform Loop Order Independence**
```lean
theorem uniform_loops_are_order_independent (stmt : Stmt) :
  isValidLoop stmt tgCtx → isLoopOrderIndependent stmt
```

### 2. **Safety Condition**
```lean
theorem loop_safety_requires_uniformity (stmt : Stmt) :
  isLoopOrderIndependent stmt → (∀ tgCtx, isValidLoop stmt tgCtx)
```

### 3. **Program-Level Guarantee**
```lean
theorem extended_minihlsl_order_independent (program : List Stmt) :
  (∀ stmt ∈ program, isValidLoop stmt tgCtx) →
  (∀ stmt ∈ program, isLoopOrderIndependent stmt)
```

## Implementation Plan

### Phase 1: Complete Formal Model
1. **Extend execution semantics** for all loop constructs
2. **Prove uniformity theorems** for loop conditions
3. **Prove order independence** for uniform loops
4. **Add counterexamples** for divergent loops

### Phase 2: Update C++ Validator
1. **Align with formal model** - only support uniform loops
2. **Add uniformity validation** for loop conditions
3. **Implement wave operation checking** in loop contexts
4. **Add proper error messages** for violations

### Phase 3: Integration Testing
1. **Test with uniform loop programs** - should pass
2. **Test with divergent loop programs** - should fail
3. **Verify order independence** with generated variants
4. **Validate against formal theorems**

## Example Valid Programs

### Uniform For Loop
```hlsl
[numthreads(64, 1, 1)]
void main() {
    float total = 0.0f;
    for (uint i = 0; i < 4; i++) {    // Uniform condition
        total += float(i);
    }
    float waveSum = WaveActiveSum(total);  // Safe: after uniform loop
}
```

### Uniform While Loop
```hlsl
[numthreads(64, 1, 1)]
void main() {
    uint counter = 0;
    while (counter < 3) {             // Uniform condition
        counter++;
        if (counter == 2) break;      // Uniform break
    }
    uint waveCount = WaveActiveSum(counter);  // Safe: after uniform loop
}
```

## Example Invalid Programs

### Divergent Loop Condition
```hlsl
[numthreads(64, 1, 1)]
void main() {
    float total = 0.0f;
    for (uint i = 0; i < WaveGetLaneIndex(); i++) {  // DIVERGENT!
        total += float(i);
    }
    float waveSum = WaveActiveSum(total);  // ERROR: Not all threads participate
}
```

### Wave Operation in Divergent Loop
```hlsl
[numthreads(64, 1, 1)]
void main() {
    for (uint i = 0; i < someCondition; i++) {
        float sum = WaveActiveSum(1.0f);  // ERROR: In potentially divergent loop
    }
}
```

## Next Steps

1. **Complete the formal proofs** in `OrderIndependenceProof_Extended.lean`
2. **Update the C++ validator** to match the formal model exactly
3. **Add comprehensive tests** for both valid and invalid cases
4. **Document the uniformity requirements** for users

This approach ensures that our C++ implementation is backed by formal mathematical guarantees, providing confidence in the correctness of our order independence validation.