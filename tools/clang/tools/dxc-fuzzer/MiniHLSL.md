# MiniHLSL Specification

## Overview

MiniHLSL is a minimal subset of HLSL designed to capture core syntax and semantics while ensuring **order-independence**: the execution order of threads/lanes does not affect the final result. This constraint enables effective testing of reconvergence behavior in shader compilers.

Inspired by miniRust, MiniHLSL prioritizes:
- **Precise operational semantics** with explicit order-independence guarantees
- **Human readability** using familiar HLSL syntax
- **Minimal core** that covers essential GPU compute patterns
- **Explicit modeling** of non-deterministic execution

## Design Principles

### 1. Order-Independence Constraint
All valid MiniHLSL programs must satisfy the **Order-Independence Property**:
> For any input, all possible interleavings of thread execution produce semantically equivalent results.

### 2. Deterministic Operations Only
MiniHLSL restricts operations to those with well-defined, deterministic semantics:
- Pure functions (no side effects)
- Commutative/associative reductions
- Uniform control flow
- Synchronized memory access patterns

### 3. Explicit Wave Operations
Wave intrinsics must be used correctly with proper convergence guarantees.

## Grammar Definition

### Core Types
```hlsl
Type ::= 
    | "int" | "uint" | "float" | "bool"
    | "int2" | "int3" | "int4"
    | "float2" | "float3" | "float4"
    | "uint2" | "uint3" | "uint4"
```

### Expressions (Order-Independent Only)
```hlsl
Expr ::= 
    | Literal
    | Variable
    | BinaryOp(Expr, Expr)      // +, -, *, /, %, ==, !=, <, <=, >, >=, &&, ||
    | UnaryOp(Expr)             // -, !, ~
    | WaveIntrinsic(Expr*)      // Wave operations with convergence guarantees
    | Select(Expr, Expr, Expr)  // condition ? true_expr : false_expr
    
Literal ::= IntLit | FloatLit | BoolLit

// Restricted binary operations (order-independent only)
BinaryOp ::= 
    | ArithOp      // +, -, *, /, % (on same-typed operands)
    | CompareOp    // ==, !=, <, <=, >, >=
    | LogicalOp    // &&, ||

// Order-independent wave intrinsics only
WaveIntrinsic ::=
    | "WaveActiveSum"(Expr)
    | "WaveActiveProduct"(Expr)
    | "WaveActiveMin"(Expr)
    | "WaveActiveMax"(Expr)
    | "WaveActiveAnd"(Expr)
    | "WaveActiveOr"(Expr)
    | "WaveActiveXor"(Expr)
    | "WaveActiveCountBits"(Expr)
    | "WaveGetLaneIndex"()
    | "WaveGetLaneCount"()
    | "WaveIsFirstLane"()
```

### Statements (Restricted for Order-Independence)
```hlsl
Stmt ::=
    | VarDecl
    | Assignment
    | UniformIf         // Condition must be uniform across wave
    | BlockStmt
    | ReturnStmt
    
VarDecl ::= Type Identifier "=" Expr ";"

Assignment ::= Variable "=" Expr ";"

// Uniform control flow: condition must be wave-uniform
UniformIf ::= "if" "(" UniformExpr ")" BlockStmt ("else" BlockStmt)?

BlockStmt ::= "{" Stmt* "}"

ReturnStmt ::= "return" Expr? ";"
```

### Uniform Expressions
```hlsl
// Expressions that evaluate to the same value across all lanes in a wave
UniformExpr ::=
    | Literal
    | UniformVariable
    | BinaryOp(UniformExpr, UniformExpr)
    | UnaryOp(UniformExpr)
    | WaveUniformIntrinsic
    
WaveUniformIntrinsic ::=
    | "WaveGetLaneCount"()
    | "WaveActiveAllEqual"(Expr)
    | "WaveActiveAllTrue"(Expr)
    | "WaveActiveAnyTrue"(Expr)
```

### Function Definition
```hlsl
Function ::= Type Identifier "(" ParamList? ")" BlockStmt

ParamList ::= Param ("," Param)*
Param ::= Type Identifier
```

### Complete Program
```hlsl
Program ::= Function*

// Entry point must be order-independent compute shader
EntryPoint ::= "[numthreads(X, Y, Z)]" "void" "main" "(" ")" BlockStmt
```

## Order-Independence Validation Rules

### Rule 1: No Racing Memory Access
- No unsynchronized writes to shared memory
- No data races between threads
- Buffer/texture access must be read-only or have non-overlapping write patterns

### Rule 2: Commutative Reductions Only
- Wave operations must be on commutative/associative operations
- `WaveActiveSum`, `WaveActiveProduct`, `WaveActiveMin`, `WaveActiveMax` are allowed
- `WavePrefixSum`, `WaveReadLaneAt` are NOT allowed (order-dependent)

### Rule 3: Uniform Control Flow
- Branch conditions must be uniform across waves OR
- All execution paths must reconverge before wave operations

### Rule 4: Deterministic Expressions
- No operations with undefined behavior
- No reliance on lane execution order
- No non-deterministic functions (e.g., random number generation)

### Rule 5: Pure Functions Only
- Functions cannot have side effects
- No global state modification
- No I/O operations

## Forbidden Constructs

The following HLSL features are explicitly forbidden in MiniHLSL:

### Control Flow
- `for`/`while` loops with wave-divergent conditions
- `break`/`continue` statements
- `switch` with wave-divergent conditions
- Non-uniform `if` statements affecting wave operations

### Memory Operations
- Atomic operations (order-dependent)
- Barriers/synchronization primitives
- Texture/buffer writes in divergent paths
- Shared memory with potential races

### Wave Operations
- Prefix/scan operations (`WavePrefixSum`, `WavePrefixProduct`)
- Lane-specific reads (`WaveReadLaneAt`, `WaveReadFirstLane`)
- Ballot operations with order-dependent usage
- Multi-prefix operations

### Data Types
- Structured buffers with mutable state
- Pointer arithmetic
- Dynamic arrays
- User-defined structs with reference semantics

## Example Valid MiniHLSL Program

```hlsl
[numthreads(32, 1, 1)]
void main() {
    uint laneIndex = WaveGetLaneIndex();
    
    // Order-independent computation
    float value = float(laneIndex * laneIndex);
    
    // Commutative reduction - order-independent
    float sum = WaveActiveSum(value);
    
    // Uniform condition across wave
    if (WaveGetLaneCount() == 32) {
        float average = sum / 32.0f;
        
        // Order-independent comparison
        bool isAboveAverage = value > average;
        uint count = WaveActiveCountBits(isAboveAverage);
    }
}
```

## Example Invalid MiniHLSL Programs

```hlsl
// INVALID: Non-uniform control flow before wave operation
[numthreads(32, 1, 1)]
void main() {
    uint lane = WaveGetLaneIndex();
    if (lane % 2 == 0) {  // Wave-divergent condition
        float sum = WaveActiveSum(1.0f);  // ERROR: Not all lanes participate
    }
}

// INVALID: Order-dependent wave operation
[numthreads(32, 1, 1)]
void main() {
    uint lane = WaveGetLaneIndex();
    float prefix = WavePrefixSum(1.0f);  // ERROR: Prefix operations are order-dependent
}

// INVALID: Loop with wave-divergent condition
[numthreads(32, 1, 1)]
void main() {
    uint lane = WaveGetLaneIndex();
    for (int i = 0; i < lane; ++i) {  // ERROR: Loop count varies per lane
        // Loop body
    }
}
```

## Semantic Validation Algorithm

1. **Parse** MiniHLSL source into AST
2. **Type Check** all expressions and statements
3. **Control Flow Analysis** to ensure uniform convergence
4. **Wave Operation Validation** to verify order-independence
5. **Memory Access Analysis** to detect potential races
6. **Order-Independence Verification** through static analysis

## Integration with Fuzzer

MiniHLSL programs can be:
1. **Generated** randomly following grammar rules
2. **Validated** for order-independence before compilation
3. **Tested** through differential compilation
4. **Verified** by comparing results across optimization levels

The fuzzer should reject any programs that violate order-independence constraints and focus testing on the valid MiniHLSL subset.