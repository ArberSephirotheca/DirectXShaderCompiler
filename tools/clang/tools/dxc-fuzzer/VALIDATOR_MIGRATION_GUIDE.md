# MiniHLSL Validator Migration Guide

## Overview

This document explains the migration from the original MiniHLSLValidator to MiniHLSLValidatorV2, which implements the new **deterministic control flow framework** established in `OrderIndependenceProof.lean`.

## Key Changes

### 🔄 **Paradigm Shift: Uniform → Deterministic**

| **Old Paradigm (V1)**  | **New Paradigm (V2)** |
|------------------------|------------------------|
| Uniform control flow   | Deterministic control flow |
| All lanes execute identically | Lanes can diverge but conditions are compile-time deterministic |
| Limited to uniform constructs | Supports non-uniform but deterministic programs |
| Based on uniformity analysis | Based on compile-time determinism analysis |

### 🆕 **Core Principle**

**V2 Core Principle**: *"Deterministic control flow guarantees order independence"*

This breakthrough enables verification of **non-uniform GPU programs** while maintaining order independence guarantees.

## Framework Comparison

### **MiniHLSLValidator (V1) - DEPRECATED**

```cpp
// Old uniform-based validation
class UniformityAnalyzer {
    bool isUniform(const Expression* expr) const;
    void markUniform(const std::string& varName);
    void markDivergent(const std::string& varName);
};

// Limited to uniform constructs
enum class ValidationError {
    NonUniformBranch,      // Forbidden
    WaveDivergentLoop,     // Forbidden
    UniformityViolation    // Core constraint
};
```

**Limitations:**
- ❌ Only uniform control flow allowed
- ❌ No support for lane-dependent branching
- ❌ Gap between theory and practical GPU programming
- ❌ Based on incomplete formal foundation

### **MiniHLSLValidatorV2 (Current) - RECOMMENDED**

```cpp
// New deterministic-based validation
class DeterministicExpressionAnalyzer {
    bool isCompileTimeDeterministic(const Expression* expr) const;
    ExpressionKind classifyExpression(const Expression* expr) const;
    ValidationResult validateDeterministicExpression(const Expression* expr) const;
};

// Support for deterministic constructs
enum class ValidationError {
    NonDeterministicCondition,        // Must be compile-time deterministic
    InvalidDeterministicExpression,   // Expression not deterministic
    MixedDeterministicContext        // Mixing deterministic/non-deterministic
};
```

**Advantages:**
- ✅ Supports non-uniform but deterministic control flow
- ✅ Based on complete formal proof framework
- ✅ Enables realistic GPU programming patterns
- ✅ Mathematically sound and practically applicable

## Technical Implementation Differences

### **Control Flow Analysis**

**V1 (Uniform-based):**
```cpp
struct ControlFlowState {
    bool isConverged = true;        // All lanes same path
    bool hasUniformBranch = true;   // Branch condition uniform
    int divergenceLevel = 0;        // Tracks divergence
};

bool isUniformCondition(const Expression* condition) const;
```

**V2 (Deterministic-based):**
```cpp
struct ControlFlowState {
    bool isDeterministic = true;           // Control flow is deterministic
    bool hasDeterministicConditions = true; // Conditions are deterministic
    int deterministicNestingLevel = 0;    // Nesting of deterministic constructs
};

bool isCompileTimeDeterministic(const Expression* expr) const;
```

### **Expression Classification**

**V1:** Uniform vs. Divergent
```cpp
// Limited classification
bool isUniform(const Expression* expr) const;
bool isDivergent(const Expression* expr) const;
```

**V2:** Deterministic Type System
```cpp
enum class ExpressionKind {
    Literal,                    // Constants: 42, 3.14f, true
    LaneIndex,                  // WaveGetLaneIndex()
    WaveIndex,                  // Current wave ID
    ThreadIndex,                // Global thread index
    WaveProperty,               // WaveGetLaneCount(), etc.
    Arithmetic,                 // +, -, *, / of deterministic expressions
    Comparison,                 // <, >, ==, != of deterministic expressions
    NonDeterministic            // Everything else
};
```

### **Memory Safety Constraints**

**V1:** Basic race condition detection
```cpp
bool hasDataRace(const MemoryAccess& access1, const MemoryAccess& access2) const;
```

**V2:** Formal safety constraints from proof
```cpp
// Implements safety constraints from OrderIndependenceProof.lean
bool hasDisjointWrites() const;           // From formal proof
bool hasOnlyCommutativeOperations() const; // From formal proof
bool hasMemoryRaceCondition(const MemoryOperation& op1, const MemoryOperation& op2) const;
```

## Migration Steps

### **For New Projects**

```cpp
// Use V2 directly
#include "MiniHLSLValidatorV2.h"

minihlsl::MiniHLSLValidatorV2 validator;
minihlsl::ValidationResult result = validator.validateProgram(program);

if (result.isValid) {
    // Program satisfies deterministic control flow requirements
    // and is guaranteed to be order-independent
}
```

### **For Existing Projects**

1. **Immediate:** Continue using V1 for backward compatibility
2. **Short-term:** Test with V2 to understand new capabilities
3. **Long-term:** Migrate to V2 for advanced deterministic features

```cpp
// Transition approach
#include "MiniHLSLValidator.h"     // V1 - for compatibility
#include "MiniHLSLValidatorV2.h"   // V2 - for new features

// Option 1: Dual validation during transition
minihlsl::MiniHLSLValidator validatorV1;
minihlsl::MiniHLSLValidatorV2 validatorV2;

auto resultV1 = validatorV1.validateProgram(program);
auto resultV2 = validatorV2.validateProgram(program);

// Option 2: Direct migration to V2
minihlsl::MiniHLSLValidatorV2 validator;
auto result = validator.validateProgram(program);
```

## Examples

### **V1 Example: Uniform-Only**

```hlsl
// V1: Only uniform control flow allowed
[numthreads(64, 1, 1)]
void CSMain(uint3 id : SV_DispatchThreadID) {
    float value = input[id.x];
    
    // ERROR in V1: Lane-dependent branching forbidden
    // if (id.x < 32) { ... }  // NOT ALLOWED
    
    // OK in V1: Uniform condition
    if (uniformCondition) {
        float result = WaveActiveSum(value);
        output[0] = result;
    }
}
```

### **V2 Example: Deterministic Control Flow**

```hlsl
// V2: Deterministic control flow allowed
[numthreads(64, 1, 1)]
void CSMain(uint3 id : SV_DispatchThreadID) {
    float value = input[id.x];
    
    // OK in V2: Compile-time deterministic condition
    if (id.x < 32) {  // Lane index < constant = deterministic
        groupshared[id.x] = value;  // Disjoint writes
    }
    
    GroupMemoryBarrierWithGroupSync();
    
    // OK in V2: Wave operation with deterministic condition
    if (WaveGetLaneIndex() == 0) {  // Deterministic per lane
        float waveSum = WaveActiveSum(value);
        output[id.x / WaveGetLaneCount()] = waveSum;
    }
}
```

## Formal Verification Integration

### **V2 Advantage: Direct Proof Mapping**

```cpp
// V2 provides direct mapping to formal proof concepts
struct FormalProofMapping {
    bool deterministicControlFlow;    // Maps to hasDetministicControlFlow
    bool orderIndependentOperations;  // Maps to operation constraints
    bool memorySafetyConstraints;     // Maps to hasDisjointWrites & hasOnlyCommutativeOps
    bool programOrderIndependence;    // Overall conclusion
};

FormalProofMapping mapping = mapToFormalProof(result, program);
```

### **Proof Alignment Report**

```cpp
std::string report = validator.generateFormalProofAlignment(program);
// Generates detailed report showing how program aligns with
// deterministic_programs_are_order_independent theorem
```

## Performance and Capabilities

| **Aspect** | **V1 (Uniform)** | **V2 (Deterministic)** |
|------------|------------------|------------------------|
| **Supported Programs** | Uniform only | Uniform + Deterministic |
| **Theoretical Foundation** | Incomplete | Complete formal proof |
| **Practical Applicability** | Limited | High |
| **GPU Programming Patterns** | Basic | Realistic |
| **Verification Guarantees** | Partial | Complete |
| **Future Extensibility** | Limited | High |

## Recommendation

### **For All New Development: Use V2**

The V2 validator provides:
- ✅ **Complete theoretical foundation** based on formal proof
- ✅ **Expanded capabilities** supporting realistic GPU patterns
- ✅ **Future-proof architecture** designed for extensibility
- ✅ **Production-ready quality** with systematic validation

### **Migration Timeline**

- **Immediate**: V2 available for new projects
- **Short-term (1-3 months)**: Test existing projects with V2
- **Long-term (3-6 months)**: Complete migration to V2
- **Future**: V1 deprecated, V2 becomes standard

## Support

- **V1 Support**: Maintenance mode only, critical bug fixes
- **V2 Support**: Active development, new features, optimizations
- **Documentation**: Comprehensive documentation for V2 framework
- **Migration Assistance**: Detailed guides and examples provided

The migration to V2 represents a **fundamental advancement** in GPU program verification, moving from limited uniform constructs to a complete deterministic control flow framework with formal mathematical foundations.