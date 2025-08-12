# Nested Control Flow Implementation

## Overview

This document describes the implementation of nested control flow generation for the MiniHLSL fuzzer. The goal is to generate complex programs with nested loops, loops inside conditionals, and conditionals inside loops, with proper handling of break/continue statements and wave operations at multiple nesting levels.

## Design Principles

1. **Recursive Generation**: Use natural recursion to handle arbitrary nesting
2. **Context Propagation**: Pass nesting context down the generation tree
3. **Flexible Wave Op Placement**: Allow wave operations at any nesting level
4. **Correct Break/Continue Semantics**: Ensure break/continue only appear where legal

## Key Data Structures

### NestingContext

```cpp
struct NestingContext {
    uint32_t currentDepth = 0;          // Current nesting depth
    uint32_t maxDepth = 3;              // Maximum allowed depth
    std::vector<Type> parentTypes;      // Stack of parent control flow types
    std::set<std::string> usedLoopVariables;  // Track used variable names
    bool allowNesting = false;          // Whether further nesting is allowed
    
    // Control probabilities
    float waveOpProbability = 0.7f;     // Probability of wave op at each level
    float nestingProbability = 0.5f;    // Probability of adding nested structure
};
```

## Implementation Details

### 1. Break/Continue Placement Rules

Break and continue statements have specific placement rules in C/C++/HLSL:

#### Valid Placements
- **Directly in loop body**: Always valid
- **Inside if within loop**: Valid - affects enclosing loop
- **Nested arbitrarily deep**: As long as there's an enclosing loop

#### Invalid Placements
- **In if without enclosing loop**: Compilation error
- **At top level**: No loop to break/continue from

#### Implementation Strategy

```cpp
// Check if current context is inside any loop
bool isCurrentlyInLoop(const NestingContext& context) {
    // Check if the immediate parent is a loop
    if (!context.parentTypes.empty()) {
        Type lastParent = context.parentTypes.back();
        return (lastParent == BlockSpec::FOR_LOOP || 
                lastParent == BlockSpec::WHILE_LOOP);
    }
    return false;
}

// Check if anywhere in the parent chain is a loop
bool hasEnclosingLoop(const NestingContext& context) {
    for (const auto& parentType : context.parentTypes) {
        if (parentType == BlockSpec::FOR_LOOP || 
            parentType == BlockSpec::WHILE_LOOP) {
            return true;
        }
    }
    return false;
}
```

### 2. Wave Operation Placement

Wave operations should be able to appear at any nesting level, not just the deepest:

```cpp
std::vector<std::unique_ptr<Statement>> generateNestedBody() {
    std::vector<std::unique_ptr<Statement>> body;
    
    // Decision 1: Add wave operation at current level?
    if (provider.ConsumeFloatingPoint<float>() < spec.nestingContext.waveOpProbability) {
        body.push_back(generateWaveOpWithCondition());
    }
    
    // Decision 2: Add nested control flow?
    if (canNest && provider.ConsumeFloatingPoint<float>() < spec.nestingContext.nestingProbability) {
        auto nestedStatements = generateNestedControlFlow();
        body.insert(body.end(), nestedStatements.begin(), nestedStatements.end());
    }
    
    // Decision 3: Add another wave operation after nesting?
    if (provider.ConsumeBool()) {
        body.push_back(generateWaveOpWithCondition());
    }
    
    // Ensure at least one statement
    if (body.empty()) {
        body.push_back(generateWaveOpWithCondition());
    }
    
    return body;
}
```

### 3. Variable Naming Strategy

To avoid variable name conflicts in nested loops:

```cpp
std::string generateLoopVariable(const NestingContext& context, 
                                BlockSpec::Type loopType,
                                ProgramState& state) {
    std::string varName;
    
    if (loopType == BlockSpec::FOR_LOOP) {
        // For nested for loops: i0, i1, i2...
        // Use nextVarIndex to ensure uniqueness
        varName = "i" + std::to_string(state.nextVarIndex++);
    } else {
        // For while loops: counter0, counter1...
        varName = "counter" + std::to_string(state.nextVarIndex++);
    }
    
    state.declaredVariables.insert(varName);
    context.usedLoopVariables.insert(varName);
    return varName;
}
```

### 4. Nested Block Generation

The main logic for creating nested blocks:

```cpp
BlockSpec createNestedBlockSpec(const BlockSpec& parentSpec,
                               ProgramState& state,
                               FuzzedDataProvider& provider) {
    BlockSpec nestedSpec;
    
    // Copy and update context
    nestedSpec.nestingContext = parentSpec.nestingContext;
    nestedSpec.nestingContext.currentDepth++;
    nestedSpec.nestingContext.parentTypes.push_back(parentSpec.type);
    
    // Check if we can nest further
    if (nestedSpec.nestingContext.currentDepth >= nestedSpec.nestingContext.maxDepth) {
        nestedSpec.nestingContext.allowNesting = false;
    }
    
    // Choose nested control flow type based on parent
    if (parentSpec.type == BlockSpec::FOR_LOOP || 
        parentSpec.type == BlockSpec::WHILE_LOOP) {
        // Inside loop: can nest another loop or if
        int choice = provider.ConsumeIntegralInRange<int>(0, 3);
        switch (choice) {
            case 0: nestedSpec.type = BlockSpec::FOR_LOOP; break;
            case 1: nestedSpec.type = BlockSpec::WHILE_LOOP; break;
            case 2: nestedSpec.type = BlockSpec::IF; break;
            case 3: nestedSpec.type = BlockSpec::IF_ELSE; break;
        }
    } else {
        // Inside if: prefer loops for interesting patterns
        nestedSpec.type = provider.ConsumeBool() ? 
                         BlockSpec::FOR_LOOP : BlockSpec::WHILE_LOOP;
    }
    
    // Handle break/continue based on context
    bool hasEnclosingLoop = hasEnclosingLoop(nestedSpec.nestingContext);
    
    if (nestedSpec.type == BlockSpec::FOR_LOOP || 
        nestedSpec.type == BlockSpec::WHILE_LOOP) {
        // Loops can always have break/continue
        nestedSpec.includeBreak = provider.ConsumeBool();
        nestedSpec.includeContinue = provider.ConsumeBool();
    } else {
        // If statements can only have break/continue if inside a loop
        nestedSpec.includeBreak = hasEnclosingLoop && provider.ConsumeBool();
        nestedSpec.includeContinue = hasEnclosingLoop && provider.ConsumeBool();
    }
    
    return nestedSpec;
}
```

## Example Generated Programs

### 1. Nested Loops with Wave Ops at Multiple Levels

```hlsl
for (int i0 = 0; i0 < 3; i0++) {
    // Wave op at outer loop level
    if (laneId < 16) {
        result = WaveActiveSum(i0);
    }
    
    // Nested loop
    for (int i1 = 0; i1 < 2; i1++) {
        // Wave op at inner loop level
        if (laneId == i0 * 2 + i1) {
            result = WaveActiveMin(i1);
        }
        
        // Break in inner loop
        if (i1 == 1 && laneId > 20) {
            break;
        }
    }
    
    // Another wave op at outer level
    if (laneId >= 16) {
        result = WaveActiveMax(i0);
    }
}
```

### 2. Loop Inside Conditional

```hlsl
if (WaveGetLaneIndex() < 16) {
    // Loop only executes for first 16 lanes
    uint counter0 = 0;
    while (counter0 < 3) {
        if (laneId == counter0) {
            result = WaveActiveSum(counter0);
        }
        counter0 = counter0 + 1;
    }
} else {
    // Different behavior for other lanes
    result = WaveActiveMin(WaveGetLaneIndex());
}
```

### 3. Complex Nesting with Break/Continue

```hlsl
for (int i0 = 0; i0 < 4; i0++) {
    if (i0 % 2 == 0) {
        // Continue in if statement (valid because if is in loop)
        if (laneId > 24) {
            continue;
        }
        
        // Nested loop in if
        for (int i1 = 0; i1 < 2; i1++) {
            result = WaveActiveSum(i0 + i1);
            
            if (result > 100) {
                break;  // Breaks inner loop only
            }
        }
    }
    
    // Wave op at outer loop level
    result = WaveActiveMin(result);
}
```

## Integration Points

### 1. Modify generateForLoop

```cpp
std::unique_ptr<Statement> generateForLoop(const BlockSpec& spec, 
                                          ProgramState& state,
                                          FuzzedDataProvider& provider) {
    // ... setup loop variable and conditions ...
    
    // Generate body with potential nesting
    auto body = generateNestedBody(spec, state, provider);
    
    // Add break/continue if specified
    if (spec.includeBreak && provider.ConsumeBool()) {
        auto breakCondition = createBreakCondition(loopVar, loopCount);
        std::vector<std::unique_ptr<Statement>> breakBody;
        breakBody.push_back(std::make_unique<BreakStmt>());
        
        body.push_back(std::make_unique<IfStmt>(
            std::move(breakCondition),
            std::move(breakBody)
        ));
    }
    
    return std::make_unique<ForStmt>(...);
}
```

### 2. Modify generateIf

```cpp
std::unique_ptr<Statement> generateIf(const BlockSpec& spec,
                                     ProgramState& state,
                                     FuzzedDataProvider& provider) {
    // ... generate condition ...
    
    // Generate then body with potential nesting
    auto thenBody = generateNestedBody(spec, state, provider);
    
    // Add break/continue if specified AND we're in a loop
    if (spec.includeBreak && provider.ConsumeBool()) {
        // This will only be true if hasEnclosingLoop was true
        thenBody.push_back(std::make_unique<BreakStmt>());
    }
    
    // Generate else body if needed
    std::vector<std::unique_ptr<Statement>> elseBody;
    if (spec.type == BlockSpec::IF_ELSE) {
        BlockSpec elseSpec = spec;
        elseSpec.nestingContext.parentTypes.push_back(BlockSpec::IF_ELSE);
        elseBody = generateNestedBody(elseSpec, state, provider);
    }
    
    return std::make_unique<IfStmt>(...);
}
```

## Testing Considerations

1. **Depth Limits**: Test that max depth is respected
2. **Variable Conflicts**: Ensure no duplicate variable names
3. **Break/Continue Validity**: Verify generated code compiles
4. **Wave Op Distribution**: Check wave ops appear at various levels
5. **Trace Complexity**: Monitor trace size with deep nesting

## Future Enhancements

1. **Switch Statements**: Add switch as another nesting option
2. **Do-While Loops**: Include do-while in nesting choices
3. **Weighted Probabilities**: Adjust probabilities based on depth
4. **Pattern Templates**: Pre-defined nesting patterns for specific scenarios
5. **Loop Bounds Relationships**: Make inner loop bounds depend on outer loop variable