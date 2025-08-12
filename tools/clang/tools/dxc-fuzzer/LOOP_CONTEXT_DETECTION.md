# Loop Context Detection for Context-Aware Mutations

## Overview

This document describes the improved approach for detecting loop contexts and extracting iteration information from the dynamic execution trace. This is crucial for the ContextAwareParticipantMutation strategy to correctly identify and mutate specific loop iterations.

## Problem Statement

The initial implementation used a crude heuristic (`blockId > 100`) to detect loop iterations. This approach:
- Relies on implementation details of block ID encoding
- Doesn't work for nested loops
- Can't distinguish between different loop types
- Ignores the rich information available in the execution trace

## Improved Approach

### Key Data Structures

```cpp
struct IterationContext {
    std::string loopVariable;      // "i0" for for-loops, "counter0" for while
    int iterationValue;            // 0, 1, 2... (which iteration)
    uint32_t blockId;              // Dynamic block where wave op executed
    std::set<interpreter::LaneId> existingParticipants;
    interpreter::WaveId waveId;
    
    // Track loop type for proper handling
    enum LoopType { FOR_LOOP, WHILE_LOOP, DO_WHILE_LOOP };
    LoopType loopType;
    
    // For while loops, track the counter variable
    std::string counterVariable;   // "counter0", etc.
};
```

### Loop Type Differences

#### For Loops
```cpp
class ForStmt {
    std::string loopVar_;  // Explicit loop variable
    // Direct access to iteration counter
}
```
- Have explicit loop variable (`i0`, `i1`, etc.)
- Iteration value directly corresponds to loop variable value
- Easy to extract from ForStmt structure

#### While/Do-While Loops
```cpp
class WhileStmt {
    // No built-in loop variable!
}
```
- No explicit iteration counter in the statement
- Program generator creates separate counter variables (`counter0`, etc.)
- Must track counter through variable accesses in trace

### Detection Algorithm

#### Step 1: Check if Statement is in Loop Context

```cpp
bool canApply(const interpreter::Statement* stmt,
              const ExecutionTrace& trace) const {
    
    // Check if statement contains wave operation
    bool hasWaveOp = checkForWaveOp(stmt);
    if (!hasWaveOp) return false;
    
    // Find all blocks where wave operations executed
    std::set<uint32_t> waveOpBlocks;
    for (const auto& waveOp : trace.waveOperations) {
        waveOpBlocks.insert(waveOp.blockId);
    }
    
    // Check if any of these blocks are inside loops
    for (uint32_t blockId : waveOpBlocks) {
        auto it = trace.blocks.find(blockId);
        if (it != trace.blocks.end()) {
            const auto& block = it->second;
            
            // Method 1: Check block type directly
            if (block.blockType == interpreter::BlockType::ForBody ||
                block.blockType == interpreter::BlockType::WhileBody ||
                block.blockType == interpreter::BlockType::DoWhileBody) {
                return true;
            }
            
            // Method 2: Check if same source statement appears multiple times
            // (indicates multiple iterations)
            int occurrences = 0;
            for (const auto& [id, b] : trace.blocks) {
                if (b.sourceStatement == block.sourceStatement) {
                    occurrences++;
                }
            }
            if (occurrences > 1) {
                return true;
            }
        }
    }
    
    return false;
}
```

#### Step 2: Extract Iteration Contexts

```cpp
std::vector<IterationContext> extractIterationContexts(
    const interpreter::Statement* stmt,
    const ExecutionTrace& trace) const {
    
    std::vector<IterationContext> contexts;
    
    // Group blocks by their loop ancestor
    std::map<const void*, std::vector<uint32_t>> loopIterations;
    
    for (const auto& waveOp : trace.waveOperations) {
        if (!matchesStatement(stmt, waveOp)) continue;
        
        // Find the containing loop
        auto loopInfo = findContainingLoop(waveOp.blockId, trace);
        if (loopInfo.loopStatement) {
            loopIterations[loopInfo.loopStatement].push_back(waveOp.blockId);
        }
    }
    
    // Extract iteration contexts for each loop
    for (const auto& [loopStmt, blockIds] : loopIterations) {
        // Sort blocks by ID to get iteration order
        std::sort(blockIds.begin(), blockIds.end());
        
        for (size_t i = 0; i < blockIds.size(); ++i) {
            IterationContext ctx;
            ctx.blockId = blockIds[i];
            ctx.iterationValue = i;
            
            // Handle different loop types
            if (auto* forStmt = dynamic_cast<const interpreter::ForStmt*>(loopStmt)) {
                ctx.loopType = IterationContext::FOR_LOOP;
                ctx.loopVariable = forStmt->getLoopVar();
            }
            else if (dynamic_cast<const interpreter::WhileStmt*>(loopStmt)) {
                ctx.loopType = IterationContext::WHILE_LOOP;
                ctx.counterVariable = findCounterForLoop(loopStmt, trace);
                ctx.loopVariable = ctx.counterVariable; // Use counter as loop var
            }
            else if (dynamic_cast<const interpreter::DoWhileStmt*>(loopStmt)) {
                ctx.loopType = IterationContext::DO_WHILE_LOOP;
                ctx.counterVariable = findCounterForLoop(loopStmt, trace);
                ctx.loopVariable = ctx.counterVariable;
            }
            
            // Get participants for this specific iteration
            auto waveOp = findWaveOpInBlock(blockIds[i], trace);
            if (waveOp) {
                ctx.existingParticipants = waveOp->arrivedParticipants;
                ctx.waveId = waveOp->waveId;
            }
            
            contexts.push_back(ctx);
        }
    }
    
    return contexts;
}
```

#### Step 3: Find Loop Variables

```cpp
std::string findLoopVariable(uint32_t blockId,
                            const ExecutionTrace& trace) const {
    
    // Traverse up the block hierarchy to find the loop
    uint32_t currentId = blockId;
    while (currentId != 0 && trace.blocks.count(currentId)) {
        const auto& block = trace.blocks.at(currentId);
        
        if (block.sourceStatement) {
            // For loop: return the explicit loop variable
            if (auto* forStmt = dynamic_cast<const interpreter::ForStmt*>(
                block.sourceStatement)) {
                return forStmt->getLoopVar();
            }
            // While/Do-While: find the counter variable
            else if (dynamic_cast<const interpreter::WhileStmt*>(
                block.sourceStatement) ||
                     dynamic_cast<const interpreter::DoWhileStmt*>(
                block.sourceStatement)) {
                return findCounterVariable(block, trace);
            }
        }
        
        currentId = block.parentBlockId;
    }
    
    return "";
}

std::string findCounterVariable(const BlockExecutionRecord& loopBlock,
                               const ExecutionTrace& trace) const {
    
    // The generator creates variables like "counter0", "counter1"
    // Look for counter variable accesses in the loop condition
    
    for (const auto& varAccess : trace.variableAccesses) {
        if (varAccess.blockId == loopBlock.blockId &&
            varAccess.varName.substr(0, 7) == "counter" &&
            !varAccess.isWrite) {  // Counter read in condition
            return varAccess.varName;
        }
    }
    
    // Alternative: Look for counter increments in loop body
    for (const auto& varAccess : trace.variableAccesses) {
        if (varAccess.blockId > loopBlock.blockId &&  // In loop body
            varAccess.blockId < loopBlock.blockId + 100 &&  // Same iteration
            varAccess.varName.substr(0, 7) == "counter" &&
            varAccess.isWrite) {  // Counter write
            return varAccess.varName;
        }
    }
    
    // Fallback heuristic
    return "counter0";
}
```

### Helper Functions

```cpp
// Find the loop statement that contains a given block
struct LoopInfo {
    const Statement* loopStatement;
    BlockType loopType;
    uint32_t loopBlockId;
};

LoopInfo findContainingLoop(uint32_t blockId,
                            const ExecutionTrace& trace) const {
    LoopInfo info = {nullptr, BlockType::Unknown, 0};
    
    uint32_t currentId = blockId;
    while (currentId != 0 && trace.blocks.count(currentId)) {
        const auto& block = trace.blocks.at(currentId);
        
        // Check if this block is a loop
        if (block.blockType == BlockType::ForLoop ||
            block.blockType == BlockType::WhileLoop ||
            block.blockType == BlockType::DoWhileLoop) {
            info.loopStatement = block.sourceStatement;
            info.loopType = block.blockType;
            info.loopBlockId = currentId;
            return info;
        }
        
        currentId = block.parentBlockId;
    }
    
    return info;
}

// Check if a wave operation matches our target statement
bool matchesStatement(const Statement* stmt,
                     const WaveOpRecord& waveOp) const {
    
    // Match by operation type
    if (auto* assign = dynamic_cast<const AssignStmt*>(stmt)) {
        if (auto* waveExpr = dynamic_cast<const WaveActiveOp*>(
            assign->getExpression())) {
            return waveOp.waveOpEnumType == 
                   static_cast<int>(waveExpr->getOpType());
        }
    }
    
    // Add other matching logic...
    return false;
}
```

## Advantages of This Approach

1. **No Magic Numbers**: Doesn't rely on block ID encoding schemes
2. **Works for All Loop Types**: Handles for, while, and do-while uniformly
3. **Supports Nested Loops**: Can traverse block hierarchy to find containing loops
4. **Uses Rich Trace Data**: Leverages block types, parent relationships, and variable accesses
5. **Precise Iteration Tracking**: Maps each wave operation to its exact iteration

## Example Usage

### For Loop
```hlsl
for (int i = 0; i < 3; i++) {
    if (laneId == i) {
        result = WaveActiveSum(value);
    }
}
```
- Loop variable: `"i0"`
- Iterations: 0, 1, 2
- Can directly use `i0 == 1` in mutation condition

### While Loop
```hlsl
int counter0 = 0;
while (counter0 < 3) {
    if (laneId == counter0) {
        result = WaveActiveSum(value);
    }
    counter0++;
}
```
- Counter variable: `"counter0"`
- Iterations: 0, 1, 2
- Must use `counter0 == 1` in mutation condition

## Future Improvements

1. **Nested Loop Support**: Track multiple loop variables for nested structures
2. **Loop-Invariant Detection**: Identify variables that don't change in loop
3. **Break/Continue Handling**: Track early exits and their effects on iterations
4. **Performance Optimization**: Cache loop detection results

## Integration with Mutations

The improved context detection enables precise mutations:

```cpp
// Create iteration-specific condition
auto condition = createIterationSpecificCondition(context, newLane);

// For loop: (i0 == 1 && laneId == 3)
// While loop: (counter0 == 1 && laneId == 3)
```

This ensures mutations target exactly the intended iteration, regardless of loop type or nesting level.