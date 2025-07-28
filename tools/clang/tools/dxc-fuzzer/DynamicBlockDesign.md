# Dynamic Block Design and Rules for MiniHLSL Interpreter

## Overview

The MiniHLSL Interpreter implements a dynamic execution block system for SIMT (Single Instruction, Multiple Thread) control flow management. This system handles divergent execution paths in GPU shader code, particularly for control flow structures like if-statements, loops, and switch statements.

## Core Concepts

### Dynamic Execution Blocks (`DynamicExecutionBlock`)

Dynamic execution blocks represent groups of threads (lanes) that are executing the same sequence of instructions in lockstep. Each block is identified by:

- **Block ID**: Unique identifier for the block
- **Block Identity**: Composite key based on execution path, control flow structure, and merge stack
- **Participating Lanes**: Set of lanes currently executing in this block (organized by wave)
- **Program Point**: Current execution location within the block

### Block Types (`BlockType`)

The system supports several types of blocks for different control flow structures:

```cpp
enum class BlockType {
    REGULAR,        // Regular sequential block
    BRANCH_THEN,    // Then branch of if statement  
    BRANCH_ELSE,    // Else branch of if statement
    MERGE,          // Merge/reconvergence point after divergent control flow
    LOOP_HEADER,    // Loop header/condition check
    LOOP_BODY,      // Loop body iteration
    LOOP_EXIT,      // Loop exit/merge point
    SWITCH_CASE,    // Switch case block
    SWITCH_DEFAULT  // Switch default block
};
```

### Block Identity (`BlockIdentity`)

Each block has a unique identity determined by:

- **Source Statement**: The AST statement that created this block
- **Block Type**: The type of control flow block
- **Condition Value**: Which branch was taken (for conditional blocks)
- **Parent Block ID**: The block this diverged from
- **Execution Path**: Sequence of statements executed to reach this point
- **Merge Stack**: Stack of convergence points for proper reconvergence

## Key Design Rules

### 1. Block Deduplication and Reuse

**Rule**: Blocks with identical `BlockIdentity` are deduplicated and reused.

```cpp
// Find existing block or create new one
uint32_t findOrCreateBlockForPath(const BlockIdentity& identity, 
                                  const std::map<WaveId, std::set<LaneId>>& unknownLanes);
```

**Implementation Details**:
- Uses `identityToBlockId` map for O(1) lookup
- Merge blocks (`MERGE`, `LOOP_EXIT`) ignore execution paths in comparison
- Divergent blocks require exact execution path match

### 2. Proactive Block Creation

**Rule**: Control flow blocks are created proactively before lanes reach them.

**For If Statements**:
```cpp
std::tuple<uint32_t, uint32_t, uint32_t> createIfBlocks(
    const void* ifStmt, uint32_t parentBlockId,
    const std::vector<MergeStackEntry>& mergeStack, bool hasElse,
    const std::vector<const void*>& executionPath);
```
- Always creates THEN block
- Creates ELSE block only if `hasElse` is true
- Always creates MERGE block for reconvergence

**For Loops**:
```cpp
std::tuple<uint32_t, uint32_t> createLoopBlocks(
    const void* loopStmt, uint32_t parentBlockId,
    const std::vector<MergeStackEntry>& mergeStack,
    const std::vector<const void*>& executionPath);
```
- Creates LOOP_HEADER block for condition checking
- Creates LOOP_EXIT block for reconvergence
- Loop body blocks are created dynamically during execution (unique per iteration)

### 3. Unknown Lane Management

**Rule**: Blocks track "unknown lanes" - threads that might potentially reach the block but haven't decided yet.

**Implementation**:
- Unknown lanes are added during block creation with all potential participants
- Lanes are moved from "unknown" to "participating" when they reach the block
- Lanes are removed from "unknown" when they choose a different path
- Block synchronization waits until all unknown lanes are resolved

### 4. Lane State Transitions

**Rule**: Lanes transition through specific states within blocks:

1. **Unknown**: Lane might reach this block but hasn't decided
2. **Participating**: Lane is actively executing in this block
3. **Arrived**: Lane has reached a synchronization point in the block
4. **Waiting**: Lane is waiting for wave operations to complete

### 5. Unique Loop Iteration Blocks

**Rule**: Each loop iteration creates a unique body block to handle different iteration-specific state.

**Implementation**:
```cpp
// Add current loop iteration to execution path to make each iteration unique
lane.executionPath.push_back(static_cast<const void*>(this));

BlockIdentity iterationBodyIdentity = 
    tg.createBlockIdentity(static_cast<const void*>(this), BlockType::LOOP_BODY, 
                          headerBlockId, currentMergeStack, true, lane.executionPath);
```

### 6. Merge Stack for Reconvergence

**Rule**: The merge stack tracks nested control flow structures to ensure proper reconvergence.

**Structure**:
```cpp
struct MergeStackEntry {
    const void* sourceStatement;        // Statement that created the divergence
    uint32_t parentBlockId;             // Block before divergence  
    std::set<uint32_t> divergentBlockIds; // Blocks that will converge
};
```

**Usage**:
- Pushed when entering divergent control flow
- Used in block identity comparison
- Popped when reconverging at merge points

### 7. Wave-Aware Block Organization

**Rule**: All block operations are organized by wave to maintain SIMT execution model.

**Data Structures**:
```cpp
std::map<WaveId, std::set<LaneId>> participatingLanes_;  // Organized by wave
std::map<WaveId, std::set<LaneId>> unknownLanes_;        // Organized by wave
std::map<WaveId, std::set<LaneId>> arrivedLanes_;        // Organized by wave
```

### 8. Instruction-Level Synchronization

**Rule**: Wave operations within blocks require all participating lanes to reach the same instruction before execution.

**Implementation**:
- Tracks instruction participants per block: `std::map<InstructionIdentity, std::map<WaveId, std::set<LaneId>>>`
- Synchronization waits for all expected participants before executing wave operations
- Instructions are identified by AST pointer and instruction type

### 9. Parent-Child Block Relationships

**Rule**: Nested control flow creates parent-child relationships between blocks.

**Usage**:
- Loop iteration blocks have loop header as parent
- Branch blocks have the pre-branch block as parent  
- Switch case blocks have the pre-switch block as parent
- Used for cleanup operations (e.g., `removeThreadFromNestedBlocks`)

### 10. Execution Path Tracking

**Rule**: Each lane maintains an execution path to differentiate between identical control flow structures at different nesting levels or iterations.

**Implementation**:
```cpp
std::vector<const void*> executionPath; // In LaneContext
```
- Updated when entering new control flow structures
- Used in block identity to create unique blocks per execution context
- Essential for loop iteration uniqueness

## Critical Implementation Notes

### Block Creation Timing
- Blocks are created **before** lanes reach them (proactive creation)
- All potential lanes are initially marked as "unknown"
- Lanes transition to "participating" when they actually reach the block

### Synchronization Points
- Wave operations require all participating lanes in a block to reach the same instruction
- Unknown lanes must be resolved before wave operations can proceed
- Block convergence is checked before allowing instruction execution

### Cleanup Operations
- When lanes exit control structures, they must be removed from nested blocks
- Special handling for loop exit to avoid removing lanes from merge blocks they should enter
- Careful management of unknown lane sets during control flow transitions

### Debug Support
- Extensive debug logging controlled by `ENABLE_INTERPRETER_DEBUG` flag
- Block state visualization with `printBlockDetails()` method
- Lane tracking through state transitions for debugging convergence issues

## Usage Examples

### If Statement Block Creation
```cpp
// Creates then, else (if present), and merge blocks
auto [thenBlockId, elseBlockId, mergeBlockId] = 
    tg.createIfBlocks(ifStmt, parentBlockId, mergeStack, hasElse, executionPath);
```

### Loop Iteration Block Creation  
```cpp
// Each iteration gets unique block based on execution path
lane.executionPath.push_back(static_cast<const void*>(this));
BlockIdentity iterationBodyIdentity = 
    tg.createBlockIdentity(this, BlockType::LOOP_BODY, headerBlockId, 
                          currentMergeStack, true, lane.executionPath);
uint32_t iterationBodyBlockId = 
    tg.findOrCreateBlockForPath(iterationBodyIdentity, iterationUnknownLanes);
```

### Lane State Management
```cpp
// Move lane from unknown to participating when it reaches a block
tg.moveThreadFromUnknownToParticipating(blockId, waveId, laneId);

// Remove lane from unknown when it chooses different path
tg.removeThreadFromUnknown(blockId, waveId, laneId);
```

This design ensures correct SIMT execution semantics while handling complex control flow patterns commonly found in GPU shader code.