# State Analysis and Simplification Plan for MiniHLSL Interpreter

## Problem Statement

The codebase has grown complex with many interdependent states that need to be updated/checked per step. This analysis evaluates which states are necessary and proposes a simplification plan.

## Current States in the System

### 1. **Lane-Level States** (`LaneContext`)
```cpp
ThreadState state;                    // Ready, WaitingForWave, WaitingForResume, Completed, Error
bool isActive;                       // Whether lane is executing
bool hasReturned;                    // Lane has executed return statement
bool isResumingFromWaveOp;          // Lane is resuming after wave operation
Value returnValue;                   // Return value if hasReturned=true
std::vector<ControlFlowStackEntry> executionStack;  // Control flow nesting (switch/loop/if)
```

### 2. **Wave Operation Sync Point States** (`WaveOperationSyncPoint`)
```cpp
const void *instruction;             // Which instruction this sync point tracks
uint32_t blockId;                   // Which block it belongs to
std::set<LaneId> expectedParticipants;   // Who should participate
std::set<LaneId> arrivedParticipants;    // Who has arrived
std::map<LaneId, Value> pendingResults;  // Results waiting to be retrieved
bool allParticipantsKnown;          // All unknown lanes resolved for block
bool allParticipantsArrived;        // All expected participants arrived
bool isComplete;                    // Ready to execute
bool hasExecuted;                   // NEW: Whether already executed
```

### 3. **Block-Level States** (`DynamicExecutionBlock`)
```cpp
std::map<WaveId, std::set<LaneId>> unknownLanes_;      // Lanes that might join
std::map<WaveId, std::set<LaneId>> participatingLanes_; // Lanes currently in block
std::map<WaveId, std::set<LaneId>> arrivedLanes_;      // Lanes that have arrived
std::map<WaveId, std::set<LaneId>> waitingLanes_;      // Lanes waiting for operation
std::map<WaveId, bool> waveAllUnknownResolved_;        // No more unknown lanes for wave
```

### 4. **Wave-Level States** (`WaveContext`)
```cpp
std::map<LaneId, uint32_t> laneToCurrentBlock;  // Which block each lane is in
std::map<LaneId, std::pair<const void *, uint32_t>> laneWaitingAtInstruction;  // What each lane waits for
std::map<std::pair<const void *, uint32_t>, WaveOperationSyncPoint> activeSyncPoints;  // Active sync points
```

## **Evaluation of State Necessity**

### **✅ ESSENTIAL STATES**

#### 1. `ThreadState state` - **CRITICAL**
**Reason**: Core scheduling primitive. Must know if lane is Ready, Waiting, or Completed.
**Keep**: Yes, but could simplify enum values.

#### 2. `hasReturned` - **CRITICAL** 
**Reason**: Must track early returns for correct cleanup.
**Keep**: Yes, fundamental for correctness.

#### 3. `executionStack` - **CRITICAL**
**Reason**: Required for nested control flow (switch/loop/if resumption).  
**Keep**: Yes, but could optimize structure.

#### 4. `expectedParticipants` & `arrivedParticipants` - **CRITICAL**
**Reason**: Core wave operation synchronization.
**Keep**: Yes, essential for correctness.

#### 5. `pendingResults` - **CRITICAL**
**Reason**: Must store wave operation results for lane retrieval.
**Keep**: Yes, required for functionality.

### **⚠️ NECESSARY BUT COULD BE SIMPLIFIED**

#### 6. `unknownLanes_` & `waveAllUnknownResolved_` - **NECESSARY BUT COMPLEX**
**Reason**: Tracks which lanes might join blocks (for switch fallthrough, etc.)
**Issue**: Creates complex interdependencies and timing issues
**Suggestion**: Could be simplified by making block membership more explicit

#### 7. `allParticipantsKnown` & `allParticipantsArrived` & `isComplete` - **REDUNDANT**
**Reason**: These are derived states that can be computed from other states
**Issue**: Multiple sources of truth create consistency problems
**Suggestion**: Compute these on-demand instead of storing

#### 8. `laneToCurrentBlock` - **NECESSARY BUT REDUNDANT**
**Reason**: Could be derived from block's `participatingLanes_`
**Issue**: Dual bookkeeping creates consistency issues
**Suggestion**: Use single source of truth

### **❌ QUESTIONABLE/REMOVABLE STATES**

#### 9. `isActive` - **MOSTLY REDUNDANT**
**Reason**: Usually derivable from `state != Completed && !hasReturned`
**Suggestion**: Remove and compute on-demand

#### 10. `isResumingFromWaveOp` - **OPTIMIZATION FLAG**
**Reason**: Performance hint, not correctness-critical
**Issue**: Another state to maintain consistency for
**Suggestion**: Remove and use different mechanism

#### 11. `waitingLanes_` - **PARTIALLY REDUNDANT**
**Reason**: Could be derived from lanes with `state == WaitingForWave`
**Suggestion**: Remove and compute from lane states

#### 12. `hasExecuted` - **BAND-AID FIX**
**Reason**: Fixes symptom of poor sync point lifecycle management
**Issue**: Adds more state complexity
**Suggestion**: Fix root cause instead

## **Recommendations for Simplification**

### **Option 1: Compute Derived States**
```cpp
// Instead of storing these flags:
bool allParticipantsKnown;
bool allParticipantsArrived; 
bool isComplete;

// Compute them:
bool isAllParticipantsKnown() const {
    return getBlock().isWaveAllUnknownResolved(waveId);
}
bool isAllParticipantsArrived() const {
    return arrivedParticipants == expectedParticipants;
}
bool isComplete() const {
    return isAllParticipantsKnown() && isAllParticipantsArrived();
}
```

### **Option 2: Simplify Block Membership**
```cpp
// Instead of complex unknown/participating/arrived/waiting sets:
enum class LaneBlockStatus { Unknown, Participating, Waiting, Left };
std::map<LaneId, LaneBlockStatus> laneStatus;
```

### **Option 3: Event-Driven Updates**
```cpp
// Instead of maintaining multiple consistent states:
class StateManager {
    void onLaneJoinBlock(LaneId lane, BlockId block);
    void onLaneLeaveBlock(LaneId lane, BlockId block);  
    void onLaneReturn(LaneId lane);
    // Automatically maintains all derived states
};
```

## **Most Critical Issues**

1. **Multiple Sources of Truth**: Lane membership tracked in both blocks and waves
2. **Complex State Dependencies**: Changes to one state require updates to 5-10 others
3. **Timing Issues**: States get out of sync during multi-step operations
4. **Band-aid Fixes**: Adding flags like `hasExecuted` instead of fixing root causes

# State Simplification Plan

Based on the analysis, here's a concrete plan to simplify the most problematic states while maintaining correctness.

## **Phase 1: Eliminate Redundant Derived States**

### **Problem**: Multiple Boolean Flags That Can Be Computed
Current problematic pattern:
```cpp
// These are all derivable from other states
syncPoint.allParticipantsKnown = ...;
syncPoint.allParticipantsArrived = ...;  
syncPoint.isComplete = ...;
syncPoint.hasExecuted = ...;  // Band-aid fix
```

### **Solution**: Replace with Computed Properties
```cpp
struct WaveOperationSyncPoint {
    // KEEP: Core data
    const void *instruction;
    uint32_t blockId;
    std::set<LaneId> expectedParticipants;
    std::set<LaneId> arrivedParticipants;
    std::map<LaneId, Value> pendingResults;
    
    // REMOVE: All boolean flags
    // bool allParticipantsKnown;
    // bool allParticipantsArrived;
    // bool isComplete;
    // bool hasExecuted;
    
    // REPLACE WITH: Computed methods
    bool isAllParticipantsArrived() const {
        return arrivedParticipants == expectedParticipants;
    }
    
    bool isAllParticipantsKnown(const ThreadgroupContext& tg, WaveId waveId) const {
        auto blockIt = tg.executionBlocks.find(blockId);
        return blockIt != tg.executionBlocks.end() && 
               blockIt->second.isWaveAllUnknownResolved(waveId);
    }
    
    bool isReadyToExecute(const ThreadgroupContext& tg, WaveId waveId) const {
        return isAllParticipantsKnown(tg, waveId) && 
               isAllParticipantsArrived() && 
               pendingResults.empty();  // Never executed
    }
    
    bool isReadyForCleanup() const {
        return !pendingResults.empty();  // Has executed, check if results consumed
    }
};
```

**Benefits**: 
- Eliminates 4 boolean flags
- No more state consistency issues
- Single source of truth for each computation

## **Phase 2: Unify Lane-Block Membership Tracking**

### **Problem**: Dual Bookkeeping
```cpp
// Block side
std::map<WaveId, std::set<LaneId>> participatingLanes_;
std::map<WaveId, std::set<LaneId>> unknownLanes_;
std::map<WaveId, std::set<LaneId>> arrivedLanes_;
std::map<WaveId, std::set<LaneId>> waitingLanes_;

// Wave side  
std::map<LaneId, uint32_t> laneToCurrentBlock;
```

### **Solution**: Single Membership Registry
```cpp
enum class LaneBlockStatus {
    Unknown,        // Lane might join this block (switch fallthrough)
    Participating,  // Lane is actively in this block
    WaitingForWave, // Lane is waiting for wave operation in this block
    Left           // Lane has left this block (returned/moved)
};

class BlockMembershipRegistry {
private:
    // Single source of truth: (waveId, laneId, blockId) -> status
    std::map<std::tuple<WaveId, LaneId, uint32_t>, LaneBlockStatus> membership_;
    
public:
    void setLaneStatus(WaveId wave, LaneId lane, uint32_t block, LaneBlockStatus status);
    LaneBlockStatus getLaneStatus(WaveId wave, LaneId lane, uint32_t block) const;
    uint32_t getCurrentBlock(WaveId wave, LaneId lane) const;
    
    // Derived queries (computed on-demand)
    std::set<LaneId> getParticipatingLanes(WaveId wave, uint32_t block) const;
    std::set<LaneId> getUnknownLanes(WaveId wave, uint32_t block) const;
    bool isWaveAllUnknownResolved(WaveId wave, uint32_t block) const;
};
```

**Benefits**:
- Single source of truth for all lane-block relationships
- No more sync issues between block/wave tracking
- Cleaner query interface

## **Phase 3: Simplify Wave Operation Lifecycle**

### **Problem**: Complex Multi-State Lifecycle
Current flow is confusing:
```
Create sync point → Set flags → Check flags → Execute → Set more flags → Check again → Cleanup
```

### **Solution**: Clear State Machine
```cpp
enum class SyncPointPhase {
    WaitingForParticipants,  // Collecting lanes
    ReadyToExecute,          // All participants known and arrived
    Executed,                // Wave operation completed, results available
    Consumed                 // All results retrieved, ready for cleanup
};

class WaveOperationSyncPoint {
private:
    SyncPointPhase phase_ = SyncPointPhase::WaitingForParticipants;
    
public:
    SyncPointPhase getPhase() const { return phase_; }
    
    void addParticipant(LaneId lane) {
        if (phase_ != SyncPointPhase::WaitingForParticipants) return;
        arrivedParticipants.insert(lane);
        expectedParticipants.insert(lane);
        // Phase transition handled by update logic
    }
    
    void execute(const std::vector<Value>& results) {
        assert(phase_ == SyncPointPhase::ReadyToExecute);
        for (LaneId lane : arrivedParticipants) {
            pendingResults[lane] = results[...];
        }
        phase_ = SyncPointPhase::Executed;
    }
    
    Value retrieveResult(LaneId lane) {
        assert(phase_ == SyncPointPhase::Executed);
        auto it = pendingResults.find(lane);
        if (it != pendingResults.end()) {
            Value result = it->second;
            pendingResults.erase(it);
            if (pendingResults.empty()) {
                phase_ = SyncPointPhase::Consumed;
            }
            return result;
        }
        throw std::runtime_error("No result for lane");
    }
    
    bool shouldExecute(const ThreadgroupContext& tg, WaveId waveId) const {
        return phase_ == SyncPointPhase::WaitingForParticipants &&
               isAllParticipantsKnown(tg, waveId) && isAllParticipantsArrived();
    }
    
    bool shouldCleanup() const {
        return phase_ == SyncPointPhase::Consumed;
    }
};
```

**Benefits**:
- Clear, linear state progression
- No ambiguous states
- Easy to reason about and debug

## **Phase 4: Event-Driven State Updates**

### **Problem**: Manual State Consistency Management
Currently, every operation requires updating 5-10 different state variables manually.

### **Solution**: Centralized State Manager
```cpp
class InterpreterStateManager {
private:
    BlockMembershipRegistry membership_;
    std::map<std::pair<const void*, uint32_t>, WaveOperationSyncPoint> syncPoints_;
    
public:
    // High-level operations that maintain all state consistency
    void onLaneJoinBlock(WaveId wave, LaneId lane, uint32_t block) {
        membership_.setLaneStatus(wave, lane, block, LaneBlockStatus::Participating);
        // Automatically update any relevant sync points
        updateSyncPointsForBlock(wave, block);
    }
    
    void onLaneReturn(WaveId wave, LaneId lane) {
        // Remove from all blocks
        uint32_t currentBlock = membership_.getCurrentBlock(wave, lane);
        if (currentBlock != 0) {
            membership_.setLaneStatus(wave, lane, currentBlock, LaneBlockStatus::Left);
        }
        
        // Remove from all sync points
        for (auto& [key, syncPoint] : syncPoints_) {
            syncPoint.removeParticipant(lane);
            if (syncPoint.getExpectedParticipants().empty()) {
                markForCleanup(key);
            }
        }
    }
    
    void onWaveOperationStart(WaveId wave, LaneId lane, const void* instruction, uint32_t block) {
        membership_.setLaneStatus(wave, lane, block, LaneBlockStatus::WaitingForWave);
        auto& syncPoint = getOrCreateSyncPoint({instruction, block});
        syncPoint.addParticipant(lane);
    }
    
private:
    void updateSyncPointsForBlock(WaveId wave, uint32_t block);
    void markForCleanup(const std::pair<const void*, uint32_t>& key);
};
```

## **Implementation Order**

1. **Start with Phase 1** (eliminate boolean flags) - Low risk, high impact
2. **Then Phase 3** (sync point state machine) - Fixes the core issue
3. **Then Phase 2** (unify membership) - Requires more changes but removes complexity
4. **Finally Phase 4** (event-driven) - Optional, but makes system much more maintainable

## **Expected Benefits**

- **Reduce state variables by ~60%** (from ~15 to ~6 core variables)
- **Eliminate state consistency bugs** (single source of truth)
- **Make debugging much easier** (clear state machine phases)
- **Fix the original switch/return issue** without band-aid flags

## Current Status

Planning to implement Phase 1 first - eliminating the redundant boolean flags from `WaveOperationSyncPoint` and replacing them with computed methods, as it's the lowest risk and highest impact change.