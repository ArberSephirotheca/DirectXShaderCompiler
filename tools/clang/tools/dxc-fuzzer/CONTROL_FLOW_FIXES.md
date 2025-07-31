# Control Flow WaitingForResume Fixes

## Summary
When implementing WaitingForResume states for scheduler flexibility, we encountered issues with wave operations executing in incorrect blocks. This document records the fixes made.

## Root Problems Identified

### 1. Wave Operations in Wrong Blocks
**Problem**: WaveActiveSum appearing in BRANCH_THEN blocks instead of LOOP_BODY blocks
**Cause**: When continue/break statements are executed inside if statements, lanes remain in incorrect blocks

### 2. Missing State Checks in Body Execution
**Problem**: Parent control flow statements continue executing subsequent statements after child statements return with WaitingForResume
**Example**: WhileStmt executes WaveActiveSum after IfStmt returns with WaitingForResume

## Fixes Applied

### Fix 1: IfStmt Continue/Break Handling
**Location**: IfStmt::execute() exception handler
```cpp
} catch (const ControlFlowException &e) {
    // Propagate break/continue to enclosing loop - do NOT move to merge block
    lane.executionStack.pop_back();
    tg.popMergePoint(wave.waveId, lane.laneId);
    
    // Clean up then/else blocks - lane will never return to them
    tg.removeThreadFromAllSets(thenBlockId, wave.waveId, lane.laneId);
    tg.removeThreadFromNestedBlocks(thenBlockId, wave.waveId, lane.laneId);
    
    if (hasElse && elseBlockId != 0) {
        tg.removeThreadFromAllSets(elseBlockId, wave.waveId, lane.laneId);
        tg.removeThreadFromNestedBlocks(elseBlockId, wave.waveId, lane.laneId);
    }
    
    // Also clean up merge block since we're not going there
    tg.removeThreadFromAllSets(mergeBlockId, wave.waveId, lane.laneId);
    tg.removeThreadFromNestedBlocks(mergeBlockId, wave.waveId, lane.laneId);
    
    throw; // Re-throw to propagate to enclosing control flow
}
```

### Fix 2: ForStmt Continue Handling with WaitingForResume
**Location**: ForStmt::execute() continue case
```cpp
} else if (e.type == ControlFlowException::Continue) {
    // Continue - go to increment phase and skip remaining statements
    std::cout << "DEBUG: ForStmt - Lane " << lane.laneId << " continuing loop" << std::endl;
    
    // Clean up - remove from all nested blocks this lane is abandoning
    if (lane.executionStack[ourStackIndex].loopBodyBlockId != 0) {
        tg.removeThreadFromAllSets(lane.executionStack[ourStackIndex].loopBodyBlockId, wave.waveId, lane.laneId);
        tg.removeThreadFromNestedBlocks(lane.executionStack[ourStackIndex].loopBodyBlockId, wave.waveId, lane.laneId);
    }
    
    tg.moveThreadFromUnknownToParticipating(headerBlockId, wave.waveId, lane.laneId);
    lane.executionStack[ourStackIndex].loopBodyBlockId = 0; // Reset for next iteration
    lane.executionStack[ourStackIndex].phase = LaneContext::ControlFlowPhase::EvaluatingIncrement;
    
    // Set state to WaitingForResume to prevent currentStatement increment
    lane.state = ThreadState::WaitingForResume;
    return; // Exit to prevent currentStatement increment, will resume later
}
```

### Fix 3: Body Execution State Checks
**Required in**: WhileStmt, DoWhileStmt, ForStmt body execution
```cpp
case ExecutingBody:
    for (size_t i = ourEntry.statementIndex; i < body_.size(); i++) {
        lane.executionStack[ourStackIndex].statementIndex = i;
        body_[i]->execute(lane, wave, tg);
        
        // Check if child statement needs to resume
        if (lane.state != ThreadState::Ready) {
            return; // Don't continue to next statement
        }
        
        if (lane.hasReturned) {
            // Pop our entry and return from loop
            lane.executionStack.pop_back();
            tg.popMergePoint(wave.waveId, lane.laneId);
            return;
        }
    }
    // Body completed, transition to next phase
```

### Fix 4: Missing Returns After Phase Transitions
**Required in**: All phase transitions that set WaitingForResume
```cpp
// Example: WhileStmt condition fails
if (!shouldContinue) {
    // Lane is exiting loop
    lane.executionStack[ourStackIndex].phase = LaneContext::ControlFlowPhase::Reconverging;
    
    // MUST have these two lines:
    lane.state = ThreadState::WaitingForResume;
    return; // Exit to allow proper phase transition
}
```

## Key Principles

1. **Always check lane state** after executing child statements
2. **Always return** after setting WaitingForResume
3. **Clean up blocks properly** when handling control flow exceptions
4. **Move lanes to correct blocks** before executing statements
5. **Don't assume block assignments** persist across WaitingForResume

## Testing Notes

These fixes were tested with:
- `/home/t-zheychen/dxc_workspace/DirectXShaderCompiler/tools/clang/tools/dxc-fuzzer/examples/control_flow_while_continue.hlsl`
- Nested control flow with continue statements
- Wave operations in various block contexts