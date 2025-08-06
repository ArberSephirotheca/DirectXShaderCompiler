// Original exception-based execute() methods for reference
// These methods handle non-uniform wave operations and show the complete semantics
// that our Result-based execute_result() methods should preserve

// =============================================================================
// IfStmt::execute() - Original Exception-Based Implementation
// =============================================================================

void IfStmt::execute(LaneContext &lane, WaveContext &wave, ThreadgroupContext &tg) {
  if (!lane.isActive)
    return;

  // Find our entry in the execution stack (if any)
  int ourStackIndex = findStackIndex(lane);
  bool isResuming = (ourStackIndex >= 0);

  if (!isResuming) {
    // Starting fresh - push initial state for condition evaluation
    lane.executionStack.emplace_back(
        static_cast<const void *>(this),
        LaneContext::ControlFlowPhase::EvaluatingCondition);
    ourStackIndex = lane.executionStack.size() - 1;
  }

  bool hasElse = !elseBlock_.empty();
  uint32_t parentBlockId = tg.getCurrentBlock(wave.waveId, lane.laneId);
  auto &ourEntry = lane.executionStack[ourStackIndex];

  try {
    switch (ourEntry.phase) {
    case LaneContext::ControlFlowPhase::EvaluatingCondition: {
      evaluateConditionAndSetup(lane, wave, tg, ourStackIndex, parentBlockId, hasElse);
      if (!isProtectedState(lane.state)) {
        lane.state = ThreadState::WaitingForResume;
      }
      return;
    }

    case LaneContext::ControlFlowPhase::ExecutingThenBlock: {
      executeThenBranch(lane, wave, tg, ourStackIndex);
      if (lane.hasReturned) {
        return;
      }
      if (!isProtectedState(lane.state)) {
        lane.state = ThreadState::WaitingForResume;
      }
      return;
    }

    case LaneContext::ControlFlowPhase::ExecutingElseBlock: {
      executeElseBranch(lane, wave, tg, ourStackIndex);
      if (lane.hasReturned) {
        return;
      }
      if (!isProtectedState(lane.state)) {
        lane.state = ThreadState::WaitingForResume;
      }
      return;
    }

    case LaneContext::ControlFlowPhase::Reconverging: {
      performReconvergence(lane, wave, tg, ourStackIndex, hasElse);
      return;
    }
    }

  } catch (const WaveOperationWaitException &) {
    throw; // Re-throw to pause parent control flow statements
  } catch (const ControlFlowException &e) {
    // Propagate break/continue to enclosing loop - do NOT move to merge block
    
    // Get block IDs before popping the stack
    uint32_t ifThenBlockId = lane.executionStack[ourStackIndex].ifThenBlockId;
    uint32_t ifElseBlockId = lane.executionStack[ourStackIndex].ifElseBlockId;
    uint32_t ifMergeBlockId = lane.executionStack[ourStackIndex].ifMergeBlockId;

    lane.executionStack.pop_back();
    tg.popMergePoint(wave.waveId, lane.laneId);

    // Clean up then/else blocks - lane will never return to them
    tg.removeThreadFromAllSets(ifThenBlockId, wave.waveId, lane.laneId);
    tg.removeThreadFromNestedBlocks(ifThenBlockId, wave.waveId, lane.laneId);

    if (hasElse && ifElseBlockId != 0) {
      tg.removeThreadFromAllSets(ifElseBlockId, wave.waveId, lane.laneId);
      tg.removeThreadFromNestedBlocks(ifElseBlockId, wave.waveId, lane.laneId);
    }

    // Also clean up merge block since we're not going there
    tg.removeThreadFromAllSets(ifMergeBlockId, wave.waveId, lane.laneId);
    tg.removeThreadFromNestedBlocks(ifMergeBlockId, wave.waveId, lane.laneId);

    // Restore active state (reconvergence)
    lane.isActive = lane.isActive && !lane.hasReturned;
    throw;
  }

  // Restore active state (reconvergence)
  lane.isActive = lane.isActive && !lane.hasReturned;
}

// Key semantics to preserve in Result-based version:
// 1. Uses evaluateConditionAndSetup() helper
// 2. Uses executeThenBranch() and executeElseBranch() helpers
// 3. Uses performReconvergence() helper
// 4. Comprehensive block cleanup on ControlFlowException
// 5. Proper merge point management
// 6. State protection checks with isProtectedState()


// Helper method for executing body statements in ForStmt
void ForStmt::executeBodyStatements(LaneContext &lane, WaveContext &wave, ThreadgroupContext &tg, 
                                   int ourStackIndex, uint32_t headerBlockId) {
  auto &ourEntry = lane.executionStack[ourStackIndex];
  
  // Execute statements - start in iteration block, naturally flow to merge blocks
  for (size_t i = ourEntry.statementIndex; i < body_.size(); i++) {
    lane.executionStack[ourStackIndex].statementIndex = i;

    uint32_t blockBeforeStatement = tg.getCurrentBlock(wave.waveId, lane.laneId);
    std::cout << "DEBUG: ForStmt - Lane " << lane.laneId << " executing statement " << i 
              << " in block " << blockBeforeStatement << std::endl;

    auto result = body_[i]->execute_result(lane, wave, tg);
    if (result.is_err()) {
      ExecutionError error = result.unwrap_err();
      if (error == ExecutionError::ControlFlowBreak) {
        throw ControlFlowException(ControlFlowException::Break);
      } else if (error == ExecutionError::ControlFlowContinue) {
        throw ControlFlowException(ControlFlowException::Continue);
      } else if (error == ExecutionError::WaveOperationWait) {
        if (!isProtectedState(lane.state)) {
          lane.state = ThreadState::WaitingForResume;
        }
        return;
      }
      // Other errors should not occur in this context
    }

    if (lane.hasReturned) {
      // Clean up iteration-specific merge point if it exists
      const void *iterationMarker = reinterpret_cast<const void *>(
          reinterpret_cast<uintptr_t>(this) + (ourEntry.loopIteration << 16) + 0x5000);

      std::vector<MergeStackEntry> currentMergeStack = tg.getCurrentMergeStack(wave.waveId, lane.laneId);
      if (!currentMergeStack.empty() && currentMergeStack.back().sourceStatement == iterationMarker) {
        tg.popMergePoint(wave.waveId, lane.laneId);
        std::cout << "DEBUG: ForStmt - Lane " << lane.laneId 
                  << " popped iteration merge point on early return" << std::endl;
      }
      // Pop our entry and return from loop
      lane.executionStack.pop_back();
      tg.popMergePoint(wave.waveId, lane.laneId);
      return;
    }

    if (lane.state != ThreadState::Ready) {
      // Child statement needs to resume - preserve current block context
      uint32_t blockAfterStatement = tg.getCurrentBlock(wave.waveId, lane.laneId);
      std::cout << "DEBUG: ForStmt - Lane " << lane.laneId << " child statement needs resume" << std::endl;
      std::cout << "  Block before: " << blockBeforeStatement 
                << ", Block after: " << blockAfterStatement << std::endl;
      return;
    }

    // Log block transitions (shows natural flow to merge blocks)
    uint32_t blockAfterStatement = tg.getCurrentBlock(wave.waveId, lane.laneId);
    if (blockBeforeStatement != blockAfterStatement) {
      std::cout << "DEBUG: ForStmt - Lane " << lane.laneId << " natural flow from block " << blockBeforeStatement
                << " to block " << blockAfterStatement << " during statement " << i << " (likely merge block)" << std::endl;
    }

    // Update statement index
    lane.executionStack[ourStackIndex].statementIndex = i + 1;
  }
}


// Helper method for body statement execution in WhileStmt
void WhileStmt::executeBodyStatements(LaneContext &lane, WaveContext &wave, ThreadgroupContext &tg,
                                     int ourStackIndex, uint32_t headerBlockId) {
  auto &ourEntry = lane.executionStack[ourStackIndex];
  
  // Execute statements
  for (size_t i = ourEntry.statementIndex; i < body_.size(); i++) {
    lane.executionStack[ourStackIndex].statementIndex = i;

    uint32_t blockBeforeStatement = tg.getCurrentBlock(wave.waveId, lane.laneId);
    std::cout << "DEBUG: WhileStmt - Lane " << lane.laneId << " executing statement " << i
              << " in block " << blockBeforeStatement << std::endl;

    auto result = body_[i]->execute_result(lane, wave, tg);
    if (result.is_err()) {
      ExecutionError error = result.unwrap_err();
      if (error == ExecutionError::ControlFlowBreak) {
        throw ControlFlowException(ControlFlowException::Break);
      } else if (error == ExecutionError::ControlFlowContinue) {
        throw ControlFlowException(ControlFlowException::Continue);
      } else if (error == ExecutionError::WaveOperationWait) {
        if (!isProtectedState(lane.state)) {
          lane.state = ThreadState::WaitingForResume;
        }
        return;
      }
      // Other errors should not occur in this context
    }
    
    if (lane.hasReturned) {
      // Clean up iteration-specific merge point if it exists
      const void *iterationMarker = reinterpret_cast<const void *>(
          reinterpret_cast<uintptr_t>(this) + (ourEntry.loopIteration << 16) + 0x5000);

      std::vector<MergeStackEntry> currentMergeStack = tg.getCurrentMergeStack(wave.waveId, lane.laneId);
      if (!currentMergeStack.empty() &&
          currentMergeStack.back().sourceStatement == iterationMarker) {
        tg.popMergePoint(wave.waveId, lane.laneId);
        std::cout << "DEBUG: WhileStmt - Lane " << lane.laneId
                  << " popped iteration merge point on early return" << std::endl;
      }

      // Pop our entry and return from loop
      lane.executionStack.pop_back();
      tg.popMergePoint(wave.waveId, lane.laneId);
      return;
    }

    if (lane.state != ThreadState::Ready) {
      uint32_t blockAfterStatement = tg.getCurrentBlock(wave.waveId, lane.laneId);
      std::cout << "DEBUG: WhileStmt - Lane " << lane.laneId << " child statement needs resume" << std::endl;
      std::cout << "  Block before: " << blockBeforeStatement << ", Block after: " << blockAfterStatement << std::endl;
      return;
    }

    // Log block transitions (shows natural flow to merge blocks)
    uint32_t blockAfterStatement = tg.getCurrentBlock(wave.waveId, lane.laneId);
    if (blockBeforeStatement != blockAfterStatement) {
      std::cout << "DEBUG: WhileStmt - Lane " << lane.laneId << " natural flow from block "
                << blockBeforeStatement << " to block " << blockAfterStatement
                << " during statement " << i << " (likely merge block)" << std::endl;
    }

    // Update statement index
    lane.executionStack[ourStackIndex].statementIndex = i + 1;
  }
}

void DoWhileStmt::executeBodyStatements(LaneContext &lane, WaveContext &wave, ThreadgroupContext &tg, 
                                int ourStackIndex, uint32_t headerBlockId) {
  auto &ourEntry = lane.executionStack[ourStackIndex];
  
  // Execute statements from where we left off
  for (size_t i = ourEntry.statementIndex; i < body_.size(); i++) {
    lane.executionStack[ourStackIndex].statementIndex = i;
    auto result = body_[i]->execute_result(lane, wave, tg);
    if (result.is_err()) {
      ExecutionError error = result.unwrap_err();
      if (error == ExecutionError::ControlFlowBreak) {
        throw ControlFlowException(ControlFlowException::Break);
      } else if (error == ExecutionError::ControlFlowContinue) {
        throw ControlFlowException(ControlFlowException::Continue);
      } else if (error == ExecutionError::WaveOperationWait) {
        if (!isProtectedState(lane.state)) {
          lane.state = ThreadState::WaitingForResume;
        }
        return;
      }
      // Other errors should not occur in this context
    }
    if (lane.hasReturned) {
      // Clean up iteration-specific merge point if it exists
      const void *iterationMarker = reinterpret_cast<const void *>(
          reinterpret_cast<uintptr_t>(this) +
          (ourEntry.loopIteration << 16) + 0x5000);

      std::vector<MergeStackEntry> currentMergeStack =
          tg.getCurrentMergeStack(wave.waveId, lane.laneId);
      if (!currentMergeStack.empty() &&
          currentMergeStack.back().sourceStatement == iterationMarker) {
        tg.popMergePoint(wave.waveId, lane.laneId);
        std::cout << "DEBUG: DoWhileStmt - Lane " << lane.laneId
                  << " popped iteration merge point on early return"
                  << std::endl;
      }

      // TODO: check if need additioanl cleanup
      // Pop our entry and return from loop
      lane.executionStack.pop_back();
      tg.popMergePoint(wave.waveId, lane.laneId);
      return;
    }
    if (lane.state != ThreadState::Ready) {
      // Child statement needs to resume - don't continue
      return;
    }
  }
}


void SwitchStmt::executeCaseStatements(LaneContext &lane, WaveContext &wave, ThreadgroupContext &tg, 
                           int ourStackIndex) {
  auto &ourEntry = lane.executionStack[ourStackIndex];
  
  // Execute all statements from current position until case/switch completion
  while (ourEntry.caseIndex < cases_.size()) {
    const auto &caseBlock = cases_[ourEntry.caseIndex];
    std::string caseLabel = caseBlock.value.has_value()
                                ? std::to_string(caseBlock.value.value())
                                : "default";

    // Execute all statements in current case from saved position
    for (size_t i = ourEntry.statementIndex;
         i < caseBlock.statements.size(); i++) {
      lane.executionStack[ourStackIndex].statementIndex = i;

      std::cout << "DEBUG: SwitchStmt - Lane " << lane.laneId
                << " executing statement " << i << " in case " << caseLabel
                << std::endl;

      // Execute the current statement
      auto result = caseBlock.statements[i]->execute_result(lane, wave, tg);
      if (result.is_err()) {
        ExecutionError error = result.unwrap_err();
        if (error == ExecutionError::ControlFlowBreak) {
          throw ControlFlowException(ControlFlowException::Break);
        } else if (error == ExecutionError::WaveOperationWait) {
          if (!isProtectedState(lane.state)) {
            lane.state = ThreadState::WaitingForResume;
          }
          return;
        }
        // Continue errors should not occur in switch case statements
        // Other errors should not occur in this context
      }

      if (lane.hasReturned) {
        std::cout << "DEBUG: SwitchStmt - Lane " << lane.laneId
                  << " popping stack due to return (depth "
                  << lane.executionStack.size() << "->"
                  << (lane.executionStack.size() - 1) << ", this=" << this
                  << ")" << std::endl;
        // TODO: verify if need additional cleanup
        lane.executionStack.pop_back();
        tg.popMergePoint(wave.waveId, lane.laneId);
        return;
      }

      if (lane.state != ThreadState::Ready) {
        // Child statement needs to resume - don't continue
        return;
      }
      lane.executionStack[ourStackIndex].statementIndex = i + 1;
    }

    // Check if lane is ready before fallthrough - if waiting for wave
    // operations, don't move to next case
    if (lane.state != ThreadState::Ready) {
      std::cout << "DEBUG: SwitchStmt - Lane " << lane.laneId
                << " completed case " << caseLabel
                << " but is not Ready (state=" << (int)lane.state
                << "), pausing before fallthrough" << std::endl;
      return;
    }

    // Current case completed - move to next case (fallthrough)
    std::cout << "DEBUG: SwitchStmt - Lane " << lane.laneId
              << " completed case " << caseLabel
              << ", falling through to next" << std::endl;

    size_t nextCaseIndex = ourEntry.caseIndex + 1;

    // Move lane to next case block if it exists
    if (nextCaseIndex < ourEntry.switchCaseBlockIds.size()) {
      uint32_t currentCaseBlockId =
          ourEntry.switchCaseBlockIds[ourEntry.caseIndex];
      uint32_t nextCaseBlockId = ourEntry.switchCaseBlockIds[nextCaseIndex];

      std::cout << "DEBUG: SwitchStmt - Lane " << lane.laneId
                << " moving from case block " << currentCaseBlockId
                << " to case block " << nextCaseBlockId << " (fallthrough)"
                << std::endl;

      // Move to next case block (fallthrough)
      // tg.assignLaneToBlock(wave.waveId, lane.laneId, nextCaseBlockId);
      tg.moveThreadFromUnknownToParticipating(nextCaseBlockId, wave.waveId,
                                              lane.laneId);

      // Remove from current case block
      tg.removeThreadFromAllSets(currentCaseBlockId, wave.waveId,
                                 lane.laneId);
    }

    // Move to next case
    lane.executionStack[ourStackIndex].caseIndex++;
    lane.executionStack[ourStackIndex].statementIndex = 0;
  }

  // All cases completed - enter reconvergence
  std::cout << "DEBUG: SwitchStmt - Lane " << lane.laneId
            << " completed all cases, entering reconvergence" << std::endl;
  ourEntry.phase = LaneContext::ControlFlowPhase::Reconverging;
}