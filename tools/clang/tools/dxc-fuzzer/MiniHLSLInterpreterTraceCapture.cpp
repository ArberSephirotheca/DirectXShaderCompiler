#include "MiniHLSLInterpreterTraceCapture.h"
#include <sstream>

namespace minihlsl {
namespace fuzzer {

TraceCaptureInterpreter::TraceCaptureInterpreter() : trace_() {}

interpreter::ExecutionResult TraceCaptureInterpreter::executeAndCaptureTrace(
    const interpreter::Program& program,
    const interpreter::ThreadOrdering& ordering,
    uint32_t waveSize) {
  
  // Clear previous trace
  trace_ = ExecutionTrace();
  
  // Record thread hierarchy first
  auto tgSize = program.numThreadsX * program.numThreadsY * program.numThreadsZ;
  auto effectiveWaveSize = program.getEffectiveWaveSize(waveSize);
  trace_.threadHierarchy.totalThreads = tgSize;
  trace_.threadHierarchy.waveSize = effectiveWaveSize;
  trace_.threadHierarchy.numWaves = (tgSize + effectiveWaveSize - 1) / effectiveWaveSize;
  
  // Execute with trace capture hooks enabled
  auto result = execute(program, ordering, waveSize);
  
  // Capture final state - shared memory
  for (const auto& [addr, value] : result.sharedMemoryState) {
    trace_.finalState.sharedMemory[addr] = value;
  }
  
  // Note: Thread variable states and wave operations are captured in onExecutionComplete hook
  
  return result;
}

// Hook implementations that override virtual methods from MiniHLSLInterpreter

void TraceCaptureInterpreter::onStatementExecute(interpreter::LaneContext &lane, 
                                                interpreter::WaveContext &wave,
                                                interpreter::ThreadgroupContext &tg) {
  // Called when a statement is about to execute
  // For now, just record basic information
  
  // Update thread hierarchy if needed
  recordThreadHierarchy(tg);
}

void TraceCaptureInterpreter::onStatementComplete(interpreter::LaneContext &lane, 
                                                 interpreter::WaveContext &wave,
                                                 interpreter::ThreadgroupContext &tg) {
  // Called after a statement completes
  // Can record timing information, state changes, etc.
}

void TraceCaptureInterpreter::onWaveOpSync(interpreter::WaveContext &wave, 
                                          interpreter::ThreadgroupContext &tg,
                                          const interpreter::SyncPointState &syncState) {
  // Record wave operation synchronization
  ExecutionTrace::WaveOpRecord record;
  record.waveId = wave.waveId;
  
  // Store basic info
  trace_.waveOperations.push_back(record);
}

void TraceCaptureInterpreter::onControlFlow(interpreter::LaneContext &lane, 
                                           interpreter::WaveContext &wave,
                                           interpreter::ThreadgroupContext &tg, 
                                           const interpreter::Statement *stmt,
                                           bool branchTaken) {
  // Record control flow decisions
  ExecutionTrace::ControlFlowDecision decision;
  decision.statement = stmt;
  decision.timestamp = trace_.controlFlowHistory.size(); // Simple timestamp
  
  // Record per-lane decision
  auto& laneDecision = decision.decisions[wave.waveId][lane.laneId];
  laneDecision.branchTaken = branchTaken;
  
  trace_.controlFlowHistory.push_back(decision);
}

void TraceCaptureInterpreter::onVariableAccess(interpreter::LaneContext &lane, 
                                              interpreter::WaveContext &wave,
                                              interpreter::ThreadgroupContext &tg, 
                                              const std::string &name,
                                              bool isWrite, 
                                              const interpreter::Value &value) {
  // Record variable accesses
  ExecutionTrace::VariableAccess access;
  access.varName = name;
  access.isWrite = isWrite;
  access.timestamp = trace_.variableAccesses.size();
  access.type = isWrite ? ExecutionTrace::VariableAccess::Write : 
                          ExecutionTrace::VariableAccess::Read;
  
  // Record value
  access.values[wave.waveId][lane.laneId] = value;
  
  trace_.variableAccesses.push_back(access);
}

void TraceCaptureInterpreter::onBarrier(interpreter::ThreadgroupContext &tg) {
  // Record barrier synchronization
  ExecutionTrace::BarrierRecord barrier;
  barrier.barrierId = trace_.barriers.size();
  
  // Record arrival pattern
  for (size_t waveId = 0; waveId < tg.waves.size(); ++waveId) {
    auto& wave = *tg.waves[waveId];
    std::set<interpreter::LaneId> arrivedLanes;
    
    for (size_t laneId = 0; laneId < wave.lanes.size(); ++laneId) {
      auto& lane = *wave.lanes[laneId];
      if (lane.state == interpreter::ThreadState::WaitingAtBarrier) {
        arrivedLanes.insert(laneId);
        
        // Record thread ID
        auto tid = waveId * tg.waveSize + laneId;
        barrier.arrivalTimes[tid] = trace_.barriers.size();
      }
    }
    
    if (!arrivedLanes.empty()) {
      barrier.arrivedLanesPerWave[waveId] = arrivedLanes;
      barrier.waveArrivalOrder.push_back(waveId);
    }
  }
  
  trace_.barriers.push_back(barrier);
}

void TraceCaptureInterpreter::recordThreadHierarchy(const interpreter::ThreadgroupContext& tg) {
  // Record thread hierarchy if not already done
  if (trace_.threadHierarchy.totalThreads == 0) {
    trace_.threadHierarchy.totalThreads = tg.threadgroupSize;
    trace_.threadHierarchy.waveSize = tg.waveSize;
    trace_.threadHierarchy.numWaves = tg.waves.size();
    
    // Record thread-to-wave mapping
    for (interpreter::ThreadId tid = 0; tid < tg.threadgroupSize; ++tid) {
      auto [waveId, laneId] = tg.getWaveAndLane(tid);
      trace_.threadHierarchy.threadToWave[tid] = waveId;
      trace_.threadHierarchy.threadToLane[tid] = laneId;
      trace_.threadHierarchy.waveToThreads[waveId].push_back(tid);
    }
  }
}

void TraceCaptureInterpreter::recordBlockExecution(
    uint32_t blockId, 
    const interpreter::DynamicExecutionBlock& block,
    const interpreter::ThreadgroupContext& tg) {
  
  ExecutionTrace::BlockExecutionRecord record;
  record.blockId = blockId;
  record.blockType = block.getBlockType();
  record.sourceStatement = block.getSourceStatement();
  record.parentBlockId = block.getParentBlockId();
  
  // Record participating lanes per wave
  auto participants = block.getParticipatingLanes(tg);
  for (const auto& [waveId, lanes] : participants) {
    record.waveParticipation[waveId].participatingLanes = lanes;
  }
  
  // Check if this is a loop iteration block
  // Loop body blocks have REGULAR type and their parent is a LOOP_HEADER
  if (block.getBlockType() == interpreter::BlockType::REGULAR && 
      record.parentBlockId != 0) {
    // Check if parent is a loop header
    auto parentIt = trace_.blocks.find(record.parentBlockId);
    if (parentIt != trace_.blocks.end() && 
        parentIt->second.blockType == interpreter::BlockType::LOOP_HEADER) {
      
      // This is likely a loop body block
      // Extract iteration number from the encoded pointer
      // The iteration is encoded as: (loopStmt + (iteration << 16) + 0x3000)
      if (block.getSourceStatement() != nullptr) {
        uintptr_t ptrValue = reinterpret_cast<uintptr_t>(block.getSourceStatement());
        
        // Extract the iteration number (bits 16-31)
        // First remove the 0x3000 offset, then shift right by 16
        if ((ptrValue & 0xFFFF) == 0x3000) {  // Check for iteration block marker
          int iteration = (ptrValue >> 16) & 0xFFFF;
          
          // Create loop iteration info
          ExecutionTrace::BlockExecutionRecord::LoopIterationInfo loopInfo;
          loopInfo.iterationValue = iteration;
          loopInfo.loopHeaderBlock = record.parentBlockId;
          
          // Try to determine loop variable by checking parent block's source statement
          // This is a heuristic - in practice we might need more sophisticated detection
          loopInfo.loopVariable = ""; // Will be filled by findLoopVariable in fuzzer
          
          record.loopIteration = loopInfo;
        }
      }
    }
  }
  
  // Store in trace
  trace_.blocks[blockId] = record;
}

void TraceCaptureInterpreter::recordWaveOperation(
    const std::string& opName,
    const std::map<interpreter::WaveId, std::set<interpreter::LaneId>>& participants,
    const std::map<interpreter::LaneId, interpreter::Value>& inputs,
    const std::map<interpreter::LaneId, interpreter::Value>& outputs,
    uint32_t blockId) {
  
  ExecutionTrace::WaveOpRecord record;
  record.opType = opName;
  record.blockId = blockId;
  
  // NEW: Collect loop iteration values from lane execution stack
  std::vector<uint32_t> loopIterations;
  
  // Get loop iterations from the first participating lane's execution stack
  // All participating lanes should have the same loop context
  if (!participants.empty() && !participants.begin()->second.empty()) {
    auto waveId = participants.begin()->first;
    auto laneId = *participants.begin()->second.begin();
    
    // Access the lane from the current threadgroup context
    // Note: This requires access to the threadgroup context
    // We'll need to pass it as a parameter or store it as a member
  }
  
  // Fallback: Try to get loop iterations from block hierarchy (for 0x3000 encoded blocks)
  uint32_t currentBlockId = blockId;
  
  while (currentBlockId != 0) {
    auto blockIt = trace_.blocks.find(currentBlockId);
    if (blockIt == trace_.blocks.end()) break;
    
    // If this block has loop iteration info, collect the iteration value
    if (blockIt->second.loopIteration.has_value()) {
      loopIterations.insert(loopIterations.begin(), 
                           blockIt->second.loopIteration->iterationValue);
    }
    
    // Move to parent block
    currentBlockId = blockIt->second.parentBlockId;
  }
  
  record.loopIterations = loopIterations;
  
  
  // Convert inputs/outputs to storage format
  for (const auto& [laneId, value] : inputs) {
    record.inputValues[laneId] = value;
  }
  for (const auto& [laneId, value] : outputs) {
    record.outputValues[laneId] = value;
  }
  
  trace_.waveOperations.push_back(record);
}

void TraceCaptureInterpreter::onExecutionComplete(const interpreter::ThreadgroupContext &tg) {
  // Extract wave operations from sync points
  extractWaveOperationsFromContext(tg);
  
  // Capture final variable states for all threads
  for (size_t waveId = 0; waveId < tg.waves.size(); ++waveId) {
    const auto& wave = *tg.waves[waveId];
    
    for (size_t laneId = 0; laneId < wave.lanes.size(); ++laneId) {
      const auto& lane = *wave.lanes[laneId];
      
      // Copy all variables for this lane
      for (const auto& [varName, value] : lane.variables) {
        trace_.finalState.laneVariables[waveId][laneId][varName] = value;
      }
      
      // Also capture return value
      trace_.finalState.returnValues[waveId][laneId] = lane.returnValue;
      
      // Capture final thread state
      trace_.finalState.finalThreadStates[waveId][laneId] = lane.state;
    }
  }
}

void TraceCaptureInterpreter::onLaneEnterBlock(interpreter::LaneContext &lane, 
                                               interpreter::WaveContext &wave,
                                               interpreter::ThreadgroupContext &tg, 
                                               uint32_t blockId) {
  // Record when a lane enters a block
  if (tg.executionBlocks.find(blockId) != tg.executionBlocks.end()) {
    auto& block = tg.executionBlocks[blockId];
    
    // Record block execution if not already recorded
    if (trace_.blocks.find(blockId) == trace_.blocks.end()) {
      recordBlockExecution(blockId, block, tg);
    }
    
    // Update participation info
    auto& record = trace_.blocks[blockId];
    record.waveParticipation[wave.waveId].participatingLanes.insert(lane.laneId);
    record.waveParticipation[wave.waveId].entryTimestamps[lane.laneId].push_back(
        trace_.variableAccesses.size()); // Use variable access count as timestamp
  }
}

void TraceCaptureInterpreter::onWaveOpExecuted(interpreter::WaveContext &wave, 
                                               interpreter::ThreadgroupContext &tg,
                                               const std::string &opName, 
                                               const interpreter::Value &result) {
  // Record wave operation execution
  ExecutionTrace::WaveOpRecord record;
  record.waveId = wave.waveId;
  record.opType = opName;
  
  // Collect loop iterations from the first active lane's execution stack
  std::vector<uint32_t> loopIterations;
  for (size_t laneId = 0; laneId < wave.lanes.size(); ++laneId) {
    if (wave.lanes[laneId]->isActive) {
      record.expectedParticipants.insert(laneId);
      record.arrivedParticipants.insert(laneId);
      record.outputValues[laneId] = result;
      
      // Get loop iterations from this lane's execution stack (only need to do once)
      if (loopIterations.empty()) {
        const auto& lane = *wave.lanes[laneId];
        for (const auto& execState : lane.executionStack) {
          // Check if this execution state is for a loop
          if (execState.loopIteration > 0 || execState.loopHeaderBlockId > 0) {
            loopIterations.push_back(execState.loopIteration);
          }
        }
      }
    }
  }
  
  record.loopIterations = loopIterations;
  
  trace_.waveOperations.push_back(record);
}

void TraceCaptureInterpreter::extractWaveOperationsFromContext(
    const interpreter::ThreadgroupContext& tg) {
  // Clear any wave operations recorded by hooks
  trace_.waveOperations.clear();
  
  // Extract from each wave's sync points
  for (size_t waveId = 0; waveId < tg.waves.size(); ++waveId) {
    const auto& wave = *tg.waves[waveId];
    
    // Iterate through all sync points in this wave
    for (const auto& [key, syncPoint] : wave.activeSyncPoints) {
      ExecutionTrace::WaveOpRecord record;
      record.waveId = waveId;
      record.instruction = key.first; // Store the instruction pointer!
      
      // Capture stable ID from the wave operation expression
      if (key.first != nullptr) {
        auto* waveOp = static_cast<const interpreter::WaveActiveOp*>(key.first);
        record.stableId = waveOp->getStableId();
      }
      
      record.opType = syncPoint.instructionType;
      record.blockId = key.second; // blockId from the key pair
      record.waveOpEnumType = syncPoint.waveOpType; // Store the enum type
      
      // Use loop iterations from sync point
      record.loopIterations = syncPoint.loopIterations;
      
      // Override with actual loop variable values if available
      // The fuzzer generates loops with predictable patterns:
      // - for(uint VAR = ...; VAR < ...; VAR++)
      // - while(VAR < ...) { VAR++; ... }
      // 
      // Strategy: Find all loop variables by pattern and sort by their numeric suffix.
      // This handles cases where the fuzzer generates non-sequential variables like
      // counter0, counter3, counter5 without hardcoding a specific range.
      //
      // LIMITATIONS:
      // 1. Relies on fuzzer's naming convention (i<N> for for-loops, counter<N> for while-loops)
      // 2. Assumes the numeric suffix indicates the intended encoding order
      // 3. Only supports up to 3 loop variables due to bit encoding constraints (6 bits total)
      // 4. May not correctly handle deeply nested loops vs. sequential loops in different branches
      //    (e.g., nested loops i0->i1->i2 vs. separate loops i0, i3, i5 in switch cases)
      // 5. The ordering by numeric suffix may not match the actual nesting order in all cases
      //
      // Future improvement: Track loop nesting depth from execution context rather than
      // relying on variable naming conventions.
      if (!syncPoint.loopVariableValues.empty()) {
        std::vector<uint32_t> actualValues;
        
        // Collect all loop variables with their numeric suffixes
        std::map<int, uint32_t> indexToValue; // map sorts by key automatically
        
        for (const auto& [varName, value] : syncPoint.loopVariableValues) {
          int index = -1;
          
          // Extract numeric suffix from i<N> pattern
          if (varName.size() > 1 && varName[0] == 'i') {
            try {
              index = std::stoi(varName.substr(1));
            } catch (const std::exception&) {
              // Not a valid number, skip
            }
          }
          // Extract numeric suffix from counter<N> pattern  
          else if (varName.size() > 7 && varName.substr(0, 7) == "counter") {
            try {
              index = std::stoi(varName.substr(7));
            } catch (const std::exception&) {
              // Not a valid number, skip
            }
          }
          
          if (index >= 0) {
            indexToValue[index] = value;
          }
        }
        
        // Take values in ascending order of index (map iteration is ordered)
        for (const auto& [index, value] : indexToValue) {
          actualValues.push_back(value);
          if (actualValues.size() >= 3) break; // Maximum 3 values for bit encoding
        }
        
        if (!actualValues.empty()) {
          record.loopIterations = actualValues;
        }
      }
      
      
      // Use actual participants from sync point
      record.expectedParticipants = syncPoint.expectedParticipants;
      record.arrivedParticipants = syncPoint.arrivedParticipants;
      
      // Copy the results for each participant
      for (const auto& [laneId, result] : syncPoint.pendingResults) {
        record.outputValues[laneId] = result;
      }
      
      trace_.waveOperations.push_back(record);
    }
  }
}

} // namespace fuzzer
} // namespace minihlsl