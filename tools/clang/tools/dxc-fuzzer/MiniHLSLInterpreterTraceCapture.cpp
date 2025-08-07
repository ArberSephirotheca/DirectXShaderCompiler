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
  trace_.threadHierarchy.totalThreads = tgSize;
  trace_.threadHierarchy.waveSize = waveSize;
  trace_.threadHierarchy.numWaves = (tgSize + waveSize - 1) / waveSize;
  
  // Execute with trace capture hooks enabled
  auto result = execute(program, ordering, waveSize);
  
  // Capture final state - shared memory
  for (const auto& [addr, value] : result.sharedMemoryState) {
    trace_.finalState.sharedMemory[addr] = value;
  }
  
  // Note: Thread variable states are captured in onExecutionComplete hook
  
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
  
  // Record participating lanes
  for (size_t laneId = 0; laneId < wave.lanes.size(); ++laneId) {
    if (wave.lanes[laneId]->isActive) {
      record.expectedParticipants.insert(laneId);
      record.arrivedParticipants.insert(laneId);
      record.outputValues[laneId] = result;
    }
  }
  
  trace_.waveOperations.push_back(record);
}

} // namespace fuzzer
} // namespace minihlsl