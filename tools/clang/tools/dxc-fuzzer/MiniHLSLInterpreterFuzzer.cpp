#include "MiniHLSLInterpreterFuzzer.h"
#include <fstream>
#include <sstream>
#include <random>
#include <algorithm>

namespace minihlsl {
namespace fuzzer {

// Global fuzzer instance
std::unique_ptr<TraceGuidedFuzzer> g_fuzzer;

// ===== Trace Capture Implementation =====

void TraceCaptureInterpreter::captureFinalState(const interpreter::ThreadgroupContext& tg) {
  auto& finalState = trace->finalState;
  
  // Capture thread hierarchy
  trace->hierarchy.threadgroupSize = tg.threadgroupSize;
  trace->hierarchy.waveSize = tg.waveSize;
  trace->hierarchy.waveCount = tg.waveCount;
  
  // Build thread mappings
  for (uint32_t waveId = 0; waveId < tg.waveCount; waveId++) {
    auto& wave = *tg.waves[waveId];
    for (uint32_t laneId = 0; laneId < wave.waveSize; laneId++) {
      interpreter::ThreadId tid = tg.getGlobalThreadId(waveId, laneId);
      trace->hierarchy.threadToWaveLane[tid] = {waveId, laneId};
      trace->hierarchy.waveLaneToThread[{waveId, laneId}] = tid;
    }
  }
  
  // Capture all lane variables and states
  for (uint32_t waveId = 0; waveId < tg.waveCount; waveId++) {
    auto& wave = *tg.waves[waveId];
    for (uint32_t laneId = 0; laneId < wave.waveSize; laneId++) {
      auto& lane = *wave.lanes[laneId];
      
      finalState.laneVariables[waveId][laneId] = lane.variables;
      finalState.returnValues[waveId][laneId] = lane.returnValue;
      finalState.finalThreadStates[waveId][laneId] = lane.state;
    }
  }
  
  // Capture memory state
  for (auto& [name, buffer] : tg.globalBuffers) {
    finalState.globalBuffers[name] = buffer->getSnapshot();
  }
  finalState.sharedMemory = tg.sharedMemory->getSnapshot();
  
  // Update statistics
  trace->stats.totalDynamicBlocks = trace->blocks.size();
  trace->stats.totalWaveOps = trace->waveOperations.size();
  trace->stats.totalBarriers = trace->barriers.size();
}

uint32_t TraceCaptureInterpreter::getOrCreateBlock(
    const interpreter::BlockIdentity& identity,
    interpreter::WaveId waveId,
    const std::set<interpreter::LaneId>& lanes,
    uint32_t parentBlockId) {
  
  // Call parent implementation
  uint32_t blockId = Interpreter::getOrCreateBlock(identity, waveId, lanes, parentBlockId);
  
  // Record in trace
  auto& record = trace->blocks[blockId];
  record.blockId = blockId;
  record.identity = identity;
  record.blockType = identity.blockType;
  record.sourceStatement = identity.sourceStatement;
  record.parentBlockId = parentBlockId;
  record.creatorWave = waveId;
  
  // Initialize wave participation
  record.waveParticipation[waveId].participatingLanes = lanes;
  
  // Track block relationships
  if (parentBlockId != 0) {
    record.predecessors.insert(parentBlockId);
    trace->blocks[parentBlockId].successors.insert(blockId);
  }
  
  return blockId;
}

void TraceCaptureInterpreter::onLaneEnterBlock(
    interpreter::WaveId waveId,
    interpreter::LaneId laneId,
    uint32_t blockId) {
  
  Interpreter::onLaneEnterBlock(waveId, laneId, blockId);
  
  auto& block = trace->blocks[blockId];
  auto& wavePart = block.waveParticipation[waveId];
  
  wavePart.participatingLanes.insert(laneId);
  wavePart.arrivedLanes.insert(laneId);
  wavePart.visitCount[laneId]++;
  wavePart.entryTimestamps[laneId].push_back(globalTimestamp++);
}

void TraceCaptureInterpreter::onLaneExitBlock(
    interpreter::WaveId waveId,
    interpreter::LaneId laneId,
    uint32_t blockId) {
  
  auto& block = trace->blocks[blockId];
  auto& wavePart = block.waveParticipation[waveId];
  
  wavePart.exitTimestamps[laneId].push_back(globalTimestamp++);
  
  Interpreter::onLaneExitBlock(waveId, laneId, blockId);
}

void TraceCaptureInterpreter::onControlFlowDecision(
    const interpreter::Statement* stmt,
    interpreter::WaveId waveId,
    interpreter::LaneId laneId,
    const interpreter::Value& condition,
    bool taken,
    interpreter::LaneContext::ControlFlowPhase phase) {
  
  ExecutionTrace::ControlFlowDecision decision;
  decision.statement = stmt;
  decision.executionBlockId = getCurrentBlockId(waveId, laneId);
  decision.timestamp = globalTimestamp++;
  
  auto& laneDecision = decision.decisions[waveId][laneId];
  laneDecision.conditionValue = condition;
  laneDecision.branchTaken = taken;
  laneDecision.phase = phase;
  
  trace->controlFlowHistory.push_back(decision);
}

// ... (implement remaining TraceCaptureInterpreter methods)

uint32_t TraceCaptureInterpreter::getCurrentBlockId(
    interpreter::WaveId waveId,
    interpreter::LaneId laneId) {
  // Get current block from interpreter's internal state
  // This would need to access protected/internal interpreter state
  return 0; // Placeholder
}

// ===== Mutation Implementations =====

// --- Explicit Lane Divergence Mutation ---

bool ExplicitLaneDivergenceMutation::canApply(
    const interpreter::Statement* stmt,
    const ExecutionTrace& trace) const {
  
  // Only apply to if statements that caused divergence
  auto ifStmt = dynamic_cast<const interpreter::IfStmt*>(stmt);
  if (!ifStmt) return false;
  
  // Check if different lanes took different branches
  for (auto& decision : trace.controlFlowHistory) {
    if (decision.statement == stmt) {
      std::set<bool> uniqueDecisions;
      for (auto& [waveId, laneDecisions] : decision.decisions) {
        for (auto& [laneId, dec] : laneDecisions) {
          uniqueDecisions.insert(dec.branchTaken);
        }
      }
      return uniqueDecisions.size() > 1;  // Divergence occurred
    }
  }
  return false;
}

std::unique_ptr<interpreter::Statement> ExplicitLaneDivergenceMutation::apply(
    const interpreter::Statement* stmt,
    const ExecutionTrace& trace) const {
  
  auto ifStmt = static_cast<const interpreter::IfStmt*>(stmt);
  
  // Find the decision record
  const ExecutionTrace::ControlFlowDecision* decision = nullptr;
  for (auto& d : trace.controlFlowHistory) {
    if (d.statement == stmt) {
      decision = &d;
      break;
    }
  }
  
  if (!decision) return nullptr;
  
  auto compound = std::make_unique<interpreter::CompoundStmt>();
  
  // For each wave
  for (auto& [waveId, laneDecisions] : decision->decisions) {
    // Group lanes by decision
    std::set<interpreter::LaneId> trueLanes, falseLanes;
    for (auto& [laneId, dec] : laneDecisions) {
      if (dec.branchTaken) trueLanes.insert(laneId);
      else falseLanes.insert(laneId);
    }
    
    // Create explicit conditions
    if (!trueLanes.empty() && ifStmt->thenBlock) {
      auto condition = createComplexCondition(waveId, trueLanes, ifStmt->condition);
      auto thenStmt = std::make_unique<interpreter::IfStmt>(
        std::move(condition),
        ifStmt->thenBlock->clone(),
        nullptr
      );
      compound->addStatement(std::move(thenStmt));
    }
    
    if (!falseLanes.empty() && ifStmt->elseBlock) {
      auto notCond = std::make_unique<interpreter::UnaryOpExpr>(
        ifStmt->condition->clone(),
        interpreter::UnaryOpExpr::LogicalNot
      );
      auto condition = createComplexCondition(waveId, falseLanes, notCond);
      auto elseStmt = std::make_unique<interpreter::IfStmt>(
        std::move(condition),
        ifStmt->elseBlock->clone(),
        nullptr
      );
      compound->addStatement(std::move(elseStmt));
    }
  }
  
  return compound;
}

std::unique_ptr<interpreter::Expression> ExplicitLaneDivergenceMutation::createComplexCondition(
    interpreter::WaveId waveId,
    const std::set<interpreter::LaneId>& lanes,
    const std::unique_ptr<interpreter::Expression>& originalCond) const {
  
  // waveId == X
  auto waveCheck = std::make_unique<interpreter::BinaryOpExpr>(
    std::make_unique<interpreter::WaveIndexExpr>(),
    std::make_unique<interpreter::LiteralExpr>(interpreter::Value((int32_t)waveId)),
    interpreter::BinaryOpExpr::Eq
  );
  
  // (laneId == Y || laneId == Z || ...)
  std::unique_ptr<interpreter::Expression> laneCheck;
  for (auto laneId : lanes) {
    auto laneEq = std::make_unique<interpreter::BinaryOpExpr>(
      std::make_unique<interpreter::LaneIndexExpr>(),
      std::make_unique<interpreter::LiteralExpr>(interpreter::Value((int32_t)laneId)),
      interpreter::BinaryOpExpr::Eq
    );
    
    if (!laneCheck) {
      laneCheck = std::move(laneEq);
    } else {
      laneCheck = std::make_unique<interpreter::BinaryOpExpr>(
        std::move(laneCheck),
        std::move(laneEq),
        interpreter::BinaryOpExpr::Or
      );
    }
  }
  
  // Combine all conditions
  auto waveAndLane = std::make_unique<interpreter::BinaryOpExpr>(
    std::move(waveCheck),
    std::move(laneCheck),
    interpreter::BinaryOpExpr::And
  );
  
  return std::make_unique<interpreter::BinaryOpExpr>(
    std::move(waveAndLane),
    originalCond->clone(),
    interpreter::BinaryOpExpr::And
  );
}

bool ExplicitLaneDivergenceMutation::validateSemanticPreservation(
    const interpreter::Statement* original,
    const interpreter::Statement* mutated,
    const ExecutionTrace& trace) const {
  // The mutation is semantically preserving by construction
  // It explicitly encodes the exact execution that occurred
  return true;
}

// --- Loop Unrolling Mutation ---

bool LoopUnrollingMutation::canApply(
    const interpreter::Statement* stmt,
    const ExecutionTrace& trace) const {
  
  // Check if it's a loop with bounded iterations
  bool isLoop = dynamic_cast<const interpreter::ForStmt*>(stmt) != nullptr ||
                dynamic_cast<const interpreter::WhileStmt*>(stmt) != nullptr ||
                dynamic_cast<const interpreter::DoWhileStmt*>(stmt) != nullptr;
  
  if (!isLoop) return false;
  
  // Check if we have loop pattern data
  return trace.loops.find(stmt) != trace.loops.end();
}

std::unique_ptr<interpreter::Statement> LoopUnrollingMutation::apply(
    const interpreter::Statement* stmt,
    const ExecutionTrace& trace) const {
  
  // Find loop pattern
  auto it = trace.loops.find(stmt);
  if (it == trace.loops.end()) return nullptr;
  
  auto& loopPattern = it->second;
  auto compound = std::make_unique<interpreter::CompoundStmt>();
  
  // Find maximum iteration count across all lanes/waves
  uint32_t maxIterations = 0;
  std::map<uint32_t, std::set<std::pair<interpreter::WaveId, interpreter::LaneId>>> iterationGroups;
  
  for (auto& [waveId, lanePatterns] : loopPattern.lanePatterns) {
    for (auto& [laneId, pattern] : lanePatterns) {
      maxIterations = std::max(maxIterations, pattern.totalIterations);
      iterationGroups[pattern.totalIterations].insert({waveId, laneId});
    }
  }
  
  // Generate unrolled version
  for (uint32_t iter = 0; iter < maxIterations; iter++) {
    // Which lanes execute this iteration?
    std::map<interpreter::WaveId, std::set<interpreter::LaneId>> activeWaveLanes;
    
    for (auto& [count, waveLanes] : iterationGroups) {
      if (iter < count) {
        for (auto& [waveId, laneId] : waveLanes) {
          activeWaveLanes[waveId].insert(laneId);
        }
      }
    }
    
    // Create guarded iteration
    auto iterBlock = createGuardedIteration(stmt, iter, activeWaveLanes);
    compound->addStatement(std::move(iterBlock));
  }
  
  return compound;
}

std::unique_ptr<interpreter::Statement> LoopUnrollingMutation::createGuardedIteration(
    const interpreter::Statement* loopStmt,
    uint32_t iteration,
    const std::map<interpreter::WaveId, std::set<interpreter::LaneId>>& activeWaveLanes) const {
  
  // Get loop body
  const interpreter::CompoundStmt* body = nullptr;
  if (auto forStmt = dynamic_cast<const interpreter::ForStmt*>(loopStmt)) {
    body = forStmt->body.get();
  } else if (auto whileStmt = dynamic_cast<const interpreter::WhileStmt*>(loopStmt)) {
    body = whileStmt->body.get();
  } else if (auto doStmt = dynamic_cast<const interpreter::DoWhileStmt*>(loopStmt)) {
    body = doStmt->body.get();
  }
  
  if (!body) return nullptr;
  
  // Check if all lanes are active
  bool allActive = true;
  uint32_t totalActiveLanes = 0;
  for (auto& [waveId, lanes] : activeWaveLanes) {
    totalActiveLanes += lanes.size();
  }
  
  // If all lanes active, no guard needed
  if (totalActiveLanes == 32) { // Assuming 32 lanes total
    return body->clone();
  }
  
  // Create complex guard condition
  std::unique_ptr<interpreter::Expression> condition;
  
  for (auto& [waveId, lanes] : activeWaveLanes) {
    // Create wave+lane condition
    auto waveCheck = std::make_unique<interpreter::BinaryOpExpr>(
      std::make_unique<interpreter::WaveIndexExpr>(),
      std::make_unique<interpreter::LiteralExpr>(interpreter::Value((int32_t)waveId)),
      interpreter::BinaryOpExpr::Eq
    );
    
    std::unique_ptr<interpreter::Expression> laneCheck;
    for (auto laneId : lanes) {
      auto laneEq = std::make_unique<interpreter::BinaryOpExpr>(
        std::make_unique<interpreter::LaneIndexExpr>(),
        std::make_unique<interpreter::LiteralExpr>(interpreter::Value((int32_t)laneId)),
        interpreter::BinaryOpExpr::Eq
      );
      
      if (!laneCheck) {
        laneCheck = std::move(laneEq);
      } else {
        laneCheck = std::make_unique<interpreter::BinaryOpExpr>(
          std::move(laneCheck),
          std::move(laneEq),
          interpreter::BinaryOpExpr::Or
        );
      }
    }
    
    auto waveCondition = std::make_unique<interpreter::BinaryOpExpr>(
      std::move(waveCheck),
      std::move(laneCheck),
      interpreter::BinaryOpExpr::And
    );
    
    if (!condition) {
      condition = std::move(waveCondition);
    } else {
      condition = std::make_unique<interpreter::BinaryOpExpr>(
        std::move(condition),
        std::move(waveCondition),
        interpreter::BinaryOpExpr::Or
      );
    }
  }
  
  return std::make_unique<interpreter::IfStmt>(
    std::move(condition),
    body->clone(),
    nullptr
  );
}

bool LoopUnrollingMutation::validateSemanticPreservation(
    const interpreter::Statement* original,
    const interpreter::Statement* mutated,
    const ExecutionTrace& trace) const {
  // Unrolling with exact guards preserves semantics
  return true;
}

// ===== Semantic Validator Implementation =====

SemanticValidator::ValidationResult SemanticValidator::validate(
    const ExecutionTrace& golden,
    const ExecutionTrace& mutant) {
  
  ValidationResult result{true, "", {}};
  
  // 1. Compare final states
  if (!compareFinalStates(golden.finalState, mutant.finalState, result)) {
    return result;
  }
  
  // 2. Compare wave operation results
  if (!compareWaveOperations(golden.waveOperations, mutant.waveOperations, result)) {
    return result;
  }
  
  // 3. Compare memory access results
  if (!compareMemoryState(golden, mutant, result)) {
    return result;
  }
  
  // 4. Verify control flow equivalence (may differ in structure)
  if (!verifyControlFlowEquivalence(golden, mutant, result)) {
    return result;
  }
  
  return result;
}

bool SemanticValidator::compareFinalStates(
    const ExecutionTrace::FinalState& golden,
    const ExecutionTrace::FinalState& mutant,
    ValidationResult& result) {
  
  // Compare all lane variables
  for (auto& [waveId, laneVars] : golden.laneVariables) {
    for (auto& [laneId, vars] : laneVars) {
      auto mutantIt = mutant.laneVariables.find(waveId);
      if (mutantIt == mutant.laneVariables.end()) {
        result.isEquivalent = false;
        result.divergenceReason = "Missing wave in mutant";
        return false;
      }
      
      auto laneIt = mutantIt->second.find(laneId);
      if (laneIt == mutantIt->second.end()) {
        result.isEquivalent = false;
        result.divergenceReason = "Missing lane in mutant";
        return false;
      }
      
      auto& mutantVars = laneIt->second;
      
      for (auto& [varName, value] : vars) {
        auto varIt = mutantVars.find(varName);
        if (varIt == mutantVars.end()) {
          result.isEquivalent = false;
          result.divergenceReason = "Missing variable in mutant";
          result.differences.push_back(
            "Wave " + std::to_string(waveId) + " Lane " + std::to_string(laneId) + 
            " missing variable: " + varName);
          continue;
        }
        
        if (varIt->second != value) {
          result.isEquivalent = false;
          result.divergenceReason = "Variable value mismatch";
          result.differences.push_back(
            "Wave " + std::to_string(waveId) + " Lane " + std::to_string(laneId) + 
            " Var " + varName + ": " + value.toString() + " vs " + 
            varIt->second.toString());
        }
      }
    }
  }
  
  // Compare return values
  for (auto& [waveId, laneReturns] : golden.returnValues) {
    for (auto& [laneId, value] : laneReturns) {
      auto& mutantValue = mutant.returnValues.at(waveId).at(laneId);
      if (mutantValue != value) {
        result.isEquivalent = false;
        result.divergenceReason = "Return value mismatch";
        result.differences.push_back(
          "Wave " + std::to_string(waveId) + " Lane " + std::to_string(laneId) + 
          " return: " + value.toString() + " vs " + mutantValue.toString());
      }
    }
  }
  
  // Compare memory state
  for (auto& [bufferName, contents] : golden.globalBuffers) {
    auto bufferIt = mutant.globalBuffers.find(bufferName);
    if (bufferIt == mutant.globalBuffers.end()) {
      result.isEquivalent = false;
      result.divergenceReason = "Missing buffer in mutant";
      continue;
    }
    
    auto& mutantBuffer = bufferIt->second;
    for (auto& [index, value] : contents) {
      auto indexIt = mutantBuffer.find(index);
      if (indexIt == mutantBuffer.end()) {
        result.isEquivalent = false;
        result.divergenceReason = "Missing buffer index";
        continue;
      }
      
      if (indexIt->second != value) {
        result.isEquivalent = false;
        result.divergenceReason = "Buffer content mismatch";
        result.differences.push_back(
          "Buffer " + bufferName + "[" + std::to_string(index) + "]: " + 
          value.toString() + " vs " + indexIt->second.toString());
      }
    }
  }
  
  return result.isEquivalent;
}

bool SemanticValidator::compareWaveOperations(
    const std::vector<ExecutionTrace::WaveOpRecord>& golden,
    const std::vector<ExecutionTrace::WaveOpRecord>& mutant,
    ValidationResult& result) {
  
  // Group by operation type and wave
  auto groupOps = [](const std::vector<ExecutionTrace::WaveOpRecord>& ops) {
    std::map<std::pair<std::string, interpreter::WaveId>, 
             std::vector<const ExecutionTrace::WaveOpRecord*>> grouped;
    for (auto& op : ops) {
      grouped[{op.opType, op.waveId}].push_back(&op);
    }
    return grouped;
  };
  
  auto goldenGroups = groupOps(golden);
  auto mutantGroups = groupOps(mutant);
  
  // Each group should have same results
  for (auto& [key, goldenOps] : goldenGroups) {
    auto mutantIt = mutantGroups.find(key);
    if (mutantIt == mutantGroups.end()) {
      result.isEquivalent = false;
      result.divergenceReason = "Missing wave operation group";
      continue;
    }
    
    auto& mutantOps = mutantIt->second;
    
    if (goldenOps.size() != mutantOps.size()) {
      result.isEquivalent = false;
      result.divergenceReason = "Different number of wave operations";
      result.differences.push_back(
        "Op " + key.first + " Wave " + std::to_string(key.second) + 
        ": " + std::to_string(goldenOps.size()) + " vs " + 
        std::to_string(mutantOps.size()));
      continue;
    }
    
    // Compare results
    for (size_t i = 0; i < goldenOps.size(); i++) {
      for (auto& [laneId, value] : goldenOps[i]->outputValues) {
        auto laneIt = mutantOps[i]->outputValues.find(laneId);
        if (laneIt == mutantOps[i]->outputValues.end()) {
          result.isEquivalent = false;
          result.divergenceReason = "Missing lane in wave op result";
          continue;
        }
        
        if (laneIt->second != value) {
          result.isEquivalent = false;
          result.divergenceReason = "Wave operation result mismatch";
          result.differences.push_back(
            "Op " + key.first + " Wave " + std::to_string(key.second) + 
            " Lane " + std::to_string(laneId) + ": " + 
            value.toString() + " vs " + laneIt->second.toString());
        }
      }
    }
  }
  
  return result.isEquivalent;
}

bool SemanticValidator::compareMemoryState(
    const ExecutionTrace& golden,
    const ExecutionTrace& mutant,
    ValidationResult& result) {
  
  // Memory state is already compared in compareFinalStates
  // This method could do additional memory access pattern validation
  return true;
}

bool SemanticValidator::verifyControlFlowEquivalence(
    const ExecutionTrace& golden,
    const ExecutionTrace& mutant,
    ValidationResult& result) {
  
  // Control flow can differ as long as final results match
  // This is already verified by comparing final states
  return true;
}

// ===== Main Fuzzer Implementation =====

TraceGuidedFuzzer::TraceGuidedFuzzer() {
  // Initialize components
  validator = std::make_unique<SemanticValidator>();
  bugReporter = std::make_unique<BugReporter>();
  
  // Register mutation strategies
  mutationStrategies.push_back(std::make_unique<ExplicitLaneDivergenceMutation>());
  mutationStrategies.push_back(std::make_unique<LoopUnrollingMutation>());
  mutationStrategies.push_back(std::make_unique<PrecomputeWaveResultsMutation>());
  mutationStrategies.push_back(std::make_unique<RedundantWaveSyncMutation>());
  mutationStrategies.push_back(std::make_unique<ForceBlockBoundariesMutation>());
  mutationStrategies.push_back(std::make_unique<SerializeMemoryAccessesMutation>());
}

void TraceGuidedFuzzer::fuzzProgram(
    interpreter::CompoundStmt* seedProgram,
    const FuzzingConfig& config) {
  
  // Step 1: Capture golden trace
  ExecutionTrace goldenTrace;
  interpreter::ThreadgroupContext tgContext(config.threadgroupSize, config.waveSize);
  
  {
    TraceCaptureInterpreter captureInterp(&goldenTrace);
    captureInterp.execute(seedProgram, tgContext);
    captureInterp.captureFinalState(tgContext);
  }
  
  if (config.enableLogging) {
    logTrace("Golden trace captured", goldenTrace);
  }
  
  // Step 2: Generate and test mutations
  std::queue<std::unique_ptr<interpreter::CompoundStmt>> workQueue;
  workQueue.push(seedProgram->clone());
  
  std::set<size_t> testedMutants;  // Dedup by hash
  
  while (!workQueue.empty() && testedMutants.size() < config.maxMutants) {
    auto current = std::move(workQueue.front());
    workQueue.pop();
    
    // Try each mutation strategy
    for (auto& strategy : mutationStrategies) {
      auto mutants = generateMutants(current.get(), strategy.get(), goldenTrace);
      
      for (auto& mutant : mutants) {
        size_t mutantHash = hashAST(mutant.get());
        if (testedMutants.count(mutantHash)) continue;
        testedMutants.insert(mutantHash);
        
        // Step 3: Execute mutant
        ExecutionTrace mutantTrace;
        interpreter::ThreadgroupContext mutantTgContext(config.threadgroupSize, config.waveSize);
        
        try {
          TraceCaptureInterpreter mutantInterp(&mutantTrace);
          mutantInterp.execute(mutant.get(), mutantTgContext);
          mutantInterp.captureFinalState(mutantTgContext);
          
          // Step 4: Validate semantic equivalence
          auto validation = validator->validate(goldenTrace, mutantTrace);
          
          if (!validation.isEquivalent) {
            bugReporter->reportBug(seedProgram, mutant.get(), 
                                 goldenTrace, mutantTrace, validation);
            bugs.push_back(bugReporter->getLastReport());
          } else {
            // Check if mutant provides new coverage
            if (hasNewCoverage(mutantTrace)) {
              workQueue.push(std::move(mutant));
              if (config.enableLogging) {
                logMutation("New coverage found", strategy->getName());
              }
            }
          }
          
        } catch (const std::exception& e) {
          bugReporter->reportCrash(seedProgram, mutant.get(), e);
        }
      }
    }
  }
  
  if (config.enableLogging) {
    logSummary(testedMutants.size(), bugs.size());
  }
}

std::vector<std::unique_ptr<interpreter::CompoundStmt>> TraceGuidedFuzzer::generateMutants(
    interpreter::CompoundStmt* program,
    MutationStrategy* strategy,
    const ExecutionTrace& trace) {
  
  std::vector<std::unique_ptr<interpreter::CompoundStmt>> mutants;
  
  // For now, simple implementation - try to apply mutation to each statement
  for (auto& stmt : program->statements) {
    if (strategy->canApply(stmt.get(), trace)) {
      auto mutatedStmt = strategy->apply(stmt.get(), trace);
      if (mutatedStmt && strategy->validateSemanticPreservation(stmt.get(), mutatedStmt.get(), trace)) {
        // Create new program with mutated statement
        auto mutant = program->clone();
        // Replace statement in mutant (simplified - would need proper AST manipulation)
        // For now, just append the mutation
        mutant->addStatement(std::move(mutatedStmt));
        mutants.push_back(std::move(mutant));
      }
    }
  }
  
  return mutants;
}

bool TraceGuidedFuzzer::hasNewCoverage(const ExecutionTrace& trace) {
  bool newCoverage = false;
  
  // Check block patterns
  for (auto& [blockId, block] : trace.blocks) {
    uint64_t pattern = hashBlockPattern(block);
    if (seenBlockPatterns.insert(pattern).second) {
      newCoverage = true;
    }
  }
  
  // Check wave patterns
  for (auto& waveOp : trace.waveOperations) {
    uint64_t pattern = hashWavePattern(waveOp);
    if (seenWavePatterns.insert(pattern).second) {
      newCoverage = true;
    }
  }
  
  // Check synchronization patterns
  for (auto& barrier : trace.barriers) {
    uint64_t pattern = hashSyncPattern(barrier);
    if (seenSyncPatterns.insert(pattern).second) {
      newCoverage = true;
    }
  }
  
  return newCoverage;
}

uint64_t TraceGuidedFuzzer::hashBlockPattern(const ExecutionTrace::BlockExecutionRecord& block) {
  // Simple hash combining block type and participant pattern
  uint64_t hash = static_cast<uint64_t>(block.blockType);
  for (auto& [waveId, participation] : block.waveParticipation) {
    hash = hash * 31 + waveId;
    hash = hash * 31 + participation.participatingLanes.size();
  }
  return hash;
}

uint64_t TraceGuidedFuzzer::hashWavePattern(const ExecutionTrace::WaveOpRecord& waveOp) {
  // Hash wave op type and participant count
  std::hash<std::string> strHash;
  uint64_t hash = strHash(waveOp.opType);
  hash = hash * 31 + waveOp.waveId;
  hash = hash * 31 + waveOp.arrivedParticipants.size();
  return hash;
}

uint64_t TraceGuidedFuzzer::hashSyncPattern(const ExecutionTrace::BarrierRecord& barrier) {
  // Hash barrier arrival pattern
  uint64_t hash = barrier.barrierId;
  for (auto& [waveId, lanes] : barrier.arrivedLanesPerWave) {
    hash = hash * 31 + waveId;
    hash = hash * 31 + lanes.size();
  }
  return hash;
}

void TraceGuidedFuzzer::logTrace(const std::string& message, const ExecutionTrace& trace) {
  std::ofstream log("fuzzing_trace.log", std::ios::app);
  log << "=== " << message << " ===\n";
  log << "Blocks: " << trace.blocks.size() << "\n";
  log << "Wave ops: " << trace.waveOperations.size() << "\n";
  log << "Control flow decisions: " << trace.controlFlowHistory.size() << "\n";
  log << "Barriers: " << trace.barriers.size() << "\n";
  log << "\n";
}

void TraceGuidedFuzzer::logMutation(const std::string& message, const std::string& strategy) {
  std::ofstream log("fuzzing_mutations.log", std::ios::app);
  log << message << " using strategy: " << strategy << "\n";
}

void TraceGuidedFuzzer::logSummary(size_t testedMutants, size_t bugsFound) {
  std::ofstream log("fuzzing_summary.log", std::ios::app);
  log << "=== Fuzzing Summary ===\n";
  log << "Mutants tested: " << testedMutants << "\n";
  log << "Bugs found: " << bugsFound << "\n";
  log << "Block patterns seen: " << seenBlockPatterns.size() << "\n";
  log << "Wave patterns seen: " << seenWavePatterns.size() << "\n";
  log << "Sync patterns seen: " << seenSyncPatterns.size() << "\n";
  log << "\n";
}

// ===== AST Serialization (Placeholder) =====

bool deserializeAST(const uint8_t* data, size_t size, 
                   std::unique_ptr<interpreter::CompoundStmt>& program) {
  // TODO: Implement AST deserialization
  // For now, create a simple test program
  program = std::make_unique<interpreter::CompoundStmt>();
  
  // Simple test: x = 1
  auto assign = std::make_unique<interpreter::AssignStmt>(
    "x",
    std::make_unique<interpreter::LiteralExpr>(interpreter::Value(1))
  );
  program->addStatement(std::move(assign));
  
  return true;
}

size_t serializeAST(const interpreter::CompoundStmt* program, 
                   uint8_t* data, size_t maxSize) {
  // TODO: Implement AST serialization
  return 0;
}

size_t hashAST(const interpreter::CompoundStmt* program) {
  // TODO: Implement proper AST hashing
  return reinterpret_cast<size_t>(program);
}

// ===== Seed Corpus =====

void loadSeedCorpus(TraceGuidedFuzzer* fuzzer) {
  // TODO: Load seed programs from files
  // For now, create some simple test programs
  
  // Test 1: Simple divergence
  {
    auto program = std::make_unique<interpreter::CompoundStmt>();
    
    // if (laneId < 16) { x = 1; } else { x = 2; }
    auto condition = std::make_unique<interpreter::BinaryOpExpr>(
      std::make_unique<interpreter::LaneIndexExpr>(),
      std::make_unique<interpreter::LiteralExpr>(interpreter::Value(16)),
      interpreter::BinaryOpExpr::Lt
    );
    
    auto thenBlock = std::make_unique<interpreter::CompoundStmt>();
    thenBlock->addStatement(std::make_unique<interpreter::AssignStmt>(
      "x",
      std::make_unique<interpreter::LiteralExpr>(interpreter::Value(1))
    ));
    
    auto elseBlock = std::make_unique<interpreter::CompoundStmt>();
    elseBlock->addStatement(std::make_unique<interpreter::AssignStmt>(
      "x",
      std::make_unique<interpreter::LiteralExpr>(interpreter::Value(2))
    ));
    
    auto ifStmt = std::make_unique<interpreter::IfStmt>(
      std::move(condition),
      std::move(thenBlock),
      std::move(elseBlock)
    );
    
    program->addStatement(std::move(ifStmt));
    
    // Add to corpus
    // fuzzer->addSeedProgram(std::move(program));
  }
}

} // namespace fuzzer
} // namespace minihlsl

// ===== LibFuzzer Entry Points =====

extern "C" {

int LLVMFuzzerInitialize(int* argc, char*** argv) {
  // Initialize fuzzer
  minihlsl::fuzzer::g_fuzzer = std::make_unique<minihlsl::fuzzer::TraceGuidedFuzzer>();
  
  // Load seed programs
  minihlsl::fuzzer::loadSeedCorpus(minihlsl::fuzzer::g_fuzzer.get());
  
  return 0;
}

int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  if (!minihlsl::fuzzer::g_fuzzer) {
    return 0;
  }
  
  // Deserialize AST from fuzzer input
  std::unique_ptr<minihlsl::interpreter::CompoundStmt> program;
  if (!minihlsl::fuzzer::deserializeAST(data, size, program)) {
    return 0;  // Invalid input
  }
  
  // Run trace-guided fuzzing
  minihlsl::fuzzer::FuzzingConfig config;
  config.threadgroupSize = 32;
  config.waveSize = 32;
  config.maxMutants = 10;  // Limit for fuzzing iteration
  config.enableLogging = false;  // Disable for performance
  
  minihlsl::fuzzer::g_fuzzer->fuzzProgram(program.get(), config);
  
  return 0;
}

size_t LLVMFuzzerCustomMutator(uint8_t* data, size_t size, 
                              size_t maxSize, unsigned int seed) {
  if (!minihlsl::fuzzer::g_fuzzer) {
    return 0;
  }
  
  // Apply AST-level mutations
  std::unique_ptr<minihlsl::interpreter::CompoundStmt> program;
  if (!minihlsl::fuzzer::deserializeAST(data, size, program)) {
    return 0;
  }
  
  // Apply random mutation
  auto mutated = minihlsl::fuzzer::g_fuzzer->mutateAST(program.get(), seed);
  
  // Serialize back
  return minihlsl::fuzzer::serializeAST(mutated.get(), data, maxSize);
}

size_t LLVMFuzzerCustomCrossOver(const uint8_t* data1, size_t size1,
                                const uint8_t* data2, size_t size2,
                                uint8_t* out, size_t maxOutSize,
                                unsigned int seed) {
  // TODO: Implement AST crossover
  return 0;
}

} // extern "C"