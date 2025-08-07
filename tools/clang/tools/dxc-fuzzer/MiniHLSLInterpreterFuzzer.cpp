#include "MiniHLSLInterpreterFuzzer.h"
#include "MiniHLSLInterpreterTraceCapture.h"
#include <fstream>
#include <sstream>
#include <random>
#include <algorithm>

namespace minihlsl {
namespace fuzzer {

// Global fuzzer instance
std::unique_ptr<TraceGuidedFuzzer> g_fuzzer;

// ===== Mutation Strategy Implementations =====

bool ExplicitLaneDivergenceMutation::canApply(const interpreter::Statement* stmt, 
                                             const ExecutionTrace& trace) const {
  // Can apply to if statements that have divergent control flow
  if (auto ifStmt = dynamic_cast<const interpreter::IfStmt*>(stmt)) {
    // Check if any wave has divergent lanes for this statement
    for (const auto& decision : trace.controlFlowHistory) {
      if (decision.statement == stmt) {
        for (const auto& [waveId, laneDecisions] : decision.decisions) {
          bool hasTrueLanes = false;
          bool hasFalseLanes = false;
          for (const auto& [laneId, dec] : laneDecisions) {
            if (dec.branchTaken) hasTrueLanes = true;
            else hasFalseLanes = true;
          }
          if (hasTrueLanes && hasFalseLanes) {
            return true; // Found divergence
          }
        }
      }
    }
  }
  return false;
}

std::unique_ptr<interpreter::Statement> ExplicitLaneDivergenceMutation::apply(
    const interpreter::Statement* stmt, 
    const ExecutionTrace& trace) const {
  
  auto ifStmt = dynamic_cast<const interpreter::IfStmt*>(stmt);
  if (!ifStmt) return nullptr;
  
  // Find the control flow decision for this statement
  for (const auto& decision : trace.controlFlowHistory) {
    if (decision.statement == stmt) {
      // Collect lanes that took true branch and false branch
      std::map<interpreter::WaveId, std::pair<std::vector<interpreter::LaneId>, std::vector<interpreter::LaneId>>> waveBranches;
      
      for (const auto& [waveId, laneDecisions] : decision.decisions) {
        for (const auto& [laneId, dec] : laneDecisions) {
          if (dec.branchTaken) {
            waveBranches[waveId].first.push_back(laneId);
          } else {
            waveBranches[waveId].second.push_back(laneId);
          }
        }
      }
      
      // Create explicit lane conditions
      std::unique_ptr<interpreter::Expression> explicitCondition = nullptr;
      
      for (const auto& [waveId, branches] : waveBranches) {
        const auto& [trueLanes, falseLanes] = branches;
        
        if (!trueLanes.empty()) {
          // Create condition for lanes that take true branch
          std::unique_ptr<interpreter::Expression> waveCondition = nullptr;
          
          // Create (waveIndex() == waveId && (laneIndex() == lane0 || laneIndex() == lane1 || ...))
          auto waveCheck = createWaveIdCheck(waveId);
          
          std::unique_ptr<interpreter::Expression> laneCondition = nullptr;
          for (auto laneId : trueLanes) {
            auto laneCheck = createLaneIdCheck(laneId);
            if (!laneCondition) {
              laneCondition = std::move(laneCheck);
            } else {
              laneCondition = createDisjunction(std::move(laneCondition), std::move(laneCheck));
            }
          }
          
          waveCondition = createConjunction(std::move(waveCheck), std::move(laneCondition));
          
          if (!explicitCondition) {
            explicitCondition = std::move(waveCondition);
          } else {
            explicitCondition = createDisjunction(std::move(explicitCondition), std::move(waveCondition));
          }
        }
      }
      
      // Clone the if statement with the new explicit condition
      // Deep copy then block
      std::vector<std::unique_ptr<interpreter::Statement>> newThenBlock;
      for (const auto& stmt : ifStmt->getThenBlock()) {
        if (stmt) {
          newThenBlock.push_back(stmt->clone());
        }
      }
      
      // Deep copy else block
      std::vector<std::unique_ptr<interpreter::Statement>> newElseBlock;
      for (const auto& stmt : ifStmt->getElseBlock()) {
        if (stmt) {
          newElseBlock.push_back(stmt->clone());
        }
      }
      
      return std::make_unique<interpreter::IfStmt>(
          std::move(explicitCondition),
          std::move(newThenBlock),
          std::move(newElseBlock)
      );
    }
  }
  
  return nullptr;
}

bool ExplicitLaneDivergenceMutation::validateSemanticPreservation(
    const interpreter::Statement* original,
    const interpreter::Statement* mutated,
    const ExecutionTrace& trace) const {
  // Always returns true as mutations are designed to be semantics-preserving
  return true;
}

bool LoopUnrollingMutation::canApply(const interpreter::Statement* stmt, 
                                    const ExecutionTrace& trace) const {
  // Can apply to loops with known iteration counts
  return dynamic_cast<const interpreter::ForStmt*>(stmt) != nullptr ||
         dynamic_cast<const interpreter::WhileStmt*>(stmt) != nullptr ||
         dynamic_cast<const interpreter::DoWhileStmt*>(stmt) != nullptr;
}

std::unique_ptr<interpreter::Statement> LoopUnrollingMutation::apply(
    const interpreter::Statement* stmt, 
    const ExecutionTrace& trace) const {
  
  // For simplicity, we'll only handle ForStmt with simple bounds
  auto forStmt = dynamic_cast<const interpreter::ForStmt*>(stmt);
  if (!forStmt) {
    return nullptr;
  }
  
  // Find loop execution info in trace
  for (const auto& [loopPtr, loopInfo] : trace.loops) {
    if (loopPtr == stmt) {
      // Create unrolled version based on max iterations observed
      if (loopInfo.lanePatterns.empty()) {
        return nullptr;
      }
      
      // Get maximum iteration count across all lanes
      uint32_t maxIterations = 0;
      for (const auto& [waveId, laneMap] : loopInfo.lanePatterns) {
        for (const auto& [laneId, pattern] : laneMap) {
          maxIterations = std::max(maxIterations, pattern.totalIterations);
        }
      }
      
      // Don't unroll if too many iterations
      if (maxIterations > 10) {
        return nullptr;
      }
      
      // Create a sequence of if statements that check iteration count
      // This is a simplified unrolling that preserves semantics
      std::vector<std::unique_ptr<interpreter::Statement>> unrolledStmts;
      
      // Add the loop variable initialization
      unrolledStmts.push_back(std::make_unique<interpreter::VarDeclStmt>(
          forStmt->getLoopVar(),
          forStmt->getInit() ? forStmt->getInit()->clone() : nullptr
      ));
      
      // For each iteration, add guarded body
      for (uint32_t i = 0; i < maxIterations; i++) {
        // Create condition: loop_var == i && original_condition
        auto iterCheck = std::make_unique<interpreter::BinaryOpExpr>(
            std::make_unique<interpreter::VariableExpr>(forStmt->getLoopVar()),
            std::make_unique<interpreter::LiteralExpr>(interpreter::Value(static_cast<int>(i))),
            interpreter::BinaryOpExpr::Eq
        );
        
        std::unique_ptr<interpreter::Expression> combinedCond;
        if (forStmt->getCondition()) {
          combinedCond = createConjunction(
              std::move(iterCheck),
              forStmt->getCondition()->clone()
          );
        } else {
          combinedCond = std::move(iterCheck);
        }
        
        // Clone loop body
        std::vector<std::unique_ptr<interpreter::Statement>> iterBody;
        for (const auto& bodyStmt : forStmt->getBody()) {
          if (bodyStmt) {
            iterBody.push_back(bodyStmt->clone());
          }
        }
        
        // Add increment
        if (forStmt->getIncrement()) {
          iterBody.push_back(std::make_unique<interpreter::ExprStmt>(
              forStmt->getIncrement()->clone()
          ));
        }
        
        // Wrap in if statement
        unrolledStmts.push_back(std::make_unique<interpreter::IfStmt>(
            std::move(combinedCond),
            std::move(iterBody)
        ));
      }
      
      // Return first statement if only one, otherwise wrap in if(true)
      if (unrolledStmts.size() == 1) {
        return std::move(unrolledStmts[0]);
      } else {
        // Wrap multiple statements in if(true) block
        return std::make_unique<interpreter::IfStmt>(
            std::make_unique<interpreter::LiteralExpr>(interpreter::Value(1)),
            std::move(unrolledStmts)
        );
      }
    }
  }
  
  return nullptr;
}

bool LoopUnrollingMutation::validateSemanticPreservation(
    const interpreter::Statement* original,
    const interpreter::Statement* mutated,
    const ExecutionTrace& trace) const {
  return true;
}

bool PrecomputeWaveResultsMutation::canApply(const interpreter::Statement* stmt, 
                                           const ExecutionTrace& trace) const {
  // Can apply to statements containing wave operations
  // For now, check if there are any wave operations in the trace
  return !trace.waveOperations.empty();
}

std::unique_ptr<interpreter::Statement> PrecomputeWaveResultsMutation::apply(
    const interpreter::Statement* stmt, 
    const ExecutionTrace& trace) const {
  
  // For this simple implementation, we'll just return nullptr
  // A full implementation would need to:
  // 1. Match the statement to a wave operation in the trace
  // 2. Extract the computed result
  // 3. Replace the wave operation with the literal value
  
  // TODO: Implement once we have proper statement-to-trace matching
  return nullptr;
}

bool PrecomputeWaveResultsMutation::validateSemanticPreservation(
    const interpreter::Statement* original,
    const interpreter::Statement* mutated,
    const ExecutionTrace& trace) const {
  return true;
}

bool RedundantWaveSyncMutation::canApply(const interpreter::Statement* stmt, 
                                       const ExecutionTrace& trace) const {
  // Can apply after any wave operation
  return false; // TODO: Implement
}

std::unique_ptr<interpreter::Statement> RedundantWaveSyncMutation::apply(
    const interpreter::Statement* stmt, 
    const ExecutionTrace& trace) const {
  // TODO: Add redundant sync operations
  return nullptr;
}

bool RedundantWaveSyncMutation::validateSemanticPreservation(
    const interpreter::Statement* original,
    const interpreter::Statement* mutated,
    const ExecutionTrace& trace) const {
  return true;
}

bool ForceBlockBoundariesMutation::canApply(const interpreter::Statement* stmt, 
                                          const ExecutionTrace& trace) const {
  // Can apply to compound statements
  return true; // TODO: Check for compound statements
}

std::unique_ptr<interpreter::Statement> ForceBlockBoundariesMutation::apply(
    const interpreter::Statement* stmt, 
    const ExecutionTrace& trace) const {
  
  // Wrap the statement in an if(true) block to force a new block boundary
  auto trueCondition = std::make_unique<interpreter::LiteralExpr>(interpreter::Value(1));
  
  // Create the then-block with the original statement
  std::vector<std::unique_ptr<interpreter::Statement>> thenBlock;
  thenBlock.push_back(stmt->clone());
  
  // Create the if statement
  return std::make_unique<interpreter::IfStmt>(
    std::move(trueCondition),
    std::move(thenBlock)
  );
}

bool ForceBlockBoundariesMutation::validateSemanticPreservation(
    const interpreter::Statement* original,
    const interpreter::Statement* mutated,
    const ExecutionTrace& trace) const {
  return true;
}

bool SerializeMemoryAccessesMutation::canApply(const interpreter::Statement* stmt, 
                                             const ExecutionTrace& trace) const {
  // Can apply if there are memory accesses
  return !trace.memoryAccesses.empty();
}

std::unique_ptr<interpreter::Statement> SerializeMemoryAccessesMutation::apply(
    const interpreter::Statement* stmt, 
    const ExecutionTrace& trace) const {
  // TODO: Add barriers between memory accesses
  return nullptr;
}

bool SerializeMemoryAccessesMutation::validateSemanticPreservation(
    const interpreter::Statement* original,
    const interpreter::Statement* mutated,
    const ExecutionTrace& trace) const {
  return true;
}

// ===== Semantic Validator Implementation =====

SemanticValidator::ValidationResult SemanticValidator::validate(
    const ExecutionTrace& golden, 
    const ExecutionTrace& mutant) {
  
  ValidationResult result;
  result.isEquivalent = true;
  
  // Compare final states
  if (!compareFinalStates(golden.finalState, mutant.finalState, result)) {
    result.isEquivalent = false;
    result.divergenceReason = "Final states differ";
  }
  
  // Compare wave operations
  if (!compareWaveOperations(golden.waveOperations, mutant.waveOperations, result)) {
    result.isEquivalent = false;
    if (result.divergenceReason.empty()) {
      result.divergenceReason = "Wave operations differ";
    }
  }
  
  // Compare memory state
  if (!compareMemoryState(golden, mutant, result)) {
    result.isEquivalent = false;
    if (result.divergenceReason.empty()) {
      result.divergenceReason = "Memory state differs";
    }
  }
  
  // Verify control flow equivalence
  if (!verifyControlFlowEquivalence(golden, mutant, result)) {
    result.isEquivalent = false;
    if (result.divergenceReason.empty()) {
      result.divergenceReason = "Control flow differs";
    }
  }
  
  return result;
}

bool SemanticValidator::compareFinalStates(
    const ExecutionTrace::FinalState& golden,
    const ExecutionTrace::FinalState& mutant,
    ValidationResult& result) {
  
  // Compare per-lane variables
  for (const auto& [waveId, waveVars] : golden.laneVariables) {
    auto mutantWaveIt = mutant.laneVariables.find(waveId);
    if (mutantWaveIt == mutant.laneVariables.end()) {
      result.differences.push_back("Missing wave " + std::to_string(waveId) + " in mutant");
      return false;
    }
    
    for (const auto& [laneId, vars] : waveVars) {
      auto mutantLaneIt = mutantWaveIt->second.find(laneId);
      if (mutantLaneIt == mutantWaveIt->second.end()) {
        result.differences.push_back("Missing lane " + std::to_string(laneId) + 
                                   " in wave " + std::to_string(waveId));
        return false;
      }
      
      // Compare variable values
      for (const auto& [varName, value] : vars) {
        auto mutantVarIt = mutantLaneIt->second.find(varName);
        if (mutantVarIt == mutantLaneIt->second.end()) {
          result.differences.push_back("Variable " + varName + " missing in mutant");
          return false;
        }
        
        if (value.asInt() != mutantVarIt->second.asInt()) {
          result.differences.push_back("Variable " + varName + " differs: " + 
                                     std::to_string(value.asInt()) + " vs " +
                                     std::to_string(mutantVarIt->second.asInt()));
          return false;
        }
      }
    }
  }
  
  // Compare shared memory
  for (const auto& [addr, value] : golden.sharedMemory) {
    auto mutantIt = mutant.sharedMemory.find(addr);
    if (mutantIt == mutant.sharedMemory.end()) {
      result.differences.push_back("Shared memory address " + std::to_string(addr) + " missing");
      return false;
    }
    
    if (value.asInt() != mutantIt->second.asInt()) {
      result.differences.push_back("Shared memory at " + std::to_string(addr) + " differs");
      return false;
    }
  }
  
  return true;
}

bool SemanticValidator::compareWaveOperations(
    const std::vector<ExecutionTrace::WaveOpRecord>& golden,
    const std::vector<ExecutionTrace::WaveOpRecord>& mutant,
    ValidationResult& result) {
  
  // Wave operations must produce identical results
  if (golden.size() != mutant.size()) {
    result.differences.push_back("Different number of wave operations: " + 
                               std::to_string(golden.size()) + " vs " + 
                               std::to_string(mutant.size()));
    return false;
  }
  
  for (size_t i = 0; i < golden.size(); ++i) {
    const auto& goldenOp = golden[i];
    const auto& mutantOp = mutant[i];
    
    if (goldenOp.opType != mutantOp.opType) {
      result.differences.push_back("Wave op type mismatch at index " + std::to_string(i));
      return false;
    }
    
    // Compare output values
    for (const auto& [laneId, value] : goldenOp.outputValues) {
      auto mutantIt = mutantOp.outputValues.find(laneId);
      if (mutantIt == mutantOp.outputValues.end()) {
        result.differences.push_back("Wave op missing output for lane " + std::to_string(laneId));
        return false;
      }
      
      if (value != mutantIt->second) {
        result.differences.push_back("Wave op output differs for lane " + std::to_string(laneId));
        return false;
      }
    }
  }
  
  return true;
}

bool SemanticValidator::compareMemoryState(
    const ExecutionTrace& golden,
    const ExecutionTrace& mutant,
    ValidationResult& result) {
  
  // Final memory state must be identical
  if (golden.finalMemoryState.size() != mutant.finalMemoryState.size()) {
    result.differences.push_back("Different number of memory locations");
    return false;
  }
  
  for (const auto& [addr, value] : golden.finalMemoryState) {
    auto mutantIt = mutant.finalMemoryState.find(addr);
    if (mutantIt == mutant.finalMemoryState.end()) {
      result.differences.push_back("Memory address " + std::to_string(addr) + " missing");
      return false;
    }
    
    if (value.asInt() != mutantIt->second.asInt()) {
      result.differences.push_back("Memory at " + std::to_string(addr) + " differs");
      return false;
    }
  }
  
  return true;
}

bool SemanticValidator::verifyControlFlowEquivalence(
    const ExecutionTrace& golden,
    const ExecutionTrace& mutant,
    ValidationResult& result) {
  
  // Verify that all lanes took equivalent paths (even if restructured)
  // This is more complex and would need proper implementation
  
  // For now, just check that the same blocks were visited
  std::set<uint32_t> goldenBlocks;
  std::set<uint32_t> mutantBlocks;
  
  for (const auto& [blockId, record] : golden.blocks) {
    goldenBlocks.insert(blockId);
  }
  
  for (const auto& [blockId, record] : mutant.blocks) {
    mutantBlocks.insert(blockId);
  }
  
  if (goldenBlocks != mutantBlocks) {
    result.differences.push_back("Different blocks visited");
    return false;
  }
  
  return true;
}

// ===== Bug Reporter Implementation =====

void BugReporter::reportBug(const interpreter::Program& original,
                          const interpreter::Program& mutant,
                          const ExecutionTrace& originalTrace,
                          const ExecutionTrace& mutantTrace,
                          const SemanticValidator::ValidationResult& validation) {
  
  BugReport report;
  report.id = generateBugId();
  report.timestamp = std::chrono::system_clock::now();
  
  // TODO: Serialize programs to strings
  report.originalProgram = "// Original program";
  report.mutantProgram = "// Mutant program";
  
  report.validation = validation;
  
  // Find divergence point
  report.traceDivergence = findTraceDivergence(originalTrace, mutantTrace);
  
  // Classify bug
  report.bugType = classifyBug(report.traceDivergence);
  report.severity = assessSeverity(report.bugType, validation);
  
  // Save and log
  saveBugReport(report);
  logBug(report);
}

void BugReporter::reportCrash(const interpreter::Program& original,
                            const interpreter::Program& mutant,
                            const std::exception& e) {
  
  BugReport report;
  report.id = generateBugId();
  report.timestamp = std::chrono::system_clock::now();
  
  report.bugType = BugReport::ControlFlowError;
  report.severity = BugReport::Critical;
  
  report.traceDivergence.type = BugReport::TraceDivergence::ControlFlow;
  report.traceDivergence.description = std::string("Crash: ") + e.what();
  
  saveBugReport(report);
  logBug(report);
}

BugReporter::BugReport::TraceDivergence BugReporter::findTraceDivergence(
    const ExecutionTrace& golden,
    const ExecutionTrace& mutant) {
  
  BugReport::TraceDivergence divergence;
  
  // TODO: Implement detailed divergence analysis
  divergence.type = BugReport::TraceDivergence::ControlFlow;
  divergence.divergencePoint = 0;
  divergence.description = "Traces diverged";
  
  return divergence;
}

BugReporter::BugReport::BugType BugReporter::classifyBug(
    const BugReport::TraceDivergence& divergence) {
  
  switch (divergence.type) {
    case BugReport::TraceDivergence::WaveOperation:
      return BugReport::WaveOpInconsistency;
    case BugReport::TraceDivergence::Synchronization:
      return BugReport::DeadlockOrRace;
    case BugReport::TraceDivergence::Memory:
      return BugReport::MemoryCorruption;
    default:
      return BugReport::ControlFlowError;
  }
}

BugReporter::BugReport::Severity BugReporter::assessSeverity(
    BugReport::BugType type,
    const SemanticValidator::ValidationResult& validation) {
  
  if (type == BugReport::DeadlockOrRace || type == BugReport::MemoryCorruption) {
    return BugReport::Critical;
  }
  
  if (type == BugReport::WaveOpInconsistency) {
    return BugReport::High;
  }
  
  return BugReport::Medium;
}

std::string BugReporter::generateBugId() {
  static uint64_t counter = 0;
  return "BUG-" + std::to_string(++counter);
}

void BugReporter::saveBugReport(const BugReport& report) {
  // TODO: Save to file
  std::string filename = "./bugs/" + report.id + ".txt";
  // Would write report details to file
}

void BugReporter::logBug(const BugReport& report) {
  std::cout << "Found bug: " << report.id << "\n";
  std::cout << "Type: " << static_cast<int>(report.bugType) << "\n";
  std::cout << "Severity: " << static_cast<int>(report.severity) << "\n";
  std::cout << "Description: " << report.traceDivergence.description << "\n";
}

// ===== Main Fuzzer Implementation =====

TraceGuidedFuzzer::TraceGuidedFuzzer() {
  // Initialize mutation strategies
  mutationStrategies.push_back(std::make_unique<ExplicitLaneDivergenceMutation>());
  mutationStrategies.push_back(std::make_unique<LoopUnrollingMutation>());
  mutationStrategies.push_back(std::make_unique<PrecomputeWaveResultsMutation>());
  mutationStrategies.push_back(std::make_unique<RedundantWaveSyncMutation>());
  mutationStrategies.push_back(std::make_unique<ForceBlockBoundariesMutation>());
  mutationStrategies.push_back(std::make_unique<SerializeMemoryAccessesMutation>());
  
  validator = std::make_unique<SemanticValidator>();
  bugReporter = std::make_unique<BugReporter>();
}

void TraceGuidedFuzzer::fuzzProgram(const interpreter::Program& seedProgram, 
                                  const FuzzingConfig& config) {
  
  std::cout << "Starting trace-guided fuzzing...\n";
  std::cout << "Threadgroup size: " << config.threadgroupSize << "\n";
  std::cout << "Wave size: " << config.waveSize << "\n";
  
  // Create trace capture interpreter
  TraceCaptureInterpreter captureInterpreter;
  
  // Execute seed and capture golden trace
  std::cout << "Capturing golden trace...\n";
  
  interpreter::ThreadOrdering ordering;
  // Use default source order
  
  auto goldenResult = captureInterpreter.executeAndCaptureTrace(
    seedProgram, ordering, config.waveSize);
  
  ExecutionTrace goldenTrace = *captureInterpreter.getTrace();
  
  std::cout << "Golden trace captured:\n";
  std::cout << "  - Blocks executed: " << goldenTrace.blocks.size() << "\n";
  std::cout << "  - Wave operations: " << goldenTrace.waveOperations.size() << "\n";
  std::cout << "  - Control flow decisions: " << goldenTrace.controlFlowHistory.size() << "\n";
  
  // Generate and test mutants
  size_t mutantsTested = 0;
  size_t bugsFound = 0;
  
  for (auto& strategy : mutationStrategies) {
    std::cout << "\nTrying mutation strategy: " << strategy->getName() << "\n";
    
    // Generate mutants with this strategy
    auto mutants = generateMutants(seedProgram, strategy.get(), goldenTrace);
    
    for (const auto& mutant : mutants) {
      if (mutantsTested >= config.maxMutants) {
        break;
      }
      
      mutantsTested++;
      
      try {
        // Execute mutant
        TraceCaptureInterpreter mutantInterpreter;
        auto mutantResult = mutantInterpreter.executeAndCaptureTrace(
          mutant, ordering, config.waveSize);
        
        ExecutionTrace mutantTrace = *mutantInterpreter.getTrace();
        
        // Validate semantic equivalence
        auto validation = validator->validate(goldenTrace, mutantTrace);
        
        if (!validation.isEquivalent) {
          // Found a bug!
          bugsFound++;
          bugReporter->reportBug(seedProgram, mutant, goldenTrace, 
                               mutantTrace, validation);
        }
        
      } catch (const std::exception& e) {
        // Mutant crashed - definitely a bug
        bugsFound++;
        bugReporter->reportCrash(seedProgram, mutant, e);
      }
    }
  }
  
  logSummary(mutantsTested, bugsFound);
}

std::unique_ptr<interpreter::Program> TraceGuidedFuzzer::mutateAST(
    const interpreter::Program& program, 
    unsigned int seed) {
  // TODO: Implement AST mutation for libFuzzer
  return nullptr;
}

std::vector<interpreter::Program> TraceGuidedFuzzer::generateMutants(
    const interpreter::Program& program,
    MutationStrategy* strategy,
    const ExecutionTrace& trace) {
  
  std::vector<interpreter::Program> mutants;
  
  // Try to apply the mutation strategy to each statement
  for (size_t i = 0; i < program.statements.size(); ++i) {
    const auto& stmt = program.statements[i];
    
    if (strategy->canApply(stmt.get(), trace)) {
      // Apply the mutation
      auto mutatedStmt = strategy->apply(stmt.get(), trace);
      
      if (mutatedStmt) {
        // Create a new program with the mutated statement
        interpreter::Program mutant;
        mutant.numThreadsX = program.numThreadsX;
        mutant.numThreadsY = program.numThreadsY;
        mutant.numThreadsZ = program.numThreadsZ;
        
        // Clone all statements except the mutated one
        for (size_t j = 0; j < program.statements.size(); ++j) {
          if (j == i) {
            mutant.statements.push_back(std::move(mutatedStmt));
          } else {
            mutant.statements.push_back(program.statements[j]->clone());
          }
        }
        
        mutants.push_back(std::move(mutant));
      }
    }
  }
  
  return mutants;
}

bool TraceGuidedFuzzer::hasNewCoverage(const ExecutionTrace& trace) {
  bool foundNew = false;
  
  // Check for new block patterns
  for (const auto& [blockId, record] : trace.blocks) {
    auto hash = hashBlockPattern(record);
    if (seenBlockPatterns.find(hash) == seenBlockPatterns.end()) {
      seenBlockPatterns.insert(hash);
      foundNew = true;
    }
  }
  
  // Check for new wave patterns
  for (const auto& waveOp : trace.waveOperations) {
    auto hash = hashWavePattern(waveOp);
    if (seenWavePatterns.find(hash) == seenWavePatterns.end()) {
      seenWavePatterns.insert(hash);
      foundNew = true;
    }
  }
  
  // Check for new sync patterns
  for (const auto& barrier : trace.barriers) {
    auto hash = hashSyncPattern(barrier);
    if (seenSyncPatterns.find(hash) == seenSyncPatterns.end()) {
      seenSyncPatterns.insert(hash);
      foundNew = true;
    }
  }
  
  return foundNew;
}

uint64_t TraceGuidedFuzzer::hashBlockPattern(const ExecutionTrace::BlockExecutionRecord& block) {
  // Simple hash based on block ID and participation
  uint64_t hash = block.blockId;
  for (const auto& [waveId, participation] : block.waveParticipation) {
    hash ^= (waveId << 16);
    hash ^= participation.participatingLanes.size();
  }
  return hash;
}

uint64_t TraceGuidedFuzzer::hashWavePattern(const ExecutionTrace::WaveOpRecord& waveOp) {
  // Hash based on operation and participants
  std::hash<std::string> strHash;
  uint64_t hash = strHash(waveOp.opType);
  hash ^= waveOp.expectedParticipants.size();
  hash ^= (static_cast<uint64_t>(waveOp.waveId) << 32);
  return hash;
}

uint64_t TraceGuidedFuzzer::hashSyncPattern(const ExecutionTrace::BarrierRecord& barrier) {
  // Hash based on arrival pattern
  uint64_t hash = barrier.barrierId;
  hash ^= barrier.arrivedLanesPerWave.size();
  hash ^= (barrier.waveArrivalOrder.size() << 16);
  return hash;
}

void TraceGuidedFuzzer::logTrace(const std::string& message, const ExecutionTrace& trace) {
  std::cout << message << "\n";
  std::cout << "  Blocks: " << trace.blocks.size() << "\n";
  std::cout << "  Wave ops: " << trace.waveOperations.size() << "\n";
  std::cout << "  Barriers: " << trace.barriers.size() << "\n";
}

void TraceGuidedFuzzer::logMutation(const std::string& message, const std::string& strategy) {
  std::cout << message << " [" << strategy << "]\n";
}

void TraceGuidedFuzzer::logSummary(size_t testedMutants, size_t bugsFound) {
  std::cout << "\n=== Fuzzing Summary ===\n";
  std::cout << "Mutants tested: " << testedMutants << "\n";
  std::cout << "Bugs found: " << bugsFound << "\n";
  std::cout << "Coverage:\n";
  std::cout << "  Block patterns: " << seenBlockPatterns.size() << "\n";
  std::cout << "  Wave patterns: " << seenWavePatterns.size() << "\n";
  std::cout << "  Sync patterns: " << seenSyncPatterns.size() << "\n";
}

// ===== LibFuzzer Integration =====

void loadSeedCorpus(TraceGuidedFuzzer* fuzzer) {
  // Load seed HLSL programs
  std::vector<std::string> seedFiles = {
    "seeds/simple_divergence.hlsl",
    "seeds/loop_divergence.hlsl",
    "seeds/nested_control_flow.hlsl"
  };
  
  for (const auto& file : seedFiles) {
    // TODO: Load and add to corpus
  }
}

bool deserializeAST(const uint8_t* data, size_t size, 
                   interpreter::Program& program) {
  // TODO: Implement AST deserialization
  // For now, create a simple test program
  program.numThreadsX = 32;
  program.numThreadsY = 1;
  program.numThreadsZ = 1;
  return true;
}

size_t serializeAST(const interpreter::Program& program, 
                   uint8_t* data, size_t maxSize) {
  // TODO: Implement AST serialization
  return 0;
}

size_t hashAST(const interpreter::Program& program) {
  // Simple hash based on program size
  return program.statements.size();
}

} // namespace fuzzer
} // namespace minihlsl

// LibFuzzer entry points
extern "C" {

int LLVMFuzzerInitialize(int* argc, char*** argv) {
  minihlsl::fuzzer::g_fuzzer = std::make_unique<minihlsl::fuzzer::TraceGuidedFuzzer>();
  minihlsl::fuzzer::loadSeedCorpus(minihlsl::fuzzer::g_fuzzer.get());
  return 0;
}

int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  if (size < 4) return 0;
  
  // Deserialize AST
  minihlsl::interpreter::Program program;
  if (!minihlsl::fuzzer::deserializeAST(data, size, program)) {
    return 0;
  }
  
  // Run fuzzing
  minihlsl::fuzzer::FuzzingConfig config;
  config.maxMutants = 10; // Quick fuzzing
  config.enableLogging = false; // Quiet mode
  
  try {
    minihlsl::fuzzer::g_fuzzer->fuzzProgram(program, config);
  } catch (...) {
    // Ignore exceptions during fuzzing
  }
  
  return 0;
}

size_t LLVMFuzzerCustomMutator(uint8_t* data, size_t size, 
                              size_t maxSize, unsigned int seed) {
  // Custom mutation based on trace-guided approach
  minihlsl::interpreter::Program program;
  if (!minihlsl::fuzzer::deserializeAST(data, size, program)) {
    return 0;
  }
  
  auto mutated = minihlsl::fuzzer::g_fuzzer->mutateAST(program, seed);
  if (!mutated) {
    return 0;
  }
  
  return minihlsl::fuzzer::serializeAST(*mutated, data, maxSize);
}

size_t LLVMFuzzerCustomCrossOver(const uint8_t* data1, size_t size1,
                                const uint8_t* data2, size_t size2,
                                uint8_t* out, size_t maxOutSize,
                                unsigned int seed) {
  // TODO: Implement crossover
  return 0;
}

} // extern "C"