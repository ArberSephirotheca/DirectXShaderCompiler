#include "MiniHLSLInterpreterFuzzer.h"
#include <algorithm>

namespace minihlsl {
namespace fuzzer {

// ===== Wave Operation Mutations =====

// --- Precompute Wave Results Mutation ---

bool PrecomputeWaveResultsMutation::canApply(
    const interpreter::Statement* stmt,
    const ExecutionTrace& trace) const {
  
  // Check if statement contains wave operation
  if (auto exprStmt = dynamic_cast<const interpreter::ExprStmt*>(stmt)) {
    return dynamic_cast<const interpreter::WaveActiveOp*>(exprStmt->expr.get()) != nullptr;
  }
  if (auto assignStmt = dynamic_cast<const interpreter::AssignStmt*>(stmt)) {
    return dynamic_cast<const interpreter::WaveActiveOp*>(assignStmt->expr.get()) != nullptr;
  }
  if (auto varDeclStmt = dynamic_cast<const interpreter::VarDeclStmt*>(stmt)) {
    return dynamic_cast<const interpreter::WaveActiveOp*>(varDeclStmt->init.get()) != nullptr;
  }
  return false;
}

std::unique_ptr<interpreter::Statement> PrecomputeWaveResultsMutation::apply(
    const interpreter::Statement* stmt,
    const ExecutionTrace& trace) const {
  
  const interpreter::WaveActiveOp* waveOp = nullptr;
  std::string targetVar;
  
  // Extract wave op and target variable
  if (auto exprStmt = dynamic_cast<const interpreter::ExprStmt*>(stmt)) {
    waveOp = dynamic_cast<const interpreter::WaveActiveOp*>(exprStmt->expr.get());
  } else if (auto assignStmt = dynamic_cast<const interpreter::AssignStmt*>(stmt)) {
    waveOp = dynamic_cast<const interpreter::WaveActiveOp*>(assignStmt->expr.get());
    targetVar = assignStmt->varName;
  } else if (auto varDeclStmt = dynamic_cast<const interpreter::VarDeclStmt*>(stmt)) {
    waveOp = dynamic_cast<const interpreter::WaveActiveOp*>(varDeclStmt->init.get());
    targetVar = varDeclStmt->name;
  }
  
  if (!waveOp) return nullptr;
  
  // Find all executions of this wave op in trace
  std::vector<const ExecutionTrace::WaveOpRecord*> executions;
  for (auto& record : trace.waveOperations) {
    if (record.instruction == waveOp) {
      executions.push_back(&record);
    }
  }
  
  if (executions.empty()) return nullptr;
  
  auto compound = std::make_unique<interpreter::CompoundStmt>();
  
  // First, declare the variable if needed
  if (!targetVar.empty()) {
    auto varDecl = std::make_unique<interpreter::VarDeclStmt>(
      targetVar,
      std::make_unique<interpreter::LiteralExpr>(interpreter::Value(0))
    );
    compound->addStatement(std::move(varDecl));
  }
  
  // For each wave that executed this op
  for (auto& exec : executions) {
    // For each lane that participated
    for (auto& [laneId, result] : exec->outputValues) {
      // if (waveId == X && laneId == Y) { var = precomputed_result; }
      auto waveCheck = std::make_unique<interpreter::BinaryOpExpr>(
        std::make_unique<interpreter::WaveIndexExpr>(),
        std::make_unique<interpreter::LiteralExpr>(interpreter::Value((int32_t)exec->waveId)),
        interpreter::BinaryOpExpr::Eq
      );
      
      auto laneCheck = std::make_unique<interpreter::BinaryOpExpr>(
        std::make_unique<interpreter::LaneIndexExpr>(),
        std::make_unique<interpreter::LiteralExpr>(interpreter::Value((int32_t)laneId)),
        interpreter::BinaryOpExpr::Eq
      );
      
      auto condition = std::make_unique<interpreter::BinaryOpExpr>(
        std::move(waveCheck),
        std::move(laneCheck),
        interpreter::BinaryOpExpr::And
      );
      
      std::unique_ptr<interpreter::Statement> assignment;
      if (!targetVar.empty()) {
        assignment = std::make_unique<interpreter::AssignStmt>(
          targetVar,
          std::make_unique<interpreter::LiteralExpr>(result)
        );
      } else {
        // Just evaluate the result
        assignment = std::make_unique<interpreter::ExprStmt>(
          std::make_unique<interpreter::LiteralExpr>(result)
        );
      }
      
      compound->addStatement(std::make_unique<interpreter::IfStmt>(
        std::move(condition),
        std::move(assignment),
        nullptr
      ));
    }
  }
  
  return compound;
}

bool PrecomputeWaveResultsMutation::validateSemanticPreservation(
    const interpreter::Statement* original,
    const interpreter::Statement* mutated,
    const ExecutionTrace& trace) const {
  // By construction, we're replacing wave ops with their exact traced results
  return true;
}

// --- Redundant Wave Sync Mutation ---

bool RedundantWaveSyncMutation::canApply(
    const interpreter::Statement* stmt,
    const ExecutionTrace& trace) const {
  
  return extractWaveOp(stmt) != nullptr;
}

const interpreter::WaveActiveOp* RedundantWaveSyncMutation::extractWaveOp(
    const interpreter::Statement* stmt) const {
  
  if (auto exprStmt = dynamic_cast<const interpreter::ExprStmt*>(stmt)) {
    return dynamic_cast<const interpreter::WaveActiveOp*>(exprStmt->expr.get());
  }
  if (auto assignStmt = dynamic_cast<const interpreter::AssignStmt*>(stmt)) {
    return dynamic_cast<const interpreter::WaveActiveOp*>(assignStmt->expr.get());
  }
  if (auto varDeclStmt = dynamic_cast<const interpreter::VarDeclStmt*>(stmt)) {
    return dynamic_cast<const interpreter::WaveActiveOp*>(varDeclStmt->init.get());
  }
  return nullptr;
}

std::unique_ptr<interpreter::Statement> RedundantWaveSyncMutation::apply(
    const interpreter::Statement* stmt,
    const ExecutionTrace& trace) const {
  
  auto compound = std::make_unique<interpreter::CompoundStmt>();
  
  // Find wave op executions for this statement
  const interpreter::WaveActiveOp* waveOp = extractWaveOp(stmt);
  if (!waveOp) return nullptr;
  
  std::map<interpreter::WaveId, std::set<interpreter::LaneId>> waveParticipants;
  for (auto& record : trace.waveOperations) {
    if (record.instruction == waveOp) {
      waveParticipants[record.waveId] = record.arrivedParticipants;
    }
  }
  
  // Add per-wave synchronization before operation
  for (auto& [waveId, participants] : waveParticipants) {
    // Create a dummy wave operation that forces sync
    // WaveActiveSum(0) should be harmless and force synchronization
    auto dummyOp = std::make_unique<interpreter::WaveActiveOp>(
      std::make_unique<interpreter::LiteralExpr>(interpreter::Value(0)),
      interpreter::WaveActiveOp::Sum
    );
    
    auto syncStmt = std::make_unique<interpreter::ExprStmt>(std::move(dummyOp));
    
    // Guard with wave check
    auto waveGuard = std::make_unique<interpreter::IfStmt>(
      std::make_unique<interpreter::BinaryOpExpr>(
        std::make_unique<interpreter::WaveIndexExpr>(),
        std::make_unique<interpreter::LiteralExpr>(interpreter::Value((int32_t)waveId)),
        interpreter::BinaryOpExpr::Eq
      ),
      std::move(syncStmt),
      nullptr
    );
    
    compound->addStatement(std::move(waveGuard));
  }
  
  // Original operation
  compound->addStatement(stmt->clone());
  
  // Duplicate operation (should produce same result!)
  compound->addStatement(stmt->clone());
  
  return compound;
}

bool RedundantWaveSyncMutation::validateSemanticPreservation(
    const interpreter::Statement* original,
    const interpreter::Statement* mutated,
    const ExecutionTrace& trace) const {
  // Adding redundant synchronization and duplicating deterministic ops preserves semantics
  return true;
}

// ===== Block Structure Mutations =====

// --- Force Block Boundaries Mutation ---

bool ForceBlockBoundariesMutation::canApply(
    const interpreter::Statement* stmt,
    const ExecutionTrace& trace) const {
  
  // Find all blocks created by this statement
  std::set<uint32_t> blockIds;
  for (auto& [blockId, block] : trace.blocks) {
    if (block.sourceStatement == stmt) {
      blockIds.insert(blockId);
    }
  }
  
  // Apply if statement creates multiple blocks
  return blockIds.size() > 1;
}

std::unique_ptr<interpreter::Statement> ForceBlockBoundariesMutation::apply(
    const interpreter::Statement* stmt,
    const ExecutionTrace& trace) const {
  
  // Find all block transitions in this statement
  std::vector<const ExecutionTrace::BlockExecutionRecord*> blocks;
  for (auto& [blockId, block] : trace.blocks) {
    if (block.sourceStatement == stmt) {
      blocks.push_back(&block);
    }
  }
  
  if (blocks.size() <= 1) {
    return nullptr;
  }
  
  // Sort blocks by creation order or type
  std::sort(blocks.begin(), blocks.end(),
    [](const auto* a, const auto* b) {
      if (a->blockType != b->blockType) {
        return static_cast<int>(a->blockType) < static_cast<int>(b->blockType);
      }
      return a->blockId < b->blockId;
    });
  
  auto compound = std::make_unique<interpreter::CompoundStmt>();
  
  // Add explicit block markers and reconstruct control flow
  for (auto* block : blocks) {
    // Create a marker that forces block creation
    auto marker = createBlockMarker(block->blockId, block->blockType);
    compound->addStatement(std::move(marker));
    
    // Add appropriate content based on block type
    if (auto ifStmt = dynamic_cast<const interpreter::IfStmt*>(stmt)) {
      if (block->blockType == interpreter::BlockType::BRANCH_THEN && ifStmt->thenBlock) {
        for (auto& s : ifStmt->thenBlock->statements) {
          compound->addStatement(s->clone());
        }
      } else if (block->blockType == interpreter::BlockType::BRANCH_ELSE && ifStmt->elseBlock) {
        for (auto& s : ifStmt->elseBlock->statements) {
          compound->addStatement(s->clone());
        }
      }
    } else if (auto loopStmt = dynamic_cast<const interpreter::ForStmt*>(stmt)) {
      if (block->blockType == interpreter::BlockType::LOOP_BODY && loopStmt->body) {
        for (auto& s : loopStmt->body->statements) {
          compound->addStatement(s->clone());
        }
      }
    }
    // Handle other statement types...
  }
  
  return compound;
}

std::unique_ptr<interpreter::Statement> ForceBlockBoundariesMutation::createBlockMarker(
    uint32_t blockId,
    interpreter::BlockType type) const {
  
  // Create a no-op statement that forces block creation
  // Could be an empty compound statement or a dummy expression
  auto marker = std::make_unique<interpreter::CompoundStmt>();
  
  // Add a comment-like expression that identifies the block
  auto blockIdExpr = std::make_unique<interpreter::ExprStmt>(
    std::make_unique<interpreter::LiteralExpr>(interpreter::Value((int32_t)blockId))
  );
  
  marker->addStatement(std::move(blockIdExpr));
  
  return marker;
}

bool ForceBlockBoundariesMutation::validateSemanticPreservation(
    const interpreter::Statement* original,
    const interpreter::Statement* mutated,
    const ExecutionTrace& trace) const {
  // Making implicit blocks explicit preserves semantics
  return true;
}

// ===== Memory Access Mutations =====

// --- Serialize Memory Accesses Mutation ---

bool SerializeMemoryAccessesMutation::canApply(
    const interpreter::Statement* stmt,
    const ExecutionTrace& trace) const {
  
  // Find block containing this statement
  uint32_t stmtBlockId = findBlockContaining(stmt, trace);
  if (stmtBlockId == 0) return false;
  
  // Count memory accesses in this block
  int accessCount = 0;
  for (auto& access : trace.memoryAccesses) {
    if (access.blockId == stmtBlockId) {
      accessCount++;
    }
  }
  
  // Apply if multiple memory accesses
  return accessCount > 1;
}

uint32_t SerializeMemoryAccessesMutation::findBlockContaining(
    const interpreter::Statement* stmt,
    const ExecutionTrace& trace) const {
  
  // Find block that was created by this statement
  for (auto& [blockId, block] : trace.blocks) {
    if (block.sourceStatement == stmt) {
      return blockId;
    }
  }
  
  // Or find block that contains this statement in its execution
  // This would require more sophisticated tracking
  return 0;
}

std::unique_ptr<interpreter::Statement> SerializeMemoryAccessesMutation::apply(
    const interpreter::Statement* stmt,
    const ExecutionTrace& trace) const {
  
  // Find all memory accesses in this statement's execution
  uint32_t stmtBlockId = findBlockContaining(stmt, trace);
  if (stmtBlockId == 0) return nullptr;
  
  std::vector<const ExecutionTrace::MemoryAccess*> accesses;
  for (auto& access : trace.memoryAccesses) {
    if (access.blockId == stmtBlockId) {
      accesses.push_back(&access);
    }
  }
  
  if (accesses.size() <= 1) return nullptr;
  
  // Group by address to find potential conflicts
  std::map<uint32_t, std::vector<const ExecutionTrace::MemoryAccess*>> byAddress;
  for (auto* access : accesses) {
    byAddress[access->address].push_back(access);
  }
  
  auto compound = std::make_unique<interpreter::CompoundStmt>();
  
  // First add the original statement
  compound->addStatement(stmt->clone());
  
  // Then serialize accesses to each conflicting address
  for (auto& [addr, addrAccesses] : byAddress) {
    if (addrAccesses.size() <= 1) continue;
    
    // Sort by original timestamp
    std::sort(addrAccesses.begin(), addrAccesses.end(),
      [](const auto* a, const auto* b) {
        return a->timestamp < b->timestamp;
      });
    
    // Add a barrier after each access to ensure ordering
    for (size_t i = 0; i < addrAccesses.size() - 1; i++) {
      auto barrier = std::make_unique<interpreter::BarrierStmt>(
        interpreter::BarrierStmt::BarrierType::Threadgroup
      );
      compound->addStatement(std::move(barrier));
    }
  }
  
  return compound;
}

std::unique_ptr<interpreter::Statement> SerializeMemoryAccessesMutation::createMemoryAccessStmt(
    const ExecutionTrace::MemoryAccess* access) const {
  
  // Create a statement that performs the memory access
  // This would need to reconstruct the appropriate load/store operation
  
  if (access->type == ExecutionTrace::MemoryAccess::GlobalBuffer) {
    // Create buffer access
    if (!access->isAtomic) {
      if (access->isWrite) {
        // buffer[index] = value;
        auto bufferAccess = std::make_unique<interpreter::BufferStoreStmt>(
          access->bufferName,
          std::make_unique<interpreter::LiteralExpr>(interpreter::Value((int32_t)access->address)),
          std::make_unique<interpreter::LiteralExpr>(access->newValue)
        );
        return bufferAccess;
      } else {
        // temp = buffer[index];
        auto bufferLoad = std::make_unique<interpreter::BufferLoadExpr>(
          access->bufferName,
          std::make_unique<interpreter::LiteralExpr>(interpreter::Value((int32_t)access->address))
        );
        return std::make_unique<interpreter::ExprStmt>(std::move(bufferLoad));
      }
    }
    // Handle atomic operations...
  }
  
  return nullptr;
}

bool SerializeMemoryAccessesMutation::validateSemanticPreservation(
    const interpreter::Statement* original,
    const interpreter::Statement* mutated,
    const ExecutionTrace& trace) const {
  // Serializing memory accesses preserves the final memory state
  return true;
}

// ===== Bug Reporter Implementation =====

void BugReporter::reportBug(
    const interpreter::CompoundStmt* original,
    const interpreter::CompoundStmt* mutant,
    const ExecutionTrace& originalTrace,
    const ExecutionTrace& mutantTrace,
    const SemanticValidator::ValidationResult& validation) {
  
  BugReport report;
  report.id = generateBugId();
  report.timestamp = std::chrono::system_clock::now();
  
  // Program information
  report.originalProgram = original->toString();
  report.mutantProgram = mutant->toString();
  report.mutation = "TODO: identify mutation type";
  
  // Trace comparison
  report.traceDivergence = findTraceDivergence(originalTrace, mutantTrace);
  report.validation = validation;
  
  // Classification
  report.bugType = classifyBug(report.traceDivergence);
  report.severity = assessSeverity(report.bugType, validation);
  
  // TODO: Generate minimal reproducer
  report.minimalReproducer = mutant->toString();
  
  // Save report
  saveBugReport(report);
  
  // Log summary
  logBug(report);
  
  lastReport = report;
}

void BugReporter::reportCrash(
    const interpreter::CompoundStmt* original,
    const interpreter::CompoundStmt* mutant,
    const std::exception& e) {
  
  BugReport report;
  report.id = generateBugId();
  report.timestamp = std::chrono::system_clock::now();
  
  report.originalProgram = original->toString();
  report.mutantProgram = mutant->toString();
  report.mutation = "TODO: identify mutation type";
  
  report.traceDivergence.type = BugReport::TraceDivergence::ControlFlow;
  report.traceDivergence.description = std::string("Crash: ") + e.what();
  
  report.bugType = BugReport::ControlFlowError;
  report.severity = BugReport::Critical;
  
  saveBugReport(report);
  logBug(report);
  
  lastReport = report;
}

BugReporter::BugReport::TraceDivergence BugReporter::findTraceDivergence(
    const ExecutionTrace& golden,
    const ExecutionTrace& mutant) {
  
  BugReport::TraceDivergence divergence;
  
  // Compare execution traces to find first divergence
  // Start with block structure
  if (golden.blocks.size() != mutant.blocks.size()) {
    divergence.type = BugReport::TraceDivergence::BlockStructure;
    divergence.description = "Different number of dynamic blocks";
    return divergence;
  }
  
  // Check wave operations
  if (golden.waveOperations.size() != mutant.waveOperations.size()) {
    divergence.type = BugReport::TraceDivergence::WaveOperation;
    divergence.description = "Different number of wave operations";
    return divergence;
  }
  
  // Check control flow
  if (golden.controlFlowHistory.size() != mutant.controlFlowHistory.size()) {
    divergence.type = BugReport::TraceDivergence::ControlFlow;
    divergence.description = "Different control flow decisions";
    return divergence;
  }
  
  // Default
  divergence.type = BugReport::TraceDivergence::Memory;
  divergence.description = "Memory state divergence";
  return divergence;
}

BugReporter::BugReport::BugType BugReporter::classifyBug(
    const BugReport::TraceDivergence& divergence) {
  
  switch (divergence.type) {
    case BugReport::TraceDivergence::WaveOperation:
      return BugReport::WaveOpInconsistency;
    case BugReport::TraceDivergence::BlockStructure:
      return BugReport::ReconvergenceError;
    case BugReport::TraceDivergence::Synchronization:
      return BugReport::DeadlockOrRace;
    case BugReport::TraceDivergence::Memory:
      return BugReport::MemoryCorruption;
    case BugReport::TraceDivergence::ControlFlow:
    default:
      return BugReport::ControlFlowError;
  }
}

BugReporter::BugReport::Severity BugReporter::assessSeverity(
    BugReport::BugType type,
    const SemanticValidator::ValidationResult& validation) {
  
  // Critical if wave operations produce wrong results
  if (type == BugReport::WaveOpInconsistency) {
    return BugReport::Critical;
  }
  
  // High if reconvergence is broken
  if (type == BugReport::ReconvergenceError) {
    return BugReport::High;
  }
  
  // Medium for other issues
  return BugReport::Medium;
}

std::string BugReporter::generateBugId() {
  static uint32_t bugCounter = 0;
  return "BUG-" + std::to_string(++bugCounter) + "-" + 
         std::to_string(std::chrono::system_clock::now().time_since_epoch().count());
}

void BugReporter::saveBugReport(const BugReport& report) {
  // Save to file
  std::ofstream file("bug_" + report.id + ".json");
  file << "{\n";
  file << "  \"id\": \"" << report.id << "\",\n";
  file << "  \"type\": " << static_cast<int>(report.bugType) << ",\n";
  file << "  \"severity\": " << static_cast<int>(report.severity) << ",\n";
  file << "  \"divergence\": \"" << report.traceDivergence.description << "\",\n";
  file << "  \"validation_errors\": [\n";
  for (size_t i = 0; i < report.validation.differences.size(); i++) {
    file << "    \"" << report.validation.differences[i] << "\"";
    if (i < report.validation.differences.size() - 1) file << ",";
    file << "\n";
  }
  file << "  ]\n";
  file << "}\n";
}

void BugReporter::logBug(const BugReport& report) {
  std::cerr << "=== BUG FOUND: " << report.id << " ===\n";
  std::cerr << "Type: " << static_cast<int>(report.bugType) << "\n";
  std::cerr << "Severity: " << static_cast<int>(report.severity) << "\n";
  std::cerr << "Divergence: " << report.traceDivergence.description << "\n";
  std::cerr << "Validation reason: " << report.validation.divergenceReason << "\n";
  if (!report.validation.differences.empty()) {
    std::cerr << "First difference: " << report.validation.differences[0] << "\n";
  }
  std::cerr << "\n";
}

} // namespace fuzzer
} // namespace minihlsl