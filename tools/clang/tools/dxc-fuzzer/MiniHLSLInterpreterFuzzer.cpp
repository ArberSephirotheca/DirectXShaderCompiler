#include "MiniHLSLInterpreterFuzzer.h"
#include "MiniHLSLInterpreterTraceCapture.h"
#include "MiniHLSLValidator.h"
#include <fstream>
#include <sstream>
#include <random>
#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <sys/stat.h>
#include <dirent.h>

namespace minihlsl {
namespace fuzzer {

// Global fuzzer instance
std::unique_ptr<TraceGuidedFuzzer> g_fuzzer;

// Global seed programs loaded from corpus
std::unique_ptr<std::vector<interpreter::Program>> g_seedPrograms;

// Helper function to serialize a program to string
std::string serializeProgramToString(const interpreter::Program& program) {
  std::stringstream ss;
  
  // Add thread configuration
  ss << "[numthreads(" << program.numThreadsX << ", " 
     << program.numThreadsY << ", " 
     << program.numThreadsZ << ")]\n";
  ss << "void main() {\n";
  
  // Add all statements
  for (const auto& stmt : program.statements) {
    ss << "  " << stmt->toString() << "\n";
  }
  
  ss << "}\n";
  return ss.str();
}

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
  // TODO: verify
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
  
  // Precompute wave operation results and replace with literals
  auto* assign = dynamic_cast<const interpreter::AssignStmt*>(stmt);
  if (!assign) {
    return stmt->clone();
  }
  
  auto* waveOp = dynamic_cast<const interpreter::WaveActiveOp*>(assign->getExpression());
  if (!waveOp) {
    return stmt->clone();
  }
  
  // Find this wave operation in the trace
  for (const auto& waveRecord : trace.waveOperations) {
    if (waveRecord.opType == waveOp->toString()) {
      // Check if all participants have the same result value
      bool allSame = true;
      interpreter::Value commonValue;
      bool firstValue = true;
      
      for (const auto& [laneId, value] : waveRecord.outputValues) {
        if (firstValue) {
          commonValue = value;
          firstValue = false;
        } else if (value.toString() != commonValue.toString()) {
          allSame = false;
          break;
        }
      }
      
      if (allSame && !firstValue) {
        // Replace with literal - this precomputes the wave result
        auto literal = std::make_unique<interpreter::LiteralExpr>(commonValue);
        return std::make_unique<interpreter::AssignStmt>(
            assign->getName(), std::move(literal));
      }
      break; // Only use the first matching wave op
    }
  }
  
  return stmt->clone();
}

bool PrecomputeWaveResultsMutation::validateSemanticPreservation(
    const interpreter::Statement* original,
    const interpreter::Statement* mutated,
    const ExecutionTrace& trace) const {
  // TODO: validate
  return true;
}

bool RedundantWaveSyncMutation::canApply(const interpreter::Statement* stmt, 
                                       const ExecutionTrace& trace) const {
  // Can apply after most statements
  // (Skip wave operation check since expr_ is private)
  return dynamic_cast<const interpreter::VarDeclStmt*>(stmt) != nullptr ||
         dynamic_cast<const interpreter::ExprStmt*>(stmt) != nullptr ||
         dynamic_cast<const interpreter::IfStmt*>(stmt) != nullptr ||
         dynamic_cast<const interpreter::ForStmt*>(stmt) != nullptr;
}

std::unique_ptr<interpreter::Statement> RedundantWaveSyncMutation::apply(
    const interpreter::Statement* stmt, 
    const ExecutionTrace& trace) const {
  // Wrap the original statement and a redundant wave operation
  // in a trivial if(true) block
  
  // Create the block containing original statement + redundant sync
  std::vector<std::unique_ptr<interpreter::Statement>> thenBlock;
  
  // Clone the original statement
  thenBlock.push_back(stmt->clone());
  
  // Add a redundant wave operation
  // Choose randomly between different redundant operations
  static std::mt19937 rng(std::random_device{}());
  std::uniform_int_distribution<int> dist(0, 2);
  
  switch (dist(rng)) {
    case 0: {
      // Add WaveActiveBallot(true) - returns mask of active lanes
      auto trueLit = std::make_unique<interpreter::LiteralExpr>(true);
      auto ballotOp = std::make_unique<interpreter::WaveActiveOp>(
          std::move(trueLit), interpreter::WaveActiveOp::Ballot);
      auto ballotStmt = std::make_unique<interpreter::ExprStmt>(std::move(ballotOp));
      thenBlock.push_back(std::move(ballotStmt));
      break;
    }
    case 1: {
      // Add WaveActiveAllTrue(true) - always returns true
      auto trueLit = std::make_unique<interpreter::LiteralExpr>(true);
      auto allTrueOp = std::make_unique<interpreter::WaveActiveOp>(
          std::move(trueLit), interpreter::WaveActiveOp::AllTrue);
      auto allTrueStmt = std::make_unique<interpreter::ExprStmt>(std::move(allTrueOp));
      thenBlock.push_back(std::move(allTrueStmt));
      break;
    }
    case 2: {
      // Add WaveActiveSum(0) - sums zeros across lanes
      auto zeroLit = std::make_unique<interpreter::LiteralExpr>(0);
      auto sumOp = std::make_unique<interpreter::WaveActiveOp>(
          std::move(zeroLit), interpreter::WaveActiveOp::Sum);
      auto sumStmt = std::make_unique<interpreter::ExprStmt>(std::move(sumOp));
      thenBlock.push_back(std::move(sumStmt));
      break;
    }
  }
  
  // Create if(true) to wrap both statements
  auto trueCond = std::make_unique<interpreter::LiteralExpr>(true);
  return std::make_unique<interpreter::IfStmt>(
      std::move(trueCond), 
      std::move(thenBlock));
}

bool RedundantWaveSyncMutation::validateSemanticPreservation(
    const interpreter::Statement* original,
    const interpreter::Statement* mutated,
    const ExecutionTrace& trace) const {
  return true;
}

bool ForceBlockBoundariesMutation::canApply(const interpreter::Statement* stmt, 
                                          const ExecutionTrace& trace) const {
  // Can apply to any statement by wrapping it in an if(true) block
  // This is most useful for compound statements or sequences of statements
  
  // Don't apply to statements that are already control flow
  if (dynamic_cast<const interpreter::IfStmt*>(stmt) ||
      dynamic_cast<const interpreter::ForStmt*>(stmt) ||
      dynamic_cast<const interpreter::WhileStmt*>(stmt) ||
      dynamic_cast<const interpreter::DoWhileStmt*>(stmt)) {
    return false; // Already has block boundaries
  }
  
  // Apply to assignment statements and expression statements
  return dynamic_cast<const interpreter::AssignStmt*>(stmt) != nullptr ||
         dynamic_cast<const interpreter::ExprStmt*>(stmt) != nullptr ||
         dynamic_cast<const interpreter::VarDeclStmt*>(stmt) != nullptr;
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
  // Force block boundaries mutation adds explicit if statements with always-true conditions
  // This should preserve semantics as the code always executes
  
  // Check that the mutated statement is an if statement
  auto* ifStmt = dynamic_cast<const interpreter::IfStmt*>(mutated);
  if (!ifStmt) {
    return false; // Expected an if statement
  }
  
  // Check that the condition is always true (literal 1)
  auto* condition = ifStmt->getCondition();
  auto* literal = dynamic_cast<const interpreter::LiteralExpr*>(condition);
  if (!literal || literal->toString() != "1") {
    return false; // Expected condition to be literal 1
  }
  
  // The mutation preserves semantics by always executing the then block
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
  // This mutation adds barriers between memory accesses to serialize them
  // This helps find race conditions by forcing deterministic ordering
  
  // First check if this statement performs a memory access
  bool isMemoryAccess = false;
  
  for (const auto& access : trace.memoryAccesses) {
    if (access.accessingThread == 0) { // Check if this access is from current thread
      // For now, we'll just clone the original statement
      // A full implementation would add memory barriers
      isMemoryAccess = true;
      break;
    }
  }
  
  if (!isMemoryAccess) {
    return nullptr; // No mutation needed
  }
  
  // Clone the original statement
  if (auto* assign = dynamic_cast<const interpreter::AssignStmt*>(stmt)) {
    return std::make_unique<interpreter::AssignStmt>(
        assign->getName(), 
        assign->getExpression()->clone());
  }
  
  // For other statement types, return nullptr for now
  return nullptr;
}

bool SerializeMemoryAccessesMutation::validateSemanticPreservation(
    const interpreter::Statement* original,
    const interpreter::Statement* mutated,
    const ExecutionTrace& trace) const {
  return true;
}

bool LanePermutationMutation::canApply(const interpreter::Statement* stmt, 
                                       const ExecutionTrace& trace) const {
  // Check if this is an assignment with a wave operation
  if (auto* assign = dynamic_cast<const interpreter::AssignStmt*>(stmt)) {
    // Use dynamic_cast to check for wave operations instead of string matching
    if (auto* waveOp = dynamic_cast<const interpreter::WaveActiveOp*>(assign->getExpression())) {
      // Check if this is an associative operation
      auto opType = waveOp->getOpType();
      bool canApply = opType == interpreter::WaveActiveOp::Sum ||
             opType == interpreter::WaveActiveOp::Product ||
             opType == interpreter::WaveActiveOp::And ||
             opType == interpreter::WaveActiveOp::Or ||
             opType == interpreter::WaveActiveOp::Xor;
      if (canApply) {
        std::cout << "[LanePermutation] canApply returns true for " << waveOp->toString() << "\n";
      }
      return canApply;
    }
  }
  return false;
}

const interpreter::WaveActiveOp* LanePermutationMutation::getWaveOp(
    const interpreter::Statement* stmt) const {
  // Try to extract wave operation from ExprStmt
  if (auto exprStmt = dynamic_cast<const interpreter::ExprStmt*>(stmt)) {
    // Check toString to identify wave operations since expr_ is private
    std::string stmtStr = stmt->toString();
    
    // Only apply to associative wave operations
    if (stmtStr.find("WaveActiveSum") != std::string::npos ||
        stmtStr.find("WaveActiveProduct") != std::string::npos ||
        stmtStr.find("WaveActiveBitAnd") != std::string::npos ||
        stmtStr.find("WaveActiveBitOr") != std::string::npos ||
        stmtStr.find("WaveActiveBitXor") != std::string::npos) {
      // We can't directly access expr_ but we know it's a wave op
      return reinterpret_cast<const interpreter::WaveActiveOp*>(stmt);
    }
  }
  return nullptr;
}

std::unique_ptr<interpreter::Statement> LanePermutationMutation::apply(
    const interpreter::Statement* stmt, 
    const ExecutionTrace& trace) const {
  
  // Only apply to assignments containing associative wave operations
  auto* assign = dynamic_cast<const interpreter::AssignStmt*>(stmt);
  if (!assign) {
    return stmt->clone();
  }
  
  auto* waveOp = dynamic_cast<const interpreter::WaveActiveOp*>(assign->getExpression());
  if (!waveOp) {
    return stmt->clone();
  }
  
  // Get the operation type and verify it's associative
  auto opType = waveOp->getOpType();
  bool isAssociative = (opType == interpreter::WaveActiveOp::Sum ||
                        opType == interpreter::WaveActiveOp::Product ||
                        opType == interpreter::WaveActiveOp::And ||
                        opType == interpreter::WaveActiveOp::Or ||
                        opType == interpreter::WaveActiveOp::Xor);
  
  if (!isAssociative) {
    return stmt->clone();
  }
  
  // Get the input expression from the wave operation
  const auto* inputExpr = waveOp->getInput();
  if (!inputExpr) {
    return stmt->clone();
  }
  
  // Find the actual active lane count from the trace
  // We need to find the specific dynamic block where this wave operation executes
  uint32_t activeLaneCount = 4; // default fallback
  uint32_t blockIdForWaveOp = 0;
  
  // First, try to find the exact wave operation in the trace
  for (const auto& waveOpRecord : trace.waveOperations) {
    // Check if this wave operation record matches our statement
    // The instruction pointer should match the wave operation we're mutating
    if (waveOpRecord.instruction == static_cast<const void*>(waveOp) ||
        waveOpRecord.opType == waveOp->toString()) {
      // Found the matching wave operation
      activeLaneCount = waveOpRecord.arrivedParticipants.size();
      blockIdForWaveOp = waveOpRecord.blockId;
      
      // Debug output
      std::cout << "[LanePermutation] Found wave op in trace with " << activeLaneCount << " active lanes in block " << blockIdForWaveOp << "\n";
      if (trace.blocks.count(blockIdForWaveOp)) {
        const auto& block = trace.blocks.at(blockIdForWaveOp);
        std::cout << "[LanePermutation] Block type: " 
                  << static_cast<int>(block.blockType) << "\n";
        // Find the wave participation info for wave 0 (assuming single wave for now)
        if (block.waveParticipation.count(0)) {
          const auto& waveInfo = block.waveParticipation.at(0);
          std::cout << "[LanePermutation] Block " << blockIdForWaveOp << " wave 0 participating lanes: ";
          for (auto laneId : waveInfo.participatingLanes) {
            std::cout << laneId << " ";
          }
          std::cout << "\n";
        }
      }
      break;
    }
  }
  
  // If we found the block, get more detailed participation info
  if (blockIdForWaveOp > 0 && trace.blocks.count(blockIdForWaveOp)) {
    const auto& block = trace.blocks.at(blockIdForWaveOp);
    // Check wave participation in this specific block
    for (const auto& [waveId, participation] : block.waveParticipation) {
      if (!participation.participatingLanes.empty()) {
        // This gives us the lanes that actually participated in this block
        activeLaneCount = participation.participatingLanes.size();
        std::cout << "[LanePermutation] Block " << blockIdForWaveOp 
                  << " has participating lanes: ";
        for (auto laneId : participation.participatingLanes) {
          std::cout << laneId << " ";
        }
        std::cout << "\n";
        break;
      }
    }
  }
  
  // Ensure we don't create invalid permutations
  if (activeLaneCount == 0) {
    std::cout << "[LanePermutation] Warning: No active lanes found, using default\n";
    activeLaneCount = 4;
  }
  
  // Choose a random permutation type
  static std::mt19937 rng(std::random_device{}());
  std::uniform_int_distribution<int> dist(0, 2);
  PermutationType permType = static_cast<PermutationType>(dist(rng));
  
  // Create compound statement
  std::vector<std::unique_ptr<interpreter::Statement>> statements;
  
  // 1. Create permuted lane index
  auto laneExpr = std::make_unique<interpreter::LaneIndexExpr>();
  auto permutedLaneExpr = createPermutationExpr(permType, std::move(laneExpr), activeLaneCount);
  statements.push_back(std::make_unique<interpreter::VarDeclStmt>(
      "_perm_lane", std::move(permutedLaneExpr)));
  
  // 2. Clone the input expression and create the WaveReadLaneAt
  auto clonedInput = inputExpr->clone();
  auto permLaneVar = std::make_unique<interpreter::VariableExpr>("_perm_lane");
  auto readLaneAt = std::make_unique<interpreter::WaveReadLaneAt>(
      std::move(clonedInput), std::move(permLaneVar));
  
  // Create a unique variable name for the permuted value
  static int permVarCounter = 0;
  std::string permVarName = "_perm_val_" + std::to_string(permVarCounter++);
  
  statements.push_back(std::make_unique<interpreter::VarDeclStmt>(
      permVarName, std::move(readLaneAt)));
  
  // 3. Create the wave operation with the permuted input
  auto permutedInput = std::make_unique<interpreter::VariableExpr>(permVarName);
  auto newWaveOp = std::make_unique<interpreter::WaveActiveOp>(std::move(permutedInput), opType);
  auto newAssign = std::make_unique<interpreter::AssignStmt>(
      assign->getName(), std::move(newWaveOp));
  statements.push_back(std::move(newAssign));
  
  // Wrap in if(true) block to create single statement
  auto trueCond = std::make_unique<interpreter::LiteralExpr>(interpreter::Value(true));
  return std::make_unique<interpreter::IfStmt>(
      std::move(trueCond), std::move(statements), 
      std::vector<std::unique_ptr<interpreter::Statement>>{});
}

std::unique_ptr<interpreter::Expression> LanePermutationMutation::createPermutationExpr(
    PermutationType type,
    std::unique_ptr<interpreter::Expression> laneExpr,
    uint32_t activeLaneCount) const {
  
  switch (type) {
    case PermutationType::Rotate: {
      // (laneId + 1) % activeLaneCount
      auto one = std::make_unique<interpreter::LiteralExpr>(1);
      auto add = std::make_unique<interpreter::BinaryOpExpr>(
          std::move(laneExpr), std::move(one), interpreter::BinaryOpExpr::Add);
      auto count = std::make_unique<interpreter::LiteralExpr>(static_cast<int>(activeLaneCount));
      return std::make_unique<interpreter::BinaryOpExpr>(
          std::move(add), std::move(count), interpreter::BinaryOpExpr::Mod);
    }
    
    case PermutationType::Reverse: {
      // (activeLaneCount - 1) - laneId
      auto countMinus1 = std::make_unique<interpreter::LiteralExpr>(
          static_cast<int>(activeLaneCount - 1));
      return std::make_unique<interpreter::BinaryOpExpr>(
          std::move(countMinus1), std::move(laneExpr), interpreter::BinaryOpExpr::Sub);
    }
    
    case PermutationType::EvenOddSwap: {
      // For even/odd swap, we need to ensure we don't go out of bounds
      // If activeLaneCount is odd, the last lane should map to itself
      // Formula: (laneId % 2 == 0) ? min(laneId + 1, activeLaneCount - 1) : (laneId - 1)
      
      auto two = std::make_unique<interpreter::LiteralExpr>(2);
      auto mod2 = std::make_unique<interpreter::BinaryOpExpr>(
          laneExpr->clone(), std::move(two), interpreter::BinaryOpExpr::Mod);
      auto zero = std::make_unique<interpreter::LiteralExpr>(0);
      auto isEven = std::make_unique<interpreter::BinaryOpExpr>(
          std::move(mod2), std::move(zero), interpreter::BinaryOpExpr::Eq);
      
      // For even lanes: min(laneId + 1, activeLaneCount - 1)
      auto one1 = std::make_unique<interpreter::LiteralExpr>(1);
      auto addOne = std::make_unique<interpreter::BinaryOpExpr>(
          laneExpr->clone(), std::move(one1), interpreter::BinaryOpExpr::Add);
      auto countMinus1Even = std::make_unique<interpreter::LiteralExpr>(interpreter::Value(static_cast<int>(activeLaneCount - 1)));
      auto minExpr = std::make_unique<interpreter::BinaryOpExpr>(
          std::move(addOne), std::move(countMinus1Even), interpreter::BinaryOpExpr::Lt);
      auto evenResult = std::make_unique<interpreter::ConditionalExpr>(
          std::move(minExpr),
          std::make_unique<interpreter::BinaryOpExpr>(
              laneExpr->clone(), 
              std::make_unique<interpreter::LiteralExpr>(1),
              interpreter::BinaryOpExpr::Add),
          std::make_unique<interpreter::LiteralExpr>(interpreter::Value(static_cast<int>(activeLaneCount - 1))));
      
      // For odd lanes: laneId - 1
      auto one2 = std::make_unique<interpreter::LiteralExpr>(1);
      auto subOne = std::make_unique<interpreter::BinaryOpExpr>(
          laneExpr->clone(), std::move(one2), interpreter::BinaryOpExpr::Sub);
      
      return std::make_unique<interpreter::ConditionalExpr>(
          std::move(isEven), std::move(evenResult), std::move(subOne));
    }
    
    default:
      // Default to rotate
      return createPermutationExpr(PermutationType::Rotate, std::move(laneExpr), activeLaneCount);
  }
}

std::unique_ptr<interpreter::Expression> LanePermutationMutation::transformExpression(
    const interpreter::Expression* expr,
    const interpreter::Expression* permutedLaneExpr) const {
  
  // Base case: LaneIndexExpr - replace with permuted version
  if (dynamic_cast<const interpreter::LaneIndexExpr*>(expr)) {
    return permutedLaneExpr->clone();
  }
  
  // Handle different expression types
  if (auto litExpr = dynamic_cast<const interpreter::LiteralExpr*>(expr)) {
    return litExpr->clone();
  }
  
  if (auto varExpr = dynamic_cast<const interpreter::VariableExpr*>(expr)) {
    return varExpr->clone();
  }
  
  if (auto binOp = dynamic_cast<const interpreter::BinaryOpExpr*>(expr)) {
    // We can't access private members, so we need to reconstruct based on toString
    // This is a limitation - for now just clone
    return expr->clone();
  }
  
  // For other expressions, just clone
  return expr->clone();
}

bool LanePermutationMutation::validateSemanticPreservation(
    const interpreter::Statement* original,
    const interpreter::Statement* mutated,
    const ExecutionTrace& trace) const {
  // Associative operations should produce the same result with permuted lanes
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
  
  // Serialize programs to strings
  report.originalProgram = serializeProgramToString(original);
  report.mutantProgram = serializeProgramToString(mutant);
  
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
  
  // Check block structure differences
  if (golden.blocks.size() != mutant.blocks.size()) {
    divergence.type = BugReport::TraceDivergence::BlockStructure;
    divergence.divergencePoint = golden.blocks.size();
    divergence.description = "Different number of blocks executed: " + 
                            std::to_string(golden.blocks.size()) + " vs " + 
                            std::to_string(mutant.blocks.size());
    return divergence;
  }
  
  // Check wave operation differences
  if (golden.waveOperations.size() != mutant.waveOperations.size()) {
    divergence.type = BugReport::TraceDivergence::WaveOperation;
    divergence.divergencePoint = golden.waveOperations.size();
    divergence.description = "Different number of wave operations: " + 
                            std::to_string(golden.waveOperations.size()) + " vs " + 
                            std::to_string(mutant.waveOperations.size());
    return divergence;
  }
  
  // Check for different wave operation results
  for (size_t i = 0; i < golden.waveOperations.size(); ++i) {
    const auto& goldenOp = golden.waveOperations[i];
    const auto& mutantOp = mutant.waveOperations[i];
    
    if (goldenOp.opType != mutantOp.opType) {
      divergence.type = BugReport::TraceDivergence::WaveOperation;
      divergence.divergencePoint = i;
      divergence.description = "Wave operation type mismatch at index " + 
                              std::to_string(i) + ": " + goldenOp.opType + 
                              " vs " + mutantOp.opType;
      return divergence;
    }
    
    // Check output values differ
    if (goldenOp.outputValues != mutantOp.outputValues) {
      divergence.type = BugReport::TraceDivergence::WaveOperation;
      divergence.divergencePoint = i;
      divergence.description = "Wave operation outputs differ at index " + 
                              std::to_string(i);
      return divergence;
    }
  }
  
  // Check control flow decisions
  if (golden.controlFlowHistory.size() != mutant.controlFlowHistory.size()) {
    divergence.type = BugReport::TraceDivergence::ControlFlow;
    divergence.divergencePoint = golden.controlFlowHistory.size();
    divergence.description = "Different number of control flow decisions: " + 
                            std::to_string(golden.controlFlowHistory.size()) + " vs " + 
                            std::to_string(mutant.controlFlowHistory.size());
    return divergence;
  }
  
  // Check barrier synchronization
  if (golden.barriers.size() != mutant.barriers.size()) {
    divergence.type = BugReport::TraceDivergence::Synchronization;
    divergence.divergencePoint = golden.barriers.size();
    divergence.description = "Different number of barriers: " + 
                            std::to_string(golden.barriers.size()) + " vs " + 
                            std::to_string(mutant.barriers.size());
    return divergence;
  }
  
  // Check memory accesses
  if (golden.memoryAccesses.size() != mutant.memoryAccesses.size()) {
    divergence.type = BugReport::TraceDivergence::Memory;
    divergence.divergencePoint = golden.memoryAccesses.size();
    divergence.description = "Different number of memory accesses: " + 
                            std::to_string(golden.memoryAccesses.size()) + " vs " + 
                            std::to_string(mutant.memoryAccesses.size());
    return divergence;
  }
  
  // Default case - no obvious divergence found
  divergence.type = BugReport::TraceDivergence::ControlFlow;
  divergence.divergencePoint = 0;
  divergence.description = "Subtle divergence in execution traces";
  
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
  // Create bugs directory if it doesn't exist
  std::string bugDir = "./bugs";
  struct stat st = {0};
  if (stat(bugDir.c_str(), &st) == -1) {
    mkdir(bugDir.c_str(), 0755);
  }
  
  std::string filename = bugDir + "/" + report.id + ".txt";
  std::ofstream file(filename);
  
  if (!file.is_open()) {
    std::cerr << "Failed to create bug report file: " << filename << "\n";
    return;
  }
  
  // Write bug report header
  file << "=== Bug Report " << report.id << " ===\n";
  file << "Timestamp: " << std::chrono::system_clock::to_time_t(report.timestamp) << "\n";
  file << "Bug Type: ";
  switch (report.bugType) {
    case BugReport::WaveOpInconsistency: file << "WaveOpInconsistency"; break;
    case BugReport::ReconvergenceError: file << "ReconvergenceError"; break;
    case BugReport::DeadlockOrRace: file << "DeadlockOrRace"; break;
    case BugReport::MemoryCorruption: file << "MemoryCorruption"; break;
    case BugReport::ControlFlowError: file << "ControlFlowError"; break;
  }
  file << "\n";
  
  file << "Severity: ";
  switch (report.severity) {
    case BugReport::Critical: file << "Critical"; break;
    case BugReport::High: file << "High"; break;
    case BugReport::Medium: file << "Medium"; break;
    case BugReport::Low: file << "Low"; break;
  }
  file << "\n\n";
  
  // Write divergence information
  file << "=== Divergence Information ===\n";
  file << "Divergence Type: ";
  switch (report.traceDivergence.type) {
    case BugReport::TraceDivergence::BlockStructure: file << "BlockStructure"; break;
    case BugReport::TraceDivergence::WaveOperation: file << "WaveOperation"; break;
    case BugReport::TraceDivergence::ControlFlow: file << "ControlFlow"; break;
    case BugReport::TraceDivergence::Synchronization: file << "Synchronization"; break;
    case BugReport::TraceDivergence::Memory: file << "Memory"; break;
  }
  file << "\n";
  file << "Divergence Point: " << report.traceDivergence.divergencePoint << "\n";
  file << "Description: " << report.traceDivergence.description << "\n\n";
  
  // Write mutation information
  file << "=== Mutation Information ===\n";
  file << "Mutation Strategy: " << report.mutation << "\n\n";
  
  // Write validation results
  file << "=== Validation Results ===\n";
  file << "Is Equivalent: " << (report.validation.isEquivalent ? "Yes" : "No") << "\n";
  file << "Divergence Reason: " << report.validation.divergenceReason << "\n";
  if (!report.validation.differences.empty()) {
    file << "Differences:\n";
    for (const auto& diff : report.validation.differences) {
      file << "  - " << diff << "\n";
    }
  }
  file << "\n";
  
  // Write original program
  file << "=== Original Program ===\n";
  file << report.originalProgram << "\n\n";
  
  // Write mutant program
  file << "=== Mutant Program ===\n";
  file << report.mutantProgram << "\n\n";
  
  // Write minimal reproducer if available
  if (!report.minimalReproducer.empty()) {
    file << "=== Minimal Reproducer ===\n";
    file << report.minimalReproducer << "\n";
  }
  
  file.close();
  std::cout << "Bug report saved to: " << filename << "\n";
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
  mutationStrategies.push_back(std::make_unique<LanePermutationMutation>());
  
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
      
      // Print mutated program if logging is enabled
      if (config.enableLogging) {
        std::cout << "\n=== Mutant " << mutantsTested << " ===\n";
        std::cout << "[numthreads(" << mutant.numThreadsX << ", " 
                  << mutant.numThreadsY << ", " << mutant.numThreadsZ << ")]\n";
        std::cout << "void main() {\n";
        for (const auto& stmt : mutant.statements) {
          std::cout << "  " << stmt->toString() << "\n";
        }
        std::cout << "}\n";
        std::cout << "==================\n";
      }
      
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
  
  // Create a random number generator with the provided seed
  std::mt19937 rng(seed);
  
  // Execute the program to get a baseline trace
  TraceCaptureInterpreter interpreter;
  interpreter::ThreadOrdering ordering = interpreter::ThreadOrdering::sequential(program.getTotalThreads());
  
  interpreter.executeAndCaptureTrace(program, ordering, 32); // Default wave size
  
  ExecutionTrace trace = *interpreter.getTrace();
  
  // Randomly select a mutation strategy
  if (mutationStrategies.empty()) {
    return nullptr;
  }
  
  std::uniform_int_distribution<size_t> strategyDist(0, mutationStrategies.size() - 1);
  size_t strategyIndex = strategyDist(rng);
  auto* strategy = mutationStrategies[strategyIndex].get();
  
  // Try to generate mutants with the selected strategy
  auto mutants = generateMutants(program, strategy, trace);
  
  if (mutants.empty()) {
    return nullptr;
  }
  
  // Return the first valid mutant
  // In the future, we could add more sophisticated selection logic
  return std::make_unique<interpreter::Program>(std::move(mutants[0]));
}

// Helper function to recursively apply mutations to statements
std::unique_ptr<interpreter::Statement> TraceGuidedFuzzer::applyMutationToStatement(
    const interpreter::Statement* stmt,
    MutationStrategy* strategy,
    const ExecutionTrace& trace,
    bool& mutationApplied) {
  
  // First, try to apply mutation to this statement directly
  if (strategy->canApply(stmt, trace)) {
    auto mutated = strategy->apply(stmt, trace);
    if (mutated) {
      mutationApplied = true;
      return mutated;
    }
  }
  
  // If no direct mutation, check for nested statements
  if (auto ifStmt = dynamic_cast<const interpreter::IfStmt*>(stmt)) {
    // Try to mutate statements in the then block
    std::vector<std::unique_ptr<interpreter::Statement>> mutatedThenStmts;
    bool foundMutation = false;
    
    for (const auto& thenStmt : ifStmt->getThenBlock()) {
      if (!foundMutation) {
        auto mutated = applyMutationToStatement(thenStmt.get(), strategy, trace, foundMutation);
        if (foundMutation) {
          mutatedThenStmts.push_back(std::move(mutated));
          mutationApplied = true;
        } else {
          mutatedThenStmts.push_back(thenStmt->clone());
        }
      } else {
        mutatedThenStmts.push_back(thenStmt->clone());
      }
    }
    
    // Try to mutate statements in the else block if we haven't found a mutation yet
    std::vector<std::unique_ptr<interpreter::Statement>> mutatedElseStmts;
    if (!foundMutation && ifStmt->hasElse()) {
      for (const auto& elseStmt : ifStmt->getElseBlock()) {
        if (!foundMutation) {
          auto mutated = applyMutationToStatement(elseStmt.get(), strategy, trace, foundMutation);
          if (foundMutation) {
            mutatedElseStmts.push_back(std::move(mutated));
            mutationApplied = true;
          } else {
            mutatedElseStmts.push_back(elseStmt->clone());
          }
        } else {
          mutatedElseStmts.push_back(elseStmt->clone());
        }
      }
    } else if (ifStmt->hasElse()) {
      // Clone else statements if we already found a mutation
      for (const auto& elseStmt : ifStmt->getElseBlock()) {
        mutatedElseStmts.push_back(elseStmt->clone());
      }
    }
    
    // If we found a mutation in nested statements, create a new IfStmt
    if (mutationApplied) {
      return std::make_unique<interpreter::IfStmt>(
          ifStmt->getCondition()->clone(),
          std::move(mutatedThenStmts),
          std::move(mutatedElseStmts));
    }
  } else if (auto forStmt = dynamic_cast<const interpreter::ForStmt*>(stmt)) {
    // Try to mutate statements in the for loop body
    std::vector<std::unique_ptr<interpreter::Statement>> mutatedBodyStmts;
    bool foundMutation = false;
    
    for (const auto& bodyStmt : forStmt->getBody()) {
      if (!foundMutation) {
        auto mutated = applyMutationToStatement(bodyStmt.get(), strategy, trace, foundMutation);
        if (foundMutation) {
          mutatedBodyStmts.push_back(std::move(mutated));
          mutationApplied = true;
        } else {
          mutatedBodyStmts.push_back(bodyStmt->clone());
        }
      } else {
        mutatedBodyStmts.push_back(bodyStmt->clone());
      }
    }
    
    // If we found a mutation in the body, create a new ForStmt
    if (mutationApplied) {
      return std::make_unique<interpreter::ForStmt>(
          forStmt->getLoopVar(),
          forStmt->getInit() ? forStmt->getInit()->clone() : nullptr,
          forStmt->getCondition() ? forStmt->getCondition()->clone() : nullptr,
          forStmt->getIncrement() ? forStmt->getIncrement()->clone() : nullptr,
          std::move(mutatedBodyStmts));
    }
  }
  
  // No mutation found, return clone
  return stmt->clone();
}

std::vector<interpreter::Program> TraceGuidedFuzzer::generateMutants(
    const interpreter::Program& program,
    MutationStrategy* strategy,
    const ExecutionTrace& trace) {
  
  std::vector<interpreter::Program> mutants;
  
  // Try to apply the mutation strategy to each statement (including nested ones)
  for (size_t i = 0; i < program.statements.size(); ++i) {
    bool mutationApplied = false;
    auto mutatedStmt = applyMutationToStatement(
        program.statements[i].get(), strategy, trace, mutationApplied);
    
    if (mutationApplied) {
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
  // Check if HLSL_SEED_DIR environment variable is set
  const char* seedDir = std::getenv("HLSL_SEED_DIR");
  if (!seedDir) {
    // Try default locations
    std::vector<std::string> defaultDirs = {
      "seeds",
      "../seeds",
      "../../tools/clang/tools/dxc-fuzzer/seeds",
    };
    
    for (const auto& dir : defaultDirs) {
      struct stat info;
      if (stat(dir.c_str(), &info) == 0 && S_ISDIR(info.st_mode)) {
        seedDir = dir.c_str();
        break;
      }
    }
  }
  
  if (!seedDir) {
    std::cerr << "No seed directory found. Set HLSL_SEED_DIR or create a 'seeds' directory.\n";
    return;
  }
  
  std::cout << "Loading seed corpus from: " << seedDir << "\n";
  
  // Load all .hlsl files from the seed directory
  DIR* dir = opendir(seedDir);
  if (!dir) {
    std::cerr << "Failed to open seed directory: " << seedDir << "\n";
    return;
  }
  
  struct dirent* entry;
  int seedCount = 0;
  while ((entry = readdir(dir)) != nullptr) {
    std::string filename = entry->d_name;
    
    // Check if it's an HLSL file
    if (filename.size() > 5 && filename.substr(filename.size() - 5) == ".hlsl") {
      std::string fullPath = std::string(seedDir) + "/" + filename;
      
      // Read the file
      std::ifstream file(fullPath);
      if (!file.is_open()) {
        std::cerr << "Failed to open seed file: " << fullPath << "\n";
        continue;
      }
      
      std::stringstream buffer;
      buffer << file.rdbuf();
      std::string hlslContent = buffer.str();
      
      // Parse HLSL and convert to interpreter AST
      std::cout << "  Loading seed: " << filename << " (" << hlslContent.size() << " bytes)";
      
      // Use the MiniHLSLValidator to parse HLSL
      minihlsl::MiniHLSLValidator validator;
      auto astResult = validator.validate_source_with_ast_ownership(hlslContent, fullPath);
      
      auto* astContext = astResult.get_ast_context();
      auto* mainFunc = astResult.get_main_function();
      
      if (!astContext || !mainFunc) {
        std::cout << " - Failed to parse or find main function\n";
        continue;
      }
      
      // Convert to interpreter program
      minihlsl::interpreter::MiniHLSLInterpreter interpreter(0);
      auto conversionResult = interpreter.convertFromHLSLAST(mainFunc, *astContext);
      if (!conversionResult.success) {
        std::cout << " - Failed to convert: " << conversionResult.errorMessage << "\n";
        continue;
      }
      
      // Add to fuzzer's seed programs (stored for mutation)
      if (!g_seedPrograms) {
        g_seedPrograms = std::make_unique<std::vector<interpreter::Program>>();
      }
      // Move the program - need to clone because Program can't be moved directly
      interpreter::Program seedProgram;
      seedProgram.numThreadsX = conversionResult.program.numThreadsX;
      seedProgram.numThreadsY = conversionResult.program.numThreadsY;
      seedProgram.numThreadsZ = conversionResult.program.numThreadsZ;
      
      for (auto& stmt : conversionResult.program.statements) {
        seedProgram.statements.push_back(std::move(stmt));
      }
      
      g_seedPrograms->push_back(std::move(seedProgram));
      
      seedCount++;
      std::cout << " - Success!\n";
    }
  }
  
  closedir(dir);
  std::cout << "Loaded " << seedCount << " seed files.\n";
}

bool deserializeAST(const uint8_t* data, size_t size, 
                   interpreter::Program& program) {
  if (size < 4) return false;
  
  // Simple approach: Use input bytes to select from seed programs
  // This gives libFuzzer meaningful starting points for mutation
  
  // Extract a seed selector from the first 4 bytes
  uint32_t selector = 0;
  memcpy(&selector, data, 4);
  
  // If we have loaded seed programs, use those
  if (g_seedPrograms && !g_seedPrograms->empty()) {
    size_t index = selector % g_seedPrograms->size();
    const auto& seedProgram = (*g_seedPrograms)[index];
    
    // Clone the seed program (deep copy)
    program.numThreadsX = seedProgram.numThreadsX;
    program.numThreadsY = seedProgram.numThreadsY;
    program.numThreadsZ = seedProgram.numThreadsZ;
    program.statements.clear();
    
    for (const auto& stmt : seedProgram.statements) {
      program.statements.push_back(stmt->clone());
    }
    
    return true;
  }
  
  // Otherwise, create different seed programs based on the selector
  switch (selector % 3) {
    case 0: {
      // Simple wave operation program
      program.numThreadsX = 4;
      program.numThreadsY = 1;
      program.numThreadsZ = 1;
      
      // var x = WaveGetLaneIndex() + 1;
      auto laneIndex = std::make_unique<interpreter::LaneIndexExpr>();
      auto one = std::make_unique<interpreter::LiteralExpr>(interpreter::Value(1));
      auto xInit = std::make_unique<interpreter::BinaryOpExpr>(
          std::move(laneIndex), std::move(one), interpreter::BinaryOpExpr::Add);
      auto xDecl = std::make_unique<interpreter::VarDeclStmt>("x", std::move(xInit));
      
      // var sum = 0;
      auto zero = std::make_unique<interpreter::LiteralExpr>(interpreter::Value(0));
      auto sumDecl = std::make_unique<interpreter::VarDeclStmt>("sum", std::move(zero));
      
      // sum = WaveActiveSum(x);
      auto xRef = std::make_unique<interpreter::VariableExpr>("x");
      auto waveSum = std::make_unique<interpreter::WaveActiveOp>(
          std::move(xRef), interpreter::WaveActiveOp::Sum);
      auto assignment = std::make_unique<interpreter::AssignStmt>("sum", std::move(waveSum));
      
      program.statements.push_back(std::move(xDecl));
      program.statements.push_back(std::move(sumDecl));
      program.statements.push_back(std::move(assignment));
      break;
    }
    
    case 1: {
      // Divergent control flow program
      program.numThreadsX = 4;
      program.numThreadsY = 1;
      program.numThreadsZ = 1;
      
      // var x = WaveGetLaneIndex() + 1;
      auto laneIndex = std::make_unique<interpreter::LaneIndexExpr>();
      auto one = std::make_unique<interpreter::LiteralExpr>(interpreter::Value(1));
      auto xInit = std::make_unique<interpreter::BinaryOpExpr>(
          std::move(laneIndex), std::move(one), interpreter::BinaryOpExpr::Add);
      auto xDecl = std::make_unique<interpreter::VarDeclStmt>("x", std::move(xInit));
      
      // var sum = 0;
      auto zero = std::make_unique<interpreter::LiteralExpr>(interpreter::Value(0));
      auto sumDecl = std::make_unique<interpreter::VarDeclStmt>("sum", std::move(zero));
      
      // if (WaveGetLaneIndex() < 2) { sum = WaveActiveSum(x); } else { sum = 0; }
      auto laneIndex2 = std::make_unique<interpreter::LaneIndexExpr>();
      auto two = std::make_unique<interpreter::LiteralExpr>(interpreter::Value(2));
      auto condition = std::make_unique<interpreter::BinaryOpExpr>(
          std::move(laneIndex2), std::move(two), interpreter::BinaryOpExpr::Lt);
      
      // Then block: sum = WaveActiveSum(x);
      std::vector<std::unique_ptr<interpreter::Statement>> thenStmts;
      auto xRef = std::make_unique<interpreter::VariableExpr>("x");
      auto waveSum = std::make_unique<interpreter::WaveActiveOp>(
          std::move(xRef), interpreter::WaveActiveOp::Sum);
      auto thenAssign = std::make_unique<interpreter::AssignStmt>("sum", std::move(waveSum));
      thenStmts.push_back(std::move(thenAssign));
      
      // Else block: sum = 0;
      std::vector<std::unique_ptr<interpreter::Statement>> elseStmts;
      auto zero2 = std::make_unique<interpreter::LiteralExpr>(interpreter::Value(0));
      auto elseAssign = std::make_unique<interpreter::AssignStmt>("sum", std::move(zero2));
      elseStmts.push_back(std::move(elseAssign));
      
      auto ifStmt = std::make_unique<interpreter::IfStmt>(
          std::move(condition), std::move(thenStmts), std::move(elseStmts));
      
      program.statements.push_back(std::move(xDecl));
      program.statements.push_back(std::move(sumDecl));
      program.statements.push_back(std::move(ifStmt));
      break;
    }
    
    default: {
      // Loop-based program
      program.numThreadsX = 8;
      program.numThreadsY = 1;
      program.numThreadsZ = 1;
      
      // var sum = 0;
      auto zero = std::make_unique<interpreter::LiteralExpr>(interpreter::Value(0));
      auto sumDecl = std::make_unique<interpreter::VarDeclStmt>("sum", std::move(zero));
      
      // for (var i = 0; i < 4; i++) { sum = WaveActiveSum(i); }
      auto iInit = std::make_unique<interpreter::LiteralExpr>(interpreter::Value(0));
      auto four = std::make_unique<interpreter::LiteralExpr>(interpreter::Value(4));
      auto iVar = std::make_unique<interpreter::VariableExpr>("i");
      auto condition = std::make_unique<interpreter::BinaryOpExpr>(
          std::move(iVar), std::move(four), interpreter::BinaryOpExpr::Lt);
      auto iVar2 = std::make_unique<interpreter::VariableExpr>("i");
      auto one = std::make_unique<interpreter::LiteralExpr>(interpreter::Value(1));
      auto increment = std::make_unique<interpreter::BinaryOpExpr>(
          std::move(iVar2), std::move(one), interpreter::BinaryOpExpr::Add);
      
      std::vector<std::unique_ptr<interpreter::Statement>> bodyStmts;
      auto iVar3 = std::make_unique<interpreter::VariableExpr>("i");
      auto waveSum = std::make_unique<interpreter::WaveActiveOp>(
          std::move(iVar3), interpreter::WaveActiveOp::Sum);
      auto bodyAssign = std::make_unique<interpreter::AssignStmt>("sum", std::move(waveSum));
      bodyStmts.push_back(std::move(bodyAssign));
      
      auto forStmt = std::make_unique<interpreter::ForStmt>(
          "i", std::move(iInit), std::move(condition), std::move(increment), std::move(bodyStmts));
      
      program.statements.push_back(std::move(sumDecl));
      program.statements.push_back(std::move(forStmt));
      break;
    }
  }
  
  return true;
}

size_t serializeAST(const interpreter::Program& program, 
                   uint8_t* data, size_t maxSize) {
  if (maxSize < 16) return 0; // Need at least 16 bytes
  
  // Simple approach: Create a fingerprint of the program that can guide future mutations
  // This is not a full serialization but provides feedback for libFuzzer
  
  // Store thread configuration (12 bytes)
  uint32_t threadInfo[3] = {
    static_cast<uint32_t>(program.numThreadsX),
    static_cast<uint32_t>(program.numThreadsY), 
    static_cast<uint32_t>(program.numThreadsZ)
  };
  memcpy(data, threadInfo, 12);
  
  // Compute a simple hash based on program structure (4 bytes)
  uint32_t hash = 0;
  hash ^= static_cast<uint32_t>(program.statements.size()) << 16;
  
  // Add type information from statements
  for (size_t i = 0; i < program.statements.size() && i < 8; ++i) {
    const auto& stmt = program.statements[i];
    uint32_t typeHash = 0;
    
    // Different statement types get different hash contributions
    if (dynamic_cast<const interpreter::VarDeclStmt*>(stmt.get())) {
      typeHash = 1;
    } else if (dynamic_cast<const interpreter::AssignStmt*>(stmt.get())) {
      typeHash = 2;
      // Check if it contains a wave operation
      auto* assign = static_cast<const interpreter::AssignStmt*>(stmt.get());
      if (dynamic_cast<const interpreter::WaveActiveOp*>(assign->getExpression())) {
        typeHash |= 0x10; // Wave operation flag
      }
    } else if (dynamic_cast<const interpreter::IfStmt*>(stmt.get())) {
      typeHash = 3;
    } else if (dynamic_cast<const interpreter::ForStmt*>(stmt.get())) {
      typeHash = 4;
    } else {
      typeHash = 5;
    }
    
    hash ^= typeHash << (i * 4);
  }
  
  memcpy(data + 12, &hash, 4);
  
  return 16; // Return the number of bytes written
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
  
  size_t result = minihlsl::fuzzer::serializeAST(*mutated, data, maxSize);
  if (result > 0 && minihlsl::fuzzer::g_fuzzer) {
    // Success - our custom mutator is working
  }
  return result;
}

size_t LLVMFuzzerCustomCrossOver(const uint8_t* data1, size_t size1,
                                const uint8_t* data2, size_t size2,
                                uint8_t* out, size_t maxOutSize,
                                unsigned int seed) {
  // Crossover two ASTs by combining statements from both programs
  minihlsl::interpreter::Program prog1, prog2;
  
  // Deserialize both inputs
  if (!minihlsl::fuzzer::deserializeAST(data1, size1, prog1) || 
      !minihlsl::fuzzer::deserializeAST(data2, size2, prog2)) {
    return 0; // Failed to deserialize
  }
  
  // Create a new program with crossover
  minihlsl::interpreter::Program crossover;
  crossover.numThreadsX = prog1.numThreadsX;
  crossover.numThreadsY = prog1.numThreadsY;
  crossover.numThreadsZ = prog1.numThreadsZ;
  
  // Use a random number generator for crossover decisions
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  
  // Crossover strategy: randomly select statements from both programs
  size_t totalStatements = prog1.statements.size() + prog2.statements.size();
  if (totalStatements == 0) return 0;
  
  // Take first half from prog1, second half from prog2 (with some randomness)
  for (size_t i = 0; i < prog1.statements.size(); ++i) {
    if (dist(rng) < 0.7f) { // 70% chance to take from prog1
      crossover.statements.push_back(prog1.statements[i]->clone());
    }
  }
  
  for (size_t i = 0; i < prog2.statements.size(); ++i) {
    if (dist(rng) < 0.3f) { // 30% chance to take from prog2
      crossover.statements.push_back(prog2.statements[i]->clone());
    }
  }
  
  // If we ended up with no statements, take at least one from each
  if (crossover.statements.empty()) {
    if (!prog1.statements.empty()) {
      crossover.statements.push_back(prog1.statements[0]->clone());
    }
    if (!prog2.statements.empty()) {
      crossover.statements.push_back(prog2.statements[0]->clone());
    }
  }
  
  // Serialize the crossover result
  size_t resultSize = minihlsl::fuzzer::serializeAST(crossover, out, maxOutSize);
  return resultSize;
}

} // extern "C"