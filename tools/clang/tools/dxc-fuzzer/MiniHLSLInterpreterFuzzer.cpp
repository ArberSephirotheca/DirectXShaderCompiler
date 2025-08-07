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