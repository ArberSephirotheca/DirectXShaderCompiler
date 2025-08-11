#include "MiniHLSLInterpreterFuzzer.h"
#include "MiniHLSLInterpreter.h"
#include <random>
#include <set>
#include <algorithm>
#include <sstream>

namespace minihlsl {
namespace fuzzer {

// PrecomputeWaveResultsMutation Implementation

// Helper to check if an expression is or contains a wave operation
bool expressionContainsWaveOp(const interpreter::Expression* expr) {
  if (!expr) return false;
  
  // Direct wave operation
  if (dynamic_cast<const interpreter::WaveActiveOp*>(expr)) {
    return true;
  }
  
  // Check binary operations
  if (auto binOp = dynamic_cast<const interpreter::BinaryOpExpr*>(expr)) {
    // Would need getters for left_ and right_
    return false; // TODO: Need getters
  }
  
  // TODO: Check other expression types that might contain wave ops
  return false;
}

// Check if statement contains wave operations
bool containsWaveOps(const interpreter::Statement* stmt) {
  // Check if this is an expression statement containing a wave op
  if (auto exprStmt = dynamic_cast<const interpreter::ExprStmt*>(stmt)) {
    return expressionContainsWaveOp(exprStmt->getExpression());
  }
  
  if (auto assignStmt = dynamic_cast<const interpreter::AssignStmt*>(stmt)) {
    return expressionContainsWaveOp(assignStmt->getExpression());
  }
  
  if (auto varDeclStmt = dynamic_cast<const interpreter::VarDeclStmt*>(stmt)) {
    return expressionContainsWaveOp(varDeclStmt->getInit());
  }
  
  return false;
}

// Find wave op record for a statement
const ExecutionTrace::WaveOpRecord* findWaveOpRecord(
    const interpreter::Statement* stmt,
    const ExecutionTrace& trace) {
  // Search through wave operations to find one matching this statement
  for (const auto& waveOp : trace.waveOperations) {
    if (waveOp.instruction == stmt) {
      return &waveOp;
    }
  }
  return nullptr;
}

// Extract variable name that stores wave op result
std::string extractTargetVariable(const interpreter::Statement* stmt) {
  if (auto assignStmt = dynamic_cast<const interpreter::AssignStmt*>(stmt)) {
    //TODO: We need getter for varName_
    return "";
  }
  
  if (auto varDeclStmt = dynamic_cast<const interpreter::VarDeclStmt*>(stmt)) {
    // TODO: We need getter for name_
    return "";
  }
  
  return "";
}

// Create statement to replace wave op with precomputed value
std::unique_ptr<interpreter::Statement> createPrecomputedReplacement(
    const interpreter::Statement* stmt,
    const ExecutionTrace::WaveOpRecord* waveOp,
    const ExecutionTrace& trace) {
  
  // For now, return nullptr as we need AST cloning
  // TODO: Implement once AST cloning is available
  return nullptr;
}



// ExplicitLaneDivergenceMutation Implementation

std::unique_ptr<interpreter::Expression> createLaneIdCheck(
    interpreter::LaneId laneId) {
  // Create: laneIndex() == laneId
  auto laneIndexExpr = std::make_unique<interpreter::LaneIndexExpr>();
  auto laneIdLiteral = std::make_unique<interpreter::LiteralExpr>(
      interpreter::Value(static_cast<int>(laneId)));
  
  return std::make_unique<interpreter::BinaryOpExpr>(
      std::move(laneIndexExpr),
      std::move(laneIdLiteral),
      interpreter::BinaryOpExpr::Eq);
}

std::unique_ptr<interpreter::Expression> createWaveIdCheck(
    interpreter::WaveId waveId) {
  // Create: waveIndex() == waveId
  auto waveIndexExpr = std::make_unique<interpreter::WaveIndexExpr>();
  auto waveIdLiteral = std::make_unique<interpreter::LiteralExpr>(
      interpreter::Value(static_cast<int>(waveId)));
  
  return std::make_unique<interpreter::BinaryOpExpr>(
      std::move(waveIndexExpr),
      std::move(waveIdLiteral),
      interpreter::BinaryOpExpr::Eq);
}

std::unique_ptr<interpreter::Expression> createConjunction(
    std::unique_ptr<interpreter::Expression> left,
    std::unique_ptr<interpreter::Expression> right) {
  // Create: left && right
  return std::make_unique<interpreter::BinaryOpExpr>(
      std::move(left),
      std::move(right),
      interpreter::BinaryOpExpr::And);
}

std::unique_ptr<interpreter::Expression> createDisjunction(
    std::unique_ptr<interpreter::Expression> left,
    std::unique_ptr<interpreter::Expression> right) {
  // Create: left || right
  return std::make_unique<interpreter::BinaryOpExpr>(
      std::move(left),
      std::move(right),
      interpreter::BinaryOpExpr::Or);
}

// Helper functions for mutations

std::unique_ptr<interpreter::Statement> createGuardedIteration(
    const interpreter::Statement* loopStmt,
    uint32_t iteration,
    const std::map<interpreter::WaveId, std::set<interpreter::LaneId>>& activeWaveLanes) {
  
  // Create a guarded version of the loop body for a specific iteration
  // TODO: Need proper AST construction
  return nullptr;
}

// LanePermutationMutation Implementation

bool LanePermutationMutation::canApply(const interpreter::Statement* stmt, 
                                       const ExecutionTrace& trace) const {
  // Only apply to assignments containing associative wave operations
  auto* assign = dynamic_cast<const interpreter::AssignStmt*>(stmt);
  if (!assign) {
    return false;
  }
  
  auto* waveOp = dynamic_cast<const interpreter::WaveActiveOp*>(assign->getExpression());
  if (!waveOp) {
    return false;
  }
  
  // Check if operation is associative
  auto opType = waveOp->getOpType();
  return (opType == interpreter::WaveActiveOp::Sum ||
          opType == interpreter::WaveActiveOp::Product ||
          opType == interpreter::WaveActiveOp::And ||
          opType == interpreter::WaveActiveOp::Or ||
          opType == interpreter::WaveActiveOp::Xor);
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
  uint32_t activeLaneCount = 4; // default fallback
  uint32_t blockIdForWaveOp = 0;
  std::set<interpreter::LaneId> participatingLanes;
  
  // First, try to find the exact wave operation in the trace
  for (const auto& waveOpRecord : trace.waveOperations) {
    // Match by enum type for precise matching
    if (waveOpRecord.waveOpEnumType >= 0 && 
        waveOpRecord.waveOpEnumType == static_cast<int>(waveOp->getOpType())) {
      activeLaneCount = waveOpRecord.arrivedParticipants.size();
      blockIdForWaveOp = waveOpRecord.blockId;
      participatingLanes = waveOpRecord.arrivedParticipants;
      break;
    }
    // Fallback to string matching if enum not available
    else if (waveOpRecord.opType.find(waveOp->toString()) != std::string::npos) {
      activeLaneCount = waveOpRecord.arrivedParticipants.size();
      blockIdForWaveOp = waveOpRecord.blockId;
      participatingLanes = waveOpRecord.arrivedParticipants;
      break;
    }
  }
  
  // If we found the block, get more detailed participation info
  if (blockIdForWaveOp > 0 && trace.blocks.count(blockIdForWaveOp)) {
    const auto& block = trace.blocks.at(blockIdForWaveOp);
    // Check wave participation in this specific block
    for (const auto& [waveId, participation] : block.waveParticipation) {
      if (!participation.participatingLanes.empty()) {
        activeLaneCount = participation.participatingLanes.size();
        if (participatingLanes.empty()) {
          participatingLanes = participation.participatingLanes;
        }
        break;
      }
    }
  }
  
  // Ensure we don't create invalid permutations
  if (activeLaneCount == 0) {
    activeLaneCount = 4; // Default fallback
  }
  
  // If we didn't find specific participating lanes, use default consecutive lanes
  if (participatingLanes.empty()) {
    for (uint32_t i = 0; i < activeLaneCount; ++i) {
      participatingLanes.insert(i);
    }
  }
  
  // Choose a random permutation type
  static std::mt19937 rng(std::random_device{}());
  std::uniform_int_distribution<int> dist(0, 2);
  PermutationType permType = static_cast<PermutationType>(dist(rng));
  
  // Create compound statement
  std::vector<std::unique_ptr<interpreter::Statement>> statements;
  
  // 1. Create permuted lane index
  auto laneExpr = std::make_unique<interpreter::LaneIndexExpr>();
  auto permutedLaneExpr = createPermutationExpr(permType, std::move(laneExpr), 
                                                 activeLaneCount, participatingLanes);
  statements.push_back(std::make_unique<interpreter::VarDeclStmt>(
      "_perm_lane", interpreter::HLSLType::Uint, std::move(permutedLaneExpr)));
  
  // 2. Clone the input expression and create the WaveReadLaneAt
  auto clonedInput = inputExpr->clone();
  auto permLaneVar = std::make_unique<interpreter::VariableExpr>("_perm_lane");
  auto readLaneAt = std::make_unique<interpreter::WaveReadLaneAt>(
      std::move(clonedInput), std::move(permLaneVar));
  
  // Create a unique variable name for the permuted value
  static int permVarCounter = 0;
  std::string permVarName = "_perm_val_" + std::to_string(permVarCounter++);
  
  // Get the type from the input expression
  interpreter::HLSLType valueType = inputExpr->getType();
  
  statements.push_back(std::make_unique<interpreter::VarDeclStmt>(
      permVarName, valueType, std::move(readLaneAt)));
  
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
    uint32_t activeLaneCount,
    const std::set<interpreter::LaneId>& participatingLanes) const {
  
  // Convert set to vector for indexed access
  std::vector<interpreter::LaneId> laneList(participatingLanes.begin(), participatingLanes.end());
  std::sort(laneList.begin(), laneList.end());
  
  // Check if participants are consecutive starting from the first lane
  bool isConsecutive = true;
  if (!laneList.empty()) {
    for (size_t i = 1; i < laneList.size(); ++i) {
      if (laneList[i] != laneList[0] + i) {
        isConsecutive = false;
        break;
      }
    }
  }
  
  switch (type) {
    case PermutationType::Rotate: {
      // General rotation that works for any set of participating lanes
      if (laneList.size() < 2) {
        return laneExpr;
      }
      
      // Optimization for consecutive participants
      if (isConsecutive && laneList.size() > 1) {
        // Use mathematical formula: ((laneId - firstLane + 1) % count) + firstLane
        auto firstLane = std::make_unique<interpreter::LiteralExpr>(static_cast<int>(laneList[0]));
        auto count = std::make_unique<interpreter::LiteralExpr>(static_cast<int>(laneList.size()));
        auto one = std::make_unique<interpreter::LiteralExpr>(1);
        
        // (laneId - firstLane + 1)
        auto offsetFromFirst = std::make_unique<interpreter::BinaryOpExpr>(
            laneExpr->clone(), firstLane->clone(), interpreter::BinaryOpExpr::Sub);
        auto offsetPlusOne = std::make_unique<interpreter::BinaryOpExpr>(
            std::move(offsetFromFirst), std::move(one), interpreter::BinaryOpExpr::Add);
        
        // % count
        auto modResult = std::make_unique<interpreter::BinaryOpExpr>(
            std::move(offsetPlusOne), std::move(count), interpreter::BinaryOpExpr::Mod);
        
        // + firstLane
        return std::make_unique<interpreter::BinaryOpExpr>(
            std::move(modResult), std::move(firstLane), interpreter::BinaryOpExpr::Add);
      }
      
      // For non-consecutive participants, we need a more complex conditional chain
      // This is handled in the original code but simplified here
      return laneExpr;
    }
    
    case PermutationType::Reverse: {
      if (laneList.size() < 2) {
        return laneExpr;
      }
      
      // For consecutive participants: firstLane + lastLane - laneId
      if (isConsecutive) {
        auto firstLane = std::make_unique<interpreter::LiteralExpr>(static_cast<int>(laneList.front()));
        auto lastLane = std::make_unique<interpreter::LiteralExpr>(static_cast<int>(laneList.back()));
        auto sum = std::make_unique<interpreter::BinaryOpExpr>(
            std::move(firstLane), std::move(lastLane), interpreter::BinaryOpExpr::Add);
        return std::make_unique<interpreter::BinaryOpExpr>(
            std::move(sum), std::move(laneExpr), interpreter::BinaryOpExpr::Sub);
      }
      
      // For non-consecutive, would need complex conditionals
      return laneExpr;
    }
    
    case PermutationType::EvenOddSwap: {
      // Simple XOR with 1 to swap even/odd lanes
      auto one = std::make_unique<interpreter::LiteralExpr>(1);
      return std::make_unique<interpreter::BinaryOpExpr>(
          std::move(laneExpr), std::move(one), interpreter::BinaryOpExpr::Xor);
    }
    
    default:
      return laneExpr;
  }
}

// WaveParticipantTrackingMutation Implementation

bool WaveParticipantTrackingMutation::canApply(const interpreter::Statement* stmt,
                                               const ExecutionTrace& trace) const {
  // Can apply to any statement with wave operations
  if (auto* assign = dynamic_cast<const interpreter::AssignStmt*>(stmt)) {
    return dynamic_cast<const interpreter::WaveActiveOp*>(assign->getExpression()) != nullptr;
  }
  if (auto* varDecl = dynamic_cast<const interpreter::VarDeclStmt*>(stmt)) {
    if (varDecl->getInit()) {
      return dynamic_cast<const interpreter::WaveActiveOp*>(varDecl->getInit()) != nullptr;
    }
  }
  return false;
}

std::unique_ptr<interpreter::Statement> WaveParticipantTrackingMutation::apply(
    const interpreter::Statement* stmt,
    const ExecutionTrace& trace) const {
  
  // Handle AssignStmt
  const interpreter::WaveActiveOp* waveOp = nullptr;
  std::string resultVarName;
  
  if (auto* assign = dynamic_cast<const interpreter::AssignStmt*>(stmt)) {
    waveOp = dynamic_cast<const interpreter::WaveActiveOp*>(assign->getExpression());
    if (!waveOp) return stmt->clone();
    resultVarName = assign->getName();
  }
  // Handle VarDeclStmt
  else if (auto* varDecl = dynamic_cast<const interpreter::VarDeclStmt*>(stmt)) {
    if (varDecl->getInit()) {
      waveOp = dynamic_cast<const interpreter::WaveActiveOp*>(varDecl->getInit());
      if (!waveOp) return stmt->clone();
      resultVarName = varDecl->getName();
    } else {
      return stmt->clone();
    }
  }
  else {
    return stmt->clone();
  }
  
  // Find expected participants from trace
  uint32_t expectedParticipants = 4; // default
  uint32_t blockId = 0;
  
  // Find the wave operation in the trace to get its block ID
  for (const auto& waveOpRecord : trace.waveOperations) {
    if (waveOpRecord.instruction == static_cast<const void*>(waveOp) ||
        waveOpRecord.opType == waveOp->toString()) {
      expectedParticipants = waveOpRecord.arrivedParticipants.size();
      blockId = waveOpRecord.blockId;
      
      // Also check the block's wave participation info for more accurate count
      if (trace.blocks.count(blockId)) {
        const auto& block = trace.blocks.at(blockId);
        // Assuming wave 0 for simplicity
        if (block.waveParticipation.count(0)) {
          const auto& waveInfo = block.waveParticipation.at(0);
          if (!waveInfo.participatingLanes.empty()) {
            expectedParticipants = waveInfo.participatingLanes.size();
          }
        }
      }
      break;
    }
  }
  
  // Create compound statement with original operation plus tracking
  std::vector<std::unique_ptr<interpreter::Statement>> statements;
  
  // 1. Original wave operation
  statements.push_back(stmt->clone());
  
  // 2. Create tracking statements
  auto trackingStmts = createTrackingStatements(waveOp, resultVarName, expectedParticipants);
  for (auto& stmt : trackingStmts) {
    statements.push_back(std::move(stmt));
  }
  
  // Wrap in if(true) to create single statement
  auto trueCond = std::make_unique<interpreter::LiteralExpr>(interpreter::Value(true));
  return std::make_unique<interpreter::IfStmt>(
      std::move(trueCond), std::move(statements),
      std::vector<std::unique_ptr<interpreter::Statement>>{});
}

std::vector<std::unique_ptr<interpreter::Statement>>
WaveParticipantTrackingMutation::createTrackingStatements(
    const interpreter::WaveActiveOp* waveOp,
    const std::string& resultVar,
    uint32_t expectedParticipants) const {
  
  std::vector<std::unique_ptr<interpreter::Statement>> statements;
  
  // 1. Count active participants: _participantCount = WaveActiveSum(1)
  auto oneExpr = std::make_unique<interpreter::LiteralExpr>(interpreter::Value(1));
  auto sumOp = std::make_unique<interpreter::WaveActiveOp>(
      std::move(oneExpr), interpreter::WaveActiveOp::Sum);
  
  statements.push_back(std::make_unique<interpreter::VarDeclStmt>(
      "_participantCount", interpreter::HLSLType::Uint, std::move(sumOp)));
  
  // 2. Check if count matches expected: isCorrect = (participantCount == expectedCount)
  auto countRef = std::make_unique<interpreter::VariableExpr>("_participantCount");
  auto expected = std::make_unique<interpreter::LiteralExpr>(static_cast<int>(expectedParticipants));
  auto compare = std::make_unique<interpreter::BinaryOpExpr>(
      std::move(countRef), std::move(expected), interpreter::BinaryOpExpr::Eq);
  
  statements.push_back(std::make_unique<interpreter::VarDeclStmt>(
      "_isCorrect", interpreter::HLSLType::Bool, std::move(compare)));
  
  // 3. Accumulate the result in a global buffer at tid.x index
  // _participant_check_sum[tid.x] += uint(_isCorrect)
  auto tidX = std::make_unique<interpreter::DispatchThreadIdExpr>(0); // component 0 = x
  
  // Convert bool to uint
  auto isCorrectRef = std::make_unique<interpreter::VariableExpr>("_isCorrect");
  auto oneVal = std::make_unique<interpreter::LiteralExpr>(1);
  auto zeroVal = std::make_unique<interpreter::LiteralExpr>(0);
  auto boolToUint = std::make_unique<interpreter::ConditionalExpr>(
      std::move(isCorrectRef), std::move(oneVal), std::move(zeroVal));
  
  // Read current buffer value
  auto bufferAccess = std::make_unique<interpreter::ArrayAccessExpr>(
      "_participant_check_sum", tidX->clone(), interpreter::HLSLType::Uint);
  
  // Add to current value
  auto addExpr = std::make_unique<interpreter::BinaryOpExpr>(
      std::move(bufferAccess), std::move(boolToUint), interpreter::BinaryOpExpr::Add);
  
  // Store back to buffer
  statements.push_back(std::make_unique<interpreter::ArrayAssignStmt>(
      "_participant_check_sum", std::move(tidX), std::move(addExpr)));
  
  return statements;
}

bool LanePermutationMutation::validateSemanticPreservation(
    const interpreter::Statement* original,
    const interpreter::Statement* mutated,
    const ExecutionTrace& trace) const {
  // Lane permutation preserves semantics by design for associative operations
  return true;
}

bool WaveParticipantTrackingMutation::validateSemanticPreservation(
    const interpreter::Statement* original,
    const interpreter::Statement* mutated,
    const ExecutionTrace& trace) const {
  // Participant tracking doesn't change the computation result
  return true;
}

// DataTransformMutation - stub implementation
bool DataTransformMutation::canApply(const interpreter::Statement* stmt,
                                    const ExecutionTrace& trace) const {
  // TODO: Implement
  return false;
}

std::unique_ptr<interpreter::Statement> DataTransformMutation::apply(
    const interpreter::Statement* stmt,
    const ExecutionTrace& trace) const {
  return stmt->clone();
}

bool DataTransformMutation::validateSemanticPreservation(
    const interpreter::Statement* original,
    const interpreter::Statement* mutated,
    const ExecutionTrace& trace) const {
  // TODO: Implement validation
  return true;
}

// RedundantComputeMutation - stub implementation
bool RedundantComputeMutation::canApply(const interpreter::Statement* stmt,
                                        const ExecutionTrace& trace) const {
  // TODO: Implement
  return false;
}

std::unique_ptr<interpreter::Statement> RedundantComputeMutation::apply(
    const interpreter::Statement* stmt,
    const ExecutionTrace& trace) const {
  return stmt->clone();
}

bool RedundantComputeMutation::validateSemanticPreservation(
    const interpreter::Statement* original,
    const interpreter::Statement* mutated,
    const ExecutionTrace& trace) const {
  // TODO: Implement validation
  return true;
}

// serializeProgramToString implementation
std::string serializeProgramToString(const interpreter::Program& program) {
  std::stringstream ss;
  
  std::cerr << "DEBUG: serializeProgramToString - globalBuffers.size() = " << program.globalBuffers.size() << "\n";
  
  // Add global buffer declarations
  for (const auto& buffer : program.globalBuffers) {
    ss << buffer.bufferType << "<";
    ss << interpreter::HLSLTypeInfo::toString(buffer.elementType);
    ss << "> " << buffer.name;
    ss << " : register(" << (buffer.isReadWrite ? "u" : "t") 
       << buffer.registerIndex << ");\n";
  }
  
  if (!program.globalBuffers.empty()) {
    ss << "\n";
  }
  
  // Add thread configuration
  ss << "[numthreads(" << program.numThreadsX << ", " 
     << program.numThreadsY << ", " 
     << program.numThreadsZ << ")]\n";
  
  // Add WaveSize attribute if specified
  if (program.waveSizeMin > 0 || program.waveSizeMax > 0 || program.waveSizePreferred > 0) {
    ss << "[WaveSize(";
    if (program.waveSizeMin > 0) ss << program.waveSizeMin;
    if (program.waveSizeMax > 0) ss << ", " << program.waveSizeMax;
    if (program.waveSizePreferred > 0) ss << ", " << program.waveSizePreferred;
    ss << ")]\n";
  }
  
  ss << "void main(";
  
  // Add function parameters with semantics
  bool first = true;
  for (const auto& param : program.entryInputs.parameters) {
    if (!first) {
      ss << ",\n          ";
    }
    first = false;
    
    // Output type
    ss << interpreter::HLSLTypeInfo::toString(param.type);
    ss << " " << param.name;
    
    // Output semantic
    if (param.semantic != interpreter::HLSLSemantic::None) {
      ss << " : " << interpreter::HLSLSemanticInfo::toString(param.semantic);
    }
  }
  
  ss << ") {\n";
  
  // Add statements
  for (const auto& stmt : program.statements) {
    ss << "  " << stmt->toString() << "\n";
  }
  
  ss << "}\n";
  
  return ss.str();
}

} // namespace fuzzer
} // namespace minihlsl