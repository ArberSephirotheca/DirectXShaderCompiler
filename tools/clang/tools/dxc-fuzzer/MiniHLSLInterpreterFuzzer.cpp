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
    if (param.type == interpreter::HLSLType::Custom && !param.customTypeStr.empty()) {
      ss << param.customTypeStr;
    }
    
    // Output parameter name
    ss << " " << param.name;
    
    // Output semantic if present
    if (param.semantic != interpreter::HLSLSemantic::None) {
      ss << " : " << interpreter::HLSLSemanticInfo::toString(param.semantic);
      if (param.semantic == interpreter::HLSLSemantic::Custom && !param.customSemanticStr.empty()) {
        ss << param.customSemanticStr;
      }
    }
  }
  
  ss << ") {\n";
  
  // Add all statements
  for (const auto& stmt : program.statements) {
    ss << "  " << stmt->toString() << "\n";
  }
  
  ss << "}\n";
  return ss.str();
}

// ===== Semantics-Preserving Mutation Implementations =====

bool LanePermutationMutation::canApply(const interpreter::Statement* stmt, 
                                       const ExecutionTrace& trace) const {
  // Check if this is an assignment with a wave operation
  if (auto* assign = dynamic_cast<const interpreter::AssignStmt*>(stmt)) {
    // Use dynamic_cast to check for wave operations instead of string matching
    if (auto* waveOp = dynamic_cast<const interpreter::WaveActiveOp*>(assign->getExpression())) {
      // Check if this is an associative operation
      auto opType = waveOp->getOpType();
      bool isAssociative = opType == interpreter::WaveActiveOp::Sum ||
                          opType == interpreter::WaveActiveOp::Product ||
                          opType == interpreter::WaveActiveOp::And ||
                          opType == interpreter::WaveActiveOp::Or ||
                          opType == interpreter::WaveActiveOp::Xor;
      
      if (!isAssociative) {
        return false;
      }
      
      // Additionally check if the operation uses thread ID variables
      // This makes the mutation more relevant for testing built-in variable handling
      const auto* inputExpr = waveOp->getInput();
      if (inputExpr && usesThreadIdVariables(inputExpr)) {
        // Prioritize mutations on operations that use thread IDs
        return true;
      }
      
      // Still allow mutations on other associative operations
      return true;
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
  std::set<interpreter::LaneId> participatingLanes; // Actual lane IDs that participate
  
  // Debug: print what we're looking for
  std::cout << "[LanePermutation] Looking for wave op: " << waveOp->toString() 
            << " ptr=" << static_cast<const void*>(waveOp) << "\n";
  
  // First, try to find the exact wave operation in the trace
  for (const auto& waveOpRecord : trace.waveOperations) {
    std::cout << "[LanePermutation] Checking trace wave op: " << waveOpRecord.opType 
              << " enumType=" << waveOpRecord.waveOpEnumType
              << " participants=" << waveOpRecord.arrivedParticipants.size() << "\n";
    
    // Match by enum type for precise matching
    if (waveOpRecord.waveOpEnumType >= 0 && 
        waveOpRecord.waveOpEnumType == static_cast<int>(waveOp->getOpType())) {
      // Found a matching wave operation type by enum
      activeLaneCount = waveOpRecord.arrivedParticipants.size();
      blockIdForWaveOp = waveOpRecord.blockId;
      participatingLanes = waveOpRecord.arrivedParticipants;
      std::cout << "[LanePermutation] Found match by enum! Active lanes: " << activeLaneCount << "\n";
      std::cout << "[LanePermutation] Participants: ";
      for (auto laneId : waveOpRecord.arrivedParticipants) {
        std::cout << laneId << " ";
      }
      std::cout << "\n";
      break;
    }
    // Fallback to string matching if enum not available
    else if (waveOpRecord.opType.find(waveOp->toString()) != std::string::npos) {
      // Found a matching wave operation type
      activeLaneCount = waveOpRecord.arrivedParticipants.size();
      blockIdForWaveOp = waveOpRecord.blockId;
      participatingLanes = waveOpRecord.arrivedParticipants;
      std::cout << "[LanePermutation] Found match by string! Active lanes: " << activeLaneCount << "\n";
      std::cout << "[LanePermutation] Participants: ";
      for (auto laneId : waveOpRecord.arrivedParticipants) {
        std::cout << laneId << " ";
      }
      std::cout << "\n";
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
  
  // Get the type from the input expression (WaveReadLaneAt returns the same type as its input)
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
        // If only one lane participates, return the same lane
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
      
      // Build a conditional expression chain: 
      // (laneId == lane0) ? lane1 : ((laneId == lane1) ? lane2 : ... )
      std::unique_ptr<interpreter::Expression> result = nullptr;
      
      for (size_t i = 0; i < laneList.size(); ++i) {
        size_t nextIdx = (i + 1) % laneList.size();
        auto currentLane = std::make_unique<interpreter::LiteralExpr>(static_cast<int>(laneList[i]));
        auto nextLane = std::make_unique<interpreter::LiteralExpr>(static_cast<int>(laneList[nextIdx]));
        
        auto condition = std::make_unique<interpreter::BinaryOpExpr>(
            laneExpr->clone(), std::move(currentLane), interpreter::BinaryOpExpr::Eq);
        
        if (!result) {
          // Last condition in the chain (or first being built)
          result = std::move(nextLane);
        } else {
          result = std::make_unique<interpreter::ConditionalExpr>(
              std::move(condition), std::move(nextLane), std::move(result));
        }
      }
      
      return result;
    }
    
    case PermutationType::Reverse: {
      // Reverse mapping for arbitrary participating lanes
      // Maps lane[i] -> lane[n-1-i]
      
      if (laneList.size() < 2) {
        return laneExpr;
      }
      
      // Optimization for consecutive participants
      if (isConsecutive && laneList.size() > 1) {
        // Use mathematical formula: (firstLane + lastLane) - laneId
        auto firstLane = std::make_unique<interpreter::LiteralExpr>(static_cast<int>(laneList[0]));
        auto lastLane = std::make_unique<interpreter::LiteralExpr>(static_cast<int>(laneList.back()));
        
        // firstLane + lastLane
        auto sum = std::make_unique<interpreter::BinaryOpExpr>(
            std::move(firstLane), std::move(lastLane), interpreter::BinaryOpExpr::Add);
        
        // (firstLane + lastLane) - laneId
        return std::make_unique<interpreter::BinaryOpExpr>(
            std::move(sum), laneExpr->clone(), interpreter::BinaryOpExpr::Sub);
      }
      
      // Build conditional chain for reverse mapping
      std::unique_ptr<interpreter::Expression> result = nullptr;
      
      for (size_t i = 0; i < laneList.size(); ++i) {
        size_t reverseIdx = laneList.size() - 1 - i;
        auto currentLane = std::make_unique<interpreter::LiteralExpr>(static_cast<int>(laneList[i]));
        auto reverseLane = std::make_unique<interpreter::LiteralExpr>(static_cast<int>(laneList[reverseIdx]));
        
        auto condition = std::make_unique<interpreter::BinaryOpExpr>(
            laneExpr->clone(), std::move(currentLane), interpreter::BinaryOpExpr::Eq);
        
        if (!result) {
          result = std::move(reverseLane);
        } else {
          result = std::make_unique<interpreter::ConditionalExpr>(
              std::move(condition), std::move(reverseLane), std::move(result));
        }
      }
      
      return result;
    }
    
    case PermutationType::EvenOddSwap: {
      // Even/odd swap for arbitrary participating lanes
      // Pairs lanes: [0]<->[1], [2]<->[3], etc.
      // If odd number of lanes, last lane maps to itself
      
      if (laneList.size() < 2) {
        return laneExpr;
      }
      
      // Optimization for consecutive participants - but we need to handle unsigned arithmetic carefully
      if (isConsecutive && laneList.size() > 1 && laneList.size() <= 4) {
        // For small consecutive sets, build specific patterns to avoid underflow
        if (laneList.size() == 2) {
          // Simple swap: lane0 <-> lane1
          auto lane0 = std::make_unique<interpreter::LiteralExpr>(static_cast<int>(laneList[0]));
          auto lane1 = std::make_unique<interpreter::LiteralExpr>(static_cast<int>(laneList[1]));
          auto isLane0 = std::make_unique<interpreter::BinaryOpExpr>(
              laneExpr->clone(), lane0->clone(), interpreter::BinaryOpExpr::Eq);
          
          // lane0 ? lane1 : lane0
          return std::make_unique<interpreter::ConditionalExpr>(
              std::move(isLane0), std::move(lane1), std::move(lane0));
        } else if (laneList.size() == 4 && laneList[0] == 0) {
          // Special case for lanes 0,1,2,3 - can use safe arithmetic
          // Pattern: 0->1, 1->0, 2->3, 3->2
          auto one = std::make_unique<interpreter::LiteralExpr>(1);
          auto two = std::make_unique<interpreter::LiteralExpr>(2);
          
          // Check if lane < 2
          auto isFirstPair = std::make_unique<interpreter::BinaryOpExpr>(
              laneExpr->clone(), two->clone(), interpreter::BinaryOpExpr::Lt);
          
          // For lanes 0,1: (lane == 0) ? 1 : 0
          auto zero = std::make_unique<interpreter::LiteralExpr>(0);
          auto isZero = std::make_unique<interpreter::BinaryOpExpr>(
              laneExpr->clone(), zero->clone(), interpreter::BinaryOpExpr::Eq);
          auto firstPairSwap = std::make_unique<interpreter::ConditionalExpr>(
              std::move(isZero), one->clone(), std::move(zero));
          
          // For lanes 2,3: (lane == 2) ? 3 : 2  
          auto three = std::make_unique<interpreter::LiteralExpr>(3);
          auto isTwo = std::make_unique<interpreter::BinaryOpExpr>(
              laneExpr->clone(), two->clone(), interpreter::BinaryOpExpr::Eq);
          auto secondPairSwap = std::make_unique<interpreter::ConditionalExpr>(
              std::move(isTwo), std::move(three), std::move(two));
          
          // Combine: (lane < 2) ? firstPairSwap : secondPairSwap
          return std::make_unique<interpreter::ConditionalExpr>(
              std::move(isFirstPair), std::move(firstPairSwap), std::move(secondPairSwap));
        }
      }
      
      // Build conditional chain for even/odd swapping
      std::unique_ptr<interpreter::Expression> result = nullptr;
      
      for (size_t i = 0; i < laneList.size(); ++i) {
        size_t pairIdx;
        if (i % 2 == 0) {
          // Even index: swap with next (if exists)
          pairIdx = (i + 1 < laneList.size()) ? i + 1 : i;
        } else {
          // Odd index: swap with previous
          pairIdx = i - 1;
        }
        
        auto currentLane = std::make_unique<interpreter::LiteralExpr>(static_cast<int>(laneList[i]));
        auto pairLane = std::make_unique<interpreter::LiteralExpr>(static_cast<int>(laneList[pairIdx]));
        
        auto condition = std::make_unique<interpreter::BinaryOpExpr>(
            laneExpr->clone(), std::move(currentLane), interpreter::BinaryOpExpr::Eq);
        
        if (!result) {
          result = std::move(pairLane);
        } else {
          result = std::make_unique<interpreter::ConditionalExpr>(
              std::move(condition), std::move(pairLane), std::move(result));
        }
      }
      
      return result;
    }
    
    default:
      // Default to rotate
      return createPermutationExpr(PermutationType::Rotate, std::move(laneExpr), 
                                   activeLaneCount, participatingLanes);
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
    // Check if this is a built-in variable that contains thread ID information
    std::string varName = varExpr->toString();
    
    // Handle built-in variables that contain thread index information
    // These variables are semantically equivalent to lane indices for wave operations
    if (varName == "tid.x" || varName == "gtid.x" || varName == "gindex" ||
        varName.find("SV_DispatchThreadID") != std::string::npos ||
        varName.find("SV_GroupThreadID") != std::string::npos ||
        varName.find("SV_GroupIndex") != std::string::npos) {
      // Replace with permuted lane expression
      return permutedLaneExpr->clone();
    }
    
    // For other variables, just clone
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

bool LanePermutationMutation::usesThreadIdVariables(const interpreter::Expression* expr) const {
  // Check if this expression uses built-in thread ID variables
  if (auto varExpr = dynamic_cast<const interpreter::VariableExpr*>(expr)) {
    std::string varName = varExpr->toString();
    return varName == "tid.x" || varName == "tid.y" || varName == "tid.z" ||
           varName == "gtid.x" || varName == "gtid.y" || varName == "gtid.z" ||
           varName == "gid.x" || varName == "gid.y" || varName == "gid.z" ||
           varName == "gindex" ||
           varName == "tid" || varName == "gtid" || varName == "gid";
  }
  
  // Check LaneIndexExpr
  if (dynamic_cast<const interpreter::LaneIndexExpr*>(expr)) {
    return true;
  }
  
  // For compound expressions, would need to recurse, but for now return false
  return false;
}

bool LanePermutationMutation::validateSemanticPreservation(
    const interpreter::Statement* original,
    const interpreter::Statement* mutated,
    const ExecutionTrace& trace) const {
  // Associative operations should produce the same result with permuted lanes
  return true;
}

// ===== DataTransformMutation Implementation =====

bool DataTransformMutation::canApply(const interpreter::Statement* stmt,
                                   const ExecutionTrace& trace) const {
  // Can apply to assignments containing wave operations
  if (auto* assign = dynamic_cast<const interpreter::AssignStmt*>(stmt)) {
    return dynamic_cast<const interpreter::WaveActiveOp*>(assign->getExpression()) != nullptr;
  }
  return false;
}

std::unique_ptr<interpreter::Statement> DataTransformMutation::apply(
    const interpreter::Statement* stmt,
    const ExecutionTrace& trace) const {
  
  auto* assign = dynamic_cast<const interpreter::AssignStmt*>(stmt);
  if (!assign) return stmt->clone();
  
  auto* waveOp = dynamic_cast<const interpreter::WaveActiveOp*>(assign->getExpression());
  if (!waveOp) return stmt->clone();
  
  // Get the input expression
  const auto* inputExpr = waveOp->getInput();
  if (!inputExpr) return stmt->clone();
  
  // Choose a random transform
  static std::mt19937 rng(std::random_device{}());
  std::uniform_int_distribution<int> dist(0, 4);
  TransformType transformType = static_cast<TransformType>(dist(rng));
  
  // Apply the transform
  auto transformedInput = applyTransform(inputExpr->clone(), transformType);
  
  // Create new wave operation with transformed input
  auto newWaveOp = std::make_unique<interpreter::WaveActiveOp>(
      std::move(transformedInput), waveOp->getOpType());
  
  return std::make_unique<interpreter::AssignStmt>(
      assign->getName(), std::move(newWaveOp));
}

std::unique_ptr<interpreter::Expression> DataTransformMutation::applyTransform(
    std::unique_ptr<interpreter::Expression> expr,
    TransformType type) const {
  
  switch (type) {
    case TransformType::MultiplyDivide: {
      // x -> (x * 4) / 4
      auto four1 = std::make_unique<interpreter::LiteralExpr>(4);
      auto four2 = std::make_unique<interpreter::LiteralExpr>(4);
      
      auto multiply = std::make_unique<interpreter::BinaryOpExpr>(
          std::move(expr), std::move(four1), interpreter::BinaryOpExpr::Mul);
      
      return std::make_unique<interpreter::BinaryOpExpr>(
          std::move(multiply), std::move(four2), interpreter::BinaryOpExpr::Div);
    }
    
    case TransformType::AddSubtract: {
      // x -> (x + 100) - 100
      auto hundred1 = std::make_unique<interpreter::LiteralExpr>(100);
      auto hundred2 = std::make_unique<interpreter::LiteralExpr>(100);
      
      auto add = std::make_unique<interpreter::BinaryOpExpr>(
          std::move(expr), std::move(hundred1), interpreter::BinaryOpExpr::Add);
      
      return std::make_unique<interpreter::BinaryOpExpr>(
          std::move(add), std::move(hundred2), interpreter::BinaryOpExpr::Sub);
    }
    
    case TransformType::DoubleNegate: {
      // x -> -(-x)
      auto negate1 = std::make_unique<interpreter::UnaryOpExpr>(
          std::move(expr), interpreter::UnaryOpExpr::Neg);
      
      return std::make_unique<interpreter::UnaryOpExpr>(
          std::move(negate1), interpreter::UnaryOpExpr::Neg);
    }
    
    case TransformType::ShiftUnshift: {
      // Since we don't have shift operators, use multiply/divide by powers of 2
      // x -> (x * 4) / 4 (similar effect for positive integers)
      auto four1 = std::make_unique<interpreter::LiteralExpr>(4);
      auto four2 = std::make_unique<interpreter::LiteralExpr>(4);
      
      auto multiply = std::make_unique<interpreter::BinaryOpExpr>(
          std::move(expr), std::move(four1), interpreter::BinaryOpExpr::Mul);
      
      return std::make_unique<interpreter::BinaryOpExpr>(
          std::move(multiply), std::move(four2), interpreter::BinaryOpExpr::Div);
    }
    
    case TransformType::BitwiseIdentity: {
      // x -> x | 0
      auto zero = std::make_unique<interpreter::LiteralExpr>(0);
      
      return std::make_unique<interpreter::BinaryOpExpr>(
          std::move(expr), std::move(zero), interpreter::BinaryOpExpr::Or);
    }
    
    default:
      return expr;
  }
}

bool DataTransformMutation::validateSemanticPreservation(
    const interpreter::Statement* original,
    const interpreter::Statement* mutated,
    const ExecutionTrace& trace) const {
  // Algebraic transformations preserve semantics
  return true;
}

// ===== RedundantComputeMutation Implementation =====

bool RedundantComputeMutation::canApply(const interpreter::Statement* stmt,
                                       const ExecutionTrace& trace) const {
  // Can apply to any wave operation
  if (auto* assign = dynamic_cast<const interpreter::AssignStmt*>(stmt)) {
    return dynamic_cast<const interpreter::WaveActiveOp*>(assign->getExpression()) != nullptr;
  }
  return false;
}

std::unique_ptr<interpreter::Statement> RedundantComputeMutation::apply(
    const interpreter::Statement* stmt,
    const ExecutionTrace& trace) const {
  
  auto* assign = dynamic_cast<const interpreter::AssignStmt*>(stmt);
  if (!assign) return stmt->clone();
  
  auto* waveOp = dynamic_cast<const interpreter::WaveActiveOp*>(assign->getExpression());
  if (!waveOp) return stmt->clone();
  
  // Create a compound statement with redundant computations
  std::vector<std::unique_ptr<interpreter::Statement>> statements;
  
  // Add redundant temporary variables
  auto tempVar1 = std::make_unique<interpreter::VarDeclStmt>(
      "_temp1", waveOp->getInput()->clone());
  statements.push_back(std::move(tempVar1));
  
  // Redundant computation: temp2 = temp1 + 0
  auto temp1Ref = std::make_unique<interpreter::VariableExpr>("_temp1");
  auto zero = std::make_unique<interpreter::LiteralExpr>(0);
  auto addZero = std::make_unique<interpreter::BinaryOpExpr>(
      std::move(temp1Ref), std::move(zero), interpreter::BinaryOpExpr::Add);
  auto tempVar2 = std::make_unique<interpreter::VarDeclStmt>(
      "_temp2", std::move(addZero));
  statements.push_back(std::move(tempVar2));
  
  // Original wave operation with temp2
  auto temp2Ref = std::make_unique<interpreter::VariableExpr>("_temp2");
  auto newWaveOp = std::make_unique<interpreter::WaveActiveOp>(
      std::move(temp2Ref), waveOp->getOpType());
  auto newAssign = std::make_unique<interpreter::AssignStmt>(
      assign->getName(), std::move(newWaveOp));
  statements.push_back(std::move(newAssign));
  
  // Wrap in if(true) to create single statement
  auto trueCond = std::make_unique<interpreter::LiteralExpr>(interpreter::Value(true));
  return std::make_unique<interpreter::IfStmt>(
      std::move(trueCond), std::move(statements),
      std::vector<std::unique_ptr<interpreter::Statement>>{});
}

bool RedundantComputeMutation::validateSemanticPreservation(
    const interpreter::Statement* original,
    const interpreter::Statement* mutated,
    const ExecutionTrace& trace) const {
  // Redundant computations preserve semantics
  return true;
}

// ===== WaveParticipantTrackingMutation Implementation =====

bool WaveParticipantTrackingMutation::canApply(const interpreter::Statement* stmt,
                                              const ExecutionTrace& trace) const {
  // Can apply to assignments containing wave operations
  if (auto* assign = dynamic_cast<const interpreter::AssignStmt*>(stmt)) {
    if (dynamic_cast<const interpreter::WaveActiveOp*>(assign->getExpression()) != nullptr) {
      // Debug: print trace info
      std::cout << "[WaveParticipantTracking] canApply: Found wave op assignment\n";
      std::cout << "[WaveParticipantTracking] Trace has " << trace.waveOperations.size() 
                << " wave operations\n";
      for (size_t i = 0; i < trace.waveOperations.size(); ++i) {
        const auto& waveOp = trace.waveOperations[i];
        std::cout << "  Wave op " << i << ": " << waveOp.opType 
                  << " in block " << waveOp.blockId 
                  << " with " << waveOp.arrivedParticipants.size() << " participants\n";
      }
      return true;
    }
  }
  return false;
}

std::unique_ptr<interpreter::Statement> WaveParticipantTrackingMutation::apply(
    const interpreter::Statement* stmt,
    const ExecutionTrace& trace) const {
  
  auto* assign = dynamic_cast<const interpreter::AssignStmt*>(stmt);
  if (!assign) return stmt->clone();
  
  auto* waveOp = dynamic_cast<const interpreter::WaveActiveOp*>(assign->getExpression());
  if (!waveOp) return stmt->clone();
  
  // Find expected participants from trace based on the actual block execution
  uint32_t expectedParticipants = 4; // default
  uint32_t blockId = 0;
  
  // Find the wave operation in the trace to get its block ID
  for (const auto& waveOpRecord : trace.waveOperations) {
    if (waveOpRecord.instruction == static_cast<const void*>(waveOp) ||
        waveOpRecord.opType == waveOp->toString()) {
      expectedParticipants = waveOpRecord.arrivedParticipants.size();
      blockId = waveOpRecord.blockId;
      
      // Debug output
      std::cout << "[WaveParticipantTracking] Found wave op in block " << blockId 
                << " with " << expectedParticipants << " participants\n";
      
      // Also check the block's wave participation info for more accurate count
      if (trace.blocks.count(blockId)) {
        const auto& block = trace.blocks.at(blockId);
        // Assuming wave 0 for simplicity - in real code would need to track wave ID
        if (block.waveParticipation.count(0)) {
          const auto& waveInfo = block.waveParticipation.at(0);
          if (!waveInfo.participatingLanes.empty()) {
            expectedParticipants = waveInfo.participatingLanes.size();
            std::cout << "[WaveParticipantTracking] Using block participation info: " 
                      << expectedParticipants << " lanes\n";
          }
        }
      }
      break;
    }
  }
  
  // Create compound statement with original operation plus tracking
  std::vector<std::unique_ptr<interpreter::Statement>> statements;
  
  // 1. Original wave operation
  statements.push_back(assign->clone());
  
  // 2. Create tracking statements that use the global tid variable
  auto trackingStmts = createTrackingStatements(waveOp, assign->getName(), expectedParticipants);
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
  
  // Convert _isCorrect to uint
  auto isCorrectRef = std::make_unique<interpreter::VariableExpr>("_isCorrect");
  auto isCorrectAsUint = std::make_unique<interpreter::ConditionalExpr>(
      std::move(isCorrectRef), 
      std::make_unique<interpreter::LiteralExpr>(1),
      std::make_unique<interpreter::LiteralExpr>(0));
  
  // Get current value: _participant_check_sum[tid.x]
  auto tidX = std::make_unique<interpreter::VariableExpr>("tid.x");
  auto bufferAccess = std::make_unique<interpreter::ArrayAccessExpr>(
      "_participant_check_sum",
      std::move(tidX));
  
  // Add to current value
  auto addExpr = std::make_unique<interpreter::BinaryOpExpr>(
      std::move(bufferAccess), std::move(isCorrectAsUint), 
      interpreter::BinaryOpExpr::Add);
  
  // Store back: _participant_check_sum[tid.x] = ...
  auto tidX2 = std::make_unique<interpreter::VariableExpr>("tid.x");
  statements.push_back(std::make_unique<interpreter::ArrayAssignStmt>(
      "_participant_check_sum", std::move(tidX2), std::move(addExpr)));
  
  return statements;
}

bool WaveParticipantTrackingMutation::hasParticipantBuffer(
    const interpreter::Program& program) const {
  // Check if program already declares a participant tracking buffer
  // For now, return false as we're using variables instead of buffers
  return false;
}

bool WaveParticipantTrackingMutation::validateSemanticPreservation(
    const interpreter::Statement* original,
    const interpreter::Statement* mutated,
    const ExecutionTrace& trace) const {
  // Tracking operations don't change the wave operation result
  return true;
}

// ===== GPUInvariantCheckMutation Implementation =====

bool GPUInvariantCheckMutation::canApply(const interpreter::Statement* stmt,
                                        const ExecutionTrace& trace) const {
  // Can apply to wave operations
  if (auto* assign = dynamic_cast<const interpreter::AssignStmt*>(stmt)) {
    return dynamic_cast<const interpreter::WaveActiveOp*>(assign->getExpression()) != nullptr;
  }
  return false;
}

std::unique_ptr<interpreter::Statement> GPUInvariantCheckMutation::apply(
    const interpreter::Statement* stmt,
    const ExecutionTrace& trace) const {
  
  auto* assign = dynamic_cast<const interpreter::AssignStmt*>(stmt);
  if (!assign) return stmt->clone();
  
  auto* waveOp = dynamic_cast<const interpreter::WaveActiveOp*>(assign->getExpression());
  if (!waveOp) return stmt->clone();
  
  // Choose an invariant type based on the operation
  InvariantType invariantType = InvariantType::ParticipantCount; // default
  
  // For Sum with constant input, we can check uniform result
  auto opType = waveOp->getOpType();
  if (opType == interpreter::WaveActiveOp::Sum ||
      opType == interpreter::WaveActiveOp::Product) {
    // Check if input is a literal constant
    if (dynamic_cast<const interpreter::LiteralExpr*>(waveOp->getInput())) {
      invariantType = InvariantType::UniformResult;
    }
  }
  
  return createInvariantCheck(waveOp, assign->getName(), invariantType, trace);
}

std::unique_ptr<interpreter::Statement> GPUInvariantCheckMutation::createInvariantCheck(
    const interpreter::WaveActiveOp* waveOp,
    const std::string& resultVar,
    InvariantType type,
    const ExecutionTrace& trace) const {
  
  std::vector<std::unique_ptr<interpreter::Statement>> statements;
  
  // First execute the original operation
  auto originalAssign = std::make_unique<interpreter::AssignStmt>(
      resultVar, waveOp->clone());
  statements.push_back(std::move(originalAssign));
  
  switch (type) {
    case InvariantType::UniformResult: {
      // For uniform inputs, all lanes should get same result
      // Check: WaveActiveAllTrue(result == WaveReadLaneFirst(result))
      
      auto resultRef1 = std::make_unique<interpreter::VariableExpr>(resultVar);
      auto resultRef2 = std::make_unique<interpreter::VariableExpr>(resultVar);
      auto zeroLane = std::make_unique<interpreter::LiteralExpr>(interpreter::Value(0));
      auto firstLaneResult = std::make_unique<interpreter::WaveReadLaneAt>(
          std::move(resultRef2), std::move(zeroLane));
      
      auto compare = std::make_unique<interpreter::BinaryOpExpr>(
          std::move(resultRef1), std::move(firstLaneResult), 
          interpreter::BinaryOpExpr::Eq);
      
      auto allTrue = std::make_unique<interpreter::WaveActiveOp>(
          std::move(compare), interpreter::WaveActiveOp::AllTrue);
      
      statements.push_back(std::make_unique<interpreter::VarDeclStmt>(
          "_invariant_check", std::move(allTrue)));
      
      // Store check result to buffer
      // ParticipantBuffer[tid] = _invariant_check ? 1 : 0
      break;
    }
    
    case InvariantType::ReductionIdentity: {
      // Check that operations with identity elements work correctly
      // e.g., WaveActiveSum(0) == 0, WaveActiveProduct(1) == 1
      
      auto resultRef = std::make_unique<interpreter::VariableExpr>(resultVar);
      auto identity = std::make_unique<interpreter::LiteralExpr>(0); // for Sum
      
      if (waveOp->getOpType() == interpreter::WaveActiveOp::Product) {
        identity = std::make_unique<interpreter::LiteralExpr>(1);
      }
      
      auto compare = std::make_unique<interpreter::BinaryOpExpr>(
          std::move(resultRef), std::move(identity),
          interpreter::BinaryOpExpr::Eq);
      
      statements.push_back(std::make_unique<interpreter::VarDeclStmt>(
          "_identity_check", std::move(compare)));
      break;
    }
    
    default:
      // Already handled by WaveParticipantTrackingMutation
      break;
  }
  
  // Wrap in if(true)
  auto trueCond = std::make_unique<interpreter::LiteralExpr>(interpreter::Value(true));
  return std::make_unique<interpreter::IfStmt>(
      std::move(trueCond), std::move(statements),
      std::vector<std::unique_ptr<interpreter::Statement>>{});
}

bool GPUInvariantCheckMutation::validateSemanticPreservation(
    const interpreter::Statement* original,
    const interpreter::Statement* mutated,
    const ExecutionTrace& trace) const {
  // Invariant checks don't change the operation result
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
  // TODO: This check is too strict and causes false positives with semantics-preserving mutations
  // that add extra blocks (e.g., if(true) wrappers). Commenting out for now.
  /*
  if (!verifyControlFlowEquivalence(golden, mutant, result)) {
    result.isEquivalent = false;
    if (result.divergenceReason.empty()) {
      result.divergenceReason = "Control flow differs";
    }
  }
  */
  
  return result;
}

bool SemanticValidator::compareFinalStates(
    const ExecutionTrace::FinalState& golden,
    const ExecutionTrace::FinalState& mutant,
    ValidationResult& result) {
  
  // Debug: Print wave structure
  if (golden.laneVariables.empty()) {
    std::cout << "DEBUG: Golden trace has no lane variables recorded\n";
  }
  if (mutant.laneVariables.empty()) {
    std::cout << "DEBUG: Mutant trace has no lane variables recorded\n";
  }
  
  // Compare per-lane variables
  // Note: We allow the mutant to have extra waves/lanes (from structural changes)
  // We only check that waves/lanes in the original exist in the mutant
  for (const auto& [waveId, waveVars] : golden.laneVariables) {
    auto mutantWaveIt = mutant.laneVariables.find(waveId);
    if (mutantWaveIt == mutant.laneVariables.end()) {
      // This might happen if waves are numbered differently
      // For now, assume single wave (wave 0) and continue
      if (waveId == 0 && !mutant.laneVariables.empty()) {
        mutantWaveIt = mutant.laneVariables.begin();
      } else {
        result.differences.push_back("Missing wave " + std::to_string(waveId) + " in mutant");
        return false;
      }
    }
    
    for (const auto& [laneId, vars] : waveVars) {
      auto mutantLaneIt = mutantWaveIt->second.find(laneId);
      if (mutantLaneIt == mutantWaveIt->second.end()) {
        result.differences.push_back("Missing lane " + std::to_string(laneId) + 
                                   " in wave " + std::to_string(waveId));
        return false;
      }
      
      // Compare variable values - only check variables that exist in the original
      // (mutations may add extra tracking variables)
      for (const auto& [varName, value] : vars) {
        // Skip variables that start with underscore (internal tracking variables)
        if (varName.empty() || varName[0] == '_') {
          continue;
        }
        
        auto mutantVarIt = mutantLaneIt->second.find(varName);
        if (mutantVarIt == mutantLaneIt->second.end()) {
          // Special case: 'tid' may be added by mutations
          if (varName == "tid") {
            continue;
          }
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
  
  // Helper lambda to extract operation type from full string
  auto extractOpType = [](const std::string& fullOp) -> std::string {
    // Extract just the operation name before the parenthesis
    // e.g., "WaveActiveSum(x)" -> "WaveActiveSum"
    size_t parenPos = fullOp.find('(');
    if (parenPos != std::string::npos) {
      return fullOp.substr(0, parenPos);
    }
    return fullOp;
  };
  
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
    
    // Compare just the operation type, not the full expression
    std::string goldenOpType = extractOpType(goldenOp.opType);
    std::string mutantOpType = extractOpType(mutantOp.opType);
    
    if (goldenOpType != mutantOpType) {
      result.differences.push_back("Wave op type mismatch at index " + std::to_string(i) + 
                                 ": " + goldenOp.opType + " vs " + mutantOp.opType);
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
  // Initialize mutation strategies - only semantics-preserving mutations
  mutationStrategies.push_back(std::make_unique<LanePermutationMutation>());
  mutationStrategies.push_back(std::make_unique<DataTransformMutation>());
  mutationStrategies.push_back(std::make_unique<RedundantComputeMutation>());
  mutationStrategies.push_back(std::make_unique<WaveParticipantTrackingMutation>());
  // TODO: Add more semantics-preserving mutations:
  // - AlgebraicIdentityMutation (more complex algebraic identities)
  // - MemoryAccessReorderMutation (reorder independent memory accesses)
  // - RegisterSpillMutation (force values through memory)
  
  validator = std::make_unique<SemanticValidator>();
  bugReporter = std::make_unique<BugReporter>();
}

void TraceGuidedFuzzer::fuzzProgram(const interpreter::Program& seedProgram, 
                                  const FuzzingConfig& config) {
  
  std::cout << "Starting trace-guided fuzzing...\n";
  std::cout << "Threadgroup size: " << config.threadgroupSize << "\n";
  std::cout << "Wave size: " << config.waveSize << "\n";
  
  // Debug: Print the seed program being fuzzed
  std::cout << "\n=== Seed Program ===\n";
  std::cout << serializeProgramToString(seedProgram);
  std::cout << "\n";
  
  // Create trace capture interpreter
  TraceCaptureInterpreter captureInterpreter;
  
  // Execute seed and capture golden trace
  std::cout << "Capturing golden trace...\n";
  
  interpreter::ThreadOrdering ordering;
  // Use default source order
  
  auto goldenResult = captureInterpreter.executeAndCaptureTrace(
    seedProgram, ordering, config.waveSize);
  
  // Check if execution succeeded
  if (!goldenResult.isValid()) {
    std::cerr << "Golden execution failed: " << goldenResult.errorMessage << "\n";
    return;
  }
  
  const ExecutionTrace& goldenTrace = *captureInterpreter.getTrace();
  
  std::cout << "Golden trace captured:\n";
  std::cout << "  - Blocks executed: " << goldenTrace.blocks.size() << "\n";
  std::cout << "  - Wave operations: " << goldenTrace.waveOperations.size() << "\n";
  std::cout << "  - Waves in final state: " << goldenTrace.finalState.laneVariables.size() << "\n";
  for (const auto& [waveId, waveVars] : goldenTrace.finalState.laneVariables) {
    std::cout << "    Wave " << waveId << " has " << waveVars.size() << " lanes\n";
  }
  std::cout << "  - Control flow decisions: " << goldenTrace.controlFlowHistory.size() << "\n";
  
  // Generate and test mutants
  size_t mutantsTested = 0;
  size_t bugsFound = 0;
  
  for (auto& strategy : mutationStrategies) {
    std::cout << "\nTrying mutation strategy: " << strategy->getName() << "\n";
    
    // Add specific debug for WaveParticipantTrackingMutation
    if (strategy->getName() == "WaveParticipantTracking") {
      std::cout << "[DEBUG] WaveParticipantTrackingMutation selected for generateMutants\n";
    }
    
    // Generate mutants with this strategy
    auto mutants = generateMutants(seedProgram, strategy.get(), goldenTrace);
    
    for (const auto& mutant : mutants) {
      if (mutantsTested >= config.maxMutants) {
        break;
      }
      
      mutantsTested++;
      
      // Always print original and mutated programs
      std::cout << "\n=== Testing Mutant " << mutantsTested << " (Strategy: " << strategy->getName() << ") ===\n";
      
      std::cout << "\n--- Original Program ---\n";
      std::cout << serializeProgramToString(seedProgram);
      
      std::cout << "\n--- Mutant Program ---\n";
      std::cout << serializeProgramToString(mutant);
      
      try {
        // Execute mutant
        TraceCaptureInterpreter mutantInterpreter;
        auto mutantResult = mutantInterpreter.executeAndCaptureTrace(
          mutant, ordering, config.waveSize);
        
        // Check if execution succeeded
        if (!mutantResult.isValid()) {
          std::cout << "Mutant execution failed: " << mutantResult.errorMessage << "\n";
          continue;
        }
        
        const ExecutionTrace& mutantTrace = *mutantInterpreter.getTrace();
        
        std::cout << "Mutant trace captured:\n";
        std::cout << "  - Waves in final state: " << mutantTrace.finalState.laneVariables.size() << "\n";
        for (const auto& [waveId, waveVars] : mutantTrace.finalState.laneVariables) {
          std::cout << "    Wave " << waveId << " has " << waveVars.size() << " lanes\n";
        }
        
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
  
  interpreter.execute(program, ordering, 32); // Default wave size
  
  const ExecutionTrace& trace = static_cast<const TraceCaptureInterpreter&>(interpreter).getTrace();
  
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
  
  // Special handling for WaveParticipantTrackingMutation
  if (dynamic_cast<WaveParticipantTrackingMutation*>(strategy)) {
    std::cout << "[DEBUG] Inside WaveParticipantTrackingMutation special handling in generateMutants\n";
    
    // Check if program contains wave operations (including in nested statements)
    bool hasWaveOps = false;
    std::function<bool(const interpreter::Statement*)> hasWaveOp;
    hasWaveOp = [&hasWaveOp](const interpreter::Statement* stmt) -> bool {
      // Check AssignStmt
      if (auto* assign = dynamic_cast<const interpreter::AssignStmt*>(stmt)) {
        if (dynamic_cast<const interpreter::WaveActiveOp*>(assign->getExpression())) {
          return true;
        }
      } 
      // Check VarDeclStmt
      else if (auto* varDecl = dynamic_cast<const interpreter::VarDeclStmt*>(stmt)) {
        if (varDecl->getInit() && 
            dynamic_cast<const interpreter::WaveActiveOp*>(varDecl->getInit())) {
          return true;
        }
      }
      // Check IfStmt
      else if (auto* ifStmt = dynamic_cast<const interpreter::IfStmt*>(stmt)) {
        for (const auto& s : ifStmt->getThenBlock()) {
          if (hasWaveOp(s.get())) return true;
        }
        for (const auto& s : ifStmt->getElseBlock()) {
          if (hasWaveOp(s.get())) return true;
        }
      } else if (auto* forStmt = dynamic_cast<const interpreter::ForStmt*>(stmt)) {
        for (const auto& s : forStmt->getBody()) {
          if (hasWaveOp(s.get())) return true;
        }
      }
      return false;
    };
    
    for (const auto& stmt : program.statements) {
      if (hasWaveOp(stmt.get())) {
        hasWaveOps = true;
        break;
      }
    }
    
    std::cout << "[DEBUG] hasWaveOps = " << hasWaveOps << "\n";
    
    if (hasWaveOps) {
      std::cout << "[DEBUG] Creating mutant for WaveParticipantTrackingMutation\n";
      
      // Create a single mutant
      interpreter::Program mutant;
      mutant.numThreadsX = program.numThreadsX;
      mutant.numThreadsY = program.numThreadsY;
      mutant.numThreadsZ = program.numThreadsZ;
      mutant.entryInputs = program.entryInputs;  // Copy entry function parameters
      mutant.globalBuffers = program.globalBuffers;  // Copy existing global buffers
      
      // Add the participant tracking buffer if it doesn't exist
      bool hasParticipantBuffer = false;
      for (const auto& buffer : mutant.globalBuffers) {
        if (buffer.name == "_participant_check_sum") {
          hasParticipantBuffer = true;
          break;
        }
      }
      
      if (!hasParticipantBuffer) {
        interpreter::GlobalBufferDecl participantBuffer;
        participantBuffer.name = "_participant_check_sum";
        participantBuffer.bufferType = "RWBuffer";
        participantBuffer.elementType = interpreter::HLSLType::Uint;
        participantBuffer.size = program.getTotalThreads();  // Size based on threadgroup
        participantBuffer.registerIndex = 1;  // Use u1 to avoid conflicts
        participantBuffer.isReadWrite = true;
        mutant.globalBuffers.push_back(participantBuffer);
        
        std::cout << "[DEBUG] Added _participant_check_sum buffer with size " 
                  << participantBuffer.size << " to mutant\n";
      }
      
      // Add buffer initialization at the beginning
      // Initialize current thread's entry: _participant_check_sum[tid.x] = 0
      auto tidX = std::make_unique<interpreter::VariableExpr>("tid.x");
      auto zero = std::make_unique<interpreter::LiteralExpr>(0);
      mutant.statements.push_back(std::make_unique<interpreter::ArrayAssignStmt>(
          "_participant_check_sum", std::move(tidX), std::move(zero)));
      
      // Apply mutation to all wave operations and clone other statements
      for (const auto& stmt : program.statements) {
        bool mutationApplied = false;
        auto mutatedStmt = applyMutationToStatement(
            stmt.get(), strategy, trace, mutationApplied);
        
        if (mutationApplied) {
          mutant.statements.push_back(std::move(mutatedStmt));
        } else {
          mutant.statements.push_back(stmt->clone());
        }
      }
      
      std::cout << "[DEBUG] Final mutant has " << mutant.statements.size() << " statements\n";
      for (size_t i = 0; i < mutant.statements.size(); ++i) {
        std::cout << "  Statement " << i << ": " << mutant.statements[i]->toString() << "\n";
      }
      
      mutants.push_back(std::move(mutant));
    }
    std::cout << "[DEBUG] Returning " << mutants.size() << " mutants from WaveParticipantTrackingMutation\n";
    return mutants;
  }
  
  // Default behavior for other mutation strategies
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
      mutant.entryInputs = program.entryInputs;  // Copy entry function parameters
      
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
  std::string seedDirStr; // Store the directory path to avoid use-after-free
  
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
        seedDirStr = dir;
        seedDir = seedDirStr.c_str();
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
      seedProgram.entryInputs = conversionResult.program.entryInputs;  // Copy entry function parameters
      
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
    program.entryInputs = seedProgram.entryInputs;  // Copy entry function parameters
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
  config.enableLogging = true; // Enable logging to see programs
  
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