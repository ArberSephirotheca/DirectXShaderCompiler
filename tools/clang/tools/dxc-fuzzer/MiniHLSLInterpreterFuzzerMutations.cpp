#include "MiniHLSLInterpreterFuzzer.h"
#include "MiniHLSLInterpreter.h"

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

} // namespace fuzzer
} // namespace minihlsl