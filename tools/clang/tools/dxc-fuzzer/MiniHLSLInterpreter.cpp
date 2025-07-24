#include "MiniHLSLInterpreter.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <thread>

// Clang AST includes for conversion
#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/StmtCXX.h"
#include "clang/Basic/OperatorKinds.h"

namespace minihlsl {
namespace interpreter {

// Value implementation
Value Value::operator+(const Value &other) const {
  if (std::holds_alternative<int32_t>(data) &&
      std::holds_alternative<int32_t>(other.data)) {
    return Value(std::get<int32_t>(data) + std::get<int32_t>(other.data));
  }
  return Value(asFloat() + other.asFloat());
}

Value Value::operator-(const Value &other) const {
  if (std::holds_alternative<int32_t>(data) &&
      std::holds_alternative<int32_t>(other.data)) {
    return Value(std::get<int32_t>(data) - std::get<int32_t>(other.data));
  }
  return Value(asFloat() - other.asFloat());
}

Value Value::operator*(const Value &other) const {
  if (std::holds_alternative<int32_t>(data) &&
      std::holds_alternative<int32_t>(other.data)) {
    return Value(std::get<int32_t>(data) * std::get<int32_t>(other.data));
  }
  return Value(asFloat() * other.asFloat());
}

Value Value::operator/(const Value &other) const {
  if (std::holds_alternative<int32_t>(data) &&
      std::holds_alternative<int32_t>(other.data)) {
    int32_t divisor = std::get<int32_t>(other.data);
    if (divisor == 0)
      throw std::runtime_error("Division by zero");
    return Value(std::get<int32_t>(data) / divisor);
  }
  float divisor = other.asFloat();
  if (divisor == 0.0f)
    throw std::runtime_error("Division by zero");
  return Value(asFloat() / divisor);
}

Value Value::operator%(const Value &other) const {
  int32_t a = asInt();
  int32_t b = other.asInt();
  if (b == 0)
    throw std::runtime_error("Modulo by zero");
  return Value(a % b);
}

bool Value::operator==(const Value &other) const {
  if (data.index() != other.data.index()) {
    return asFloat() == other.asFloat();
  }
  return data == other.data;
}

bool Value::operator!=(const Value &other) const { return !(*this == other); }

bool Value::operator<(const Value &other) const {
  if (std::holds_alternative<int32_t>(data) &&
      std::holds_alternative<int32_t>(other.data)) {
    return std::get<int32_t>(data) < std::get<int32_t>(other.data);
  }
  return asFloat() < other.asFloat();
}

bool Value::operator<=(const Value &other) const {
  return (*this < other) || (*this == other);
}

bool Value::operator>(const Value &other) const { return !(*this <= other); }

bool Value::operator>=(const Value &other) const { return !(*this < other); }

Value Value::operator&&(const Value &other) const {
  return Value(asBool() && other.asBool());
}

Value Value::operator||(const Value &other) const {
  return Value(asBool() || other.asBool());
}

Value Value::operator!() const { return Value(!asBool()); }

int32_t Value::asInt() const {
  if (std::holds_alternative<int32_t>(data)) {
    return std::get<int32_t>(data);
  } else if (std::holds_alternative<float>(data)) {
    return static_cast<int32_t>(std::get<float>(data));
  } else {
    return std::get<bool>(data) ? 1 : 0;
  }
}

float Value::asFloat() const {
  if (std::holds_alternative<float>(data)) {
    return std::get<float>(data);
  } else if (std::holds_alternative<int32_t>(data)) {
    return static_cast<float>(std::get<int32_t>(data));
  } else {
    return std::get<bool>(data) ? 1.0f : 0.0f;
  }
}

bool Value::asBool() const {
  if (std::holds_alternative<bool>(data)) {
    return std::get<bool>(data);
  } else if (std::holds_alternative<int32_t>(data)) {
    return std::get<int32_t>(data) != 0;
  } else {
    return std::get<float>(data) != 0.0f;
  }
}

std::string Value::toString() const {
  std::stringstream ss;
  if (std::holds_alternative<int32_t>(data)) {
    ss << std::get<int32_t>(data);
  } else if (std::holds_alternative<float>(data)) {
    ss << std::fixed << std::setprecision(6) << std::get<float>(data);
  } else {
    ss << (std::get<bool>(data) ? "true" : "false");
  }
  return ss.str();
}

// WaveContext implementation
uint64_t WaveContext::getActiveMask() const {
  uint64_t mask = 0;
  for (size_t i = 0; i < lanes.size() && i < 64; ++i) {
    if (lanes[i]->isActive) {
      mask |= (1ULL << i);
    }
  }
  return mask;
}

std::vector<LaneId> WaveContext::getActiveLanes() const {
  std::vector<LaneId> active;
  for (size_t i = 0; i < lanes.size(); ++i) {
    if (lanes[i]->isActive) {
      active.push_back(i);
    }
  }
  return active;
}

bool WaveContext::allLanesActive() const {
  return std::all_of(lanes.begin(), lanes.end(),
                     [](const auto &lane) { return lane->isActive; });
}

uint32_t WaveContext::countActiveLanes() const {
  return std::count_if(lanes.begin(), lanes.end(),
                       [](const auto &lane) { return lane->isActive; });
}

std::vector<LaneId> WaveContext::getCurrentlyActiveLanes() const {
  std::vector<LaneId> active;
  for (size_t i = 0; i < lanes.size(); ++i) {
    if (lanes[i]->isActive) {
      active.push_back(i);
    }
  }
  return active;
}

uint32_t WaveContext::countCurrentlyActiveLanes() const {
  return std::count_if(lanes.begin(), lanes.end(),
                       [](const auto &lane) { return lane->isActive; });
}

// SharedMemory implementation
Value SharedMemory::read(MemoryAddress addr, ThreadId tid) {
  std::lock_guard<std::mutex> lock(mutex_);
  accessHistory_[addr].insert(tid);
  auto it = memory_.find(addr);
  return it != memory_.end() ? it->second : Value(0);
}

void SharedMemory::write(MemoryAddress addr, Value value, ThreadId tid) {
  std::lock_guard<std::mutex> lock(mutex_);
  memory_[addr] = value;
  accessHistory_[addr].insert(tid);
}

Value SharedMemory::atomicAdd(MemoryAddress addr, Value value, ThreadId tid) {
  std::lock_guard<std::mutex> lock(mutex_);
  Value oldValue = memory_[addr];
  memory_[addr] = oldValue + value;
  accessHistory_[addr].insert(tid);
  return oldValue;
}

bool SharedMemory::hasConflictingAccess(MemoryAddress addr, ThreadId tid1,
                                        ThreadId tid2) const {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = accessHistory_.find(addr);
  if (it == accessHistory_.end())
    return false;
  return it->second.count(tid1) > 0 && it->second.count(tid2) > 0;
}

std::map<MemoryAddress, Value> SharedMemory::getSnapshot() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return memory_;
}

void SharedMemory::clear() {
  std::lock_guard<std::mutex> lock(mutex_);
  memory_.clear();
  accessHistory_.clear();
}

// BarrierState implementation
void BarrierState::reset(uint32_t totalThreads) {
  std::lock_guard<std::mutex> lock(mutex);
  waitingThreads.clear();
  arrivedThreads.clear();
}

bool BarrierState::tryArrive(ThreadId tid) {
  std::lock_guard<std::mutex> lock(mutex);
  arrivedThreads.insert(tid);
  return true;
}

void BarrierState::waitForAll() {
  // In our interpreter model, barriers are handled at the orchestration level
  // This is a simplified implementation
  std::this_thread::sleep_for(std::chrono::milliseconds(1));
}

// ThreadgroupContext implementation
ThreadgroupContext::ThreadgroupContext(uint32_t tgSize, uint32_t wSize)
    : threadgroupSize(tgSize), waveSize(wSize) {
  waveCount = (threadgroupSize + waveSize - 1) / waveSize;

    
      BlockIdentity initialIdentity =
          createBlockIdentity(nullptr, true, 0, {});

      // Create initial block with no unknown lanes (all lanes are guaranteed to
      // start here)
      uint32_t initialBlockId =
          findOrCreateBlockForPath(initialIdentity, {});

  // Initialize waves
  for (uint32_t w = 0; w < waveCount; ++w) {
    auto wave = std::make_unique<WaveContext>();
    wave->waveId = w;
    wave->waveSize = waveSize;

    // Initialize lanes in wave
    uint32_t lanesInWave = std::min(waveSize, threadgroupSize - w * waveSize);
    std::set<LaneId> allLanes;
    for (uint32_t l = 0; l < lanesInWave; ++l) {
      auto lane = std::make_unique<LaneContext>();
      lane->laneId = l;
      wave->lanes.push_back(std::move(lane));
      allLanes.insert(l);
    }
    waves.push_back(std::move(wave));
    // Create initial execution block with all lanes as known participants
    if (!allLanes.empty()) {
      // Create identity for initial block (no condition, all lanes start here

      // Mark all lanes as arrived at the initial block
      for (LaneId laneId : allLanes) {
        // TODO: implement this function for WaveContext
        markLaneArrived(w, laneId, initialBlockId);
      }
    }
  }



  sharedMemory = std::make_shared<SharedMemory>();
}

ThreadId ThreadgroupContext::getGlobalThreadId(WaveId wid, LaneId lid) const {
  return wid * waveSize + lid;
}

std::pair<WaveId, LaneId>
ThreadgroupContext::getWaveAndLane(ThreadId tid) const {
  WaveId wid = tid / waveSize;
  LaneId lid = tid % waveSize;
  return {wid, lid};
}

std::vector<ThreadId> ThreadgroupContext::getReadyThreads() const {
  std::vector<ThreadId> ready;
  for (ThreadId tid = 0; tid < threadgroupSize; ++tid) {
    auto [waveId, laneId] = getWaveAndLane(tid);
    if (waveId < waves.size() && laneId < waves[waveId]->lanes.size()) {
      const auto &lane = waves[waveId]->lanes[laneId];
      if (lane->state == ThreadState::Ready) {
        ready.push_back(tid);
      }
    }
  }
  return ready;
}

std::vector<ThreadId> ThreadgroupContext::getWaitingThreads() const {
  std::vector<ThreadId> waiting;
  for (ThreadId tid = 0; tid < threadgroupSize; ++tid) {
    auto [waveId, laneId] = getWaveAndLane(tid);
    if (waveId < waves.size() && laneId < waves[waveId]->lanes.size()) {
      const auto &lane = waves[waveId]->lanes[laneId];
      if (lane->state == ThreadState::WaitingAtBarrier ||
          lane->state == ThreadState::WaitingForWave) {
        waiting.push_back(tid);
      }
    }
  }
  return waiting;
}

bool ThreadgroupContext::canExecuteWaveOp(
    WaveId waveId, const std::set<LaneId> &activeLanes) const {
  // Phase 1: Simple implementation - always allow wave ops
  // TODO: Add proper active lane checking in Phase 2
  return true;
}

bool ThreadgroupContext::canReleaseBarrier(uint32_t barrierId) const {
  // Phase 1: Simple implementation - always release barriers
  // TODO: Add proper barrier participation analysis in Phase 2
  return true;
}

// Global dynamic block creation methods (delegates to appropriate wave)
// std::pair<uint32_t, uint32_t> ThreadgroupContext::createIfBlocks(
//     const void *ifStmt, uint32_t parentBlockId,
//     const std::vector<MergeStackEntry> &mergeStack, bool hasElse,
//     WaveId waveId) {
//   if (waveId < waves.size()) {
//     return waves[waveId]->createIfBlocks(ifStmt, parentBlockId, mergeStack,
//                                          hasElse);
//   }
//   return {0, 0};
// }

// uint32_t ThreadgroupContext::createLoopIterationBlock(
//     const void *loopStmt, uint32_t parentBlockId,
//     const std::vector<MergeStackEntry> &mergeStack, WaveId waveId) {
//   if (waveId < waves.size()) {
//     return waves[waveId]->createLoopIterationBlock(loopStmt, parentBlockId,
//                                                    mergeStack);
//   }
//   return 0;
// }

// std::vector<uint32_t> ThreadgroupContext::createSwitchCaseBlocks(
//     const void *switchStmt, uint32_t parentBlockId,
//     const std::vector<MergeStackEntry> &mergeStack,
//     const std::vector<int> &caseValues, bool hasDefault, WaveId waveId) {
//   if (waveId < waves.size()) {
//     return waves[waveId]->createSwitchCaseBlocks(
//         switchStmt, parentBlockId, mergeStack, caseValues, hasDefault);
//   }
//   return {};
// }

// void ThreadgroupContext::moveThreadFromUnknownToParticipating(uint32_t blockId,
//                                                               LaneId laneId,
//                                                               WaveId waveId) {
//   if (waveId < waves.size()) {
//     waves[waveId]->moveThreadFromUnknownToParticipating(blockId, laneId);
//   }
// }

// void ThreadgroupContext::removeThreadFromUnknown(uint32_t blockId,
//                                                  LaneId laneId, WaveId waveId) {
//   if (waveId < waves.size()) {
//     waves[waveId]->removeThreadFromUnknown(blockId, laneId);
//   }
// }

// void ThreadgroupContext::removeThreadFromNestedBlocks(uint32_t parentBlockId,
//                                                       LaneId laneId,
//                                                       WaveId waveId) {
//   if (waveId < waves.size()) {
//     waves[waveId]->removeThreadFromNestedBlocks(parentBlockId, laneId);
//   }
// }

// ThreadOrdering implementation
ThreadOrdering ThreadOrdering::sequential(uint32_t threadCount) {
  ThreadOrdering ordering;
  ordering.description = "Sequential";
  for (uint32_t i = 0; i < threadCount; ++i) {
    ordering.executionOrder.push_back(i);
  }
  return ordering;
}

ThreadOrdering ThreadOrdering::reverseSequential(uint32_t threadCount) {
  ThreadOrdering ordering;
  ordering.description = "Reverse Sequential";
  for (int i = threadCount - 1; i >= 0; --i) {
    ordering.executionOrder.push_back(i);
  }
  return ordering;
}

ThreadOrdering ThreadOrdering::random(uint32_t threadCount, uint32_t seed) {
  ThreadOrdering ordering;
  ordering.description = "Random (seed=" + std::to_string(seed) + ")";

  // Create sequential order then shuffle
  for (uint32_t i = 0; i < threadCount; ++i) {
    ordering.executionOrder.push_back(i);
  }

  std::mt19937 rng(seed);
  std::shuffle(ordering.executionOrder.begin(), ordering.executionOrder.end(),
               rng);

  return ordering;
}

ThreadOrdering ThreadOrdering::evenOddInterleaved(uint32_t threadCount) {
  ThreadOrdering ordering;
  ordering.description = "Even-Odd Interleaved";

  // First all even threads, then all odd threads
  for (uint32_t i = 0; i < threadCount; i += 2) {
    ordering.executionOrder.push_back(i);
  }
  for (uint32_t i = 1; i < threadCount; i += 2) {
    ordering.executionOrder.push_back(i);
  }

  return ordering;
}

ThreadOrdering ThreadOrdering::waveInterleaved(uint32_t threadCount,
                                               uint32_t waveSize) {
  ThreadOrdering ordering;
  ordering.description = "Wave Interleaved";

  uint32_t waveCount = (threadCount + waveSize - 1) / waveSize;

  // Execute threads wave by wave in reverse order
  for (int w = waveCount - 1; w >= 0; --w) {
    uint32_t startThread = w * waveSize;
    uint32_t endThread = std::min(startThread + waveSize, threadCount);

    for (uint32_t t = startThread; t < endThread; ++t) {
      ordering.executionOrder.push_back(t);
    }
  }

  return ordering;
}

// Expression implementations
Value VariableExpr::evaluate(LaneContext &lane, WaveContext &,
                             ThreadgroupContext &) const {
  auto it = lane.variables.find(name_);
  if (it == lane.variables.end()) {
    throw std::runtime_error("Undefined variable: " + name_);
  }
  return it->second;
}

Value LaneIndexExpr::evaluate(LaneContext &lane, WaveContext &,
                              ThreadgroupContext &) const {
  return Value(static_cast<int32_t>(lane.laneId));
}

Value WaveIndexExpr::evaluate(LaneContext &, WaveContext &wave,
                              ThreadgroupContext &) const {
  return Value(static_cast<int32_t>(wave.waveId));
}

Value ThreadIndexExpr::evaluate(LaneContext &lane, WaveContext &wave,
                                ThreadgroupContext &tg) const {
  ThreadId tid = tg.getGlobalThreadId(wave.waveId, lane.laneId);
  return Value(static_cast<int32_t>(tid));
}

BinaryOpExpr::BinaryOpExpr(std::unique_ptr<Expression> left,
                           std::unique_ptr<Expression> right, OpType op)
    : left_(std::move(left)), right_(std::move(right)), op_(op) {}

Value BinaryOpExpr::evaluate(LaneContext &lane, WaveContext &wave,
                             ThreadgroupContext &tg) const {
  Value leftVal = left_->evaluate(lane, wave, tg);
  Value rightVal = right_->evaluate(lane, wave, tg);

  switch (op_) {
  case Add:
    return leftVal + rightVal;
  case Sub:
    return leftVal - rightVal;
  case Mul:
    return leftVal * rightVal;
  case Div:
    return leftVal / rightVal;
  case Mod:
    return leftVal % rightVal;
  case Eq:
    return Value(leftVal == rightVal);
  case Ne:
    return Value(leftVal != rightVal);
  case Lt:
    return Value(leftVal < rightVal);
  case Le:
    return Value(leftVal <= rightVal);
  case Gt:
    return Value(leftVal > rightVal);
  case Ge:
    return Value(leftVal >= rightVal);
  case And:
    return leftVal && rightVal;
  case Or:
    return leftVal || rightVal;
  }
  throw std::runtime_error("Unknown binary operator");
}

bool BinaryOpExpr::isDeterministic() const {
  return left_->isDeterministic() && right_->isDeterministic();
}

std::string BinaryOpExpr::toString() const {
  static const char *opStrings[] = {
      "+", "-", "*", "/", "%", "==", "!=", "<", "<=", ">", ">=", "&&", "||"};
  return "(" + left_->toString() + " " + opStrings[op_] + " " +
         right_->toString() + ")";
}

UnaryOpExpr::UnaryOpExpr(std::unique_ptr<Expression> expr, OpType op)
    : expr_(std::move(expr)), op_(op) {}

Value UnaryOpExpr::evaluate(LaneContext &lane, WaveContext &wave,
                            ThreadgroupContext &tg) const {
  Value val = expr_->evaluate(lane, wave, tg);

  switch (op_) {
  case Neg:
    return Value(-val.asFloat());
  case Not:
    return !val;
  }
  throw std::runtime_error("Unknown unary operator");
}

bool UnaryOpExpr::isDeterministic() const { return expr_->isDeterministic(); }

std::string UnaryOpExpr::toString() const {
  static const char *opStrings[] = {"-", "!"};
  return opStrings[op_] + expr_->toString();
}

// Wave operation implementations
WaveActiveOp::WaveActiveOp(std::unique_ptr<Expression> expr, OpType op)
    : expr_(std::move(expr)), op_(op) {}

Value WaveActiveOp::evaluate(LaneContext &lane, WaveContext &wave,
                             ThreadgroupContext &tg) const {
  // Wave operations require all active lanes to participate
  if (!lane.isActive) {
    throw std::runtime_error("Inactive lane executing wave operation");
  }

  // Use instruction-level synchronization: both participants known AND all
  // arrived
  const void *instruction =
      static_cast<const void *>(this); // Use 'this' as instruction pointer

  if (!wave.canExecuteWaveInstruction(lane.laneId, instruction)) {
    // Mark this lane as arrived at this specific instruction
    wave.markLaneArrivedAtInstruction(lane.laneId, instruction, "WaveActiveOp");

    // In a full cooperative scheduler, this would suspend the lane and schedule
    // others For now, throw an exception to indicate we need to wait
    throw std::runtime_error("Lane must wait: participants unknown or not all "
                             "arrived at instruction");
  }

  // All participants known AND all have arrived at this instruction - safe to
  // execute
  auto participants = wave.getInstructionParticipants(instruction);
  std::vector<Value> values;
  for (LaneId laneId : participants) {
    values.push_back(expr_->evaluate(*wave.lanes[laneId], wave, tg));
  }

  // Release the sync point after execution
  wave.releaseSyncPoint(instruction);

  if (values.empty()) {
    throw std::runtime_error("No active lanes for wave operation");
  }

  switch (op_) {
  case Sum: {
    Value sum = values[0];
    for (size_t i = 1; i < values.size(); ++i) {
      sum = sum + values[i];
    }
    return sum;
  }
  case Product: {
    Value product = values[0];
    for (size_t i = 1; i < values.size(); ++i) {
      product = product * values[i];
    }
    return product;
  }
  case Min: {
    Value minVal = values[0];
    for (size_t i = 1; i < values.size(); ++i) {
      if (values[i] < minVal)
        minVal = values[i];
    }
    return minVal;
  }
  case Max: {
    Value maxVal = values[0];
    for (size_t i = 1; i < values.size(); ++i) {
      if (values[i] > maxVal)
        maxVal = values[i];
    }
    return maxVal;
  }
  case And: {
    Value result = values[0];
    for (size_t i = 1; i < values.size(); ++i) {
      result = result && values[i];
    }
    return result;
  }
  case Or: {
    Value result = values[0];
    for (size_t i = 1; i < values.size(); ++i) {
      result = result || values[i];
    }
    return result;
  }
  case Xor: {
    int32_t result = values[0].asInt();
    for (size_t i = 1; i < values.size(); ++i) {
      result ^= values[i].asInt();
    }
    return Value(result);
  }
  case CountBits: {
    int32_t count = 0;
    for (const auto &val : values) {
      if (val.asBool())
        count++;
    }
    return Value(count);
  }
  }
  throw std::runtime_error("Unknown wave operation");
}

std::string WaveActiveOp::toString() const {
  static const char *opNames[] = {"WaveActiveSum", "WaveActiveProduct",
                                  "WaveActiveMin", "WaveActiveMax",
                                  "WaveActiveAnd", "WaveActiveOr",
                                  "WaveActiveXor", "WaveActiveCountBits"};
  return std::string(opNames[op_]) + "(" + expr_->toString() + ")";
}

Value WaveGetLaneCountExpr::evaluate(LaneContext &, WaveContext &wave,
                                     ThreadgroupContext &) const {
  return Value(static_cast<int32_t>(wave.countActiveLanes()));
}

// Statement implementations
VarDeclStmt::VarDeclStmt(const std::string &name,
                         std::unique_ptr<Expression> init)
    : name_(name), init_(std::move(init)) {}

void VarDeclStmt::execute(LaneContext &lane, WaveContext &wave,
                          ThreadgroupContext &tg) {
  if (!lane.isActive)
    return;

  Value initVal = init_ ? init_->evaluate(lane, wave, tg) : Value(0);
  lane.variables[name_] = initVal;
}

std::string VarDeclStmt::toString() const {
  return "var " + name_ + " = " + (init_ ? init_->toString() : "0") + ";";
}

AssignStmt::AssignStmt(const std::string &name,
                       std::unique_ptr<Expression> expr)
    : name_(name), expr_(std::move(expr)) {}

void AssignStmt::execute(LaneContext &lane, WaveContext &wave,
                         ThreadgroupContext &tg) {
  if (!lane.isActive)
    return;

  lane.variables[name_] = expr_->evaluate(lane, wave, tg);
}

std::string AssignStmt::toString() const {
  return name_ + " = " + expr_->toString() + ";";
}

IfStmt::IfStmt(std::unique_ptr<Expression> cond,
               std::vector<std::unique_ptr<Statement>> thenBlock,
               std::vector<std::unique_ptr<Statement>> elseBlock)
    : condition_(std::move(cond)), thenBlock_(std::move(thenBlock)),
      elseBlock_(std::move(elseBlock)) {}

void IfStmt::execute(LaneContext &lane, WaveContext &wave,
                     ThreadgroupContext &tg) {
  if (!lane.isActive)
    return;

  // Get current block before if/else divergence
  uint32_t parentBlockId = tg.getCurrentBlock(wave.waveId, lane.laneId);

  bool condValue = condition_->evaluate(lane, wave, tg).asBool();

  // Push merge point for if/else divergence
  std::set<uint32_t>
      divergentBlocks; // Will be populated with then/else block IDs
  wave.pushMergePoint(lane.laneId, static_cast<const void *>(this),
                      parentBlockId, divergentBlocks);

  // Get current merge stack for block creation
  std::vector<MergeStackEntry> currentMergeStack =
      wave.getCurrentMergeStack(lane.laneId);

  // PROACTIVE: Create blocks that actually exist in the code
  bool hasElse = !elseBlock_.empty();
  auto [thenBlockId, elseBlockId] =
      tg.createIfBlocks(static_cast<const void *>(this), parentBlockId,
                        currentMergeStack, hasElse);

  // Update blocks based on this lane's condition result
  if (condValue) {
    // This lane goes to then block
    tg.moveThreadFromUnknownToParticipating(thenBlockId, lane.laneId,
                                            wave.waveId);

    // If else block exists, remove this lane from it
    if (hasElse) {
      tg.removeThreadFromUnknown(elseBlockId, lane.laneId, wave.waveId);
      // Also remove from nested blocks of else block
      tg.removeThreadFromNestedBlocks(elseBlockId, lane.laneId, wave.waveId);
    }

    // Execute then block
    if (lane.isActive) {
      for (auto &stmt : thenBlock_) {
        stmt->execute(lane, wave, tg);
        if (lane.hasReturned) {
          wave.popMergePoint(lane.laneId);
          return;
        }
      }
    }
  } else {
    // This lane chose false path
    if (hasElse) {
      // Lane goes to else block
      tg.moveThreadFromUnknownToParticipating(elseBlockId, lane.laneId,
                                              wave.waveId);
      tg.removeThreadFromUnknown(thenBlockId, lane.laneId, wave.waveId);
      // Also remove from nested blocks of then block
      tg.removeThreadFromNestedBlocks(thenBlockId, lane.laneId, wave.waveId);

      // Execute else block
      if (lane.isActive) {
        for (auto &stmt : elseBlock_) {
          stmt->execute(lane, wave, tg);
          if (lane.hasReturned) {
            wave.popMergePoint(lane.laneId);
            return;
          }
        }
      }
    } else {
      // No else block - lane stays in parent block and skips if entirely
      tg.removeThreadFromUnknown(thenBlockId, lane.laneId, wave.waveId);
      // Also remove from nested blocks of then block
      tg.removeThreadFromNestedBlocks(thenBlockId, lane.laneId, wave.waveId);
      // Lane remains assigned to parentBlockId
    }
  }

  // Pop merge point and return to parent block (reconvergence)
  wave.popMergePoint(lane.laneId);
  tg.assignLaneToBlock(wave.waveId, lane.laneId, parentBlockId);

  // Restore active state (reconvergence)
  lane.isActive = lane.isActive && !lane.hasReturned;
}

bool IfStmt::requiresAllLanesActive() const {
  // Check if any statement in branches requires all lanes
  for (const auto &stmt : thenBlock_) {
    if (stmt->requiresAllLanesActive())
      return true;
  }
  for (const auto &stmt : elseBlock_) {
    if (stmt->requiresAllLanesActive())
      return true;
  }
  return false;
}

std::string IfStmt::toString() const {
  std::string result = "if (" + condition_->toString() + ") {\n";
  for (const auto &stmt : thenBlock_) {
    result += "    " + stmt->toString() + "\n";
  }
  result += "}";
  if (!elseBlock_.empty()) {
    result += " else {\n";
    for (const auto &stmt : elseBlock_) {
      result += "    " + stmt->toString() + "\n";
    }
    result += "}";
  }
  return result;
}

ForStmt::ForStmt(const std::string &var, std::unique_ptr<Expression> init,
                 std::unique_ptr<Expression> cond,
                 std::unique_ptr<Expression> inc,
                 std::vector<std::unique_ptr<Statement>> body)
    : loopVar_(var), init_(std::move(init)), condition_(std::move(cond)),
      increment_(std::move(inc)), body_(std::move(body)) {}

void ForStmt::execute(LaneContext &lane, WaveContext &wave,
                      ThreadgroupContext &tg) {
  if (!lane.isActive)
    return;

  // Get current block before entering loop
  uint32_t parentBlockId = tg.getCurrentBlock(wave.waveId,lane.laneId);

  // Push merge point for loop divergence
  // Each iteration will create different blocks, but they'll all converge after
  // the loop
  std::set<uint32_t>
      divergentBlocks; // Will be populated as iterations create blocks
  wave.pushMergePoint(lane.laneId, static_cast<const void *>(this),
                      parentBlockId, divergentBlocks);

  // Initialize loop variable
  lane.variables[loopVar_] = init_->evaluate(lane, wave, tg);

  // TODO: do we want them to reconverge at the loop condition?
  // TODO: do we want them to reconverge at the same point after the loop even
  // threads have different iterations? Execute loop
  while (condition_->evaluate(lane, wave, tg).asBool()) {
    try {
      // PROACTIVE: Create iteration block for this iteration
      std::vector<MergeStackEntry> currentMergeStack =
          wave.getCurrentMergeStack(lane.laneId);
      uint32_t iterationBlockId = tg.createLoopIterationBlock(
          static_cast<const void *>(this), parentBlockId, currentMergeStack);

      // This lane enters the iteration block
      tg.moveThreadFromUnknownToParticipating(iterationBlockId, lane.laneId,
                                              wave.waveId);

      // Execute body in this iteration's block
      for (auto &stmt : body_) {
        stmt->execute(lane, wave, tg);
        if (lane.hasReturned) {
          wave.popMergePoint(lane.laneId);
          return;
        }
      }
    } catch (const ControlFlowException &e) {
      if (e.type == ControlFlowException::Break) {
        wave.popMergePoint(lane.laneId);
        tg.assignLaneToBlock(wave.waveId, lane.laneId, parentBlockId);
        return; // Exit the loop
      } else if (e.type == ControlFlowException::Continue) {
        // Continue to increment
      }
    }

    // Increment
    lane.variables[loopVar_] = increment_->evaluate(lane, wave, tg);
  }

  // Pop merge point when exiting loop normally
  wave.popMergePoint(lane.laneId);

  // Return to parent block after loop completion
  tg.assignLaneToBlock(wave.waveId, lane.laneId, parentBlockId);
}

std::string ForStmt::toString() const {
  std::string result = "for (" + loopVar_ + " = " + init_->toString() + "; ";
  result += condition_->toString() + "; ";
  result += loopVar_ + " = " + increment_->toString() + ") {\n";
  for (const auto &stmt : body_) {
    result += "    " + stmt->toString() + "\n";
  }
  result += "}";
  return result;
}

ReturnStmt::ReturnStmt(std::unique_ptr<Expression> expr)
    : expr_(std::move(expr)) {}

void ReturnStmt::execute(LaneContext &lane, WaveContext &wave,
                         ThreadgroupContext &tg) {
  if (!lane.isActive)
    return;

  if (expr_) {
    lane.returnValue = expr_->evaluate(lane, wave, tg);
  }

  // Handle comprehensive global cleanup for early return
  handleGlobalEarlyReturn(lane, wave, tg);
}

void ReturnStmt::handleGlobalEarlyReturn(LaneContext &lane, WaveContext &wave,
                                         ThreadgroupContext &tg) {
  LaneId returningLaneId = lane.laneId;

  // 1. Mark lane as returned and inactive
  lane.hasReturned = true;
  lane.isActive = false;
  lane.state = ThreadState::Completed;

  // 2. Remove lane from all execution blocks and update their states
  updateBlockResolutionStates(tg, wave, returningLaneId);

  // 3. Remove lane from all wave operations and update their states
  updateWaveOperationStates(tg, wave, returningLaneId);

  // 4. Remove lane from all barrier operations and update their states
  updateBarrierStates(tg, returningLaneId);

  // 5. Clear the merge stack (early exit from all nested control flow)
  wave.laneMergeStacks[returningLaneId].clear();

  // 6. Remove lane from current block assignment
  wave.laneToCurrentBlock.erase(returningLaneId);

  // 7. Remove from lane waiting map
  wave.laneWaitingAtInstruction.erase(returningLaneId);
}

void ReturnStmt::updateBlockResolutionStates(ThreadgroupContext &tg, WaveContext &wave,
                                             LaneId returningLaneId) {
  // Remove lane from ALL execution blocks and check resolution states
  WaveId waveId = wave.waveId;
  for (auto &[blockId, block] : tg.executionBlocks) {
    // Remove lane from all block participant sets
    block.removeUnknownLane(waveId, returningLaneId);
    block.removeArrivedLane(waveId, returningLaneId);
    block.removeWaitingLane(waveId, returningLaneId);
    block.removeParticipatingLane(waveId, returningLaneId);

    // Remove from per-instruction participants in this block
    for (auto &[instruction, participants] : block.getInstructionParticipants()) {
        block.removeInstructionParticipant(instruction, waveId, returningLaneId);
    }

    // Check if block resolution state changed
    // Block is resolved when no unknown lanes remain (all lanes have chosen to
    // join or return)
    bool wasResolved = block.isWaveAllUnknownResolved(waveId);
    block.setWaveAllUnknownResolved(waveId, block.getUnknownLanesForWave(waveId).empty());

    // If block just became resolved, wake up any lanes waiting for resolution
    if (!wasResolved && block.isWaveAllUnknownResolved(waveId)) {
      // All lanes in this block can now proceed with wave operations
      for (LaneId laneId : block.getWaitingLanesForWave(waveId)) {
        if (laneId < wave.lanes.size() && wave.lanes[laneId] &&
            wave.lanes[laneId]->state == ThreadState::WaitingForWave) {
          // Check if this lane's wave operations can now proceed
          bool canProceed = true;
          auto waitingIt = wave.laneWaitingAtInstruction.find(laneId);
          if (waitingIt != wave.laneWaitingAtInstruction.end()) {
            const void *instruction = waitingIt->second;
            canProceed = wave.canExecuteWaveInstruction(laneId, instruction);
          }

          if (canProceed) {
            wave.lanes[laneId]->state = ThreadState::Ready;
          }
        }
      }
    }
  }
}

void ReturnStmt::updateWaveOperationStates(ThreadgroupContext &tg, WaveContext &wave,
                                           LaneId returningLaneId) {
  // Remove lane from ALL wave operation sync points and update completion
  // states
  std::vector<const void *>
      completedInstructions; // Track instructions that become complete

  for (auto &[instruction, syncPoint] : wave.activeSyncPoints) {
    bool wasExpected =
        syncPoint.expectedParticipants.count(returningLaneId) > 0;

    syncPoint.expectedParticipants.erase(returningLaneId);
    syncPoint.arrivedParticipants.erase(returningLaneId);

    // Update sync point completion status
    bool wasComplete = syncPoint.isComplete;
    syncPoint.allParticipantsArrived =
        (syncPoint.arrivedParticipants == syncPoint.expectedParticipants);

    // Early return helps resolve "all participants known" - one less unknown
    // participant
    if (wasExpected && !syncPoint.allParticipantsKnown) {
      // Check if all expected participants are now known by examining block
      // states
      uint32_t blockId = syncPoint.blockId;
      auto blockIt = tg.executionBlocks.find(blockId);
      if (blockIt != tg.executionBlocks.end()) {
        syncPoint.allParticipantsKnown = blockIt->second.isWaveAllUnknownResolved(wave.waveId);
      }
    }

    syncPoint.isComplete =
        syncPoint.allParticipantsKnown && syncPoint.allParticipantsArrived;

    // If sync point just became complete, mark it for processing
    if (!wasComplete && syncPoint.isComplete) {
      completedInstructions.push_back(instruction);
    }
  }

  // Wake up lanes waiting at newly completed sync points
  for (const void *instruction : completedInstructions) {
    auto &syncPoint = wave.activeSyncPoints[instruction];
    for (LaneId waitingLaneId : syncPoint.arrivedParticipants) {
      if (waitingLaneId < wave.lanes.size() && wave.lanes[waitingLaneId] &&
          wave.lanes[waitingLaneId]->state == ThreadState::WaitingForWave) {
        wave.lanes[waitingLaneId]->state = ThreadState::Ready;
      }
    }
  }
}

void ReturnStmt::updateBarrierStates(ThreadgroupContext &tg,
                                     LaneId returningLaneId) {
  // Convert lane ID to thread ID
  ThreadId returningThreadId = 0;
  bool found = false;
  for (WaveId waveId = 0; waveId < tg.waves.size(); ++waveId) {
    if (returningLaneId < tg.waves[waveId]->lanes.size()) {
      returningThreadId = tg.getGlobalThreadId(waveId, returningLaneId);
      found = true;
      break;
    }
  }

  if (!found)
    return;

  // Remove thread from all active barriers
  std::vector<uint32_t>
      completedBarriers; // Track barriers that become complete

  for (auto &[barrierId, barrier] : tg.activeBarriers) {
    barrier.participatingThreads.erase(returningThreadId);
    barrier.arrivedThreads.erase(returningThreadId);

    // Update barrier completion status
    bool wasComplete = barrier.isComplete;
    barrier.isComplete =
        (barrier.arrivedThreads == barrier.participatingThreads);

    // If barrier just became complete, mark it for processing
    if (!wasComplete && barrier.isComplete) {
      completedBarriers.push_back(barrierId);
    }
  }

  // Wake up threads waiting at newly completed barriers
  for (uint32_t barrierId : completedBarriers) {
    auto &barrier = tg.activeBarriers[barrierId];
    for (ThreadId waitingThreadId : barrier.arrivedThreads) {
      auto [waveId, laneId] = tg.getWaveAndLane(waitingThreadId);
      if (waveId < tg.waves.size() && laneId < tg.waves[waveId]->lanes.size()) {
        auto &lane = tg.waves[waveId]->lanes[laneId];
        if (lane && lane->state == ThreadState::WaitingAtBarrier) {
          lane->state = ThreadState::Ready;
        }
      }
    }
  }
}

std::string ReturnStmt::toString() const {
  return "return" + (expr_ ? " " + expr_->toString() : "") + ";";
}

void BarrierStmt::execute(LaneContext &lane, WaveContext &wave,
                          ThreadgroupContext &tg) {
  // Phase 1: Simple barrier implementation
  // TODO: Add proper cooperative barrier handling in Phase 2

  // For now, barriers are no-ops since we don't have full cooperative
  // scheduling In a real GPU, this would synchronize all threads in the
  // threadgroup ThreadId tid = tg.getGlobalThreadId(wave.waveId, lane.laneId);
  // // Would be used in full implementation
}

ExprStmt::ExprStmt(std::unique_ptr<Expression> expr) : expr_(std::move(expr)) {}

void ExprStmt::execute(LaneContext &lane, WaveContext &wave,
                       ThreadgroupContext &tg) {
  if (!lane.isActive)
    return;

  // Execute the expression (evaluate it but don't store the result)
  if (expr_) {
    expr_->evaluate(lane, wave, tg);
  }
}

std::string ExprStmt::toString() const {
  if (expr_) {
    return expr_->toString() + ";";
  }
  return "ExprStmt();";
}

SharedWriteStmt::SharedWriteStmt(MemoryAddress addr,
                                 std::unique_ptr<Expression> expr)
    : addr_(addr), expr_(std::move(expr)) {}

void SharedWriteStmt::execute(LaneContext &lane, WaveContext &wave,
                              ThreadgroupContext &tg) {
  if (!lane.isActive)
    return;

  Value value = expr_->evaluate(lane, wave, tg);
  ThreadId tid = tg.getGlobalThreadId(wave.waveId, lane.laneId);
  tg.sharedMemory->write(addr_, value, tid);
}

std::string SharedWriteStmt::toString() const {
  return "g_shared[" + std::to_string(addr_) + "] = " + expr_->toString() + ";";
}

Value SharedReadExpr::evaluate(LaneContext &lane, WaveContext &wave,
                               ThreadgroupContext &tg) const {
  ThreadId tid = tg.getGlobalThreadId(wave.waveId, lane.laneId);
  return tg.sharedMemory->read(addr_, tid);
}

std::string SharedReadExpr::toString() const {
  return "g_shared[" + std::to_string(addr_) + "]";
}

// MiniHLSLInterpreter implementation with cooperative scheduling
ExecutionResult
MiniHLSLInterpreter::executeWithOrdering(const Program &program,
                                         const ThreadOrdering &ordering) {
  ExecutionResult result;

  // Create threadgroup context
  const uint32_t waveSize = 32; // Standard wave size
  ThreadgroupContext tgContext(program.getTotalThreads(), waveSize);

  try {
    uint32_t orderingIndex = 0;
    uint32_t maxIterations = program.getTotalThreads() *
                             program.statements.size() * 10; // Safety limit
    uint32_t iteration = 0;

    // Cooperative scheduling main loop
    while (iteration < maxIterations) {
      iteration++;

      // Process completed wave operations and barriers
      processWaveOperations(tgContext);
      processBarriers(tgContext);

      // Get threads ready for execution
      auto readyThreads = tgContext.getReadyThreads();
      if (readyThreads.empty()) {
        // Check if all threads are completed
        bool allCompleted = true;
        for (const auto &wave : tgContext.waves) {
          for (const auto &lane : wave->lanes) {
            if (lane->state != ThreadState::Completed &&
                lane->state != ThreadState::Error) {
              allCompleted = false;
              break;
            }
          }
          if (!allCompleted)
            break;
        }

        if (allCompleted) {
          break; // All threads finished
        }

        // Check for deadlock
        auto waitingThreads = tgContext.getWaitingThreads();
        if (!waitingThreads.empty()) {
          result.errorMessage =
              "Deadlock detected: threads waiting but no progress possible";
          break;
        }

        continue; // Wait for synchronization to complete
      }

      // Select next thread to execute according to ordering
      ThreadId nextTid =
          selectNextThread(readyThreads, ordering, orderingIndex);

      // Execute one step of the selected thread
      bool continueExecution = executeOneStep(nextTid, program, tgContext);
      if (!continueExecution) {
        break; // Fatal error occurred
      }
    }

    if (iteration >= maxIterations) {
      result.errorMessage =
          "Execution timeout: possible infinite loop or deadlock";
    }

    // Collect return values in thread order
    for (ThreadId tid = 0; tid < program.getTotalThreads(); ++tid) {
      auto [waveId, laneId] = tgContext.getWaveAndLane(tid);
      if (waveId < tgContext.waves.size() &&
          laneId < tgContext.waves[waveId]->lanes.size()) {
        result.threadReturnValues.push_back(
            tgContext.waves[waveId]->lanes[laneId]->returnValue);
      }
    }

  } catch (const std::exception &e) {
    result.errorMessage = std::string("Runtime error: ") + e.what();
  }

  // Collect final state
  result.sharedMemoryState = tgContext.sharedMemory->getSnapshot();

  // Collect global variables from first thread
  if (!tgContext.waves.empty() && !tgContext.waves[0]->lanes.empty()) {
    result.globalVariables = tgContext.waves[0]->lanes[0]->variables;
  }

  return result;
}

// Phase 1: Simple implementation to fix wave operations
bool MiniHLSLInterpreter::executeOneStep(ThreadId tid, const Program &program,
                                         ThreadgroupContext &tgContext) {
  auto [waveId, laneId] = tgContext.getWaveAndLane(tid);
  if (waveId >= tgContext.waves.size())
    return false;

  auto &wave = *tgContext.waves[waveId];
  if (laneId >= wave.lanes.size())
    return false;

  auto &lane = *wave.lanes[laneId];

  // Check if thread is ready to execute
  if (lane.state != ThreadState::Ready)
    return true;

  // Check if we have more statements to execute
  if (lane.currentStatement >= program.statements.size()) {
    lane.state = ThreadState::Completed;
    return true;
  }

  // Execute the current statement
  try {
    const auto &stmt = program.statements[lane.currentStatement];
    stmt->execute(lane, wave, tgContext);
    lane.currentStatement++;

    if (lane.hasReturned) {
      lane.state = ThreadState::Completed;
    }
  } catch (const std::exception &e) {
    lane.state = ThreadState::Error;
    lane.errorMessage = e.what();
  }

  return true;
}

void MiniHLSLInterpreter::processWaveOperations(ThreadgroupContext &tgContext) {
  // Phase 1: Simple implementation - wave ops complete immediately
  // TODO: Add proper cooperative scheduling in Phase 2
}

void MiniHLSLInterpreter::processBarriers(ThreadgroupContext &tgContext) {
  // Phase 1: Simple implementation - barriers complete immediately
  // TODO: Add proper barrier analysis in Phase 2
}

ThreadId
MiniHLSLInterpreter::selectNextThread(const std::vector<ThreadId> &readyThreads,
                                      const ThreadOrdering &ordering,
                                      uint32_t &orderingIndex) {
  // Simple round-robin selection from ready threads
  // Try to follow the ordering preference when possible
  if (readyThreads.empty())
    return 0;

  // Look for the next thread in ordering that's ready
  for (uint32_t i = 0; i < ordering.executionOrder.size(); ++i) {
    uint32_t idx = (orderingIndex + i) % ordering.executionOrder.size();
    ThreadId tid = ordering.executionOrder[idx];

    if (std::find(readyThreads.begin(), readyThreads.end(), tid) !=
        readyThreads.end()) {
      orderingIndex = (idx + 1) % ordering.executionOrder.size();
      return tid;
    }
  }

  // Fallback to first ready thread
  return readyThreads[0];
}

bool MiniHLSLInterpreter::areResultsEquivalent(const ExecutionResult &r1,
                                               const ExecutionResult &r2,
                                               double epsilon) {
  // Check error states
  if (!r1.isValid() || !r2.isValid()) {
    return false;
  }

  // Check shared memory state
  if (r1.sharedMemoryState.size() != r2.sharedMemoryState.size()) {
    return false;
  }

  for (const auto &[addr, val1] : r1.sharedMemoryState) {
    auto it = r2.sharedMemoryState.find(addr);
    if (it == r2.sharedMemoryState.end()) {
      return false;
    }

    // Compare values with epsilon for floating point
    if (std::abs(val1.asFloat() - it->second.asFloat()) > epsilon) {
      return false;
    }
  }

  // Check return values (order might differ, so we sort)
  auto returns1 = r1.threadReturnValues;
  auto returns2 = r2.threadReturnValues;

  std::sort(
      returns1.begin(), returns1.end(),
      [](const Value &a, const Value &b) { return a.asFloat() < b.asFloat(); });
  std::sort(
      returns2.begin(), returns2.end(),
      [](const Value &a, const Value &b) { return a.asFloat() < b.asFloat(); });

  if (returns1.size() != returns2.size()) {
    return false;
  }

  for (size_t i = 0; i < returns1.size(); ++i) {
    if (std::abs(returns1[i].asFloat() - returns2[i].asFloat()) > epsilon) {
      return false;
    }
  }

  return true;
}

MiniHLSLInterpreter::VerificationResult
MiniHLSLInterpreter::verifyOrderIndependence(const Program &program,
                                             uint32_t numOrderings) {
  VerificationResult verification;

  // Generate test orderings
  verification.orderings =
      generateTestOrderings(program.getTotalThreads(), numOrderings);

  // Execute with each ordering
  for (const auto &ordering : verification.orderings) {
    verification.results.push_back(executeWithOrdering(program, ordering));
  }

  // Check if all results are equivalent
  verification.isOrderIndependent = true;
  if (!verification.results.empty()) {
    const auto &reference = verification.results[0];

    for (size_t i = 1; i < verification.results.size(); ++i) {
      if (!areResultsEquivalent(reference, verification.results[i])) {
        verification.isOrderIndependent = false;

        // Generate divergence report
        std::stringstream report;
        report << "Order dependence detected!\n";
        report << "Reference ordering: "
               << verification.orderings[0].description << "\n";
        report << "Divergent ordering: "
               << verification.orderings[i].description << "\n";
        report << "Differences in shared memory or return values detected.\n";

        verification.divergenceReport = report.str();
        break;
      }
    }
  }

  return verification;
}

ExecutionResult MiniHLSLInterpreter::execute(const Program &program,
                                             const ThreadOrdering &ordering) {
  return executeWithOrdering(program, ordering);
}

std::vector<ThreadOrdering>
MiniHLSLInterpreter::generateTestOrderings(uint32_t threadCount,
                                           uint32_t numOrderings) {
  std::vector<ThreadOrdering> orderings;

  // Always include sequential ordering
  orderings.push_back(ThreadOrdering::sequential(threadCount));

  // Add reverse sequential
  if (numOrderings > 1) {
    orderings.push_back(ThreadOrdering::reverseSequential(threadCount));
  }

  // Add even-odd interleaved
  if (numOrderings > 2) {
    orderings.push_back(ThreadOrdering::evenOddInterleaved(threadCount));
  }

  // Add wave interleaved
  if (numOrderings > 3) {
    orderings.push_back(ThreadOrdering::waveInterleaved(threadCount, 32));
  }

  // Fill remaining with random orderings
  for (uint32_t i = orderings.size(); i < numOrderings; ++i) {
    std::uniform_int_distribution<uint32_t> dist(0, UINT32_MAX);
    uint32_t seed = dist(rng_);
    orderings.push_back(ThreadOrdering::random(threadCount, seed));
  }

  return orderings;
}

// Helper function implementations
std::unique_ptr<Expression> makeLiteral(Value v) {
  return std::make_unique<LiteralExpr>(v);
}

std::unique_ptr<Expression> makeVariable(const std::string &name) {
  return std::make_unique<VariableExpr>(name);
}

std::unique_ptr<Expression> makeLaneIndex() {
  return std::make_unique<LaneIndexExpr>();
}

std::unique_ptr<Expression> makeWaveIndex() {
  return std::make_unique<WaveIndexExpr>();
}

std::unique_ptr<Expression> makeThreadIndex() {
  return std::make_unique<ThreadIndexExpr>();
}

std::unique_ptr<Expression> makeBinaryOp(std::unique_ptr<Expression> left,
                                         std::unique_ptr<Expression> right,
                                         BinaryOpExpr::OpType op) {
  return std::make_unique<BinaryOpExpr>(std::move(left), std::move(right), op);
}

std::unique_ptr<Expression> makeWaveSum(std::unique_ptr<Expression> expr) {
  return std::make_unique<WaveActiveOp>(std::move(expr), WaveActiveOp::Sum);
}

std::unique_ptr<Statement> makeVarDecl(const std::string &name,
                                       std::unique_ptr<Expression> init) {
  return std::make_unique<VarDeclStmt>(name, std::move(init));
}

std::unique_ptr<Statement> makeAssign(const std::string &name,
                                      std::unique_ptr<Expression> expr) {
  return std::make_unique<AssignStmt>(name, std::move(expr));
}

std::unique_ptr<Statement>
makeIf(std::unique_ptr<Expression> cond,
       std::vector<std::unique_ptr<Statement>> thenBlock,
       std::vector<std::unique_ptr<Statement>> elseBlock) {
  return std::make_unique<IfStmt>(std::move(cond), std::move(thenBlock),
                                  std::move(elseBlock));
}

// HLSL AST conversion implementation (simplified version)
MiniHLSLInterpreter::ConversionResult
MiniHLSLInterpreter::convertFromHLSLAST(const clang::FunctionDecl *func,
                                        clang::ASTContext &context) {
  ConversionResult result;
  result.success = false;

  if (!func || !func->hasBody()) {
    result.errorMessage = "Function has no body or is null";
    return result;
  }

  std::cout << "Converting HLSL function: " << func->getName().str()
            << std::endl;

  try {
    // Extract thread configuration from function attributes
    extractThreadConfiguration(func, result.program);

    // Get the function body
    const clang::CompoundStmt *body =
        clang::dyn_cast<clang::CompoundStmt>(func->getBody());
    if (!body) {
      result.errorMessage = "Function body is not a compound statement";
      return result;
    }

    // Convert the function body to interpreter statements
    convertCompoundStatement(body, result.program, context);

    std::cout << "Converted AST to interpreter program with "
              << result.program.statements.size() << " statements" << std::endl;

    result.success = true;
    return result;

  } catch (const std::exception &e) {
    result.errorMessage =
        std::string("Exception during conversion: ") + e.what();
    return result;
  }
}

// AST traversal helper methods
void MiniHLSLInterpreter::extractThreadConfiguration(
    const clang::FunctionDecl *func, Program &program) {
  // Default configuration
  program.numThreadsX = 1;
  program.numThreadsY = 1;
  program.numThreadsZ = 1;

  // Look for HLSLNumThreadsAttr
  if (const clang::HLSLNumThreadsAttr *attr =
          func->getAttr<clang::HLSLNumThreadsAttr>()) {
    program.numThreadsX = attr->getX();
    program.numThreadsY = attr->getY();
    program.numThreadsZ = attr->getZ();
    std::cout << "Found numthreads attribute: [" << program.numThreadsX << ", "
              << program.numThreadsY << ", " << program.numThreadsZ << "]"
              << std::endl;
  } else {
    std::cout << "No numthreads attribute found, using default [1, 1, 1]"
              << std::endl;
  }
}

void MiniHLSLInterpreter::convertCompoundStatement(
    const clang::CompoundStmt *compound, Program &program,
    clang::ASTContext &context) {
  std::cout << "Converting compound statement with " << compound->size()
            << " child statements" << std::endl;

  for (const auto *stmt : compound->children()) {
    if (auto convertedStmt = convertStatement(stmt, context)) {
      program.statements.push_back(std::move(convertedStmt));
    }
  }
}

std::unique_ptr<Statement>
MiniHLSLInterpreter::convertStatement(const clang::Stmt *stmt,
                                      clang::ASTContext &context) {
  if (!stmt)
    return nullptr;

  std::cout << "Converting statement: " << stmt->getStmtClassName()
            << std::endl;

  // Handle different statement types
  if (auto binOp = clang::dyn_cast<clang::BinaryOperator>(stmt)) {
    return convertBinaryOperator(binOp, context);
  } else if (auto callExpr = clang::dyn_cast<clang::CallExpr>(stmt)) {
    return convertCallExpression(callExpr, context);
  } else if (auto declStmt = clang::dyn_cast<clang::DeclStmt>(stmt)) {
    return convertDeclarationStatement(declStmt, context);
  }
  // Note: ExprStmt doesn't exist in DXC's Clang fork
  // Expression statements are handled differently or don't exist
  else if (auto ifStmt = clang::dyn_cast<clang::IfStmt>(stmt)) {
    return convertIfStatement(ifStmt, context);
  } else if (auto forStmt = clang::dyn_cast<clang::ForStmt>(stmt)) {
    return convertForStatement(forStmt, context);
  } else if (auto whileStmt = clang::dyn_cast<clang::WhileStmt>(stmt)) {
    return convertWhileStatement(whileStmt, context);
  } else if (auto doStmt = clang::dyn_cast<clang::DoStmt>(stmt)) {
    return convertDoStatement(doStmt, context);
  } else if (auto switchStmt = clang::dyn_cast<clang::SwitchStmt>(stmt)) {
    return convertSwitchStatement(switchStmt, context);
  } else if (auto breakStmt = clang::dyn_cast<clang::BreakStmt>(stmt)) {
    return convertBreakStatement(breakStmt, context);
  } else if (auto continueStmt = clang::dyn_cast<clang::ContinueStmt>(stmt)) {
    return convertContinueStatement(continueStmt, context);
  } else if (auto compound = clang::dyn_cast<clang::CompoundStmt>(stmt)) {
    // Nested compound statement - this should not happen in our current design
    std::cout << "Warning: nested compound statement found, skipping"
              << std::endl;
    return nullptr;
  } else {
    std::cout << "Unsupported statement type: " << stmt->getStmtClassName()
              << std::endl;
    return nullptr;
  }
}

std::unique_ptr<Statement>
MiniHLSLInterpreter::convertBinaryOperator(const clang::BinaryOperator *binOp,
                                           clang::ASTContext &context) {
  if (!binOp->isAssignmentOp()) {
    std::cout << "Non-assignment binary operator, skipping" << std::endl;
    return nullptr;
  }

  // Handle assignment: LHS = RHS
  auto lhs = convertExpression(binOp->getLHS(), context);
  auto rhs = convertExpression(binOp->getRHS(), context);

  if (!rhs) {
    std::cout << "Failed to convert assignment RHS" << std::endl;
    return nullptr;
  }

  // Determine the target variable name
  std::string targetVar = "unknown";
  if (auto declRef = clang::dyn_cast<clang::DeclRefExpr>(binOp->getLHS())) {
    targetVar = declRef->getDecl()->getName().str();
  } else if (clang::isa<clang::CXXOperatorCallExpr>(binOp->getLHS())) {
    targetVar = "buffer_write"; // Array access assignment
  }

  return makeAssign(targetVar, std::move(rhs));
}

std::unique_ptr<Statement>
MiniHLSLInterpreter::convertCallExpression(const clang::CallExpr *callExpr,
                                           clang::ASTContext &context) {
  if (auto funcDecl = callExpr->getDirectCallee()) {
    std::string funcName = funcDecl->getName().str();
    std::cout << "Converting function call: " << funcName << std::endl;

    // Check for barrier functions
    if (funcName == "GroupMemoryBarrierWithGroupSync" ||
        funcName == "AllMemoryBarrierWithGroupSync" ||
        funcName == "DeviceMemoryBarrierWithGroupSync") {
      return std::make_unique<BarrierStmt>();
    }

    // Check for wave intrinsic functions
    if (funcName == "WaveActiveSum" && callExpr->getNumArgs() == 1) {
      auto arg = convertExpression(callExpr->getArg(0), context);
      if (arg) {
        return std::make_unique<ExprStmt>(
            std::make_unique<WaveActiveOp>(std::move(arg), WaveActiveOp::Sum));
      }
    } else if (funcName == "WaveActiveProduct" && callExpr->getNumArgs() == 1) {
      auto arg = convertExpression(callExpr->getArg(0), context);
      if (arg) {
        return std::make_unique<ExprStmt>(std::make_unique<WaveActiveOp>(
            std::move(arg), WaveActiveOp::Product));
      }
    } else if (funcName == "WaveActiveMin" && callExpr->getNumArgs() == 1) {
      auto arg = convertExpression(callExpr->getArg(0), context);
      if (arg) {
        return std::make_unique<ExprStmt>(
            std::make_unique<WaveActiveOp>(std::move(arg), WaveActiveOp::Min));
      }
    } else if (funcName == "WaveActiveMax" && callExpr->getNumArgs() == 1) {
      auto arg = convertExpression(callExpr->getArg(0), context);
      if (arg) {
        return std::make_unique<ExprStmt>(
            std::make_unique<WaveActiveOp>(std::move(arg), WaveActiveOp::Max));
      }
    } else if (funcName == "WaveActiveAnd" && callExpr->getNumArgs() == 1) {
      auto arg = convertExpression(callExpr->getArg(0), context);
      if (arg) {
        return std::make_unique<ExprStmt>(
            std::make_unique<WaveActiveOp>(std::move(arg), WaveActiveOp::And));
      }
    } else if (funcName == "WaveActiveOr" && callExpr->getNumArgs() == 1) {
      auto arg = convertExpression(callExpr->getArg(0), context);
      if (arg) {
        return std::make_unique<ExprStmt>(
            std::make_unique<WaveActiveOp>(std::move(arg), WaveActiveOp::Or));
      }
    } else if (funcName == "WaveActiveXor" && callExpr->getNumArgs() == 1) {
      auto arg = convertExpression(callExpr->getArg(0), context);
      if (arg) {
        return std::make_unique<ExprStmt>(
            std::make_unique<WaveActiveOp>(std::move(arg), WaveActiveOp::Xor));
      }
    } else if (funcName == "WaveActiveCountBits" &&
               callExpr->getNumArgs() == 1) {
      auto arg = convertExpression(callExpr->getArg(0), context);
      if (arg) {
        return std::make_unique<ExprStmt>(std::make_unique<WaveActiveOp>(
            std::move(arg), WaveActiveOp::CountBits));
      }
    } else if (funcName == "WaveGetLaneIndex" && callExpr->getNumArgs() == 0) {
      return std::make_unique<ExprStmt>(std::make_unique<LaneIndexExpr>());
    } else if (funcName == "WaveGetLaneCount" && callExpr->getNumArgs() == 0) {
      return std::make_unique<ExprStmt>(
          std::make_unique<WaveGetLaneCountExpr>());
    }

    // Handle other function calls as needed
    std::cout << "Unsupported function call: " << funcName << std::endl;
  }

  return nullptr;
}

std::unique_ptr<Statement> MiniHLSLInterpreter::convertDeclarationStatement(
    const clang::DeclStmt *declStmt, clang::ASTContext &context) {
  std::cout << "Converting declaration statement" << std::endl;

  for (const auto *decl : declStmt->decls()) {
    if (auto varDecl = clang::dyn_cast<clang::VarDecl>(decl)) {
      std::string varName = varDecl->getName().str();
      std::cout << "Declaring variable: " << varName << std::endl;

      // If it has an initializer, create an assignment
      if (varDecl->hasInit()) {
        auto initExpr = convertExpression(varDecl->getInit(), context);
        if (initExpr) {
          return makeAssign(varName, std::move(initExpr));
        }
      }
    }
  }

  return nullptr;
}

std::unique_ptr<Statement>
MiniHLSLInterpreter::convertIfStatement(const clang::IfStmt *ifStmt,
                                        clang::ASTContext &context) {
  std::cout << "Converting if statement" << std::endl;

  // Convert the condition expression
  auto condition = convertExpression(ifStmt->getCond(), context);
  if (!condition) {
    std::cout << "Failed to convert if condition" << std::endl;
    return nullptr;
  }

  // Convert the then block
  std::vector<std::unique_ptr<Statement>> thenBlock;
  if (auto thenStmt = ifStmt->getThen()) {
    if (auto compound = clang::dyn_cast<clang::CompoundStmt>(thenStmt)) {
      // Handle compound statement (block with {})
      for (auto stmt : compound->body()) {
        if (auto convertedStmt = convertStatement(stmt, context)) {
          thenBlock.push_back(std::move(convertedStmt));
        }
      }
    } else {
      // Handle single statement
      if (auto convertedStmt = convertStatement(thenStmt, context)) {
        thenBlock.push_back(std::move(convertedStmt));
      }
    }
  }

  // Convert the else block (if it exists)
  std::vector<std::unique_ptr<Statement>> elseBlock;
  if (auto elseStmt = ifStmt->getElse()) {
    if (auto compound = clang::dyn_cast<clang::CompoundStmt>(elseStmt)) {
      // Handle compound statement (block with {})
      for (auto stmt : compound->body()) {
        if (auto convertedStmt = convertStatement(stmt, context)) {
          elseBlock.push_back(std::move(convertedStmt));
        }
      }
    } else {
      // Handle single statement (including else if)
      if (auto convertedStmt = convertStatement(elseStmt, context)) {
        elseBlock.push_back(std::move(convertedStmt));
      }
    }
  }

  return std::make_unique<IfStmt>(std::move(condition), std::move(thenBlock),
                                  std::move(elseBlock));
}

std::unique_ptr<Statement>
MiniHLSLInterpreter::convertForStatement(const clang::ForStmt *forStmt,
                                         clang::ASTContext &context) {
  std::cout << "Converting for statement" << std::endl;

  // For loops in HLSL typically have the structure: for (init; condition;
  // increment) { body } We need to extract each component

  // Extract the loop variable from the init statement
  std::string loopVar;
  std::unique_ptr<Expression> init = nullptr;

  if (auto initStmt = forStmt->getInit()) {
    if (auto declStmt = clang::dyn_cast<clang::DeclStmt>(initStmt)) {
      // Handle variable declaration: int i = 0
      for (const auto *decl : declStmt->decls()) {
        if (auto varDecl = clang::dyn_cast<clang::VarDecl>(decl)) {
          loopVar = varDecl->getName().str();
          if (varDecl->hasInit()) {
            init = convertExpression(varDecl->getInit(), context);
          }
          break; // Take the first variable
        }
      }
    } else {
      // Handle assignment: i = 0
      std::cout << "For loop init is not a declaration statement" << std::endl;
      return nullptr;
    }
  }

  // Convert the condition expression
  std::unique_ptr<Expression> condition = nullptr;
  if (auto condExpr = forStmt->getCond()) {
    condition = convertExpression(condExpr, context);
  }

  // Convert the increment expression
  std::unique_ptr<Expression> increment = nullptr;
  if (auto incExpr = forStmt->getInc()) {
    increment = convertExpression(incExpr, context);
  }

  // Convert the loop body
  std::vector<std::unique_ptr<Statement>> body;
  if (auto bodyStmt = forStmt->getBody()) {
    if (auto compound = clang::dyn_cast<clang::CompoundStmt>(bodyStmt)) {
      // Handle compound statement (block with {})
      for (auto stmt : compound->body()) {
        if (auto convertedStmt = convertStatement(stmt, context)) {
          body.push_back(std::move(convertedStmt));
        }
      }
    } else {
      // Handle single statement
      if (auto convertedStmt = convertStatement(bodyStmt, context)) {
        body.push_back(std::move(convertedStmt));
      }
    }
  }

  // Validate that we have the required components
  if (loopVar.empty() || !init || !condition || !increment) {
    std::cout << "For loop missing required components (var: " << loopVar
              << ", init: " << (init ? "yes" : "no")
              << ", condition: " << (condition ? "yes" : "no")
              << ", increment: " << (increment ? "yes" : "no") << ")"
              << std::endl;
    return nullptr;
  }

  return std::make_unique<ForStmt>(loopVar, std::move(init),
                                   std::move(condition), std::move(increment),
                                   std::move(body));
}

std::unique_ptr<Statement>
MiniHLSLInterpreter::convertWhileStatement(const clang::WhileStmt *whileStmt,
                                           clang::ASTContext &context) {
  std::cout << "Converting while statement" << std::endl;

  // Convert the condition expression
  auto condition = convertExpression(whileStmt->getCond(), context);
  if (!condition) {
    std::cout << "Failed to convert while condition" << std::endl;
    return nullptr;
  }

  // Convert the loop body
  std::vector<std::unique_ptr<Statement>> body;
  if (auto bodyStmt = whileStmt->getBody()) {
    if (auto compound = clang::dyn_cast<clang::CompoundStmt>(bodyStmt)) {
      for (auto stmt : compound->body()) {
        if (auto convertedStmt = convertStatement(stmt, context)) {
          body.push_back(std::move(convertedStmt));
        }
      }
    } else {
      if (auto convertedStmt = convertStatement(bodyStmt, context)) {
        body.push_back(std::move(convertedStmt));
      }
    }
  }

  return std::make_unique<WhileStmt>(std::move(condition), std::move(body));
}

std::unique_ptr<Statement>
MiniHLSLInterpreter::convertDoStatement(const clang::DoStmt *doStmt,
                                        clang::ASTContext &context) {
  std::cout << "Converting do-while statement" << std::endl;

  // Convert the loop body
  std::vector<std::unique_ptr<Statement>> body;
  if (auto bodyStmt = doStmt->getBody()) {
    if (auto compound = clang::dyn_cast<clang::CompoundStmt>(bodyStmt)) {
      for (auto stmt : compound->body()) {
        if (auto convertedStmt = convertStatement(stmt, context)) {
          body.push_back(std::move(convertedStmt));
        }
      }
    } else {
      if (auto convertedStmt = convertStatement(bodyStmt, context)) {
        body.push_back(std::move(convertedStmt));
      }
    }
  }

  // Convert the condition expression
  auto condition = convertExpression(doStmt->getCond(), context);
  if (!condition) {
    std::cout << "Failed to convert do-while condition" << std::endl;
    return nullptr;
  }

  return std::make_unique<DoWhileStmt>(std::move(body), std::move(condition));
}

std::unique_ptr<Statement>
MiniHLSLInterpreter::convertSwitchStatement(const clang::SwitchStmt *switchStmt,
                                            clang::ASTContext &context) {
  std::cout << "Converting switch statement" << std::endl;

  // Convert the condition expression
  auto condition = convertExpression(switchStmt->getCond(), context);
  if (!condition) {
    std::cout << "Failed to convert switch condition" << std::endl;
    return nullptr;
  }

  auto switchResult = std::make_unique<SwitchStmt>(std::move(condition));

  // Convert the switch body - we need to handle case statements specially
  if (auto body = switchStmt->getBody()) {
    if (auto compound = clang::dyn_cast<clang::CompoundStmt>(body)) {
      std::vector<std::unique_ptr<Statement>> currentCase;
      std::optional<int> currentCaseValue;
      bool isDefault = false;

      for (auto stmt : compound->body()) {
        if (auto caseStmt = clang::dyn_cast<clang::CaseStmt>(stmt)) {
          // Save previous case if any
          if (currentCaseValue.has_value() || isDefault) {
            if (isDefault) {
              switchResult->addDefault(std::move(currentCase));
            } else {
              switchResult->addCase(currentCaseValue.value(),
                                    std::move(currentCase));
            }
            currentCase.clear();
            isDefault = false;
          }

          // Get case value
          if (auto lhs = caseStmt->getLHS()) {
            if (auto intLit = clang::dyn_cast<clang::IntegerLiteral>(lhs)) {
              currentCaseValue = intLit->getValue().getSExtValue();
            }
          }

          // Convert case body
          if (auto substmt = caseStmt->getSubStmt()) {
            if (auto converted = convertStatement(substmt, context)) {
              currentCase.push_back(std::move(converted));
            }
          }
        } else if (clang::isa<clang::DefaultStmt>(stmt)) {
          // Save previous case if any
          if (currentCaseValue.has_value() || isDefault) {
            if (isDefault) {
              switchResult->addDefault(std::move(currentCase));
            } else {
              switchResult->addCase(currentCaseValue.value(),
                                    std::move(currentCase));
            }
            currentCase.clear();
          }

          isDefault = true;
          currentCaseValue.reset();
        } else {
          // Regular statement in current case
          if (auto converted = convertStatement(stmt, context)) {
            currentCase.push_back(std::move(converted));
          }
        }
      }

      // Save last case
      if (currentCaseValue.has_value() || isDefault) {
        if (isDefault) {
          switchResult->addDefault(std::move(currentCase));
        } else {
          switchResult->addCase(currentCaseValue.value(),
                                std::move(currentCase));
        }
      }
    }
  }

  return switchResult;
}

std::unique_ptr<Statement>
MiniHLSLInterpreter::convertBreakStatement(const clang::BreakStmt *breakStmt,
                                           clang::ASTContext &context) {
  std::cout << "Converting break statement" << std::endl;
  return std::make_unique<BreakStmt>();
}

std::unique_ptr<Statement> MiniHLSLInterpreter::convertContinueStatement(
    const clang::ContinueStmt *continueStmt, clang::ASTContext &context) {
  std::cout << "Converting continue statement" << std::endl;
  return std::make_unique<ContinueStmt>();
}

std::unique_ptr<Expression>
MiniHLSLInterpreter::convertExpression(const clang::Expr *expr,
                                       clang::ASTContext &context) {
  if (!expr)
    return nullptr;

  std::cout << "Converting expression: " << expr->getStmtClassName()
            << std::endl;

  // Handle different expression types
  if (auto binOp = clang::dyn_cast<clang::BinaryOperator>(expr)) {
    return convertBinaryExpression(binOp, context);
  } else if (auto declRef = clang::dyn_cast<clang::DeclRefExpr>(expr)) {
    std::string varName = declRef->getDecl()->getName().str();
    return makeVariable(varName);
  } else if (auto intLit = clang::dyn_cast<clang::IntegerLiteral>(expr)) {
    int64_t value = intLit->getValue().getSExtValue();
    return makeLiteral(Value(static_cast<int>(value)));
  } else if (auto floatLit = clang::dyn_cast<clang::FloatingLiteral>(expr)) {
    double value = floatLit->getValueAsApproximateDouble();
    return makeLiteral(Value(static_cast<float>(value)));
  } else if (auto boolLit = clang::dyn_cast<clang::CXXBoolLiteralExpr>(expr)) {
    bool value = boolLit->getValue();
    return makeLiteral(Value(value));
  } else if (auto parenExpr = clang::dyn_cast<clang::ParenExpr>(expr)) {
    return convertExpression(parenExpr->getSubExpr(), context);
  } else if (auto implicitCast =
                 clang::dyn_cast<clang::ImplicitCastExpr>(expr)) {
    return convertExpression(implicitCast->getSubExpr(), context);
  } else if (auto operatorCall =
                 clang::dyn_cast<clang::CXXOperatorCallExpr>(expr)) {
    return convertOperatorCall(operatorCall, context);
  } else if (auto callExpr = clang::dyn_cast<clang::CallExpr>(expr)) {
    return convertCallExpressionToExpression(callExpr, context);
  } else if (auto condOp = clang::dyn_cast<clang::ConditionalOperator>(expr)) {
    return convertConditionalOperator(condOp, context);
  } else {
    std::cout << "Unsupported expression type: " << expr->getStmtClassName()
              << std::endl;
    return nullptr;
  }
}

std::unique_ptr<Expression>
MiniHLSLInterpreter::convertCallExpressionToExpression(
    const clang::CallExpr *callExpr, clang::ASTContext &context) {
  if (auto funcDecl = callExpr->getDirectCallee()) {
    std::string funcName = funcDecl->getName().str();
    std::cout << "Converting function call to expression: " << funcName
              << std::endl;

    // Check for wave intrinsic functions that return values
    if (funcName == "WaveActiveSum" && callExpr->getNumArgs() == 1) {
      auto arg = convertExpression(callExpr->getArg(0), context);
      if (arg) {
        return std::make_unique<WaveActiveOp>(std::move(arg),
                                              WaveActiveOp::Sum);
      }
    } else if (funcName == "WaveActiveProduct" && callExpr->getNumArgs() == 1) {
      auto arg = convertExpression(callExpr->getArg(0), context);
      if (arg) {
        return std::make_unique<WaveActiveOp>(std::move(arg),
                                              WaveActiveOp::Product);
      }
    } else if (funcName == "WaveActiveMin" && callExpr->getNumArgs() == 1) {
      auto arg = convertExpression(callExpr->getArg(0), context);
      if (arg) {
        return std::make_unique<WaveActiveOp>(std::move(arg),
                                              WaveActiveOp::Min);
      }
    } else if (funcName == "WaveActiveMax" && callExpr->getNumArgs() == 1) {
      auto arg = convertExpression(callExpr->getArg(0), context);
      if (arg) {
        return std::make_unique<WaveActiveOp>(std::move(arg),
                                              WaveActiveOp::Max);
      }
    } else if (funcName == "WaveActiveAnd" && callExpr->getNumArgs() == 1) {
      auto arg = convertExpression(callExpr->getArg(0), context);
      if (arg) {
        return std::make_unique<WaveActiveOp>(std::move(arg),
                                              WaveActiveOp::And);
      }
    } else if (funcName == "WaveActiveOr" && callExpr->getNumArgs() == 1) {
      auto arg = convertExpression(callExpr->getArg(0), context);
      if (arg) {
        return std::make_unique<WaveActiveOp>(std::move(arg), WaveActiveOp::Or);
      }
    } else if (funcName == "WaveActiveXor" && callExpr->getNumArgs() == 1) {
      auto arg = convertExpression(callExpr->getArg(0), context);
      if (arg) {
        return std::make_unique<WaveActiveOp>(std::move(arg),
                                              WaveActiveOp::Xor);
      }
    } else if (funcName == "WaveActiveCountBits" &&
               callExpr->getNumArgs() == 1) {
      auto arg = convertExpression(callExpr->getArg(0), context);
      if (arg) {
        return std::make_unique<WaveActiveOp>(std::move(arg),
                                              WaveActiveOp::CountBits);
      }
    } else if (funcName == "WaveGetLaneIndex" && callExpr->getNumArgs() == 0) {
      return std::make_unique<LaneIndexExpr>();
    } else if (funcName == "WaveGetLaneCount" && callExpr->getNumArgs() == 0) {
      return std::make_unique<WaveGetLaneCountExpr>();
    } else if (funcName == "WaveIsFirstLane" && callExpr->getNumArgs() == 0) {
      return std::make_unique<WaveIsFirstLaneExpr>();
    } else if (funcName == "WaveActiveAllEqual" &&
               callExpr->getNumArgs() == 1) {
      auto arg = convertExpression(callExpr->getArg(0), context);
      if (arg) {
        return std::make_unique<WaveActiveAllEqualExpr>(std::move(arg));
      }
    } else if (funcName == "WaveActiveAllTrue" && callExpr->getNumArgs() == 1) {
      auto arg = convertExpression(callExpr->getArg(0), context);
      if (arg) {
        return std::make_unique<WaveActiveAllTrueExpr>(std::move(arg));
      }
    } else if (funcName == "WaveActiveAnyTrue" && callExpr->getNumArgs() == 1) {
      auto arg = convertExpression(callExpr->getArg(0), context);
      if (arg) {
        return std::make_unique<WaveActiveAnyTrueExpr>(std::move(arg));
      }
    }

    std::cout << "Unsupported function call in expression context: " << funcName
              << std::endl;
  }

  return nullptr;
}

std::unique_ptr<Expression> MiniHLSLInterpreter::convertConditionalOperator(
    const clang::ConditionalOperator *condOp, clang::ASTContext &context) {
  std::cout << "Converting conditional operator (ternary)" << std::endl;

  // Convert the condition
  auto condition = convertExpression(condOp->getCond(), context);
  if (!condition) {
    std::cout << "Failed to convert conditional operator condition"
              << std::endl;
    return nullptr;
  }

  // Convert the true expression
  auto trueExpr = convertExpression(condOp->getTrueExpr(), context);
  if (!trueExpr) {
    std::cout << "Failed to convert conditional operator true expression"
              << std::endl;
    return nullptr;
  }

  // Convert the false expression
  auto falseExpr = convertExpression(condOp->getFalseExpr(), context);
  if (!falseExpr) {
    std::cout << "Failed to convert conditional operator false expression"
              << std::endl;
    return nullptr;
  }

  return std::make_unique<ConditionalExpr>(
      std::move(condition), std::move(trueExpr), std::move(falseExpr));
}

std::unique_ptr<Expression>
MiniHLSLInterpreter::convertBinaryExpression(const clang::BinaryOperator *binOp,
                                             clang::ASTContext &context) {
  auto lhs = convertExpression(binOp->getLHS(), context);
  auto rhs = convertExpression(binOp->getRHS(), context);

  if (!lhs || !rhs)
    return nullptr;

  // Map Clang binary operator to interpreter binary operator
  BinaryOpExpr::OpType opType;
  switch (binOp->getOpcode()) {
  case clang::BO_Add:
    opType = BinaryOpExpr::Add;
    break;
  case clang::BO_Sub:
    opType = BinaryOpExpr::Sub;
    break;
  case clang::BO_Mul:
    opType = BinaryOpExpr::Mul;
    break;
  case clang::BO_Div:
    opType = BinaryOpExpr::Div;
    break;
  case clang::BO_Rem:
    opType = BinaryOpExpr::Mod;
    break;
  default:
    std::cout << "Unsupported binary operator" << std::endl;
    return nullptr;
  }

  return makeBinaryOp(std::move(lhs), std::move(rhs), opType);
}

std::unique_ptr<Expression> MiniHLSLInterpreter::convertOperatorCall(
    const clang::CXXOperatorCallExpr *opCall, clang::ASTContext &context) {
  clang::OverloadedOperatorKind op = opCall->getOperator();

  if (op == clang::OO_Subscript) {
    // Array access: buffer[index]
    std::cout << "Converting array access operator[]" << std::endl;

    if (opCall->getNumArgs() >= 2) {
      // Get the base expression (the array/buffer being accessed)
      auto baseExpr = opCall->getArg(0);
      std::string bufferName;

      // Try to extract buffer name from DeclRefExpr
      if (auto declRef = clang::dyn_cast<clang::DeclRefExpr>(baseExpr)) {
        bufferName = declRef->getDecl()->getName().str();
        std::cout << "Array access on buffer: " << bufferName << std::endl;
      } else {
        std::cout << "Complex base expression for array access" << std::endl;
        bufferName = "unknown_buffer";
      }

      // Convert the index expression
      auto indexExpr = convertExpression(opCall->getArg(1), context);
      if (indexExpr) {
        // For shared memory buffers, we need to compute the memory address
        // For now, we'll use a simple model where each buffer starts at address
        // 0 and each element is 4 bytes (size of int/float)

        // Create a shared memory read expression
        // Note: In a real implementation, we'd need to track buffer base
        // addresses and handle different data types/sizes MemoryAddress
        // baseAddr = 0; // Simplified: assume buffer starts at 0 (unused for
        // now)

        // Create an expression that computes: index * sizeof(element)
        auto sizeofElement =
            makeLiteral(Value(4)); // Assume 4 bytes per element
        auto offset = std::make_unique<BinaryOpExpr>(
            std::move(indexExpr), std::move(sizeofElement), BinaryOpExpr::Mul);

        // For now, just use the index directly as the address (simplified)
        // In a real implementation, we'd add the base address
        return std::make_unique<SharedReadExpr>(0); // Placeholder

        // TODO: Properly implement SharedReadExpr that takes a dynamic address
        // expression return
        // std::make_unique<SharedReadExpr>(std::move(offset));
      }
    }
  }

  std::cout << "Unsupported operator call" << std::endl;
  return nullptr;
}

// ConditionalExpr implementation
ConditionalExpr::ConditionalExpr(std::unique_ptr<Expression> condition,
                                 std::unique_ptr<Expression> trueExpr,
                                 std::unique_ptr<Expression> falseExpr)
    : condition_(std::move(condition)), trueExpr_(std::move(trueExpr)),
      falseExpr_(std::move(falseExpr)) {}

Value ConditionalExpr::evaluate(LaneContext &lane, WaveContext &wave,
                                ThreadgroupContext &tg) const {
  if (!lane.isActive)
    return Value(0);

  auto condValue = condition_->evaluate(lane, wave, tg);
  bool cond = condValue.asBool();

  if (cond) {
    return trueExpr_->evaluate(lane, wave, tg);
  } else {
    return falseExpr_->evaluate(lane, wave, tg);
  }
}

bool ConditionalExpr::isDeterministic() const {
  return condition_->isDeterministic() && trueExpr_->isDeterministic() &&
         falseExpr_->isDeterministic();
}

std::string ConditionalExpr::toString() const {
  return "(" + condition_->toString() + " ? " + trueExpr_->toString() + " : " +
         falseExpr_->toString() + ")";
}

// WaveIsFirstLaneExpr implementation
Value WaveIsFirstLaneExpr::evaluate(LaneContext &lane, WaveContext &wave,
                                    ThreadgroupContext &tg) const {
  if (!lane.isActive)
    return Value(false);

  // Find the first active lane in the wave
  for (LaneId lid = 0; lid < wave.lanes.size(); ++lid) {
    if (wave.lanes[lid]->isActive) {
      return Value(lane.laneId == lid);
    }
  }

  return Value(false);
}

// WaveActiveAllEqualExpr implementation
WaveActiveAllEqualExpr::WaveActiveAllEqualExpr(std::unique_ptr<Expression> expr)
    : expr_(std::move(expr)) {}

Value WaveActiveAllEqualExpr::evaluate(LaneContext &lane, WaveContext &wave,
                                       ThreadgroupContext &tg) const {
  if (!lane.isActive)
    return Value(false);

  // Only check lanes executing together in the same control flow block
  auto lanesInSameBlock = tg.getWaveActiveLanesInSameBlock(wave.waveId, lane.laneId);

  if (lanesInSameBlock.empty()) {
    return Value(true); // Vacuously true if no lanes in block
  }

  // Get value from the first lane in the block
  Value firstValue =
      expr_->evaluate(*wave.lanes[lanesInSameBlock[0]], wave, tg);

  // Compare all other lanes in the block with the first
  for (size_t i = 1; i < lanesInSameBlock.size(); ++i) {
    LaneId laneId = lanesInSameBlock[i];
    Value val = expr_->evaluate(*wave.lanes[laneId], wave, tg);
    if (val.asInt() != firstValue.asInt()) {
      return Value(false);
    }
  }

  return Value(true);
}

std::string WaveActiveAllEqualExpr::toString() const {
  return "WaveActiveAllEqual(" + expr_->toString() + ")";
}

// WaveActiveAllTrueExpr implementation
WaveActiveAllTrueExpr::WaveActiveAllTrueExpr(std::unique_ptr<Expression> expr)
    : expr_(std::move(expr)) {}

Value WaveActiveAllTrueExpr::evaluate(LaneContext &lane, WaveContext &wave,
                                      ThreadgroupContext &tg) const {
  if (!lane.isActive)
    return Value(false);

  // Only check lanes executing together in the same control flow block
  auto lanesInSameBlock = tg.getWaveActiveLanesInSameBlock(wave.waveId, lane.laneId);
  for (LaneId laneId : lanesInSameBlock) {
    Value val = expr_->evaluate(*wave.lanes[laneId], wave, tg);
    if (!val.asBool()) {
      return Value(false);
    }
  }

  return Value(true);
}

std::string WaveActiveAllTrueExpr::toString() const {
  return "WaveActiveAllTrue(" + expr_->toString() + ")";
}

// WaveActiveAnyTrueExpr implementation
WaveActiveAnyTrueExpr::WaveActiveAnyTrueExpr(std::unique_ptr<Expression> expr)
    : expr_(std::move(expr)) {}

Value WaveActiveAnyTrueExpr::evaluate(LaneContext &lane, WaveContext &wave,
                                      ThreadgroupContext &tg) const {
  if (!lane.isActive)
    return Value(false);

  // Only check lanes executing together in the same control flow block
  auto lanesInSameBlock = tg.getWaveActiveLanesInSameBlock(wave.waveId, lane.laneId);
  for (LaneId laneId : lanesInSameBlock) {
    Value val = expr_->evaluate(*wave.lanes[laneId], wave, tg);
    if (val.asBool()) {
      return Value(true);
    }
  }

  return Value(false);
}

std::string WaveActiveAnyTrueExpr::toString() const {
  return "WaveActiveAnyTrue(" + expr_->toString() + ")";
}

// WhileStmt implementation
WhileStmt::WhileStmt(std::unique_ptr<Expression> cond,
                     std::vector<std::unique_ptr<Statement>> body)
    : condition_(std::move(cond)), body_(std::move(body)) {}

void WhileStmt::execute(LaneContext &lane, WaveContext &wave,
                        ThreadgroupContext &tg) {
  if (!lane.isActive)
    return;

  // Get current block before entering loop
  uint32_t parentBlockId = tg.getCurrentBlock(wave.waveId, lane.laneId);

  // Execute while loop with condition checking
  while (true) {
    auto condValue = condition_->evaluate(lane, wave, tg);
    if (!condValue.asBool())
      break;

    try {
      // PROACTIVE: Create iteration block for this iteration
      std::vector<MergeStackEntry> currentMergeStack =
          wave.getCurrentMergeStack(lane.laneId);
      uint32_t iterationBlockId = tg.createLoopIterationBlock(
          static_cast<const void *>(this), parentBlockId, currentMergeStack);

      // This lane enters the iteration block
      tg.moveThreadFromUnknownToParticipating(iterationBlockId, lane.laneId,
                                              wave.waveId);

      // Execute body in this iteration's block
      for (auto &stmt : body_) {
        stmt->execute(lane, wave, tg);
        if (!lane.isActive)
          return; // Early exit if lane becomes inactive
      }
    } catch (const ControlFlowException &e) {
      if (e.type == ControlFlowException::Break) {
        break; // Exit the loop
      } else if (e.type == ControlFlowException::Continue) {
        continue; // Go to next iteration
      }
    }
  }
}

std::string WhileStmt::toString() const {
  std::string result = "while (" + condition_->toString() + ") {\n";
  for (const auto &stmt : body_) {
    result += "  " + stmt->toString() + "\n";
  }
  result += "}";
  return result;
}

// DoWhileStmt implementation
DoWhileStmt::DoWhileStmt(std::vector<std::unique_ptr<Statement>> body,
                         std::unique_ptr<Expression> cond)
    : body_(std::move(body)), condition_(std::move(cond)) {}

void DoWhileStmt::execute(LaneContext &lane, WaveContext &wave,
                          ThreadgroupContext &tg) {
  if (!lane.isActive)
    return;

  // Get current block before entering loop
  uint32_t parentBlockId = tg.getCurrentBlock(wave.waveId, lane.laneId);

  // Execute do-while loop - body executes at least once
  do {
    try {
      // PROACTIVE: Create iteration block for this iteration
      std::vector<MergeStackEntry> currentMergeStack =
          wave.getCurrentMergeStack(lane.laneId);
      uint32_t iterationBlockId = tg.createLoopIterationBlock(
          static_cast<const void *>(this), parentBlockId, currentMergeStack);

      // This lane enters the iteration block
      tg.moveThreadFromUnknownToParticipating(iterationBlockId, lane.laneId,
                                              wave.waveId);

      // Execute body in this iteration's block
      for (auto &stmt : body_) {
        stmt->execute(lane, wave, tg);
        if (!lane.isActive)
          return; // Early exit if lane becomes inactive
      }
    } catch (const ControlFlowException &e) {
      if (e.type == ControlFlowException::Break) {
        break; // Exit the loop
      } else if (e.type == ControlFlowException::Continue) {
        // Continue to condition check
      }
    }

    auto condValue = condition_->evaluate(lane, wave, tg);
    if (!condValue.asBool())
      break;
  } while (true);
}

std::string DoWhileStmt::toString() const {
  std::string result = "do {\n";
  for (const auto &stmt : body_) {
    result += "  " + stmt->toString() + "\n";
  }
  result += "} while (" + condition_->toString() + ");";
  return result;
}

// SwitchStmt implementation
SwitchStmt::SwitchStmt(std::unique_ptr<Expression> cond)
    : condition_(std::move(cond)) {}

void SwitchStmt::addCase(int value,
                         std::vector<std::unique_ptr<Statement>> stmts) {
  cases_.push_back({value, std::move(stmts)});
}

void SwitchStmt::addDefault(std::vector<std::unique_ptr<Statement>> stmts) {
  cases_.push_back({std::nullopt, std::move(stmts)});
}

void SwitchStmt::execute(LaneContext &lane, WaveContext &wave,
                         ThreadgroupContext &tg) {
  if (!lane.isActive)
    return;

  // Get current block before switch divergence
  uint32_t parentBlockId = tg.getCurrentBlock(wave.waveId, lane.laneId);

  // Evaluate switch condition
  auto condValue = condition_->evaluate(lane, wave, tg);
  int switchValue = condValue.asInt();

  // PROACTIVE: Create all case blocks
  std::vector<int> caseValues;
  bool hasDefault = false;
  for (const auto &caseBlock : cases_) {
    if (caseBlock.value.has_value()) {
      caseValues.push_back(caseBlock.value.value());
    } else {
      hasDefault = true;
    }
  }

  std::vector<MergeStackEntry> currentMergeStack =
      wave.getCurrentMergeStack(lane.laneId);
  std::vector<uint32_t> caseBlockIds =
      tg.createSwitchCaseBlocks(static_cast<const void *>(this), parentBlockId,
                                currentMergeStack, caseValues, hasDefault);

  // Find which case this lane should execute
  int matchingCaseIndex = -1;
  for (size_t i = 0; i < cases_.size(); ++i) {
    if (cases_[i].value.has_value() && cases_[i].value.value() == switchValue) {
      matchingCaseIndex = i;
      break;
    } else if (!cases_[i].value.has_value()) {
      // Default case - only use if no exact match found
      if (matchingCaseIndex == -1) {
        matchingCaseIndex = i;
      }
    }
  }

  if (matchingCaseIndex != -1) {
    // Lane goes to the matching case block
    uint32_t chosenBlockId = caseBlockIds[matchingCaseIndex];
    tg.moveThreadFromUnknownToParticipating(chosenBlockId, lane.laneId,
                                            wave.waveId);

    // Remove lane from all other case blocks
    for (size_t i = 0; i < caseBlockIds.size(); ++i) {
      if (i != matchingCaseIndex) {
        tg.removeThreadFromUnknown(caseBlockIds[i], lane.laneId, wave.waveId);
        tg.removeThreadFromNestedBlocks(caseBlockIds[i], lane.laneId,
                                        wave.waveId);
      }
    }
  } else {
    // No matching case - remove lane from all case blocks
    for (uint32_t blockId : caseBlockIds) {
      tg.removeThreadFromUnknown(blockId, lane.laneId, wave.waveId);
      tg.removeThreadFromNestedBlocks(blockId, lane.laneId, wave.waveId);
    }
    return; // Lane doesn't execute any case
  }

  // Find matching case (for execution)
  bool foundMatch = false;
  bool fallthrough = false;

  for (const auto &caseBlock : cases_) {
    if (!foundMatch && caseBlock.value.has_value()) {
      if (caseBlock.value.value() == switchValue) {
        foundMatch = true;
        fallthrough = true;
      }
    } else if (!foundMatch && !caseBlock.value.has_value()) {
      // Default case
      foundMatch = true;
      fallthrough = true;
    }

    if (fallthrough) {
      try {
        for (auto &stmt : caseBlock.statements) {
          stmt->execute(lane, wave, tg);
          if (!lane.isActive)
            return;
        }
        // Continue to next case (fallthrough)
      } catch (const ControlFlowException &e) {
        if (e.type == ControlFlowException::Break) {
          break; // Exit switch
        }
        // Continue statements don't apply to switch
      }
    }
  }
}

std::string SwitchStmt::toString() const {
  std::string result = "switch (" + condition_->toString() + ") {\n";
  for (const auto &caseBlock : cases_) {
    if (caseBlock.value.has_value()) {
      result += "  case " + std::to_string(caseBlock.value.value()) + ":\n";
    } else {
      result += "  default:\n";
    }
    for (const auto &stmt : caseBlock.statements) {
      result += "    " + stmt->toString() + "\n";
    }
  }
  result += "}";
  return result;
}

// BreakStmt implementation
void BreakStmt::execute(LaneContext &lane, WaveContext &wave,
                        ThreadgroupContext &tg) {
  if (!lane.isActive)
    return;
  throw ControlFlowException(ControlFlowException::Break);
}

// ContinueStmt implementation
void ContinueStmt::execute(LaneContext &lane, WaveContext &wave,
                           ThreadgroupContext &tg) {
  if (!lane.isActive)
    return;
  throw ControlFlowException(ControlFlowException::Continue);
}

// Dynamic execution block methods
uint32_t ThreadgroupContext::createExecutionBlock(
    const std::map<WaveId, std::set<LaneId>> &lanes, const void *sourceStmt) {
  uint32_t blockId = nextBlockId++;

  DynamicExecutionBlock block;
  block.setBlockId(blockId);
  for (const auto& [waveId, laneSet] : lanes) {
    for (LaneId laneId : laneSet) {
      block.addParticipatingLane(waveId, laneId);
    }
  }
  block.setProgramPoint(0);
  block.setSourceStatement(sourceStmt);
  
  // Calculate total lanes across all waves
  size_t totalLanes = 0;
  for (const auto& [waveId, laneSet] : lanes) {
    totalLanes += laneSet.size();
  }
  block.setIsConverged(totalLanes == threadgroupSize);

  executionBlocks[blockId] = block;

  // Assign all lanes to this block
  for (const auto &[waveId, laneSet] : lanes) {
    for (LaneId laneId : laneSet) {
      assignLaneToBlock(waveId, laneId, blockId);
    }
  }

  return blockId;
}

void ThreadgroupContext::assignLaneToBlock(WaveId waveId, LaneId laneId,
                                           uint32_t blockId) {
  waves[waveId]->laneToCurrentBlock[laneId] = blockId;

  // Add lane to the block's participating lanes
  auto it = executionBlocks.find(blockId);
  if (it != executionBlocks.end()) {
    it->second.addParticipatingLane(waveId, laneId);

    // Check if converged - all lanes from all waves are in this block
    size_t totalLanesInBlock = 0;
    for (const auto &[wid, lanes] : it->second.getParticipatingLanes()) {
      totalLanesInBlock += lanes.size();
    }
    // todo: now convergence is at threadgroup level
    it->second.setIsConverged(totalLanesInBlock == threadgroupSize);
  }
}


// get all active lanes of same wave in the same dynamic block
std::vector<LaneId>
ThreadgroupContext::getWaveActiveLanesInSameBlock(WaveId waveId, LaneId laneId) const {
  auto it = waves[waveId]->laneToCurrentBlock.find(laneId);
  if (it == waves[waveId]->laneToCurrentBlock.end()) {
    return {}; // Lane not in any block
  }

  uint32_t blockId = it->second;
  auto blockIt = executionBlocks.find(blockId);
  if (blockIt == executionBlocks.end()) {
    return {}; // Block not found
  }

  // Return only currently active lanes from the same wave in the same block
  std::vector<LaneId> result;
  auto participatingLanes = blockIt->second.getParticipatingLanesForWave(waveId);
  for (LaneId otherLaneId : participatingLanes) {
      if (otherLaneId < waves[waveId]->lanes.size() &&
          waves[waveId]->lanes[otherLaneId]->isActive &&
          waves[waveId]->lanes[otherLaneId]->state == ThreadState::Ready &&
          !waves[waveId]->lanes[otherLaneId]->hasReturned) {
        result.push_back(otherLaneId);
      }
    }
  return result;
}

uint32_t ThreadgroupContext::getCurrentBlock(WaveId waveId,
                                             LaneId laneId) const {
  auto it = waves[waveId]->laneToCurrentBlock.find(laneId);
  return it != waves[waveId]->laneToCurrentBlock.end() ? it->second : 0;
}

bool ThreadgroupContext::areInSameBlock(WaveId wave1, LaneId lane1,
                                        WaveId wave2, LaneId lane2) const {
  auto it1 = waves[wave1]->laneToCurrentBlock.find(lane1);
  auto it2 = waves[wave2]->laneToCurrentBlock.find(lane2);

  if (it1 == waves[wave1]->laneToCurrentBlock.end() ||
      it2 == waves[wave2]->laneToCurrentBlock.end()) {
    return false;
  }

  return it1->second == it2->second;
}

void ThreadgroupContext::mergeExecutionPaths(
    const std::vector<uint32_t> &blockIds, uint32_t targetBlockId) {
  // Create or update the target block
  std::map<WaveId, std::set<LaneId>> mergedLanes;

  for (uint32_t blockId : blockIds) {
    auto it = executionBlocks.find(blockId);
    if (it != executionBlocks.end()) {
      // Merge lanes from this block, organized by wave
      for (const auto &[waveId, laneSet] : it->second.getParticipatingLanes()) {
        mergedLanes[waveId].insert(laneSet.begin(), laneSet.end());
      }
    }
  }

  // Count total lanes across all waves
  size_t totalMergedLanes = 0;
  for (const auto &[waveId, laneSet] : mergedLanes) {
    totalMergedLanes += laneSet.size();
  }

  // Create the target block with merged lanes
  if (executionBlocks.find(targetBlockId) == executionBlocks.end()) {
    DynamicExecutionBlock targetBlock;
    targetBlock.setBlockId(targetBlockId);
    for (const auto& [waveId, laneSet] : mergedLanes) {
      for (LaneId laneId : laneSet) {
        targetBlock.addParticipatingLane(waveId, laneId);
      }
    }
    targetBlock.setProgramPoint(0);
    targetBlock.setIsConverged(totalMergedLanes == threadgroupSize);
    executionBlocks[targetBlockId] = targetBlock;
  } else {
    // Update existing target block
    auto& targetBlock = executionBlocks[targetBlockId];
    // Clear existing lanes and add merged ones
    for (const auto& [waveId, _] : targetBlock.getParticipatingLanes()) {
      auto lanes = targetBlock.getParticipatingLanesForWave(waveId);
      for (LaneId laneId : lanes) {
        targetBlock.removeParticipatingLane(waveId, laneId);
      }
    }
    for (const auto& [waveId, laneSet] : mergedLanes) {
      for (LaneId laneId : laneSet) {
        targetBlock.addParticipatingLane(waveId, laneId);
      }
    }
    targetBlock.setIsConverged(totalMergedLanes == threadgroupSize);
  }

  // Reassign all lanes to the target block
  for (const auto &[waveId, laneSet] : mergedLanes) {
    for (LaneId laneId : laneSet) {
      waves[waveId]->laneToCurrentBlock[laneId] = targetBlockId;
    }
  }

  // Remove the source blocks (except target if it was in the list)
  for (uint32_t blockId : blockIds) {
    if (blockId != targetBlockId) {
      executionBlocks.erase(blockId);
    }
  }
}

// Cooperative scheduling methods
void ThreadgroupContext::markLaneArrived(WaveId waveId, LaneId laneId,
                                         uint32_t blockId) {
  auto it = executionBlocks.find(blockId);
  if (it != executionBlocks.end()) {
    it->second.addArrivedLane(waveId, laneId);
    it->second.removeUnknownLane(waveId, laneId); // No longer unknown

    // // If this wave's unknown set is now empty, remove it
    // if (it->second.getUnknownLanesForWave(waveId).empty()) {
    //     it->second.removeUnknownLane(waveId)
    // //   it->second.unknownLanes.erase(waveId);
    // }

    // Update lane assignment
    waves[waveId]->laneToCurrentBlock[laneId] = blockId;

    // Check if all unknown lanes are now resolved (no waves have unknown lanes)
    // it->second.allUnknownResolved = it->second.unknownLanes.empty();
    it->second.setWaveAllUnknownResolved(waveId, it->second.getUnknownLanesForWave(waveId).empty());
  }
}

void ThreadgroupContext::markLaneWaitingForWave(WaveId waveId, LaneId laneId,
                                                uint32_t blockId) {
  auto it = executionBlocks.find(blockId);
  if (it != executionBlocks.end()) {
    // it->second.waitingLanes[waveId].insert(laneId);
    it->second.addWaitingLane(waveId, laneId);
    // Change lane state to waiting
    if (waveId < waves.size() && laneId < waves[waveId]->lanes.size()) {
      waves[waveId]->lanes[laneId]->state = ThreadState::WaitingForWave;
    }
  }
}

bool ThreadgroupContext::canExecuteWaveOperation(WaveId waveId,
                                                 LaneId laneId) const {
  auto it = waves[waveId]->laneToCurrentBlock.find(laneId);
  if (it == waves[waveId]->laneToCurrentBlock.end()) {
    return false; // Lane not in any block
  }

  uint32_t blockId = it->second;
  auto blockIt = executionBlocks.find(blockId);
  if (blockIt == executionBlocks.end()) {
    return false; // Block not found
  }

  const auto &block = blockIt->second;

  // Can execute if all unknown lanes are resolved (we know the complete
  // participant set)
  return block.areAllUnknownLanesResolvedForWave(waveId);
}

std::vector<LaneId>
ThreadgroupContext::getWaveOperationParticipants(WaveId waveId,
                                                 LaneId laneId) const {
  auto it = waves[waveId]->laneToCurrentBlock.find(laneId);
  if (it == waves[waveId]->laneToCurrentBlock.end()) {
    return {}; // Lane not in any block
  }

  uint32_t blockId = it->second;
  auto blockIt = executionBlocks.find(blockId);
  if (blockIt == executionBlocks.end()) {
    return {}; // Block not found
  }

  const auto &block = blockIt->second;

  // Return all lanes from the same wave that have arrived at this block and are
  // still active
  std::vector<LaneId> participants;
  auto arrivedLanes = block.getArrivedLanesForWave(waveId);
  for (LaneId participantId : arrivedLanes) {
    if (participantId < waves[waveId]->lanes.size() &&
        waves[waveId]->lanes[participantId]->isActive &&
        !waves[waveId]->lanes[participantId]->hasReturned) {
      participants.push_back(participantId);
    }
  }

  return participants;
}

// void ThreadgroupContext::resolveUnknownLane(WaveId waveId, LaneId laneId,
//                                             uint32_t chosenBlockId) {
//   // Remove this lane from all blocks' unknown sets
//   for (auto &[blockId, block] : executionBlocks) {
//     block.unknownLanes.erase(laneId);

//     // Update resolution status
//     block.allUnknownResolved = block.unknownLanes.empty();
//   }

//   // Mark the lane as arrived at the chosen block
//   markLaneArrived(waveId, laneId, chosenBlockId);
// }

bool ThreadgroupContext::areAllUnknownLanesResolved(uint32_t blockId) const {
  auto it = executionBlocks.find(blockId);
  if (it == executionBlocks.end()) {
    return true; // Block not found, consider resolved
  }

  return it->second.areAllUnknownLanesResolved();
//   return it->second.allUnknownResolved;
}

// TODO: check the functional correctness
// Block deduplication methods
uint32_t ThreadgroupContext::findOrCreateBlockForPath(
    const BlockIdentity &identity,
    const std::map<WaveId, std::set<LaneId>> &unknownLanes) {
  // Check if block with this identity already exists
  auto it = identityToBlockId.find(identity);
  if (it != identityToBlockId.end()) {
    // Block exists, add unknown lanes to it
    uint32_t existingBlockId = it->second;
    auto &existingBlock = executionBlocks[existingBlockId];

    // Add new unknown lanes to existing block
    for (const auto& [waveId, laneSet] : unknownLanes) {
      for (LaneId laneId : laneSet) {
        existingBlock.addUnknownLane(waveId, laneId);
      }
    }
    // Update resolution status for all waves
    for (const auto& [waveId, _] : unknownLanes) {
      existingBlock.setWaveAllUnknownResolved(waveId, existingBlock.areAllUnknownLanesResolvedForWave(waveId));
    }

    return existingBlockId;
  }

  // Create new block with this identity
  uint32_t newBlockId = nextBlockId++;

  DynamicExecutionBlock newBlock;
  newBlock.setBlockId(newBlockId);
  newBlock.setIdentity(identity);
  newBlock.setParentBlockId(identity.parentBlockId);
  newBlock.setSourceStatement(identity.sourceStatement);
  
  // Add unknown lanes
  for (const auto& [waveId, laneSet] : unknownLanes) {
    for (LaneId laneId : laneSet) {
      newBlock.addUnknownLane(waveId, laneId);
    }
  }
  
  // Update resolution status for all waves
  for (const auto& [waveId, _] : unknownLanes) {
    newBlock.setWaveAllUnknownResolved(waveId, newBlock.areAllUnknownLanesResolvedForWave(waveId));
  }
  
  newBlock.setIsConverged(false); // Will be updated as lanes arrive

  // Store the new block
  executionBlocks[newBlockId] = newBlock;
  identityToBlockId[identity] = newBlockId;

  return newBlockId;
}

uint32_t
ThreadgroupContext::findBlockByIdentity(const BlockIdentity &identity) const {
  auto it = identityToBlockId.find(identity);
  return it != identityToBlockId.end() ? it->second : 0;
}

BlockIdentity ThreadgroupContext::createBlockIdentity(
    const void *sourceStmt, bool conditionValue, uint32_t parentBlockId,
    const std::vector<MergeStackEntry> &mergeStack) const {
  BlockIdentity identity;
  identity.sourceStatement = sourceStmt;
  identity.conditionValue = conditionValue;
  identity.parentBlockId = parentBlockId;
  identity.mergeStack = mergeStack;
  return identity;
}

// Instruction-level synchronization methods
bool ThreadgroupContext::canExecuteWaveInstruction(
    WaveId waveId, LaneId laneId, const void *instruction) const {
  // Get the current block for this lane
  uint32_t currentBlockId = getCurrentBlock(waveId, laneId);
  if (currentBlockId == 0) {
    return false; // Lane not assigned to any block
  }

  // Create instruction identity for lookup
  InstructionIdentity instrIdentity;
  instrIdentity.instruction = instruction;
  instrIdentity.instructionType =
      "WaveActiveOp"; // This should be passed as parameter in a full
                      // implementation
  instrIdentity.sourceExpression = instruction;

  // Use the new per-block approach for instruction synchronization
  bool canExecuteInBlock =
      canExecuteInstructionInBlock(currentBlockId, instrIdentity);

  // Also check the global sync point system for backward compatibility
  // This allows gradual migration to the new system
  bool canExecuteGlobal =
      areAllParticipantsKnownForWaveInstruction(waveId, instruction) &&
      haveAllParticipantsArrivedAtWaveInstruction(waveId, instruction);

  // Both approaches should agree for proper synchronization
  return canExecuteInBlock && canExecuteGlobal;
}

void ThreadgroupContext::markLaneArrivedAtWaveInstruction(
    WaveId waveId, LaneId laneId, const void *instruction,
    const std::string &instructionType) {
  // Create instruction identity
  InstructionIdentity instrIdentity =
      createInstructionIdentity(instruction, instructionType, instruction);

  // Get current block for this lane
  uint32_t currentBlockId = getCurrentBlock(waveId, laneId);
  if (currentBlockId == 0) {
    throw std::runtime_error(
        "Lane not assigned to any block when arriving at instruction");
  }

  // Add instruction to block with this lane as participant
  std::map<WaveId, std::set<LaneId>> participants = {{waveId, {laneId}}};
  addInstructionToBlock(currentBlockId, instrIdentity, participants);

  // Create or update sync point for this instruction
  createOrUpdateWaveSyncPoint(instruction, waveId, laneId, instructionType);

  auto &syncPoint = waves[waveId]->activeSyncPoints[instruction];
  syncPoint.arrivedParticipants.insert(laneId);

  // Mark lane as waiting at this instruction
  waves[waveId]->laneWaitingAtInstruction[laneId] = instruction;
  waves[waveId]->lanes[laneId]->state = ThreadState::WaitingForWave;

  // Update completion status
  syncPoint.allParticipantsArrived =
      (syncPoint.arrivedParticipants == syncPoint.expectedParticipants);
  syncPoint.isComplete =
      syncPoint.allParticipantsKnown && syncPoint.allParticipantsArrived;
}

bool ThreadgroupContext::areAllParticipantsKnownForWaveInstruction(
    WaveId waveId, const void *instruction) const {
  auto it = waves[waveId]->activeSyncPoints.find(instruction);
  if (it == waves[waveId]->activeSyncPoints.end()) {
    return false; // No sync point created yet
  }

  return it->second.allParticipantsKnown;
}

bool ThreadgroupContext::haveAllParticipantsArrivedAtWaveInstruction(
    WaveId waveId, const void *instruction) const {
  auto it = waves[waveId]->activeSyncPoints.find(instruction);
  if (it == waves[waveId]->activeSyncPoints.end()) {
    return false; // No sync point created yet
  }

  return it->second.allParticipantsArrived;
}

std::vector<LaneId> ThreadgroupContext::getWaveInstructionParticipants(
    WaveId waveId, const void *instruction) const {
  auto it = waves[waveId]->activeSyncPoints.find(instruction);
  if (it == waves[waveId]->activeSyncPoints.end()) {
    return {}; // No sync point
  }

  const auto &syncPoint = it->second;
  std::vector<LaneId> participants;

  // Return expected participants (from the execution block)
  for (LaneId laneId : syncPoint.expectedParticipants) {
    if (laneId < waves[waveId]->lanes.size() &&
        waves[waveId]->lanes[laneId]->isActive &&
        !waves[waveId]->lanes[laneId]->hasReturned) {
      participants.push_back(laneId);
    }
  }

  return participants;
}

void ThreadgroupContext::createOrUpdateWaveSyncPoint(
    const void *instruction, WaveId waveId, LaneId laneId,
    const std::string &instructionType) {
  auto it = waves[waveId]->activeSyncPoints.find(instruction);

  if (it == waves[waveId]->activeSyncPoints.end()) {
    // Create new sync point
    WaveOperationSyncPoint syncPoint;
    syncPoint.instruction = instruction;
    syncPoint.instructionType = instructionType;

    // Determine which block this lane is in and get expected participants
    uint32_t blockId = getCurrentBlock(waveId, laneId);
    syncPoint.blockId = blockId;

    // Get expected participants from the block
    auto blockParticipants = getWaveOperationParticipants(waveId, laneId);
    for (LaneId participantId : blockParticipants) {
      syncPoint.expectedParticipants.insert(participantId);
    }

    // Check if all participants are known (no unknown lanes in block)
    syncPoint.allParticipantsKnown = areAllUnknownLanesResolved(blockId);

    waves[waveId]->activeSyncPoints[instruction] = syncPoint;
  } else {
    // Update existing sync point
    auto &syncPoint = it->second;

    // Update participants knowledge if block resolved more lanes
    uint32_t blockId = getCurrentBlock(waveId, laneId);
    syncPoint.allParticipantsKnown = areAllUnknownLanesResolved(blockId);

    // Update expected participants if needed
    auto blockParticipants = getWaveOperationParticipants(waveId, laneId);
    for (LaneId participantId : blockParticipants) {
      syncPoint.expectedParticipants.insert(participantId);
    }
  }
}

void ThreadgroupContext::releaseWaveSyncPoint(WaveId waveId,
                                              const void *instruction) {
  auto it = waves[waveId]->activeSyncPoints.find(instruction);
  if (it != waves[waveId]->activeSyncPoints.end()) {
    const auto &syncPoint = it->second;

    // Release all waiting lanes
    for (LaneId laneId : syncPoint.arrivedParticipants) {
      if (laneId < waves[waveId]->lanes.size()) {
        waves[waveId]->lanes[laneId]->state = ThreadState::Ready;
      }
      waves[waveId]->laneWaitingAtInstruction.erase(laneId);
    }

    // Remove the sync point
    waves[waveId]->activeSyncPoints.erase(it);
  }
}

// Merge stack management methods
void ThreadgroupContext::pushMergePoint(
    WaveId waveId, LaneId laneId, const void *sourceStmt, uint32_t parentBlockId,
    const std::set<uint32_t> &divergentBlocks) {
  MergeStackEntry entry;
  entry.sourceStatement = sourceStmt;
  entry.parentBlockId = parentBlockId;
  entry.divergentBlockIds = divergentBlocks;

  // Add to lane-specific merge stack
  waves[waveId]->laneMergeStacks[laneId].push_back(entry);
}

void ThreadgroupContext::popMergePoint(WaveId waveId, LaneId laneId) {
  auto it = waves[waveId]->laneMergeStacks.find(laneId);
  if (it != waves[waveId]->laneMergeStacks.end() && !it->second.empty()) {
    it->second.pop_back();
  }
}

std::vector<MergeStackEntry>
ThreadgroupContext::getCurrentMergeStack(WaveId waveId, LaneId laneId) const {
  auto it = waves[waveId]->laneMergeStacks.find(laneId);
  if (it != waves[waveId]->laneMergeStacks.end()) {
    return it->second;
  }
  return {};
}

void ThreadgroupContext::updateMergeStack(
    WaveId waveId, LaneId laneId, const std::vector<MergeStackEntry> &mergeStack) {
  waves[waveId]->laneMergeStacks[laneId] = mergeStack;
}

// Instruction identity management methods
void ThreadgroupContext::addInstructionToBlock(
    uint32_t blockId, const InstructionIdentity &instruction,
    const std::map<WaveId, std::set<LaneId>> &participants) {
  auto blockIt = executionBlocks.find(blockId);
  if (blockIt == executionBlocks.end()) {
    return; // Block doesn't exist
  }

  DynamicExecutionBlock &block = blockIt->second;

  // Add instruction to the ordered list if not already present
  bool found = false;
  for (const auto &existingInstr : block.getInstructionList()) {
    if (existingInstr == instruction) {
      found = true;
      break;
    }
  }

  if (!found) {
    block.addInstruction(instruction);
  }

  // Merge participants for this instruction (don't replace, add to existing)
  for (const auto &[waveId, newLanes] : participants) {
    for (LaneId laneId : newLanes) {
      block.addInstructionParticipant(instruction, waveId, laneId);
    }
  }
}

InstructionIdentity ThreadgroupContext::createInstructionIdentity(
    const void *instruction, const std::string &instructionType,
    const void *sourceExpr) const {
  InstructionIdentity identity;
  identity.instruction = instruction;
  identity.instructionType = instructionType;
  identity.sourceExpression = sourceExpr;
  return identity;
}

std::vector<InstructionIdentity>
ThreadgroupContext::getBlockInstructions(uint32_t blockId) const {
  auto blockIt = executionBlocks.find(blockId);
  if (blockIt != executionBlocks.end()) {
    return blockIt->second.getInstructionList();
  }
  return {};
}

std::map<WaveId, std::set<LaneId>>
ThreadgroupContext::getInstructionParticipantsInBlock(
    uint32_t blockId, const InstructionIdentity &instruction) const {
  auto blockIt = executionBlocks.find(blockId);
  if (blockIt != executionBlocks.end()) {
    const auto& instructionParticipants = blockIt->second.getInstructionParticipants();
    auto instrIt = instructionParticipants.find(instruction);
    if (instrIt != instructionParticipants.end()) {
      return instrIt->second;
    }
  }
  return {};
}

bool ThreadgroupContext::canExecuteInstructionInBlock(
    uint32_t blockId, const InstructionIdentity &instruction) const {
  auto blockIt = executionBlocks.find(blockId);
  if (blockIt == executionBlocks.end()) {
    return false; // Block doesn't exist
  }

  const DynamicExecutionBlock &block = blockIt->second;

  // Check if all unknown lanes in this block are resolved
  if (!block.areAllUnknownLanesResolved()) {
    return false; // Still have unknown lanes
  }

  // Get expected participants (all active lanes in the block)
  std::map<WaveId, std::set<LaneId>> expectedParticipants =
      block.getParticipatingLanes();

  // Get actual participants who have arrived at this instruction
  auto arrivedParticipants =
      getInstructionParticipantsInBlock(blockId, instruction);

  // Check if all expected participants have arrived
  return arrivedParticipants == expectedParticipants;
}

std::map<WaveId, std::set<LaneId>>
ThreadgroupContext::getExpectedParticipantsInBlock(
    uint32_t blockId, const InstructionIdentity &instruction) const {
  auto blockIt = executionBlocks.find(blockId);
  if (blockIt != executionBlocks.end()) {
    // For wave operations, all participating lanes in the block should
    // participate
    return blockIt->second.getParticipatingLanes();
  }
  return {};
}

// Proactive block creation methods
std::pair<uint32_t, uint32_t> ThreadgroupContext::createIfBlocks(
    const void *ifStmt, uint32_t parentBlockId,
    const std::vector<MergeStackEntry> &mergeStack, bool hasElse) {
  // Get all lanes that could potentially take either path
  std::map<WaveId, std::set<LaneId>> allPotentialLanes =
      getCurrentBlockParticipants(parentBlockId);

  // Always create the then block
  BlockIdentity thenIdentity =
      createBlockIdentity(ifStmt, true, parentBlockId, mergeStack);
  uint32_t thenBlockId =
      findOrCreateBlockForPath(thenIdentity, allPotentialLanes);

  uint32_t elseBlockId = 0; // 0 means no else block

  // Only create else block if it exists in the code
  if (hasElse) {
    BlockIdentity elseIdentity =
        createBlockIdentity(ifStmt, false, parentBlockId, mergeStack);
    elseBlockId = findOrCreateBlockForPath(elseIdentity, allPotentialLanes);
  }

  return {thenBlockId, elseBlockId};
}

void ThreadgroupContext::moveThreadFromUnknownToParticipating(uint32_t blockId,
                                                              WaveId waveId,
                                                              LaneId laneId) {
  auto blockIt = executionBlocks.find(blockId);
  if (blockIt == executionBlocks.end())
    return;

  DynamicExecutionBlock &block = blockIt->second;

  // Move lane from unknown to participating and arrived
  block.removeUnknownLane(waveId, laneId);
  block.addParticipatingLane(waveId, laneId);
  block.addArrivedLane(waveId, laneId);

  // Update resolution status
  block.setWaveAllUnknownResolved(waveId, block.areAllUnknownLanesResolvedForWave(waveId));

  // Update lane assignment
  assignLaneToBlock(waveId, laneId, blockId);
}

void ThreadgroupContext::removeThreadFromUnknown(uint32_t blockId,
                                                 WaveId waveId, LaneId laneId) {
  auto blockIt = executionBlocks.find(blockId);
  if (blockIt == executionBlocks.end())
    return;

  DynamicExecutionBlock &block = blockIt->second;

  // Remove lane from unknown (it chose a different path)
  block.removeUnknownLane(waveId, laneId);

  // Update resolution status
  block.setWaveAllUnknownResolved(waveId, block.areAllUnknownLanesResolvedForWave(waveId));
}

void ThreadgroupContext::removeThreadFromNestedBlocks(uint32_t parentBlockId,
                                                      WaveId waveId,
                                                      LaneId laneId) {
  // Find all blocks that are nested within the parent block and remove the lane
  for (auto &[blockId, block] : executionBlocks) {
    if (block.getParentBlockId() == parentBlockId) {
      // This is a direct child of the parent block
      removeThreadFromUnknown(blockId, waveId, laneId);

      // Recursively remove from nested blocks of this child
      removeThreadFromNestedBlocks(blockId, waveId, laneId);
    }
  }
}

std::map<WaveId, std::set<LaneId>>
ThreadgroupContext::getCurrentBlockParticipants(uint32_t blockId) const {
  auto blockIt = executionBlocks.find(blockId);
  if (blockIt != executionBlocks.end()) {
    // Return union of participating, arrived, waiting, and unknown lanes
    std::map<WaveId, std::set<LaneId>> allLanes;
    const auto &block = blockIt->second;

    for (const auto &[waveId, lanes] : block.getParticipatingLanes()) {
      allLanes[waveId].insert(lanes.begin(), lanes.end());
    }
    for (const auto &[waveId, lanes] : block.getArrivedLanes()) {
      allLanes[waveId].insert(lanes.begin(), lanes.end());
    }
    for (const auto &[waveId, lanes] : block.getWaitingLanes()) {
      allLanes[waveId].insert(lanes.begin(), lanes.end());
    }
    for (const auto &[waveId, lanes] : block.getUnknownLanes()) {
      allLanes[waveId].insert(lanes.begin(), lanes.end());
    }

    return allLanes;
  }
  return {};
}

uint32_t ThreadgroupContext::createLoopIterationBlock(
    const void *loopStmt, uint32_t parentBlockId,
    const std::vector<MergeStackEntry> &mergeStack) {
  // Get all lanes that could potentially enter the loop iteration
  std::map<WaveId, std::set<LaneId>> allPotentialLanes =
      getCurrentBlockParticipants(parentBlockId);

  // Create iteration block (always true condition for loop bodies)
  BlockIdentity iterationIdentity =
      createBlockIdentity(loopStmt, true, parentBlockId, mergeStack);
  return findOrCreateBlockForPath(iterationIdentity, allPotentialLanes);
}

std::vector<uint32_t> ThreadgroupContext::createSwitchCaseBlocks(
    const void *switchStmt, uint32_t parentBlockId,
    const std::vector<MergeStackEntry> &mergeStack,
    const std::vector<int> &caseValues, bool hasDefault) {
  // Get all lanes that could potentially take any case path
  std::map<WaveId, std::set<LaneId>> allPotentialLanes =
      getCurrentBlockParticipants(parentBlockId);

  std::vector<uint32_t> caseBlockIds;

  // Create a block for each case value
  for (int caseValue : caseValues) {
    BlockIdentity caseIdentity =
        createBlockIdentity(switchStmt, true, parentBlockId, mergeStack);
    uint32_t caseBlockId =
        findOrCreateBlockForPath(caseIdentity, allPotentialLanes);
    caseBlockIds.push_back(caseBlockId);
  }

  // Create default block if it exists
  if (hasDefault) {
    BlockIdentity defaultIdentity =
        createBlockIdentity(switchStmt, false, parentBlockId, mergeStack);
    uint32_t defaultBlockId =
        findOrCreateBlockForPath(defaultIdentity, allPotentialLanes);
    caseBlockIds.push_back(defaultBlockId);
  }

  return caseBlockIds;
}

} // namespace interpreter
} // namespace minihlsl